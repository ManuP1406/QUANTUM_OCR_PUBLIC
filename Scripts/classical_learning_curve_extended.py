
"""
  1. Pre-compute ALL GrakelGraphs for every combination of parameters
     (label_type, grid_size, n_dev) on the full pool (max 400 graphs)
  2. For each seed and train_size, extract subgraphs for indexing
  3. GRID SIZE OPTIMIZED via cross-validation for position/angle/combined
  4. Grid search on n_iter and C also for wl_none (baseline)
  5. Kernels tested: WL with different label types
  6. NORMALIZED KERNELS to be invariant to graph size
"""

import os
import sys
import time
import pickle
import numpy as np
from itertools import combinations
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

import networkx as nx
from grakel.kernels import WeisfeilerLehman, RandomWalk
from grakel import Graph as GrakelGraph




def load_data(nx_pkl_path, classes, max_per_class, max_test_per_class):
    """Load NetworkX graphs and positions."""
    with open(nx_pkl_path, "rb") as f:
        data = pickle.load(f)

    G_all    = data["G"]
    y_all    = data["y"]
    pos_all  = data["G_positions"]
    G_test   = data["G_test"]
    y_test   = data["y_test"]
    pos_test = data["G_test_positions"]

    # Training pool (max_per_class per class)
    train_idx = []
    for c in classes:
        idx = np.where(y_all == c)[0][:max_per_class]
        train_idx.extend(idx)
    G_train   = [G_all[i]   for i in train_idx]
    y_train   = y_all[train_idx]
    pos_train = [pos_all[i] for i in train_idx]

    # Fixed test set
    test_idx = []
    for c in classes:
        idx = np.where(y_test == c)[0][:max_test_per_class]
        test_idx.extend(idx)
    G_test_sel   = [G_test[i]   for i in test_idx]
    y_test_sel   = y_test[test_idx]
    pos_test_sel = [pos_test[i] for i in test_idx]

    return (G_train, np.array(y_train), pos_train,
            G_test_sel, np.array(y_test_sel), pos_test_sel)




def _pos_labels(pos_dict, grid_size=8):
    """Position-based labels (grid)."""
    if not pos_dict:
        return {}
    all_x = [x for x, _ in pos_dict.values()]
    all_y = [y for _, y in pos_dict.values()]
    w = max(all_x) + 1 if all_x else 1
    h = max(all_y) + 1 if all_y else 1
    
    out = {}
    for node, (x, y) in pos_dict.items():
        rx = int(min(x // (w / grid_size), grid_size - 1))
        ry = int(min(y // (h / grid_size), grid_size - 1))
        out[node] = rx * grid_size + ry
    return out


def _angle_labels_simple(G_nx, pos_dict, n_dev=4):
    """Labels based on normalized degree (simple and robust)."""
    if G_nx.number_of_nodes() == 0:
        return {}
    
    degrees = [G_nx.degree(i) for i in range(G_nx.number_of_nodes())]
    max_deg = max(degrees) if degrees else 1
    
    out = {}
    for i, deg in enumerate(degrees):
        # Robust normalization to avoid division by zero
        if max_deg > 0:
            bin_idx = int((deg / max_deg) * (n_dev - 1))
        else:
            bin_idx = 0
        out[i] = max(0, min(bin_idx, n_dev - 1))
    return out


def _build_grakel(G_nx, node_labels):
    """Build GrakelGraph. Assumes nodes already renamed 0..n-1."""
    n = G_nx.number_of_nodes()
    # Ensure every node has a label
    for i in range(n):
        node_labels.setdefault(i, 0)
    adj = {i: list(G_nx.neighbors(i)) for i in range(n)}
    return GrakelGraph(adj, node_labels=node_labels)


def build_grakel_set(G_list, pos_list, label_type="none", grid_size=8, n_dev=4):
    """Build list of GrakelGraphs (version for pre-computation)."""
    out = []
    for G_orig, pos in zip(G_list, pos_list):
        # Rename nodes to 0..n-1 to ensure consistency
        mapping = {old: new for new, old in enumerate(G_orig.nodes())}
        G = nx.relabel_nodes(G_orig, mapping)
        
        # Positions with new IDs
        pd = {mapping[old]: (pos[old][0], pos[old][1]) 
              for old in G_orig.nodes()}
        
        n = G.number_of_nodes()
        
        if label_type == "none":
            lbl = {i: 1 for i in range(n)}
        elif label_type == "position":
            lbl = _pos_labels(pd, grid_size)
        elif label_type == "angle":
            lbl = _angle_labels_simple(G, pd, n_dev)
        elif label_type == "combined":
            pl = _pos_labels(pd, grid_size)
            al = _angle_labels_simple(G, pd, n_dev)
            # Unique encoding that combines position and degree
            lbl = {i: pl.get(i, 0) * 100 + al.get(i, 0) for i in range(n)}
        else:
            lbl = {i: 1 for i in range(n)}
        
        out.append(_build_grakel(G, lbl))
    
    return out




def _cv_svm_normalized(K_train, y_train, C_vals=(0.1, 1.0, 10.0), cv=3):
    """
    Cross-validation for SVM on ALREADY NORMALIZED kernel.
    The input kernel is already normalized (invariant to graph size).
    """
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return 1.0
    
    n_splits = min(cv, min(np.bincount(y_train)))
    if n_splits < 2:
        return 1.0
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_acc, best_C = 0.0, 1.0
    
    for C in C_vals:
        fold_accs = []
        for train_idx, val_idx in skf.split(K_train, y_train):
            # Kernel already normalized, no additional normalization
            K_tr = K_train[np.ix_(train_idx, train_idx)]
            K_va = K_train[np.ix_(val_idx, train_idx)]
            
            svm = SVC(C=C, kernel='precomputed', cache_size=500)
            svm.fit(K_tr, y_train[train_idx])
            pred = svm.predict(K_va)
            fold_accs.append(accuracy_score(y_train[val_idx], pred))
        
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_C = C
    
    return best_C



def _wl_grid_search_no_gridsize(G_tr_gk, y_tr, 
                                 n_iter_vals=(2, 3, 4), 
                                 C_vals=(0.1, 1.0, 10.0), 
                                 cv=3):
  
    unique_classes = np.unique(y_tr)
    if len(unique_classes) < 2:
        return 3, 1.0
    
    n_splits = min(cv, min(np.bincount(y_tr)))
    if n_splits < 2:
        return 3, 1.0
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_acc, best_n, best_C = 0.0, 3, 1.0
    
    for n_iter in n_iter_vals:
        # Kernel normalized internally
        wl = WeisfeilerLehman(n_iter=n_iter, normalize=True)
        
        try:
            K_full = wl.fit_transform(G_tr_gk)
        except Exception as e:
            print(f"            Error with n_iter={n_iter}: {e}")
            continue
        
        # Verify that the kernel is valid
        if np.any(np.isnan(K_full)) or np.any(np.isinf(K_full)):
            print(f"            Invalid kernel for n_iter={n_iter}")
            continue
        
        for C in C_vals:
            fold_accs = []
            for train_idx, val_idx in skf.split(K_full, y_tr):
                # Kernel already normalized, no further normalization needed
                K_tr = K_full[np.ix_(train_idx, train_idx)]
                K_va = K_full[np.ix_(val_idx, train_idx)]
                
                svm = SVC(C=C, kernel='precomputed', cache_size=500)
                svm.fit(K_tr, y_tr[train_idx])
                pred = svm.predict(K_va)
                fold_accs.append(accuracy_score(y_tr[val_idx], pred))
            
            mean_acc = np.mean(fold_accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_n = n_iter
                best_C = C
    
    return best_n, best_C


def _wl_grid_search_with_gridsize(
    G_original_list, pos_list, y_tr,
    label_type="position",
    n_iter_vals=(2, 3, 4),
    grid_size_vals=(4, 6, 8, 10),
    C_vals=(0.1, 1.0, 10.0),
    cv=3
):
    """
    Grid search on (n_iter, grid_size, C) for WL kernel.
    The kernel is normalized internally by GraKeL.
    Rebuilds GrakelGraphs for each grid_size.
    """
    unique_classes = np.unique(y_tr)
    if len(unique_classes) < 2:
        return 3, 8, 1.0
    
    n_splits = min(cv, min(np.bincount(y_tr)))
    if n_splits < 2:
        return 3, 8, 1.0
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_acc = 0.0
    best_n = 3
    best_gs = 8
    best_C = 1.0
    
    for grid_size in grid_size_vals:
        # Rebuild graphs for this grid_size
        G_gk = build_grakel_set(G_original_list, pos_list, label_type, 
                                grid_size=grid_size, n_dev=4)
        
        for n_iter in n_iter_vals:
            # Kernel normalized internally
            wl = WeisfeilerLehman(n_iter=n_iter, normalize=True)
            
            try:
                K_full = wl.fit_transform(G_gk)
            except Exception as e:
                print(f"            Error with n_iter={n_iter}, grid_size={grid_size}: {e}")
                continue
            
            # Verify kernel validity
            if np.any(np.isnan(K_full)) or np.any(np.isinf(K_full)):
                print(f"            Invalid kernel for n_iter={n_iter}, grid_size={grid_size}")
                continue
            
            for C in C_vals:
                fold_accs = []
                for train_idx, val_idx in skf.split(K_full, y_tr):
                    # Kernel already normalized
                    K_tr = K_full[np.ix_(train_idx, train_idx)]
                    K_va = K_full[np.ix_(val_idx, train_idx)]
                    
                    svm = SVC(C=C, kernel='precomputed', cache_size=500)
                    svm.fit(K_tr, y_tr[train_idx])
                    pred = svm.predict(K_va)
                    fold_accs.append(accuracy_score(y_tr[val_idx], pred))
                
                mean_acc = np.mean(fold_accs)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_n = n_iter
                    best_gs = grid_size
                    best_C = C
    
    return best_n, best_gs, best_C


def train_wl_svm_with_gridsize(G_tr_nx, pos_tr, y_tr, G_te_nx, pos_te, y_te,
                                label_type="position",
                                n_iter_vals=(2, 3, 4),
                                grid_size_vals=(4, 6, 8, 10),
                                C_vals=(0.1, 1.0, 10.0)):
    """
    Train WL+SVM with grid search on (n_iter, grid_size, C).
    Kernel normalized internally by GraKeL.
    """
    # Grid search on training data
    best_n, best_gs, best_C = _wl_grid_search_with_gridsize(
        G_tr_nx, pos_tr, y_tr,
        label_type=label_type,
        n_iter_vals=n_iter_vals,
        grid_size_vals=grid_size_vals,
        C_vals=C_vals,
        cv=3
    )
    
    # Rebuild with the best parameters
    G_tr_gk = build_grakel_set(G_tr_nx, pos_tr, label_type, 
                                grid_size=best_gs, n_dev=4)
    G_te_gk = build_grakel_set(G_te_nx, pos_te, label_type,
                                grid_size=best_gs, n_dev=4)
    
    # Normalized kernel
    wl = WeisfeilerLehman(n_iter=best_n, normalize=True)
    K_tr = wl.fit_transform(G_tr_gk)
    K_te = wl.transform(G_te_gk)
    
    # Kernels already normalized, no further normalization needed
    svm = SVC(C=best_C, kernel='precomputed', cache_size=500)
    svm.fit(K_tr, y_tr)
    
    return svm.predict(K_te), {'n_iter': best_n, 'grid_size': best_gs, 'C': best_C}


def train_wl_svm_fixed(G_tr_gk, y_tr, G_te_gk, n_iter=3, C=1.0):
    """
    Train WL kernel with fixed parameters (for baseline).
    Kernel normalized internally by GraKeL.
    """
    # Normalized kernel
    wl = WeisfeilerLehman(n_iter=n_iter, normalize=True)
    
    try:
        K_tr = wl.fit_transform(G_tr_gk)
        K_te = wl.transform(G_te_gk)
    except Exception as e:
        print(f"Error computing WL kernel: {e}")
        # Fallback: random predictions
        return np.random.choice(np.unique(y_tr), size=len(G_te_gk))
    
    # Kernel already normalized
    svm = SVC(C=C, kernel='precomputed', cache_size=500)
    svm.fit(K_tr, y_tr)
    
    return svm.predict(K_te)




def train_random_walk(G_tr_gk, y_tr, G_te_gk, lamda=0.01):
    """Random walk kernel with normalization."""
    # Kernel normalized internally
    rw = RandomWalk(normalize=True, lamda=lamda)
    
    try:
        K_tr = rw.fit_transform(G_tr_gk)
        K_te = rw.transform(G_te_gk)
    except Exception as e:
        print(f"Error computing RandomWalk kernel: {e}")
        return np.random.choice(np.unique(y_tr), size=len(G_te_gk))
    
    # Kernel already normalized, find best C via CV
    best_C = _cv_svm_normalized(K_tr, y_tr, C_vals=(0.1, 1.0, 10.0))
    
    svm = SVC(C=best_C, kernel='precomputed', cache_size=500)
    svm.fit(K_tr, y_tr)
    
    return svm.predict(K_te)




def _metrics(y_true, y_pred):
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _aggregate(per_size_results, train_sizes):
    """Aggregate results across all seeds for each train size."""
    scalar_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    aggregated = {}
    
    for size in train_sizes:
        seed_list = per_size_results[size]
        
        if len(seed_list) == 0:
            aggregated[size] = {f"mean_{k}": 0.0 for k in scalar_keys}
            aggregated[size].update({f"std_{k}": 0.0 for k in scalar_keys})
            aggregated[size]["n_seeds"] = 0
            aggregated[size]["mean_confusion_matrix"] = []
            aggregated[size]["std_confusion_matrix"] = []
            continue
        
        agg = {"n_seeds": len(seed_list)}
        
        for k in scalar_keys:
            vals = [s[k] for s in seed_list]
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        
        # Aggregation of confusion matrices
        cms = np.array([s["confusion_matrix"] for s in seed_list])
        agg["mean_confusion_matrix"] = np.mean(cms, axis=0).tolist()
        agg["std_confusion_matrix"] = np.std(cms, axis=0, ddof=1).tolist() if len(cms) > 1 else np.zeros_like(cms[0]).tolist()
        
        aggregated[size] = agg
    
    return aggregated



def run_classical_learning_curve(
    nx_pkl_path,
    classes=list(range(10)),
    train_sizes=[10, 15, 20, 25, 30, 35, 40],
    max_per_class=40,
    max_test_per_class=15,
    n_seeds=5,
    # Models to run
    run_wl_position=True,   # WL with positions (with grid size search)
    run_wl_angle=True,      # WL with angles (with grid size search)
    run_wl_combined=True,   # WL combined (with grid size search)
    run_wl_none=True,       # WL without labels (with grid search on n_iter and C)
    run_rw=False,           # Random Walk (very slow)
    # Grid size optimizations
    wl_n_iter_vals=(2, 3, 4),
    wl_grid_size_vals=(4, 6, 8, 10, 12),
    wl_c_vals=(0.1, 1.0, 10.0),
    results_save_path="results/classical_learning_curve.pkl",
    summary_save_path="results/classical_learning_curve_summary.txt",
):
    """
    Run learning curve with classical models and grid size optimization.
    ALL kernels are normalized to be invariant to graph size.
    """
    
    t_start = time.time()
    classes = list(classes)
    
    print("\n" + "=" * 70)
    print("CLASSICAL MODELS  --  LEARNING CURVE (GRID SIZE OPTIMIZATION)")
    print("=" * 70)
    print(f"  Classes           : {classes}")
    print(f"  Train sizes       : {train_sizes}")
    print(f"  Seeds             : {n_seeds}")
    print(f"  Max per class     : {max_per_class}")
    print(f"  Max test          : {max_test_per_class}")
    print(f"  WL position       : {run_wl_position} (grid size search)")
    print(f"  WL angle          : {run_wl_angle} (grid size search)")
    print(f"  WL combined       : {run_wl_combined} (grid size search)")
    print(f"  WL none           : {run_wl_none} (grid search on n_iter and C)")
    print(f"  Random Walk       : {run_rw}")
    print(f"  Grid size values  : {wl_grid_size_vals}")
    print(f"  n_iter values     : {wl_n_iter_vals}")
    print(f"  C values          : {wl_c_vals}")
    print("  KERNEL NORMALIZATION: ENABLED (invariant to graph size)")
    print("=" * 70)
    
    # ── Load data ─────────────────────────────────────────────────────────
    G_train, y_train, pos_train, G_test, y_test, pos_test = load_data(
        nx_pkl_path, classes, max_per_class, max_test_per_class
    )
    print(f"\nTraining pool: {len(G_train)} graphs")
    print(f"Fixed test set: {len(G_test)} graphs")
    
    # Filter out graphs that are too small (helps kernel stability)
    min_nodes = 2
    valid_train_idx = [i for i, g in enumerate(G_train) if g.number_of_nodes() >= min_nodes]
    G_train = [G_train[i] for i in valid_train_idx]
    y_train = y_train[valid_train_idx]
    pos_train = [pos_train[i] for i in valid_train_idx]
    print(f"After filter (nodes>={min_nodes}): {len(G_train)} graphs")
    
    # Indices per class
    class_indices = {c: np.where(y_train == c)[0].tolist() for c in classes}
    for c, idx in class_indices.items():
        print(f"  Class {c}: {len(idx)} available samples")
    
    # ── Initialize result structures ────────────────────────────────────
    model_names = []
    if run_wl_position:
        model_names.append("WL_position_grid")
    if run_wl_angle:
        model_names.append("WL_angle_grid")
    if run_wl_combined:
        model_names.append("WL_combined_grid")
    if run_wl_none:
        model_names.append("WL_none_grid")
    if run_rw:
        model_names.append("RW_normalized")
    
    per_size = {name: {sz: [] for sz in train_sizes} for name in model_names}
    best_params_log = {sz: [] for sz in train_sizes}
    
    # ── Main loop ─────────────────────────────────────────────────────
    for size in train_sizes:
        print(f"\n{'─' * 60}")
        print(f"  Train size = {size}/class  ({size * len(classes)} total)")
        print(f"{'─' * 60}")
        
        for seed in range(n_seeds):
            print(f"\n    Seed {seed+1}/{n_seeds}:")
            rng = np.random.default_rng(seed)
            
            # Balanced index selection per class
            sel_idx = []
            for c in classes:
                if len(class_indices[c]) < size:
                    print(f"      Warning: class {c} has only {len(class_indices[c])} samples, {size} requested")
                    chosen = class_indices[c]
                else:
                    chosen = rng.choice(class_indices[c], size=size, replace=False)
                sel_idx.extend(chosen.tolist())
            sel_idx = np.array(sel_idx)
            
            G_sub = [G_train[i] for i in sel_idx]
            pos_sub = [pos_train[i] for i in sel_idx]
            y_sub = y_train[sel_idx]
            
            def _record(name, y_pred, t0, extra_info=None):
                """Record results for a model."""
                m = _metrics(y_test, y_pred)
                m.update({"seed": seed, "train_size_pc": size, 
                         "training_time_s": time.time() - t0})
                per_size[name][size].append(m)
                info_str = f"      {name:20s} acc={m['accuracy']:.4f} ({m['training_time_s']:.1f}s)"
                if extra_info:
                    info_str += f" [{extra_info}]"
                print(info_str)
            
            # ── WL with POSITIONS (with grid size search) ─────────────────────
            if run_wl_position:
                t0 = time.time()
                try:
                    y_pred, best_params = train_wl_svm_with_gridsize(
                        G_sub, pos_sub, y_sub,
                        G_test, pos_test, y_test,
                        label_type="position",
                        n_iter_vals=wl_n_iter_vals,
                        grid_size_vals=wl_grid_size_vals,
                        C_vals=wl_c_vals
                    )
                    extra = f"n={best_params['n_iter']},gs={best_params['grid_size']},C={best_params['C']}"
                    _record("WL_position_grid", y_pred, t0, extra)
                    
                    best_params_log[size].append({
                        'seed': seed,
                        'label_type': 'position',
                        'n_iter': best_params['n_iter'],
                        'grid_size': best_params['grid_size'],
                        'C': best_params['C']
                    })
                except Exception as e:
                    print(f"      WL_position_grid ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ── WL with ANGLES (with grid size search) ────────────────────────
            if run_wl_angle:
                t0 = time.time()
                try:
                    # For angle, grid_size_vals is reinterpreted as n_dev_vals
                    y_pred, best_params = train_wl_svm_with_gridsize(
                        G_sub, pos_sub, y_sub,
                        G_test, pos_test, y_test,
                        label_type="angle",
                        n_iter_vals=wl_n_iter_vals,
                        grid_size_vals=[4, 6, 8, 10],  # Interpreted as n_dev
                        C_vals=wl_c_vals
                    )
                    extra = f"n={best_params['n_iter']},n_dev={best_params['grid_size']},C={best_params['C']}"
                    _record("WL_angle_grid", y_pred, t0, extra)
                    
                    best_params_log[size].append({
                        'seed': seed,
                        'label_type': 'angle',
                        'n_iter': best_params['n_iter'],
                        'grid_size': best_params['grid_size'],
                        'C': best_params['C']
                    })
                except Exception as e:
                    print(f"      WL_angle_grid ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ── WL combined ────────────────────────────────────────────────
            if run_wl_combined:
                t0 = time.time()
                try:
                    y_pred, best_params = train_wl_svm_with_gridsize(
                        G_sub, pos_sub, y_sub,
                        G_test, pos_test, y_test,
                        label_type="combined",
                        n_iter_vals=wl_n_iter_vals,
                        grid_size_vals=wl_grid_size_vals,
                        C_vals=wl_c_vals
                    )
                    extra = f"n={best_params['n_iter']},gs={best_params['grid_size']},C={best_params['C']}"
                    _record("WL_combined_grid", y_pred, t0, extra)
                    
                    best_params_log[size].append({
                        'seed': seed,
                        'label_type': 'combined',
                        'n_iter': best_params['n_iter'],
                        'grid_size': best_params['grid_size'],
                        'C': best_params['C']
                    })
                except Exception as e:
                    print(f"      WL_combined_grid ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ── WL without labels (WITH grid search on n_iter and C) ──────────
            if run_wl_none:
                t0 = time.time()
                try:
                    # Build graphs for this subset
                    G_tr_gk = build_grakel_set(G_sub, pos_sub, "none")
                    G_te_gk = build_grakel_set(G_test, pos_test, "none")
                    
                    # Grid search for n_iter and C
                    best_n, best_C = _wl_grid_search_no_gridsize(
                        G_tr_gk, y_sub,
                        n_iter_vals=wl_n_iter_vals,
                        C_vals=wl_c_vals,
                        cv=3
                    )
                    
                    y_pred = train_wl_svm_fixed(G_tr_gk, y_sub, G_te_gk, 
                                                n_iter=best_n, C=best_C)
                    
                    extra = f"n={best_n},C={best_C}"
                    _record("WL_none_grid", y_pred, t0, extra)
                    
                    best_params_log[size].append({
                        'seed': seed,
                        'label_type': 'none',
                        'n_iter': best_n,
                        'grid_size': None,
                        'C': best_C
                    })
                except Exception as e:
                    print(f"      WL_none_grid ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ── Random Walk (with normalization) ─────────────────────────
            if run_rw:
                t0 = time.time()
                try:
                    G_tr_gk = build_grakel_set(G_sub, pos_sub, "none")
                    G_te_gk = build_grakel_set(G_test, pos_test, "none")
                    y_pred = train_random_walk(G_tr_gk, y_sub, G_te_gk)
                    _record("RW_normalized", y_pred, t0)
                except Exception as e:
                    print(f"      RW_normalized ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    # ── Result aggregation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULT AGGREGATION")
    print("=" * 70)
    
    aggregated = {}
    for name in model_names:
        aggregated[name] = _aggregate(per_size[name], train_sizes)
        print(f"\n  {name}:")
        for sz in train_sizes:
            if sz in aggregated[name] and aggregated[name][sz]["n_seeds"] > 0:
                a = aggregated[name][sz]
                print(f"    size={sz:3d}  acc={a['mean_accuracy']:.4f} ± {a['std_accuracy']:.4f}")
            else:
                print(f"    size={sz:3d}  NO RESULTS")
    
    # ── Saving ────────────────────────────────────────────────────────
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "train_sizes": train_sizes,
        "max_per_class": max_per_class,
        "max_test_per_class": max_test_per_class,
        "n_seeds": n_seeds,
        "model_names": model_names,
        "wl_n_iter_vals": list(wl_n_iter_vals),
        "wl_grid_size_vals": list(wl_grid_size_vals),
        "wl_c_vals": list(wl_c_vals),
        "total_time_s": time.time() - t_start,
        "best_params_log": best_params_log,
        "kernel_normalization": True,  # IMPORTANT: indicates that kernels are normalized
    }
    
    results = {
        "train_sizes": train_sizes,
        "per_size": per_size,
        "aggregated": aggregated,
        "meta": meta,
    }
    
    if results_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(results_save_path)), exist_ok=True)
        with open(results_save_path, "wb") as f:
            pickle.dump(results, f, protocol=4)
        print(f"\nResults saved → {results_save_path}")
    
    if summary_save_path:
        _save_summary(results, summary_save_path)
        print(f"Summary saved → {summary_save_path}")
    
    print(f"\nTotal time: {meta['total_time_s']:.1f}s")
    
    # Print summary of best grid_size
    if best_params_log and any(best_params_log.values()):
        print("\n" + "=" * 60)
        print("SUMMARY OF BEST PARAMETERS FOUND")
        print("=" * 60)
        for size, params_list in best_params_log.items():
            if params_list:
                print(f"\n  Size={size}:")
                # Group by label_type
                by_type = {}
                for p in params_list:
                    lt = p['label_type']
                    if lt not in by_type:
                        by_type[lt] = []
                    by_type[lt].append(p)
                
                for lt, plist in by_type.items():
                    if lt == 'none':
                        n_vals = [p['n_iter'] for p in plist]
                        c_vals = [p['C'] for p in plist]
                        most_n = Counter(n_vals).most_common(1)[0]
                        most_c = Counter(c_vals).most_common(1)[0]
                        print(f"    {lt}: n_iter={most_n[0]} ({most_n[1]}/{len(plist)}), C={most_c[0]} ({most_c[1]}/{len(plist)})")
                    else:
                        gs_vals = [p['grid_size'] for p in plist if p['grid_size'] is not None]
                        n_vals = [p['n_iter'] for p in plist]
                        if gs_vals:
                            most_gs = Counter(gs_vals).most_common(1)[0]
                            most_n = Counter(n_vals).most_common(1)[0]
                            print(f"    {lt}: grid_size={most_gs[0]} ({most_gs[1]}/{len(plist)}), n_iter={most_n[0]}")
    
    return results


def _save_summary(results, path):
    """Save readable summary with grid size results."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = results["meta"]
    agg = results["aggregated"]
    sizes = results["train_sizes"]
    
    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSICAL MODELS -- LEARNING CURVE (GRID SIZE OPTIMIZATION)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {meta['timestamp']}\n")
        f.write(f"Total time: {meta['total_time_s']:.1f}s\n")
        f.write(f"KERNEL NORMALIZATION: ENABLED (invariant to graph size)\n\n")
        f.write(f"Classes: {meta['classes']}\n")
        f.write(f"Sizes: {meta['train_sizes']}\n")
        f.write(f"Seeds: {meta['n_seeds']}\n")
        f.write(f"Grid sizes tested: {meta['wl_grid_size_vals']}\n")
        f.write(f"n_iter tested: {meta['wl_n_iter_vals']}\n")
        f.write(f"C tested: {meta['wl_c_vals']}\n\n")
        
        for name in meta["model_names"]:
            f.write(f"\n{'─' * 40}\n")
            f.write(f"{name}\n")
            f.write(f"{'─' * 40}\n")
            f.write(f"{'Size':>8}  {'Acc mean':>10}  {'Acc std':>10}  {'F1 mean':>10}  {'F1 std':>10}\n")
            f.write("-" * 55 + "\n")
            
            for sz in sizes:
                if sz in agg[name] and agg[name][sz]["n_seeds"] > 0:
                    a = agg[name][sz]
                    f.write(f"{sz:>8}  {a['mean_accuracy']:>10.4f}  {a['std_accuracy']:>10.4f}  "
                           f"{a['mean_f1_macro']:>10.4f}  {a['std_f1_macro']:>10.4f}\n")
                else:
                    f.write(f"{sz:>8}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}\n")
        
        # Also save the best parameters
        if meta.get('best_params_log', {}):
            f.write("\n\n" + "─" * 80 + "\n")
            f.write("BEST PARAMETERS FOUND\n")
            f.write("─" * 80 + "\n")
            for size, params_list in meta['best_params_log'].items():
                if params_list:
                    f.write(f"\nSize={size}:\n")
                    for p in params_list[:10]:
                        if p['grid_size'] is None:
                            f.write(f"  seed={p['seed']} ({p['label_type']}): n_iter={p['n_iter']}, C={p['C']}\n")
                        else:
                            f.write(f"  seed={p['seed']} ({p['label_type']}): n_iter={p['n_iter']}, "
                                   f"grid_size={p['grid_size']}, C={p['C']}\n")
        
        f.write("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    date = time.strftime("%d-%m-%Y")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(script_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    results = run_classical_learning_curve(
        nx_pkl_path=os.path.join(script_dir, "..", "Data", "mnist_graphs_nx_filtered.pkl"),
        classes=list(range(10)),
        train_sizes=[10, 15, 20, 25, 30, 35, 40],
        max_per_class=80,
        max_test_per_class=15,
        n_seeds=5,
        # Models to run
        run_wl_position=True,   # WL with positions (with grid size search)
        run_wl_angle=True,      # WL with angles (with grid size search)
        run_wl_combined=True,   # WL combined (with grid size search)
        run_wl_none=True,       # WL without labels (with grid search)
        run_rw=False,           # Random Walk disabled (too slow)
        # Optimizations
        wl_n_iter_vals=(2, 3, 4),
        wl_grid_size_vals=(4, 6, 8, 10, 12),
        wl_c_vals=(0.1, 1.0, 10.0),
        results_save_path=os.path.join(res_dir, f"classical_lc_gridsearch_{date}.pkl"),
        summary_save_path=os.path.join(res_dir, f"classical_lc_summary_{date}.txt"),
    )
