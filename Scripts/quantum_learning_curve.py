"""
Learning Curve Experiment for Quantum Kernel Classification.

Strategy:
  1. Compute quantum distributions ONCE for all available training graphs
     (max_per_class) and all test graphs → cache to disk.
  2. For each train_size in train_sizes:
       For each random seed in range(n_seeds):
         - Subsample `train_size` examples PER CLASS from the cached distributions
         - Train pair-classifiers (SVM only, cheap)
         - Predict on the FULL test set using cached test distributions
         - Record metrics
  3. Compute mean ± std across seeds for every train_size.
  4. Save a rich results dict + a human-readable summary.

No quantum re-simulation is ever needed after the first run,
provided the cache file is kept on disk.
"""

import os
import sys
import time
import pickle
import json

import numpy as np
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)
from joblib import Parallel, delayed

# ── local imports (adjust paths if needed) ──────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantumKernels import QuantumMulticlassClassifier, QuantumPairClassifier
from quantum_kernel_utils.quantum_core import kernel_js
from quantum_kernel_utils.graph_preprocessing import (
    compute_local_scale_factors,
    scale_all_graphs_local,
)
from quantum_kernel_utils.pulser_functions import R_interatomic
from graph_kernel_utils.data_loader import load_graphs_data, filter_and_limit_classes
from quantum_kernel_utils.graph_preprocessing import graph_to_node_list

from quantumKernels import QuantumEvolutionKernel
from quantum_kernel_utils.quantum_core import quantum_loop



def _compute_distributions_for_graphs(
    graphs_node_lists,
    circuit_params,
    LAYERS,
    N_samples,
    encoding_type,
    encoding_method,
    use_correlation,
    min_atom_distance,
    n_jobs_quantum,
):
    """
    Scale graphs and run quantum simulations. Returns a list of distributions
    (one per graph).  Uses the same local-scaling logic as QuantumEvolutionKernel.
    """
    # Scale
    scale_factors = compute_local_scale_factors(graphs_node_lists, min_atom_distance)
    graphs_scaled  = scale_all_graphs_local(graphs_node_lists, scale_factors)

    # Centre each graph
    bad = 0
    for graph in graphs_scaled:
        coords = np.array([node[:2] for node in graph])
        coords -= coords.mean(axis=0)
        for i, node in enumerate(graph):
            node[0] = coords[i, 0]
            node[1] = coords[i, 1]
        if np.any(np.linalg.norm(coords, axis=1) > 50):
            bad += 1
    if bad:
        raise ValueError(f"{bad} graphs have nodes > 50 from origin after centering.")

    distributions = Parallel(n_jobs=n_jobs_quantum)(
        delayed(quantum_loop)(
            circuit_params,
            graph,
            LAYERS,
            N_samples,
            encoding_type,
            encoding_method,
            use_correlation=use_correlation,
        )
        for graph in graphs_scaled
    )
    return distributions


def _train_pair_from_distributions(
    c1, c2,
    train_dists,      # list of distributions for this pair
    train_labels,     # 1-D array with values in {c1, c2}
    n_features,
    circuit_params,
    n_jobs,
    n_jobs_quantum,
    kernel_kwargs,    # LAYERS, gamma, find_optimal_gamma, use_weights, …
):
    """Train one binary SVM from pre-computed distributions. Returns fitted QuantumPairClassifier."""
    pair_clf = QuantumPairClassifier(
        class_pair=(c1, c2),
        distributions=train_dists,
        labels=train_labels,
        n_features=n_features,
        params_in=circuit_params,
        n_jobs=n_jobs,
        n_jobs_quantum=n_jobs_quantum,
        **kernel_kwargs,
    )
    pair_clf.fit()
    print(f"\n  {'='*50}")
    print(f"  FINAL WEIGHTS for {c1} vs {c2}:")
    if pair_clf.weights is not None:
        print(f"    Number of weights: {len(pair_clf.weights)}")
        print(f"    Weights (first 10): {np.round(pair_clf.weights[:10], 4)}")
        print(f"    Gamma: {pair_clf.gamma:.6f}")
        print(f"    Sum of weights: {np.sum(pair_clf.weights):.6f}")
    else:
        print(f"    No weights optimized (use_weights=False)")
        print(f"    Gamma: {pair_clf.gamma:.6f}")
    print(f"  {'='*50}\n")
    return (c1, c2), pair_clf


def _predict_from_pair_classifiers(pair_classifiers, test_distributions, classes):
    """
    One-vs-one voting using pre-computed test distributions.
    No quantum simulation needed.
    """
    n_test       = len(test_distributions)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    total_votes  = np.zeros((n_test, len(classes)))

    for (c1, c2), pair_clf in pair_classifiers.items():
        K_test = kernel_js(
            test_distributions,
            pair_clf.train_distributions,
            pair_clf.gamma,
            pair_clf.weights,
        )
        pred = pair_clf.model.predict(K_test)
        for i, p in enumerate(pred):
            if p == 0:
                total_votes[i, class_to_idx[c1]] += 1
            else:
                total_votes[i, class_to_idx[c2]] += 1

    return np.array([classes[np.argmax(v)] for v in total_votes])


def _metrics(y_true, y_pred):
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted":     float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }



def _load_and_convert(
    orig_pkl_path, rep_pkl_path,
    classes, max_per_class, max_test_per_class,
):
    """Return converted node-lists + labels for train and test."""

    def _convert(G_orig, G_pos_orig, G_pos_rep):
        out = []
        for i, g in enumerate(G_orig):
            nl = graph_to_node_list(g, G_pos_orig[i], G_pos_rep[i], features_type="angle+pos") 
            out.append(nl)
        return out

   
    G_orig, y_orig, G_orig_pos, G_test_orig, y_test_orig, G_test_orig_pos, _, _ = \
        load_graphs_data(orig_pkl_path)
    G_rep,  y_rep,  G_rep_pos,  G_test_rep,  y_test_rep,  G_test_rep_pos,  _, _ = \
        load_graphs_data(rep_pkl_path)

    assert np.array_equal(y_orig, y_rep), "Label mismatch (train)"
    assert np.array_equal(y_test_orig, y_test_rep), "Label mismatch (test)"

  
    G_orig,      y_orig,      G_orig_pos,      _ = filter_and_limit_classes(
        G_orig, y_orig, G_orig_pos, classes, max_per_class, None, dataset_type="training")
    G_rep,       y_rep,       G_rep_pos,       _ = filter_and_limit_classes(
        G_rep,  y_rep,  G_rep_pos,  classes, max_per_class, None,
        dataset_type="training", verbose=False)

    G_test_orig, y_test_orig, G_test_orig_pos, _ = filter_and_limit_classes(
        G_test_orig, y_test_orig, G_test_orig_pos, classes, max_test_per_class, None, dataset_type="test")
    G_test_rep,  y_test_rep,  G_test_rep_pos,  _ = filter_and_limit_classes(
        G_test_rep,  y_test_rep,  G_test_rep_pos,  classes, max_test_per_class, None,
        dataset_type="test", verbose=False)

    train_graphs = _convert(G_orig, G_orig_pos, G_rep_pos)
    test_graphs  = _convert(G_test_orig, G_test_orig_pos, G_test_rep_pos)

    return train_graphs, np.array(y_orig), test_graphs, np.array(y_test_orig)




def _cache_save(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"  [cache] Saved → {path}")


def _cache_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)




def run_learning_curve_experiment(
    
    orig_pkl_path=None,
    rep_pkl_path=None,
   
    classes=[1, 3, 5, 8],
    train_sizes=[10, 15, 20, 25, 30, 35, 40],  # per-class
    max_per_class=40,            # pool to draw from (must be ≥ max(train_sizes))
    max_test_per_class=100,
    n_seeds=5,                   # random subsets per train_size → gives std
 
    circuit_params=None,         # fixed params; if None, use default below
    LAYERS=1,
    gamma=1.0,
    encoding_type="with_hyperfine",
    encoding_method="multi_run", #"multi_run" QUI
    N_samples=200,
    use_correlation="z",
    min_atom_distance=4.1,
    optimize_parameters=True,    # per-pair gamma / weight optimisation
  
    n_jobs=4,                    # parallel pair training
    n_jobs_quantum=2,            # parallel quantum simulations
   
    distributions_cache_path="cache/distributions_cache.pkl",

    results_save_path="results/learning_curve_results.pkl",
    summary_save_path="results/learning_curve_summary.txt",
):
    """
    Run a learning-curve experiment and return a results dictionary.

    Returns
    -------
    dict with keys:
        train_sizes   : list of per-class sizes tested
        per_size      : dict  size → list of per-seed metric dicts
        aggregated    : dict  size → {mean_acc, std_acc, mean_f1, std_f1, …}
        meta          : experiment hyper-parameters
    """

    assert max(train_sizes) <= max_per_class, \
        f"max(train_sizes)={max(train_sizes)} > max_per_class={max_per_class}"

    t_start_total = time.time()


    base_dir = os.path.dirname(os.path.abspath(__file__))

    def _abs(p, default_rel):
        if p is None:
            p = os.path.join(base_dir, "..", "Data", default_rel)
        return os.path.abspath(p)

    orig_pkl_path = _abs(orig_pkl_path, "mnist_graphs_grakel_filtered.pkl")
    rep_pkl_path  = _abs(rep_pkl_path,  "mnist_graphs_grakel_repulsion_filtered.pkl")

    print("\n" + "=" * 70)
    print("QUANTUM KERNEL  —  LEARNING CURVE EXPERIMENT")
    print("=" * 70)
    print(f"  Classes          : {classes}")
    print(f"  Train sizes (pc) : {train_sizes}")
    print(f"  Seeds per size   : {n_seeds}")
    print(f"  Max pool (pc)    : {max_per_class}")
    print(f"  Max test (pc)    : {max_test_per_class}")
    print(f"  LAYERS           : {LAYERS}")
    print(f"  N_samples        : {N_samples}")
    print(f"  Optimize params  : {optimize_parameters}")
    print(f"  n_jobs           : {n_jobs}   |  n_jobs_quantum: {n_jobs_quantum}")
    print(f"  Dist. cache      : {distributions_cache_path}")
    print("=" * 70)

   
    if circuit_params is None:
        # Will be set after we know n_features; placeholder handled below
        _circuit_params_provided = False
    else:
        _circuit_params_provided = True

   
    cache_exists = (
        distributions_cache_path is not None
        and os.path.exists(distributions_cache_path)
    )
    # cache_exists = False# QUI

    if cache_exists:
        print(f"\n[Step 1] Loading cached distributions from:\n  {distributions_cache_path}")
        cache = _cache_load(distributions_cache_path)
        train_dists   = cache["train_dists"]
        train_labels  = cache["train_labels"]
        test_dists    = cache["test_dists"]
        test_labels   = cache["test_labels"]
        n_features    = cache["n_features"]
        circuit_params = cache.get("circuit_params", circuit_params)
        print(f"  Loaded {len(train_dists)} train + {len(test_dists)} test distributions.")
        print(f"  circuit_params = {circuit_params[:4]} …")
    else:
        print("\n[Step 1] Computing quantum distributions (first run) …")

        # Load & convert graphs
        train_graphs, train_labels, test_graphs, test_labels = _load_and_convert(
            orig_pkl_path, rep_pkl_path,
            classes, max_per_class, max_test_per_class,
        )
        n_features = len(train_graphs[0][0]) - 2
        print(f"  n_features per node: {n_features}")
        print(f"  Train graphs: {len(train_graphs)}  |  Test graphs: {len(test_graphs)}")

        # Build circuit params if not provided
        if not _circuit_params_provided:
            t_params     = [240.1] * LAYERS
            theta_params = [0.1 * np.pi] * LAYERS
            freq_params  = [1.0] * (LAYERS * n_features) \
                           if encoding_type == "with_hyperfine" else []
            circuit_params = t_params + theta_params + freq_params
            print(f"  Using default circuit_params: {circuit_params[:4]} …")

        # ── training distributions ──
        print(f"\n  Computing TRAIN distributions ({len(train_graphs)} graphs) …")
        t0 = time.time()
        train_dists = _compute_distributions_for_graphs(
            train_graphs, circuit_params,
            LAYERS, N_samples, encoding_type, encoding_method, use_correlation,
            min_atom_distance, n_jobs_quantum,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        # ── test distributions ──
        print(f"\n  Computing TEST distributions ({len(test_graphs)} graphs) …")
        t0 = time.time()
        test_dists = _compute_distributions_for_graphs(
            test_graphs, circuit_params,
            LAYERS, N_samples, encoding_type, encoding_method, use_correlation,
            min_atom_distance, n_jobs_quantum,
        )
        print(f"  Done in {time.time() - t0:.1f}s")

        # ── save cache ──
        if distributions_cache_path is not None:
            cache = dict(
                train_dists=train_dists,
                train_labels=train_labels,
                test_dists=test_dists,
                test_labels=test_labels,
                n_features=n_features,
                circuit_params=circuit_params,
                classes=classes,
                max_per_class=max_per_class,
                max_test_per_class=max_test_per_class,
                LAYERS=LAYERS,
                N_samples=N_samples,
                encoding_type=encoding_type,
                encoding_method=encoding_method,
                use_correlation=use_correlation,
            )
            _cache_save(distributions_cache_path, cache)

    train_dists  = list(train_dists)   # ensure indexable
    test_dists   = list(test_dists)
    train_labels = np.array(train_labels)
    test_labels  = np.array(test_labels)

    # SVM / kernel kwargs forwarded to pair classifiers
    kernel_kwargs = dict(
        LAYERS=LAYERS,
        gamma=gamma,
        find_optimal_gamma=optimize_parameters,
        use_weights=optimize_parameters,
    )

 
    print("\n" + "=" * 70)
    print("[Step 2] Sweeping train_sizes × seeds")
    print("=" * 70)

    # Build per-class index lists (into the full pool)
    class_indices = {
        c: np.where(train_labels == c)[0].tolist()
        for c in classes
    }
    for c, idx in class_indices.items():
        print(f"  Class {c}: {len(idx)} samples available in pool")

    per_size_results = {}   # size → list[metric_dict]

    for size in train_sizes:
        per_size_results[size] = []
        print(f"\n{'─'*60}")
        print(f"  Training size = {size} per class  ({size * len(classes)} total)")
        print(f"{'─'*60}")

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            t_seed = time.time()

            
            sel_idx = []
            for c in classes:
                avail = class_indices[c]
                if len(avail) < size:
                    raise ValueError(
                        f"Class {c} has only {len(avail)} samples in pool, "
                        f"but train_size={size} was requested."
                    )
                chosen = rng.choice(avail, size=size, replace=False).tolist()
                sel_idx.extend(chosen)

            sel_idx       = np.array(sel_idx)
            sub_dists     = [train_dists[i]  for i in sel_idx]
            sub_labels    = train_labels[sel_idx]

           
            pairs     = list(combinations(classes, 2))
            pair_data = []
            for c1, c2 in pairs:
                mask   = np.isin(sub_labels, [c1, c2])
                p_idx  = np.where(mask)[0]
                if len(p_idx) < 2:
                    continue
                pair_data.append(dict(
                    c1=c1, c2=c2,
                    dists=[sub_dists[i] for i in p_idx],
                    labels=sub_labels[mask],
                ))

            n_par = min(len(pair_data), n_jobs)
            results_pairs = Parallel(n_jobs=n_par)(
                delayed(_train_pair_from_distributions)(
                    pd["c1"], pd["c2"],
                    pd["dists"], pd["labels"],
                    n_features, circuit_params,
                    n_jobs, n_jobs_quantum,
                    kernel_kwargs,
                )
                for pd in pair_data
            )
            pair_classifiers = dict(results_pairs)

           
            y_pred = _predict_from_pair_classifiers(
                pair_classifiers, test_dists, classes
            )

            m = _metrics(test_labels, y_pred)
            m["training_time_s"] = time.time() - t_seed
            m["seed"]            = seed
            m["train_size_pc"]   = size
            per_size_results[size].append(m)

            print(
                f"  size={size:3d}  seed={seed}  "
                f"acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  "
                f"({m['training_time_s']:.1f}s)"
            )


    print("\n" + "=" * 70)
    print("[Step 3] Aggregating results")
    print("=" * 70)

    scalar_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    aggregated  = {}

    for size in train_sizes:
        seed_list = per_size_results[size]
        agg = {"n_seeds": len(seed_list)}
        for k in scalar_keys:
            vals = [s[k] for s in seed_list]
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        # Aggregate confusion matrices
        cms = np.array([s["confusion_matrix"] for s in seed_list])
        agg["mean_confusion_matrix"] = np.mean(cms, axis=0).tolist()
        agg["std_confusion_matrix"]  = np.std(cms, axis=0, ddof=1).tolist() if len(cms) > 1 \
                                        else np.zeros_like(cms[0]).tolist()

        aggregated[size] = agg
        print(
            f"  size={size:3d} → "
            f"acc {agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f}  |  "
            f"f1  {agg['mean_f1_macro']:.4f} ± {agg['std_f1_macro']:.4f}"
        )

   
    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        classes=classes,
        train_sizes=train_sizes,
        max_per_class=max_per_class,
        max_test_per_class=max_test_per_class,
        n_seeds=n_seeds,
        LAYERS=LAYERS,
        gamma=gamma,
        encoding_type=encoding_type,
        encoding_method=encoding_method,
        N_samples=N_samples,
        use_correlation=use_correlation,
        optimize_parameters=optimize_parameters,
        n_jobs=n_jobs,
        n_jobs_quantum=n_jobs_quantum,
        circuit_params=list(circuit_params),
        total_time_s=time.time() - t_start_total,
    )

    final_results = dict(
        train_sizes=train_sizes,
        per_size=per_size_results,
        aggregated=aggregated,
        meta=meta,
    )

    if results_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(results_save_path)), exist_ok=True)
        with open(results_save_path, "wb") as f:
            pickle.dump(final_results, f, protocol=4)
        print(f"\nResults saved → {results_save_path}")

    if summary_save_path:
        _save_summary(final_results, summary_save_path)
        print(f"Summary saved → {summary_save_path}")

    print(f"\nTotal experiment time: {meta['total_time_s']:.1f}s")
    return final_results




def _save_summary(results, path):
    meta  = results["meta"]
    agg   = results["aggregated"]
    sizes = results["train_sizes"]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        W = 72
        f.write("=" * W + "\n")
        f.write("QUANTUM KERNEL  —  LEARNING CURVE SUMMARY\n")
        f.write("=" * W + "\n\n")

        f.write(f"Timestamp        : {meta['timestamp']}\n")
        f.write(f"Total time       : {meta['total_time_s']:.1f}s\n\n")

        f.write("── CONFIGURATION ──\n")
        f.write(f"  Classes          : {meta['classes']}\n")
        f.write(f"  Train sizes (pc) : {meta['train_sizes']}\n")
        f.write(f"  Seeds per size   : {meta['n_seeds']}\n")
        f.write(f"  Max pool (pc)    : {meta['max_per_class']}\n")
        f.write(f"  Max test (pc)    : {meta['max_test_per_class']}\n")
        f.write(f"  LAYERS           : {meta['LAYERS']}\n")
        f.write(f"  Gamma            : {meta['gamma']}\n")
        f.write(f"  Encoding         : {meta['encoding_type']} / {meta['encoding_method']}\n")
        f.write(f"  N_samples        : {meta['N_samples']}\n")
        f.write(f"  use_correlation  : {meta['use_correlation']}\n")
        f.write(f"  optimize_params  : {meta['optimize_parameters']}\n")
        f.write(f"  circuit_params   : {meta['circuit_params'][:4]} …\n\n")

        
        col = 12
        header = (
            f"{'size':>{col}}"
            f"{'acc mean':>{col}}"
            f"{'acc std':>{col}}"
            f"{'f1 mean':>{col}}"
            f"{'f1 std':>{col}}"
            f"{'prec mean':>{col}}"
            f"{'rec mean':>{col}}"
        )
        f.write("── RESULTS TABLE (mean ± std across seeds) ──\n")
        f.write(header + "\n")
        f.write("-" * (col * 7) + "\n")
        for sz in sizes:
            a = agg[sz]
            row = (
                f"{sz:>{col}}"
                f"{a['mean_accuracy']:>{col}.4f}"
                f"{a['std_accuracy']:>{col}.4f}"
                f"{a['mean_f1_macro']:>{col}.4f}"
                f"{a['std_f1_macro']:>{col}.4f}"
                f"{a['mean_precision_macro']:>{col}.4f}"
                f"{a['mean_recall_macro']:>{col}.4f}"
            )
            f.write(row + "\n")
        f.write("\n")


        f.write("── PER-SEED DETAIL ──\n")
        for sz in sizes:
            f.write(f"\n  size = {sz} samples/class\n")
            for sd in results["per_size"][sz]:
                f.write(
                    f"    seed={sd['seed']}  "
                    f"acc={sd['accuracy']:.4f}  "
                    f"f1={sd['f1_macro']:.4f}  "
                    f"prec={sd['precision_macro']:.4f}  "
                    f"rec={sd['recall_macro']:.4f}  "
                    f"time={sd['training_time_s']:.1f}s\n"
                )

        
        f.write("\n── MEAN CONFUSION MATRICES ──\n")
        classes = meta["classes"]
        for sz in sizes:
            cm = np.array(agg[sz]["mean_confusion_matrix"])
            cm_std = np.array(agg[sz]["std_confusion_matrix"])
            f.write(f"\n  size = {sz} samples/class\n")
            header_cm = " " * 8 + "".join(f"{c:>8}" for c in classes)
            f.write(header_cm + "\n")
            for i, c in enumerate(classes):
                row_str = f"{c:>8}" + "".join(
                    f"{cm[i,j]:>6.1f}±{cm_std[i,j]:>4.1f}" for j in range(len(classes))
                )
                f.write(row_str + "\n")

        f.write("\n" + "=" * W + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * W + "\n")




def load_and_print_results(results_path, summary_path=None):
    """Reload a saved results dict and optionally regenerate the summary."""
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    print_aggregated_table(results)
    if summary_path:
        _save_summary(results, summary_path)
    return results


def print_aggregated_table(results):
    """Print a concise aggregated table to stdout."""
    print("\n" + "=" * 72)
    print("LEARNING CURVE — AGGREGATED RESULTS")
    print("=" * 72)
    col = 12
    print(
        f"{'size':>{col}}"
        f"{'acc mean':>{col}}"
        f"{'acc std':>{col}}"
        f"{'f1 mean':>{col}}"
        f"{'f1 std':>{col}}"
    )
    print("-" * (col * 5))
    for sz in results["train_sizes"]:
        a = results["aggregated"][sz]
        print(
            f"{sz:>{col}}"
            f"{a['mean_accuracy']:>{col}.4f}"
            f"{a['std_accuracy']:>{col}.4f}"
            f"{a['mean_f1_macro']:>{col}.4f}"
            f"{a['std_f1_macro']:>{col}.4f}"
        )
    print("=" * 72)



if __name__ == "__main__":
    date = time.strftime("%d-%m-%Y")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cache_dir   = os.path.join(script_dir, "cache")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(cache_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results = run_learning_curve_experiment(
       
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        train_sizes=[10, 15, 20, 25, 30, 35, 40],
        max_per_class=80,
        max_test_per_class=15,
        n_seeds=5,
   
        LAYERS=1,
        N_samples=500,
        optimize_parameters=True,
       
        n_jobs=45,
        n_jobs_quantum=49,
     
        distributions_cache_path=os.path.join(
            cache_dir, f"distributions_{date}.pkl"
        ),
       
        results_save_path=os.path.join(
           
            results_dir, f"learning_curve_{date}_wass_sum_det.pkl"
        ),
        summary_save_path=os.path.join(
            results_dir, f"learning_curve_summary_{date}_wass_sum_det.txt"
        ),
    )

    print_aggregated_table(results)
