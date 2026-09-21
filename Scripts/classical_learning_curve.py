"""
Classical Learning Curve Experiment.

Mirrors the structure of learning_curve_experiment.py (quantum) but for:
  - WL + SVM variants (no labels, position, angle, combined, OvO)
  - GNN models (GCN, GAT) — optional, controlled by run_gnn flag

For each (train_size, seed):
  - Subsample train_size examples per class from the full pool
  - Train model on subsample
  - Evaluate on the FIXED full test set
  - Record metrics

Results saved as pickle in the same format as the quantum script,
ready to be loaded by plot_learning_curve_comparison.py.
"""

import os
import sys
import time
import pickle
import numpy as np
from itertools import combinations
from collections import defaultdict

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm
from torch_geometric.data import Data, DataLoader

import networkx as nx
from grakel.kernels import WeisfeilerLehman
from grakel import Graph as GrakelGraph




def load_data(nx_pkl_path, classes, max_per_class, max_test_per_class):
    with open(nx_pkl_path, "rb") as f:
        data = pickle.load(f)

    G_all   = data["G"]
    y_all   = data["y"]
    pos_all = data["G_positions"]
    G_test  = data["G_test"]
    y_test  = data["y_test"]
    pos_test = data["G_test_positions"]

    # ── pool (training) ──
    train_idx = []
    for c in classes:
        idx = np.where(y_all == c)[0][:max_per_class]
        train_idx.extend(idx)
    G_train = [G_all[i] for i in train_idx]
    y_train = y_all[train_idx]
    pos_train = [pos_all[i] for i in train_idx]

    # ── fixed test ──
    test_idx = []
    for c in classes:
        idx = np.where(y_test == c)[0][:max_test_per_class]
        test_idx.extend(idx)
    G_test_sel  = [G_test[i]    for i in test_idx]
    y_test_sel  = y_test[test_idx]
    pos_test_sel = [pos_test[i] for i in test_idx]

    return (G_train, np.array(y_train), pos_train,
            G_test_sel, np.array(y_test_sel), pos_test_sel)




def _pos_labels(pos_dict, grid_size=8):
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


def _angle_labels(G_nx, pos_dict, n_dev=4, n_dir=6):
    dev_bins = np.linspace(0, np.pi, n_dev + 1)
    out = {}
    for node in G_nx.nodes():
        nbrs = list(G_nx.neighbors(node))
        if len(nbrs) <= 1:
            out[node] = 0
            continue
        angles = []
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                p0 = np.array(pos_dict.get(node,      [0, 0]))
                pi = np.array(pos_dict.get(nbrs[i],   [0, 0]))
                pj = np.array(pos_dict.get(nbrs[j],   [0, 0]))
                v1, v2 = pi - p0, pj - p0
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                angles.append(np.arccos(np.clip(np.dot(v1/n1, v2/n2), -1, 1)))
        if not angles:
            out[node] = 0
            continue
        dev = abs(np.pi - max(angles))
        out[node] = max(0, min(np.digitize(dev, dev_bins) - 1, n_dev - 1)) * n_dir
    return out


def _build_grakel(G_nx, labels):
    adj = {i: list(G_nx.neighbors(i)) for i in range(G_nx.number_of_nodes())}
    return GrakelGraph(adj, node_labels=labels)


def build_grakel_set(G_list, pos_list, label_type="none", grid_size=8, n_dev=4):
    out = []
    for G, pos in zip(G_list, pos_list):
        pd = {i: (pos[i][0], pos[i][1]) for i in range(G.number_of_nodes())}
        if label_type == "none":
            lbl = {i: 1 for i in range(G.number_of_nodes())}
        elif label_type == "position":
            lbl = _pos_labels(pd, grid_size)
        elif label_type == "angle":
            lbl = _angle_labels(G, pd, n_dev)
        elif label_type == "combined":
            pl = _pos_labels(pd, grid_size)
            al = _angle_labels(G, pd, n_dev)
            lbl = {n: pl.get(n, 0) * 100 + al.get(n, 0)
                   for n in set(pl) | set(al)}
        else:
            lbl = {i: 1 for i in range(G.number_of_nodes())}
        out.append(_build_grakel(G, lbl))
    return out




def norm_kernel(K_train, K_test=None):
    eps = 1e-10
    d   = np.sqrt(np.diag(K_train) + eps)
    Kn  = K_train / np.outer(d, d)
    if K_test is None:
        return Kn
    return Kn, K_test


def _wl_cv_best(G_tr, y_tr, n_iter_vals=(2,3,4,5), C_vals=(0.1,1,10,100), cv=3):
    """Return best (n_iter, C) via stratified CV on training set."""
    skf = StratifiedKFold(n_splits=min(cv, len(np.unique(y_tr))), shuffle=True, random_state=42)
    best_acc, best_n, best_C = 0, 3, 1
    for n_iter in n_iter_vals:
        for C in C_vals:
            fold_accs = []
            for tr_i, va_i in skf.split(G_tr, y_tr):
                G_t = [G_tr[i] for i in tr_i]
                G_v = [G_tr[i] for i in va_i]
                wl = WeisfeilerLehman(n_iter=n_iter)
                Kt = wl.fit_transform(G_t)
                Kv = wl.transform(G_v)
                Ktn, Kvn = norm_kernel(Kt, Kv)
                svm = SVC(C=C, kernel="precomputed")
                svm.fit(Ktn, y_tr[tr_i])
                fold_accs.append(accuracy_score(y_tr[va_i], svm.predict(Kvn)))
            if np.mean(fold_accs) > best_acc:
                best_acc, best_n, best_C = np.mean(fold_accs), n_iter, C
    return best_n, best_C


def train_wl_svm(G_tr, y_tr, G_te, label_type, grid_size=8, n_dev=4):
    """Train WL+SVM and return predictions on G_te."""
    best_n, best_C = _wl_cv_best(G_tr, y_tr)
    wl = WeisfeilerLehman(n_iter=best_n)
    Ktr = wl.fit_transform(G_tr)
    Kte = wl.transform(G_te)
    Ktrn, Kten = norm_kernel(Ktr, Kte)
    svm = SVC(C=best_C, kernel="precomputed")
    svm.fit(Ktrn, y_tr)
    return svm.predict(Kten)


def train_ovo_wl(G_tr, y_tr, G_te, classes):
    """One-vs-one WL per pair, returns predictions on G_te."""
    pairs = list(combinations(classes, 2))
    pair_classifiers = {}

    for c1, c2 in pairs:
        mask = np.isin(y_tr, [c1, c2])
        G_p  = [G_tr[i] for i in range(len(G_tr)) if mask[i]]
        y_p  = y_tr[mask]
        if len(np.unique(y_p)) < 2:
            continue
        best_n, best_C = _wl_cv_best(G_p, y_p,
                                     n_iter_vals=(1,2,3,4,5),
                                     C_vals=(0.1,1,10,100),
                                     cv=min(3, len(np.unique(y_p))))
        wl = WeisfeilerLehman(n_iter=best_n)
        Ktr = wl.fit_transform(G_p)
        d   = np.sqrt(np.diag(Ktr) + 1e-10)
        Ktrn = Ktr / np.outer(d, d)
        svm  = SVC(C=best_C, kernel="precomputed")
        svm.fit(Ktrn, y_p)
        pair_classifiers[(c1, c2)] = {"svm": svm, "wl": wl, "d_train": d}

    # predict
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    votes = np.zeros((len(G_te), len(classes)))
    for (c1, c2), info in pair_classifiers.items():
        Kte  = info["wl"].transform(G_te)
        wl2  = WeisfeilerLehman(n_iter=info["wl"].n_iter)
        Kten = Kte 
        pred = info["svm"].predict(Kten)
        for i, p in enumerate(pred):
            votes[i, cls_to_idx[p]] += 1
    return np.array([classes[np.argmax(v)] for v in votes])



def nx_to_pyg(G_nx, pos, label):
    n   = G_nx.number_of_nodes()
    pa  = np.array([pos[i] for i in range(n)], dtype=np.float32)
    if pa.size > 0 and pa.max() > pa.min():
        pa = (pa - pa.min()) / (pa.max() - pa.min() + 1e-8)
    x   = torch.tensor(pa, dtype=torch.float)
    deg = torch.tensor([G_nx.degree(i) for i in range(n)], dtype=torch.float).unsqueeze(1)
    if deg.max() > 0:
        deg = deg / (deg.max() + 1e-8)
    x   = torch.cat([x, deg], dim=1)
    edges = list(G_nx.edges()) or [(0, 0)]
    ei  = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei, y=torch.tensor([label], dtype=torch.long))


class OptimizedGCN(nn.Module):
    def __init__(self, in_ch, hid, out_ch, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid);  self.bn1 = BatchNorm(hid)
        self.conv2 = GCNConv(hid, hid);    self.bn2 = BatchNorm(hid)
        self.lin   = nn.Linear(hid * 2, out_ch)
        self.drop  = dropout
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, self.drop, self.training)
        x = F.relu(self.bn2(self.conv2(x, ei)))
        x = torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=1)
        return F.log_softmax(self.lin(x), dim=1)


class OptimizedGAT(nn.Module):
    def __init__(self, in_ch, hid, out_ch, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_ch, hid, heads=heads, dropout=dropout)
        self.bn1  = BatchNorm(hid * heads)
        self.gat2 = GATConv(hid * heads, hid * 2, heads=1, dropout=dropout)
        self.bn2  = BatchNorm(hid * 2)
        self.lin  = nn.Linear(hid * 2, out_ch)
        self.drop = dropout
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = F.elu(self.bn1(self.gat1(x, ei)))
        x = F.dropout(x, self.drop, self.training)
        x = F.elu(self.bn2(self.gat2(x, ei)))
        return F.log_softmax(self.lin(global_mean_pool(x, b)), dim=1)


def train_gnn_model(model, G_tr, pos_tr, y_tr, G_te, pos_te, y_te,
                    epochs=80, lr=0.01, patience=15, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    def to_pyg(G_list, pos_list, y_list):
        return [nx_to_pyg(G, p, lbl)
                for G, p, lbl in zip(G_list, pos_list, y_list)
                if G.number_of_nodes() >= 2]

    # val split (20% of training)
    n = len(G_tr)
    val_n = max(1, int(n * 0.2))
    rng   = np.random.default_rng(42)
    val_i = rng.choice(n, val_n, replace=False)
    tr_i  = np.setdiff1d(np.arange(n), val_i)

    tr_data = to_pyg([G_tr[i] for i in tr_i],  [pos_tr[i] for i in tr_i],  y_tr[tr_i])
    va_data = to_pyg([G_tr[i] for i in val_i], [pos_tr[i] for i in val_i], y_tr[val_i])
    te_data = to_pyg(G_te, pos_te, y_te)

    tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=batch_size)
    te_loader = DataLoader(te_data, batch_size=batch_size)

    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    best_val, patience_cnt, best_state = 0, 0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for d in tr_loader:
            d = d.to(device)
            opt.zero_grad()
            loss = F.nll_loss(model(d), d.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for d in va_loader:
                d = d.to(device)
                pred = model(d).argmax(1)
                correct += (pred == d.y).sum().item()
                total   += d.y.size(0)
        val_acc = correct / total if total > 0 else 0
        sched.step(total_loss / len(tr_loader))

        if val_acc > best_val:
            best_val = val_acc
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for d in te_loader:
            d = d.to(device)
            preds.extend(model(d).argmax(1).cpu().numpy())
            labels.extend(d.y.cpu().numpy())

    return np.array(preds), np.array(labels)




def _metrics(y_true, y_pred):
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted":     float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }




def _aggregate(per_size_results, train_sizes):
    scalar_keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    aggregated  = {}
    for size in train_sizes:
        seed_list = per_size_results[size]
        agg = {"n_seeds": len(seed_list)}
        for k in scalar_keys:
            vals = [s[k] for s in seed_list]
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        cms = np.array([s["confusion_matrix"] for s in seed_list])
        agg["mean_confusion_matrix"] = np.mean(cms, axis=0).tolist()
        agg["std_confusion_matrix"]  = (np.std(cms, axis=0, ddof=1).tolist()
                                        if len(cms) > 1
                                        else np.zeros_like(cms[0]).tolist())
        aggregated[size] = agg
    return aggregated




def run_classical_learning_curve(
    nx_pkl_path,
    classes=list(range(10)),
    train_sizes=[10, 15, 20, 25, 30, 35, 40],
    max_per_class=40,
    max_test_per_class=15,
    n_seeds=5,
    # which models to run
    run_wl=True,
    run_ovo=True,
    run_gnn=False,       # off by default (slow)
    results_save_path="results/classical_learning_curve.pkl",
    summary_save_path="results/classical_learning_curve_summary.txt",
):
    t_start = time.time()
    classes = list(classes)

    print("\n" + "=" * 70)
    print("CLASSICAL MODELS  —  LEARNING CURVE EXPERIMENT")
    print("=" * 70)
    print(f"  Classes     : {classes}")
    print(f"  Train sizes : {train_sizes}")
    print(f"  Seeds       : {n_seeds}")
    print(f"  Run WL+SVM  : {run_wl}   Run OvO: {run_ovo}   Run GNN: {run_gnn}")
    print("=" * 70)

   
    G_train, y_train, pos_train, G_test, y_test, pos_test = load_data(
        nx_pkl_path, classes, max_per_class, max_test_per_class
    )
    print(f"\nPool: {len(G_train)} train  |  Fixed test: {len(G_test)}")

    # Pre-build all grakel sets for the full pool (indexed later)
    if run_wl or run_ovo:
        print("Building grakel label sets for full pool …")
        pool_gk = {
            "none":     build_grakel_set(G_train, pos_train, "none"),
            "pos8":     build_grakel_set(G_train, pos_train, "position", grid_size=8),
            "angle":    build_grakel_set(G_train, pos_train, "angle"),
            "combined": build_grakel_set(G_train, pos_train, "combined"),
        }
        test_gk = {
            "none":     build_grakel_set(G_test, pos_test, "none"),
            "pos8":     build_grakel_set(G_test, pos_test, "position", grid_size=8),
            "angle":    build_grakel_set(G_test, pos_test, "angle"),
            "combined": build_grakel_set(G_test, pos_test, "combined"),
        }

    # per-class pool indices
    class_indices = {c: np.where(y_train == c)[0].tolist() for c in classes}

    # model names we will track
    model_names = []
    if run_wl:
        model_names += ["WL_none", "WL_pos8", "WL_angle", "WL_combined"]
    if run_ovo:
        model_names += ["OvO_pos8"]
    if run_gnn:
        model_names += ["GCN", "GAT"]

   
    per_size = {name: {sz: [] for sz in train_sizes} for name in model_names}

    for size in train_sizes:
        print(f"\n{'─'*60}")
        print(f"  Train size = {size}/class  ({size*len(classes)} total)")
        print(f"{'─'*60}")

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            sel_idx = []
            for c in classes:
                chosen = rng.choice(class_indices[c], size=size, replace=False)
                sel_idx.extend(chosen.tolist())
            sel_idx  = np.array(sel_idx)
            y_sub    = y_train[sel_idx]

            # ── WL + SVM ──
            if run_wl:
                for ltype, key in [("none","WL_none"), ("pos8","WL_pos8"),
                                   ("angle","WL_angle"), ("combined","WL_combined")]:
                    t0 = time.time()
                    G_sub_gk = [pool_gk[ltype][i] for i in sel_idx]
                    y_pred   = train_wl_svm(G_sub_gk, y_sub,
                                            test_gk[ltype], ltype)
                    m = _metrics(y_test, y_pred)
                    m.update({"seed": seed, "train_size_pc": size,
                               "training_time_s": time.time() - t0})
                    per_size[key][size].append(m)
                    print(f"  {key:15s}  size={size}  seed={seed}  "
                          f"acc={m['accuracy']:.4f}  ({m['training_time_s']:.1f}s)")

            # ── OvO ──
            if run_ovo:
                t0     = time.time()
                G_sub  = [pool_gk["pos8"][i] for i in sel_idx]
                y_pred = train_ovo_wl(G_sub, y_sub, test_gk["pos8"], classes)
                m = _metrics(y_test, y_pred)
                m.update({"seed": seed, "train_size_pc": size,
                           "training_time_s": time.time() - t0})
                per_size["OvO_pos8"][size].append(m)
                print(f"  {'OvO_pos8':15s}  size={size}  seed={seed}  "
                      f"acc={m['accuracy']:.4f}  ({m['training_time_s']:.1f}s)")

            # ── GNN ──
            if run_gnn:
                G_sub_nx  = [G_train[i]   for i in sel_idx]
                pos_sub   = [pos_train[i] for i in sel_idx]
                n_classes = len(classes)

                for gnn_name, model_cls in [("GCN", OptimizedGCN),
                                            ("GAT", OptimizedGAT)]:
                    t0    = time.time()
                    model = (model_cls(3, 32, n_classes, dropout=0.3)
                             if gnn_name == "GCN"
                             else model_cls(3, 32, n_classes, heads=4, dropout=0.3))
                    y_pred, y_true = train_gnn_model(
                        model, G_sub_nx, pos_sub, y_sub,
                        G_test, pos_test, y_test
                    )
                    m = _metrics(y_true, y_pred)
                    m.update({"seed": seed, "train_size_pc": size,
                               "training_time_s": time.time() - t0})
                    per_size[gnn_name][size].append(m)
                    print(f"  {gnn_name:15s}  size={size}  seed={seed}  "
                          f"acc={m['accuracy']:.4f}  ({m['training_time_s']:.1f}s)")

    
    print("\n" + "=" * 70)
    print("AGGREGATING")
    print("=" * 70)

    aggregated = {}
    for name in model_names:
        aggregated[name] = _aggregate(per_size[name], train_sizes)
        print(f"\n  {name}")
        for sz in train_sizes:
            a = aggregated[name][sz]
            print(f"    size={sz:3d}  acc {a['mean_accuracy']:.4f} ± {a['std_accuracy']:.4f}"
                  f"  f1 {a['mean_f1_macro']:.4f} ± {a['std_f1_macro']:.4f}")

    meta = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        classes=classes,
        train_sizes=train_sizes,
        max_per_class=max_per_class,
        max_test_per_class=max_test_per_class,
        n_seeds=n_seeds,
        model_names=model_names,
        total_time_s=time.time() - t_start,
    )

    results = dict(
        train_sizes=train_sizes,
        per_size=per_size,
        aggregated=aggregated,
        meta=meta,
    )

    if results_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(results_save_path)), exist_ok=True)
        with open(results_save_path, "wb") as f:
            pickle.dump(results, f, protocol=4)
        print(f"\nResults saved → {results_save_path}")

    if summary_save_path:
        _save_summary(results, summary_save_path)
        print(f"Summary saved → {summary_save_path}")

    print(f"\nTotal time: {meta['total_time_s']:.1f}s")
    return results


def _save_summary(results, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta  = results["meta"]
    agg   = results["aggregated"]
    sizes = results["train_sizes"]
    names = meta["model_names"]
    col   = 12

    with open(path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("CLASSICAL MODELS  —  LEARNING CURVE SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Timestamp : {meta['timestamp']}\n")
        f.write(f"Total time: {meta['total_time_s']:.1f}s\n\n")
        f.write(f"Classes   : {meta['classes']}\n")
        f.write(f"Sizes     : {meta['train_sizes']}\n")
        f.write(f"Seeds     : {meta['n_seeds']}\n\n")

        for name in names:
            f.write(f"\n── {name} ──\n")
            header = (f"{'size':>{col}}"
                      f"{'acc mean':>{col}}"
                      f"{'acc std':>{col}}"
                      f"{'f1 mean':>{col}}"
                      f"{'f1 std':>{col}}\n")
            f.write(header)
            f.write("-" * (col * 5) + "\n")
            for sz in sizes:
                a = agg[name][sz]
                f.write(f"{sz:>{col}}"
                        f"{a['mean_accuracy']:>{col}.4f}"
                        f"{a['std_accuracy']:>{col}.4f}"
                        f"{a['mean_f1_macro']:>{col}.4f}"
                        f"{a['std_f1_macro']:>{col}.4f}\n")

        f.write("\n" + "=" * 72 + "\nEND\n" + "=" * 72 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    date       = time.strftime("%d-%m-%Y")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir    = os.path.join(script_dir, "results")

    run_classical_learning_curve(
        nx_pkl_path=os.path.join(script_dir, "..", "Data",
                                 "mnist_graphs_nx_filtered.pkl"),
        classes=list(range(10)),
        train_sizes=[10, 15, 20, 25, 30, 35, 40],
        max_per_class=80,
        max_test_per_class=15,
        n_seeds=5,
        run_wl=True,
        run_ovo=True,
        run_gnn=True,    # set True if you want GNNs (slow)
        results_save_path=os.path.join(res_dir, f"classical_lc_{date}.pkl"),
        summary_save_path=os.path.join(res_dir, f"classical_lc_summary_{date}.txt"),
    )
