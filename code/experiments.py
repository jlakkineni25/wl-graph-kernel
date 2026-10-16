"""
Run all WL-kernel experiments and save results to ../results.

Protocol: 10-fold stratified cross-validation, repeated 3 times (seeds 0,1,2).
Inside every outer training split, hyper-parameters (SVM C, and where relevant
h / lambda) are chosen by an inner 5-fold CV, so test folds are never
used for model selection (nested CV).
"""
import json
import warnings
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from wl_kernels import (load_tudataset, wl_relabel, feature_matrices,
                        per_iteration_grams, wl_kernel, normalize)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

CS = [10.0 ** k for k in range(-3, 4)]
SEEDS = [0, 1, 2]
H_MAX = 10
H_SELECT = list(range(0, 8))          # h grid used for nested selection
LAMBDAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
# cap on libsvm iterations: unnormalised kernels with large C otherwise run for hours
MAX_ITER = 200_000
METRICS = ["acc", "precision", "recall", "f1", "roc_auc"]


# --------------------------------------------------------------------------
# Nested cross-validation with precomputed kernels
# --------------------------------------------------------------------------
def _inner_score(K, y, tr, C, inner_splits):
    accs = []
    for a, b in inner_splits:
        clf = SVC(kernel="precomputed", C=C, max_iter=MAX_ITER)
        clf.fit(K[np.ix_(tr[a], tr[a])], y[tr[a]])
        accs.append(accuracy_score(y[tr[b]], clf.predict(K[np.ix_(tr[b], tr[a])])))
    return np.mean(accs)


def _outer_fold(kernels, y, tr, te, seed):
    """Inner CV picks (hyper-parameter key, C); refit on the training part, score the test part."""
    inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]))
    best, best_score = None, -1
    for key, K in kernels.items():
        for C in CS:
            s = _inner_score(K, y, tr, C, inner)
            if s > best_score:
                best, best_score = (key, C), s
    key, C = best
    K = kernels[key]
    clf = SVC(kernel="precomputed", C=C, max_iter=MAX_ITER).fit(K[np.ix_(tr, tr)], y[tr])
    K_test = K[np.ix_(te, tr)]
    pred = clf.predict(K_test)
    score = clf.decision_function(K_test)          # > 0 means class 1, used for ROC-AUC
    return {
        "acc": accuracy_score(y[te], pred),
        "precision": precision_score(y[te], pred, average="macro", zero_division=0),
        "recall": recall_score(y[te], pred, average="macro", zero_division=0),
        "f1": f1_score(y[te], pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y[te], score),
        "confusion": confusion_matrix(y[te], pred, labels=[0, 1]).tolist(),
        "y_true": y[te].tolist(), "score": score.tolist(),
        "key": str(key), "C": C,
    }


def nested_cv(kernels, y, keep_folds=False):
    """
    kernels: dict  hyper-parameter-key -> precomputed Gram matrix.
    Returns mean/std of Accuracy, Precision, Recall, F1 (macro) and ROC-AUC over
    all outer folds. keep_folds=True also stores per-fold labels and SVM scores
    (needed for ROC curves and confusion matrices).
    """
    jobs = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            jobs.append((tr, te, seed))
    out = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_outer_fold)(kernels, y, tr, te, s) for tr, te, s in jobs)
    res = {}
    for m in METRICS:
        v = np.array([o[m] for o in out]) * 100
        res[f"{m}_mean"], res[f"{m}_std"], res[f"fold_{m}"] = float(v.mean()), float(v.std()), v.tolist()
    res["acc_rep_std"] = float(np.array(res["fold_acc"]).reshape(len(SEEDS), -1).mean(axis=1).std())
    res["chosen"] = [o["key"] for o in out]
    res["chosen_C"] = [o["C"] for o in out]
    res["confusion_total"] = np.sum([o["confusion"] for o in out], axis=0).tolist()
    if keep_folds:
        res["folds"] = [{"y_true": o["y_true"], "score": o["score"]} for o in out]
    return res


# --------------------------------------------------------------------------
# Runtime measurement
# --------------------------------------------------------------------------
RUNTIME_REPEATS = 5


def _kernel_time(graphs, h, use_edge_labels):
    """Time the full pipeline graphs -> normalised N x N kernel matrix."""
    t = time.perf_counter()
    it = wl_relabel(graphs, h, use_edge_labels=use_edge_labels)
    normalize(wl_kernel(per_iteration_grams(feature_matrices(it)), h))
    return time.perf_counter() - t


def measure_runtime(graphs, has_edge_labels):
    """
    Kernel computation time for h = 0..H_MAX. Each point is the best of
    RUNTIME_REPEATS runs, which removes operating-system noise (a single
    millisecond-scale timing is easily disturbed by other processes).
    """
    rt = {"WL": [], "WL edge-label": []}
    for h in range(H_MAX + 1):
        rt["WL"].append(min(_kernel_time(graphs, h, False) for _ in range(RUNTIME_REPEATS)))
        if has_edge_labels:
            rt["WL edge-label"].append(min(_kernel_time(graphs, h, True) for _ in range(RUNTIME_REPEATS)))
    return rt


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def run(name):
    print(f"\n===== {name} =====", flush=True)
    graphs, y = load_tudataset(name, os.path.join(ROOT, "data"))
    has_edge_labels = any(len(np.unique(np.concatenate(g.edge_labels))) > 1
                          for g in graphs if g.n > 1 and len(np.concatenate(g.edge_labels)))
    results = {"dataset": name, "n_graphs": len(graphs),
               "avg_nodes": float(np.mean([g.n for g in graphs])),
               "avg_edges": float(np.mean([sum(len(a) for a in g.neighbors) / 2 for g in graphs])),
               "class_counts": np.bincount(y).tolist()}

    # ---- features + per-iteration Gram matrices -------------------------
    t0 = time.perf_counter()
    it = wl_relabel(graphs, H_MAX)
    grams = per_iteration_grams(feature_matrices(it))
    results["n_labels_per_iter"] = [int(max(l.max() for l in L) + 1) for L in it]
    if has_edge_labels:
        it_e = wl_relabel(graphs, H_MAX, use_edge_labels=True)
        grams_e = per_iteration_grams(feature_matrices(it_e))
        results["n_labels_per_iter_edge"] = [int(max(l.max() for l in L) + 1) for L in it_e]
    print(f"features built in {time.perf_counter() - t0:.1f}s", flush=True)

    builders = {
        "WL (raw)": lambda h: wl_kernel(grams, h),
        "WL (normalised)": lambda h: normalize(wl_kernel(grams, h)),
    }
    if has_edge_labels:
        builders["WL edge-label"] = lambda h: normalize(wl_kernel(grams_e, h))

    # ---- E1: accuracy vs number of iterations h ---------------------------
    results["h_curve"] = {}
    for vname, build in builders.items():
        curve = []
        for h in range(H_MAX + 1):
            r = nested_cv({h: build(h)}, y)
            curve.append({"h": h, "acc_std": r["acc_std"],
                          **{f"{m}_mean": r[f"{m}_mean"] for m in METRICS}})
            print(f"  [h-curve] {vname:16s} h={h:2d} acc={r['acc_mean']:.2f}±{r['acc_std']:.2f}", flush=True)
        results["h_curve"][vname] = curve

    # ---- E2: iteration weighting lambda -----------------------------------
    results["lambda_curve"] = []
    for lam in LAMBDAS:
        r = nested_cv({h: normalize(wl_kernel(grams, h, lam)) for h in H_SELECT}, y)
        results["lambda_curve"].append({"lambda": lam, **{k: r[k] for k in ("acc_mean", "acc_std", "f1_mean", "f1_std")},
                                        "chosen": r["chosen"]})
        print(f"  [lambda] {lam:4.2f} acc={r['acc_mean']:.2f}±{r['acc_std']:.2f}", flush=True)

    # ---- E3: final comparison with full nested model selection ------------
    summary = {}
    for vname, build in builders.items():
        summary[vname] = nested_cv({h: build(h) for h in H_SELECT}, y, keep_folds=True)
    summary["WL weighted (lambda tuned)"] = nested_cv(
        {(h, lam): normalize(wl_kernel(grams, h, lam)) for h in H_SELECT for lam in LAMBDAS}, y, keep_folds=True)
    for k, r in summary.items():
        print(f"  [summary] {k:28s} acc={r['acc_mean']:.2f}±{r['acc_std']:.2f}  prec={r['precision_mean']:.2f}"
              f"  rec={r['recall_mean']:.2f}  f1={r['f1_mean']:.2f}  auc={r['roc_auc_mean']:.2f}", flush=True)
    results["summary"] = summary

    # ---- E4: runtime ------------------------------------------------------
    results["runtime"] = measure_runtime(graphs, has_edge_labels)

    # ---- kernel matrix for visualisation (WL normalised, h=3) -------------
    order = np.argsort(y, kind="stable")
    np.save(os.path.join(RES, f"kernel_{name}.npy"), normalize(wl_kernel(grams, 3))[np.ix_(order, order)])
    results["kernel_order_labels"] = y[order].tolist()

    with open(os.path.join(RES, f"results_{name}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved results_{name}.json", flush=True)


def runtime_only(name):
    """Re-measure only the runtime and update the existing results file (takes seconds)."""
    path = os.path.join(RES, f"results_{name}.json")
    results = json.load(open(path))
    graphs, _ = load_tudataset(name, os.path.join(ROOT, "data"))
    results["runtime"] = measure_runtime(graphs, has_edge_labels=(name == "MUTAG"))
    json.dump(results, open(path, "w"), indent=1)
    print(f"{name}: runtime updated", {k: [round(t, 3) for t in v] for k, v in results["runtime"].items()})


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for ds in (args or ["MUTAG", "PROTEINS"]):
        runtime_only(ds) if "--runtime-only" in sys.argv else run(ds)
