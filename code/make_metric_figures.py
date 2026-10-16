"""
Figures and tables for the standard vs modified comparison on all five metrics.
Reads the "summary" section of ../results/results_<dataset>.json (written by experiments.py).
Output: ../figures/fig8_metrics_comparison.png, fig9_roc_curves.png,
        fig10_confusion_matrices.png and ../results/metrics_table.csv
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3dc"

# same style as make_figures.py
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})

DATASETS = ["MUTAG", "PROTEINS"]
METRICS = ["acc", "precision", "recall", "f1", "roc_auc"]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
STANDARD = "Standard WL subtree"
COLORS = {
    STANDARD: "#8a8a85",
    "M1: Normalised WL": "#2a78d6",
    "M2: Iteration-weighted WL": "#eda100",
    "M3: Edge-label WL": "#1baf7a",
}
# experiments.py variant name -> display name used in the comparison
DISPLAY = {
    "WL (raw)": STANDARD,
    "WL (normalised)": "M1: Normalised WL",
    "WL weighted (lambda tuned)": "M2: Iteration-weighted WL",
    "WL edge-label": "M3: Edge-label WL",
}


def corrected_ttest(a, b, test_train_ratio=1 / 9):
    """Nadeau & Bengio (2003) corrected resampled t-test over CV folds."""
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0
    t = d.mean() / np.sqrt((1 / len(d) + test_train_ratio) * var)
    return float(2 * stats.t.sf(abs(t), df=len(d) - 1))


R = {}
for d in DATASETS:
    summary = json.load(open(os.path.join(RES, f"results_{d}.json")))["summary"]
    R[d] = {DISPLAY[k]: {**v, **{m: {"mean": v[f"{m}_mean"], "std": v[f"{m}_std"],
                                    "values": v[f"fold_{m}"]} for m in METRICS}}
            for k, v in summary.items()}
    base = R[d][STANDARD]
    for name, r in R[d].items():
        if name != STANDARD:
            r["p_values"] = {m: corrected_ttest(r[m]["values"], base[m]["values"]) for m in METRICS}

# 1. grouped bars: every metric, every variant -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))
for ax, d in zip(axes, DATASETS):
    names = [v for v in COLORS if v in R[d]]
    width = 0.8 / len(names)
    x = np.arange(len(METRICS))
    for i, v in enumerate(names):
        means = [R[d][v][m]["mean"] for m in METRICS]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, means, width * 0.92,
               color=COLORS[v], label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    lo = min(R[d][v][m]["mean"] for v in names for m in METRICS)
    ax.set_ylim(lo - 4, 100 if d == "MUTAG" else lo + 16)
    ax.set_ylabel("Score (%)")
    ax.set_title(d, color=INK, fontweight="bold")
    ax.grid(axis="x", visible=False)
axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
fig.suptitle("Standard vs modified WL kernels on five metrics (mean over 30 test folds)", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig8_metrics_comparison.png"), dpi=200)
plt.close(fig)
print("saved fig8_metrics_comparison.png")

# 2. mean ROC curves --------------------------------------------------------
grid = np.linspace(0, 1, 201)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, d in zip(axes, DATASETS):
    for v in [v for v in COLORS if v in R[d]]:
        tprs = []
        for f in R[d][v]["folds"]:
            fpr, tpr, _ = roc_curve(f["y_true"], f["score"])
            t = np.interp(grid, fpr, tpr)
            t[0] = 0.0
            tprs.append(t)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        auc = R[d][v]["roc_auc"]["mean"]
        ax.plot(grid, mean_tpr, lw=2.4 if v == STANDARD else 1.8, color=COLORS[v],
                ls="--" if v == STANDARD else "-", label=f"{v}  (AUC {auc:.1f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(d, color=INK, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("Mean ROC curves over 30 test folds", color=INK)
fig.savefig(os.path.join(FIG, "fig9_roc_curves.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig9_roc_curves.png")

# 3. confusion matrices: standard vs best modification (by accuracy) ----------
CLASS_NAMES = {"MUTAG": ["non-mutagenic", "mutagenic"], "PROTEINS": ["class 0", "class 1"]}
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for row, d in enumerate(DATASETS):
    best = max((v for v in R[d] if v != STANDARD), key=lambda v: R[d][v]["acc"]["mean"])
    for col, v in enumerate([STANDARD, best]):
        cm = np.array(R[d][v]["confusion_total"], dtype=float)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax = axes[row, col]
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({int(cm[i, j] / 3)})", ha="center", va="center",
                        color="white" if cm_pct[i, j] > 55 else INK, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES[d]); ax.set_yticklabels(CLASS_NAMES[d], rotation=90, va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{d}: {v}", color=INK, fontsize=10)
        ax.grid(False)
fig.suptitle("Confusion matrices (row %, counts averaged over 3 repetitions)", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig10_confusion_matrices.png"), dpi=200)
plt.close(fig)
print("saved fig10_confusion_matrices.png")

# 4. table with mean ± std and corrected-t p-value vs standard ------------
with open(os.path.join(RES, "metrics_table.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Dataset", "Kernel"] + METRIC_NAMES + [f"p ({m})" for m in METRIC_NAMES])
    for d in DATASETS:
        for v in sorted(R[d], key=lambda v: (v != STANDARD, v)):   # standard first, then M1, M2, M3
            r = R[d][v]
            p = r.get("p_values", {})
            w.writerow([d, v] + [f"{r[m]['mean']:.2f} ± {r[m]['std']:.2f}" for m in METRICS]
                       + [f"{p[m]:.3f}" if m in p else "-" for m in METRICS])
print("saved metrics_table.csv")
