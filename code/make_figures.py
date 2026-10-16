"""
Create all figures and the results table from ../results/*.json.
Output: ../figures/*.png and ../results/summary_table.csv
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

DATASETS = ["MUTAG", "PROTEINS"]
R = {d: json.load(open(os.path.join(RES, f"results_{d}.json"))) for d in DATASETS}

# fixed colour per variant (same colour = same variant in every figure)
COLORS = {
    "WL (raw)": "#8a8a85",
    "WL (normalised)": "#2a78d6",
    "WL edge-label": "#1baf7a",
    "WL weighted (lambda tuned)": "#eda100",
}
MARKERS = {"WL (raw)": "o", "WL (normalised)": "s", "WL edge-label": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3dc"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print("saved", name)


# 1. Accuracy vs number of WL iterations h --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, d in zip(axes, DATASETS):
    for v, curve in R[d]["h_curve"].items():
        h = [c["h"] for c in curve]
        acc = np.array([c["acc_mean"] for c in curve])
        ax.plot(h, acc, lw=2, marker=MARKERS[v], ms=6, color=COLORS[v], label=v)
    ax.set_title(d, color=INK, fontweight="bold")
    ax.set_xlabel("WL iterations h")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(0, 11))
axes[0].legend(loc="lower right")
fig.suptitle("Effect of the number of WL iterations (10-fold CV x 3)", color=INK)
save(fig, "fig1_accuracy_vs_h.png")

# 2. Final comparison bar chart -------------------------------------------
variants = list(COLORS)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, d in zip(axes, DATASETS):
    s = R[d]["summary"]
    names = [v for v in variants if v in s]
    acc = [s[v]["acc_mean"] for v in names]
    err = [s[v]["acc_std"] for v in names]
    x = np.arange(len(names))
    ax.bar(x, acc, yerr=err, color=[COLORS[v] for v in names], width=0.7,
           edgecolor="white", linewidth=2, capsize=3, error_kw={"ecolor": MUTED, "lw": 1})
    for xi, a in zip(x, acc):
        ax.text(xi, a + 0.4, f"{a:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (lambda tuned)", "\n(λ tuned)").replace(" ", "\n", 1)
                        if len(n) > 10 else n for n in names], fontsize=8)
    lo = min(acc) - max(err) - 3
    ax.set_ylim(lo, max(acc) + max(err) + 2)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(d, color=INK, fontweight="bold")
    ax.grid(axis="x", visible=False)
fig.suptitle("Variant comparison with nested model selection (mean ± std over 30 folds)", color=INK)
save(fig, "fig2_variant_comparison.png")

# 3. Iteration weighting lambda -------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for d, c in zip(DATASETS, ["#2a78d6", "#eb6834"]):
    lc = R[d]["lambda_curve"]
    ax.plot([x["lambda"] for x in lc], [x["acc_mean"] for x in lc], lw=2, marker="o", ms=7,
            color=c, label=d)
ax.axvline(1.0, color=MUTED, ls="--", lw=1)
ax.text(1.02, ax.get_ylim()[0] + 0.5, "λ = 1 (original WL)", color=MUTED, fontsize=8)
ax.set_xlabel("λ  (weight of iteration i is λ^i)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Iteration weighting", color=INK, fontweight="bold")
ax.legend()
save(fig, "fig3_lambda.png")

# 4. Runtime and number of distinct WL labels ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, d in zip(axes, DATASETS):
    rt = R[d]["runtime"]
    h = np.arange(len(rt["WL"]))
    for key, color, m in [("WL", "#2a78d6", "s"), ("WL edge-label", "#1baf7a", "D")]:
        if rt.get(key):
            ax.plot(h, rt[key], lw=2, marker=m, ms=6, color=color, label=key)
    ax.set_xlabel("WL iterations h")
    ax.set_ylabel("Kernel computation time (s)")
    ax.set_xticks(range(0, 11))
    ax.set_title(d, color=INK, fontweight="bold")
axes[0].legend()
fig.suptitle("Runtime grows linearly with h, as predicted by O(N·h·m)", color=INK)
save(fig, "fig4_runtime.png")

fig, ax = plt.subplots(figsize=(6, 4))
for d, c in zip(DATASETS, ["#2a78d6", "#eb6834"]):
    n = R[d]["n_labels_per_iter"]
    ax.plot(range(len(n)), n, lw=2, marker="o", ms=6, color=c, label=d)
ax.set_yscale("log")
ax.set_xlabel("WL iterations h")
ax.set_ylabel("Distinct labels (features), log scale")
ax.set_title("Feature space growth", color=INK, fontweight="bold")
ax.legend()
save(fig, "fig5_feature_growth.png")

# 5. Kernel matrix heatmaps -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, d in zip(axes, DATASETS):
    K = np.load(os.path.join(RES, f"kernel_{d}.npy"))
    lab = np.array(R[d]["kernel_order_labels"])
    im = ax.imshow(K, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    b = np.argmax(lab == 1)
    ax.axhline(b - 0.5, color="#eb6834", lw=1.2)
    ax.axvline(b - 0.5, color="#eb6834", lw=1.2)
    ax.set_title(f"{d}: normalised WL kernel (h = 3)", color=INK, fontsize=10)
    ax.set_xlabel("graphs (sorted by class)")
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8, label="similarity")
save(fig, "fig6_kernel_heatmap.png")

# Summary table -----------------------------------------------------------
with open(os.path.join(RES, "summary_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Dataset", "Variant", "Accuracy mean", "Accuracy std", "Macro-F1 mean", "Macro-F1 std"])
    for d in DATASETS:
        for v, s in R[d]["summary"].items():
            w.writerow([d, v, f"{s['acc_mean']:.2f}", f"{s['acc_std']:.2f}",
                        f"{s['f1_mean']:.2f}", f"{s['f1_std']:.2f}"])
print("saved summary_table.csv")
