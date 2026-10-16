"""
Statistical significance of the differences between WL variants.

Every variant was evaluated on exactly the same 30 folds (10-fold CV x 3 seeds),
so the fold accuracies are paired. Two tests are reported against the
standard baseline (original WL subtree kernel):

1. Corrected resampled t-test (Nadeau & Bengio, 2003). Cross-validation folds
   share training data, so a plain paired t-test underestimates the variance
   and finds "significant" differences too easily. The correction inflates the
   variance by (1/n + n_test/n_train).
2. Wilcoxon signed-rank test: non-parametric, no normality assumption.

Output: ../results/significance.csv
"""
import csv
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
BASELINE = "WL (raw)"      # standard WL subtree kernel (Shervashidze et al. 2011)
TEST_TRAIN_RATIO = 1 / 9        # 10-fold CV: 1 part test, 9 parts train
ALPHA = 0.05


def corrected_ttest(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / n + TEST_TRAIN_RATIO) * var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return t, p


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return 1.0
    return stats.wilcoxon(a, b, zero_method="zsplit").pvalue


rows = []
for name in ["MUTAG", "PROTEINS"]:
    summary = json.load(open(os.path.join(RES, f"results_{name}.json")))["summary"]
    base = summary[BASELINE]["fold_acc"]
    print(f"\n===== {name}  (baseline: {BASELINE}, {np.mean(base):.2f}%) =====")
    print(f"{'Variant':28s} {'diff':>7s} {'t-test p':>9s} {'Wilcoxon p':>11s}  verdict")
    for v, s in summary.items():
        if v == BASELINE:
            continue
        acc = s["fold_acc"]
        diff = np.mean(acc) - np.mean(base)
        t, p_t = corrected_ttest(acc, base)
        p_w = wilcoxon(acc, base)
        verdict = "significant" if (p_t < ALPHA and p_w < ALPHA) else "not significant"
        print(f"{v:28s} {diff:+7.2f} {p_t:9.3f} {p_w:11.3f}  {verdict}")
        rows.append([name, v, f"{np.mean(acc):.2f}", f"{diff:+.2f}", f"{t:.2f}",
                     f"{p_t:.3f}", f"{p_w:.3f}", verdict])

with open(os.path.join(RES, "significance.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Dataset", "Variant", "Accuracy", "Diff vs baseline", "Corrected t",
                "p (corrected t-test)", "p (Wilcoxon)", "Verdict (alpha=0.05)"])
    w.writerows(rows)
print("\nsaved results/significance.csv")
