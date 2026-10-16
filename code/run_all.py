"""
Run the whole pipeline in order:
  1. experiments.py        all experiments + all five metrics  (slow: ~5 min MUTAG, ~45 min PROTEINS)
  2. significance.py       corrected t-test / Wilcoxon vs the standard kernel
  3. make_figures.py       figures 1-6
  4. make_wl_example.py    figure 7 (worked WL example)
  5. make_metric_figures.py figures 8-10 + metrics_table.csv

Usage:  python run_all.py                 (both datasets)
        python run_all.py MUTAG           (one dataset; later steps need both result files)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
datasets = sys.argv[1:] or ["MUTAG", "PROTEINS"]
steps = [["experiments.py", *datasets], ["significance.py"], ["make_figures.py"],
         ["make_wl_example.py"], ["make_metric_figures.py"]]

start = time.perf_counter()
for step in steps:
    print(f"\n########## {' '.join(step)} ##########", flush=True)
    t = time.perf_counter()
    subprocess.run([sys.executable, *step], cwd=HERE, check=True)
    print(f"---------- done in {time.perf_counter() - t:.0f}s", flush=True)
print(f"\nALL DONE in {(time.perf_counter() - start) / 60:.1f} min")
