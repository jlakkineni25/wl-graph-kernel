"""
Worked example of one WL relabelling iteration on two toy graphs, with the
resulting feature vectors and kernel value. Uses the same wl_relabel code as
the experiments. Output: ../figures/fig7_wl_example.png
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from wl_kernels import Graph, wl_relabel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

LETTERS = ["A", "B", "C"]
INK, MUTED = "#1f1f1e", "#6b6b66"
# h=0 colours (A, B, C) followed by six distinct colours for the h=1 labels
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a5cd1", "#4a4a46", "#c43d3d", "#0f8c8c"]


def make_graph(labels, edges):
    n = len(labels)
    nbrs = [[] for _ in range(n)]
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)
    return Graph(n, np.array(labels), [np.array(a) for a in nbrs],
                 [np.zeros(len(a), dtype=int) for a in nbrs])


# G: triangle with a tail      H: path A-B-C-A   (same node labels, different structure)
POS = np.array([[0, 1], [1.2, 1], [0.6, 0], [1.8, 0]])
G_EDGES = [(0, 1), (1, 2), (2, 0), (2, 3)]
H_EDGES = [(0, 1), (1, 2), (2, 3)]
G = make_graph([0, 1, 2, 0], G_EDGES)
H = make_graph([0, 1, 2, 0], H_EDGES)
graphs = [G, H]
it = wl_relabel(graphs, 1)

# human-readable signature of each iteration-1 label, e.g. "C | A,A,B"
sig_of = {}
for g, lab0, lab1 in zip(graphs, it[0], it[1]):
    for v in range(g.n):
        nb = sorted(LETTERS[g.node_labels[u]] for u in g.neighbors[v])
        sig_of[int(lab1[v])] = f"{LETTERS[g.node_labels[v]]} | {','.join(nb)}"
NEW = {k: str(k + 1) for k in sig_of}          # display ids 1,2,3,...


def draw(ax, g, edges, labels, text_fn, title):
    for u, v in edges:
        ax.plot(*POS[[u, v]].T, color=MUTED, lw=2, zorder=1)
    for v in range(g.n):
        ax.scatter(*POS[v], s=1300, color=PALETTE[labels[v] % len(PALETTE)],
                   edgecolor="white", linewidth=2.5, zorder=2)
        ax.text(*POS[v], text_fn(v), ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", zorder=3)
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlim(-0.5, 2.3); ax.set_ylim(-0.55, 1.55)
    ax.set_aspect("equal"); ax.axis("off")


fig = plt.figure(figsize=(13, 6.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.45], wspace=0.08, hspace=0.25)
for row, (g, edges, name) in enumerate([(G, G_EDGES, "G"), (H, H_EDGES, "H")]):
    ax0 = fig.add_subplot(gs[row, 0])
    draw(ax0, g, edges, it[0][row], lambda v, g=g: LETTERS[g.node_labels[v]],
         f"Graph {name}, h = 0 (original labels)")
    ax1 = fig.add_subplot(gs[row, 1])
    draw(ax1, g, edges, [l + 3 for l in it[1][row]], lambda v, r=row: NEW[int(it[1][r][v])],
         f"Graph {name}, h = 1 (after one relabel)")
    for v in range(g.n):    # signature under each node
        dy = 0.36 if POS[v, 1] > 0.5 else -0.36     # top row: above, bottom row: below
        ax1.text(POS[v, 0], POS[v, 1] + dy, sig_of[int(it[1][row][v])],
                 ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1))

# right panel: compression table, feature vectors, kernel value
ax = fig.add_subplot(gs[:, 2]); ax.axis("off")
c0 = [Counter(l.tolist()) for l in it[0]]
c1 = [Counter(l.tolist()) for l in it[1]]
n1 = len(sig_of)
phi0 = [[c[i] for i in range(3)] for c in c0]
phi1 = [[c[i] for i in range(n1)] for c in c1]
k0 = int(np.dot(*phi0)); k1 = int(np.dot(*phi1))
kgg = int(np.dot(phi0[0], phi0[0]) + np.dot(phi1[0], phi1[0]))
khh = int(np.dot(phi0[1], phi0[1]) + np.dot(phi1[1], phi1[1]))

lines = ["Step 1  signature = (own label | sorted neighbour labels)",
         "Step 2  compress each new signature to a short id", ""]
lines += [f"    {sig:<10s} ->  {NEW[k]}" for k, sig in sorted(sig_of.items())]
lines += ["", "Step 3  count labels  (feature vector phi)", "",
          "               A B C   " + " ".join(NEW[k] for k in range(n1)),
          f"   phi(G)  =   {' '.join(map(str, phi0[0]))}   {' '.join(map(str, phi1[0]))}",
          f"   phi(H)  =   {' '.join(map(str, phi0[1]))}   {' '.join(map(str, phi1[1]))}",
          "", "Step 4  kernel = dot product, summed over h", "",
          f"   K(G,H) = {k0} (h=0)  +  {k1} (h=1)  =  {k0 + k1}",
          f"   K(G,G) = {kgg},   K(H,H) = {khh}",
          f"   normalised  K = {k0 + k1}/sqrt({kgg}*{khh}) = {(k0 + k1) / np.sqrt(kgg * khh):.2f}"]
ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
        fontsize=10.5, color=INK, linespacing=1.45, transform=ax.transAxes)

fig.suptitle("Weisfeiler-Lehman relabelling: a worked example", fontsize=14,
             color=INK, fontweight="bold")
fig.text(0.5, 0.015, "At h = 0 the two graphs look identical (same label counts). "
         "After one WL iteration the triangle in G produces labels that H does not have, "
         "so their similarity drops.", ha="center", fontsize=9.5, color=MUTED)
fig.savefig(os.path.join(FIG, "fig7_wl_example.png"), dpi=200, bbox_inches="tight")
print("saved fig7_wl_example.png")
