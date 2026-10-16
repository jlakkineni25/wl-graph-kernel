"""
Weisfeiler-Lehman graph kernels implemented from scratch, following
Shervashidze et al., "Weisfeiler-Lehman Graph Kernels", JMLR 12 (2011).

Standard kernel
---------------
WL subtree kernel  K(G,H) = sum_{i=0}^{h} <phi_i(G), phi_i(H)>

Modifications (all within the framework of the same paper)
----------------------------------------------------------
M1. Cosine normalisation of the kernel
M2. Iteration weighting, w_i = lambda**i   (the paper's general WL kernel)
M3. Edge-label WL: bond types included in the relabelling signature
"""
from collections import Counter
import time

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
class Graph:
    """Light-weight graph: node labels, adjacency lists and edge labels."""

    def __init__(self, n, node_labels, neighbors, edge_labels):
        self.n = n
        self.node_labels = node_labels   # np.ndarray[int], shape (n,)
        self.neighbors = neighbors       # list[np.ndarray[int]]
        self.edge_labels = edge_labels   # list[np.ndarray[int]] aligned with neighbors


def load_tudataset(name, root="data"):
    """Load a TU benchmark dataset through PyTorch Geometric and convert it."""
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root, name)
    graphs, y = [], []
    for d in ds:
        n = d.num_nodes
        labels = d.x.argmax(dim=1).numpy() if d.x is not None else np.zeros(n, dtype=int)
        src, dst = d.edge_index.numpy()
        if d.edge_attr is not None:
            elab = d.edge_attr.argmax(dim=1).numpy()
        else:
            elab = np.zeros(len(src), dtype=int)
        nbrs = [[] for _ in range(n)]
        elabs = [[] for _ in range(n)]
        for s, t, e in zip(src, dst, elab):
            nbrs[s].append(t)
            elabs[s].append(e)
        graphs.append(Graph(n, labels.astype(int),
                            [np.array(a, dtype=int) for a in nbrs],
                            [np.array(a, dtype=int) for a in elabs]))
        y.append(int(d.y.item()))
    return graphs, np.array(y)


# --------------------------------------------------------------------------
# WL relabelling
# --------------------------------------------------------------------------
def wl_relabel(graphs, h, use_edge_labels=False):
    """
    Run h iterations of WL colour refinement on all graphs jointly.

    Returns a list of length h+1; entry i is a list (one per graph) of the
    compressed node labels after i iterations. Label ids are shared across
    graphs so that identical rooted subtrees get identical ids.
    """
    # iteration 0: compress original labels
    lookup = {}
    current = []
    for g in graphs:
        current.append(np.array([lookup.setdefault(int(l), len(lookup)) for l in g.node_labels]))
    all_iters = [current]

    for _ in range(h):
        lookup = {}
        new = []
        for g, lab in zip(graphs, current):
            out = np.empty(g.n, dtype=int)
            for v in range(g.n):
                if use_edge_labels:
                    multiset = tuple(sorted(zip(lab[g.neighbors[v]].tolist(),
                                                g.edge_labels[v].tolist())))
                else:
                    multiset = tuple(sorted(lab[g.neighbors[v]].tolist()))
                sig = (int(lab[v]), multiset)
                out[v] = lookup.setdefault(sig, len(lookup))
            new.append(out)
        current = new
        all_iters.append(current)
    return all_iters


def feature_matrices(wl_iters):
    """One sparse count matrix (n_graphs x n_labels_i) per WL iteration."""
    mats = []
    for labels in wl_iters:
        n_feat = max(int(l.max()) for l in labels) + 1
        rows, cols, vals = [], [], []
        for gi, lab in enumerate(labels):
            c = Counter(lab.tolist())
            rows.extend([gi] * len(c))
            cols.extend(c.keys())
            vals.extend(c.values())
        mats.append(sp.csr_matrix((vals, (rows, cols)), shape=(len(labels), n_feat), dtype=float))
    return mats


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
def normalize(K):
    """Cosine normalisation: K(G,H) / sqrt(K(G,G) K(H,H))."""
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)


def per_iteration_grams(mats):
    """Gram matrix phi_i phi_i^T for every iteration i."""
    return [np.asarray((m @ m.T).todense()) for m in mats]


def wl_kernel(grams, h, lam=1.0):
    """
    WL subtree kernel  K = sum_{i=0}^{h} w_i * <phi_i(G), phi_i(H)>,  w_i = lam**i.
    lam = 1 gives the original Shervashidze et al. kernel.
    """
    K = np.zeros_like(grams[0])
    for i in range(h + 1):
        K += (lam ** i) * grams[i]
    return K


# --------------------------------------------------------------------------
# Convenience timing helper
# --------------------------------------------------------------------------
def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t
