"""Size, memory and time of an eta certificate (eta.certify_eta) before it is attempted: roadmap item N-12.

`plan_eta` reproduces the bookkeeping of certify_eta without forming any matrix: the reduced boundary dimensions of
L, C, R and of the two terms Gamma_i = L u C, Gamma_j = C u R (graph.slot_reduction, graph.union_coords), the S^z
blocks of the bordered matrix M = [[-p G_A, q C], [q C^T, -q G_B]] with the labels certify_eta uses, the pairing of
blocks by the global spin flip (a pair is factored once; a block mapped to itself splits into its two flip
characters), and the site permutations that preserve the edges and each of L, C, R (the only ones certify_eta can
use). The object arrays Z (nL x nC x fi) and Y (nC x nR x fj) that certify_eta builds before any block are
reported too.

Memory and time of one block of n rows: the floating-point layer from cost.dense_inertia_mb / dense_inertia_seconds,
and the exact re-check (checker.inertia_exact: an object matrix, the dense LDL^T, the dyadic factors as int64 and as
FLINT matrices, and the row-block residual products) from two constants measured on this machine, bytes per entry
and seconds per n^3, which the caller may replace.
"""
import itertools
from math import comb

import numpy as np

from .graph import Graph, slot_reduction, union_coords

# Exact inertia (checker.inertia_exact) on this machine: placeholders replaced by measured values in
# experiments/n12_honeycomb_estimate.py (see its record).
EXACT_BYTES_PER_ENTRY = 80.0
EXACT_SECONDS_PER_N3 = 1.0e-8


def _lite(g, X):
    slots = g.boundary_slots(X)
    red, reps, downs = slot_reduction(slots)
    inner = sum(1 for u, v in g.edges if u in X and v in X)
    phys = int(np.prod([g.z[s] + 1 for s in X]))
    return {"slots": slots, "red": red, "downs": np.array(downs), "n": len(reps), "inner": inner, "phys": phys}


def symmetries(g: Graph, L, C, R):
    """Site permutations (lists) that map the edge set onto itself and each of L, C, R onto itself."""
    n = len(g.z)
    part = {}
    for k, X in enumerate((L, C, R)):
        for s in X:
            part[s] = k
    adj = [set() for _ in range(n)]
    for u, v in g.edges:
        adj[u].add(v); adj[v].add(u)
    order = sorted(range(n))
    found = []

    def extend(img):
        k = len(img)
        if k == n:
            found.append(list(img))
            return
        s = order[k]
        used = set(img)
        for t in range(n):
            if t in used or part.get(t) != part.get(s) or len(adj[t]) != len(adj[s]) or g.free[t] != g.free[s]:
                continue
            if all(((img[order[j]] in adj[t]) == (order[j] in adj[s])) for j in range(k)):
                img.append(t)
                extend(img)
                img.pop()
    extend([])
    return found


def plan_eta(g: Graph, L, C, R, sigmas=(), use_flip=True, bytes_per_entry=None, seconds_per_n3=None):
    """Dimensions of every object certify_eta(g, L, C, R, thresholds, sigmas, use_flip) would form, and the cost of
    its largest block. Without sigmas the block sizes are exact; with sigmas each block is divided by the order of
    the group they generate, an estimate (the characters need not split a block evenly). The number of site
    permutations available (symmetries()) is reported separately: certify_eta uses only the sigmas it is given."""
    from ...cost import dense_inertia_mb, dense_inertia_seconds
    bpe = EXACT_BYTES_PER_ENTRY if bytes_per_entry is None else bytes_per_entry
    spn = EXACT_SECONDS_PER_N3 if seconds_per_n3 is None else seconds_per_n3
    rL, rC, rR = (_lite(g, X) for X in (L, C, R))
    Ti, downs_i, sing_i, slots_i = union_coords(g, [L, C], [rL, rC])
    Tj, downs_j, sing_j, slots_j = union_coords(g, [C, R], [rC, rR])
    fi, fj = len(Ti), len(Tj)
    U = sorted(set(L) | set(C) | set(R))
    rU = _lite(g, U)
    labA, labB = {}, {}
    for b in range(fi):
        for r in range(rR["n"]):
            k = int(sing_i + downs_i[b] + rR["downs"][r] + rR["inner"])
            labA[k] = labA.get(k, 0) + 1
    for l in range(rL["n"]):
        for d in range(fj):
            k = int(sing_j + rL["downs"][l] + downs_j[d] + rL["inner"])
            labB[k] = labB.get(k, 0) + 1
    keys = sorted(set(labA) | set(labB))
    size = {k: labA.get(k, 0) + labB.get(k, 0) for k in keys}
    K0 = 2 * (sing_i + rR["inner"]) + len(slots_i) + len(rR["slots"])
    K0b = 2 * (sing_j + rL["inner"]) + len(rL["slots"]) + len(slots_j)
    avail = len(symmetries(g, L, C, R))
    nsig = 2 ** len(sigmas) if sigmas else 1          # sigmas generate commuting involutions in every use so far
    factored = []                      # (keys, rows of one factorization, number of factorizations)
    done = set()
    for k in keys:
        if k in done:
            continue
        kp = K0 - k if use_flip and K0 == K0b else None
        if kp is not None and kp != k and kp in size:
            done |= {k, kp}
            factored.append({"keys": [k, kp], "rows": size[k], "factorizations": 1})
        elif kp == k:
            done.add(k)
            factored.append({"keys": [k], "rows": -(-size[k] // 2), "factorizations": 2})
        else:
            done.add(k)
            factored.append({"keys": [k], "rows": size[k], "factorizations": 1})
    if nsig > 1:                       # site symmetries split each block about evenly (lower estimate of size)
        for f in factored:
            f["rows_with_site_symmetry"] = -(-f["rows"] // nsig)
    big = max(factored, key=lambda f: f.get("rows_with_site_symmetry", f["rows"]))
    n = big.get("rows_with_site_symmetry", big["rows"])
    tot_n3 = sum(f["factorizations"] * float(f.get("rows_with_site_symmetry", f["rows"])) ** 3 for f in factored)
    return {
        "sites": len(g.z), "edges": len(g.edges),
        "phys_dims": {"L": rL["phys"], "C": rC["phys"], "R": rR["phys"]},
        "reduced": {"L": rL["n"], "C": rC["n"], "R": rR["n"], "Gamma_i": fi, "Gamma_j": fj, "union": rU["n"]},
        "dim_V": rL["n"] * rC["n"] * rR["n"], "dim_A": fi * rR["n"], "dim_B": rL["n"] * fj,
        "bordered_rows": fi * rR["n"] + rL["n"] * fj,
        "Sz_blocks": {str(k): size[k] for k in keys},
        "site_symmetries_available": avail, "site_symmetry_group_used": nsig, "flip_pairs_consistent": K0 == K0b,
        "factorizations": factored, "largest_block_rows": n,
        "Z_Y_object_entries": rL["n"] * rC["n"] * fi + rC["n"] * rR["n"] * fj,
        "Z_Y_object_GiB": 8.0 * (rL["n"] * rC["n"] * fi + rC["n"] * rR["n"] * fj) / 2 ** 30,
        "largest_block": {"rows": n, "float_layer_MiB": dense_inertia_mb(n), "float_layer_seconds": dense_inertia_seconds(n),
                          "exact_GiB": bpe * n * n / 2 ** 30, "exact_seconds": spn * float(n) ** 3,
                          "dense_float64_copy_GiB": 8.0 * n * n / 2 ** 30},
        "all_blocks_exact_seconds": spn * tot_n3,
        "model": {"exact_bytes_per_entry": bpe, "exact_seconds_per_n3": spn},
        "binomial_check": {"C(22,11)": comb(22, 11)},
    }
