"""Sparse Heisenberg Hamiltonians on sites with different spins, restricted to fixed total S^z.

Used for the spin-1 crown cluster, where each outer pair of the reduced model carries a conserved
pair spin S~ in {0, 1, 2} and acts as a single composite spin-S~ site:
    c1 S_a.S_b             = c1 (S~(S~+1) - 2 S(S+1)) / 2          (constant on the block)
    c3 (S_a + S_b).S_h     = c3 S~ . S_h                            (exact on the S~ irrep)
Basis states are tuples of 2m (integers) per site; S.S is applied exactly: zz part from m_i m_j and the
flip part from the ladder coefficients sqrt(S(S+1) - m(m+1)), which are in general irrational.
Couplings and matrix elements are returned in floating point; callers that need exactness use
integer-valued models (all spins 1/2) or bound the rounding separately.
"""
from itertools import product

import numpy as np
import scipy.sparse as sp


def sz0_basis(two_spins):
    """All tuples (2m_1, ..., 2m_n) with sum 0; returns an int array (states x n) and a lookup."""
    ranges = [np.arange(-ts, ts + 1, 2) for ts in two_spins]
    # build incrementally keeping only partial sums that can still reach zero
    maxrest = np.concatenate([np.cumsum([ts for ts in two_spins][::-1])[::-1][1:], [0]])
    states = np.zeros((1, 0), dtype=np.int64)
    for k, r in enumerate(ranges):
        cand = np.concatenate([np.repeat(states, len(r), axis=0), np.tile(r, states.shape[0])[:, None]], axis=1)
        partial = cand.sum(axis=1)
        keep = np.abs(partial) <= maxrest[k]
        states = cand[keep]
    states = states[states.sum(axis=1) == 0]
    return states


def encode(states, two_spins):
    base = np.array([ts + 1 for ts in two_spins], dtype=np.int64)
    digits = (states + np.array(two_spins)) // 2
    code = np.zeros(states.shape[0], dtype=np.int64)
    for k in range(states.shape[1]):
        code = code * base[k] + digits[:, k]
    return code


def heisenberg(two_spins, bonds, const=0.0):
    """H = sum_(c,i,j) c S_i.S_j + const on the S^z=0 sector. Returns (csr matrix, states)."""
    states = sz0_basis(two_spins)
    codes = encode(states, two_spins)
    order = np.argsort(codes)
    states, codes = states[order], codes[order]
    n = len(codes)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, float(const))]
    diag = np.zeros(n)
    for c, i, j in bonds:
        mi, mj = states[:, i] / 2.0, states[:, j] / 2.0
        diag += c * mi * mj
        Si, Sj = two_spins[i] / 2.0, two_spins[j] / 2.0
        # S+_i S-_j / 2 : m_i -> m_i + 1, m_j -> m_j - 1
        for di, dj in ((1, -1), (-1, 1)):
            ok = (np.abs(states[:, i] + 2 * di) <= two_spins[i]) & (np.abs(states[:, j] + 2 * dj) <= two_spins[j])
            src = np.nonzero(ok)[0]
            if src.size == 0:
                continue
            t = states[src].copy()
            t[:, i] += 2 * di
            t[:, j] += 2 * dj
            dst = np.searchsorted(codes, encode(t, two_spins))
            a = np.sqrt(Si * (Si + 1) - mi[src] * (mi[src] + di))
            b = np.sqrt(Sj * (Sj + 1) - mj[src] * (mj[src] + dj))
            rows.append(dst); cols.append(src); vals.append(0.5 * c * a * b)
    vals[0] = vals[0] + diag
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    H.sum_duplicates()
    return H, states
