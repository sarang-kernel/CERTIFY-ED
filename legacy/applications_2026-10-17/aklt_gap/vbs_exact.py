"""Exact (integer) AKLT valence-bond-solid maps for spin-3/2 on degree-3 graphs.

Scaled basis. With D = diag(1, sqrt3, sqrt3, 1) per site (|3/2>, |1/2>, |-1/2>, |-3/2>), the scaled
operators D^-1 S^a D have integer entries (S+ -> [3, 2, 1] above the diagonal, S- -> [1, 2, 3] below,
S^z unchanged), and a VBS amplitude on |m_k> (k down virtual spins) becomes proportional to
1/binom(3, k); multiplied by 3 it is the integer 3/binom(3,k) in {3, 1, 1, 3}. Physical inner products
in scaled coordinates carry the weight D^2 = diag(1, 3, 3, 1) per site. So all vectors, Gram matrices
and the (scaled) AKLT Hamiltonian are exact integers or rationals.
"""
import itertools
import math
from fractions import Fraction
from math import comb

import numpy as np
import scipy.sparse as sp

D_LOC = 4
WEIGHT = np.array([1, 3, 3, 1], dtype=np.int64)          # D^2 per site
AMP = np.array([3, 1, 1, 3], dtype=np.int64)             # 3 / binom(3, k)


def scaled_spin_ops():
    Sz = [Fraction(3, 2), Fraction(1, 2), Fraction(-1, 2), Fraction(-3, 2)]
    Sp = np.zeros((4, 4), dtype=object); Sm = np.zeros((4, 4), dtype=object)
    Sp[:] = Fraction(0); Sm[:] = Fraction(0)
    for k, v in zip((1, 2, 3), (3, 2, 1)):
        Sp[k - 1, k] = Fraction(v)
    for k, v in zip((1, 2, 3), (1, 2, 3)):
        Sm[k, k - 1] = Fraction(v)
    Z = np.zeros((4, 4), dtype=object); Z[:] = Fraction(0)
    for k in range(4):
        Z[k, k] = Sz[k]
    return Z, Sp, Sm


def _kron(a, b):
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n * m, n * m), dtype=object)
    for i in range(n):
        for j in range(n):
            out[i * m:(i + 1) * m, j * m:(j + 1) * m] = a[i, j] * b
    return out


def scaled_P3_integer():
    """Return (P, den): integer 16x16 matrix P with P / den = D^-1 (x) D^-1  P3  D (x) D exactly."""
    Z, Sp, Sm = scaled_spin_ops()
    X = _kron(Z, Z) + (_kron(Sp, Sm) + _kron(Sm, Sp)) * Fraction(1, 2)
    I = np.zeros((16, 16), dtype=object); I[:] = Fraction(0)
    for i in range(16):
        I[i, i] = Fraction(1)
    S = Fraction(3, 2)
    xs = {J: Fraction(J * (J + 1), 2) - S * (S + 1) for J in range(4)}
    P = I.copy()
    for J in (0, 1, 2):
        P = P.dot((X - xs[J] * I) * (1 / (xs[3] - xs[J])))
    den = 1
    for v in P.ravel():
        den = den * v.denominator // math.gcd(den, v.denominator)
    Pi = np.array([[int(v * den) for v in row] for row in P], dtype=np.int64)
    return Pi, den


def vbs_map(n_sites, inner_edges, free_slots):
    """Integer matrix Psi (4^n x 2^f): columns are scaled VBS states for each boundary configuration.

    inner_edges: list of (u, v) inside the subgraph (each carries a singlet |ud> - |du>).
    free_slots: list of sites, one entry per free virtual index (degree-3 sites: 3 - inner degree).
    Every site must have exactly three virtual spins (inner edges + free slots)."""
    deg = np.zeros(n_sites, dtype=int)
    for u, v in inner_edges:
        deg[u] += 1; deg[v] += 1
    for s in free_slots:
        deg[s] += 1
    assert np.all(deg == 3), deg
    f = len(free_slots)
    Psi = np.zeros((D_LOC ** n_sites, 2 ** f), dtype=np.int64)
    powers = D_LOC ** np.arange(n_sites - 1, -1, -1)
    for bconf in range(2 ** f):
        bdown = [(bconf >> (f - 1 - i)) & 1 for i in range(f)]
        for sconf in itertools.product((0, 1), repeat=len(inner_edges)):
            downs = np.zeros(n_sites, dtype=int)
            sign = 1
            for (u, v), s in zip(inner_edges, sconf):
                if s == 0:          # |u up, v down>
                    downs[v] += 1
                else:               # - |u down, v up>
                    downs[u] += 1
                    sign = -sign
            for site, bd in zip(free_slots, bdown):
                downs[site] += bd
            amp = sign * int(np.prod(AMP[downs]))
            Psi[int(downs @ powers), bconf] += amp
    return Psi


def hamiltonian_scaled(n_sites, edges):
    """Sparse integer matrix den * D^-1 H D, H = sum_edges P3, on 4^n (site 0 most significant)."""
    P, den = scaled_P3_integer()
    N = D_LOC ** n_sites
    rows, cols, vals = [], [], []
    idx = np.arange(N)
    digits = [(idx // D_LOC ** (n_sites - 1 - s)) % D_LOC for s in range(n_sites)]
    for u, v in edges:
        a, b = digits[u], digits[v]
        colpair = a * D_LOC + b
        for rp in range(16):
            coef = P[rp, colpair]
            nz = coef != 0
            if not np.any(nz):
                continue
            a2, b2 = divmod(rp, D_LOC)
            tgt = idx[nz] + (a2 - a[nz]) * D_LOC ** (n_sites - 1 - u) + (b2 - b[nz]) * D_LOC ** (n_sites - 1 - v)
            rows.append(tgt); cols.append(idx[nz]); vals.append(coef[nz])
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N),
                      dtype=np.int64)
    return H, den


def weight_vector(n_sites):
    w = np.ones(1, dtype=np.int64)
    for _ in range(n_sites):
        w = np.kron(w, WEIGHT)
    return w
