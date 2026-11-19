"""Shared helpers of the independent adversarial searches (docs/proofs/independent/adversarial_<id>.py).

Written for the independent re-derivations (2026-10-30) without reading docs/proofs/_common.py, so that the
ground truth here is computed by different code from the one the original verify_<id>.py scripts use. Exact
ground truth: mpmath at 60 significant digits (eigenvalues) or fractions.Fraction (rational quantities).
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def F(x) -> Fraction:
    """Exact rational value of a binary64 number (or of a Python int/Fraction)."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    return Fraction(float(x))


def mpf(x) -> mp.mpf:
    """Exact mpf of a binary64 number (binary64 -> mpf is exact at dps >= 17 for the mantissa; 60 digits used)."""
    return mp.mpf(float(x))


def mpc_(x):
    x = complex(x)
    return mp.mpc(mp.mpf(x.real), mp.mpf(x.imag))


def to_mp(A):
    A = np.asarray(A)
    if np.iscomplexobj(A):
        return mp.matrix([[mpc_(v) for v in row] for row in A])
    return mp.matrix([[mpf(v) for v in row] for row in A])


def herm(M):
    return M.transpose_conj() if hasattr(M, "transpose_conj") else M.H


def eig_herm(M):
    """Eigenvalues (ascending) and eigenvectors of an mp Hermitian matrix."""
    w, V = mp.eigh(M)
    idx = sorted(range(len(w)), key=lambda i: w[i])
    return [w[i] for i in idx], V


def spec_norm(M):
    """Spectral norm via the largest eigenvalue of M^H M (mp)."""
    if M.rows == 0 or M.cols == 0:
        return mp.mpf(0)
    w, _ = mp.eigh(herm(M) * M)
    return mp.sqrt(max(mp.re(v) for v in w)) if max(mp.re(v) for v in w) > 0 else mp.mpf(0)


def fro_norm(M):
    s = mp.mpf(0)
    for i in range(M.rows):
        for j in range(M.cols):
            s += abs(M[i, j]) ** 2
    return mp.sqrt(s)


def herm_fun(M, f):
    w, V = mp.eigh(M)
    D = mp.diag([f(mp.re(v)) for v in w])
    return V * D * herm(V)


def has_matching(left, right, radius):
    """Is there an injection j -> i with |right[i] - left[j]| <= radius? (augmenting paths, exact mp compare)."""
    n_l, n_r = len(left), len(right)
    match_r = [-1] * n_r

    def try_assign(j, seen):
        for i in range(n_r):
            if not seen[i] and abs(right[i] - left[j]) <= radius:
                seen[i] = True
                if match_r[i] == -1 or try_assign(match_r[i], seen):
                    match_r[i] = j
                    return True
        return False

    for j in range(n_l):
        if not try_assign(j, [False] * n_r):
            return False
    return True


def bottleneck_distance(left, right):
    """Smallest radius r such that an injection with |right[i] - left[j]| <= r exists (min over the finite set of
    candidate distances)."""
    cands = sorted({abs(a - b) for a in left for b in right})
    lo, hi = 0, len(cands) - 1
    if not has_matching(left, right, cands[hi]):
        return mp.inf
    while lo < hi:
        mid = (lo + hi) // 2
        if has_matching(left, right, cands[mid]):
            hi = mid
        else:
            lo = mid + 1
    return cands[lo]


def sorted_pairing_distance(left, right):
    """max_j |sorted(left)_j - sorted(right)_j| (requires equal lengths)."""
    a, b = sorted(left), sorted(right)
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b)) if a else mp.mpf(0)


class Tally:
    def __init__(self, name):
        self.name = name
        self.random = 0
        self.adversarial = 0
        self.skipped = 0
        self.violations = []

    def add(self, ok, adversarial, note=""):
        if adversarial:
            self.adversarial += 1
        else:
            self.random += 1
        if not ok:
            self.violations.append(note)

    def skip(self):
        self.skipped += 1

    def line(self):
        s = (f"{self.name}: {self.random} random + {self.adversarial} adversarial cases, "
             f"{self.skipped} skipped (hypothesis not met), {len(self.violations)} violations")
        for v in self.violations[:10]:
            s += f"\n    VIOLATION {v}"
        return s


def report(tallies):
    total = 0
    for t in tallies:
        print(t.line())
        total += len(t.violations)
    print(f"RESULT: {total} violations")
    return total
