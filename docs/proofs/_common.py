"""Shared helpers of the docs/proofs verification scripts (mpmath ground truth, exact rationals).

Not part of the package. Every script in this directory can be run from the repository root with
`python docs/proofs/verify_<id>.py [--cases N] [--seed S]`; it prints the case counts and the number of
violations and exits with status 1 when a violation was found.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import mpmath as mp
import numpy as np

mp.mp.dps = 60
TOL = mp.mpf("1e-40")     # slack for the 60-digit ground truth, multiplied by the operand scale in each script;
#                           the floating-point bounds exceed the exact quantities by at least a relative 1e-16 of
#                           the quantity itself, so a violation is never masked by it


def to_mp(A):
    """numpy array (float or complex binary64 entries, taken exactly) -> mp.matrix."""
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
    M = mp.matrix(A.shape[0], A.shape[1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            z = A[i, j]
            if np.iscomplexobj(A):
                M[i, j] = mp.mpc(mp.mpf(float(z.real)), mp.mpf(float(z.imag)))
            else:
                M[i, j] = mp.mpf(float(z))
    return M


def herm(M):
    return M.transpose_conj()


def eigh(M):
    """Eigenvalues (ascending, list of mpf) and eigenvectors (columns) of a Hermitian mp.matrix."""
    n = M.rows
    if n == 0:
        return [], mp.matrix(0, 0)
    E, Q = mp.eighe(M)
    vals = [mp.re(E[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i])
    Qs = mp.matrix(n, n)
    for new, old in enumerate(order):
        for r in range(n):
            Qs[r, new] = Q[r, old]
    return [vals[i] for i in order], Qs


def norm2(M):
    """Spectral norm of an mp.matrix (sqrt of the largest eigenvalue of M^H M)."""
    if M.rows == 0 or M.cols == 0:
        return mp.mpf(0)
    S = herm(M) * M if M.rows >= M.cols else M * herm(M)
    vals, _ = eigh(S)
    return mp.sqrt(max(vals[-1], mp.mpf(0)))


def fro(M):
    return mp.sqrt(sum(abs(M[i, j]) ** 2 for i in range(M.rows) for j in range(M.cols)))


def func_herm(M, f):
    """f(M) for Hermitian M by diagonalisation."""
    vals, Q = eigh(M)
    D = mp.diag([f(v) for v in vals])
    return Q * D * herm(Q)


def bottleneck(mu_list, lam_list):
    """min over injections j -> i_j of max_j |lam_{i_j} - mu_j| (brute force; g <= 4, n <= 7)."""
    g, n = len(mu_list), len(lam_list)
    best = None
    for inj in itertools.permutations(range(n), g):
        d = max(abs(lam_list[i] - mu_list[j]) for j, i in enumerate(inj))
        if best is None or d < best:
            best = d
    return best


def bottleneck_sorted(mu_list, lam_list):
    """max_j |sorted(lam)_j - sorted(mu)_j| for lists of equal length."""
    a = sorted(mu_list)
    b = sorted(lam_list)
    return max(abs(x - y) for x, y in zip(a, b))


def frac(x) -> Fraction:
    return Fraction(float(x))


class Counter:
    def __init__(self):
        self.random = 0
        self.adversarial = 0
        self.skipped = 0
        self.violations = []

    def add(self, adversarial, ok, note=""):
        if adversarial:
            self.adversarial += 1
        else:
            self.random += 1
        if not ok:
            self.violations.append(note)

    def report(self, name):
        print(f"{name}: {self.random} random + {self.adversarial} adversarial cases, "
              f"{self.skipped} skipped (hypothesis not met), {len(self.violations)} violations")
        for v in self.violations[:20]:
            print("  VIOLATION:", v)
        return len(self.violations) == 0
