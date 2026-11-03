"""Computational verification of docs/proofs/D-01_sparse-residual.md (a-posteriori sparse L D L^H residual).

For random sparse Hermitian S (real and complex) the script factorizes S with SuperLU exactly as
certify_ed.inertia.inertia_sparse does, calls ldl_residual.sparse_ldl_residual_bound on the stored factors and
checks, in exact rational arithmetic on the stored binary64 numbers,

  (V1) ||E||_inf <= beta             with E = L D L^H - P S P^T the exact residual of the stored factors, for the
       beta of the default row partition (what inertia_sparse records) and for the beta of a partition into
       many small row blocks (block_mb = 0.001, min_block_rows = 1: the mirrored column sums are exercised),
  (V2) |E_ij| <= |E_c,ij|/(1 - u) + gamma_{t_ij+1} G_ij  entry by entry, where E_c is the residual recomputed in
       floating point by the same SciPy operations (the rounding-model step of the proof),
  (V3) #{lambda < -beta} <= nu <= #{lambda <= beta}   (Lemma 4 of docs/theory.md) against 60-digit eigenvalues,

and that the a-posteriori beta never exceeds 2^-30 max(1, max|S_ij|) on the random cases (it measures a
backward-stable factorization; without pivoting the growth is bounded by the small dimension here). Adversarial cases: a zero pivot (exactly singular S), ill-conditioned
pivots (growth 1/eps without pivoting), an eigenvalue exactly at sigma, entries at scale 1e150 (overflow in the
product -> non-finite factors), 1e-150 and 1e300, n = 1, a diagonal matrix, a complex Hermitian tridiagonal,
dense patterns. Each adversarial case either satisfies (V1)-(V3) or is reported inconclusive by the code
(counted as skipped: hypothesis not met).

Usage: python docs/proofs/verify_D-01.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from _common import Counter, eigh, to_mp  # noqa: E402
from certify_ed.fp import U, gamma_dot  # noqa: E402
from certify_ed.inertia import inertia_sparse  # noqa: E402
from certify_ed.ldl_residual import sparse_ldl_residual_bound  # noqa: E402

UF = Fraction(U)
SLACK = 1 + Fraction(1, 2 ** 40)


def fr(z):
    """Exact rational pair (re, im) of a binary64 number."""
    if np.iscomplexobj(z):
        return (Fraction(float(np.real(z))), Fraction(float(np.imag(z))))
    return (Fraction(float(z)), Fraction(0))


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def sqrt_upper(q: Fraction) -> Fraction:
    """A rational upper bound on sqrt(q), q >= 0, with relative slack 2^-40 (any magnitude: q is scaled by an
    even power of two into [1, 4) before the floating-point square root)."""
    if q == 0:
        return Fraction(0)
    e = 0
    while q >= 4:
        q /= 4
        e += 1
    while q < 1:
        q *= 4
        e -= 1
    x = Fraction(float(np.sqrt(float(q)))) * SLACK
    while x * x < q:
        x = x * SLACK
    return x * Fraction(2) ** e


def exact_residual(L, D, SP):
    """E = L D L^H - SP entry by entry as exact rational pairs (dense, small n), and the exact G = |L||D||L|^T
    and the overlap counts t_ij, computed with a single pass over the stored nonzeros."""
    n = L.shape[0]
    Ld = L.toarray()
    SPd = SP.toarray()
    Lf = [[fr(Ld[i, k]) for k in range(n)] for i in range(n)]
    Df = [Fraction(float(np.real(D[k]))) for k in range(n)]
    absL = [[sqrt_upper(Lf[i][k][0] ** 2 + Lf[i][k][1] ** 2) if Ld[i, k] != 0 else Fraction(0) for k in range(n)]
            for i in range(n)]
    E = [[None] * n for _ in range(n)]
    G = [[Fraction(0)] * n for _ in range(n)]
    T = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = (Fraction(0), Fraction(0))
            g = Fraction(0)
            t = 0
            for k in range(min(i, j) + 1):
                if Ld[i, k] != 0 and Ld[j, k] != 0:
                    p = cmul(Lf[i][k], (Lf[j][k][0], -Lf[j][k][1]))
                    acc = (acc[0] + p[0] * Df[k], acc[1] + p[1] * Df[k])
                    g += absL[i][k] * abs(Df[k]) * absL[j][k]
                    t += 1
            s = fr(SPd[i, j])
            E[i][j] = (acc[0] - s[0], acc[1] - s[1])
            G[i][j] = g
            T[i][j] = t
    return E, G, T


def factor(S):
    return spla.splu(sp.csc_matrix(S), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options={"SymmetricMode": True, "Equil": False})


def random_sparse_hermitian(rng, n, cplx, density=0.25, scale=1.0, shift=0.0):
    k = max(n, int(density * n * n))
    rows = rng.integers(0, n, k)
    cols = rng.integers(0, n, k)
    vals = rng.standard_normal(k) * scale
    if cplx:
        vals = vals + 1j * rng.standard_normal(k) * scale
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A = (A + A.conj().T) * 0.5
    A = A + sp.diags(rng.standard_normal(n) * scale + shift)
    A = A.tocsr()
    A.sort_indices()
    return A


def check_case(S, ctr, adversarial, note, expect_inconclusive=False):
    n = S.shape[0]
    res = inertia_sparse(S, 0.0)
    if not res.conclusive:
        if expect_inconclusive:
            ctr.skipped += 1
            return
        ctr.add(adversarial, False, f"{note}: unexpectedly inconclusive ({res.note})")
        return
    f = factor(S)
    L = f.L.tocsr()
    D = f.U.diagonal()
    rb = sparse_ldl_residual_bound(L, D, S, f.perm_c, block_mb=0.001, threads=1, min_block_rows=1)   # many blocks
    # the default partition (inertia_sparse: one block at these sizes) and the many-block one give two bounds; both
    # must dominate the exact residual (they differ in the order of the floating-point row sums and in m)
    beta = Fraction(min(rb.beta, res.beta))
    ip = np.argsort(f.perm_c)
    SP = sp.csr_matrix(S)[ip][:, ip]
    E, G, T = exact_residual(L, D, SP)
    Eabs = [[sqrt_upper(E[i][j][0] ** 2 + E[i][j][1] ** 2) for j in range(n)] for i in range(n)]
    # V1
    nrm = max(sum(Eabs[i]) for i in range(n))
    ok1 = nrm <= beta
    # V2: the entrywise rounding model, with E_c recomputed by the same SciPy operations
    cplx = np.iscomplexobj(L.data) or np.iscomplexobj(S.data)
    DLt = (sp.diags(np.real(D)) @ L.conj().T).tocsc()
    Ec = (L @ DLt - SP).toarray()
    c = np.diff(L.indptr)
    ok2 = True
    for i in range(n):
        for j in range(n):
            t = T[i][j]
            g = Fraction(gamma_dot(t + 1, cplx)) if t > 0 else Fraction(0)
            ec = Fraction(float(abs(Ec[i, j]))) * SLACK        # |E_c| rounded once by abs(): slack 2^-40
            if Eabs[i][j] > ec / (1 - UF) + g * G[i][j]:
                ok2 = False
            if t > min(c[i], c[j]):
                ok2 = False
    # V3: Lemma 4 against 60-digit eigenvalues
    vals, _ = eigh(to_mp(S.toarray()))
    ok3 = True
    for bb in (rb.beta, res.beta):
        b = mp.mpf(bb)
        below = sum(1 for v in vals if v < -b)
        upto = sum(1 for v in vals if v <= b)
        ok3 = ok3 and (below <= res.negative <= upto)
    scale = float(np.max(np.abs(S.data))) if S.nnz else 1.0
    ok4 = adversarial or rb.beta <= 2.0 ** -30 * max(1.0, scale)   # sanity: beta is a measured O(u ||S||) quantity
    ctr.add(adversarial, ok1 and ok2 and ok3 and ok4,
            f"{note}: V1 {ok1} (|E|={float(nrm):.3e} beta={rb.beta:.3e}) V2 {ok2} V3 {ok3} "
            f"({below} <= {res.negative} <= {upto}) small-beta {ok4}")


def gamma_exact(k: int) -> Fraction:
    return k * UF / (1 - k * UF)


def check_scalar_inequalities(ctr):
    """(Lambda-R1) and (Lambda-R2) of the document, exactly, on a grid of m and n."""
    grid = list(range(1, 20001)) + [2 ** j for j in range(15, 28)]
    one_m_u = 1 - UF
    bad1 = [m for m in grid if m >= 2 and not
            ((1 - gamma_exact(m + 2)) * (1 + gamma_exact(m) * one_m_u ** 2) * (1 + 4 * UF) * (1 + 8 * UF) * one_m_u ** 3
             >= 1 / one_m_u)]
    bad2 = [n for n in grid if not
            ((1 - gamma_exact(2 * n + 6)) * one_m_u ** 3 * (1 + gamma_exact(2 * n + 8) * one_m_u ** 2) * (1 + 8 * UF) ** 2
             * one_m_u ** 3 >= 1)]
    ctr.add(True, not bad1, f"Lambda-R1 fails at m = {bad1[:5]}")
    ctr.add(True, not bad2, f"Lambda-R2 fails at n = {bad2[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ctr = Counter()
    for case in range(args.cases):
        n = int(rng.integers(2, 14))
        cplx = bool(case % 2)
        density = [0.15, 0.3, 0.6][case % 3]
        scale = [1.0, 1e-3, 1e3][(case // 3) % 3]
        shift = [0.0, 3.0 * scale, -3.0 * scale][(case // 9) % 3]
        S = random_sparse_hermitian(rng, n, cplx, density, scale, shift)
        check_case(S, ctr, False, f"random case {case} n={n} cplx={cplx}")
    # adversarial
    check_case(sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])), ctr, True, "exact zero pivot", expect_inconclusive=True)
    check_case(sp.csr_matrix(np.zeros((3, 3))), ctr, True, "zero matrix", expect_inconclusive=True)
    for eps in (1e-6, 1e-12, 1e-16):
        B = np.array([[eps, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
        check_case(sp.csr_matrix(B), ctr, True, f"ill-conditioned pivot eps={eps}")
    # an eigenvalue exactly at sigma = 0: rank 1, eigenvalues 0 and 2; pivots 1 and exactly 0
    B = np.array([[1.0, 1.0], [1.0, 1.0]])
    check_case(sp.csr_matrix(B), ctr, True, "eigenvalue exactly at sigma (rank 1)", expect_inconclusive=True)
    B = np.array([[4.0, 2.0, 0.0], [2.0, 1.0, 1e-17], [0.0, 1e-17, 1.0]])   # second pivot 0 exactly
    check_case(sp.csr_matrix(B), ctr, True, "cancelling pivot", expect_inconclusive=True)
    # an eigenvalue at sigma within rounding but a nonzero pivot: 2x2 [[1, 1], [1, 1 + 2^-52]]
    B = np.array([[1.0, 1.0], [1.0, 1.0 + 2.0 ** -52]])
    check_case(sp.csr_matrix(B), ctr, True, "eigenvalue 2^-52 from sigma")
    for sc in (1e150, 1e-150, 1e300):
        S = random_sparse_hermitian(rng, 6, False, 0.5, sc)
        check_case(S, ctr, True, f"scale {sc}", expect_inconclusive=(sc >= 1e150))
    check_case(sp.csr_matrix(np.array([[-3.0]])), ctr, True, "n = 1")
    check_case(sp.diags([1.0, -2.0, 3.0, -4.0]).tocsr(), ctr, True, "diagonal")
    n = 9
    T = sp.diags([np.full(n - 1, 0.5j), np.linspace(-1, 1, n) + 0.3, np.full(n - 1, -0.5j)], [-1, 0, 1]).tocsr()
    check_case(T, ctr, True, "complex Hermitian tridiagonal")
    for seed in range(5):
        r2 = np.random.default_rng(seed)
        S = random_sparse_hermitian(r2, 12, bool(seed % 2), 0.9, 1.0)
        check_case(S, ctr, True, f"dense-pattern n=12 seed={seed}")
    check_scalar_inequalities(ctr)
    ok = ctr.report("D-01 sparse residual")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
