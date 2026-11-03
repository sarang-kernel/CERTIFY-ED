"""Independent adversarial check of D-01 (a-posteriori backward error of the sparse L D L^H), from the statement only.

Part A (exact rationals): for SuperLU's stored factors L, D_r of random sparse Hermitian S (real and complex, n up to
60, entries with wide dynamic range, pivots with imaginary parts for complex data), the exact residual E = L D_r L^H -
P S P^T is formed in Fractions and ||E||_inf (>= ||E||_2) is compared with beta - shift_term; also Theorem D-01 (a)
entrywise: |E_ij| <= |E_c,ij|/(1 - u) + g_{t_ij+1} G_ij with E_c recomputed as SciPy forms it.
Part B (mp): the inertia sandwich #{lambda(A) < sigma - beta} <= nu <= #{lambda(A) <= sigma + beta} against mp
eigenvalues of A, at random shifts and at shifts within 1e-9 of an eigenvalue; small block_mb to exercise several row
blocks; threads 1 and 4 must give the same beta.
Part C (self-correction): the factors are CORRUPTED (random perturbation of L entries by 1e-8 .. 1, a sign flip of a
pivot, a wrong permutation) and sparse_ldl_residual_bound is applied to the corrupted factors with nu recounted from the
corrupted D: the sandwich must still hold (beta grows with the residual).

Usage: python docs/proofs/independent/adversarial_D-01_sparse-residual.py [--cases 40] [--seed 103]
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, mpf, to_mp, eig_herm, Tally, report  # noqa: E402
from certify_ed.inertia import inertia_sparse  # noqa: E402
from certify_ed.ldl_residual import sparse_ldl_residual_bound  # noqa: E402
from certify_ed.fp import U, gamma_dot  # noqa: E402

mp.mp.dps = 50


def fc(z):
    z = complex(z)
    return (F(z.real), F(z.imag))


def rand_sparse_herm(rng, n, cplx, dens, scale_spread):
    A = rng.standard_normal((n, n)) * (rng.random((n, n)) < dens) * 10.0 ** rng.integers(-scale_spread, scale_spread + 1, size=(n, n))
    if cplx:
        A = A + 1j * rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
    A = A + A.conj().T + np.diag(rng.standard_normal(n) * 3)
    return sp.csc_matrix(A)


def factor(S):
    f = spla.splu(S, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True, "Equil": False})
    if not np.array_equal(f.perm_r, f.perm_c):
        return None
    L = f.L.tocsr()
    if not np.all(L.diagonal() == 1.0):
        return None
    return L, f.U.diagonal(), f.perm_c


def exact_residual_inf(L, Dr, SP):
    """||L D_r L^H - S_P||_inf exactly (Fractions) and the entrywise exact |E| as a dict."""
    Lc = sp.csr_matrix(L); n = Lc.shape[0]
    rows = [{int(k): fc(v) for k, v in zip(Lc.indices[Lc.indptr[i]:Lc.indptr[i + 1]], Lc.data[Lc.indptr[i]:Lc.indptr[i + 1]])} for i in range(n)]
    D = [F(float(d)) for d in Dr]
    SPc = sp.coo_matrix(SP)
    Sd = {(int(r), int(c)): fc(v) for r, c, v in zip(SPc.row, SPc.col, SPc.data)}
    E = {}
    for i in range(n):
        for j in range(n):
            acc = (F(0), F(0))
            ri, rj = rows[i], rows[j]
            for k, a in ri.items():
                b = rj.get(k)
                if b is None:
                    continue
                # a * D_k * conj(b)
                acc = (acc[0] + D[k] * (a[0] * b[0] + a[1] * b[1]), acc[1] + D[k] * (a[1] * b[0] - a[0] * b[1]))
            s = Sd.get((i, j), (F(0), F(0)))
            e = (acc[0] - s[0], acc[1] - s[1])
            if e != (F(0), F(0)):
                E[(i, j)] = e
    # |E_ij| is irrational in general: form the moduli in 50-digit mp from the exact rational components (the margin
    # of the comparison is far above 1e-45).
    rows_mp = [mp.mpf(0)] * n
    for (i, j), e in E.items():
        rows_mp[i] += mp.sqrt((mp.mpf(e[0].numerator) / e[0].denominator) ** 2 + (mp.mpf(e[1].numerator) / e[1].denominator) ** 2)
    return max(rows_mp) if n else mp.mpf(0), E


def part_A(rng, cases, T_inf, T_entry):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(6, 61))
        S = rand_sparse_herm(rng, n, cplx, float(rng.choice([0.05, 0.15, 0.4])), int(rng.choice([0, 3, 8])))
        fac = factor(S)
        if fac is None:
            T_inf.skip(); continue
        L, dU, perm = fac
        Dr = np.real(dU)
        if np.any(Dr == 0):
            T_inf.skip(); continue
        rb = sparse_ldl_residual_bound(L, dU, S, perm, block_mb=float(rng.choice([0.001, 0.01, 64.0])))
        ip = np.argsort(perm); SP = sp.csr_matrix(S)[ip][:, ip].tocsr()
        Einf, E = exact_residual_inf(L, Dr, SP)
        T_inf.add(Einf <= mpf(rb.beta - rb.shift_term), n >= 40 or cplx, f"A n={n} cplx={cplx} ||E||_inf={mp.nstr(Einf, 8)} beta-shift={rb.beta - rb.shift_term:.6e} (resid {rb.residual_inf:.3e} round {rb.rounding_inf:.3e})")
        # (a) entrywise with E_c as SciPy forms it
        DLt = (sp.diags(Dr) @ L.conj().T).tocsc()
        Ec = (L @ DLt - SP).tocsr().toarray()
        absL = abs(L).tocsr(); G = (absL @ sp.diags(np.abs(Dr)) @ absL.T).toarray()
        c = np.diff(L.indptr)
        Lb = (L.toarray() != 0)
        ok = True
        for (r, cidx), e in E.items():
            t = int(np.sum(Lb[r] & Lb[cidx]))
            lhs = mp.sqrt((mp.mpf(e[0].numerator) / e[0].denominator) ** 2 + (mp.mpf(e[1].numerator) / e[1].denominator) ** 2)
            rhs = mpf(abs(Ec[r, cidx])) / (1 - mpf(U)) + mpf(gamma_dot(t + 1, cplx)) * mpf(G[r, cidx])
            ok = ok and lhs <= rhs
            if t > min(c[r], c[cidx]):
                ok = False
        T_entry.add(ok, cplx, f"A-(a) n={n} cplx={cplx}: an entry violates (a)")


def part_B(rng, cases, T, T_thr):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(6, 41))
        A = rand_sparse_herm(rng, n, cplx, float(rng.choice([0.1, 0.3])), int(rng.choice([0, 2])))
        lam, _ = eig_herm(to_mp(A.toarray()))
        for adv, sigma in [(False, float(rng.uniform(float(lam[0]) - 1, float(lam[-1]) + 1))),
                           (True, float(lam[int(rng.integers(0, n))]) + float(rng.choice([1e-9, -1e-9, 1e-13])))]:
            res = inertia_sparse(A, sigma, block_mb=float(rng.choice([0.001, 64.0])), threads=1)
            if not res.conclusive:
                T.skip(); continue
            lo = sum(1 for l in lam if l < mpf(sigma) - mpf(res.beta)); hi = sum(1 for l in lam if l <= mpf(sigma) + mpf(res.beta))
            T.add(lo <= res.negative <= hi, adv, f"B n={n} cplx={cplx} sigma={sigma!r} nu={res.negative} beta={res.beta:.3e} N(lo)={lo} N(hi)={hi}")
            res4 = inertia_sparse(A, sigma, block_mb=0.001, threads=4)
            res1 = inertia_sparse(A, sigma, block_mb=0.001, threads=1)
            T_thr.add(res4.beta == res1.beta and res4.negative == res1.negative, False, f"B-threads n={n}: beta {res1.beta!r} vs {res4.beta!r}")


def part_C(rng, cases, T):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(6, 41))
        A = rand_sparse_herm(rng, n, cplx, 0.2, 0)
        sigma = float(rng.uniform(-3, 3))
        S = (A - sigma * sp.identity(n, dtype=A.dtype, format="csc")).tocsc()
        fac = factor(S)
        if fac is None:
            T.skip(); continue
        L, dU, perm = fac
        L = L.tolil(); Dr = np.real(dU).copy(); perm = np.asarray(perm).copy()
        kind = i % 3
        if kind == 0:                                   # perturb L entries
            rows, cols_ = L.nonzero()
            for r_, c_ in zip(rows, cols_):
                if r_ != c_ and rng.random() < 0.3:
                    L[r_, c_] = L[r_, c_] * (1 + float(rng.choice([1e-8, 1e-3, 1.0])) * rng.standard_normal())
        elif kind == 1:                                 # flip a pivot sign
            Dr[int(rng.integers(0, n))] *= -1
        else:                                           # wrong permutation
            perm = rng.permutation(n)
        L = L.tocsr()
        if np.any(Dr == 0):
            T.skip(); continue
        rb = sparse_ldl_residual_bound(L, Dr, S, perm)
        nu = int(np.sum(Dr < 0))
        lam, _ = eig_herm(to_mp(A.toarray()))
        lo = sum(1 for l in lam if l < mpf(sigma) - mpf(rb.beta)); hi = sum(1 for l in lam if l <= mpf(sigma) + mpf(rb.beta))
        T.add(lo <= nu <= hi, True, f"C n={n} kind={kind} nu={nu} beta={rb.beta:.3e} N(lo)={lo} N(hi)={hi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=103)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_inf = Tally("Theorem D-01 (c): exact ||E||_inf <= beta - shift term (exact rationals / mp moduli)")
    T_entry = Tally("Theorem D-01 (a): entrywise |E_ij| <= |E_c,ij|/(1-u) + g_{t+1} G_ij, t <= min(c_i, c_j)")
    T_B = Tally("inertia sandwich #{lambda < sigma - beta} <= nu <= #{lambda <= sigma + beta} (mp)")
    T_thr = Tally("beta independent of the thread count")
    T_C = Tally("self-correction: corrupted factors / flipped pivot / wrong permutation keep the sandwich")
    part_A(rng, a.cases, T_inf, T_entry)
    part_B(rng, a.cases, T_B, T_thr)
    part_C(rng, a.cases, T_C)
    report([T_inf, T_entry, T_B, T_thr, T_C])


if __name__ == "__main__":
    main()
