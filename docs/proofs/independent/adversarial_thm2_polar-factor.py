"""Independent adversarial search for Theorem 2 / polar-factor lemma / Kahan (docs/proofs/thm2_polar-factor.md).

Targets the hypotheses the original proof leans on: eta -> 1 (invertibility of the Gram matrix), zero residuals
(exact eigenvectors, non-orthogonal within a degenerate level), repeated eigenvalues, scale 1e+-150 and subnormal
operator entries, g = n and g = 1, complex entries. Ground truth in mpmath at 60 digits from the stored binary64
numbers; the implemented bound (`enclosure._cluster_bound`, `certify._subspace_shift`,
`sectors.decomposition_shift`) is compared with the exact rho and with the exact bottleneck matching distance.

Usage: python docs/proofs/independent/adversarial_thm2_polar-factor.py [--cases 300] [--seed 3]
"""
import argparse
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import (Tally, bottleneck_distance, eig_herm, herm, herm_fun, mpf, report,  # noqa: E402
                        spec_norm, to_mp)
from certify_ed.enclosure import OperatorAdapter, _cluster_bound, residuals  # noqa: E402
from certify_ed.certify import _subspace_shift  # noqa: E402
from certify_ed.fp import inf_norm_upper  # noqa: E402

SLACK = mp.mpf("1e-45")   # for the irrational (eigenvalue) comparisons only; code-vs-exact comparisons are exact


def hermitian(rng, n, cplx, scale=1.0):
    M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    return (M + M.conj().T) / 2 * scale


def cases(rng, i):
    """Yield (A, X, mu, tag, adversarial)."""
    cplx = bool(i % 3 == 0)
    kind = i % 11
    if kind == 0:
        n = int(rng.integers(2, 7)); g = int(rng.integers(1, n + 1))
        A = hermitian(rng, n, cplx)
        X = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0)
        X /= np.linalg.norm(X, axis=0)
        mu = rng.standard_normal(g)
        return A, X, mu, "random", False
    if kind == 1:   # eta -> 1: two nearly parallel columns, c = 1 - 2^-k
        n = int(rng.integers(2, 6)); k = int(rng.integers(2, 50))
        c = 1.0 - 2.0 ** -k
        X = np.zeros((n, 2), dtype=complex if cplx else float)
        X[0, 0] = 1.0; X[0, 1] = c; X[1, 1] = np.sqrt(1 - c * c) * (1j if cplx else 1.0)
        A = hermitian(rng, n, cplx)
        mu = np.array([A[0, 0].real, A[0, 0].real])
        return A, X, mu, f"eta_near_1(k={k})", True
    if kind == 2:   # zero residual: exact eigenvectors (diagonal A) with non-orthogonal columns inside a 2-fold level
        n = int(rng.integers(3, 7))
        d = rng.standard_normal(n); d[1] = d[0]
        A = np.diag(d).astype(complex if cplx else float)
        X = np.zeros((n, 2), dtype=A.dtype)
        th = rng.uniform(0.05, 1.0)
        X[0, 0] = 1.0; X[0, 1] = np.cos(th); X[1, 1] = np.sin(th) * (np.exp(0.7j) if cplx else 1.0)
        return A, X, np.array([d[0], d[0]]), "zero_residual_degenerate", True
    if kind == 3:   # repeated eigenvalues, g = n, random non-orthogonal X
        n = int(rng.integers(2, 6))
        d = rng.choice([-1.0, 0.0, 1.0], size=n)
        U = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0))[0]
        A = U @ np.diag(d) @ U.conj().T
        X = U + 0.2 * rng.standard_normal((n, n))
        return A, X, d + 0.1 * rng.standard_normal(n), "repeated_square", True
    if kind == 4:   # scale 1e150 / 1e-150
        n = int(rng.integers(2, 6)); g = int(rng.integers(1, n + 1))
        s = 1e150 if i % 2 else 1e-150
        A = hermitian(rng, n, cplx, s)
        X = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0)
        X /= np.linalg.norm(X, axis=0)
        mu = rng.standard_normal(g) * s
        return A, X, mu, f"scale({s:g})", True
    if kind == 5:   # subnormal operator entries
        n = int(rng.integers(2, 5)); g = int(rng.integers(1, n + 1))
        A = hermitian(rng, n, cplx, 2.0 ** -1060)
        X = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0)
        X /= np.linalg.norm(X, axis=0)
        mu = rng.standard_normal(g) * 2.0 ** -1060
        return A, X, mu, "subnormal_operator", True
    if kind == 6:   # g = 1, unnormalised column
        n = int(rng.integers(1, 6))
        A = hermitian(rng, n, cplx)
        X = (rng.standard_normal((n, 1)) + (1j * rng.standard_normal((n, 1)) if cplx else 0.0)) * rng.choice([0.3, 1.0, 1.4])
        return A, X, rng.standard_normal(1), "g_eq_1", True
    if kind == 7:   # Lambda far from the Rayleigh quotient (large residual), orthonormal X
        n = int(rng.integers(2, 6)); g = int(rng.integers(1, n + 1))
        A = hermitian(rng, n, cplx)
        X = np.linalg.qr(rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0))[0]
        return A, X, rng.standard_normal(g) * 50, "far_lambda_orthonormal", True
    if kind == 8:   # A = 0 and A = cI
        n = int(rng.integers(2, 6)); g = int(rng.integers(1, n + 1))
        c = 0.0 if i % 2 else float(rng.standard_normal())
        A = np.eye(n) * c
        X = rng.standard_normal((n, g)) * rng.choice([0.5, 1.0])
        return A, X, rng.standard_normal(g), f"scalar_operator(c={c:.3g})", True
    if kind == 9:   # columns with wildly different norms (eta from the diagonal)
        n = int(rng.integers(2, 6)); g = int(rng.integers(2, n + 1))
        A = hermitian(rng, n, cplx)
        X = np.linalg.qr(rng.standard_normal((n, g)))[0] * (1 + rng.uniform(-0.9, 0.9, size=g))
        return A, X, rng.standard_normal(g), "column_scales", True
    # kind == 10: mu all equal, X spans an eigenspace of a 3-fold level exactly
    n = int(rng.integers(3, 7))
    d = rng.standard_normal(n); d[0] = d[1] = d[2] = 0.5
    A = np.diag(d)
    X = np.zeros((n, 3)); X[:3, :3] = np.linalg.qr(rng.standard_normal((3, 3)))[0] @ np.diag([1.0, 0.8, 1.2])
    return A, X, np.array([0.5, 0.5, 0.5]), "threefold_level", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_lemma = Tally("polar-factor lemma (a)-(d)")
    T_kahan = Tally("Kahan: bottleneck distance <= ||A Qhat - Qhat Lambda||")
    T_thm2 = Tally("Theorem 2: bottleneck distance <= rho")
    T_code = Tally("implemented rho, eta, radius >= exact (and radius >= matching distance)")
    T_sub = Tally("_subspace_shift >= exact rho (its inputs taken as exact)")
    T_neg = Tally("negative control: eta >= 1 -> no distinct matching claimed, code returns None")
    for i in range(a.cases):
        A, X, mu, tag, adv = cases(rng, i)
        A = np.asarray(A); X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        n, g = X.shape
        Am, Xm = to_mp(A), to_mp(X)
        G = herm(Xm) * Xm
        Id = mp.eye(g)
        eta = spec_norm(G - Id)
        lam, _ = eig_herm(Am)
        mus = [mpf(m) for m in mu]
        Lm = mp.diag(mus)
        R = Am * Xm - Xm * Lm
        nR, nA, nL = spec_norm(R), spec_norm(Am), max(abs(m) for m in mus)
        if not eta < 1:
            T_lemma.skip()
            cb = _cluster_bound(OperatorAdapter(A), np.asarray(mu, float), X,
                                residuals(OperatorAdapter(A), np.asarray(mu, float), X), list(range(g)))
            T_neg.add(cb is None, adv, f"{tag}: eta={mp.nstr(eta, 8)} but code returned {cb}")
            continue
        delta = 1 - mp.sqrt(1 - eta)
        rho = nR + (nA + nL) * delta
        M = herm_fun(G, mp.sqrt)
        Minv = herm_fun(G, lambda v: 1 / mp.sqrt(v))
        Q = Xm * Minv
        okl = (spec_norm(herm(Q) * Q - Id) <= SLACK
               and spec_norm(M - Id) <= delta + SLACK
               and spec_norm(Minv) <= 1 / mp.sqrt(1 - eta) + SLACK
               and spec_norm(Xm - Q) <= delta + SLACK)
        rQ = spec_norm(Am * Q - Q * Lm)
        okl = okl and rQ <= rho + SLACK * (1 + nA + nL)
        T_lemma.add(okl, adv, f"{tag}: lemma parts fail, eta={mp.nstr(eta, 8)}")
        d = bottleneck_distance(mus, lam)
        T_kahan.add(d <= rQ + SLACK * (1 + nA + nL), adv, f"{tag}: d={mp.nstr(d, 12)} rQ={mp.nstr(rQ, 12)}")
        T_thm2.add(d <= rho + SLACK * (1 + nA + nL), adv, f"{tag}: d={mp.nstr(d, 12)} rho={mp.nstr(rho, 12)}")
        ad = OperatorAdapter(A)
        res = residuals(ad, np.asarray(mu, float), X)
        cb = _cluster_bound(ad, np.asarray(mu, float), X, res, list(range(g)))
        if cb is None:
            T_code.skip()
        else:
            radius, eta_c, rho_c, info = cb
            ok = (mpf(eta_c) >= eta and mpf(rho_c) >= rho and mpf(radius) >= d
                  and mpf(info["eps_star"]) >= d and radius == min(rho_c, info["eps_star"]))
            T_code.add(ok, adv, f"{tag}: eta_c={eta_c} eta={mp.nstr(eta, 12)} rho_c={rho_c} rho={mp.nstr(rho, 12)} "
                                f"radius={radius} d={mp.nstr(d, 12)} eps*={info['eps_star']}")
        # _subspace_shift with exact-valued float inputs rounded up (eta, ||R|| as the stored floats)
        iso = float(mp.nstr(eta, 17, strip_zeros=False)); iso = iso if mpf(iso) >= eta else np.nextafter(iso, 2.0)
        inv = float(mp.nstr(nR, 17, strip_zeros=False)); inv = inv if mpf(inv) >= nR else np.nextafter(inv, np.inf)
        e = _subspace_shift(iso, inv, inf_norm_upper(A), float(nL) if mpf(float(nL)) >= nL else np.nextafter(float(nL), np.inf))
        if e is None:
            T_sub.skip()
        else:
            T_sub.add(mpf(e) >= mpf(inv) + (mpf(inf_norm_upper(A)) + nL) * (1 - mp.sqrt(1 - mpf(iso))), adv,
                      f"{tag}: e={e}")
    # duplicate-column necessity of eta < 1: x twice, exact eigenvector, R = 0, but only one eigenvalue index
    for k in range(20):
        n = 3
        A = np.diag([0.0, 1.0 + k, 2.0 + 2 * k])
        X = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        Am, Xm = to_mp(A), to_mp(X)
        lam, _ = eig_herm(Am)
        d = bottleneck_distance([mpf(0), mpf(0)], lam)
        cb = _cluster_bound(OperatorAdapter(A), np.zeros(2), X, residuals(OperatorAdapter(A), np.zeros(2), X), [0, 1])
        T_neg.add(d > 0 and cb is None, True, f"dup column k={k}: d={d} cb={cb}")
    sys.exit(1 if report([T_lemma, T_kahan, T_thm2, T_code, T_sub, T_neg]) else 0)


if __name__ == "__main__":
    main()
