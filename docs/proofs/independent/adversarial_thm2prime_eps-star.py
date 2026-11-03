"""Independent adversarial search for Theorem 2' (docs/proofs/thm2prime_eps-star.md).

Exact ground truth (mpmath, 60 digits) for the stored binary64 numbers: eps* exactly, ||X^H R||, ||Rc - R||_F
(Rc = the computed residual columns), ||X||_2, and the bottleneck matching distance. Checks the theorem
(distance <= eps*) and every intermediate bound of enclosure._cluster_bound in the direction it claims, with the
hypothesis boundaries: eta -> 1, R = 0 with spread = 0 (eps* = 0 exactly), large ||X^H R|| (Lambda far from the
Rayleigh quotient), g = n, g = 1, complex data, scale 1e+-150, and operator/vector entries at the subnormal scale
(where the additive underflow terms of the model are real).

Usage: python docs/proofs/independent/adversarial_thm2prime_eps-star.py [--cases 400] [--seed 5]
"""
import argparse
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, bottleneck_distance, eig_herm, fro_norm, herm, mpf, report, spec_norm, to_mp  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, _cluster_bound, residuals  # noqa: E402
from certify_ed.fp import norm2_upper  # noqa: E402

SLACK = mp.mpf("1e-45")


def make(rng, i):
    cplx = i % 3 == 0
    kind = i % 10
    def rnd(*shape):
        return rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0.0)
    def hermitian(n, scale=1.0):
        M = rnd(n, n)
        return (M + M.conj().T) / 2 * scale
    if kind == 0:
        n = int(rng.integers(2, 7)); g = int(rng.integers(1, n + 1))
        X = rnd(n, g); X /= np.linalg.norm(X, axis=0)
        return hermitian(n), X, rng.standard_normal(g), "random", False
    if kind == 1:   # eta -> 1 (Frobenius eta below 1: columns at angle with cos = c, Frobenius sqrt(2) c < 1)
        n = int(rng.integers(2, 6)); c = 0.7 - 2.0 ** -int(rng.integers(3, 40))
        X = np.zeros((n, 2), dtype=complex if cplx else float)
        X[0, 0] = 1.0; X[0, 1] = c; X[1, 1] = np.sqrt(1 - c * c)
        A = hermitian(n)
        return A, X, np.array([A[0, 0].real, A[1, 1].real]), f"eta_near_1(c={c!r})", True
    if kind == 2:   # exact eigenvectors of one degenerate level, non-orthogonal: eps* should be ~0, rho not
        n = int(rng.integers(3, 7)); d = rng.standard_normal(n); d[0] = d[1] = d[2] = float(rng.standard_normal())
        A = np.diag(d).astype(complex if cplx else float)
        X = np.zeros((n, 3), dtype=A.dtype)
        X[:3, :3] = np.linalg.qr(rnd(3, 3))[0] @ np.diag([1.0, 0.7, 1.3])
        return A, X, np.array([d[0]] * 3), "degenerate_exact_vectors", True
    if kind == 3:   # orthonormal X, Lambda far from the Rayleigh quotient (||X^H R|| large)
        n = int(rng.integers(2, 6)); g = int(rng.integers(1, n + 1))
        X = np.linalg.qr(rnd(n, g))[0]
        return hermitian(n), X, rng.standard_normal(g) * 100, "far_lambda", True
    if kind == 4:   # g = n
        n = int(rng.integers(2, 6))
        X = rnd(n, n) * 0.3 + np.eye(n)
        return hermitian(n), X, rng.standard_normal(n), "square", True
    if kind == 5:   # g = 1, scaled column
        n = int(rng.integers(1, 6))
        return hermitian(n), rnd(n, 1) * rng.choice([0.4, 1.0, 1.3]), rng.standard_normal(1), "g_eq_1", True
    if kind == 6:   # scale 1e+-150
        n = int(rng.integers(2, 6)); g = int(rng.integers(1, n + 1)); s = 1e150 if i % 2 else 1e-150
        X = rnd(n, g); X /= np.linalg.norm(X, axis=0)
        return hermitian(n, s), X, rng.standard_normal(g) * s, f"scale({s:g})", True
    if kind == 7:   # subnormal operator and claimed values: the additive underflow terms are real here
        n = int(rng.integers(2, 5)); g = int(rng.integers(1, n + 1)); s = 2.0 ** -int(rng.integers(1030, 1070))
        X = rnd(n, g); X /= np.linalg.norm(X, axis=0)
        return hermitian(n, s), X, rng.standard_normal(g) * s, f"subnormal(2^{int(np.log2(s))})", True
    if kind == 8:   # subnormal eigenvector entries with a normal operator
        n = int(rng.integers(2, 5)); g = int(rng.integers(1, n + 1))
        X = rnd(n, g) * 2.0 ** -1040
        return hermitian(n), X, rng.standard_normal(g), "subnormal_vectors", True
    # kind 9: repeated claimed values, mu = eigenvalue exactly, X spanning two levels (spread 0, R != 0)
    n = int(rng.integers(3, 7)); d = rng.standard_normal(n); d[0] = 1.0; d[1] = 1.0
    A = np.diag(d); X = np.zeros((n, 2)); X[0, 0] = 1.0; X[1, 1] = 1.0; X[2, 1] = 0.3
    return A, X, np.array([1.0, 1.0]), "spread_zero_residual_nonzero", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_thm = Tally("Theorem 2': bottleneck matching distance <= exact eps*")
    T_parts = Tally("intermediate bounds of _cluster_bound >= exact (xnorm, Rerr, XR, inv, inv_sqrt, spread)")
    T_eps = Tally("implemented eps_star >= exact eps* for the stored numbers; radius = min(rho, eps_star)")
    T_rem = Tally("remark: degenerate exact vectors give eps* < rho; far Lambda gives rho < eps*")
    for i in range(a.cases):
        A, X, mu, tag, adv = make(rng, i)
        n, g = X.shape
        Am, Xm = to_mp(A), to_mp(X)
        G = herm(Xm) * Xm
        eta = spec_norm(G - mp.eye(g))
        if not eta < 1:
            T_thm.skip(); continue
        mus = [mpf(m) for m in mu]
        R = Am * Xm - Xm * mp.diag(mus)
        nR = spec_norm(R); XR = spec_norm(herm(Xm) * R)
        delta = 1 - mp.sqrt(1 - eta); spread = max(mus) - min(mus)
        eps = nR / mp.sqrt(1 - eta) + XR / (1 - eta) + delta * spread / mp.sqrt(1 - eta)
        lam, _ = eig_herm(Am)
        d = bottleneck_distance(mus, lam)
        scale = 1 + spec_norm(Am) + max(abs(m) for m in mus)
        T_thm.add(d <= eps + SLACK * scale, adv, f"{tag}: d={mp.nstr(d, 12)} eps*={mp.nstr(eps, 12)}")
        ad = OperatorAdapter(A)
        res = residuals(ad, np.asarray(mu, float), X)
        cb = _cluster_bound(ad, np.asarray(mu, float), X, res, list(range(g)))
        if cb is None:
            T_eps.skip(); continue
        radius, eta_c, rho_c, info = cb
        # intermediate quantities, recomputed from the report exactly as the code forms them
        Rc = res.residual_columns(list(range(g)), np.asarray(X))
        Rcm = to_mp(Rc)
        Rerr_exact = fro_norm(Rcm - R)
        Rerr_code = norm2_upper(np.asarray(res.E_norm[:g], dtype=float))
        xnorm_exact = spec_norm(Xm)
        okp = (mpf(info["XR"]) >= XR and mpf(info["R_fro"]) >= nR and mpf(info["spread"]) >= spread
               and mpf(info["delta"]) >= delta and mpf(eta_c) >= eta)
        ok_rerr = mpf(Rerr_code) >= Rerr_exact
        T_parts.add(okp and ok_rerr, adv, f"{tag}: XR_code={info['XR']} XR={mp.nstr(XR, 12)} Rerr_code={Rerr_code} "
                                           f"Rerr_exact={mp.nstr(Rerr_exact, 12)} xnorm={mp.nstr(xnorm_exact, 12)} ok_rerr={ok_rerr}")
        # eps* with the code's eta (>= exact eta): the code bounds eps*(eta_code) >= eps*(eta) by monotonicity
        oke = mpf(info["eps_star"]) >= eps and radius == min(rho_c, info["eps_star"]) and mpf(radius) >= d
        T_eps.add(oke, adv, f"{tag}: eps_code={info['eps_star']} eps*={mp.nstr(eps, 12)} radius={radius} d={mp.nstr(d, 12)}")
        if tag == "degenerate_exact_vectors":
            T_rem.add(info["eps_star"] < rho_c and info["used"] == "eps_star", True, f"{tag}: eps*={info['eps_star']} rho={rho_c}")
        if tag == "far_lambda":
            T_rem.add(rho_c <= info["eps_star"] and info["used"] == "rho", True, f"{tag}: eps*={info['eps_star']} rho={rho_c}")
    sys.exit(1 if report([T_thm, T_parts, T_eps, T_rem]) else 0)


if __name__ == "__main__":
    main()
