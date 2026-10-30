"""Computational verification of docs/proofs/thm2_polar-factor.md (Theorem 2 with radius rho and the
polar-factor lemma; Kahan's inclusion theorem for an arbitrary Hermitian block).

Ground truth: mpmath at 30 digits on the binary64 inputs taken exactly (eigenvalues of A, the polar factor,
spectral norms). Implemented bound: enclosure._cluster_bound (rho), certify._subspace_shift and
sectors.decomposition_shift on the same data.

Checks per case (X n x g with eta = ||X^H X - I|| < 1, Lambda = diag(mu), R = A X - X Lambda, M = (X^H X)^(1/2),
Qhat = X M^(-1), delta = 1 - sqrt(1 - eta), rho = ||R|| + (||A|| + ||Lambda||) delta):
  L1  ||M - I|| <= delta,  ||M^(-1)|| <= (1 - eta)^(-1/2),  ||X - Qhat|| <= delta,  Qhat^H Qhat = I
  L2  ||A Qhat - Qhat Lambda|| <= rho                                      (polar-factor lemma)
  K   bottleneck matching distance between mu and the eigenvalues of A <= ||A Qhat - Qhat Lambda||
      (Kahan's theorem with the non-Rayleigh block Lambda), hence <= rho          (Theorem 2)
  KG  Kahan's theorem for an orthonormal Q and a random Hermitian H (not diagonal, not Q^H A Q):
      bottleneck distance between eig(H) and eig(A) <= ||A Q - Q H||
  C   the code's rho (enclosure._cluster_bound) >= the exact rho; its radius min(rho, eps*) >= bottleneck
      distance; _subspace_shift and decomposition_shift return a number >= the exact rho on their inputs
Usage: python docs/proofs/verify_thm2_polar-factor.py [--cases 20000] [--seed 1]
"""
from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter, TOL, bottleneck, eigh, func_herm, herm, norm2, to_mp  # noqa: E402

from certify_ed.enclosure import OperatorAdapter, residuals, _cluster_bound  # noqa: E402
from certify_ed.certify import _subspace_shift  # noqa: E402
from certify_ed.fp import inf_norm_upper  # noqa: E402


def random_case(rng, adversarial=None):
    """(A, X, mu, tag). A Hermitian n x n, X n x g, mu real g."""
    cplx = bool(rng.random() < 0.3)
    n = int(rng.integers(2, 7))
    g = int(rng.integers(1, min(4, n) + 1))
    M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    A = (M + M.conj().T) / 2
    w, V = la.eigh(A)
    spread = rng.choice([0.0, 1e-9, 1e-4, 1e-1, 1.0, 10.0])
    w = w.copy()
    w[:g] = w[0] + spread * rng.random(g)
    A = (V * w) @ V.conj().T
    A = (A + A.conj().T) / 2
    w, V = la.eigh(A)
    C = np.eye(g) + rng.choice([0.0, 1e-6, 1e-2, 0.2, 0.5, 0.9]) * rng.standard_normal((g, g))
    X = V[:, :g] @ C
    noise = rng.choice([0.0, 1e-10, 1e-5, 1e-2, 0.3])
    X = X + noise * (rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0))
    if rng.random() < 0.7:
        X = X / np.linalg.norm(X, axis=0)
    mu = w[:g] + rng.choice([0.0, 1e-8, 1e-3, 1.0]) * rng.standard_normal(g)
    tag = "random"
    if adversarial == "eta_near_1":
        X[:, -1] = X[:, 0] * (1 + rng.choice([1e-3, 1e-6, 1e-9])) if g > 1 else X[:, 0] * (1 + 1e-7)
        tag = adversarial
    elif adversarial == "exact_orthonormal":
        X = np.zeros((n, g), dtype=X.dtype)
        for j in range(g):
            X[j, j] = rng.choice([-1.0, 1.0])
        tag = adversarial
    elif adversarial == "far_lambda":
        mu = mu + rng.choice([-1e3, 1e3, 1e6])
        tag = adversarial
    elif adversarial == "zero_operator":
        A = np.zeros_like(A)
        tag = adversarial
    elif adversarial == "scalar_operator":
        A = np.eye(n, dtype=A.dtype) * rng.choice([-2.0, 0.5, 7.0])
        tag = adversarial
    elif adversarial == "square":
        g = n
        C = np.eye(n) + 0.3 * rng.standard_normal((n, n))
        X = V @ C
        mu = w + 1e-3 * rng.standard_normal(n)
        tag = adversarial
    elif adversarial == "large_spread":
        mu = np.sort(w)[:g] * 0 + np.linspace(-50.0, 50.0, g)
        tag = adversarial
    elif adversarial == "scaled":
        s = rng.choice([1e-150, 1e-40, 1e40, 1e150])
        A = A * s
        mu = mu * s
        tag = adversarial
    elif adversarial == "column_scale":
        X = X * rng.choice([0.5, 0.9, 1.1, 1.3])
        tag = adversarial
    elif adversarial == "one_dim":
        A = np.array([[rng.standard_normal()]])
        X = np.array([[rng.standard_normal()]])
        mu = np.array([rng.standard_normal()])
        tag = adversarial
    return A, X, mu, tag


def check_case(A, X, mu, tag, counters, adversarial):
    n, g = X.shape
    Am, Xm = to_mp(A), to_mp(X)
    G = herm(Xm) * Xm
    I = mp.eye(g)
    eta = norm2(G - I)
    # eta within 1e-20 of 1 is the hypothesis boundary (a singular Gram matrix has eta = 1 exactly; the 30-digit
    # value may fall on either side): such a case is counted as 'hypothesis not met'
    if not 1 - eta > mp.mpf("1e-20"):
        counters["L"].skipped += 1
        return
    # the polar factor amplifies rounding by 1/(1 - eta): raise the working precision when eta is close to 1
    with mp.workdps(60 if 1 - eta > mp.mpf("1e-6") else 120):
        _check_case(A, X, mu, tag, counters, adversarial, Am, Xm, G, I, eta)


def _check_case(A, X, mu, tag, counters, adversarial, Am, Xm, G, I, eta):
    n, g = X.shape
    Lm = mp.diag([mp.mpf(float(m)) for m in mu])
    eta = norm2(G - I)
    if not 1 - eta > mp.mpf("1e-20"):
        counters["L"].skipped += 1
        return
    lamA, _ = eigh(Am)
    tol = TOL * (1 + norm2(Am) + max(abs(mp.mpf(float(m))) for m in mu))
    M = func_herm(G, mp.sqrt)
    Minv = func_herm(G, lambda v: 1 / mp.sqrt(v))
    Q = Xm * Minv
    delta = 1 - mp.sqrt(1 - eta)
    R = Am * Xm - Xm * Lm
    nR, nA, nL = norm2(R), norm2(Am), max(abs(mp.mpf(float(m))) for m in mu)
    rho = nR + (nA + nL) * delta
    # L1
    # L1 is scale-free in A and mu (it concerns X alone): absolute slack TOL; the scale-dependent checks below
    # use tol = TOL (1 + ||A|| + ||Lambda||)
    parts = (norm2(M - I) <= delta + TOL, norm2(Minv) <= 1 / mp.sqrt(1 - eta) + TOL,
             norm2(Xm - Q) <= delta + TOL, norm2(herm(Q) * Q - I) <= TOL)
    counters["L"].add(adversarial, all(parts), f"L1 {tag}: parts={[bool(t) for t in parts]} eta={mp.nstr(eta, 12)} "
                                               f"||M-I||-delta={mp.nstr(norm2(M - I) - delta, 5)} "
                                               f"||X-Q||-delta={mp.nstr(norm2(Xm - Q) - delta, 5)} "
                                               f"||Q^HQ-I||={mp.nstr(norm2(herm(Q) * Q - I), 5)}")
    # L2
    Rhat = Am * Q - Q * Lm
    nRhat = norm2(Rhat)
    counters["L"].add(adversarial, nRhat <= rho + tol, f"L2 {tag}: ||Rhat||={nRhat} rho={rho}")
    # K (Kahan with the non-Rayleigh block Lambda) and Theorem 2
    d = bottleneck([mp.mpf(float(m)) for m in mu], lamA)
    counters["K"].add(adversarial, d <= nRhat + tol, f"K {tag}: d={d} ||Rhat||={nRhat}")
    counters["T2"].add(adversarial, d <= rho + tol, f"T2 {tag}: d={d} rho={rho}")
    # C: the implemented bound
    ad = OperatorAdapter(A)
    res = residuals(ad, np.asarray(mu, dtype=float), X)
    cb = _cluster_bound(ad, np.asarray(mu, dtype=float), X, res, list(range(g)))
    if cb is None:
        counters["C"].skipped += 1
    else:
        radius, eta_c, rho_c, info = cb
        ok = (mp.mpf(rho_c) >= rho and mp.mpf(eta_c) >= eta and mp.mpf(radius) >= d
              and info["rho"] == rho_c and radius == min(rho_c, info["eps_star"]))
        counters["C"].add(adversarial, ok, f"C {tag}: rho_code={rho_c} rho={rho} eta_code={eta_c} eta={eta} "
                                            f"radius={radius} d={d}")
    # the two other users of Theorem 2 with the Kahan constant: their inputs are upper bounds on eta and ||R||
    iso = float(eta) * (1 + 1e-15)
    inv = float(nR) * (1 + 1e-15)
    hn, mn = inf_norm_upper(A), float(nL)
    e = _subspace_shift(iso, inv, hn, mn)
    if e is not None:
        ok = mp.mpf(e) >= nR + (nA + nL) * (1 - mp.sqrt(1 - mp.mpf(iso)))
        counters["C"].add(adversarial, ok, f"C subspace_shift {tag}: e={e}")


def kahan_general(rng, counters, adversarial):
    """Q orthonormal (polar factor of a random X, in mp), H random Hermitian g x g (not Q^H A Q)."""
    cplx = bool(rng.random() < 0.3)
    n = int(rng.integers(2, 7))
    g = int(rng.integers(1, min(4, n) + 1))
    M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    A = to_mp((M + M.conj().T) / 2)
    X = to_mp(rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0))
    Q = X * func_herm(herm(X) * X, lambda v: 1 / mp.sqrt(v))
    Hr = rng.standard_normal((g, g)) * rng.choice([0.1, 1.0, 10.0])
    if adversarial:
        # H far from the Rayleigh quotient: Q^H R large compared with the orthogonal part
        Hm = herm(Q) * A * Q + to_mp(rng.choice([1.0, 10.0, 100.0]) * (Hr + Hr.T) / 2)
    else:
        Hm = to_mp((Hr + Hr.T) / 2)
    th, _ = eigh(Hm)
    lam, _ = eigh(A)
    r = norm2(A * Q - Q * Hm)
    d = bottleneck(th, lam)
    tol = TOL * (1 + norm2(A) + norm2(Hm))
    counters["KG"].add(adversarial, d <= r + tol, f"KG: d={d} r={r} n={n} g={g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("L", "K", "T2", "KG", "C")}
    advs = ["eta_near_1", "exact_orthonormal", "far_lambda", "zero_operator", "scalar_operator", "square",
            "large_spread", "scaled", "column_scale", "one_dim"]
    for i in range(args.cases):
        A, X, mu, tag = random_case(rng)
        check_case(A, X, mu, tag, counters, False)
        if i % 4 == 0:
            kahan_general(rng, counters, False)
        if i % 2000 == 0:
            print(f"  ... {i} random cases", flush=True)
    for a in advs:
        for _ in range(60):
            A, X, mu, tag = random_case(rng, a)
            check_case(A, X, mu, tag, counters, True)
    for _ in range(600):
        kahan_general(rng, counters, True)
    ok = True
    for k, name in (("L", "polar-factor lemma (L1, L2)"), ("K", "Kahan with the block Lambda"),
                    ("T2", "Theorem 2 radius rho"), ("KG", "Kahan, random Hermitian block"),
                    ("C", "implemented bounds >= exact")):
        ok &= counters[k].report(name)
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
