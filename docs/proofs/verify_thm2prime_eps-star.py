"""Computational verification of docs/proofs/thm2prime_eps-star.md (Theorem 2', radius eps*).

Ground truth: mpmath at 30 digits on the binary64 inputs taken exactly. Implemented bound: enclosure._cluster_bound.

Checks per case (notation of the document; Hp = Qhat^H A Qhat, tau = ||X^H R||/(1 - eta) + delta spread/sqrt(1 - eta)):
  E1  ||A Qhat - Qhat Hp|| <= ||R|| / sqrt(1 - eta)
  E2  ||Hp - Lambda|| <= tau, and Hp - Lambda = M^-1 X^H R M^-1 + (M Lambda M^-1 - Lambda) to 1e-24
  E3  sorted pairing of eig(Hp) and eig(Lambda) within tau (Weyl)
  E4  bottleneck matching distance between mu and eig(A) <= eps*
  C   code: eps_star >= exact eps*, radius = min(rho, eps_star) >= bottleneck distance
Usage: python docs/proofs/verify_thm2prime_eps-star.py [--cases 20000] [--seed 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter, TOL, bottleneck, bottleneck_sorted, eigh, func_herm, herm, norm2, to_mp  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verify_thm2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_thm2_polar-factor.py"))
_thm2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thm2)
random_case = _thm2.random_case

from certify_ed.enclosure import OperatorAdapter, residuals, _cluster_bound  # noqa: E402


def extra_case(rng, kind):
    A, X, mu, _ = random_case(rng)
    if kind == "spread":
        g = mu.size
        mu = mu + np.linspace(0.0, rng.choice([10.0, 100.0]), g)
    elif kind == "ritz":
        Xn = X / np.linalg.norm(X, axis=0)
        mu = np.real(np.einsum("ij,ij->j", Xn.conj(), A @ Xn))
    return A, X, mu, kind


def check_case(A, X, mu, tag, counters, adversarial):
    Am, Xm = to_mp(A), to_mp(X)
    G = herm(Xm) * Xm
    I = mp.eye(X.shape[1])
    eta = norm2(G - I)
    # eta within 1e-20 of 1 is the hypothesis boundary (a singular Gram matrix has eta = 1 exactly; the 30-digit
    # value may fall on either side): such a case is counted as 'hypothesis not met'
    if not 1 - eta > mp.mpf("1e-20"):
        counters["E"].skipped += 1
        return
    with mp.workdps(60 if 1 - eta > mp.mpf("1e-6") else 120):
        _check_case(A, X, mu, tag, counters, adversarial, Am, Xm, G, I)


def _check_case(A, X, mu, tag, counters, adversarial, Am, Xm, G, I):
    n, g = X.shape
    mus = [mp.mpf(float(m)) for m in mu]
    Lm = mp.diag(mus)
    eta = norm2(G - I)
    if not 1 - eta > mp.mpf("1e-20"):
        counters["E"].skipped += 1
        return
    lamA, _ = eigh(Am)
    Minv = func_herm(G, lambda v: 1 / mp.sqrt(v))
    M = func_herm(G, mp.sqrt)
    Q = Xm * Minv
    delta = 1 - mp.sqrt(1 - eta)
    R = Am * Xm - Xm * Lm
    nR = norm2(R)
    tol = TOL * (1 + norm2(Am) + max(abs(m) for m in mus))   # ground truth relative to the operand scale
    nXR = norm2(herm(Xm) * R)
    spread = max(mus) - min(mus)
    s = mp.sqrt(1 - eta)
    tau = nXR / (1 - eta) + delta * spread / s
    eps_star = nR / s + tau
    Hp = herm(Q) * Am * Q
    # E1
    counters["E"].add(adversarial, norm2(Am * Q - Q * Hp) <= nR / s + tol, f"E1 {tag}")
    # E2
    lhs = Hp - Lm
    rhs = Minv * herm(Xm) * R * Minv + (M * Lm * Minv - Lm)
    counters["E"].add(adversarial, norm2(lhs - rhs) <= tol and norm2(lhs) <= tau + tol,
                      f"E2 {tag}: ||Hp-L||={norm2(lhs)} tau={tau}")
    # E3
    th, _ = eigh(Hp)
    counters["E"].add(adversarial, bottleneck_sorted(th, mus) <= tau + tol, f"E3 {tag}")
    # E4
    d = bottleneck(mus, lamA)
    counters["T"].add(adversarial, d <= eps_star + tol, f"E4 {tag}: d={d} eps*={eps_star}")
    # C
    ad = OperatorAdapter(A)
    res = residuals(ad, np.asarray(mu, dtype=float), X)
    cb = _cluster_bound(ad, np.asarray(mu, dtype=float), X, res, list(range(g)))
    if cb is None:
        counters["C"].skipped += 1
        return
    radius, eta_c, rho_c, info = cb
    ok = (mp.mpf(info["eps_star"]) >= eps_star and mp.mpf(radius) >= d
          and radius == min(rho_c, info["eps_star"]) and info["used"] in ("rho", "eps_star"))
    counters["C"].add(adversarial, ok, f"C {tag}: eps*_code={info['eps_star']} eps*={eps_star} radius={radius} d={d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("E", "T", "C")}
    for i in range(args.cases):
        A, X, mu, tag = random_case(rng)
        check_case(A, X, mu, tag, counters, False)
        if i % 2000 == 0:
            print(f"  ... {i} random cases", flush=True)
    advs = ["eta_near_1", "exact_orthonormal", "far_lambda", "zero_operator", "scalar_operator", "square",
            "large_spread", "scaled", "column_scale", "one_dim"]
    for a in advs:
        for _ in range(60):
            A, X, mu, tag = random_case(rng, a)
            check_case(A, X, mu, tag, counters, True)
    for kind in ("spread", "ritz"):
        for _ in range(100):
            A, X, mu, tag = extra_case(rng, kind)
            check_case(A, X, mu, tag, counters, True)
    ok = True
    for k, name in (("E", "steps (i)-(ii) inequalities E1-E3"), ("T", "Theorem 2' radius eps*"),
                    ("C", "implemented eps* >= exact, radius >= matching distance")):
        ok &= counters[k].report(name)
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
