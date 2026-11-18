"""Computational verification of docs/proofs/cor5_pairing.md (Corollary 5 and the sorted-pairing lemma).

P  (exact rationals) the sorted-pairing lemma, with the non-uniform-radius counterexample as a documented
   expectation (its failure count is reported, not a violation);
C  (the code) enclosure.enclose + certify._certify_lowest on random Hermitian matrices with planted clusters and
   a model shift Delta: when the result is CERTIFIED, the conclusion is checked against the eigenvalues (mpmath,
   60 digits) of A + E for a planted Hermitian E with ||E|| <= Delta; negative controls (a missing eigenvalue, a
   duplicated vector) must not be CERTIFIED.
Usage: python docs/proofs/verify_cor5_pairing.py [--cases 20000] [--seed 3]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter, TOL, eigh, to_mp  # noqa: E402

from certify_ed.certify import Options, _certify_lowest  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, enclose  # noqa: E402
from certify_ed.fp import add_up  # noqa: E402


def frac_rand(rng, lo=-10, hi=10, den=1000):
    return Fraction(int(rng.integers(lo * den, hi * den)), den)


def pairing_case(rng, uniform=True):
    m = int(rng.integers(1, 9))
    a = sorted(frac_rand(rng) for _ in range(m))
    if rng.random() < 0.2 and m > 1:
        a[1] = a[0]                                   # a tie
    r = [Fraction(int(rng.integers(0, 2000)), 1000)] * m
    if not uniform:
        r = [Fraction(int(rng.integers(0, 2000)), 1000) for _ in range(m)]
    perm = rng.permutation(m)
    b = [None] * m
    for j in range(m):
        t = Fraction(int(rng.integers(-1000, 1001)), 1000)       # in [-1, 1]
        b[perm[j]] = a[j] + r[j] * t
    bs = sorted(b)
    # radii attached to the sorted a (a is sorted already)
    return all(abs(bs[j] - a[j]) <= r[j] for j in range(m))


def planted_claim(rng, n, k, cluster_mix, noise, mu_pert, degenerate):
    M = rng.standard_normal((n, n))
    A = (M + M.T) / 2
    w, V = la.eigh(A)
    w = w.copy()
    if degenerate and k >= 2:
        j = int(rng.integers(0, k - 1))
        size = int(rng.integers(2, min(4, k - j) + 1))
        w[j:j + size] = w[j] + rng.choice([0.0, 1e-9, 1e-5]) * rng.random(size)
        A = (V * w) @ V.T
        A = (A + A.T) / 2
        w, V = la.eigh(A)
        C = np.eye(size) + cluster_mix * rng.standard_normal((size, size))
        X = V[:, :k].copy()
        X[:, j:j + size] = V[:, j:j + size] @ C
    else:
        X = V[:, :k].copy()
    X = X + noise * rng.standard_normal((n, k))
    mu = w[:k] + mu_pert * rng.standard_normal(k)
    return A, X, mu, w


def check_code(A, X, mu, Delta, E, counters, adversarial, expect_not_certified=False, tag=""):
    n = A.shape[0]
    k = mu.size
    enc = enclose(OperatorAdapter(A), mu, X)
    st, ev, window = _certify_lowest(A, np.sort(mu), enc, Options(), eps_shift=Delta)
    if expect_not_certified:
        counters["N"].add(adversarial, st != "CERTIFIED", f"negative control {tag} CERTIFIED")
        return
    if st != "CERTIFIED":
        counters["C"].skipped += 1
        return
    lam = eigh(to_mp(A + E))[0]
    order = np.argsort(mu, kind="stable")
    mus = np.sort(mu)
    rad = add_up(enc.radius[order], Delta)
    tol = TOL * (1 + float(np.abs(A).sum()))
    ok = True
    for j in range(k):
        ok &= abs(lam[j] - mp.mpf(float(mus[j]))) <= mp.mpf(float(rad[j])) + tol
    nb = mp.mpf(ev["next_eigenvalue_lower_bound"])
    ok &= sum(1 for v in lam if v < nb) == k
    ok &= lam[k] >= nb - tol if k < n else True
    if k < n:
        ok &= lam[k] - lam[k - 1] >= mp.mpf(ev["gap_lower_bound"]) - tol
    counters["C"].add(adversarial, ok, f"C {tag}: n={n} k={k} Delta={Delta} radii={rad.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("P", "C", "N")}
    # P: the lemma in exact rationals
    for _ in range(args.cases):
        counters["P"].add(False, pairing_case(rng, uniform=True), "sorted pairing, uniform r")
    nonuniform_fail = sum(0 if pairing_case(rng, uniform=False) else 1 for _ in range(2000))
    for _ in range(500):          # ties and r = 0
        m = int(rng.integers(1, 6))
        a = sorted([Fraction(int(rng.integers(0, 3)))] * m)
        counters["P"].add(True, pairing_case(rng, uniform=True) and (a == sorted(a)), "ties")
    # C: the code with a model shift
    n_code = max(200, args.cases // 50)
    for i in range(n_code):
        n = int(rng.integers(4, 10))
        k = int(rng.integers(1, n))
        Delta = float(rng.choice([0.0, 1e-12, 1e-8, 1e-4]))
        Em = rng.standard_normal((n, n))
        Em = (Em + Em.T) / 2
        E = Em / max(la.norm(Em, 2), 1e-300) * Delta * float(rng.choice([0.0, 0.5, 1.0]))
        A, X, mu, w = planted_claim(rng, n, k, cluster_mix=float(rng.choice([0.0, 1e-3, 0.3, 0.6])),
                                    noise=float(rng.choice([0.0, 1e-10, 1e-6])),
                                    mu_pert=float(rng.choice([0.0, 1e-9, 1e-6])), degenerate=bool(rng.random() < 0.6))
        check_code(A, X, mu, Delta, E, counters, False, tag=f"case {i}")
    # negative controls
    for _i in range(n_code // 2):
        n = int(rng.integers(4, 10))
        k = int(rng.integers(2, n))
        A, X, mu, w = planted_claim(rng, n, k + 1, 0.0, 0.0, 0.0, False)
        j = int(rng.integers(0, k))
        keep = [t for t in range(k + 1) if t != j]      # skips eigenvalue j < k: an unclaimed eigenvalue below the top
        check_code(A, X[:, keep], mu[keep], 0.0, np.zeros_like(A), counters, True, True, "missing eigenvalue")
        A, X, mu, w = planted_claim(rng, n, k, 0.0, 0.0, 0.0, False)
        X2 = X.copy()
        X2[:, 1] = X2[:, 0]
        mu2 = mu.copy()
        mu2[1] = mu2[0]
        check_code(A, X2, mu2, 0.0, np.zeros_like(A), counters, True, True, "duplicated vector")
    # informational: a planted E larger than the declared Delta
    wrong = 0
    for _i in range(100):
        n = int(rng.integers(4, 8))
        k = int(rng.integers(1, n))
        Em = rng.standard_normal((n, n))
        Em = (Em + Em.T) / 2
        E = Em / la.norm(Em, 2) * 1e-3
        A, X, mu, w = planted_claim(rng, n, k, 0.0, 0.0, 0.0, False)
        enc = enclose(OperatorAdapter(A), mu, X)
        st, ev, _ = _certify_lowest(A, np.sort(mu), enc, Options(), eps_shift=1e-12)
        if st == "CERTIFIED":
            lam = eigh(to_mp(A + E))[0]
            rad = add_up(enc.radius[np.argsort(mu, kind="stable")], 1e-12)
            wrong += any(abs(lam[j] - mp.mpf(float(np.sort(mu)[j]))) > mp.mpf(float(rad[j])) for j in range(k))
    ok = True
    for key, name in (("P", "sorted-pairing lemma (exact rationals)"), ("C", "Corollary 5 against mpmath eigenvalues of A + E"),
                      ("N", "negative controls not CERTIFIED")):
        ok &= counters[key].report(name)
    print(f"documented counterexample: non-uniform radii break the sorted pairing in {nonuniform_fail} of 2000 cases "
          f"(expected > 0)")
    print(f"informational: with ||E|| = 1e-3 > declared Delta = 1e-12 the conclusion fails for A + E in {wrong} of 100 "
          f"CERTIFIED cases (Delta is needed)")
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
