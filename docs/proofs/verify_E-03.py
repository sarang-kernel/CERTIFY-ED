"""Computational verification of docs/proofs/E-03_generalized.md (generalized eigenproblems A x = lambda B x).

Ground truth: mpmath at 60 digits -- Cholesky B = L L^H, C = L^-1 A L^-H, its eigenvalues (= the pencil's), the exact
B^-1-norm of the residual and the B-norm of x. Parts:
  A  G1: beta_c <= lambda_min(B); G2: r_i >= ||A x - mu B x||_{B^-1} / ||x||_B and an eigenvalue within r_i of mu_i
     (random pencils n = 3..8, real and complex, B conditioning 1 .. 1e8, claimed pairs exact or perturbed 1e-12..1e-2)
  B  G3: when CERTIFIED, exactly k eigenvalues below complete_below and lambda_{k+1} >= next_eigenvalue_lower_bound
  C  adversarial: B indefinite is refused (no statement certified); a wrong eigenvalue keeps a valid radius and G3 is not
     CERTIFIED; swapped pairs: radii valid, G3 not CERTIFIED; near-singular B (lambda_min ~ 1e-10); NEGATIVE CONTROL: a
     false declared lambda_min(B) (10 x the true one, inertia disabled) must undercut the exact radius in some cases
     while the record says CERTIFIED_UNDER_HYPOTHESIS and names declared_lambda_min_B
Run from the repository root: python docs/proofs/verify_E-03.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _common import Counter, eigh, herm, to_mp  # noqa: E402

import certify_ed.generalized as G  # noqa: E402
from certify_ed.generalized import certify_generalized  # noqa: E402

mp.mp.dps = 60


def rand_pencil(rng, n, cplx, cond):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    A = A + A.conj().T
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
    d = np.logspace(0, -np.log10(cond), n)
    B = (Q * d) @ Q.conj().T
    B = (B + B.conj().T) / 2
    return A, B


def pencil_truth(A, B):
    Am, Bm = to_mp(A), to_mp(B)
    L = mp.cholesky(Bm)
    Linv = mp.inverse(L)
    C = Linv * Am * herm(Linv)
    C = (C + herm(C)) / 2
    lam, _ = eigh(C)
    lamB, _ = eigh(Bm)
    return lam, lamB, Bm, Am


def exact_radius(Am, Bm, mu, x):
    xm = to_mp(x)
    s = Am * xm - mp.mpf(mu) * (Bm * xm)
    Binv = mp.inverse(Bm)
    num = mp.sqrt(mp.re((herm(s) * Binv * s)[0]))
    den = mp.sqrt(mp.re((herm(xm) * Bm * xm)[0]))
    return num / den


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cA, cB, cC = Counter(), Counter(), Counter()
    g3_certified = 0
    neg_broken = 0
    neg_total = 0

    for case in range(a.cases):
        n = int(rng.integers(3, 9))
        cplx = bool(case % 3 == 1)
        cond = float(10.0 ** rng.integers(0, 9))
        A, B = rand_pencil(rng, n, cplx, cond)
        w, V = la.eigh(A, B)
        k = int(rng.integers(1, min(4, n)))
        X = V[:, :k].copy()
        mu = w[:k].copy()
        if case % 2:
            eps = float(10.0 ** rng.uniform(-12, -2))
            X = X + eps * (rng.standard_normal(X.shape) + (1j * rng.standard_normal(X.shape) if cplx else 0))
            mu = mu + eps * rng.standard_normal(k) * 0.1
        rec = certify_generalized(A, B, mu, X, kind="lowest")
        lam, lamB, Bm, Am = pencil_truth(A, B)
        g1 = stmt(rec, "G1.B_positive_definite")
        okA = g1["status"] == "CERTIFIED" and mp.mpf(g1["evidence"]["beta_c"]) <= lamB[0]
        cA.add(False, okA, f"case {case}: beta_c {g1['evidence'].get('beta_c')} vs lambda_min(B) {float(lamB[0])} ({g1['status']})")
        if g1["status"] != "CERTIFIED":
            continue
        for i, e in enumerate(rec["enclosures"]):
            r_ex = exact_radius(Am, Bm, mu[i], X[:, i])
            r_im = mp.mpf(e["radius"])
            near = min(abs(lv - mp.mpf(mu[i])) for lv in lam)
            cA.add(False, r_im >= r_ex and near <= r_im,
                   f"case {case} pair {i}: radius {e['radius']} < exact {float(r_ex)} or nearest eigenvalue at {float(near)}")
        g3 = stmt(rec, "G3.completeness")
        if g3["status"] == "CERTIFIED":
            g3_certified += 1
            cb = mp.mpf(g3["evidence"]["complete_below"])
            below = sum(1 for lv in lam if lv < cb)
            nlb = mp.mpf(g3["evidence"]["next_eigenvalue_lower_bound"])
            ok = below == k and (k >= n or lam[k] >= nlb)
            cB.add(False, ok, f"case {case}: G3 certified but {below} eigenvalues below {float(cb)} (k = {k}) or lambda_(k+1) < bound")

    # ---- C adversarial
    for case in range(max(8, a.cases // 4)):
        n = 6
        cplx = bool(case % 2)
        A, B = rand_pencil(rng, n, cplx, 10.0)
        w, V = la.eigh(A, B)
        lam, lamB, Bm, Am = pencil_truth(A, B)
        # B indefinite: refused
        Bi = B - 2 * float(np.linalg.eigvalsh(B)[-1]) * np.outer(V[:, 0], V[:, 0].conj()) / float(np.vdot(V[:, 0], V[:, 0]).real)
        Bi = (Bi + Bi.conj().T) / 2
        rec = certify_generalized(A, Bi, w[:2], V[:, :2], kind="lowest")
        cC.add(True, all(s["status"] == "NOT_CHECKED" and s["evidence"].get("reason_code") == "UNSUPPORTED_MODEL" for s in rec["statements"]),
               f"adv {case}: indefinite B not refused")
        # wrong eigenvalue
        mu = w[:3].copy()
        mu[1] += 0.3 * (w[3] - w[0])
        rec = certify_generalized(A, B, mu, V[:, :3], kind="lowest")
        ok = stmt(rec, "G3.completeness")["status"] != "CERTIFIED"
        for i, e in enumerate(rec["enclosures"]):
            r_ex = exact_radius(Am, Bm, mu[i], V[:, i])
            ok = ok and mp.mpf(e["radius"]) >= r_ex and min(abs(lv - mp.mpf(mu[i])) for lv in lam) <= mp.mpf(e["radius"])
        cC.add(True, ok, f"adv {case}: wrong eigenvalue: radius invalid or G3 certified")
        # swapped pairs
        rec = certify_generalized(A, B, w[:3], V[:, [1, 0, 2]], kind="lowest")
        ok = stmt(rec, "G3.completeness")["status"] != "CERTIFIED"
        for i, e in enumerate(rec["enclosures"]):
            r_ex = exact_radius(Am, Bm, w[i], V[:, [1, 0, 2]][:, i])
            ok = ok and mp.mpf(e["radius"]) >= r_ex
        cC.add(True, ok, f"adv {case}: swapped pairs: radius invalid or G3 certified")
        # near-singular B
        An, Bn = rand_pencil(rng, n, cplx, 1e10)
        wn, Vn = la.eigh(An, Bn)
        rec = certify_generalized(An, Bn, wn[:2], Vn[:, :2], kind="lowest")
        lamn, lamBn, Bmn, Amn = pencil_truth(An, Bn)
        g1 = stmt(rec, "G1.B_positive_definite")
        ok = True
        if g1["status"] == "CERTIFIED":
            ok = mp.mpf(g1["evidence"]["beta_c"]) <= lamBn[0]
            for i, e in enumerate(rec["enclosures"]):
                ok = ok and mp.mpf(e["radius"]) >= exact_radius(Amn, Bmn, wn[i], Vn[:, i])
        else:
            ok = all(s["status"] in ("INCONCLUSIVE", "NOT_CHECKED") for s in rec["statements"])
        cC.add(True, ok, f"adv {case}: near-singular B: {g1['status']} with an invalid bound")
        # negative control: false declared lambda_min(B) with the inertia path disabled
        saved = G.certify_B_positive_definite
        G.certify_B_positive_definite = lambda Bx, **kw: {"ok": False, "indefinite": False, "reason": "disabled (negative control)",
                                                          "inertia": None, "inertia_at_zero": None, "beta_c": None, "beta": None}
        try:
            false_beta = 10.0 * float(lamB[0])
            Xp = V[:, :2] + 1e-4 * rng.standard_normal(V[:, :2].shape)
            rec = certify_generalized(A, B, w[:2], Xp, kind="lowest", lambda_min_B=false_beta, lambda_min_B_source="WRONG")
        finally:
            G.certify_B_positive_definite = saved
        g1 = stmt(rec, "G1.B_positive_definite")
        g2 = stmt(rec, "G2.eigenpairs")
        named = (g1["status"] == "CERTIFIED_UNDER_HYPOTHESIS" and g1["hypotheses"][0]["name"] == "declared_lambda_min_B"
                 and g2["status"] == "CERTIFIED_UNDER_HYPOTHESIS" and g2["hypotheses"][0]["source"] == "WRONG")
        cC.add(True, named, f"neg {case}: the record does not name the false hypothesis")
        for i, e in enumerate(rec["enclosures"]):
            neg_total += 1
            if mp.mpf(e["radius"]) < exact_radius(Am, Bm, w[i], Xp[:, i]):
                neg_broken += 1

    ok = True
    ok &= cA.report("A G1 lower bound on lambda_min(B); G2 radii vs the exact B^-1-norm residual and the nearest eigenvalue")
    ok &= cB.report(f"B G3 completeness ({g3_certified} certified records checked against the exact spectrum)")
    ok &= cC.report("C adversarial: indefinite B, wrong eigenvalue, swapped pairs, near-singular B, false declared lambda_min(B) named")
    print(f"   negative control (declared lambda_min(B) = 10 x true, inertia disabled): radius below the exact B^-1-norm residual in "
          f"{neg_broken}/{neg_total} pairs")
    ok &= neg_broken > 0
    total_r = cA.random + cB.random + cC.random
    total_a = cA.adversarial + cB.adversarial + cC.adversarial
    print(f"TOTAL: {total_r} random + {total_a} adversarial cases; violations: {0 if ok else 'see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
