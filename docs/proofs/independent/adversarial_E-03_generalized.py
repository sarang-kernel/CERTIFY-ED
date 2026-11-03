"""Independent adversarial check of E-03 (generalized eigenproblems), written from the statement only.

Ground truth: mp at 60 digits (Cholesky of B, C = L^-1 A L^-H, eigenvalues of C = pencil eigenvalues) and exact
rationals for the floating-point bounds of Proposition 2.
Part A: Proposition 2 -- S >= ||A x - mu B x||_2 and D_lo <= x^H B x (exact Fractions); r >= ||s||_{B^-1}/||x||_B (mp) and
an eigenvalue within r of mu (Lemma 2): random A, B (B = G^H G + shift, condition numbers up to 1e8), real/complex,
x random / near-eigenvectors (residual down to 1e-13) / scaled 1e-3..1e3, mu random or Rayleigh-like.
Part B: Proposition 1 -- beta_c <= lambda_min(B) (mp) for well- and ill-conditioned B; indefinite B refused.
Part C: Lemma 3 / Proposition 3 -- pencil_count_below: N(sigma - c) <= nu <= #{lambda <= sigma + c} at random shifts
(including shifts within 1e-9 of an eigenvalue); certify_generalized 'lowest' k = 1..3: G2 enclosures each contain a
pencil eigenvalue, G3 CERTIFIED implies the claimed are exactly the k lowest and lambda_{k+1} >= the recorded bound;
negative controls: a skipped eigenvalue (claim 1st and 3rd as the lowest two), a duplicated vector, a declared
lambda_min_B larger than the truth with an inconclusive B (hypothesis path never CERTIFIED).

Usage: python docs/proofs/independent/adversarial_E-03_generalized.py [--cases 80] [--seed 97]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, mpf, to_mp, herm, eig_herm, Tally, report  # noqa: E402
from certify_ed.generalized import certify_generalized, certify_B_positive_definite, pencil_residual_radii, pencil_count_below  # noqa: E402

mp.mp.dps = 60


def fc(z):
    z = complex(z)
    return (F(z.real), F(z.imag))


def rand_herm(rng, n, cplx, scale=1.0):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    return (A + A.conj().T) / 2 * scale


def rand_spd(rng, n, cplx, cond):
    G = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    Q, _ = np.linalg.qr(G)
    d = np.logspace(0, -np.log10(cond), n)
    B = (Q * d) @ Q.conj().T
    return (B + B.conj().T) / 2


def pencil_mp(A, B):
    """(eigenvalues sorted, L) of the pencil in mp: B = L L^H, C = L^-1 A L^-H."""
    Bm, Am = to_mp(B), to_mp(A)
    L = mp.cholesky(Bm)
    Li = mp.inverse(L)
    C = Li * Am * herm(Li)
    C = (C + herm(C)) / 2
    lam, _ = eig_herm(C)
    return lam, L, Li


def exact_residual_parts(A, B, mu, x):
    """Exact (Fractions): ||A x - mu B x||_2^2 and x^H B x for binary64 data."""
    n = x.size
    xf = [fc(v) for v in x]
    muf = F(float(mu))
    s2 = F(0); xBx = F(0)
    for r in range(n):
        ar = (F(0), F(0)); br = (F(0), F(0))
        for c in range(n):
            a = fc(A[r, c]); b = fc(B[r, c]); v = xf[c]
            ar = (ar[0] + a[0] * v[0] - a[1] * v[1], ar[1] + a[0] * v[1] + a[1] * v[0])
            br = (br[0] + b[0] * v[0] - b[1] * v[1], br[1] + b[0] * v[1] + b[1] * v[0])
        sr = (ar[0] - muf * br[0], ar[1] - muf * br[1])
        s2 += sr[0] ** 2 + sr[1] ** 2
        xBx += xf[r][0] * br[0] + xf[r][1] * br[1]
    return s2, xBx


def part_A(rng, cases, T_fl, T_r):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 8))
        cond = float(rng.choice([1.0, 1e2, 1e5, 1e8]))
        B = rand_spd(rng, n, cplx, cond)
        A = rand_herm(rng, n, cplx, float(rng.choice([1e-2, 1.0, 1e2])))
        lam, L, Li = pencil_mp(A, B)
        pd = certify_B_positive_definite(B)
        if not pd["ok"]:
            T_fl.skip(); continue
        beta_c = pd["beta_c"]
        kind = i % 4
        import scipy.linalg as la
        w, V = la.eigh(A, B)
        if kind == 1:
            j = int(rng.integers(0, n)); x = V[:, j] + rng.standard_normal(n) * float(rng.choice([1e-13, 1e-8, 1e-3])); mu = w[j]
        elif kind == 2:
            x = rng.standard_normal(n) * float(rng.choice([1e-3, 1e3])); mu = float(rng.uniform(w[0], w[-1]))
        else:
            x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0); mu = float(rng.uniform(w[0] - 1, w[-1] + 1))
        x = np.asarray(x, dtype=complex if cplx else float)
        rr = pencil_residual_radii(A, B, np.array([mu]), x[:, None], beta_c)
        s2, xBx = exact_residual_parts(A, B, mu, x)
        ok_fl = F(rr["residual_upper"][0]) ** 2 >= s2 and F(rr["xBx_lower"][0]) <= xBx
        T_fl.add(ok_fl, kind in (1, 2), f"A-fl n={n} kind={kind} cond={cond} S={rr['residual_upper'][0]:.3e} |s|={float(s2) ** 0.5:.3e} Dlo={rr['xBx_lower'][0]:.6e} xBx={float(xBx):.6e}")
        r = rr["radius"][0]
        if not np.isfinite(r):
            T_r.skip(); continue
        xm = to_mp(x.reshape(-1, 1)); sm = to_mp(A) * xm - mpf(mu) * to_mp(B) * xm
        num = mp.sqrt(mp.re((herm(Li * sm) * (Li * sm))[0, 0])); den = mp.sqrt(mp.re((herm(xm) * to_mp(B) * xm)[0, 0]))
        true_r = num / den
        ok = true_r <= mpf(r) and any(abs(l - mpf(mu)) <= mpf(r) for l in lam)
        T_r.add(ok, kind in (1, 2), f"A n={n} kind={kind} cond={cond} r={r:.3e} true={mp.nstr(true_r, 6)} min|lam-mu|={mp.nstr(min(abs(l - mpf(mu)) for l in lam), 6)}")


def part_B(rng, cases, T, T_neg):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 10))
        cond = float(rng.choice([1.0, 1e3, 1e8, 1e12, 1e15]))
        B = rand_spd(rng, n, cplx, cond) * float(rng.choice([1e-3, 1.0, 1e3]))
        lam, _ = eig_herm(to_mp(B))
        pd = certify_B_positive_definite(B)
        if pd.get("indefinite"):
            T.add(False, True, f"B n={n} cond={cond}: positive definite B refused as indefinite")
            continue
        if not pd["ok"]:
            T.skip(); continue
        T.add(mpf(pd["beta_c"]) <= lam[0], cond >= 1e8, f"B n={n} cond={cond} beta_c={pd['beta_c']:.6e} lam_min={mp.nstr(lam[0], 8)}")
        # negative control: an indefinite B
        Bn = B - np.eye(n) * float(lam[int(rng.integers(0, n))] * mpf("1.5"))
        pdn = certify_B_positive_definite(Bn)
        T_neg.add(not pdn["ok"], True, f"B-neg n={n}: indefinite B accepted ({pdn})")


def part_C(rng, cases, T_cnt, T_e2e, T_neg):
    import scipy.linalg as la
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(3, 8))
        B = rand_spd(rng, n, cplx, float(rng.choice([1.0, 1e3, 1e6])))
        A = rand_herm(rng, n, cplx)
        lam, L, Li = pencil_mp(A, B)
        pd = certify_B_positive_definite(B)
        if not pd["ok"]:
            T_cnt.skip(); continue
        beta_c = pd["beta_c"]
        # Lemma 3 at random shifts and at shifts next to eigenvalues
        for adv, sig in [(False, float(rng.uniform(float(lam[0]) - 1, float(lam[-1]) + 1))),
                         (True, float(lam[int(rng.integers(0, n))]) * (1 + 1e-9) + 1e-12)]:     # next to an eigenvalue
            c = pencil_count_below(A, B, sig, beta_c)
            if not c["conclusive"]:
                T_cnt.skip(); continue
            lo_s, hi_s = mpf(c["lower_shift"]), mpf(c["upper_shift"])
            N_lo = sum(1 for l in lam if l < lo_s); N_hi = sum(1 for l in lam if l <= hi_s)
            T_cnt.add(N_lo <= c["nu"] <= N_hi, adv, f"C-count n={n} sigma={sig!r} nu={c['nu']} N(lo)={N_lo} N(hi)={N_hi}")
        # end to end
        w, V = la.eigh(A, B)
        k = int(rng.integers(1, min(3, n - 1) + 1))
        X = V[:, :k] * (1 + rng.standard_normal(k) * 1e-9)
        rec = certify_generalized(A, B, w[:k], X, kind="lowest")
        st = {s["id"]: s for s in rec["statements"]}
        ok = True
        if st["G2.eigenpairs"]["status"] == "CERTIFIED":
            for e in rec["enclosures"]:
                ok = ok and any(mpf(e["lower"]) <= l <= mpf(e["upper"]) for l in lam)
        if st["G3.completeness"]["status"] == "CERTIFIED":
            ev = st["G3.completeness"]["evidence"]
            below = sum(1 for l in lam if l < mpf(ev["complete_below"]))
            ok = ok and below == k and lam[k] >= mpf(ev["next_eigenvalue_lower_bound"])
            # the k enclosures hold exactly lambda_1..lambda_k
            for j, e in enumerate(sorted(rec["enclosures"], key=lambda e: e["mu"])):
                ok = ok and mpf(e["lower"]) <= lam[j] <= mpf(e["upper"])
        T_e2e.add(ok, st["G3.completeness"]["status"] == "CERTIFIED", f"C-e2e n={n} k={k} G2={st['G2.eigenpairs']['status']} G3={st['G3.completeness']['status']}")
        # negative controls
        if n >= 3:
            Xs = V[:, [0, 2]]; recs = certify_generalized(A, B, w[[0, 2]], Xs, kind="lowest")
            g3 = [s for s in recs["statements"] if s["id"] == "G3.completeness"][0]
            T_neg.add(g3["status"] != "CERTIFIED", True, f"C-neg skipped eigenvalue n={n}: G3 {g3['status']}")
            Xd = V[:, [0, 0]]; recd = certify_generalized(A, B, w[[0, 0]], Xd, kind="lowest")
            g3 = [s for s in recd["statements"] if s["id"] == "G3.completeness"][0]
            T_neg.add(g3["status"] != "CERTIFIED", True, f"C-neg duplicated vector n={n}: G3 {g3['status']}")
        # hypothesis path: a nearly singular B where the inertia cannot certify, with a declared (too large) bound
        if i % 5 == 0:
            Bs = rand_spd(rng, n, cplx, 1e17)
            pds = certify_B_positive_definite(Bs)
            if not pds["ok"] and not pds.get("indefinite"):
                rech = certify_generalized(A, Bs, w[:1], V[:, :1], kind="lowest", lambda_min_B=1.0, lambda_min_B_source="wrong (test)")
                sts = {s["id"]: s["status"] for s in rech["statements"]}
                T_neg.add(all(v != "CERTIFIED" for v in sts.values()), True, f"C-neg hypothesis path: {sts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=80)
    ap.add_argument("--seed", type=int, default=97)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_fl = Tally("Proposition 2: S >= ||s||_2 and D_lo <= x^H B x (exact rationals)")
    T_r = Tally("Lemma 2 / Proposition 2: r >= ||s||_{B^-1}/||x||_B and an eigenvalue within r (mp)")
    T_B = Tally("Proposition 1: beta_c <= lambda_min(B) (mp; condition numbers up to 1e15)")
    T_Bn = Tally("negative control: indefinite B not certified positive definite")
    T_cnt = Tally("Lemma 3: N(sigma - c) <= nu <= #{lambda <= sigma + c} (mp)")
    T_e2e = Tally("certify_generalized 'lowest': G2 enclosures and G3 completeness vs mp pencil eigenvalues")
    T_neg = Tally("negative controls: skipped eigenvalue / duplicated vector / wrong declared lambda_min_B never CERTIFIED")
    part_A(rng, a.cases, T_fl, T_r)
    part_B(rng, a.cases, T_B, T_Bn)
    part_C(rng, a.cases, T_cnt, T_e2e, T_neg)
    report([T_fl, T_r, T_B, T_Bn, T_cnt, T_e2e, T_neg])


if __name__ == "__main__":
    main()
