"""Computational verification of docs/proofs/E-05_variational.md (variational certificates).

Parts (exact rational arithmetic where the quantity is rational, mpmath at 60 digits otherwise):
  A  Rayleigh quotient: theta_lo <= theta_exact <= theta_hi for the stored vector (exact rationals), lambda_1 <= upper bound
  B  existence radius: r_impl^2 >= r_exact^2 = <psi,H^2 psi>/<psi,psi> - theta^2 (rational) and an eigenvalue within r
  C  Temple: lower <= lambda_1 with rho = fl_down(lambda_2) (hypothesis true), rho = lambda_2 - delta; adversarial
     two-level mixtures (equality case), psi orthogonal to the ground state (INCONCLUSIVE expected), complex data;
     NEGATIVE CONTROL: rho = lambda_3 > lambda_2 (hypothesis false) on two-level mixtures must break the bound
  D  MPS contraction: ||psi_hat - psi_exact||_2 <= contraction_error (exact rational contraction) and Lemma 5
  E  tiny norms: psi scaled by 2^-600 keeps every bound valid; 2^-1000 is reported as UNDERFLOW_RISK
Run from the repository root: python docs/proofs/verify_E-05.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _common import Counter, eigh, frac, to_mp  # noqa: E402

from certify_ed.variational import certify_variational, state_from_mps  # noqa: E402

mp.mp.dps = 60


def rand_herm(rng, n, cplx, scale=1.0):
    A = rng.standard_normal((n, n)) * scale
    if cplx:
        A = A + 1j * rng.standard_normal((n, n)) * scale
    A = A + A.conj().T
    return A


def exact_quadratic(H, x):
    """(<x,x>, <x,Hx>, <x,H^2 x>) as exact Fractions (real parts; H Hermitian)."""
    n = H.shape[0]
    cplx = np.iscomplexobj(H) or np.iscomplexobj(x)

    def fr(z):
        return (Fraction(float(np.real(z))), Fraction(float(np.imag(z)))) if cplx else (Fraction(float(z)), Fraction(0))

    Hf = [[fr(H[i, j]) for j in range(n)] for i in range(n)]
    xf = [fr(x[i]) for i in range(n)]

    def cmul(a, b):
        return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

    def cadd(a, b):
        return (a[0] + b[0], a[1] + b[1])

    def matvec(v):
        out = []
        for i in range(n):
            s = (Fraction(0), Fraction(0))
            for j in range(n):
                s = cadd(s, cmul(Hf[i][j], v[j]))
            out.append(s)
        return out

    def inner(a, b):       # a^H b, real part
        s = Fraction(0)
        for i in range(n):
            s += a[i][0] * b[i][0] + a[i][1] * b[i][1]
        return s

    Hx = matvec(xf)
    return inner(xf, xf), inner(xf, Hx), inner(Hx, Hx)


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cA, cB, cC, cD, cE = Counter(), Counter(), Counter(), Counter(), Counter()
    neg_control_broken = 0
    neg_control_total = 0

    for case in range(a.cases):
        n = int(rng.integers(2, 7))
        cplx = bool(case % 3 == 2)
        scale = float(10.0 ** rng.integers(-3, 4))
        H = rand_herm(rng, n, cplx, scale)
        lam, Q = eigh(to_mp(H))
        lam1, lam2 = lam[0], lam[1]
        # trial state: ground state plus a perturbation of random size, or random
        mode = case % 4
        v0 = np.array([complex(Q[i, 0]) for i in range(n)]) if cplx else np.array([float(mp.re(Q[i, 0])) for i in range(n)])
        if mode == 0:
            x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0)
        else:
            eps = float(10.0 ** rng.uniform(-8, -1))
            x = v0 + eps * (rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0))
        x = x * float(10.0 ** rng.integers(-2, 3))
        rho = float(mp.mpf(lam2))
        if mp.mpf(rho) > lam2:
            rho = float(np.nextafter(rho, -np.inf))
        rec = certify_variational(H, x, gap_hypothesis=rho, gap_source="verify: exact lambda_2 rounded down")
        xx, xHx, xH2x = exact_quadratic(H, x)
        theta = xHx / xx
        th = rec["theta"]
        okA = frac(th["lower"]) <= theta <= frac(th["upper"]) and mp.mpf(lam1) <= mp.mpf(stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"])
        cA.add(False, okA, f"case {case}: theta {float(theta)} not in [{th['lower']}, {th['upper']}] or lambda_1 above the bound")
        r2_exact = xH2x / xx - theta * theta
        r = frac(stmt(rec, "V2.existence")["evidence"]["radius"])
        iv = stmt(rec, "V2.existence")["evidence"]["interval"]
        has_eig = any(mp.mpf(iv[0]) <= lv <= mp.mpf(iv[1]) for lv in lam)
        cB.add(False, r * r >= r2_exact and has_eig, f"case {case}: r^2 {float(r * r)} < exact {float(r2_exact)} or no eigenvalue in the interval")
        s3 = stmt(rec, "V3.temple_lower")
        if s3["status"] == "CERTIFIED_UNDER_HYPOTHESIS":
            okC = mp.mpf(s3["evidence"]["lower_bound"]) <= lam1 and s3["hypotheses"][0]["name"] == "declared_gap"
            cC.add(False, okC, f"case {case}: Temple lower {s3['evidence']['lower_bound']} > lambda_1 {float(lam1)}")
        else:
            # Temple cannot apply when theta >= rho; the record must then be INCONCLUSIVE, never certified
            okC = s3["status"] == "INCONCLUSIVE" and frac(th["upper"]) >= Fraction(rho)
            cC.add(False, okC, f"case {case}: unexpected V3 status {s3['status']} with theta {float(theta)}, rho {rho}")

    # ---- C adversarial: two-level mixtures (Temple equality), orthogonal trial state, false hypothesis (negative control)
    for case in range(max(10, a.cases // 3)):
        n = int(rng.integers(3, 7))
        cplx = bool(case % 2)
        H = rand_herm(rng, n, cplx)
        lam, Q = eigh(to_mp(H))
        v = [np.array([complex(Q[i, k]) for i in range(n)]) if cplx else np.array([float(mp.re(Q[i, k])) for i in range(n)])
             for k in range(3)]
        ang = float(rng.uniform(0.05, 1.2))
        x = np.cos(ang) * v[0] + np.sin(ang) * v[1]
        rho_true = float(np.nextafter(float(lam[1]), -np.inf))
        rec = certify_variational(H, x, gap_hypothesis=rho_true, gap_source="exact")
        s3 = stmt(rec, "V3.temple_lower")
        if s3["status"] == "CERTIFIED_UNDER_HYPOTHESIS":
            lb = mp.mpf(s3["evidence"]["lower_bound"])
            # equality case: the exact Temple bound equals lambda_1 up to the floating-point slack of the vectors
            cC.add(True, lb <= lam[0] and lam[0] - lb < mp.mpf(1e-6) * (1 + abs(lam[0])),
                   f"adv {case}: two-level mixture lower {float(lb)} vs lambda_1 {float(lam[0])}")
        else:
            cC.add(True, False, f"adv {case}: two-level mixture not certified ({s3['status']})")
        # orthogonal to the ground state: theta >= lambda_2 >= rho -> INCONCLUSIVE
        y = np.cos(ang) * v[1] + np.sin(ang) * v[2]
        rec = certify_variational(H, y, gap_hypothesis=rho_true, gap_source="exact")
        s3 = stmt(rec, "V3.temple_lower")
        cC.add(True, s3["status"] == "INCONCLUSIVE" and "theta < rho" in s3["summary"],
               f"adv {case}: orthogonal trial state gave {s3['status']}")
        # negative control: hypothesis false (rho = lambda_3 > lambda_2) on the two-level mixture
        rho_false = float(lam[2])
        rec = certify_variational(H, x, gap_hypothesis=rho_false, gap_source="WRONG (negative control)")
        s3 = stmt(rec, "V3.temple_lower")
        neg_control_total += 1
        if s3["status"] == "CERTIFIED_UNDER_HYPOTHESIS":
            if mp.mpf(s3["evidence"]["lower_bound"]) > lam[0]:
                neg_control_broken += 1
            ok = s3["hypotheses"][0]["name"] == "declared_gap" and "HYPOTHESIS" in s3["summary"] and s3["status"] != "CERTIFIED"
            cC.add(True, ok, f"neg {case}: the record does not name the hypothesis")
        elif s3["status"] == "INCONCLUSIVE":
            pass    # theta >= rho_false: Temple not applied, nothing claimed

    # ---- D: MPS contraction (exact rational contraction of binary64 tensors)
    for case in range(max(10, a.cases // 3)):
        L = int(rng.integers(2, 6))
        d = 2
        D = int(rng.integers(1, 5))
        cplx = bool(case % 2)
        scale = float(10.0 ** rng.integers(-2, 3))

        def t(shape, scale=scale, cplx=cplx):
            A = rng.standard_normal(shape) * scale
            return A + 1j * rng.standard_normal(shape) * scale if cplx else A

        tens = [t((1, d, D))] + [t((D, d, D)) for _ in range(L - 2)] + [t((D, d, 1))]
        st = state_from_mps(tens)
        # exact contraction with Fractions (complex as pairs)

        def fr(z):
            return (Fraction(float(np.real(z))), Fraction(float(np.imag(z))))

        T = [[fr(tens[0][0, s, b]) for b in range(D)] for s in range(d)]      # list over rows (d^j) of D-vectors
        for A in tens[1:]:
            Dr = A.shape[2]
            newT = []
            for row in T:
                for s in range(d):
                    vec = []
                    for b in range(Dr):
                        re = Fraction(0)
                        im = Fraction(0)
                        for k in range(D):
                            ar, ai = fr(A[k, s, b])
                            re += row[k][0] * ar - row[k][1] * ai
                            im += row[k][0] * ai + row[k][1] * ar
                        vec.append((re, im))
                    newT.append(vec)
            T = newT
        psi_exact = [row[0] for row in T]
        err2 = sum((Fraction(float(np.real(st.vector[i]))) - psi_exact[i][0]) ** 2 +
                   (Fraction(float(np.imag(st.vector[i]))) - psi_exact[i][1]) ** 2 for i in range(len(psi_exact)))
        e = Fraction(st.contraction_error)
        cD.add(False, err2 <= e * e, f"mps {case}: ||psi_hat - psi||^2 = {float(err2)} > e^2 = {float(e * e)}")
        # Lemma 5 on a small Hermitian H: |theta(psi) - theta(psi_hat)| <= theta_source_vs_contracted
        n = d ** L
        H = rand_herm(rng, n, cplx)
        rec = certify_variational(H, st)
        dth = rec["state"].get("theta_source_vs_contracted")
        if dth is not None and np.isfinite(dth):
            psi_e = np.array([complex(float(p[0]), float(p[1])) for p in psi_exact])      # rounded only for the exact theta below
            xx, xHx, _ = exact_quadratic(H, st.vector)
            Hm = to_mp(H)
            pe = mp.matrix([mp.mpc(mp.mpf(p[0].numerator) / p[0].denominator, mp.mpf(p[1].numerator) / p[1].denominator) for p in psi_exact])
            th_src = mp.re((pe.transpose_conj() * Hm * pe)[0]) / mp.re((pe.transpose_conj() * pe)[0])
            th_hat = mp.mpf(xHx.numerator) / xHx.denominator / (mp.mpf(xx.numerator) / xx.denominator)
            cD.add(True, abs(th_src - th_hat) <= mp.mpf(dth), f"mps {case}: Lemma 5 |dtheta| = {float(abs(th_src - th_hat))} > {dth}")
            del psi_e

    # ---- E: tiny norms
    for case in range(10):
        n = 4
        H = rand_herm(rng, n, False)
        lam, Q = eigh(to_mp(H))
        x = (rng.standard_normal(n)) * 2.0 ** -450
        rec = certify_variational(H, x, gap_hypothesis=float(np.nextafter(float(lam[1]), -np.inf)), gap_source="exact")
        # entries at 2^-600: the squares underflow, the norm cannot be resolved: nothing may be CERTIFIED
        rec_mid = certify_variational(H, x * 2.0 ** -150)
        cE.add(True, all(s["status"] in ("NOT_CHECKED", "INCONCLUSIVE") for s in rec_mid["statements"][:3]),
               f"tiny {case}: 2^-600 entries gave a certified statement")
        xx, xHx, xH2x = exact_quadratic(H, x)
        theta = xHx / xx
        th = rec["theta"]
        ok = frac(th["lower"]) <= theta <= frac(th["upper"])
        r = frac(stmt(rec, "V2.existence")["evidence"]["radius"])
        ok = ok and r * r >= xH2x / xx - theta * theta
        s3 = stmt(rec, "V3.temple_lower")
        ok = ok and (s3["status"] != "CERTIFIED_UNDER_HYPOTHESIS" or mp.mpf(s3["evidence"]["lower_bound"]) <= lam[0])
        cE.add(True, ok, f"tiny {case}: a bound failed at scale 2^-600")
        x2 = x * 2.0 ** -500          # entries near 2^-950 < 2^-900
        rec2 = certify_variational(H, x2)
        cE.add(True, all(s["status"] == "INCONCLUSIVE" and s["evidence"].get("reason_code") == "UNDERFLOW_RISK"
                         for s in rec2["statements"][:3]), f"tiny {case}: 2^-1000 entries not reported as UNDERFLOW_RISK")

    ok = True
    ok &= cA.report("A Rayleigh quotient enclosure and upper bound")
    ok &= cB.report("B existence radius")
    ok &= cC.report("C Temple lower bound (hypothesis true, equality case, orthogonal state, false hypothesis named)")
    print(f"   negative control (rho = lambda_3 > lambda_2 declared): bound broken in {neg_control_broken}/{neg_control_total} two-level cases")
    ok &= neg_control_broken == neg_control_total
    ok &= cD.report("D MPS contraction bound and Lemma 5")
    ok &= cE.report("E tiny norms")
    total_r = cA.random + cB.random + cC.random + cD.random + cE.random
    total_a = cA.adversarial + cB.adversarial + cC.adversarial + cD.adversarial + cE.adversarial
    print(f"TOTAL: {total_r} random + {total_a} adversarial cases; violations: {0 if ok else 'see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
