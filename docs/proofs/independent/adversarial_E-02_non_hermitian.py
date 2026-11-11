"""Independent adversarial check of E-02 (non-Hermitian eigenproblems), written from the statement (sections 1 and 3)
only.

Ground truth: mpmath at 60 digits for every quantity (the residuals of the stored numbers, ||I - W V||_F, the spectral
norms, the eigenvalues), with no use of fractions.Fraction or of the helpers of docs/proofs/verify_E-02.py; exact rational
comparison only for the separation test, where a ulp matters.
Part A: Propositions 5 to 7 end to end on targeted families -- perturbed Jordan blocks J_m(lam) + 2^-k e_m e_1^T
(eigenvalues on a circle of radius 2^(-k/m)), pairs of nearly parallel eigenvectors (cond_2(V) up to about 1e15, theta
close to 1), graded matrices diag(2^(-10 i)) S with mixed scales, complex claims of real matrices, and a supplied inverse
W that is deliberately wrong by a relative 1e-8 .. 1e-2 (theta then larger, beta must still bound ||V^-1||) or random
(theta >= 1: N2 INCONCLUSIVE). For every CERTIFIED statement: eps >= ||r||/||x||, theta >= ||I - W V||_F, beta >= ||V^-1||,
rho >= ||V^-1 R||, all eigenvalues in the discs, the count per group.
Part B: disc_groups at the boundary: centres placed at distance 2 rho (1 + k 2^-52), k = -3 .. 3, in the real, the
imaginary and a diagonal direction; every pair it puts in different groups must satisfy |mu_i - mu_j| > 2 rho exactly.
Part C: negative controls: a radius from the per-pair residuals (the Hermitian Theorem 1 radius) and the radius
||R||_F without beta, on the perturbed Jordan family; each must fail in some cases.
Part D: the opt-out path (underflow_check=False) with complex stored entries whose components are subnormal (their
moduli are rounded to the subnormal grid, outside H-mod) and claimed vectors scaled by 2^100 .. 2^500: every CERTIFIED
eps, rho and count must still hold (mp at 500 digits, enough to form the residual of such inputs exactly).
Usage: python docs/proofs/independent/adversarial_E-02_non_hermitian.py [--cases 60] [--seed 113]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from certify_ed.nonhermitian import certify_nonhermitian, disc_groups, inverse_bound  # noqa: E402

mp.mp.dps = 60


def M(A):
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
    out = mp.matrix(A.shape[0], A.shape[1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            z = complex(A[i, j])
            out[i, j] = mp.mpc(mp.mpf(z.real), mp.mpf(z.imag))
    return out


def spec_norm(X):
    """Largest singular value: sqrt of the largest eigenvalue of X^H X (mp.eighe)."""
    H = X.transpose_conj() * X
    E, _ = mp.eighe(H)
    return mp.sqrt(max(mp.re(E[i]) for i in range(H.rows)))


def fro(X):
    return mp.sqrt(sum(abs(X[i, j]) ** 2 for i in range(X.rows) for j in range(X.cols)))


def check(A, mu, V, rec, W=None):
    """Return the list of failed checks for the CERTIFIED statements of rec (mp ground truth)."""
    bad = []
    st = {s["id"]: s["status"] for s in rec["statements"]}
    n, k = np.asarray(V).shape
    Am, Vm = M(A), M(V)
    mum = [mp.mpc(complex(z).real, complex(z).imag) for z in np.ravel(mu)]
    slack = mp.mpf("1e-45")
    if st["N1.backward_errors"] == "CERTIFIED":
        for j in range(k):
            x = Vm[:, j]
            r = Am * x - mum[j] * x
            if mp.mpf(rec["pairs"][j]["eps_right"]) < fro(r) / fro(x) * (1 - slack):
                bad.append(f"N1 pair {j}")
    if st["N2.eigenvector_inverse"] == "CERTIFIED":
        inv = rec["eigenvector_matrix"]
        Wm = M(W if W is not None else np.linalg.inv(V))
        G = mp.eye(n) - Wm * Vm
        if mp.mpf(inv["theta"]) < fro(G) * (1 - slack):
            bad.append("N2 theta")
        Vi = mp.inverse(Vm) if True else None
        if mp.mpf(inv["beta"]) < spec_norm(Vi) * (1 - slack):
            bad.append("N2 beta")
        if st["N3.inclusion"] == "CERTIFIED":
            rho = mp.mpf(rec["radius"]["rho"])
            R = Am * Vm - Vm * mp.diag(mum)
            if rho < spec_norm(Vi * R) * (1 - slack):
                bad.append("N3 rho")
            lam = mp.eig(Am, left=False, right=False)
            scale = 1 + max(abs(l_) for l_ in lam)
            for l_ in lam:
                if min(abs(l_ - m_) for m_ in mum) > rho + slack * scale:
                    bad.append("N3 eigenvalue outside")
            if st["N4.counting"] == "CERTIFIED":
                gof = [d["group"] for d in rec["discs"]]
                size = {g: gof.count(g) for g in set(gof)}
                cnt = {g: 0 for g in size}
                for l_ in lam:
                    i0 = min(range(n), key=lambda i: abs(l_ - mum[i]))
                    cnt[gof[i0]] += 1
                if cnt != size:
                    bad.append(f"N4 counts {cnt} != {size}")
    return bad


def run(cases=60, seed=113):
    rng = np.random.default_rng(seed)
    tally = {"A_cases": 0, "A_certified_N3": 0, "A_N2_inconclusive": 0, "B_pairs": 0, "C1": [0, 0], "C2": [0, 0]}
    failures = []
    for c in range(cases):
        fam = c % 5
        if fam == 0:                                   # perturbed Jordan block
            m = int(rng.integers(2, 6))
            lam = float(rng.integers(-4, 5)) / 2
            A = lam * np.eye(m) + np.diag(np.ones(m - 1), 1)
            A[m - 1, 0] = 2.0 ** -int(rng.integers(8, 40))
            w, V = np.linalg.eig(A)
            W = None
        elif fam == 1:                                 # nearly parallel eigenvectors
            n = int(rng.integers(2, 6))
            V0 = rng.standard_normal((n, n))
            V0[:, 1] = V0[:, 0] + 10.0 ** -rng.uniform(4, 15) * rng.standard_normal(n)
            d = rng.standard_normal(n)
            A = V0 @ np.diag(d) @ np.linalg.inv(V0)
            w, V = np.linalg.eig(A)
            W = None
        elif fam == 2:                                 # graded scales
            n = int(rng.integers(3, 7))
            S = np.diag(2.0 ** (-10.0 * np.arange(n)))
            A = S @ rng.standard_normal((n, n)) @ np.linalg.inv(S)
            w, V = np.linalg.eig(A)
            W = None
        elif fam == 3:                                 # wrong supplied inverse
            n = int(rng.integers(2, 7))
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            w, V = np.linalg.eig(A)
            W = np.linalg.inv(V) * (1 + 10.0 ** -rng.uniform(2, 8) * rng.standard_normal((n, n)))
        else:                                          # random supplied inverse
            n = int(rng.integers(2, 6))
            A = rng.standard_normal((n, n))
            w, V = np.linalg.eig(A)
            W = rng.standard_normal((n, n))
        rec = certify_nonhermitian(A, w, V, W=W)
        tally["A_cases"] += 1
        st = {s["id"]: s["status"] for s in rec["statements"]}
        tally["A_certified_N3"] += st["N3.inclusion"] == "CERTIFIED"
        tally["A_N2_inconclusive"] += st["N2.eigenvector_inverse"] == "INCONCLUSIVE"
        failures += [f"A{c} fam{fam}: {b}" for b in check(A, w, V, rec, W)]
        if fam == 0 and st["N1.backward_errors"] == "CERTIFIED":
            Am = M(A)
            lam_t = mp.eig(Am, left=False, right=False)
            for j in range(len(w)):
                dist = min(abs(l_ - mp.mpc(complex(w[j]).real, complex(w[j]).imag)) for l_ in lam_t)
                x = M(V)[:, j]
                r = fro(Am * x - mp.mpc(complex(w[j]).real, complex(w[j]).imag) * x) / fro(x)
                tally["C1"][1] += 1
                tally["C1"][0] += int(r < dist)                 # Theorem 1 radius misused: misses the eigenvalue
            rr = np.sqrt(sum(rec["pairs"][j]["residual_upper"] ** 2 for j in range(len(w))))
            tally["C2"][1] += 1
            far = max(min(abs(l_ - mp.mpc(complex(z).real, complex(z).imag)) for z in w) for l_ in lam_t)
            tally["C2"][0] += int(mp.mpf(rr) < far)             # Bauer-Fike without beta: some eigenvalue outside
    # Part B: disc_groups at the separation boundary
    for c in range(cases):
        rho = float(2.0 ** rng.uniform(-40, 10))
        base = complex(rng.uniform(-5, 5), rng.uniform(-5, 5))
        k = int(rng.integers(-3, 4))
        dirs = [1, 1j, (1 + 1j) / np.sqrt(2)]
        dvec = dirs[c % 3]
        mu = np.array([base, base + dvec * (2 * rho) * (1 + k * 2.0 ** -52), base + 0.5 * dvec * rho])
        g = disc_groups(mu, rho)
        for i in range(3):
            for j in range(i + 1, 3):
                tally["B_pairs"] += 1
                if g["group_of"][i] != g["group_of"][j]:
                    zi, zj = complex(mu[i]), complex(mu[j])
                    d2 = (Fraction(zi.real) - Fraction(zj.real)) ** 2 + (Fraction(zi.imag) - Fraction(zj.imag)) ** 2
                    if not d2 > (2 * Fraction(rho)) ** 2:
                        failures.append(f"B{c}: pair ({i},{j}) separated at distance <= 2 rho")
    # Part D: opt-out with subnormal complex components and huge vectors
    tally["D_cases"] = 0
    for c in range(max(4, cases // 2)):
        n = int(rng.integers(2, 5))
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        for _ in range(int(rng.integers(1, n * n + 1))):
            i, j = rng.integers(0, n, 2)
            A[i, j] = complex(float(rng.integers(1, 7)) * 2.0 ** -1074, float(rng.integers(1, 7)) * 2.0 ** -1074)
        try:
            w, V = np.linalg.eig(A)
        except np.linalg.LinAlgError:          # LAPACK geev may not converge on such inputs: no claim to check
            tally["D_skipped"] = tally.get("D_skipped", 0) + 1
            continue
        V = V * 2.0 ** float(rng.integers(100, 500))
        rec = certify_nonhermitian(A, w, V, underflow_check=False)
        tally["D_cases"] += 1
        with mp.workdps(500):
            failures += [f"D{c}: {b}" for b in check(A, w, V, rec)]
    print(f"D: {tally['D_cases']} opt-out matrices with subnormal complex components and vectors up to 2^500 "
          f"({tally.get('D_skipped', 0)} skipped: LAPACK geev did not converge)")
    print(f"A: {tally['A_cases']} matrices (N3 certified in {tally['A_certified_N3']}, N2 INCONCLUSIVE in "
          f"{tally['A_N2_inconclusive']}); B: {tally['B_pairs']} centre pairs at the separation boundary")
    print(f"C negative controls: Theorem 1 radius misses the nearest eigenvalue in {tally['C1'][0]}/{tally['C1'][1]} pairs; "
          f"Bauer-Fike without beta leaves an eigenvalue outside in {tally['C2'][0]}/{tally['C2'][1]} matrices")
    for f_ in failures[:20]:
        print("  VIOLATION:", f_)
    print(f"violations: {len(failures)}")
    return {"violations": len(failures), "tally": tally}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=60)
    p.add_argument("--seed", type=int, default=113)
    a = p.parse_args()
    r = run(a.cases, a.seed)
    sys.exit(1 if r["violations"] else 0)
