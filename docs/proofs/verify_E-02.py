"""Computational verification of docs/proofs/E-02_non_hermitian.md (non-Hermitian eigenproblems, E-02).

Ground truth: exact rational arithmetic (fractions.Fraction, complex numbers as pairs) for every floating-point
inequality of Propositions 5 to 7 (the residual of the stored numbers, the entrywise bound E, the norm bound S, the
backward error eps, the matrix I - W V and its bound B, theta, the separation lower bounds), and mpmath at 60 digits for
the spectral quantities (||V^-1||_2, ||W||_2, ||V||_2, ||V^-1 R||_2, sigma_min(A - mu I), and the eigenvalues of A when
they are not planted). Parts:
  A  random matrices (n = 2..8, real and complex, Gaussian and non-normal Schur forms, claims from numpy.linalg.eig,
     half of them perturbed by 1e-12..1e-3 in the values or the vectors; left vectors from scipy.linalg.eig): every
     stage of Propositions 5 and 6 exactly; N1 eps_i >= ||r_i||/||x_i|| exactly and sigma_min(A - mu_i I) <= eps_i;
     N2 theta >= ||I - W V||_F exactly, beta >= ||V^-1||_2, c_W >= ||W||_2, c_V >= ||V||_2; N3 rho >= ||V^-1 R||_2 and
     every eigenvalue in the union of the discs; N4 cross-group discs disjoint (exactly) and the count of each group
  B  planted spectra: A = S J S^-1 with S unimodular integer and J a dyadic Jordan form (distinct, repeated
     semisimple, defective, complex, clustered at 2^-30), so the eigenvalues and their multiplicities are exact: N3 and
     N4 checked exactly against them
  C  adversarial: the Bauer-Fike equality family (2 x 2, distance = ||V^-1 R||_2 exactly), exact Jordan blocks,
     underflow inside the products (A ~ 2^-895, vector entries ~ 2^-150), a subnormal entry (UNDERFLOW_RISK; with
     the scan off the bounds must still hold), a duplicated eigenvector (V singular), overflow (2^500), companion
     matrices of prod (z - j), column scalings 2^+-30, a wrong eigenvalue, real ill-conditioned matrices with
     conjugate pairs
  D  negative controls (must be detected): eps_i read as an enclosure radius, Bauer-Fike without the factor
     ||V^-1|| (radius ||R||_F), the radius halved, and the stage check with E = 0
Run from the repository root: python docs/proofs/verify_E-02.py [--cases N] [--seed S]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _common import Counter, eigh, herm, norm2, to_mp  # noqa: E402

import certify_ed.nonhermitian as NH  # noqa: E402
from certify_ed.nonhermitian import certify_nonhermitian, disc_groups, inverse_bound, residual_bounds  # noqa: E402

mp.mp.dps = 60
TOL = mp.mpf("1e-40")


# ------------------------------------------------------------------------------------------------ exact complex rationals
def cf(z):
    z = complex(z)
    return (Fraction(z.real), Fraction(z.imag))


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def csub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def abs2(a):
    return a[0] * a[0] + a[1] * a[1]


def fmat(M):
    M = np.asarray(M)
    if M.ndim == 1:
        M = M[:, None]
    return [[cf(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]


def exact_residual(Af, mu, Xf, j):
    """r = A x_j - mu x_j exactly (list of complex rationals)."""
    n = len(Af)
    m = cf(mu)
    out = []
    for i in range(n):
        s = (Fraction(0), Fraction(0))
        for l in range(n):
            p = cmul(Af[i][l], Xf[l][j])
            s = (s[0] + p[0], s[1] + p[1])
        out.append(csub(s, cmul(m, Xf[i][j])))
    return out


def exact_I_minus_WV(Wf, Vf):
    n = len(Vf)
    G = []
    for i in range(n):
        row = []
        for j in range(n):
            s = (Fraction(0), Fraction(0))
            for l in range(n):
                p = cmul(Wf[i][l], Vf[l][j])
                s = (s[0] + p[0], s[1] + p[1])
            row.append(((Fraction(1) if i == j else Fraction(0)) - s[0], -s[1]))
        G.append(row)
    return G


def cdiv(a, b):
    d = abs2(b)
    num = cmul(a, (b[0], -b[1]))
    return (num[0] / d, num[1] / d)


def exact_inverse(Vf):
    """Inverse of a square complex rational matrix by Gauss-Jordan elimination, or None when it is singular."""
    n = len(Vf)
    M = [list(row) + [((Fraction(1) if i == j else Fraction(0)), Fraction(0)) for j in range(n)] for i, row in enumerate(Vf)]
    for c in range(n):
        piv = next((r for r in range(c, n) if abs2(M[r][c]) != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        p = M[c][c]
        M[c] = [cdiv(x, p) for x in M[c]]
        for r in range(n):
            if r != c and abs2(M[r][c]) != 0:
                f = M[r][c]
                M[r] = [csub(x, cmul(f, y)) for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def fr_to_mp(c):
    return mp.mpc(mp.mpf(c[0].numerator) / c[0].denominator, mp.mpf(c[1].numerator) / c[1].denominator)


def mp_from_frac(F):
    M = mp.matrix(len(F), len(F[0]))
    for i in range(len(F)):
        for j in range(len(F[0])):
            M[i, j] = fr_to_mp(F[i][j])
    return M


def ge(bound, value):
    """bound >= value for a binary64 bound and an exact rational value (squares compared exactly when value >= 0)."""
    if not math.isfinite(bound):
        return True
    return Fraction(bound) >= value


def mp_eigs(Am):
    E = mp.eig(Am, left=False, right=False)
    return [mp.mpc(e) for e in E]


def sigma_min(Am, mu):
    """Smallest singular value of A - mu I by mpmath's complex SVD (not through M^H M, whose smallest eigenvalue is not
    resolved at 60 digits when sigma_min / ||M|| < 1e-30)."""
    n = Am.rows
    M = Am - mp.mpc(complex(mu)) * mp.eye(n)
    S = mp.svd_c(M, compute_uv=False)
    return min(mp.re(S[i]) for i in range(n))


# ------------------------------------------------------------------------------------------------ checks of one record
def check_record(A, mu, V, rec, eigs=None, eig_exact=None, Y=None, cnt=None, adv=False, label="", stage=True):
    """Check every CERTIFIED statement of `rec` and the floating-point stages behind it. eigs: mp eigenvalues; eig_exact:
    exact eigenvalues as complex rationals (with multiplicity). Returns True when nothing is violated."""
    ok_all = True
    st = {s["id"]: s for s in rec["statements"]}
    n, k = V.shape
    Af = fmat(A)
    Vf = fmat(V)
    if st["N1.backward_errors"]["status"] == "CERTIFIED":
        rr = residual_bounds(np.asarray(A), np.asarray(mu), np.asarray(V), return_arrays=True)
        Am = to_mp(A)
        for j in range(k):
            r = exact_residual(Af, mu[j], Vf, j)
            nr2 = sum(abs2(c) for c in r)
            nx2 = sum(abs2(Vf[i][j]) for i in range(n))
            ok = True
            if stage:
                # Proposition 5 (iii) entrywise, with the per-entry additive allowance (10 n + 20) u_min of (iv)
                allow = Fraction(10 * n + 20, 2 ** 1075)
                for i in range(n):
                    d = csub(cf(rr["S_hat"][i, j]), r[i])
                    ok = ok and abs2(d) <= (Fraction(float(rr["E"][i, j])) + allow) ** 2
            ok = ok and ge(rr["upper"][j], 0) and (Fraction(rr["upper"][j]) ** 2 >= nr2)
            eps = rec["pairs"][j]["eps_right"]
            ok = ok and Fraction(eps) ** 2 * nx2 >= nr2
            sm = sigma_min(Am, mu[j])
            ok = ok and sm <= mp.mpf(eps) * (1 + TOL) + TOL
            if Y is not None:
                Ah = np.asarray(A).conj().T
                rl = exact_residual(fmat(Ah), np.conj(mu[j]), fmat(Y), j)
                ny2 = sum(abs2(cf(Y[i, j])) for i in range(n))
                ok = ok and Fraction(rec["pairs"][j]["eps_left"]) ** 2 * ny2 >= sum(abs2(c) for c in rl)
            ok_all &= ok
            if cnt is not None:
                cnt.add(adv, ok, f"{label}: N1 pair {j}")
    if st["N2.eigenvector_inverse"]["status"] == "CERTIFIED":
        ib = inverse_bound(np.asarray(V), return_arrays=True)
        Wf = fmat(ib["W"])
        G = exact_I_minus_WV(Wf, Vf)
        ok = Fraction(ib["theta"]) ** 2 >= sum(abs2(G[i][j]) for i in range(n) for j in range(n))
        if stage:
            allow = Fraction(10 * n + 20, 2 ** 1075)
            for i in range(n):
                for j in range(n):
                    ok = ok and abs2(csub(cf(ib["G_hat"][i, j]), G[i][j])) <= (Fraction(float(ib["B"][i, j])) + allow) ** 2
        Vm = to_mp(V)
        Vinv_f = exact_inverse(Vf)
        if Vinv_f is None:                       # N2 certified a singular matrix: a violation
            if cnt is not None:
                cnt.add(adv, False, f"{label}: N2 certified an exactly singular V")
            return False
        Vinv = mp_from_frac(Vinv_f)
        ok = ok and mp.mpf(ib["beta"]) >= norm2(Vinv) * (1 - TOL)
        ok = ok and mp.mpf(ib["c_W"]) >= norm2(mp_from_frac(Wf)) * (1 - TOL)
        ok = ok and mp.mpf(ib["c_V"]) >= norm2(Vm) * (1 - TOL)
        ok_all &= ok
        if cnt is not None:
            cnt.add(adv, ok, f"{label}: N2")
        if st["N3.inclusion"]["status"] == "CERTIFIED":
            rho = rec["radius"]["rho"]
            Rf = [[None] * n for _ in range(n)]
            for j in range(n):
                r = exact_residual(Af, mu[j], Vf, j)
                for i in range(n):
                    Rf[i][j] = r[i]
            ok = mp.mpf(rho) >= norm2(mp_from_frac(matmul_f(Vinv_f, Rf))) * (1 - TOL)
            if eig_exact is not None:
                for lam in eig_exact:
                    ok = ok and any(abs2(csub(lam, cf(mu[i]))) <= Fraction(rho) ** 2 for i in range(n))
            elif eigs is not None:
                scale = 1 + max(abs(e) for e in eigs)
                for lam in eigs:
                    ok = ok and min(abs(lam - mp.mpc(complex(mu[i]))) for i in range(n)) <= mp.mpf(rho) + TOL * scale
            ok_all &= ok
            if cnt is not None:
                cnt.add(adv, ok, f"{label}: N3")
        if st["N4.counting"]["status"] == "CERTIFIED":
            rho = rec["radius"]["rho"]
            gof = [d["group"] for d in rec["discs"]]
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    dij = abs2(csub(cf(mu[i]), cf(mu[j])))
                    if gof[i] != gof[j]:
                        ok = ok and dij > (2 * Fraction(rho)) ** 2
            sizes = {}
            for gidx in gof:
                sizes[gidx] = sizes.get(gidx, 0) + 1
            counts = {gidx: 0 for gidx in sizes}
            if eig_exact is not None:
                for lam in eig_exact:
                    inside = [gof[i] for i in range(n) if abs2(csub(lam, cf(mu[i]))) <= Fraction(rho) ** 2]
                    if inside:
                        counts[inside[0]] += 1
                        ok = ok and len(set(inside)) == 1
                    else:
                        ok = False
            elif eigs is not None:
                for lam in eigs:
                    d = [abs(lam - mp.mpc(complex(mu[i]))) for i in range(n)]
                    i0 = min(range(n), key=lambda i: d[i])
                    counts[gof[i0]] += 1
            ok = ok and all(counts[gidx] == sizes[gidx] for gidx in sizes)
            ok_all &= ok
            if cnt is not None:
                cnt.add(adv, ok, f"{label}: N4")
    # separation lower bounds (Proposition 7 (i)), exactly, whenever discs were formed
    if "discs" in rec:
        mu_a = np.asarray(mu).ravel()
        two_rho = 2.0 * rec["radius"]["rho"]
        re, im = np.real(mu_a).astype(float), np.imag(mu_a).astype(float)
        delta = np.maximum(np.abs(np.subtract.outer(re, re)), np.abs(np.subtract.outer(im, im))) * NH._ONE_MINUS_2U
        ok = True
        for i in range(n):
            for j in range(n):
                if i != j:
                    ok = ok and Fraction(float(delta[i, j])) ** 2 <= abs2(csub(cf(mu_a[i]), cf(mu_a[j])))
        ok_all &= ok
        if cnt is not None:
            cnt.add(adv, ok, f"{label}: separation bound")
    return ok_all


# ------------------------------------------------------------------------------------------------ generators
def rand_matrix(rng, n, cplx, family):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    if family == "schur":
        c = float(rng.choice([1.0, 10.0, 100.0]))
        T = np.triu(A, 1) * c + np.diag(np.diag(A))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
        A = Q @ T @ Q.conj().T
    return A


def unimodular(rng, n, m=2):
    L = np.tril(rng.integers(-m, m + 1, (n, n)), -1) + np.eye(n, dtype=np.int64)
    Uu = np.triu(rng.integers(-m, m + 1, (n, n)), 1) + np.eye(n, dtype=np.int64)
    S = [[Fraction(int(x)) for x in row] for row in (L @ Uu)]
    Linv = np.round(np.linalg.inv(L)).astype(np.int64)
    Uinv = np.round(np.linalg.inv(Uu)).astype(np.int64)
    Si = [[Fraction(int(x)) for x in row] for row in (Uinv @ Linv)]
    return S, Si


def matmul_f(Af, Bf):
    n, m, p = len(Af), len(Bf), len(Bf[0])
    out = []
    for i in range(n):
        row = []
        for j in range(p):
            s = (Fraction(0), Fraction(0))
            for l in range(m):
                q = cmul(Af[i][l], Bf[l][j])
                s = (s[0] + q[0], s[1] + q[1])
            row.append(s)
        out.append(row)
    return out


def planted(rng, n, kind):
    """A = S J S^-1 exactly (dyadic entries, exactly representable), with its eigenvalues (with multiplicity)."""
    lam = [Fraction(int(rng.integers(-16, 17)), 4) for _ in range(n)]
    lamc = [(l_, Fraction(0)) for l_ in lam]
    J = [[(Fraction(0), Fraction(0)) for _ in range(n)] for _ in range(n)]
    if kind == "complex":
        lamc = [(Fraction(int(rng.integers(-16, 17)), 4), Fraction(int(rng.integers(-16, 17)), 4)) for _ in range(n)]
    elif kind == "repeated":
        lamc[1] = lamc[0]
        if n > 3:
            lamc[3] = lamc[2]
    elif kind == "cluster":
        lamc[1] = (lamc[0][0] + Fraction(1, 2 ** 30), Fraction(0))
    for i in range(n):
        J[i][i] = lamc[i]
    if kind == "defective":
        m = 2 if n < 5 else 3
        for i in range(1, m):
            J[i][i] = J[0][0]
            lamc[i] = lamc[0]
            J[i - 1][i] = (Fraction(1), Fraction(0))
    S, Si = unimodular(rng, n)
    Sc = [[(x, Fraction(0)) for x in row] for row in S]
    Sic = [[(x, Fraction(0)) for x in row] for row in Si]
    Af = matmul_f(matmul_f(Sc, J), Sic)
    cplx = any(Af[i][j][1] != 0 for i in range(n) for j in range(n))
    A = np.empty((n, n), dtype=complex if cplx else float)
    for i in range(n):
        for j in range(n):
            re, im = Af[i][j]
            assert Fraction(float(re)) == re and Fraction(float(im)) == im, "planted entry not representable"
            A[i, j] = complex(float(re), float(im)) if cplx else float(re)
    return A, lamc


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


# ------------------------------------------------------------------------------------------------ run
def run(n_random: int = 60, seed: int = 20261111, verbose: bool = False, quick: bool = False):
    """All parts; quick=True shrinks the fixed adversarial families (for the pinning test)."""
    rng = np.random.default_rng(seed)
    cA, cB, cC = Counter(), Counter(), Counter()
    neg = {"D1_eps_as_radius": [0, 0], "D2_no_conditioning": [0, 0], "D3_half_radius": [0, 0], "D4_zero_E": [0, 0]}
    certified = {"N1": 0, "N2": 0, "N3": 0, "N4": 0}

    def tally(rec):
        for s in rec["statements"]:
            if s["status"] == "CERTIFIED":
                certified[s["id"][:2]] += 1

    # ---- A: random
    for case in range(n_random):
        n = int(rng.integers(2, 9))
        cplx = bool(rng.integers(0, 2))
        family = "gauss" if case % 2 == 0 else "schur"
        A = rand_matrix(rng, n, cplx, family)
        w, Yl, V = la.eig(A, left=True, right=True)
        mode = case % 4
        if mode == 1:
            w = w * (1 + 10.0 ** rng.uniform(-12, -3) * rng.standard_normal(n))
        elif mode == 2:
            V = V + 10.0 ** rng.uniform(-12, -3) * (rng.standard_normal(V.shape) + 1j * rng.standard_normal(V.shape))
        rec = certify_nonhermitian(A, w, V, left_eigenvectors=Yl)
        tally(rec)
        eigs = mp_eigs(to_mp(A))
        check_record(A, w, V, rec, eigs=eigs, Y=Yl, cnt=cA, adv=False, label=f"A{case} {family} n={n}")
        # D4: the stage check must fail when E is replaced by 0 (the harness can detect a missing rounding term)
        rr = residual_bounds(np.asarray(A), np.asarray(w), np.asarray(V), return_arrays=True)
        Af, Vf = fmat(A), fmat(V)
        r0 = exact_residual(Af, w[0], Vf, 0)
        neg["D4_zero_E"][1] += 1
        if any(abs2(csub(cf(rr["S_hat"][i, 0]), r0[i])) > 0 for i in range(n)):
            neg["D4_zero_E"][0] += 1

    # ---- B: planted exact spectra
    kinds = ("distinct", "complex", "repeated", "defective", "cluster")
    for case in range(max(5, n_random // 2)):
        n = int(rng.integers(3, 8))
        kind = kinds[case % len(kinds)]
        A, lamc = planted(rng, n, kind)
        w, V = np.linalg.eig(A)
        rec = certify_nonhermitian(A, w, V)
        tally(rec)
        check_record(A, w, V, rec, eig_exact=lamc, cnt=cB, adv=False, label=f"B{case} {kind} n={n}")

    # ---- C: adversarial
    tight = []
    for a in ((1.0,) if quick else (0.0, 1.0, -3.5)):
        for b in ((1.0, 2.0 ** 10) if quick else (1.0, 2.0 ** 10, 2.0 ** -10)):
            for tau in ((2.0 ** -5, 2.0 ** -20) if quick else (0.5, 2.0 ** -5, 2.0 ** -10, 2.0 ** -20)):
                A = np.array([[a, b], [0.0, a]])
                V = np.array([[1.0, 1.0], [tau, -tau]])
                mu = np.array([a + b * tau, a - b * tau])
                rec = certify_nonhermitian(A, mu, V)
                tally(rec)
                lam = [(Fraction(a), Fraction(0))] * 2
                check_record(A, mu, V, rec, eig_exact=lam, cnt=cC, adv=True, label=f"C1 a={a} b={b} tau={tau}")
                dist = Fraction(b) * Fraction(tau)
                if stmt(rec, "N3.inclusion")["status"] == "CERTIFIED":
                    rho = rec["radius"]["rho"]
                    tight.append(float(Fraction(rho) / dist - 1))
                    neg["D3_half_radius"][1] += 1
                    if Fraction(rho) / 2 < dist:
                        neg["D3_half_radius"][0] += 1
                eps = rec["pairs"][0]["eps_right"]
                neg["D1_eps_as_radius"][1] += 1
                if Fraction(eps) < dist:
                    neg["D1_eps_as_radius"][0] += 1
                rr = residual_bounds(A, mu, V)
                rF = float(np.sqrt(np.sum(rr["upper"] ** 2)))
                neg["D2_no_conditioning"][1] += 1
                if Fraction(rF) < dist:
                    neg["D2_no_conditioning"][0] += 1
    for m in ((2, 4) if quick else range(2, 7)):
        for lamv in ((1.0,) if quick else (0.0, 1.0, -2.25)):
            A = lamv * np.eye(m) + np.diag(np.ones(m - 1), 1)
            w, V = np.linalg.eig(A)
            rec = certify_nonhermitian(A, w, V)
            tally(rec)
            check_record(A, w, V, rec, eig_exact=[(Fraction(lamv), Fraction(0))] * m, cnt=cC, adv=True,
                         label=f"C2 Jordan m={m} lam={lamv}")
    for case in range(2 if quick else 6):
        n = int(rng.integers(3, 6))
        A0, lamc = planted(rng, n, "distinct")
        A = A0 * 2.0 ** -895
        lam = [(l_[0] * Fraction(2) ** -895, l_[1] * Fraction(2) ** -895) for l_ in lamc]
        w, V = np.linalg.eig(A)
        V = V * 2.0 ** -150
        rec = certify_nonhermitian(A, w, V)
        tally(rec)
        check_record(A, w, V, rec, eig_exact=lam, cnt=cC, adv=True, label=f"C3 underflow inside n={n}")
    for case in range(1 if quick else 4):
        n = int(rng.integers(3, 6))
        A, lamc = planted(rng, n, "distinct")
        A = A.astype(float)
        A[0, n - 1] = 1e-310                 # a stored subnormal entry (the spectrum is then taken from mpmath)
        w, V = np.linalg.eig(A)
        rec = certify_nonhermitian(A, w, V)
        ok = all(s["status"] == "INCONCLUSIVE" and s["evidence"].get("reason_code") == "UNDERFLOW_RISK" for s in rec["statements"])
        cC.add(True, ok, f"C4 subnormal entry: not refused")
        rec = certify_nonhermitian(A, w, V, underflow_check=False)
        tally(rec)
        eigs = mp_eigs(to_mp(A))
        check_record(A, w, V, rec, eigs=eigs, cnt=cC, adv=True, label="C4 subnormal entry, scan off")
    for case in range(1 if quick else 4):
        n = int(rng.integers(3, 7))
        A = rand_matrix(rng, n, bool(case % 2), "gauss")
        w, V = np.linalg.eig(A)
        V[:, 1] = V[:, 0]
        w[1] = w[0]
        rec = certify_nonhermitian(A, w, V)
        cC.add(True, stmt(rec, "N2.eigenvector_inverse")["status"] != "CERTIFIED", "C5 duplicated vector: N2 certified")
        check_record(A, w, V, rec, eigs=mp_eigs(to_mp(A)), cnt=cC, adv=True, label="C5 duplicated vector")
    for case in range(1 if quick else 3):
        n = int(rng.integers(3, 6))
        A, lamc = planted(rng, n, "distinct")
        A = A * 2.0 ** 500
        lam = [(l_[0] * Fraction(2) ** 500, l_[1]) for l_ in lamc]
        w, V = np.linalg.eig(A)
        rec = certify_nonhermitian(A, w, V)
        tally(rec)
        check_record(A, w, V, rec, eig_exact=lam, cnt=cC, adv=True, label="C6 overflow scale")
    for n in ((3, 5) if quick else range(3, 8)):
        coeffs = np.poly(np.arange(1, n + 1))            # integer coefficients of prod (z - j)
        A = np.zeros((n, n))
        A[0, :] = -coeffs[1:]
        A[1:, :-1] = np.eye(n - 1)
        assert np.all(A == np.round(A))
        w, V = np.linalg.eig(A)
        rec = certify_nonhermitian(A, w, V)
        tally(rec)
        check_record(A, w, V, rec, eig_exact=[(Fraction(j), Fraction(0)) for j in range(1, n + 1)], cnt=cC, adv=True,
                     label=f"C7 companion n={n}")
    for case in range(2 if quick else 4):
        n = int(rng.integers(3, 7))
        A, lamc = planted(rng, n, "complex" if case % 2 else "distinct")
        w, V = np.linalg.eig(A)
        V = V * (2.0 ** rng.integers(-30, 31, n))[None, :]
        rec = certify_nonhermitian(A, w, V)
        tally(rec)
        check_record(A, w, V, rec, eig_exact=lamc, cnt=cC, adv=True, label="C8 column scaling")
        w2 = w.copy()
        w2[0] = w2[0] + 0.5
        rec = certify_nonhermitian(A, w2, V)
        tally(rec)
        check_record(A, w2, V, rec, eig_exact=lamc, cnt=cC, adv=True, label="C10 wrong eigenvalue")
    for case in range(1 if quick else 4):
        n = 2 * int(rng.integers(2, 4))
        blocks = []
        for b_ in range(n // 2):
            a_, c_ = Fraction(int(rng.integers(-8, 9)), 4), Fraction(int(rng.integers(1, 9)), 4)
            blocks.append((a_, c_))
        J = [[(Fraction(0), Fraction(0)) for _ in range(n)] for _ in range(n)]
        lam = []
        for b_, (a_, c_) in enumerate(blocks):
            i = 2 * b_
            J[i][i] = (a_, Fraction(0)); J[i + 1][i + 1] = (a_, Fraction(0))
            J[i][i + 1] = (-c_, Fraction(0)); J[i + 1][i] = (c_, Fraction(0))
            lam += [(a_, c_), (a_, -c_)]
        S, Si = unimodular(rng, n, m=4)
        Af = matmul_f(matmul_f([[(x, Fraction(0)) for x in row] for row in S], J),
                      [[(x, Fraction(0)) for x in row] for row in Si])
        A = np.array([[float(Af[i][j][0]) for j in range(n)] for i in range(n)])
        w, V = np.linalg.eig(A)
        rec = certify_nonhermitian(A, w, V)
        tally(rec)
        check_record(A, w, V, rec, eig_exact=lam, cnt=cC, adv=True, label="C11 real conjugate pairs")

    ok = True
    ok &= cA.report("A random matrices: Propositions 5-7 stages exactly, N1-N4 against mpmath")
    ok &= cB.report("B planted exact spectra (distinct, complex, repeated, defective, clustered): N3, N4 exactly")
    ok &= cC.report("C adversarial: Bauer-Fike equality, Jordan blocks, underflow, subnormal, singular V, overflow, "
                    "companion, scalings, wrong eigenvalue, conjugate pairs")
    print(f"   certified statements checked: {certified}")
    if tight:
        print(f"   Bauer-Fike equality family: rho / ||V^-1 R||_2 - 1 between {min(tight):.3e} and {max(tight):.3e}")
    for key, (hit, tot) in neg.items():
        print(f"   negative control {key}: detected in {hit}/{tot} cases")
    detected = all(hit > 0 for hit, tot in neg.values())
    ok &= detected
    total_r = cA.random + cB.random + cC.random
    total_a = cA.adversarial + cB.adversarial + cC.adversarial
    nviol = len(cA.violations) + len(cB.violations) + len(cC.violations)
    print(f"TOTAL: {total_r} random + {total_a} adversarial cases; violations: {nviol}; negative controls detected: {detected}")
    return {"violations": nviol, "random_cases": total_r, "adversarial_cases": total_a, "negative_controls": neg,
            "certified": certified, "tightness": (min(tight), max(tight)) if tight else None, "ok": bool(ok)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--quick", action="store_true", help="shrink the fixed adversarial families")
    a = p.parse_args(argv)
    r = run(a.cases, a.seed, quick=a.quick)
    return 0 if r["ok"] and r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
