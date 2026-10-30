"""Computational verification of docs/proofs/C-03_entanglement.md (certify_ed.entanglement).

Parts (each prints its case count and its number of violations; the document records the numbers):
  A  Proposition 1: T = ||rho_A - rho_hat_A||_1 / 2 <= s + eta/(1 - eta) (s = exact sin theta_max, eta exact)
  B  Lemma 3: the Ostrowski-Weyl enclosure of the spectrum of the exact reduced matrix contains its eigenvalues
  C  Fannes-Audenaert: |S(rho) - S(sigma)| <= g(T) = T ln(d - 1) + h(T) on random pairs; monotonicity of g on
     [0, 1 - 1/d]; the adversarial case showing why g(T_up) may only be used when T_up <= 1 - 1/d
  D  Weyl: |lambda_i(rho_A) - lambda_i(rho_hat_A)| <= 2 T
  E  Interval functions of entanglement.py: -p ln p, h(T), ln(d - 1) enclosures contain the mpmath values;
     end-to-end: certify_entanglement's interval (fed the exact sin theta) contains the exact eigenvector's entropy

Exact quantities: rational arithmetic (fractions.Fraction) where the quantity is rational (eta, t, the reduced
matrix, the enclosures); mpmath at 50 digits where it is not (trace norms, eigenvalues, entropies). The mpmath
values are compared with a margin of 1e-40, far below any gap that matters.

Run:  python docs/proofs/verify_C-03.py [seed]
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from certify_ed import entanglement as ent, ivl  # noqa: E402

mp.mp.dps = 50
MARGIN = mp.mpf(10) ** -40
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261030
rng = np.random.default_rng(SEED)


def to_mp(F):
    """mpmath matrix from a numpy array (float or complex) with exact binary64 -> mpf conversion."""
    F = np.asarray(F)
    M = mp.matrix(F.shape[0], F.shape[1])
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            v = F[i, j]
            M[i, j] = mp.mpc(mp.mpf(float(v.real)), mp.mpf(float(v.imag))) if np.iscomplexobj(F) else mp.mpf(float(v))
    return M


def frac_mp(q: Fraction):
    return mp.mpf(q.numerator) / q.denominator


def trace_norm(M):
    return sum(mp.svd_c(M, compute_uv=False)) if any(isinstance(M[i, j], mp.mpc) for i in range(M.rows) for j in range(M.cols)) \
        else sum(mp.svd_r(M, compute_uv=False))


def herm_eigs(M):
    cplx = any(isinstance(M[i, j], mp.mpc) and M[i, j].imag != 0 for i in range(M.rows) for j in range(M.cols))
    if cplx:
        return sorted(mp.eigh(M, eigvals_only=True))
    R = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            R[i, j] = mp.re(M[i, j])
    return sorted(mp.eigsy(R, eigvals_only=True))


def partial_trace(Z, dA, dB):
    """tr_B of a (dA dB) x (dA dB) mpmath matrix, index = a * dB + b."""
    R = mp.matrix(dA, dA)
    for a in range(dA):
        for c in range(dA):
            R[a, c] = sum(Z[a * dB + b, c * dB + b] for b in range(dB))
    return R


def entropy(M):
    return -sum(p * mp.log(p) for p in herm_eigs(M) if p > mp.mpf(10) ** -45)


def orth_basis(n, g, cplx):
    Z = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
    Q, _ = np.linalg.qr(Z)
    return Q


# ---------------------------------------------------------------------------------------------- A: Proposition 1
def part_A(n_random=120):
    cases = viol = 0
    adversarial = 0

    def check(Q, X, dA, dB, label=None):
        nonlocal cases, viol
        n, g = X.shape
        Qm, Xm = to_mp(Q), to_mp(X)
        P = Qm * Qm.transpose_conj()
        # exact polar factor of X and sin theta_max = ||Qhat Qhat^H - P||_2
        U_, S_, Vh = (mp.svd_c(Xm) if np.iscomplexobj(X) else mp.svd_r(Xm))
        Qhat = U_ * Vh
        D = Qhat * Qhat.transpose_conj() - P
        s = max(mp.svd_c(D, compute_uv=False) if np.iscomplexobj(X) else mp.svd_r(D, compute_uv=False))
        # exact eta = ||X^H X - I||_2 (mpmath; the rational value is the same number) and t = tr(X^H X)
        G = Xm.transpose_conj() * Xm
        eta = max(abs(e) for e in herm_eigs(G - mp.eye(g)))
        t = sum(G[i, i] for i in range(g))
        rho = partial_trace(P, dA, dB) / g
        rho_hat = partial_trace(Xm * Xm.transpose_conj(), dA, dB) / t
        T = trace_norm(rho - rho_hat) / 2
        bound = s + (0 if g == 1 else eta / (1 - eta)) if eta < 1 else mp.inf
        cases += 1
        if T > bound + MARGIN:
            viol += 1
            print("  VIOLATION A", label, g, dA, dB, mp.nstr(T, 12), mp.nstr(bound, 12))
        return T, bound, s, eta

    for _ in range(n_random):
        dA, dB = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        n = dA * dB
        g = int(rng.integers(1, min(4, n // 2) + 1))
        cplx = bool(rng.integers(0, 2))
        Q = orth_basis(n, g, cplx)
        # X: a rotated, non-orthonormal perturbation of a basis of range(Q)
        amp = 10.0 ** rng.uniform(-6, -0.3)
        W = orth_basis(n, n, cplx)
        Xp = Q + amp * W[:, :g]
        M = np.eye(g) + 0.3 * amp * (rng.standard_normal((g, g)))
        X = Xp @ M
        check(Q, X, dA, dB)
    # adversarial: (i) g = 1 with a badly normalised vector (eta large: the bound must not depend on it),
    # (ii) rotation close to 90 degrees, (iii) eta close to 1 for g = 2, (iv) X exactly a basis of range(P)
    for scale in (0.1, 3.0, 50.0):
        Q = orth_basis(6, 1, False)
        X = scale * (Q + 1e-3 * orth_basis(6, 1, False))
        T, bound, s, eta = check(Q, X, 2, 3, "g=1 scale %g" % scale)
        adversarial += 1
    Q = orth_basis(8, 1, True)
    W = orth_basis(8, 2, True)
    for ang in (1.2, 1.5, 1.56):
        X = math.cos(ang) * Q + math.sin(ang) * (W[:, 1:2] - Q * (Q.conj().T @ W[:, 1:2]))
        check(Q, X, 2, 4, "angle %g" % ang)
        adversarial += 1
    Q = orth_basis(9, 2, False)
    for f in (0.3, 0.05):
        X = Q @ np.diag([1.0, f])                      # eta = 1 - f^2 close to 1
        check(Q, X, 3, 3, "eta=%g" % (1 - f * f))
        adversarial += 1
    Q = orth_basis(12, 3, True)
    Qr, _ = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    T, bound, s, eta = check(Q, Q @ Qr, 3, 4, "exact basis")
    adversarial += 1
    print(f"A  Proposition 1 (trace-distance bound): {cases} cases ({cases - adversarial} random + {adversarial} adversarial), "
          f"{viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- B: Lemma 3
def part_B(n_random=60):
    cases = viol = 0
    adversarial = 0

    def check(x, dA, dB, label=None):
        nonlocal cases, viol
        a_idx, b_idx = np.repeat(np.arange(dA), dB), np.tile(np.arange(dB), dA)
        Mr, Mi, t = ent.reduced_density_exact(x, a_idx, b_idx, dA, dB)
        encl = ent.enclose_psd_eigenvalues(Mr, Mi, t)
        M = mp.matrix(dA, dA)
        for i in range(dA):
            for j in range(dA):
                M[i, j] = frac_mp(Mr[i, j] / t) + (1j * frac_mp(Mi[i, j] / t) if Mi is not None else 0)
        ev = herm_eigs(M)
        cases += 1
        for k in range(dA):
            if not (frac_mp(encl["lo"][k]) - MARGIN <= ev[k] <= frac_mp(encl["hi"][k]) + MARGIN):
                viol += 1
                print("  VIOLATION B", label, k, mp.nstr(ev[k], 20), float(encl["lo"][k]), float(encl["hi"][k]))
                break

    for _ in range(n_random):
        dA, dB = int(rng.integers(2, 7)), int(rng.integers(1, 7))
        g = int(rng.integers(1, 3))
        cplx = bool(rng.integers(0, 2))
        X = rng.standard_normal((dA * dB, g)) + (1j * rng.standard_normal((dA * dB, g)) if cplx else 0)
        X *= 10.0 ** rng.uniform(-3, 3)
        check(X, dA, dB)
    # adversarial: degenerate spectrum (maximally entangled), rank-deficient (product state), tiny and huge scale,
    # a near-degenerate pair of Schmidt weights
    dA = dB = 4
    check(np.eye(4).ravel() / 2, dA, dB, "maximally entangled")
    adversarial += 1
    check(np.outer(rng.standard_normal(4), rng.standard_normal(4)).ravel(), dA, dB, "product state")
    adversarial += 1
    check(rng.standard_normal(16) * 1e-150, dA, dB, "scale 1e-150")
    adversarial += 1
    check(rng.standard_normal(16) * 1e150, dA, dB, "scale 1e150")
    adversarial += 1
    Y = np.diag([0.5, 0.5 + 1e-12, 0.5, 0.5])
    check(Y.ravel(), dA, dB, "near-degenerate")
    adversarial += 1
    print(f"B  Lemma 3 (exact spectrum enclosure): {cases} cases ({cases - adversarial} random + {adversarial} adversarial), "
          f"{viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- C: Fannes-Audenaert
def g_fa(T, d):
    h = -(T * mp.log(T) if T > 0 else 0) - ((1 - T) * mp.log(1 - T) if T < 1 else 0)
    return T * mp.log(d - 1) + h


def random_state(d, cplx, rank=None):
    r = rank or d
    Z = rng.standard_normal((d, r)) + (1j * rng.standard_normal((d, r)) if cplx else 0)
    Zm = to_mp(Z)
    R = Zm * Zm.transpose_conj()
    return R / sum(R[i, i] for i in range(d))


def part_C(n_random=150):
    cases = viol = 0
    mono_cases = mono_viol = 0
    for _ in range(n_random):
        d = int(rng.integers(2, 7))
        cplx = bool(rng.integers(0, 2))
        rho = random_state(d, cplx, int(rng.integers(1, d + 1)))
        sig = random_state(d, cplx, int(rng.integers(1, d + 1)))
        if rng.integers(0, 2):
            lam = mp.mpf(10) ** rng.uniform(-8, 0)
            sig = (1 - lam) * rho + lam * sig           # close pairs as well
        T = trace_norm(rho - sig) / 2
        dS = abs(entropy(rho) - entropy(sig))
        cases += 1
        if dS > g_fa(T, d) + MARGIN:
            viol += 1
            print("  VIOLATION C", d, mp.nstr(T, 12), mp.nstr(dS, 12), mp.nstr(g_fa(T, d), 12))
    # equality cases (Audenaert): rho pure, sigma = (1 - T) rho + T * (I - rho)/(d - 1), any T <= 1 - 1/d
    eq_cases = 0
    for d in (2, 3, 5, 8):
        for T in (Fraction(1, 10), Fraction(1, 3), 1 - Fraction(1, d)):
            Tm = frac_mp(T)
            rho = mp.zeros(d, d)
            rho[0, 0] = 1
            sig = (1 - Tm) * rho + Tm * (mp.eye(d) - rho) / (d - 1)
            dS = abs(entropy(rho) - entropy(sig))
            cases += 1
            eq_cases += 1
            if dS > g_fa(Tm, d) + MARGIN or abs(dS - g_fa(Tm, d)) > MARGIN * 100:
                viol += 1
                print("  VIOLATION C (equality case)", d, T, mp.nstr(dS, 20), mp.nstr(g_fa(Tm, d), 20))
    # monotonicity of g on [0, 1 - 1/d]
    for d in range(2, 12):
        grid = [mp.mpf(k) / 2000 * (1 - mp.mpf(1) / d) for k in range(2001)]
        vals = [g_fa(T, d) for T in grid]
        for a, b in zip(vals, vals[1:]):
            mono_cases += 1
            if b < a - MARGIN:
                mono_viol += 1
    # beyond the range g decreases: the naive use of g(T_up) with T < T_up fails (d = 2 example of the document)
    rho = mp.matrix([[mp.mpf(1) / 2, 0], [0, mp.mpf(1) / 2]])
    sig = mp.matrix([[1, 0], [0, 0]])
    T = trace_norm(rho - sig) / 2                      # = 1/2
    dS = abs(entropy(rho) - entropy(sig))              # = ln 2
    T_up = mp.mpf("0.9")
    beyond = (T < T_up) and (dS > g_fa(T_up, 2))
    print(f"C  Fannes-Audenaert: {cases} pairs ({cases - eq_cases} random + {eq_cases} equality cases), {viol} violations; "
          f"monotonicity of g on [0, 1 - 1/d]: {mono_cases} grid steps, {mono_viol} violations; "
          f"beyond-range adversarial case (T = 1/2 < T_up = 0.9, d = 2): |dS| = {mp.nstr(dS, 8)} > g(T_up) = "
          f"{mp.nstr(g_fa(T_up, 2), 8)}: {'reproduced' if beyond else 'NOT reproduced'}")
    return cases + mono_cases, viol + mono_viol, beyond


# ---------------------------------------------------------------------------------------------- D: Weyl
def part_D(n_random=80):
    cases = viol = 0
    for _ in range(n_random):
        d = int(rng.integers(2, 7))
        cplx = bool(rng.integers(0, 2))
        rho = random_state(d, cplx)
        sig = random_state(d, cplx)
        lam = mp.mpf(10) ** rng.uniform(-10, 0)
        sig = (1 - lam) * rho + lam * sig
        T = trace_norm(rho - sig) / 2
        er, es = herm_eigs(rho), herm_eigs(sig)
        cases += 1
        if max(abs(a - b) for a, b in zip(er, es)) > 2 * T + MARGIN:
            viol += 1
            print("  VIOLATION D")
    print(f"D  Weyl (entanglement spectrum): {cases} random pairs, {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- E: intervals, end to end
def part_E(n_random=200):
    cases = viol = 0
    for _ in range(n_random):
        p = Fraction(int(rng.integers(1, 10 ** 9)), 10 ** 9)
        iv = ent._f_point(p)
        lo, hi = iv.fractions()
        exact = -frac_mp(p) * mp.log(frac_mp(p))
        cases += 1
        if not (frac_mp(lo) - MARGIN <= exact <= frac_mp(hi) + MARGIN):
            viol += 1
            print("  VIOLATION E f", p)
        T = p / 2
        hT = frac_mp(Fraction(ent.binary_entropy_upper(T), ivl.ONE))
        exact_h = -(frac_mp(T) * mp.log(frac_mp(T)) + (1 - frac_mp(T)) * mp.log(1 - frac_mp(T)))
        cases += 1
        if hT + MARGIN < exact_h:
            viol += 1
            print("  VIOLATION E h", T)
        d = int(rng.integers(2, 65))
        fa = ent.fannes_audenaert_bound(T, d)
        if fa is not None:
            cases += 1
            if frac_mp(Fraction(fa, ivl.ONE)) + MARGIN < g_fa(frac_mp(T), d):
                viol += 1
                print("  VIOLATION E g", T, d)
    # range enclosure of f over intervals
    for _ in range(100):
        a = Fraction(int(rng.integers(0, 10 ** 6)), 10 ** 6)
        b = min(Fraction(1), a + Fraction(int(rng.integers(0, 10 ** 6)), 10 ** 6))
        iv = ent._f_interval(a, b)
        lo, hi = iv.fractions()
        cases += 1
        for p in (a, b, (a + b) / 2, a + (b - a) * Fraction(1, 7)):
            exact = -frac_mp(p) * mp.log(frac_mp(p)) if p > 0 else 0
            if not (frac_mp(lo) - MARGIN <= exact <= frac_mp(hi) + MARGIN):
                viol += 1
                print("  VIOLATION E range", a, b, p)
                break
    # end to end: random Hermitian H with a planted isolated eigenvector; the claimed vector is rotated by a known
    # angle; the record (fed s = exact sin theta rounded up) must contain the exact eigenvector's entropy
    e2e = e2e_viol = 0
    for _ in range(40):
        dA, dB = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        n = dA * dB
        cplx = bool(rng.integers(0, 2))
        Q = orth_basis(n, n, cplx)
        psi = Q[:, 0]
        ang = rng.uniform(0.0, 0.6)
        x = math.cos(ang) * psi + math.sin(ang) * Q[:, 1]
        x = x * rng.uniform(0.5, 2.0)
        s_exact = Fraction(math.sin(ang))
        s_up = float(ivl.to_float_up(ivl.from_fraction(s_exact).hi)) if ang > 0 else 0.0
        a_idx, b_idx = np.repeat(np.arange(dA), dB), np.tile(np.arange(dB), dA)
        Mr, Mi, t = ent.reduced_density_exact(x, a_idx, b_idx, dA, dB)
        encl = ent.enclose_psd_eigenvalues(Mr, Mi, t)
        S_hat = ent.entropy_interval(encl["lo"], encl["hi"])
        fa = ent.fannes_audenaert_bound(Fraction(s_up), dA)
        e2e += 1
        S_exact = entropy(partial_trace(to_mp(psi[:, None]) * to_mp(psi[:, None]).transpose_conj(), dA, dB))
        if fa is None:
            if Fraction(s_up) <= 1 - Fraction(1, dA):
                e2e_viol += 1
                print("  VIOLATION E2E: range refused inside the range")
            continue
        lo = frac_mp(Fraction(S_hat.lo - fa, ivl.ONE))
        hi = frac_mp(Fraction(S_hat.hi + fa, ivl.ONE))
        if not (lo - MARGIN <= S_exact <= hi + MARGIN):
            e2e_viol += 1
            print("  VIOLATION E2E", dA, dB, ang, mp.nstr(S_exact, 15), mp.nstr(lo, 15), mp.nstr(hi, 15))
    print(f"E  interval functions: {cases} cases, {viol} violations; end to end (exact sin theta fed in): {e2e} cases, "
          f"{e2e_viol} violations")
    return cases + e2e, viol + e2e_viol


if __name__ == "__main__":
    print(f"verify_C-03: seed {SEED}, mpmath dps {mp.mp.dps}")
    tot_c = tot_v = 0
    for part in (part_A, part_B, part_C, part_D, part_E):
        res = part()
        tot_c += res[0]
        tot_v += res[1]
    print(f"TOTAL: {tot_c} cases, {tot_v} violations")
