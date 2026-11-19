"""Computational verification of docs/proofs/E-07_davis_kahan.md (certify_ed.davis_kahan; C-12, C-13, E-07).

Parts (mpmath at 50 digits for the exact quantities; exact rationals where stated):
  A  Lemma 1 (Frobenius sin Theta): ||(I - P) Qhat||_F <= ||(I - P) R'||_F / delta for random A, Qhat with planted gaps
  B  Lemma 2 (polar identities): ||(I - Qhat Qhat^H) A Qhat||_F <= ||R||_F / sqrt(1 - eta) and
     ||H' - Lambda||_2 <= ||X^H R||_2 / (1 - eta) + delta spread / sqrt(1 - eta)
  C  Proposition 1 (E-07): exact sin theta_max = ||Qhat Qhat^H - P||_2 <= s_DK (exact ingredients); comparison with the
     Theorem 6 quotient; negative control: a planted eigenvalue inside the claimed gap (dist overstated)
  D  Proposition 2 (C-12): every eigenvalue of X^H S X (exact) is within D of an eigenvalue of S for random A, S with
     ||[A, S]|| = c planted; the alignment lemma ||Qhat - Q U|| <= s + s^2; negative control: c understated
  E  Proposition 3 (C-13): |tr(P_1 P_2) - tr(X_1 X_1^H X_2 X_2^H)| <= g (s_1 + eta_1) + g (1 + eta_1)(s_2 + eta_2)
  F  rounding bounds in exact rationals: commutator_bound and the fidelity rounding against exact products

Run:  python docs/proofs/verify_E-07.py [seed]
"""
from __future__ import annotations

import math
import os
import sys

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from certify_ed import davis_kahan as dk  # noqa: E402

mp.mp.dps = 50
MARGIN = mp.mpf(10) ** -40
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261030
rng = np.random.default_rng(SEED)


def to_mp(F):
    F = np.asarray(F)
    M = mp.matrix(F.shape[0], F.shape[1])
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            v = F[i, j]
            M[i, j] = mp.mpc(float(np.real(v)), float(np.imag(v))) if np.iscomplexobj(F) else mp.mpf(float(v))
    return M


def snorm(M):
    return max(mp.svd_c(M, compute_uv=False)) if M.rows and M.cols else mp.mpf(0)


def fnorm(M):
    return mp.sqrt(sum(abs(M[i, j]) ** 2 for i in range(M.rows) for j in range(M.cols)))


def eigs_h(M):
    cplx = any(isinstance(M[i, j], mp.mpc) and M[i, j].imag != 0 for i in range(M.rows) for j in range(M.cols))
    if cplx:
        return sorted(mp.re(e) for e in mp.eigh(M, eigvals_only=True))
    R = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            R[i, j] = mp.re(M[i, j])
    return sorted(mp.re(e) for e in mp.eigsy(R, eigvals_only=True))


def polar(X):
    U_, S_, Vh = mp.svd_c(X)
    return U_ * Vh, mp.diag(S_)


def planted(n, g, gap, eps, eta_scale, cplx):
    Qf, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
    lam = np.concatenate([rng.uniform(-0.1, 0.1, g), rng.choice([-1, 1], n - g) * (gap + rng.uniform(0, 1, n - g))])
    Qm = to_mp(Qf)
    A = Qm * mp.diag([mp.mpf(float(x)) for x in lam]) * Qm.transpose_conj()
    A = (A + A.transpose_conj()) / 2
    Q = Qm[:, :g]
    W = rng.standard_normal((n - g, g)) + (1j * rng.standard_normal((n - g, g)) if cplx else 0)
    W = Qf[:, g:] @ (W / np.linalg.norm(W, 2))
    M = np.eye(g) + eta_scale * rng.standard_normal((g, g))
    X = (Qf[:, :g] + eps * W) @ M
    P = Q * Q.transpose_conj()
    return A, to_mp(X), P, lam, Q


def ingredients(A, X, lam, g):
    """Exact (50-digit) ||R||_F, ||X^H R||_2, eta, delta, spread, dist, mu for the claim (X, mu = Rayleigh quotients)."""
    n = A.rows
    mu = [mp.re((X[:, j].transpose_conj() * A * X[:, j])[0, 0] / (X[:, j].transpose_conj() * X[:, j])[0, 0]) for j in range(g)]
    R = A * X - X * mp.diag(mu)
    G = X.transpose_conj() * X
    eta = max(abs(e) for e in eigs_h(G - mp.eye(g)))
    delta = 1 - mp.sqrt(1 - eta)
    spread = max(mu) - min(mu)
    others = [mp.mpf(float(x)) for x in lam[g:]]
    dist = min(abs(o - min(max(o, min(mu)), max(mu))) for o in others)
    return dict(R_fro=fnorm(R), XR=snorm(X.transpose_conj() * R), eta=eta, delta=delta, spread=spread, dist=dist, mu=mu, R=R, n=n)


# ---------------------------------------------------------------------------------------------- A: Frobenius sin Theta
def part_A(n_random=60):
    cases = viol = 0
    for _ in range(n_random):
        n, g = int(rng.integers(3, 10)), int(rng.integers(1, 4))
        g = min(g, n - 1)
        cplx = bool(rng.integers(0, 2))
        A, X, P, lam, Q = planted(n, g, rng.uniform(0.3, 3), 10.0 ** rng.uniform(-4, -0.5), 0.0, cplx)
        Qh, _ = polar(X)
        Hp = Qh.transpose_conj() * A * Qh
        Rp = A * Qh - Qh * Hp
        ritz = eigs_h(Hp)
        others = [mp.mpf(float(x)) for x in lam[g:]]
        delta = min(abs(o - r) for o in others for r in ritz)
        lhs = fnorm((mp.eye(n) - P) * Qh)
        rhs = fnorm((mp.eye(n) - P) * Rp) / delta
        cases += 1
        if lhs > rhs + MARGIN:
            viol += 1
            print("  VIOLATION A", mp.nstr(lhs, 12), mp.nstr(rhs, 12))
    print(f"A  Lemma 1 (Frobenius sin Theta): {cases} random cases, {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- B: polar identities
def part_B(n_random=60):
    cases = viol = 0
    for _ in range(n_random):
        n, g = int(rng.integers(3, 10)), int(rng.integers(1, 4))
        g = min(g, n - 1)
        cplx = bool(rng.integers(0, 2))
        A, X, P, lam, Q = planted(n, g, rng.uniform(0.3, 3), 10.0 ** rng.uniform(-4, -0.5), 10.0 ** rng.uniform(-4, -0.3), cplx)
        ing = ingredients(A, X, lam, g)
        if ing["eta"] >= 1:
            continue
        Qh, _ = polar(X)
        Hp = Qh.transpose_conj() * A * Qh
        lhs1 = fnorm((mp.eye(n) - Qh * Qh.transpose_conj()) * A * Qh)
        rhs1 = ing["R_fro"] / mp.sqrt(1 - ing["eta"])
        lhs2 = snorm(Hp - mp.diag(ing["mu"]))
        rhs2 = ing["XR"] / (1 - ing["eta"]) + ing["delta"] * ing["spread"] / mp.sqrt(1 - ing["eta"])
        cases += 2
        if lhs1 > rhs1 + MARGIN:
            viol += 1
            print("  VIOLATION B (i)")
        if lhs2 > rhs2 + MARGIN:
            viol += 1
            print("  VIOLATION B (ii)", mp.nstr(lhs2, 12), mp.nstr(rhs2, 12))
    print(f"B  Lemma 2 (polar identities (i), (ii)): {cases} random cases, {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- C: Proposition 1
def part_C(n_random=100):
    cases = viol = 0
    sharper = 0
    compared = 0
    adversarial = 0
    negative_hits = 0

    def check(A, X, P, lam, g, label=None, dist_override=None):
        nonlocal cases, viol
        ing = ingredients(A, X, lam, g)
        if ing["eta"] >= 1:
            return None
        dist = ing["dist"] if dist_override is None else dist_override
        rec = dk.sin_theta_davis_kahan(float(ing["R_fro"]) * (1 + 1e-14), float(ing["XR"]) * (1 + 1e-14), float(ing["eta"]) * (1 + 1e-14),
                                       float(ing["delta"]) * (1 + 1e-14) + 1e-300, float(ing["spread"]) * (1 + 1e-14) + 1e-300,
                                       float(dist) * (1 - 1e-14))
        Qh, _ = polar(X)
        sin_exact = snorm(Qh * Qh.transpose_conj() - P)
        cases += 1
        if rec["bound"] is not None and sin_exact > mp.mpf(rec["bound"]) + mp.mpf(10) ** -14:
            viol += 1
            if label != "negative":
                print("  VIOLATION C", label, mp.nstr(sin_exact, 12), rec["bound"])
            return True
        return False

    for _ in range(n_random):
        n, g = int(rng.integers(3, 12)), int(rng.integers(1, 4))
        g = min(g, n - 1)
        cplx = bool(rng.integers(0, 2))
        A, X, P, lam, Q = planted(n, g, rng.uniform(0.3, 3), 10.0 ** rng.uniform(-6, -1), 10.0 ** rng.uniform(-6, -1), cplx)
        check(A, X, P, lam, g)
        # comparison with the Theorem 6 quotient rho / (dist - eps*) using the same exact ingredients (information only)
        ing = ingredients(A, X, lam, g)
        if ing["eta"] >= 1:
            continue
        An = snorm(A)
        rho = ing["R_fro"] + (An + max(abs(m) for m in ing["mu"])) * ing["delta"]
        eps_star = ing["R_fro"] / mp.sqrt(1 - ing["eta"]) + ing["XR"] / (1 - ing["eta"]) + ing["delta"] * ing["spread"] / mp.sqrt(1 - ing["eta"])
        s6 = rho / (ing["dist"] - eps_star) if ing["dist"] > eps_star else mp.inf
        rec = dk.sin_theta_davis_kahan(float(ing["R_fro"]), float(ing["XR"]), float(ing["eta"]), float(ing["delta"]) + 1e-300,
                                       float(ing["spread"]) + 1e-300, float(ing["dist"]))
        compared += 1
        if rec["bound"] is not None and mp.mpf(rec["bound"]) <= s6:
            sharper += 1
    # adversarial: eta close to 1; the rotation close to the gap limit; g = n - 1; exact eigenvectors (sin theta = 0)
    for eta_scale in (0.3, 0.45):
        A, X, P, lam, Q = planted(8, 2, 1.0, 1e-3, eta_scale, True)
        adversarial += int(check(A, X, P, lam, 2, f"eta_scale={eta_scale}") is not None)
    A, X, P, lam, Q = planted(8, 1, 0.5, 0.4, 0.0, False)
    adversarial += int(check(A, X, P, lam, 1, "large rotation") is not None)
    A, X, P, lam, Q = planted(6, 5, 1.0, 1e-2, 1e-2, False)
    adversarial += int(check(A, X, P, lam, 5, "g = n - 1") is not None)
    A, X, P, lam, Q = planted(7, 2, 1.0, 0.0, 1e-3, True)
    adversarial += int(check(A, X, P, lam, 2, "exact eigenvectors") is not None)
    # negative control: dist overstated (an eigenvalue inside the claimed gap): the quotient may undercut sin theta
    for _ in range(20):
        A, X, P, lam, Q = planted(10, 1, 1.0, 0.3, 0.0, False)
        negative_hits += int(bool(check(A, X, P, lam, 1, "negative", dist_override=mp.mpf(50))))
    cases -= 20
    viol -= negative_hits
    print(f"C  Proposition 1 (E-07, exact sin theta <= s_DK): {cases} cases ({cases - adversarial} random + {adversarial} adversarial), "
          f"{viol} violations; s_DK <= Theorem 6 quotient in {sharper} of {compared} random cases; negative control (dist overstated to 50): "
          f"bound undercut in {negative_hits}/20 cases (expected > 0)")
    return cases, viol, negative_hits


# ---------------------------------------------------------------------------------------------- D: Proposition 2
def part_D(n_random=60):
    cases = viol = 0
    align_cases = align_viol = 0
    for trial in range(n_random):
        n, g = int(rng.integers(3, 10)), int(rng.integers(1, 3))
        g = min(g, n - 1)
        cplx = bool(rng.integers(0, 2))
        A, X, P, lam, Q = planted(n, g, rng.uniform(0.5, 3), 10.0 ** rng.uniform(-5, -1), 10.0 ** rng.uniform(-5, -1), cplx)
        # S: a diagonal (in A's eigenbasis) operator plus a perturbation -> [A, S] = c planted
        S0 = Q * mp.diag([mp.mpf(int(rng.integers(-2, 3))) for _ in range(g)]) * Q.transpose_conj()
        Zp = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
        Zm = to_mp(Zp)
        Zm = (Zm + Zm.transpose_conj()) / 2
        amp = mp.mpf(10) ** rng.uniform(-6, -1)
        # complete S0 to a full-rank Hermitian with the other eigenvectors of A (commuting part) and add the perturbation
        # the commuting part of S outside range P: a random Hermitian compressed to (I - P)
        Zc = (mp.eye(n) - P) * Zm * (mp.eye(n) - P)
        S = S0 + Zc + amp * Zm
        S = (S + S.transpose_conj()) / 2
        c_exact = snorm(A * S - S * A)
        c_used = c_exact
        ing = ingredients(A, X, lam, g)
        if ing["eta"] >= 1:
            continue
        Qh, _ = polar(X)
        s = snorm(Qh * Qh.transpose_conj() - P)
        eta = ing["eta"]
        Sn = snorm(S)
        # level enclosure J: the exact cluster eigenvalues (width w), gamma_0 = distance from its midpoint to the rest
        cl = sorted(float(x) for x in lam[:g])
        JL, JU = mp.mpf(cl[0]), mp.mpf(cl[-1])
        w = JU - JL
        mid = (JL + JU) / 2
        gamma_0 = min(abs(mp.mpf(float(o)) - mid) for o in lam[g:])
        M = X.transpose_conj() * S * X
        M = (M + M.transpose_conj()) / 2
        est = eigs_h(M)
        eigS = eigs_h(S)
        D = (2 * s + 2 * s * s + eta) * Sn + (c_used + Sn * w / 2) / gamma_0
        bad = any(min(abs(e - m) for e in eigS) > D + MARGIN for m in est)
        cases += 1
        if bad:
            viol += 1
            print("  VIOLATION D", trial)
        # alignment lemma: ||Qhat - Q U|| <= s + s^2 with U the polar factor of Q^H Qhat
        C_ = Q.transpose_conj() * Qh
        U_, _ = polar(C_)
        align_cases += 1
        if snorm(Qh - Q * U_) > s + s * s + MARGIN:
            align_viol += 1
            print("  VIOLATION D (alignment)")
    # tight case: A = diag(0, gamma), S = c' sigma_x, X = e_0 exactly (s = eta = w = 0): the label is 0 while the eigenvalues
    # of S are +-c', so the deviation equals c' = ||[A, S]|| / gamma exactly; the bound holds with equality and is broken
    # as soon as c is understated
    tight_cases = tight_viol = 0
    negative_hits = 0
    for _ in range(20):
        gam = mp.mpf(10) ** rng.uniform(-2, 1)
        cp = mp.mpf(10) ** rng.uniform(-3, 0)
        A2 = mp.matrix([[0, 0], [0, gam]])
        S2 = mp.matrix([[0, cp], [cp, 0]])
        c_exact = snorm(A2 * S2 - S2 * A2)              # = gamma c'
        dev = cp                                         # |0 - (+-c')|
        tight_cases += 1
        if dev > c_exact / gam + MARGIN:
            tight_viol += 1
            print("  VIOLATION D (tight)")
        negative_hits += int(dev > (c_exact / 2) / gam + MARGIN)
    print(f"D  Proposition 2 (C-12 label deviation): {cases} random cases + {tight_cases} tight adversarial cases (equality), "
          f"{viol + tight_viol} violations; alignment lemma: {align_cases} cases, {align_viol} violations; negative control "
          f"(c halved in the tight case): bound broken in {negative_hits}/20 cases (expected 20)")
    return cases + align_cases + tight_cases, viol + align_viol + tight_viol, negative_hits


# ---------------------------------------------------------------------------------------------- E: Proposition 3
def part_E(n_random=80):
    cases = viol = 0
    for _ in range(n_random):
        n, g = int(rng.integers(3, 10)), int(rng.integers(1, 4))
        g = min(g, n // 2)
        cplx = bool(rng.integers(0, 2))
        outs = []
        for _k in range(2):
            Qf, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
            Qm = to_mp(Qf)
            P = Qm[:, :g] * Qm[:, :g].transpose_conj()
            a = rng.uniform(0, 0.5)
            W = Qf[:, g:2 * g]
            X = (math.cos(a) * Qf[:, :g] + math.sin(a) * W) @ (np.eye(g) + 10.0 ** rng.uniform(-6, -0.5) * rng.standard_normal((g, g)))
            Xm = to_mp(X)
            Qh, _ = polar(Xm)
            s = snorm(Qh * Qh.transpose_conj() - P)
            eta = max(abs(e) for e in eigs_h(Xm.transpose_conj() * Xm - mp.eye(g)))
            outs.append((P, Xm, s, eta))
        (P1, X1, s1, e1), (P2, X2, s2, e2) = outs
        if e1 >= 1 or e2 >= 1:
            continue
        exact = mp.re(sum((P1 * P2)[i, i] for i in range(n)))
        comp = mp.re(sum((X1 * X1.transpose_conj() * X2 * X2.transpose_conj())[i, i] for i in range(n)))
        bound = g * (s1 + e1) + g * (1 + e1) * (s2 + e2)
        cases += 1
        if abs(exact - comp) > bound + MARGIN:
            viol += 1
            print("  VIOLATION E", mp.nstr(abs(exact - comp), 12), mp.nstr(bound, 12))
    print(f"E  Proposition 3 (C-13 overlap bound, exact projectors): {cases} random cases, {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- F: rounding, exact rationals
def part_F(n_random=30):
    cases = viol = 0

    for _ in range(n_random):
        n = int(rng.integers(3, 12))
        cplx = bool(rng.integers(0, 2))
        H = sp.random(n, n, density=rng.uniform(0.1, 0.6), random_state=int(rng.integers(1 << 30)), format="csr")
        S = sp.random(n, n, density=rng.uniform(0.1, 0.6), random_state=int(rng.integers(1 << 30)), format="csr")
        if cplx:
            H = H.multiply(np.exp(2j * np.pi * rng.random((n, 1)))).tocsr()
            S = S.multiply(np.exp(2j * np.pi * rng.random((n, 1)))).tocsr()
        H = (H + H.conj().T).tocsr() * 10.0 ** rng.uniform(-3, 3)
        S = (S + S.conj().T).tocsr()
        rec = dk.commutator_bound(H, S)
        Hd, Sd = H.toarray(), S.toarray()
        # exact commutator (50 digits) and its infinity norm, compared with the code's bound on ||[H, S]||_2 <= ||C||_inf
        Cm = to_mp(Hd) * to_mp(Sd) - to_mp(Sd) * to_mp(Hd)
        exact_inf = max(sum(abs(Cm[i, j]) for j in range(n)) for i in range(n))
        cases += 1
        if exact_inf > mp.mpf(rec["bound"]) + MARGIN:
            viol += 1
            print("  VIOLATION F (commutator)", mp.nstr(exact_inf, 15), rec["bound"])
        # fidelity rounding: exact tr(X1 X1^H X2 X2^H) vs the computed value within the rounding term
        g = int(rng.integers(1, 4))
        X1 = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
        X2 = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
        fr = dk.certified_fidelity(X1, 0.0, 0.0, X2, 0.0, 0.0)
        X1m, X2m = to_mp(X1), to_mp(X2)
        exact = mp.re(sum((X1m * X1m.transpose_conj() * X2m * X2m.transpose_conj())[i, i] for i in range(n)))
        cases += 1
        if abs(exact - fr["value"]) > mp.mpf(fr["terms"]["rounding"]) + MARGIN:
            viol += 1
            print("  VIOLATION F (fidelity rounding)", mp.nstr(abs(exact - fr["value"]), 6), fr["terms"]["rounding"])
    print(f"F  rounding bounds (commutator infinity norm, fidelity value) against 50-digit exact products: {cases} cases, {viol} violations")
    return cases, viol


if __name__ == "__main__":
    print(f"verify_E-07: seed {SEED}, mpmath dps {mp.mp.dps}")
    tot_c = tot_v = 0
    for part in (part_A, part_B, part_C, part_D, part_E, part_F):
        res = part()
        tot_c += res[0]
        tot_v += res[1]
    print(f"TOTAL: {tot_c} cases, {tot_v} violations")
