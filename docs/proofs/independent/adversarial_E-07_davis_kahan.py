"""Independent adversarial check of E-07 / C-12 / C-13, written from the statement only (mp, 60 digits).

Part A: Proposition 1 -- sin_theta_davis_kahan with EXACT inputs (||R||_F, ||X^H R||_2, delta, spread, dist, all
rounded up/down as the code expects) against the exact sin theta_max = ||Qhat Qhat^H - P||_2 for planted clusters:
g = 1..4, eta up to 0.9, residuals from 0 to the separation, complex data, clusters next to the rest, and model-shift
cases (A_true = A + Delta, ||Delta||_2 = sigma with Delta X = sigma Y for orthonormal Y outside the cluster: the
Frobenius worst case ||Delta X||_F = sigma sqrt(g)); the residuals are those of A, the truth is P of A_true.
Part B: Lemma 3 -- ||Qhat - Q U|| <= s + s^2 at the exact s = sin theta_max, incl. s close to 1 and g = n - 1.
Part C: Proposition 2 -- every eigenvalue of X^H S X within (2s + 2s^2 + eta)||S|| + (c + ||S|| w/2)/gamma_0 of an
eigenvalue of S, with c = ||[A, S]||_2 exact, S = commuting part + non-commuting part of controlled size, J the exact
hull of the cluster (w = 0 and w > 0), gamma_0 exact; and approximate_symmetry_label end to end on the same data.
Part D: Proposition 3 / Lemma 5 (c) -- certified_fidelity against the exact tr(P1 P2) with true s_i, eta_i, g = 1..3,
complex, parallel / orthogonal / generic pairs, and X scaled by 1 +- eta.
Part E: Lemma 5 (a) -- commutator_bound vs the exact ||[H, S]||_2 (exact rational commutator, mp norm), sparse and
dense, cancellation-prone entries.

Usage: python docs/proofs/independent/adversarial_E-07_davis_kahan.py [--cases 100] [--seed 83]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, mpf, to_mp, herm, eig_herm, spec_norm, herm_fun, Tally, report  # noqa: E402
from certify_ed.davis_kahan import sin_theta_davis_kahan, commutator_bound, approximate_symmetry_label, certified_fidelity  # noqa: E402
from certify_ed.eigenspace import EigenspaceCertificate  # noqa: E402

mp.mp.dps = 60


def up_(v):
    return float(np.nextafter(float(mp.mpf(v)), np.inf))


def dn_(v):
    return float(np.nextafter(float(mp.mpf(v)), -np.inf))


def rand_unitary(rng, n, cplx):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    Q, _ = np.linalg.qr(A)
    return Q


def mp_unitary(rng, n, cplx):
    """binary64 QR factor re-orthonormalised in mp (polar factor): unitary to ~1e-58."""
    Q, _ = polar_Q(to_mp(rand_unitary(rng, n, cplx)))
    return Q


def fro(M):
    return mp.sqrt(sum(abs(M[i, j]) ** 2 for i in range(M.rows) for j in range(M.cols)))


def cols(V, idx):
    out = mp.matrix(V.rows, len(idx))
    for j, k in enumerate(idx):
        for r in range(V.rows):
            out[r, j] = V[r, k]
    return out


def polar_Q(X):
    G = herm(X) * X
    return X * herm_fun(G, lambda v: 1 / mp.sqrt(v)), G


# ------------------------------------------------------------------------------------------------- Part A
def part_A(rng, cases, T, T_shift):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(3, 9)); g = int(rng.integers(1, min(4, n - 1) + 1))
        V = mp_unitary(rng, n, cplx)
        lam_c = [mpf(float(x)) for x in rng.uniform(-0.05, 0.05, size=g)]
        if i % 5 == 0:
            lam_c = [mpf(0)] * g                                     # exactly degenerate cluster
        gap = mpf(float(rng.choice([0.3, 1.0, 5.0])))
        lam_o = [gap + mpf(float(x)) for x in rng.uniform(0, 3, size=n - g)]
        if i % 3 == 0 and n - g >= 2:
            lam_o[1] = -gap - mpf(float(rng.uniform(0, 1)))         # rest on both sides
        lam = lam_c + lam_o
        A = V * mp.diag(lam) * herm(V)
        P = cols(V, range(g)) * herm(cols(V, range(g)))
        # X: cluster basis mixed (eta) and tilted towards the rest (residual)
        Cm = to_mp(np.eye(g) + rng.standard_normal((g, g)) * float(rng.choice([0.0, 0.01, 0.1, 0.4])) / np.sqrt(g))
        tilt = float(rng.choice([0.0, 1e-8, 1e-3, 0.05, 0.3]))
        Out = to_mp(rng.standard_normal((n - g, g)) * tilt)
        X = cols(V, range(g)) * Cm + cols(V, range(g, n)) * Out
        mu = [mp.re((herm(X) * A * X)[j, j] / (herm(X) * X)[j, j]) for j in range(g)]   # Rayleigh quotients
        if i % 7 == 0:
            mu = [m + mpf(float(x)) for m, x in zip(mu, rng.uniform(-1e-3, 1e-3, size=g))]
        Lam = mp.diag(mu)
        R = A * X - X * Lam
        G = herm(X) * X
        eta = spec_norm(G - mp.eye(g))
        if not eta < 1:
            T.skip(); continue
        delta = 1 - mp.sqrt(1 - eta)
        spread = max(mu) - min(mu)
        dist = min(abs(l - m) for l in lam_o for m in (min(mu), max(mu)) if True)
        dist = min(min(abs(l - min(mu)), abs(l - max(mu))) if not (min(mu) <= l <= max(mu)) else mpf(0) for l in lam_o)
        if not dist > 0:
            T.skip(); continue
        Qh, _ = polar_Q(X)
        sin_true = spec_norm(Qh * herm(Qh) - P)
        rec = sin_theta_davis_kahan(up_(fro(R)), up_(spec_norm(herm(X) * R)), up_(eta), up_(delta), up_(spread), dn_(dist))
        if rec["bound"] is None:
            T.skip(); continue
        T.add(sin_true <= mpf(rec["bound"]) + mpf("1e-50"), tilt >= 0.05 or i % 5 == 0 or i % 7 == 0,
              f"P1 n={n} g={g} eta={mp.nstr(eta, 4)} tilt={tilt} sin={mp.nstr(sin_true, 8)} s_DK={rec['bound']:.8e}")
        # model shift: A_true = A + Delta, Delta X-part isometric onto g orthonormal directions outside the cluster
        if n - g >= g:
            sigma = mpf(float(rng.choice([1e-6, 1e-3, 0.02])))
            Qc = cols(V, range(g)); Yo = cols(V, range(g, 2 * g))
            Delta = sigma * (Yo * herm(Qc) + Qc * herm(Yo))           # Hermitian, ||Delta||_2 = sigma, Delta Qc = sigma Yo
            A_true = A + Delta
            lam_t, V_t = eig_herm(A_true)
            # the cluster of A_true: the g eigenvalues closest to the cluster values
            order = sorted(range(n), key=lambda k: min(abs(lam_t[k] - m) for m in mu))[:g]
            P_t = cols(V_t, order) * herm(cols(V_t, order))
            others_t = [lam_t[k] for k in range(n) if k not in order]
            dist_t = min(min(abs(l - min(mu)), abs(l - max(mu))) if not (min(mu) <= l <= max(mu)) else mpf(0) for l in others_t)
            if not dist_t > 0:
                T_shift.skip(); continue
            sin_t = spec_norm(Qh * herm(Qh) - P_t)
            rec2 = sin_theta_davis_kahan(up_(fro(R)), up_(spec_norm(herm(X) * R)), up_(eta), up_(delta), up_(spread), dn_(dist_t),
                                         model_shift=up_(sigma))
            if rec2["bound"] is None:
                T_shift.skip(); continue
            T_shift.add(sin_t <= mpf(rec2["bound"]) + mpf("1e-50"), True,
                        f"P1-shift n={n} g={g} sigma={mp.nstr(sigma, 3)} eta={mp.nstr(eta, 4)} tilt={tilt} sin={mp.nstr(sin_t, 8)} s_DK={rec2['bound']:.8e} (Frobenius shift would be {mp.nstr(sigma * mp.sqrt(g), 4)})")


# ------------------------------------------------------------------------------------------------- Part B
def part_B(rng, cases, T):
    for i in range(cases):
        cplx = i % 2 == 0
        n = int(rng.integers(2, 8)); g = int(rng.integers(1, n))
        Q = cols(mp_unitary(rng, n, cplx), range(g))
        Qh = cols(mp_unitary(rng, n, cplx), range(g))
        if i % 3 == 0:                                   # Qhat close to Q
            Qh = Q * to_mp(rand_unitary(rng, g, cplx)) + to_mp(rng.standard_normal((n, g)) * float(rng.choice([1e-6, 1e-2, 0.2])))
            Qh, _ = polar_Q(Qh)
        P = Q * herm(Q)
        s = spec_norm(Qh * herm(Qh) - P)
        C = herm(Q) * Qh
        # unitary polar factor U of C = U K: U = C (C^H C)^{-1/2} when C is nonsingular
        K2 = herm(C) * C
        if min(mp.re(x) for x in mp.eigh(K2)[0]) <= mpf("1e-40"):
            T.skip(); continue
        Uu = C * herm_fun(K2, lambda v: 1 / mp.sqrt(v))
        lhs = spec_norm(Qh - Q * Uu)
        T.add(lhs <= s + s * s + mpf("1e-45"), s > mpf("0.5") or g == n - 1, f"L3 n={n} g={g} s={mp.nstr(s, 6)} lhs={mp.nstr(lhs, 8)} bound={mp.nstr(s + s * s, 8)}")


# ------------------------------------------------------------------------------------------------- Part C
def part_C(rng, cases, T, T_code):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(3, 8)); g = int(rng.integers(1, min(3, n - 1) + 1))
        V = mp_unitary(rng, n, cplx)
        lam_c = [mpf(float(x)) for x in rng.uniform(-0.02, 0.02, size=g)]
        gap = mpf(float(rng.choice([0.5, 1.0, 3.0])))
        lam_o = [gap + mpf(float(x)) for x in rng.uniform(0, 2, size=n - g)]
        if i % 3 == 0 and n - g >= 2:
            lam_o[0] = -gap - mpf(float(rng.uniform(0, 1)))
        A = V * mp.diag(lam_c + lam_o) * herm(V)
        # S: commuting part (same eigenbasis, integer eigenvalues) + non-commuting Hermitian part of size kappa
        Sc = V * mp.diag([mpf(int(x)) for x in rng.integers(-2, 3, size=n)]) * herm(V)
        kappa = float(rng.choice([0.0, 1e-9, 1e-4, 1e-2]))
        Nm = to_mp(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)) * mpf(kappa)
        S = Sc + (Nm + herm(Nm)) / 2
        comm = A * S - S * A
        c = spec_norm(comm)
        S_norm = spec_norm(S)
        qS, _ = eig_herm(S)
        # cluster vectors X (mixed + tilted), exact s and eta
        Cm = to_mp(np.eye(g) + rng.standard_normal((g, g)) * float(rng.choice([0.0, 0.05, 0.3])) / np.sqrt(g))
        Out = to_mp(rng.standard_normal((n - g, g)) * float(rng.choice([0.0, 1e-6, 1e-2, 0.1])))
        X = cols(V, range(g)) * Cm + cols(V, range(g, n)) * Out
        G = herm(X) * X
        eta = spec_norm(G - mp.eye(g))
        if not eta < 1:
            T.skip(); continue
        Qh, _ = polar_Q(X)
        P = cols(V, range(g)) * herm(cols(V, range(g)))
        s = spec_norm(Qh * herm(Qh) - P)
        if not s < 1:
            T.skip(); continue
        # J: exact hull of the cluster, possibly widened
        JL, JU = min(lam_c), max(lam_c)
        if i % 4 == 0:
            JL -= mpf(float(rng.uniform(0, 0.1))); JU += mpf(float(rng.uniform(0, 0.1)))
        w = JU - JL; zeta = (JL + JU) / 2
        gamma0 = min(abs(l - zeta) for l in lam_o)
        D = (2 * s + 2 * s * s + eta) * S_norm + (c + S_norm * w / 2) / gamma0
        Mx = herm(X) * S * X; Mx = (Mx + herm(Mx)) / 2
        mh, _ = eig_herm(Mx)
        ok = all(min(abs(q - m) for q in qS) <= D + mpf("1e-45") for m in mh)
        T.add(ok, kappa >= 1e-2 or i % 4 == 0, f"P2 n={n} g={g} kappa={kappa} s={mp.nstr(s, 4)} eta={mp.nstr(eta, 4)} D={mp.nstr(D, 6)}")
        # the code: EigenspaceCertificate with exact (rounded) s, eta and sep = gamma0 - w/2 (so that sep + w/2 = gamma0)
        Xf = np.array([[complex(X[r, k]) if cplx else float(mp.re(X[r, k])) for k in range(g)] for r in range(n)])
        Sf = np.array([[complex(S[r, k]) if cplx else float(mp.re(S[r, k])) for k in range(n)] for r in range(n)])
        Sf = (Sf + Sf.conj().T) / 2
        sep = dn_(gamma0 - w / 2)
        ec = EigenspaceCertificate(list(range(g)), [float(x) for x in lam_c], sep, 0.0, up_(eta), up_(s), up_(s + eta))
        rec = approximate_symmetry_label(ec, Xf, sp.csr_matrix(Sf), "S", up_(c * (1 + mpf("1e-12")) + mpf("1e-300")), (dn_(JL), up_(JU)))
        if rec.get("status") == "INCONCLUSIVE" and "distance" in rec.get("reason", ""):
            T_code.skip(); continue
        Dc = mpf(rec["bound"])
        # the stored S (binary64) differs from the mp S by rounding; compare against the eigenvalues of the stored S
        qSf = [mpf(x) for x in np.linalg.eigvalsh(Sf)]
        okc = all(min(abs(q - mpf(m)) for q in qSf) <= Dc + mpf("1e-12") * (1 + abs(mpf(m))) for m in rec["estimates"])
        T_code.add(okc, kappa >= 1e-2, f"P2-code n={n} g={g} kappa={kappa} D={rec['bound']:.6e} status={rec.get('status')}")


# ------------------------------------------------------------------------------------------------- Part D
def part_D(rng, cases, T):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 8)); g = int(rng.integers(1, min(3, n) + 1))
        V1 = mp_unitary(rng, n, cplx); V2 = mp_unitary(rng, n, cplx)
        kind = i % 4
        if kind == 1:
            V2 = V1                                                     # identical subspaces
        if kind == 2 and n >= 2 * g:
            V2 = cols(V1, list(range(g, 2 * g)) + list(range(0, g)) + list(range(2 * g, n)))   # orthogonal subspaces
        P1 = cols(V1, range(g)) * herm(cols(V1, range(g))); P2 = cols(V2, range(g)) * herm(cols(V2, range(g)))
        tr_true = mp.re(sum((P1 * P2)[r, r] for r in range(n)))
        Xs = []
        for V in (V1, V2):
            Cm = to_mp(np.eye(g) + rng.standard_normal((g, g)) * float(rng.choice([0.0, 0.01, 0.2])) / np.sqrt(g))
            Out = to_mp(rng.standard_normal((n - g, g)) * float(rng.choice([0.0, 1e-7, 1e-3, 0.05]))) if n > g else mp.matrix(0, g)
            X = cols(V, range(g)) * Cm + (cols(V, range(g, n)) * Out if n > g else 0)
            Xs.append(X)
        pars = []
        for X, P in zip(Xs, (P1, P2)):
            G = herm(X) * X; eta = spec_norm(G - mp.eye(g))
            if not eta < 1:
                break
            Qh, _ = polar_Q(X); s = spec_norm(Qh * herm(Qh) - P)
            pars.append((up_(s), up_(eta)))
        if len(pars) < 2:
            T.skip(); continue
        tofl = lambda X: np.array([[complex(X[r, k]) if cplx else float(mp.re(X[r, k])) for k in range(g)] for r in range(n)])  # noqa: E731
        X1f, X2f = tofl(Xs[0]), tofl(Xs[1])
        rec = certified_fidelity(X1f, pars[0][0], pars[0][1], X2f, pars[1][0], pars[1][1])
        # the stored X differ from the mp X by rounding: s, eta are for the mp X; the Theorem-6 bound ||X X^H - P|| <= s + eta
        # must hold for the stored X -> recompute s + eta for the stored X exactly and use the larger
        ok_hyp = True
        for Xf, P, (s_, e_) in zip((X1f, X2f), (P1, P2), pars):
            Xm = to_mp(Xf); pe = spec_norm(Xm * herm(Xm) - P)
            ok_hyp = ok_hyp and pe <= mpf(s_) + mpf(e_) + mpf("1e-14")
        lo, hi = rec["interval"]
        T.add((not ok_hyp) or (mpf(lo) - mpf("1e-14") <= tr_true <= mpf(hi) + mpf("1e-14")), kind in (1, 2),
              f"P3 n={n} g={g} kind={kind} tr={mp.nstr(tr_true, 10)} interval=({lo}, {hi})")


# ------------------------------------------------------------------------------------------------- Part E
def part_E(rng, cases, T):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 8))
        dens = float(rng.choice([0.3, 1.0]))
        def rh():
            A = rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
            if cplx:
                A = A + 1j * rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
            if i % 5 == 0:
                A = rng.choice([-1.0, 1.0, 1 + 2.0 ** -52], size=(n, n)) * (1 + 0j if cplx else 1.0)
            return A + A.conj().T
        H, S = rh(), rh() * float(rng.choice([1.0, 1e-30, 1e30]))
        if i % 4 == 0:
            S = np.diag(rng.integers(-3, 4, size=n)).astype(complex if cplx else float)   # a diagonal label operator
        b = commutator_bound(sp.csr_matrix(H), sp.csr_matrix(S))["bound"]
        Hm, Sm = to_mp(H), to_mp(S)
        c_true = spec_norm(Hm * Sm - Sm * Hm)
        T.add(c_true <= mpf(b), i % 5 == 0 or i % 4 == 0, f"L5a n={n} cplx={cplx} c={mp.nstr(c_true, 8)} bound={b:.8e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=83)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    TA = Tally("Proposition 1: sin theta_max <= s_DK (exact inputs rounded outward; mp truth)")
    TAs = Tally("Proposition 1 with a model shift sigma (A_true = A + Delta, ||Delta X||_F = sigma sqrt(g))")
    TB = Tally("Lemma 3: ||Qhat - Q U||_2 <= s + s^2 (mp)")
    TC = Tally("Proposition 2: labels within D of an eigenvalue of S (exact c, gamma_0; mp)")
    TCc = Tally("approximate_symmetry_label end to end (stored S; exact hypotheses rounded outward)")
    TD = Tally("Proposition 3 / Lemma 5(c): certified_fidelity interval contains tr(P1 P2) (mp)")
    TE = Tally("Lemma 5(a): commutator_bound >= ||[H, S]||_2 (mp from stored matrices)")
    part_A(rng, a.cases, TA, TAs)
    part_B(rng, a.cases, TB)
    part_C(rng, a.cases, TC, TCc)
    part_D(rng, a.cases, TD)
    part_E(rng, a.cases, TE)
    report([TA, TAs, TB, TC, TCc, TD, TE])


if __name__ == "__main__":
    main()
