"""Computational verification of docs/proofs/C-04_correlation_lift.md (certify_ed.correlations, sector lift).

Parts:
  A  the identity tr(P_s V^H O V) = tr(V P_s V^H O) in exact rational arithmetic (random rational V, P_s, O)
  B  Proposition 1 (isometry term): |tr(Pi_V O) - tr(Pi_Q O)| <= 2 g ||O||_2 delta (1 + sqrt(1 + iota)),
     delta = 1 - sqrt(1 - iota), with Q the exact polar factor of V (mpmath, 50 digits)
  C  Lemma 1 (compression rounding): ||fl(V^H O V)_h - V^H O V||_F <= rnd (compress_operator) in exact rationals
  D  end to end: the lifted record's interval contains tr(Pi_Q O) for a planted block decomposition
Each part prints its case count and its violations.

Run:  python docs/proofs/verify_C-04.py [seed]
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from certify_ed.correlations import certify_observable_lifted, compress_operator  # noqa: E402
from certify_ed.eigenspace import EigenspaceCertificate  # noqa: E402

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


def spec_norm(M):
    return max(mp.svd_c(M, compute_uv=False))


def polar(V):
    U_, S_, Vh = mp.svd_c(V)
    return U_ * Vh, S_


def rand_frac_matrix(r, c, den=97):
    return [[Fraction(int(rng.integers(-50, 51)), den) for _ in range(c)] for _ in range(r)]


def fmat(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def ftrans(A):
    return [list(r) for r in zip(*A)]


def ftrace(A):
    return sum(A[i][i] for i in range(len(A)))


def finv(A):
    """Exact inverse of a rational matrix (Gauss-Jordan)."""
    n = len(A)
    M = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


# ---------------------------------------------------------------------------------------------- A: the identity
def part_A(n_random=60):
    cases = viol = 0
    for _ in range(n_random):
        n, d, g = int(rng.integers(3, 8)), int(rng.integers(2, 5)), int(rng.integers(1, 3))
        d = min(d, n)
        g = min(g, d)
        V = rand_frac_matrix(n, d)
        Y = rand_frac_matrix(d, g)
        YtY = fmat(ftrans(Y), Y)
        Ps = fmat(fmat(Y, finv(YtY)), ftrans(Y))              # exact rational orthogonal projector of rank g
        O = rand_frac_matrix(n, n)
        O = [[(O[i][j] + O[j][i]) / 2 for j in range(n)] for i in range(n)]
        lhs = ftrace(fmat(Ps, fmat(ftrans(V), fmat(O, V))))
        rhs = ftrace(fmat(fmat(fmat(V, Ps), ftrans(V)), O))
        cases += 1
        if lhs != rhs:
            viol += 1
    print(f"A  identity tr(P_s V^H O V) = tr(V P_s V^H O) (exact rationals, real case): {cases} cases, {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- B: isometry term
def part_B(n_random=100):
    cases = viol = 0
    adversarial = 0

    def check(V, Ps, O, label=None):
        nonlocal cases, viol
        g = int(mp.re(sum(Ps[i, i] for i in range(Ps.rows))) + mp.mpf("0.5"))
        Q, S_ = polar(V)
        iota = spec_norm(V.transpose_conj() * V - mp.eye(V.cols))
        PiV = V * Ps * V.transpose_conj()
        PiQ = Q * Ps * Q.transpose_conj()
        diff = abs(sum(((PiV - PiQ) * O)[i, i] for i in range(O.rows)))
        if iota >= 1:
            bound = mp.inf
        else:
            delta = 1 - mp.sqrt(1 - iota)
            bound = 2 * g * spec_norm(O) * delta * (1 + mp.sqrt(1 + iota))
        cases += 1
        if diff > bound + MARGIN:
            viol += 1
            print("  VIOLATION B", label, mp.nstr(diff, 12), mp.nstr(bound, 12))

    def proj(d, g, cplx=True):
        Z = rng.standard_normal((d, g)) + (1j * rng.standard_normal((d, g)) if cplx else 0)
        Qz, _ = np.linalg.qr(Z)
        Qm = to_mp(Qz)
        return Qm * Qm.transpose_conj()

    def herm(n, cplx=True):
        A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
        Am = to_mp(A)
        return (Am + Am.transpose_conj()) / 2

    for _ in range(n_random):
        n, d = int(rng.integers(3, 9)), int(rng.integers(1, 5))
        d = min(d, n)
        g = int(rng.integers(1, d + 1))
        Qn, _ = np.linalg.qr(rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d)))
        iota = 10.0 ** rng.uniform(-8, -0.1)
        sv = 1 + iota * rng.uniform(-1, 1, size=d)
        sv[0] = 1 + iota if rng.integers(0, 2) else 1 - iota
        Rz, _ = np.linalg.qr(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
        V = to_mp(Qn) * to_mp(np.diag(np.sqrt(sv))) * to_mp(Rz)
        check(V, proj(d, g), herm(n))
    # adversarial: iota close to 1; g = d (P_s = I); O a projector aligned with the stretched direction; iota = 0
    n, d = 8, 4
    Qn, _ = np.linalg.qr(rng.standard_normal((n, d)))
    for iota in (0.9, 0.999, 0.999999):
        sv = np.array([1 + iota, 1.0, 1.0, 1 - iota])
        V = to_mp(Qn) * to_mp(np.diag(np.sqrt(sv)))
        check(V, proj(d, 1, False), herm(n, False), f"iota={iota}")
        check(V, mp.eye(d), herm(n, False), f"iota={iota}, g=d")
        e0 = mp.matrix(d, d)
        e0[0, 0] = 1
        q0 = to_mp(Qn[:, :1])
        check(V, e0, q0 * q0.transpose_conj(), f"iota={iota}, aligned")
        adversarial += 3
    check(to_mp(Qn), proj(d, 2, False), herm(n, False), "iota=0")
    adversarial += 1
    print(f"B  Proposition 1 (isometry term): {cases} cases ({cases - adversarial} random + {adversarial} adversarial), {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- C: compression rounding
def exact_err2(V, O, Oh):
    Vd, Od, Ohd = V.toarray(), O.toarray(), Oh.toarray()
    n, d = Vd.shape

    def F(z):
        return Fraction(float(np.real(z))), Fraction(float(np.imag(z)))

    err2 = Fraction(0)
    for i in range(d):
        for j in range(d):
            re = im = Fraction(0)
            for a in range(n):
                if Vd[a, i] == 0:
                    continue
                va_r, va_i = F(np.conj(Vd[a, i]))
                for b in range(n):
                    if Od[a, b] == 0 or Vd[b, j] == 0:
                        continue
                    o_r, o_i = F(Od[a, b])
                    vb_r, vb_i = F(Vd[b, j])
                    p_r, p_i = va_r * o_r - va_i * o_i, va_r * o_i + va_i * o_r
                    re += p_r * vb_r - p_i * vb_i
                    im += p_r * vb_i + p_i * vb_r
            c_r, c_i = F(Ohd[i, j])
            err2 += (re - c_r) ** 2 + (im - c_i) ** 2
    return err2


def part_C(n_random=40):
    cases = viol = 0
    adversarial = 0

    def check(V, O, label=None):
        nonlocal cases, viol
        Oh, rnd, _ = compress_operator(V, O)
        e2 = exact_err2(V, O, Oh)
        cases += 1
        if e2 > Fraction(rnd) ** 2:
            viol += 1
            print("  VIOLATION C", label, float(e2) ** 0.5, rnd)

    for _ in range(n_random):
        n, d = int(rng.integers(6, 26)), int(rng.integers(2, 7))
        cplx = bool(rng.integers(0, 2))
        V = sp.random(n, d, density=rng.uniform(0.05, 0.6), random_state=int(rng.integers(1 << 30)), format="csr")
        O = sp.random(n, n, density=rng.uniform(0.05, 0.5), random_state=int(rng.integers(1 << 30)), format="csr")
        if cplx:
            V = V.multiply(np.exp(2j * np.pi * rng.random((n, 1)))).tocsr()
            O = O.multiply(np.exp(2j * np.pi * rng.random((n, 1)))).tocsr()
        V = V.multiply(10.0 ** rng.uniform(-3, 3)).tocsr()
        O = (O + O.conj().T).tocsr()
        check(V, O)
    # adversarial: wide dynamic range, dense O, one nonzero per column of V (w_V = 1), cancellation (+-1 entries)
    n, d = 16, 4
    Vw = sp.csr_matrix(rng.standard_normal((n, d)) * 10.0 ** rng.uniform(-8, 8, size=(n, d)))
    Ow = sp.csr_matrix(rng.standard_normal((n, n)) * 10.0 ** rng.uniform(-8, 8, size=(n, n)))
    check(Vw, (Ow + Ow.T).tocsr(), "dynamic range")
    check(sp.csr_matrix(rng.standard_normal((n, d))), sp.csr_matrix((lambda A: A + A.T)(rng.standard_normal((n, n)))), "dense")
    V1 = sp.csr_matrix((np.ones(d), (rng.choice(n, d, replace=False), np.arange(d))), shape=(n, d))
    check(V1, sp.csr_matrix((lambda A: A + A.T)(rng.standard_normal((n, n)))), "w_V = 1")
    Vc = sp.csr_matrix(rng.choice([-1.0, 1.0], size=(n, d)))
    Oc = sp.csr_matrix(rng.choice([-1.0, 1.0], size=(n, n)) + 1e-9 * rng.standard_normal((n, n)))
    check(Vc, (Oc + Oc.T).tocsr(), "cancellation")
    adversarial += 4
    print(f"C  Lemma 1 (compression rounding, exact rationals): {cases} cases ({cases - adversarial} random + {adversarial} adversarial), {viol} violations")
    return cases, viol


# ---------------------------------------------------------------------------------------------- D: end to end
def part_D(n_random=40):
    cases = viol = 0
    for _ in range(n_random):
        n, d = int(rng.integers(6, 12)), int(rng.integers(2, 5))
        cplx = bool(rng.integers(0, 2))
        Qf, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
        iota = 10.0 ** rng.uniform(-10, -2)
        sv = np.sqrt(1 + iota * rng.uniform(-1, 1, size=d))
        V = Qf[:, :d] @ np.diag(sv)
        Hb = rng.standard_normal((d, d)) + (1j * rng.standard_normal((d, d)) if cplx else 0)
        Hs = (Hb + Hb.conj().T) / 2
        w, Y = np.linalg.eigh(Hs)
        O = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
        O = (O + O.conj().T) / 2
        # exact (50-digit) eigenvector of the stored H_s, its sin theta to the float eigenvector, exact Pi_Q
        Hm = to_mp(Hs)
        E, Qm = mp.eighe(Hm) if cplx else mp.eigsy(to_mp(Hs.real))
        k = int(np.argmin([abs(E[i] - w[0]) for i in range(d)]))
        psi = Qm[:, k]
        x = Y[:, 0]
        xm = to_mp(x[:, None])
        xm = xm / mp.sqrt((xm.transpose_conj() * xm)[0, 0])
        ov = abs((psi.transpose_conj() * xm)[0, 0])
        s = mp.sqrt(max(mp.mpf(0), 1 - ov ** 2))
        s_up = float(s) * (1 + 1e-12) + 1e-300
        eta_exact = abs(float(np.real(np.vdot(x, x))) - 1)
        ec = EigenspaceCertificate([0], [float(w[0])], 1.0, 0.0, eta_exact * (1 + 1e-12), s_up, s_up + eta_exact * (1 + 1e-12))
        iota_up = float(spec_norm(to_mp(V).transpose_conj() * to_mp(V) - mp.eye(d))) * (1 + 1e-12)
        rec = certify_observable_lifted(ec, x[:, None], sp.csr_matrix(V), sp.csr_matrix(O), "O", iota_up)
        Qp, _ = polar(to_mp(V))
        PiQ = Qp * (psi * psi.transpose_conj()) * Qp.transpose_conj()
        tQ = mp.re(sum((PiQ * to_mp(O))[i, i] for i in range(n)))
        cases += 1
        if abs(tQ - rec["value"]) > rec["bound"]:
            viol += 1
            print("  VIOLATION D", mp.nstr(tQ, 15), rec["value"], rec["bound"])
    print(f"D  end to end (planted block decomposition, exact Pi_Q): {cases} cases, {viol} violations")
    return cases, viol


if __name__ == "__main__":
    print(f"verify_C-04: seed {SEED}, mpmath dps {mp.mp.dps}")
    tot_c = tot_v = 0
    for part in (part_A, part_B, part_C, part_D):
        c, v = part()
        tot_c += c
        tot_v += v
    print(f"TOTAL: {tot_c} cases, {tot_v} violations")
