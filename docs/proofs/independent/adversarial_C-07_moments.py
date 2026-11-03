"""Independent adversarial check of C-07 (certified moments and sum rules), written from the statement only.

Part A (mp, 60 digits): Lemma 1 -- |tr(P M) - tr(X X^H M)| <= (2 g s + g eta) ||M|| with the TRUE s = ||Qhat Qhat^H - P||
and eta, for planted eigenspaces X = Q C + tilt, g = 1..4, eta up to 0.9, random Hermitian M.
Part B (exact rationals): Lemma 3/4 -- the entrywise/vector bounds e_y, eps_k, D_k of the repeated application and the
chain bound E of hermitian_matrix_with_bound against exact products of the stored binary64 data (sparse H, O with
cancellation-prone entries, scales 1e-3..1e3, complex).
Part C (mp): certified_moments end to end -- the M1 intervals contain the exact m_k = tr(P O H^k O)/g of the exact
eigenspace (mp eigenvectors of the stored H at 60 digits; s = true sin theta, eta = true Gram defect, rounded up), K up
to 4, random sparse/dense H and O, g = 1..3; M2 and M3 CERTIFIED (consistent) in every case; the M2/M3 Theorem-6
intervals contain the exact tr(P O^2)/g and tr(P[O,[H,O]])/(2g). Negative control: X replaced by vectors far from
the eigenspace with the same (false) s -- reports how often the sum rules flag the inconsistency (informational).

Usage: python docs/proofs/independent/adversarial_C-07_moments.py [--cases 60] [--seed 101]
"""
from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, mpf, to_mp, herm, eig_herm, spec_norm, herm_fun, Tally, report  # noqa: E402
from certify_ed.moments import certified_moments, hermitian_matrix_with_bound  # noqa: E402
from certify_ed.fp import gamma, gamma_dot, norm2_upper  # noqa: E402

mp.mp.dps = 60


def fc(z):
    z = complex(z)
    return (F(z.real), F(z.imag))


def rand_unitary(rng, n, cplx):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    Q, _ = np.linalg.qr(A)
    Q, _ = polar_Q(to_mp(Q))
    return Q


def polar_Q(X):
    G = herm(X) * X
    return X * herm_fun(G, lambda v: 1 / mp.sqrt(v)), G


def cols(V, idx):
    out = mp.matrix(V.rows, len(idx))
    for j, k in enumerate(idx):
        for r in range(V.rows):
            out[r, j] = V[r, k]
    return out


def rand_herm_sparse(rng, n, cplx, dens, scale, pm=False):
    A = rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
    if pm:
        A = rng.choice([-1.0, 1.0, 1 + 2.0 ** -52], size=(n, n)) * (rng.random((n, n)) < dens)
    if cplx:
        A = A + 1j * rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
    return sp.csr_matrix((A + A.conj().T) * scale)


def tr(M):
    return sum(M[i, i] for i in range(M.rows))


# ------------------------------------------------------------------------------------------------- Part A
def part_A(rng, cases, T):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 8)); g = int(rng.integers(1, min(4, n) + 1))
        V = rand_unitary(rng, n, cplx)
        P = cols(V, range(g)) * herm(cols(V, range(g)))
        Cm = to_mp(np.eye(g) + rng.standard_normal((g, g)) * float(rng.choice([0.0, 0.05, 0.3, 0.6])) / np.sqrt(g))
        X = cols(V, range(g)) * Cm
        if g < n:
            X = X + cols(V, range(g, n)) * to_mp(rng.standard_normal((n - g, g)) * float(rng.choice([0.0, 1e-6, 0.05, 0.4])))
        G = herm(X) * X; eta = spec_norm(G - mp.eye(g))
        if not eta < 1:
            T.skip(); continue
        Qh, _ = polar_Q(X); s = spec_norm(Qh * herm(Qh) - P)
        Md = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
        Mm = to_mp(Md); Mm = (Mm + herm(Mm)) / 2
        if i % 5 == 0:
            Mm = P - Qh * herm(Qh)                             # aligned with the projector error
        lhs = abs(mp.re(tr(P * Mm) - tr(X * herm(X) * Mm)))
        rhs = (2 * g * s + g * eta) * spec_norm(Mm)
        T.add(lhs <= rhs + mpf("1e-45"), i % 5 == 0 or eta > mpf("0.3"), f"L1 n={n} g={g} s={mp.nstr(s, 4)} eta={mp.nstr(eta, 4)} lhs={mp.nstr(lhs, 6)} rhs={mp.nstr(rhs, 6)}")


# ------------------------------------------------------------------------------------------------- Part B
def exact_matvec(M, x):
    Mc = sp.coo_matrix(M); n = x.size
    xf = [fc(v) for v in x]; y = [(F(0), F(0)) for _ in range(n)]
    for r, c, v in zip(Mc.row, Mc.col, Mc.data):
        a = fc(v); b = xf[c]
        y[r] = (y[r][0] + a[0] * b[0] - a[1] * b[1], y[r][1] + a[0] * b[1] + a[1] * b[0])
    return y


def part_B(rng, cases, T_vec, T_chain):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 12))
        H = rand_herm_sparse(rng, n, cplx, float(rng.choice([0.2, 1.0])), float(rng.choice([1e-3, 1.0, 1e3])), pm=i % 3 == 0)
        O = rand_herm_sparse(rng, n, cplx, float(rng.choice([0.2, 1.0])), float(rng.choice([1e-3, 1.0, 1e3])), pm=i % 3 == 1)
        x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0)
        x = np.asarray(x, dtype=complex if cplx else float)
        wH = int(np.diff(H.indptr).max()); wO = int(np.diff(O.indptr).max())
        # Lemma 3: e_y >= ||fl(O x) - O x||, eps >= ||fl(H z) - H z||
        y_fl = O @ x
        y_ex = exact_matvec(O, x)
        e_y = float(norm2_upper(gamma_dot(wO, cplx) * (abs(O) @ np.abs(x)) / (1 - gamma(wO))))
        err2 = sum((fc(y_fl[k])[0] - y_ex[k][0]) ** 2 + (fc(y_fl[k])[1] - y_ex[k][1]) ** 2 for k in range(n))
        ok = F(e_y) ** 2 >= err2
        z_fl = H @ y_fl
        z_ex = exact_matvec(H, y_fl)
        eps = float(norm2_upper(gamma_dot(wH, cplx) * (abs(H) @ np.abs(y_fl)) / (1 - gamma(wH))))
        err2z = sum((fc(z_fl[k])[0] - z_ex[k][0]) ** 2 + (fc(z_fl[k])[1] - z_ex[k][1]) ** 2 for k in range(n))
        ok = ok and F(eps) ** 2 >= err2z
        T_vec.add(ok, i % 3 in (0, 1), f"L3 n={n} cplx={cplx} e_y={e_y:.3e} err={float(err2) ** 0.5:.3e} eps={eps:.3e} errz={float(err2z) ** 0.5:.3e}")
        # Lemma 4: chains O O, O H O, H O O
        for factors, widths in (([O, O], [wO]), ([O, H, O], [wO, n]), ([H, O, O], [wH, n])):
            Tm, E = hermitian_matrix_with_bound(factors, widths)
            # exact product
            Fs = [[[fc(v) for v in row] for row in f.toarray()] for f in factors]
            prod = Fs[0]
            for Fm in Fs[1:]:
                new = [[(F(0), F(0)) for _ in range(n)] for _ in range(n)]
                for r in range(n):
                    for c in range(n):
                        acc = (F(0), F(0))
                        for k in range(n):
                            a = prod[r][k]; b = Fm[k][c]
                            acc = (acc[0] + a[0] * b[0] - a[1] * b[1], acc[1] + a[0] * b[1] + a[1] * b[0])
                        new[r][c] = acc
                prod = new
            okc = True
            for r in range(n):
                for c in range(n):
                    t = fc(Tm[r, c]); e = prod[r][c]
                    okc = okc and (t[0] - e[0]) ** 2 + (t[1] - e[1]) ** 2 <= F(E[r, c]) ** 2
            T_chain.add(okc, i % 3 in (0, 1), f"L4 n={n} cplx={cplx} chain of {len(factors)}: entrywise bound violated")


# ------------------------------------------------------------------------------------------------- Part C
def part_C(rng, cases, T_m, T_sr, T_neg):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(3, 9)); g = int(rng.integers(1, min(3, n - 1) + 1)); K = int(rng.integers(1, 5))
        H = rand_herm_sparse(rng, n, cplx, float(rng.choice([0.3, 1.0])), float(rng.choice([1e-2, 1.0, 1e2])))
        O = rand_herm_sparse(rng, n, cplx, float(rng.choice([0.3, 1.0])), float(rng.choice([1e-2, 1.0, 1e2])), pm=i % 4 == 0)
        Hm = to_mp(H.toarray()); Om = to_mp(O.toarray())
        lam, V = eig_herm(Hm)
        # cluster: g lowest eigenvalues; level = exact hull widened by 1e-12
        P = cols(V, range(g)) * herm(cols(V, range(g)))
        Vf = np.array([[complex(V[r, k]) if cplx else float(mp.re(V[r, k])) for k in range(g)] for r in range(n)])
        X = Vf @ (np.eye(g) + rng.standard_normal((g, g)) * float(rng.choice([0.0, 1e-8, 1e-3])))
        Xm = to_mp(X); G = herm(Xm) * Xm; eta = spec_norm(G - mp.eye(g))
        Qh, _ = polar_Q(Xm); s = spec_norm(Qh * herm(Qh) - P)
        up_ = lambda v: float(np.nextafter(float(v), np.inf)) * (1 + 1e-14) + 1e-300  # noqa: E731
        level = [float(np.nextafter(float(lam[0]), -np.inf)) - 1e-12, float(np.nextafter(float(lam[g - 1]), np.inf)) + 1e-12]
        rec = certified_moments(H, O, X, up_(s), up_(eta), K=K, level=level)
        st = {s_["id"]: s_ for s_ in rec["statements"]}
        if st["M1.moments"]["status"] != "CERTIFIED":
            T_m.skip(); continue
        ok = True
        Hk = mp.eye(n)
        for mrec in rec["moments"]:
            k = mrec["k"]
            Hk = mp.eye(n) if k == 0 else Hk * Hm
            m_true = mp.re(tr(P * Om * Hk * Om)) / g
            ok = ok and mpf(mrec["interval"][0]) <= m_true <= mpf(mrec["interval"][1])
        T_m.add(ok, i % 4 == 0, f"M1 n={n} g={g} K={K}: an m_k outside its interval")
        # sum rules: consistent, and the Theorem-6 intervals contain the exact quantities
        oksr = st["M2.sum_rule_zeroth"]["status"] == "CERTIFIED" and st["M3.sum_rule_first"]["status"] == "CERTIFIED"
        t2 = st["M2.sum_rule_zeroth"]["evidence"]["theorem6_interval"]
        m0_true = mp.re(tr(P * Om * Om)) / g
        oksr = oksr and mpf(t2[0]) <= m0_true <= mpf(t2[1])
        rhs = st["M3.sum_rule_first"]["evidence"]["rhs_interval"]
        Cm = 2 * Om * Hm * Om - Om * Om * Hm - Hm * Om * Om
        c_true = mp.re(tr(P * Cm)) / (2 * g)
        oksr = oksr and mpf(rhs[0]) <= c_true <= mpf(rhs[1])
        lhs = st["M3.sum_rule_first"]["evidence"]["lhs_interval"]
        oksr = oksr and mpf(lhs[0]) <= c_true <= mpf(lhs[1])
        T_sr.add(oksr, i % 4 == 0, f"M2/M3 n={n} g={g}: {st['M2.sum_rule_zeroth']['status']} {st['M3.sum_rule_first']['status']} m0={mp.nstr(m0_true, 8)} t6={t2} c={mp.nstr(c_true, 8)} rhs={rhs} lhs={lhs}")
        # negative control (informational): wrong vectors with the same claimed s
        Xw = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0.0)
        Xw, _ = np.linalg.qr(Xw)
        recw = certified_moments(H, O, Xw, up_(s), up_(eta), K=K, level=level)
        stw = {s_["id"]: s_["status"] for s_ in recw["statements"]}
        T_neg.add(True, stw["M2.sum_rule_zeroth"] == "FAILED" or stw["M3.sum_rule_first"] == "FAILED", f"neg {stw}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_A = Tally("Lemma 1: |tr(P M) - tr(X X^H M)| <= (2 g s + g eta) ||M|| (mp, true s and eta)")
    T_vec = Tally("Lemma 3: e_y >= ||fl(O x) - O x||, eps >= ||fl(H z) - H z|| (exact rationals)")
    T_chain = Tally("Lemma 4: entrywise chain bound of hermitian_matrix_with_bound (exact rationals)")
    T_m = Tally("M1: certified moment intervals contain the exact m_k of the exact eigenspace (mp)")
    T_sr = Tally("M2/M3: CERTIFIED (consistent) and the Theorem-6 / lhs intervals contain the exact quantities (mp)")
    T_neg = Tally("informational: wrong vectors with the claimed s -- 'adversarial' count = sum rules reported FAILED")
    part_A(rng, a.cases, T_A)
    part_B(rng, max(10, a.cases // 3), T_vec, T_chain)
    part_C(rng, a.cases, T_m, T_sr, T_neg)
    report([T_A, T_vec, T_chain, T_m, T_sr, T_neg])


if __name__ == "__main__":
    main()
