"""Independent adversarial check of C-04 (lift through the sector isometry), written from the statement only.

Part A (exact rationals): Lemma 1 -- ||Otilde_h - V^H O V||_F <= r_c for compress_operator, real and complex data,
sparse O/V with wide row/column widths, cancellation-prone rows (+-1 patterns), entries 1 + 2^-52, scales 1e-40 ..
1e40 (no underflow); also exact Hermiticity of Otilde_h.  The exact Frobenius norm squared is a Fraction.
Part B (mp, 60 digits): Proposition 1 with the TRUE iota = ||V^H V - I||_2: V = Q M with chosen singular values
(iota up to 0.999), random rank-g P_s (g = 1 .. d_s), random Hermitian O; checks the stated 2g bound and records how
often the g bound (independent proof) holds too.
Part C (end to end): certify_observable_lifted on a planted block/cluster with exact (mp) Pi_Q, Pi_V from the polar
factor of the stored V; the hypotheses (eta, s, projector_error) are supplied as exact values rounded up; real and
complex, Hermitian and non-Hermitian O; checks |v - tr(Pi_Q O)| <= bound and |v - tr(Pi_V O)| <= bound_V.

Usage: python docs/proofs/independent/adversarial_C-04_correlation_lift.py [--cases 150] [--seed 61]
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
from certify_ed.correlations import certify_observable_lifted, compress_operator  # noqa: E402
from certify_ed.eigenspace import EigenspaceCertificate  # noqa: E402

mp.mp.dps = 60


# ----------------------------------------------------------------------------------------------- exact helpers
def fc(z):
    """complex binary64 -> (Fraction re, Fraction im)"""
    z = complex(z)
    return (F(z.real), F(z.imag))


def fmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def fconj(a):
    return (a[0], -a[1])


def exact_compress(V, O):
    """V^H O V exactly as a dict {(l, k): (re, im)} from the stored binary64 sparse matrices."""
    Vc = sp.coo_matrix(V)
    Oc = sp.coo_matrix(O)
    Vcols = {}
    for r, c, v in zip(Vc.row, Vc.col, Vc.data):
        Vcols.setdefault(int(c), []).append((int(r), fc(v)))
    Orows = {}
    for r, c, v in zip(Oc.row, Oc.col, Oc.data):
        Orows.setdefault(int(r), []).append((int(c), fc(v)))
    # W = O V (exact, sparse dict of columns)
    W = {}
    for k, col in Vcols.items():
        acc = {}
        for j, vjk in col:
            # column k of W gets O[:, j] * v_jk: iterate rows i with O_ij != 0 -> need O by column; O Hermitian so
            # O[i, j] = conj(O[j, i]); use row j of O: O[j, i] -> O[i, j] = conj(O[j, i])
            for i, oji in Orows.get(j, []):
                oij = fconj(oji)
                t = fmul(oij, vjk)
                a = acc.get(i, (F(0), F(0)))
                acc[i] = (a[0] + t[0], a[1] + t[1])
        W[k] = acc
    out = {}
    for l, col in Vcols.items():
        for k, wk in W.items():
            s = (F(0), F(0))
            for i, vil in col:
                w = wk.get(i)
                if w is not None:
                    t = fmul(fconj(vil), w)
                    s = (s[0] + t[0], s[1] + t[1])
            if s != (F(0), F(0)):
                out[(l, k)] = s
    return out


def part_A(rng, cases, T):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(4, 14))
        d = int(rng.integers(1, min(n, 6) + 1))
        kind = i % 6
        scale = float(rng.choice([1.0, 1e-40, 1e40, 3.7]))
        # O Hermitian sparse
        dens = float(rng.choice([0.15, 0.4, 1.0]))
        A = rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
        if cplx:
            A = A + 1j * rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
        if kind == 2:                       # cancellation-prone: +-1 entries, dense
            A = rng.choice([-1.0, 1.0], size=(n, n)) * (1 + 0j if cplx else 1.0)
        if kind == 3:                       # entries 1 + 2^-52 and 1 - 2^-53
            A = rng.choice([1 + 2.0 ** -52, 1 - 2.0 ** -53, -1.0], size=(n, n)) * (1 + 0j if cplx else 1.0)
        O = (A + A.conj().T) / 2
        if kind == 3 or kind == 2:
            O = A + A.conj().T           # exact Hermitian (sums of a few powers of two)
        O = O * scale
        O = sp.csr_matrix(O)
        # V: sparse-ish n x d
        Vd = rng.standard_normal((n, d)) * (rng.random((n, d)) < float(rng.choice([0.3, 0.6, 1.0])))
        if cplx:
            Vd = Vd + 1j * rng.standard_normal((n, d))
        if kind == 4:                        # exact 0/1 isometry-like columns (sector basis)
            Vd = np.zeros((n, d), dtype=complex if cplx else float)
            for c in range(d):
                Vd[int(rng.integers(0, n)), c] = 1.0
        if kind == 5:
            Vd = Vd * 2.0 ** -40
        if not Vd.any():
            Vd[0, 0] = 1.0
        V = sp.csr_matrix(Vd)
        Oh, rnd, nrm = compress_operator(V, O)
        # exactness of the Hermitian part
        Ohd = Oh.toarray()
        herm_ok = np.array_equal(Ohd, Ohd.conj().T)
        ex = exact_compress(V, O)
        err2 = F(0)
        keys = set(ex) | {(int(r), int(c)) for r, c in zip(*Oh.nonzero())}
        for (l, k) in keys:
            e = ex.get((l, k), (F(0), F(0)))
            h = fc(Ohd[l, k])
            err2 += (h[0] - e[0]) ** 2 + (h[1] - e[1]) ** 2
        ok = herm_ok and err2 <= F(rnd) ** 2
        T.add(ok, kind in (2, 3, 4, 5), f"A n={n} d={d} cplx={cplx} kind={kind} scale={scale} err={float(err2) ** 0.5:.3e} rnd={rnd:.3e} herm={herm_ok}")


# ----------------------------------------------------------------------------------------------- Proposition 1
def rand_unitary(rng, n, cplx):
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
    Q, _ = np.linalg.qr(A)
    return Q


def part_B(rng, cases, T, T_g):
    for i in range(cases):
        cplx = i % 2 == 0
        n = int(rng.integers(3, 9)); d = int(rng.integers(1, n + 1))
        g = int(rng.integers(1, d + 1)) if i % 4 else d
        iota_t = float(rng.choice([1e-10, 1e-3, 0.1, 0.5, 0.9, 0.999]))
        # V = Q M, Q isometry, M eigenvalues in [sqrt(1-iota), sqrt(1+iota)] with both ends attained when d >= 2
        Qf = to_mp(rand_unitary(rng, n, cplx)[:, :d])
        Um = to_mp(rand_unitary(rng, d, cplx))
        ev = [mp.sqrt(1 + mpf(iota_t) * mpf(float(x))) for x in rng.uniform(-1, 1, size=d)]
        if d >= 2:
            ev[0] = mp.sqrt(1 - mpf(iota_t)); ev[1] = mp.sqrt(1 + mpf(iota_t))
        elif i % 2:
            ev[0] = mp.sqrt(1 - mpf(iota_t))
        M = Um * mp.diag(ev) * herm(Um)
        V = Qf * M
        G = herm(V) * V
        iota = spec_norm(G - mp.eye(d))                  # the true iota
        if not iota < 1:
            T.skip(); continue
        delta = 1 - mp.sqrt(1 - iota)
        # P_s: random rank-g projector in the block
        Wm = to_mp(rand_unitary(rng, d, cplx)[:, :g])
        Ps = Wm * herm(Wm)
        # O Hermitian
        Od = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
        Om = to_mp(Od); Om = (Om + herm(Om)) / 2
        if i % 5 == 0:                                    # O aligned with range V (worst case for the trace)
            Om = Qf * herm(Qf) * mpf(float(rng.choice([1.0, -3.0])))
        Onorm = spec_norm(Om)
        # polar factor from the exact SVD-free route: Q = V (V^H V)^{-1/2}
        Minv = herm_fun(G, lambda v: 1 / mp.sqrt(v))
        Qp = V * Minv
        PiV = V * Ps * herm(V); PiQ = Qp * Ps * herm(Qp)
        diff = abs(sum((PiV - PiQ)[r, c] * Om[c, r] for r in range(n) for c in range(n)))
        bound2 = 2 * g * Onorm * delta * (1 + mp.sqrt(1 + iota))
        bound1 = g * Onorm * delta * (1 + mp.sqrt(1 + iota))
        T.add(diff <= bound2 * (1 + mpf("1e-40")), iota_t >= 0.9 or g == d, f"B n={n} d={d} g={g} iota={mp.nstr(iota, 6)} diff={mp.nstr(diff, 8)} bound={mp.nstr(bound2, 8)}")
        T_g.add(diff <= bound1 * (1 + mpf("1e-40")), iota_t >= 0.9 or g == d, f"B(g) n={n} d={d} g={g} iota={mp.nstr(iota, 6)} diff={mp.nstr(diff, 8)} g-bound={mp.nstr(bound1, 8)}")


# ----------------------------------------------------------------------------------------------- end to end
def part_C(rng, cases, T):
    for i in range(cases):
        cplx = i % 3 == 1
        n = int(rng.integers(6, 13)); d = int(rng.integers(2, min(n, 7) + 1))
        g = int(rng.integers(1, min(d, 3) + 1))
        # block H_s with a planted g-cluster (binary64), X = its eigenvectors (binary64) mildly perturbed
        B = rng.standard_normal((d, d)) + (1j * rng.standard_normal((d, d)) if cplx else 0.0)
        Hs = (B + B.conj().T) / 2
        w, U = np.linalg.eigh(Hs)
        X = U[:, :g].copy()
        X = X * (1 + float(rng.choice([0.0, 1e-9, 1e-4]))) + rng.standard_normal(X.shape) * float(rng.choice([0.0, 1e-10, 1e-6]))
        Hs_mp = to_mp(Hs); lam, Vm = eig_herm(Hs_mp)
        Vg = mp.matrix(d, g)
        for j in range(g):
            for r in range(d):
                Vg[r, j] = Vm[r, j]
        Ps = Vg * herm(Vg)
        Xm = to_mp(X)
        eta = spec_norm(herm(Xm) * Xm - mp.eye(g))
        if not eta < 1:
            T.skip(); continue
        perr = spec_norm(Xm * herm(Xm) - Ps)
        s_true = max(perr - eta, mpf(0))
        up_ = lambda v: float(mp.mpf(v) * (1 + mpf("1e-14")) + mpf("1e-300"))  # noqa: E731
        sep = float(w[g] - w[g - 1]) if g < d else 1.0
        ec = EigenspaceCertificate(list(range(g)), [float(w[k]) for k in range(g)], sep, 0.0, up_(eta), up_(s_true), up_(perr))
        # V: n x d, near-isometry with chosen iota (stored binary64), sometimes an exact 0/1 sector basis
        if i % 4 == 0:
            Vd = np.zeros((n, d), dtype=complex if cplx else float)
            rows = rng.choice(n, size=d, replace=False)
            for c in range(d):
                Vd[rows[c], c] = 1.0
        else:
            Q0 = rand_unitary(rng, n, cplx)[:, :d]
            E = rng.standard_normal((d, d)) * float(rng.choice([0.0, 1e-8, 1e-3, 0.2]))
            Vd = Q0 @ (np.eye(d) + (E + E.T) / 2)
        V = sp.csr_matrix(Vd)
        Vm_ = to_mp(Vd); G = herm(Vm_) * Vm_
        iota_true = spec_norm(G - mp.eye(d))
        if not iota_true < 1:
            T.skip(); continue
        iota = up_(mp.sqrt(sum(abs((G - mp.eye(d))[r, c]) ** 2 for r in range(d) for c in range(d))))   # Frobenius, as S1
        # O: Hermitian (sparse) or non-Hermitian with exactly representable parts
        A = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.4) * float(rng.choice([1.0, 1e-30, 1e30]))
        if cplx:
            A = A + 1j * rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.4)
        nonherm = i % 5 == 4
        if nonherm:                                      # dyadic entries: the Hermitian parts are formed exactly
            A = rng.integers(-4, 5, size=(n, n)) * 0.125 * (rng.random((n, n)) < 0.5)
            if cplx:
                A = A + 1j * rng.integers(-4, 5, size=(n, n)) * 0.125
            A = A * float(rng.choice([1.0, 2.0 ** -90, 2.0 ** 90]))
        O = A if nonherm else (A + A.conj().T)
        Oc = sp.csr_matrix(O)
        rec = certify_observable_lifted(ec, X, V, Oc, "O", iota)
        if rec.get("status") == "INCONCLUSIVE":
            T.skip(); continue
        Om = to_mp(np.asarray(Oc.toarray()))
        Minv = herm_fun(G, lambda v: 1 / mp.sqrt(v)); Qp = Vm_ * Minv
        PiV = Vm_ * Ps * herm(Vm_); PiQ = Qp * Ps * herm(Qp)
        trV = sum((PiV * Om)[r, r] for r in range(n)); trQ = sum((PiQ * Om)[r, r] for r in range(n))
        if nonherm:
            vr, vi = rec["value"]
            v = mp.mpc(vr, vi)
            okQ = abs(v - trQ) <= mpf(rec["bound"]); okV = abs(v - trV) <= mpf(rec["bound_V"])
            # the parts are bounded separately: real and imaginary parts each within the part bounds
            okQ = okQ or (abs(mp.re(v - trQ)) <= mpf(rec["parts"]["hermitian"]["bound"]) and abs(mp.im(v - trQ)) <= mpf(rec["parts"]["anti_hermitian"]["bound"]))
        else:
            v = mpf(rec["value"])
            okQ = abs(v - mp.re(trQ)) <= mpf(rec["bound"]); okV = abs(v - mp.re(trV)) <= mpf(rec["bound_V"])
        T.add(okQ and okV, i % 4 == 0 or nonherm, f"C n={n} d={d} g={g} cplx={cplx} nonherm={nonherm} iota={iota:.3e} |v-trQ|={mp.nstr(abs(v - trQ), 6)} bound={rec['bound']:.6e} |v-trV|={mp.nstr(abs(v - trV), 6)} bound_V={rec['bound_V']:.6e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=150)
    ap.add_argument("--seed", type=int, default=61)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_A = Tally("Lemma 1: ||Otilde_h - V^H O V||_F <= r_c (exact rationals) and Otilde_h exactly Hermitian")
    T_B = Tally("Proposition 1: |tr(Pi_V O) - tr(Pi_Q O)| <= 2 g ||O|| delta (1 + sqrt(1 + iota)) (mp, true iota)")
    T_Bg = Tally("Proposition 1 with the sharper constant g (independent proof)")
    T_C = Tally("certify_observable_lifted end to end: |v - tr(Pi_Q O)| <= bound and |v - tr(Pi_V O)| <= bound_V (mp)")
    part_A(rng, a.cases, T_A)
    part_B(rng, a.cases, T_B, T_Bg)
    part_C(rng, max(20, a.cases // 3), T_C)
    report([T_A, T_B, T_Bg, T_C])


if __name__ == "__main__":
    main()
