"""Independent adversarial search for INT-05 (docs/proofs/int05_underflow.md).

Exact ground truth: fractions.Fraction for everything rational (binary64 numbers, their products and sums, the
exact residual r = Ax - mu x for real data, the squared norm), mpmath (60 digits) for square roots and complex
moduli. Targets: (M-U) on products/sums/sqrt straddling 2^-1022 and the 2^-1074 grid; Lemma U1 on dot products
with subnormal products (real and complex); Lemma U2 on exact_dot with tiny, subnormal x huge, cancelling and
mixed pairs; Lemma U3 on vectors with entries 2^-1074 k (norm below 2^-1022, N = 1..6, complex); Lemma U4 and
Theorem U5 on residuals with subnormal x, subnormal A, A subnormal with x ~ 2^1000 (the hypot-eta family for
complex data), zero residual, w = n and w = 1; Theorem U6 against the old chain re-formed as in the frozen
reference; Corollary U7's gram_additive_term on a BLAS Gram matrix of subnormal vectors.

Usage: python docs/proofs/independent/adversarial_int05_underflow.py [--cases 400] [--seed 23]
"""
import argparse
import math
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, Tally, mpf, report  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, _rounding_bound, residuals  # noqa: E402
from certify_ed.fp import (U, ULP_MIN, _GUARD, exact_dot, gamma, gram_additive_term, norm2_lower, norm2_upper,  # noqa: E402
                           norm_underflow_term, residual_additive_term, residual_entry_count)

mp.mp.dps = 60
UMIN = Fraction(1, 2 ** 1075)
MINN = Fraction(1, 2 ** 1022)


def rand_float(rng, lo=-1080, hi=1000):
    e = int(rng.integers(lo, hi))
    m = 1.0 + float(rng.random())
    return float(rng.choice([-1.0, 1.0])) * math.ldexp(m, e)


def mp_sqrt_frac(q):
    return mp.sqrt(mp.mpf(q.numerator) / mp.mpf(q.denominator))


def model_ok(z_exact, z_fl, additive_allowed=True):
    """(M-U): |fl - z| <= u|z| if |z| >= 2^-1022 else <= u_min (and exact for the additions, checked by caller)."""
    d = abs(F(z_fl) - z_exact)
    if abs(z_exact) >= MINN:
        return d <= F(U) * abs(z_exact)
    return d <= UMIN if additive_allowed else d == 0


def part_model(rng, T):
    for i in range(400):
        a, b = rand_float(rng, -1080, 500), rand_float(rng, -1080, 500)
        op = i % 4
        if op == 0:
            T.add(model_ok(F(a) * F(b), a * b), True, f"(M-U) product {a} {b}")
        elif op == 1:
            T.add(model_ok(F(a) + F(b), a + b, additive_allowed=False), True, f"(M-U) sum {a} {b}")
        elif op == 2:
            if b != 0 and math.isfinite(a / b):
                T.add(model_ok(F(a) / F(b), a / b), True, f"(M-U) quotient {a} {b}")
        else:
            a = abs(a); s = math.sqrt(a)
            T.add(abs(mpf(s) - mp_sqrt_frac(F(a))) <= mpf(U) * mp_sqrt_frac(F(a)), True, f"(M-U) sqrt {a}")


def part_dot(rng, T):
    for i in range(300):
        k = int(rng.integers(1, 9)); cplx = i % 3 == 0
        lo, hi = (-600, -480) if i % 2 else (-1074, -500)
        a = np.array([rand_float(rng, lo, hi) for _ in range(k)]); b = np.array([rand_float(rng, lo, hi) for _ in range(k)])
        if cplx:
            a = a + 1j * np.array([rand_float(rng, lo, hi) for _ in range(k)]); b = b + 1j * np.array([rand_float(rng, lo, hi) for _ in range(k)])
            s = np.vdot(a.conj(), b)   # sum a_j b_j
            ex_re = sum(F(x.real) * F(y.real) - F(x.imag) * F(y.imag) for x, y in zip(a, b))
            ex_im = sum(F(x.real) * F(y.imag) + F(x.imag) * F(y.real) for x, y in zip(a, b))
            err = mp.sqrt((mpf(s.real) - mp.mpf(ex_re.numerator) / ex_re.denominator) ** 2 + (mpf(s.imag) - mp.mpf(ex_im.numerator) / ex_im.denominator) ** 2)
            absum = sum(mp.sqrt(F(x.real) ** 2 + F(x.imag) ** 2) * mp.sqrt(F(y.real) ** 2 + F(y.imag) ** 2) for x, y in zip(a, b)) if False else \
                sum(mp_sqrt_frac(F(x.real) ** 2 + F(x.imag) ** 2) * mp_sqrt_frac(F(y.real) ** 2 + F(y.imag) ** 2) for x, y in zip(a, b))
            bound = mpf(gamma(k + 2)) * absum + 6 * k * mpf(2.0 ** -1074) / 2
            T.add(err <= bound, True, f"U1 complex k={k} err={mp.nstr(err, 6)} bound={mp.nstr(bound, 6)}")
        else:
            s = float(np.dot(a, b))
            ex = sum(F(x) * F(y) for x, y in zip(a, b))
            bound = F(gamma(k)) * sum(abs(F(x) * F(y)) for x, y in zip(a, b)) + k * (1 + F(gamma(max(k - 1, 1)))) * UMIN
            T.add(abs(F(s) - ex) <= bound, True, f"U1 real k={k}")


def part_exact_dot(rng, T):
    kinds = ["tiny", "sub_huge", "cancel", "mixed", "normal"]
    for i in range(300):
        kind = kinds[i % 5]; k = int(rng.integers(1, 7)); cplx = i % 7 == 0
        if kind == "tiny":
            x = np.array([rand_float(rng, -600, -480) for _ in range(k)]); y = np.array([rand_float(rng, -600, -480) for _ in range(k)])
        elif kind == "sub_huge":
            x = np.array([rand_float(rng, -1074, -1023) for _ in range(k)]); y = np.array([rand_float(rng, 50, 1000) for _ in range(k)])
        elif kind == "cancel":
            x = np.array([rand_float(rng, -500, -400) for _ in range(k)]); y = np.array([rand_float(rng, -500, -400) for _ in range(k)])
            if k >= 2:
                x[1], y[1] = x[0], -y[0] * (1 + 2 ** -40)
        elif kind == "mixed":
            x = np.array([rand_float(rng, -1074, 0) for _ in range(k)]); y = np.array([rand_float(rng, -1074, 0) for _ in range(k)])
        else:
            x = rng.standard_normal(k); y = rng.standard_normal(k)
        if cplx:
            x = x + 1j * np.array([rand_float(rng, -1074, 0) for _ in range(k)]); y = y + 1j * np.array([rand_float(rng, -1074, 0) for _ in range(k)])
        s, err = exact_dot(x, y)
        if cplx:
            ex_re = sum(F(a.real) * F(b.real) + F(a.imag) * F(b.imag) for a, b in zip(x, y))      # x^H y
            ex_im = sum(F(a.real) * F(b.imag) - F(a.imag) * F(b.real) for a, b in zip(x, y))
            d = mp.sqrt((mpf(s.real) - mp.mpf(ex_re.numerator) / ex_re.denominator) ** 2 + (mpf(s.imag) - mp.mpf(ex_im.numerator) / ex_im.denominator) ** 2)
            T.add(d <= mpf(err), True, f"U2 complex {kind} k={k} d={mp.nstr(d, 6)} err={err}")
        else:
            ex = sum(F(a) * F(b) for a, b in zip(x, y))
            T.add(abs(F(s) - ex) <= F(err), True, f"U2 {kind} k={k} s={s} err={err} exact={float(ex)}")


def part_norm(rng, T):
    for i in range(300):
        n = int(rng.integers(1, 7)); cplx = i % 3 == 0
        kind = i % 4
        if kind == 0:
            v = np.array([math.ldexp(float(rng.integers(1, 2 ** 20)), -1074) for _ in range(n)])
        elif kind == 1:
            v = np.array([rand_float(rng, -1074, -1000) for _ in range(n)])
        elif kind == 2:
            v = np.array([rand_float(rng, -1074, -500) for _ in range(n)])
        else:
            v = np.array([rand_float(rng, -540, -530) for _ in range(n)])       # squares straddle 2^-1074
        if cplx:
            v = v + 1j * np.array([rand_float(rng, -1074, -500) for _ in range(n)])
        sq = sum(F(z.real) ** 2 + F(z.imag) ** 2 for z in v) if cplx else sum(F(z) ** 2 for z in v)
        nv = mp_sqrt_frac(sq)
        up_, lo_ = norm2_upper(v), norm2_lower(v)
        T.add(mpf(up_) >= nv and mpf(lo_) <= nv, True, f"U3 n={n} cplx={cplx} up={up_} lo={lo_} exact={mp.nstr(nv, 25)}")


def part_residual(rng, T4, T5, T6):
    for i in range(300):
        kind = i % 7; cplx = i % 2 == 1 and kind != 6
        n = int(rng.integers(1, 6))
        M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0); A = (M + M.conj().T) / 2
        x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0); mu = float(rng.standard_normal())
        if kind == 0:
            x = x * 2.0 ** -1050
        elif kind == 1:
            A = A * 2.0 ** -1060; mu *= 2.0 ** -1060
        elif kind == 2:   # A subnormal, x huge (hypot-eta family for complex data)
            A = np.round(A * 8) / 8 * 2.0 ** -1074; x = x * 2.0 ** 1000; mu = 0.0
        elif kind == 3:   # zero residual exact dyadic eigenpair at tiny scale
            d = rng.integers(-8, 9, size=n).astype(float) / 4; A = np.diag(d); x = np.zeros(n); x[0] = 2.0 ** -1070; mu = float(d[0]); cplx = False
        elif kind == 4:   # x entries straddling 2^-1022
            x = np.array([rand_float(rng, -1074, -1000) for _ in range(n)]) + (1j * np.array([rand_float(rng, -1074, -1000) for _ in range(n)]) if cplx else 0.0)
        elif kind == 5:   # w = 1 sparse, subnormal everything
            n = max(2, n // 2 * 2); A = sp.lil_matrix((n, n), dtype=complex if cplx else float)
            for k in range(0, n, 2):
                v = rand_float(rng, -1074, -1020) * (1 + (1j if cplx else 0)); A[k, k + 1] = v; A[k + 1, k] = np.conj(v)
            A = A.tocsr(); x = np.array([rand_float(rng, -1074, -1020) for _ in range(n)]) * (1 + (0.5j if cplx else 0)); mu = rand_float(rng, -1074, -1020)
        else:             # the old chain comparison (Theorem U6): real, tiny scale
            w, V = np.linalg.eigh(A.real); j = int(rng.integers(0, n)); x = V[:, j] * 10.0 ** float(rng.choice([0, -50, -120, -150, -200])); mu = float(w[j]); A = A.real
        ad = OperatorAdapter(A); X = np.asarray(x)[:, None]
        res = residuals(ad, np.array([mu]), X)
        Rc = res.residual_columns([0], X)[:, 0]; E = _rounding_bound(ad, np.array([mu]), X)[:, 0]
        Ad = A.toarray() if sp.issparse(A) else A
        cw = residual_entry_count(ad.width, ad.complex or np.iscomplexobj(x))
        ok4 = True; r_sq = Fraction(0)
        for t in range(n):
            re_ = sum(F(complex(Ad[t, j]).real) * F(complex(x[j]).real) - F(complex(Ad[t, j]).imag) * F(complex(x[j]).imag) for j in range(n)) - F(mu) * F(complex(x[t]).real)
            im_ = sum(F(complex(Ad[t, j]).real) * F(complex(x[j]).imag) + F(complex(Ad[t, j]).imag) * F(complex(x[j]).real) for j in range(n)) - F(mu) * F(complex(x[t]).imag)
            d2 = (F(complex(Rc[t]).real) - re_) ** 2 + (F(complex(Rc[t]).imag) - im_) ** 2
            r_sq += re_ ** 2 + im_ ** 2
            ok4 &= d2 <= (F(E[t]) + F(cw) * UMIN) ** 2
        T4.add(ok4, True, f"U4 kind={kind} cplx={cplx}: |Rc - r| > E + c_w u_min")
        xsq = sum(F(complex(z).real) ** 2 + F(complex(z).imag) ** 2 for z in x)
        rel = float(res.rel_bound[0])
        if xsq == 0:
            T5.add(rel == float("inf"), True, "U5 zero x")
        else:
            T5.add(mpf(rel) ** 2 * (mp.mpf(xsq.numerator) / xsq.denominator) >= mp.mpf(r_sq.numerator) / r_sq.denominator, True,
                   f"U5 kind={kind} cplx={cplx} rel={rel}")
        if kind == 6:
            R_old = float(np.linalg.norm(Rc)) * (1 + gamma(n + 2)) * _GUARD
            E_old = float(np.linalg.norm(E)) * (1 + gamma(n + 2)) * _GUARD
            xl_old = max(0.0, float(np.linalg.norm(x)) * (1 - gamma(n + 2)) / _GUARD)
            rel_old = (R_old + E_old) / xl_old * (1 + 4 * U) if xl_old > 0 else float("inf")
            xl_new = float(res.xnorm_lower[0])
            if np.isfinite(rel_old) and np.isfinite(rel) and xl_new > 0:
                Tn = norm_underflow_term(n); A_ = residual_additive_term(n, ad.width, False)
                D = 2 * _GUARD * ((2 * Tn + A_) / xl_new + (R_old + E_old) * Tn / (xl_new * xl_old)) + 4 * ULP_MIN
                ok6 = rel_old <= rel <= rel_old + D and (rel == rel_old if rel_old > 1e-145 else True)
                T6.add(ok6, True, f"U6 old={rel_old} new={rel} D={D}")
            else:
                T6.add(not np.isfinite(rel_old) or rel >= rel_old, True, f"U6 inf case old={rel_old} new={rel}")


def part_gram(rng, T):
    for i in range(100):
        n = int(rng.integers(1, 8)); g = int(rng.integers(1, 4)); cplx = i % 3 == 0
        X = np.array([[rand_float(rng, -560, -520) for _ in range(g)] for _ in range(n)])
        if cplx:
            X = X + 1j * np.array([[rand_float(rng, -560, -520) for _ in range(g)] for _ in range(n)])
        G = X.conj().T @ X
        # exact Gram
        fro2 = mp.mpf(0)
        for a in range(g):
            for b in range(g):
                re_ = sum(F(complex(X[t, a]).real) * F(complex(X[t, b]).real) + F(complex(X[t, a]).imag) * F(complex(X[t, b]).imag) for t in range(n))
                im_ = sum(F(complex(X[t, a]).real) * F(complex(X[t, b]).imag) - F(complex(X[t, a]).imag) * F(complex(X[t, b]).real) for t in range(n))
                d_re = F(complex(G[a, b]).real) - re_; d_im = F(complex(G[a, b]).imag) - im_
                # relative part allowance: gamma_dot(n) sum |x||y| (mp), additive part: gram_additive_term covers the Frobenius norm
                rel_allow = mpf(gamma(n + 4) * 2 if cplx else gamma(n)) * sum(mp_sqrt_frac(F(complex(X[t, a]).real) ** 2 + F(complex(X[t, a]).imag) ** 2) * mp_sqrt_frac(F(complex(X[t, b]).real) ** 2 + F(complex(X[t, b]).imag) ** 2) for t in range(n))
                dd = mp.sqrt(mp.mpf(d_re.numerator) / d_re.denominator ** 1 * 0 + (mp.mpf(d_re.numerator) / d_re.denominator) ** 2 + (mp.mpf(d_im.numerator) / d_im.denominator) ** 2)
                fro2 += max(mp.mpf(0), dd - rel_allow) ** 2
        T.add(mp.sqrt(fro2) <= mpf(gram_additive_term(g, n, cplx)), True, f"U7 gram n={n} g={g} cplx={cplx} additive part={mp.nstr(mp.sqrt(fro2), 6)} term={gram_additive_term(g, n, cplx)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_m = Tally("(M-U) on products, exact sums, quotients, square roots across 2^-1022"); part_model(rng, T_m)
    T_1 = Tally("U1 dot products with subnormal products (real/complex)"); part_dot(rng, T_1)
    T_2 = Tally("U2 exact_dot error bound (tiny, subnormal x huge, cancelling, mixed)"); part_exact_dot(rng, T_2)
    T_3 = Tally("U3 norm2_upper >= ||v||, norm2_lower <= ||v|| for tiny vectors"); part_norm(rng, T_3)
    T_4 = Tally("U4 |Rc_i - r_i| <= E_i + c_w u_min (subnormal x / A / A-subnormal-x-huge / zero residual / w=1)")
    T_5 = Tally("U5 rel >= ||Ax - mu x||/||x|| (exact)")
    T_6 = Tally("U6 rel_new in [rel_old, rel_old + D]; bit-identical above 1e-145")
    part_residual(rng, T_4, T_5, T_6)
    T_7 = Tally("U7 gram_additive_term >= additive part of a BLAS Gram matrix of tiny vectors"); part_gram(rng, T_7)
    sys.exit(1 if report([T_m, T_1, T_2, T_3, T_4, T_5, T_6, T_7]) else 0)


if __name__ == "__main__":
    main()
