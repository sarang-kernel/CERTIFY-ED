"""Independent adversarial search for Theorem 1-fl (docs/proofs/thm1_fp-accounting.md).

Ground truth for the stored binary64 numbers: exact rationals (fractions.Fraction) for a_i = (|A||x|)_i, q_i,
r = Ax - mu x and the entrywise error |Rc_i - r_i| when the data are real; mpmath (60 digits) for complex data
and for the norms. Stage checks (S2)-(S5) of enclosure.residuals / _rounding_bound and the final radius, at the
hypothesis boundaries: zero residual (exact dyadic eigenpairs), w = 1 (one nonzero per row), w = n (dense),
n = 1, scales 1e+-150, subnormal vector entries, complex data, repeated claimed values, x with a single nonzero.
A targeted family probes (S2) for complex data, where the hypot roundings of |a_ij| and |x_j| are not covered by
the guard gamma_{w+1} (see independent/thm1_fp-accounting.md, Step 2): moduli just below a rounding midpoint
(|z| = sqrt(re^2 + 2u) with re = 1 + 2u j) and products whose mantissa makes the division and the (1 + 4u)
product round down.

Usage: python docs/proofs/independent/adversarial_thm1_fp-accounting.py [--cases 400] [--seed 19]
"""
import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, mpf, report  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, _rounding_bound, residuals  # noqa: E402
from certify_ed.fp import U, gamma, norm2_lower, norm2_upper  # noqa: E402

mp.mp.dps = 60


def mp_abs(z):
    z = complex(z)
    return mp.sqrt(mp.mpf(z.real) ** 2 + mp.mpf(z.imag) ** 2)


def exact_a_q_r(A, x, mu):
    """a_i, q_i and r_i (exact) as mp numbers (exact rationals when the data are real)."""
    n = A.shape[0]
    Ad = A.toarray() if sp.issparse(A) else np.asarray(A)
    a, q, r = [], [], []
    for i in range(n):
        ai = mp.mpf(0); ri = mp.mpc(0)
        for j in range(n):
            z = complex(Ad[i, j]); xj = complex(x[j])
            if z != 0 and xj != 0:
                ai += mp_abs(z) * mp_abs(xj)
                ri += mp.mpc(mp.mpf(z.real), mp.mpf(z.imag)) * mp.mpc(mp.mpf(xj.real), mp.mpf(xj.imag))
        xi = complex(x[i])
        ri -= mp.mpf(float(mu)) * mp.mpc(mp.mpf(xi.real), mp.mpf(xi.imag))
        a.append(ai); q.append(abs(mp.mpf(float(mu))) * mp_abs(xi)); r.append(ri)
    return a, q, r


def norm_mp(v):
    return mp.sqrt(sum(mp_abs(z) ** 2 for z in v))


def make(rng, i):
    kind = i % 12
    cplx = kind in (6, 7, 11)
    def rnd(*shape):
        return rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0.0)
    if kind == 0:   # random dense
        n = int(rng.integers(1, 7)); M = rnd(n, n); A = (M + M.conj().T) / 2
        return A, rnd(n), float(rng.standard_normal()), "dense_random", False
    if kind == 1:   # exact dyadic eigenpair: zero residual
        n = int(rng.integers(2, 7)); d = rng.integers(-8, 9, size=n).astype(float) / 4
        A = np.diag(d); x = np.zeros(n); x[0] = 1.0
        return A, x, float(d[0]), "zero_residual_dyadic", True
    if kind == 2:   # w = 1 sparse (permutation-like Hermitian: pairs)
        n = int(rng.integers(2, 7)) // 2 * 2; A = sp.lil_matrix((n, n))
        for k in range(0, n, 2):
            v = float(rng.standard_normal()); A[k, k + 1] = v; A[k + 1, k] = v
        return A.tocsr(), rnd(n), float(rng.standard_normal()), "w_eq_1", True
    if kind == 3:   # n = 1
        return np.array([[float(rng.standard_normal())]]), rnd(1), float(rng.standard_normal()), "n_eq_1", True
    if kind == 4:   # scale 1e+-150
        s = 1e150 if i % 24 < 12 else 1e-150
        n = int(rng.integers(1, 6)); M = rnd(n, n); A = (M + M.conj().T) / 2 * s
        return A, rnd(n) * s, float(rng.standard_normal()) * s, f"scale({s:g})", True
    if kind == 5:   # subnormal vector entries, normal A
        n = int(rng.integers(1, 6)); M = rnd(n, n); A = (M + M.conj().T) / 2
        return A, rnd(n) * 2.0 ** -1050, float(rng.standard_normal()), "subnormal_x", True
    if kind == 6:   # complex dense
        n = int(rng.integers(1, 7)); M = rnd(n, n); A = (M + M.conj().T) / 2
        return A, rnd(n), float(rng.standard_normal()), "complex_dense", True
    if kind == 7:   # complex sparse w = 1
        n = int(rng.integers(2, 7)) // 2 * 2; A = sp.lil_matrix((n, n), dtype=complex)
        for k in range(0, n, 2):
            v = complex(rng.standard_normal(), rng.standard_normal()); A[k, k + 1] = v; A[k + 1, k] = np.conj(v)
        return A.tocsr(), rnd(n), float(rng.standard_normal()), "complex_w_eq_1", True
    if kind == 8:   # x with a single nonzero, mu = 0
        n = int(rng.integers(2, 7)); M = rnd(n, n); A = (M + M.conj().T) / 2; x = np.zeros(n); x[int(rng.integers(0, n))] = 3.0
        return A, x, 0.0, "single_nonzero_mu0", True
    if kind == 9:   # A = 0
        n = int(rng.integers(1, 6)); return np.zeros((n, n)), rnd(n), float(rng.standard_normal()), "A_zero", True
    if kind == 10:  # large mu, tiny A
        n = int(rng.integers(1, 6)); M = rnd(n, n); A = (M + M.conj().T) / 2 * 1e-100
        return A, rnd(n), 1e100 * float(rng.standard_normal()), "mu_dominates", True
    # 11: complex, targeted (S2) family: moduli just below rounding midpoints, mantissas tuned for the division/product
    n = 2; j = int(rng.integers(2 ** 20, 2 ** 26)); k = int(rng.integers(2 ** 20, 2 ** 26))
    mant = float(rng.choice([1.0, 1.249, 1.4995, 1.1175, 1.9995]))
    z = complex(mant * (1 + 2 * U * j), mant * 2.0 ** -26 * float(rng.choice([1.0, 0.9999999, 0.5])))
    xv = complex(mant * (1 + 2 * U * k), mant * 2.0 ** -26)
    A = sp.csr_matrix(np.array([[0, z], [np.conj(z), 0]])); A.eliminate_zeros()
    return A, np.array([0j, xv]), 0.0, "complex_S2_targeted", True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_s2 = Tally("(S2) absHX_up_i >= a_i (real data)")
    T_s2c = Tally("(S2) absHX_up_i >= a_i (complex data; not established by the proof route)")
    T_s3 = Tally("(S3) E_i >= |Rc_i - r_i| entrywise")
    T_s4 = Tally("(S4) norm2_upper >= ||v||(1 + 3u), norm2_lower <= ||v||(1 - 3u)")
    T_s5 = Tally("(S5)+Thm 1: rel >= (||Rc|| + ||E||)/||x|| >= ||Ax - mu x||/||x||")
    T_ovf = Tally("outside hypotheses: overflowing norms give inf/NaN (never a finite too-small bound)")
    for i in range(a.cases):
        A, x, mu, tag, adv = make(rng, i)
        ad = OperatorAdapter(A)
        n = x.size
        am, qm, rm = exact_a_q_r(A, x, mu)
        X = np.asarray(x)[:, None]
        absHX = ad.abs_matvec(X)
        if absHX.ndim == 1:
            absHX = absHX[:, None]
        gw = gamma(ad.width + 1)
        absup = absHX / (1.0 - gw) * (1 + 4 * U)
        subnormal = tag == "subnormal_x"      # outside the standard-model hypothesis of this document (INT-05 covers it)
        ok2 = all(mpf(absup[t, 0]) >= am[t] for t in range(n))
        if subnormal:
            T_s2.skip(); T_s3.skip()
        else:
            (T_s2c if np.iscomplexobj(x) or ad.complex else T_s2).add(ok2, adv, f"{tag}: absHX_up < a_i; min rel margin/u = "
            f"{mp.nstr(min((mpf(absup[t, 0]) - am[t]) / am[t] / mpf(U) for t in range(n) if am[t] > 0), 6) if any(am) else 'n/a'}")
        E = _rounding_bound(ad, np.array([mu]), X)
        res = residuals(ad, np.array([mu]), X)
        Rc = res.residual_columns([0], X)[:, 0]
        err = []
        for t in range(n):
            rc = complex(Rc[t]); d = mp.mpc(mp.mpf(rc.real), mp.mpf(rc.imag)) - rm[t]
            err.append(mp.sqrt(mp.re(d) ** 2 + mp.im(d) ** 2))
        if not subnormal:
            T_s3.add(all(mpf(E[t, 0]) >= err[t] for t in range(n)), adv, f"{tag}: E_i < |Rc_i - r_i|")
        for v in (x, Rc):
            nv = norm_mp(v)
            up_, lo_ = norm2_upper(np.asarray(v)), norm2_lower(np.asarray(v))
            if not np.isfinite(float(np.linalg.norm(np.asarray(v)))):
                # the squares overflow (||v|| > 1e154): outside the hypotheses (fl(.) not a normal number); the
                # code then returns a NaN radius, which certify() treats fail-safe (IMP-22); counted separately
                T_ovf.add(up_ == float("inf"), True, f"{tag}: overflow but norm2_upper={up_}")
                continue
            ok4 = mpf(up_) >= nv * (1 + 3 * mpf(U)) and mpf(lo_) <= nv * (1 - 3 * mpf(U))
            T_s4.add(ok4, adv, f"{tag}: norm guards up={up_} lo={lo_} exact={mp.nstr(nv, 20)}")
        exact_r = mp.sqrt(sum(mp.re(v) ** 2 + mp.im(v) ** 2 for v in rm))
        nx = norm_mp(x)
        rel = float(res.rel_bound[0])
        if nx == 0:
            T_s5.add(rel == float("inf"), adv, f"{tag}: zero x but rel finite")
        elif not np.isfinite(rel):
            T_ovf.add(np.isnan(rel) or rel == float("inf"), True, f"{tag}: non-finite rel {rel}")
        else:
            ok5 = mpf(rel) >= (norm_mp(Rc) + mp.sqrt(sum(mpf(E[t, 0]) ** 2 for t in range(n)))) / nx and mpf(rel) >= exact_r / nx
            T_s5.add(ok5, adv, f"{tag}: rel={rel} exact radius={mp.nstr(exact_r / nx, 20)}")
    sys.exit(1 if report([T_s2, T_s2c, T_s3, T_s4, T_s5, T_ovf]) else 0)


if __name__ == "__main__":
    main()
