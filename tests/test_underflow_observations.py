"""Observations, not proofs: moment bounds on inputs whose products underflow although every scan passes.

The review of round 10 found inputs that pass every scan of the underflow check (the entries of X, H and O and the norm
powers h_1..h_K) and still make quantities of the M1 and M5 bounds subnormal or zero: a small norm of O, or a norm of
H near 1e-10 with K = 27 together with a small norm of O. The no-underflow hypothesis of C-07 Lemma 3 and C-07b Lemma 7
is assumed for such quantities, not established. These tests record what the intervals do there, against the exact
moments computed in Fractions from the binary64 inputs: (1/g) sum_i <O x_i, H^k O x_i> for M1 and
(1/g) sum_i <O x_i, T_k((H - c) s) O x_i> for M5, with the centre c and the scale s of the record.

The inputs use s = eta = 0, so the projector term of the bounds is zero and only the floating-point terms are observed;
with a tiny O the products ||O||^2 h_k (the record field M_norm_upper) and the projector term also round to zero
(open item). Three regimes are recorded, every case with a computed moment below the normal range:

- norm of O near 1e-170: the exact moments are below the smallest subnormal number and the computed ones are zero, so
  containment is trivial;
- norm of O near 5e-159 (M1) or 5e-160 (M5): the exact moments are subnormal, the computed ones are zero, and the
  intervals hold because exact_dot adds 2^-966 for every product it leaves out: with that term set to zero they fail;
- norm of H near 1e-10 with K = 27 and a norm of O near 1e-30 to 1e-100: the intervals hold with the additive term
  of norm2_upper (T_N, about 4e-162 for N = 7) or with that of exact_dot, and fail with both set to zero.

These are observations on these inputs; that the additive terms of INT-05 cover every such case is not derived
(UNVERIFIED, D-12).
"""
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed import fp
from certify_ed import moments as mo
from certify_ed import spectral_bounds as sb

TINY = 2.2250738585072014e-308          # the smallest normal binary64 number


def _case(seed, n, g, hscale, oscale):
    rng = np.random.default_rng(seed)

    def sym(scale):
        A = rng.standard_normal((n, n)) * scale
        A = A + A.T
        A[np.abs(A) < 0.3 * scale] = 0
        return sp.csr_matrix(A)

    return sym(hscale), sym(oscale), np.linalg.qr(rng.standard_normal((n, g)))[0]


def _rows(A):
    return [[F(float(v)) for v in row] for row in A.toarray()]


def _mv(M, v):
    return [sum((a * b for a, b in zip(row, v)), F(0)) for row in M]


def _start(O, X):
    Of = _rows(O)
    return [_mv(Of, [F(float(x)) for x in X[:, i]]) for i in range(X.shape[1])]


def _mean_dot(Ys, Zs):
    return sum(sum(a * b for a, b in zip(y, z)) for y, z in zip(Ys, Zs)) / len(Ys)


def exact_power_moments(H, O, X, K):
    Hf, Ys = _rows(H), _start(O, X)
    Zs, out = [list(y) for y in Ys], []
    for k in range(K + 1):
        if k:
            Zs = [_mv(Hf, z) for z in Zs]
        out.append(_mean_dot(Ys, Zs))
    return out


def exact_chebyshev_moments(H, O, X, K, c, s):
    Hf, Ys, c, s = _rows(H), _start(O, X), F(c), F(s)

    def A(v):
        return [(w - c * x) * s for w, x in zip(_mv(Hf, v), v)]

    Zm, Z, out = None, [list(y) for y in Ys], []
    for j in range(K + 1):
        if j == 1:
            Zm, Z = Z, [A(z) for z in Z]
        elif j >= 2:
            Zm, Z = Z, [[2 * a - b for a, b in zip(A(z), zm)] for z, zm in zip(Z, Zm)]
        out.append(_mean_dot(Ys, Z))
    return out


def _below_normal(value, exact):
    return (value == 0.0 and exact != 0) or 0.0 < abs(value) < TINY


def _m1(seed, hscale, oscale, K):
    H, O, X = _case(seed, 7, 2, hscale, oscale)
    rec = mo.certified_moments(H, O, X, 0.0, 0.0, K=K, sum_rules=False)
    status = {s["id"]: s["status"] for s in rec["statements"]}["M1.moments"]
    exact = exact_power_moments(H, O, X, K)
    outside = [m["k"] for m, e in zip(rec["moments"], exact)
               if not F(m["value"]) - F(m["bound"]) <= e <= F(m["value"]) + F(m["bound"])]
    below = sum(_below_normal(m["value"], e) for m, e in zip(rec["moments"], exact))
    return status, rec["underflow_check"]["findings"], outside, below


def _m5(seed, oscale, K):
    H, O, X = _case(100 + seed, 7, 2, 1.0, oscale)
    rec = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=K)
    status = {s["id"]: s["status"] for s in rec["statements"]}["M5.chebyshev_moments"]
    ch = rec["chebyshev"]
    exact = exact_chebyshev_moments(H, O, X, K, ch["center"], ch["scale"])
    outside = [j for j, (m, e) in enumerate(zip(ch["moments"], exact))
               if not F(m["interval"][0]) <= e <= F(m["interval"][1])]
    below = sum(_below_normal(m["value"], e) for m, e in zip(ch["moments"], exact))
    return status, rec["underflow_check"]["findings"], outside, below


M1_CASES = ([(s, 1.0, 0.5e-170, 3) for s in range(3)] + [(s, 1.0, 5e-159, 3) for s in range(3)]
            + [(s, 1e-10, o, 27) for o in (0.5e-30, 0.5e-50, 0.5e-100) for s in range(2)])
M5_CASES = [(s, o) for o in (0.5e-170, 5e-160) for s in range(3)]


@pytest.mark.parametrize("seed,hscale,oscale,K", M1_CASES)
def test_m1_intervals_hold_where_quantities_underflow(seed, hscale, oscale, K):
    status, findings, outside, below = _m1(seed, hscale, oscale, K)
    assert status == "CERTIFIED" and not findings                    # every scan passes
    assert not outside                                               # the exact moments are inside
    assert below                                                     # and the case is in the regime


@pytest.mark.parametrize("seed,oscale", M5_CASES)
def test_m5_intervals_hold_where_quantities_underflow(seed, oscale):
    status, findings, outside, below = _m5(seed, oscale, 9)
    assert status == "CERTIFIED" and not findings
    assert not outside and below


@pytest.mark.parametrize("seed", range(3))
def test_without_the_additive_term_of_exact_dot_the_intervals_fail(seed, monkeypatch):
    """Control: at a norm of O near 5e-159 (M1) or 5e-160 (M5) the term 2^-966 per left-out product carries the
    intervals."""
    monkeypatch.setattr(fp, "TINY_BOUND", 0.0)
    assert _m1(seed, 1.0, 5e-159, 3)[2]
    assert _m5(seed, 5e-160, 9)[2]


@pytest.mark.parametrize("oscale", [0.5e-30, 0.5e-50])
def test_without_both_additive_terms_the_small_norm_intervals_fail(oscale, monkeypatch):
    """Control: with a norm of H near 1e-10 and K = 27, T_N of norm2_upper or the term of exact_dot carries the
    intervals; with both set to zero some fail."""
    monkeypatch.setattr(fp, "TINY_BOUND", 0.0)
    monkeypatch.setattr(fp, "norm_underflow_term", lambda N: 0.0)
    assert _m1(0, 1e-10, oscale, 27)[2]


def test_the_exact_reference_sees_a_wrong_interval():
    """Negative control: an interval moved off the exact moment by twice its half-width is outside the check."""
    H, O, X = _case(0, 7, 2, 1.0, 0.5)
    rec = mo.certified_moments(H, O, X, 0.0, 0.0, K=3, sum_rules=False)
    m, e = rec["moments"][2], exact_power_moments(H, O, X, 3)[2]
    moved = F(m["value"]) + 3 * F(m["bound"])
    assert not (moved - F(m["bound"]) <= e <= moved + F(m["bound"]))
