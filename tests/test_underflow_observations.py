"""Observations, not proofs: moment bounds on inputs whose products underflow although every scan passes.

The review of round 10 found inputs that pass every scan of the underflow check (the entries of X, H and O and the norm
powers h_1..h_K) and still make products of the M1 and M5 bounds subnormal or zero: a norm of O near 1e-170, or a norm
of H near 1e-10 with K = 27. The no-underflow hypothesis of C-07 Lemma 3 and C-07b Lemma 7 is assumed for such
products, not established. These tests record that, for such inputs, the exact moments lie inside the intervals.

The exact moments are computed in Fractions from the binary64 inputs, with s = eta = 0, so the projection term of the
bounds is zero and the bounds are the floating-point terms alone: (1/g) sum_i <O x_i, H^k O x_i> for M1 and
(1/g) sum_i <O x_i, T_k((H - c) s) O x_i> for M5, with the centre c and the scale s of the record. A passing test is an
observation on these inputs; why the intervals hold (possibly the additive term of exact_dot) is not derived
(UNVERIFIED, D-12).
"""
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

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


@pytest.mark.parametrize("seed,hscale,oscale,K", [(s, 1.0, 0.5e-170, 3) for s in range(3)]
                         + [(s, 1e-10, 0.5, 27) for s in range(3)]
                         + [(s, 1e-10, 0.5e-100, 27) for s in range(2)])
def test_m1_intervals_hold_where_products_underflow(seed, hscale, oscale, K):
    H, O, X = _case(seed, 7, 2, hscale, oscale)
    rec = mo.certified_moments(H, O, X, 0.0, 0.0, K=K, sum_rules=False)
    status = {s["id"]: s["status"] for s in rec["statements"]}["M1.moments"]
    assert status == "CERTIFIED" and not rec["underflow_check"]["findings"]      # every scan passes
    exact = exact_power_moments(H, O, X, K)
    for m, e in zip(rec["moments"], exact):
        assert F(m["value"]) - F(m["bound"]) <= e <= F(m["value"]) + F(m["bound"]), (m["k"], m["value"], m["bound"])
    if oscale < 1e-100:                         # the regime is the one of the review: values below the normal range
        assert any(_below_normal(m["value"], e) for m, e in zip(rec["moments"], exact))


@pytest.mark.parametrize("seed", range(3))
def test_m5_intervals_hold_where_products_underflow(seed):
    H, O, X = _case(100 + seed, 7, 2, 1.0, 0.5e-170)
    rec = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=9)
    status = {s["id"]: s["status"] for s in rec["statements"]}["M5.chebyshev_moments"]
    assert status == "CERTIFIED" and not rec["underflow_check"]["findings"]
    ch = rec["chebyshev"]
    exact = exact_chebyshev_moments(H, O, X, 9, ch["center"], ch["scale"])
    for m, e in zip(ch["moments"], exact):
        lo, hi = m["interval"]
        assert F(lo) <= e <= F(hi), (m["value"], m["interval"])
    assert any(_below_normal(m["value"], e) for m, e in zip(ch["moments"], exact))


def test_the_exact_reference_sees_a_wrong_interval():
    """Negative control: an interval moved off the exact moment by twice its half-width is outside the check."""
    H, O, X = _case(0, 7, 2, 1.0, 0.5)
    rec = mo.certified_moments(H, O, X, 0.0, 0.0, K=3, sum_rules=False)
    m, e = rec["moments"][2], exact_power_moments(H, O, X, 3)[2]
    moved = F(m["value"]) + 3 * F(m["bound"])
    assert not (moved - F(m["bound"]) <= e <= moved + F(m["bound"]))
