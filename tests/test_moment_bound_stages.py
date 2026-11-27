"""Stage-by-stage check of the M1 and M5 bounds in exact rationals (round 9 of the review).

For two members at K = 1 the test forms the floating-point stages as certified_moments (C-07 Lemma 3) and
certified_chebyshev_moments (C-07b Lemma 7) do, requires the recorded bound to equal them bit for bit, so that a change
of an upward step is seen, and checks every upward stage and the bound against their exact counterparts in Fractions:
the power of the norm bound, the per-member terms and their upward sum, and the bound against the exact value of its
argument plus 2u|value|."""
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed import moments as mo
from certify_ed import spectral_bounds as sb
from certify_ed.fp import (U, ULP_MIN, abs_up, add_up, exact_dot, exact_sum, gamma, gamma_dot, max_nnz_per_row,
                           modulus_up, norm2_upper, up)


def _case(seed, n, g, cplx):
    rng = np.random.default_rng(seed)

    def herm(scale):
        A = rng.standard_normal((n, n)) * scale
        if cplx:
            A = A + 1j * rng.standard_normal((n, n)) * scale
        A = A + A.conj().T
        A[np.abs(A) < 0.3 * scale] = 0
        return sp.csr_matrix(A)

    X = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
    return herm(1.0), herm(0.5), np.linalg.qr(X)[0]


def _members_and_bound(Y, Z, t1, t2, g, proj, sum_bounds):
    """The last stages of both bounds: members, their sum, the summation term, the value and the bound."""
    vals, absv, members = [], [], []
    for i in range(g):
        v, e = exact_dot(Y[:, i], Z[:, i])
        vals.append(float(np.real(v)))
        absv.append(abs(float(np.real(v))))
        mem = float(add_up(add_up(t1[i], t2[i]), float(e)))
        assert F(mem) >= F(t1[i]) + F(t2[i]) + F(float(e))          # the per-member sum is upward
        members.append(mem)
    fp_terms = sum_bounds(members)
    total = exact_sum(vals)
    sum_rnd = gamma(g + 1) * exact_sum(absv) + U * abs(total)
    value = total / g
    bound = float(add_up(up((proj + fp_terms + sum_rnd) / g), 2 * U * abs(value)))
    exact_arg = (F(proj) + sum(F(m) for m in members) + F(gamma(g + 1)) * sum(F(a) for a in absv)
                 + F(U) * abs(F(total)))
    assert F(bound) >= exact_arg / g + 2 * F(U) * abs(F(value))        # every rounding of the last steps is covered
    return value, bound


@pytest.mark.parametrize("seed,cplx", [(s, c) for s in range(6) for c in (False, True)])
def test_m1_bound_stages_in_exact_rationals(seed, cplx):
    H, O, X = _case(seed, 7, 2, cplx)
    rec = mo.certified_moments(H, O, X, 0.0, 0.0, K=1, sum_rules=False)
    g = X.shape[1]
    wH, wO = max_nnz_per_row(H), max_nnz_per_row(O)
    Hn, On = rec["operator"]["H_norm_upper"], rec["operator"]["O_norm_upper"]
    gO, gH, gwO, gwH = gamma_dot(wO, cplx), gamma_dot(wH, cplx), gamma(wO), gamma(wH)
    absO, absH = abs_up(O), abs_up(H)
    Y = O @ X
    e_y = [float(up(norm2_upper(gO * (absO @ modulus_up(X[:, i])) / (1 - gwO)))) for i in range(g)]
    yn = [float(norm2_upper(Y[:, i])) for i in range(g)]
    proj_norm1 = up(2 * g * rec["state"]["sin_theta_bound"] + g * rec["state"]["eta_used"])
    Z, D, Hk = Y.copy(), [0.0] * g, 1.0
    for k in range(2):
        if k:
            eps = [float(up(norm2_upper(gH * (absH @ modulus_up(Z[:, i])) / (1 - gwH)))) for i in range(g)]
            Z = H @ Z
            D = [float(up(Hn * D[i] + eps[i])) for i in range(g)]
            Hk = float(up(Hk * Hn)) + ULP_MIN
        assert F(Hk) >= F(Hn) ** k                                      # the power of the norm bound, upward
        t1 = [float(up(Hk * (2 * yn[i] + e_y[i]) * e_y[i])) for i in range(g)]
        t2 = [float(up(yn[i] * D[i])) for i in range(g)]
        for i in range(g):
            assert F(t1[i]) >= F(Hn) ** k * (2 * F(yn[i]) + F(e_y[i])) * F(e_y[i])
            assert F(t2[i]) >= F(yn[i]) * F(D[i])
        proj = up(proj_norm1 * On * On * Hk)
        value, bound = _members_and_bound(Y, Z, t1, t2, g, proj, mo._sum_bounds)
        m = rec["moments"][k]
        assert (m["value"], m["bound"]) == (value, bound), (k, m["value"], value, m["bound"], bound)


@pytest.mark.parametrize("seed,cplx", [(s, c) for s in range(6) for c in (False, True)])
def test_m5_bound_stages_in_exact_rationals(seed, cplx):
    H, O, X = _case(100 + seed, 7, 2, cplx)
    rec = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=1)
    g = X.shape[1]
    ch = rec["chebyshev"]
    c, sc = ch["center"], ch["scale"]
    wH, wO = max_nnz_per_row(H), max_nnz_per_row(O)
    gO, gwO = gamma_dot(wO, cplx), gamma(wO)
    gA, gwA = gamma_dot(wH + 2, cplx), gamma(wH + 4)
    absO, absH = abs_up(O), abs_up(H)
    On = rec["operator"]["O_norm_upper"]
    Y = O @ X
    e_y = [float(up(norm2_upper(gO * (absO @ modulus_up(X[:, i])) / (1 - gwO)))) for i in range(g)]
    yn = [float(norm2_upper(Y[:, i])) for i in range(g)]
    proj = float(up(up(2 * g * rec["state"]["sin_theta_bound"] + g * rec["state"]["eta_used"]) * On * On))
    step1 = [float(up((2 * yn[i] + e_y[i]) * e_y[i])) for i in range(g)]
    for i in range(g):
        assert F(step1[i]) >= (2 * F(yn[i]) + F(e_y[i])) * F(e_y[i])
    Z, E = Y.copy(), [0.0] * g
    for j in range(2):
        if j:
            delta = []
            for i in range(g):
                mz = modulus_up(Z[:, i])
                delta.append(float(up(norm2_upper(gA * sc * (absH @ mz + abs(c) * mz) / (1 - gwA)))))
            Z = ((H @ Z) - c * Z) * sc
            S = [float(up(0.0 + delta[i])) for i in range(g)]
            E = [float(up(0.0 + S[i])) for i in range(g)]
        t2 = [float(up(yn[i] * E[i])) for i in range(g)]
        for i in range(g):
            assert F(t2[i]) >= F(yn[i]) * F(E[i])
        value, bound = _members_and_bound(Y, Z, step1, t2, g, proj, sb._sum_bounds)
        m = ch["moments"][j]
        assert (m["value"], m["bound"]) == (value, bound), (j, m["value"], value, m["bound"], bound)
