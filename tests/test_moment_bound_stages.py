"""Stage-by-stage check of the M1 and M5 bounds in exact rationals (round 9 of the review).

For two members at K = 3 the test forms the floating-point stages as certified_moments (C-07 Lemma 3) and
certified_chebyshev_moments (C-07b Lemma 7) do, requires the recorded bound to equal them bit for bit, so that a change
of an upward step is seen, and checks every upward stage and the bound against their exact counterparts in Fractions:
the power of the norm bound, the recursion of the error bounds, the per-member terms and their upward sum, and the bound
against the exact value of its argument plus 2u|value|."""
from fractions import Fraction as F

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed import moments as mo
from certify_ed import spectral_bounds as sb
from certify_ed.fp import (U, ULP_MIN, abs_up, add_up, exact_dot, exact_sum, gamma, gamma_dot, max_nnz_per_row,
                           modulus_up, norm2_upper, up)


def _case(seed, n, g, cplx, hscale=1.0):
    rng = np.random.default_rng(seed)

    def herm(scale):
        A = rng.standard_normal((n, n)) * scale
        if cplx:
            A = A + 1j * rng.standard_normal((n, n)) * scale
        A = A + A.conj().T
        A[np.abs(A) < 0.3 * scale] = 0
        return sp.csr_matrix(A)

    X = rng.standard_normal((n, g)) + (1j * rng.standard_normal((n, g)) if cplx else 0)
    return herm(hscale), herm(0.5), np.linalg.qr(X)[0]


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


@pytest.mark.parametrize("seed,cplx,hscale", [(s, c, 1.0) for s in range(6) for c in (False, True)]
                         + [(0, False, 1e-160), (1, True, 1e-160)])
def test_m1_bound_stages_in_exact_rationals(seed, cplx, hscale):
    """hscale 1e-160 puts the second power of the norm bound in the subnormal range and the third below it, where only
    ULP_MIN keeps it above the exact power. The underflow check then gives no statement; with the check off the stages
    are still pinned bit for bit, but the products that underflow are not compared with their exact values."""
    H, O, X = _case(seed, 7, 2, cplx, hscale)
    if hscale != 1.0:
        guarded = mo.certified_moments(H, O, X, 0.0, 0.0, K=3, sum_rules=False)
        st = {s["id"]: s for s in guarded["statements"]}
        assert st["M1.moments"]["status"] == "INCONCLUSIVE"
        assert st["M1.moments"]["evidence"]["reason_code"] == "UNDERFLOW_RISK"
        assert any("||H||^k from k = 2" in f["where"] for f in guarded["underflow_check"]["findings"])
    rec = mo.certified_moments(H, O, X, 0.0, 0.0, K=3, sum_rules=False, underflow_check=hscale == 1.0)
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
    for k in range(4):
        if k:
            eps = [float(up(norm2_upper(gH * (absH @ modulus_up(Z[:, i])) / (1 - gwH)))) for i in range(g)]
            Z = H @ Z
            Dn = [float(up(Hn * D[i] + eps[i])) for i in range(g)]
            for i in range(g if hscale == 1.0 else 0):
                assert F(Dn[i]) >= F(Hn) * F(D[i]) + F(eps[i])           # the recursion of D, upward
            D = Dn
            Hk = float(up(Hk * Hn)) + ULP_MIN
        assert F(Hk) >= F(Hn) ** k                                      # the power of the norm bound, upward
        t1 = [float(up(Hk * (2 * yn[i] + e_y[i]) * e_y[i])) for i in range(g)]
        t2 = [float(up(yn[i] * D[i])) for i in range(g)]
        for i in range(g if hscale == 1.0 else 0):
            assert F(t1[i]) >= F(Hn) ** k * (2 * F(yn[i]) + F(e_y[i])) * F(e_y[i])
            assert F(t2[i]) >= F(yn[i]) * F(D[i])
        proj = up(proj_norm1 * On * On * Hk)
        value, bound = _members_and_bound(Y, Z, t1, t2, g, proj, mo._sum_bounds)
        m = rec["moments"][k]
        assert (m["value"], m["bound"]) == (value, bound), (k, m["value"], value, m["bound"], bound)
        assert m["M_norm_upper"] == float(up(On * On * Hk)), (k, m["M_norm_upper"], Hk)    # pins the power itself


@pytest.mark.parametrize("seed,cplx", [(s, c) for s in range(6) for c in (False, True)])
def test_m5_bound_stages_in_exact_rationals(seed, cplx):
    H, O, X = _case(100 + seed, 7, 2, cplx)
    rec = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=3)
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
    def matvec_error(Z):
        out = []
        for i in range(g):
            mz = modulus_up(Z[:, i])
            out.append(float(up(norm2_upper(gA * sc * (absH @ mz + abs(c) * mz) / (1 - gwA)))))
        return out

    Zm, Z, S, E = None, Y.copy(), [0.0] * g, [0.0] * g
    for j in range(4):
        if j == 1:
            delta = matvec_error(Z)
            Zm, Z = Z, ((H @ Z) - c * Z) * sc
        elif j >= 2:
            W = ((H @ Z) - c * Z) * sc
            eps = matvec_error(Z)
            Zn = 2.0 * W - Zm
            zn = [float(norm2_upper(Zn[:, i])) for i in range(g)]
            delta = [float(up(2.0 * eps[i] + (U / (1 - U)) * zn[i])) for i in range(g)]
            for i in range(g):                                            # the local error, upward
                assert F(delta[i]) >= 2 * F(eps[i]) + F(U) / (1 - F(U)) * F(zn[i])
            Zm, Z = Z, Zn
        if j >= 1:
            Sn = [float(up(S[i] + delta[i])) for i in range(g)]
            En = [float(up(E[i] + Sn[i])) for i in range(g)]
            for i in range(g):                                            # the two accumulations, upward
                assert F(Sn[i]) >= F(S[i]) + F(delta[i]) and F(En[i]) >= F(E[i]) + F(Sn[i])
            S, E = Sn, En
        t2 = [float(up(yn[i] * E[i])) for i in range(g)]
        for i in range(g):
            assert F(t2[i]) >= F(yn[i]) * F(E[i])
        value, bound = _members_and_bound(Y, Z, step1, t2, g, proj, sb._sum_bounds)
        m = ch["moments"][j]
        assert (m["value"], m["bound"]) == (value, bound), (j, m["value"], value, m["bound"], bound)


def test_plain_additions_in_place_of_the_upward_ones_are_seen(monkeypatch):
    """Negative control: with np.add in place of add_up in both modules the bit-for-bit comparison fails."""
    monkeypatch.setattr(mo, "add_up", np.add)
    monkeypatch.setattr(sb, "add_up", np.add)
    with pytest.raises(AssertionError):
        test_m1_bound_stages_in_exact_rationals(0, False, 1.0)
    with pytest.raises(AssertionError):
        test_m5_bound_stages_in_exact_rationals(0, False)
