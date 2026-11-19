"""INT-04: the code-facing accounting of the Theorem 1 radius (docs/theory.md section 2, numbered list).

A fixed 3 x 3 real case (and a few random ones) recomputed in exact rational arithmetic (fractions.Fraction;
square roots as Decimal with directed rounding). Stage by stage, the code's quantity is at least what the
description says it bounds and at most that value times the factors and rounding envelopes listed there.
Changing a factor in enclosure.residuals or fp.norm2_upper/lower, or adding a rounding, moves one of these.
"""
from decimal import Decimal, getcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction

import numpy as np
import pytest

from certify_ed.enclosure import OperatorAdapter, residuals
from certify_ed.fp import U, _GUARD, gamma, gamma_dot, norm2_lower, norm2_upper

getcontext().prec = 120
u = Fraction(U)


def F(a):
    return Fraction(float(a))


def gam(m):                                    # exact gamma_m = m u / (1 - m u)
    return m * u / (1 - m * u)


def sqrt_up(q: Fraction) -> Fraction:
    getcontext().rounding = ROUND_CEILING
    d = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
    return Fraction(d) * (1 + Fraction(1, 10 ** 100))


def sqrt_down(q: Fraction) -> Fraction:
    getcontext().rounding = ROUND_FLOOR
    d = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
    return Fraction(d) * (1 - Fraction(1, 10 ** 100))


def norm_bounds(v):
    s = sum(F(t) ** 2 for t in v)
    return sqrt_down(s), sqrt_up(s)


def _case_fixed():
    A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    x = np.array([1.0, 1.4142135623730951, 1.0])        # near the eigenvector for 2 - sqrt(2)
    mu = np.array([2.0 - 1.4142135623730951])
    return A, x[:, None], mu


def _cases_random(seed=7, count=5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = int(rng.integers(3, 7))
        M = rng.standard_normal((n, n))
        A = (M + M.T) / 2
        w, V = np.linalg.eigh(A)
        j = int(rng.integers(0, n))
        x = V[:, j] * float(rng.uniform(0.5, 3.0)) + 1e-9 * rng.standard_normal(n)
        out.append((A, x[:, None], np.array([w[j] + 1e-7])))
    return out


@pytest.mark.parametrize("case", [_case_fixed()] + _cases_random())
def test_theorem1_radius_accounting(case):
    A, X, mu = case
    n = A.shape[0]
    w = n                                              # dense: every row has n entries
    ad = OperatorAdapter(A)
    assert ad.width == w
    res = residuals(ad, mu, X)
    x = X[:, 0]
    R_c, E_c, rel_c = res.R[:, 0], res.E[:, 0], float(res.rel_bound[0])
    Af = [[F(a) for a in row] for row in A]
    xf = [F(t) for t in x]
    muf = F(mu[0])

    # ---- the exact Theorem 1 radius for the stored numbers is dominated (soundness of the whole chain)
    r_exact = [sum(Af[i][j] * xf[j] for j in range(n)) - muf * xf[i] for i in range(n)]
    r2 = sum(t ** 2 for t in r_exact)
    x2 = sum(t ** 2 for t in xf)
    assert F(rel_c) ** 2 * x2 >= r2

    # ---- step 2: absHX_up >= |A||x| exactly; <= |A||x| (1 + gamma_w)/(1 - gamma_{w+1}) (1 + 4u) (1 + u)^5
    absAx = [sum(abs(Af[i][j]) * abs(xf[j]) for j in range(n)) for i in range(n)]
    absHX_c = ad.abs_matvec(X)[:, 0]
    gw = gamma(w + 1)
    absHX_up_c = absHX_c / (1.0 - gw) * (1 + 4 * U)                  # line 163, re-evaluated in binary64
    env2 = (1 + gam(w)) / (1 - gam(w + 1)) * (1 + 4 * u) * (1 + u) ** 5
    for i in range(n):
        assert F(absHX_up_c[i]) >= absAx[i]
        assert F(absHX_up_c[i]) <= absAx[i] * env2
        # the factor is present: at least the line's formula on its own input times (1 + 4u), less the roundings
        assert F(absHX_up_c[i]) >= F(absHX_c[i]) / (1 - gam(w + 1)) * (1 + 4 * u) * (1 - u) ** 5

    # ---- step 3: E >= g (absHX_up + |x||mu|) exactly (the slack of step 1 covers the fourth rounding and g);
    #      E <= g (absHX_up + |x||mu|) (1 + 4u) (1 + u)^6
    g = gamma_dot(w + 2, False)
    gf = gam(w + 2)
    assert F(g) <= gf * (1 + u) ** 2 and F(g) >= gf * (1 - u) ** 2
    env3 = (1 + 4 * u) * (1 + u) ** 6
    for i in range(n):
        base = gf * (F(absHX_up_c[i]) + abs(xf[i]) * abs(muf))
        assert F(E_c[i]) >= base * (1 - 2 * u - 16 * u ** 2)      # the documented shortfall before the slack
        assert F(E_c[i]) <= base * env3
        assert F(E_c[i]) >= base * (1 + 4 * u) * (1 - u) ** 6     # the factor is present
        # and E dominates the true entrywise rounding error of R (step 1), with the slack to spare
        assert F(E_c[i]) >= abs(F(R_c[i]) - r_exact[i])
        assert F(E_c[i]) >= gf * (absAx[i] + abs(xf[i]) * abs(muf))

    # ---- steps 4-5: rel >= (||R|| + ||E||)/||x|| exactly, and <= that times the guards and envelopes
    nR_lo, nR_hi = norm_bounds(R_c)
    nE_lo, nE_hi = norm_bounds(E_c)
    nx_lo, nx_hi = norm_bounds(x)
    assert F(rel_c) >= (nR_lo + nE_lo) / nx_hi
    gn = gam(n + 2)
    env_norm_up = (1 + gam(n)) ** 1 * (1 + u) * (1 + gn) * F(_GUARD) * (1 + u) ** 3   # BLAS, sqrt, guard, 3 roundings
    env_norm_lo = (1 - gam(n)) * (1 - u) * (1 - gn) / F(_GUARD) / (1 + u) ** 3
    env5 = (1 + 4 * u) * (1 + u) ** 3                                                     # sum, division, product
    assert F(rel_c) <= (nR_hi + nE_hi) * env_norm_up / (nx_lo * env_norm_lo) * env5
    # the factors are present: the computed radius is at least the formula on the code's own upper and xl
    # times (1 + 4u), less its three roundings; and upper, xl carry their guards
    assert F(rel_c) >= F(res.upper[0]) / F(res.xnorm_lower[0]) * (1 + 4 * u) * (1 - u) ** 3
    assert F(res.upper[0]) >= (nR_lo + nE_lo) * (1 + gn) * F(_GUARD) * (1 - gam(n)) * (1 - u) ** 5
    assert F(res.xnorm_lower[0]) <= nx_hi * (1 - gn) / F(_GUARD) * (1 + gam(n)) * (1 + u) ** 5
    # the exported pieces agree with the recomputation
    assert F(res.upper[0]) >= nR_lo + nE_lo
    assert F(res.xnorm_lower[0]) <= nx_hi
    # step 4 on its own input s = fl(||v||): the guards (1 +- gamma_{n+2}) and (1 + 8u) are present, three roundings
    for v in (R_c, E_c, x):
        s = F(float(np.linalg.norm(v)))
        assert s * (1 + gn) * F(_GUARD) * (1 - u) ** 3 <= F(norm2_upper(v)) <= s * (1 + gn) * F(_GUARD) * (1 + u) ** 3
        if s > 0:
            assert s * (1 - gn) / F(_GUARD) * (1 - u) ** 3 <= F(norm2_lower(v)) <= s * (1 - gn) / F(_GUARD) * (1 + u) ** 3


def test_factor_constants_are_as_documented():
    assert _GUARD == 1.0 + 8 * U
    assert gamma(3) == 3 * U / (1 - 3 * U)
    assert gamma_dot(5, False) == gamma(5) and gamma_dot(5, True) == 2 * gamma(9)
    # (1 - gamma_3)(1 + 4u) >= 1: a factor (1 + 4u) covers three roundings; (1 - gamma_4)(1 + 4u) < 1: not four
    assert (1 - gam(3)) * (1 + 4 * u) >= 1
    assert (1 - gam(4)) * (1 + 4 * u) < 1
    # gamma_{n+2} >= gamma_n / 2 + u (the norm guard undoes the BLAS error with slack)
    for n in (1, 2, 10, 1000, 10 ** 6):
        assert gam(n + 2) >= gam(n) / 2 + u


def test_zero_vector_gives_infinite_radius():
    A, X, mu = _case_fixed()
    res = residuals(OperatorAdapter(A), mu, np.zeros_like(X))
    assert res.rel_bound[0] == np.inf
