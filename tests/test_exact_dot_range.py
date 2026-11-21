"""fp.exact_dot and fp.exact_sum at the top of the binary64 range (docs/proofs/int05_underflow.md, Lemma U2b).

Before, a pair with a factor above about 2^997 (the split 2^27 + 1 times the factor overflows) or a product within 2^-25
of the overflow threshold (the partial products overflow) gave s = err = nan or inf although the exact product is a
binary64 number, and a sum whose partial sums overflow gave nan although its exact value is in the range. Such values
are no statement (the callers give INCONCLUSIVE), so no result was wrong; the bound |s - x^H y| <= err of Lemma U2 did not
hold for them. Every value is checked against exact rational arithmetic.
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed import fp
from certify_ed.enclosure import OperatorAdapter, rayleigh_interval
from certify_ed.fp import HUGE_FACTOR, HUGE_PRODUCT, exact_dot, exact_sum

BIG = float(np.finfo(float).max)
ROOT = float(np.sqrt(BIG))


def _exact(x, y):
    return sum((Fraction(a) * Fraction(b) for a, b in zip(x, y)), Fraction(0))


def _rounds_in_range(q):
    try:
        return math.isfinite(float(q))
    except OverflowError:
        return False


PAIRS = [
    (BIG, 1.0), (1.0, BIG), (-BIG, 0.999999), (BIG * (1 - 2.0 ** -30), 1.0), (1.7e308, 1.0 - 2.0 ** -40),
    (2.0 ** 1000, 2.0 ** 23), (ROOT, ROOT), (-ROOT, ROOT), (2.0 ** 996, 3.0), (2.0 ** 995, 2.0 ** -1070),
    (2.0 ** 1023, 5e-324), (2.0 ** 1023, 2.0 ** -1060), (np.nextafter(HUGE_FACTOR, 0), 1.5), (HUGE_FACTOR, 1.5),
    (np.nextafter(HUGE_PRODUCT, 0), 1.0), (HUGE_PRODUCT, 1.0), (2.0 ** 498, 2.0 ** 498), (2.0 ** 511.5, 2.0 ** 512.4),
    (1.5 * 2.0 ** 997, 2.0 ** -10), (1.9 * 2.0 ** 996, 2.0 ** -30),       # split overflow, product below HUGE_PRODUCT
]


@pytest.mark.parametrize("a,b", PAIRS)
def test_one_pair_at_the_top_of_the_range_is_exact(a, b):
    """One pair: s is the correctly rounded exact product (both orders), err covers it; before, nan or inf."""
    for x, y in ((a, b), (b, a)):
        s, err = exact_dot(np.array([x]), np.array([y]))
        q = Fraction(x) * Fraction(y)
        assert math.isfinite(s) and math.isfinite(err)
        assert s == float(q)
        assert abs(Fraction(s) - q) <= Fraction(err)


def test_random_vectors_at_the_top_of_the_range():
    """Random real and complex vectors with entries over the whole upper exponent range, mixed with ordinary and tiny
    ones: |s - x^H y| <= err whenever the exact value rounds into the range, a value that is not finite otherwise."""
    rng = np.random.default_rng(11)
    checked = 0
    for trial in range(400):
        n = int(rng.integers(1, 12))
        ex = rng.integers(-1000, 1024, size=(2, n)) if trial % 3 == 0 else rng.integers(450, 1024, size=(2, n))
        x = rng.uniform(1, 2, n) * np.ldexp(1.0, ex[0] - 1) * rng.choice([-1, 1], n)
        y = rng.uniform(0, 1, n) * np.ldexp(1.0, np.minimum(ex[1], 1023 - np.maximum(ex[0], 0) + rng.integers(-3, 2, n)))
        if trial % 2:
            xc = x + 1j * rng.permutation(x)
            yc = y + 1j * rng.permutation(y)[::-1]
            if not (np.isfinite(xc).all() and np.isfinite(yc).all()):
                continue
            s, err = exact_dot(xc, yc)
            re = _exact(xc.real, yc.real) + _exact(xc.imag, yc.imag)
            im = _exact(xc.real, yc.imag) - _exact(xc.imag, yc.real)
            with np.errstate(over="ignore"):
                prods = np.concatenate([xc.real * yc.real, xc.imag * yc.imag, xc.real * yc.imag, xc.imag * yc.real])
            if np.isfinite(prods).all() and _rounds_in_range(re) and _rounds_in_range(im):
                assert math.isfinite(s.real) and math.isfinite(s.imag) and math.isfinite(err)
                assert abs(Fraction(s.real) - re) + abs(Fraction(s.imag) - im) <= Fraction(err)
                checked += 1
            continue
        s, err = exact_dot(x, y)
        q = _exact(x, y)
        with np.errstate(over="ignore"):
            fin = np.isfinite(x * y).all()
        if fin and _rounds_in_range(q):
            assert math.isfinite(s) and math.isfinite(err)
            assert abs(Fraction(s) - q) <= Fraction(err)
            checked += 1
        else:
            assert not (math.isfinite(s) and math.isfinite(err))
    assert checked > 200


def test_beyond_the_range_stays_not_finite():
    """An overflowing product, or an exact inner product beyond the range, gives no finite pair (as before)."""
    for x, y in (([BIG], [2.0]), ([BIG, BIG], [1.0, 1.0]), ([ROOT * 2, 1.0], [ROOT, 0.0])):
        s, err = exact_dot(np.array(x), np.array(y))
        assert not (math.isfinite(s) and math.isfinite(err))
    s, err = exact_dot(np.array([BIG + 0j, 1j * ROOT]), np.array([1.0 - 1e-3j, ROOT + 0j]))   # Im = -1.001 BIG
    assert not math.isfinite(err)


def test_complex_error_term_does_not_overflow():
    """Both parts near the top: err = 2u|re| + 2u|im| is finite (2u(|re| + |im|) overflowed)."""
    z = np.array([BIG + 0j, 1j * ROOT])
    w = np.array([1.0 - 1e-3j, -ROOT + 0j])
    s, err = exact_dot(z, w)
    re, im = Fraction(BIG), Fraction(BIG) * Fraction(-1e-3) + Fraction(ROOT) ** 2
    assert math.isfinite(err)
    assert abs(Fraction(s.real) - re) + abs(Fraction(s.imag) - im) <= Fraction(err)


@pytest.mark.parametrize("pad", [0, 5000], ids=["fsum-path", "cascade-path"])
def test_exact_sum_with_overflowing_partial_sums(pad):
    """Partial sums that overflow, exact sum in the range: the correctly rounded exact sum (before: nan), on both the
    math.fsum path and the vectorized cascade (more than EXACT_SUM_MIN_TERMS terms)."""
    half_ulp = 2.0 ** 970                                  # half the spacing of the binary64 numbers below BIG
    cases = [
        [BIG, BIG, -BIG], [-BIG, -BIG, BIG, 1.0], [BIG, BIG, -BIG, 5e-324],
        [BIG, half_ulp, -5e-324],                          # just below the midpoint above BIG: rounds to BIG
        [BIG, 2.0 ** 1023, -2.0 ** 1023, -BIG, 2.0 ** -1074],
    ]
    for c in cases:
        t = np.array(c + [1.0, -1.0] * (pad // 2))
        q = sum((Fraction(v) for v in c), Fraction(0))
        assert exact_sum(t) == float(q)
    for c in ([BIG, BIG], [BIG, half_ulp], [-BIG, -BIG, 1.0]):   # exact sums that round beyond the range
        assert math.isnan(exact_sum(np.array(c + [1.0, -1.0] * (pad // 2))))


def test_cascade_decision_at_the_largest_number():
    """The cascade's decision at h = +-BIG uses the midpoint BIG + 2^970 as the rounding boundary on the overflow side
    (np.nextafter gives an infinite gap there): sums just below the midpoint give +-BIG, sums at or above it nan."""
    pad = [1.0, -1.0] * 2500
    for sign in (1.0, -1.0):
        for tail, beyond in (([2.0 ** 970 - 2.0 ** 917], False), ([2.0 ** 969, 2.0 ** 969 - 2.0 ** 916], False),
                             ([2.0 ** 969, 2.0 ** 969], True), ([2.0 ** 970 - 2.0 ** 917, 2.0 ** 916, 2.0 ** 916], True)):
            t = np.array([sign * BIG] + [sign * v for v in tail] + pad)
            got = exact_sum(t)
            assert math.isnan(got) if beyond else got == sign * BIG


def test_rounding_interval_at_the_largest_number():
    """The decision of the cascade at h = +-BIG: a remainder that can reach the midpoint BIG + 2^970 is not inside the
    rounding interval (np.nextafter's infinite gap would have accepted it); the side towards zero is as elsewhere."""
    near = 2.0 ** 970 - 2.0 ** 917                      # the largest binary64 number below the half-gap
    for sign in (1.0, -1.0):
        h = sign * BIG
        assert fp._inside_rounding_interval(h, sign * 2.0 ** 960, 0.0)
        assert not fp._inside_rounding_interval(h, sign * near, 2.0 ** 917)
        assert not fp._inside_rounding_interval(h, sign * 2.0 ** 970, 0.0)
        assert fp._inside_rounding_interval(h, -sign * 2.0 ** 960, 0.0)
        assert fp._inside_rounding_interval(h, -sign * 2.0 ** 969, 0.0)       # towards zero the half-gap is 2^970 too
        assert not fp._inside_rounding_interval(h, -sign * 2.0 ** 970, 0.0)
        assert not fp._inside_rounding_interval(h, -sign * near, 2.0 ** 918)
    assert fp._inside_rounding_interval(1.0, 2.0 ** -60, 2.0 ** -60)
    assert not fp._inside_rounding_interval(1.0, 2.0 ** -53, 0.0)


def test_wide_sum_is_the_rounded_exact_sum():
    """The integer fallback against exact rationals on random terms over the whole exponent range, subnormals included."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(2, 40))
        t = rng.uniform(1, 2, n) * np.ldexp(1.0, rng.integers(-1074, 1024, n)) * rng.choice([-1, 1], n)
        t = t[np.isfinite(t)]
        q = sum((Fraction(v) for v in t.tolist()), Fraction(0))
        got = fp._wide_sum(t)
        if _rounds_in_range(q):
            assert got == float(q)
        else:
            assert math.isnan(got)


def test_rayleigh_interval_at_the_top_of_the_range():
    """A Rayleigh quotient near 2^1000 now has a finite enclosing interval (before: nan, nan)."""
    H = sp.diags([2.0 ** 1000, 3.0, 5.0]).tocsr()
    x = np.array([1.0, 0.0, 1e-3])
    lo, hi, mid, _ = rayleigh_interval(OperatorAdapter(H), x)
    q = (Fraction(2.0 ** 1000) + Fraction(5.0) * Fraction(1e-3) ** 2) / (1 + Fraction(1e-3) ** 2)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert Fraction(lo) <= q <= Fraction(hi)
