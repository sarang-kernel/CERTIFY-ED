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
from certify_ed.generalized import certify_generalized
from certify_ed.variational import certify_variational

BIG = float(np.finfo(float).max)
ROOT = float(np.sqrt(BIG))


def _exact(x, y):
    return sum((Fraction(a) * Fraction(b) for a, b in zip(x, y)), Fraction(0))


def _retained(x, y):
    """The exact sum of the products exact_dot keeps: those with |fl(ab)| >= TINY_PRODUCT (Lemma U2)."""
    with np.errstate(over="ignore", under="ignore"):
        keep = [abs(float(a) * float(b)) >= fp.TINY_PRODUCT for a, b in zip(x, y)]
    return sum((Fraction(a) * Fraction(b) for a, b, k in zip(x, y, keep) if k), Fraction(0))


_BEYOND = Fraction(BIG) + Fraction(2) ** 970      # an exact value of this magnitude or more rounds beyond the range


def _rounds_in_range(q):
    """q rounds to a finite binary64 number, decided in exact rationals: |q| < BIG + 2^970 (the midpoint above BIG rounds
    to the even neighbour, which is beyond the range)."""
    return abs(q) < _BEYOND


def _is_nearest(got, q):
    """got is the binary64 number nearest to q, ties to the even significand, decided in exact rationals against both
    neighbours of got (not through float(q): Fraction.__float__ rests on the correctly rounded integer division that the
    code under test also uses, hypothesis H1 of Lemma U2b)."""
    if not (isinstance(got, float) and math.isfinite(got)) or not _rounds_in_range(q):
        return False
    d = abs(Fraction(got) - q)
    with np.errstate(over="ignore"):
        nbs = (float(np.nextafter(got, math.inf)), float(np.nextafter(got, -math.inf)))
    for nb in nbs:
        if math.isfinite(nb):
            dn = abs(Fraction(nb) - q)
            if dn < d or (dn == d and int(np.array(got).view(np.int64)) & 1):
                return False
    return True


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
        assert _is_nearest(float(s), q)
        assert abs(Fraction(s) - q) <= Fraction(err)


def test_random_vectors_at_the_top_of_the_range():
    """Random real and complex vectors with entries over the whole upper exponent range, mixed with ordinary and tiny
    ones: |s - x^H y| <= err whenever every product is finite and the retained sum (the products not left out as tiny,
    Lemma U2) rounds into the range, a value that is not finite otherwise."""
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
            re_k = _retained(xc.real, yc.real) + _retained(xc.imag, yc.imag)
            im_k = _retained(xc.real, yc.imag) - _retained(xc.imag, yc.real)
            with np.errstate(over="ignore"):
                prods = np.concatenate([xc.real * yc.real, xc.imag * yc.imag, xc.real * yc.imag, xc.imag * yc.real])
            if np.isfinite(prods).all() and _rounds_in_range(re_k) and _rounds_in_range(im_k):
                assert math.isfinite(s.real) and math.isfinite(s.imag) and math.isfinite(err)
                assert abs(Fraction(s.real) - re) + abs(Fraction(s.imag) - im) <= Fraction(err)
                checked += 1
            continue
        s, err = exact_dot(x, y)
        q = _exact(x, y)
        with np.errstate(over="ignore"):
            fin = np.isfinite(x * y).all()
        if fin and _rounds_in_range(_retained(x, y)):
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
        assert _is_nearest(exact_sum(t), q)
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
            assert _is_nearest(got, q)
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


def test_statements_decided_at_the_top_of_the_range():
    """A trial vector of norm just below sqrt(BIG): x^H x lies within 2^-25 of the overflow threshold, where exact_dot
    gave inf before, so V1 and V2 of certify_variational and G2 and G3 of certify_generalized were INCONCLUSIVE
    (NONFINITE_BOUND). They are now decided, on exact inner products, and their bounds contain the exact eigenvalue
    lambda_1 = 1 of H = diag(1, 2, 3); V3 and M3 give the NOT_CHECKED of an ordinary vector."""
    H = sp.diags([1.0, 2.0, 3.0]).tocsr()
    x = np.array([ROOT * (1 - 2.0 ** -30), 0.0, 0.0])
    st = {s["id"]: s for s in certify_variational(H, x)["statements"]}
    v1, v2 = st["V1.rayleigh_upper"], st["V2.existence"]
    assert v1["status"] == "CERTIFIED" and v1["evidence"]["upper_bound"] >= 1.0, v1
    lo, hi = v2["evidence"]["interval"]
    assert v2["status"] == "CERTIFIED" and lo <= 1.0 <= hi, v2
    gs = {s["id"]: s for s in certify_generalized(H, sp.identity(3, format="csr"), np.array([1.0]), x[:, None])["statements"]}
    assert gs["G2.eigenpairs"]["status"] == "CERTIFIED", gs["G2.eigenpairs"]
    assert gs["G3.completeness"]["status"] == "CERTIFIED" and gs["G3.completeness"]["evidence"]["count_from_inertia"] == 1
    assert 1.0 < gs["G3.completeness"]["evidence"]["next_eigenvalue_lower_bound"] <= 2.0
    # V3 and M3 need a gap or a level that is not given: NOT_CHECKED as for an ordinary vector (they were INCONCLUSIVE,
    # NONFINITE_BOUND); the certificate of certify (S2 CERTIFIED) does not depend on the inner products here
    v3 = st["V3.temple_lower"]
    assert v3["status"] == "NOT_CHECKED" and v3["evidence"].get("reason_code") == "NEEDS_DEPENDENCY", v3
    from certify_ed.moments import certified_moments
    ms = {s["id"]: s for s in certified_moments(H, H, x[:, None], 1e-12, 1e-12, K=2)["statements"]}
    assert ms["M3.sum_rule_first"]["status"] == "NOT_CHECKED", ms["M3.sum_rule_first"]
    # M1 is decided on the exact inner products (INCONCLUSIVE, NONFINITE_BOUND at main; input outside its hypotheses,
    # see the changelog); M2 stays INCONCLUSIVE as at main: its second route, tr(P O^2)/g, overflows
    assert ms["M1.moments"]["status"] == "CERTIFIED", ms["M1.moments"]
    m2 = ms["M2.sum_rule_zeroth"]
    assert m2["status"] == "INCONCLUSIVE" and m2["evidence"].get("reason_code") == "NONFINITE_BOUND", m2
    import certify_ed as ce
    cert = ce.certify(ce.Claim(np.array([1.0]), x[:, None], operator=H, kind="lowest"), None, ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"


def test_complex_error_term_unchanged_where_finite():
    """The complex error term keeps the form 2u(|re| + |im|) wherever that sum is finite, so it is bit-identical with the
    code before, also where the components are about 2^-996 with bits near 2^-1070 (products just above TINY_PRODUCT
    that cancel): there 2u|re| + 2u|im| rounds differently, in about a quarter of these cases."""
    rng = np.random.default_rng(3)
    eps, b = 2.0 ** -30, np.ldexp(1.0, -483)
    differs = 0
    for _ in range(400):
        r = rng.integers(1, 2 ** 20, 8)
        xr = b * (1 + eps * np.array([r[0], r[1] + 1]))
        yr = b * (1 + eps * np.array([r[2], -(r[2] - 1)]))
        xi = b * (1 + eps * np.array([r[4], r[5] + 1]))
        yi = b * (1 + eps * np.array([r[6], -(r[6] - 1)]))
        s, err = exact_dot(xr + 1j * xi, np.array([yr[0], -yr[1]]) + 1j * np.array([yi[0], -yi[1]]))
        assert err == 2 * fp.U * (abs(s.real) + abs(s.imag))          # every product is above TINY_PRODUCT
        differs += err != 2 * fp.U * abs(s.real) + 2 * fp.U * abs(s.imag)
    assert differs > 50
    for _ in range(200):                                                 # ordinary magnitudes
        x = rng.standard_normal(5) + 1j * rng.standard_normal(5)
        y = rng.standard_normal(5) + 1j * rng.standard_normal(5)
        s, err = exact_dot(x, y)
        assert err == 2 * fp.U * (abs(s.real) + abs(s.imag))


def test_finiteness_is_decided_by_the_retained_sum():
    """The products below TINY_PRODUCT are left out (their total magnitude goes into err), so at the midpoint above BIG
    the retained sum decides whether the pair is finite: the exact inner product can round into the range while the
    retained sum does not (nan), or the reverse (a finite pair, whose bound holds). Both inputs of a review."""
    x, y = np.array([BIG, 2.0 ** 970, 2.0 ** -500]), np.array([1.0, 1.0, -2.0 ** -500])
    s, err = exact_dot(x, y)
    assert _rounds_in_range(_exact(x, y)) and not _rounds_in_range(_retained(x, y))
    assert math.isnan(s) and math.isnan(err)
    x = np.array([BIG, 2.0 ** 970, -2.0 ** -966] + [0.9 * 2.0 ** -467] * 3)
    y = np.array([1.0, 1.0, 1.0] + [2.0 ** -500] * 3)
    s, err = exact_dot(x, y)
    q = _exact(x, y)
    assert not _rounds_in_range(q) and _rounds_in_range(_retained(x, y))
    assert s == BIG and math.isfinite(err) and abs(Fraction(s) - q) <= Fraction(err)


@pytest.mark.parametrize("pad", [0, 5000], ids=["fsum-path", "cascade-path"])
def test_two_sum_intermediate_overflow_with_a_finite_sum(pad):
    """-3 * 2^970 then BIG: the TwoSum step's sum BIG - 2^971 is finite but its intermediate s + 3 * 2^970 = BIG + 2^970
    rounds to infinity, so the error term is nan; the cascade then falls back to the exact route (nearest value)."""
    c = [-3.0 * 2.0 ** 970, BIG]
    t = np.array(c + [1.0, -1.0] * (pad // 2))
    assert _is_nearest(exact_sum(t), sum((Fraction(v) for v in c), Fraction(0)))


def _wide_sum_loop(t):
    """The integer sum of the previous _wide_sum, one term at a time (float.as_integer_ratio): the reference."""
    total = 0
    for x in np.asarray(t, dtype=float).ravel().tolist():
        num, den = x.as_integer_ratio()
        total += num << (1075 - den.bit_length())
    try:
        return total / (1 << 1074)
    except OverflowError:
        return math.nan


def _same_float(a, b):
    return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1.0, a) == math.copysign(1.0, b))


def test_wide_sum_equals_the_integer_loop():
    """fp._wide_sum (frexp, int64 run sums, one division) equals the term-by-term integer sum bit for bit: random
    exponents over the whole range, subnormals of both signs (m shifted out exactly), runs of one exponent across the
    1024-term run boundary, exact cancellation, even integers, and sums at and beyond the top of the range."""
    rng = np.random.default_rng(29)
    cases = []
    for trial in range(600):
        n = int(rng.integers(1, 2500))
        kind = trial % 6
        if kind == 0:
            t = rng.standard_normal(n) * np.ldexp(1.0, rng.integers(-1074, 1000, n))
        elif kind == 1:
            t = rng.integers(-2 ** 52, 2 ** 52, n).astype(float) * 2.0 ** -1074
        elif kind == 2:
            t = rng.choice([-1.0, 1.0], n) * 2.0 ** 1023 * (1 + rng.integers(0, 2 ** 52, n) * 2.0 ** -52)
        elif kind == 3:
            h = rng.standard_normal(n // 2 + 1) * 2.0 ** float(rng.integers(-1000, 1000))
            t = np.r_[h, -h, rng.standard_normal(1) * 2.0 ** -1060]
        elif kind == 4:
            t = np.r_[BIG, BIG, -BIG, rng.standard_normal(n) * 2.0 ** 969]
        else:
            t = np.r_[rng.integers(-2 ** 60, 2 ** 60, n).astype(float), [4.0, -8.0, 2.0 ** 1000]]
        cases.append(t[np.isfinite(t)])
    for n in (1023, 1024, 1025, 2048, 2049):
        cases.append(np.full(n, 2.0 ** 1012) * (1 + 2.0 ** -52))
        cases.append(np.full(n, -(2.0 ** -1074) * (2 ** 52 - 1)))
    cases.append(np.full(3000, 2.0 ** 1023))                       # beyond the range: nan
    cases.append(np.array([BIG, -BIG]))                             # exactly zero
    for t in cases:
        got, want = fp._wide_sum(t), _wide_sum_loop(t)
        assert _same_float(got, want), (t[:4], got, want)
    assert math.isnan(fp._wide_sum(np.full(3000, 2.0 ** 1023)))


def test_zero_subnormal_and_non_finite_companions_of_top_of_range_pairs():
    """Pairs at the top of the range next to zeros (both signs), subnormal and tiny partners and ordinary terms: the
    value is the nearest binary64 number to the sum of the kept products and the bound holds (main gave nan for the
    first and third). An infinite or nan companion gives the IEEE result, as at main."""
    tiny = 2.0 ** -1074
    finite = [([BIG, 0.0, -0.0, tiny], [1.0, 5.0, 3.0, 1.0]),
              ([BIG, 1.0], [-0.0, 2.0]),
              ([BIG, 2.0 ** -1060, ROOT, -0.0], [1.0, 2.0 ** -20, -ROOT * 0.5, BIG]),
              ([0.0, BIG, tiny, -tiny], [BIG, 1.0, BIG, 2.0 ** 52])]
    for x, y in finite:
        x, y = np.array(x), np.array(y)
        s, err = exact_dot(x, y)
        q = sum((Fraction(a) * Fraction(b) for a, b in zip(x.tolist(), y.tolist())), Fraction(0))
        assert _is_nearest(s, _retained(x, y)), (x, y, s)
        assert math.isfinite(err) and abs(Fraction(s) - q) <= Fraction(err), (x, y, s, err)
    with np.errstate(all="ignore"):
        assert exact_dot(np.array([BIG, np.inf]), np.array([1.0, 1.0]))[0] == math.inf
        assert exact_dot(np.array([BIG, -np.inf]), np.array([1.0, 1.0]))[0] == -math.inf
        assert math.isnan(exact_dot(np.array([BIG, np.nan]), np.array([1.0, 1.0]))[0])
        assert math.isnan(exact_dot(np.array([BIG, 0.0]), np.array([1.0, np.inf]))[0])
