"""Property-based tests of fp.py and enclosure.py against exact rational arithmetic (roadmap H-02, 3.2.2).

Every stored binary64 number is a rational number, so each bound these modules return can be compared with the
exact quantity it bounds, computed with fractions.Fraction. hypothesis draws the inputs: magnitudes from 2^-600
to 2^200 (subnormal products and squares included, no overflow), real and complex data, dense and sparse
operators, exact and perturbed eigenpairs. Norms are compared through their squares, so no square root is
evaluated; where a modulus must be summed (inf_norm_upper of complex data) an upward-rounded rational square root
with 2300 fractional bits is used, which is finer than every square of a binary64 number.

Properties: gamma(k) = gamma_k (1 + theta_2) (Lambda3 of docs/proofs/thm1_fp-accounting.md); norm2_upper(v) >= ||v||
and norm2_lower(v) <= ||v|| for every magnitude (INT-05), with the slack (1 + 3u) and (1 - 3u) of stage (S4) when
no square leaves the normal range; |exact_dot(x, y) - x^H y| <= err; fro_upper and inf_norm_upper above the exact
norms; for the residual of a claimed pair: E_i >= |Rc_i - r_i| (stage S3), upper >= ||r||, xnorm_lower <= ||x||,
and the Theorem 1 radius rel >= ||A x - mu x|| / ||x||; the Rayleigh interval contains the exact Rayleigh quotient;
the cluster bound's eta is at least ||X^H X - I||_F. The examples are derandomized (the same on every run); set
CERTIFY_ED_HYPOTHESIS_EXAMPLES for more.
"""
import math
import os
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

import certify_ed.fp as fp  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, _cluster_bound, rayleigh_interval, residuals  # noqa: E402

N_EXAMPLES = int(os.environ.get("CERTIFY_ED_HYPOTHESIS_EXAMPLES", "120"))
SETTINGS = settings(max_examples=N_EXAMPLES, deadline=None, derandomize=True, database=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much, HealthCheck.data_too_large])
U = Fraction(1, 2 ** 53)


# ---------------------------------------------------------------- exact helpers

def F(x):
    return Fraction(float(x))


def cparts(z):
    z = complex(z)
    return Fraction(z.real), Fraction(z.imag)


def abs2(z):
    a, b = cparts(z)
    return a * a + b * b


def norm2_sq(v):
    return sum((abs2(z) for z in np.asarray(v).ravel()), Fraction(0))


def sqrt_up(q: Fraction, bits: int = 2300) -> Fraction:
    """A rational >= sqrt(q), within 2^-bits."""
    if q == 0:
        return Fraction(0)
    s = math.isqrt((q.numerator << (2 * bits)) // q.denominator)
    return Fraction(s + 1, 1 << bits)


def exact_gamma(k):
    return k * U / (1 - k * U)


# ---------------------------------------------------------------- strategies

@st.composite
def finite(draw, lo=-600, hi=200, zero=True):
    """A binary64 number m 2^e with |m| in [1, 2), e in [lo, hi], or 0 (subnormal results arise in the products)."""
    if zero and draw(st.integers(0, 9)) == 0:
        return 0.0
    m = draw(st.floats(1.0, 2.0, exclude_max=True, allow_nan=False))
    e = draw(st.integers(lo, hi))
    s = -1.0 if draw(st.booleans()) else 1.0
    return s * math.ldexp(m, e)


@st.composite
def vectors(draw, n=None, lo=-600, hi=200, complex_=None):
    n = draw(st.integers(1, 24)) if n is None else n
    c = draw(st.booleans()) if complex_ is None else complex_
    re = [draw(finite(lo, hi)) for _ in range(n)]
    if c:
        im = [draw(finite(lo, hi)) for _ in range(n)]
        return np.array(re) + 1j * np.array(im)
    return np.array(re)


@st.composite
def normal_vectors(draw):
    """Entries 0 or of magnitude in [2^-400, 2^200], at most 24 of them: every square and the sum of squares are
    normal numbers (no underflow, no overflow), the hypothesis of stage (S4)."""
    return draw(vectors(lo=-400, hi=200))


@st.composite
def hermitian(draw, n=None, lo=-300, hi=60):
    n = draw(st.integers(1, 6)) if n is None else n
    c = draw(st.booleans())
    A = np.zeros((n, n), dtype=complex if c else float)
    for i in range(n):
        A[i, i] = draw(finite(lo, hi))
        for j in range(i + 1, n):
            a = draw(finite(lo, hi))
            b = draw(finite(lo, hi)) if c else 0.0
            A[i, j] = a + 1j * b if c else a
            A[j, i] = np.conj(A[i, j])
    return A


@st.composite
def claimed_pair(draw, lo=-300):
    """(A as dense or sparse, mu, x): an eigenpair of A computed in floating point, perturbed or not, or a random
    vector with a random value. lo = -300 keeps every product in the normal range; lo = -600 lets products and
    squares underflow (the Theorem 1 chain is stated for every magnitude, INT-05)."""
    A = draw(hermitian(lo=lo))
    n = A.shape[0]
    kind = draw(st.sampled_from(["eig", "eig_perturbed", "random"]))
    if kind == "random":
        x = draw(vectors(n=n, lo=lo, hi=60, complex_=draw(st.booleans())))
        mu = draw(finite(lo, 60))
    else:
        w, V = np.linalg.eigh(A)
        j = draw(st.integers(0, n - 1))
        x, mu = V[:, j].copy(), float(w[j])
        if kind == "eig_perturbed":
            k = draw(st.integers(0, n - 1))
            x[k] = x[k] + draw(finite(-60, -10, zero=False))
            mu = mu * (1 + draw(st.floats(-1e-6, 1e-6)))
        scale = math.ldexp(1.0, draw(st.integers(-250, 40)))          # exact scaling by a power of two
        x = x * scale
    if not np.any(x != 0):
        x = np.zeros(n, dtype=x.dtype)
        x[0] = 1.0
    op = sp.csr_matrix(A) if draw(st.booleans()) else A
    return op, float(mu), np.asarray(x)


def _dense(op):
    return op.toarray() if sp.issparse(op) else np.asarray(op)


def _exact_residual(op, mu, x):
    """r = A x - mu x in exact complex rationals, as (re, im) per entry."""
    A = _dense(op)
    n = A.shape[0]
    out = []
    for i in range(n):
        re, im = Fraction(0), Fraction(0)
        for j in range(n):
            a, b = cparts(A[i, j])
            c, d = cparts(x[j])
            re += a * c - b * d
            im += a * d + b * c
        c, d = cparts(x[i])
        out.append((re - F(mu) * c, im - F(mu) * d))
    return out


# ---------------------------------------------------------------- fp.py

@SETTINGS
@given(st.integers(1, 2 ** 26))
def test_gamma_is_gamma_k_with_two_roundings(k):
    g = Fraction(fp.gamma(k))
    ex = exact_gamma(k)
    assert abs(g - ex) <= exact_gamma(2) * ex


@SETTINGS
@given(st.integers(1, 2 ** 26 - 4), st.booleans())
def test_gamma_dot_constants(k, cplx):
    gd = Fraction(fp.gamma_dot(k, cplx))
    ex = 2 * exact_gamma(k + 4) if cplx else exact_gamma(k)
    assert abs(gd - ex) <= 2 * exact_gamma(2) * ex


@SETTINGS
@given(vectors())
def test_norm_guards_bracket_the_exact_norm_for_every_magnitude(v):
    ex = norm2_sq(v)
    up_ = Fraction(fp.norm2_upper(v))
    lo_ = Fraction(fp.norm2_lower(v))
    assert up_ >= 0 and up_ * up_ >= ex
    assert 0 <= lo_ and lo_ * lo_ <= ex


@SETTINGS
@given(normal_vectors())
def test_norm_guards_keep_the_3u_slack_in_the_normal_range(v):
    """Stage (S4) of thm1_fp-accounting.md: norm2_upper >= ||v|| (1 + 3u), norm2_lower <= ||v|| (1 - 3u)."""
    ex = norm2_sq(v)
    up_ = Fraction(fp.norm2_upper(v))
    lo_ = Fraction(fp.norm2_lower(v))
    assert up_ * up_ >= ex * (1 + 3 * U) ** 2
    assert lo_ * lo_ <= ex * (1 - 3 * U) ** 2


@SETTINGS
@given(st.integers(1, 24).flatmap(lambda n: st.tuples(vectors(n=n, lo=-560, hi=200), vectors(n=n, lo=-560, hi=200))))
def test_exact_dot_error_bound(xy):
    x, y = xy
    s, err = fp.exact_dot(x, y)
    re, im = Fraction(0), Fraction(0)
    for a, b in zip(x, y):
        ar, ai = cparts(a)
        br, bi = cparts(b)
        re += ar * br + ai * bi                 # conj(a) b
        im += ar * bi - ai * br
    sr, si = cparts(s)
    E = Fraction(err)
    assert E >= 0
    assert (sr - re) ** 2 + (si - im) ** 2 <= E * E


@SETTINGS
@given(hermitian(lo=-500, hi=150), st.booleans())
def test_fro_and_inf_norm_upper_bound_the_exact_norms(A, sparse):
    M = sp.csr_matrix(A) if sparse else A
    fro = Fraction(fp.fro_upper(M))
    assert fro * fro >= norm2_sq(A)
    inf = Fraction(fp.inf_norm_upper(M))
    rows = [sum((sqrt_up(abs2(z)) for z in row), Fraction(0)) for row in _dense(M)]
    assert inf >= max(rows)                                        # each sqrt_up is >= the exact modulus


# ---------------------------------------------------------------- enclosure.py

@SETTINGS
@given(st.sampled_from([-300, -600]).flatmap(lambda lo: claimed_pair(lo=lo)))
def test_theorem1_radius_bounds_the_exact_residual_norm(pair):
    op, mu, x = pair
    adapter = OperatorAdapter(op)
    res = residuals(adapter, np.array([mu]), x[:, None])
    r = _exact_residual(op, mu, x)
    r2 = sum((a * a + b * b for a, b in r), Fraction(0))
    x2 = norm2_sq(x)
    upper, xl, rel = Fraction(res.upper[0]), Fraction(res.xnorm_lower[0]), res.rel_bound[0]
    assert upper * upper >= r2                                     # (S5) upper >= ||r||
    assert xl * xl <= x2                                           # xnorm_lower <= ||x||
    if math.isfinite(rel):
        R = Fraction(rel)
        assert R * R * x2 >= r2                                    # Theorem 1-fl: rel >= ||A x - mu x|| / ||x||
    else:
        assert xl == 0
    # (S3) the entrywise rounding bound covers the computed residual's error: a standard-model statement, checked when
    # no product a_ij x_j or mu x_i underflows (below that the INT-05 additive term in `upper` covers the shortfall,
    # and the norm-level checks above apply)
    A = _dense(op)
    prods = [abs2(A[i, j]) * abs2(x[j]) for i in range(A.shape[0]) for j in range(A.shape[0])] + [F(mu) ** 2 * abs2(z) for z in x]
    if any(0 < p < Fraction(1, 2 ** 1900) for p in prods):           # |product| < 2^-950
        return
    E = res.E[:, 0]
    Rc = res.R[:, 0]
    for i, (a, b) in enumerate(r):
        cr, ci = cparts(Rc[i])
        assert Fraction(E[i]) ** 2 >= (cr - a) ** 2 + (ci - b) ** 2


@SETTINGS
@given(claimed_pair())
def test_rayleigh_interval_contains_the_exact_rayleigh_quotient(pair):
    op, _mu, x = pair
    adapter = OperatorAdapter(op)
    ri = rayleigh_interval(adapter, x)
    if ri is None:
        return
    lo, hi, _th, _r = ri
    A = _dense(op)
    num_re = Fraction(0)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            a, b = cparts(A[i, j])
            xr, xi = cparts(x[i])
            yr, yi = cparts(x[j])
            # Re(conj(x_i) a_ij x_j)
            pr, pi = a * yr - b * yi, a * yi + b * yr
            num_re += xr * pr + xi * pi
    theta = num_re / norm2_sq(x)
    assert Fraction(lo) <= theta <= Fraction(hi)


@SETTINGS
@given(hermitian(n=None, lo=-40, hi=20), st.integers(0, 2 ** 31 - 1), st.floats(0.0, 1e-3))
def test_cluster_eta_bounds_the_exact_gram_defect(A, seed, noise):
    n = A.shape[0]
    if n < 2:
        return
    w, V = np.linalg.eigh(A)
    rng = np.random.default_rng(seed)
    X = V[:, :2] + noise * rng.standard_normal((n, 2))
    adapter = OperatorAdapter(A)
    res = residuals(adapter, w[:2], X)
    cb = _cluster_bound(adapter, w[:2], X, res, [0, 1])
    if cb is None:
        return
    eta = Fraction(cb[1])
    defect = Fraction(0)
    for i in range(2):
        for j in range(2):
            re, im = Fraction(0), Fraction(0)
            for k in range(n):
                ar, ai = cparts(X[k, i])
                br, bi = cparts(X[k, j])
                re += ar * br + ai * bi
                im += ar * bi - ai * br
            if i == j:
                re -= 1
            defect += re * re + im * im
    assert eta * eta >= defect
