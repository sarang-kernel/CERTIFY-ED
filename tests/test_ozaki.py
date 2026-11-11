"""D-03 prototype (certify_ed/ozaki.py, UNVERIFIED): error-free splits, Gram products and residuals against exact
rational arithmetic (python-flint fmpq_mat), negative controls, and the refusal paths. docs/proofs/D-03_ozaki_draft.md."""
import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed import ozaki as oz
from certify_ed.enclosure import OperatorAdapter, enclose, residuals

from conftest import needs_flint

flint = pytest.importorskip("flint")


def q(x):
    return flint.fmpq(*float(x).as_integer_ratio())


def qmat(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return flint.fmpq_mat(A.shape[0], A.shape[1], [q(v) for v in A.ravel()])


def qentries(M):
    return [[M[i, j] for j in range(M.ncols())] for i in range(M.nrows())]


def exact_gram(X, Y):
    """(Re, Im) of X^H Y as lists of fmpq."""
    Xr, Xi = np.real(X).astype(float), np.imag(X).astype(float)
    Yr, Yi = np.real(Y).astype(float), np.imag(Y).astype(float)
    re = qmat(Xr).transpose() * qmat(Yr) + qmat(Xi).transpose() * qmat(Yi)
    im = qmat(Xr).transpose() * qmat(Yi) - qmat(Xi).transpose() * qmat(Yr)
    return qentries(re), qentries(im)


def exact_residual(H, mu, X):
    Hd = H.toarray() if sp.issparse(H) else np.asarray(H)
    Hr, Hi = np.real(Hd).astype(float), np.imag(Hd).astype(float)
    Xr, Xi = np.real(X).astype(float), np.imag(X).astype(float)
    M = qmat(np.diag(mu))
    re = qmat(Hr) * qmat(Xr) - qmat(Hi) * qmat(Xi) - qmat(Xr) * M
    im = qmat(Hr) * qmat(Xi) + qmat(Hi) * qmat(Xr) - qmat(Xi) * M
    return qentries(re), qentries(im)


def violations(V, E, ex_re, ex_im):
    """entries with |V - exact| > E (complex: squared moduli compared exactly)."""
    bad = 0
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            dr = q(np.real(V[i, j])) - ex_re[i][j]
            di = q(np.imag(V[i, j])) - ex_im[i][j]
            if dr * dr + di * di > q(E[i, j]) ** 2:
                bad += 1
    return bad


def mixed_columns(rng, n, g, complex_=False):
    X = rng.standard_normal((n, g)) * np.logspace(0, -14, n)[:, None]
    X[rng.integers(0, n, 3), 0] *= 2.0 ** -200            # tiny entries far below the slices: remainder terms
    if complex_:
        X = X + 1j * rng.standard_normal((n, g)) * np.logspace(-3, -9, n)[:, None]
    return X


@needs_flint
def test_split_is_exact_and_bounded():
    rng = np.random.default_rng(0)
    for _case in range(20):
        n, g = int(rng.integers(1, 60)), int(rng.integers(1, 5))
        A = mixed_columns(rng, n, g)
        beta = int(rng.integers(3, 27))
        S = oz.split_columns(A, beta, max_slices=int(rng.integers(1, 9)))
        total = qmat(S.remainder)
        for xp, s in zip(S.slices, S.units):
            total = total + qmat(xp)
            m = np.ldexp(xp, -s[None, :].astype(np.int32))
            assert np.all(m == np.rint(m)) and np.all(np.abs(m) <= 2.0 ** beta)
        assert total == qmat(A)


@needs_flint
@pytest.mark.parametrize("cplx", [False, True])
def test_gram_against_exact_rationals(cplx):
    rng = np.random.default_rng(1 + cplx)
    for case in range(12):
        n, g, h = int(rng.integers(2, 120)), int(rng.integers(1, 5)), int(rng.integers(1, 4))
        X = mixed_columns(rng, n, g, cplx)
        Y = None if case % 3 == 0 else mixed_columns(rng, n, h, cplx and case % 2 == 0)
        r = oz.ozaki_gram(X, Y, max_slices=8 if case % 2 else 2)
        ex_re, ex_im = exact_gram(X, X if Y is None else Y)
        assert violations(r.G, r.E, ex_re, ex_im) == 0


@needs_flint
def test_slice_products_are_exact_on_blas():
    rng = np.random.default_rng(3)
    n = 4000
    X = rng.standard_normal((n, 3))
    beta = oz.split_beta(n)
    S = oz.split_columns(X, beta)
    for xp in S.slices:
        for yq in S.slices:
            assert qmat(xp).transpose() * qmat(yq) == qmat(xp.T @ yq)


@needs_flint
def test_negative_control_beta_too_large_breaks_exactness():
    # slices four bits wider than split_beta allows: n products of 2 beta bits overflow the 53-bit significand even
    # in BLAS's partial accumulators, and the inner product rounds (the exactness check of the previous test can fail)
    n = 3000
    beta = oz.split_beta(n) + 4
    rng = np.random.default_rng(5)
    x = rng.integers(2 ** (beta - 1), 2 ** beta, (n, 1)).astype(float)
    y = rng.integers(2 ** (beta - 1), 2 ** beta, (n, 1)).astype(float)
    assert qmat(x).transpose() * qmat(y) != qmat(x.T @ y)


@needs_flint
def test_column_dots_against_exact():
    rng = np.random.default_rng(4)
    for cplx in (False, True):
        X = mixed_columns(rng, 90, 4, cplx)
        Y = mixed_columns(rng, 90, 4, cplx)
        d, e = oz.ozaki_column_dots(X, Y, max_slices=3)
        ex_re, ex_im = exact_gram(X, Y)
        bad = 0
        for c in range(4):
            dr, di = q(np.real(d[c])) - ex_re[c][c], q(np.imag(d[c])) - ex_im[c][c]
            bad += dr * dr + di * di > q(e[c]) ** 2
        assert bad == 0


def random_hermitian(rng, n, density, cplx, dyadic):
    A = sp.random(n, n, density=density, random_state=int(rng.integers(1 << 30)), format="csr")
    if dyadic:
        A.data = np.round(A.data * 8) / 8
    if cplx:
        B = sp.random(n, n, density=density, random_state=int(rng.integers(1 << 30)), format="csr")
        A = A + 1j * B
    A = (A + A.conj().T) * 0.5 + sp.diags(rng.standard_normal(n) * (0.125 if dyadic else 1.0))
    return sp.csr_matrix(A)


@needs_flint
@pytest.mark.parametrize("cplx,dyadic", [(False, True), (False, False), (True, False)])
def test_residual_against_exact_rationals(cplx, dyadic):
    rng = np.random.default_rng(10 + 2 * cplx + dyadic)
    for case in range(6):
        n = int(rng.integers(5, 70))
        H = random_hermitian(rng, n, 0.2, cplx, dyadic)
        w, V = np.linalg.eigh(H.toarray())
        k = int(rng.integers(1, 4))
        mu, X = w[:k], V[:, :k]
        if case % 2:
            X = X + 1e-9 * rng.standard_normal(X.shape)
        for max_slices in (8, 1):                       # 1: remainders of H, X and mu are bounded, not summed
            r = oz.ozaki_residual(H if case % 3 else H.toarray(), mu, X, max_slices=max_slices)
            ex_re, ex_im = exact_residual(H, mu, X)
            assert violations(r.R, r.E, ex_re, ex_im) == 0


@needs_flint
def test_negative_controls_residual():
    rng = np.random.default_rng(20)
    H = random_hermitian(rng, 40, 0.3, False, False)
    w, V = np.linalg.eigh(H.toarray())
    mu, X = w[:3], V[:, :3]
    ex_re, ex_im = exact_residual(H, mu, X)
    # (a) the plain floating-point residual is not exact: a zero error bound is violated
    Rfp = H @ X - X * mu[None, :]
    assert violations(Rfp, np.zeros(Rfp.shape), ex_re, ex_im) > 0
    # (b) the remainder bound carries the error when the split is truncated: shrinking it is detected
    r = oz.ozaki_residual(H, mu, X, max_slices=1)
    assert r.info["remainder"]
    assert violations(r.R, r.E * 1e-3, ex_re, ex_im) > 0


def test_tighter_than_default_on_random_sparse():
    rng = np.random.default_rng(30)
    for case in range(5):
        H = random_hermitian(rng, 300, 0.02, case % 2 == 1, False)
        w, V = np.linalg.eigh(H.toarray())
        old = residuals(OperatorAdapter(H), w[:4], V[:, :4])
        new = oz.ozaki_residual(H, w[:4], V[:, :4])
        assert np.all(new.rel_bound <= old.rel_bound)


def test_enclose_hook_default_unchanged_and_ozaki_report():
    rng = np.random.default_rng(31)
    H = random_hermitian(rng, 200, 0.03, False, True)
    w, V = np.linalg.eigh(H.toarray())
    mu, X = w[:5], V[:, :5]
    ad = OperatorAdapter(H)
    e0 = enclose(ad, mu, X)
    e1 = enclose(ad, mu, X, residual_report=residuals(ad, mu, X))
    assert np.array_equal(e0.radius, e1.radius)
    e2 = enclose(ad, mu, X, residual_report=oz.residual_report(H, mu, X))
    assert np.all(e2.radius <= e0.radius) and e2.all_distinct == e0.all_distinct


def test_refusals():
    with pytest.raises(oz.NotApplicable):
        oz.ozaki_gram(np.array([[2.0 ** 500], [1.0]]))
    with pytest.raises(oz.NotApplicable):
        oz.ozaki_gram(np.array([[np.nan], [1.0]]))
    with pytest.raises(oz.NotApplicable):
        oz.ozaki_residual(object(), [0.0], np.ones((2, 1)))
    assert oz.split_beta(1) == 26 and oz.split_beta(2) == 26 and oz.split_beta(3) == 25      # 3 * 2^52 > 2^53
    assert oz.split_beta(12870) == 19 and oz.split_beta(2 ** 26) == 13


@needs_flint
def test_subnormal_and_zero_columns():
    X = np.zeros((6, 3))
    X[:, 1] = [1.0, 2.0 ** -1060, 5e-324, 0.0, -3.0, 2.0 ** -500]
    X[:, 2] = [1e-310, 0.0, 0.0, 0.0, 0.0, 0.0]                      # a subnormal-only column: no slice above 2^-537
    for Y in (X, X[:, :2], np.array([[1e-300], [1e-301]])):
        r = oz.ozaki_gram(Y)                                           # the tiny parts are bounded as remainder
        ex_re, ex_im = exact_gram(Y, Y)
        assert violations(r.G, r.E, ex_re, ex_im) == 0
    S = oz.split_columns(X, 25)
    assert min(int(u.min()) for u in S.units) >= oz.MIN_UNIT and not np.any(S.slices[0][:, 2])
