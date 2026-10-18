"""Validate the verifier: certified statements must hold against independent references."""
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from certify_ed.enclosure import OperatorAdapter, enclose, kato_temple, residuals
from certify_ed.inertia import inertia
from certify_ed.fp import exact_dot
from certify_ed.corroborate import jacobi_eigvalsh
from certify_ed.construct import SpecOperator, build_bitwise
from certify_ed import spec as S

from conftest import needs_flint


def _rand_herm(n, ev, rng, cplx):
    Z = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    Q, _ = np.linalg.qr(Z)
    A = (Q * ev) @ Q.conj().T
    return (A + A.conj().T) / 2


def _spectrum_kind(kind, n, rng):
    if kind == "generic":
        return rng.standard_normal(n)
    if kind == "degenerate":
        return np.round(rng.standard_normal(n), 1)
    if kind == "near-degenerate":
        return np.r_[0.5 + 1e-11 * rng.standard_normal(3), rng.standard_normal(n - 3)]
    return np.linspace(-1, 1, n)


@needs_flint
@pytest.mark.parametrize("seed", range(24))
def test_enclosures_contain_arb_eigenvalues(seed):
    from certify_ed.rigorous import arb_eigenvalues, arb_residual_radii
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 20))
    kind = ["generic", "degenerate", "near-degenerate", "equispaced"][seed % 4]
    A = _rand_herm(n, _spectrum_kind(kind, n, rng), rng, cplx=seed % 2 == 1)
    w, V = la.eigh(A)
    if seed % 3 == 0:
        V = V + 1e-7 * rng.standard_normal(V.shape)
    E = enclose(OperatorAdapter(A), w, V)
    ref = arb_eigenvalues(A)
    for i in range(n):
        lo, hi = E.interval(i)
        assert any(r[0] <= hi and r[1] >= lo for r in ref)
    for g in E.groups:
        if g.distinct:
            lo = min(E.mu[i] - E.radius[i] for i in g.members)
            hi = max(E.mu[i] + E.radius[i] for i in g.members)
            assert sum(1 for r in ref if r[0] <= hi and r[1] >= lo) >= g.size
    rr = arb_residual_radii(A, w, V)
    assert np.all(E.residual.rel_bound >= rr * (1 - 1e-12))


def test_duplicate_vectors_not_certified_distinct():
    rng = np.random.default_rng(0)
    A = _rand_herm(12, np.linspace(0, 1, 12), rng, False)
    w, V = la.eigh(A)
    E = enclose(OperatorAdapter(A), np.r_[w[:3], w[0]], np.c_[V[:, :3], V[:, 0]])
    assert not E.all_distinct


@pytest.mark.parametrize("seed", range(40))
def test_inertia_sandwich(seed):
    """#(lambda < s - beta) <= nu <= #(lambda <= s + beta), dense and sparse, real and complex."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    kind = ["generic", "degenerate", "equispaced"][seed % 3]
    A = _rand_herm(n, _spectrum_kind(kind, n, rng), rng, cplx=seed % 2 == 1)
    w = np.linalg.eigvalsh(A)
    for s in rng.uniform(w.min() - 0.2, w.max() + 0.2, 4):
        for meth, M in (("dense", A), ("sparse", sp.csr_matrix(A))):
            r = inertia(M, s, method=meth)
            if r.conclusive:
                assert np.sum(w < s - r.beta) <= r.negative <= np.sum(w <= s + r.beta)


def test_inertia_detects_exact_eigenvalue_as_inconclusive_or_consistent():
    A = np.diag([0.0, 1.0, 2.0])
    r = inertia(A, 1.0, method="dense")
    assert (not r.conclusive) or (0 <= r.negative <= 2)


def test_exact_dot_accuracy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(10000) * 10.0 ** rng.integers(-8, 8, 10000)
    y = rng.standard_normal(10000)
    s, e = exact_dot(x, y)
    from fractions import Fraction
    exact = sum(Fraction(a) * Fraction(b) for a, b in zip(x.tolist(), y.tolist()))
    assert abs(Fraction(s) - exact) <= Fraction(e) + Fraction(1, 10 ** 300)


def test_kato_temple_quadratic():
    """With a loose residual, Kato-Temple is much narrower than the residual interval."""
    rng = np.random.default_rng(2)
    A = _rand_herm(40, np.linspace(0, 4, 40), rng, False)
    w, V = la.eigh(A)
    x = V[:, 0] + 1e-5 * rng.standard_normal(40)
    x /= np.linalg.norm(x)
    ad = OperatorAdapter(A)
    r = residuals(ad, np.array([x @ A @ x]), x[:, None]).rel_bound[0]
    kt = kato_temple(ad, x, -10.0, w[1] - 1e-3)
    assert kt is not None
    lo, hi, _, _ = kt
    assert lo <= w[0] <= hi
    assert hi - lo < 1e-3 * 2 * r


def test_matrix_free_residuals_are_rigorous_bounds():
    spec = S.heisenberg_chain(10)
    H = build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    ad_mf = OperatorAdapter(SpecOperator(spec))
    ad_m = OperatorAdapter(H)
    r_mf = residuals(ad_mf, w[:3], V[:, :3])
    r_m = residuals(ad_m, w[:3], V[:, :3])
    assert r_mf.rigorous and r_m.rigorous
    assert np.all(r_mf.upper >= r_m.computed)


@pytest.mark.parametrize("n,cplx", [(9, False), (30, True), (64, False)])
def test_jacobi_matches_lapack(n, cplx):
    rng = np.random.default_rng(n)
    A = _rand_herm(n, rng.standard_normal(n), rng, cplx)
    assert np.max(np.abs(jacobi_eigvalsh(A) - np.linalg.eigvalsh(A))) < 1e-11
