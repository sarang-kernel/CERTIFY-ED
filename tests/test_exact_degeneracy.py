"""C-16: exact eigenvalue counts and multiplicities in a rational bracket (certify_ed.exact_degeneracy), as the S8
evidence field `exact_bracket`. Heisenberg multiplets (exactly degenerate), a planted near-degenerate pair that
is NOT degenerate (negative control), a contradicted count (negative control), sector blocks against the full
matrix, and integer matrices with planted multiplicities as exact ground truth."""
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.spec import ModelSpec, Term
from conftest import HAVE_FLINT

pytestmark = pytest.mark.skipif(not HAVE_FLINT, reason="python-flint not installed")


def _s8(cert):
    return [s for s in cert.statements if s["id"] == "S8.degeneracy"][0]


def _with_field(spec, h):
    """The spec plus a uniform field h sum_i S^z_i (splits every multiplet by h per unit of S^z)."""
    return ModelSpec(spec.n_sites, spec.local, list(spec.terms) + [Term("z", [(complex(h), (i,)) for i in range(spec.n_sites)])],
                     state_order=spec.state_order, name=spec.name + "+field")


def test_heisenberg_singlet_and_triplet_are_exactly_degenerate():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec,
                      ce.Options(eigenspaces="all", exact_recheck=True))
    s8 = _s8(cert)
    assert s8["status"] == "CERTIFIED", s8["summary"]
    recs = s8["evidence"]["clusters"]
    by_g = {r["g_claimed"]: r for r in recs}
    for g in (1, 3):
        xb = by_g[g]["exact_bracket"]
        assert xb["status"] == "CERTIFIED" and xb["equal"] is True and xb["count"] == g
        assert xb["multiplicities"] == {str(g): 1} and xb["no_floating_point_assumption"]
        assert "all equal" in xb["statement"] and "exact arithmetic" in xb["statement"]
        lo, hi = xb["bracket"]
        assert lo <= by_g[g]["bracket_inner"][0] and by_g[g]["bracket_inner"][1] <= hi
    assert "exactly degenerate" in s8["summary"] and any("C16" in a for a in s8["assumptions"])
    # without exact_recheck no exact bracket is attempted
    cert2 = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, ce.Options(eigenspaces="all"))
    assert all("exact_bracket" not in r for r in _s8(cert2)["evidence"]["clusters"])


def test_planted_near_degenerate_pair_is_not_degenerate():
    """A field of 2^-30 splits the triplet into three levels 2^-30 apart: S8 brackets them as one cluster of 3
    ('exactly 3 in the bracket', true), and the exact bracket says they are NOT equal."""
    spec = _with_field(ce.heisenberg_chain(8), 2.0 ** -30)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    assert 0 < w[2] - w[1] < 1e-8 and 0 < w[3] - w[2] < 1e-8
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec,
                      ce.Options(eigenspaces="all", exact_recheck=True, cluster_tol=1e-6))
    s8 = _s8(cert)
    recs = s8["evidence"]["clusters"]
    trip = [r for r in recs if r["g_claimed"] == 3][0]
    assert trip["status"] == "CERTIFIED" and trip["degeneracy"] == 3          # Prop 14: a count, true
    xb = trip["exact_bracket"]
    assert xb["status"] == "CERTIFIED" and xb["equal"] is False and xb["count"] == 3 and xb["distinct_values"] == 3
    assert xb["multiplicities"] == {"1": 3} and "NOT all equal" in xb["statement"]
    assert "1 not degenerate" in s8["summary"]


def test_contradicted_count_is_failed():
    """Negative control at the API: a bracket holding the whole triplet with g_claimed = 2."""
    from certify_ed.checker import exact_matrix
    from certify_ed.exact_degeneracy import exact_bracket
    spec = ce.heisenberg_chain(6)
    w = la.eigvalsh(ce.build_bitwise(spec).toarray())
    ex = exact_matrix(spec)
    r = exact_bracket(ex, w[1] - 1e-9, w[3] + 1e-9, 2)
    assert r["status"] == "FAILED" and r["count"] == 3 and "contradicted" in r["reason"]
    r = exact_bracket(ex, w[1] - 1e-9, w[3] + 1e-9, 3)
    assert r["status"] == "CERTIFIED" and r["equal"] is True
    # the ground state alone
    r = exact_bracket(ex, w[0] - 1e-9, w[0] + 1e-9, 1)
    assert r["status"] == "CERTIFIED" and r["equal"] is True and r["multiplicities"] == {"1": 1}


def test_integer_matrix_with_planted_multiplicities():
    """Exact ground truth: an integer matrix similar to diag(planted) through a unimodular matrix."""
    from certify_ed.checker import ExactMatrix
    from certify_ed.exact_degeneracy import exact_charpoly, roots_in_bracket
    rng = np.random.default_rng(3)
    planted = [(-2, 3), (0, 1), (1, 4), (5, 2)]              # (eigenvalue, multiplicity), total 10
    d = [v for v, m in planted for _ in range(m)]
    n = len(d)
    U = np.eye(n, dtype=object)
    for _ in range(12):                                        # random unimodular: elementary row operations
        i, j = rng.integers(0, n, 2)
        if i != j:
            U[i] = U[i] + int(rng.integers(-2, 3)) * U[j]
    Uinv = np.array(la.inv(U.astype(float)).round().astype(int), dtype=object)
    assert np.all(U.dot(Uinv) == np.eye(n, dtype=object))
    M = U.dot(np.diag(np.array(d, dtype=object))).dot(Uinv)
    rows, cols = np.nonzero(M != 0)
    ex = ExactMatrix(rows.astype(np.int64), cols.astype(np.int64), M[rows, cols], np.zeros(rows.size, dtype=object), 1,
                     np.ones(n, dtype=object))
    p, den, factor = exact_charpoly(ex)
    assert p.degree() == n and den == 1 and factor == 1
    r = roots_in_bracket(p, Fraction(-3), Fraction(2))          # -2 (x3), 0, 1 (x4)
    assert r["count"] == 8 and r["distinct"] == 3 and r["multiplicities"] == {3: 1, 1: 1, 4: 1}
    r = roots_in_bracket(p, Fraction(1), Fraction(5))            # ends ON roots: nudged outward, 1 (x4) and 5 (x2)
    assert r["nudged"] and r["count"] == 6 and r["distinct"] == 2
    r = roots_in_bracket(p, Fraction(1, 2), Fraction(3, 4))      # empty
    assert r["count"] == 0 and r["distinct"] == 0


def test_sector_blocks_add_up_to_the_k0_block_count():
    """Periodic Heisenberg N = 8, Sz = 0, momentum k = 0 (the 1D Gaussian irreps of translation x reflection x flip
    with k fixed to 0 cover the whole k = 0 subspace): the sectors' exact counts in a bracket sum to the count in
    the k = 0 block, and the lowest k = 0 level is exactly simple."""
    from certify_ed.basis import cyclic_isometry, reflection, spin_flip, translation
    from certify_ed.checker import exact_matrix
    from certify_ed.exact_degeneracy import exact_bracket
    from certify_ed.exact_sectors import FiniteGroup, check_commutes, sector_matrices
    spec = ce.heisenberg_chain(8)
    st = ce.u1_state_set(spec, 4)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    V0 = cyclic_isometry(st, T, 0)
    w0 = la.eigvalsh((V0.conj().T @ H @ V0).toarray())
    ex = exact_matrix(spec, st)
    gens = [T, reflection(spec), spin_flip(spec)]
    group = FiniteGroup(gens, st)
    assert check_commutes(ex, group)[0]
    mats, dim_sum = sector_matrices(ex, group, gens, {0: 0})
    assert dim_sum == V0.shape[1]
    # the lowest k = 0 level (the ground state): exactly one eigenvalue, in exactly one sector
    a, b = w0[0] - 1e-9, w0[0] + 1e-9
    per_sector = [exact_bracket((Are, Aim, D), a, b, 0, den=ex.den) for _qn, _m, (Are, Aim, D) in mats]
    assert all(r["status"] != "NOT_CHECKED" for r in per_sector)
    assert sum(r["count"] for r in per_sector) == 1
    # a wider bracket: the k lowest k = 0 levels up to a gap, counted sector by sector
    k = 1
    while w0[k] - w0[k - 1] < 1e-6:
        k += 1
    a, b = w0[0] - 1e-9, w0[k - 1] + 1e-9
    assert w0[k] - w0[k - 1] > 1e-6
    assert sum(exact_bracket((Are, Aim, D), a, b, 0, den=ex.den)["count"] for _qn, _m, (Are, Aim, D) in mats) == k
    # the full Sz = 0 block holds at least these k eigenvalues in the same bracket (other momenta may add more)
    assert exact_bracket(ex, a, b, k)["count"] >= k
