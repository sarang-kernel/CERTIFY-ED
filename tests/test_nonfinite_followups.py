"""Follow-ups to 'Non-finite bounds give no statement': the S3 reason code when only the inertia backward error
overflows, the N1, N3 and G2 records (inf and the reason codes of the non-finite branch instead of nan and
NOT_APPLICABLE), the moments record (no nan; no ValueError for complex entries whose products overflow), and
fp.exact_sum on infinite terms (IEEE semantics; math.fsum raises for inf - inf)."""
import math
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.fp import exact_dot, exact_sum
from certify_ed.generalized import certify_generalized
from certify_ed.moments import certified_moments
from certify_ed.nonhermitian import certify_nonhermitian
from certify_ed.enclosure import OperatorAdapter, kato_temple, rayleigh_interval
from certify_ed.inertia import inertia, inertia_dense
from certify_ed.variational import TrialState, certify_variational

BIG = np.nextafter(np.finfo(float).max, 0)      # within 12u of the largest float: its upper modulus is inf


def _nan_paths(o, p=""):
    if isinstance(o, float) and math.isnan(o):
        yield p
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _nan_paths(v, f"{p}/{k}")
    elif isinstance(o, (list, tuple)):
        for k, v in enumerate(o):
            yield from _nan_paths(v, f"{p}/{k}")


def _stmt(rec, prefix):
    return next(s for s in rec["statements"] if s["id"].startswith(prefix))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def test_s3_names_the_overflowed_backward_error():
    """Entries of 1e200 make the backward error of the factorization overflow (its squares exceed the largest float),
    while the claimed pairs have finite radii: S2 CERTIFIED, S3 INCONCLUSIVE with NONFINITE_BOUND (before: no code)."""
    H = sp.diags([1.0, 2.0, 3.0, 1e200]).tocsr()
    cert = ce.certify(ce.Claim(eigenvalues=np.array([1.0, 2.0, 3.0]), eigenvectors=np.eye(4)[:, :3], operator=H,
                               kind="lowest"), None, ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    s3 = cert.statement("S3.completeness")
    assert s3["status"] == "INCONCLUSIVE" and s3["evidence"]["reason_code"] == "NONFINITE_BOUND", s3
    # control: ordinary entries give a certified count and no code
    H = sp.diags([1.0, 2.0, 3.0, 10.0]).tocsr()
    cert = ce.certify(ce.Claim(eigenvalues=np.array([1.0, 2.0, 3.0]), eigenvectors=np.eye(4)[:, :3], operator=H,
                               kind="lowest"), None, ce.Options(seed=1))
    assert cert.statement("S3.completeness")["status"] == "CERTIFIED"


@pytest.mark.parametrize("vectors", [True, False], ids=["window", "vectorless-window"])
def test_s3_code_on_the_window_paths(vectors):
    """The same overflow on the window path (with and without vectors): the count at an edge is inconclusive, and
    the statement names the cause (NONFINITE_BOUND, set by the inertia result, not found in the summary text)."""
    def s3(top):
        H = sp.diags([1.0, 2.0, 3.0, top]).tocsr()
        claim = ce.Claim(eigenvalues=np.array([1.0, 2.0, 3.0]), eigenvectors=np.eye(4)[:, :3] if vectors else None,
                         operator=H, kind="window", window=(0.5, 3.5))
        return ce.certify(claim, None, ce.Options(seed=1)).statement("S3.completeness")
    s = s3(1e200)
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    if vectors:
        assert s3(10.0)["status"] == "CERTIFIED"


def _overflowing(hermitian):
    A = np.diag([1.0, 2.0, 3.0]).astype(complex)
    A[1, 2] = BIG + 1j
    A[2, 1] = BIG - 1j if hermitian else BIG + 1j
    return A


def test_n1_and_g2_follow_the_rules_of_the_non_finite_branch():
    x = np.zeros((3, 1), complex)
    x[0, 0] = 1
    n = certify_nonhermitian(_overflowing(False), np.array([1.0 + 0j]), x, kind="subset")
    s = _stmt(n, "N1")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert n["pairs"][0]["residual_upper"] == math.inf and not list(_nan_paths(n))
    g = certify_generalized(_overflowing(True), np.eye(3), np.array([1.0]), x, kind="subset")
    s = _stmt(g, "G2")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert "overflowed" in s["summary"] and "bounded away from zero" not in s["summary"]
    assert g["enclosures"][0]["radius"] == math.inf and not list(_nan_paths(g))


def test_n3_names_an_overflowed_disc_radius():
    """Eigenvalues of modulus near the largest float: the Bauer-Fike radius overflows; N1 and N3 store inf and name
    the cause (before: nan and NOT_APPLICABLE); the eigenvector conditioning (N2) is still certified."""
    A = np.diag([BIG, -BIG, 1.0])
    w, V = np.linalg.eig(A)
    n = certify_nonhermitian(A, w, V, kind="all")
    for sid in ("N1", "N3"):
        s = _stmt(n, sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert _stmt(n, "N2")["status"] == "CERTIFIED"
    assert not list(_nan_paths(n))


def test_g2_names_an_overflowed_xbx():
    """x^H B x is inf - inf for B of entries near the largest float and x = (1, 1, -1): the radius is not finite,
    and G2 names the overflow (before: ValueError from math.fsum). lambda_min(B) is declared, so G1 holds."""
    B = np.full((3, 3), BIG)
    x = np.array([[1.0], [1.0], [-1.0]])
    g = certify_generalized(np.eye(3), B, np.array([1.0]), x, kind="subset", lambda_min_B=1.0,
                            lambda_min_B_source="test")
    s = _stmt(g, "G2")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert "overflowed" in s["summary"] and not list(_nan_paths(g))


def test_variational_rayleigh_interval_of_inf_minus_inf():
    """x^H A x is inf - inf for A of entries near the largest float and x = (1, 1, -1): the Rayleigh interval has no
    finite end, V1-V3 state nothing (NONFINITE_BOUND; before: ValueError from math.fsum), and the theta record
    holds null and inf, not nan."""
    rec = certify_variational(np.full((3, 3), BIG), np.array([1.0, 1.0, -1.0]))
    for sid in ("V1", "V2", "V3"):
        s = _stmt(rec, sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert not list(_nan_paths(rec))


def test_complex_vectors_whose_squares_overflow():
    """Complex eigenvectors scaled by 1e155: the Gram entries and the squared norms overflow (before: ValueError
    from math.fsum); S2 and S3 are inconclusive with UNRESOLVED_VECTOR_NORM and no record holds nan."""
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    X = V[:, :3] * ((1 + 1j) / np.sqrt(2)) * 1e155
    cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=X, operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    for sid in ("S2.eigenpairs", "S3.completeness"):
        s = cert.statement(sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNRESOLVED_VECTOR_NORM", s
    assert not list(_nan_paths(cert.body))


def test_an_unresolvable_vector_norm_keeps_its_own_code():
    """A vector whose norm cannot be bounded away from zero in binary64 (entries of 1e-170: the squares underflow)
    names that cause, as S2 does (UNRESOLVED_VECTOR_NORM, normalise the vectors)."""
    x = np.zeros((3, 1))
    x[0, 0] = 1e-170
    n = certify_nonhermitian(np.diag([1.0, 2.0, 3.0]), np.array([1.0]), x, kind="subset", underflow_check=False)
    s = _stmt(n, "N1")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNRESOLVED_VECTOR_NORM", s
    g = certify_generalized(np.diag([1.0, 2.0, 3.0]), np.eye(3), np.array([1.0]), x, kind="subset")
    s = _stmt(g, "G2")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNRESOLVED_VECTOR_NORM", s


@pytest.mark.parametrize("top", [1e200, 1e300])
def test_moments_of_a_large_hamiltonian_raise_nothing(top):
    """||H||^k overflows (before: OverflowError from the power of a float): M1-M3 state nothing."""
    rec = certified_moments(np.diag([1.0, 2.0, top]), np.eye(3), np.eye(3)[:, :1], 1e-14, 1e-14)
    for sid in ("M1", "M2", "M3"):
        s = _stmt(rec, sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert not list(_nan_paths(rec))


@pytest.mark.parametrize("v", [1e308 + 1e308j, 1.5e308 * (1 + 1j), BIG + 1j])
@pytest.mark.parametrize("sparse", [False, True])
def test_moments_of_an_overflowing_observable_state_nothing_and_hold_no_nan(v, sparse):
    O = np.zeros((3, 3), complex)
    O[0, 1], O[1, 0], O[2, 2] = v, np.conj(v), 1.0
    rec = certified_moments(np.diag([1.0, 2.0, 3.0]), sp.csr_matrix(O) if sparse else O, np.eye(3)[:, :1], 1e-14,
                            1e-14)
    for sid in ("M1", "M2", "M3"):
        s = _stmt(rec, sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert not list(_nan_paths(rec))


def test_exact_sum_of_infinite_terms():
    assert math.isnan(exact_sum([math.inf, -math.inf, 1.0]))
    assert math.isnan(exact_sum([1.0, math.nan]))
    assert exact_sum([math.inf, 1.0] + [2.0] * 40) == math.inf
    assert exact_sum([-math.inf, -1.0]) == -math.inf
    terms = [1e16, 1.0, -1e16, 3.0] * 20
    assert exact_sum(terms) == math.fsum(terms)


def test_exact_sum_of_finite_terms_that_overflow():
    """The exact sum exceeds the largest float: nan (no value; math.fsum raises OverflowError)."""
    assert math.isnan(exact_sum([BIG, BIG]))
    assert math.isnan(exact_sum([-BIG, -BIG, 1.0]))


def test_exact_dot_keeps_inf_times_zero():
    """inf * 0 is nan, and the error-free split is not applied to a product that is not finite (before: the pair
    was dropped as tiny and the dot product came out as 1)."""
    s, e = exact_dot(np.array([math.inf, 1.0]), np.array([0.0, 1.0]))
    assert math.isnan(s) and math.isnan(e)
    s, e = exact_dot(np.array([math.inf, 1.0]), np.array([1.0, 1.0]))
    assert s == math.inf


def test_a_sum_rule_on_an_unbounded_interval_states_nothing():
    """The formed double commutator of M3 has an infinite deviation bound (entries of 1e154 in H): its interval is
    (-inf, inf), which overlaps every interval; M3 is INCONCLUSIVE with NONFINITE_BOUND (before: CERTIFIED)."""
    H = np.array([[2.5, 2.5, 1e154, 2.5], [2.5, 1e154, 2.5, 1.0], [1e154, 2.5, 1e154, 2.5], [2.5, 1.0, 2.5, 0.0]])
    O = np.array([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1e-170, 1.0], [0.0, 1e-170, 1e-170, 1e-170],
                  [1.0, 1.0, 1e-170, 1e-170]])
    x = np.array([[-0.14685366], [-0.72157702], [0.67062429], [-0.08957494]])
    x /= np.linalg.norm(x)
    rec = certified_moments(H, O, x, 1e-14, 1e-14, K=1, level=(-1.0, 1.0), underflow_check=False)
    s = _stmt(rec, "M3")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert s["evidence"]["matrix_deviation_bound"] == math.inf and "not finite" in s["summary"]
    assert _stmt(rec, "M1")["status"] == "CERTIFIED"         # the moments themselves have finite bounds
    assert not list(_nan_paths(rec))


def test_the_count_records_of_finite_data_are_unchanged():
    """The inertia records of S3 carry nonfinite_bound only when it is set: a finite count has the keys it had."""
    H = sp.diags([1.0, 2.0, 3.0, 10.0]).tocsr()
    claim = ce.Claim(eigenvalues=np.array([1.0, 2.0, 3.0]), eigenvectors=np.eye(4)[:, :3], operator=H, kind="lowest")
    counts = ce.certify(claim, None, ce.Options(seed=1)).statement("S3.completeness")["evidence"]["counts"]
    assert counts and all("nonfinite_bound" not in c for c in counts)
    H = sp.diags([1.0, 2.0, 3.0, 1e200]).tocsr()
    claim = ce.Claim(eigenvalues=np.array([1.0, 2.0, 3.0]), eigenvectors=np.eye(4)[:, :3], operator=H, kind="lowest")
    counts = ce.certify(claim, None, ce.Options(seed=1)).statement("S3.completeness")["evidence"]["counts"]
    assert any(c.get("nonfinite_bound") is True for c in counts)


def test_the_contracted_state_record_holds_no_nan():
    """A trial state with a contraction error (an MPS) whose Rayleigh interval has no finite end: the Lemma 5 term
    theta_source_vs_contracted is stored as inf (before: nan)."""
    H = np.array([[1.0, 1.797e308, 2.5], [1.797e308, -1e200, -1e154], [2.5, -1e154, 2.5]])
    x = np.array([-0.919, -0.368, -0.140])
    st = TrialState(x / np.linalg.norm(x), "mps", {}, contraction_error=1e-12)
    rec = certify_variational(H, st)
    for sid in ("V1", "V2", "V3"):
        assert _stmt(rec, sid)["status"] == "INCONCLUSIVE"
    assert rec["state"]["theta_source_vs_contracted"] == math.inf and not list(_nan_paths(rec))


def test_the_s2_record_holds_no_nan_for_an_inf_minus_inf_residual():
    """The computed residual norm is nan when H x has inf - inf; the record stores inf, as for the bound."""
    H = sp.csr_matrix(np.array([[1e200, 2.5, 0.0], [2.5, 0.0, 1.797e308], [0.0, 1.797e308, -1e200]]))
    x = np.full((3, 1), 6e199)
    cert = ce.certify(ce.Claim(eigenvalues=np.array([0.0]), eigenvectors=x, operator=H, kind="subset"), None,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "INCONCLUSIVE"
    assert not list(_nan_paths(cert.body))


def test_an_overflowed_factorization_gives_no_count():
    """A factorization with infinite entries in D (B of entries near the largest float): no exact sign, so no count
    (before: OverflowError or ValueError from the conversion to a fraction)."""
    big = 1.7976931348623155e308
    B = np.array([[1, 0, 1.7e308, 1.7e308], [0, big, big, 1], [1.7e308, big, 0, 1.7e308], [1.7e308, 1, 1.7e308, big]],
                 dtype=complex)
    r = inertia_dense(B, 0.0)
    assert not r.conclusive and r.negative is None
    x = np.eye(4)[:, :1]
    g = certify_generalized(np.eye(4), B, np.array([1.0]), x, kind="subset")
    assert _stmt(g, "G1")["status"] != "CERTIFIED" and not list(_nan_paths(g))


def test_rayleigh_interval_and_kato_temple_of_an_overflowing_vector():
    """x^H x overflows (entries of 1e155): the Rayleigh interval has no finite end, (nan, nan, nan, inf), and the
    Kato-Temple bound declines (None); a norm too small to resolve keeps None."""
    ad = OperatorAdapter(sp.diags([1.0, 2.0, 3.0]).tocsr())
    x = np.array([1e155, 1e155, 0.0])
    ri = rayleigh_interval(ad, x)
    assert ri is not None and all(math.isnan(v) for v in ri[:3]) and ri[3] == math.inf
    assert kato_temple(ad, x, 0.0, 10.0) is None
    # x^H A x = inf - inf with a finite norm of x: (nan, nan, nan, inf); without the guard (-inf, inf, 0, inf)
    ad2 = OperatorAdapter(np.array([[BIG, -BIG], [-BIG, BIG]]))
    x2 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    ri = rayleigh_interval(ad2, x2)
    assert all(math.isnan(v) for v in ri[:3]) and ri[3] == math.inf and kato_temple(ad2, x2, -1.0, 1.0) is None
    assert rayleigh_interval(ad, np.zeros(3)) is None
    lo, hi, th, r = rayleigh_interval(ad, np.array([1.0, 0.0, 0.0]))
    assert lo <= 1.0 <= hi and 0.0 <= r < 1e-14


def test_moments_whose_values_sum_past_the_largest_float():
    """Two claimed vectors with m_0 terms of about 1.7e308 each: their exact sum exceeds the largest float, exact_sum
    gives nan (OverflowError before), and M1 to M3 state nothing."""
    O = np.diag([1.3e154, 1.3e154, 1.0])
    rec = certified_moments(np.diag([1.0, 2.0, 3.0]), O, np.eye(3)[:, :2], 1e-14, 1e-14)
    for sid in ("M1", "M2", "M3"):
        s = _stmt(rec, sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert not list(_nan_paths(rec))


@pytest.mark.parametrize("sigma", [math.inf, -math.inf, math.nan])
def test_an_inertia_count_at_a_shift_that_is_not_finite(sigma):
    """No count and no factorization (LAPACK raises ValueError for inf or nan); the result names the overflow, and a
    nan shift is recorded as null."""
    r = inertia(np.diag([1.0, 2.0, 3.0]), sigma)
    assert not r.conclusive and r.negative is None and r.nonfinite_bound and r.beta == math.inf
    assert not list(_nan_paths(r.to_dict()))


def test_the_s1_invariants_record_holds_no_nan():
    """Couplings of 1e160: the squared Frobenius norms overflow on both sides and their difference is inf - inf; the
    record stores inf (before: nan)."""
    spec = ce.heisenberg_chain(4, J=1e160)
    cert = ce.certify(ce.Claim(eigenvalues=np.array([0.0]), eigenvectors=None, operator=None, kind="subset"), spec,
                      ce.Options(seed=1))
    inv = cert.statement("S1.model_fidelity")["evidence"]["spec_invariants_vs_reference"]
    assert inv["fro2_diff"] == math.inf and inv["consistent"] is None and not list(_nan_paths(cert.body))


def test_the_sum_rule_records_hold_no_nan_when_a_formed_product_overflows():
    """The moments are finite but O^2 and the double commutator overflow (O = diag(1e154, 1)): M2 and M3 state nothing,
    and their records store a non-finite interval as null and a non-finite bound as inf (before: nan)."""
    rec = certified_moments(np.diag([10.0, 2.0]), np.diag([1e154, 1.0]), np.array([[0.0], [1.0]]), 0.0, 0.0, K=1,
                            level=(1.5, 2.5))
    assert _stmt(rec, "M1")["status"] == "CERTIFIED"
    for sid in ("M2", "M3"):
        s = _stmt(rec, sid)
        assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert not list(_nan_paths(rec))


@pytest.mark.parametrize("method", ["dense", "sparse", "sparse-apriori"])
def test_a_count_whose_shifted_matrix_overflows(method):
    """A finite shift for which A - sigma I overflows (-1.5e308 - 1.6e308 is beyond the range): no count, no exception
    (before: ValueError from LAPACK on the dense path; no reason code on the sparse path), and S3 of a vectorless
    window at the edge of the range names the overflow."""
    D = np.diag([-1.5e308, 0.0, 1.5e308])
    r = inertia(sp.csr_matrix(D) if method != "dense" else D, -1.6e308, method=method)
    assert not r.conclusive and r.negative is None and r.nonfinite_bound
    claim = ce.Claim(eigenvalues=np.array([-1.5e308]), eigenvectors=None, operator=sp.csr_matrix(D), kind="window",
                     window=(-1.6e308, -1.4e308))
    s = ce.certify(claim, None, ce.Options(seed=1, inertia_method=method)).statement("S3.completeness")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s


def test_s8_names_an_overflowed_bracketing_count():
    """The bracketing counts of S8 are refused for the same overflow as S3 (entries of 1e200): S8 is INCONCLUSIVE with
    NONFINITE_BOUND (before: no code); the control with ordinary entries is CERTIFIED."""
    def s8(top):
        H = sp.diags([1.0, 1.0, 3.0, top]).tocsr()
        claim = ce.Claim(eigenvalues=np.array([1.0, 1.0]), eigenvectors=np.eye(4)[:, :2], operator=H, kind="lowest")
        return ce.certify(claim, None, ce.Options(seed=1)).statement("S8.degeneracy")
    s = s8(1e200)
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert s8(10.0)["status"] == "CERTIFIED"


def test_a_moment_interval_whose_end_overflows_states_nothing():
    """A finite value and a finite bound whose sum overflows (m_0 next to the largest float): M1 is INCONCLUSIVE with
    no interval and the finite bound kept (before: CERTIFIED on [1.7976931348623131e308, inf]); the same input with c
    scaled by 1 - 2^-40 is CERTIFIED (its interval ends below the largest float)."""
    c = 1.3407807929942596e154
    a = 0.7071067811865475
    X = np.array([[a], [a], [0.0]])

    def m1(c):
        rec = certified_moments(np.diag([1.0, 1.0, 3.0]), np.diag([c, c, 1.0]), X, 0.0, 0.0, K=0, level=(0.5, 1.5))
        return rec, _stmt(rec, "M1")
    rec, s = m1(c)
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    m = rec["moments"][0]
    assert m["interval"] == [None, None] and math.isfinite(m["bound"]) and not list(_nan_paths(rec))
    assert m1(c * (1 - 2.0 ** -40))[1]["status"] == "CERTIFIED"


def test_the_cluster_bound_record_holds_no_nan():
    """The Theorem 2' term of a cluster overflows to inf - inf (entries of 1.7e308): S2 states nothing, and the group
    record stores eps_star and XR as inf (before: nan); the radius is rho, never a nan eps_star."""
    H = np.array([[1.7e308, 1, 0, 1], [1, 1, 1.7e308 + 1e300j, 0], [0, 1.7e308 - 1e300j, 0, 2.5],
                  [1, 0, 2.5, 1.7e308]], dtype=complex)
    X = np.array([[-0.5283649431124846 - 0.0685491197023318j, -0.015247715572997738 - 0.5159036663258222j],
                  [0.1777258675720891 - 0.45480298238794126j, -0.6068573042194345 + 0.2730350214706092j],
                  [-0.24590006055627978 - 0.178670158981735j, 0.01219100913005814 + 0.34632907909114086j],
                  [-0.29472282449866805 + 0.5463038161242746j, -0.24907250521438387 + 0.3296322784952174j]])
    claim = ce.Claim(eigenvalues=np.array([-1.797e308, -1.797e308]), eigenvectors=X, operator=sp.csr_matrix(H),
                     kind="subset")
    cert = ce.certify(claim, None, ce.Options(seed=1, inertia_method="sparse-apriori"))
    assert cert.statement("S2.eigenpairs")["status"] != "CERTIFIED"
    assert not list(_nan_paths(cert.body))


@pytest.mark.parametrize("method", ["sparse", "sparse-apriori"])
def test_sparse_factors_that_overflow_name_the_overflow(method):
    """A - sigma I is finite but the stored factors overflow (pivot 1e-200 against 1e200): no count, with the flag
    that gives S3 and S8 their reason code (before: no count and no flag)."""
    r = inertia(sp.csr_matrix(np.array([[1e-200, 1e200], [1e200, 1.0]])), 0.0, method=method)
    assert not r.conclusive and r.negative is None and r.nonfinite_bound


def test_s8_names_a_non_finite_level_enclosure():
    """Couplings of 1.3e154: the level enclosure of the cluster is not finite, no count is attempted, and S8 names the
    overflow as S2 does (before: INCONCLUSIVE without a code)."""
    spec = ce.heisenberg_chain(4, J=1.3e154)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    s = cert.statement("S8.degeneracy")
    assert s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "NONFINITE_BOUND", s
    assert not list(_nan_paths(cert.body))
