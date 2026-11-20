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
from certify_ed.fp import exact_sum
from certify_ed.generalized import certify_generalized
from certify_ed.moments import certified_moments
from certify_ed.nonhermitian import certify_nonhermitian

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
