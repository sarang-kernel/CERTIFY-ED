"""Non-finite upper moduli (docs/proofs/fp_modulus-up.md, section 1, Risks and limits; UNVERIFIED, D-12).

fp.modulus_up returns inf for a finite z when |z|(1 + 12u) exceeds the largest float (Lemma M (iii)); for z = x + i
with x one step below the largest float it is inf where np.abs is finite. In a dense product the inf meets zero
entries and gives nan. Each entry point below must then state nothing: no CERTIFIED statement and no finite bound
(the rules for non-finite bounds, tests/test_nonfinite_bounds.py, here reached through the upper moduli). The same
inputs with a large finite entry (1e300 + i) are certified, so the tests see the change of status.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.davis_kahan import commutator_bound
from certify_ed.eigenspace import EigenspaceCertificate, certify_observable
from certify_ed.fp import modulus_up
from certify_ed.generalized import certify_generalized
from certify_ed.moments import certified_moments
from certify_ed.nonhermitian import certify_nonhermitian
from certify_ed.sweep import matrix_affinity_defect
from certify_ed.variational import certify_variational

Z_INF = complex(np.nextafter(np.finfo(float).max, 0), 1.0)
Z_FIN = complex(1e300, 1.0)
E0 = np.eye(3, dtype=complex)[:, :1]


def _hermitian(z):
    H = np.diag([1.0, 2.0, 3.0]).astype(complex)
    H[1, 2], H[2, 1] = z, np.conj(z)
    return H


def _status(statements, sid):
    return next(s["status"] for s in statements if s["id"] == sid)


def test_the_upper_modulus_is_inf_where_numpy_is_finite():
    assert np.isfinite(np.abs(Z_INF)) and modulus_up(Z_INF) == np.inf and np.isfinite(modulus_up(Z_FIN))
    with np.errstate(all="raise"):                      # no floating-point error is raised inside modulus_up
        assert modulus_up(np.array([Z_INF, 0j, complex(np.inf, np.nan)]))[0] == np.inf
    with np.errstate(invalid="ignore"):
        assert np.isnan(modulus_up(_hermitian(Z_INF)) @ modulus_up(E0)).any()   # inf * 0 = nan


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("z,certified", [(Z_INF, False), (Z_FIN, True)])
def test_entry_points_state_nothing_on_a_non_finite_bound(z, certified):
    claim = ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=E0, operator=sp.csr_matrix(_hermitian(z)), kind="subset")
    cert = ce.certify(claim, None, ce.Options(seed=1))
    assert (_status(cert.statements, "S2.eigenpairs") == "CERTIFIED") == certified
    rec = certify_variational(sp.csr_matrix(_hermitian(z)), E0[:, 0])
    for sid in ("V1.rayleigh_upper", "V2.existence"):
        assert (_status(rec["statements"], sid) == "CERTIFIED") == certified
    if not certified:
        assert _status(rec["statements"], "V3.temple_lower") == "INCONCLUSIVE"
    A = np.diag([1.0, 2.0, 3.0]).astype(complex)
    A[1, 2] = z
    rec = certify_nonhermitian(A, np.array([1.0 + 0j]), E0)
    assert (_status(rec["statements"], "N1.backward_errors") == "CERTIFIED") == certified
    rec = certify_generalized(_hermitian(z), np.eye(3), np.array([1.0]), E0, kind="lowest")
    assert (_status(rec["statements"], "G2.eigenpairs") == "CERTIFIED") == certified
    ec = EigenspaceCertificate([0], [1.0], 1.0, 0.0, 0.0, 0.0, 0.0)
    r = certify_observable(ec, E0, sp.csr_matrix(_hermitian(z)), "O")
    assert np.isfinite(r["bound"]) == certified and not np.isnan(r["bound"])
    c = commutator_bound(sp.csr_matrix(_hermitian(z)), sp.identity(3, format="csr"))["bound"]
    d = matrix_affinity_defect(sp.csr_matrix(_hermitian(z)), sp.csr_matrix(_hermitian(z)), 0.5,
                               sp.identity(3, format="csr", dtype=complex))
    assert np.isfinite(c) == certified and np.isfinite(d) == certified
    rec = certified_moments(sp.csr_matrix(np.diag([1.0, 2.0, 3.0])), sp.csr_matrix(_hermitian(z)), E0, 0.0, 0.0, K=1,
                            level=(0.5, 1.5))
    st = {s["id"]: s["status"] for s in rec["statements"]}
    if certified:
        assert st["M1.moments"] == "CERTIFIED"
    else:
        assert st["M1.moments"] == st["M2.sum_rule_zeroth"] == st["M3.sum_rule_first"] == "INCONCLUSIVE"
        assert all(s["evidence"]["reason_code"] == "NONFINITE_BOUND" for s in rec["statements"]
                   if s["id"] in ("M1.moments", "M2.sum_rule_zeroth", "M3.sum_rule_first"))
        rec = certify_variational(sp.csr_matrix(_hermitian(z)), E0[:, 0])
        assert all(s["evidence"]["reason_code"] == "NONFINITE_BOUND" for s in rec["statements"]
                   if s["id"] in ("V1.rayleigh_upper", "V2.existence", "V3.temple_lower"))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_s2_and_s3_say_that_the_radius_is_not_finite():
    """S2 and S3 on an infinite upper modulus: INCONCLUSIVE with the reason code NONFINITE_BOUND, a summary that says
    the radius is not finite (not that an eigenvalue lies within it), and no nan in the certificate (an overflowed
    bound is stored as inf). A zero claimed vector keeps its own wording and no reason code."""
    claim = ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=E0, operator=sp.csr_matrix(_hermitian(Z_INF)),
                     kind="lowest")
    cert = ce.certify(claim, None, ce.Options(seed=1))
    s2, s3 = cert.statement("S2.eigenpairs"), cert.statement("S3.completeness")
    assert s2["status"] == s3["status"] == "INCONCLUSIVE"
    assert s2["evidence"]["reason_code"] == s3["evidence"]["reason_code"] == "NONFINITE_BOUND"
    assert "is not finite" in s2["summary"] and "within its radius" not in s2["summary"]
    assert "overflowed" in s3["summary"] and s2["evidence"]["max_radius"] == np.inf
    for key, v in cert.arrays.items():
        a = np.asarray(v)
        assert not (a.dtype.kind in "fc" and np.isnan(a).any()), key
    zero = ce.Claim(eigenvalues=np.array([1.0]), eigenvectors=np.zeros((3, 1), dtype=complex),
                    operator=sp.csr_matrix(_hermitian(Z_FIN)), kind="lowest")
    s3z = ce.certify(zero, None, ce.Options(seed=1)).statement("S3.completeness")
    assert s3z["summary"].startswith("some claimed pair has no finite enclosure (zero eigenvector)")
    assert "reason_code" not in s3z["evidence"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inertia_gives_no_count_on_a_non_finite_backward_error():
    from certify_ed.inertia import inertia_dense
    r = inertia_dense(_hermitian(Z_INF), 0.5)
    assert r.negative is None and r.beta == np.inf and not r.conclusive and "not finite" in r.note
    r = inertia_dense(_hermitian(complex(1e100, 1.0)), 0.5)
    assert r.conclusive and np.isfinite(r.beta)


