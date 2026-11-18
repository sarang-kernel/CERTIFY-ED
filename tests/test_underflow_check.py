"""INT-01 (3.2.0.dev0): the no-underflow hypothesis is enforced, conservatively, with reason code UNDERFLOW_RISK.

Planted inputs below 2^-900 (or 2^-450 where the quantity is squared by a norm) make S2 and the statements
that rest on it INCONCLUSIVE; inputs without a finding get exactly the statuses of a run without the check;
the opt-out records itself. Negative control: a finding never leaves S2 CERTIFIED."""
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify_ed import Claim, Options, certify, heisenberg_chain, build_bitwise, tfim_chain
from certify_ed.fp import UNDERFLOW_THRESHOLD, tiny_nonzero, underflow_scan, min_nonzero_magnitude
from certify_ed.scope import REASON_CODES

DEPENDENT = ("S2.eigenpairs", "S3.completeness", "S4.eigenspaces", "S8.degeneracy")


def _statuses(cert):
    return {s["id"]: (s["status"], s["evidence"].get("reason_code")) for s in cert.statements}


def _codes(cert):
    return {s["id"]: s["evidence"].get("reason_code") for s in cert.statements}


@pytest.fixture(scope="module")
def chain():
    spec = heisenberg_chain(6, 1.0)
    H = build_bitwise(spec).toarray()
    w, V = la.eigh(H)
    return spec, H, w, V


def _lowest(H, w, V, k=4, **kw):
    return Claim(eigenvalues=w[:k], eigenvectors=V[:, :k], operator=H, kind="lowest", **kw)


# ---------------------------------------------------------------- helpers in fp.py

def test_tiny_nonzero_reports_only_nonzero_entries_below_threshold():
    a = np.array([0.0, 1.0, 2.0 ** -950, -2.0 ** -1060, 2.0 ** -899])
    f = tiny_nonzero(a)
    assert f["count"] == 2 and f["first_indices"] == [2, 3]
    assert f["min_magnitude"] == 2.0 ** -1060
    assert tiny_nonzero(np.array([0.0, 0.0, 1.0])) is None            # exact zeros are fine
    assert tiny_nonzero(np.zeros(0)) is None
    assert tiny_nonzero(np.array([1, 2, 3])) is None                   # integer data: not an operand of fl()


def test_tiny_nonzero_scans_complex_components_and_sparse_data():
    z = np.array([1.0 + 0j, 1.0 + 1j * 2.0 ** -950])
    f = tiny_nonzero(z)
    assert f["count"] == 1 and f["first_indices"] == [3]               # imaginary part of the second entry
    S = sp.csr_matrix(np.array([[1.0, 0.0], [2.0 ** -1000, 1.0]]))
    assert tiny_nonzero(S)["count"] == 1
    assert min_nonzero_magnitude(S) == 2.0 ** -1000
    assert min_nonzero_magnitude(np.zeros(3)) is None


def test_underflow_scan_names_parts_and_thresholds():
    parts = [("a", np.array([2.0 ** -500])), ("b", np.array([2.0 ** -500]), 2.0 ** -450)]
    out = underflow_scan(parts)
    assert [f["where"] for f in out] == ["b"] and out[0]["threshold"] == 2.0 ** -450
    assert UNDERFLOW_THRESHOLD == 2.0 ** -900


def test_reason_code_is_registered():
    assert "UNDERFLOW_RISK" in REASON_CODES


# ---------------------------------------------------------------- planted inputs

def test_planted_subnormal_eigenvector_entry(chain):
    spec, H, w, V = chain
    V2 = V[:, :4].copy()
    V2[3, 1] = 2.0 ** -1060                                             # a subnormal, far below 2^-900
    cert = certify(Claim(eigenvalues=w[:4], eigenvectors=V2, operator=H, kind="lowest"), spec, Options())
    st = _statuses(cert)
    assert st["S1.model_fidelity"] == ("CERTIFIED", None)               # S1 does not rest on S2
    for sid in DEPENDENT:
        assert st[sid] == ("INCONCLUSIVE", "UNDERFLOW_RISK"), (sid, st[sid])
    s2 = cert.statements[1]
    f = s2["evidence"]["underflow"]["findings"]
    assert f[0]["where"] == "claim.eigenvectors" and f[0]["min_magnitude"] == 2.0 ** -1060
    assert f[0]["first_indices"] == [3 * 4 + 1]                        # flat index of X[3, 1]
    assert "would otherwise be" in s2["summary"]
    assert cert.verify_scope()["ok"]


def test_planted_tiny_matrix_entry(chain):
    spec, H, w, V = chain
    H3 = H.copy()
    H3[0, 1] += 2.0 ** -950
    H3[1, 0] += 2.0 ** -950                                             # Hermitian; S1 reports the difference itself
    cert = certify(_lowest(H3, w, V), spec, Options())
    st = _statuses(cert)
    for sid in DEPENDENT:
        assert st[sid] == ("INCONCLUSIVE", "UNDERFLOW_RISK"), (sid, st[sid])
    f = cert.statements[1]["evidence"]["underflow"]["findings"]
    assert any(x["where"] == "operator.entries" and x["min_magnitude"] == 2.0 ** -950 for x in f)
    assert cert.verify_scope()["ok"]


def test_planted_tiny_sparse_matrix_entry(chain):
    spec, H, w, V = chain
    H3 = sp.lil_matrix(H)
    H3[2, 5] = H3[2, 5] + 2.0 ** -1000
    H3[5, 2] = H3[5, 2] + 2.0 ** -1000
    cert = certify(_lowest(H3.tocsr(), w, V), spec, Options())
    assert _statuses(cert)["S2.eigenpairs"] == ("INCONCLUSIVE", "UNDERFLOW_RISK")


def test_radius_forced_tiny_by_exact_eigenpair_of_scaled_operator():
    # exact eigenpairs of a diagonal operator scaled to 1e-258 with eigenvectors of norm 1e125: every stored
    # entry and the claimed values are above 2^-900 = 1.2e-271, the exact residual is zero, and the computed
    # Theorem 1 radius gamma |H| ~ 1e-274 is below the threshold. Until track/backlog-proofs the radius was
    # scanned and S2 was INCONCLUSIVE (UNDERFLOW_RISK); the Theorem 1 radius is now evaluated under the
    # underflow-tolerant model (INT-05, docs/proofs/int05_underflow.md), so the tiny radius is a bound for
    # every magnitude and S2 is CERTIFIED; the scan covers the inputs only
    D = np.diag([1.0, 2.0, 3.0, 4.0]) * 1e-258
    X = np.eye(4)[:, :2] * 1e125
    assert np.abs(D[D != 0]).min() > UNDERFLOW_THRESHOLD
    claim = Claim(eigenvalues=[1e-258, 2e-258], eigenvectors=X, operator=D, kind="lowest")
    cert = certify(claim, None, Options())
    st = _statuses(cert)
    assert st["S2.eigenpairs"] == ("CERTIFIED", None)
    rec = cert.body["arithmetic"]["underflow_check"]
    assert rec["findings"] == [] and "enclosure.radius" not in rec["scanned"]
    radius = np.asarray(cert.arrays["radius"])
    assert np.all(radius > 0) and radius.max() < UNDERFLOW_THRESHOLD
    # the exact eigenvalues are the claimed values: the tiny radii are correct bounds
    assert np.all(np.abs(np.array([1e-258, 2e-258]) - np.array([1e-258, 2e-258])) <= radius)


def test_tiny_claimed_eigenvalue_is_reported():
    D = np.diag([0.0, 1.0, 2.0, 3.0])
    claim = Claim(eigenvalues=[2.0 ** -1000, 1.0], eigenvectors=np.eye(4)[:, :2], operator=D, kind="subset")
    cert = certify(claim, None, Options())
    s2 = cert.statements[1]
    assert (s2["status"], s2["evidence"]["reason_code"]) == ("INCONCLUSIVE", "UNDERFLOW_RISK")
    assert s2["evidence"]["underflow"]["findings"][0]["where"] == "claim.eigenvalues"


def test_matrix_free_operator_is_probed():
    # a black-box matvec whose output has a tiny nonzero component: the probe matvec finds it
    n = 8
    d = np.arange(1.0, n + 1)
    d[-1] = 2.0 ** -1000                                                # a tiny diagonal element

    def mv(x):
        return d * np.asarray(x).reshape(-1)

    L = spla.LinearOperator((n, n), matvec=mv, dtype=float)
    claim = Claim(eigenvalues=[1.0, 2.0], eigenvectors=np.eye(n)[:, :2], operator=L, kind="subset")
    cert = certify(claim, None, Options())
    s2 = cert.statements[1]
    assert s2["evidence"]["reason_code"] == "UNDERFLOW_RISK"
    assert "operator.probe_matvec" in cert.body["arithmetic"]["underflow_check"]["scanned"]
    assert any(f["where"] == "operator.probe_matvec" for f in s2["evidence"]["underflow"]["findings"])


def test_spec_operator_without_stored_matrix_is_probed():
    spec = tfim_chain(5, 1.0, 0.7)
    H = build_bitwise(spec).toarray()
    w, V = la.eigh(H)
    claim = Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], kind="lowest")        # operator=None: spec operator
    cert = certify(claim, spec, Options(model_fidelity=False))
    scanned = cert.body["arithmetic"]["underflow_check"]["scanned"]
    assert "operator.probe_matvec" in scanned and "operator.norm_bound" in scanned
    assert cert.statements[1]["evidence"].get("reason_code") is None


def test_complex_data_components_are_scanned():
    H = np.array([[2.0, 1j], [-1j, 2.0]])
    w, V = la.eigh(H)
    V2 = V.copy()
    V2[0, 0] = V2[0, 0] + 1j * 2.0 ** -1000
    cert = certify(Claim(eigenvalues=w, eigenvectors=V2, operator=H, kind="all"), None, Options())
    assert _statuses(cert)["S2.eigenpairs"] == ("INCONCLUSIVE", "UNDERFLOW_RISK")
    cert_ok = certify(Claim(eigenvalues=w, eigenvectors=V, operator=H, kind="all"), None, Options())
    assert cert_ok.statements[1]["status"] == "CERTIFIED"


# ---------------------------------------------------------------- unchanged fixtures and the opt-out

@pytest.mark.parametrize("kind", ["lowest", "subset", "all"])
def test_standard_fixtures_unchanged_by_underflow_check(chain, kind):
    spec, H, w, V = chain
    k = w.size if kind == "all" else 4
    claim = Claim(eigenvalues=w[:k], eigenvectors=V[:, :k], operator=H, kind=kind)
    with_check = certify(claim, spec, Options())
    without = certify(claim, spec, Options(underflow_check=False))
    assert _statuses(with_check) == _statuses(without)
    assert "UNDERFLOW_RISK" not in _codes(with_check).values()
    rec = with_check.body["arithmetic"]["underflow_check"]
    assert rec["performed"] and rec["findings"] == [] and "claim.eigenvectors" in rec["scanned"]
    assert "enclosure.radius" not in rec["scanned"]        # INT-05: outputs are no longer scanned
    assert with_check.body["scope"]["options"]["underflow_check"] is True


def test_lapack_eigenvectors_with_numerically_zero_entries_are_not_flagged():
    # eigh of a 10-site Heisenberg chain returns entries of order 1e-265 (numerical zeros of a symmetry
    # sector): above 2^-900, so the hypothesis as stated holds and the statuses are those of 3.1.0
    spec = heisenberg_chain(10, 1.0)
    static = spec.to_quspin_static()
    static[2][1][3][0] *= -1                                            # the test_certify fixture with one flipped bond
    from certify_ed import ModelSpec
    bad = ModelSpec.from_static(spec.n_sites, spec.local, static)
    H = build_bitwise(bad)
    w, V = la.eigh(H.toarray())
    a = np.abs(V[:, :3])
    assert (a[a > 0] < 2.0 ** -600).any()                               # the fixture does carry such entries
    cert = certify(Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest"), spec, Options())
    assert cert.statements[1]["status"] == "CERTIFIED"                  # S1 FAILED, S2 as in 3.1.0
    assert cert.body["arithmetic"]["underflow_check"]["findings"] == []
    # the intermediate quantities of the bound do fall below 2^-900 for this input (INT-05, not scanned)
    from certify_ed.enclosure import OperatorAdapter, residuals
    res = residuals(OperatorAdapter(H), w[:3], V[:, :3])
    assert np.abs(res.E[res.E > 0]).min() < 2.0 ** -900


def test_opt_out_records_itself_and_assumes_the_hypothesis(chain):
    spec, H, w, V = chain
    V2 = V[:, :4].copy()
    V2[3, 1] = 2.0 ** -1060
    claim = Claim(eigenvalues=w[:4], eigenvectors=V2, operator=H, kind="lowest")
    cert = certify(claim, spec, Options(underflow_check=False))
    assert cert.body["scope"]["options"]["underflow_check"] is False
    rec = cert.body["arithmetic"]["underflow_check"]
    assert rec["performed"] is False and rec["scanned"] == [] and rec["findings"] == []
    assert "UNDERFLOW_RISK" not in _codes(cert).values()
    # the opt-out does not make anything CERTIFIED that the planted vector would not have given anyway
    assert cert.statements[1]["status"] in ("FAILED", "INCONCLUSIVE", "CERTIFIED")
    assert cert.verify_scope()["ok"]


def test_cli_flag_maps_to_the_option():
    import argparse
    from certify_ed.cli import _options_from_args
    a = argparse.Namespace(assume_no_underflow=True)
    assert _options_from_args(a).underflow_check is False
    assert _options_from_args(argparse.Namespace()).underflow_check is True


# ---------------------------------------------------------------- negative control

def test_negative_control_finding_never_leaves_s2_certified(chain):
    """A finding in any scanned quantity, with every other input exact, must not be CERTIFIED."""
    spec, H, w, V = chain
    base = certify(_lowest(H, w, V), spec, Options())
    assert base.statements[1]["status"] == "CERTIFIED"
    for where in ("vector", "matrix", "eigenvalue"):
        Hx, wx, Vx = H.copy(), w.copy(), V[:, :4].copy()
        if where == "vector":
            Vx[0, 0] = 2.0 ** -1000
        elif where == "matrix":
            Hx[3, 3] = Hx[3, 3] + 2.0 ** -1000 if Hx[3, 3] == 0 else Hx[3, 3]
            Hx[0, 2] = 2.0 ** -1000
            Hx[2, 0] = 2.0 ** -1000
        else:
            wx[0] = 2.0 ** -1000
        cert = certify(Claim(eigenvalues=wx[:4], eigenvectors=Vx, operator=Hx, kind="lowest"), spec, Options())
        assert cert.statements[1]["status"] != "CERTIFIED", where
        assert cert.statements[1]["evidence"]["reason_code"] == "UNDERFLOW_RISK", where
        for s in cert.statements:
            if s["id"] in DEPENDENT:
                assert s["status"] != "CERTIFIED", (where, s["id"])
