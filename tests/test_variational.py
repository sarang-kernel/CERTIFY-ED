"""E-05 variational certificates (certify_ed.variational): Rayleigh upper bound, existence radius, Temple lower
bound under a declared or a certified gap, adapters, declared estimates, negative controls, record schema, CLI."""
import json

import numpy as np
import pytest

from certify_ed.certify import Claim, Options, certify
from certify_ed.construct import build_bitwise
from certify_ed.schema import SchemaValidationError, validate_record
from certify_ed.spec import heisenberg_chain
from certify_ed.variational import (VARIATIONAL_SCHEMA, certify_variational, format_variational, state_from_mps,
                                    state_from_sparse_ci)


@pytest.fixture(scope="module")
def heis():
    spec = heisenberg_chain(8)
    H = build_bitwise(spec)
    w, V = np.linalg.eigh(H.toarray())
    return spec, H, w, V


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


def test_rayleigh_and_existence_on_perturbed_ground_state(heis):
    spec, H, w, V = heis
    rng = np.random.default_rng(1)
    psi = V[:, 0] + 1e-3 * rng.standard_normal(256)
    rec = certify_variational(H, psi)
    assert rec["schema"] == VARIATIONAL_SCHEMA
    v1 = stmt(rec, "V1.rayleigh_upper")
    assert v1["status"] == "CERTIFIED" and v1["hypotheses"] == []
    theta = float(psi @ (H @ psi) / (psi @ psi))
    assert w[0] <= v1["evidence"]["upper_bound"]
    assert theta <= v1["evidence"]["upper_bound"] <= theta + 1e-12
    v2 = stmt(rec, "V2.existence")
    lo, hi = v2["evidence"]["interval"]
    assert v2["status"] == "CERTIFIED" and lo <= w[0] <= hi
    r_exact = np.linalg.norm(H @ psi - theta * psi) / np.linalg.norm(psi)
    assert r_exact <= v2["evidence"]["radius"] <= r_exact * (1 + 1e-10) + 1e-13
    assert stmt(rec, "V3.temple_lower")["status"] == "NOT_CHECKED"
    assert stmt(rec, "V3.temple_lower")["evidence"]["reason_code"] == "NEEDS_DEPENDENCY"
    assert "next_action" in stmt(rec, "V3.temple_lower")["evidence"]
    assert validate_record(rec) in (True, None)


def test_temple_under_declared_gap_names_the_hypothesis(heis):
    spec, H, w, V = heis
    rng = np.random.default_rng(2)
    psi = V[:, 0] + 1e-2 * rng.standard_normal(256)
    rec = certify_variational(None, psi, spec=spec, gap_hypothesis=float(w[1]), gap_source="ED of the same chain (test)")
    assert rec["operator"]["model_shift"] > 0
    v3 = stmt(rec, "V3.temple_lower")
    assert v3["status"] == "CERTIFIED_UNDER_HYPOTHESIS"
    assert v3["hypotheses"][0]["name"] == "declared_gap"
    assert "ED of the same chain" in v3["hypotheses"][0]["source"]
    assert "HYPOTHESIS" in v3["summary"] and "declared" in v3["summary"].lower()
    assert v3["evidence"]["lower_bound"] <= w[0] <= stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"]
    assert "gap_certificate" in v3["evidence"]["next_action"]
    assert not any(s["status"] == "CERTIFIED" for s in rec["statements"] if s["id"] == "V3.temple_lower")


def test_temple_with_certificate_is_certified_without_hypothesis(heis):
    spec, H, w, V = heis
    cert = certify(Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, Options())
    assert cert.status_of("S3.completeness") == "CERTIFIED"
    rng = np.random.default_rng(3)
    psi = V[:, 0] + 1e-2 * rng.standard_normal(256)
    rec = certify_variational(None, psi, spec=spec, gap_certificate=cert)
    v3 = stmt(rec, "V3.temple_lower")
    assert v3["status"] == "CERTIFIED" and v3["hypotheses"] == []
    assert v3["evidence"]["gap_certificate"]["ok"] and v3["evidence"]["rho"] <= w[1]
    assert v3["evidence"]["lower_bound"] <= w[0]
    # k = 4 certificate: the second smallest lower endpoint
    cert4 = certify(Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, Options(eigenspaces="none", degeneracy=False))
    assert cert4.status_of("S3.completeness") == "CERTIFIED"
    rec4 = certify_variational(None, psi, spec=spec, gap_certificate=cert4)
    v3 = stmt(rec4, "V3.temple_lower")
    assert v3["status"] == "CERTIFIED" and v3["evidence"]["rho"] <= w[1] and v3["evidence"]["lower_bound"] <= w[0]
    assert "second smallest" in v3["evidence"]["gap_certificate"]["source"]


def test_negative_control_false_gap_hypothesis_breaks_the_bound(heis):
    """Two-level mixture: Temple with rho = lambda_2 is an equality; with rho = lambda_4 > lambda_2 (false hypothesis)
    the 'lower bound' exceeds lambda_1 -- and the record says it rests on the hypothesis."""
    spec, H, w, V = heis
    psi = np.cos(0.6) * V[:, 0] + np.sin(0.6) * V[:, 1]
    rec_true = certify_variational(H, psi, gap_hypothesis=float(np.nextafter(w[1], -np.inf)), gap_source="exact")
    lb = stmt(rec_true, "V3.temple_lower")["evidence"]["lower_bound"]
    assert lb <= w[0] and w[0] - lb < 1e-9
    rec_false = certify_variational(H, psi, gap_hypothesis=float(w[4]), gap_source="WRONG")
    v3 = stmt(rec_false, "V3.temple_lower")
    assert v3["status"] == "CERTIFIED_UNDER_HYPOTHESIS" and v3["status"] != "CERTIFIED"
    assert v3["evidence"]["lower_bound"] > w[0]          # the bound is false
    assert v3["hypotheses"][0]["name"] == "declared_gap" and v3["hypotheses"][0]["source"] == "WRONG"


def test_negative_control_orthogonal_trial_state_is_inconclusive(heis):
    spec, H, w, V = heis
    psi = V[:, 1]
    rec = certify_variational(H, psi, gap_hypothesis=float(w[1]), gap_source="exact")
    v3 = stmt(rec, "V3.temple_lower")
    assert v3["status"] == "INCONCLUSIVE" and v3["evidence"]["reason_code"] == "NOT_APPLICABLE"
    assert "theta < rho" in v3["summary"]
    assert stmt(rec, "V1.rayleigh_upper")["status"] == "CERTIFIED"      # the upper bound still holds (lambda_1 <= lambda_2)
    assert w[0] <= stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"]


def test_negative_control_mismatched_certificate(heis):
    spec, H, w, V = heis
    other = heisenberg_chain(8, J2=0.3)
    Ho = build_bitwise(other)
    wo, Vo = np.linalg.eigh(Ho.toarray())
    cert = certify(Claim(wo[:1], Vo[:, :1], operator=Ho, kind="lowest"), other, Options())
    rec = certify_variational(None, V[:, 0], spec=spec, gap_certificate=cert)
    v3 = stmt(rec, "V3.temple_lower")
    assert v3["status"] == "NOT_CHECKED" and "spec hash differs" in v3["evidence"]["gap_certificate"]["reason"]


def test_tiny_norms_and_zero_vector(heis):
    spec, H, w, V = heis
    rec = certify_variational(H, V[:, 0] * 2.0 ** -450)
    assert stmt(rec, "V1.rayleigh_upper")["status"] == "CERTIFIED"
    assert w[0] <= stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"] <= w[0] + 1e-12
    rec = certify_variational(H, V[:, 0] * 2.0 ** -950)
    assert all(s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNDERFLOW_RISK" for s in rec["statements"][:3])
    rec = certify_variational(H, V[:, 0] * 2.0 ** -950, underflow_check=False)
    assert rec["underflow_check"]["enabled"] is False
    assert all(s["status"] == "NOT_CHECKED" for s in rec["statements"][:3])     # the norm cannot be resolved
    rec = certify_variational(H, np.zeros(256))
    assert all(s["status"] == "NOT_CHECKED" for s in rec["statements"][:3])
    rec = certify_variational(H, np.full(256, np.nan))
    assert all(s["status"] == "NOT_CHECKED" for s in rec["statements"][:3])


def test_complex_operator_and_state():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((12, 12)) + 1j * rng.standard_normal((12, 12))
    A = A + A.conj().T
    w, V = np.linalg.eigh(A)
    psi = V[:, 0] + 1e-3 * (rng.standard_normal(12) + 1j * rng.standard_normal(12))
    rec = certify_variational(A, psi, gap_hypothesis=float(w[1]), gap_source="eigh")
    assert w[0] <= stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"]
    assert stmt(rec, "V3.temple_lower")["evidence"]["lower_bound"] <= w[0]
    lo, hi = stmt(rec, "V2.existence")["evidence"]["interval"]
    assert lo <= w[0] <= hi
    with pytest.raises(ValueError, match="not Hermitian"):
        certify_variational(rng.standard_normal((12, 12)), psi)


def test_mps_adapter(heis):
    spec, H, w, V = heis
    # product state up,down,up,down,... as an MPS with D = 1: exact
    up, dn = np.array([[[1.0], [0.0]]]), np.array([[[0.0], [1.0]]])
    st = state_from_mps([up if i % 2 == 0 else dn for i in range(8)])
    assert st.contraction_error < 1e-14            # D = 1: one product per step, gamma_1 per entry
    assert np.count_nonzero(st.vector) == 1
    rec = certify_variational(H, st)
    assert abs(stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"] - (-2.0)) < 1e-12       # Neel energy -N/4
    rng = np.random.default_rng(6)
    tens = [rng.standard_normal((1, 2, 3))] + [rng.standard_normal((3, 2, 3)) for _ in range(6)] + [rng.standard_normal((3, 2, 1))]
    st = state_from_mps(tens)
    exact = tens[0][0]
    for A in tens[1:]:
        exact = np.tensordot(exact, A, axes=([-1], [0])).reshape(-1, A.shape[2])
    assert np.linalg.norm(st.vector - exact[:, 0]) <= st.contraction_error
    rec = certify_variational(H, st)
    assert rec["state"]["form"] == "mps" and rec["state"]["contraction_error"] > 0
    assert np.isfinite(rec["state"]["theta_source_vs_contracted"])
    assert w[0] <= stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"]
    # size limit
    big = state_from_mps([np.ones((1, 2, 1))] * 23)
    assert big.vector is None and big.reason_code == "SIZE_LIMIT"
    rec = certify_variational(H, {"mps": [np.ones((1, 2, 1))] * 23})
    assert all(s["status"] == "NOT_CHECKED" and s["evidence"]["reason_code"] == "SIZE_LIMIT" for s in rec["statements"][:3])
    assert "2^22" in stmt(rec, "V1.rayleigh_upper")["evidence"]["next_action"]
    with pytest.raises(ValueError):
        state_from_mps([np.ones((1, 2, 2)), np.ones((3, 2, 1))])


def test_sparse_ci_adapter_and_declared_estimates(heis):
    spec, H, w, V = heis
    idx = np.flatnonzero(np.abs(V[:, 0]) > 0.2)
    st = state_from_sparse_ci(idx, V[idx, 0], 256)
    assert st.record["nnz"] == idx.size
    rec = certify_variational(H, st, declared={"energy": -3.65, "variance": 1e-4, "samples": 10000, "method": "VMC"})
    assert w[0] <= stmt(rec, "V1.rayleigh_upper")["evidence"]["upper_bound"]
    v4 = stmt(rec, "V4.declared_estimates")
    assert v4["status"] == "DECLARED" and v4["evidence"]["reason_code"] == "DECLARED_ESTIMATE"
    assert v4["evidence"]["declared"]["method"] == "VMC" and "not certified" in v4["summary"]
    assert "next_action" in v4["evidence"]
    with pytest.raises(ValueError, match="duplicate"):
        state_from_sparse_ci([1, 1], [0.5, 0.5], 256)
    # declared only, no vector: nothing certified
    rec = certify_variational(H, None, declared={"energy": -3.65})
    assert all(s["status"] == "NOT_CHECKED" for s in rec["statements"][:3])
    assert stmt(rec, "V4.declared_estimates")["status"] == "DECLARED"


def test_record_schema_rejects_hypothesis_free_certified_under_hypothesis(heis):
    spec, H, w, V = heis
    rec = certify_variational(H, V[:, 0] + 1e-3, gap_hypothesis=float(w[1]), gap_source="x")
    assert validate_record(rec) in (True, None)
    bad = json.loads(json.dumps(rec))
    stmt(bad, "V3.temple_lower")["hypotheses"] = []
    with pytest.raises(SchemaValidationError):
        validate_record(bad)
    bad2 = json.loads(json.dumps(rec))
    bad2["statements"][0]["status"] = "VERIFIED"
    with pytest.raises(SchemaValidationError):
        validate_record(bad2)
    bad3 = json.loads(json.dumps(rec))
    bad3["description"] = "edited"
    with pytest.raises(SchemaValidationError, match="record_sha256"):
        validate_record(bad3)


def test_cli_variational(tmp_path, heis):
    from certify_ed.cli import main
    spec, H, w, V = heis
    np.save(tmp_path / "state.npy", V[:, 0] + 1e-3)
    with open(tmp_path / "spec.json", "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f)
    out = tmp_path / "rec.json"
    rc = main(["variational", "--spec", str(tmp_path / "spec.json"), "--state", str(tmp_path / "state.npy"),
               "--gap", str(float(w[1])), "--gap-source", "test", "--declared", "energy=-3.65,method=VMC", "--json", str(out)])
    assert rc == 0
    rec = json.load(open(out, encoding="utf-8"))
    assert rec["schema"] == VARIATIONAL_SCHEMA
    assert stmt(rec, "V3.temple_lower")["status"] == "CERTIFIED_UNDER_HYPOTHESIS"
    assert stmt(rec, "V4.declared_estimates")["status"] == "DECLARED"
    assert "hypothesis declared_gap" in format_variational(rec)
