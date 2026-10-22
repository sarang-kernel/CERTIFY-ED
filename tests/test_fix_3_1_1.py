"""Fixes of 3.1.1 (known issues of 3.1.0): INT-02 scope-block validator, INT-03 exported radii rounded upward."""
import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import Certificate, canonical_json


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _cert(heis8, operator=None, **opts):
    spec, H, w, V = heis8
    op = H if operator is None else operator
    return ce.certify(ce.Claim(w[:4], V[:, :4], operator=op, kind="lowest"), spec, ce.Options(**opts))


def _rewrite_with_valid_hash(path, edit):
    """Edit a saved certificate body and recompute its body hash, as a writer with a bug would."""
    body = json.loads(Path(path).read_text())
    integ = body.pop("integrity")
    edit(body)
    integ["body_sha256"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    body["integrity"] = integ
    Path(path).write_text(json.dumps(body))


# ------------------------------------------------------------------------------------ INT-02
@pytest.mark.parametrize("opts", [{}, {"skip": ["S3"]}, {"only": ["S2"]}, {"budget_seconds": 0.0}])
def test_scope_validator_accepts_what_certify_writes(heis8, tmp_path, opts):
    c = _cert(heis8, **opts)
    v = c.verify_scope()
    assert v["ok"] is True, v["problems"]
    rep = Certificate.load(c.save(str(tmp_path / "cert"))).verify_integrity()
    assert rep["ok"] and rep["scope_ok"] is True and rep["body_hash_ok"]


def test_scope_validator_rejects_a_statement_dropped_from_the_scope_block(heis8, tmp_path):
    c = _cert(heis8)
    p = c.save(str(tmp_path / "cert"))
    victim = c.body["scope"]["in_scope"][-1]["id"]

    def drop(body):
        body["scope"]["in_scope"] = [x for x in body["scope"]["in_scope"] if x["id"] != victim]
    _rewrite_with_valid_hash(p, drop)
    with pytest.raises(ValueError, match="scope"):
        Certificate.load(p)                                   # load verifies by default and refuses it
    rep = Certificate.load(p, verify=False).verify_integrity()
    assert rep["body_hash_ok"], "the hash was recomputed; only the scope check can catch this"
    assert rep["scope_ok"] is False and rep["ok"] is False
    assert any(victim in m for m in rep["scope_problems"])


def test_scope_validator_rejects_an_in_scope_statement_missing_from_the_statements(heis8):
    c = copy.deepcopy(_cert(heis8))
    victim = c.body["scope"]["in_scope"][0]["id"]
    c.body["statements"] = [s for s in c.body["statements"] if s["id"] != victim]
    v = c.verify_scope()
    assert v["ok"] is False and any("absent from the statements" in m for m in v["problems"])


def test_scope_validator_rejects_narrowed_but_certified_and_unknown_codes(heis8):
    c = _cert(heis8, skip=["S3"])
    bad = copy.deepcopy(c)
    for s in bad.body["statements"]:
        if s["id"] == "S3.completeness":
            s["status"] = "CERTIFIED"
    assert bad.verify_scope()["ok"] is False
    bad2 = copy.deepcopy(c)
    bad2.body["scope"]["narrowed"][0]["reason_code"] = "BECAUSE"
    assert any("unknown reason code" in m for m in bad2.verify_scope()["problems"])


def test_scope_validator_says_not_checked_for_legacy_certificates(heis8):
    c = copy.deepcopy(_cert(heis8))
    del c.body["scope"]
    assert c.verify_scope()["ok"] is None


# ------------------------------------------------------------------------------------ INT-03
@pytest.mark.parametrize("eps", [1e-14, 1e-16])
def test_exported_radii_bound_the_exact_sum_with_a_model_shift(heis8, eps):
    spec, H, w, V = heis8
    Hp = (H + eps * ce.build_bitwise(ce.heisenberg_chain(8, J2=0.3))).tocsr()   # producer matrix within tolerance
    c = _cert(heis8, operator=Hp)
    by = {s["id"]: s for s in c.statements}
    assert by["S1.model_fidelity"]["status"] == "CERTIFIED"
    ev = by["S2.eigenpairs"]["evidence"]
    assert ev["model_shift"] > 0 and ev["radius_refers_to"] == "declared model (spec)"
    shift = Fraction(ev["model_shift"])
    assert Fraction(float(c.arrays["radius"].max())) >= Fraction(ev["max_radius"]) + shift
    assert Fraction(float(c.arrays["existence_radius"].max())) >= Fraction(ev["max_existence_radius"]) + shift


def test_exported_radii_with_the_builder_shift_bound_the_exact_sum(heis8):
    # CERTIFY-ED's own matrix still carries the builder's rounding bound as model shift (nonzero), so INT-03
    # concerned almost every exported radius of 3.1.0
    c = _cert(heis8)
    ev = {s["id"]: s for s in c.statements}["S2.eigenpairs"]["evidence"]
    shift = Fraction(ev["model_shift"])
    assert shift > 0
    assert Fraction(float(c.arrays["radius"].max())) >= Fraction(ev["max_radius"]) + shift


def test_exported_radii_unchanged_without_a_model_shift(heis8):
    spec, H, w, V = heis8
    c = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"))     # no spec: no model shift
    ev = {s["id"]: s for s in c.statements}["S2.eigenpairs"]["evidence"]
    assert ev["model_shift"] == 0.0
    assert float(c.arrays["radius"].max()) == ev["max_radius"]
    assert float(c.arrays["existence_radius"].max()) == ev["max_existence_radius"]
