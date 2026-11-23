"""JSON Schema of the certificate (schema/certificate-2.1.schema.json), validation at load, the producer-bundle
schema, and the conformance check of the reference bundles (B-02, F-09).

Soundness note: nothing here changes a status. The negative controls show that a certificate carrying an
unknown status, an unknown reason code, a CORROBORATED non-S5 statement or a duplicated scope entry is
rejected by the validator, so a third-party writer cannot pass such a file off as conformant.
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import STATEMENT_IDS, Certificate
from certify_ed.conformance import check_bundle, format_report
from certify_ed.io.bundle import write_bundle
from certify_ed.schema import (CERTIFICATE_SCHEMA, PRODUCER_BUNDLE_SCHEMA, SCHEMA_FILES, SchemaValidationError,
                               have_jsonschema, render, schema_errors, structural_check, validate_bundle_meta,
                               validate_certificate_body)
from certify_ed.scope import REASON_CODES
from conftest import HAVE_FLINT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "conformance", "reference")
needs_jsonschema = pytest.mark.skipif(not have_jsonschema(), reason="jsonschema not installed")


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


@pytest.fixture(scope="module")
def fixture_certificates(heis8, tmp_path_factory):
    """Real certificates of every shape the code writes: with/without spec, values only, skip/only/budget,
    every preset, a window claim, a planted error (FAILED), stored eigenvectors."""
    spec, H, w, V = heis8
    d = tmp_path_factory.mktemp("certs")
    low = lambda k=4: ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest")
    cases = {
        "lowest_spec": (low(), spec, ce.Options()),
        "lowest_nospec": (low(), None, ce.Options()),
        "values_only": (ce.Claim(w[:4], None, operator=H, kind="subset"), spec, ce.Options()),
        "skip_S3": (low(), spec, ce.Options(skip=["S3"], narrowing_reason="test")),
        "only_S1S2": (low(), spec, ce.Options(only=["S1", "S2"])),
        "budget": (low(), spec, ce.Options(budget_seconds=0.0, budget_mb=0.001)),
        "window": (ce.Claim(w[2:6], V[:, 2:6], operator=H, kind="window",
                            window=(float(w[2]) - 1e-3, float(w[5]) + 1e-3)), spec, ce.Options()),
        "planted_fail": (ce.Claim(w[:3] + np.array([0.0, 0.3, 0.0]), V[:, :3], operator=H, kind="lowest"), spec,
                         ce.Options()),
        "store_vectors": (ce.Claim(w[:3], V[:, :3], operator=H, kind="lowest"), spec,
                          ce.Options(store_eigenvectors=True)),
    }
    for name in ("lab-minimal", "benchmark", "whole-spectrum", "labels") + (("proof-lemma",) if HAVE_FLINT else ()):
        kind_all = name == "whole-spectrum"
        claim = ce.Claim(w, V, operator=H, kind="all") if kind_all else low()
        cases["preset_" + name] = (claim, spec, ce.Options.from_preset(name, temperatures=[0.5, 2.0]))
    out = {}
    for name, (claim, sp_, opts) in cases.items():
        cert = ce.certify(claim, sp_, opts)
        out[name] = cert.save(str(d / name))
    return out


# ------------------------------------------------------------------------------------ the schema itself
def test_schema_enumerations_match_the_code():
    defs = CERTIFICATE_SCHEMA["$defs"]
    assert defs["statement_id"]["enum"] == list(STATEMENT_IDS)
    assert set(defs["reason_code"]["enum"]) == set(REASON_CODES)
    assert set(defs["status"]["enum"]) == {"CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED", "CORROBORATED"}
    assert CERTIFICATE_SCHEMA["properties"]["schema"]["const"] == "certify-ed/certificate/2.1"
    assert PRODUCER_BUNDLE_SCHEMA["properties"]["schema"]["const"] == "certify-ed/producer-bundle/1.0"


def test_published_schema_files_are_in_sync_with_the_definitions():
    """schema/*.schema.json is rendered from certify_ed.schema; a drift means someone edited one side only."""
    for name, schema in SCHEMA_FILES.items():
        path = os.path.join(ROOT, "schema", name)
        assert os.path.exists(path), f"{path} missing: python -m certify_ed.schema --write schema"
        with open(path, encoding="utf-8") as f:
            assert f.read() == render(schema), f"{name} differs from certify_ed.schema (re-render it)"
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@needs_jsonschema
def test_schemas_are_valid_draft_2020_12():
    import jsonschema
    for schema in (CERTIFICATE_SCHEMA, PRODUCER_BUNDLE_SCHEMA):
        jsonschema.Draft202012Validator.check_schema(schema)


# ------------------------------------------------------------------------------------ fixtures validate
def test_every_fixture_certificate_validates(fixture_certificates):
    for name, path in fixture_certificates.items():
        with open(path) as f:
            body = json.load(f)
        errs = schema_errors(body, CERTIFICATE_SCHEMA)
        assert errs in (None, []), (name, errs[:5])
        assert structural_check(body) == [], name
        # load with validation (default) and with validation required when jsonschema is present
        cert = Certificate.load(path)
        assert cert.status_of("S2.eigenpairs") is not None
        if have_jsonschema():
            assert Certificate.load(path, validate_schema=True).validate_schema() is True
        assert Certificate.load(path, validate_schema=False).body == body


def test_statuses_of_the_fixtures_cover_the_vocabulary(fixture_certificates):
    seen = set()
    for path in fixture_certificates.values():
        with open(path) as f:
            seen |= {s["status"] for s in json.load(f)["statements"]}
    assert {"CERTIFIED", "FAILED", "NOT_CHECKED", "CORROBORATED"} <= seen


def test_unsaved_certificate_validates_as_it_would_be_saved(heis8):
    spec, H, w, V = heis8
    cert = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest"), spec, ce.Options.from_preset("lab-minimal"))
    assert "integrity" not in cert.body
    assert cert.validate_schema() in (True, None)


# ------------------------------------------------------------------------------------ negative controls
def _mutants(body):
    def m(f):
        b = copy.deepcopy(body)
        f(b)
        return b
    return {
        "unknown status": m(lambda b: b["statements"][1].__setitem__("status", "VERIFIED")),
        "unknown reason code (scope)": m(lambda b: b["scope"]["out_of_scope"][0].__setitem__("reason_code", "BECAUSE")),
        "unknown reason code (reason_codes)": m(lambda b: b["scope"]["reason_codes"].__setitem__("MADE_UP", "x")),
        "CORROBORATED on S2": m(lambda b: b["statements"][1].__setitem__("status", "CORROBORATED")),
        "statement twice in scope": m(lambda b: b["scope"]["in_scope"].append({"id": "S1.model_fidelity", "status": "CERTIFIED"})),
        "statement missing from scope": m(lambda b: b["scope"]["in_scope"].pop(0)),
        "in_scope status disagrees": m(lambda b: b["scope"]["in_scope"][0].__setitem__("status", "FAILED")),
        "narrowed but certified": m(lambda b: b["scope"]["narrowed"][0].__setitem__("status", "CERTIFIED")),
        "wrong schema id": m(lambda b: b.__setitem__("schema", "certify-ed/certificate/2.0")),
        "unknown statement id": m(lambda b: b["statements"].append({"id": "S9.extra", "status": "CERTIFIED", "summary": ""})),
        "claim kind": m(lambda b: b["claim"].__setitem__("kind", "some")),
        "n_eigenvalues mismatch": m(lambda b: b["claim"].__setitem__("n_eigenvalues", 99)),
        "no scope block": m(lambda b: b.pop("scope")),
    }


@needs_jsonschema
def test_negative_controls_are_rejected(fixture_certificates):
    with open(fixture_certificates["lowest_spec"]) as f:
        body = json.load(f)
    validate_certificate_body(body)
    for mutant in _mutants(body).values():
        with pytest.raises(SchemaValidationError):
            validate_certificate_body(mutant)


def test_structural_layer_rejects_the_scope_level_negative_controls(fixture_certificates):
    """The structural layer alone (what runs without the optional jsonschema) catches the scope-level mutants."""
    with open(fixture_certificates["lowest_spec"]) as f:
        body = json.load(f)
    for label, mutant in _mutants(body).items():
        if "scope" in label or "reason code" in label or "mismatch" in label:
            assert structural_check(mutant) != [], label


@needs_jsonschema
def test_unknown_status_and_reason_code_fail_at_load(fixture_certificates, tmp_path):
    """A tampered file: the body hash is recomputed so that integrity passes and only the schema can object."""
    import hashlib
    from certify_ed.spec import canonical_json
    with open(fixture_certificates["lowest_spec"]) as f:
        body = json.load(f)
    side = os.path.join(os.path.dirname(fixture_certificates["lowest_spec"]), body["arrays"]["sidecar"])
    for label, mut in (("status", lambda b: b["statements"][1].__setitem__("status", "UNKNOWN_STATUS")),
                       ("reason", lambda b: b["scope"]["narrowed"][0].__setitem__("reason_code", "NOT_A_CODE"))):
        b = copy.deepcopy(body)
        mut(b)
        b.pop("integrity")
        digest = hashlib.sha256(canonical_json(b).encode()).hexdigest()
        b["integrity"] = {"algorithm": "sha256", "canonicalization": "certify-ed-canonical-json-1", "body_sha256": digest}
        p = tmp_path / f"tampered_{label}.json"
        p.write_text(canonical_json(b))
        import shutil
        shutil.copy(side, tmp_path / body["arrays"]["sidecar"])
        # the hash check alone cannot see it; verify_scope (fix/3.1.1) catches the reason code but not the
        # status, so the hashes pass and only the schema objects to both
        c_nov = Certificate.load(str(p), verify=False, validate_schema=False)
        rep = c_nov.verify_integrity()
        assert rep["body_hash_ok"] and rep["arrays_ok"]
        assert rep["ok"] == (label == "status")
        with pytest.raises(SchemaValidationError):
            Certificate.load(str(p))                                                       # the schema does
        with pytest.raises(SchemaValidationError):
            Certificate.load(str(p), validate_schema=True)


def test_load_without_jsonschema_falls_back_to_structural_checks(fixture_certificates, monkeypatch):
    import certify_ed.schema as cs
    monkeypatch.setattr(cs, "have_jsonschema", lambda: False)
    with open(fixture_certificates["lowest_spec"]) as f:
        body = json.load(f)
    assert validate_certificate_body(body) is None
    with pytest.raises(ImportError):
        validate_certificate_body(body, require=True)
    bad = copy.deepcopy(body)
    bad["scope"]["in_scope"].append({"id": "S1.model_fidelity", "status": "CERTIFIED"})
    with pytest.raises(SchemaValidationError):
        validate_certificate_body(bad)


# ------------------------------------------------------------------------------------ bundle schema
def test_bundle_meta_schema_accepts_written_bundles_and_rejects_bad_fields(tmp_path, heis8):
    spec, H, w, V = heis8
    d = str(tmp_path / "b")
    write_bundle(d, w[:2], V[:, :2], H, spec, fmt="npz", kind="lowest", meta={"temperatures": [0.5]})
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(d, "spec.json")) as f:
        spec_d = json.load(f)
    assert validate_bundle_meta(meta, spec_d) in (True, None)
    if have_jsonschema():
        for bad in ({**meta, "kind": "everything"}, {**meta, "kind": "window"}, {**meta, "temperatures": [-1.0]},
                    {**meta, "eigenvectors_layout": "diagonal"}):
            with pytest.raises(SchemaValidationError):
                validate_bundle_meta(bad, spec_d)
        with pytest.raises(SchemaValidationError):
            validate_bundle_meta(meta, {**spec_d, "schema": "other"})


# ------------------------------------------------------------------------------------ conformance
REFERENCE_BUNDLES = ["dense-arrays", "npz", "matrix-free"]


@pytest.mark.parametrize("name", REFERENCE_BUNDLES)
def test_reference_bundles_are_conformant(name, tmp_path):
    rep = check_bundle(os.path.join(REFERENCE, name), save=str(tmp_path / "cert"))
    assert rep["ok"], format_report(rep)
    assert [s["step"] for s in rep["steps"]] == ["layout", "meta_schema", "import", "certify", "certificate_schema",
                                                  "expected_statuses"]
    assert rep["statuses"]["S2.eigenpairs"] == "CERTIFIED"
    assert os.path.exists(rep["certificate"])
    if name == "matrix-free":
        with open(rep["certificate"]) as f:
            body = json.load(f)
        assert "operator" not in body["claim"]["producer"]["files"]     # no operator file was read
        assert body["subject"]["spec"] is not None


def test_reference_bundles_match_the_generator(tmp_path):
    """The committed reference bundles are what make_reference_bundles.py writes (meta, spec, expected, eigenvalues)."""
    sys.path.insert(0, os.path.join(ROOT, "conformance"))
    try:
        import make_reference_bundles as mk
    finally:
        sys.path.pop(0)
    made = mk.make(str(tmp_path))
    assert sorted(os.path.basename(p) for p in made) == sorted(REFERENCE_BUNDLES)
    from certify_ed.io import read_array
    for p in made:
        name = os.path.basename(p)
        ref = os.path.join(REFERENCE, name)
        assert sorted(os.listdir(ref)) == sorted(os.listdir(p)), name
        for fn in ("meta.json", "spec.json", "expected.json"):
            with open(os.path.join(ref, fn)) as f1, open(os.path.join(p, fn)) as f2:
                assert json.load(f1) == json.load(f2), (name, fn)
        ev_ref = read_array(os.path.join(ref, [f for f in os.listdir(ref) if f.startswith("eigenvalues")][0]))
        ev_new = read_array(os.path.join(p, [f for f in os.listdir(p) if f.startswith("eigenvalues")][0]))
        assert np.allclose(ev_ref, ev_new, rtol=0, atol=1e-12), name


def test_conformance_negative_controls(tmp_path, heis8):
    """A bundle whose eigenvalues are wrong is not conformant when expected.json pins CERTIFIED; a bundle with
    a bad meta.json fails the schema step; a directory without eigenvalues fails at layout."""
    spec, H, w, V = heis8
    d = str(tmp_path / "wrong")
    write_bundle(d, w[:3] + np.array([0.0, 0.2, 0.0]), V[:, :3], H, spec, fmt="npz", kind="lowest")
    with open(os.path.join(d, "expected.json"), "w") as f:
        json.dump({"S2.eigenpairs": "CERTIFIED"}, f)
    rep = check_bundle(d)
    assert rep["statuses"]["S2.eigenpairs"] == "FAILED" and not rep["ok"]
    assert rep["steps"][-1]["step"] == "expected_statuses" and not rep["steps"][-1]["ok"]
    assert rep["steps"][-2]["ok"]            # the certificate of a failed claim is still a valid certificate
    if have_jsonschema():
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        meta["kind"] = "nonsense"
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
        rep = check_bundle(d)
        assert not rep["ok"] and not [s for s in rep["steps"] if s["step"] == "meta_schema"][0]["ok"]
    e = str(tmp_path / "empty")
    os.makedirs(e)
    rep = check_bundle(e)
    assert not rep["ok"] and rep["steps"] == [{"step": "layout", "ok": False, "detail": rep["steps"][0]["detail"]}]
    assert "NOT CONFORMANT" in format_report(rep)


def test_conform_cli(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "conform", os.path.join(REFERENCE, "npz"),
                        "--json", str(out), "-o", str(tmp_path / "cert")], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CONFORMANT" in r.stdout and "NOT CONFORMANT" not in r.stdout
    with open(out) as f:
        assert json.load(f)["ok"] is True
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "conform", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "NOT CONFORMANT" in r.stdout


def test_schema_module_cli_checks_sync(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "certify_ed.schema", "--check", "schema"], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout
    r = subprocess.run([sys.executable, "-m", "certify_ed.schema", "--check", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "DIFFERS" in r.stdout
