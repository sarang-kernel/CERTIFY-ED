"""The independent C verifier (verifier/): build, agreement with certify-ed on a conformance fixture, and the
negative controls (a tampered status, a tampered hash, an unsound radius must each be caught).

Skipped when no C compiler is found (verifier/build.py: CEV_CC, then gcc/clang/cc on PATH); the full validation
on the corpus is verifier/validation/run_validation.py (verifier/validation/RESULTS.md records the counts).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verifier"))


@pytest.fixture(scope="module")
def cev_binary(tmp_path_factory):
    from build import build, find_cc
    if find_cc() is None:
        pytest.skip("no C compiler available for the verifier")
    out = str(tmp_path_factory.mktemp("cev") / ("cev.exe" if os.name == "nt" else "cev"))
    return build(None, out, quiet=True)


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    import certify_ed as ce
    from certify_ed.importers import claim_from_bundle
    bundle = os.path.join(ROOT, "conformance", "reference", "npz")
    claim, spec, _meta = claim_from_bundle(bundle)
    cert = ce.certify(claim, spec, ce.Options(store_eigenvectors=True))
    d = tmp_path_factory.mktemp("cert")
    path = cert.save(str(d / "npz_case"))
    return path, os.path.join(bundle, "operator.npz"), cert


def run(cev, cert, operator, out_json):
    r = subprocess.run([cev, cert, "--operator", operator, "--quiet", "--json", out_json], capture_output=True, text=True)
    summary = json.load(open(out_json)) if os.path.exists(out_json) else None
    return r.returncode, summary, r.stderr


def reseal(body):
    b = dict(body)
    b.pop("integrity", None)
    digest = hashlib.sha256(json.dumps(b, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    b["integrity"] = {"algorithm": "sha256", "canonicalization": "certify-ed-canonical-json-1", "body_sha256": digest}
    return b


def write(path, body):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def test_agrees_with_certify_ed_on_conformance_fixture(cev_binary, certificate, tmp_path):
    path, operator, cert = certificate
    assert {s["id"]: s["status"] for s in cert.statements}["S2.eigenpairs"] == "CERTIFIED"
    rc, summary, err = run(cev_binary, path, operator, str(tmp_path / "s.json"))
    assert rc == 0, (err, summary)
    assert summary["counts"]["fail"] == 0
    ids = {c["id"]: c["status"] for c in summary["checks"]}
    for key in ("H1.body_sha256", "H3.eigenvectors_sha256", "H4.spec_sha256", "H5.operator_sha256",
                "R2.residual_bound", "R3.existence_radius", "S2.status", "S3.inertia", "S3.ordering"):
        assert ids[key] == "PASS", (key, ids)


def test_catches_tampered_status(cev_binary, certificate, tmp_path):
    """Negative control: a FAILED S2 forged into CERTIFIED (statement and scope block, body resealed)."""
    import certify_ed as ce
    from certify_ed.importers import claim_from_bundle
    bundle = os.path.join(ROOT, "conformance", "reference", "npz")
    claim, spec, _ = claim_from_bundle(bundle)
    mu = np.array(claim.eigenvalues, dtype=float)
    mu[0] += 1e-5                       # a wrong eigenvalue: certify-ed FAILS S2
    bad = ce.Claim(mu, claim.eigenvectors, claim.operator, kind=claim.kind, accuracy_target=claim.accuracy_target)
    cert = ce.certify(bad, spec, ce.Options(store_eigenvectors=True))
    assert cert.status_of("S2.eigenpairs") == "FAILED"
    path = cert.save(str(tmp_path / "failed"))
    rc, summary, _ = run(cev_binary, path, os.path.join(bundle, "operator.npz"), str(tmp_path / "a.json"))
    assert rc == 0 and {c["id"]: c["status"] for c in summary["checks"]}["S2.failed"] == "REPRODUCED"
    body = json.load(open(path, encoding="utf-8"))
    for s in body["statements"]:
        if s["id"] == "S2.eigenpairs":
            s["status"] = "CERTIFIED"
    for e in body["scope"]["in_scope"]:
        if e["id"] == "S2.eigenpairs":
            e["status"] = "CERTIFIED"
    write(path, reseal(body))
    rc, summary, _ = run(cev_binary, path, os.path.join(bundle, "operator.npz"), str(tmp_path / "b.json"))
    assert rc == 1
    assert {c["id"]: c["status"] for c in summary["checks"]}["S2.status"] == "FAIL"


def test_catches_tampered_hash(cev_binary, certificate, tmp_path):
    path, operator, _ = certificate
    body = json.load(open(path, encoding="utf-8"))
    body["claim"]["description"] = body["claim"].get("description", "") + " (edited)"
    out = str(tmp_path / "h.json")
    write(out, body)                                   # body changed, hash not resealed
    with open(os.path.join(os.path.dirname(path), body["arrays"]["sidecar"]), "rb") as f:
        with open(os.path.join(tmp_path, body["arrays"]["sidecar"]), "wb") as g:
            g.write(f.read())
    rc, summary, _ = run(cev_binary, out, operator, str(tmp_path / "s.json"))
    assert rc == 1
    assert {c["id"]: c["status"] for c in summary["checks"]}["H1.body_sha256"] == "FAIL"


def test_catches_unsound_radius(cev_binary, certificate, tmp_path):
    """Negative control: a residual bound lowered below the exact residual, with every hash resealed, is caught
    only by the recomputation from the operator and the stored vectors (R2/R3)."""
    path, operator, _ = certificate
    body = json.load(open(path, encoding="utf-8"))
    side = os.path.join(os.path.dirname(path), body["arrays"]["sidecar"])
    arrays = {k: v for k, v in np.load(side).items()}
    i = int(np.argmax(arrays["residual_bound"]))
    rec = next(r for r in body["enclosures"] if r["index"] == i)
    for key in ("existence_radius", "residual_bound"):
        shift = float(arrays[key][i]) - rec[key]
        rec[key] = rec[key] / 1e4
        arrays[key][i] = rec[key] + max(shift, 0.0)
    from certify_ed.certificate_v2 import array_sha256
    for key, arr in arrays.items():
        body["arrays"]["sha256"][key] = array_sha256(arr)
    body["arrays"]["sidecar"] = "forged.arrays.npz"
    out = str(tmp_path / "forged.json")
    write(out, reseal(body))
    np.savez(str(tmp_path / "forged.arrays.npz"), **arrays)
    rc, summary, _ = run(cev_binary, out, operator, str(tmp_path / "s.json"))
    assert rc == 1
    ids = {c["id"]: c["status"] for c in summary["checks"]}
    assert ids["H1.body_sha256"] == "PASS" and ids["H2.residual_bound"] == "PASS"     # the forgery is consistent
    assert ids["R2.residual_bound"] == "FAIL" and ids["R3.existence_radius"] == "FAIL"


def test_canonical_json_matches_python(cev_binary, tmp_path):
    """The verifier's canonical JSON (shortest round-trip floats, sorted keys, escapes) equals json.dumps."""
    rng = np.random.default_rng(7)
    vals = list(rng.uniform(-10, 10, 200)) + list(np.ldexp(rng.integers(1, 2**53, 200), rng.integers(-1100, 900, 200)))
    vals += [1e16, 1e15, 0.0001, 0.00001, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.078769683837890625, -0.0, 1e22, 1e23]
    obj = {"v": [float(v) for v in vals if np.isfinite(v)], "i": [0, -1, 2**70], "s": ["a\"b\\c\n\t\u0001\u00e9\u4e2d\U0001f600 \x7f"]}
    src = str(tmp_path / "f.json")
    with open(src, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    dump = str(tmp_path / "canon.json")
    r = subprocess.run([cev_binary, src, "--quiet", "--dump-canonical", dump], capture_output=True, text=True)
    assert r.returncode in (0, 1)
    assert open(dump, "rb").read() == json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_npz_reader_handles_compressed_and_fortran(cev_binary, certificate, tmp_path):
    """savez_compressed members (deflate) and Fortran-ordered vectors hash identically to np.savez output."""
    path, operator, _ = certificate
    body = json.load(open(path, encoding="utf-8"))
    side = os.path.join(os.path.dirname(path), body["arrays"]["sidecar"])
    arrays = {k: v for k, v in np.load(side).items()}
    arrays["eigenvectors"] = np.asfortranarray(arrays["eigenvectors"])
    body["arrays"]["sidecar"] = "c.arrays.npz"
    out = str(tmp_path / "c.json")
    write(out, reseal(body))
    np.savez_compressed(str(tmp_path / "c.arrays.npz"), **arrays)
    with zipfile.ZipFile(str(tmp_path / "c.arrays.npz")) as z:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in z.infolist())
    rc, summary, err = run(cev_binary, out, operator, str(tmp_path / "s.json"))
    assert rc == 0, (err, summary)
    ids = {c["id"]: c["status"] for c in summary["checks"]}
    assert ids["H2.eigenvectors"] == "PASS" and ids["R2.residual_bound"] == "PASS"
