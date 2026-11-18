r"""Tests of Verify/cecheck_bundle.py (N-04).

    cd lean
    set PYTHONPATH=..\..\archive\archive_v5\code\certify-ed
    python -m pytest -q Verify\test_cecheck_bundle.py --basetemp work\verify\pytest

The export is compared byte for byte with export/export_certs.py; the mapping from cecheck's checks to statement
verdicts is tested on synthetic reports; the end-to-end tests run the built cecheck (skipped when
.lake/build/bin/cecheck is missing) on a certificate of the conformance fixture `npz` and on two forgeries of it,
which must be caught: a body changed after sealing (integrity FAIL) and a resealed existence radius that is too
small (S2 NOT_CONFIRMED).
"""
from __future__ import annotations

import copy
import filecmp
import hashlib
import json
import os
import sys

import pytest

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cecheck_bundle as cb  # noqa: E402

ce = pytest.importorskip("certify_ed", reason="certify_ed not on PYTHONPATH")
CODE = os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__)))
BUNDLE = os.path.join(CODE, "conformance", "reference", "npz")
HAVE_CECHECK = os.path.isfile(cb.EXE)


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    from certify_ed.certify import Options, certify
    from certify_ed.importers import claim_from_bundle
    claim, spec, _meta = claim_from_bundle(BUNDLE)
    c = certify(claim, spec, Options(store_eigenvectors=True))
    return c.save(str(tmp_path_factory.mktemp("cert") / "npz"))


def _reseal(body):
    from certify_ed.certificate_v2 import canonical_json
    b = dict(body)
    integ = dict(b.pop("integrity", {}))
    integ["body_sha256"] = hashlib.sha256(canonical_json(b).encode()).hexdigest()
    b["integrity"] = integ
    return canonical_json(b)


def _forge(cert, tmp_path, name, change, reseal):
    import shutil
    with open(cert, encoding="utf-8") as f:
        body = json.load(f)
    body = change(copy.deepcopy(body))
    side = body["arrays"]["sidecar"]
    shutil.copyfile(os.path.join(os.path.dirname(cert), side), tmp_path / side)
    p = tmp_path / (name + ".json")
    p.write_text(_reseal(body) if reseal else json.dumps(body), encoding="utf-8")
    return str(p)


def test_export_is_byte_identical_to_export_certs(cert, tmp_path):
    import export_certs as ec
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    ec.export_one("npz", cert, BUNDLE, str(a))
    cb.export(cert, str(b), bundle=BUNDLE)
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    for f in os.listdir(a):
        assert filecmp.cmp(a / f, b / f, shallow=False), f
    cb.export(cert, str(c), operator=os.path.join(BUNDLE, "operator.npz"))
    for f in os.listdir(a):
        if f != "manifest.json":
            assert filecmp.cmp(a / f, c / f, shallow=False), f


def _body(statuses):
    return {"statements": [{"id": k, "status": v} for k, v in statuses.items()]}


def _rep(**checks):
    return {"checks": [{"id": k.replace("__", "."), "status": v, "detail": ""} for k, v in checks.items()]}


GOOD = dict(H1__body_sha256="PASS", H5__operator_sha256="PASS", C3__enclosure_records="PASS",
            E1__residual_bound="PASS", E2__existence_radius="PASS", E5__disjoint="PASS")


def test_mapping_certified_statuses():
    body = _body({"S1.model_fidelity": "CERTIFIED", "S2.eigenpairs": "CERTIFIED", "S3.completeness": "CERTIFIED",
                  "S4.eigenspaces": "CERTIFIED", "S8.degeneracy": "CERTIFIED"})
    ok = _rep(**GOOD, S2__status__exact="PASS", S2__status__rule="PASS", S3__status__exact="PASS",
              S3__status__rule="PASS", S8__C0__exact="PASS", K0__kato_temple="PASS")
    v = cb.statement_verdicts(body, ok)
    assert v["integrity"]["verdict"] == "PASS"
    assert [v[s]["verdict"] for s in ("S2.eigenpairs", "S3.completeness", "S8.degeneracy")] == ["CONFIRMED"] * 3
    assert v["S1.model_fidelity"]["verdict"] == v["S4.eigenspaces"]["verdict"] == "NOT_RECHECKED"
    assert v["kato_temple"]["verdict"] == "CONFIRMED"
    assert all(x["level"] == "N" for x in v.values())
    # one failing check of a statement, or of the integrity group, withdraws the confirmation
    for bad, hit in ((dict(E2__existence_radius="FAIL"), "S2.eigenpairs"), (dict(S3__status__exact="FAIL"),
                     "S3.completeness"), (dict(S8__C0__exact="FAIL"), "S8.degeneracy"),
                     (dict(S2__status__rule="FAIL"), "S2.eigenpairs")):
        checks = {**GOOD, "S2__status__exact": "PASS", "S2__status__rule": "PASS", "S3__status__exact": "PASS",
                  "S3__status__rule": "PASS", "S8__C0__exact": "PASS", **bad}
        assert cb.statement_verdicts(body, _rep(**checks))[hit]["verdict"] == "NOT_CONFIRMED"
    tampered = _rep(**{**GOOD, "H1__body_sha256": "FAIL"}, S2__status__exact="PASS", S2__status__rule="PASS",
                    S3__status__exact="PASS", S8__C0__exact="PASS")
    v = cb.statement_verdicts(body, tampered)
    assert v["integrity"]["verdict"] == "FAIL"
    assert {v[s]["verdict"] for s in ("S2.eigenpairs", "S3.completeness", "S8.degeneracy")} == {"NOT_CONFIRMED"}
    # a CERTIFIED S8 with no cluster re-counted is not confirmed
    assert cb.statement_verdicts(body, _rep(**GOOD))["S8.degeneracy"]["verdict"] == "NOT_CONFIRMED"


def test_mapping_failed_and_other_statuses():
    body = _body({"S2.eigenpairs": "FAILED", "S3.completeness": "FAILED"})
    v = cb.statement_verdicts(body, _rep(**GOOD, S2__failed="REPRODUCED", S2__failed__refuted="INFO",
                                         S3__failed="REFUTED"))
    assert v["S2.eigenpairs"]["verdict"] == "REPRODUCED" and v["S3.completeness"]["verdict"] == "REFUTED"
    v = cb.statement_verdicts(body, _rep(**GOOD, S2__failed="REPRODUCED", S2__failed__refuted="REFUTED",
                                         S3__failed="NOT_REPRODUCED"))
    assert v["S2.eigenpairs"]["verdict"] == "REFUTED" and v["S3.completeness"]["verdict"] == "NOT_REPRODUCED"
    v = cb.statement_verdicts(body, _rep(**GOOD, S2__failed="NOT_REPRODUCED"))
    assert v["S2.eigenpairs"]["verdict"] == "NOT_REPRODUCED"
    body = _body({"S2.eigenpairs": "INCONCLUSIVE", "S3.completeness": "NOT_CHECKED", "S8.degeneracy": "INCONCLUSIVE"})
    v = cb.statement_verdicts(body, _rep(**GOOD, S2__status="INFO"))
    assert {v[s]["verdict"] for s in ("S2.eigenpairs", "S3.completeness", "S8.degeneracy")} == {"NO_CLAIM"}
    assert v["S2.eigenpairs"]["exact_radius_checks_hold"] is True


def test_missing_cecheck_is_an_error(cert, tmp_path):
    code, res = cb.run(cert, bundle=BUNDLE, exe=str(tmp_path / "no_cecheck.exe"), quiet=True)
    assert code == 2 and res["verdict"] == "ERROR" and "statements" not in res


@pytest.mark.skipif(not HAVE_CECHECK, reason="cecheck not built (lake build cecheck)")
def test_end_to_end_genuine_and_forgeries(cert, tmp_path):
    code, res = cb.run(cert, bundle=BUNDLE, quiet=True)
    assert code == 0 and res["verdict"] == "PASS"
    st = {k: v["verdict"] for k, v in res["statements"].items()}
    assert st["integrity"] == "PASS" and st["S2.eigenpairs"] == "CONFIRMED" and st["S3.completeness"] == "CONFIRMED"
    assert res["cecheck"]["sha256"] and res["cecheck"]["report_sha256"]

    def tamper(b):
        b["claim"]["description"] = "changed after sealing"
        return b
    code, res = cb.run(_forge(cert, tmp_path, "tamper", tamper, reseal=False), bundle=BUNDLE, quiet=True)
    assert code == 1 and res["statements"]["integrity"]["verdict"] == "FAIL"

    def radius(b):
        b["enclosures"][0]["existence_radius"] *= 1e-6
        return b
    code, res = cb.run(_forge(cert, tmp_path, "radius", radius, reseal=True), bundle=BUNDLE, quiet=True)
    assert code == 1 and res["statements"]["integrity"]["verdict"] == "PASS"
    assert res["statements"]["S2.eigenpairs"]["verdict"] == "NOT_CONFIRMED"
