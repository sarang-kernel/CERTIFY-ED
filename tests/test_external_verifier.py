"""The optional external-verifier path (certify_ed.external, `certify-ed verify --external`, roadmap item N-04).

An external verifier is run on a saved certificate and its verdict is written to a separate record. These tests use
stub programs (small Python scripts written into tmp_path) in place of a real verifier, and check that

* the exit-code convention maps 0 / 1 / other / timeout / missing program to PASS / FAIL / ERROR / TIMEOUT / ERROR;
* the verifier's report is hashed and its verdict, level and statements are copied, as the verifier's words;
* the certificate file is byte-for-byte unchanged and its statuses are the same after every outcome, including FAIL;
* a placeholder without a value stops the run before the program starts;
* the CLI returns 0 / 2 / 1 for PASS / not PASS / failed integrity, and runs no verifier when integrity fails;
* no module that computes a status imports certify_ed.external (the verdict cannot flow back).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUB = """
import json, sys, time
mode, cert, report = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.load(open(cert, encoding="utf-8"))
if len(sys.argv) > 4:
    open(sys.argv[4], "w").write("ran")
if mode == "sleep":
    time.sleep(30)
st = {s["id"]: {"recorded": s["status"], "verdict": "CONFIRMED" if mode == "pass" else "NOT_CONFIRMED"}
      for s in body["statements"]}
json.dump({"verdict": "PASS" if mode == "pass" else "FAIL", "level": "N", "statements": st}, open(report, "w"))
print("stub verifier", mode)
sys.exit({"pass": 0, "fail": 1, "error": 3}.get(mode, 0))
"""


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    import certify_ed as ce
    rng = np.random.default_rng(7)
    M = rng.standard_normal((8, 8))
    H = (M + M.T) / 2
    w, V = np.linalg.eigh(H)
    cert = ce.quick(H, w[:2], V[:, :2], kind="lowest", store_eigenvectors=True)
    d = tmp_path_factory.mktemp("ext")
    path = cert.save(str(d / "case"))
    stub = d / "stub.py"
    stub.write_text(STUB, encoding="utf-8")
    return path, str(stub)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _statuses(path):
    from certify_ed.certificate_v2 import Certificate
    return {s["id"]: s["status"] for s in Certificate.load(path, verify=True).statements}


def _cmd(stub, mode, *extra):
    return json.dumps([sys.executable, stub, mode, "{certificate}", "{report}", *extra])


@pytest.mark.parametrize("mode,verdict", [("pass", "PASS"), ("fail", "FAIL"), ("error", "ERROR")])
def test_verdict_follows_the_exit_code_and_changes_nothing(saved, tmp_path, mode, verdict):
    from certify_ed.external import SCHEMA, STATUS_EFFECT, verify_external
    path, stub = saved
    before_sha, before = _sha(path), _statuses(path)
    rec = verify_external(path, _cmd(stub, mode), record=str(tmp_path / "rec.json"), timeout=60)
    assert rec["verdict"] == verdict
    assert rec["schema"] == SCHEMA and rec["status_effect"] == STATUS_EFFECT
    assert rec["run"]["exit_code"] == {"pass": 0, "fail": 1, "error": 3}[mode]
    assert "stub verifier" in rec["run"]["stdout_tail"]
    # the report is the verifier's own words, hashed
    rep = rec["report"]
    assert rep["sha256"] == _sha(rep["path"]) and rep["level"] == "N"
    assert set(rep["statements"]) == set(before)
    # the certificate is untouched and its statuses are the same
    assert _sha(path) == before_sha
    assert _statuses(path) == before
    assert rec["certificate"]["file_sha256"] == before_sha
    assert rec["certificate"]["statuses"] == before
    with open(path, encoding="utf-8") as f:
        assert rec["certificate"]["body_sha256"] == json.load(f)["integrity"]["body_sha256"]
    # the record on disk is the returned record
    with open(tmp_path / "rec.json", encoding="utf-8") as f:
        assert json.load(f) == rec


def test_timeout_and_missing_program(saved, tmp_path):
    from certify_ed.external import verify_external
    path, stub = saved
    rec = verify_external(path, _cmd(stub, "sleep"), timeout=2)
    assert rec["verdict"] == "TIMEOUT" and rec["run"]["exit_code"] is None
    missing = json.dumps([str(tmp_path / "no_such_verifier.exe"), "{certificate}"])
    rec = verify_external(path, missing, timeout=10)
    assert rec["verdict"] == "ERROR" and rec["run"]["error"]
    assert rec["report"] is None


def test_unset_placeholder_stops_before_running(saved, tmp_path):
    from certify_ed.external import verify_external
    path, stub = saved
    marker = tmp_path / "marker.txt"
    cmd = json.dumps([sys.executable, stub, "pass", "{certificate}", "{report}", str(marker), "{bundle}"])
    with pytest.raises(ValueError, match="bundle"):
        verify_external(path, cmd)
    assert not marker.exists()
    rec = verify_external(path, cmd, bundle=str(tmp_path))
    assert rec["verdict"] == "PASS" and marker.exists()
    assert os.path.abspath(str(tmp_path)) in rec["verifier"]["command"]


def test_stale_report_is_removed(saved, tmp_path):
    from certify_ed.external import verify_external
    path, stub = saved
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    cmd = json.dumps([sys.executable, "-c", "import sys; sys.exit(1)", "{report}", "{certificate}"])
    rec = verify_external(path, cmd, report=str(report))
    assert rec["verdict"] == "FAIL" and rec["report"] is None and not report.exists()


def test_certificate_appended_and_string_templates():
    from certify_ed.external import expand_command, split_command
    assert split_command('["a b", "{certificate}"]') == ["a b", "{certificate}"]
    if os.name == "nt":
        assert split_command(r'"C:\Program Files\v.exe" --x C:\d\{report}') == [r"C:\Program Files\v.exe", "--x",
                                                                                 r"C:\d\{report}"]
    else:
        assert split_command("'/opt/my v' --x {report}") == ["/opt/my v", "--x", "{report}"]
    assert expand_command(["v", "{certificate}", "--o={report}"], {"certificate": "c", "report": "r"}) == \
        ["v", "c", "--o=r"]
    with pytest.raises(ValueError):
        split_command("")


def test_cli_exit_codes_and_env_var(saved, tmp_path, monkeypatch, capsys):
    from certify_ed.cli import main
    from certify_ed.external import ENV_VAR
    path, stub = saved
    before = _statuses(path)
    rec = str(tmp_path / "cli_rec.json")
    assert main(["verify", path]) == 0
    assert main(["verify", path, "--external", _cmd(stub, "pass"), "--record", rec]) == 0
    with open(rec, encoding="utf-8") as f:
        assert json.load(f)["verdict"] == "PASS"
    assert main(["verify", path, "--external", _cmd(stub, "fail"), "--record", rec]) == 2
    monkeypatch.setenv(ENV_VAR, _cmd(stub, "error"))
    assert main(["verify", path, "--record", rec]) == 2
    with open(rec, encoding="utf-8") as f:
        assert json.load(f)["verdict"] == "ERROR"
    out = capsys.readouterr().out
    assert "statuses unchanged" in out
    assert _statuses(path) == before


def test_cli_integrity_failure_runs_no_verifier(saved, tmp_path):
    from certify_ed.cli import main
    path, stub = saved
    with open(path, encoding="utf-8") as f:
        body = json.load(f)
    body["claim"]["description"] = "tampered after sealing"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body), encoding="utf-8")
    side = body["arrays"]["sidecar"]
    with open(os.path.join(os.path.dirname(path), side), "rb") as f:
        (tmp_path / side).write_bytes(f.read())
    marker = tmp_path / "marker.txt"
    cmd = json.dumps([sys.executable, stub, "pass", "{certificate}", "{report}", str(marker)])
    assert main(["verify", str(bad), "--external", cmd]) == 1
    assert not marker.exists()


def test_no_status_code_imports_the_external_module():
    from certify_ed.provenance import NON_COMPUTATIONAL
    assert "certify_ed/external.py" in NON_COMPUTATIONAL
    pkg = os.path.join(ROOT, "certify_ed")
    users = []
    for dirpath, _dirs, files in os.walk(pkg):
        for fn in files:
            if fn.endswith(".py") and not (fn == "external.py" and os.path.samefile(dirpath, pkg)):
                p = os.path.join(dirpath, fn)
                with open(p, encoding="utf-8") as f:
                    if re.search(r"^\s*(from\s+\S*external\s+import|import\s+\S*external)", f.read(), re.M):
                        users.append(os.path.relpath(p, ROOT).replace(os.sep, "/"))
    assert sorted(users) == ["certify_ed/api.py", "certify_ed/cli.py"]
