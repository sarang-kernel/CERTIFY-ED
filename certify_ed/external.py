"""Optional external verifiers (roadmap item N-04): run a configured program on a saved certificate and keep its
verdict in a record of its own.

An external verifier is any program that re-checks a saved certificate independently of certify-ed, for example
the exact checker `cecheck` of the Lean re-verification (driver `Verify/cecheck_bundle.py` of the repository
certify-ed-project) or the C verifier of `verifier/`. This module runs such a program and writes what it said into
an external-verification record (schema `certify-ed/external-verification/1`), a separate JSON file. Nothing here
reads that verdict back: the certificate file is not modified, and no status, bound or count of certify-ed depends
on the external program. The record says so in its field `status_effect`.

The command is a template. It is either a JSON list of arguments or a string split into arguments (POSIX shell
rules, except on Windows, where backslashes are kept and surrounding quotes are removed). Each argument may contain
the placeholders

    {certificate}   absolute path of the certificate JSON
    {sidecar}       absolute path of its array sidecar (from arrays.sidecar of the certificate)
    {bundle}        a producer bundle directory (given by the caller)
    {operator}      an operator file (given by the caller)
    {report}        a file the program may write its JSON report to (default: a file next to the record)

A placeholder without a value is an error before anything runs. When no argument contains {certificate}, the
certificate path is appended as the last argument. The program runs without a shell. Its exit code decides the
recorded verdict, by the convention of `cecheck` and of the C verifier: 0 is PASS (no check failed), 1 is FAIL (a
check failed), anything else is ERROR (the input could not be read or the program did not finish its work); a
program that exceeds the timeout is TIMEOUT. A JSON report written to {report} is hashed, and its fields `verdict`,
`level` and `statements`, when present, are copied into the record as the verifier's own words.

    from certify_ed.external import verify_external
    rec = verify_external("cert.json", "python cecheck_bundle.py {certificate} --bundle {bundle} --json {report}",
                          bundle="bundle_dir", record="cert.external.json")
    rec["verdict"]          # 'PASS' / 'FAIL' / 'ERROR' / 'TIMEOUT'

The environment variable CERTIFY_ED_EXTERNAL_VERIFIER supplies the command when none is given.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Union

SCHEMA = "certify-ed/external-verification/1"
ENV_VAR = "CERTIFY_ED_EXTERNAL_VERIFIER"
PLACEHOLDERS = ("certificate", "sidecar", "bundle", "operator", "report")
STATUS_EFFECT = ("none: the verdict of an external verifier is recorded only; no status, bound or count of the "
                 "certificate depends on it, and the certificate file is not modified")
_TAIL = 4000

__all__ = ["SCHEMA", "ENV_VAR", "STATUS_EFFECT", "split_command", "expand_command", "verdict_from_exit_code",
           "verify_external"]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_command(command: Union[str, Sequence[str]]) -> List[str]:
    """The argument list of a command template: a sequence is taken as it is, a string starting with '[' is read
    as a JSON list, any other string is split by shell rules (POSIX rules except on Windows, see the module
    text)."""
    if not isinstance(command, str):
        args = [str(a) for a in command]
    elif command.lstrip().startswith("["):
        args = json.loads(command)
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError("a JSON command must be a list of strings")
    elif os.name == "nt":
        args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a
                for a in shlex.split(command, posix=False)]
    else:
        args = shlex.split(command)
    if not args:
        raise ValueError("empty external verifier command")
    return args


def expand_command(args: Sequence[str], fields: Dict[str, Optional[str]]) -> List[str]:
    """Substitute the placeholders {certificate}, {sidecar}, {bundle}, {operator} and {report} in every argument.
    Raises ValueError for a placeholder whose value is None."""
    out = []
    for a in args:
        for name in PLACEHOLDERS:
            key = "{" + name + "}"
            if key in a:
                v = fields.get(name)
                if v is None:
                    raise ValueError(f"the external verifier command uses {key}, but no {name} was given")
                a = a.replace(key, v)
        out.append(a)
    return out


def verdict_from_exit_code(code: Optional[int]) -> str:
    """0 -> PASS, 1 -> FAIL, any other code (or no code) -> ERROR."""
    if code == 0:
        return "PASS"
    if code == 1:
        return "FAIL"
    return "ERROR"


def _recorded_statuses(body: Dict[str, Any]) -> Dict[str, str]:
    return {str(s.get("id")): str(s.get("status")) for s in (body.get("statements") or []) if isinstance(s, dict)}


def _program_info(prog: str) -> Dict[str, Any]:
    path = prog if os.path.isfile(prog) else shutil.which(prog)
    info: Dict[str, Any] = {"name": prog, "path": os.path.abspath(path) if path else None}
    info["sha256"] = _sha256_file(path) if path and os.path.isfile(path) else None
    return info


def _text(x) -> str:
    if x is None:
        return ""
    return x.decode(errors="replace") if isinstance(x, bytes) else x


def verify_external(certificate: str, command: Union[str, Sequence[str], None] = None, *,
                    bundle: Optional[str] = None, operator: Optional[str] = None, sidecar: Optional[str] = None,
                    report: Optional[str] = None, record: Optional[str] = None, timeout: Optional[float] = None,
                    cwd: Optional[str] = None) -> Dict[str, Any]:
    """Run an external verifier on the saved certificate `certificate` and return the external-verification
    record (a dict); the record is also written to `record` when that is given. `command` defaults to the
    environment variable CERTIFY_ED_EXTERNAL_VERIFIER. The certificate is read as JSON only to record its hashes
    and its statuses; it is not verified, modified or re-sealed here (`Certificate.load` does the integrity
    check)."""
    if command is None:
        command = os.environ.get(ENV_VAR)
        if not command:
            raise ValueError(f"no external verifier command given and {ENV_VAR} is not set")
    cert_path = os.path.abspath(certificate)
    with open(cert_path, "rb") as f:
        raw = f.read()
    body = json.loads(raw.decode("utf-8"))
    side = (body.get("arrays") or {}).get("sidecar")
    if sidecar is None and side:
        sidecar = os.path.join(os.path.dirname(cert_path), side)
    if report is None:
        stem = cert_path[:-5] if cert_path.endswith(".json") else cert_path
        base = record[:-5] if record and record.endswith(".json") else (record or stem + ".external")
        report = base + ".report.json"
    fields = {"certificate": cert_path, "sidecar": os.path.abspath(sidecar) if sidecar else None,
              "bundle": os.path.abspath(bundle) if bundle else None,
              "operator": os.path.abspath(operator) if operator else None, "report": os.path.abspath(report)}
    template = split_command(command)
    args = expand_command(template, fields)
    if not any("{certificate}" in a for a in template):
        args.append(cert_path)
    if os.path.exists(fields["report"]):
        os.remove(fields["report"])          # a stale report must not be taken for this run's
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    exit_code: Optional[int] = None
    stdout = stderr = ""
    error = None
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        exit_code, stdout, stderr = r.returncode, _text(r.stdout), _text(r.stderr)
        verdict = verdict_from_exit_code(exit_code)
    except subprocess.TimeoutExpired as e:
        verdict = "TIMEOUT"
        stdout, stderr = _text(e.stdout), _text(e.stderr)
        error = f"timeout after {timeout} s"
    except OSError as e:
        verdict = "ERROR"
        error = f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - t0
    rep_info = None
    if os.path.isfile(fields["report"]):
        rep_info = {"path": fields["report"], "sha256": _sha256_file(fields["report"])}
        try:
            with open(fields["report"], encoding="utf-8") as f:
                rep = json.load(f)
            if isinstance(rep, dict):
                for key in ("verdict", "level", "statements"):
                    if key in rep:
                        rep_info[key] = rep[key]
        except (ValueError, UnicodeDecodeError) as e:
            rep_info["error"] = f"report is not JSON: {e}"
    from . import __version__
    rec = {
        "schema": SCHEMA,
        "certify_ed_version": __version__,
        "certificate": {"path": cert_path, "file_sha256": hashlib.sha256(raw).hexdigest(),
                        "body_sha256": (body.get("integrity") or {}).get("body_sha256"),
                        "statuses": _recorded_statuses(body)},
        "verifier": {"command": args, "program": _program_info(args[0])},
        "run": {"started_utc": started, "seconds": round(seconds, 3), "exit_code": exit_code,
                "timeout_seconds": timeout, "error": error,
                "stdout_tail": stdout[-_TAIL:], "stderr_tail": stderr[-_TAIL:]},
        "verdict": verdict,
        "report": rep_info,
        "status_effect": STATUS_EFFECT,
    }
    if record:
        with open(record, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
            f.write("\n")
    return rec
