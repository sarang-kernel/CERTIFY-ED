"""Conformance check of a producer bundle: schema, import, certify, and schema of the result.

`check_bundle(path)` runs, in order:

  1. the bundle layout (`certify_ed.io.bundle_files`: an eigenvalues.<ext> file exists),
  2. meta.json and spec.json against schema/producer-bundle-1.0.schema.json (when jsonschema is installed),
  3. the import (`read_bundle`: arrays read, shapes agree, no non-finite entries),
  4. `certify()` with the bundle's suggested scope (or the given options),
  5. the certificate body against schema/certificate-2.1.schema.json plus the structural checks,
  6. optionally, the statuses against `expected.json` in the bundle (`{"S1.model_fidelity": "CERTIFIED", ...}`),
     which is what the reference bundles under conformance/ carry.

The report says which step failed and why. A conformant bundle is one that reaches step 5 with a valid
certificate; whether its statements are CERTIFIED is a property of the data, not of conformance, except
where `expected.json` pins them. This module reads and reports; it does not compute a bound
(provenance.NON_COMPUTATIONAL).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

__all__ = ["check_bundle", "format_report"]


def _step(report: Dict[str, Any], name: str, ok: bool, detail: str = "", **extra) -> bool:
    rec = {"step": name, "ok": bool(ok), "detail": detail}
    rec.update(extra)
    report["steps"].append(rec)
    if not ok:
        report["ok"] = False
    return ok


def check_bundle(path: str, options=None, allow_code: bool = False, save: Optional[str] = None,
                 require_jsonschema: bool = False) -> Dict[str, Any]:
    """Run the conformance steps on a bundle directory; returns a report dict (see module docstring)."""
    from .certify import Options, certify
    from .io.bundle import bundle_files, read_bundle
    from .io.formats import FormatError
    from .schema import SchemaValidationError, have_jsonschema, validate_bundle_meta, validate_certificate_body

    report: Dict[str, Any] = {"bundle": os.path.abspath(path), "ok": True, "steps": [],
                              "jsonschema": have_jsonschema(), "statuses": {}, "certificate": None}
    # 1. layout
    try:
        files = bundle_files(path)
    except FormatError as e:
        _step(report, "layout", False, str(e))
        return report
    _step(report, "layout", True, ", ".join(os.path.basename(v) for v in files.values()),
          files={k: os.path.basename(v) for k, v in files.items()})
    # 2. meta / spec schema
    meta, spec_d = {}, None
    try:
        if "meta" in files:
            with open(files["meta"], encoding="utf-8") as f:
                meta = json.load(f)
        if "spec" in files:
            with open(files["spec"], encoding="utf-8") as f:
                spec_d = json.load(f)
    except (OSError, ValueError) as e:
        _step(report, "meta_schema", False, f"meta.json / spec.json is not valid JSON: {e}")
        return report
    try:
        r = validate_bundle_meta(meta, spec_d, require=require_jsonschema)
        _step(report, "meta_schema", True, "valid" if r else "not validated (jsonschema not installed)")
    except SchemaValidationError as e:
        _step(report, "meta_schema", False, str(e))
    except ImportError as e:
        _step(report, "meta_schema", False, str(e))
        return report
    # 3. import
    try:
        b = read_bundle(path, allow_code=allow_code)
    except (FormatError, ValueError, OSError) as e:
        _step(report, "import", False, f"{type(e).__name__}: {e}")
        return report
    claim = b.claim
    _step(report, "import", True,
          f"{claim.eigenvalues.size} eigenvalues, eigenvectors {'yes' if claim.eigenvectors is not None else 'no'}, "
          f"operator {'yes' if claim.operator is not None else 'no'}, spec {'yes' if b.spec is not None else 'no'}, "
          f"kind {claim.kind}")
    if b.spec is None and claim.operator is None:
        _step(report, "certify", False, "neither an operator file nor spec.json: nothing to certify against")
        return report
    # 4. certify
    if options is None:
        kw = {}
        for key in ("budget_seconds", "budget_mb"):
            if meta.get(key) is not None:
                kw[key] = float(meta[key])
        if meta.get("temperatures"):
            kw["temperatures"] = [float(t) for t in meta["temperatures"]]
        if meta.get("only"):
            kw["only"] = meta["only"] if isinstance(meta["only"], list) else [s.strip() for s in str(meta["only"]).split(",")]
        if meta.get("skip"):
            kw["skip"] = meta["skip"] if isinstance(meta["skip"], list) else [s.strip() for s in str(meta["skip"]).split(",")]
        options = Options.from_preset(meta["preset"], **kw) if meta.get("preset") else Options(**kw)
    t0 = time.time()
    try:
        cert = certify(claim, b.spec, options)
    except Exception as e:      # a conformance report must say what happened, whatever it was
        _step(report, "certify", False, f"{type(e).__name__}: {e}")
        return report
    report["statuses"] = {s["id"]: s["status"] for s in cert.statements}
    _step(report, "certify", True, ", ".join(f"{k.split('.')[0]} {v}" for k, v in report["statuses"].items()),
          seconds=round(time.time() - t0, 3))
    # 5. certificate schema (as saved)
    out = cert.save(save) if save else None
    body = cert.body if out else cert._finalize_body(None)
    report["certificate"] = out
    try:
        r = validate_certificate_body(body, require=require_jsonschema)
        _step(report, "certificate_schema", True, "valid" if r else "structure ok (jsonschema not installed)")
    except SchemaValidationError as e:
        _step(report, "certificate_schema", False, str(e))
    except ImportError as e:
        _step(report, "certificate_schema", False, str(e))
    # 6. expected statuses
    exp_path = os.path.join(path, "expected.json")
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            expected = json.load(f)
        diffs = [f"{k}: expected {v}, got {report['statuses'].get(k)}" for k, v in expected.items()
                 if report["statuses"].get(k) != v]
        _step(report, "expected_statuses", not diffs, "; ".join(diffs) if diffs else f"{len(expected)} statuses as expected")
    return report


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"bundle: {report['bundle']}"]
    for s in report["steps"]:
        lines.append(f"  [{'ok' if s['ok'] else 'FAIL':>4s}] {s['step']:20s} {s['detail']}")
    if report.get("certificate"):
        lines.append(f"  certificate: {report['certificate']}")
    if not report["jsonschema"]:
        lines.append("  note: jsonschema is not installed; schema steps ran the structural checks only "
                     "(pip install certify-ed[schema])")
    lines.append("CONFORMANT" if report["ok"] else "NOT CONFORMANT")
    return "\n".join(lines)
