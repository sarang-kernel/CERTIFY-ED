"""Offline HTML view of a certificate: one self-contained file, readable without a network connection,
a JavaScript engine or the package.

`render_html(cert, integrity, schema_result)` writes a single HTML document (inline CSS, no external
resources, no scripts; sections fold with `<details>`) from a Certificate: a header (model, claim kind
and k, package version, hashes, run environment, floating-point probe), the statements in plain words
(status, summary, next action, key evidence), the scope block (in scope / narrowed by the user with the
reason / out of scope with the reason code and its explanation from docs/SCOPE_MATRIX.md), the enclosure
table (one row per claimed pair: claimed value, radius, existence radius, enclosure interval, cluster),
the Kato-Temple intervals, the completeness (gap and inertia count) data, the cluster data of S4 and S8,
the thermodynamic intervals of S7, the exact re-check data of S6, the methods paragraph of
`certify_ed.report`, and a footer with the integrity report (body hash, array hashes, scope block,
schema validation).

Every number shown is a value of the certificate, formatted by `report.round_decimal` in the direction
that cannot overstate the result: radii, residual bounds, error bounds and upper ends of enclosures are
rounded up; gap bounds, separation bounds and lower ends of enclosures are rounded down; certified
windows and isolation intervals are rounded inward; claimed values, shifts, counts and estimates are
printed exactly (`repr`). Each number carries the exact value in a `data-exact` attribute, which is what
tests/test_viewer.py checks against the certificate.

`view_file(path, out)` loads a certificate JSON (and its arrays sidecar when present) WITHOUT raising on
a bad hash, runs `verify_integrity`, `verify_scope` and the schema validation, and renders; a failed
check becomes a banner at the top of the page. `certify-ed view CERT.json -o CERT.html` is the command.

This module reads a certificate and writes text; it cannot change a number in one
(provenance.NON_COMPUTATIONAL).
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .certificate_v2 import STATEMENT_IDS, Certificate
from .report import methods_paragraph, round_decimal
from .scope import REASON_CODES
from .spec import canonical_json

__all__ = ["render_html", "view_file", "integrity_report", "schema_report", "STATEMENT_TITLES", "STATUS_WORDS"]

STATEMENT_TITLES = {
    "S1.model_fidelity": "Model fidelity: the producer's operator is the declared model",
    "S2.eigenpairs": "Eigenpairs: each claimed value lies within a certified radius of a distinct eigenvalue",
    "S3.completeness": "Completeness: no eigenvalue is missing from the claimed set",
    "S4.eigenspaces": "Eigenspaces: the claimed vectors span the eigenspaces of the clusters",
    "S5.corroboration": "Corroboration: independent solvers agree (evidence, not a certified statement)",
    "S6.exact_recheck": "Exact re-check: the enclosures re-derived in exact integer arithmetic",
    "S7.thermodynamics": "Thermodynamics: intervals for Z, F, U, C from a complete spectrum",
    "S8.degeneracy": "Degeneracy: the number of eigenvalues in each cluster's level enclosure",
}
STATUS_WORDS = {"CERTIFIED": "certified", "FAILED": "refuted", "INCONCLUSIVE": "undecided",
                "NOT_CHECKED": "not checked", "CORROBORATED": "corroborated"}
_STATUS_EXPLAIN = {
    "CERTIFIED": "the statement holds under the hypotheses listed in the certificate",
    "FAILED": "the statement is refuted by the check",
    "INCONCLUSIVE": "the check ran but could not decide; the next action says what would decide it",
    "NOT_CHECKED": "the check was not run; the scope block says why",
    "CORROBORATED": "an independent recomputation agrees; this is evidence, not a certified statement",
}

# rounding direction by field name: 'up' may not understate an error, 'down' may not overstate a margin
_UP = {"radius", "max_radius", "median_radius", "max_existence_radius", "existence_radius", "residual_bound",
       "residual_computed", "model_shift", "model_shift_bound", "sector_lift_shift", "hermiticity_max_abs", "eta",
       "rho", "sin_theta_bound", "projector_error_bound", "splitting", "width", "residual_width", "beta",
       "beta_effective", "beta_a", "beta_b", "unit_roundoff", "norm_upper_bound", "upper", "delta", "estimate_mb",
       "seconds", "total_seconds", "bound", "eps", "eps_star", "max_abs_diff", "relative_residual"}
_DOWN = {"lower", "gap_lower_bound", "next_eigenvalue_lower_bound", "separation_lower_bound", "next_lower_bound",
         "complete_below"}
_OUTWARD = {"level_enclosure", "bracket_inner", "bracket_outer", "interval", "window_enclosure"}
_INWARD = {"certified_window", "isolation_interval"}


# ------------------------------------------------------------------------------------------------- numbers
def _esc(x: Any) -> str:
    return _html.escape(str(x), quote=True)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _special(x: Any) -> Optional[str]:
    """'inf' / '-inf' / 'nan' as written by Certificate._jsonable, or a non-finite float."""
    if isinstance(x, str) and x in ("inf", "-inf", "nan"):
        return x
    if _is_number(x):
        xf = float(x)
        if xf != xf:
            return "nan"
        if xf in (float("inf"), float("-inf")):
            return "inf" if xf > 0 else "-inf"
    return None


def _num(x: Any, mode: str = "exact", sig: int = 4, tag: str = "span") -> str:
    """One number of the certificate as an HTML element: displayed text rounded in `mode` ('up', 'down',
    'exact'), the exact value in data-exact. Non-numeric input is escaped as text."""
    sp_ = _special(x)
    if sp_ is not None:
        return f'<{tag} class="num exact" data-exact="{sp_}">{sp_}</{tag}>'
    if not _is_number(x):
        return f"<{tag}>{_esc(x)}</{tag}>" if x is not None else f"<{tag}></{tag}>"
    if isinstance(x, (int, np.integer)):
        return f'<{tag} class="num exact" data-exact="{int(x)}">{int(x)}</{tag}>'
    xf = float(x)
    exact = repr(xf)
    text = exact if mode == "exact" else round_decimal(xf, sig, mode)
    return f'<{tag} class="num {mode}" data-exact="{exact}">{text}</{tag}>'


def _cell(x: Any, mode: str = "exact", sig: int = 4) -> str:
    return _num(x, mode, sig, tag="td")


def _interval(iv: Any, how: str = "outward", sig: int = 6) -> str:
    """[lo, hi] rounded outward (enclosures) or inward (certified windows, isolation intervals)."""
    if not isinstance(iv, (list, tuple)) or len(iv) != 2:
        return _esc(iv)
    lo_mode, hi_mode = ("down", "up") if how == "outward" else ("up", "down")
    return f"[{_num(iv[0], lo_mode, sig)}, {_num(iv[1], hi_mode, sig)}]"


def _mode_for(key: str) -> str:
    if key in _UP:
        return "up"
    if key in _DOWN:
        return "down"
    return "exact"


def _value(key: str, v: Any, sig: int = 4) -> str:
    """Generic rendering of an evidence value: numbers by the direction table, two-element numeric lists as
    intervals by the direction table, short lists inline, dicts and long lists as folded JSON."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "-"
    if _is_number(v) or _special(v) is not None:
        return _num(v, _mode_for(key), sig)
    if isinstance(v, str):
        return _esc(v)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_number(a) or _special(a) for a in v) and (
            key in _OUTWARD or key in _INWARD):
        return _interval(v, "inward" if key in _INWARD else "outward")
    if isinstance(v, (list, tuple)) and len(v) <= 12 and all(_is_number(a) or isinstance(a, str) for a in v):
        return "[" + ", ".join(_num(a, _mode_for(key), sig) if _is_number(a) else _esc(a) for a in v) + "]"
    return _json_fold(v)


def _json_fold(v: Any, summary: str = "details (JSON)") -> str:
    try:
        txt = json.dumps(v, indent=1, sort_keys=True, default=str)
    except (TypeError, ValueError):
        txt = str(v)
    return f"<details><summary>{_esc(summary)}</summary><pre>{_esc(txt)}</pre></details>"


def _table(headers: Sequence[str], rows: List[str], cls: str = "") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    return (f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def _kv(rows: List[Tuple[str, str]]) -> str:
    return '<table class="kv">' + "".join(f"<tr><th>{_esc(k)}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def _details(title: str, inner: str, open_: bool = True, anchor: str = "") -> str:
    aid = f' id="{_esc(anchor)}"' if anchor else ""
    return (f'<details{aid}{" open" if open_ else ""}><summary><h2>{_esc(title)}</h2></summary>'
            f'<div class="body">{inner}</div></details>')


# ------------------------------------------------------------------------------------------- the checks
def integrity_report(cert: Certificate) -> Dict[str, Any]:
    """verify_integrity plus the stored and recomputed body digests (so a mismatch can be shown)."""
    rep = dict(cert.verify_integrity())
    body = dict(cert.body)
    integ = body.pop("integrity", {}) or {}
    rep["stored_body_sha256"] = integ.get("body_sha256")
    rep["recomputed_body_sha256"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    declared = set((cert.body.get("arrays") or {}).get("sha256", {}).keys())
    rep["arrays_declared"] = sorted(declared)
    rep["arrays_loaded"] = sorted(cert.arrays.keys())
    rep["arrays_missing"] = sorted(declared - set(cert.arrays.keys()))
    return rep


def schema_report(cert: Certificate) -> Dict[str, Any]:
    """{'ok': True | None | False, 'detail': str}: True validated with jsonschema, None structural checks only
    (jsonschema not installed), False with the validator's message."""
    try:
        from .schema import SchemaValidationError, validate_certificate_body
    except Exception as e:                       # pragma: no cover - the schema module is part of the package
        return {"ok": None, "detail": f"schema module unavailable: {e}"}
    try:
        r = validate_certificate_body(cert.body, require=False)
    except SchemaValidationError as e:
        return {"ok": False, "detail": str(e)}
    except Exception as e:                       # a malformed body must still render
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    if r:
        return {"ok": True, "detail": "the body validates against schema/certificate-2.1.schema.json "
                                      "(jsonschema) and passes the structural checks"}
    return {"ok": None, "detail": "structural checks passed; the JSON Schema itself was not applied "
                                  "(the optional jsonschema package is not installed)"}


# ---------------------------------------------------------------------------------------------- sections
def _banner(integ: Optional[dict], schema: Optional[dict]) -> str:
    out = []
    if integ is not None:
        if integ.get("ok"):
            out.append('<div class="banner ok">INTEGRITY VERIFIED: the body hash, the array hashes and the scope '
                       'block of this certificate check out.</div>')
        else:
            why = []
            if not integ.get("body_hash_ok"):
                why.append("the body hash does not match the content (the JSON was modified after it was saved)")
            if not integ.get("arrays_ok"):
                bad = ", ".join(integ.get("bad_arrays") or []) or "arrays"
                why.append(f"array hash mismatch or missing array: {bad}")
            if integ.get("scope_ok") is False:
                why.append("scope block inconsistent: " + "; ".join(integ.get("scope_problems") or []))
            out.append('<div class="banner fail">INTEGRITY CHECK FAILED: ' + _esc("; ".join(why) or "see footer")
                       + ". Do not rely on any number on this page.</div>")
    if schema is not None and schema.get("ok") is False:
        out.append('<div class="banner warn">SCHEMA VALIDATION FAILED: ' + _esc(schema.get("detail", "")) + "</div>")
    return "".join(out)


def _header(cert: Certificate) -> str:
    body = cert.body
    sub = body.get("subject") or {}
    spec = sub.get("spec") or {}
    claim = body.get("claim") or {}
    env = body.get("environment") or {}
    op = sub.get("operator") or {}
    rows: List[Tuple[str, str]] = []
    if spec:
        params = spec.get("parameters") or {}
        ptxt = ", ".join(f"{_esc(k)} = {_num(v) if _is_number(v) else _esc(v)}" for k, v in sorted(params.items())
                         if v is not None and not isinstance(v, bool))
        local = spec.get("local_space") or {}
        ltxt = ""
        if local.get("kind") == "spin" and local.get("spin"):
            ltxt = f", spin-{_esc(local['spin'])} sites"
        rows.append(("Model", f"{_esc(spec.get('name', 'declared model'))} with {_num(spec.get('n_sites'))} sites"
                     f"{ltxt}" + (f" ({ptxt})" if ptxt else "")))
    else:
        rows.append(("Model", _esc(sub.get("description") or "no model specification declared")))
    if op:
        rows.append(("Operator", f"kind {_esc(op.get('kind', '?'))}, dimension {_num(op.get('dimension'))}"
                     + (f", norm bound {_num(op.get('norm_upper_bound'), 'up')}" if op.get("norm_upper_bound") is not None
                        else "")))
    if sub.get("sector"):
        rows.append(("Sector", _json_fold(sub["sector"], "sector block")))
    kind = claim.get("kind", "?")
    ktxt = f"{_esc(kind)}, k = {_num(claim.get('n_eigenvalues'))}"
    if claim.get("window") is not None:
        ktxt += f", window {_interval(claim['window'], 'outward', 10)}"
    ktxt += ", with eigenvectors" if claim.get("has_eigenvectors") else ", eigenvalues only"
    if claim.get("accuracy_target") is not None:
        ktxt += f", accuracy target {_num(claim['accuracy_target'])}"
    rows.append(("Claim", ktxt))
    if claim.get("description"):
        rows.append(("Claim description", _esc(claim["description"])))
    prod = claim.get("producer") or {}
    if prod:
        rows.append(("Producer", _esc(prod.get("name", "")) + (" " + _json_fold(prod, "producer record") if len(prod) > 1 else "")))
    rows.append(("CERTIFY-ED version", _esc(env.get("certify_ed") or "unknown")))
    rows.append(("Schema", _esc(body.get("schema") or "")))
    rows.append(("Created (UTC)", _esc(env.get("created_utc") or "")))
    hashes = [("body SHA-256", (body.get("integrity") or {}).get("body_sha256")),
              ("spec SHA-256", sub.get("spec_sha256")), ("operator SHA-256", op.get("matrix_sha256")),
              ("eigenvectors SHA-256", claim.get("eigenvectors_sha256")),
              ("computational source hash", env.get("source_hash")), ("git commit", env.get("git_commit"))]
    rows.append(("Hashes", "<br>".join(f"{_esc(k)}: <code>{_esc(v)}</code>" for k, v in hashes if v)))
    if env.get("source_dirty"):
        rows.append(("Source state", "uncommitted changes in the computational source at run time"))
    stack = [f"{lab} {_esc(env[key])}" for key, lab in (("python", "Python"), ("numpy", "NumPy"), ("scipy", "SciPy"),
                                                       ("python_flint", "python-flint"), ("flint_c_library", "FLINT"))
             if env.get(key)]
    blas = env.get("blas")
    if isinstance(blas, list) and blas:
        b0 = blas[0]
        stack.append("BLAS " + _esc(" ".join(str(b0.get(k)) for k in ("internal_api", "version") if b0.get(k)))
                     + (f" ({_esc(b0.get('num_threads'))} threads)" if b0.get("num_threads") else ""))
    elif isinstance(blas, dict) and blas.get("name"):
        stack.append(f"BLAS {_esc(blas.get('name'))} {_esc(blas.get('version', ''))}")
    if env.get("platform"):
        stack.append(_esc(env["platform"]))
    rows.append(("Environment", ", ".join(stack) if stack else "not recorded"))
    secs = (body.get("timing") or {}).get("total_seconds")
    if secs is not None:
        rows.append(("Certification time", f"{_num(secs, 'up', 2)} s"))
    ar = body.get("arithmetic") or {}
    probe = ar.get("runtime_probe") or {}
    held = probe.get("standard_model_hypotheses_hold")
    if held is True:
        ptxt = '<span class="ok">passed</span>'
    elif held is False:
        ptxt = ('<span class="bad">FAILED</span>: ' + _esc(", ".join(probe.get("failed", [])))
                + " (no floating-point bound is certified)")
    else:
        ptxt = "not recorded"
    checks = probe.get("checks") or {}
    if checks:
        ptxt += " " + _json_fold(checks, f"{len(checks)} probe checks")
    rows.append(("Floating-point probe", ptxt))
    if ar.get("unit_roundoff") is not None:
        rows.append(("Unit roundoff", _num(ar["unit_roundoff"], "up")))
    uc = ar.get("underflow_check")
    if isinstance(uc, dict):
        if uc.get("performed"):
            n = len(uc.get("findings") or [])
            utxt = ('<span class="ok">performed</span>, no finding' if n == 0 else
                    f'<span class="bad">performed, {n} finding(s)</span> ' + _json_fold(uc["findings"], "findings"))
            if uc.get("threshold") is not None:
                utxt += f"; threshold {_num(uc['threshold'])}"
        else:
            utxt = "skipped by the user (Options(underflow_check=False)): the no-underflow hypothesis is assumed"
        rows.append(("No-underflow scan", utxt))
    arrays = body.get("arrays") or {}
    if arrays.get("sidecar"):
        shapes = arrays.get("shapes") or {}
        rows.append(("Arrays sidecar", f"<code>{_esc(arrays['sidecar'])}</code>: " + ", ".join(
            f"{_esc(k)} {_esc('x'.join(str(d) for d in shapes.get(k, [])))}" for k in sorted(shapes))))
    return _kv(rows)


def _key_evidence(sid: str, st: str, ev: Dict[str, Any]) -> str:
    """Per-statement key evidence in words and numbers (the generic fallback lists the scalar fields)."""
    items: List[str] = []
    if ev.get("reason_code"):
        code = ev["reason_code"]
        items.append(f"reason code <code>{_esc(code)}</code>: {_esc(REASON_CODES.get(code, 'unknown code'))}")
    if ev.get("reason"):
        items.append("reason: " + _esc(ev["reason"]))
    if sid == "S1.model_fidelity":
        if ev.get("spec_name"):
            items.append(f"declared model {_esc(ev['spec_name'])}")
        if ev.get("hermiticity_max_abs") is not None:
            items.append(f"max |H - H*| entry {_num(ev['hermiticity_max_abs'], 'up')}")
        if ev.get("model_shift_bound") is not None:
            items.append(f"model shift bound {_num(ev['model_shift_bound'], 'up')} (producer operator vs spec)")
        for key in ("comparison", "diagnosis", "reference", "internal_cross_construction", "sector_audit"):
            if ev.get(key):
                items.append(f"{_esc(key.replace('_', ' '))}: " + _value(key, ev[key]))
    elif sid == "S2.eigenpairs":
        if ev.get("n_pairs") is not None:
            items.append(f"{_num(ev['n_pairs'])} claimed pairs")
        if ev.get("max_radius") is not None:
            items.append(f"largest radius {_num(ev['max_radius'], 'up')} (rounded up)")
        if ev.get("median_radius") is not None:
            items.append(f"median radius {_num(ev['median_radius'], 'up')}")
        if ev.get("max_existence_radius") is not None:
            items.append(f"largest existence radius {_num(ev['max_existence_radius'], 'up')}")
        if ev.get("model_shift") is not None:
            items.append(f"model shift {_num(ev['model_shift'], 'up')} added to every radius")
        if ev.get("sector_lift_shift") is not None:
            items.append(f"sector lift shift {_num(ev['sector_lift_shift'], 'up')}")
        if ev.get("radius_refers_to"):
            items.append("radii refer to the " + _esc(ev["radius_refers_to"]))
        tgt = ev.get("accuracy_target") or {}
        if isinstance(tgt, dict) and tgt.get("value") is not None:
            items.append(f"accuracy target {_num(tgt['value'])} ({_esc(tgt.get('source', ''))})")
        if ev.get("rigorous") is not None:
            items.append("bounds are rigorous (standard model)" if ev["rigorous"] else "bounds are NOT rigorous")
        if ev.get("non_finite"):
            items.append("non-finite entries: " + _value("non_finite", ev["non_finite"]))
        groups = ev.get("groups")
        if isinstance(groups, list) and groups:
            items.append(f"{_num(len(groups))} cluster group(s) of size > 1: "
                         + "; ".join(f"members {_value('members', g.get('members'))} radius {_num(g.get('radius'), 'up')}"
                                     f" ({_esc(g.get('method', ''))}, bound {_esc((g.get('bound') or {}).get('used', ''))})"
                                     for g in groups))
    elif sid == "S3.completeness":
        if ev.get("next_eigenvalue_lower_bound") is not None:
            items.append(f"exactly the claimed number of eigenvalues lie below "
                         f"{_num(ev['next_eigenvalue_lower_bound'], 'down', 10)} (rounded down)")
        if ev.get("gap_lower_bound") is not None:
            items.append(f"gap to the next eigenvalue at least {_num(ev['gap_lower_bound'], 'down')} (rounded down)")
        if ev.get("certified_window") is not None:
            items.append(f"certified window {_interval(ev['certified_window'], 'inward', 10)} (rounded inward)")
        if ev.get("complete_below") is not None:
            items.append(f"the claim is complete below {_num(ev['complete_below'], 'down', 10)}")
        if ev.get("next_eigenvalue_estimate") is not None:
            items.append(f"next eigenvalue estimate {_num(ev['next_eigenvalue_estimate'])} (an estimate, not a bound)")
        for key in ("missing_at_least", "missing_below_top_claim_at_least"):
            if ev.get(key) is not None:
                items.append(f'<span class="bad">at least {_num(ev[key])} eigenvalue(s) missing from the claim</span>')
        if ev.get("count_from_inertia") is not None:
            items.append(f"inertia count in the window {_num(ev['count_from_inertia'])} vs "
                         f"{_num(ev.get('claimed_certified_inside'))} claimed pairs certified inside")
        if ev.get("top_multiplet"):
            items.append("top multiplet: " + _value("top_multiplet", ev["top_multiplet"]))
        if ev.get("unclaimed_near_top") is not None:
            items.append(f"unclaimed eigenvalues near the top of the claim: {_num(ev['unclaimed_near_top'])}")
        if ev.get("beta_effective") is not None:
            items.append(f"factorization error bound beta {_num(ev['beta_effective'], 'up')}")
    elif sid in ("S4.eigenspaces", "S8.degeneracy"):
        cl = ev.get("clusters")
        if isinstance(cl, list):
            n_c = sum(1 for c in cl if c.get("status") == "CERTIFIED")
            items.append(f"{_num(n_c)} of {_num(len(cl))} cluster(s) certified; the per-cluster table is below")
    elif sid == "S6.exact_recheck":
        if ev.get("no_floating_point_assumption"):
            items.append("no floating-point assumption: exact integer / rational arithmetic")
        if ev.get("den") is not None:
            items.append(f"common denominator {_num(ev['den'])}")
        if ev.get("next_lower_bound") is not None:
            items.append(f"exact lower bound on the next eigenvalue {_num(ev['next_lower_bound'], 'down', 10)}")
        if ev.get("radii") is not None:
            items.append("exact radii (rounded up): " + _value("radius", ev["radii"]))
    elif sid == "S7.thermodynamics":
        if ev.get("n_intervals") is not None:
            items.append(f"{_num(ev['n_intervals'])} spectrum intervals, dimension {_num(ev.get('dimension'))}")
        if ev.get("hypothesis"):
            items.append("hypothesis: " + _esc(ev["hypothesis"]))
        if ev.get("method"):
            items.append("method: " + _esc(ev["method"]))
        if ev.get("looseness"):
            items.append("looseness: " + _esc(ev["looseness"]))
    elif sid == "S5.corroboration":
        for key in ("solvers", "agreement", "max_abs_diff", "results"):
            if ev.get(key) is not None:
                items.append(f"{_esc(key.replace('_', ' '))}: " + _value(key, ev[key]))
    if ev.get("budget"):
        items.append("budget: " + _value("budget", ev["budget"]))
    if ev.get("narrowed"):
        items.append("narrowed: " + _value("narrowed", ev["narrowed"]))
    shown = {"reason_code", "reason", "next_action", "spec_name", "hermiticity_max_abs", "model_shift_bound",
             "comparison", "diagnosis", "reference", "internal_cross_construction", "sector_audit", "n_pairs",
             "max_radius", "median_radius", "max_existence_radius", "model_shift", "sector_lift_shift",
             "radius_refers_to", "accuracy_target", "rigorous", "groups", "non_finite", "next_eigenvalue_lower_bound",
             "gap_lower_bound", "certified_window", "complete_below", "next_eigenvalue_estimate", "missing_at_least",
             "missing_below_top_claim_at_least", "count_from_inertia", "claimed_certified_inside", "top_multiplet",
             "unclaimed_near_top", "beta_effective", "clusters", "no_floating_point_assumption", "den",
             "next_lower_bound", "radii", "n_intervals", "dimension", "hypothesis", "method", "looseness", "budget",
             "narrowed", "counts", "rows", "enclosures", "components", "status", "solvers", "agreement",
             "max_abs_diff", "results", "spec_sha256", "spec_invariants_vs_reference", "estimate_mb"}
    rest = {k: v for k, v in ev.items() if k not in shown}
    for k in sorted(rest):
        items.append(f"{_esc(k.replace('_', ' '))}: " + _value(k, rest[k]))
    return "<ul>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>" if items else "<p>no evidence fields</p>"


def _statements(cert: Certificate) -> str:
    rows = []
    for s in cert.statements:
        sid, st = s["id"], s["status"]
        ev = s.get("evidence") or {}
        act = ev.get("next_action")
        rows.append(
            f'<tr id="stmt-{_esc(sid)}"><td><code>{_esc(sid)}</code><br><span class="title">'
            f'{_esc(STATEMENT_TITLES.get(sid, sid))}</span></td>'
            f'<td class="status {_esc(st)}">{_esc(st)}<br><span class="word">{_esc(STATUS_WORDS.get(st, st.lower()))}'
            f'</span></td>'
            f'<td>{_esc(" ".join(str(s.get("summary", "")).split()))}'
            + (f'<div class="next"><b>Next action:</b> {_esc(act)}</div>' if act else "")
            + (f'<div class="assume">Assumptions: {_esc("; ".join(str(a) for a in s.get("assumptions") or []))}</div>'
               if s.get("assumptions") else "")
            + f'</td><td>{_key_evidence(sid, st, ev)}</td></tr>')
    legend = "<p class='legend'>" + " ".join(
        f'<span class="status {k}">{k}</span> {_esc(v)}.' for k, v in _STATUS_EXPLAIN.items()) + "</p>"
    return (f"<p><b>{_esc(cert.summary())}</b></p>" + legend
            + _table(["statement", "status", "what the check found", "key evidence"], rows, "statements"))


def _scope(cert: Certificate) -> str:
    sc = cert.body.get("scope")
    if not isinstance(sc, dict):
        return "<p>This certificate has no scope block (written before schema 2.1).</p>"
    codes = dict(REASON_CODES)
    codes.update({k: v for k, v in (sc.get("reason_codes") or {}).items() if isinstance(v, str)})
    out = [f"<p>Preset: <b>{_esc(sc.get('preset') or 'none')}</b>; operator scope: {_esc(sc.get('operator_scope', ''))}.</p>"]
    ins = sc.get("in_scope") or []
    out.append("<h3>In scope</h3>")
    out.append(_table(["statement", "status", "implied by"], [
        f'<tr><td><code>{_esc(x.get("id"))}</code></td><td class="status {_esc(x.get("status", ""))}">'
        f'{_esc(x.get("status", ""))}</td><td>{_esc(x.get("implied") or "-")}</td></tr>' for x in ins])
               if ins else "<p>none</p>")
    nar = sc.get("narrowed") or []
    out.append("<h3>Narrowed</h3>")
    out.append(_table(["statement", "by", "reason given", "reason code", "explanation", "status"], [
        f'<tr><td><code>{_esc(x.get("id"))}</code></td><td>{_esc(x.get("by", ""))}</td><td>{_esc(x.get("reason", ""))}</td>'
        f'<td><code>{_esc(x.get("reason_code", ""))}</code></td><td>{_esc(codes.get(x.get("reason_code"), "unknown reason code"))}</td>'
        f'<td class="status {_esc(x.get("status", ""))}">{_esc(x.get("status", ""))}</td></tr>' for x in nar])
               if nar else "<p>none</p>")
    outs = sc.get("out_of_scope") or []
    st_out = [x for x in outs if x.get("quantity") in STATEMENT_IDS]
    q_out = [x for x in outs if x.get("quantity") not in STATEMENT_IDS]
    out.append("<h3>Statements not checked</h3>")
    out.append(_table(["statement", "reason code", "explanation (scope matrix)", "detail"], [
        f'<tr><td><code>{_esc(x.get("quantity"))}</code></td><td><code>{_esc(x.get("reason_code", ""))}</code></td>'
        f'<td>{_esc(codes.get(x.get("reason_code"), "unknown reason code"))}</td><td>{_esc(x.get("detail", ""))}</td></tr>'
        for x in st_out]) if st_out else "<p>none</p>")
    out.append("<h3>Quantities out of scope of this version</h3>")
    out.append(_table(["quantity", "reason code", "explanation (scope matrix)", "detail"], [
        f'<tr><td>{_esc(x.get("quantity"))}</td><td><code>{_esc(x.get("reason_code", ""))}</code></td>'
        f'<td>{_esc(codes.get(x.get("reason_code"), "unknown reason code"))}</td><td>{_esc(x.get("detail", ""))}</td></tr>'
        for x in q_out]) if q_out else "<p>none</p>")
    bud = sc.get("budgets") or {}
    btxt = (f"time {_num(bud.get('seconds')) if bud.get('seconds') is not None else 'none'} s, memory "
            f"{_num(bud.get('mb')) if bud.get('mb') is not None else 'none'} MB")
    if bud.get("events"):
        btxt += "; events: " + _json_fold(bud["events"], f"{len(bud['events'])} budget event(s)")
    out.append(f"<h3>Budgets</h3><p>{btxt}</p>")
    if sc.get("options"):
        out.append("<h3>Options recorded</h3>" + _json_fold(sc["options"], "options"))
    if sc.get("requested"):
        out.append("<h3>Inputs supplied</h3>" + _json_fold(sc["requested"], "requested"))
    return "".join(out)


def _enclosures(cert: Certificate, max_pairs: Optional[int]) -> str:
    enc = cert.body.get("enclosures") or []
    if not enc:
        return "<p>No enclosure data (S2 was not run or had no pairs).</p>"
    n = len(enc)
    shown = enc if max_pairs is None or n <= max_pairs else enc[:max_pairs]
    # the sidecar radii are the enclosure radii plus the certified model shift (INT-03, rounded up): they refer
    # to the declared model, the per-pair records to the producer's operator
    r_model = cert.arrays.get("radius")
    x_model = cert.arrays.get("existence_radius")
    with_model = (r_model is not None and x_model is not None and len(r_model) == n and len(x_model) == n)
    rows = []
    for e in shown:
        i = e.get("index")
        extra = ""
        if with_model and isinstance(i, int) and 0 <= i < n:
            extra = _cell(float(r_model[i]), "up") + _cell(float(x_model[i]), "up")
        elif with_model:
            extra = "<td></td><td></td>"
        rows.append("<tr>" + _cell(i) + _cell(e.get("claimed")) + _cell(e.get("radius"), "up")
                    + _cell(e.get("existence_radius"), "up") + extra
                    + _cell(e.get("lower"), "down", 10) + _cell(e.get("upper"), "up", 10)
                    + f"<td>{_esc(e.get('group'))} (size {_esc(e.get('group_size'))}, {_esc(e.get('method', ''))})</td>"
                    + f"<td>{'yes' if e.get('distinct') else 'no'}</td>" + _cell(e.get("residual_bound"), "up") + "</tr>")
    s2 = cert.statement("S2.eigenpairs") or {}
    shift = (s2.get("evidence") or {}).get("model_shift")
    note = (f"<p>{_num(n)} pairs; showing the first {_num(len(shown))} (max_pairs).</p>" if len(shown) < n
            else f"<p>{_num(n)} claimed pair(s).</p>")
    note += ("<p>Radii and residual bounds are rounded up; the enclosure [lower, upper] is rounded outward; the "
             "claimed value is printed exactly. The first radius pair and the enclosure refer to the producer's "
             "operator; ")
    if with_model:
        note += ("the columns marked 'declared model' are the sidecar arrays: the same radii plus the certified "
                 "model shift" + (f" ({_num(shift, 'up')})" if shift is not None else "") +
                 ", which is what the methods paragraph quotes.</p>")
    else:
        note += ("no arrays sidecar was loaded, so the radii with respect to the declared model (producer radius "
                 "plus model shift" + (f" {_num(shift, 'up')}" if shift is not None else "") + ") are not tabulated.</p>")
    headers = ["pair", "claimed value mu", "radius (up)", "existence radius (up)"]
    if with_model:
        headers += ["radius, declared model (up)", "existence radius, declared model (up)"]
    headers += ["lower (down)", "upper (up)", "cluster (group, size, method)", "distinct", "residual bound (up)"]
    return note + _table(headers, rows, "enc")


def _kato_temple(cert: Certificate) -> str:
    kt = cert.body.get("kato_temple") or []
    if not kt:
        return "<p>No Kato-Temple interval improved on a residual enclosure (or the step was not run).</p>"
    rows = ["<tr>" + _cell(x.get("index")) + _cell(x.get("lower"), "down", 10) + _cell(x.get("upper"), "up", 10)
            + _cell(x.get("width"), "up") + _cell(x.get("residual_width"), "up")
            + f"<td>{_interval(x.get('isolation_interval'), 'inward', 10)}</td></tr>" for x in kt]
    return (f"<p>{_num(len(kt))} pair(s) with a Kato-Temple interval narrower than the residual enclosure "
            "(Theorem 2; the isolation interval is rounded inward).</p>"
            + _table(["pair", "lower (down)", "upper (up)", "width (up)", "residual width (up)", "isolation interval"],
                     rows, "kt"))


def _counts_table(counts: Any) -> str:
    if not isinstance(counts, list) or not counts:
        return ""
    rows = ["<tr>" + _cell(c.get("sigma")) + _cell(c.get("negative")) + f"<td>{_esc(c.get('method', ''))}</td>"
            + _cell(c.get("beta"), "up") + f"<td>{'yes' if c.get('conclusive') else 'no'}</td>"
            + f"<td>{_esc(c.get('note', ''))}</td></tr>" for c in counts if isinstance(c, dict)]
    return _table(["shift sigma", "eigenvalues below (inertia)", "method", "beta (up)", "conclusive", "note"], rows,
                  "counts")


def _completeness(cert: Certificate) -> str:
    s3 = cert.statement("S3.completeness")
    if s3 is None:
        return "<p>S3 is not in this certificate.</p>"
    ev = s3.get("evidence") or {}
    rows: List[Tuple[str, str]] = [("Status", f'<span class="status {_esc(s3["status"])}">{_esc(s3["status"])}</span>')]
    for key, label, mode, sig in (("next_eigenvalue_lower_bound", "Lower bound on the next eigenvalue", "down", 10),
                                  ("gap_lower_bound", "Gap to the next eigenvalue, at least", "down", 4),
                                  ("complete_below", "Complete below", "down", 10),
                                  ("next_eigenvalue_estimate", "Next eigenvalue (estimate)", "exact", 4),
                                  ("beta_effective", "Factorization error bound beta", "up", 4),
                                  ("top_shift", "Top shift used", "exact", 4),
                                  ("dimension", "Dimension", "exact", 4),
                                  ("missing_at_least", "Eigenvalues missing, at least", "exact", 4),
                                  ("missing_below_top_claim_at_least", "Missing below the top claim, at least", "exact", 4),
                                  ("count_from_inertia", "Inertia count in the window", "exact", 4),
                                  ("claimed_certified_inside", "Claimed pairs certified inside the window", "exact", 4)):
        if ev.get(key) is not None:
            rows.append((label, _num(ev[key], mode, sig)))
    if ev.get("certified_window") is not None:
        rows.append(("Certified window (rounded inward)", _interval(ev["certified_window"], "inward", 10)))
    if ev.get("window") is not None:
        rows.append(("Claimed window", _interval(ev["window"], "outward", 10)))
    if ev.get("top_multiplet"):
        rows.append(("Top multiplet", _value("top_multiplet", ev["top_multiplet"])))
    out = _kv(rows)
    ct = _counts_table(ev.get("counts"))
    if ct:
        out += "<h3>Inertia counts</h3>" + ct
    return out


def _clusters(cert: Certificate, sid: str) -> str:
    s = cert.statement(sid)
    if s is None:
        return f"<p>{_esc(sid)} is not in this certificate.</p>"
    cl = (s.get("evidence") or {}).get("clusters")
    if not isinstance(cl, list) or not cl:
        return f"<p>{_esc(sid)}: {_esc(s.get('summary', ''))}</p>"
    if sid == "S4.eigenspaces":
        rows = []
        for c in cl:
            rows.append(f'<tr><td>{_value("members", c.get("members"))}</td>'
                        f'<td class="status {_esc(c.get("status", ""))}">{_esc(c.get("status", ""))}</td>'
                        + _cell(c.get("separation_lower_bound"), "down") + _cell(c.get("sin_theta_bound"), "up")
                        + _cell(c.get("projector_error_bound"), "up") + _cell(c.get("rho"), "up") + _cell(c.get("eta"), "up")
                        + _cell(c.get("splitting"), "up")
                        + f"<td>{_esc(c.get('reason', ''))}{_json_fold(c.get('observables'), 'observables') if c.get('observables') else ''}"
                          f"{_json_fold(c.get('quantum_numbers'), 'quantum numbers') if c.get('quantum_numbers') else ''}</td></tr>")
        return _table(["members", "status", "separation (down)", "sin theta bound (up)", "projector error (up)",
                       "rho (up)", "eta (up)", "splitting (up)", "notes"], rows, "clusters")
    rows = []
    for c in cl:
        rows.append(f'<tr><td>{_value("members", c.get("members"))}</td>'
                    f'<td class="status {_esc(c.get("status", ""))}">{_esc(c.get("status", ""))}</td>'
                    + _cell(c.get("g_claimed")) + _cell(c.get("degeneracy") if c.get("degeneracy") is not None
                                                        else c.get("degeneracy_at_least"))
                    + f"<td>{_interval(c.get('level_enclosure'), 'outward', 10) if c.get('level_enclosure') else '-'}</td>"
                    + _cell(c.get("width"), "up")
                    + f"<td>{_interval(c.get('bracket_outer'), 'outward', 10) if c.get('bracket_outer') else '-'}</td>"
                    + _cell(c.get("count_difference"))
                    + f"<td>{_esc(c.get('statement', ''))} {_esc(c.get('reason', ''))}</td></tr>")
    out = _table(["members", "status", "claimed g", "degeneracy (or at least)", "level enclosure (outward)",
                  "width (up)", "outer bracket (outward)", "count in bracket", "statement / reason"], rows, "clusters")
    counts = [c for x in cl for c in (x.get("counts") or [])]
    if counts:
        out += "<h3>Inertia counts behind the brackets</h3>" + _counts_table(counts)
    return out


def _thermo(cert: Certificate) -> str:
    s = cert.statement("S7.thermodynamics")
    if s is None:
        return ""
    ev = s.get("evidence") or {}
    rows = ev.get("rows")
    if not isinstance(rows, list) or not rows:
        return f"<p>S7: {_esc(s.get('summary', ''))}</p>"
    tr = []
    for r in rows:
        tr.append("<tr>" + _cell(r.get("T")) + "".join(
            f"<td>{_interval(r.get(q), 'outward', 10) if r.get(q) is not None else '-'}</td>"
            for q in ("Z", "lnZ", "F", "U", "C", "variance_E")) + "</tr>")
    head = (f"<p>Energy offset {_num(ev.get('energy_offset'))}; intervals rounded outward; k_B = 1.</p>"
            if ev.get("energy_offset") is not None else "")
    return head + _table(["T", "Z", "ln Z", "F", "U", "C", "Var(E)"], tr, "thermo")


def _exact(cert: Certificate) -> str:
    s = cert.statement("S6.exact_recheck")
    if s is None:
        return ""
    ev = s.get("evidence") or {}
    out = []
    enc = ev.get("enclosures")
    if isinstance(enc, list) and enc:
        out.append(_table(["eigenvalue indices", "kind", "lower (down)", "upper (up)"], [
            f"<tr><td>{_value('eigenvalue_indices', e.get('eigenvalue_indices'))}</td><td>{_esc(e.get('kind', ''))}</td>"
            + _cell(e.get("lower"), "down", 12) + _cell(e.get("upper"), "up", 12) + "</tr>" for e in enc], "exact"))
    cnt = ev.get("counts")
    if isinstance(cnt, list) and cnt:
        out.append("<h3>Exact counts</h3>" + _table(["point", "count", "claimed below", "separated", "delta (up)"], [
            "<tr>" + _cell(c.get("point")) + _cell(c.get("count")) + _cell(c.get("claimed_below"))
            + f"<td>{'yes' if c.get('separated') else 'no'}</td>" + _cell(c.get("delta"), "up") + "</tr>" for c in cnt],
            "counts"))
    return "".join(out) if out else f"<p>S6: {_esc(s.get('summary', ''))}</p>"


def _methods(cert: Certificate) -> str:
    try:
        text = methods_paragraph(cert, style="plain")
    except Exception as e:                       # the page must still render
        return f"<p>methods paragraph unavailable: {_esc(type(e).__name__)}: {_esc(e)}</p>"
    para, _, bib = text.partition("@software")
    return (f'<p class="methods">{_esc(para.strip())}</p>'
            + (f"<pre>@software{_esc(bib)}</pre>" if bib else "")
            + '<p class="small">Numbers in this paragraph are rounded in the safe direction by certify_ed.report '
              '(radii up, gap bounds down); see <code>certify-ed statement</code> for the LaTeX form.</p>')


def _footer(cert: Certificate, integ: Optional[dict], schema: Optional[dict]) -> str:
    rows: List[Tuple[str, str]] = []
    if integ is None:
        rows.append(("Integrity", "not checked (render_html called without an integrity report)"))
    else:
        ok = lambda b: '<span class="ok">ok</span>' if b else '<span class="bad">FAILED</span>'
        rows.append(("Body hash (verify_integrity)", ok(integ.get("body_hash_ok"))
                     + f"<br>stored <code>{_esc(integ.get('stored_body_sha256'))}</code>"
                     + (f"<br>recomputed <code>{_esc(integ.get('recomputed_body_sha256'))}</code>"
                        if not integ.get("body_hash_ok") else "")))
        rows.append(("Array hashes", ok(integ.get("arrays_ok"))
                     + (f"<br>bad or missing: {_esc(', '.join(integ.get('bad_arrays') or []))}" if not integ.get("arrays_ok") else "")
                     + (f"<br>declared: {_esc(', '.join(integ.get('arrays_declared') or []) or 'none')}")))
        sc_ok = integ.get("scope_ok")
        rows.append(("Scope block (verify_scope)", ('<span class="warn">not applicable (no scope block)</span>' if sc_ok is None
                                                    else ok(sc_ok))
                     + ("<br>" + "<br>".join(_esc(p) for p in integ.get("scope_problems") or []) if integ.get("scope_problems") else "")))
        rows.append(("Overall", ok(integ.get("ok"))))
    if schema is None:
        rows.append(("Schema validation", "not run"))
    else:
        s_ok = schema.get("ok")
        rows.append(("Schema validation", ('<span class="ok">ok</span>' if s_ok else
                                           '<span class="warn">structural checks only</span>' if s_ok is None else
                                           '<span class="bad">FAILED</span>') + "<br>" + _esc(schema.get("detail", ""))))
    nv = cert.body.get("not_verified") or []
    extra = ("<h3>Not verified by this certificate</h3><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in nv) + "</ul>") if nv else ""
    return _kv(rows) + extra + ('<p class="small">A certificate hash shows that the file was not modified after it '
                                'was saved; it does not show that the computation behind it was honest. Re-running '
                                '<code>certify</code> on the stored claim is the check that does not trust the producer.</p>')


_CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:1200px;margin:1.5em auto;padding:0 1em;color:#111;background:#fff;line-height:1.4}
h1{font-size:1.5em;margin:0 0 .3em 0} h2{font-size:1.15em;display:inline;margin:0} h3{font-size:1em;margin:1em 0 .3em 0}
summary{cursor:pointer;padding:.4em 0;border-bottom:1px solid #999;margin-top:1.2em}
details>.body{padding:.5em 0 0 0}
table{border-collapse:collapse;font-size:.9em;margin:.5em 0;width:100%}
th,td{border:1px solid #bbb;padding:.3em .5em;vertical-align:top;text-align:left}
th{background:#eee} table.kv th{width:16em}
td.num,span.num{font-family:'DejaVu Sans Mono',Consolas,monospace;white-space:nowrap}
td.num{text-align:right}
.status{font-weight:bold} .status.CERTIFIED{color:#1b5e20} .status.FAILED{color:#b71c1c}
.status.INCONCLUSIVE{color:#e65100} .status.NOT_CHECKED{color:#555} .status.CORROBORATED{color:#0d47a1}
.word{font-weight:normal;color:#444;font-size:.9em} .title{color:#333;font-size:.9em}
.banner{padding:.8em 1em;margin:.8em 0;border:2px solid;font-weight:bold}
.banner.ok{border-color:#1b5e20;background:#e8f5e9;color:#1b5e20}
.banner.fail{border-color:#b71c1c;background:#ffebee;color:#b71c1c;font-size:1.2em}
.banner.warn{border-color:#e65100;background:#fff3e0;color:#e65100}
.ok{color:#1b5e20;font-weight:bold} .bad{color:#b71c1c;font-weight:bold} .warn{color:#e65100;font-weight:bold}
.next{margin-top:.4em;color:#333} .assume{margin-top:.3em;color:#555;font-size:.9em}
.legend{font-size:.9em;color:#333} .small{font-size:.85em;color:#444} .methods{text-align:justify}
ul{margin:.2em 0;padding-left:1.2em} pre{white-space:pre-wrap;font-size:.85em;background:#f7f7f7;padding:.5em}
code{font-family:'DejaVu Sans Mono',Consolas,monospace;font-size:.9em;word-break:break-all}
"""


def render_html(cert: Certificate, integrity: Optional[dict] = None, schema_result: Optional[dict] = None,
                title: Optional[str] = None, max_pairs: Optional[int] = None) -> str:
    """The HTML document (a str) for `cert`. `integrity` is `integrity_report(cert)` (None: not shown, and the
    page says so); `schema_result` is `schema_report(cert)`; `max_pairs` caps the enclosure table rows."""
    body = cert.body
    sub = body.get("subject") or {}
    spec = sub.get("spec") or {}
    name = title or f"CERTIFY-ED certificate: {spec.get('name') or sub.get('description') or 'certificate'}"
    sections = [
        _banner(integrity, schema_result),
        f"<h1>{_esc(name)}</h1>",
        _header(cert),
        _details("Statements", _statements(cert), True, "statements"),
        _details("Scope", _scope(cert), True, "scope"),
        _details("Eigenvalue enclosures (S2)", _enclosures(cert, max_pairs),
                 len(body.get("enclosures") or []) <= 64, "enclosures"),
        _details("Kato-Temple intervals", _kato_temple(cert), bool(body.get("kato_temple")), "kato-temple"),
        _details("Completeness and gap (S3)", _completeness(cert), cert.statement("S3.completeness") is not None,
                 "completeness"),
        _details("Eigenspace clusters (S4)", _clusters(cert, "S4.eigenspaces"), False, "eigenspaces"),
        _details("Degeneracy clusters (S8)", _clusters(cert, "S8.degeneracy"), False, "degeneracy"),
    ]
    th = _thermo(cert)
    if th:
        sections.append(_details("Thermodynamic intervals (S7)", th, True, "thermodynamics"))
    ex = _exact(cert)
    if ex:
        sections.append(_details("Exact re-check (S6)", ex, True, "exact"))
    sections.append(_details("Methods paragraph", _methods(cert), True, "methods"))
    sections.append(_details("Integrity report", _footer(cert, integrity, schema_result), True, "integrity"))
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{_esc(name)}</title><style>{_CSS}</style></head><body>\n" + "\n".join(sections)
            + '\n<p class="small">Rendered by certify_ed.viewer; no external resources, no scripts. '
              'The page opens offline in any browser.</p></body></html>\n')


def _load_tolerant(path: str) -> Tuple[Certificate, List[str]]:
    """Load the body and whatever arrays can be read; never raise on a bad hash or a missing sidecar."""
    notes: List[str] = []
    with open(path, encoding="utf-8") as f:
        body = json.load(f)
    arrays: Dict[str, np.ndarray] = {}
    side = (body.get("arrays") or {}).get("sidecar")
    if side:
        sp_path = os.path.join(os.path.dirname(os.path.abspath(path)), side)
        try:
            with np.load(sp_path) as z:
                arrays = {k: z[k] for k in z.files}
        except (OSError, ValueError) as e:
            notes.append(f"arrays sidecar {side} could not be read: {e}")
    return Certificate(body, arrays), notes


def view_file(path: str, out: Optional[str] = None, max_pairs: Optional[int] = None) -> Dict[str, Any]:
    """Render the certificate at `path` to `out` (default: same stem with .html). Returns
    {'html': out, 'integrity': report, 'schema': report, 'notes': [...]}; never raises on a tampered file."""
    cert, notes = _load_tolerant(path)
    integ = integrity_report(cert)
    if notes:
        integ["ok"] = False
        integ["arrays_ok"] = False
        integ["bad_arrays"] = sorted(set(integ.get("bad_arrays") or []) | set(integ.get("arrays_declared") or []))
    schema = schema_report(cert)
    html = render_html(cert, integ, schema, max_pairs=max_pairs)
    if out is None:
        out = (path[:-5] if path.lower().endswith(".json") else path) + ".html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return {"html": out, "integrity": integ, "schema": schema, "notes": notes}
