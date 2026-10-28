"""Certificate format v2: a machine-readable record of what was checked.

A certificate is a JSON document plus an optional NPZ sidecar holding large
arrays (eigenvectors, per-pair data). Integrity:

  * every array in the sidecar is hashed (SHA-256 over dtype, shape and
    little-endian C-order bytes) and the hashes are stored in the JSON body;
  * the JSON body is serialised canonically (sorted keys, fixed separators,
    shortest round-trip float representation, no NaN) and its SHA-256 is
    stored under "integrity.body_sha256".

The hash detects modification of a saved certificate. It does NOT show that
the computation behind it was honest: a certificate can always be regenerated
from different inputs. Re-verification means running `certify` again on the
stored claim (the eigenvalues; the eigenvectors only if they were stored with
Options(store_eigenvectors=True)) with an operator and a specification that the
checker supplies; it is the only check that does not trust the producer of the
certificate. `Certificate.load(verify=True)` checks integrity only.

Since schema 2.1 the body carries a mandatory `scope` section (certify_ed.scope): the preset, the statements
in scope, the statements the user narrowed out with the reason, and the quantities out of scope with a
reason code (docs/SCOPE_MATRIX.md). Optional `spectrum_enclosure` holds the Prop 9 interval list.

Status vocabulary for statements:
  CERTIFIED     the statement holds, with the stated assumptions
  FAILED        the statement is refuted (e.g. matrix differs from spec,
                eigenvalues provably missing)
  INCONCLUSIVE  checks ran but could not decide (e.g. zero pivot, overlapping
                existence-only intervals)
  NOT_CHECKED   the check was not requested, its inputs were unavailable, or it did not fit
                in memory
  CORROBORATED  (S5 only) independent recomputation agrees; evidence, not a certified statement
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, Optional

import numpy as np
import scipy
import scipy.sparse as sp

from .spec import canonical_json

CERT_SCHEMA = "certify-ed/certificate/2.1"
STATUSES = ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED")
STATEMENT_IDS = ("S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces", "S5.corroboration",
                 "S6.exact_recheck", "S7.thermodynamics", "S8.degeneracy")


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def matrix_sha256(A) -> str:
    """Hash of a canonical CSR form (sorted indices, summed duplicates, no explicit zeros)."""
    if not sp.issparse(A):
        A = sp.csr_matrix(np.asarray(A))
    A = sp.csr_matrix(A, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    h = hashlib.sha256()
    h.update(json.dumps(list(A.shape)).encode())
    for arr in (A.indptr.astype(np.int64), A.indices.astype(np.int64),
                A.data.astype(np.complex128 if np.iscomplexobj(A.data) else np.float64)):
        h.update(array_sha256(arr).encode())
    return h.hexdigest()


def _jsonable(x):
    """Convert numpy scalars/arrays and non-finite floats into JSON-safe values."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if f != f:
            return "nan"
        if f in (float("inf"), float("-inf")):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(x, (np.complexfloating, complex)):
        return [_jsonable(x.real), _jsonable(x.imag)]
    return x


def environment_record() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        from threadpoolctl import threadpool_info
        env["blas"] = [{k: d.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                       for d in threadpool_info()]
    except Exception:
        try:
            cfg = np.show_config(mode="dicts")
            env["blas"] = cfg.get("Build Dependencies", {}).get("blas", {})
        except Exception:
            env["blas"] = "unavailable"
    try:
        from . import __version__
        env["certify_ed"] = __version__
    except Exception:
        pass
    # IMP-26: records of different code states all said '3.0.0'; the computational source hash (and the
    # commit, when run from a checkout) distinguishes them. None outside a repository.
    try:
        from .provenance import source_fingerprint
        fp = source_fingerprint()
        env["source_hash"] = fp.get("source_hash")
        env["source_dirty"] = fp.get("source_dirty")
        env["git_commit"] = fp.get("git_commit")
    except Exception:
        env["source_hash"] = None
    try:
        import flint
        env["python_flint"] = getattr(flint, "__version__", "installed")
        env["flint_c_library"] = str(getattr(flint, "__FLINT_VERSION__", "unknown"))
    except Exception:
        pass
    return env


class Certificate:
    """A certificate: body (statements, evidence, claim, subject, environment) and arrays, with
    canonical-JSON hashing for integrity (save/load/verify_integrity, HDF5 export)."""

    def __init__(self, body: Dict[str, Any], arrays: Optional[Dict[str, np.ndarray]] = None):
        self.body = body
        self.arrays = arrays or {}

    # ------------------------------------------------------------- summary
    @property
    def statements(self):
        return self.body.get("statements", [])

    def status_of(self, sid: str) -> Optional[str]:
        for s in self.statements:
            if s["id"] == sid:
                return s["status"]
        return None

    def statement(self, sid: str) -> Optional[dict]:
        """The statement record with this id, or None."""
        for s in self.statements:
            if s["id"] == sid:
                return s
        return None

    def counts_by_status(self) -> Dict[str, list]:
        """status -> short ids ('S1', ...) of the statements with that status, in certificate order."""
        out: Dict[str, list] = {}
        for s in self.statements:
            out.setdefault(s["status"], []).append(s["id"].split(".")[0])
        return out

    def max_radius(self) -> Optional[float]:
        """Largest certified S2 radius (sidecar 'radius' when present, else S2 evidence); None without S2."""
        r = self.arrays.get("radius")
        if r is not None and len(r):
            return float(np.max(r))
        s2 = self.statement("S2.eigenpairs")
        if s2 and "max_radius" in (s2.get("evidence") or {}):
            return float(s2["evidence"]["max_radius"])
        return None

    def summary(self) -> str:
        """One sentence: what was certified and what was not (statement counts by status, the largest S2 radius,
        whether completeness holds and with which gap). `report()` is the full multi-line text."""
        from .report import round_decimal
        by = self.counts_by_status()
        n = len(self.statements)
        parts = []
        cert = by.get("CERTIFIED", [])
        parts.append(f"{len(cert)} of {n} statements CERTIFIED" + (f" ({', '.join(cert)})" if cert else ""))
        for st in ("FAILED", "INCONCLUSIVE", "NOT_CHECKED", "CORROBORATED"):
            ids = by.get(st, [])
            if ids:
                parts.append(f"{len(ids)} {st} ({', '.join(ids)})")
        head = ", ".join(parts)
        tail = []
        s2 = self.statement("S2.eigenpairs")
        r = self.max_radius()
        if s2 and s2["status"] == "CERTIFIED" and r is not None:
            tail.append(f"max certified radius {round_decimal(r, 3, 'up')} (rounded up)")
        s3 = self.statement("S3.completeness")
        if s3 is not None:
            ev = s3.get("evidence") or {}
            if s3["status"] == "CERTIFIED":
                g = ev.get("gap_lower_bound")
                tail.append("completeness holds" + (f" with gap >= {round_decimal(g, 3, 'down')}" if g is not None else ""))
            elif s3["status"] == "FAILED":
                tail.append("completeness is refuted")
            else:
                tail.append(f"completeness {s3['status']}")
        sc = self.body.get("scope")
        if isinstance(sc, dict) and sc.get("preset"):
            tail.append(f"preset {sc['preset']}")
        sentence = head + ("; " + "; ".join(tail) if tail else "") + "."
        return sentence[0].upper() + sentence[1:]

    def _rows(self):
        return [(s["id"], s["status"], " ".join(str(s.get("summary", "")).split()),
                 (s.get("evidence") or {}).get("next_action")) for s in self.statements]

    def _repr_markdown_(self) -> str:
        """Markdown table (statement, status, one-line evidence with the next action) for notebooks."""
        lines = [f"**{self.summary()}**", "", "| statement | status | evidence |", "|---|---|---|"]
        for sid, st, summ, act in self._rows():
            cell = summ.replace("|", "\\|")
            if act:
                cell += f" *next: {act}*".replace("|", "\\|")
            lines.append(f"| `{sid}` | {st} | {cell} |")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """HTML table (statement, status, one-line evidence with the next action) for Jupyter."""
        import html as _html
        colour = {"CERTIFIED": "#2e7d32", "FAILED": "#c62828", "INCONCLUSIVE": "#ef6c00", "NOT_CHECKED": "#616161",
                  "CORROBORATED": "#1565c0"}
        rows = []
        for sid, st, summ, act in self._rows():
            cell = _html.escape(summ)
            if act:
                cell += f'<br><span style="color:#616161">next: {_html.escape(act)}</span>'
            rows.append(f'<tr><td><code>{_html.escape(sid)}</code></td>'
                        f'<td style="color:{colour.get(st, "#000")};font-weight:bold">{_html.escape(st)}</td>'
                        f'<td>{cell}</td></tr>')
        return (f'<div class="certify-ed"><p><b>{_html.escape(self.summary())}</b></p>'
                '<table><thead><tr><th>statement</th><th>status</th><th>evidence</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>')

    def methods_paragraph(self, style: str = "plain") -> str:
        """Methods-section paragraph with every number taken from this certificate (certify_ed.report)."""
        from .report import methods_paragraph
        return methods_paragraph(self, style=style)

    def report(self) -> str:
        """Full multi-line text: subject, claim, every statement with its summary, scope, not-verified list."""
        lines = [f"CERTIFY-ED certificate ({self.body.get('schema', CERT_SCHEMA)})",
                 f"  subject : {self.body.get('subject', {}).get('description', '')}",
                 f"  claim   : {self.body.get('claim', {}).get('description', '')}"]
        for s in self.statements:
            lines.append(f"  [{s['status']:>12s}] {s['id']}: {s.get('summary', '')}")
        sc = self.body.get("scope")
        if isinstance(sc, dict):
            lines.append(f"  scope   : preset {sc.get('preset') or '-'}; in scope "
                         f"{', '.join(x['id'].split('.')[0] for x in sc.get('in_scope', [])) or '-'}")
            nar = sc.get("narrowed", [])
            if nar:
                lines.append("  narrowed: " + "; ".join(f"{x['id'].split('.')[0]} ({x['by']}: {x['reason']})" for x in nar))
            out = [x for x in sc.get("out_of_scope", []) if x["quantity"] in STATEMENT_IDS]
            if out:
                lines.append("  not checked: " + "; ".join(f"{x['quantity'].split('.')[0]} [{x['reason_code']}]" for x in out))
            lines.append(f"  out of scope: {sum(1 for x in sc.get('out_of_scope', []) if x['quantity'] not in STATEMENT_IDS)} "
                         f"quantities with reason codes (see scope.out_of_scope, docs/SCOPE_MATRIX.md)")
        ns = self.body.get("not_verified", [])
        if ns:
            lines.append("  not verified:")
            lines += [f"    - {x}" for x in ns]
        return "\n".join(lines)

    # ----------------------------------------------------------- save/load
    def _finalize_body(self, sidecar_name: Optional[str]) -> dict:
        body = _jsonable(dict(self.body))
        body["schema"] = CERT_SCHEMA
        if self.arrays:
            body["arrays"] = {"sidecar": sidecar_name,
                              "sha256": {k: array_sha256(v) for k, v in sorted(self.arrays.items())},
                              "shapes": {k: list(v.shape) for k, v in sorted(self.arrays.items())},
                              "dtypes": {k: v.dtype.str for k, v in sorted(self.arrays.items())}}
        body.pop("integrity", None)
        digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        body["integrity"] = {"algorithm": "sha256", "canonicalization": "certify-ed-canonical-json-1",
                             "body_sha256": digest}
        return body

    def save(self, path: str) -> str:
        base = path[:-5] if path.endswith(".json") else path
        sidecar = os.path.basename(base) + ".arrays.npz" if self.arrays else None
        body = self._finalize_body(sidecar)
        with open(base + ".json", "w") as f:
            f.write(canonical_json(body))
        if self.arrays:
            np.savez(base + ".arrays.npz", **self.arrays)
        self.body = body
        return base + ".json"

    @staticmethod
    def load(path: str, verify: bool = True) -> "Certificate":
        with open(path) as f:
            body = json.load(f)
        arrays = {}
        side = body.get("arrays", {}).get("sidecar")
        if side:
            with np.load(os.path.join(os.path.dirname(path), side)) as z:
                arrays = {k: z[k] for k in z.files}
        cert = Certificate(body, arrays)
        if verify:
            rep = cert.verify_integrity()
            if not rep["ok"]:
                raise ValueError(f"certificate integrity check failed: {rep}")
        return cert

    def verify_integrity(self) -> dict:
        body = dict(self.body)
        integ = body.pop("integrity", {})
        digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        rep = {"body_hash_ok": digest == integ.get("body_sha256"), "arrays_ok": True, "bad_arrays": []}
        for k, h in body.get("arrays", {}).get("sha256", {}).items():
            if k not in self.arrays or array_sha256(self.arrays[k]) != h:
                rep["arrays_ok"] = False
                rep["bad_arrays"].append(k)
        sc = self.verify_scope()
        rep["scope_ok"], rep["scope_problems"] = sc["ok"], sc["problems"]
        rep["ok"] = rep["body_hash_ok"] and rep["arrays_ok"] and sc["ok"] is not False
        return rep

    def verify_scope(self) -> dict:
        """Check the scope block (INT-02): every statement id is accounted for exactly once (in scope, narrowed, or
        out of scope), reason codes are known, every statement in scope is present, a narrowed statement that is
        present is NOT_CHECKED, and no statement appears outside the scope lists. A narrowed statement may never be
        silently absent from the scope block. Returns {'ok': True | False | None, 'problems': [...]}; None for a
        certificate written before the scope block existed (schema < 2.1)."""
        from .scope import REASON_CODES
        sc = self.body.get("scope")
        if sc is None:
            return {"ok": None, "problems": ["no scope block (certificate written before schema 2.1)"]}
        problems = []
        ids_in = [x.get("id") for x in sc.get("in_scope", [])]
        ids_nar = [x.get("id") for x in sc.get("narrowed", [])]
        ids_out = [x.get("quantity") for x in sc.get("out_of_scope", []) if x.get("quantity") in STATEMENT_IDS]
        listed = ids_in + ids_nar + ids_out
        for sid in STATEMENT_IDS:
            n = listed.count(sid)
            if n != 1:
                problems.append(f"{sid}: listed {n} times in the scope block (must be exactly once)")
        for x in sc.get("narrowed", []) + sc.get("out_of_scope", []):
            if x.get("reason_code") not in REASON_CODES:
                problems.append(f"{x.get('id') or x.get('quantity')}: unknown reason code {x.get('reason_code')!r}")
        present = {s["id"]: s for s in self.statements}
        for sid in ids_in:
            if sid not in present:
                problems.append(f"{sid}: in scope but absent from the statements")
        for sid in ids_nar:
            if sid in present and present[sid].get("status") != "NOT_CHECKED":
                problems.append(f"{sid}: narrowed but reported {present[sid].get('status')}")
        for sid in present:
            if sid not in listed:
                problems.append(f"{sid}: reported but not accounted for in the scope block")
        return {"ok": not problems, "problems": problems}

    def export_hdf5(self, path: str) -> str:
        """Optional HDF5 export (eigenvalues, residual bounds, radii, statements as JSON attribute)."""
        import h5py
        with h5py.File(path, "w") as f:
            for k, v in self.arrays.items():
                f.create_dataset(k, data=v)
            # same body (with array hashes and integrity record) as the JSON file
            f.attrs["certificate_json"] = canonical_json(self._finalize_body(None))
        return path
