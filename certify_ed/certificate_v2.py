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
from different inputs. Re-verification (`reverify`) recomputes the statements
from the stored claim and a user-supplied operator/spec and is the only check
that does not trust the producer of the certificate. See THREAT_MODEL in the
documentation.

Status vocabulary for statements:
  CERTIFIED     the statement holds, with the stated assumptions
  FAILED        the statement is refuted (e.g. matrix differs from spec,
                eigenvalues provably missing)
  INCONCLUSIVE  checks ran but could not decide (e.g. zero pivot, overlapping
                existence-only intervals)
  NOT_CHECKED   the check was not requested or its inputs were unavailable
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import platform
import sys
from typing import Any, Dict, Optional

import numpy as np
import scipy
import scipy.sparse as sp

from .spec import canonical_json

CERT_SCHEMA = "certify-ed/certificate/2.0"
STATUSES = ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED")


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
    return env


class Certificate:
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

    def summary(self) -> str:
        lines = [f"CERTIFY-ED certificate ({self.body.get('schema')})",
                 f"  subject : {self.body.get('subject', {}).get('description', '')}",
                 f"  claim   : {self.body.get('claim', {}).get('description', '')}"]
        for s in self.statements:
            lines.append(f"  [{s['status']:>12s}] {s['id']}: {s.get('summary', '')}")
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
        rep["ok"] = rep["body_hash_ok"] and rep["arrays_ok"]
        return rep

    def export_hdf5(self, path: str) -> str:
        """Optional HDF5 export (eigenvalues, residual bounds, radii, statements as JSON attribute)."""
        import h5py
        with h5py.File(path, "w") as f:
            for k, v in self.arrays.items():
                f.create_dataset(k, data=v)
            f.attrs["certificate_json"] = canonical_json(self.body)
        return path
