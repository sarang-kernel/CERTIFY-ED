"""The producer bundle: a directory any ED code can write, which CERTIFY-ED reads as a claim.

    bundle/
      eigenvalues.<ext>      required   1-D real array
      eigenvectors.<ext>     optional   n x k array (columns are the claimed vectors)
      operator.<ext>         optional   the producer's matrix (dense/sparse) or operator.py (matrix-free)
      basis.<ext>            optional   1-D integer state list (U(1) block in the spec's encoding) or an
                                        n_full x n isometry (symmetry-adapted block)
      spec.json              optional   ModelSpec.to_dict() of the declared model
      meta.json              optional   kind, window, producer, accuracy_target, description, keys,
                                        state_order, layout, temperatures, preset, options, ...

`<ext>` is any of the formats of `certify_ed.io.formats` (.npy .npz .h5 .csv .tsv .txt .dat .json .mtx);
the first match of each stem wins (the order of FORMAT_PREFERENCE). Full documentation:
docs/producer_bundle.md. Nothing is executed unless `allow_code=True` (operator.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

from ..certify import Claim
from ..construct import StateSet
from ..precision import is_narrow_float
from ..spec import ModelSpec
from .formats import (ARRAY_EXT, FormatError, read_array, read_basis, read_operator, write_array,
                      write_basis, write_operator)

STEMS = ("eigenvalues", "eigenvectors", "operator", "basis")
FORMAT_PREFERENCE = (".npz", ".npy", ".h5", ".hdf5", ".hdf", ".mtx", ".json", ".csv", ".tsv", ".txt", ".dat", ".py")
META_SCHEMA = "certify-ed/producer-bundle/1.0"

__all__ = ["Bundle", "read_bundle", "write_bundle", "bundle_files", "STEMS", "META_SCHEMA"]


@dataclass
class Bundle:
    claim: Claim
    spec: Optional[ModelSpec]
    meta: Dict[str, Any]
    files: Dict[str, str] = field(default_factory=dict)
    path: str = ""


def bundle_files(path: str) -> Dict[str, str]:
    """Resolve stem -> file for the four data stems plus spec.json / meta.json."""
    if not os.path.isdir(path):
        raise FormatError(f"{path}: not a directory (a producer bundle is a directory)")
    names = sorted(os.listdir(path))
    out = {}
    for stem in STEMS:
        cands = [n for n in names if os.path.splitext(n)[0] == stem and os.path.splitext(n)[1].lower() in FORMAT_PREFERENCE]
        if cands:
            cands.sort(key=lambda n: FORMAT_PREFERENCE.index(os.path.splitext(n)[1].lower()))
            out[stem] = os.path.join(path, cands[0])
    for extra in ("spec.json", "meta.json"):
        if extra in names:
            out[extra[:-5]] = os.path.join(path, extra)
    if "eigenvalues" not in out:
        raise FormatError(f"{path}: no eigenvalues.<ext> file (extensions: {ARRAY_EXT})")
    return out


def read_bundle(path: str, allow_code: bool = False, kind: Optional[str] = None, window=None) -> Bundle:
    """Read a producer bundle into a Claim (+ ModelSpec when spec.json is present)."""
    files = bundle_files(path)
    meta: Dict[str, Any] = {}
    if "meta" in files:
        with open(files["meta"]) as f:
            meta = json.load(f)
    keys = meta.get("keys", {}) or {}
    cp = meta.get("complex_pairs", {}) or {}
    ev = read_array(files["eigenvalues"], keys.get("eigenvalues"), "eigenvalues",
                    complex_pairs=bool(cp.get("eigenvalues", False)))
    ev = np.asarray(ev)
    if ev.ndim == 2 and ev.shape[1] == 1:
        ev = ev[:, 0]
    if ev.ndim != 1:
        raise FormatError(f"eigenvalues must be 1-D, got shape {ev.shape}")
    if np.iscomplexobj(ev):
        if np.any(ev.imag != 0):
            raise FormatError("eigenvalues must be real (a Hermitian claim)")
        ev = ev.real
    X = None
    if "eigenvectors" in files:
        X = read_array(files["eigenvectors"], keys.get("eigenvectors"), "eigenvectors",
                       complex_pairs=bool(cp.get("eigenvectors", False)))
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] != ev.size and X.shape[0] == ev.size:
            if meta.get("eigenvectors_layout") == "rows":
                X = X.T
            elif X.shape[0] != X.shape[1]:
                X = X.T            # unambiguous: rows were the vectors
        if X.shape[1] != ev.size:
            raise FormatError(f"eigenvectors shape {X.shape} does not match {ev.size} eigenvalues "
                              f"(set meta.eigenvectors_layout = 'rows' if the vectors are rows)")
    op = None
    if "operator" in files:
        lay = (meta.get("layout", {}) or {}).get("operator", "auto")
        op = read_operator(files["operator"], keys.get("operator"), layout=lay, allow_code=allow_code,
                           shape=meta.get("operator_shape"), one_based=bool(meta.get("one_based", False)))
    spec = None
    if "spec" in files:
        with open(files["spec"]) as f:
            spec = ModelSpec.from_dict(json.load(f))
    states, iso = None, None
    if "basis" in files:
        kind_b, B = read_basis(files["basis"], keys.get("basis"))
        if kind_b == "states":
            order = meta.get("state_order") or (spec.state_order if spec is not None else "descending")
            states = StateSet(np.sort(B.astype(np.int64)), order, label=meta.get("basis_label", "producer basis"))
        else:
            iso = B if sp.issparse(B) and is_narrow_float(B.dtype) else sp.csr_matrix(B)   # no single-precision sum
    kind = kind or meta.get("kind", "subset")
    window = window if window is not None else meta.get("window")
    if window is not None:
        from ..certify import parse_window
        window = parse_window(window)
    producer = dict(meta.get("producer", {}) or {})
    producer.setdefault("name", "producer bundle")
    producer["bundle"] = os.path.abspath(path)
    producer["files"] = {k: os.path.basename(v) for k, v in files.items()}
    claim = Claim(ev, X, operator=op, kind=kind, window=window, states=states, sector_isometry=iso,
                  producer=producer, accuracy_target=meta.get("accuracy_target"),
                  description=meta.get("description", ""))
    return Bundle(claim, spec, meta, files, os.path.abspath(path))


def write_bundle(path: str, eigenvalues, eigenvectors=None, operator=None, spec: Optional[ModelSpec] = None,
                 basis=None, fmt: str = "npz", meta: Optional[Dict[str, Any]] = None, kind: Optional[str] = None,
                 window=None, operator_fmt: Optional[str] = None) -> str:
    """Write a producer bundle (the `of=` side; also what the round-trip tests use).

    fmt: extension without the dot for the arrays (npz, npy, h5, csv, tsv, txt, dat, json, mtx);
    operator_fmt defaults to fmt (sparse operators go to npz/mtx/h5/json/text COO)."""
    os.makedirs(path, exist_ok=True)
    fmt = fmt.lstrip(".")
    ofmt = (operator_fmt or fmt).lstrip(".")
    if operator is not None and sp.issparse(operator) and operator_fmt is None and ofmt == "npy":
        ofmt = "npz"           # .npy has no sparse form
    write_array(os.path.join(path, f"eigenvalues.{fmt}"), np.asarray(eigenvalues, dtype=float), role="eigenvalues")
    if eigenvectors is not None:
        write_array(os.path.join(path, f"eigenvectors.{fmt}"), np.asarray(eigenvectors), role="eigenvectors")
    if operator is not None:
        write_operator(os.path.join(path, f"operator.{ofmt}"), operator)
    if basis is not None:
        write_basis(os.path.join(path, f"basis.{fmt}"), basis)
    if spec is not None:
        with open(os.path.join(path, "spec.json"), "w") as f:
            json.dump(spec.to_dict(), f, indent=1)
    m = {"schema": META_SCHEMA}
    if kind is not None:
        m["kind"] = kind
    if window is not None:
        m["window"] = [float(window[0]), float(window[1])]
    if meta:
        m.update(meta)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(m, f, indent=1, default=float)
    return path
