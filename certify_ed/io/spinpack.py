"""Readers for SPINPACK text output (J. Schulenburg, Uni Magdeburg), roadmap F-02.

STATUS OF THE LAYOUT: ASSUMED. No SPINPACK file specification was available offline when this reader was
written (the package's documentation is the source tree and its web page, which describe the model
definition file `daten.def` and the result files only by example). The reader therefore implements two
tolerant text layouts and names them in the Claim's `producer` record, so that a certificate says which
assumption it rests on:

  * `layout="table"` (default): a whitespace-separated text table with optional `#` comment lines; one line per
    eigenvalue; the energy in the column `energy_column` (default: the last column); the other columns are
    sector labels (Sz, momentum, symmetry index) and are copied to the producer record. Lines are kept in file
    order; `select={column_index: value}` keeps one sector.
  * `layout="keyvalue"`: lines containing tokens `E=<value>` or `E<n>=<value>` (also `E0=`, `E1=`), one energy
    per token in file order; everything else is ignored.

`read_spinpack_def(path)` parses a `daten.def`-style model file into a dictionary of `key = value` / `key value`
pairs (sites, spin, couplings as text) for the provenance record only; nothing is derived from it.

When a real SPINPACK run is at hand the layouts above should be checked against it; if they are wrong, the
energy file can always be imported through `claim_from_file` / a producer bundle after a one-line conversion.
SPINPACK eigenvectors are not read (no layout available); S2-S4 need vectors from another route.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from .formats import FormatError

__all__ = ["read_spinpack_energies", "read_spinpack_def", "claim_from_spinpack", "SPINPACK_LAYOUT_STATUS"]

SPINPACK_LAYOUT_STATUS = ("assumed: no file specification was available; table = energy in a text column, "
                          "keyvalue = E=<value> tokens")


def read_spinpack_energies(path: str, layout: str = "table", energy_column: int = -1) -> Dict[str, object]:
    """Energies from a SPINPACK text output file under one of the assumed layouts (module docstring).
    Returns {'energies': array, 'labels': (n x m) float array of the other columns (table layout) or None,
    'layout': str}."""
    if layout not in ("table", "keyvalue"):
        raise ValueError("layout must be 'table' or 'keyvalue'")
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if layout == "keyvalue":
        rx = re.compile(r"(?<![A-Za-z])E\d*\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)")
        vals: List[float] = []
        for line in lines:
            if line.strip().startswith("#"):
                continue
            for v in rx.findall(line):
                vals.append(float(v.replace("d", "e").replace("D", "E")))
        if not vals:
            raise FormatError(f"{path}: no 'E=<value>' tokens found (keyvalue layout)")
        return {"energies": np.array(vals, dtype=float), "labels": None, "layout": "keyvalue"}
    rows: List[List[float]] = []
    for ln, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        toks = line.replace(",", " ").split()
        try:
            rows.append([float(t.replace("d", "e").replace("D", "E")) for t in toks])
        except ValueError as e:
            raise FormatError(f"{path}:{ln}: non-numeric entry in {raw!r} ({e}); the table layout expects "
                              f"numeric columns (comment lines start with '#')") from e
    if not rows:
        raise FormatError(f"{path}: no data lines (table layout)")
    width = {len(r) for r in rows}
    if len(width) != 1:
        raise FormatError(f"{path}: lines have different numbers of columns {sorted(width)} (table layout)")
    table = np.array(rows, dtype=float)
    col = energy_column if energy_column >= 0 else table.shape[1] + energy_column
    if not (0 <= col < table.shape[1]):
        raise FormatError(f"{path}: energy_column {energy_column} out of range for {table.shape[1]} columns")
    labels = np.delete(table, col, axis=1)
    return {"energies": table[:, col].copy(), "labels": labels, "layout": "table"}


def read_spinpack_def(path: str) -> Dict[str, str]:
    """`key = value` / `key value` pairs of a daten.def-style file as strings (comments after '#' dropped)."""
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                k, v = parts
            out[k.strip()] = v.strip()
    if not out:
        raise FormatError(f"{path}: no key/value lines")
    return out


def claim_from_spinpack(energies: str, layout: str = "table", energy_column: int = -1, select: Optional[Dict[int, float]] = None,
                        definition: Optional[str] = None, operator=None, states=None, kind: str = "subset", window=None,
                        n_states: Optional[int] = None, **claim_kw):
    """Claim (eigenvalues only) from a SPINPACK energy file under an assumed layout; `select` maps a label column
    index (in the table without the energy column) to a value. The default kind is 'subset' because without a
    documented layout the file is not known to list the lowest states in order; pass kind='lowest' when it does."""
    from ..certify import Claim
    r = read_spinpack_energies(energies, layout=layout, energy_column=energy_column)
    w, labels = r["energies"], r["labels"]
    if select:
        if labels is None:
            raise FormatError("select= needs the table layout (label columns)")
        mask = np.ones(w.size, dtype=bool)
        for c, v in select.items():
            if not (0 <= int(c) < labels.shape[1]):
                raise FormatError(f"select column {c} out of range for {labels.shape[1]} label columns")
            mask &= labels[:, int(c)] == float(v)
        w, labels = w[mask], labels[mask]
        if w.size == 0:
            raise FormatError(f"no lines match {select}")
    if n_states is not None:
        w = w[:n_states]
        labels = None if labels is None else labels[:n_states]
    producer = {"name": "SPINPACK", "energy_file": os.path.abspath(energies), "layout": r["layout"],
                "layout_status": SPINPACK_LAYOUT_STATUS,
                "labels": None if labels is None else labels.tolist()}
    if definition is not None:
        producer["definition"] = read_spinpack_def(definition)
    producer.update(claim_kw.pop("producer", {}) or {})
    return Claim(w, None, operator=operator, kind=kind, window=window, states=states, producer=producer, **claim_kw)
