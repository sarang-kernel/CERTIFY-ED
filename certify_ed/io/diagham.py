"""Readers for DiagHam output (nick-ux.org/diagham), roadmap F-02.

Layouts:

  * Binary vectors `*.vec` (documented, DiagHam wiki "Binary vectors"): little endian; for real vectors with
    fewer than 2^31 components, a 4-byte int (the dimension) followed by dimension doubles. The wiki's own
    reading example is `np.fromfile(f, '<i4', count=1)` then `np.fromfile(f, '<f8', count=dim)`.
    Complex vectors (ComplexVector::WriteVector) are ASSUMED to be the same header followed by 2 * dimension
    doubles with real and imaginary parts interleaved, by analogy with the documented binary matrix format
    (Re, Im consecutive). The reader decides real vs complex from the file size and refuses files whose size
    matches neither; `complex=True/False` forces the interpretation (still size-checked). Vectors with 2^31
    components or more (a different header) are not supported.
  * Spectrum text files `*.dat` written by the diagonalization programs: one line per eigenvalue, whitespace
    separated columns of quantum numbers followed by the energy in the last column, optional `#` comment or
    header lines (for instance `# Sz E` or `# Lz E`). The exact set of columns depends on the program
    (Sz; Lz; Kx Ky; ...): `read_diagham_spectrum` returns the numeric table and the header names when present,
    and `claim_from_diagham(..., select={"Sz": 0})` keeps one sector. The column convention (quantum numbers
    first, energy last) is the one of DiagHam's spectrum files and of its `FindGroundState`-style tools; it is
    an observation of the output of the spin and FQHE programs, not a statement of the manual.

The state order of a DiagHam basis relative to a certify-ed encoding is not established: as for HPhi, the
caller supplies the operator in DiagHam's order (`operator=`) and, for a U(1) block, the state list.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .formats import FormatError

__all__ = ["read_diagham_vector", "read_diagham_spectrum", "claim_from_diagham"]


def read_diagham_vector(path: str, complex: Optional[bool] = None, header_bytes: int = 4) -> np.ndarray:
    """Read a DiagHam binary vector. `header_bytes` is 4 (documented int header); 8 is accepted for files
    written with a 64-bit dimension (not documented; size-checked like the rest)."""
    if header_bytes not in (4, 8):
        raise ValueError("header_bytes must be 4 or 8")
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(header_bytes)
        if len(head) < header_bytes:
            raise FormatError(f"{path}: file too short ({size} bytes) for a {header_bytes}-byte dimension header")
        dim = int(np.frombuffer(head, dtype="<i4" if header_bytes == 4 else "<i8")[0])
        if dim <= 0:
            raise FormatError(f"{path}: dimension header {dim} (vectors with >= 2^31 components or a different "
                              f"header are not supported)")
        n_real = header_bytes + 8 * dim
        n_cplx = header_bytes + 16 * dim
        if complex is None:
            if size == n_real:
                complex = False
            elif size == n_cplx:
                complex = True
            else:
                raise FormatError(f"{path}: size {size} bytes is neither {n_real} (real, dimension {dim}) nor "
                                  f"{n_cplx} (complex, dimension {dim}); not a DiagHam vector, or truncated")
        else:
            want = n_cplx if complex else n_real
            if size != want:
                raise FormatError(f"{path}: size {size} bytes, expected {want} for a {'complex' if complex else 'real'} "
                                  f"vector of dimension {dim}")
        data = np.frombuffer(f.read(size - header_bytes), dtype="<f8")
    if complex:
        v = data[0::2] + 1j * data[1::2]
    else:
        v = np.array(data, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise FormatError(f"{path}: non-finite components")
    return v


def read_diagham_spectrum(path: str) -> Dict[str, object]:
    """Parse a DiagHam spectrum text file: {'columns': [names or None], 'table': (n_lines x n_columns) float
    array, 'energies': last column}. All data lines must have the same number of columns."""
    names: Optional[List[str]] = None
    rows: List[List[float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                toks = line.lstrip("#").split()
                if toks and names is None and not rows and not all(_is_number(t) for t in toks):
                    names = toks
                continue
            toks = line.split()
            try:
                rows.append([float(t) for t in toks])
            except ValueError as e:
                raise FormatError(f"{path}:{ln}: non-numeric entry in {raw!r} ({e})") from e
    if not rows:
        raise FormatError(f"{path}: no data lines")
    width = {len(r) for r in rows}
    if len(width) != 1:
        raise FormatError(f"{path}: lines have different numbers of columns {sorted(width)}")
    table = np.array(rows, dtype=float)
    if names is not None and len(names) != table.shape[1]:
        names = None                     # a comment line that was not a header
    return {"columns": names, "table": table, "energies": table[:, -1].copy()}


def _is_number(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def claim_from_diagham(spectrum: str, vectors: Optional[Sequence[str]] = None, select: Optional[Dict] = None,
                       operator=None, states=None, kind: str = "lowest", window=None, n_states: Optional[int] = None,
                       complex: Optional[bool] = None, **claim_kw):
    """Claim from a DiagHam spectrum file and optional binary vectors (one file per eigenvalue, in the order of
    the selected lines). `select` maps a column name (from the `#` header) or a 0-based column index to the value
    to keep, e.g. {"Sz": 0}; the lines are kept in file order (DiagHam writes them sorted by energy within a
    sector; `kind='lowest'` needs that order and `certify()` checks it)."""
    from ..certify import Claim
    sp_ = read_diagham_spectrum(spectrum)
    table, names = sp_["table"], sp_["columns"]
    mask = np.ones(table.shape[0], dtype=bool)
    for key, val in (select or {}).items():
        if isinstance(key, str):
            if names is None or key not in names:
                raise FormatError(f"{spectrum}: no column named {key!r} (header: {names})")
            col = names.index(key)
        else:
            col = int(key)
        if col >= table.shape[1] - 1:
            raise FormatError(f"{spectrum}: column {col} is the energy column or beyond")
        mask &= table[:, col] == float(val)
    w = table[mask, -1]
    if w.size == 0:
        raise FormatError(f"{spectrum}: no lines match {select}")
    if n_states is not None:
        w = w[:n_states]
    X = None
    if vectors is not None:
        vectors = list(vectors)
        if n_states is not None:
            vectors = vectors[:n_states]
        if len(vectors) != w.size:
            raise FormatError(f"{len(vectors)} vector files for {w.size} selected eigenvalues")
        cols = [read_diagham_vector(p, complex=complex) for p in vectors]
        dims = {c.size for c in cols}
        if len(dims) != 1:
            raise FormatError(f"vector dimensions differ between files: {sorted(dims)}")
        X = np.column_stack(cols)
    producer = {"name": "DiagHam", "spectrum_file": os.path.abspath(spectrum), "columns": names,
                "select": {str(k): v for k, v in (select or {}).items()},
                "quantum_numbers": table[mask][: w.size, :-1].tolist()}
    if vectors is not None:
        producer["vector_files"] = [os.path.basename(p) for p in vectors]
    producer.update(claim_kw.pop("producer", {}) or {})
    return Claim(w, X, operator=operator, kind=kind, window=window, states=states, producer=producer, **claim_kw)
