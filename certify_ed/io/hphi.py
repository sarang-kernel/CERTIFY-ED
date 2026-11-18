"""Readers for the output files of HPhi (issp-center-dev/HPhi), roadmap F-02.

Layouts, as documented in the HPhi manual (file specification, output files; version 3.5):

  * `##_energy.dat` (Lanczos / LOBCG): keyword lines, one block per state,

        State 0
         Energy  -2.4500706750607750
         Doublon  0.0000000000000000
         Sz  0.0000000000333962

    The single-state Lanczos output has the same keyword lines without the `State n` header. The older
    one-line form `i= 0 Energy=-11.228483 N= 16.000000 Sz= 0.000000 Doublon= 0.000000` is accepted too.
  * `Eigenvalue.dat` (full diagonalization): one line per eigenvalue, `eigennumber eigenvalue`
    (whitespace separated; the manual states "an eigennumber and an eigenvalue are outputted for each line").
  * `##_eigenvec_&&_rank_$$.dat` (OutputEigenVec=1): binary, written (manual section eigenvec.dat) as

        fwrite(&number_of_iterations, sizeof(int), 1, fp);
        fwrite(&local_size, sizeof(unsigned long int), 1, fp);
        fwrite(&eigen_vector[0], sizeof(complex double), local_size + 1, fp);

    i.e. a 4-byte int, an 8-byte unsigned long (LP64; on an LLP64 platform such as Windows `unsigned long` is
    4 bytes, which this reader detects from the file size), then local_size + 1 complex doubles of which the
    first (`eigen_vector[0]`) is not used. Native byte order, little endian assumed. With MPI `local_size` is
    the rank-local part and the files of all ranks concatenate in rank order (`claim_from_hphi(ranks=...)`).

What is NOT established from the documentation: the order of HPhi's basis states relative to a certify-ed
ModelSpec encoding (HPhi's state index for a given spin/electron configuration). The reader returns the vector
components in HPhi's order; to certify against a spec the caller supplies the operator in the same order
(`operator=`) and, for a U(1) block, the state list (`states=`) in the spec's encoding. A Claim with the vectors
and an operator alone (no spec) certifies S2-S4 without S1.

Every reader checks the file against the layout and raises `FormatError` with the observed and the expected
quantities when they disagree; it never guesses a layout silently.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from .formats import FormatError

__all__ = ["read_hphi_energy", "read_hphi_eigenvalue_file", "read_hphi_eigenvec", "claim_from_hphi"]

_ENERGY_KEYS = ("Energy", "Doublon", "Sz", "N")


def read_hphi_energy(path: str) -> Dict[str, np.ndarray]:
    """Parse `##_energy.dat`: returns {'Energy': array, 'Doublon': array, 'Sz': array, 'state': array} with one
    entry per state in file order (missing keywords are NaN). Accepts the block form (`State n` + keyword lines),
    the single-state keyword form and the one-line `i= 0 Energy=... ` form."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    rows: List[Dict[str, float]] = []
    cur: Optional[Dict[str, float]] = None
    one_line = re.compile(r"^\s*i\s*=\s*(\d+)\s+(.*)$")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = one_line.match(line)
        if m:
            row = {"state": float(m.group(1))}
            for key, val in re.findall(r"([A-Za-z]+)\s*=\s*([-+0-9.eEdD]+)", m.group(2)):
                row[key] = float(val.replace("d", "e").replace("D", "E"))
            rows.append(row)
            cur = None
            continue
        m = re.match(r"^State\s+(\d+)\s*$", line)
        if m:
            cur = {"state": float(m.group(1))}
            rows.append(cur)
            continue
        m = re.match(r"^([A-Za-z_]+)\s+([-+0-9.eEdD]+)\s*$", line)
        if m:
            key, val = m.group(1), float(m.group(2).replace("d", "e").replace("D", "E"))
            if cur is None or (key in cur and key == "Energy"):
                cur = {"state": float(len(rows))}
                rows.append(cur)
            cur[key] = val
            continue
        raise FormatError(f"{path}: line {raw!r} is not an HPhi energy.dat line (expected 'State n', "
                          f"'<Keyword> <value>' or 'i= n Energy=...')")
    if not rows or not any("Energy" in r for r in rows):
        raise FormatError(f"{path}: no 'Energy' entries found (not an HPhi energy.dat file?)")
    out = {"state": np.array([r.get("state", i) for i, r in enumerate(rows)], dtype=float)}
    for k in _ENERGY_KEYS:
        out[k] = np.array([r.get(k, np.nan) for r in rows], dtype=float)
    if np.any(np.isnan(out["Energy"])):
        raise FormatError(f"{path}: a state block has no Energy line")
    return out


def read_hphi_eigenvalue_file(path: str) -> np.ndarray:
    """Parse `Eigenvalue.dat` of the full diagonalization: `eigennumber eigenvalue` per line; returns the
    eigenvalues in file order (the eigennumbers must be 0, 1, 2, ... in order)."""
    nums, vals = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise FormatError(f"{path}:{ln}: expected 'eigennumber eigenvalue', got {raw!r}")
            try:
                nums.append(int(parts[0]))
                vals.append(float(parts[1]))
            except ValueError as e:
                raise FormatError(f"{path}:{ln}: {e}") from e
    if not vals:
        raise FormatError(f"{path}: empty Eigenvalue.dat")
    if nums != list(range(len(nums))):
        raise FormatError(f"{path}: eigennumbers are not 0..{len(nums) - 1} in order (first entries {nums[:5]})")
    return np.array(vals, dtype=float)


def read_hphi_eigenvec(path: str, expected_size: Optional[int] = None) -> Dict[str, object]:
    """Read one `##_eigenvec_&&_rank_$$.dat` file: {'iterations': int, 'local_size': int, 'vector': complex array
    of length local_size, 'size_t_bytes': 8 | 4}. The unused component eigen_vector[0] is dropped. The file size
    must equal 4 + size_t_bytes + 16 * (local_size + 1) for size_t_bytes = 8 (LP64, documented) or 4 (LLP64);
    otherwise FormatError."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12:
            raise FormatError(f"{path}: file too short for the HPhi eigenvector header ({size} bytes)")
        n_iter = int(np.frombuffer(head[:4], dtype="<i4")[0])
        n8 = int(np.frombuffer(head[4:12], dtype="<u8")[0])
        n4 = int(np.frombuffer(head[4:8], dtype="<u4")[0])
        if size == 4 + 8 + 16 * (n8 + 1):
            local_size, width, offset = n8, 8, 12
        elif size == 4 + 4 + 16 * (n4 + 1):
            local_size, width, offset = n4, 4, 8
        else:
            raise FormatError(f"{path}: size {size} bytes matches neither 4 + 8 + 16*(local_size+1) with "
                              f"local_size={n8} (= {4 + 8 + 16 * (n8 + 1)}) nor the 4-byte-long variant with "
                              f"local_size={n4} (= {4 + 4 + 16 * (n4 + 1)}); not an HPhi eigenvec file, or truncated")
        f.seek(offset)
        data = np.frombuffer(f.read(16 * (local_size + 1)), dtype="<c16")
    if expected_size is not None and local_size != expected_size:
        raise FormatError(f"{path}: local_size {local_size} differs from the expected {expected_size}")
    vec = np.array(data[1:], dtype=np.complex128)
    if not np.all(np.isfinite(vec)):
        raise FormatError(f"{path}: non-finite components")
    return {"iterations": n_iter, "local_size": local_size, "vector": vec, "size_t_bytes": width}


def _vector_files(pattern_or_list, ranks: Optional[int]) -> List[List[str]]:
    """Expand the eigenvector file specification to [[rank files of state 0], [rank files of state 1], ...]."""
    if isinstance(pattern_or_list, str):
        d, base = os.path.split(pattern_or_list)
        d = d or "."
        rx = re.compile(r"^" + re.escape(base).replace(r"\{state\}", r"(\d+)").replace(r"\{rank\}", r"(\d+)") + r"$")
        if "{state}" not in base:
            raise FormatError("eigenvectors pattern must contain {state} (and {rank} for MPI output), e.g. "
                              "'output/zvo_eigenvec_{state}_rank_{rank}.dat'")
        found: Dict[int, Dict[int, str]] = {}
        for n in os.listdir(d):
            m = rx.match(n)
            if m:
                g = [int(x) for x in m.groups()]
                st, rk = (g[0], g[1]) if len(g) == 2 else (g[0], 0)
                found.setdefault(st, {})[rk] = os.path.join(d, n)
        if not found:
            raise FormatError(f"no files match {pattern_or_list}")
        out = []
        for st in sorted(found):
            rks = found[st]
            if sorted(rks) != list(range(len(rks))):
                raise FormatError(f"state {st}: rank files are not 0..{len(rks) - 1} ({sorted(rks)})")
            if ranks is not None and len(rks) != ranks:
                raise FormatError(f"state {st}: {len(rks)} rank files, expected {ranks}")
            out.append([rks[r] for r in range(len(rks))])
        if sorted(found) != list(range(len(found))):
            raise FormatError(f"eigenvector states are not 0..{len(found) - 1} ({sorted(found)})")
        return out
    out = []
    for item in pattern_or_list:
        out.append([item] if isinstance(item, str) else list(item))
    return out


def claim_from_hphi(energy: str, eigenvectors=None, operator=None, states=None, kind: str = "lowest",
                    window=None, ranks: Optional[int] = None, n_states: Optional[int] = None, **claim_kw):
    """Claim from HPhi output: `energy` is `##_energy.dat` or `Eigenvalue.dat`; `eigenvectors` a pattern with
    `{state}` (and `{rank}`) or a list of files (one per state, or a list of rank files per state); the vector
    of a state is the concatenation of its rank files in rank order. `operator` (HPhi's matrix in HPhi's state
    order, from the user) and `states` (the block's state list in the spec's encoding) are passed to the Claim.
    `n_states` keeps the first n eigenvalues (HPhi writes `exct` states; the vectors must match)."""
    from ..certify import Claim
    base = os.path.basename(energy)
    if base.lower().endswith("eigenvalue.dat"):
        w = read_hphi_eigenvalue_file(energy)
        meta = {"file": "Eigenvalue.dat"}
    else:
        e = read_hphi_energy(energy)
        w = e["Energy"]
        meta = {"file": "energy.dat", "Sz": e["Sz"].tolist(), "Doublon": e["Doublon"].tolist()}
    if n_states is not None:
        w = w[:n_states]
    X = None
    vec_meta = None
    if eigenvectors is not None:
        groups = _vector_files(eigenvectors, ranks)
        if n_states is not None:
            groups = groups[:n_states]
        if len(groups) != w.size:
            raise FormatError(f"{len(groups)} eigenvector states for {w.size} eigenvalues (use n_states= to "
                              f"select, or check the pattern)")
        cols, iters = [], []
        for files in groups:
            parts = [read_hphi_eigenvec(p) for p in files]
            cols.append(np.concatenate([p["vector"] for p in parts]))
            iters.append([p["iterations"] for p in parts])
        n = {c.size for c in cols}
        if len(n) != 1:
            raise FormatError(f"eigenvector lengths differ between states: {sorted(n)}")
        X = np.column_stack(cols)
        if np.all(X.imag == 0):
            X = X.real.copy()
        vec_meta = {"files": [[os.path.basename(p) for p in g] for g in groups], "iterations": iters,
                    "dimension": int(X.shape[0])}
    producer = {"name": "HPhi", "energy_file": os.path.abspath(energy), **meta}
    if vec_meta:
        producer["eigenvectors"] = vec_meta
    producer.update(claim_kw.pop("producer", {}) or {})
    return Claim(w, X, operator=operator, kind=kind, window=window, states=states, producer=producer, **claim_kw)
