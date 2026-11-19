"""Synthetic fixture writers for the output formats of HPhi, DiagHam, SPINPACK and ALPS (roadmap F-02).

Each writer follows the layout documented in the corresponding reader module (certify_ed/io/hphi.py,
diagham.py, spinpack.py, alps.py), including the layouts marked as assumptions there. They exist so that the
readers can be tested without the codes; they are not the codes' own writers, so a test passing here shows
that reader and writer agree on the documented layout, not that the documentation is right.
"""
from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------------- HPhi
def write_hphi_energy(path: str, energies: Sequence[float], sz: Optional[Sequence[float]] = None,
                      doublon: Optional[Sequence[float]] = None, form: str = "block") -> str:
    """`##_energy.dat`. form: 'block' (State n + keyword lines, LOBCG), 'single' (keyword lines only, Lanczos,
    one state), 'oneline' (the old `i= 0 Energy=... N=... Sz=... Doublon=...` line)."""
    energies = [float(e) for e in energies]
    sz = [0.0] * len(energies) if sz is None else list(sz)
    doublon = [0.0] * len(energies) if doublon is None else list(doublon)
    lines = []
    if form == "block":
        for i, (e, s, d) in enumerate(zip(energies, sz, doublon)):
            lines += [f"State {i}", f" Energy  {e:.16f}", f" Doublon  {d:.16f}", f" Sz  {s:.16f}", ""]
    elif form == "single":
        if len(energies) != 1:
            raise ValueError("the single-state form holds one state")
        lines = [f" Energy  {energies[0]:.16f}", f" Doublon  {doublon[0]:.16f}", f" Sz  {sz[0]:.16f}"]
    elif form == "oneline":
        for i, (e, s, d) in enumerate(zip(energies, sz, doublon)):
            lines.append(f"i= {i} Energy={e:.6f} N= 16.000000 Sz= {s:.6f} Doublon= {d:.6f}")
    else:
        raise ValueError(form)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_hphi_eigenvalue_file(path: str, energies: Sequence[float]) -> str:
    """`Eigenvalue.dat`: eigennumber and eigenvalue per line."""
    with open(path, "w", newline="\n") as f:
        for i, e in enumerate(energies):
            f.write(f"{i} {float(e):.16e}\n")
    return path


def write_hphi_eigenvec(path: str, vector: np.ndarray, iterations: int = 100, size_t_bytes: int = 8,
                        unused_first: complex = 0.0) -> str:
    """`##_eigenvec_&&_rank_$$.dat`: int32 iterations, uint{64|32} local_size, complex128 x (local_size + 1),
    the first component unused."""
    v = np.asarray(vector, dtype=np.complex128).reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", int(iterations)))
        f.write(struct.pack("<Q" if size_t_bytes == 8 else "<I", int(v.size)))
        np.concatenate([[np.complex128(unused_first)], v]).astype("<c16").tofile(f)
    return path


def write_hphi_run(dirpath: str, energies: Sequence[float], vectors: Optional[np.ndarray] = None,
                   prefix: str = "zvo", ranks: int = 1, **kw) -> Dict[str, str]:
    """A directory like HPhi's `output/`: `<prefix>_energy.dat` and `<prefix>_eigenvec_<n>_rank_<r>.dat`;
    with ranks > 1 each vector is split into contiguous rank-local parts."""
    os.makedirs(dirpath, exist_ok=True)
    out = {"energy": write_hphi_energy(os.path.join(dirpath, f"{prefix}_energy.dat"), energies, **kw)}
    if vectors is not None:
        V = np.asarray(vectors)
        parts = np.array_split(np.arange(V.shape[0]), ranks)
        for n in range(V.shape[1]):
            for r, idx in enumerate(parts):
                write_hphi_eigenvec(os.path.join(dirpath, f"{prefix}_eigenvec_{n}_rank_{r}.dat"), V[idx, n])
        out["pattern"] = os.path.join(dirpath, f"{prefix}_eigenvec_{{state}}_rank_{{rank}}.dat")
    return out


# --------------------------------------------------------------------------- DiagHam
def write_diagham_vector(path: str, vector: np.ndarray, header_bytes: int = 4) -> str:
    """`*.vec`: int32 dimension (int64 with header_bytes=8), then dimension doubles (real) or 2 x dimension
    doubles Re, Im interleaved (complex)."""
    v = np.asarray(vector).reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<i" if header_bytes == 4 else "<q", int(v.size)))
        if np.iscomplexobj(v):
            np.column_stack([v.real, v.imag]).reshape(-1).astype("<f8").tofile(f)
        else:
            v.astype("<f8").tofile(f)
    return path


def write_diagham_spectrum(path: str, table: np.ndarray, columns: Optional[Sequence[str]] = None) -> str:
    """Spectrum `*.dat`: optional `# col1 col2 ... E` header, then one line per eigenvalue with the quantum
    numbers in the leading columns and the energy last."""
    table = np.asarray(table, dtype=float)
    with open(path, "w", newline="\n") as f:
        if columns is not None:
            f.write("# " + " ".join(columns) + "\n")
        for row in table:
            cells = [f"{int(x)}" if float(x).is_integer() else f"{x:.6g}" for x in row[:-1]]
            f.write(" ".join(cells + [f"{row[-1]:.16e}"]) + "\n")
    return path


# -------------------------------------------------------------------------- SPINPACK
def write_spinpack_table(path: str, energies: Sequence[float], labels: Optional[np.ndarray] = None,
                         comment: str = "# sz k energy (assumed SPINPACK table layout)") -> str:
    energies = np.asarray(energies, dtype=float)
    labels = np.zeros((energies.size, 0)) if labels is None else np.asarray(labels, dtype=float)
    with open(path, "w", newline="\n") as f:
        if comment:
            f.write(comment + "\n")
        for lab, e in zip(labels, energies):
            f.write(" ".join(f"{int(x)}" if float(x).is_integer() else f"{x:g}" for x in lab) + f" {e:.16e}\n")
    return path


def write_spinpack_keyvalue(path: str, energies: Sequence[float]) -> str:
    with open(path, "w", newline="\n") as f:
        f.write("# spinpack-like log (assumed keyvalue layout)\n")
        for i, e in enumerate(energies):
            f.write(f"n={i} E{i}={float(e):.16e} iter=12\n")
    return path


def write_spinpack_def(path: str, n_sites: int, spin: str = "1/2", couplings: Sequence[str] = ()) -> str:
    with open(path, "w", newline="\n") as f:
        f.write(f"# daten.def-style model file (fixture)\nN = {n_sites}\nspin = {spin}\n")
        for i, c in enumerate(couplings):
            f.write(f"J{i} = {c}\n")
    return path


# ------------------------------------------------------------------------------ ALPS
def write_alps_h5(path: str, sectors: List[Dict], complex_vectors: bool = False) -> str:
    """`*.out.h5` with /spectrum/sectors/<n>/{energies, quantumnumbers/<name>, eigenvectors/{shape,values},
    basis/{shape,values}}. Each sector dict: energies (k,), quantumnumbers {name: value}, optional
    eigenvectors (dim x k, columns are vectors) and basis (dim x n_sites local-state indices)."""
    import h5py
    with h5py.File(path, "w") as f:
        for n, s in enumerate(sectors):
            g = f.create_group(f"spectrum/sectors/{n}")
            g.create_dataset("energies", data=np.asarray(s["energies"], dtype=float))
            for name, val in (s.get("quantumnumbers") or {}).items():
                g.create_dataset(f"quantumnumbers/{name}", data=np.float64(val))
            if s.get("eigenvectors") is not None:
                X = np.asarray(s["eigenvectors"])
                rows = X.T                                            # [eigenvector][basis state]
                g.create_dataset("eigenvectors/shape", data=np.array(rows.shape, dtype=np.int64))
                if complex_vectors or np.iscomplexobj(rows):
                    g.create_dataset("eigenvectors/values", data=rows.reshape(-1).astype(np.complex128))
                else:
                    g.create_dataset("eigenvectors/values", data=rows.reshape(-1).astype(float))
            if s.get("basis") is not None:
                B = np.asarray(s["basis"], dtype=np.int64)
                g.create_dataset("basis/shape", data=np.array(B.shape, dtype=np.int64))
                g.create_dataset("basis/values", data=B.reshape(-1))
    return path
