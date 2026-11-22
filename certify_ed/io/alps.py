"""Reader for the HDF5 output of the ALPS diagonalization applications (sparsediag, fulldiag), roadmap F-02.

Layout (ALPS 2.x `*.task*.out.h5`, as used by `pyalps.loadSpectra` / `loadEigenstateMeasurements`, and the
2026 `SAVE_EIGENVECTORS` addition, ALPSim/ALPS pull request 168):

    /spectrum/sectors/<n>/energies                 1-D float, the eigenvalues of quantum-number sector n
    /spectrum/sectors/<n>/quantumnumbers/<name>    scalar per conserved quantum number (Sz, N, TOTAL_MOMENTUM, ...)
    /spectrum/sectors/<n>/eigenvectors/shape       [n_eigenvectors, dim]              (SAVE_EIGENVECTORS=1)
    /spectrum/sectors/<n>/eigenvectors/values      flat row-major coefficients, real or complex
    /spectrum/sectors/<n>/basis/shape              [dim, n_sites]                     (SAVE_EIGENVECTORS=1)
    /spectrum/sectors/<n>/basis/values             flat row-major local-state indices per basis state
    /spectrum/site_type, /spectrum/site_basis/<type>/<qn>   local-state quantum numbers (not used here)

`energies` and `quantumnumbers` are the documented pyalps layout; the eigenvector and basis datasets are from
the pull-request text (shape + values pairs). The reader checks that `values` has shape[0] * shape[1] entries
and that the number of eigenvectors equals the number of energies. Local-state indices are returned as an
array; mapping them to a certify-ed encoding (digit k of site i) is left to the caller
(`states_from_alps_basis` does it for the common case where ALPS's local-state index equals the spec's digit,
which holds for spin-S sites listed Sz = -S .. S and for boson/fermion occupation numbers).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..precision import carries_imaginary_parts, claimed_reals, dropped_imaginary_parts
from .formats import FormatError

__all__ = ["read_alps_spectrum", "states_from_alps_basis", "claim_from_alps"]


def _dataset(f, path: str):
    if path not in f:
        raise FormatError(f"{f.filename}: no dataset {path}")
    return f[path][...]


def _energies(ds, path: str, sector: str) -> np.ndarray:
    """The energies of one sector as a 1-D array. Complex numbers are read as h5py gives them (a compound of r and i,
    numpy complex) or in the layout of the ALPS HDF5 archive, a trailing dimension of 2 (real, imaginary) with the
    attribute '__complex__'; any other array of more than one dimension is refused, since flattening it would make
    its components count as energies."""
    a = np.asarray(ds[...])
    if a.dtype.names and set(a.dtype.names) >= {"r", "i"}:
        a = a["r"] + 1j * a["i"]
    if "__complex__" in ds.attrs and a.ndim >= 1 and a.shape[-1] == 2 and a.dtype.kind in "fiu":
        a = a[..., 0] + 1j * a[..., 1]
    if sum(1 for s in a.shape if s != 1) > 1:
        raise FormatError(f"{path}: sector {sector}: energies of shape {a.shape} (expected one dimension)")
    return a.reshape(-1)


def read_alps_spectrum(path: str, with_vectors: bool = True) -> List[Dict[str, object]]:
    """All sectors of an ALPS out.h5: [{'index': n, 'energies': array, 'quantumnumbers': {name: value},
    'eigenvectors': (dim x n) array or None, 'basis': (dim x n_sites) int array or None,
    'energies_as_given': array, 'dropped_imaginary_parts': record or None}] in sector order.

    'energies' are binary64 reals (precision.claimed_reals). Energies stored as complex numbers with a nonzero
    imaginary part are also kept as stored ('energies_as_given'; otherwise it is 'energies'), and
    'dropped_imaginary_parts' records the count and largest magnitude of the finite nonzero imaginary parts that
    'energies' drops; claim_from_alps passes the stored values, so that certify records the same."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("reading ALPS output needs h5py: pip install 'certify-ed[hdf5]'") from e
    out = []
    with h5py.File(path, "r") as f:
        if "spectrum" not in f or "sectors" not in f["spectrum"]:
            raise FormatError(f"{path}: no /spectrum/sectors group (not an ALPS diagonalization output file)")
        sectors = f["spectrum/sectors"]
        keys = sorted(sectors.keys(), key=lambda k: (len(k), k))
        if not keys:
            raise FormatError(f"{path}: /spectrum/sectors is empty")
        for k in keys:
            g = sectors[k]
            if "energies" not in g:
                raise FormatError(f"{path}: /spectrum/sectors/{k} has no 'energies' dataset")
            raw = _energies(g["energies"], path, k)
            w = claimed_reals(raw)
            if not np.all(np.isfinite(w)):
                raise FormatError(f"{path}: sector {k}: non-finite energies")
            given = raw if carries_imaginary_parts(raw) else w
            qn: Dict[str, float] = {}
            if "quantumnumbers" in g:
                for name in g["quantumnumbers"]:
                    v = g["quantumnumbers"][name][...]
                    qn[name] = float(np.asarray(v).reshape(-1)[0]) if np.asarray(v).size == 1 else np.asarray(v).tolist()
            X = basis = None
            if with_vectors and "eigenvectors" in g:
                shape = tuple(int(s) for s in np.asarray(g["eigenvectors/shape"][...]).reshape(-1))
                vals = np.asarray(g["eigenvectors/values"][...])
                if len(shape) != 2:
                    raise FormatError(f"{path}: sector {k}: eigenvectors/shape {shape} is not [n_vectors, dim]")
                if vals.dtype.names and set(vals.dtype.names) >= {"r", "i"}:
                    vals = vals["r"] + 1j * vals["i"]
                vals = vals.reshape(-1)
                if vals.size != shape[0] * shape[1]:
                    raise FormatError(f"{path}: sector {k}: eigenvectors/values has {vals.size} entries, shape "
                                      f"{shape} needs {shape[0] * shape[1]}")
                if shape[0] != w.size:
                    raise FormatError(f"{path}: sector {k}: {shape[0]} eigenvectors for {w.size} energies")
                X = vals.reshape(shape).T.copy()                      # columns are the vectors
                if not np.all(np.isfinite(X)):
                    raise FormatError(f"{path}: sector {k}: non-finite eigenvector entries")
                if "basis" in g:
                    bshape = tuple(int(s) for s in np.asarray(g["basis/shape"][...]).reshape(-1))
                    bvals = np.asarray(g["basis/values"][...]).reshape(-1)
                    if len(bshape) != 2 or bvals.size != bshape[0] * bshape[1]:
                        raise FormatError(f"{path}: sector {k}: basis/shape {bshape} does not match "
                                          f"{bvals.size} values")
                    if bshape[0] != shape[1]:
                        raise FormatError(f"{path}: sector {k}: basis has {bshape[0]} states, vectors have "
                                          f"dimension {shape[1]}")
                    basis = bvals.reshape(bshape).astype(np.int64)
            out.append({"index": k, "energies": w, "quantumnumbers": qn, "eigenvectors": X, "basis": basis,
                        "energies_as_given": given, "dropped_imaginary_parts": dropped_imaginary_parts(raw)})
    return out


def states_from_alps_basis(basis: np.ndarray, q: int, state_order: str = "descending"):
    """StateSet from an ALPS basis table (dim x n_sites local-state indices), assuming index = spec digit and
    site 0 the most significant digit (spec.site_order 'site0_msb'). Returns (StateSet, permutation) where
    permutation[j] is the row of the ALPS table that holds the j-th state of the StateSet, so that
    X[permutation] reorders ALPS vectors into the StateSet's order."""
    from ..construct import StateSet
    basis = np.asarray(basis, dtype=np.int64)
    if basis.ndim != 2:
        raise FormatError("basis must be dim x n_sites")
    if np.any(basis < 0) or np.any(basis >= q):
        raise FormatError(f"basis holds local-state indices outside 0..{q - 1}")
    n_sites = basis.shape[1]
    weights = q ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    ints = basis @ weights
    if np.unique(ints).size != ints.size:
        raise FormatError("basis table has repeated states")
    st = StateSet(np.sort(ints), state_order, label=f"ALPS basis ({ints.size} states)")
    # StateSet index j <-> integer st.states[j]; find the ALPS row for each
    order_alps = np.argsort(ints)
    ints_sorted = ints[order_alps]
    pos = np.searchsorted(ints_sorted, st.states_in_order())
    perm = order_alps[pos]
    return st, perm


def claim_from_alps(path: str, sector: int = 0, spec=None, operator=None, states=None, kind: str = "lowest",
                    window=None, n_states: Optional[int] = None, **claim_kw):
    """Claim from one sector of an ALPS out.h5. With `spec` and a saved basis table the vectors are reordered
    into the spec's state encoding and the Claim carries the block's StateSet (then S1 can run against
    `operator=` or the spec's own matrix). Without a basis table the vectors are passed in ALPS's order and the
    caller supplies `operator`/`states`."""
    from ..certify import Claim
    secs = read_alps_spectrum(path)
    if not (0 <= sector < len(secs)):
        raise FormatError(f"{path}: sector {sector} out of range (file has {len(secs)} sectors)")
    s = secs[sector]
    w, X = s["energies_as_given"], s["eigenvectors"]     # values with a nonzero imaginary part as stored (certify
    #                                                     records the parts it drops), real values as before
    if n_states is not None:
        w = w[:n_states]
        X = None if X is None else X[:, :n_states]
    producer = {"name": "ALPS", "file": os.path.abspath(path), "sector": s["index"], "quantumnumbers": s["quantumnumbers"],
                "n_sectors": len(secs), "all_sector_quantumnumbers": [t["quantumnumbers"] for t in secs]}
    if spec is not None and s["basis"] is not None and states is None:
        st, perm = states_from_alps_basis(s["basis"], spec.q, spec.state_order)
        states = st
        if X is not None:
            X = X[perm]
        if operator is not None:
            operator = operator[perm][:, perm]
        producer["basis"] = "from /spectrum/sectors/basis, reordered into the spec encoding"
    producer.update(claim_kw.pop("producer", {}) or {})
    return Claim(w, X, operator=operator, kind=kind, window=window, states=states, producer=producer, **claim_kw)
