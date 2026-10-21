"""Importers: turn a producer's output into a `Claim` (+ `ModelSpec` when possible).

Supported producers
  * QuSpin (native): hamiltonian object + basis + static list. Handles full
    bases, U(1) bases (Nup / Nf / Nb) and symmetry-adapted bases (momentum,
    parity, spin inversion, ...) via basis.get_proj.
  * Plain arrays / NPZ files: any code that can write its matrix (or not) and
    its eigenpairs; this is the route for XDiag, HPhi, custom C++/Fortran/Julia
    codes. A spec can be attached if the encoding of basis states is known.
  * HDF5 (the format Referee 1 proposed for publishing residual data):
    datasets 'eigenvalues', optional 'eigenvectors', optional CSR triplet
    'H/data', 'H/indices', 'H/indptr', 'H/shape'.
  * scipy LinearOperator / callable matvec: matrix-free producers.
  * Any file format and the producer bundle directory (certify_ed.io; docs/producer_bundle.md):
    claim_from_file, claim_from_bundle.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .certify import Claim
from .construct import StateSet
from .spec import spec_from_quspin


def _quspin_symmetry_blocks(basis) -> dict:
    blocks = dict(getattr(basis, "blocks", {}) or {})
    blocks.pop("a", None)
    return {k: v for k, v in blocks.items() if v is not None}


def claim_from_quspin(H, basis, static: Sequence, eigenvalues, eigenvectors=None,
                      kind: str = "lowest", window=None, **claim_kw):
    """Build (Claim, ModelSpec) from QuSpin objects.

    The spec is rebuilt from `static` and the basis's local space; the matrix is
    NOT taken from QuSpin for reference purposes. QuSpin's matrix is the
    producer operator that gets compared with the reference.
    """
    spec = spec_from_quspin(static, basis)
    Hq = H.tocsr() if hasattr(H, "tocsr") else sp.csr_matrix(H)
    producer = {"name": "QuSpin", "basis": type(basis).__name__, "Ns": int(basis.Ns)}
    try:
        import quspin
        producer["version"] = quspin.__version__
    except Exception:
        pass
    sym = _quspin_symmetry_blocks(basis)
    producer["symmetry_blocks"] = {k: (int(v) if np.isscalar(v) else str(v)) for k, v in sym.items()}
    if sym:
        V = basis.get_proj(np.complex128)
        V = sp.csr_matrix(V)
        return Claim(eigenvalues, eigenvectors, operator=Hq, kind=kind, window=window,
                     sector_isometry=V, producer=producer, **claim_kw), spec
    if int(basis.Ns) == spec.dim:
        return Claim(eigenvalues, eigenvectors, operator=Hq, kind=kind, window=window,
                     producer=producer, **claim_kw), spec
    states = np.sort(np.asarray(basis.states, dtype=np.int64))
    st = StateSet(states, "descending", label=f"QuSpin U(1) basis ({basis.Ns} states)")
    return Claim(eigenvalues, eigenvectors, operator=Hq, kind=kind, window=window,
                 states=st, producer=producer, **claim_kw), spec


def claim_from_arrays(eigenvalues, eigenvectors=None, operator=None, kind="subset",
                      window=None, producer=None, **kw) -> Claim:
    """Claim from in-memory arrays (eigenvalues, optional eigenvectors and operator)."""
    return Claim(eigenvalues, eigenvectors, operator=operator, kind=kind, window=window,
                 producer=producer or {"name": "arrays"}, **kw)


def claim_from_npz(path: str, kind="subset", window=None, **kw) -> Claim:
    """NPZ with keys: eigenvalues [, eigenvectors] [, H_data, H_indices, H_indptr, H_shape | H]."""
    z = np.load(path, allow_pickle=False)
    op = None
    if "H" in z.files:
        op = z["H"]
    elif "H_data" in z.files:
        op = sp.csr_matrix((z["H_data"], z["H_indices"], z["H_indptr"]), shape=tuple(z["H_shape"]))
    ev = z["eigenvalues"]
    X = z["eigenvectors"] if "eigenvectors" in z.files else None
    return Claim(ev, X, operator=op, kind=kind, window=window,
                 producer={"name": "npz", "path": path}, **kw)


def claim_from_hdf5(path: str, kind="subset", window=None, **kw) -> Claim:
    """HDF5 with datasets eigenvalues [, eigenvectors] [, group H with data/indices/indptr/shape]."""
    import h5py
    with h5py.File(path, "r") as f:
        ev = f["eigenvalues"][...]
        X = f["eigenvectors"][...] if "eigenvectors" in f else None
        op = None
        if "H" in f and isinstance(f["H"], h5py.Group):
            g = f["H"]
            op = sp.csr_matrix((g["data"][...], g["indices"][...], g["indptr"][...]),
                               shape=tuple(g["shape"][...]))
    return Claim(ev, X, operator=op, kind=kind, window=window,
                 producer={"name": "hdf5", "path": path}, **kw)


def export_npz(path: str, eigenvalues, eigenvectors=None, H=None) -> str:
    """Helper for producers in other languages' Python bindings or post-processing."""
    d = {"eigenvalues": np.asarray(eigenvalues)}
    if eigenvectors is not None:
        d["eigenvectors"] = np.asarray(eigenvectors)
    if H is not None:
        H = sp.csr_matrix(H)
        d.update(H_data=H.data, H_indices=H.indices, H_indptr=H.indptr, H_shape=np.array(H.shape))
    np.savez(path, **d)
    return path

def claim_from_file(eigenvalues: str, eigenvectors: str = None, operator: str = None, basis: str = None,
                    kind: str = "subset", window=None, keys: dict = None, allow_code: bool = False, **kw) -> Claim:
    """Claim from separate files in any supported format (certify_ed.io.formats): .npy .npz .h5 .csv .tsv
    .txt .dat .json .mtx (operator also .py with allow_code=True). This is `dd if=` for single files; a
    directory of such files is a producer bundle (claim_from_bundle)."""
    from .io.formats import read_array, read_basis, read_operator
    keys = keys or {}
    ev = np.asarray(read_array(eigenvalues, keys.get("eigenvalues"), "eigenvalues"), dtype=float)
    X = None if eigenvectors is None else np.asarray(read_array(eigenvectors, keys.get("eigenvectors"), "eigenvectors"))
    op = None if operator is None else read_operator(operator, keys.get("operator"), allow_code=allow_code)
    states, iso = None, None
    if basis is not None:
        kind_b, B = read_basis(basis, keys.get("basis"))
        if kind_b == "states":
            states = StateSet(np.sort(B), kw.pop("state_order", "descending"), label="producer basis")
        else:
            iso = B
    return Claim(ev, X, operator=op, kind=kind, window=window, states=states, sector_isometry=iso,
                 producer={"name": "files", "eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
                           "operator": operator, "basis": basis}, **kw)


def claim_from_bundle(path: str, allow_code: bool = False, kind: str = None, window=None):
    """(Claim, ModelSpec | None, meta) from a producer bundle directory (docs/producer_bundle.md)."""
    from .io.bundle import read_bundle
    b = read_bundle(path, allow_code=allow_code, kind=kind, window=window)
    return b.claim, b.spec, b.meta

