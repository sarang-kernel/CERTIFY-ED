"""Readers and writers for the file formats producers actually write (the `dd if=` side).

The reader accepts what a lab produces and decides by extension:

  arrays (eigenvalues, eigenvectors, basis state lists, isometries)
    .npy                      one array
    .npz                      array under `key` (default: a conventional name, else the only array)
    .h5 .hdf5 .hdf            dataset under `key` (default: conventional names, see DEFAULT_KEYS)
    .csv .tsv .txt .dat       numeric columns; '#'/'%' comments and one header line are skipped;
                              complex entries as Python literals (1+2j); `complex_pairs=True` reads a
                              trailing (re, im) column pair
    .json                     a list (nested for 2-D), or an object with the array under `key`;
                              complex entries as {"re": .., "im": ..} or strings "1+2j"
    .mtx                      Matrix Market (dense or coordinate)

  operators (the producer's matrix)
    .npy                      dense
    .npz                      scipy.sparse.save_npz output, or CSR triplet H_data/H_indices/H_indptr/H_shape,
                              or COO row/col/data/shape, or a dense array (key 'H' or the only array)
    .mtx                      Matrix Market
    .h5 .hdf5                 group with data/indices/indptr/shape (CSR) or row/col/data/shape (COO),
                              or a dense dataset
    .csv .tsv .txt .dat .json dense table; or COO triplets i j value [imag] with layout='coo'
                              (JSON: {"format": "coo"|"csr", "shape": [n, n], ...})
    .py                       matrix-free: a module defining `matvec(x)` and `shape` (or `dim`), or a
                              function `operator()` returning a scipy LinearOperator; loaded only
                              with allow_code=True because it executes the producer's code

Everything non-finite is rejected at the door (review IMP-21: a NaN eigenvalue must not reach the
enclosure code).
"""
from __future__ import annotations

import importlib.util
import json
import os
import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp

ARRAY_EXT = (".npy", ".npz", ".h5", ".hdf5", ".hdf", ".csv", ".tsv", ".txt", ".dat", ".json", ".mtx")
OPERATOR_EXT = ARRAY_EXT + (".py",)

# conventional dataset / key names tried when `key` is not given
DEFAULT_KEYS = {
    "eigenvalues": ("eigenvalues", "evals", "E", "energies", "w", "values", "lambda"),
    "eigenvectors": ("eigenvectors", "evecs", "V", "vectors", "psi", "X"),
    "operator": ("H", "operator", "matrix", "A", "hamiltonian"),
    "basis": ("basis", "states", "isometry", "V", "projector", "P"),
}

__all__ = ["read_array", "read_operator", "read_basis", "write_array", "write_operator", "write_basis",
           "sniff_kind", "DEFAULT_KEYS", "ARRAY_EXT", "OPERATOR_EXT", "FormatError"]


class FormatError(ValueError):
    pass


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _finite(a, what: str):
    data = a.data if sp.issparse(a) else np.asarray(a)
    if data.dtype.kind in "fc" and not np.all(np.isfinite(data)):
        raise FormatError(f"{what}: non-finite entries (NaN/inf) are not accepted")
    return a


def _pick_key(available, key: Optional[str], role: str, path: str):
    available = list(available)
    if key is not None:
        if key not in available:
            raise FormatError(f"{path}: no entry {key!r}; available: {available}")
        return key
    for k in DEFAULT_KEYS.get(role, ()):
        if k in available:
            return k
    if len(available) == 1:
        return available[0]
    raise FormatError(f"{path}: give key= (available: {available})")


def _mmread(path: str):
    from scipy.io import mmread
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)      # scipy's spmatrix/sparray default change
        return mmread(path)


# ------------------------------------------------------------------------------------------ text
def _conv(p: str):
    try:
        return float(p)
    except ValueError:
        return complex(p.replace("i", "j").replace(" ", ""))


def _parse_text_table(path: str, delimiter: Optional[str]) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("%"):
                continue
            parts = s.split(delimiter) if delimiter else s.replace(",", " ").split()
            rows.append([p.strip() for p in parts if p.strip() != ""])
    if not rows:
        raise FormatError(f"{path}: no numeric rows")
    try:
        [_conv(p) for p in rows[0]]
    except ValueError:
        rows = rows[1:]          # one header line
    if not rows:
        raise FormatError(f"{path}: header only")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise FormatError(f"{path}: ragged rows")
    vals = [[_conv(p) for p in r] for r in rows]
    cplx = any(isinstance(v, complex) for r in vals for v in r)
    return np.array(vals, dtype=complex if cplx else float)


def _text_delimiter(path: str) -> Optional[str]:
    return {".csv": ",", ".tsv": "\t"}.get(_ext(path))


def _write_text(path: str, a: np.ndarray, delimiter: str, header: str = ""):
    a = np.asarray(a)
    rows = a if a.ndim == 2 else a.reshape(-1, 1)
    with open(path, "w") as f:
        if header:
            f.write("# " + header + "\n")
        for r in rows:
            f.write(delimiter.join(repr(complex(v)) if np.iscomplexobj(a) else repr(float(v)) for v in r) + "\n")


# ------------------------------------------------------------------------------------------ json
def _map_nested(obj, fn):
    if isinstance(obj, list):
        return [_map_nested(x, fn) for x in obj]
    return fn(obj)


def _json_to_array(obj) -> np.ndarray:
    def conv(v):
        if isinstance(v, str):
            return complex(v.replace("i", "j").replace(" ", ""))
        if isinstance(v, dict) and "re" in v:
            return complex(float(v["re"]), float(v.get("im", 0.0)))
        return v
    return np.array(_map_nested(obj, conv))


def _jsonable_array(a: np.ndarray):
    if np.iscomplexobj(a):
        return _map_nested(a.tolist(), lambda z: {"re": z.real, "im": z.imag})
    return a.tolist()


def _pairs_to_complex(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim < 1 or a.shape[-1] != 2:
        raise FormatError("complex pairs need a trailing axis of length 2")
    return a[..., 0] + 1j * a[..., 1]


# ------------------------------------------------------------------------------------------ arrays
def read_array(path: str, key: Optional[str] = None, role: str = "eigenvalues",
               complex_pairs: bool = False) -> np.ndarray:
    """Read one numeric array (1-D or 2-D) from any supported file; see the module docstring."""
    e = _ext(path)
    if e == ".npy":
        a = np.load(path, allow_pickle=False)
    elif e == ".npz":
        with np.load(path, allow_pickle=False) as z:
            a = z[_pick_key(z.files, key, role, path)]
    elif e in (".h5", ".hdf5", ".hdf"):
        import h5py
        with h5py.File(path, "r") as f:
            names = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            k = key if (key is not None and key in f) else _pick_key(names, key, role, path)
            a = f[k][...]
    elif e in (".csv", ".tsv", ".txt", ".dat"):
        a = _parse_text_table(path, _text_delimiter(path))
        if a.ndim == 2 and a.shape[1] == 1 and not complex_pairs:
            a = a[:, 0]
    elif e == ".json":
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            k = _pick_key([k for k, v in obj.items() if isinstance(v, list)], key, role, path)
            obj = obj[k]
        a = _json_to_array(obj)
    elif e == ".mtx":
        a = _mmread(path)
        a = a.toarray() if sp.issparse(a) else np.asarray(a)
        if a.ndim == 2 and a.shape[1] == 1 and role != "eigenvectors":
            a = a.ravel()
    else:
        raise FormatError(f"{path}: unsupported extension {e!r} (known: {ARRAY_EXT})")
    if complex_pairs:
        a = _pairs_to_complex(a)
    a = np.asarray(a)
    if a.dtype.kind not in "fciub":
        raise FormatError(f"{path}: non-numeric data (dtype {a.dtype})")
    return _finite(a, path)


def write_array(path: str, a, key: Optional[str] = None, role: str = "eigenvalues") -> str:
    """Write an array in the format given by the extension (the inverse of read_array)."""
    a = np.asarray(a)
    e = _ext(path)
    key = key or DEFAULT_KEYS[role][0]
    if e == ".npy":
        np.save(path, a)
    elif e == ".npz":
        np.savez(path, **{key: a})
    elif e in (".h5", ".hdf5", ".hdf"):
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset(key, data=a)
    elif e in (".csv", ".tsv", ".txt", ".dat"):
        _write_text(path, a, {".csv": ",", ".tsv": "\t"}.get(e, " "), header=key)
    elif e == ".json":
        with open(path, "w") as f:
            json.dump({key: _jsonable_array(a)}, f)
    elif e == ".mtx":
        from scipy.io import mmwrite
        mmwrite(path, a.reshape(-1, 1) if a.ndim == 1 else a)
    else:
        raise FormatError(f"{path}: unsupported extension {e!r}")
    return path


# ------------------------------------------------------------------------------------------ operators
def _csr_from_group(g):
    keys = set(g.keys())
    if {"data", "indices", "indptr", "shape"} <= keys:
        return sp.csr_matrix((g["data"][...], g["indices"][...], g["indptr"][...]), shape=tuple(g["shape"][...]))
    if {"row", "col", "data", "shape"} <= keys:
        return sp.coo_matrix((g["data"][...], (g["row"][...], g["col"][...])), shape=tuple(g["shape"][...])).tocsr()
    raise FormatError("sparse group needs data/indices/indptr/shape or row/col/data/shape")


def _coo_from_table(t: np.ndarray, shape=None, one_based: bool = False):
    if t.ndim != 2 or t.shape[1] not in (3, 4):
        raise FormatError("COO layout needs columns i j value [imag]")
    i = t[:, 0].real.astype(np.int64) - (1 if one_based else 0)
    j = t[:, 1].real.astype(np.int64) - (1 if one_based else 0)
    v = t[:, 2] if t.shape[1] == 3 else t[:, 2].real + 1j * t[:, 3].real
    if shape is None:
        n = int(max(i.max(), j.max())) + 1
        shape = (n, n)
    return sp.coo_matrix((v, (i, j)), shape=tuple(shape)).tocsr()


def _coo_header_shape(path: str):
    """Shape recorded by write_operator in the first comment line ('# coo shape n n: ...')."""
    with open(path) as f:
        first = f.readline()
    if first.startswith("#") and "coo shape" in first:
        parts = first.split("coo shape", 1)[1].split(":")[0].split()
        return (int(parts[0]), int(parts[1]))
    return None


def read_operator(path: str, key: Optional[str] = None, layout: str = "auto", allow_code: bool = False,
                  shape=None, one_based: bool = False):
    """Read the producer's operator: dense ndarray, scipy CSR matrix, or (allow_code) a LinearOperator."""
    e = _ext(path)
    if e == ".py":
        if not allow_code:
            raise FormatError(f"{path}: a matrix-free operator executes the producer's code; pass allow_code=True")
        return _load_callable(path)
    A = _read_matrix(path, key, layout, shape, one_based)
    if A.shape[0] != A.shape[1]:
        raise FormatError(f"{path}: operator must be square, got shape {A.shape}")
    return _finite(A, path)


def _read_matrix(path: str, key: Optional[str], layout: str = "auto", shape=None, one_based: bool = False):
    """A 2-D matrix, dense ndarray or CSR, from any matrix-capable format (no squareness check)."""
    e = _ext(path)
    if e == ".npy":
        A = np.load(path, allow_pickle=False)
    elif e == ".npz":
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            if {"data", "indices", "indptr", "shape", "format"} <= files:
                A = sp.load_npz(path).tocsr()
            elif {"H_data", "H_indices", "H_indptr", "H_shape"} <= files:
                A = sp.csr_matrix((z["H_data"], z["H_indices"], z["H_indptr"]), shape=tuple(z["H_shape"]))
            elif {"row", "col", "data", "shape"} <= files:
                A = sp.coo_matrix((z["data"], (z["row"], z["col"])), shape=tuple(z["shape"])).tocsr()
            else:
                A = z[_pick_key(z.files, key, "operator", path)]
    elif e == ".mtx":
        A = _mmread(path)
        A = A.tocsr() if sp.issparse(A) else np.asarray(A)
    elif e in (".h5", ".hdf5", ".hdf"):
        import h5py
        with h5py.File(path, "r") as f:
            k = key if key is not None else next((k for k in DEFAULT_KEYS["operator"] if k in f), None)
            if k is None:
                names = list(f.keys())
                k = names[0] if len(names) == 1 else None
            if k is None:
                raise FormatError(f"{path}: give key= (available: {list(f.keys())})")
            A = _csr_from_group(f[k]) if isinstance(f[k], h5py.Group) else f[k][...]
    elif e in (".csv", ".tsv", ".txt", ".dat"):
        t = _parse_text_table(path, _text_delimiter(path))
        hdr_shape = _coo_header_shape(path)
        if layout == "coo" or (layout == "auto" and hdr_shape is not None):
            A = _coo_from_table(t, shape or hdr_shape, one_based)
        else:
            A = t
    elif e == ".json":
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj.get("format") in ("coo", "csr"):
            shp = tuple(obj["shape"])
            if obj["format"] == "coo":
                A = sp.coo_matrix((_json_to_array(obj["data"]), (np.asarray(obj["row"]), np.asarray(obj["col"]))),
                                  shape=shp).tocsr()
            else:
                A = sp.csr_matrix((_json_to_array(obj["data"]), np.asarray(obj["indices"]), np.asarray(obj["indptr"])),
                                  shape=shp)
        else:
            if isinstance(obj, dict):
                obj = obj[_pick_key([k for k, v in obj.items() if isinstance(v, list)], key, "operator", path)]
            A = _json_to_array(obj)
    else:
        raise FormatError(f"{path}: unsupported operator extension {e!r} (known: {OPERATOR_EXT})")
    if not sp.issparse(A):
        A = np.asarray(A)
        if A.ndim != 2:
            raise FormatError(f"{path}: expected a 2-D matrix, got shape {A.shape}")
    return A


def _load_callable(path: str):
    from scipy.sparse.linalg import LinearOperator
    spec = importlib.util.spec_from_file_location("certify_ed_producer_operator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "operator"):
        op = mod.operator()
        if not hasattr(op, "matvec"):
            raise FormatError(f"{path}: operator() must return an object with matvec")
        return op
    if not hasattr(mod, "matvec"):
        raise FormatError(f"{path}: define matvec(x) and shape (or dim), or operator()")
    n = int(getattr(mod, "dim", None) or mod.shape[0])
    dtype = np.dtype(getattr(mod, "dtype", float))
    return LinearOperator((n, n), matvec=mod.matvec, rmatvec=getattr(mod, "rmatvec", mod.matvec), dtype=dtype)


def write_operator(path: str, A, key: str = "H") -> str:
    """Write an operator: dense (.npy, .npz, text, .json, .mtx, .h5) or sparse (.npz via scipy, .mtx,
    .h5 CSR group, .json COO, text COO triplets with the shape in the header)."""
    e = _ext(path)
    if sp.issparse(A):
        A = sp.csr_matrix(A)
        if e == ".npz":
            sp.save_npz(path, A)
        elif e == ".mtx":
            from scipy.io import mmwrite
            mmwrite(path, A)
        elif e in (".h5", ".hdf5", ".hdf"):
            import h5py
            with h5py.File(path, "w") as f:
                g = f.create_group(key)
                for k, v in (("data", A.data), ("indices", A.indices), ("indptr", A.indptr), ("shape", np.array(A.shape))):
                    g.create_dataset(k, data=v)
        elif e == ".json":
            C = A.tocoo()
            with open(path, "w") as f:
                json.dump({"format": "coo", "shape": list(A.shape), "row": C.row.tolist(), "col": C.col.tolist(),
                           "data": _jsonable_array(C.data)}, f)
        elif e in (".csv", ".tsv", ".txt", ".dat"):
            C = A.tocoo()
            cols = [C.row, C.col, C.data.real] + ([C.data.imag] if np.iscomplexobj(C.data) else [])
            _write_text(path, np.column_stack(cols).astype(float), {".csv": ",", ".tsv": "\t"}.get(e, " "),
                        header=f"coo shape {A.shape[0]} {A.shape[1]}: i j value" + (" imag" if len(cols) == 4 else ""))
        else:
            raise FormatError(f"{path}: sparse operators go to .npz/.mtx/.h5/.json/text, not {e!r}")
        return path
    A = np.asarray(A)
    if e == ".npz":
        np.savez(path, **{key: A})
    else:
        write_array(path, A, key=key, role="operator")
    return path


# ------------------------------------------------------------------------------------------ bases
def read_basis(path: str, key: Optional[str] = None):
    """A sector basis: ('states', 1-D int64 array of basis states) or ('isometry', CSR matrix)."""
    e = _ext(path)
    A = None
    if e in (".mtx", ".h5", ".hdf5", ".hdf") or (e == ".npz" and _npz_is_sparse(path)):
        try:
            A = _read_matrix(path, key)
        except FormatError:
            A = None
    if A is None:
        A = read_array(path, key, "basis")
    if (sp.issparse(A) or np.asarray(A).ndim == 2) and A.shape[1] == 1 and A.shape[0] > 1:
        A = (A.toarray() if sp.issparse(A) else np.asarray(A)).ravel()      # an n x 1 column is a state list
    if sp.issparse(A) or np.asarray(A).ndim == 2:
        return ("isometry", sp.csr_matrix(_finite(A, path)))
    a = np.asarray(A)
    if a.dtype.kind in "fc":
        if not np.all(np.mod(a.real, 1) == 0) or np.any(a.imag != 0):
            raise FormatError(f"{path}: a 1-D basis must be a list of integer states")
        a = a.real.astype(np.int64)
    return ("states", a.astype(np.int64))


def write_basis(path: str, basis) -> str:
    """Inverse of read_basis: a 1-D integer array (state list) or a 2-D/sparse isometry."""
    if sp.issparse(basis) or np.asarray(basis).ndim == 2:
        return write_operator(path, basis, key="V") if sp.issparse(basis) else write_array(path, np.asarray(basis), "V", "basis")
    return write_array(path, np.asarray(basis, dtype=np.int64), "states", "basis")


def _npz_is_sparse(path: str) -> bool:
    with np.load(path, allow_pickle=False) as z:
        return {"data", "indices", "indptr", "shape", "format"} <= set(z.files)


def sniff_kind(path: str) -> str:
    """'array' | 'code' | 'unknown' from the extension only."""
    e = _ext(path)
    if e == ".py":
        return "code"
    return "array" if e in ARRAY_EXT else "unknown"
