"""Binary64 inputs (3.3.0.dev0): the one helper through which a stored operator, vectors or another matrix reach a
rigorous bound.

Every rounding constant of CERTIFY-ED (fp.U = 2^-53, the gamma_n, the guards) is that of IEEE binary64. numpy forms a
product of two single-precision arrays in single precision (u = 2^-24) and reduces a single-precision array (a norm,
a row sum) in single precision, so a bound evaluated on such data does not hold: in 3.0.0 to 3.2.3 a float32 2 x 2
operator with float32 vectors received an S2 certificate of radius 8.9e-16 for a claimed value 9.3e-10 away from
both eigenvalues. Every entry point applies `binary64` before the data reach a product:

  float64, complex128, a long double that is binary64     the same object, unchanged
  float16 and float32 -> float64, complex64 -> complex128  widened; the conversion is exact, so every statement still
                                                           concerns the values as stored
  bool and integers -> float64                             when every value is exactly representable; refused otherwise
  a wider format (80- or 128-bit long double), object,     refused: PrecisionError (a ValueError)
  strings, dates, ...

A sparse matrix keeps its format, its index arrays and any repeated or unsorted indices (they are summed later, in
binary64, by the step that converts the format); LIL and DOK matrices, which hold one value per position, are converted
by their own astype. Every entry point widens the caller's object before any format
conversion (tocsr, csr_matrix, toarray), since such a conversion sums repeated entries in the input's own precision.
Claimed eigenvectors on disk (numpy memmap, HDF5 dataset) are wrapped in an enclosure.StreamedColumns that widens each
block as it is read (stream=True); any other memmap (an operator, a symmetry diagonal, an observable) becomes an array
in memory. Operators with a matvec (LinearOperators, spec operators) and callables are returned unchanged: their
products are their own. A list or tuple of numbers, or an array-like (an object with __array__ and no matvec), becomes
an ndarray whatever its dtype, so that an operator given this way is a stored matrix in binary64 as in a narrower
format; it is recorded when its numbers were converted.

Claimed eigenvalues are always converted with dtype=float by the callers; `check_values` only refuses the types for
which that conversion would round (a wider float, an integer beyond 2^53 that binary64 cannot hold).

Records: `widened` collects one entry per converted input (the caller's dtype, the target dtype and source_sha256, the
hash of the input as given); certify() writes them to the certificate's arithmetic.input_widening and the other
records to their `input_widening` field, only when there are any, so a binary64 input gives the same record as before.
Every converted input is recorded, integers and lists included; the claimed eigenvalues, which were always converted
with dtype=float, are checked and not recorded. Hashes are taken of the binary64 data, after the conversion:
matrix_sha256 has always hashed binary64 entries (an operator's hash does not change), and an eigenvector hash is that
of the converted array, which is the array the arrays sidecar stores.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)
_EXACT_INT = 2 ** 53                 # every integer of magnitude <= 2^53 is a binary64 number
_CHUNK_BYTES = 64 * 2 ** 20          # rows of an on-disk integer array read at once for the exactness check


class PrecisionError(ValueError):
    """An input whose values binary64 cannot hold exactly, or that is not numeric data; raised before any product, and
    no certificate or record is issued."""


def _is_binary64(dt: np.dtype) -> bool:
    fi = np.finfo(dt)
    return (fi.nmant == 52 and fi.maxexp == 1024 and fi.minexp == -1022
            and dt.itemsize == (8 if dt.kind == "f" else 16))


def _fits_binary64(dt: np.dtype) -> bool:
    fi = np.finfo(dt)
    return fi.nmant <= 52 and fi.maxexp <= 1024 and fi.minexp >= -1022


def target_dtype(dtype, what: str = "input") -> Optional[np.dtype]:
    """The dtype an array of `dtype` is converted to (float64 or complex128), or None when it is binary64 already.
    Integer and bool dtypes give float64 (the values are checked by the caller). PrecisionError for a floating type
    wider than binary64 and for anything that is not numeric."""
    dt = np.dtype(dtype)
    if dt == _F64 or dt == _C128:
        return None
    if dt.kind in "fc":
        if _is_binary64(dt):
            return None
        if _fits_binary64(dt):
            return _F64 if dt.kind == "f" else _C128
        raise PrecisionError(f"{what}: dtype {dt.name} is wider than binary64 ({np.finfo(dt).nmant + 1}-bit "
                             "significand); the bounds count binary64 arithmetic, so the data must be binary64 or "
                             "narrower (round it to float64 or complex128 yourself if that is what you mean)")
    if dt.kind in "biu":
        return _F64
    raise PrecisionError(f"{what}: dtype {dt.name} is not numeric data; the bounds need binary64 floating-point "
                         "numbers (float64 or complex128; narrower types are widened exactly)")


def _exact_ints(a: np.ndarray, what: str) -> None:
    """PrecisionError unless every integer of the in-memory array `a` is a binary64 number."""
    if a.dtype.kind == "b" or a.size == 0:
        return
    lo, hi = int(a.min()), int(a.max())
    if -_EXACT_INT <= lo and hi <= _EXACT_INT:
        return
    for v in a.ravel().tolist():
        if abs(v) > _EXACT_INT and int(float(v)) != v:
            raise PrecisionError(f"{what}: the integer {v} is not exactly representable in binary64; refused (the "
                                 "bounds would concern a rounded value)")


def _exact_ints_streamed(src, shape, dtype: np.dtype, what: str) -> None:
    if dtype.kind == "b" or not shape or shape[0] == 0:
        return
    row = dtype.itemsize * (int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1)
    step = max(1, _CHUNK_BYTES // max(1, row))
    for r in range(0, shape[0], step):
        _exact_ints(np.asarray(src[r:r + step]), what)


_SPARSE_PARTS = {"csr": ("data", "indices", "indptr"), "csc": ("data", "indices", "indptr"),
                 "bsr": ("data", "indices", "indptr"), "coo": ("data", "row", "col"), "dia": ("data", "offsets")}


def source_sha256(a) -> str:
    """SHA-256 of an input as the caller stored it, before any conversion: certificate_v2.array_sha256 (dtype, shape,
    C-order bytes; on-disk arrays in row chunks) of an array, of np.asarray of a list or an array-like; for a sparse
    matrix the format, the shape and array_sha256 of each stored array (data and index arrays as given; lil and dok
    through their coo form)."""
    from .certificate_v2 import array_sha256
    from .enclosure import StreamedColumns, is_streamed
    if sp.issparse(a):
        parts = _SPARSE_PARTS.get(a.format)
        if parts is None:
            a, parts = a.tocoo(), _SPARSE_PARTS["coo"]
        h = hashlib.sha256()
        h.update(a.format.encode())
        h.update(json.dumps([int(d) for d in a.shape]).encode())
        for name in parts:
            h.update(array_sha256(np.asarray(getattr(a, name))).encode())
        return h.hexdigest()
    if isinstance(a, StreamedColumns) and a.widens:
        a = a.source
    return array_sha256(a if (isinstance(a, np.ndarray) or is_streamed(a)) else np.asarray(a))


def _note(widened: Optional[List[Dict[str, Any]]], what: str, a, dt, tgt) -> None:
    if widened is not None:
        widened.append({"input": what, "dtype": np.dtype(dt).name, "converted_to": np.dtype(tgt).name,
                        "source_sha256": source_sha256(a)})


def _sparse_binary64(a, tgt):
    """A new matrix of the same format with the stored entries converted to `tgt` and copies of the index arrays:
    every stored entry is kept as given (scipy's astype sums repeated entries, in the new format, where the sum may
    round), and the caller's entries are not copied before the conversion. LIL and DOK matrices hold one value per
    position (no repeated entries; LIL's `data` is an array of lists, DOK has none): their own astype is used."""
    if a.format in ("csr", "csc", "bsr"):
        kw = {"blocksize": a.blocksize} if a.format == "bsr" else {}
        w = type(a)((a.data.astype(tgt), a.indices.copy(), a.indptr.copy()), shape=a.shape, **kw)
        if (w.indices.dtype, w.indptr.dtype) != (a.indices.dtype, a.indptr.dtype):  # the constructor may narrow int64
            w.indices, w.indptr = a.indices.copy(), a.indptr.copy()
        return w
    if a.format == "coo":
        return type(a)((a.data.astype(tgt), (a.row.copy(), a.col.copy())), shape=a.shape)
    if a.format == "dia":
        return type(a)((a.data.astype(tgt), a.offsets.copy()), shape=a.shape)
    return a.astype(tgt)                          # lil, dok: one value per position


def binary64(a, what: str = "input", widened: Optional[List[Dict[str, Any]]] = None, stream: bool = False):
    """`a` with binary64 entries (see the module docstring): the same object when it is binary64 already or is not
    array data, else an exact conversion, noted in `widened` when that list is given (with the caller's dtype and
    source_sha256 of the caller's data). PrecisionError for data that binary64 cannot hold.

    stream: vectors on disk (numpy memmap, HDF5 dataset) in a narrower format are wrapped in a widening
    enclosure.StreamedColumns (claimed eigenvectors only); without it they are converted to an array in memory, as
    any matrix is."""
    if a is None:
        return None
    if sp.issparse(a):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        if a.dtype.kind in "iu":
            _exact_ints(a.tocoo().data if a.format in ("dok", "lil") else a.data, what)
        w = _sparse_binary64(a, tgt)
        _note(widened, what, a, a.dtype, tgt)
        return w
    from .enclosure import StreamedColumns, is_streamed       # enclosure imports this module
    if is_streamed(a):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        src = a.source if isinstance(a, StreamedColumns) else a
        if np.dtype(a.dtype).kind in "biu":
            _exact_ints_streamed(src, tuple(int(d) for d in a.shape), np.dtype(src.dtype), what)
        _note(widened, what, a, a.dtype, tgt)
        if stream:
            return StreamedColumns(src, dtype=tgt)
        return np.asarray(src[...] if not isinstance(src, np.ndarray) else src).astype(tgt)
    if isinstance(a, np.ndarray):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        if a.dtype.kind in "biu":
            _exact_ints(a, what)
        _note(widened, what, a, a.dtype, tgt)
        return a.astype(tgt)
    if isinstance(a, (list, tuple, np.generic)) or (hasattr(a, "__array__") and not hasattr(a, "matvec")):
        # numeric array data given as a list or tuple of numbers or as an array-like (an object with __array__: a data
        # frame, a tensor on the CPU, a zarr array) becomes an ndarray whatever its dtype, so that an operator given
        # this way is a stored matrix in binary64 as in any narrower format; binary64 data are not recorded. Numpy
        # scalars in binary64 stay scalars. Operators with a matvec are the producer's and stay as given.
        try:
            arr = np.asarray(a)
        except (ValueError, TypeError):           # ragged or mixed: left to the caller, as before
            return a
        if arr.dtype.kind not in "biufc":
            return a
        if target_dtype(arr.dtype, what) is None:
            return a if isinstance(a, np.generic) else arr
        return binary64(arr, what, widened)
    return a


def check_values(values, what: str = "eigenvalues") -> None:
    """PrecisionError when converting `values` with dtype=float would round them: a floating type wider than
    binary64, or an integer that binary64 cannot hold. Narrower numeric types convert exactly. Non-numeric values
    (an object array of Fraction, Decimal or mpmath numbers) are not checked: the callers convert them with
    dtype=float as before, which rounds them, and the record then holds the converted values, which are the values
    every statement refers to."""
    try:
        arr = np.asarray(values)
    except (ValueError, TypeError):
        return
    if arr.dtype.kind not in "biufc":
        return
    target_dtype(arr.dtype, what)
    if arr.dtype.kind in "iu":
        _exact_ints(arr, what)


def widening_record(widened: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The record of the conversions (each input once), or None when there were none."""
    if not widened:
        return None
    seen, rows = set(), []
    for w in widened:
        key = (w["input"], w["dtype"], w["converted_to"], w.get("source_sha256"))
        if key not in seen:
            seen.add(key)
            rows.append(dict(w))
    return {"converted": rows, "exact": True,
            "note": ("inputs stored in a narrower format than binary64 (or as integers) were converted to binary64 "
                     "before any product; the conversion is exact, so the statements concern the values as stored; "
                     "the record's hashes are of the converted binary64 data, source_sha256 of each input as given")}
