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

A sparse matrix keeps its format, its index arrays and any repeated or unsorted indices; vectors on disk (numpy
memmap, HDF5 dataset) are wrapped in an enclosure.StreamedColumns that widens each block as it is read. Objects
without a dtype (LinearOperators, spec operators, callables) are returned unchanged: their products are their own.
A list or tuple of numbers is converted to an array only when its numbers need it.

Claimed eigenvalues are always converted with dtype=float by the callers; `check_values` only refuses the types for
which that conversion would round (a wider float, an integer beyond 2^53 that binary64 cannot hold).

Records: `widened` collects one entry per converted input; certify() writes them to the certificate's
arithmetic.input_widening and the other records to their `input_widening` field, only when there are any, so a
binary64 input gives the same record as before. Hashes are taken of the binary64 data, after the conversion:
matrix_sha256 has always hashed binary64 entries (an operator's hash does not change), and an eigenvector hash is that
of the converted array, which is the array the arrays sidecar stores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)
_EXACT_INT = 2 ** 53                 # every integer of magnitude <= 2^53 is a binary64 number
_CHUNK_BYTES = 64 * 2 ** 20          # rows of an on-disk integer array read at once for the exactness check


class PrecisionError(ValueError):
    """An input whose values binary64 cannot hold exactly, or that is not numeric data."""


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


def _note(widened: Optional[List[Dict[str, Any]]], what: str, dt, tgt) -> None:
    if widened is not None:
        widened.append({"input": what, "dtype": np.dtype(dt).name, "converted_to": np.dtype(tgt).name})


def binary64(a, what: str = "input", widened: Optional[List[Dict[str, Any]]] = None):
    """`a` with binary64 entries (see the module docstring): the same object when it is binary64 already or is not
    array data, else an exact conversion, noted in `widened` when that list is given. PrecisionError for data that
    binary64 cannot hold."""
    if a is None:
        return None
    if sp.issparse(a):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        if a.dtype.kind in "iu":
            _exact_ints(a.tocoo().data if a.format in ("dok", "lil") else a.data, what)
        if a.format in ("csr", "csc", "bsr", "coo", "dia"):
            # a copy with the entries converted: scipy's astype would sum repeated entries (in the new format, where
            # the sum may round), while every stored entry is kept here, as given
            w = a.copy()
            if a.format in ("csr", "csc", "bsr") and \
                    (w.indices.dtype, w.indptr.dtype) != (a.indices.dtype, a.indptr.dtype):     # copy() may narrow int64
                w.indices, w.indptr = a.indices.copy(), a.indptr.copy()
            w.data = a.data.astype(tgt)
        else:                                       # lil, dok: no repeated entries
            w = a.astype(tgt)
        _note(widened, what, a.dtype, tgt)
        return w
    from .enclosure import StreamedColumns, is_streamed       # enclosure imports this module
    if is_streamed(a):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        src = a.source if isinstance(a, StreamedColumns) else a
        if np.dtype(a.dtype).kind in "biu":
            _exact_ints_streamed(src, tuple(int(d) for d in a.shape), np.dtype(src.dtype), what)
        _note(widened, what, a.dtype, tgt)
        return StreamedColumns(src, dtype=tgt)
    if isinstance(a, np.ndarray):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        if a.dtype.kind in "biu":
            _exact_ints(a, what)
        _note(widened, what, a.dtype, tgt)
        return a.astype(tgt)
    if isinstance(a, (list, tuple, np.generic)):
        try:
            arr = np.asarray(a)
        except (ValueError, TypeError):           # ragged or mixed: left to the caller, as before
            return a
        if arr.dtype.kind not in "biufc" or target_dtype(arr.dtype, what) is None:
            return a
        return binary64(arr, what, widened)
    return a


def check_values(values, what: str = "eigenvalues") -> None:
    """PrecisionError when converting `values` with dtype=float would round them: a floating type wider than
    binary64, or an integer that binary64 cannot hold. Every other numeric type converts exactly."""
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
        key = (w["input"], w["dtype"], w["converted_to"])
        if key not in seen:
            seen.add(key)
            rows.append(dict(w))
    return {"converted": rows, "exact": True,
            "note": ("inputs stored in a narrower format than binary64 (or as integers) were converted to binary64 "
                     "before any product; the conversion is exact, so the statements concern the values as stored; "
                     "hashes are of the converted binary64 data")}
