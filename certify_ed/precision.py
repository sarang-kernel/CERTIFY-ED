"""Binary64 inputs (3.3.0.dev0): the one helper through which a stored operator, vectors or another matrix reach a
rigorous bound.

Every rounding constant of CERTIFY-ED (fp.U = 2^-53, the gamma_n, the guards) is that of IEEE binary64. numpy forms a
product of two single-precision arrays in single precision (u = 2^-24) and reduces a single-precision array (a norm,
a row sum) in single precision, so a bound evaluated on such data does not hold: in 3.0.0 to 3.2.3 a float32 2 x 2
operator with float32 vectors received an S2 certificate of radius 8.9e-16 for a claimed value 9.3e-10 away from
both eigenvalues. The public functions listed in tests/test_precision_registry.py apply `binary64` before the data
reach a product or a format conversion:

  float64, complex128, a long double that is binary64     the same object, unchanged
  float16 and float32 -> float64, complex64 -> complex128  converted exactly, so every statement still concerns the
                                                           values as stored; recorded
  integers                                                 checked (every value a binary64 number), then converted, or
                                                           with keep_integers used as given (certify(), as in 3.x);
                                                           not recorded
  bool                                                     converted (exact), not recorded
  object arrays                                            accepted when float(v) == v (or complex(v) == v) for every
                                                           value, converted, not recorded
  a wider format (80- or 128-bit long double), an inexact  refused: PrecisionError (a ValueError)
  integer or object value, strings, dates, ...

The format decision is classify_format (significand bits, exponent range, item size), so that the refusal of a wider
long double is tested on platforms where long double is binary64. A sparse matrix keeps its format, its index arrays
and any repeated or unsorted indices; the entry points then apply `summed`, which sums repeated entries on a copy, in
binary64, before any product (so that every product, norm and modulus of a call refers to the same summed matrix). LIL
and DOK matrices, which hold one value per position, are converted by their own astype; DIA padding is not read.
Format conversions (tocsr, csr_matrix, toarray) come after `binary64`, since such a conversion sums repeated entries
in the input's own precision. Claimed eigenvectors on disk (numpy memmap, HDF5 dataset) are wrapped in an
enclosure.StreamedColumns that widens each block as it is read (stream=True); any other matrix on disk (an operator,
a symmetry diagonal, an observable) becomes an array in memory, also in binary64. Operators with a matvec
(LinearOperators, spec operators) and callables are returned unchanged: their products are their own. A list or tuple
of numbers, or an array-like (an object with __array__ and no matvec), becomes an ndarray whatever its dtype, so that
an operator given this way is a stored matrix; ragged lists and lists with strings (the operator forms of
operators.py) are returned as given. The low-level kernels call `require_binary64` and refuse floating data other
than binary64, so an entry point that misses its conversion fails instead of certifying.

Claimed eigenvalues are converted with dtype=float by the callers, as in 3.x (a complex array: the real part, with
numpy's ComplexWarning; `claimed_reals` makes a value with a non-finite imaginary part nan, so that the rule for
non-finite claimed values applies to it); `check_values` refuses, under the same rule, the values for which that
conversion would round.

Records: `widened` collects one entry per input converted from a narrower floating format (the caller's dtype, the
target dtype and source_sha256, the hash of the input as given); certify() writes them to the certificate's
arithmetic.input_widening and the other records to their `input_widening` field, only when there are any, so that a
binary64 or integer input gives the record of 3.x. Hashes are taken of the binary64 data, after the conversion:
matrix_sha256 widens before it sums repeated entries, so the hash of the caller's matrix is the hash in its
certificate; it differs from the 3.x hash only for a matrix in a narrower format whose repeated entries sum differently
in that format. An eigenvector hash is that of the converted array, which is the array the arrays sidecar stores.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
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


def classify_format(nmant: int, maxexp: int, minexp: int, itemsize: int, kind: str = "f") -> str:
    """'binary64', 'narrower' or 'wider' for a floating format given by its stored significand bits (np.finfo.nmant),
    its exponent range (maxexp, minexp) and its item size ('f' real, 'c' complex). The decision of target_dtype,
    factored out so that it can be tested for formats a platform does not have (the x87 80-bit and the IEEE 128-bit
    long double)."""
    if nmant == 52 and maxexp == 1024 and minexp == -1022 and itemsize == (8 if kind == "f" else 16):
        return "binary64"
    if nmant <= 52 and maxexp <= 1024 and minexp >= -1022:
        return "narrower"
    return "wider"


def _format_of(dt: np.dtype) -> str:
    fi = np.finfo(dt)
    return classify_format(int(fi.nmant), int(fi.maxexp), int(fi.minexp), dt.itemsize, dt.kind)


def _is_binary64(dt: np.dtype) -> bool:
    return _format_of(dt) == "binary64"


def is_narrow_float(dtype) -> bool:
    """True for a floating dtype narrower than binary64 (float16, float32, complex64): data whose products numpy forms
    in a precision the bounds do not count, and whose repeated sparse entries a format conversion sums in that
    precision."""
    dt = np.dtype(dtype)
    return dt.kind in "fc" and _format_of(dt) == "narrower"


def require_binary64(a, what: str):
    """Fail closed: PrecisionError when floating data that are not binary64 reach a step whose bounds count binary64
    arithmetic, so that an entry point that missed its conversion gives a refusal and not a certificate. Integer and
    boolean data pass (numpy converts them exactly in a product with binary64 data); objects without a dtype pass."""
    dt = getattr(a, "dtype", None)
    if dt is not None:
        dt = np.dtype(dt)
        if dt.kind in "fc" and _format_of(dt) != "binary64":
            raise PrecisionError(f"{what}: {dt.name} data reached a step whose bounds count binary64 arithmetic; the "
                                 "entry point did not convert them (precision.binary64); refused")
    return a


def target_dtype(dtype, what: str = "input") -> Optional[np.dtype]:
    """The dtype an array of `dtype` is converted to (float64 or complex128), or None when it is binary64 already.
    Integer and bool dtypes give float64 (the values are checked by the caller). PrecisionError for a floating type
    wider than binary64 and for anything that is not numeric."""
    dt = np.dtype(dtype)
    if dt == _F64 or dt == _C128:
        return None
    if dt.kind in "fc":
        fmt = _format_of(dt)
        if fmt == "binary64":                     # also in a non-native byte order: numpy's arithmetic is binary64
            return None
        if fmt == "narrower":
            return _F64 if dt.kind == "f" else _C128
        raise PrecisionError(f"{what}: dtype {dt.name} is wider than binary64 ({np.finfo(dt).nmant + 1}-bit "
                             "significand); the bounds count binary64 arithmetic, so the data must be binary64 or "
                             "narrower (round it to float64 or complex128 yourself if that is what you mean)")
    if dt.kind in "biu":
        return _F64
    raise PrecisionError(f"{what}: dtype {dt.name} is not numeric data; the bounds need binary64 floating-point "
                         "numbers (float64 or complex128; narrower types are widened exactly)")


def _inexact_int(v, what: str) -> PrecisionError:
    return PrecisionError(f"{what}: the integer {v} is not exactly representable in binary64; refused (the bounds "
                          "would concern a rounded value)")


def _exact_ints(a: np.ndarray, what: str) -> None:
    """PrecisionError unless every integer of the in-memory array `a` is a binary64 number. The values beyond 2^53
    are selected by one vectorized comparison and only those are checked one by one."""
    a = np.asarray(a)
    if a.dtype.kind == "b" or a.size == 0:
        return
    big = a > _EXACT_INT
    if a.dtype.kind == "i":
        big |= a < -_EXACT_INT
    if not big.any():
        return
    for v in a[big].tolist():
        if int(float(v)) != v:
            raise _inexact_int(v, what)


def _check_python_ints(obj, arr: np.ndarray, what: str) -> None:
    """The integers of a nested list or tuple (Python integers, numpy integer scalars, integer arrays), which
    np.asarray rounds silently when the list also holds a float: PrecisionError for one beyond 2^53 that binary64
    cannot hold. `arr` is np.asarray of the list; a rounded integer has magnitude >= 2^53 there, so the list is walked
    only when `arr` holds such a value."""
    if arr.size == 0 or not np.any(np.abs(arr) >= 2.0 ** 53):
        return
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, np.ndarray):
            if o.dtype.kind in "iu":
                _exact_ints(o, what)
            elif o.dtype.kind == "O":
                stack.extend(o.ravel().tolist())
        elif isinstance(o, (bool, np.bool_)):
            continue
        elif isinstance(o, (int, np.integer)):
            v = int(o)
            if abs(v) > _EXACT_INT:
                try:
                    exact = int(float(v)) == v
                except OverflowError:
                    exact = False
                if not exact:
                    raise _inexact_int(v, what)


def _exact_objects(a: np.ndarray, what: str) -> np.ndarray:
    """An object array of numbers (Python integers, Fraction, Decimal, mpmath, numpy scalars) as a float64 or
    complex128 array, when every value v is exactly a binary64 number (float(v) == v, or complex(v) == v);
    PrecisionError for any other value and for non-numbers. One rule for every input position."""
    vals, cplx = [], False
    for v in a.ravel().tolist():
        if isinstance(v, (str, bytes)) or v is None or isinstance(v, (list, tuple, dict)):
            raise PrecisionError(f"{what}: {type(v).__name__} values are not numeric data")
        try:
            if isinstance(v, (complex, np.complexfloating)):
                raise TypeError
            f = float(v)
            ok = f == v or f != f                  # NaN goes on to the finite check, as in a float array
        except OverflowError:
            ok = False
        except (TypeError, ValueError):
            try:
                f = complex(v)
            except OverflowError:
                f, ok = 0j, False
            except (TypeError, ValueError):
                raise PrecisionError(f"{what}: {type(v).__name__} values are not numeric data") from None
            else:
                ok = f == v or f != f
            cplx = True
        if not ok:
            r = repr(v)
            r = r if len(r) <= 60 else r[:57] + "..."
            raise PrecisionError(f"{what}: the value {r} is not exactly a binary64 number (float(v) != v); refused "
                                 "(convert it yourself if a rounded value is what you mean)")
        vals.append(f)
    return np.array(vals, dtype=_C128 if cplx else _F64).reshape(a.shape)


def _stored_values(a):
    """The values a sparse matrix holds: DIA without its padding, LIL and DOK through COO, CSR, CSC and BSR up to
    nnz."""
    if a.format in ("dia", "lil", "dok"):
        return a.tocoo().data
    if a.format in ("csr", "csc", "bsr"):
        return a.data[:a.indptr[-1]]              # values past nnz (BSR: blocks) are not held
    return a.data


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
    """Record a conversion of a narrower floating format (float16, float32, complex64). Exact conversions of integer,
    boolean and object data and byte-order changes are not recorded: those inputs were not affected and keep the
    certificate they had."""
    if widened is not None and np.dtype(dt).kind in "fc":
        widened.append({"input": what, "dtype": np.dtype(dt).name, "converted_to": np.dtype(tgt).name,
                        "source_sha256": source_sha256(a)})


def _sparse_binary64(a, tgt):
    """A new matrix of the same format with the stored entries converted to `tgt` and copies of the index arrays:
    every stored entry is kept as given (scipy's astype sums repeated entries: up to scipy 1.10 before the conversion,
    in the matrix's own format and in place on the caller's matrix, from scipy 1.11 after it), and the caller's entries
    are not copied before the conversion. LIL and DOK matrices hold one value per
    position (no repeated entries; LIL's `data` is an array of lists, DOK has none): their own astype is used."""
    if a.format in ("csr", "csc", "bsr"):
        kw = {"blocksize": a.blocksize} if a.format == "bsr" else {}
        w = type(a)((a.data.astype(tgt), a.indices.copy(), a.indptr.copy()), shape=a.shape, **kw)
        if (w.indices.dtype, w.indptr.dtype) != (a.indices.dtype, a.indptr.dtype):  # the constructor may narrow int64
            w.indices, w.indptr = w.indices.astype(a.indices.dtype), w.indptr.astype(a.indptr.dtype)
        return w
    if a.format == "coo":
        coords = tuple(c.copy() for c in a.coords) if hasattr(a, "coords") else (a.row.copy(), a.col.copy())
        return type(a)((a.data.astype(tgt), coords), shape=a.shape)       # coords: 1-D sparse arrays too
    if a.format == "dia":
        return type(a)((a.data.astype(tgt), a.offsets.copy()), shape=a.shape)
    return a.astype(tgt)                          # lil, dok: one value per position


def _compressed_with_repeats(a):
    """A copy of the CSR or CSC matrix `a` whose indices are sorted within each row (column) and whose repeated
    entries are kept, adjacent; `a` is not changed. scipy's own conversions would add the repeated entries."""
    nnz = int(a.indptr[-1])
    c = type(a)((a.data[:nnz].copy(), a.indices[:nnz].copy(), a.indptr.copy()), shape=a.shape)
    c.has_sorted_indices = False
    c.sort_indices()                                    # within each row (column); repeated entries stay
    return c


def _exact_float_sum(v: np.ndarray) -> float:
    """The correctly rounded exact sum of the binary64 values v: math.fsum, or the exact rational sum where fsum's
    partial sums overflow (inf when the exact sum is beyond the range). With a value that is not finite the result is
    decided by the set of values, whatever their order: nan for a nan or for both infinities, else the infinity."""
    if not np.all(np.isfinite(v)):
        if np.isnan(v).any() or (np.isposinf(v).any() and np.isneginf(v).any()):
            return math.nan
        return math.inf if np.isposinf(v).any() else -math.inf
    try:
        return math.fsum(v.tolist())
    except OverflowError:
        s = sum(Fraction(x) for x in v.tolist())
        try:
            return float(s)
        except OverflowError:
            return math.inf if s > 0 else -math.inf


def _fsum_groups(v: np.ndarray, starts: np.ndarray, k: int) -> np.ndarray:
    """_exact_float_sum of each group v[st : st + k], st in `starts` (groups of one size), with the per-group numpy
    work done once per slice of at most _FSUM_ROWS groups (and _GROUP_CHUNK values): one gather, one finiteness test,
    one conversion to Python floats, then math.fsum on each row (_exact_float_sum for a row with a value that is not
    finite or whose partial sums overflow)."""
    out = []
    step = max(1, min(_FSUM_ROWS, _GROUP_CHUNK // k))       # bounded buffers, as for the vectorized blocks
    for lo in range(0, starts.size, step):
        rows = v[starts[lo:lo + step, None] + np.arange(k)]
        fin = np.isfinite(rows).all(axis=1).tolist()
        for i, (r, f) in enumerate(zip(rows.tolist(), fin)):
            if f:
                try:
                    out.append(math.fsum(r))
                    continue
                except OverflowError:
                    pass
            out.append(_exact_float_sum(rows[i]))
    return np.array(out, dtype=np.float64)


_U = 2.0 ** -53
_GROUP_CHUNK = 1 << 22                                  # values per vectorized block (32 MiB of float64)
_GROUP_VEC_MAX = 64             # groups of up to this many values are always vectorized
_GROUP_VEC_SHARED = 512         # larger groups are vectorized when this many share their size (break-even near 500)
_GROUP_VEC_CAP = 4096           # groups of more values always take math.fsum
_FSUM_ROWS = 1 << 16            # groups per slice of the math.fsum path


def _exact_real_sums(v: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The correctly rounded exact sum of each group v[starts[k] : starts[k] + counts[k]] of real binary64 values.
    Groups of one or two values: one addition (exact or correctly rounded, and commutative). Groups of three to
    _GROUP_VEC_MAX values, and larger groups up to _GROUP_VEC_CAP values when at least _GROUP_VEC_SHARED groups have
    the same size, are vectorized over the groups of each size (a block holds groups of one size only, at most
    _GROUP_CHUNK values, so the buffer is the values themselves, and each pass over a column of a larger size covers
    at least _GROUP_VEC_SHARED groups, except in the last block of a size): three passes of Knuth's TwoSum along each group (error-free: the exact sum is
    kept) leave a leading value h and a remainder e; when e lies strictly inside the rounding interval of h with the
    bound on its own rounding error, the correctly rounded sum is h. The other groups (the remaining sizes, a sum within
    that slack of a rounding boundary, h = 0, a value that is not finite, a partial sum that overflows) take
    math.fsum's value."""
    s = np.add.reduceat(v, starts) if v.size else np.zeros(0)
    big = np.flatnonzero(counts >= 3)
    cb = counts[big]
    order = np.argsort(cb, kind="stable")              # once: the groups of each size in position order
    sizes, first, shared = np.unique(cb[order], return_index=True, return_counts=True)
    blocks = []
    for kmax, lo, g in zip(sizes.tolist(), first.tolist(), shared.tolist()):
        same = big[order[lo:lo + g]]
        if kmax > _GROUP_VEC_MAX and (g < _GROUP_VEC_SHARED or kmax > _GROUP_VEC_CAP):
            s[same] = _fsum_groups(v, starts[same], kmax)   # math.fsum on each: linear, in C
            continue
        step = max(1, _GROUP_CHUNK // kmax)             # groups of one size per block: no padding
        blocks += [(same[i:i + step], kmax) for i in range(0, same.size, step)]
    for grp, kmax in blocks:
        P = np.empty((grp.size, kmax), order="F")       # column-major: each pass walks contiguous columns
        st = starts[grp]
        for j in range(kmax):
            P[:, j] = v[st + j]
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(3):
                for j in range(1, kmax):
                    x, y = P[:, j - 1], P[:, j]
                    t = x + y
                    z = t - x
                    P[:, j - 1] = (x - (t - z)) + (y - z)
                    P[:, j] = t
            h = P[:, -1]
            rest = P[:, :-1]
            e = rest.sum(axis=1)                        # |e - (exact sum of rest)| <= slack
            slack = 2.0 * kmax * _U * np.abs(rest).sum(axis=1)
            r = h + e                                   # the candidate: the rounding of h + e
            # h - r is exact, and so is d, by FastTwoSum (|e| <= |h| is required below); the slack still counts one
            # rounding of d
            d = (h - r) + e
            slack = slack + 2.0 * _U * np.abs(d)
            up_gap = (np.nextafter(r, np.inf) - r) / 2.0
            dn_gap = (r - np.nextafter(r, -np.inf)) / 2.0
            # the exact sum is r + (d +- slack); r is its correct rounding when that offset lies strictly inside
            # the rounding interval of r (no midpoint, so no tie, in between). At the largest float the upper gap
            # is infinite and decides nothing: such a candidate takes the fallback
            ok = (np.isfinite(r) & np.isfinite(d) & np.isfinite(slack) & (r != 0.0) & (np.abs(e) <= np.abs(h))
                  & np.isfinite(up_gap) & np.isfinite(dn_gap)
                  & (np.nextafter(d + slack, np.inf) < up_gap) & (np.nextafter(d - slack, -np.inf) > -dn_gap))
            # one nonzero remainder: the exact sum is h + e with e exact, and r = fl(h + e) is its correct rounding
            # (one IEEE addition, ties to even included)
            ok |= np.isfinite(r) & (np.count_nonzero(rest, axis=1) <= 1)
        s[grp[ok]] = r[ok]
        rej = grp[~ok]
        if rej.size:
            s[rej] = _fsum_groups(v, starts[rej], kmax)
    return s


def _group_sums(v: np.ndarray, starts: np.ndarray, counts: np.ndarray, what: str, dtype: np.dtype) -> np.ndarray:
    """The sum of each group v[starts[k] : starts[k] + counts[k]] of a value array whose groups are contiguous: for
    binary64 data the correctly rounded exact sum (each real part on its own), for integer data the exact sum, in
    the caller's integer type when every sum fits and in int64 otherwise (PrecisionError beyond int64 or uint64)."""
    if v.dtype.kind in "iu":
        wide = v.astype(np.int64) if v.dtype.itemsize < 8 else v
        s = np.add.reduceat(wide, starts)               # exact for 8- to 32-bit data (fewer than 2^31 values a group)
        if v.dtype.itemsize == 8:                       # 64-bit data: groups whose moduli could overflow, exactly
            bound = np.add.reduceat(np.abs(v.astype(np.float64)), starts)
            info = np.iinfo(v.dtype)
            for k in np.flatnonzero(bound * (1.0 + 2.0 ** -40) >= 2.0 ** 62).tolist():
                t = sum(int(x) for x in v[starts[k]:starts[k] + counts[k]].tolist())
                if not info.min <= t <= info.max:
                    raise PrecisionError(f"{what}: a sum of repeated integer entries ({t}) is outside the range of "
                                         f"{v.dtype}; refused (give the matrix with its entries summed, or in "
                                         "binary64)")
                s[k] = t
        info = np.iinfo(dtype)
        if s.size and (int(s.min()) < info.min or int(s.max()) > info.max):
            return s.astype(np.int64) if v.dtype.itemsize < 8 else s
        return s.astype(dtype, copy=False)
    if np.iscomplexobj(v):                              # each part on its own (1j * inf would make the real part nan)
        out = np.empty(starts.size, dtype=np.complex128)
        out.real = _exact_real_sums(np.ascontiguousarray(v.real), starts, counts)
        out.imag = _exact_real_sums(np.ascontiguousarray(v.imag), starts, counts)
        return out
    return _exact_real_sums(v, starts, counts)


def needs_summing(a) -> bool:
    """True when `summed` would copy `a`: a CSR, CSC, BSR or COO matrix not flagged canonical, or a DIA matrix with a
    repeated offset."""
    if not sp.issparse(a):
        return False
    if a.format == "dia":
        offsets = np.asarray(a.offsets)
        return np.unique(offsets).size != offsets.size
    return a.format in ("csr", "csc", "bsr", "coo") and not getattr(a, "has_canonical_format", False)


def summed(a, what: str = "matrix"):
    """`a` with its repeated entries summed, `a` itself unchanged. The same object when `a` is not a sparse matrix,
    when it is a CSR, CSC, BSR or COO matrix flagged canonical (sorted indices, no repeated entries; the flag is
    trusted, as scipy trusts it), a LIL or DOK matrix, or a DIA matrix with distinct offsets; else a canonical copy of
    the same format, class and index dtypes in which each position holds the sum of its stored entries. A matrix whose
    positions are distinct but whose indices are unsorted is copied too, so that every later step sees one stored
    order and none sorts the caller's arrays in place (scipy's abs and sum_duplicates do).

    The sum at a position is defined independently of the format, the stored order and the platform: for binary64
    data the correctly rounded exact sum of the stored values (each real part on its own), for integer data the exact
    sum, kept in the caller's integer type when every sum fits and in int64 otherwise, and checked like the entries
    (every value a binary64 number; PrecisionError otherwise, and beyond the range of int64 or uint64). Booleans are
    converted to float64 first (True + True = 2), as binary64 converts them; a narrower floating format is refused
    (require_binary64), so the entry points apply `binary64` first, and `binary64` sums integer sparse data this way
    before it converts them.

    The entry points apply it before any product and before any format conversion (scipy's conversions add repeated
    entries in stored order): a product formed from the stored entries of a matrix with repeated entries carries
    rounding errors of the size of the repeated entries, which the moduli of the summed matrix (fp.abs_up) need not
    bound, so every product, norm and modulus of a call refers to one matrix, the summed one
    (docs/proofs/fp_modulus-up.md, section 1). Cost: a copy of the stored arrays, a sort of the positions (scipy's sort
    within rows for CSR and CSC), and the sums of _exact_real_sums, whose memory is a small multiple of the stored
    values."""
    if not needs_summing(a):
        return a
    fmt = a.format
    require_binary64(a, f"precision.summed: {what}")   # a narrower format would be summed in its own precision
    if len(a.shape) == 1:                               # a 1-D coo_array (scipy >= 1.13): positions are its coords
        key = np.asarray(a.coords[0], dtype=np.int64)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        first = np.ones(ks.size, dtype=bool)
        first[1:] = ks[1:] != ks[:-1]
        starts = np.flatnonzero(first)
        counts = np.diff(np.append(starts, ks.size))
        v = np.asarray(a.data)[order]
        v = v.astype(np.float64) if v.dtype.kind == "b" else v
        sums = _group_sums(v, starts, counts, what, v.dtype)
        if sums.dtype.kind in "iu":
            _exact_ints(sums, what)
        cls = sp.coo_array if isinstance(a, getattr(sp, "sparray", ())) else sp.coo_matrix
        out = cls((sums, (ks[starts].astype(a.coords[0].dtype),)), shape=a.shape)
        out.sum_duplicates()
        return out
    if fmt in ("csr", "csc"):
        return _summed_compressed(a, what)
    coo = a if fmt == "coo" else a.tocoo()             # BSR and DIA through their stored entries (none summed)
    row, col = np.asarray(coo.row), np.asarray(coo.col)
    order = np.lexsort((col, row))                      # row-major; no combined key, so no overflow for any shape
    rs, cs_ = row[order], col[order]
    first = np.ones(rs.size, dtype=bool)
    first[1:] = (rs[1:] != rs[:-1]) | (cs_[1:] != cs_[:-1])
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, rs.size))
    v = np.asarray(coo.data)[order]
    v = v.astype(np.float64) if v.dtype.kind == "b" else v
    sums = _group_sums(v, starts, counts, what, v.dtype)
    if sums.dtype.kind in "iu":
        _exact_ints(sums, what)
    cls = sp.coo_array if isinstance(a, getattr(sp, "sparray", ())) else sp.coo_matrix
    out = cls((sums, (rs[starts], cs_[starts])), shape=a.shape)
    out.has_canonical_format = True                     # sorted row-major, no repeated position
    if fmt == "bsr":
        res = out.tobsr(blocksize=a.blocksize)
        if (res.indices.dtype, res.indptr.dtype) != (a.indices.dtype, a.indptr.dtype):
            res.indices, res.indptr = res.indices.astype(a.indices.dtype), res.indptr.astype(a.indptr.dtype)
        return res
    if fmt == "dia":
        return out.todia()
    if hasattr(out, "coords"):                          # scipy >= 1.13: one tuple, its index dtypes set together
        out.coords = (out.row.astype(a.row.dtype, copy=False), out.col.astype(a.col.dtype, copy=False))
    else:
        out.row, out.col = out.row.astype(a.row.dtype, copy=False), out.col.astype(a.col.dtype, copy=False)
    out.has_canonical_format = True
    return out


def _summed_compressed(a, what: str):
    """summed for CSR and CSC input: a copy sorted within rows (columns) by scipy, the groups found by comparing
    neighbours, the sums of _group_sums, and the canonical result in the caller's class and index dtypes."""
    c = _compressed_with_repeats(a)
    idx, indptr, v = c.indices, c.indptr, c.data
    nnz = idx.size
    first = np.ones(nnz, dtype=bool)
    if nnz:
        first[1:] = idx[1:] != idx[:-1]
        row_start = indptr[:-1][indptr[:-1] < nnz]
        first[row_start] = True                         # a new row (column) starts a new position
    if first.all():                                     # no position repeats: the sorted copy
        sums, new_idx, new_ptr = (v.astype(np.float64) if v.dtype.kind == "b" else v), idx, indptr
        if sums.dtype.kind in "iu":
            _exact_ints(sums, what)
    else:
        starts = np.flatnonzero(first)
        counts = np.diff(np.append(starts, nnz))
        vv = v.astype(np.float64) if v.dtype.kind == "b" else v
        sums = _group_sums(vv, starts, counts, what, vv.dtype)
        if sums.dtype.kind in "iu":
            _exact_ints(sums, what)
        new_idx = idx[starts]
        new_ptr = np.concatenate([[0], np.cumsum(first)])[indptr].astype(indptr.dtype, copy=False)
    out = type(a)((sums, new_idx, new_ptr), shape=a.shape)
    out.has_sorted_indices = True
    out.has_canonical_format = True
    if (out.indices.dtype, out.indptr.dtype) != (a.indices.dtype, a.indptr.dtype):
        out.indices, out.indptr = out.indices.astype(a.indices.dtype), out.indptr.astype(a.indptr.dtype)
    return out


def _materialize(a, what: str) -> np.ndarray:
    try:
        return np.asarray(a)
    except Exception as e:                         # a device array, a lazy array that cannot be read, ...
        raise PrecisionError(f"{what}: {type(a).__name__} could not be converted to a numpy array "
                             f"({type(e).__name__}: {e}); give a numpy array or a scipy.sparse matrix") from None


def binary64(a, what: str = "input", widened: Optional[List[Dict[str, Any]]] = None, stream: bool = False,
             keep_integers: bool = False):
    """`a` with binary64 entries (see the module docstring): the same object when it is binary64 already or is not
    array data, else an exact conversion; conversions of a narrower floating format are noted in `widened` when that
    list is given (with the caller's dtype and source_sha256 of the caller's data). PrecisionError for data that
    binary64 cannot hold.

    stream: vectors on disk (numpy memmap, HDF5 dataset) in a narrower format are wrapped in a widening
    enclosure.StreamedColumns (claimed eigenvectors only); without it they are converted to an array in memory, as
    any matrix is (an HDF5 dataset in binary64 as well; a binary64 memmap is an array already).
    keep_integers: integer data are checked (every value a binary64 number) and returned as given, so that certify()
    works on the caller's objects exactly as 3.x did; the steps that multiply them convert them. Boolean data (which
    3.x could not certify) are converted (exactly, not recorded) in any case."""
    if a is None:
        return None
    if sp.issparse(a):
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            if not a.dtype.isnative:              # binary64 in the other byte order: scipy.sparse needs native data
                return _sparse_binary64(a, a.dtype.newbyteorder("="))
            return a
        if a.dtype.kind in "biu":
            _exact_ints(_stored_values(a), what)
            if keep_integers and a.dtype.kind in "iu":
                return a
            if a.dtype.kind in "iu" and needs_summing(a):
                # repeated integer entries are summed exactly first (refused where the exact sum is not a binary64
                # number or leaves int64), so that every route sees the matrix certify() sees (summed)
                a = summed(a, what)
        w = _sparse_binary64(a, tgt)
        _note(widened, what, a, a.dtype, tgt)
        return w
    from .enclosure import StreamedColumns, is_streamed       # enclosure imports this module
    if is_streamed(a):
        tgt = target_dtype(a.dtype, what)
        src = a.source if isinstance(a, StreamedColumns) else a
        if tgt is None:
            if stream or isinstance(a, np.ndarray):
                return a
            return np.asarray(src if isinstance(src, np.ndarray) else src[...])
        if np.dtype(a.dtype).kind in "biu":
            _exact_ints_streamed(src, tuple(int(d) for d in a.shape), np.dtype(src.dtype), what)
            if keep_integers and np.dtype(a.dtype).kind in "iu":
                return a
        _note(widened, what, a, a.dtype, tgt)
        if stream:
            return StreamedColumns(src, dtype=tgt)
        return np.asarray(src[...] if not isinstance(src, np.ndarray) else src).astype(tgt)
    if isinstance(a, np.ndarray):
        if a.dtype.kind == "O":
            return _exact_objects(a, what)
        tgt = target_dtype(a.dtype, what)
        if tgt is None:
            return a
        if a.dtype.kind in "biu":
            _exact_ints(a, what)
            if keep_integers and a.dtype.kind in "iu":
                return a
        _note(widened, what, a, a.dtype, tgt)
        return a.astype(tgt)
    if isinstance(a, np.generic):
        if a.dtype.kind in "fc" and target_dtype(a.dtype, what) is None:
            return a                              # a binary64 scalar stays a scalar
        return binary64(np.asarray(a), what, widened, keep_integers=keep_integers)
    if isinstance(a, (list, tuple)):
        # numeric data given as a (nested) list or tuple become an ndarray whatever their dtype, so that an operator
        # given this way is a stored matrix as in any format; ragged lists and lists with strings (the operator forms
        # of operators.py) are left to the caller
        try:
            arr = np.asarray(a)
        except OverflowError:                     # a Python integer of 2^1024 or more next to a float
            raise PrecisionError(f"{what}: an integer of magnitude 2^1024 or more is not a binary64 number; "
                                 "refused") from None
        except (ValueError, TypeError):
            return a
        if arr.dtype.kind in "USV":
            return a
        if arr.dtype.kind == "O":
            flat = arr.ravel().tolist()
            if any(isinstance(v, (str, bytes, list, tuple, dict)) for v in flat):
                return a                          # mixed: an operator form
            return _exact_objects(arr, what)      # Python integers beyond 2^64, Fractions, Decimals, ...
        if arr.dtype.kind in "fc":
            _check_python_ints(a, arr, what)      # integers beyond 2^53 that np.asarray rounded next to a float
        return binary64(arr, what, widened, keep_integers=keep_integers)
    if hasattr(a, "__array__") and not hasattr(a, "matvec"):
        # an array-like (a data frame, a tensor on the CPU, a zarr array: an object with __array__) is materialized
        # in memory whatever its dtype; one whose conversion fails is refused. Operators with a matvec stay as given.
        return binary64(_materialize(a, what), what, widened, keep_integers=keep_integers)
    return a


def check_values(values, what: str = "eigenvalues") -> None:
    """PrecisionError when converting the claimed `values` to binary64 (dtype=float, or complex) would change them,
    under the rule of binary64(): a floating type wider than binary64, an integer that binary64 cannot hold (also an
    integer in a list next to floats, a Python integer beyond 2^64 or of 2^1024 or more), an object value v with
    float(v) != v (a Fraction 1/3, a Decimal 0.1), and non-numeric data. Narrower numeric types convert exactly. A
    complex array of the eigenvalues of a Hermitian claim is converted by the callers as in 3.x (dtype=float: the real
    part, with numpy's ComplexWarning)."""
    try:
        arr = np.asarray(values)
    except OverflowError:
        raise PrecisionError(f"{what}: an integer of magnitude 2^1024 or more is not a binary64 number; refused") from None
    except (ValueError, TypeError):
        raise PrecisionError(f"{what}: not a numeric array") from None
    if arr.dtype.kind == "O":
        arr = _exact_objects(arr, what)
    elif isinstance(values, (list, tuple)) and arr.dtype.kind in "fc":
        _check_python_ints(values, arr, what)
    if arr.dtype.kind not in "biufc":
        raise PrecisionError(f"{what}: dtype {arr.dtype.name} is not numeric data")
    target_dtype(arr.dtype, what)
    if arr.dtype.kind in "iu":
        _exact_ints(arr, what)

def claimed_reals(values) -> np.ndarray:
    """The claimed eigenvalues of a Hermitian claim as binary64 reals, converted as in 3.x (dtype=float; a complex array:
    the real part, with numpy's ComplexWarning), except that a value whose imaginary part is not finite becomes nan.
    Such a value is not finite, and the callers' rule for non-finite claimed values applies to it (INCONCLUSIVE, or
    refused where non-finite values are refused); its real part alone would otherwise be certified. Call after
    check_values. The conversion is that of 3.x on the values as given, so a list of Python complex numbers is refused
    with numpy's TypeError, as before."""
    out = np.array(values, dtype=float)
    arr = np.asarray(values)
    if arr.dtype.kind == "c":
        out[~np.isfinite(arr.imag)] = np.nan
    return out


def dropped_imaginary_parts(values) -> Optional[Dict[str, Any]]:
    """What claimed_reals drops from complex claimed values: the number and the largest magnitude of the finite nonzero
    imaginary parts, or None (real values, or no such part). The largest magnitude is max(im.max(), -im.min()), exact."""
    arr = np.asarray(values)
    if arr.dtype.kind != "c":
        return None
    im = arr.imag.ravel()
    im = im[np.isfinite(im) & (im != 0)]
    if not im.size:
        return None
    return {"count": int(im.size), "max_abs": float(max(im.max(), -im.min()))}


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
            "note": ("inputs stored in a narrower floating format than binary64 (float16, float32, complex64) were "
                     "converted to binary64 before any product; the conversion is exact, so the statements concern "
                     "the values as stored; "
                     "the record's hashes are of the converted binary64 data, source_sha256 of each input as given")}
