"""Compatibility helpers for the declared version floors: the eigenvalues of a 1 x 1 Hermitian matrix.

scipy.linalg.eigh and eigvalsh refuse a 1 x 1 matrix in scipy 1.8.1 and 1.10.1 (LAPACK from OpenBLAS) with the
drivers evr (the default) and evd, real and complex: the workspace query gives a size the wrapper refuses (liwork = 1
where it requires 10 n for evr); the drivers ev and evx run. Every call of the package that can meet a 1 x 1
standard problem goes through `eigh` below (tests/test_version_floors.py audits the calls).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.linalg as la

__all__ = ["eigh"]


def _lapack_dtype(a: np.ndarray) -> np.dtype:
    """The dtype scipy's LAPACK wrappers compute in for a: single precision kept, everything else double."""
    if a.dtype.kind == "c":
        return np.dtype(np.complex64 if a.dtype == np.complex64 else np.complex128)
    return np.dtype(np.float32 if a.dtype in (np.float16, np.float32) else np.float64)


def eigh(a, eigvals_only: bool = False, driver: Optional[str] = None,
         subset_by_index: Optional[Sequence[int]] = None):
    """scipy.linalg.eigh of a Hermitian matrix a, with the 1 x 1 case taken from its entry.

    The eigenvalue of a 1 x 1 Hermitian matrix is the real part of its entry and its eigenvector is [1]: what LAPACK
    returns where it runs. As in scipy, a non-finite entry is refused (check_finite) and the dtypes are those of the
    LAPACK routine (single precision kept, integers in double)."""
    a = np.asarray(a)
    if a.shape != (1, 1):
        kw = {} if driver is None else {"driver": driver}
        if subset_by_index is not None:
            kw["subset_by_index"] = subset_by_index
        return la.eigh(a, eigvals_only=eigvals_only, **kw)
    if not np.all(np.isfinite(a)):
        raise ValueError("array must not contain infs or NaNs")
    if subset_by_index is not None and tuple(int(i) for i in subset_by_index) != (0, 0):
        raise ValueError(f"subset_by_index {tuple(subset_by_index)} is out of range for a 1 x 1 matrix")
    dt = _lapack_dtype(a)
    w = np.real(a).reshape(1).astype(np.empty(0, dt).real.dtype)
    if eigvals_only:
        return w
    return w, np.ones((1, 1), dtype=dt)
