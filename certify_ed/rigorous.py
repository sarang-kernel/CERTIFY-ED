"""Interval-arithmetic path (Arb, via python-flint) that does not rely on the
floating-point model assumed by the float64 bounds.

* `arb_residual_radii` recomputes the Theorem-1 radii ||A x - mu x|| / ||x|| in
  ball arithmetic. Inputs are float64 numbers and are therefore exact; every
  arithmetic operation is outward-rounded by Arb, so the returned radii are
  mathematically rigorous for the stored operator. Cost is O(nnz * k) ball
  operations; intended for dimensions up to a few thousand.
* `arb_eigenvalues` computes rigorous eigenvalue enclosures of a dense matrix
  with Arb's certified eigenvalue routine (van der Hoeven-Mourrain or Rump
  algorithm, with multiple eigenvalues handled as clusters). Used in the test
  suite to validate the float64 bounds, and available to users for small n.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from .precision import binary64

try:
    import flint
    HAVE_FLINT = True
except Exception:  # pragma: no cover
    HAVE_FLINT = False


def _require():
    if not HAVE_FLINT:
        raise ImportError("python-flint is required for the interval-arithmetic path "
                          "(pip install python-flint)")


def _upper(b) -> float:
    """Upper endpoint of an arb ball as a float (rounded up)."""
    import math
    m = float(b.mid())
    r = float(b.rad())
    u = m + r
    return math.nextafter(math.nextafter(u, math.inf), math.inf)


def _lower(b) -> float:
    import math
    m = float(b.mid())
    r = float(b.rad())
    l = m - r
    return math.nextafter(math.nextafter(l, -math.inf), -math.inf)


def arb_residual_radii(A, mu, X, prec: int = 128, lower: bool = False) -> np.ndarray:
    """Rigorous ||A x_i - mu_i x_i||_2 / ||x_i||_2 for each column, in ball arithmetic.

    Returns the upper endpoint of each ball (rounded up), or the lower endpoint (rounded down)
    when ``lower`` is True."""
    _require()
    old = flint.ctx.prec
    flint.ctx.prec = prec
    try:
        A = sp.csr_matrix(binary64(A, "operator"))    # exact, before the conversion; wider formats refused (precision.py)
        X = binary64(np.asarray(X), "eigenvectors")
        if X.ndim == 1:
            X = X[:, None]
        mu = np.asarray(mu, dtype=float)
        n, k = X.shape
        cplx = np.iscomplexobj(A.data) or np.iscomplexobj(X)
        Num = flint.acb if cplx else flint.arb
        indptr, indices, data = A.indptr, A.indices, A.data
        out = np.zeros(k)
        for j in range(k):
            x = X[:, j]
            xb = [Num(complex(v)) if cplx else Num(float(v)) for v in x]
            m = flint.arb(float(mu[j]))
            s2 = flint.arb(0)
            xn2 = flint.arb(0)
            for i in range(n):
                acc = Num(0)
                for p in range(indptr[i], indptr[i + 1]):
                    a = data[p]
                    acc += (Num(complex(a)) if cplx else Num(float(a))) * xb[indices[p]]
                acc -= m * xb[i]
                s2 += (abs(acc) ** 2) if cplx else acc * acc
                xn2 += (abs(xb[i]) ** 2) if cplx else xb[i] * xb[i]
            r = s2.sqrt() / xn2.sqrt()
            out[j] = _lower(r) if lower else _upper(r)
        return out
    finally:
        flint.ctx.prec = old


def arb_eigenvalues(A, prec: int = 128, algorithm: str = None) -> List[Tuple[float, float]]:
    """Rigorous enclosures [lo, hi] of all eigenvalues of a dense Hermitian matrix
    (real parts; imaginary parts are verified to contain 0). Clusters of
    numerically multiple eigenvalues are returned as repeated identical balls."""
    _require()
    old = flint.ctx.prec
    flint.ctx.prec = prec
    try:
        M = binary64(A if sp.issparse(A) else np.asarray(A), "operator")    # exact, before toarray (precision.py)
        M = M.toarray() if sp.issparse(M) else M
        n = M.shape[0]
        E, err = None, None
        for p in (prec, 2 * prec, 4 * prec, 8 * prec):
            # Arb may fail to isolate tight clusters at a given precision; retry with more bits
            flint.ctx.prec = p
            rows = [[flint.acb(complex(M[i, j])) for j in range(n)] for i in range(n)]
            try:
                E = flint.acb_mat(rows).eig(multiple=True, algorithm=algorithm)
                break
            except ValueError as e:
                err = e
        if E is None:
            raise ArithmeticError(f"Arb could not enclose the eigenvalues up to {8 * prec} bits: {err}")
        out = []
        for e in E:
            if not e.imag.contains(0):
                raise ArithmeticError("eigenvalue enclosure excludes the real axis: input not Hermitian")
            out.append((_lower(e.real), _upper(e.real)))
        return sorted(out)
    finally:
        flint.ctx.prec = old
