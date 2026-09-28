"""Floating-point error model used by every bound in CERTIFY-ED v2.

All "rigorous" statements produced by the float64 code paths are rigorous
*under the standard model of floating-point arithmetic* (Higham 2002, eq. 2.4):

    fl(a op b) = (a op b)(1 + delta),   |delta| <= u,   u = 2**-53,

for IEEE-754 binary64 with round-to-nearest, no underflow or overflow, and
BLAS / sparse kernels whose inner products obey the usual gamma_n bound in any
summation order. Fused multiply-add satisfies the model. The model is violated
by: flush-to-zero/denormals-are-zero modes combined with tiny values,
alternative matrix-multiplication algorithms (e.g. Strassen-type) inside BLAS,
and reduced-precision tensor-core paths (e.g. TF32). The certificate records
these assumptions; `certify_ed.rigorous` provides an interval-arithmetic path
(Arb via python-flint) that does not rely on them, for small dimensions.

Every helper below returns an UPPER bound on a true (exact-arithmetic) quantity
given floating-point inputs. Where a computed floating-point number is used as
an upper bound it is inflated by a relative guard so that the rounding of the
final operation cannot make the bound too small.
"""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp

U = 2.0 ** -53          # unit roundoff for binary64
_GUARD = 1.0 + 8 * U    # relative guard applied to computed upper bounds


def gamma(n: float) -> float:
    """gamma_n = n u / (1 - n u) (Higham 2002, Lemma 3.1)."""
    nu = n * U
    if nu >= 0.5:
        raise ValueError(f"gamma_{n} undefined: n*u = {nu} too large")
    return nu / (1.0 - nu)


def gamma_dot(n: int, complex_arith: bool) -> float:
    """Error constant for an inner product with at most n nonzero terms.

    Real arithmetic: gamma_n (Higham Eq. 3.5). Complex arithmetic: we use the
    deliberately conservative 2*gamma_{n+4}, which dominates the complex
    inner-product bounds of Higham 2002 Section 3.6.
    """
    n = max(int(n), 1)
    return 2.0 * gamma(n + 4) if complex_arith else gamma(n)


def up(x: float) -> float:
    """Guarded upper bound for a non-negative computed quantity."""
    return float(x) * _GUARD + 0.0


def norm2_upper(v: np.ndarray) -> float:
    """Upper bound on the exact 2-norm of a stored float vector."""
    n = v.size
    s = float(np.linalg.norm(v))
    return s * (1.0 + gamma(n + 2)) * _GUARD


def norm2_lower(v: np.ndarray) -> float:
    """Lower bound on the exact 2-norm of a stored float vector."""
    n = v.size
    s = float(np.linalg.norm(v))
    return max(0.0, s * (1.0 - gamma(n + 2)) / _GUARD)


def fro_upper(A) -> float:
    """Upper bound on the exact Frobenius norm of a stored dense/sparse array."""
    if sp.issparse(A):
        data = A.data if hasattr(A, "data") else A.tocsr().data
        return norm2_upper(np.asarray(data))
    return norm2_upper(np.asarray(A).ravel())


def inf_norm_upper(A) -> float:
    """Upper bound on ||A||_inf (max absolute row sum) of a stored matrix.

    For Hermitian A this also bounds ||A||_2 (since ||A||_2 <= sqrt(||A||_1 ||A||_inf)
    and ||A||_1 = ||A||_inf).
    """
    if sp.issparse(A):
        A = A.tocsr()
        absA = abs(A)
        rows = np.asarray(absA.sum(axis=1)).ravel()
        width = int(np.diff(A.indptr).max()) if A.shape[0] else 1
    else:
        rows = np.abs(np.asarray(A)).sum(axis=1)
        width = A.shape[1]
    m = float(rows.max()) if rows.size else 0.0
    return m * (1.0 + gamma(width + 1)) * _GUARD


def max_nnz_per_row(A) -> int:
    if sp.issparse(A):
        A = A.tocsr()
        return int(np.diff(A.indptr).max()) if A.shape[0] else 0
    return int(np.asarray(A).shape[1])


def is_complex(*arrays) -> bool:
    return any(np.iscomplexobj(a.data if sp.issparse(a) else a) for a in arrays)


def float_env_record() -> dict:
    """Description of the arithmetic model assumed by float64 bounds."""
    return {
        "arithmetic": "IEEE-754 binary64, round-to-nearest",
        "unit_roundoff": U,
        "model": "standard model fl(a op b)=(a op b)(1+d), |d|<=u (Higham 2002, Eq. 2.4)",
        "assumptions": [
            "no underflow/overflow in the computations used for bounds",
            "BLAS and sparse kernels satisfy gamma_n inner-product bounds in any summation order",
            "no reduced-precision matrix-multiply modes (e.g. TF32) and no Strassen-type BLAS",
        ],
        "complex_inner_product_constant": "2*gamma_{n+4} (conservative)",
    }


def safe_sqrt_upper(x: float) -> float:
    return math.sqrt(max(x, 0.0)) * _GUARD


# ---------------------------------------------------------------------------
# error-free transformations: accurately rounded inner products
# ---------------------------------------------------------------------------

_SPLITTER = 134217729.0  # 2**27 + 1 (Dekker)


def _two_product(a: np.ndarray, b: np.ndarray):
    """p, e with p + e == a*b exactly (Dekker 1971), barring overflow/underflow."""
    p = a * b
    c = _SPLITTER * a
    ah = c - (c - a)
    al = a - ah
    c = _SPLITTER * b
    bh = c - (c - b)
    bl = b - bh
    e = al * bl - (((p - ah * bh) - al * bh) - ah * bl)
    return p, e


def exact_dot(x: np.ndarray, y: np.ndarray):
    """Return (s, err) with |s - x^H y| <= err, where s is the correctly rounded
    value of the exact inner product (computed with TwoProduct and math.fsum).

    err = 2u|s| per real component (fsum is correctly rounded; the factor 2
    covers the conversion of the complex result). Underflow is excluded by the
    stated arithmetic assumptions.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        xr, xi = np.real(x).astype(float), np.imag(x).astype(float)
        yr, yi = np.real(y).astype(float), np.imag(y).astype(float)
        p1, e1 = _two_product(xr, yr)
        p2, e2 = _two_product(xi, yi)
        p3, e3 = _two_product(xr, yi)
        p4, e4 = _two_product(xi, yr)
        re = math.fsum(np.concatenate([p1, e1, p2, e2]).tolist())
        im = math.fsum(np.concatenate([p3, e3, -p4, -e4]).tolist())
        s = complex(re, im)
        err = 2 * U * (abs(re) + abs(im))
        return s, err
    p, e = _two_product(x.astype(float), y.astype(float))
    s = math.fsum(np.concatenate([p, e]).tolist())
    return s, 2 * U * abs(s)


def exact_gram(X: np.ndarray):
    """Gram matrix X^H X with an entrywise error bound (accurately rounded entries)."""
    g = X.shape[1]
    G = np.zeros((g, g), dtype=complex if np.iscomplexobj(X) else float)
    E = np.zeros((g, g))
    for i in range(g):
        for j in range(i, g):
            s, e = exact_dot(X[:, i], X[:, j])
            G[i, j] = s
            E[i, j] = e
            if i != j:
                G[j, i] = np.conj(s)
                E[j, i] = e
    return G, E
