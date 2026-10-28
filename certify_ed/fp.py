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

# No-underflow hypothesis (INT-01, enforced since 3.2.0.dev0 unless Options(underflow_check=False)): every
# nonzero operand, radius and residual bound of the floating-point statements has magnitude >= 2^-900. The
# standard model fl(a op b) = (a op b)(1 + delta) does not hold for results in the subnormal range
# (below 2^-1022), where the absolute rounding error is bounded but the relative one is not; the margin
# between 2^-900 and 2^-1022 is what the hypothesis leaves for the intermediate quantities of the bounds.
UNDERFLOW_THRESHOLD = 2.0 ** -900


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


# Directed rounding of one addition or subtraction. Under round-to-nearest the computed a +/- b is within
# half an ulp of the exact value, so one step outward (nextafter) bounds the exact value. Interval
# endpoints (mu - r, mu + r), shifts (sigma - beta) and separations are formed with these, so that a
# reported endpoint is on the safe side of the exact one and comparisons between endpoints are conservative.
def sub_down(a, b):
    """Lower bound on the exact a - b (elementwise for arrays)."""
    return np.nextafter(np.subtract(a, b), -np.inf)


def add_up(a, b):
    """Upper bound on the exact a + b (elementwise for arrays)."""
    return np.nextafter(np.add(a, b), np.inf)


def sub_up(a, b):
    """Upper bound on the exact a - b."""
    return np.nextafter(np.subtract(a, b), np.inf)


def add_down(a, b):
    """Lower bound on the exact a + b."""
    return np.nextafter(np.add(a, b), -np.inf)


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


def tiny_nonzero(a, threshold: float = UNDERFLOW_THRESHOLD, max_indices: int = 8):
    """Nonzero entries of magnitude below `threshold` in a stored array (the no-underflow hypothesis).

    Complex data are scanned component by component (the real and imaginary parts are the operands of
    the arithmetic). Exact zeros are not reported; non-finite entries are not this function's concern.
    Returns None when nothing is found, else {'count', 'first_indices', 'min_magnitude'} with the flat
    indices (component index for complex data) of the first few offenders.
    """
    if a is None:
        return None
    if sp.issparse(a):
        a = a.data
    arr = np.asarray(a)
    if arr.size == 0 or arr.dtype.kind not in "fc":
        return None
    comp = np.ascontiguousarray(arr).view(arr.real.dtype).ravel() if arr.dtype.kind == "c" else arr.ravel()
    mag = np.abs(comp)
    with np.errstate(invalid="ignore"):
        hit = (mag > 0) & (mag < threshold)
    if not hit.any():
        return None
    idx = np.nonzero(hit)[0]
    return {"count": int(idx.size), "first_indices": idx[:max_indices].tolist(),
            "min_magnitude": float(mag[idx].min())}


def min_nonzero_magnitude(a):
    """Smallest nonzero |entry| (component-wise for complex data); None when there is none."""
    if a is None:
        return None
    if sp.issparse(a):
        a = a.data
    arr = np.asarray(a)
    if arr.size == 0 or arr.dtype.kind not in "fc":
        return None
    comp = np.ascontiguousarray(arr).view(arr.real.dtype).ravel() if arr.dtype.kind == "c" else arr.ravel()
    mag = np.abs(comp)
    mag = mag[(mag > 0) & np.isfinite(mag)]
    return float(mag.min()) if mag.size else None


def underflow_scan(parts, threshold: float = UNDERFLOW_THRESHOLD):
    """Scan named arrays for the no-underflow hypothesis -> list of findings.

    `parts` holds (name, array) or (name, array, threshold) tuples; the default threshold is
    UNDERFLOW_THRESHOLD. Each finding is tiny_nonzero(array) plus the name under 'where' and the threshold used; an empty list
    means every nonzero magnitude is at or above its threshold (the hypothesis holds for what was scanned).
    """
    out = []
    for part in parts:
        name, arr = part[0], part[1]
        thr = part[2] if len(part) > 2 else threshold
        f = tiny_nonzero(arr, thr)
        if f is not None:
            f["where"] = name
            f["threshold"] = float(thr)
            out.append(f)
    return out


def float_env_record() -> dict:
    """Description of the arithmetic model assumed by float64 bounds."""
    return {
        "arithmetic": "IEEE-754 binary64, round-to-nearest",
        "unit_roundoff": U,
        "model": "standard model fl(a op b)=(a op b)(1+d), |d|<=u (Higham 2002, Eq. 2.4)",
        "assumptions": [
            "no underflow/overflow in the computations used for bounds (nonzero operands, radii and residual "
            "bounds of magnitude >= 2^-900; see 'underflow_check')",
            "BLAS and sparse kernels satisfy gamma_n inner-product bounds in any summation order",
            "no reduced-precision matrix-multiply modes (e.g. TF32) and no Strassen-type BLAS",
        ],
        "complex_inner_product_constant": "2*gamma_{n+4} (conservative)",
    }


def probe_float_environment() -> dict:
    """Run-time test of the hypotheses behind the standard-model bounds, in this process.

    Shared libraries compiled with -ffast-math can switch on flush-to-zero (FTZ) and
    denormals-are-zero (DAZ) for the whole process when they are imported (documented for thousands
    of PyPI binaries); a directed rounding mode (toward +inf, -inf or zero) or a reduced-precision BLAS
    also break the model. The probes use NumPy arrays so that nothing is constant-folded."""
    t = np.array([2.0 ** -1022, 2.0 ** -1074, 1.0, 2.0 ** -60, 2.0 ** -53, 3.0 * 2.0 ** -54], dtype=np.float64)
    checks = {}
    # compare bit patterns: under DAZ a comparison with a subnormal treats it as zero
    checks["gradual_underflow"] = bool((t[0:1] / 2.0).view(np.uint64)[0] == np.uint64(1 << 51))  # FTZ
    checks["subnormal_operands"] = bool((t[1:2] * 1.0).view(np.uint64)[0] == np.uint64(1))       # DAZ
    checks["round_to_nearest_up"] = bool((t[2:3] + t[3:4])[0] == 1.0)                # round-up: > 1
    checks["round_to_nearest_down"] = bool((-t[2:3] - t[3:4])[0] == -1.0)            # round-down
    # round-toward-zero passes both tests above (IMP-24): 1 + 3*2^-54 = 1 + 0.75 ulp must round to 1 + 2^-52
    # (toward zero or down: 1), and its mirror -1 - 3*2^-54 to -1 - 2^-52 (toward zero or up: -1)
    checks["round_to_nearest_not_toward_zero_pos"] = bool((t[2:3] + t[5:6])[0] == 1.0 + 2.0 ** -52)
    checks["round_to_nearest_not_toward_zero_neg"] = bool((-t[2:3] - t[5:6])[0] == -1.0 - 2.0 ** -52)
    checks["ties_to_even"] = bool((t[2:3] + t[4:5])[0] == 1.0)
    # a BLAS dot product whose exact result needs the full 53-bit significand
    a = np.full(64, 1.0 + 2.0 ** -40)
    b = np.zeros(64)
    b[0] = 1.0
    checks["blas_full_precision"] = bool(np.dot(a, b) == 1.0 + 2.0 ** -40)
    M = np.zeros((4, 4))
    M[0, 0] = 1.0 + 2.0 ** -40
    checks["gemm_full_precision"] = bool((M @ np.eye(4))[0, 0] == 1.0 + 2.0 ** -40)
    ok = all(checks.values())
    return {"standard_model_hypotheses_hold": ok, "checks": checks,
            "failed": [k for k, v in checks.items() if not v]}


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
