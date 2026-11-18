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

# Underflow-tolerant model (INT-05, docs/proofs/int05_underflow.md). One binary64 operation with exact result z
# returns fl(z) = z (1 + delta) + eta, |delta| <= u, |eta| <= u_min = 2^-1075: for |z| >= 2^-1022 the grid
# spacing at z is <= 2^-52 |z| (so eta = 0), below it the spacing is 2^-1074 (so delta = 0). An addition or
# subtraction whose exact result lies below 2^-1022 is exact (both operands are multiples of 2^-1074, so is
# the sum, and every such multiple below 2^-1022 is a subnormal number): eta = 0 for + and -. 2^-1075 itself
# is not a binary64 number; additive terms are formed in units of ULP_MIN = 2^-1074 >= 2 u_min, or with
# math.ldexp(., -1075) followed by + ULP_MIN (the ldexp rounds to the subnormal grid, by at most u_min).
ULP_MIN = 2.0 ** -1074          # smallest positive binary64 number (= 2 u_min)
_SQRT_HALF_2M537 = math.sqrt(0.5) * 2.0 ** -537      # sqrt(u_min) = sqrt(0.5) 2^-537, rounded once

# No-underflow hypothesis (INT-01, enforced since 3.2.0.dev0 unless Options(underflow_check=False)): every
# nonzero operand, radius and residual bound of the floating-point statements has magnitude >= 2^-900. The
# standard model fl(a op b) = (a op b)(1 + delta) does not hold for results in the subnormal range
# (below 2^-1022), where the absolute rounding error is bounded but the relative one is not; the margin
# between 2^-900 and 2^-1022 is what the hypothesis leaves for the intermediate quantities of the bounds.
UNDERFLOW_THRESHOLD = 2.0 ** -900

# Range of the floating-point guards (3.2.2, roadmap N-01, finding L-3 of the Lean formalization). The guards of
# Theorem 1 (docs/proofs/thm1_fp-accounting.md: Lambda4 for the norms, Lambda5 for |A||x|, Lambda6 for E) and of the
# sparse residual (docs/proofs/D-01_sparse-residual.md, hypothesis H3) are established for vectors of 1 <= n <= 2^26
# entries and row widths 1 <= w <= 2^26 only: Lambda4 with N = 2n fails from n = 84886744 on, and the sentence that
# extended Lambda6 beyond 2^26 fails at w + 2 = floor(2^53 / 3) (erratum of 10 Nov 2026 in thm1_fp-accounting.md).
# Outside the range no bound is returned: the guarded functions raise GuardRangeError, and certify() reports the
# statements that rest on them as INCONCLUSIVE with reason code FP_GUARD_RANGE. Inside the range nothing changes.
GUARD_RANGE_MAX = 2 ** 26


class GuardRangeError(ValueError):
    """A floating-point guard is applied outside the range in which it is established (GUARD_RANGE_MAX)."""

    def __init__(self, quantity: str, value: int, where: str = ""):
        self.quantity, self.value, self.where = str(quantity), int(value), str(where)
        self.limit = int(GUARD_RANGE_MAX)
        super().__init__(f"{self.quantity} = {self.value} exceeds {self.limit} = 2^26, the range of the floating-point "
                         f"guards{' in ' + self.where if self.where else ''} (docs/proofs/thm1_fp-accounting.md, "
                         f"Lambda4 to Lambda6)")

    def to_dict(self) -> dict:
        return {"quantity": self.quantity, "value": self.value, "limit": self.limit, "where": self.where}


def guard_range_findings(n=None, w=None, where: str = "") -> list:
    """[{'quantity', 'value', 'limit', 'where'}] for each of n (entries of a normed vector, or the dimension) and w
    (row width) that exceeds GUARD_RANGE_MAX (read at call time); [] when both are inside the range or None."""
    out = []
    for name, val in (("n", n), ("w", w)):
        if val is not None and int(val) > GUARD_RANGE_MAX:
            out.append(GuardRangeError(name, int(val), where).to_dict())
    return out


def check_guard_range(n=None, w=None, where: str = "") -> None:
    """Raise GuardRangeError when n or w exceeds GUARD_RANGE_MAX (see guard_range_findings)."""
    for name, val in (("n", n), ("w", w)):
        if val is not None and int(val) > GUARD_RANGE_MAX:
            raise GuardRangeError(name, int(val), where)


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


# Upper moduli (3.3.0.dev0, docs/proofs/fp_modulus-up.md; UNVERIFIED until Sarang's check, D-12). NumPy's complex
# absolute value is not correctly rounded and no bound on its error is established: on the Windows reference host
# (numpy 2.4.4, AVX512F) its relative error reaches 2.47u (z = 0.02019294356747192 - 0.5012539518584045j). modulus_up
# forms the modulus of complex data from correctly rounded *, +, sqrt and exact scaling by powers of two; it is >= the
# exact modulus, so a bound that takes its moduli from it needs no rounding count for them.
_SMALLEST_NORMAL = 2.0 ** -1022
_SAFE_LO, _SAFE_HI = 2.0 ** -500, 2.0 ** 500     # max(|x|, |y|) in this range: no scaling is needed
_CHUNK = 1 << 18                                  # entries per pass of modulus_up (bounds its temporaries)


def _modulus_up_scaled(x, y):
    """modulus_up for components x, y >= 0 (or inf, nan) of any magnitude: scaled by the power of two 2^-e with
    max(x, y) = f 2^e, f in [1/2, 1), and rescaled; zero, inf and nan components are handled by the caller."""
    e = np.frexp(np.maximum(x, y))[1]
    np.negative(e, out=e)
    m = np.ldexp(x, e)                           # the larger component lands in [1/2, 1), exactly
    b = np.ldexp(y, e)                           # the smaller one may round to the subnormal grid
    np.multiply(m, m, out=m)
    np.multiply(b, b, out=b)
    np.add(m, b, out=m)
    np.sqrt(m, out=m)
    np.multiply(m, _GUARD, out=m)
    np.negative(e, out=e)
    np.ldexp(m, e, out=m)                        # exact unless the result overflows (inf) or is subnormal
    small = m < _SMALLEST_NORMAL
    if np.any(small):
        m[small] = np.nextafter(m[small], np.inf)    # the rescaling may have rounded a subnormal result down
    return m


def modulus_up(z):
    """m >= |z| for every entry of a real or complex array, same shape (float64 for complex, integer and boolean input;
    the input's type for real floating input); a float for a scalar (a NumPy scalar for a real type wider than
    binary64, which a float would round).

    Real floating input: np.abs(z), which is exact. Integer and boolean input: the absolute values as float64, one step
    up (np.nextafter) from 2^53 on, where the conversion may round (np.abs itself wraps at the most negative integer).
    Complex input z = x + iy, complex64 or complex128 (docs/proofs/fp_modulus-up.md, Lemma M):
    m = fl(fl(sqrt(fl(fl(x x) + fl(y y)))) (1 + 8u)) when max(|x|, |y|) is 0 or lies in [2^-500, 2^500]; otherwise
    the same with the components first scaled by the power of two 2^-e, max(|x|, |y|) = f 2^e with f in [1/2, 1)
    (np.frexp, np.ldexp), the result scaled back by 2^e and moved one step up (np.nextafter) when it is below
    2^-1022, where the rescaling may have rounded it down. The factor (1 + 8u) covers the five roundings and the
    rounding of a scaled smaller component to the subnormal grid. A zero component gives the other's absolute value
    (exact); an infinite component gives inf; otherwise a nan component gives nan (as np.abs does); an m above the
    largest float is inf, never a smaller finite value. Tightness: a finite m satisfies m <= |z| (1 + 12u) + 2^-1073,
    and m <= |z| (1 + 12u) when |z| >= 2^-1022. Floating-point warnings are off inside (np.errstate); nothing is
    raised for inf or nan.

    Rejected with TypeError: scipy sparse matrices (use abs_up), object arrays (their entries need not be binary64
    numbers) and complex types wider than complex128 (np.clongdouble where long double is wider than binary64), whose
    components would be rounded on the way to float64. The array is processed in passes of at most _CHUNK entries
    and is not copied, whatever its memory layout, so the temporaries beyond the float64 result are bounded by about
    6 _CHUNK float64 numbers (12 MiB; 6.8 MiB measured on 2^24 entries)."""
    if sp.issparse(z):
        raise TypeError("modulus_up takes arrays and scalars; use fp.abs_up for a scipy sparse matrix")
    z = np.asarray(z)
    kind = z.dtype.kind
    if kind == "f":
        m = np.abs(z)
        return float(m) if m.ndim == 0 and np.finfo(z.dtype).nmant <= 52 else m
    if kind in "iub":
        m = np.abs(z.astype(np.float64))
        big = m >= 2.0 ** 53
        if big.any():
            m = np.where(big, np.nextafter(m, np.inf), m)
        return m if m.shape else float(m)
    if kind != "c":
        raise TypeError(f"modulus_up: no upper modulus for dtype {z.dtype} (convert to a numeric dtype first)")
    if np.finfo(z.dtype).nmant > 52:
        raise TypeError(f"modulus_up: {z.dtype} is wider than complex128; its components are not binary64 numbers")
    if z.ndim == 0:
        z = z.reshape(1)
        m = np.empty(1, dtype=np.float64)
        _modulus_up_into(z, m)
        return float(m[0])
    m = np.empty(z.shape, dtype=np.float64)
    _modulus_up_into(z, m)
    return m


def _modulus_up_into(z, out):
    """out[...] = modulus_up(z) for a complex array z of any shape and strides, out a C-contiguous float64 array of
    z's shape. Works in passes of at most _CHUNK entries along the leading axes and makes no copy of z, so the
    temporaries do not depend on the size or the memory layout of z."""
    if z.ndim > 1 and z.size > _CHUNK:
        row = z.size // z.shape[0]
        if row >= _CHUNK:
            for a in range(z.shape[0]):
                _modulus_up_into(z[a], out[a])
        else:
            step = _CHUNK // row
            for a in range(0, z.shape[0], step):
                _modulus_up_into(z[a:a + step], out[a:a + step])
        return
    o = out.reshape(-1)                              # a view: out is C-contiguous
    if z.ndim > 1:                                   # at most _CHUNK entries; their components in C order
        x = np.abs(z.real, order="C").astype(np.float64, copy=False).reshape(-1)
        y = np.abs(z.imag, order="C").astype(np.float64, copy=False).reshape(-1)
        _modulus_up_block(x, y, o)
        return
    for a in range(0, z.size, _CHUNK):
        c = z[a:a + _CHUNK]
        x = np.abs(c.real).astype(np.float64, copy=False)
        y = np.abs(c.imag).astype(np.float64, copy=False)
        _modulus_up_block(x, y, o[a:a + _CHUNK])


def _modulus_up_block(x, y, m):
    """modulus_up of the complex numbers x + iy with x, y >= 0 float64 (the absolute values of the components, exact),
    written into m (float64, same size)."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        big = np.maximum(x, y)
        fast = ((big >= _SAFE_LO) & (big <= _SAFE_HI)) | (big == 0.0)    # 0 + 0i: exact on this branch too
        del big
        np.multiply(x, x, out=m)
        b = np.multiply(y, y)
        np.add(m, b, out=m)
        del b
        np.sqrt(m, out=m)
        np.multiply(m, _GUARD, out=m)
        if not fast.all():
            i = np.flatnonzero(~fast)
            m[i] = _modulus_up_scaled(x[i], y[i])
            np.copyto(m, np.inf, where=np.isinf(x) | np.isinf(y))
    zx, zy = x == 0.0, y == 0.0
    if zx.any():
        np.copyto(m, y, where=zx)
    if zy.any():
        np.copyto(m, x, where=zy)


def abs_up(A):
    """The entrywise moduli of a stored dense array or scipy sparse matrix, each rounded up: abs(A) for real floating
    data (exact), modulus_up entrywise for complex, integer and boolean data (float64; integers stepped up from 2^53).

    A sparse matrix in CSR, CSC, BSR or COO format has its duplicate entries summed in place first (A.sum_duplicates(),
    in the type of its data), as abs(A) does; the result, and every product formed with A afterwards, refer to that
    summed matrix (docs/proofs/fp_modulus-up.md, section 1: a product formed before the call is not covered). DIA keeps
    its stored diagonals; LIL, DOK and other formats are converted to CSR (a copy, so A itself is not changed). The
    result has A's format (CSR for LIL, DOK and the others) and A's class (sparse matrix or sparse array)."""
    if sp.issparse(A):
        if A.format not in ("csr", "csc", "bsr", "coo", "dia"):
            A = A.tocsr()
        if A.data.dtype.kind == "f":
            return abs(A)
        if A.format != "dia":
            A.sum_duplicates()
        data = modulus_up(A.data)
        if A.format in ("csr", "csc", "bsr"):
            return type(A)((data, A.indices.copy(), A.indptr.copy()), shape=A.shape)
        if A.format == "coo":
            return type(A)((data, tuple(c.copy() for c in A.coords)), shape=A.shape)
        return type(A)((data, A.offsets.copy()), shape=A.shape)
    return modulus_up(A)


def norm_underflow_term(N: int) -> float:
    """T_N >= sqrt(N u_min) sqrt((1 + gamma_N)/(1 - gamma_N)): the mass the squares of N real components can lose
    to underflow in fl(||v||) = sqrt(sum fl(v_i^2)), expressed as a norm (docs/proofs/int05_underflow.md,
    Lemma U3). N = n for a real vector, 2n for a complex one. About 1e-161 sqrt(N): below the resolution of
    every radius above 1e-145, so it changes no bound of a normal-sized quantity. (1 + 2 gamma_{N+2}) >=
    sqrt((1 + gamma_N)/(1 - gamma_N)) for N u <= 1/3; sqrt(0.5 N) 2^-537 = sqrt(N 2^-1075) (2^-537 is a normal
    number; 2^-1075 is not); the two guards cover the five roundings of this line (sqrt, 1 + 2 gamma, two
    products, gamma itself)."""
    N = max(int(N), 1)
    return math.sqrt(0.5 * N) * 2.0 ** -537 * (1.0 + 2.0 * gamma(N + 2)) * _GUARD * _GUARD


def _real_components(v: np.ndarray) -> int:
    return 2 * v.size if np.iscomplexobj(v) else v.size


def residual_entry_count(w: int, complex_arith: bool) -> float:
    """c_w: the number of u_min = 2^-1075 by which one entry of the computed residual R_i = fl(fl(A x)_i -
    fl(mu x_i)) plus the computed entrywise bound E_i can fall short of the exact |A x - mu x|_i under
    fl(z) = z(1 + delta) + eta (docs/proofs/int05_underflow.md, Lemma U4): w + 1 real products in R_i for real
    data (4w + 2 for complex data: four per complex product, two for mu x_i), each eta propagated by at most
    (1 + gamma_{w+1}); in E_i the w products of |A||x| propagated by at most 2 g (1 + 4u), then three operations
    at scale g (1 + 4u)^2 and two at scale (1 + 4u). w = adapter.width."""
    w = max(int(w), 1)
    g = gamma_dot(w + 2, complex_arith)
    products = (4 * w + 2) if complex_arith else (w + 1)
    return (products * (1.0 + gamma(w + 1)) + 2.0 * g * w * (1.0 + gamma(w)) * (1 + 4 * U)
            + 3.0 * g * (1 + 4 * U) ** 2 + 2 * (1 + 4 * U))


def residual_additive_term(n: int, w: int, complex_arith: bool) -> float:
    """A_{n,w} >= sqrt(n) c_w u_min: the 2-norm of the vector of entrywise additive terms, added to the bound on
    ||A x - mu x|| in enclosure.residuals (INT-05). Formed as ldexp(sqrt(n) c_w (1 + 8u)^2, -1075) + ULP_MIN:
    the guards cover the roundings of the normal-range factor, the ldexp rounds to the subnormal grid by at
    most u_min, and ULP_MIN = 2 u_min covers that (the addition is exact in the subnormal range)."""
    n = max(int(n), 1)
    t = math.sqrt(float(n)) * residual_entry_count(w, complex_arith) * _GUARD * _GUARD
    return math.ldexp(t, -1075) + ULP_MIN


def gram_additive_term(g: int, n: int, complex_arith: bool) -> float:
    """Upper bound on the Frobenius norm of the additive (underflow) part of the rounding error of a BLAS
    product X^H Y with X, Y n x g: each of the g^2 entries is a sum of n products (3n real products of
    which at most 2n enter one component for complex data), each contributing one eta <= u_min propagated
    by at most (1 + gamma_n); the matrix of these bounds has Frobenius norm <= g n (1 + gamma_n) u_min times
    3 for complex data (|z| <= |Re z| + |Im z|, 2n + 2n real products) (docs/proofs/int05_underflow.md)."""
    g, n = max(int(g), 1), max(int(n), 1)
    t = g * (3.0 if complex_arith else 1.0) * n * (1.0 + gamma(n)) * _GUARD * _GUARD
    return math.ldexp(t, -1075) + ULP_MIN


def norm2_upper(v: np.ndarray) -> float:
    """Upper bound on the exact 2-norm of a stored float vector (any magnitude: the standard-model factor
    (1 + gamma_{n+2})(1 + 8u) undoes the relative errors of the squares, the sum and the square root; the
    additive norm_underflow_term covers the squares that underflow, INT-05). The guard (1 + gamma_{n+2})(1 + 8u) is
    established for 1 <= n <= 2^26 entries (Lambda4); above that GuardRangeError is raised before anything is
    computed (3.2.2)."""
    n = v.size
    check_guard_range(n=n, where="fp.norm2_upper")
    s = float(np.linalg.norm(v))
    return s * (1.0 + gamma(n + 2)) * _GUARD + norm_underflow_term(_real_components(v))


def norm2_lower(v: np.ndarray) -> float:
    """Lower bound on the exact 2-norm of a stored float vector (0 when the vector is too small to resolve).
    Established for 1 <= n <= 2^26 entries (Lambda4); above that GuardRangeError is raised (3.2.2)."""
    n = v.size
    check_guard_range(n=n, where="fp.norm2_lower")
    s = float(np.linalg.norm(v))
    if not math.isfinite(s):
        # the sum of squares overflowed (||v|| > ~1.3e154): inf is not a lower bound; 0 makes every quotient by
        # it +inf (no enclosure) instead of 0 (independent re-derivation 2026-10-30, docs/proofs/VERDICTS.md,
        # int05_underflow: a radius 2 ULP_MIN was returned for ||x|| ~ 1e160 with a small residual)
        return 0.0
    return max(0.0, s * (1.0 - gamma(n + 2)) / _GUARD - norm_underflow_term(_real_components(v)))


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


def _scan_chunk(arr: np.ndarray, threshold: float):
    """(flat component indices below threshold, their magnitudes) of one in-memory chunk, C order."""
    comp = np.ascontiguousarray(arr).view(arr.real.dtype).ravel() if arr.dtype.kind == "c" else arr.ravel()
    mag = np.abs(comp)
    with np.errstate(invalid="ignore"):
        hit = (mag > 0) & (mag < threshold)
    idx = np.nonzero(hit)[0]
    return idx, mag[idx]


_SCAN_CHUNK_BYTES = 64 * 2 ** 20


def tiny_nonzero(a, threshold: float = UNDERFLOW_THRESHOLD, max_indices: int = 8):
    """Nonzero entries of magnitude below `threshold` in a stored array (the no-underflow hypothesis).

    Complex data are scanned component by component (the real and imaginary parts are the operands of
    the arithmetic). Exact zeros are not reported; non-finite entries are not this function's concern.
    Returns None when nothing is found, else {'count', 'first_indices', 'min_magnitude'} with the flat
    indices (component index for complex data) of the first few offenders. A large 2-D array, a numpy
    memmap or an on-disk column source (enclosure.StreamedColumns, h5py dataset) is scanned in row chunks
    with the same C-order flat indices, so that no whole copy is made.
    """
    if a is None:
        return None
    if sp.issparse(a):
        a = a.data
    shape = tuple(int(d) for d in getattr(a, "shape", ()))
    dtype = np.dtype(getattr(a, "dtype", None) or np.asarray(a).dtype)
    if dtype.kind not in "fc" or int(np.prod(shape, dtype=np.int64)) == 0:
        return None
    in_memory = isinstance(a, np.ndarray) and not isinstance(a, np.memmap)
    if len(shape) < 2 or (in_memory and a.nbytes <= _SCAN_CHUNK_BYTES):
        arr = np.asarray(a)
        idx, mags = _scan_chunk(arr, threshold)
        if idx.size == 0:
            return None
        return {"count": int(idx.size), "first_indices": idx[:max_indices].tolist(),
                "min_magnitude": float(mags.min())}
    src = getattr(a, "source", a)
    row_comp = int(np.prod(shape[1:], dtype=np.int64)) * (2 if dtype.kind == "c" else 1)
    step = max(1, _SCAN_CHUNK_BYTES // max(1, row_comp * dtype.itemsize))
    count, first, mn = 0, [], None
    for r in range(0, shape[0], step):
        idx, mags = _scan_chunk(np.asarray(src[r:r + step]), threshold)
        if idx.size:
            count += int(idx.size)
            if len(first) < max_indices:
                first += (idx[:max_indices - len(first)] + r * row_comp).tolist()
            m = float(mags.min())
            mn = m if mn is None else min(mn, m)
    if count == 0:
        return None
    return {"count": count, "first_indices": first, "min_magnitude": mn}


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


EXACT_SUM_MIN_TERMS = 4096   # below this math.fsum is as fast as the vectorised cascade


def _two_sum_seq(T: np.ndarray):
    """Sequential floating-point sum of T with every rounding error recovered exactly: s_i = fl(s_{i-1} + t_i)
    by cumsum, err_i = (s_{i-1} + t_i) - s_i by Knuth's TwoSum (exact in binary64, no overflow). Returns
    (s_n, nonzero errors); s_n + sum(errors) == sum(T) exactly."""
    s = np.cumsum(T)
    a = np.concatenate(([0.0], s[:-1]))
    bb = s - a
    err = (a - (s - bb)) + (T - bb)
    return float(s[-1]), err[err != 0.0]


def exact_sum(terms, max_rounds: int = 6) -> float:
    """The correctly rounded sum of a float array: the same value math.fsum returns.

    Cascaded error-free summation (Ogita, Rump and Oishi 2005, SumK, with Knuth's TwoSum): after each round
    the exact sum equals the round's float sum plus the exact sum of its recovered errors; after K rounds
    sum(terms) = S_1 + ... + S_K + rho with |rho| <= sum |remaining errors| (bounded with an upward guard).
    The S_k are folded with TwoSum into h + e exactly; when e together with every slack lies strictly
    inside the rounding interval of h, the correctly rounded sum is h. Otherwise (a sum within the slack of
    a rounding boundary, a zero result, non-finite input, or fewer than EXACT_SUM_MIN_TERMS terms) the value
    is math.fsum's, so the result never depends on the path taken.
    """
    t = np.asarray(terms, dtype=float).ravel()
    t = t[t != 0.0]
    if t.size < EXACT_SUM_MIN_TERMS or not np.all(np.isfinite(t)):
        return math.fsum(t.tolist())
    S = []
    T = t
    for _ in range(max_rounds):
        s, T = _two_sum_seq(T)
        S.append(s)
        if T.size == 0:
            break
    B = float(np.sum(np.abs(T))) * (1.0 + U * (T.size + 2)) * _GUARD if T.size else 0.0
    h = S[0]
    e1 = 0.0
    if len(S) > 1:
        hs = h + S[1]
        bb = hs - h
        e1 = (h - (hs - bb)) + (S[1] - bb)
        h = hs
    c, slack = e1, B
    for s_ in S[2:]:
        c = c + s_
        slack += abs(c) * 2 * U
    hs = h + c
    bb = hs - h
    e = (h - (hs - bb)) + (c - bb)
    h = hs
    if h == 0.0 or not math.isfinite(h):
        return math.fsum(t.tolist())
    up_gap = (np.nextafter(h, np.inf) - h) / 2.0
    dn_gap = (h - np.nextafter(h, -np.inf)) / 2.0
    if np.nextafter(e + slack, np.inf) < up_gap and np.nextafter(e - slack, -np.inf) > -dn_gap:
        return h
    return math.fsum(t.tolist())


# Dekker's TwoProduct (Veltkamp split of two normal numbers, then the partial products) is exact when every
# intermediate lies on a grid no finer than 2^-1074: all of them are multiples of ulp(a) ulp(b), which is
# >= 2^-1074 whenever |a b| >= 2^-968 (docs/proofs/int05_underflow.md, Lemma U4). Pairs whose computed product
# is below TINY_PRODUCT = 2^-967 are left out of the error-free sum and their total magnitude, at most
# TINY_BOUND each, is added to the error bound. A pair with a subnormal factor and a product above the
# threshold has its other factor above 2^54; it is rescaled by 2^60 and 2^-60 (exact) before the split.
TINY_PRODUCT = 2.0 ** -967
TINY_BOUND = 2.0 ** -966          # >= 2^-967 + u_min, the largest exact product a left-out pair can have
_MIN_NORMAL = 2.0 ** -1022


def _two_product_split(a: np.ndarray, b: np.ndarray):
    """(p, e, n_tiny): TwoProduct of the pairs whose product is at least TINY_PRODUCT in magnitude (exact),
    and the number of pairs left out (their products are below TINY_BOUND in magnitude)."""
    big = np.abs(a * b) >= TINY_PRODUCT
    n_tiny = int(a.size - np.count_nonzero(big))
    if n_tiny:
        a, b = a[big], b[big]
    sa = np.abs(a) < _MIN_NORMAL          # a subnormal factor (then |b| >= 2^54, and the product is unchanged)
    sb = np.abs(b) < _MIN_NORMAL
    if np.any(sa) or np.any(sb):
        a, b = a.astype(float, copy=True), b.astype(float, copy=True)
        a[sa] *= 2.0 ** 60
        b[sa] *= 2.0 ** -60
        a[sb] *= 2.0 ** -60
        b[sb] *= 2.0 ** 60
    p, e = _two_product(a, b)
    return p, e, n_tiny


def exact_dot(x: np.ndarray, y: np.ndarray):
    """Return (s, err) with |s - x^H y| <= err, where s is the correctly rounded
    value of the exact inner product of the terms whose products do not underflow (computed with
    TwoProduct and a correctly rounded summation: exact_sum, which is math.fsum's value by construction).

    err = 2u|s| per real component (the summation is correctly rounded; the factor 2
    covers the conversion of the complex result) plus the total magnitude the left-out products can
    have, n_tiny * TINY_BOUND per real component (INT-05; zero for every input whose
    products are above 2^-967, i.e. for every pair of vectors with entries above 2^-483 in magnitude).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        xr, xi = np.real(x).astype(float), np.imag(x).astype(float)
        yr, yi = np.real(y).astype(float), np.imag(y).astype(float)
        p1, e1, t1 = _two_product_split(xr, yr)
        p2, e2, t2 = _two_product_split(xi, yi)
        p3, e3, t3 = _two_product_split(xr, yi)
        p4, e4, t4 = _two_product_split(xi, yr)
        re = exact_sum(np.concatenate([p1, e1, p2, e2]))
        im = exact_sum(np.concatenate([p3, e3, -p4, -e4]))
        s = complex(re, im)
        err = 2 * U * (abs(re) + abs(im)) + (t1 + t2 + t3 + t4) * TINY_BOUND
        return s, err
    p, e, t = _two_product_split(x.astype(float), y.astype(float))
    s = exact_sum(np.concatenate([p, e]))
    return s, 2 * U * abs(s) + t * TINY_BOUND


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
