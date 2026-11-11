"""Error-free products by integer splitting (Ozaki scheme): roadmap D-03, prototype, NOT the default path.

UNVERIFIED (D-12): the bounds below are derived in docs/proofs/D-03_ozaki_draft.md (draft; machine-checked by
docs/proofs/verify_D-03.py and tests/test_ozaki.py; independent re-derivation and Sarang's check pending). Nothing
in certify() calls this module; it is reached only by direct calls and by experiments/d03_ozaki_tightness.py.

Idea (Ozaki, Ogita, Oishi and Rump 2012). Split every operand into slices whose entries are integers times a power
of two shared by one row (left factor) or one column (right factor), with at most beta bits each. A product of two
slices is then an integer times 2^(s + s'), and an inner product of m such terms stays an integer of magnitude at
most m 2^(2 beta) <= 2^53 units: every partial sum is a binary64 number, so BLAS (any summation order, with or
without FMA) and SciPy's sparse kernels compute the slice products WITHOUT ROUNDING. The exact product is the sum of
the slice products plus the remainder terms; the slice products are summed with Sum2 (Ogita, Rump and Oishi 2005,
Algorithm 4.4), whose error is u |result| plus a second-order term, and the remainder is bounded entrywise.

What the module returns, with u = 2^-53:

* `split_columns(A, beta, max_slices)`: slices A_1..A_K and the remainder R, A = A_1 + ... + A_K + R exactly,
  (A_p)_ij = m 2^(s_pj) with |m| <= 2^beta integer (Lemmas S1 to S3 of the draft).
* `ozaki_gram(X, Y)`: G and E with |G_ab - (X^H Y)_ab| <= E_ab for every entry (Theorem G).
* `ozaki_residual(H, mu, X)`: R and E with |R_ic - (H X - X diag(mu))_ic| <= E_ic (Theorem R), and the
  Theorem-1 quantities (upper bound on ||H x_c - mu_c x_c||_2, radius) built from them as enclosure.residuals does.

Hypotheses the code checks and refuses otherwise (`NotApplicable`): finite data, |entries| < 2^480 (no overflow
anywhere), at most 2^26 terms per inner product and dimension at most 2^26, every slice unit s >= -1074 (slicing stops
there and leaves the rest to the remainder) and every unit pair s + s' >= -1074 (slice products are exact also in the
subnormal range), a real claimed spectrum mu, and a stored dense or sparse operator (spec and black-box operators are
not supported).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .fp import U, ULP_MIN, _GUARD, gamma, norm2_upper, norm2_lower, up

DEFAULT_MAX_SLICES = 8
MAX_EXPONENT = 480          # |entries| < 2^480: products of two slices and their sums stay far below 2^1024
MIN_UNIT_SUM = -1074        # s + s' >= -1074: an integer of at most 53 bits times 2^(s + s') is a binary64 number


class NotApplicable(ValueError):
    """The data violate a hypothesis of the error-free split (the caller keeps the default path)."""


MAX_TERMS = 2 ** 26          # the range of the floating-point guards (hypothesis (H3) of thm1_fp-accounting.md, L-3)


def split_beta(terms: int) -> int:
    """Largest beta with terms * 2^(2 beta) <= 2^53: slice products with `terms` terms per inner product are exact.
    More than 2^26 terms is refused (the guards of the remainder bounds are established up to 2^26 only)."""
    if int(terms) > MAX_TERMS:
        raise NotApplicable(f"{terms} terms per inner product > 2^26")
    c = max(0, (max(1, int(terms)) - 1).bit_length())       # ceil(log2(terms))
    beta = (53 - c) // 2
    if beta < 1:
        raise NotApplicable(f"inner products of {terms} terms leave no bits for a slice")
    return beta


def _exponent(amax) -> np.ndarray:
    """t with amax < 2^t (frexp: amax = f 2^t, 1/2 <= f < 1); 0 where amax == 0 (an all-zero vector)."""
    amax = np.asarray(amax, dtype=float)
    _, e = np.frexp(amax)
    return np.where(amax > 0, e, 0).astype(np.int64)


def _check_finite(*arrays):
    for a in arrays:
        if a is not None and not np.all(np.isfinite(a)):
            raise NotApplicable("non-finite entries")


@dataclass
class Split:
    """A = sum(slices) + remainder exactly; slices[p] has entries m 2^units[p][j] (j the shared index), |m| <= 2^beta."""
    slices: List[np.ndarray]
    units: List[np.ndarray]          # per slice: the exponent s_pj of every column
    remainder: np.ndarray
    beta: int
    t: np.ndarray                    # exponents with max |A_.j| < 2^t_j
    nonzero: np.ndarray              # mask of the columns that are not identically zero

    @property
    def k(self) -> int:
        return len(self.slices)

    @property
    def exhausted(self) -> bool:
        return not np.any(self.remainder)

    def min_unit(self) -> Optional[int]:
        """Smallest unit exponent among the nonzero columns and the slices used (None: everything is zero)."""
        if not np.any(self.nonzero) or not self.slices:
            return None
        return int(min(int(np.min(u[self.nonzero])) for u in self.units))


def _slice_values(v: np.ndarray, t, beta: int, max_slices: int):
    """Slices of the float array v with exponents t (broadcastable to v): x_p = 2^s_p rint(r_{p-1} 2^-s_p),
    s_p = t - p beta, r_p = r_{p-1} - x_p (every step exact, Lemmas S1 and S2 of the draft)."""
    r = np.array(v, dtype=float, copy=True)
    out, units = [], []
    for p in range(1, max(1, int(max_slices)) + 1):
        s = np.asarray(t - p * beta)
        if s.size and int(np.min(s)) < MIN_UNIT_SUM:
            break                             # Lemma S1 needs s >= -1074: the rest stays in the remainder
        s = s.astype(np.int32)
        m = np.rint(np.ldexp(r, -s))          # |m| <= 2^beta; the scaling by a power of two is exact where m != 0
        xp = np.ldexp(m, s)                   # exact: an integer of at most beta + 1 bits times 2^s, s >= -1074
        r = r - xp                            # exact (Lemma S2)
        out.append(xp)
        units.append(s)
        if not np.any(r):
            break
    return out, units, r


def split_columns(A: np.ndarray, beta: int, max_slices: int = DEFAULT_MAX_SLICES) -> Split:
    """Column-wise split of a real 2-D array (the exponent of column j is shared by its entries)."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    _check_finite(A)
    amax = np.max(np.abs(A), axis=0) if A.shape[0] else np.zeros(A.shape[1])
    t = _exponent(amax)
    if np.any(t > MAX_EXPONENT):
        raise NotApplicable("entries of magnitude >= 2^480")
    sl, un, r = _slice_values(A, t[None, :], beta, max_slices)
    return Split(sl, [np.broadcast_to(u, (1, A.shape[1])).ravel().astype(np.int64) for u in un], r, beta, t, amax > 0)


def split_complex_columns(A: np.ndarray, beta: int, max_slices: int = DEFAULT_MAX_SLICES) -> Tuple[Split, Split]:
    """Real and imaginary parts split with one exponent per column (the larger of the two parts)."""
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
    Ar = np.ascontiguousarray(A.real, dtype=float)
    Ai = np.ascontiguousarray(A.imag, dtype=float) if np.iscomplexobj(A) else np.zeros_like(Ar)
    _check_finite(Ar, Ai)
    if A.shape[0]:
        amax = np.maximum(np.max(np.abs(Ar), axis=0), np.max(np.abs(Ai), axis=0))
    else:
        amax = np.zeros(A.shape[1])
    t = _exponent(amax)
    if np.any(t > MAX_EXPONENT):
        raise NotApplicable("entries of magnitude >= 2^480")
    out = []
    for part in (Ar, Ai):
        sl, un, r = _slice_values(part, t[None, :], beta, max_slices)
        out.append(Split(sl, [np.broadcast_to(u, (1, A.shape[1])).ravel().astype(np.int64) for u in un], r, beta, t,
                         amax > 0))
    return out[0], out[1]


@dataclass
class RowSplit:
    """Row-wise split of a matrix (sparse: CSR on its stored entries; dense: all entries); real and imaginary
    parts share one exponent per row."""
    real: list
    imag: list                       # empty for real data
    units: List[np.ndarray]          # per slice: exponent per row
    rem_real: object
    rem_imag: object                 # None for real data
    beta: int
    nonzero: np.ndarray

    @property
    def k(self) -> int:
        return len(self.real)

    @property
    def exhausted(self) -> bool:
        def nz(M):
            if M is None:
                return False
            return bool(np.any(M.data) if sp.issparse(M) else np.any(M))
        return not (nz(self.rem_real) or nz(self.rem_imag))

    def min_unit(self) -> Optional[int]:
        if not np.any(self.nonzero) or not self.units:
            return None
        return int(min(int(np.min(u[self.nonzero])) for u in self.units))


def _pad(parts, k, like):
    return parts + [np.zeros_like(like) for _ in range(k - len(parts))]


def split_rows(H, beta: int, max_slices: int = DEFAULT_MAX_SLICES) -> RowSplit:
    """Split a stored matrix by rows (sparse: on its stored entries; dense: on all entries)."""
    if sp.issparse(H):
        H = sp.csr_matrix(H)
        H.sort_indices()
        cplx = np.iscomplexobj(H.data)
        dr = np.ascontiguousarray(H.data.real, dtype=float)
        di = np.ascontiguousarray(H.data.imag, dtype=float) if cplx else None
        _check_finite(dr, di)
        n = H.shape[0]
        counts = np.diff(H.indptr)
        absmax = np.abs(dr) if di is None else np.maximum(np.abs(dr), np.abs(di))
        rmax = np.zeros(n)
        if H.nnz:
            rows_nz = counts > 0
            rmax[rows_nz] = np.maximum.reduceat(absmax, H.indptr[:-1][rows_nz])
        t = _exponent(rmax)
        if np.any(t > MAX_EXPONENT):
            raise NotApplicable("operator entries of magnitude >= 2^480")
        t_entry = np.repeat(t, counts)
        pr, _, rr = _slice_values(dr, t_entry, beta, max_slices)
        pi, ri = [], None
        if cplx:
            pi, _, ri = _slice_values(di, t_entry, beta, max_slices)
            kk = max(len(pr), len(pi))
            pr, pi = _pad(pr, kk, dr), _pad(pi, kk, dr)

        def mk(d):
            return sp.csr_matrix((d, H.indices.copy(), H.indptr.copy()), shape=H.shape)

        units = [(t - (p + 1) * beta).astype(np.int64) for p in range(len(pr))]
        return RowSplit([mk(d) for d in pr], [mk(d) for d in pi], units, mk(rr),
                        None if ri is None else mk(ri), beta, rmax > 0)
    A = np.asarray(H)
    cplx = np.iscomplexobj(A)
    Ar = np.ascontiguousarray(A.real, dtype=float)
    Ai = np.ascontiguousarray(A.imag, dtype=float) if cplx else None
    _check_finite(Ar, Ai)
    rmax = np.max(np.abs(Ar), axis=1) if Ai is None else np.maximum(np.max(np.abs(Ar), axis=1), np.max(np.abs(Ai), axis=1))
    t = _exponent(rmax)
    if np.any(t > MAX_EXPONENT):
        raise NotApplicable("operator entries of magnitude >= 2^480")
    pr, _, rr = _slice_values(Ar, t[:, None], beta, max_slices)
    pi, ri = [], None
    if cplx:
        pi, _, ri = _slice_values(Ai, t[:, None], beta, max_slices)
        kk = max(len(pr), len(pi))
        pr, pi = _pad(pr, kk, Ar), _pad(pi, kk, Ar)
    units = [(t - (p + 1) * beta).astype(np.int64) for p in range(len(pr))]
    return RowSplit(pr, pi, units, rr, ri, beta, rmax > 0)


# ---------------------------------------------------------------------------------------------------------------
# accurate summation of exact terms (Sum2, Ogita, Rump and Oishi 2005, Algorithm 4.4, vectorised over entries)
# ---------------------------------------------------------------------------------------------------------------

class _Sum2:
    """Running Sum2 over arrays of one shape: fl(s + c) is the result, A = fl(sum |t|), T the number of terms.

    Bound used (draft, Lemma A; Ogita, Rump and Oishi 2005, Proposition 4.5; additions only, so it holds with
    gradual underflow): |fl(s + c) - sum t| <= u |sum t| + gamma_{T-1}^2 sum |t|."""

    def __init__(self, shape):
        self.s = np.zeros(shape)
        self.c = np.zeros(shape)
        self.A = np.zeros(shape)
        self.T = 0

    def add(self, t):
        t = np.asarray(t, dtype=float)
        if self.T == 0:
            self.s = np.array(np.broadcast_to(t, self.s.shape), dtype=float, copy=True)
        else:
            x = self.s + t                     # TwoSum (Knuth): x + y == s + t exactly
            z = x - self.s
            y = (self.s - (x - z)) + (t - z)
            self.s = x
            self.c += y
        self.A += np.abs(t)
        self.T += 1

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        """(value, error bound) entrywise, error >= |value - exact sum| (draft, Lemma A in floating point:
        u |v| / (1 - u) <= u |v| (1 + 2u); sum |t| <= A / (1 - gamma_{T-1}); _GUARD covers the roundings of this
        line; 2 ULP_MIN the underflow of its two products)."""
        v = self.s + self.c
        coef = sum2_coefficient(self.T)
        err = (np.abs(v) * (U * (1.0 + 2.0 * U)) + self.A * coef) * _GUARD + 2.0 * ULP_MIN
        return v, err


def sum2_coefficient(T: int) -> float:
    """A binary64 number >= gamma_{T-1}^2 / ((1 - gamma_{T-1}) (1 - u)) (Lemma A of the draft; checked exactly on a
    grid of T by docs/proofs/verify_D-03.py)."""
    T = max(int(T), 2)
    g = gamma(T - 1)
    return up(up(g * g) / (1.0 - gamma(T)) / (1.0 - U))


def _abs_upper(re, im):
    """|Re| + |Im| >= |z| (exact moduli; one rounding of the sum, covered by the callers' guards)."""
    return np.abs(re) if im is None else np.abs(re) + np.abs(im)


def _units_ok(left, right):
    ml = [m for m in left if m is not None]
    mr = [m for m in right if m is not None]
    return not (ml and mr and min(ml) + min(mr) < MIN_UNIT_SUM), (min(ml) if ml else None), (min(mr) if mr else None)


# ---------------------------------------------------------------------------------------------------------------
# Gram matrices and column inner products
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class GramResult:
    G: np.ndarray
    E: np.ndarray                    # |G - X^H Y| <= E entrywise
    slices_x: int
    slices_y: int
    beta: int
    exhausted: bool                  # no remainder: the bound is the Sum2 term only
    products: int                    # number of BLAS slice products formed
    seconds: float
    info: dict = field(default_factory=dict)


def _remainder_gram(X, Y, Xr, Xi, Yr, Yi, cplx, reduce):
    """Upper bound on |R_X|^T |Y| + |Xs|^T |R_Y| (Xs = X - R_X, exact) with `reduce` the product (gemm or column sum)."""
    n = X.shape[0]
    RXa = _abs_upper(Xr.remainder, Xi.remainder if cplx else None)
    RYa = _abs_upper(Yr.remainder, Yi.remainder if cplx else None)
    Ya = _abs_upper(np.real(Y).astype(float), np.imag(Y).astype(float) if cplx else None)
    Xsa = _abs_upper(np.real(X).astype(float) - Xr.remainder,
                     (np.imag(X).astype(float) - Xi.remainder) if cplx else None)
    B = reduce(RXa, Ya) + reduce(Xsa, RYa)
    # n nonnegative products per entry and product: gamma_n; the moduli |Re| + |Im| and the final sum: two more
    # roundings (gamma_{n+3} covers them); underflow: at most 2n products per entry, each losing <= 2^-1075
    return B * (1.0 + gamma(n + 3)) * _GUARD + 2.0 * n * ULP_MIN


def ozaki_gram(X: np.ndarray, Y: Optional[np.ndarray] = None, max_slices: int = DEFAULT_MAX_SLICES) -> GramResult:
    """X^H Y (Y = X when omitted) with an entrywise error bound, by error-free slice products on BLAS.

    Theorem G of the draft: with X = sum_p X_p + R_X and Y = sum_q Y_q + R_Y split column-wise (beta for n terms),
    every gemm X_p^T Y_q is exact; X^H Y = sum_{p,q} X_p^H Y_q + R_X^H Y + Xs^H R_Y, and E is the Sum2 bound plus an
    upper bound on |R_X|^T |Y| + |Xs|^T |R_Y|."""
    t0 = time.time()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    same = Y is None
    Y = X if same else np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = int(X.shape[0])
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")
    cplx = bool(np.iscomplexobj(X) or np.iscomplexobj(Y))
    beta = split_beta(n)
    Xr, Xi = split_complex_columns(X, beta, max_slices)
    Yr, Yi = (Xr, Xi) if same else split_complex_columns(Y, beta, max_slices)
    ok, ma, mb = _units_ok((Xr.min_unit(), Xi.min_unit()), (Yr.min_unit(), Yi.min_unit()))
    if not ok:
        raise NotApplicable(f"slice units too small for exact products (2^{ma} x 2^{mb})")
    g, h = X.shape[1], Y.shape[1]
    kx, ky = max(Xr.k, Xi.k), max(Yr.k, Yi.k)
    zx, zy = np.zeros(X.shape), np.zeros(Y.shape)
    acc_re = _Sum2((g, h))
    acc_im = _Sum2((g, h)) if cplx else None
    nprod = 0
    for p in range(kx):
        xr = Xr.slices[p] if p < Xr.k else zx
        xi = (Xi.slices[p] if p < Xi.k else zx) if cplx else None
        for q in range(ky):
            yr = Yr.slices[q] if q < Yr.k else zy
            acc_re.add(xr.T @ yr)                               # exact (Lemma P)
            nprod += 1
            if cplx:
                yi = Yi.slices[q] if q < Yi.k else zy
                acc_re.add(xi.T @ yi)
                acc_im.add(xr.T @ yi)
                acc_im.add(-(xi.T @ yr))
                nprod += 3
    G_re, E_re = acc_re.result()
    if cplx:
        G_im, E_im = acc_im.result()
        G = G_re + 1j * G_im                                    # exact assembly of two floats
        E = (E_re + E_im) * _GUARD                              # |z| <= |Re z| + |Im z|; one addition, guarded
    else:
        G, E = G_re, E_re
    exhausted = Xr.exhausted and Xi.exhausted and Yr.exhausted and Yi.exhausted
    if not exhausted:
        E = (E + _remainder_gram(X, Y, Xr, Xi, Yr, Yi, cplx, lambda A, B: A.T @ B)) * _GUARD
    return GramResult(G, E, kx, ky, beta, bool(exhausted), nprod, time.time() - t0,
                      {"terms_per_entry": acc_re.T, "complex": cplx})


def ozaki_column_dots(X: np.ndarray, Y: np.ndarray, max_slices: int = DEFAULT_MAX_SLICES):
    """(d, e): d_c ~ x_c^H y_c for every column c with |d_c - x_c^H y_c| <= e_c (the diagonal of Theorem G, formed
    by elementwise slice products and column sums: each column sum has n exact terms, the same argument)."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1:
        X, Y = X[:, None], Y[:, None]
    n, k = X.shape
    cplx = bool(np.iscomplexobj(X) or np.iscomplexobj(Y))
    beta = split_beta(n)
    Xr, Xi = split_complex_columns(X, beta, max_slices)
    Yr, Yi = split_complex_columns(Y, beta, max_slices)
    ok, ma, mb = _units_ok((Xr.min_unit(), Xi.min_unit()), (Yr.min_unit(), Yi.min_unit()))
    if not ok:
        raise NotApplicable(f"slice units too small for exact products (2^{ma} x 2^{mb})")
    acc_re = _Sum2((k,))
    acc_im = _Sum2((k,)) if cplx else None
    z = np.zeros(X.shape)
    for p in range(max(Xr.k, Xi.k)):
        xr = Xr.slices[p] if p < Xr.k else z
        xi = Xi.slices[p] if p < Xi.k else z
        for q in range(max(Yr.k, Yi.k)):
            yr = Yr.slices[q] if q < Yr.k else z
            yi = Yi.slices[q] if q < Yi.k else z
            acc_re.add(np.sum(xr * yr, axis=0))
            if cplx:
                acc_re.add(np.sum(xi * yi, axis=0))
                acc_im.add(np.sum(xr * yi, axis=0))
                acc_im.add(-np.sum(xi * yr, axis=0))
    d, e = acc_re.result()
    if cplx:
        di, ei = acc_im.result()
        d = d + 1j * di
        e = (e + ei) * _GUARD
    if not (Xr.exhausted and Xi.exhausted and Yr.exhausted and Yi.exhausted):
        e = (e + _remainder_gram(X, Y, Xr, Xi, Yr, Yi, cplx, lambda A, B: np.sum(A * B, axis=0))) * _GUARD
    return d, e


# ---------------------------------------------------------------------------------------------------------------
# residuals R = H X - X diag(mu)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class OzakiResidual:
    R: np.ndarray                    # residual vectors, entrywise within E of the exact H X - X diag(mu)
    E: np.ndarray
    upper: np.ndarray                # >= ||H x_c - mu_c x_c||_2 (exact arithmetic on the stored data)
    xnorm_lower: np.ndarray
    rel_bound: np.ndarray            # Theorem 1 radius upper / xnorm_lower, rounded upward as in enclosure.residuals
    E_norm: np.ndarray
    computed: np.ndarray             # ||R_c||_2 as computed
    info: dict


def ozaki_residual(H, mu, X, max_slices: int = DEFAULT_MAX_SLICES, block_cols: Optional[int] = None) -> OzakiResidual:
    """Residual vectors of the claimed pairs (mu_c, x_c) of a stored matrix H (dense ndarray or scipy.sparse) with
    an entrywise error bound that has no gamma_w (|H| |x| + |mu| |x|) term (Theorem R of the draft).

    Row i of H X has w_i <= w terms (w = stored entries per row; n for a dense H), so beta is chosen for w terms;
    the diagonal term mu_c x_ic is one exact product per slice pair. The terms of entry (i, c) are summed with
    Sum2; the remainders of H, X and mu are bounded by |H| |R_X| + |R_H| |Xs| + |Xs| |r_mu| + |R_X| |mu|."""
    t0 = time.time()
    mu = np.asarray(mu, dtype=float).ravel()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if mu.size != k:
        raise ValueError("one claimed value per column")
    _check_finite(mu)
    if sp.issparse(H):
        Hc = sp.csr_matrix(H)
        w = int(np.diff(Hc.indptr).max()) if n else 1
    elif isinstance(H, np.ndarray):
        Hc = np.asarray(H)
        w = int(Hc.shape[1])
    else:
        raise NotApplicable("only a stored dense or sparse matrix is supported")
    if Hc.shape != (n, n):
        raise ValueError("H must be n x n")
    cplx_h = bool(np.iscomplexobj(Hc.data if sp.issparse(Hc) else Hc))
    cplx = bool(cplx_h or np.iscomplexobj(X))
    if n > MAX_TERMS:
        raise NotApplicable(f"dimension {n} > 2^26")
    beta = split_beta(max(w, 1))
    HS = split_rows(Hc, beta, max_slices)
    XR, XI = split_complex_columns(X, beta, max_slices)
    MU = split_columns(mu[None, :], beta, max_slices)         # one exponent per claimed value
    ok1, mh, mx = _units_ok((HS.min_unit(),), (XR.min_unit(), XI.min_unit()))
    ok2, mm, _ = _units_ok((MU.min_unit(),), (XR.min_unit(), XI.min_unit()))
    if not (ok1 and ok2):
        raise NotApplicable(f"slice units too small for exact products (H 2^{mh}, mu 2^{mm}, X 2^{mx})")
    kb = k if block_cols is None else max(1, int(block_cols))
    R = np.empty((n, k), dtype=complex if cplx else float)
    E = np.empty((n, k))
    absH = abs(Hc) if sp.issparse(Hc) else np.abs(Hc)
    abs_mu = np.abs(mu)
    rem_any = not (HS.exhausted and XR.exhausted and XI.exhausted and MU.exhausted)
    n_terms = 0
    kx = max(XR.k, XI.k)
    for a in range(0, k, kb):
        b = min(k, a + kb)
        cols = slice(a, b)
        zb = np.zeros((n, b - a))
        acc_re = _Sum2((n, b - a))
        acc_im = _Sum2((n, b - a)) if cplx else None
        for p in range(kx):
            xr = XR.slices[p][:, cols] if p < XR.k else zb
            xi = (XI.slices[p][:, cols] if p < XI.k else zb) if cplx else None
            for q in range(HS.k):
                hr = HS.real[q]
                hi = HS.imag[q] if HS.imag else None
                acc_re.add(np.asarray(hr @ xr))                           # exact (Lemma P, at most w terms)
                if cplx:
                    acc_im.add(np.asarray(hr @ xi))
                    if hi is not None:
                        acc_re.add(-np.asarray(hi @ xi))
                        acc_im.add(np.asarray(hi @ xr))
            for r_ in range(MU.k):
                m_r = MU.slices[r_][:, cols]                              # shape (1, b - a)
                acc_re.add(-(xr * m_r))                                   # one exact product per entry
                if cplx:
                    acc_im.add(-(xi * m_r))
        v_re, e_re = acc_re.result()
        n_terms = max(n_terms, acc_re.T)
        if cplx:
            v_im, e_im = acc_im.result()
            R[:, cols] = v_re + 1j * v_im
            Eb = up(1.0) * (e_re + e_im)
        else:
            R[:, cols] = v_re
            Eb = e_re
        if rem_any:
            Xb = X[:, cols]
            RXa = _abs_upper(XR.remainder[:, cols], XI.remainder[:, cols] if cplx else None)
            Xsa = _abs_upper(np.real(Xb).astype(float) - XR.remainder[:, cols],
                             (np.imag(Xb).astype(float) - XI.remainder[:, cols]) if cplx else None)
            RHa = abs(HS.rem_real) if sp.issparse(HS.rem_real) else np.abs(HS.rem_real)
            if HS.rem_imag is not None:
                RHa = RHa + (abs(HS.rem_imag) if sp.issparse(HS.rem_imag) else np.abs(HS.rem_imag))
            B = (np.asarray(absH @ RXa) + np.asarray(RHa @ Xsa) + Xsa * np.abs(MU.remainder[:, cols])
                 + RXa * abs_mu[cols][None, :])
            # w nonnegative products per entry in each matvec, three additions, the moduli (one rounding each for
            # complex data, hypot): gamma_{w+6} covers them; underflow: at most 2w + 2 products per entry
            B = B * (1.0 + gamma(w + 6)) * _GUARD * _GUARD + (2 * w + 4) * ULP_MIN
            Eb = up(1.0) * (Eb + B)
        E[:, cols] = Eb
    computed = np.linalg.norm(R, axis=0)
    xl = np.array([norm2_lower(X[:, c]) for c in range(k)])
    R_up = np.array([norm2_upper(R[:, c]) for c in range(k)])
    E_up = np.array([norm2_upper(E[:, c]) for c in range(k)])
    upper = R_up + E_up                    # ||r|| <= ||R|| + ||R - r||; the rounding of this sum is absorbed below
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(xl > 0, upper / xl, np.inf) * (1 + 4 * U) + 2 * ULP_MIN
    info = {"beta": beta, "w": w, "slices_H": HS.k, "slices_X": kx, "slices_mu": MU.k,
            "remainder": bool(rem_any), "terms_per_entry": int(n_terms), "complex": cplx,
            "seconds": time.time() - t0}
    return OzakiResidual(R, E, upper, xl, rel, E_up, computed, info)


def residual_report(H, mu, X, max_slices: int = DEFAULT_MAX_SLICES):
    """An enclosure.ResidualReport whose bounds come from ozaki_residual (prototype, UNVERIFIED): pass it to
    `enclosure.enclose(..., residual_report=...)` to obtain the Theorem 1, 2 and 2' radii with the D-03 residuals.
    The stored residual R is the accurately summed one and E_norm its entrywise error bound, so the Theorem 2'
    term ||X^H R|| uses ||Rc - R|| <= ||E|| exactly as with the default report."""
    from .enclosure import OperatorAdapter, ResidualReport
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    mu = np.asarray(mu, dtype=float).ravel()
    r = ozaki_residual(H, mu, X, max_slices)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = r.computed / np.linalg.norm(X, axis=0)
    return ResidualReport(r.computed, r.upper, r.xnorm_lower, r.rel_bound, est, r.R, r.E_norm, True,
                          OperatorAdapter(H), mu, X)


__all__ = ["NotApplicable", "split_beta", "split_columns", "split_complex_columns", "split_rows", "Split",
           "RowSplit", "ozaki_gram", "ozaki_column_dots", "ozaki_residual", "GramResult", "OzakiResidual",
           "DEFAULT_MAX_SLICES", "MIN_UNIT_SUM", "MAX_EXPONENT", "MAX_TERMS", "residual_report", "sum2_coefficient"]
