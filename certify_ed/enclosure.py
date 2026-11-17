"""Eigenvalue enclosures for claimed eigenpairs of a Hermitian operator.

Statements produced here concern the operator AS STORED (the floating-point
matrix or the exact operator defined by a spec, see `operator_kind`).

Evaluation (3.2.0.dev0, roadmap C-17 / A-06). The bounds are the ones below; what changed is how they are
evaluated. `residuals` processes the claimed vectors in column blocks: for a sparse or spec operator the
product H X[:, a:b] is column-for-column the same floating-point result as H X (one sparse row sum per
entry, in index order), so the block width does not change a single bit and the vectors may live in a
numpy memmap or an HDF5 dataset that is never loaded whole. For a dense operator BLAS gemm is not
invariant under column blocking (different kernels for different widths), so a dense operator always
takes one block: the vectors are then read into memory, which is no restriction since a dense n x n
operator is at least as large as n x k vectors. The rounding-bound matrix E is reduced to its column
norms at once and not stored; the residual matrix R is stored for in-memory vectors and recomputed for
the few columns of a cluster when the vectors are streamed. The Gram entries of a cluster are cached
within one `enclose` call so that a group that grows by repeated merging does not recompute them.

Theorem 1 (single pair; e.g. Parlett 1998, Thm 4.5.1). For Hermitian A, any
nonzero x and any real mu, some eigenvalue lambda_j satisfies
    |lambda_j - mu| <= ||A x - mu x|| / ||x||.

Theorem 2 (cluster; docs/proofs/thm2_polar-factor.md). Let X (n x g) satisfy
||X^* X - I||_2 <= eta < 1, let Lambda = diag(mu_1..mu_g) be real, and let
R = A X - X Lambda. Put delta = 1 - sqrt(1 - eta) and
    rho = ||R||_2 + (||A||_2 + ||Lambda||_2) delta.
Then there is an injective map i -> j(i) with |lambda_{j(i)} - mu_i| <= rho for
i = 1..g. Proof: the polar factor Qhat = X (X^* X)^(-1/2) is an isometry with
||X - Qhat|| <= delta, so ||A Qhat - Qhat Lambda|| <= rho (polar-factor lemma),
and Kahan's inclusion theorem (any Hermitian g x g block, radius = residual
norm) pairs the eigenvalues of Lambda with distinct eigenvalues of A. Before
track/backlog-proofs the radius was eps = (w + sqrt(w^2 + 4 rho^2))/2 >= rho,
from the Weyl-perturbation proof of the appendix; rho is the Kahan constant.

Theorem 2' (sharper cluster radius; docs/proofs/thm2prime_eps-star.md).
Under the same hypotheses, with spread(Lambda) = max(mu) - min(mu),
    eps* = ||R||_2 / sqrt(1 - eta) + ||X^* R||_2 / (1 - eta) + delta spread(Lambda) / sqrt(1 - eta)
also admits such an injection (Kahan's theorem for Qhat^* A Qhat, then Weyl for
Qhat^* A Qhat - Lambda). The cluster radius used is min(rho, eps*); both are
proven (D-59), so the min is sound.

Each cluster member keeps its Theorem-1 radius as its *existence radius*; the
cluster radius is the one under which the members are matched to distinct
eigenvalues and the one used by every indexed statement downstream.

Disjoint groups therefore certify distinct eigenvalues; overlapping groups are
merged and re-bounded until the enlarged intervals of different groups are
disjoint. When vectors are missing or too far from orthonormal (eta >= 1),
the group only certifies existence (at least one eigenvalue per interval),
not multiplicity; multiplicities then come from inertia counts.

Theorem 3 (Kato 1949; Temple 1928). If the open interval (alpha, beta) contains
exactly one distinct eigenvalue lambda of A, x is a unit vector with Rayleigh
quotient theta in (alpha, beta) and r = ||A x - theta x||, then
    theta - r^2/(beta - theta) <= lambda <= theta + r^2/(theta - alpha).
Used to sharpen isolated enclosures once inertia counts certify isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .fp import (U, ULP_MIN, gamma, gamma_dot, up, norm2_upper, norm2_lower, fro_upper,
                 inf_norm_upper, max_nnz_per_row, exact_dot, exact_gram, residual_additive_term, gram_additive_term,
                 sub_down, add_up, sub_up, check_guard_range)
from .gpu import GPUUnavailable

EXACT_GRAM_BUDGET = 20_000_000   # n*g*(g+1)/2 above which the gamma_n Gram bound is used

# ---- device for the products H X and |H| |X| (roadmap D-05, gpu.py): OperatorAdapter(op, device=...). None or
# 'cpu': scipy on the CPU; 'gpu': the ordered CSR kernels of gpu.py, bit for bit equal to the CPU products, with a
# fallback to the CPU recorded in the adapter's device_record.


def _device_error_text(err, limit: int = 300) -> str:
    """The reason recorded for a device failure: the message of a GPUUnavailable, otherwise type and message, on one
    line and at most `limit` characters (an NVRTC error carries the whole compiler log)."""
    text = str(err) if isinstance(err, GPUUnavailable) else f"{type(err).__name__}: {err}"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."

# ---------------------------------------------------------------------------
# operator adapters
# ---------------------------------------------------------------------------

class OperatorAdapter:
    """Uniform access to H x, |H| |x| bounds and ||H||_2 bounds.

    kind:
      'matrix'   : dense ndarray or scipy.sparse matrix (bounds rigorous)
      'spec'     : certify_ed.construct.SpecOperator (bounds rigorous, via abs_matvec)
      'blackbox' : LinearOperator / callable without |H| access (rounding of
                   the producer's matvec cannot be bounded; statements are
                   marked 'floating-point estimate')
    """

    def __init__(self, op, norm_bound: Optional[float] = None, device: Optional[str] = None):
        self.op = op
        if sp.issparse(op) or isinstance(op, np.ndarray):
            self.kind = "matrix"
            self.A = op.tocsr() if sp.issparse(op) else np.asarray(op)
            self.shape = self.A.shape
            self.complex = bool(np.iscomplexobj(self.A.data if sp.issparse(self.A) else self.A))
            self.width = max_nnz_per_row(self.A)
            # |A| is kept for a sparse matrix (cheap); for a dense one it is formed when needed, so that the
            # adapter does not hold a second n x n array (the product |A| |X| is the same gemm either way)
            self._absA = abs(self.A) if sp.issparse(self.A) else None
            self._norm = inf_norm_upper(self.A)
        elif hasattr(op, "abs_matvec") and hasattr(op, "spec"):
            self.kind = "spec"
            self.shape = op.shape
            self.complex = op.dtype == np.complex128
            spec = op.spec
            max_len = max((len(t.op) for t in spec.terms), default=1)
            # each entry of H x is a sum of at most m = n_terms 2^max_len monomial contributions, each a
            # product of the coupling, <= max_len local amplitudes (one rounding each where irrational) and
            # x_j: at most 2 max_len + 1 roundings per contribution, m - 1 in the sum (width = m + 2 max_len + 2
            # matches certify._builder_shift; docs/proofs/appA_model-shift.md). Before track/backlog-proofs the
            # width was m + max_len + 2, short by max_len - 4 roundings for strings of five or more letters.
            self.width = spec.n_terms * (2 ** max_len) + 2 * max_len + 2
            if hasattr(op, "rounding_width"):
                # D-02 (matrixfree.MatrixFreeOperator): per-row monomial count plus the amplitude roundings,
                # docs/proofs/D-02_matrix-free.md; never larger than the count above
                self.width = min(self.width, int(op.rounding_width))
            ones = np.ones(op.shape[0])
            # the row sums of |H|_abs carry at most width (3.1 count) or width + 1 (D-02: W - 1) roundings
            g_norm = gamma(self.width + 2) if hasattr(op, "rounding_width") else gamma(self.width)
            self._norm = up(float(np.max(op.abs_matvec(ones)))) * (1 + g_norm)
            if hasattr(op, "norm_bound"):
                # D-02 Lemma N: ||H|| <= sum_t |c_t| prod ||o_l||; both bounds are proven, the smaller is kept
                self._norm = min(self._norm, float(op.norm_bound()))
        else:
            self.kind = "blackbox"
            self.shape = op.shape
            self.complex = np.dtype(getattr(op, "dtype", np.complex128)).kind == "c"
            self.width = None
            self._norm = norm_bound
        if norm_bound is not None:
            self._norm = norm_bound if self._norm is None else min(self._norm, norm_bound)
        self._init_device(device)

    def _init_device(self, device):
        """Set up the GPU products when asked for (device='gpu') and possible."""
        self._gpu = None
        self.device_record = None
        if device in (None, "cpu"):
            return
        rec: Dict[str, Any] = {"requested": str(device), "used": False}
        if device != "gpu":
            rec["reason"] = f"unknown device {device!r} (expected 'cpu' or 'gpu')"
        elif not (self.kind == "matrix" and sp.issparse(self.A)):
            rec["reason"] = ("the GPU products are implemented for stored sparse matrices only; this operator is "
                             + ("a dense matrix" if self.kind == "matrix" else f"of kind '{self.kind}'"))
        else:
            from . import gpu
            try:
                self._gpu = gpu.DeviceCSR(self.A, self._absA)
                rec["used"] = True
                rec["gpu"] = self._gpu.record
            except Exception as e:          # GPUUnavailable, or any CuPy/CUDA error: the CPU forms the products
                rec["reason"] = _device_error_text(e)
        self.device_record = rec

    def _gpu_fallback(self, err):
        """Stop using the GPU and record why. If products had been taken from the device before the failure
        (gpu_products_at_fallback > 0), they are not trusted: certify() then computes the certificate again with the
        CPU products."""
        gpu_rec = self.device_record.get("gpu") or {}
        self.device_record["fallback"] = _device_error_text(err)
        self.device_record["gpu_products_at_fallback"] = int(gpu_rec.get("products", 0))
        self.device_record["used"] = self.device_record["gpu_products_at_fallback"] > 0
        try:
            self._gpu.close()
        except Exception:
            pass
        finally:
            self._gpu = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_gpu"] = None       # device arrays stay in this process; a copy computes on the CPU
        return d

    @property
    def rigorous(self) -> bool:
        return self.kind in ("matrix", "spec")

    def matvec(self, X):
        if self.kind == "matrix":
            if getattr(self, "_gpu", None) is not None:
                try:
                    return self._gpu.matmul(X)
                except Exception as e:
                    self._gpu_fallback(e)
            return self.A @ X
        if self.kind == "spec":
            return self.op.matvec(X)
        if hasattr(self.op, "matmat") and X.ndim == 2:
            return self.op.matmat(X)
        if X.ndim == 2:
            return np.column_stack([self.op.matvec(X[:, i]) for i in range(X.shape[1])])
        return self.op.matvec(X)

    def abs_matvec(self, X):
        if self.kind == "matrix":
            absA = self._absA if self._absA is not None else np.abs(self.A)
            aX = np.abs(X)
            if getattr(self, "_gpu", None) is not None:
                try:
                    return self._gpu.matmul(aX, absolute=True)
                except Exception as e:
                    self._gpu_fallback(e)
            return absA @ aX
        if self.kind == "spec":
            return self.op.abs_matvec(np.abs(X))
        raise RuntimeError("no |H| access for black-box operator")

    def both_products(self, X):
        """(H X, |H| |X|), the same arrays as (matvec(X), abs_matvec(X)); with the GPU products of D-05 both are
        formed in one pass over H."""
        if self.kind == "matrix" and getattr(self, "_gpu", None) is not None:
            try:
                return self._gpu.products(X)
            except Exception as e:
                self._gpu_fallback(e)
        return self.matvec(X), self.abs_matvec(X)

    def norm_upper(self) -> Optional[float]:
        return self._norm

    @property
    def dense(self) -> bool:
        """A stored dense matrix: its products go through BLAS gemm, whose result for a column depends on how
        many columns are multiplied at once (verified on OpenBLAS); such an operator is applied to all
        claimed vectors in one call so that the enclosure does not depend on a block size."""
        return self.kind == "matrix" and not sp.issparse(self.A)

    @property
    def column_block_invariant(self) -> bool:
        """Whether H X[:, a:b] equals the columns a:b of H X bit for bit: sparse and spec operators (one
        ordered sum per entry). Dense and black-box operators are applied in one block."""
        return self.kind == "spec" or (self.kind == "matrix" and sp.issparse(self.A))


# ---------------------------------------------------------------------------
# residuals
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_MB = 512.0   # memory for the dense n x b temporaries of one column block (residuals)
_BLOCK_TEMPORARIES = 4     # X block, H X block, |X| block, |H| |X| block (the E block reuses the last)


class StreamedColumns:
    """Column access to an on-disk eigenvector matrix (an h5py dataset, or any object with shape, dtype and
    2-D slicing) with numpy-style indexing X[:, i], X[:, a:b], X[:, [i, j, ...]] (any order). What certify()
    wraps an HDF5 dataset in; a numpy memmap needs no wrapper."""

    def __init__(self, source):
        self.source = source
        self.shape = tuple(int(d) for d in source.shape)
        self.dtype = np.dtype(source.dtype)
        self.ndim = len(self.shape)
        if self.ndim != 2:
            raise ValueError(f"streamed eigenvectors must be 2-D (n x k), got shape {self.shape}")

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None):
            col = key[1]
            if isinstance(col, slice):
                return np.asarray(self.source[:, col])
            if np.isscalar(col) or isinstance(col, (int, np.integer)):
                return np.asarray(self.source[:, int(col)])
            return read_columns(self.source, list(col))
        raise IndexError("StreamedColumns supports X[:, i], X[:, a:b] and X[:, [i, j, ...]] only")

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self.source[...])
        return a if dtype is None else a.astype(dtype)


def is_streamed(X) -> bool:
    """Whether the claimed vectors are a numpy memmap, an HDF5 dataset or a StreamedColumns wrapper (read in
    column blocks, never loaded whole by this module)."""
    if isinstance(X, (np.memmap, StreamedColumns)):
        return True
    mod = type(X).__module__
    return mod.startswith("h5py") and hasattr(X, "shape") and hasattr(X, "dtype")


def column_blocks(k: int, n: int, bytes_per_entry: int, block_mb: Optional[float] = None,
                  temporaries: int = _BLOCK_TEMPORARIES) -> List[tuple]:
    """Column ranges [(a, b), ...] covering 0..k such that `temporaries` dense n x (b - a) arrays fit in
    block_mb. No block has width 1 unless k == 1: a one-column block is reduced by numpy along a
    contiguous axis with a different summation order than a wider block, and the results of this module
    must not depend on the block width."""
    if k <= 0:
        return []
    mb = DEFAULT_BLOCK_MB if block_mb is None else float(block_mb)
    per_col = max(1, int(n)) * int(bytes_per_entry) * max(1, int(temporaries))
    b = int(max(2, min(k, (mb * 2 ** 20) // per_col)))
    out = [(a, min(k, a + b)) for a in range(0, k, b)]
    if len(out) > 1 and out[-1][1] - out[-1][0] == 1:
        out[-2] = (out[-2][0], k)
        out.pop()
    return out


def read_columns(X, idx) -> np.ndarray:
    """X[:, idx] as an in-memory array, for an ndarray, a memmap or an HDF5 dataset (which needs increasing
    indices: they are read sorted and put back in the requested order). A list of indices gives a
    column-major (Fortran-ordered) array, the layout numpy's own X[:, idx] produces: the small products
    X_g^H R_g of a cluster go through BLAS, whose result depends on the operand layout."""
    if isinstance(idx, slice):
        return np.asarray(X[:, idx])
    idx = np.asarray(list(idx), dtype=np.int64)
    if isinstance(X, np.ndarray):
        return np.asfortranarray(X[:, idx])
    if isinstance(X, StreamedColumns):
        X = X.source
    order = np.argsort(idx, kind="stable")
    uniq, inv = np.unique(idx[order], return_inverse=True)
    cols = np.asarray(X[:, uniq.tolist()])          # increasing, no duplicates
    out = np.empty((cols.shape[0], idx.size), dtype=cols.dtype, order="F")
    out[:, order] = cols[:, inv]
    return out


@dataclass
class ResidualReport:
    computed: np.ndarray        # ||fl(Hx - mu x)||_2 as computed
    upper: np.ndarray           # rigorous upper bound on exact ||Hx - mu x||_2 (inf if not rigorous)
    xnorm_lower: np.ndarray
    rel_bound: np.ndarray       # upper / xnorm_lower  (Theorem 1 radius; inf if not rigorous)
    rel_estimate: np.ndarray    # computed / ||x|| (floating-point estimate, always finite)
    R: Optional[np.ndarray]     # computed residual vectors (n x k); None when the vectors are streamed
    E_norm: np.ndarray          # norm2_upper of each column of the componentwise rounding bound E (zeros if
    #                             not rigorous); the quantity every bound uses, so E itself is not stored
    rigorous: bool
    adapter: Optional[OperatorAdapter] = None      # to recompute residual columns when R is not stored
    mu: Optional[np.ndarray] = None
    X: Any = None

    def residual_columns(self, idx, Xg: Optional[np.ndarray] = None) -> np.ndarray:
        """The computed residual vectors R[:, idx]: stored, or recomputed as H X[:, idx] - X[:, idx] diag(mu)
        (bit-identical for a sparse or spec operator, see the module docstring)."""
        idx = list(idx)
        if self.R is not None:
            return np.asfortranarray(self.R[:, idx])
        Xg = read_columns(self.X, idx) if Xg is None else Xg
        HXg = self.adapter.matvec(Xg)
        if HXg.ndim == 1:
            HXg = HXg[:, None]
        return np.asfortranarray(HXg - Xg * self.mu[idx][None, :])     # the layout of a stored R[:, idx]

    @property
    def E(self) -> np.ndarray:
        """The componentwise rounding bound E (n x k), recomputed on request (not stored since 3.2.0.dev0)."""
        n, k = int(self.X.shape[0]), int(self.mu.size)
        if not self.rigorous:
            return np.zeros((n, k))
        if not self.adapter.column_block_invariant:
            return _rounding_bound(self.adapter, self.mu, read_columns(self.X, slice(0, k)))
        return np.column_stack([_rounding_bound(self.adapter, self.mu[a:b], read_columns(self.X, slice(a, b)))
                                for a, b in column_blocks(k, n, 16)])


def _rounding_bound(adapter: OperatorAdapter, mu: np.ndarray, X: np.ndarray, absHX: Optional[np.ndarray] = None):
    """E = gamma_dot(w+2) (|H||X| / (1 - gamma_{w+1}) (1 + 4u) + |X| |mu|) (1 + 4u), componentwise, as in the
    Theorem-1 accounting (docs/theory.md section 1); for complex data gamma_{w+3} replaces gamma_{w+1} (the two extra
    modulus roundings). Computed in place on `absHX` when given.

    The guards Lambda5 and Lambda6 behind these constants are established for row widths w <= 2^26 and the norms
    that follow for n <= 2^26 entries; outside that range fp.GuardRangeError is raised (3.2.2, finding L-3)."""
    check_guard_range(n=int(np.shape(X)[0]), w=adapter.width, where="enclosure._rounding_bound")
    cplx = adapter.complex or np.iscomplexobj(X)
    g = gamma_dot(adapter.width + 2, cplx)
    # real data: |H||x| is formed with w products and w - 1 additions (gamma_{w+1} covers them). Complex data: the
    # moduli |a_ij| and |x_j| are each rounded once more (hypot), so the division needs gamma_{w+3} for
    # absHX_up >= (|H||x|)_i to hold (docs/proofs/VERDICTS.md, thm1 correction; looser than before)
    gw = gamma(adapter.width + (3 if cplx else 1))
    E = adapter.abs_matvec(X) if absHX is None else absHX
    if E.ndim == 1:
        E = E[:, None]
    E /= (1.0 - gw)
    E *= (1 + 4 * U)
    aX = np.abs(X)
    aX *= np.abs(mu)[None, :]
    E += aX
    E *= g
    E *= (1 + 4 * U)
    return E


def residuals(adapter: OperatorAdapter, mu: np.ndarray, X, block_mb: Optional[float] = None,
              keep_R: Optional[bool] = None, products: Optional[tuple] = None) -> ResidualReport:
    """Residual norms and their rigorous upper bounds for the claimed pairs (mu_i, X[:, i]).

    X: ndarray, numpy memmap or HDF5 dataset (n x k; a 1-D ndarray is one column). The columns are processed
    in blocks of `block_mb` megabytes of temporaries (DEFAULT_BLOCK_MB) for a sparse or spec operator, in
    one block for a dense or black-box operator (OperatorAdapter.column_block_invariant). keep_R: store the
    residual matrix (default: yes for in-memory vectors, no for streamed ones). products: (H X, |H| |X|)
    already formed for in-memory X (one block; the |H| |X| array is consumed in place)."""
    mu = np.asarray(mu, dtype=float)
    streamed = is_streamed(X)
    if not streamed:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
    n, k = int(X.shape[0]), int(X.shape[1])
    if keep_R is None:
        # streamed vectors: recompute the few residual columns a cluster needs; a dense operator reads the
        # vectors whole anyway and its gemm is not block-invariant, so its R is kept
        keep_R = (not streamed) or not adapter.column_block_invariant
    cplx_x = np.dtype(X.dtype).kind == "c"
    bytes_per = 16 if (cplx_x or adapter.complex) else 8
    # sparse/spec operators: column blocks cap the temporaries, in memory or streamed; the result does not
    # depend on the block width (tests/test_whole_spectrum.py). Dense/black-box operators: one block.
    blocks = column_blocks(k, n, bytes_per, block_mb) if adapter.column_block_invariant else [(0, k)]
    if products is not None:
        blocks = [(0, k)]
    computed = np.empty(k)
    xl = np.empty(k)
    xn = np.empty(k)
    R_up = np.empty(k)
    E_up = np.zeros(k)
    R = None
    for a, b in blocks:
        Xb = X if (a, b) == (0, k) and not streamed else np.asarray(X[:, a:b])
        if Xb.ndim == 1:
            Xb = Xb[:, None]
        mub = mu[a:b]
        gpu_absHX = None
        if products is not None:
            HX = np.asarray(products[0])
        elif adapter.rigorous and getattr(adapter, "_gpu", None) is not None:
            HX, gpu_absHX = adapter.both_products(Xb)   # D-05: both products in one pass, bit for bit the CPU ones
        else:
            HX = adapter.matvec(Xb)
        if HX.ndim == 1:
            HX = HX[:, None]
        Rb = HX - Xb * mub[None, :]
        del HX
        computed[a:b] = np.linalg.norm(Rb, axis=0)
        xn[a:b] = np.linalg.norm(Xb, axis=0)
        for i in range(b - a):
            xl[a + i] = norm2_lower(Xb[:, i])
            R_up[a + i] = norm2_upper(Rb[:, i])
        if adapter.rigorous:
            if gpu_absHX is not None:
                absHX, gpu_absHX = gpu_absHX, None
            else:
                absHX = None if products is None else np.asarray(products[1], dtype=float)
            Eb = _rounding_bound(adapter, mub, Xb, absHX)
            del absHX
            for i in range(b - a):
                E_up[a + i] = norm2_upper(Eb[:, i])
            del Eb
        if keep_R:
            if R is None:
                R = Rb if (a, b) == (0, k) else np.empty((n, k), dtype=Rb.dtype)
            if R is not Rb:
                R[:, a:b] = Rb
        del Rb
        gpu_absHX = None
    if adapter.rigorous:
        # ||A x - mu x|| <= ||R|| + ||E|| + sqrt(n) c_w u_min: the last term is the INT-05 additive term (the
        # per-entry shortfall of R and E under fl(z) = z(1 + delta) + eta); it is about 1e-160 and changes the
        # stored sum only when the bound itself is below about 1e-145 (docs/proofs/int05_underflow.md)
        upper = (R_up + E_up) + residual_additive_term(n, adapter.width, adapter.complex or cplx_x)
    else:
        upper = np.full(k, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # four roundings (the two sums, the division, the product): (1 + 4u) covers three, the fourth is
        # absorbed by the slack of norm2_upper (docs/proofs/thm1_fp-accounting.md, step 5); + 2 ULP_MIN covers
        # the absolute errors of a division and a product whose results are subnormal
        rel = np.where(xl > 0, upper / xl, np.inf) * (1 + 4 * U) + 2 * ULP_MIN
        est = computed / xn
    return ResidualReport(computed, upper, xl, rel, est, R, E_up, adapter.rigorous, adapter, mu, X)


# ---------------------------------------------------------------------------
# enclosures
# ---------------------------------------------------------------------------

@dataclass
class Group:
    members: List[int]           # indices into the claimed list (sorted by claimed value)
    radius: float                # certified radius for every member
    distinct: bool               # True: members matched to distinct eigenvalues
    method: str                  # 'single' | 'cluster' | 'existence-only' | 'kato-temple'
    eta: Optional[float] = None  # orthonormality defect bound (cluster)
    rho: Optional[float] = None  # ||Rhat||_2 bound (cluster)
    note: str = ""
    bound: Optional[dict] = None # cluster: {'rho' (= 'eps'): Theorem 2, 'eps_star': Theorem 2', 'used': the smaller}

    @property
    def size(self):
        return len(self.members)


@dataclass
class EnclosureSet:
    mu: np.ndarray               # claimed values
    radius: np.ndarray           # per-pair certified radius (cluster radius for cluster members: the
    #                              radius of the distinct matching, used by every indexed statement)
    group_of: np.ndarray         # group index per pair
    groups: List[Group]
    residual: ResidualReport
    rigorous: bool
    all_distinct: bool
    existence_radius: Optional[np.ndarray] = None   # per-pair Theorem-1 radius: an eigenvalue lies within
    #                              it of the claimed value (<= radius); None only for legacy construction

    def __post_init__(self):
        if self.existence_radius is None:
            self.existence_radius = np.array(self.radius, dtype=float, copy=True)

    def interval(self, i):
        """(lower, upper): bounds on the exact endpoints mu_i -/+ r_i (rounded outward)."""
        return (float(sub_down(self.mu[i], self.radius[i])), float(add_up(self.mu[i], self.radius[i])))

    def to_records(self) -> List[dict]:
        out = []
        for i in range(self.mu.size):
            g = self.groups[self.group_of[i]]
            out.append({"index": int(i), "claimed": float(self.mu[i]), "radius": float(self.radius[i]),
                        "lower": float(sub_down(self.mu[i], self.radius[i])),
                        "upper": float(add_up(self.mu[i], self.radius[i])),
                        "existence_radius": float(self.existence_radius[i]),
                        "group": int(self.group_of[i]), "group_size": g.size,
                        "method": g.method, "distinct": bool(g.distinct),
                        "residual_computed": float(self.residual.computed[i]),
                        "residual_bound": float(self.residual.upper[i])})
        return out


def _exact_gram_cached(Xg: np.ndarray, idx: Sequence[int], cache: Optional[Dict[tuple, tuple]]):
    """exact_gram(Xg) with the (i, j) inner products memoised across calls (same columns -> same bits), so
    that a group that grows by repeated merging computes each pair once."""
    if cache is None:
        return exact_gram(Xg)
    g = Xg.shape[1]
    G = np.zeros((g, g), dtype=complex if np.iscomplexobj(Xg) else float)
    E = np.zeros((g, g))
    for a in range(g):
        for b in range(a, g):
            key = (int(idx[a]), int(idx[b]))
            if key not in cache:
                cache[key] = exact_dot(Xg[:, a], Xg[:, b])
            s, e = cache[key]
            G[a, b] = s
            E[a, b] = e
            if a != b:
                G[b, a] = np.conj(s)
                E[b, a] = e
    return G, E


def _cluster_bound(adapter: OperatorAdapter, mu: np.ndarray, X, res: ResidualReport, idx: Sequence[int],
                   cache: Optional[Dict[tuple, tuple]] = None):
    """Cluster radius for the group idx: min of the Theorem 2 radius rho = ||R|| + (||A|| + ||Lambda||) delta
    and the Theorem 2' radius eps* = ||R||/sqrt(1-eta) + ||X^H R||/(1-eta) + delta*spread(mu)/sqrt(1-eta)
    (docs/proofs/thm2_polar-factor.md, docs/proofs/thm2prime_eps-star.md: section 3 of each lists the lines
    below with their rounding counts; `up` = (1 + 8u) covers at most six round-to-nearest operations).
    Returns (radius, eta, rho, {'eps', 'rho', 'eps_star', 'used'}) or None when eta >= 1 or ||A|| is
    unknown ('eps' repeats 'rho': the key the certificate schema has carried since 3.1.0).
    `cache` memoises the exact Gram inner products between calls (enclose passes one per call)."""
    idx = list(idx)
    Xg = read_columns(X, idx)
    n, g = Xg.shape
    cplx = np.iscomplexobj(Xg)
    gd = gamma_dot(n, cplx)
    if n * g * (g + 1) // 2 <= EXACT_GRAM_BUDGET:
        # accurately rounded Gram entries: error <= Eg_ij each (error-free transformations, fp.exact_dot);
        # ||X^H X - I||_2 <= ||fl(G - I)||_F /(1 - u) + ||Eg||_F  (two additions inside up)
        G, Eg = _exact_gram_cached(Xg, idx, cache)
        eta = up(fro_upper(G - np.eye(g)) + fro_upper(Eg) + 4 * U * g)
    else:
        # BLAS Gram: entrywise error <= gamma_dot(n) (|X|^H |X|)_ij, Frobenius norm <= gamma_dot(n) ||X||_F^2
        G = Xg.conj().T @ Xg
        xf2 = fro_upper(Xg) ** 2
        eta = up(fro_upper(G - np.eye(g)) + gd * xf2) + gram_additive_term(g, n, cplx) + 2 * ULP_MIN
    if not eta < 1.0:
        return None
    # delta >= 1 - sqrt(1 - eta): three roundings (1 - eta, sqrt, 1 - .) cost at most 2.6u absolute, up()
    # adds >= 6u relative and the final + 4u is absolute (docs/proofs/thm2_polar-factor.md, section 3)
    delta = 1.0 - math.sqrt(1.0 - eta)
    delta = up(delta) + 4 * U
    # ||R||_2 <= ||R||_F = ||(||r_1||, ..., ||r_g||)||_2 <= norm2_upper of the vector of Theorem-1 bounds
    Rg_fro = norm2_upper(np.asarray(res.upper[idx], dtype=float))
    Hn = adapter.norm_upper()
    if Hn is None or not np.isfinite(Hn):
        return None
    lam_n = float(np.max(np.abs(mu[idx])))              # = ||Lambda||_2 exactly
    rho = up(Rg_fro + (Hn + lam_n) * delta) + ULP_MIN   # three roundings inside up; one product may be subnormal
    # Theorem 2': ||X^H R||_2 <= ||fl(X^H Rc)||_F + gamma_dot(n) ||X||_F ||Rc||_F + ||X||_2 ||Rc - R||_F,
    # with Rc the computed residual columns and ||Rc - R||_F <= norm2_upper of the per-column E bounds
    Rc = res.residual_columns(idx, Xg)
    XR = Xg.conj().T @ Rc
    xnorm = up(math.sqrt(1.0 + eta))                    # >= ||X||_2 = sqrt(lambda_max(X^H X)); two roundings
    Rerr = norm2_upper(np.asarray(res.E_norm[idx], dtype=float))
    XR_bound = (up(fro_upper(XR) + fro_upper(Xg) * fro_upper(Rc) * gd + xnorm * Rerr)   # five roundings
                + gram_additive_term(g, n, cplx) + 3 * ULP_MIN)      # underflow in the BLAS product; 3 products
    inv_sqrt = up(1.0 / math.sqrt(1.0 - eta))          # >= (1 - eta)^(-1/2); three roundings
    inv = up(1.0 / (1.0 - eta))                        # >= (1 - eta)^(-1);   two roundings
    mu_g = mu[list(idx)]
    spread = up(float(sub_up(float(np.max(mu_g)), float(np.min(mu_g)))))
    eps_star = up(Rg_fro * inv_sqrt + XR_bound * inv + delta * spread * inv_sqrt) + 4 * ULP_MIN   # six roundings
    used = "eps_star" if eps_star < rho else "rho"
    # the ingredients are exported for the Davis-Kahan sin(theta) alternative of S4 (E-07, davis_kahan.py): upper
    # bounds on ||R||_F, ||X^H R||_2, delta = 1 - sqrt(1 - eta) and spread(mu) (3.2; additive, used nowhere else here)
    return min(rho, eps_star), eta, rho, {"eps": float(rho), "rho": float(rho), "eps_star": float(eps_star),
                                         "used": used, "R_fro": float(Rg_fro), "XR": float(XR_bound),
                                         "delta": float(delta), "spread": float(spread)}


def nonfinite_columns(X, block_mb: Optional[float] = None) -> List[int]:
    """Indices of columns of X with a non-finite entry, read in column blocks (X may be streamed)."""
    if not is_streamed(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
    n, k = int(X.shape[0]), int(X.shape[1])
    bad = []
    for a, b in column_blocks(k, n, 16 if np.dtype(X.dtype).kind == "c" else 8, block_mb, temporaries=2):
        Xb = np.asarray(X[:, a:b])
        bad += (a + np.nonzero(~np.all(np.isfinite(Xb), axis=0))[0]).tolist()
    return bad


def enclose(adapter: OperatorAdapter, mu: Sequence[float], X, block_mb: Optional[float] = None,
            residual_report: Optional[ResidualReport] = None) -> EnclosureSet:
    """Certified intervals for claimed eigenpairs (mu_i, x_i) of the stored operator.

    X: ndarray (n x k), or a numpy memmap / HDF5 dataset that is read in column blocks (`block_mb`, see
    `residuals`); with a dense operator the vectors are read whole. `residual_report`: a ResidualReport for
    (adapter, mu, X) computed elsewhere (3.3.0.dev0, roadmap D-03: `ozaki.residual_report`, UNVERIFIED and not the
    default); when None (every caller in the package) the report is `residuals(adapter, mu, X, block_mb)`."""
    mu = np.asarray(mu, dtype=float)
    k = mu.size
    if X is None:
        raise ValueError("eigenvectors are required for enclosures (eigenvalue-only claims "
                         "are handled by inertia counts)")
    if not is_streamed(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
    # IMP-21/IMP-22: NaN compares False everywhere, so a non-finite claimed value would pass every
    # overlap and accuracy test; refuse it at the entry
    if not np.all(np.isfinite(mu)):
        raise ValueError("claimed eigenvalues must be finite (found NaN/inf at indices "
                         f"{np.nonzero(~np.isfinite(mu))[0].tolist()[:8]})")
    bad_cols = nonfinite_columns(X, block_mb)
    if bad_cols:
        raise ValueError("claimed eigenvectors must be finite (non-finite entries in columns "
                         f"{bad_cols[:8]})")
    order = np.argsort(mu, kind="stable")
    res = residuals(adapter, mu, X, block_mb=block_mb) if residual_report is None else residual_report
    gram_cache: Dict[tuple, tuple] = {}
    radius = res.rel_bound.copy() if res.rigorous else res.rel_estimate.copy()

    # start: every pair its own group; merge by overlap of current intervals
    groups: List[List[int]] = [[int(i)] for i in order]
    rad_of_group = [float(radius[g[0]]) for g in groups]
    kinds = ["single"] * len(groups)
    meta = [None] * len(groups)

    def overlaps(ga, ra, gb, rb):
        hi_a = add_up(max(mu[i] for i in ga), ra)
        lo_b = sub_down(min(mu[i] for i in gb), rb)
        return hi_a >= lo_b          # outward-rounded ends: touching intervals are merged

    changed = True
    while changed:
        changed = False
        merged, mr, mk, mm = [groups[0]], [rad_of_group[0]], [kinds[0]], [meta[0]]
        for g, r, kd, me in zip(groups[1:], rad_of_group[1:], kinds[1:], meta[1:]):
            if overlaps(merged[-1], mr[-1], g, r):
                newg = merged[-1] + g
                cb = _cluster_bound(adapter, mu, X, res, newg, gram_cache) if res.rigorous else None
                if cb is None:
                    # existence only: each member keeps its own Theorem-1 radius
                    merged[-1] = newg
                    mr[-1] = float(max(radius[i] for i in newg))
                    mk[-1] = "existence-only"
                    mm[-1] = None
                else:
                    eps, eta, rho, info = cb
                    merged[-1] = newg
                    mr[-1] = eps
                    mk[-1] = "cluster"
                    mm[-1] = (eta, rho, info)
                changed = True
            else:
                merged.append(g); mr.append(r); mk.append(kd); mm.append(me)
        groups, rad_of_group, kinds, meta = merged, mr, mk, mm

    group_objs: List[Group] = []
    group_of = np.zeros(k, dtype=int)
    final_radius = np.zeros(k)
    for gi, (g, r, kd, me) in enumerate(zip(groups, rad_of_group, kinds, meta)):
        members = sorted(g, key=lambda i: mu[i])
        if kd == "existence-only":
            rads = [float(radius[i]) for i in members]
        else:
            rads = [float(r)] * len(members)
        # a non-finite radius (zero vector) encloses nothing and never counts as a distinct match
        distinct = kd in ("single", "cluster") and res.rigorous and all(math.isfinite(x) for x in rads)
        for i, ri in zip(members, rads):
            group_of[i] = gi
            final_radius[i] = ri
        group_objs.append(Group(members, float(max(rads)), distinct, kd if res.rigorous else "estimate",
                                eta=None if me is None else float(me[0]),
                                rho=None if me is None else float(me[1]),
                                bound=None if me is None else me[2]))
    # safety net: a group may claim distinct eigenvalues only if its union interval is
    # disjoint from every other group's union interval
    if len(group_objs) > 1:
        L = np.array([min(sub_down(mu[i], final_radius[i]) for i in g.members) for g in group_objs])
        Uu = np.array([max(add_up(mu[i], final_radius[i]) for i in g.members) for g in group_objs])
        order_g = np.argsort(L, kind="stable")
        Ls, Us = L[order_g], Uu[order_g]
        prefmax = np.maximum.accumulate(np.r_[-np.inf, Us[:-1]])
        nextL = np.r_[Ls[1:], np.inf]
        clash = (Ls <= prefmax) | (Us >= nextL)
        for pos, gi in enumerate(order_g):
            if clash[pos] and group_objs[gi].distinct:
                group_objs[gi].distinct = False
                group_objs[gi].note = "union interval overlaps another group"
    all_distinct = all(g.distinct for g in group_objs)
    # existence radius: Theorem 1 per member (never larger than the cluster radius; min() guards rounding)
    existence = np.minimum(np.asarray(radius, dtype=float), final_radius)
    return EnclosureSet(mu, final_radius, group_of, group_objs, res, res.rigorous, all_distinct, existence)


def cluster_groups(enc: EnclosureSet, cluster_tol: float) -> List[List[int]]:
    """Partition claimed pairs into clusters: members of the same enclosure group,
    or consecutive claimed values closer than cluster_tol."""
    order = np.argsort(enc.mu, kind="stable")
    parent = list(range(enc.mu.size))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for g in enc.groups:
        for m in g.members[1:]:
            parent[find(m)] = find(g.members[0])
    for a, b in zip(order[:-1], order[1:]):
        if enc.mu[b] - enc.mu[a] <= cluster_tol:
            parent[find(b)] = find(a)
    out = {}
    for i in order:
        out.setdefault(find(i), []).append(int(i))
    return list(out.values())


# ---------------------------------------------------------------------------
# Kato-Temple sharpening
# ---------------------------------------------------------------------------

def rayleigh_interval(adapter: OperatorAdapter, x: np.ndarray, Ax: Optional[np.ndarray] = None,
                      absAx: Optional[np.ndarray] = None):
    """Rigorous enclosure [lo, hi] of theta = x^* A x / x^* x and the Theorem-1 radius at theta_hat.

    A x is computed in floating point (componentwise error <= gamma_w |A||x|); the two inner
    products are evaluated with error-free transformations (see fp.exact_dot). `Ax` and `absAx` may be
    supplied when fl(A x) and |A| |x| were already formed (certify.py forms them for all isolated pairs in
    one product each); the bound is the same function of them.
    """
    x = np.asarray(x).ravel()
    # the |A||x| guard below is the one of Theorem 1 (Lambda5): established for w <= 2^26 and n <= 2^26 (3.2.2)
    check_guard_range(n=x.size, w=adapter.width, where="enclosure.rayleigh_interval")
    cplx = adapter.complex or np.iscomplexobj(x)
    Ax = adapter.matvec(x) if Ax is None else np.asarray(Ax).ravel()
    num_c, num_e = exact_dot(x, Ax)
    num = float(np.real(num_c))
    den_c, den_e = exact_dot(x, x)
    den = float(np.real(den_c))
    gdw = gamma_dot(adapter.width + 2, cplx)
    gw = gamma(adapter.width + 1)
    absAx_raw = adapter.abs_matvec(x) if absAx is None else np.asarray(absAx).ravel()
    absAx = absAx_raw / (1 - gw)
    xa = np.abs(x)
    # |x^*(Ax - fl(Ax))| <= |x|^T gdw |A||x|, bounded with a guarded float dot of non-negatives
    num_err = up(num_e + gdw * float(xa @ absAx) * (1 + gamma(x.size + 2)))
    den_err = up(den_e)
    den_lo = den - den_err
    if den_lo <= 0:
        return None
    cands = [(num - num_err) / (den + den_err), (num - num_err) / den_lo,
             (num + num_err) / (den + den_err), (num + num_err) / den_lo]
    lo, hi = min(cands), max(cands)
    lo -= abs(lo) * 4 * U
    hi += abs(hi) * 4 * U
    theta_hat = num / den
    res = residuals(adapter, np.array([theta_hat]), x[:, None],
                    products=(Ax[:, None], absAx_raw[:, None].copy()))
    return lo, hi, theta_hat, float(res.rel_bound[0])


def kato_temple(adapter: OperatorAdapter, x: np.ndarray, alpha: float, beta: float,
                Ax: Optional[np.ndarray] = None, absAx: Optional[np.ndarray] = None):
    """Certified [lo, hi] for the unique eigenvalue in (alpha, beta), or None if
    the hypotheses cannot be verified (theta not safely inside, or margins
    smaller than the residual). `Ax`, `absAx`: see rayleigh_interval."""
    ri = rayleigh_interval(adapter, x, Ax, absAx)
    if ri is None:
        return None
    th_lo, th_hi, th, r_hat = ri
    # r = ||Ax - theta x|| <= ||Ax - th x|| + |theta - th| (unit-norm normalisation in rel_bound)
    r = up(r_hat + max(th_hi - th, th - th_lo))
    if not (alpha < sub_down(th_lo, r) and add_up(th_hi, r) < beta):
        return None
    r2 = np.nextafter(r * r, np.inf) + ULP_MIN       # + ULP_MIN: the product's absolute error when r^2 is subnormal
    lower = float(sub_down(th_lo, np.nextafter(r2 / sub_down(beta, th_lo), np.inf)))
    upper = float(add_up(th_hi, np.nextafter(r2 / sub_down(th_hi, alpha), np.inf)))
    lower -= abs(lower) * 8 * U
    upper += abs(upper) * 8 * U
    return float(lower), float(upper), float(th), float(r)
