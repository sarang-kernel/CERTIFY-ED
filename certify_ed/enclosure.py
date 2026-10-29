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

Theorem 2 (cluster; proved in the paper's appendix). Let X (n x g) satisfy
||X^* X - I||_2 <= eta < 1, let Lambda = diag(mu_1..mu_g) be real, and let
R = A X - X Lambda. Put delta = 1 - sqrt(1 - eta),
    rho = ||R||_2 + (||A||_2 + ||Lambda||_2) delta,
    w   = any upper bound on ||Qhat^* Rhat||_2 (w <= rho),
    eps = (w + sqrt(w^2 + 4 rho^2)) / 2,
where Qhat is the orthonormal polar factor of X. Then there is an injective
map i -> j(i) with |lambda_{j(i)} - mu_i| <= eps for i = 1..g.

Theorem 2' (sharper cluster radius; docs/theory.md section 2, UNVERIFIED, D-12).
Under the same hypotheses, with spread(Lambda) = max(mu) - min(mu),
    eps* = ||R||_2 / sqrt(1 - eta) + ||X^* R||_2 / (1 - eta) + delta spread(Lambda) / sqrt(1 - eta)
also admits such an injection (Kahan's theorem for Qhat^* A Qhat, then Weyl for
Qhat^* A Qhat - Lambda). The cluster radius used is min(eps, eps*).

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

from .fp import (U, gamma, gamma_dot, up, norm2_upper, norm2_lower, fro_upper,
                 inf_norm_upper, max_nnz_per_row, safe_sqrt_upper, exact_dot, exact_gram,
                 sub_down, add_up, sub_up)

EXACT_GRAM_BUDGET = 20_000_000   # n*g*(g+1)/2 above which the gamma_n Gram bound is used


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

    def __init__(self, op, norm_bound: Optional[float] = None):
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
            # each matrix element is a sum over couplings of products of <= max_len+1 factors
            self.width = spec.n_terms * (2 ** max_len) + max_len + 2
            ones = np.ones(op.shape[0])
            self._norm = up(float(np.max(op.abs_matvec(ones)))) * (1 + gamma(self.width))
        else:
            self.kind = "blackbox"
            self.shape = op.shape
            self.complex = np.dtype(getattr(op, "dtype", np.complex128)).kind == "c"
            self.width = None
            self._norm = norm_bound
        if norm_bound is not None:
            self._norm = norm_bound if self._norm is None else min(self._norm, norm_bound)

    @property
    def rigorous(self) -> bool:
        return self.kind in ("matrix", "spec")

    def matvec(self, X):
        if self.kind == "matrix":
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
            return absA @ np.abs(X)
        if self.kind == "spec":
            return self.op.abs_matvec(np.abs(X))
        raise RuntimeError("no |H| access for black-box operator")

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
    Theorem-1 accounting (docs/theory.md section 1). Computed in place on `absHX` when given."""
    cplx = adapter.complex or np.iscomplexobj(X)
    g = gamma_dot(adapter.width + 2, cplx)
    gw = gamma(adapter.width + 1)
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
        HX = adapter.matvec(Xb) if products is None else np.asarray(products[0])
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
            Eb = _rounding_bound(adapter, mub, Xb, None if products is None else np.asarray(products[1], dtype=float))
            for i in range(b - a):
                E_up[a + i] = norm2_upper(Eb[:, i])
            del Eb
        if keep_R:
            if R is None:
                R = Rb if (a, b) == (0, k) else np.empty((n, k), dtype=Rb.dtype)
            if R is not Rb:
                R[:, a:b] = Rb
        del Rb
    if adapter.rigorous:
        upper = R_up + E_up
    else:
        upper = np.full(k, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(xl > 0, upper / xl, np.inf) * (1 + 4 * U)
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
    bound: Optional[dict] = None # cluster: {'eps': Theorem 2, 'eps_star': Theorem 2', 'used': which is smaller}

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
    """Cluster radius for the group idx: min of the Theorem 2 radius eps and the Theorem 2' radius
    eps* = ||R||/sqrt(1-eta) + ||X^H R||/(1-eta) + delta*spread(mu)/sqrt(1-eta) (docs/theory.md).
    Returns (radius, eta, rho, {'eps', 'eps_star', 'used'}) or None when eta >= 1 or ||A|| is unknown.
    `cache` memoises the exact Gram inner products between calls (enclose passes one per call)."""
    idx = list(idx)
    Xg = read_columns(X, idx)
    n, g = Xg.shape
    cplx = np.iscomplexobj(Xg)
    gd = gamma_dot(n, cplx)
    if n * g * (g + 1) // 2 <= EXACT_GRAM_BUDGET:
        # accurately rounded Gram entries: error <= 2u|G_ij| each (error-free transformations)
        G, Eg = _exact_gram_cached(Xg, idx, cache)
        eta = up(fro_upper(G - np.eye(g)) + fro_upper(Eg) + 4 * U * g)
    else:
        G = Xg.conj().T @ Xg
        xf2 = fro_upper(Xg) ** 2
        eta = up(fro_upper(G - np.eye(g)) + gd * xf2)
    if not eta < 1.0:
        return None
    delta = 1.0 - math.sqrt(1.0 - eta)
    delta = up(delta) + 4 * U
    Rg_fro = safe_sqrt_upper(float(np.sum(res.upper[idx] ** 2)))
    Hn = adapter.norm_upper()
    if Hn is None or not np.isfinite(Hn):
        return None
    lam_n = float(np.max(np.abs(mu[idx])))
    rho = up(Rg_fro + (Hn + lam_n) * delta)
    # w bound: ||Qhat^* Rhat|| <= ||X^* R_exact|| + ||X|| ||Rhat - R|| + delta ||Rhat||
    Rc = res.residual_columns(idx, Xg)
    XR = Xg.conj().T @ Rc
    xnorm = math.sqrt(1.0 + eta)
    Rerr = safe_sqrt_upper(float(sum(res.E_norm[j] ** 2 for j in idx)))
    # ||X^* R_exact||_2 <= ||fl(X^* Rc)||_F + rounding of that product + ||X|| ||R_exact - Rc||
    XR_bound = up(fro_upper(XR) + gd * fro_upper(Xg) * fro_upper(Rc) + xnorm * Rerr)
    w = XR_bound + xnorm * (Hn + lam_n) * delta + delta * rho
    w = min(up(w), rho)
    eps = up(0.5 * (w + math.sqrt(w * w + 4.0 * rho * rho)))
    # Theorem 2' (IMP-13): every factor an upper bound; up() covers the few roundings of each product
    inv_sqrt = up(1.0 / math.sqrt(1.0 - eta))
    inv = up(1.0 / (1.0 - eta))
    mu_g = mu[list(idx)]
    spread = up(float(sub_up(float(np.max(mu_g)), float(np.min(mu_g)))))
    eps_star = up(Rg_fro * inv_sqrt + XR_bound * inv + delta * spread * inv_sqrt)
    used = "eps_star" if eps_star < eps else "eps"
    return min(eps, eps_star), eta, rho, {"eps": float(eps), "eps_star": float(eps_star), "used": used}


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


def enclose(adapter: OperatorAdapter, mu: Sequence[float], X, block_mb: Optional[float] = None) -> EnclosureSet:
    """Certified intervals for claimed eigenpairs (mu_i, x_i) of the stored operator.

    X: ndarray (n x k), or a numpy memmap / HDF5 dataset that is read in column blocks (`block_mb`, see
    `residuals`); with a dense operator the vectors are read whole."""
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
    res = residuals(adapter, mu, X, block_mb=block_mb)
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
    r2 = np.nextafter(r * r, np.inf)
    lower = float(sub_down(th_lo, np.nextafter(r2 / sub_down(beta, th_lo), np.inf)))
    upper = float(add_up(th_hi, np.nextafter(r2 / sub_down(th_hi, alpha), np.inf)))
    lower -= abs(lower) * 8 * U
    upper += abs(upper) * 8 * U
    return float(lower), float(upper), float(th), float(r)
