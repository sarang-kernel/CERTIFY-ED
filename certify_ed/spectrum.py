"""Verified enclosure of the whole spectrum of a Hermitian matrix, optionally sector by sector.

This is the symveig functionality (Vehale & Goel, arXiv:2606.16217) rebuilt on the CERTIFY-ED engine,
with three corrections:

1. Counts. A per-vector residual interval contains *at least one* eigenvalue, not a distinct one. A
   merged chain of k overlapping intervals therefore holds k eigenvalues only if the k vectors are
   shown to span a k-dimensional near-invariant subspace. Here that is the cluster theorem of
   certify_ed.enclosure (certified orthonormality defect eta < 1). Groups that do not pass are
   reported as 'existence-only': at least one eigenvalue, count not certified. The spectrum is
   declared *complete* (union of intervals contains every eigenvalue, with the stated counts) only
   when every group is counted and the counts add up to the dimension.
2. Sectors. A sector decomposition is valid only if the matrix has no entries between different
   values of the conserved quantity. This is checked exactly on the stored entries before any sector
   is used; a violation raises ValueError (symveig 1.0.0 silently enclosed the wrong spectrum).
3. Rounding. Residual bounds use |A||x| with row-width gamma factors instead of gamma_{4n} ||A||_F ||x||,
   so single-eigenvalue intervals do not grow with the dimension (E8: median half-widths several times
   smaller at L = 12). Cluster intervals carry the cost of the count guarantee (Davis-Kahan-type radius)
   and can be wider than symveig's merged intervals, which carry no count guarantee.

The floating-point environment assumptions are those of certify_ed (checked at run time by
certify_ed.fp.probe_float_environment when used through certify()).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .enclosure import OperatorAdapter, enclose
from .fp import add_up, sub_down
from .precision import binary64, summed


@dataclass
class SpectrumInterval:
    lo: float
    hi: float
    count: int                   # number of claimed eigenpairs merged into this interval
    counted: bool                # True: the interval contains at least `count` distinct eigenvalues
    #                              (exactly `count` when the enclosure is complete)
    method: str                  # 'single' | 'cluster' | 'existence-only' | 'merged-across-sectors'
    sectors: List[float] = field(default_factory=list)

    @property
    def midpoint(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def halfwidth(self):
        return 0.5 * (self.hi - self.lo)


@dataclass
class SpectrumEnclosure:
    intervals: List[SpectrumInterval]
    dimension: int
    complete: bool               # every interval counted and counts sum to the dimension
    max_halfwidth: float
    sector_dims: dict

    def contains(self, value: float) -> bool:
        return any(iv.lo <= value <= iv.hi for iv in self.intervals)


def _sector_blocks(H, S_diag, atol):
    """Exact check that H has no entries between different S values; return {s: index array}."""
    S = np.asarray(S_diag, dtype=float)
    keys = np.round(S / atol).astype(np.int64) if atol > 0 else S
    Hc = sp.coo_matrix(H)
    nz = Hc.data != 0
    bad = keys[Hc.row[nz]] != keys[Hc.col[nz]]
    if np.any(bad):
        i = int(np.nonzero(bad)[0][0])
        raise ValueError(f"matrix couples different sectors (entry ({Hc.row[nz][i]}, {Hc.col[nz][i]}) "
                         f"between S = {S[Hc.row[nz][i]]} and {S[Hc.col[nz][i]]}): the quantity is not conserved")
    out = {}
    for k in np.unique(keys):
        idx = np.nonzero(keys == k)[0]
        out[float(S[idx[0]])] = idx
    return out


def intervals_from_enclosure(enc, shift: float = 0.0) -> List[SpectrumInterval]:
    """One SpectrumInterval per enclosure group (outward-rounded ends; `shift` widens every radius, e.g. by
    the S1 model shift so that the intervals refer to the declared model)."""
    out = []
    for g in enc.groups:
        lo = min(sub_down(enc.mu[i], add_up(enc.radius[i], shift)) for i in g.members)
        hi = max(add_up(enc.mu[i], add_up(enc.radius[i], shift)) for i in g.members)
        out.append(SpectrumInterval(float(lo), float(hi), g.size, bool(g.distinct), g.method))
    return out


def merge_intervals(ivs: List[SpectrumInterval], n: int) -> SpectrumEnclosure:
    """Merge overlapping intervals (counts add when both parts are counted) and decide completeness:
    every merged interval counted and the counts summing to n (Proposition 9)."""
    ivs = sorted(ivs, key=lambda v: v.lo)
    merged: List[SpectrumInterval] = []
    for iv in ivs:
        if merged and iv.lo <= merged[-1].hi:
            m = merged[-1]
            merged[-1] = SpectrumInterval(m.lo, max(m.hi, iv.hi), m.count + iv.count, m.counted and iv.counted,
                                          "merged-across-sectors",
                                          m.sectors + iv.sectors)
        else:
            merged.append(iv)
    complete = all(v.counted for v in merged) and sum(v.count for v in merged) == n
    return SpectrumEnclosure(merged, n, bool(complete), max(v.halfwidth for v in merged) if merged else 0.0, {})


def _enclose_block(A, eig=None):
    dense = A.toarray() if sp.issparse(A) else np.asarray(A)
    if eig is None:
        if dense.shape == (1, 1):            # scipy 1.8 cannot size the LAPACK workspace for n = 1
            mu, X = np.real(dense).reshape(1).astype(float), np.ones((1, 1), dtype=dense.dtype)
        else:
            mu, X = la.eigh(dense)
    else:
        mu, X = eig
    enc = enclose(OperatorAdapter(A if sp.issparse(A) else dense), mu, X)
    return intervals_from_enclosure(enc), enc


def enclose_spectrum(H, S_diag=None, eigh_result=None, atol: float = 1e-12) -> SpectrumEnclosure:
    """Verified enclosure of all eigenvalues of the Hermitian matrix H (dense or sparse).

    S_diag: optional diagonal of a conserved quantity in the working basis; each sector is then
    diagonalised and enclosed separately (the full matrix is never diagonalised).
    eigh_result: optional (mu, X) for the whole matrix (no sectors), e.g. a producer's result; it is
    verified, not trusted. A matrix in a narrower format than binary64 is converted exactly first (precision.py), so
    the enclosure refers to the matrix as stored."""
    H = summed(binary64(H if sp.issparse(H) else np.asarray(H), "H"), "H")
    n = H.shape[0]
    if not np.all(np.isfinite(H.data if sp.issparse(H) else H)):
        raise ValueError("H must have finite entries")
    if (sp.issparse(H) and abs(H - H.conj().T).max() != 0) or (not sp.issparse(H) and not np.array_equal(H, np.conj(H).T)):
        raise ValueError("H must be exactly Hermitian (no symmetrisation is applied: the enclosure refers "
                         "to the matrix as given)")
    if eigh_result is not None:
        # IMP-21: a NaN claimed value gave a [nan, nan] interval that counted towards a 'complete' enclosure
        mu_in, X_in = eigh_result
        if not (np.all(np.isfinite(mu_in)) and np.all(np.isfinite(X_in))):
            raise ValueError("eigh_result must be finite (NaN/inf claimed eigenvalues or vectors)")
    if S_diag is None:
        ivs, _ = _enclose_block(H, eigh_result)
        dims = {None: n}
    else:
        if eigh_result is not None:
            raise ValueError("eigh_result is for the unsectored path")
        blocks = _sector_blocks(H, S_diag, atol)
        Hc = sp.csr_matrix(H) if sp.issparse(H) else None
        ivs, dims = [], {}
        for s, idx in blocks.items():
            A = Hc[idx][:, idx] if Hc is not None else np.asarray(H)[np.ix_(idx, idx)]
            part, _ = _enclose_block(A)
            for iv in part:
                iv.sectors = [s]
            ivs += part
            dims[s] = int(idx.size)
    # merge overlaps across sectors: eigenvalues of different sectors are distinct eigenvalues of H,
    # so counts add when both parts are counted
    res = merge_intervals(ivs, n)
    res.sector_dims = dims
    return res
