"""symveig 1.0.0 API (Vehale & Goel, arXiv:2606.16217) on the CERTIFY-ED engine.

    from certify_ed.compat.symveig import enclose_global, enclose_sectors, summary_stats

Same call signatures and result type as symveig 1.0.0, with these differences (see the 2026-10-18
audit, docs/symveig_audit.md):
* `multiplicity` is a certified count only when `counted` is True (cluster theorem with a certified
  orthonormality defect). symveig 1.0.0 reported merged multiplicities without proof; for supplied
  eigenpairs they could be wrong.
* `enclose_sectors` checks exactly that H has no entries between different S values and raises
  ValueError otherwise (symveig 1.0.0 silently enclosed the wrong spectrum).
* H must be exactly Hermitian; pass `symmetrize=True` to enclose the Hermitian part 0.5 (H + H^H)
  instead (symveig 1.0.0 did this silently).
* Residual bounds use |H||x| with row-width gamma factors, so widths no longer grow with the dimension.
* `tier2` (the unverified Rump-Lange refinement) is accepted and ignored.
"""
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from ..spectrum import enclose_spectrum


@dataclass
class Enclosure:
    midpoint: float
    halfwidth: float
    multiplicity: int
    tier: str = "single"
    counted: bool = True
    subspace_basis: Optional[np.ndarray] = None
    subspace_halfwidth: float = 0.0

    @property
    def lo(self) -> float:
        return self.midpoint - self.halfwidth

    @property
    def hi(self) -> float:
        return self.midpoint + self.halfwidth

    def __repr__(self):
        return (f"Enclosure([{self.lo:.6e}, {self.hi:.6e}], mult={self.multiplicity}, tier={self.tier}, "
                f"counted={self.counted})")


def _prepare(H, symmetrize):
    if symmetrize:
        H = 0.5 * (H + H.conj().T)
    return H


def _convert(enc) -> List[Enclosure]:
    out = []
    for iv in enc.intervals:
        # lo/hi are the certified endpoints; midpoint +- halfwidth reproduces them up to rounding, so
        # widen by one ulp-level margin to keep containment exact for users of lo/hi
        mid = iv.midpoint
        half = float(np.nextafter(max(mid - iv.lo, iv.hi - mid), np.inf))
        out.append(Enclosure(mid, half, iv.count, tier=iv.method, counted=bool(iv.counted and enc.complete)))
    return sorted(out, key=lambda e: e.midpoint)


def enclose_global(H, tier2: bool = False, return_subspaces: bool = False, eigh_result: tuple = None,
                   assume_hermitian: bool = False, symmetrize: bool = False) -> List[Enclosure]:
    """Verified enclosures of all eigenvalues of the Hermitian matrix H (dense or sparse)."""
    if tier2:
        warnings.warn("tier2 is not supported (unverified in symveig 1.0.0); ignored", stacklevel=2)
    if return_subspaces:
        warnings.warn("return_subspaces is not supported; use certify_ed.certify for eigenspace statements",
                      stacklevel=2)
    return _convert(enclose_spectrum(_prepare(H, symmetrize), eigh_result=eigh_result))


def enclose_sectors(H, S_diag, tier2: bool = False, atol: float = 1e-12,
                    symmetrize: bool = False) -> List[Enclosure]:
    """Sector-decomposed verified enclosures; S_diag is the diagonal of a conserved quantity."""
    if tier2:
        warnings.warn("tier2 is not supported (unverified in symveig 1.0.0); ignored", stacklevel=2)
    return _convert(enclose_spectrum(_prepare(H, symmetrize), S_diag=S_diag, atol=atol))


def summary_stats(enclosures: List[Enclosure]) -> dict:
    if not enclosures:
        return {"count": 0}
    half = np.array([e.halfwidth for e in enclosures])
    mults = np.array([e.multiplicity for e in enclosures])
    tiers = {}
    for e in enclosures:
        tiers[e.tier] = tiers.get(e.tier, 0) + 1
    return {"count": len(enclosures), "total_eigs": int(mults.sum()), "max_halfwidth": float(half.max()),
            "median_halfwidth": float(np.median(half)), "min_halfwidth": float(half.min()),
            "max_multiplicity": int(mults.max()), "all_counted": all(e.counted for e in enclosures),
            "tier_counts": tiers}


# ------------------------------------------------------------------------ models (symveig.models)
def _dense_with_sz(spec):
    from ..construct import build_bitwise
    H = build_bitwise(spec).toarray()
    N, q = spec.n_sites, spec.q
    s = np.arange(spec.dim)
    sz = np.zeros(spec.dim)
    for site in range(N):
        sz += ((s // q ** (N - 1 - site)) % q) - 0.5
    return H, {"Sz": sz}


def heisenberg_1d(L, periodic=True, J=1.0):
    from ..spec import heisenberg_chain
    return _dense_with_sz(heisenberg_chain(L, J=J, boundary="periodic" if periodic else "open"))


def j1j2_heisenberg_1d(L, periodic=True, J1=1.0, J2=0.5):
    from ..spec import heisenberg_chain
    return _dense_with_sz(heisenberg_chain(L, J=J1, J2=J2, boundary="periodic" if periodic else "open"))


def heisenberg_2d(Lx, Ly, periodic=True, J=1.0):
    from ..spec import heisenberg_square
    return _dense_with_sz(heisenberg_square(Lx, Ly, J=J, boundary="periodic" if periodic else "open"))
