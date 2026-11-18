"""Gap records (roadmap C-01): certified intervals for differences of consecutive eigenvalues.

Nothing here is a new statement. The record reads what S2 and S3 already certify and writes it as
intervals for the gaps:

* S2 (Theorems 1-2) gives one interval per claimed pair, each containing a distinct eigenvalue; the
  radii carry the model shift so that the intervals refer to the operator the statements refer to.
* S3 CERTIFIED (Corollary 5 for 'lowest', the two inertia counts of the window path for 'window', the
  count = dimension argument for 'all') says that the claimed enclosures contain every eigenvalue in the
  certified window and nothing else is there. The pigeonhole step of docs/theory.md Section 6 (used by
  Proposition 9 for whole spectra) then gives: a maximal chain of overlapping enclosures ("level")
  containing c claimed pairs contains exactly c eigenvalues, and the levels are ordered. For a 'lowest'
  claim the levels therefore carry eigenvalue ranks (1-based: level j covers ranks R_j .. R_j + c_j - 1).
* For two consecutive levels L = [lo_L, hi_L] (ranks up to r) and U = [lo_U, hi_U] (ranks from r + 1),
  lambda_{r+1} - lambda_r lies in [lo_U - hi_L, hi_U - lo_L], rounded outward (fp.sub_down / fp.sub_up).
* For a 'lowest' claim the gap above the highest claimed level has the lower bound S3 already reports
  (`gap_lower_bound`, Lemma 4 / Corollary 5) and no upper bound: the next eigenvalue is not claimed.
  When a producer wants that gap bounded on both sides, it claims k + 1 pairs; the gap between the
  k-th and (k+1)-th level then has both ends from S2 (upper end mu_{k+1} + r_{k+1} - lo_k, Theorem 1).

Gaps inside a level (between the c eigenvalues of a cluster) are reported as [0, hi - lo]: the
enclosures do not separate them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .fp import add_up, sub_down, sub_up

__all__ = ["merge_enclosures", "gap_records", "rank_interval"]


def merge_enclosures(mu: Sequence[float], radius: Sequence[float]) -> List[dict]:
    """Maximal chains of overlapping enclosures [mu_i - r_i, mu_i + r_i] (ends rounded outward), sorted.

    Returns records {"lower", "upper", "members" (indices into mu), "count"}; "ranks" ([first, last],
    1-based) is filled in by the caller when the ordering is certified. The merge is the one of
    spectrum.merge_intervals, applied pair by pair."""
    mu = np.asarray(mu, dtype=float)
    rad = np.asarray(radius, dtype=float)
    lo = np.asarray(sub_down(mu, rad), dtype=float)
    hi = np.asarray(add_up(mu, rad), dtype=float)
    order = np.argsort(lo, kind="stable")
    out: List[dict] = []
    for i in order:
        i = int(i)
        if out and lo[i] <= out[-1]["upper"]:
            m = out[-1]
            m["upper"] = float(max(m["upper"], hi[i]))
            m["members"].append(i)
            m["count"] += 1
        else:
            out.append({"lower": float(lo[i]), "upper": float(hi[i]), "members": [i], "count": 1})
    return out


def _assign_ranks(levels: List[dict], first_rank: int = 1) -> None:
    r = first_rank
    for lv in levels:
        lv["ranks"] = [r, r + lv["count"] - 1]
        r += lv["count"]


def rank_interval(levels: List[dict], rank: int) -> Optional[dict]:
    """The level record whose certified ranks contain `rank` (1-based), or None."""
    for lv in levels:
        rk = lv.get("ranks")
        if rk is not None and rk[0] <= rank <= rk[1]:
            return lv
    return None


def gap_records(mu: Sequence[float], radius: Sequence[float], kind: str, s3_evidence: dict,
                refers_to: str) -> Dict[str, object]:
    """The `gaps` record of a certificate body.

    mu, radius: claimed values and the S2 radii *including* the model shift (as the S3 statement uses
    them). kind: the claim kind ('lowest', 'window' or 'all'); the caller guarantees that S2 and S3 are
    CERTIFIED. s3_evidence: the S3 evidence (gap_lower_bound for 'lowest'). Every interval end is rounded
    outward; an upper end of inf means "no upper bound"."""
    levels = merge_enclosures(mu, radius)
    ranked = kind in ("lowest", "all")
    if ranked:
        _assign_ranks(levels, 1)
    gaps: List[dict] = []
    for j in range(len(levels) - 1):
        L, Uv = levels[j], levels[j + 1]
        rec = {"interval": [float(sub_down(Uv["lower"], L["upper"])), float(sub_up(Uv["upper"], L["lower"]))],
               "lower_from": "S2", "upper_from": "S2", "within_level": False,
               "levels": [j, j + 1]}
        if ranked:
            rec["between_ranks"] = [L["ranks"][1], Uv["ranks"][0]]
        gaps.append(rec)
    for j, lv in enumerate(levels):
        if lv["count"] > 1:
            rec = {"interval": [0.0, float(sub_up(lv["upper"], lv["lower"]))], "lower_from": "S2", "upper_from": "S2",
                   "within_level": True, "levels": [j, j]}
            if ranked:
                rec["between_ranks"] = list(lv["ranks"])
            gaps.append(rec)
    top = None
    if kind == "lowest" and "gap_lower_bound" in s3_evidence:
        k = int(sum(lv["count"] for lv in levels))
        top = {"interval": [float(s3_evidence["gap_lower_bound"]), float("inf")], "lower_from": "S3",
               "upper_from": None, "within_level": False, "levels": [len(levels) - 1, None],
               "between_ranks": [k, k + 1],
               "note": "the eigenvalue above the highest claimed level is not claimed: no upper bound"}
        gaps.append(top)
    gaps.sort(key=lambda g: (g["levels"][0], g["within_level"]))
    basis = ("S2 enclosures (Theorems 1-2, radii including the model shift) ordered by the S3 count "
             + {"lowest": "(Corollary 5)", "all": "(count = dimension)", "window": "(two inertia counts, Lemma 4)"}[kind]
             + "; levels are maximal chains of overlapping enclosures, each holding exactly its count of eigenvalues "
               "(pigeonhole step of docs/theory.md Section 6)"
             + ("; the gap above the highest claimed level has the Lemma 4 / Corollary 5 lower bound of S3 and no "
                "upper bound" if top is not None else ""))
    return {"refers_to": refers_to, "ranked": ranked,
            "rank_origin": ("eigenvalue rank in the full spectrum (1 = lowest)" if ranked else
                            "levels ordered inside the claimed window; ranks in the spectrum are not certified"),
            "n_levels": len(levels), "levels": levels, "gaps": gaps, "top_gap": top, "basis": basis}
