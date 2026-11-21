"""S8: the number of eigenvalues in each certified cluster, by bracketing inertia counts (Proposition 14).

For a cluster C of g pairs certified distinct (Theorem 2) with level enclosure J = [J_L, J_U], two counts
at sigma_lo < J_L and sigma_hi > J_U give nu_lo, beta_lo and nu_hi, beta_hi. With
I_in = (sigma_lo + beta_lo, sigma_hi - beta_hi) and I_out = [sigma_lo - beta_lo, sigma_hi + beta_hi],
#I_in <= nu_hi - nu_lo <= #I_out. If J is inside I_in and nu_hi - nu_lo = g, I_in holds exactly g
eigenvalues: 'degeneracy g certified' (a count in an interval of stated width; exact equality of the g
eigenvalues is not a floating-point statement). Otherwise INCONCLUSIVE with the counts.
Derivation and machine check: docs/proofs/P14_degeneracy_bracketing.md (docs/theory.md 9.1); independent
re-derivation pending (D-12).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .fp import add_up, sub_down
from .inertia import inertia as inertia_count

__all__ = ["degeneracy_counts"]


def _count(A, sigma: float, method: str, cache: Dict[float, object], budget_mb=None):
    if sigma not in cache:
        cache[sigma] = inertia_count(A, sigma, method=method, budget_mb=budget_mb)
    return cache[sigma]


def _probe(A, target: float, direction: int, margin: float, eps: float, method: str, cache: Dict[float, object],
           budget_mb=None):
    """A conclusive count at a shift beyond `target` (direction -1: below, +1: above) whose effective backward
    error is smaller than its distance to `target`; up to four tries, moving the shift outward."""
    sigma = target + direction * margin
    r = None
    for _ in range(4):
        r = _count(A, sigma, method, cache, budget_mb)
        if r.resource_failure:
            return sigma, r
        beta = float(add_up(r.beta, eps))
        dist = float(sub_down(sigma, target)) if direction > 0 else float(sub_down(target, sigma))
        if r.conclusive and beta < dist:
            return sigma, r
        sigma = target + direction * max(4 * beta, 2 * abs(sigma - target))
    return sigma, r


def degeneracy_counts(A, enc, clusters: Sequence[Sequence[int]], method: str = "auto", eps_shift: float = 0.0,
                      margin_rel: float = 1e-8, budget_mb=None) -> List[dict]:
    """Proposition 14 for every cluster (lists of indices into enc.mu). Returns one record per cluster."""
    cache: Dict[float, object] = {}
    scale = max(1.0, float(np.max(np.abs(enc.mu)))) if enc.mu.size else 1.0
    out = []
    for C in clusters:
        C = [int(i) for i in C]
        g = len(C)
        rad = add_up(enc.radius[C], eps_shift)
        JL = float(np.min(sub_down(enc.mu[C], rad)))
        JU = float(np.max(add_up(enc.mu[C], rad)))
        rec = {"members": C, "g_claimed": g, "level_enclosure": [JL, JU], "width": float(JU - JL),
               "distinct": bool(all(enc.groups[enc.group_of[i]].distinct for i in C))}
        if not rec["distinct"]:
            rec.update(status="INCONCLUSIVE", reason="cluster pairs not certified distinct (S2)")
            out.append(rec)
            continue
        if not np.isfinite(JL) or not np.isfinite(JU):
            rec.update(status="INCONCLUSIVE", reason="non-finite level enclosure", nonfinite_bound=True)
            out.append(rec)
            continue
        margin = max(margin_rel * scale, 16 * float(np.max(rad)))
        s_lo, r_lo = _probe(A, JL, -1, margin, eps_shift, method, cache, budget_mb)
        s_hi, r_hi = _probe(A, JU, +1, margin, eps_shift, method, cache, budget_mb)
        rec["counts"] = [r_lo.to_dict(), r_hi.to_dict()]
        if r_lo.resource_failure or r_hi.resource_failure:
            rec.update(status="NOT_CHECKED", reason="factorization not completed: " + (r_lo.note or r_hi.note))
            out.append(rec)
            continue
        if not (r_lo.conclusive and r_hi.conclusive):
            rec.update(status="INCONCLUSIVE", reason="inertia count inconclusive at a bracket shift")
            out.append(rec)
            continue
        b_lo = float(add_up(r_lo.beta, eps_shift))
        b_hi = float(add_up(r_hi.beta, eps_shift))
        I_in = (float(add_up(s_lo, b_lo)), float(sub_down(s_hi, b_hi)))
        I_out = (float(sub_down(s_lo, b_lo)), float(add_up(s_hi, b_hi)))
        diff = int(r_hi.negative - r_lo.negative)
        rec.update(bracket_inner=list(I_in), bracket_outer=list(I_out), count_difference=diff,
                   shifts=[s_lo, s_hi], beta=[b_lo, b_hi])
        if not (I_in[0] < JL and JU < I_in[1]):
            rec.update(status="INCONCLUSIVE", reason="backward error of the factorization reaches the level enclosure")
        elif diff == g:
            rec.update(status="CERTIFIED", degeneracy=g,
                       statement=f"exactly {g} eigenvalues in the open interval ({I_in[0]:.15g}, {I_in[1]:.15g}) "
                                 f"(level enclosure width {JU - JL:.2e}); Proposition 14 [derived: "
                                 f"docs/proofs/P14_degeneracy_bracketing.md; UNVERIFIED by independent "
                                 f"re-derivation, D-12]")
        elif diff > g:
            rec.update(status="INCONCLUSIVE", degeneracy_at_least=g,
                       reason=f"{diff - g} further eigenvalue(s) within the outer bracket "
                              f"[{I_out[0]:.15g}, {I_out[1]:.15g}]: the level may be larger or a neighbour is close")
        else:
            rec.update(status="FAILED", reason=f"bracket counts {diff} eigenvalues but {g} distinct pairs are claimed "
                                               f"inside it: Theorem 2 or the counts are contradicted")
        out.append(rec)
    return out
