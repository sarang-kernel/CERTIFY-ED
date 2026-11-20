"""Certified low-energy spectrum of H assembled from symmetry sectors.

Typical ED practice: diagonalize every symmetry block H_s = V_s^* H V_s
separately, pool the eigenvalues, sort, and report "the K lowest states" with
their quantum numbers. The pooled list is only the spectrum of H if (i) every
V_s is an isometry onto an H-invariant subspace, (ii) the subspaces are
mutually orthogonal, (iii) they span the whole Hilbert space, and (iv) no block
is missing eigenvalues below the cut. Each condition fails in practice through
a different bug (wrong representative phase or normalisation, a forgotten or
double-counted sector, Lanczos missing a state inside one block).

Statement (proof in the paper's appendix). Let W = [V_1 ... V_m] (n x n),
eta = ||W^*W - I||_2 < 1, D = ||H W - W B||_2 with B = diag(H_1..H_m), and
eps = rho = D + (||H|| + ||B||)(1 - sqrt(1 - eta)) (Weyl for the square W; see
docs/theory.md, Theorem 8 note). Suppose in every sector the claimed
eigenvalues below the cut are certified distinct with radius <= r and inertia
certifies that H_s has no further eigenvalue below E_cut - beta_s. Let
c_(1) <= ... <= c_(K) be the pooled claimed values. Then the K lowest
eigenvalues of H satisfy |lambda_j - c_(j)| <= r + eps, and
lambda_{K+1} >= E_cut - max_s beta_s - eps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .construct import build_bitwise, hermitian_part
from .enclosure import OperatorAdapter, enclose
from .fp import U, ULP_MIN, add_up, fro_upper, gamma_dot, inf_norm_upper, norm2_upper, sub_down, up
from .inertia import inertia as inertia_count
from .precision import binary64, check_values, summed, widening_record
from .sectors import audit_sector, restrict_to_u1
from .certify import _builder_shift
from .spec import ModelSpec


@dataclass
class SectorResult:
    label: str
    V: sp.spmatrix                      # n x d_s isometry (rows in the spec's full-space encoding)
    H: sp.spmatrix                      # d_s x d_s sector block (as used by the producer)
    eigenvalues: np.ndarray             # producer's claimed eigenvalues below the cut (may be empty)
    eigenvectors: Optional[np.ndarray]  # d_s x k


_UNSET = object()


def _binary64_sector(s: SectorResult, widened: Optional[list] = None) -> SectorResult:
    """s with its isometry, block and vectors converted exactly to binary64 where they are stored in a narrower format
    (precision.binary64, noted in `widened`), before the U(1) restriction, the audits and the decomposition sum or
    multiply them; the same object when nothing is converted. The claimed eigenvalues are checked
    (precision.check_values)."""
    check_values(s.eigenvalues, f"sector {s.label}: eigenvalues")
    V = summed(binary64(s.V, f"sector {s.label}: isometry", widened), f"sector {s.label}: isometry")
    H = summed(binary64(s.H, f"sector {s.label}: block", widened), f"sector {s.label}: block")
    X = binary64(s.eigenvectors, f"sector {s.label}: eigenvectors", widened)
    if V is s.V and H is s.H and X is s.eigenvectors:
        return s
    return replace(s, V=V, H=H, eigenvectors=X)


def _sector_record(spec: ModelSpec, s: SectorResult, E_cut: float, H, inertia_method: str,
                   label_symmetries: Sequence[str], cache: dict, red=_UNSET):
    """Everything certify_low_energy computes for one sector: (record, contribution, H). `H` is the Hermitian part of
    the full matrix when it is known or had to be built (returned so that the next sector reuses it); the
    contribution holds the numbers the assembly takes maxima over (Bn, Hn, m_build), the pooled values, the radius
    and beta of a COMPLETE sector, and whether the producer block was symmetrised. 3.3.0.dev0 (D-04): the body of
    the 3.2 loop, unchanged, so that sectors can run on separate workers. `red`: the U(1) restriction decided by the
    caller (a tuple, or None for the full matrix H); by default it is decided here as in the 3.2 loop."""
    Hs = summed(sp.csr_matrix(binary64(s.H, "sector block")), "sector block")   # exact; summed (precision.py)
    asym = abs(Hs - Hs.conj().T)
    herm_note = None
    symmetrised = False
    if asym.nnz and asym.max() > 0:
        # IMP-20: record it, as certify() does with 'hermitian_part'; the statement concerns the Hermitian part
        herm_note = {"max_abs_asymmetry": float(asym.max()),
                     "note": "producer block not exactly Hermitian; the statement concerns its Hermitian part"}
        symmetrised = True
        Hs = ((Hs + Hs.conj().T) / 2).tocsr()
    Bn = inf_norm_upper(Hs)
    if red is _UNSET:
        red = restrict_to_u1(spec, s.V, cache) if H is None else None
    if red is not None:
        Ha, Va, st_a = red
        Ha = hermitian_part(Ha)          # Theorem 8 needs a Hermitian H; the builder shift covers this
    else:
        if H is None:
            H = hermitian_part(build_bitwise(spec))
        Ha, Va, st_a = H, s.V, None
    Hn = inf_norm_upper(Ha)
    key = None if st_a is None else (st_a.size, int(st_a.states_asc[0]), int(st_a.states_asc[-1]))
    if key not in cache.setdefault("_builder_shift", {}):
        cache["_builder_shift"][key] = _builder_shift(spec, st_a)
    m_build = cache["_builder_shift"][key]
    aud = audit_sector(Ha, Va, Hs, label=s.label)
    rec = {"label": s.label, "dim": int(Hs.shape[0]), "isometry_defect": aud.isometry_defect,
           "invariance_defect": aud.invariance_defect, "audit_passed": aud.passed,
           "hermitian_part": herm_note}
    if label_symmetries:
        from .sectors import audit_label
        rec["label_audit"] = {g: audit_label(spec, s.V, g, cache) for g in label_symmetries}
    contrib = {"Bn": Bn, "Hn": Hn, "m_build": m_build, "symmetrised": symmetrised, "complete": False,
               "pooled": [], "rad": 0.0, "beta": 0.0}
    mu = np.asarray(s.eigenvalues, dtype=float)
    mu_below = mu[mu < E_cut]
    r = inertia_count(Hs, E_cut, method=inertia_method)
    rec["inertia"] = r.to_dict()
    if not r.conclusive:
        rec["status"] = "INCONCLUSIVE"
        return rec, contrib, H
    if mu_below.size:
        sel = np.where(mu < E_cut)[0]
        enc = enclose(OperatorAdapter(Hs), mu[sel], np.asarray(s.eigenvectors)[:, sel])
        inside = bool(np.all(add_up(enc.mu, enc.radius) < sub_down(E_cut, r.beta)))
        distinct = enc.all_distinct
        rad = float(enc.radius.max())
    else:
        inside, distinct, rad = True, True, 0.0
    rec.update(n_claimed=int(mu_below.size), count=int(r.negative), max_radius=rad,
               beta=float(r.beta))
    if distinct and inside and r.negative == mu_below.size:
        rec["status"] = "COMPLETE"
        contrib.update(complete=True, pooled=mu_below.tolist(), rad=rad, beta=float(r.beta))
    elif r.negative > mu_below.size:
        rec["status"] = "MISSING"
        rec["missing_at_least"] = int(r.negative - mu_below.size)
    else:
        rec["status"] = "INCONCLUSIVE"
    return rec, contrib, H


def _sector_job(job):
    """One sector in a worker process (parallel.run_tasks), with the restriction the serial loop would use."""
    spec, s, E_cut, H, inertia_method, label_symmetries, red = job
    rec, contrib, _ = _sector_record(spec, s, E_cut, H, inertia_method, label_symmetries, {}, red)
    return rec, contrib


def _sector_resource_reason(result) -> str:
    rec, _ = result
    inr = rec.get("inertia") or {}
    return "inertia count: resource failure" if inr.get("resource_failure") else ""


def _sector_predicted_mb(s: SectorResult, n_full: int, has_full: bool) -> float:
    """Advisory peak of one sector (cost.py): the inertia count of the block, its enclosure columns, and the full
    matrix when the sector is not a U(1) block (estimated from the block's fill per row)."""
    from .cost import dense_inertia_mb, sparse_inertia_mb
    Hs = sp.csr_matrix(s.H)
    d, nnz = int(Hs.shape[0]), int(Hs.nnz)
    inert = dense_inertia_mb(d) if d <= 4096 else sparse_inertia_mb(d, nnz)
    vec = d * max(1, np.asarray(s.eigenvalues).size) * 8 * 4 / 2 ** 20
    full = 0.0 if has_full else (nnz / max(1, d)) * n_full * 12 * 3 / 2 ** 20
    return float(inert + vec + full)


def certify_low_energy(spec: ModelSpec, sectors: Sequence[SectorResult], E_cut: float,
                       H_full=None, inertia_method: str = "auto", label_symmetries: Sequence[str] = (),
                       workers: int = 1, cap_mb: Optional[float] = None) -> Dict:
    """The statement of the module docstring. workers (3.3.0.dev0, D-04): the sectors are audited, counted and
    enclosed on this many processes under the rules of `parallel` (memory budget per worker derived from the process
    cap; a sector that does not fit a worker's budget or fails there is redone here); default 1 = the 3.2 loop.
    The result does not depend on `workers` (tests/test_parallel.py); with workers > 1 it carries the scheduling
    record under 'parallel'."""
    t0 = time.time()
    widened: list = []
    sectors = [_binary64_sector(s, widened) for s in sectors]   # exact, before any conversion or product (precision.py)
    H = None if H_full is None else hermitian_part(summed(sp.csr_matrix(binary64(H_full, "H_full", widened)), "H_full"))
    n = spec.dim
    cache = {}
    Hn = inf_norm_upper(H) if H is not None else 0.0
    per = []
    ok_all = True
    pooled, rmax, beta_max = [], 0.0, 0.0
    Bn = 0.0
    m_build = 0.0     # rounding of CERTIFY-ED's own reference blocks (statements refer to the exact spec)
    symmetrised = []
    sched = None
    if int(workers) > 1 and len(sectors) > 1:
        from .parallel import run_tasks
        # decide, in sector order, what the 3.2 loop uses for each sector: its U(1) restriction while no sector has
        # needed the full matrix, the full matrix from the first sector without a restriction on
        jobs, H_cur = [], H
        for s in sectors:
            red = restrict_to_u1(spec, s.V, cache) if H_cur is None else None
            if red is None and H_cur is None:
                H_cur = hermitian_part(build_bitwise(spec))
            jobs.append((spec, s, E_cut, None if red is not None else H_cur, inertia_method, tuple(label_symmetries), red))
        pred = [_sector_predicted_mb(s, n, H is not None) for s in sectors]
        outs, rep = run_tasks(_sector_job, jobs, workers=int(workers), predicted_mb=pred, cap_mb=cap_mb,
                              needs_serial=_sector_resource_reason)
        sched = rep.to_dict()
    else:
        outs = []
        for s in sectors:
            rec, contrib, H = _sector_record(spec, s, E_cut, H, inertia_method, label_symmetries, cache)
            outs.append((rec, contrib))
    for s, (rec, contrib) in zip(sectors, outs):
        if contrib["symmetrised"]:
            symmetrised.append(s.label)
        Bn = max(Bn, contrib["Bn"])
        Hn = max(Hn, contrib["Hn"])
        m_build = max(m_build, contrib["m_build"])
        if rec.get("status") != "COMPLETE":
            ok_all = False
        else:
            pooled.extend(contrib["pooled"])
            rmax = max(rmax, contrib["rad"])
            beta_max = max(beta_max, contrib["beta"])
        per.append(rec)

    # decomposition: W^*W - I and total invariance defect
    W = sp.hstack([sp.csr_matrix(s.V) for s in sectors]).tocsc()
    cplx = np.iscomplexobj(W.data)
    covered = int(W.shape[1])
    G = (W.conj().T @ W - sp.identity(covered, format="csc")).tocsr()
    wmax = int(np.diff(W.indptr).max()) if covered else 1
    eta = up(fro_upper(G) + gamma_dot(wmax, cplx) * fro_upper(W) ** 2)
    # ||H W - W B||_2 <= ||.||_F = ||(inv_1, ..., inv_m)||_2 (block columns); norm2_upper covers the roundings of
    # the sum of squares and their underflow (was up(sqrt(sum of squares)) before track/backlog-proofs)
    D = norm2_upper(np.array([p["invariance_defect"] for p in per], dtype=float))
    out = {"n": n, "covered_dim": covered, "n_sectors": len(sectors), "eta": eta,
           "invariance_total": D, "E_cut": E_cut, "sectors": per,
           "symmetrised_blocks": symmetrised}
    widening = widening_record(widened)
    if widening is not None:                 # only when an input was converted: binary64 inputs keep their record
        out["input_widening"] = widening
    status, summary = "CERTIFIED", ""
    nonfinite = [p["label"] for p in per
                 if not (math.isfinite(p["isometry_defect"]) and math.isfinite(p["invariance_defect"]))]
    bad_audit = [p["label"] for p in per if not p["audit_passed"]]
    if nonfinite:
        # an audit bound that is not finite (a bound of the accounting overflowed) is no finding about the sector
        # bases
        status = "INCONCLUSIVE"
        summary = (f"the audit bounds of {len(nonfinite)} sector(s) are not finite (a bound of the floating-point "
                   f"accounting overflowed; e.g. {nonfinite[:4]}): nothing is stated")
    elif bad_audit:
        status = "FAILED"
        summary = (f"{len(bad_audit)} sector basis/bases are not orthonormal H-invariant subspaces "
                   f"(e.g. {bad_audit[:4]})")
    elif covered != n:
        status = "FAILED"
        summary = f"sectors cover {covered} of {n} dimensions"
    elif not eta < 1:
        status = "FAILED"
        summary = f"sector bases are not orthonormal (eta={eta:.2e})"
    elif not ok_all:
        bad = [p["label"] for p in per if p.get("status") != "COMPLETE"]
        status = "FAILED" if any(p.get("status") == "MISSING" for p in per) else "INCONCLUSIVE"
        summary = f"sectors not certified complete below the cut: {bad[:8]}"
    if status == "CERTIFIED" and pooled and add_up(max(pooled), rmax) >= sub_down(E_cut, beta_max):
        # an unclaimed eigenvalue of one block could lie in [E_cut - beta_max, E_cut - beta_s) below a
        # claimed value of another block; the pooled list is then not provably the K lowest
        status = "INCONCLUSIVE"
        summary = (f"highest claimed value {max(pooled):.10g} (+{rmax:.1e}) is not below "
                   f"E_cut - max beta = {E_cut - beta_max:.10g}")
    if status == "CERTIFIED":
        # W is square here (covered == n), so its polar factor is unitary: Qhat^H H Qhat - B = Qhat^H Rhat
        # with ||Rhat|| <= D + (||H|| + ||B||) delta = rho, and Weyl pairs the sorted eigenvalues within
        # rho (docs/theory.md, Theorem 8 note; was (1 + sqrt 5)/2 rho before 3.1.0, review IMP-12/IMP-20)
        delta = up(1 - math.sqrt(1 - eta)) + 4 * U
        rho = up(D + (Hn + Bn) * delta) + ULP_MIN
        eps = up(rho + m_build)
        pooled = np.sort(np.asarray(pooled))
        K = int(pooled.size)
        out.update(eps_decomposition=eps, builder_shift=m_build, max_sector_radius=rmax, K=K,
                   lowest=pooled.tolist(), radius=float(add_up(rmax, eps)),
                   next_lower_bound=float(sub_down(sub_down(E_cut, beta_max), eps)))
        summary = (f"the {K} lowest eigenvalues of H equal the pooled sector values within "
                   f"{rmax + eps:.2e}; the next eigenvalue is >= {E_cut - beta_max - eps:.10g}")
    out.update(status=status, summary=summary, seconds=time.time() - t0)
    if sched is not None:
        out["parallel"] = sched
    return out
