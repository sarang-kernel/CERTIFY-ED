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
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .construct import build_bitwise, hermitian_part
from .enclosure import OperatorAdapter, enclose
from .fp import U, ULP_MIN, add_up, fro_upper, gamma_dot, inf_norm_upper, norm2_upper, sub_down, up
from .inertia import inertia as inertia_count
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


def certify_low_energy(spec: ModelSpec, sectors: Sequence[SectorResult], E_cut: float,
                       H_full=None, inertia_method: str = "auto", label_symmetries: Sequence[str] = ()) -> Dict:
    t0 = time.time()
    H = None if H_full is None else hermitian_part(sp.csr_matrix(H_full))
    n = spec.dim
    cache = {}
    Hn = inf_norm_upper(H) if H is not None else 0.0
    per = []
    ok_all = True
    pooled, rmax, beta_max = [], 0.0, 0.0
    Bn = 0.0
    m_build = 0.0     # rounding of CERTIFY-ED's own reference blocks (statements refer to the exact spec)
    symmetrised = []
    for s in sectors:
        Hs = sp.csr_matrix(s.H)
        asym = abs(Hs - Hs.conj().T)
        herm_note = None
        if asym.nnz and asym.max() > 0:
            # IMP-20: record it, as certify() does with 'hermitian_part'; the statement concerns the Hermitian part
            herm_note = {"max_abs_asymmetry": float(asym.max()),
                         "note": "producer block not exactly Hermitian; the statement concerns its Hermitian part"}
            symmetrised.append(s.label)
            Hs = ((Hs + Hs.conj().T) / 2).tocsr()
        Bn = max(Bn, inf_norm_upper(Hs))
        red = restrict_to_u1(spec, s.V, cache) if H is None else None
        if red is not None:
            Ha, Va, st_a = red
            Ha = hermitian_part(Ha)          # Theorem 8 needs a Hermitian H; the builder shift covers this
        else:
            if H is None:
                H = hermitian_part(build_bitwise(spec))
            Ha, Va, st_a = H, s.V, None
        Hn = max(Hn, inf_norm_upper(Ha))
        key = None if st_a is None else (st_a.size, int(st_a.states_asc[0]), int(st_a.states_asc[-1]))
        if key not in cache.setdefault("_builder_shift", {}):
            cache["_builder_shift"][key] = _builder_shift(spec, st_a)
        m_build = max(m_build, cache["_builder_shift"][key])
        aud = audit_sector(Ha, Va, Hs, label=s.label)
        rec = {"label": s.label, "dim": int(Hs.shape[0]), "isometry_defect": aud.isometry_defect,
               "invariance_defect": aud.invariance_defect, "audit_passed": aud.passed,
               "hermitian_part": herm_note}
        if label_symmetries:
            from .sectors import audit_label
            rec["label_audit"] = {g: audit_label(spec, s.V, g, cache) for g in label_symmetries}
        mu = np.asarray(s.eigenvalues, dtype=float)
        mu_below = mu[mu < E_cut]
        r = inertia_count(Hs, E_cut, method=inertia_method)
        rec["inertia"] = r.to_dict()
        if not r.conclusive:
            rec["status"] = "INCONCLUSIVE"
            ok_all = False
            per.append(rec)
            continue
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
            pooled.extend(mu_below.tolist())
            rmax = max(rmax, rad)
            beta_max = max(beta_max, r.beta)
        elif r.negative > mu_below.size:
            rec["status"] = "MISSING"
            rec["missing_at_least"] = int(r.negative - mu_below.size)
            ok_all = False
        else:
            rec["status"] = "INCONCLUSIVE"
            ok_all = False
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
    status, summary = "CERTIFIED", ""
    bad_audit = [p["label"] for p in per if not p["audit_passed"]]
    if bad_audit:
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
    return out
