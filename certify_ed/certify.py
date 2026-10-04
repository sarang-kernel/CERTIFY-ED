"""Orchestration: turn a claimed ED result into a certificate of explicit statements.

Statements (ids are stable and appear in certificates):

  S1.model_fidelity   the producer's operator equals the operator defined by the
                      ModelSpec (entrywise, or invariance + isometry for a
                      symmetry sector, or agreement on random probes for a
                      matrix-free producer)
  S2.eigenpairs       each claimed eigenvalue lies within a certified radius of a
                      true eigenvalue, distinct claimed pairs matched to distinct
                      eigenvalues (enclosure.Theorems 1-2, optionally 3)
  S3.completeness     no eigenvalue is missing from the claimed set inside the
                      claimed window (inertia counts), and a certified lower
                      bound on the first unclaimed eigenvalue (gap)
  S4.eigenspaces      basis-independent statements about claimed clusters:
                      projector error, certified observables, quantum numbers
  S5.corroboration    optional recomputation with algorithmically distinct
                      solvers, with their shared components declared

Every statement records its evidence, its assumptions and its cost in seconds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import Certificate, environment_record, matrix_sha256, array_sha256
from .construct import (StateSet, build_bitwise, build_kron, cross_construct, full_state_set,
                        matrix_difference, product_tolerance, SpecOperator)
from .eigenspace import (certify_eigenspace, certify_observable, certify_quantum_numbers,
                         commutes_diagonal, rotation_spread)
from .enclosure import OperatorAdapter, enclose, kato_temple, _cluster_bound, cluster_groups
from .fp import U, float_env_record, inf_norm_upper, up
from .inertia import inertia as inertia_count
from .invariants import compare_invariants, matrix_invariants, spec_invariants
from .sectors import audit_sector, restrict_to_u1
from .spec import ModelSpec


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A claimed spectral result, as handed over by a producer.

    kind:
      'lowest' : the claimed eigenvalues are the len(eigenvalues) lowest of the operator
      'window' : the claimed eigenvalues are all eigenvalues in `window` = (a, b)
      'all'    : the claim is the full spectrum
      'subset' : no completeness claim (only accuracy of the given pairs)
    operator: the producer's operator (dense/sparse matrix or LinearOperator);
      None means "use the matrix CERTIFY-ED builds from the spec".
    states: basis states of the producer's operator when it is a U(1)-restricted
      block expressed in the spec's integer encoding (e.g. QuSpin basis.states).
    sector_isometry: for symmetry-adapted blocks (momentum, parity, ...), the
      producer's isometry V (rows in the spec's full-space encoding).
    """
    eigenvalues: Sequence[float]
    eigenvectors: Optional[np.ndarray] = None
    operator: Any = None
    kind: str = "subset"
    window: Optional[Tuple[float, float]] = None
    states: Optional[StateSet] = None
    sector_isometry: Optional[sp.spmatrix] = None
    producer: Dict[str, Any] = field(default_factory=dict)
    accuracy_target: Optional[float] = None   # None -> default 1e-8*max(1,||H||); inf -> no target
    description: str = ""


@dataclass
class Options:
    model_fidelity: bool = True
    kron_cross_check_max_dim: int = 2 ** 14
    reference_max_dim: int = 2 ** 22
    inertia: bool = True
    inertia_method: str = "auto"
    inertia_max_dim: int = 60000
    gap_fractions: Sequence[float] = (0.9, 0.5, 0.2, 0.05)
    kato_temple: bool = True
    eigenspaces: str = "clusters"            # 'none' | 'clusters' | 'all'
    cluster_tol: float = 0.0                 # claimed values closer than this form one cluster (S4)
    observables: Dict[str, Any] = field(default_factory=dict)       # name -> operator in claim space
    symmetries: Dict[str, Tuple[np.ndarray, Optional[float]]] = field(default_factory=dict)
    #   name -> (diagonal of S in the claim space, lattice spacing of its eigenvalues or None)
    corroborate: Sequence[str] = ()
    store_eigenvectors: bool = False
    probe_vectors: int = 3
    seed: int = 12345


DEFAULT_REL_ACCURACY = 1e-8


def _stmt(sid, status, summary, evidence=None, assumptions=None, seconds=0.0):
    assert status in ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED")
    return {"id": sid, "status": status, "summary": summary, "evidence": evidence or {},
            "assumptions": assumptions or [], "seconds": round(float(seconds), 6)}


def _subspace_shift(iso: float, inv: float, H_norm: float, M_norm: float) -> Optional[float]:
    """Theorem 2 with X = V (sector isometry, g = d_s) and Hermitian M = H_s."""
    if not iso < 1:
        return None
    delta = up(1 - math.sqrt(1 - iso)) + 4 * U
    rho = up(inv + (H_norm + M_norm) * delta)
    return up(0.5 * (rho + math.sqrt(rho * rho + 4 * rho * rho)))


# ---------------------------------------------------------------------------
# S1: model fidelity
# ---------------------------------------------------------------------------

def _diagnose_mismatch(spec: ModelSpec, states: StateSet, Hp: sp.spmatrix, Href: sp.spmatrix,
                       max_couplings: int = 400) -> dict:
    """Explain a matrix mismatch: global scale factor, or per-coupling attribution."""
    D = (sp.csr_matrix(Hp) - Href).tocsr()
    out = {}
    num = (Href.multiply(Hp.conj())).sum()
    den = (Href.multiply(Href.conj())).sum()
    if abs(den) > 0:
        alpha = complex(num / den)
        resid = sp.linalg.norm(sp.csr_matrix(Hp) - alpha.real * Href) / max(sp.linalg.norm(Href), 1e-300)
        out["best_scale_factor"] = alpha.real
        out["relative_residual_after_scaling"] = float(resid)
    if spec.n_terms <= max_couplings:
        # least-squares fit  D ~ sum_t a_t O_t  over all couplings of the spec (the O_t need not be
        # orthogonal once restricted to a sector, so a joint fit is required)
        n = D.shape[0]
        labels, cols = [], []
        for t in spec.terms:
            for c, s in t.couplings:
                sub = ModelSpec(spec.n_sites, spec.local, [type(t)(t.op, [(1.0, s)])],
                                state_order=spec.state_order)
                O = sp.coo_matrix(build_bitwise(sub, states))
                labels.append((t.op, list(s), complex(c)))
                cols.append(O)
        keys_all = np.unique(np.concatenate([o.row.astype(np.int64) * n + o.col for o in cols] +
                                            [sp.coo_matrix(D).row.astype(np.int64) * n + sp.coo_matrix(D).col]))
        def vec(M):
            M = sp.coo_matrix(M)
            v = np.zeros(keys_all.size, dtype=complex)
            v[np.searchsorted(keys_all, M.row.astype(np.int64) * n + M.col)] += M.data
            return v
        Mmat = np.column_stack([vec(o) for o in cols])
        d = vec(D)
        a, _, rank, sv = np.linalg.lstsq(Mmat, d, rcond=None)
        resid = float(np.linalg.norm(Mmat @ a - d) / max(np.linalg.norm(d), 1e-300))
        attrib = []
        for (op_, s_, c_), ai in zip(labels, a):
            if abs(ai) > 1e-9 * max(1.0, abs(c_)):
                attrib.append({"op": op_, "sites": s_, "spec_coefficient": [c_.real, c_.imag],
                               "producer_minus_spec": [float(ai.real), float(ai.imag)]})
        attrib.sort(key=lambda r: -abs(complex(*r["producer_minus_spec"])))
        out["coupling_fit"] = {
            "method": "least squares over all spec couplings on the producer's basis",
            "rank": int(rank), "n_couplings": len(labels),
            "relative_residual": resid,
            "interpretation": ("difference fully explained by changed coupling coefficients"
                               if resid < 1e-8 else
                               "difference NOT expressible as changed coefficients of the spec's operator "
                               "strings (wrong operator content, basis/ordering convention, or missing term types)"),
            "couplings": attrib[:16], "n_couplings_implicated": len(attrib)}
    return out


def _builder_shift(spec: ModelSpec, states: Optional[StateSet] = None) -> float:
    """Upper bound on ||H_ref - H_spec||_2, where H_spec is the spec operator on `states` in exact
    arithmetic (couplings taken as the given binary floating-point numbers) and H_ref is CERTIFY-ED's
    floating-point build of it.

    Every entry of H_ref is a sum of at most m = n_terms * 2^Lmax contributions, each a product of at
    most Lmax + 1 factors, so |H_ref - H_spec| <= gamma_{m + Lmax + 2} |H|_abs entrywise, where |H|_abs
    sums the absolute values of the contributions. For a Hermitian specification |H|_abs is symmetric,
    hence ||H_ref - H_spec||_2 <= gamma * max_i (|H|_abs 1)_i, evaluated with the matrix-free operator
    (the computed row sums of nonnegative terms are inflated by a further gamma factor)."""
    from .fp import gamma
    op = SpecOperator(spec, states)
    n = op.shape[0]
    if n == 0 or spec.n_terms == 0:
        return 0.0
    max_len = max((len(t.op) for t in spec.terms), default=1)
    width = spec.n_terms * (2 ** max_len) + max_len + 2
    row = float(np.max(op.abs_matvec(np.ones(n))))
    g = gamma(width)
    return up(g * row * (1 + gamma(width)))


def check_model_fidelity(claim: Claim, spec: ModelSpec, opts: Options, H_full_ref=None):
    t0 = time.time()
    ev = {"spec_sha256": spec.sha256(), "spec_name": spec.name}
    op = claim.operator
    if claim.sector_isometry is not None:
        V_audit = claim.sector_isometry
        red = restrict_to_u1(spec, claim.sector_isometry) if H_full_ref is None else None
        if red is not None:
            H_full_ref, V_audit, st_u1 = red
            ev["audit_space"] = {"kind": "U(1) block (exactly invariant; leak 0)", "label": st_u1.label,
                                 "dimension": int(st_u1.size)}
        elif H_full_ref is None:
            H_full_ref = build_bitwise(spec)
            ev["audit_space"] = {"kind": "full Hilbert space", "dimension": int(spec.dim)}
        Hs = sp.csr_matrix(op)
        herm = float(abs(Hs - Hs.conj().T).max()) if Hs.nnz else 0.0
        ev["hermiticity_max_abs"] = herm
        # the Hermitian part is what S2-S4 certify; audit exactly that matrix
        Hs = ((Hs + Hs.conj().T) / 2).tocsr() if herm > 0 else Hs
        audit = audit_sector(H_full_ref, V_audit, Hs, label="producer sector")
        ev["sector_audit"] = audit.to_dict()
        ev["model_shift_bound"] = _builder_shift(spec, st_u1 if red is not None else None)
        status = "CERTIFIED" if audit.passed else "FAILED"
        summ = (f"producer sector block is an invariant, orthonormal restriction of the spec "
                f"operator (isometry defect {audit.isometry_defect:.1e}, invariance defect "
                f"{audit.invariance_defect:.1e})" if status == "CERTIFIED" else
                f"sector audit failed (isometry defect {audit.isometry_defect:.1e}, invariance "
                f"defect {audit.invariance_defect:.1e}, hermiticity {herm:.1e})")
        return _stmt("S1.model_fidelity", status, summ, ev,
                     ["full-space reference built by CERTIFY-ED from the spec (bitwise path)"],
                     time.time() - t0), H_full_ref, audit

    states = claim.states if claim.states is not None else full_state_set(spec)
    if states.size > opts.reference_max_dim:
        return _stmt("S1.model_fidelity", "NOT_CHECKED",
                     f"reference build skipped (dimension {states.size} > {opts.reference_max_dim})",
                     ev, [], time.time() - t0), None, None
    Href, leak = build_bitwise(spec, states, return_leak=True)
    ev["reference"] = {"path": "bitwise", "dimension": int(states.size), "nnz": int(Href.nnz),
                       "state_set": states.label, "leak_out_of_state_set": leak}
    if states.label == "full" and spec.dim <= opts.kron_cross_check_max_dim:
        cc = cross_construct(spec, kron_max_dim=opts.kron_cross_check_max_dim)
        ev["internal_cross_construction"] = {k: cc[k] for k in ("agree", "max_abs_diff", "tolerance") if k in cc}
        if cc.get("agree") is False:
            return _stmt("S1.model_fidelity", "INCONCLUSIVE",
                         "CERTIFY-ED's two reference builders disagree (internal error)", ev,
                         [], time.time() - t0), Href, None
    if states.label == "full":
        si = spec_invariants(spec) if spec.n_terms <= 400 else None
        if si is not None:
            ev["spec_invariants_vs_reference"] = compare_invariants(si, matrix_invariants(Href))
    if leak > 0:
        return _stmt("S1.model_fidelity", "FAILED",
                     f"the spec does not conserve the quantity defining the producer's basis "
                     f"(amplitude leaving the state set: {leak:.3e})", ev, [], time.time() - t0), Href, None
    ev["model_shift_bound"] = _builder_shift(spec, states)   # replaced below when a producer matrix is compared
    if op is None:
        ev["operator_source"] = "built by CERTIFY-ED from the spec"
        return _stmt("S1.model_fidelity", "CERTIFIED",
                     "operator built from the spec by CERTIFY-ED (no producer matrix to compare)",
                     ev, [], time.time() - t0), Href, None
    if sp.issparse(op) or isinstance(op, np.ndarray):
        Hp = sp.csr_matrix(op)
        herm = float(abs(Hp - Hp.conj().T).max()) if Hp.nnz else 0.0
        ev["hermiticity_max_abs"] = herm
        diff = matrix_difference(Hp, Href)
        tol = product_tolerance(spec, diff.get("max_abs_entry", 1.0))
        ev["comparison"] = dict(diff, tolerance=tol)
        ok = diff["shape_match"] and diff["max_abs_diff"] <= tol
        if ok:
            Hh = ((Hp + Hp.conj().T) / 2).tocsr() if herm > 0 else Hp
            Dm = (Hh - Href).tocsr()
            # ||D||_2 <= sqrt(||D||_1 ||D||_inf) <= max(||D||_1, ||D||_inf)
            ev["model_shift_bound"] = up(max(inf_norm_upper(Dm), inf_norm_upper(Dm.conj().T.tocsr()))
                                         + ev["model_shift_bound"])
            return _stmt("S1.model_fidelity", "CERTIFIED",
                         f"producer matrix equals the spec matrix entrywise (max |diff| "
                         f"{diff['max_abs_diff']:.1e} <= {tol:.1e})", ev,
                         ["spec encoding (site order, state order, local conventions) matches the producer's"],
                         time.time() - t0), Href, None
        if diff["shape_match"]:
            ev["diagnosis"] = _diagnose_mismatch(spec, states, Hp, Href)
        return _stmt("S1.model_fidelity", "FAILED",
                     f"producer matrix differs from the spec matrix (max |diff| {diff['max_abs_diff']:.3e}, "
                     f"{diff.get('n_differing_entries', '?')} entries)", ev, [], time.time() - t0), Href, None
    # black-box operator: probabilistic agreement on random vectors
    rng = np.random.default_rng(opts.seed)
    worst = 0.0
    for _ in range(opts.probe_vectors):
        z = rng.standard_normal(states.size) + (1j * rng.standard_normal(states.size)
                                                if not spec.is_real() else 0.0)
        a = op.matvec(z) if hasattr(op, "matvec") else op @ z
        b = Href @ z
        worst = max(worst, float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)))
    ev["probe"] = {"n_vectors": opts.probe_vectors, "max_relative_difference": worst}
    tol = 1e3 * U * max(1, spec.n_terms)
    st = "CERTIFIED" if worst <= tol else "FAILED"
    return _stmt("S1.model_fidelity", st,
                 f"matrix-free producer agrees with the spec on {opts.probe_vectors} random vectors "
                 f"(max rel. diff {worst:.1e})" if st == "CERTIFIED" else
                 f"matrix-free producer disagrees with the spec (max rel. diff {worst:.3e})",
                 ev, ["probabilistic: a difference orthogonal to all probes would be missed "
                      "(probability zero for Gaussian probes in exact arithmetic)"],
                 time.time() - t0), Href, None


# ---------------------------------------------------------------------------
# S3: completeness
# ---------------------------------------------------------------------------

def _certify_lowest(A, mu_sorted, enc, opts, eps_shift=0.0):
    """Completeness of a 'k lowest' claim. A: matrix used for inertia.

    Probe order (each probe = one sparse/dense LDL^T factorization): first near a
    floating-point estimate of the (k+1)-th eigenvalue (a hint; correctness rests only
    on the inertia counts), then progressively closer to the highest claimed value,
    finally at a minimal margin. Typical cost when the claim is complete: one factorization.
    The first probe that counts more than k eigenvalues triggers one count just below the
    highest claimed value; if eigenvalues are missing there the claim fails at once
    (typical cost of a FAILED statement: two factorizations).
    """
    k = mu_sorted.size
    rad = enc.radius[np.argsort(enc.mu, kind="stable")] + eps_shift
    hi = mu_sorted + rad
    lo = mu_sorted - rad
    hi_k = float(hi.max())
    counts = []
    scale = max(1.0, float(np.max(np.abs(mu_sorted))))
    margin = max(1e-8 * scale, 16 * float(rad.max()))
    distinct = enc.all_distinct
    est = _estimate_next(A, k, seed=opts.seed)
    sigmas = []
    if est is not None and est > hi_k + margin:
        sigmas += [hi_k + f * (est - hi_k) for f in opts.gap_fractions if f * (est - hi_k) > margin]
    sigmas.append(hi_k + margin)
    ev = {"counts": counts, "next_eigenvalue_estimate": est}
    r, sigma = None, None
    last_reason = None
    low_probe = {}

    lo_top = float(lo[np.argmax(mu_sorted)])

    def _below_top():
        """Inertia count just below the lower end of the highest claimed enclosure (cached).

        The shift starts at a minimal margin below; if the factorization's backward error is not
        smaller than the distance to the enclosure, the shift is moved down to four times the
        backward error (at most three retries), as for the probe above the claim."""
        if not low_probe:
            s_low = lo_top - margin
            for _ in range(4):
                r_low = inertia_count(A, s_low, method=opts.inertia_method)
                counts.append(r_low.to_dict())
                if not r_low.conclusive or r_low.beta + eps_shift < lo_top - s_low:
                    break
                s_low = lo_top - 4 * (r_low.beta + eps_shift)
            low_probe.update(s_low=s_low, r=r_low)
        return low_probe["s_low"], low_probe["r"]

    def _missing_below(s_low, r_low):
        """Lower bound on eigenvalues missing below the top claim, or None if not conclusive.

        Eigenvalues <= s_low + beta number at least nu; claimed enclosures reaching below s_low + beta
        number c; if nu > c, at least nu - c eigenvalues below the top enclosure are unclaimed."""
        if r_low.conclusive and r_low.beta + eps_shift < lo_top - s_low:
            bl = r_low.beta + eps_shift
            return int(r_low.negative - np.sum(lo <= s_low + bl))
        return None

    for idx, sigma in enumerate(sigmas):
        tries = 0
        while True:
            r = inertia_count(A, sigma, method=opts.inertia_method)
            counts.append(r.to_dict())
            # the final (minimal-margin) probe sits close to a claimed eigenvalue, where a
            # no-pivoting factorization can lose accuracy: widen the margin to the observed beta
            if (idx == len(sigmas) - 1 and r.conclusive and r.beta + eps_shift >= (sigma - hi_k) / 2
                    and tries < 3):
                sigma = hi_k + 4 * (r.beta + eps_shift)
                tries += 1
                continue
            break
        if not r.conclusive:
            last_reason = f"factorization at sigma={sigma:.10g} inconclusive ({r.note})"
            continue
        beta = r.beta + eps_shift
        if r.negative == k and distinct and hi_k < sigma - beta:
            # exactly k eigenvalues below sigma - beta, and they are the claimed ones
            ev.update(top_shift=sigma, beta_effective=beta, next_eigenvalue_lower_bound=sigma - beta,
                      certified_window=["-inf", sigma - beta], gap_lower_bound=(sigma - beta) - hi_k)
            return "CERTIFIED", ev, (float("-inf"), sigma - beta)
        if r.negative == k and distinct:
            last_reason = (f"factorization backward error ({beta:.2e}) too large relative to the "
                           f"margin above the highest claimed value")
        if r.negative <= k:
            break           # descending further cannot help
        # more than k eigenvalues below sigma: decide at once whether any lies below the top claim
        s_low, r_low = _below_top()
        miss = _missing_below(s_low, r_low)
        if miss is not None and miss > 0:
            ev.update(top_shift=sigma, beta_effective=beta, missing_below_top_claim_at_least=miss,
                      witness_shift=s_low)
            return "FAILED", ev, None
    if r is None or not r.conclusive:
        return "INCONCLUSIVE", dict(ev, reason=last_reason or "no conclusive inertia count"), None
    beta = r.beta + eps_shift
    ev.update(top_shift=sigma, beta_effective=beta)
    if r.negative == k and not distinct:
        return "INCONCLUSIVE", dict(ev, reason="claimed pairs not all certified distinct (S2); "
                                               "the count matches but cannot be attributed"), None
    if r.negative == k:
        return "INCONCLUSIVE", dict(ev, reason=last_reason), None
    if r.negative < k:
        return "INCONCLUSIVE", dict(ev, reason=f"inertia counts {r.negative} eigenvalues below "
                                               f"{sigma:.10g} but {k} claimed pairs lie below it: "
                                               "claimed pairs are not all distinct eigenpairs"), None
    # Something is not accounted for. Is anything missing BELOW the top claimed value?
    s_low, r_low = _below_top()
    miss = _missing_below(s_low, r_low)
    if miss is not None:
        bl = r_low.beta + eps_shift
        if miss > 0:
            ev["missing_below_top_claim_at_least"] = miss
            ev["witness_shift"] = s_low
            return "FAILED", ev, None
        c_lt = int(np.sum(hi < s_low - bl))
        if distinct and r_low.negative == c_lt:
            ev["complete_below"] = s_low - bl
    if r.negative > k:
        ev["unclaimed_near_top"] = int(r.negative - k)
        if "complete_below" in ev:
            n_top_claimed = int(np.sum(lo > ev["complete_below"]))
            ev["top_multiplet"] = {"claimed": n_top_claimed,
                                   "at_least": int(r.negative - r_low.negative),
                                   "interval": [ev["complete_below"], sigma + beta]}
            ev["reason"] = (f"the claim is complete below {ev['complete_below']:.10g}, but the highest "
                            f"claimed level is a (quasi-)degenerate multiplet of at least "
                            f"{r.negative - r_low.negative} eigenvalues within "
                            f"[{ev['complete_below']:.10g}, {sigma + beta:.10g}], of which only "
                            f"{n_top_claimed} are claimed: 'the k lowest' is ill-posed for this k")
        else:
            ev["reason"] = ("at least one unclaimed eigenvalue lies within "
                            f"{margin + beta:.2e} above the highest claimed one")
        return "INCONCLUSIVE", ev, None
    ev["reason"] = "count above the claim could not be classified"
    return "INCONCLUSIVE", ev, None


def _estimate_next(A, k, seed=12345):
    """Floating-point estimate of the (k+1)-th smallest eigenvalue (a hint only).

    The Lanczos start vector is drawn from a generator seeded with ``seed`` so that the shifts, and
    hence the certified gap and the run time, are reproducible (SciPy otherwise seeds from OS entropy)."""
    try:
        n = A.shape[0]
        if k + 1 > n:
            return None
        if not sp.issparse(A) or n <= 1500:
            M = A.toarray() if sp.issparse(A) else np.asarray(A)
            import scipy.linalg as _la
            return float(_la.eigvalsh(M, subset_by_index=[k, k])[0])
        import scipy.sparse.linalg as _spla
        v0 = np.random.default_rng(seed).standard_normal(n)
        w = _spla.eigsh(A, k=min(k + 6, n - 2), which="SA", return_eigenvectors=False, tol=1e-10, v0=v0)
        return float(np.sort(w)[k])
    except Exception:
        return None


def _certify_window(A, enc, a, b, opts, eps_shift=0.0):
    ra = inertia_count(A, a, method=opts.inertia_method)
    rb = inertia_count(A, b, method=opts.inertia_method)
    ev = {"counts": [ra.to_dict(), rb.to_dict()], "window": [a, b]}
    if not (ra.conclusive and rb.conclusive):
        return "INCONCLUSIVE", dict(ev, reason="inertia inconclusive at a window edge"), None
    ba, bb = ra.beta + eps_shift, rb.beta + eps_shift
    n = rb.negative - ra.negative
    lo = enc.mu - enc.radius - eps_shift
    hi = enc.mu + enc.radius + eps_shift
    inside = (lo > a + ba) & (hi < b - bb)
    touching = (hi >= a - ba) & (lo <= b + bb)
    c_in = int(np.sum(inside)) if enc.all_distinct else None
    c_all = int(np.sum(touching))
    ev.update({"count_from_inertia": int(n), "claimed_certified_inside": c_in,
               "claimed_touching": c_all, "beta_a": ba, "beta_b": bb})
    if c_in is not None and n == c_in:
        ev["certified_window"] = [a + ba, b - bb]
        return "CERTIFIED", ev, (a + ba, b - bb)
    if n > c_all:
        ev["missing_at_least"] = int(n - c_all)
        return "FAILED", ev, None
    return "INCONCLUSIVE", ev, None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def certify(claim: Claim, spec: Optional[ModelSpec] = None, options: Optional[Options] = None) -> Certificate:
    opts = options or Options()
    t_start = time.time()
    statements = []
    mu = np.asarray(claim.eigenvalues, dtype=float)
    X = None if claim.eigenvectors is None else np.asarray(claim.eigenvectors)
    if X is not None and X.ndim == 1:
        X = X[:, None]

    # ---------------- S1
    H_full_ref, audit = None, None
    Href = None
    if spec is not None and opts.model_fidelity:
        s1, Href_or_full, audit = check_model_fidelity(claim, spec, opts)
        statements.append(s1)
        if claim.sector_isometry is not None:
            H_full_ref = Href_or_full
        else:
            Href = Href_or_full
    else:
        statements.append(_stmt("S1.model_fidelity", "NOT_CHECKED",
                                "no model specification supplied" if spec is None else "disabled"))

    s1_status = statements[0]["status"]
    op_note = None
    is_matrix = lambda o: sp.issparse(o) or isinstance(o, np.ndarray)
    if claim.operator is None:
        if Href is not None:
            op = Href
        elif spec is not None:
            op = SpecOperator(spec, claim.states)
            op_note = "operator applied matrix-free from the spec (no stored matrix)"
        else:
            raise ValueError("no operator: supply claim.operator or a spec")
    elif is_matrix(claim.operator):
        op = claim.operator
    else:
        # black-box producer matvec: bound the spec operator instead, if S1 tied them together
        if spec is not None and s1_status == "CERTIFIED":
            op = Href if Href is not None else SpecOperator(spec, claim.states)
            op_note = ("producer is matrix-free; S2-S4 bounds are computed for the spec operator, "
                       "which S1 found to agree with the producer on random probes")
        else:
            op = claim.operator
            op_note = "black-box operator without a verified spec: rounding in the producer's matvec is not bounded"
    herm_note = None
    if sp.issparse(op) or isinstance(op, np.ndarray):
        opm = sp.csr_matrix(op) if sp.issparse(op) else np.asarray(op)
        asym = opm - opm.conj().T
        dev = float(abs(asym).max()) if sp.issparse(asym) else float(np.abs(asym).max())
        if dev > 0:
            op = (opm + opm.conj().T) / 2
            herm_note = {"max_abs_asymmetry": dev,
                         "note": "producer matrix not exactly Hermitian; statements concern its Hermitian part"}
    adapter = OperatorAdapter(op)
    # model shift: bound on ||H_operator_used - H_spec||_2 (exact spec operator), so that S2/S3
    # statements refer to the declared model; 0 when there is no spec or S1 did not certify it
    model_shift = 0.0
    if spec is not None and statements and statements[0]["status"] == "CERTIFIED":
        model_shift = float(statements[0]["evidence"].get("model_shift_bound", 0.0))
    eps_shift = model_shift
    if audit is not None:
        e = _subspace_shift(audit.isometry_defect, audit.invariance_defect,
                            inf_norm_upper(H_full_ref), adapter.norm_upper())
        eps_shift = float("inf") if e is None else up(e + model_shift)

    # ---------------- S2
    t0 = time.time()
    enc = None
    if X is None:
        statements.append(_stmt("S2.eigenpairs", "NOT_CHECKED",
                                "eigenvalue-only claim: no vectors to bound (see S3 for counts)"))
    else:
        enc = enclose(adapter, mu, X)
        ev = {"n_pairs": int(mu.size), "max_radius": float(enc.radius.max()),
              "median_radius": float(np.median(enc.radius)),
              "groups": [{"members": g.members, "method": g.method, "radius": g.radius,
                          "distinct": g.distinct} for g in enc.groups if g.size > 1],
              "rigorous": enc.rigorous, "sector_lift_shift": eps_shift if audit is not None else None,
              "model_shift": model_shift,
              "radius_refers_to": ("declared model (spec)" if model_shift > 0 or (spec is not None and
                                   statements[0]["status"] == "CERTIFIED") else "producer operator")}
        if not enc.rigorous:
            st, summ = "INCONCLUSIVE", "residuals computed but rounding not boundable for this operator (estimates only)"
        elif not enc.all_distinct:
            st, summ = "INCONCLUSIVE", ("each claimed value has an eigenvalue within its radius, but some "
                                        "claimed pairs could not be matched to distinct eigenvalues "
                                        "(duplicate or non-orthogonal vectors)")
        else:
            st = "CERTIFIED"
            summ = (f"{mu.size} claimed values matched to distinct eigenvalues; max radius "
                    f"{enc.radius.max():.2e}" + (f" (sector block); {enc.radius.max() + eps_shift:.2e} "
                                                 f"for the full operator" if audit is not None else
                                                 (f" (+{model_shift:.1e} model shift)" if model_shift > 0 else "")))
        # accuracy presumption: unless the producer states otherwise, a claimed eigenpair is
        # presumed accurate to DEFAULT_REL_ACCURACY * ||H|| (a true but loose enclosure is FAILED)
        Hn_ = adapter.norm_upper() or 1.0
        if claim.accuracy_target is None:
            target, tsrc = DEFAULT_REL_ACCURACY * max(1.0, Hn_), "default (1e-8 * max(1, ||H||))"
        else:
            target, tsrc = float(claim.accuracy_target), "stated by producer"
        ev["accuracy_target"] = {"value": target, "source": tsrc}
        if st == "CERTIFIED" and enc.radius.max() + eps_shift > target:
            st = "FAILED"
            worst = int(np.argmax(enc.radius))
            summ = (f"certified radius {enc.radius.max() + eps_shift:.2e} (pair {worst}) exceeds the "
                    f"accuracy target {target:.1e} [{tsrc}]")
        statements.append(_stmt("S2.eigenpairs", st, summ, ev,
                                ["standard floating-point model (see 'arithmetic')"] if enc.rigorous else [],
                                time.time() - t0))

    # ---------------- S3
    t0 = time.time()
    window = None
    inertia_op = op if is_matrix(op) else Href
    # S3/S4 concern the declared model when S1 certified the producer's matrix (model shift added);
    # for a symmetry sector they concern the producer's sector block (see compose.py for the lift)
    eps_s3 = model_shift if audit is None else 0.0
    if claim.kind == "all":
        if enc is not None and enc.all_distinct and mu.size == adapter.shape[0]:
            statements.append(_stmt("S3.completeness", "CERTIFIED",
                                    f"{mu.size} distinct certified eigenvalues = dimension: spectrum complete",
                                    {"dimension": int(adapter.shape[0])}, [], time.time() - t0))
            window = (float("-inf"), float("inf"))
        else:
            statements.append(_stmt("S3.completeness", "INCONCLUSIVE",
                                    "full-spectrum claim but pairs not all certified distinct or count != dimension",
                                    {"n_claimed": int(mu.size), "dimension": int(adapter.shape[0])}))
    elif claim.kind in ("lowest", "window"):
        if not opts.inertia:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED", "inertia counting disabled"))
        elif inertia_op is None:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED",
                                    "no stored matrix to factorize (matrix-free producer without spec)"))
        elif inertia_op.shape[0] > opts.inertia_max_dim:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED",
                                    f"factorization skipped: dimension {inertia_op.shape[0]} > inertia_max_dim "
                                    f"{opts.inertia_max_dim}", {"dimension": int(inertia_op.shape[0])}))
        elif enc is None:
            statements.append(_stmt("S3.completeness", "NOT_CHECKED",
                                    "eigenvalue-only claim: completeness needs certified intervals"))
        else:
            if claim.kind == "lowest":
                st, ev, window = _certify_lowest(inertia_op, np.sort(mu), enc, opts, eps_s3)
                if st == "CERTIFIED":
                    summ = (f"exactly {mu.size} eigenvalues below {ev['next_eigenvalue_lower_bound']:.10g}; "
                            f"certified gap to the next >= {ev['gap_lower_bound']:.3e}")
                elif st == "FAILED":
                    summ = (f"at least {ev['missing_at_least'] if 'missing_at_least' in ev else ev['missing_below_top_claim_at_least']}"
                            f" eigenvalue(s) below the highest claimed value are missing from the claim")
                else:
                    summ = ev.get("reason", "inconclusive")
            else:
                a, b = claim.window
                st, ev, window = _certify_window(inertia_op, enc, a, b, opts, eps_s3)
                summ = {"CERTIFIED": f"claim contains every eigenvalue in ({window[0] if window else a:.10g}, "
                                     f"{window[1] if window else b:.10g})" if st == "CERTIFIED" else "",
                        "FAILED": f"at least {ev.get('missing_at_least')} eigenvalue(s) in the window are not claimed",
                        "INCONCLUSIVE": "counts could not be matched to the claimed set"}[st]
            statements.append(_stmt("S3.completeness", st, summ, ev,
                                    ["Sylvester inertia of a computed LDL^T factorization with explicitly "
                                     "bounded backward error; standard floating-point model"],
                                    time.time() - t0))
    else:
        statements.append(_stmt("S3.completeness", "NOT_CHECKED", "claim kind 'subset' makes no completeness claim"))

    # ---------------- Kato-Temple sharpening for isolated pairs (Theorem 3)
    kt = []
    if (enc is not None and window is not None and opts.kato_temple and enc.rigorous
            and enc.all_distinct and audit is None):
        order = np.argsort(enc.mu, kind="stable")
        for pos, i in enumerate(order):
            if enc.groups[enc.group_of[i]].size != 1:
                continue
            j_lo = order[pos - 1] if pos > 0 else None
            j_hi = order[pos + 1] if pos + 1 < len(order) else None
            # isolation interval for the operator in `adapter`; the S3 window refers to the declared
            # model, so it is shrunk by the model shift
            alpha = (enc.mu[j_lo] + enc.radius[j_lo]) if j_lo is not None else window[0] + eps_s3
            beta_ = (enc.mu[j_hi] - enc.radius[j_hi]) if j_hi is not None else window[1] - eps_s3
            if not np.isfinite(beta_):
                continue
            res = kato_temple(adapter, X[:, i], alpha, beta_)
            if res is not None:
                lower, upper, th, r = res
                lower, upper = lower - eps_s3, upper + eps_s3      # refer to the declared model
                if upper - lower < 2 * float(enc.radius[i]):
                    kt.append({"index": int(i), "lower": lower, "upper": upper,
                               "width": upper - lower, "residual_width": 2 * float(enc.radius[i]),
                               "isolation_interval": [float(alpha), float(beta_)]})
    # ---------------- S4 eigenspaces
    t0 = time.time()
    eig_records = []
    if enc is None or opts.eigenspaces == "none":
        statements.append(_stmt("S4.eigenspaces", "NOT_CHECKED", "not requested or no eigenvectors"))
    elif window is None:
        statements.append(_stmt("S4.eigenspaces", "NOT_CHECKED",
                                "requires a certified complete window (S3) to bound separations"))
    else:
        clusters = cluster_groups(enc, opts.cluster_tol)
        groups = [c for c in clusters if (len(c) > 1 or opts.eigenspaces == "all")]
        n_ok = 0
        rad = enc.radius
        for C in groups:
            Cset = set(C)
            a_ = min(enc.mu[i] for i in C)
            b_ = max(enc.mu[i] for i in C)
            rec = {"members": C, "claimed": [float(enc.mu[i]) for i in C], "splitting": float(b_ - a_)}
            cb = _cluster_bound(adapter, enc.mu, X, enc.residual, C)
            if cb is None:
                rec.update(status="INCONCLUSIVE", reason="cluster vectors too far from orthonormal")
                eig_records.append(rec)
                continue
            eps, eta, rho = cb
            JL = min(a_ - eps, min(enc.mu[i] - rad[i] for i in C))
            JU = max(b_ + eps, max(enc.mu[i] + rad[i] for i in C))
            others = [i for i in range(enc.mu.size) if i not in Cset]
            clash = [i for i in others if not (enc.mu[i] + rad[i] < JL or enc.mu[i] - rad[i] > JU)]
            if clash or not (JL > window[0] and JU < window[1]):
                rec.update(status="INCONCLUSIVE",
                           reason="cluster interval overlaps other claimed pairs or the window edge")
                eig_records.append(rec)
                continue
            U_below = max([enc.mu[i] + rad[i] for i in others if enc.mu[i] < a_] + [window[0]])
            L_above = min([enc.mu[i] - rad[i] for i in others if enc.mu[i] > b_] + [window[1]])
            sep = min(a_ - U_below, L_above - b_)
            rec["separation_lower_bound"] = float(sep)
            ec = certify_eigenspace(X, C, enc.mu, rho, eta, sep)
            if ec is None:
                rec.update(status="INCONCLUSIVE", reason="residual not small relative to separation")
                eig_records.append(rec)
                continue
            for name, O in opts.observables.items():
                certify_observable(ec, X, O, name)
                ec.observables[name]["per_vector_spread_under_rotation"] = rotation_spread(X, C, O)
            for name, (sdiag, lattice) in opts.symmetries.items():
                okc = commutes_diagonal(op, sdiag) if is_matrix(op) else False
                certify_quantum_numbers(ec, X, sp.diags(sdiag), name, lattice=lattice,
                                        commutation_certified=okc)
            r = ec.to_dict()
            r["splitting"] = rec["splitting"]
            r["status"] = "CERTIFIED"
            n_ok += 1
            eig_records.append(r)
        if not groups:
            statements.append(_stmt("S4.eigenspaces", "NOT_CHECKED", "no clusters among the claimed pairs",
                                    {}, [], time.time() - t0))
        else:
            st = "CERTIFIED" if n_ok == len(groups) else ("INCONCLUSIVE" if n_ok < len(groups) else "CERTIFIED")
            statements.append(_stmt("S4.eigenspaces", st,
                                    f"{n_ok}/{len(groups)} clusters certified at projector level",
                                    {"clusters": eig_records}, ["Davis-Kahan sin(theta) theorem; Ostrowski's theorem"],
                                    time.time() - t0))

    # ---------------- S5 corroboration
    if opts.corroborate:
        from .corroborate import corroborate
        t0 = time.time()
        cor = corroborate(op, mu, enc, list(opts.corroborate))
        statements.append(_stmt("S5.corroboration", cor["status"], cor["summary"], cor, [], time.time() - t0))

    # ---------------- assemble
    subject = {"description": (spec.name if spec else "operator supplied without spec"),
               "spec": spec.to_dict() if spec else None,
               "spec_sha256": spec.sha256() if spec else None,
               "operator": {"kind": adapter.kind, "dimension": int(adapter.shape[0]),
                            "matrix_sha256": matrix_sha256(op) if is_matrix(op) else None,
                            "norm_upper_bound": adapter.norm_upper()},
               "sector": None if claim.sector_isometry is None else {
                   "isometry_sha256": matrix_sha256(claim.sector_isometry),
                   "dimension": int(claim.sector_isometry.shape[1]),
                   "lift_shift": eps_shift}}
    claim_rec = {"description": claim.description or f"{claim.kind} claim, {mu.size} eigenvalues",
                 "kind": claim.kind, "window": claim.window, "n_eigenvalues": int(mu.size),
                 "has_eigenvectors": X is not None, "producer": claim.producer,
                 "accuracy_target": claim.accuracy_target,
                 "eigenvalues": mu.tolist(),
                 "eigenvectors_sha256": array_sha256(X) if X is not None else None}
    not_verified = [
        "that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)",
        "correctness of the producer's code in general (only this result is checked)",
    ]
    if enc is not None and enc.rigorous:
        not_verified.append("interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')")
    if claim.kind == "subset":
        not_verified.append("completeness: no claim that other eigenvalues are absent")
    elif window is not None:
        not_verified.append(f"anything outside the certified window {list(window)}")
    if opts.eigenspaces == "clusters":
        not_verified.append("eigenvector subspaces of isolated (non-cluster) pairs beyond their residuals")
    body = {"subject": subject, "claim": claim_rec, "statements": statements,
            "enclosures": enc.to_records() if enc is not None else [],
            "kato_temple": kt,
            "arithmetic": float_env_record(), "environment": environment_record(),
            "hermitian_part": herm_note,
            "operator_note": op_note,
            "scope": ("sector block: S2-S4 concern the producer's sector matrix; S2 radii in 'arrays' "
                      "include the certified shift to the full operator") if audit is not None else "operator",
            "not_verified": not_verified,
            "timing": {"total_seconds": time.time() - t_start}}
    arrays = {"eigenvalues": mu}
    if enc is not None:
        arrays["radius"] = enc.radius + (eps_shift if np.isfinite(eps_shift) else 0.0)
        arrays["residual_bound"] = enc.residual.upper
    if opts.store_eigenvectors and X is not None:
        arrays["eigenvectors"] = X
    return Certificate(body, arrays)
