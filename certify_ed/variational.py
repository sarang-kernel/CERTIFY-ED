"""Variational certificates for states that are not exact-diagonalisation eigenvectors (E-05, 3.2).

A trial state psi (a dense vector, a matrix product state contracted to a vector, or a sparse CI vector) and a
Hermitian operator H (a stored matrix, or the operator built from a declared ModelSpec) give three statements
about the lowest eigenvalue lambda_1 of H and one declaration (docs/proofs/E-05_variational.md):

  V1.rayleigh_upper   lambda_1 <= theta = <psi, H psi> / <psi, psi>, unconditionally (Rayleigh-Ritz), with the
                      floating-point accounting of the quotient (upper bound on the exact quotient of the stored
                      vector) and the model shift sigma when the operator is the one built from a spec
  V2.existence        some eigenvalue lies in [theta - r, theta + r], r >= ||H psi - theta psi|| / ||psi|| (Theorem 1)
  V3.temple_lower     lambda_1 >= theta - r^2 / (rho - theta) for any rho with theta < rho <= lambda_2 (Temple 1928;
                      Kato 1949). rho comes either from a DECLARED gap hypothesis (status
                      CERTIFIED_UNDER_HYPOTHESIS; the hypothesis is named in the record and in the statement
                      text) or from the S3 statement of a certificate of an ED run for the same spec (then no
                      hypothesis: status CERTIFIED)
  V4.declared_estimates  stochastic estimates handed over by the producer (energy, variance from samples) are
                      recorded as DECLARED with reason code DECLARED_ESTIMATE; nothing rests on them

Status vocabulary of these records (distinct from the certificate statuses so that a hypothesis is never read
as a certification): CERTIFIED, CERTIFIED_UNDER_HYPOTHESIS, DECLARED, INCONCLUSIVE, NOT_CHECKED, FAILED (the last
only for refuted consistency statements, C-07).

The statements concern the operator as stored (a matrix) or, for a spec, the operator the spec defines in exact
arithmetic: the floating-point build differs from it by at most the model shift sigma (certify._builder_shift,
docs/proofs/appA_model-shift.md), which is added to the upper bound and to the radius. The standard model of
floating-point arithmetic is assumed with the no-underflow hypothesis (entries of psi and of the operator at or
above 2^-900 in magnitude, or exactly zero); a finding makes the statements INCONCLUSIVE (UNDERFLOW_RISK)
unless underflow_check=False, which is recorded.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import _jsonable, array_sha256, matrix_sha256
from .enclosure import OperatorAdapter, rayleigh_interval
from .fp import U, ULP_MIN, UNDERFLOW_THRESHOLD, add_up, gamma, modulus_up, norm2_upper, sub_down, underflow_scan, up
from .precision import binary64, summed, widening_record
from .spec import ModelSpec, canonical_json

VARIATIONAL_SCHEMA = "certify-ed/variational/1.0"
VARIATIONAL_STATUSES = ("CERTIFIED", "CERTIFIED_UNDER_HYPOTHESIS", "DECLARED", "INCONCLUSIVE", "NOT_CHECKED", "FAILED")
VARIATIONAL_STATEMENT_IDS = ("V1.rayleigh_upper", "V2.existence", "V3.temple_lower", "V4.declared_estimates")
MPS_MAX_DIM = 2 ** 22

STANDARD_MODEL = ("IEEE-754 binary64 round-to-nearest, standard model fl(a op b) = (a op b)(1 + delta), |delta| <= u, "
                  "no-underflow hypothesis (nonzero operands >= 2^-900)")

__all__ = ["VARIATIONAL_SCHEMA", "VARIATIONAL_STATUSES", "VARIATIONAL_STATEMENT_IDS", "MPS_MAX_DIM",
           "TrialState", "state_from_dense", "state_from_mps", "state_from_sparse_ci", "gap_from_certificate",
           "certify_variational", "format_variational"]


# ------------------------------------------------------------------------------------------------ adapters

@dataclass
class TrialState:
    """A trial vector with the record of how it was obtained. `vector` is None when the adapter refused (the
    record then carries `reason_code`); `contraction_error` bounds ||vector - psi_source||_2 where the source
    (an MPS) was contracted in floating point (0 for a dense or sparse-CI vector, which are taken as stored)."""
    vector: Optional[np.ndarray]
    form: str                                   # 'dense' | 'mps' | 'sparse_ci'
    record: Dict[str, Any] = field(default_factory=dict)
    contraction_error: float = 0.0
    reason_code: Optional[str] = None
    reason: str = ""


def state_from_dense(psi, widened: Optional[list] = None) -> TrialState:
    """The stored vector psi in binary64 (precision.binary64): a narrower floating format (float16, float32, complex64;
    noted in `widened`), integers and exact object values (a Fraction 1/2) are converted exactly; a wider format and
    values binary64 cannot hold (an integer beyond 2^53, a Fraction 1/3; 3.x rounded them) are refused."""
    v = binary64(np.asarray(psi), "trial state", widened)
    if v.ndim == 2 and 1 in v.shape:
        v = v.ravel()
    if v.ndim != 1:
        raise ValueError(f"a dense trial state must be a vector, got shape {v.shape}")
    return TrialState(v, "dense", {"form": "dense", "dimension": int(v.size), "complex": bool(np.iscomplexobj(v))})


def state_from_sparse_ci(indices, amplitudes, n: int, widened: Optional[list] = None) -> TrialState:
    """psi[indices[k]] = amplitudes[k], zero elsewhere (exact: a scatter, no arithmetic). Duplicate indices are
    refused (their sum would round). Amplitudes in a narrower format are converted exactly to binary64
    (precision.binary64), a wider format is refused."""
    idx = np.asarray(indices, dtype=np.int64).ravel()
    amp = binary64(np.asarray(amplitudes).ravel(), "sparse CI amplitudes", widened)
    n = int(n)
    if idx.size != amp.size:
        raise ValueError(f"indices ({idx.size}) and amplitudes ({amp.size}) must have the same length")
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"indices must lie in [0, {n}), got [{idx.min()}, {idx.max()}]")
    if np.unique(idx).size != idx.size:
        raise ValueError("duplicate indices in the sparse CI vector: the amplitudes would have to be summed (rounding)")
    v = np.zeros(n, dtype=complex if np.iscomplexobj(amp) else float)
    v[idx] = amp
    return TrialState(v, "sparse_ci", {"form": "sparse_ci", "dimension": n, "nnz": int(idx.size),
                                       "complex": bool(np.iscomplexobj(amp))})


def state_from_mps(tensors: Sequence[Any], max_dim: int = MPS_MAX_DIM, widened: Optional[list] = None) -> TrialState:
    """Contract an open-boundary MPS (tensors A^(j) of shape (D_{j-1}, d_j, D_j), D_0 = D_L = 1; a 2-D first or
    last tensor is accepted) to the dense vector psi[s_1 ... s_L] = A^(1)[s_1] ... A^(L)[s_L] (site 1 is the most
    significant index). The contraction runs left to right in floating point; the entrywise bound
    |psi_hat - psi| <= c_L |A^(1)| ... |A^(L)| with c_L = prod_j (1 + g_j) - 1, g_j = gamma_dot(D_{j-1}) the inner-product
    constant of step j (docs/proofs/E-05_variational.md, Lemma 4), gives `contraction_error` >= ||psi_hat - psi||_2.
    Refused (vector None, reason_code SIZE_LIMIT) when prod d_j > max_dim. Tensors in a narrower format are converted
    exactly to binary64 first (precision.binary64): the contraction bound counts binary64 arithmetic."""
    if len(tensors) == 0:
        raise ValueError("an MPS needs at least one tensor")
    As = []
    for j, A in enumerate(tensors):
        A = binary64(np.asarray(A), f"MPS tensor {j}", widened)
        if A.ndim == 2:
            A = A[None, :, :] if j == 0 else A[:, :, None]
        if A.ndim != 3:
            raise ValueError(f"tensor {j} must have shape (D_left, d, D_right), got {A.shape}")
        As.append(A)
    if As[0].shape[0] != 1 or As[-1].shape[2] != 1:
        raise ValueError("open-boundary MPS: the first tensor needs D_left = 1 and the last D_right = 1")
    for j in range(len(As) - 1):
        if As[j].shape[2] != As[j + 1].shape[0]:
            raise ValueError(f"bond dimension mismatch between tensors {j} and {j + 1}: {As[j].shape[2]} vs {As[j + 1].shape[0]}")
    dims = [int(A.shape[1]) for A in As]
    n = 1
    for d in dims:
        n *= d
    cplx = any(np.iscomplexobj(A) for A in As)
    rec = {"form": "mps", "n_tensors": len(As), "physical_dims": dims, "bond_dims": [int(A.shape[2]) for A in As[:-1]],
           "dimension": int(n), "complex": cplx}
    if n > max_dim:
        return TrialState(None, "mps", rec, reason_code="SIZE_LIMIT",
                          reason=f"the contracted vector has dimension {n} > max_dim = {max_dim}")
    T = As[0][0].astype(complex if cplx else float)            # (d_1, D_1)
    P = modulus_up(As[0][0]).astype(float)                       # |A^(1)| rounded up (fp.modulus_up), same shape
    m_total = 0                                                 # M = sum_j m_j, c_L = prod (1 + g_j) - 1 <= gamma_M
    for A in As[1:]:
        D = int(A.shape[0])
        T = np.tensordot(T, A, axes=([1], [0])).reshape(-1, A.shape[2])
        P = np.tensordot(P, modulus_up(A).astype(float), axes=([1], [0])).reshape(-1, A.shape[2])
        m_total += (2 * (D + 4)) if cplx else D                  # 1 + 2 gamma_{D+4} <= 1 + gamma_{2(D+4)}
    psi = T[:, 0]
    # |psi_hat - psi| <= c_L |A^(1)|...|A^(L)| <= gamma_M |A^(1)|...|A^(L)| entrywise (Higham Lemma 3.3); the computed
    # absolute product underestimates the exact one by at most a relative gamma_M (non-negative inner products),
    # undone by the division; the guard of up() covers the roundings of gamma_M, 1 - gamma_M, the division and the product
    gM = gamma(max(m_total, 1))
    err_entry = up(gM) * (P[:, 0] / (1 - gM))
    e = float(norm2_upper(err_entry)) if len(As) > 1 else 0.0
    rec["contraction_constant"] = float(up(gM)) if len(As) > 1 else 0.0
    return TrialState(psi, "mps", rec, contraction_error=e)


def _as_state(psi, widened: Optional[list] = None) -> TrialState:
    if isinstance(psi, TrialState):
        return psi
    if isinstance(psi, dict):
        if "mps" in psi:
            return state_from_mps(psi["mps"], psi.get("max_dim", MPS_MAX_DIM), widened)
        if "indices" in psi:
            return state_from_sparse_ci(psi["indices"], psi["amplitudes"], psi["n"], widened)
        raise ValueError("a dict trial state needs either {'mps': [tensors]} or {'indices', 'amplitudes', 'n'}")
    if isinstance(psi, (list, tuple)) and len(psi) and np.ndim(psi[0]) >= 2:
        return state_from_mps(psi, widened=widened)
    return state_from_dense(psi, widened)


# ------------------------------------------------------------------------------------------------ gap sources

def gap_from_certificate(cert, spec: Optional[ModelSpec], n: int) -> Dict[str, Any]:
    """A certified lower bound on lambda_2 from a certificate of a 'lowest' claim on the same operator: the second
    smallest lower endpoint of the S2 enclosures when k >= 2 (two distinct eigenvalues lie in distinct enclosures,
    so lambda_2 is at least the second smallest lower endpoint; Corollary 5), or S3's next_eigenvalue_lower_bound
    when k = 1. Needs S2 and S3 CERTIFIED, the same spec hash (when a spec is declared) and the same dimension.
    Returns {'ok': bool, 'rho': float | None, 'reason': str, ...}."""
    body = cert.body
    subj = body.get("subject", {})
    out = {"ok": False, "rho": None, "certificate_sha256": None, "reason": ""}
    try:
        from .crossings import certificate_hash
        out["certificate_sha256"] = certificate_hash(cert)
    except Exception:
        pass
    if spec is not None and subj.get("spec_sha256") != spec.sha256():
        out["reason"] = "the certificate's spec hash differs from the declared spec"
        return out
    dim = (subj.get("operator") or {}).get("dimension")
    if dim is not None and int(dim) != int(n):
        out["reason"] = f"the certificate's operator has dimension {dim}, the trial state {n}"
        return out
    if body.get("claim", {}).get("kind") != "lowest":
        out["reason"] = "the certificate must be about a 'lowest' claim (its S3 certifies the ranks)"
        return out
    if cert.status_of("S2.eigenpairs") != "CERTIFIED" or cert.status_of("S3.completeness") != "CERTIFIED":
        out["reason"] = (f"S2 is {cert.status_of('S2.eigenpairs')} and S3 is {cert.status_of('S3.completeness')}; both must "
                         "be CERTIFIED")
        return out
    mu = np.asarray(cert.arrays.get("eigenvalues"), dtype=float)
    rad = np.asarray(cert.arrays.get("radius"), dtype=float)
    if mu is None or rad is None or mu.size == 0 or not np.all(np.isfinite(rad)):
        out["reason"] = "the certificate has no finite radii"
        return out
    if mu.size >= 2:
        lows = np.sort(sub_down(mu, rad))
        rho = float(lows[1])
        src = "second smallest lower endpoint of the certified enclosures (k >= 2; Corollary 5 pairing)"
    else:
        ev = (cert.statement("S3.completeness") or {}).get("evidence") or {}
        nlb = ev.get("next_eigenvalue_lower_bound")
        if nlb is None or not math.isfinite(float(nlb)):
            out["reason"] = "the certificate (k = 1) carries no next_eigenvalue_lower_bound"
            return out
        rho = float(nlb)
        src = "S3 next_eigenvalue_lower_bound (k = 1)"
    out.update(ok=True, rho=rho, source=src, k=int(mu.size),
               refers_to=((cert.statement("S2.eigenpairs") or {}).get("evidence") or {}).get("radius_refers_to"))
    return out


# ------------------------------------------------------------------------------------------------ statements

def _vstmt(sid: str, status: str, summary: str, evidence: Optional[dict] = None, hypotheses: Optional[list] = None,
           assumptions: Optional[list] = None, reason_code: Optional[str] = None) -> dict:
    assert status in VARIATIONAL_STATUSES, status
    ev = dict(evidence or {})
    if reason_code is not None:
        ev["reason_code"] = reason_code
    if status == "CERTIFIED_UNDER_HYPOTHESIS":
        assert hypotheses, "a statement under hypothesis must name it"
    return {"id": sid, "status": status, "summary": summary, "evidence": ev,
            "hypotheses": list(hypotheses or []), "assumptions": list(assumptions or [])}


def _record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if not k.startswith("_") and k != "record_sha256"}
    return hashlib.sha256(canonical_json(_jsonable(body)).encode()).hexdigest()


def certify_variational(operator=None, psi=None, spec: Optional[ModelSpec] = None, states=None, *,
                        gap_hypothesis: Optional[float] = None, gap_source: str = "",
                        gap_certificate=None, declared: Optional[Dict[str, Any]] = None,
                        underflow_check: bool = True, seed: int = 12345, description: str = "") -> Dict[str, Any]:
    """Variational statements about the lowest eigenvalue of H for a trial state psi (E-05).

    operator: a dense or scipy.sparse Hermitian matrix (the statements then concern the stored matrix), or None
      with `spec` given (the operator is built from the spec on `states`, model shift sigma recorded and added).
    psi: a dense vector, a TrialState, {'mps': [tensors]} (contracted; SIZE_LIMIT above 2^22), or
      {'indices': ..., 'amplitudes': ..., 'n': ...} (sparse CI).
    gap_hypothesis: a DECLARED rho with rho <= lambda_2, with `gap_source` the producer's text (where the value comes
      from); V3 is then CERTIFIED_UNDER_HYPOTHESIS 'declared_gap'. gap_certificate: a Certificate of a 'lowest'
      claim for the same operator (S2, S3 CERTIFIED) whose enclosures give a certified rho; V3 is then CERTIFIED.
      Both given: the certified value is used and the declared one recorded.
    declared: producer's stochastic estimates, e.g. {'energy': E, 'variance': v, 'samples': N, 'method': 'VMC'}:
      recorded as DECLARED (DECLARED_ESTIMATE), never certified.
    Returns a JSON-able record (schema VARIATIONAL_SCHEMA) with `statements` and `record_sha256`."""
    from .certify import _builder_shift, _private_copies
    from .construct import SpecOperator
    statements: List[dict] = []
    rec: Dict[str, Any] = {"schema": VARIATIONAL_SCHEMA, "description": description or "variational trial state"}
    assume = [STANDARD_MODEL]
    widened: List[dict] = []         # inputs converted exactly to binary64 (precision.py), recorded under input_widening

    # operator
    sigma = 0.0
    if operator is None:
        if spec is None:
            raise ValueError("give an operator (matrix) or a spec")
        op = SpecOperator(spec, states)
        adapter = OperatorAdapter(op)
        sigma = float(_builder_shift(spec, states))
        refers_to = "declared model (spec); the floating-point build differs by at most the model shift, added to the bounds"
        op_rec = {"kind": "spec", "dimension": int(adapter.shape[0]), "spec_sha256": spec.sha256(),
                  "spec_name": spec.name, "model_shift": sigma,
                  "block": None if states is None else {"label": getattr(states, "label", ""), "dimension": int(states.size)}}
    else:
        if not (sp.issparse(operator) or isinstance(operator, np.ndarray)):
            raise ValueError("the operator must be a stored matrix (dense ndarray or scipy.sparse); matrix-free "
                             "operators have no rounding bound (docs/SCOPE_MATRIX.md)")
        # a narrower format becomes an exact binary64 copy (precision.py), before tocsr, which would sum repeated
        # entries in the narrower format
        Aw = binary64(operator if sp.issparse(operator) else np.asarray(operator), "operator", widened)
        Aw = summed(Aw, "operator")          # repeated entries summed on a copy, before any product
        A = Aw.tocsr() if sp.issparse(Aw) else Aw
        if A is operator:
            A = _private_copies()(A)       # the adapter sums duplicates in place: the caller's matrix stays as given
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"the operator must be square, got shape {A.shape}")
        D = (A - A.conj().T)
        if sp.issparse(D):
            herm = float(abs(D).max()) if D.nnz else 0.0
        else:
            herm = float(np.max(np.abs(D))) if D.size else 0.0
        if herm != 0.0:
            raise ValueError(f"the stored operator is not Hermitian (max |A - A^H| = {herm:.3e}); symmetrise it first "
                             "(certify_ed.construct.hermitian_part) -- the statements assume a Hermitian operator")
        adapter = OperatorAdapter(A)
        refers_to = "the stored matrix (no model fidelity statement here: run certify-ed certify for S1)"
        op_rec = {"kind": "matrix", "dimension": int(A.shape[0]), "matrix_sha256": matrix_sha256(A),
                  "spec_sha256": None if spec is None else spec.sha256(), "model_shift": 0.0}
    op_rec["norm_upper_bound"] = adapter.norm_upper()
    op_rec["refers_to"] = refers_to
    rec["operator"] = op_rec
    n = int(adapter.shape[0])

    # state
    st = _as_state(psi, widened) if psi is not None else None
    if st is not None and st.vector is not None:
        vec = binary64(st.vector, "trial state", widened)      # a TrialState given as such
        if vec is not st.vector:
            st = replace(st, vector=vec)
    if st is not None and st.vector is not None and st.vector.size != n:
        raise ValueError(f"the trial state has dimension {st.vector.size}, the operator {n}")
    state_rec = dict(st.record) if st is not None else {"form": None}
    if st is not None:
        state_rec["contraction_error"] = float(st.contraction_error)
        if st.vector is not None:
            state_rec["vector_sha256"] = array_sha256(st.vector)
    rec["state"] = state_rec
    rec["gap_hypothesis"] = None if gap_hypothesis is None else {"rho": float(gap_hypothesis), "source": str(gap_source)}
    widening = widening_record(widened)
    if widening is not None:                 # only when an input was converted: binary64 inputs keep their record
        rec["input_widening"] = widening

    def all_not_checked(code, why):
        for sid in VARIATIONAL_STATEMENT_IDS[:3]:
            statements.append(_vstmt(sid, "NOT_CHECKED", why, {}, reason_code=code, assumptions=assume))

    if st is None or st.vector is None:
        if st is None:
            all_not_checked("NEEDS_EIGENVECTORS", "no trial state was given (declared estimates only)")
        else:
            all_not_checked(st.reason_code or "NOT_APPLICABLE", st.reason)
    else:
        x = st.vector
        if not np.all(np.isfinite(x)):
            all_not_checked("NOT_APPLICABLE", "the trial state has non-finite entries")
        else:
            # no-underflow hypothesis (standard-model paths: Rayleigh quotient, Temple arithmetic)
            findings = []
            if underflow_check:
                parts = [("state.entries", x, UNDERFLOW_THRESHOLD)]
                if adapter.kind == "matrix":
                    entries = adapter.A.data[:adapter.A.nnz] if sp.issparse(adapter.A) else adapter.A    # stored values
                    parts.append(("operator.entries", entries, UNDERFLOW_THRESHOLD))
                else:
                    parts.append(("operator.norm_bound", np.array([adapter.norm_upper()], dtype=float), UNDERFLOW_THRESHOLD))
                    if sigma > 0:
                        parts.append(("operator.model_shift", np.array([sigma]), UNDERFLOW_THRESHOLD))
                findings = underflow_scan(parts)
            rec["underflow_check"] = {"enabled": bool(underflow_check), "findings": findings}
            ri = None if findings else rayleigh_interval(adapter, x)
            if findings:
                from .certify import _underflow_statement_note
                note = _underflow_statement_note(findings)
                for sid in VARIATIONAL_STATEMENT_IDS[:3]:
                    statements.append(_vstmt(sid, "INCONCLUSIVE", note, {"findings": findings},
                                             reason_code="UNDERFLOW_RISK", assumptions=assume))
            elif ri is None:
                all_not_checked("NOT_APPLICABLE", "the norm of the trial state is too small to resolve (||psi||^2 - its error "
                                "bound <= 0); rescale the state")
            else:
                th_lo, th_hi, th, r_hat = ri
                # r >= ||H x - theta x|| / ||x|| for the exact theta: r_hat is the Theorem 1 radius at theta_hat and
                # |theta - theta_hat| <= max(th_hi - th, th - th_lo) (kato_temple uses the same step); + sigma for the spec
                r_stored = float(up(r_hat + max(th_hi - th, th - th_lo)))
                r = float(add_up(r_stored, sigma)) if sigma > 0 else r_stored
                upper = float(add_up(th_hi, sigma)) if sigma > 0 else float(th_hi)
                upper = float(np.nextafter(upper, np.inf))
                theta_rec = {"computed": float(th), "lower": float(th_lo), "upper": float(th_hi),
                             "residual_radius_stored": r_stored, "residual_radius": r, "model_shift": sigma}
                for key in ("computed", "lower", "upper"):        # no nan in the record: a value that is not finite
                    if not math.isfinite(theta_rec[key]):         # is stored as null, a radius as inf
                        theta_rec[key] = None
                for key in ("residual_radius_stored", "residual_radius"):
                    if not math.isfinite(theta_rec[key]):
                        theta_rec[key] = math.inf
                rec["theta"] = theta_rec
                if st.contraction_error > 0:
                    # Lemma 5: |theta(psi) - theta(psi_hat)| <= (2 r ||psi_hat|| e + (||H|| + |theta|) e^2) / (||psi_hat|| - e)^2
                    e = st.contraction_error
                    xn = float(norm2_upper(x))
                    from .fp import norm2_lower
                    xl = float(norm2_lower(x))
                    Hn = adapter.norm_upper() or float("inf")
                    if xl > e:
                        dth = up((2 * r_stored * xn * e + (Hn + max(abs(th_lo), abs(th_hi))) * e * e) / ((xl - e) ** 2))
                    else:
                        dth = float("inf")
                    rec["state"]["theta_source_vs_contracted"] = float(dth) if math.isfinite(dth) else math.inf
                    rec["state"]["note"] = ("the statements are about the contracted vector psi_hat as stored; the source "
                                            "MPS psi has ||psi_hat - psi|| <= contraction_error and a Rayleigh quotient "
                                            "within theta_source_vs_contracted of theta(psi_hat) (Lemma 5)")
                if not all(math.isfinite(v) for v in (th_lo, th_hi, r, upper)):
                    # a bound of the accounting is inf or nan (a product, a sum or a modulus overflowed; inf * 0 =
                    # nan in a product): no statement rests on it
                    note = ("the floating-point accounting of theta is not finite (a bound of the accounting "
                            "overflowed): no bound is stated")
                    ev_nf = {"theta": theta_rec, "refers_to": refers_to}
                    for sid in VARIATIONAL_STATEMENT_IDS[:3]:
                        statements.append(_vstmt(sid, "INCONCLUSIVE", note, ev_nf, assumptions=assume,
                                                 reason_code="NONFINITE_BOUND"))
                else:
                    statements.append(_vstmt("V1.rayleigh_upper", "CERTIFIED",
                                             f"lambda_1 <= {upper:.17g} (Rayleigh quotient of the trial state, upper endpoint"
                                             + (f" + model shift {sigma:.3e})" if sigma > 0 else ")"),
                                             {"upper_bound": upper, "theta": theta_rec, "refers_to": refers_to},
                                             assumptions=assume))
                    lo_ex, hi_ex = float(sub_down(th_lo, r)), float(add_up(th_hi, r))
                    statements.append(_vstmt("V2.existence", "CERTIFIED",
                                             f"some eigenvalue lies in [{lo_ex:.17g}, {hi_ex:.17g}] (Theorem 1 at theta, radius {r:.3e})",
                                             {"interval": [lo_ex, hi_ex], "radius": r, "refers_to": refers_to},
                                             assumptions=assume))
                    statements.append(_temple(th_lo, th_hi, r, sigma, gap_hypothesis, gap_source, gap_certificate, spec, n,
                                              assume, refers_to))
    # V4: declared estimates
    if declared:
        d = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and k != "samples" else v)
             for k, v in dict(declared).items()}
        statements.append(_vstmt("V4.declared_estimates", "DECLARED",
                                 "producer-declared estimates (" + ", ".join(f"{k} = {v}" for k, v in d.items()) +
                                 "): recorded as declared, not certified; no statement above rests on them",
                                 {"declared": d, "reason": "stochastic estimates (energy, variance from samples) have no "
                                  "floating-point or sampling error bound that this tool can check"},
                                 reason_code="DECLARED_ESTIMATE"))
    else:
        statements.append(_vstmt("V4.declared_estimates", "NOT_CHECKED", "no producer estimates were declared", {},
                                 reason_code="NOT_APPLICABLE"))
    from .diagnostics import annotate_record
    rec["statements"] = statements
    annotate_record(rec)
    rec["basis"] = ("Rayleigh-Ritz (V1), Theorem 1 (V2), Temple's inequality (V3); floating-point accounting of theta "
                    "and r as in docs/proofs/E-05_variational.md")
    rec["record_sha256"] = _record_hash(rec)
    return rec


def _temple(th_lo: float, th_hi: float, r: float, sigma: float, gap_hypothesis, gap_source: str, gap_certificate,
            spec, n: int, assume: list, refers_to: str) -> dict:
    """V3: lambda_1 >= theta - r^2 / (rho - theta) with the hypothesis theta < rho <= lambda_2 (Temple 1928, Kato 1949).

    With the operator built from a spec (model shift sigma): the hypothesis rho <= lambda_2(H_spec) gives
    rho - sigma <= lambda_2(H_ref) (Weyl), Temple applies to H_ref with rho' = rho - sigma and the Rayleigh data of
    H_ref (th_lo, th_hi, the stored radius), and lambda_1(H_spec) >= lambda_1(H_ref) - sigma. `r` here already contains
    sigma (it is the radius of V2), which only enlarges r^2."""
    hyps = []
    ev: Dict[str, Any] = {"refers_to": refers_to}
    rho = None
    certified = False
    if gap_certificate is not None:
        g = gap_from_certificate(gap_certificate, spec, n)
        ev["gap_certificate"] = {k: v for k, v in g.items()}
        if g["ok"]:
            rho = float(g["rho"])
            certified = True
    if gap_hypothesis is not None:
        ev["declared_gap"] = {"rho": float(gap_hypothesis), "source": str(gap_source)}
        if rho is None:
            rho = float(gap_hypothesis)
            hyps.append({"name": "declared_gap", "statement": f"lambda_2 >= {rho!r} (the second eigenvalue of the operator is "
                         "at least rho; not checked by this tool)", "value": rho, "source": str(gap_source) or "(no source given)"})
    if rho is None:
        why = "no lower bound on lambda_2: declare one (gap_hypothesis, with its source) or supply a certificate of an ED run "
        why += "for the same operator (gap_certificate)"
        if gap_certificate is not None:
            why += f"; the supplied certificate was not usable: {ev['gap_certificate'].get('reason')}"
        return _vstmt("V3.temple_lower", "NOT_CHECKED", why, ev, reason_code="NEEDS_DEPENDENCY", assumptions=assume)
    rho_eff = float(sub_down(rho, sigma)) if sigma > 0 else float(rho)
    ev["rho"] = rho
    ev["rho_effective"] = rho_eff
    if not th_hi < rho_eff:
        return _vstmt("V3.temple_lower", "INCONCLUSIVE",
                      f"Temple's inequality needs theta < rho: theta <= {th_hi:.17g} is not below rho" +
                      (f" - sigma = {rho_eff:.17g}" if sigma > 0 else f" = {rho_eff:.17g}") +
                      " (the trial state is not close enough to the ground state for this gap)",
                      ev, hypotheses=hyps, reason_code="NOT_APPLICABLE", assumptions=assume)
    r2 = np.nextafter(r * r, np.inf) + ULP_MIN
    corr = np.nextafter(r2 / sub_down(rho_eff, th_hi), np.inf)
    lower = float(sub_down(th_lo, corr))
    if sigma > 0:
        lower = float(sub_down(lower, sigma))
    lower -= abs(lower) * 8 * U
    ev.update(lower_bound=lower, correction=float(corr), r=r, theta_lower=th_lo, theta_upper=th_hi,
              source="certificate" if certified else "declared hypothesis")
    status = "CERTIFIED" if certified else "CERTIFIED_UNDER_HYPOTHESIS"
    text = f"lambda_1 >= {lower:.17g} (Temple: theta - r^2/(rho - theta) with rho = {rho!r}"
    if certified:
        text += ", a certified lower bound on lambda_2 from the supplied certificate's S2/S3)"
    else:
        text += (f"; UNDER THE DECLARED HYPOTHESIS lambda_2 >= {rho!r}, source: {gap_source or '(no source given)'}; "
                 "the bound is false if the hypothesis is)")
    return _vstmt("V3.temple_lower", status, text, ev, hypotheses=hyps, assumptions=assume)


def format_variational(rec: dict) -> str:
    lines = [f"variational record ({rec['schema']}): {rec.get('description', '')}",
             f"  operator: {rec['operator']['kind']}, dimension {rec['operator']['dimension']}, model shift {rec['operator'].get('model_shift', 0.0):.3e}",
             f"  state: {rec['state'].get('form')}" + (f", contraction error {rec['state']['contraction_error']:.3e}"
                                                        if rec['state'].get('contraction_error') else "")]
    for s in rec["statements"]:
        lines.append(f"  {s['id']:22s} {s['status']:27s} {s['summary']}")
        for h in s.get("hypotheses", []):
            lines.append(f"      hypothesis {h['name']}: {h['statement']}; source: {h['source']}")
    return "\n".join(lines)
