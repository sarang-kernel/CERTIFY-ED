"""Generalized Hermitian eigenproblems A x = lambda B x with B positive definite (E-03, 3.2; dense path).

With B = L L^H (Cholesky) the pencil is similar to the Hermitian matrix C = L^{-1} A L^{-H}: the eigenvalues of
the pencil are those of C, with eigenvectors y = L^H x. Three statements (docs/proofs/E-03_generalized.md):

  G1.B_positive_definite  lambda_min(B) >= beta_c > 0, certified by the inertia of B - beta I (Bunch-Kaufman LDL^H
                          with its backward error, inertia.inertia_dense) -- or DECLARED by the user when the
                          inertia is inconclusive (hypothesis `declared_lambda_min_B`, status
                          CERTIFIED_UNDER_HYPOTHESIS for everything that rests on it). A B whose inertia at 0 shows a
                          negative eigenvalue is refused (UNSUPPORTED_MODEL): nothing is certified.
  G2.eigenpairs           for each claimed pair (x, mu): some eigenvalue of the pencil satisfies
                          |lambda - mu| <= r = ||A x - mu B x||_{B^-1} / ||x||_B <= ||A x - mu B x||_2 / (sqrt(beta_c) ||x||_B)
                          (Theorem 1 for C, ||L^{-1}||_2 = lambda_min(B)^{-1/2}), with the standard-model accounting of
                          the residual and of x^H B x
  G3.completeness         for a 'lowest' claim: the inertia nu of M = A - sigma B at a shift sigma above the enclosures
                          counts the pencil eigenvalues below sigma (Sylvester's law for the congruence
                          A - sigma B = L (C - sigma I) L^H), with the collar b / beta_c for the backward error b of the
                          factorization plus the rounding of forming M; nu = k with pairwise disjoint enclosures makes
                          the k enclosures the k lowest eigenvalues and gives lambda_{k+1} >= sigma - b/beta_c

The statements concern the stored matrices A and B. Dense path only (n <= max_dim, default 4000). Cluster handling
(overlapping enclosures) is not implemented: such a claim keeps its existence radii and G3 is INCONCLUSIVE.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import _jsonable, array_sha256, matrix_sha256
from .fp import U, abs_up, add_up, exact_dot, gamma, gamma_dot, inf_norm_upper, modulus_up, norm2_upper, sub_down, up
from .inertia import inertia_dense
from .precision import binary64, check_values, claimed_reals, widening_record
from .spec import canonical_json

GENERALIZED_SCHEMA = "certify-ed/generalized/1.0"
GENERALIZED_STATEMENT_IDS = ("G1.B_positive_definite", "G2.eigenpairs", "G3.completeness")
DEFAULT_MAX_DIM = 4000
STANDARD_MODEL = ("IEEE-754 binary64 round-to-nearest, standard model fl(a op b) = (a op b)(1 + delta), |delta| <= u, "
                  "no-underflow hypothesis (nonzero operands >= 2^-900); LAPACK Bunch-Kaufman factorization with the "
                  "explicitly formed residual as backward error (inertia.inertia_dense)")

__all__ = ["GENERALIZED_SCHEMA", "GENERALIZED_STATEMENT_IDS", "DEFAULT_MAX_DIM", "certify_generalized",
           "certify_B_positive_definite", "pencil_residual_radii", "pencil_count_below", "format_generalized"]


def _dense_hermitian(M, name: str, widened: Optional[list] = None) -> np.ndarray:
    # exact for narrower formats (noted), a wider one refused (precision.py); before toarray, which sums repeated
    # entries in the input's precision
    M = binary64(M if sp.issparse(M) else np.asarray(M), name, widened)
    M = M.toarray() if sp.issparse(M) else M
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} has non-finite entries")
    D = M - M.conj().T
    if M.size and float(np.max(np.abs(D))) != 0.0:
        raise ValueError(f"{name} is not Hermitian (max |{name} - {name}^H| = {float(np.max(np.abs(D))):.3e}); the statements "
                         "assume Hermitian A and B")
    if not (np.issubdtype(M.dtype, np.floating) or np.issubdtype(M.dtype, np.complexfloating)):
        M = M.astype(complex if np.iscomplexobj(M) else float)
    return M


def _vstmt(sid, status, summary, evidence=None, hypotheses=None, assumptions=None, reason_code=None):
    from .variational import _vstmt as _v
    return _v(sid, status, summary, evidence, hypotheses, assumptions, reason_code)


# ------------------------------------------------------------------------------------------------ G1

def certify_B_positive_definite(B: np.ndarray, fraction: float = 0.5, tries: int = 6) -> Dict[str, Any]:
    """A certified lower bound beta_c on lambda_min(B) from the inertia of B - beta I.

    beta = fraction * (smallest computed eigenvalue of B); if the Bunch-Kaufman count of negative eigenvalues of
    B - beta I is 0 with backward error b, then #{lambda(B) < beta - b} = 0, i.e. lambda_min(B) >= beta_c := beta - b
    (rounded down). Smaller fractions are tried when the count is inconclusive. Also records the inertia of B itself
    (shift 0): a negative count >= 1 there means B is not positive definite (refused by the caller).
    Returns {'ok', 'beta_c', 'beta', 'inertia', 'inertia_at_zero', 'reason'}."""
    B = binary64(np.asarray(B), "B")               # exact (precision.py)
    n = B.shape[0]
    out: Dict[str, Any] = {"ok": False, "beta_c": None, "beta": None, "inertia": None, "inertia_at_zero": None, "reason": ""}
    at0 = inertia_dense(B, 0.0)
    out["inertia_at_zero"] = at0.to_dict()
    if at0.conclusive and at0.negative is not None and at0.negative > 0:
        out["reason"] = (f"B is not positive definite: the inertia of B at shift 0 counts {at0.negative} negative eigenvalue(s) "
                         f"(backward error {at0.beta:.3e}; at least {at0.negative} eigenvalue(s) of B are <= {at0.beta:.3e})")
        out["indefinite"] = True
        return out
    out["indefinite"] = False
    try:
        lam_min = float(np.linalg.eigvalsh(B)[0]) if n else 0.0
    except Exception:
        lam_min = 0.0
    out["lambda_min_computed"] = lam_min
    if not lam_min > 0:
        out["reason"] = f"the computed smallest eigenvalue of B is {lam_min:.3e} <= 0: no positive shift to test"
        return out
    f = float(fraction)
    for _ in range(tries):
        beta = f * lam_min
        res = inertia_dense(B, beta)
        if res.conclusive and res.negative == 0:
            beta_c = float(sub_down(beta, res.beta))
            beta_c -= abs(beta_c) * 4 * U
            out.update(ok=beta_c > 0, beta_c=beta_c, beta=float(beta), inertia=res.to_dict(),
                       reason="" if beta_c > 0 else f"the backward error {res.beta:.3e} exceeds the shift {beta:.3e}")
            return out
        out["inertia"] = res.to_dict()
        f *= 0.25
    out["reason"] = "the inertia of B - beta I was inconclusive (zero pivot) or found a negative count at every shift tried"
    return out


# ------------------------------------------------------------------------------------------------ G2

def pencil_residual_radii(A: np.ndarray, B: np.ndarray, mu: np.ndarray, X: np.ndarray, beta_c: float) -> Dict[str, Any]:
    """Per pair: upper bound on ||A x - mu B x||_2, lower bound on ||x||_B = sqrt(x^H B x), and the radius
    r = ||A x - mu B x||_2 / (sqrt(beta_c) ||x||_B) >= ||A x - mu B x||_{B^-1} / ||x||_B (Lemma 2 of the derivation).

    Rounding (standard model, n terms per inner product, gamma_dot for complex data): with a = fl(A x), b = fl(B x),
    c = fl(mu b), s = fl(a - c): |s - (A x - mu B x)| <= gd_n |A||x| + |mu| gd_{n+1} |B||x| + u (1 + 2u) |s| entrywise;
    the computed |A||x|, |B||x| are divided by (1 - gamma_n). x^H B x = x^H b + x^H (B x - b) with the error-free
    x^H b (fp.exact_dot) and |x^H (B x - b)| <= gd_n |x|^T |B||x| (as in enclosure.rayleigh_interval)."""
    A, B = binary64(np.asarray(A), "A"), binary64(np.asarray(B), "B")    # exact (precision.py)
    X = binary64(np.asarray(X), "eigenvectors")
    n, k = X.shape
    cplx = np.iscomplexobj(A) or np.iscomplexobj(B) or np.iscomplexobj(X)
    gd = gamma_dot(n, cplx)
    gd1 = gamma_dot(n + 1, cplx)
    gn = gamma(n)
    AX = A @ X
    BX = B @ X
    aX = modulus_up(X)                                       # upper moduli of complex data (fp.modulus_up)
    absAX = abs_up(A) @ aX / (1 - gn)
    absBX = abs_up(B) @ aX / (1 - gn)
    s_up = np.empty(k)
    s_fl = np.empty(k)
    xb_lo = np.empty(k)
    xb_val = np.empty(k)
    for i in range(k):
        c = mu[i] * BX[:, i]
        s = AX[:, i] - c
        E = gd * absAX[:, i] + abs(mu[i]) * gd1 * absBX[:, i] + U * (1 + 2 * U) * modulus_up(s)
        s_fl[i] = float(np.linalg.norm(s))
        s_up[i] = float(up(norm2_upper(s) + norm2_upper(E)))
        num_c, num_e = exact_dot(X[:, i], BX[:, i])
        val = float(np.real(num_c))
        err = up(num_e + gd * float(modulus_up(X[:, i]) @ absBX[:, i]) * (1 + gamma(n + 2)))
        xb_val[i] = val
        xb_lo[i] = val - err
    with np.errstate(divide="ignore", invalid="ignore"):
        # denominator sqrt(beta_c x^H B x) rounded down: product and sqrt, (1 - 4u) covers both
        den = np.sqrt(beta_c * np.maximum(xb_lo, 0.0)) * (1 - 4 * U)
        rad = np.where(den > 0, s_up / den, np.inf)
    # a bound that is not finite (an overflow, or inf * 0 = nan) is stored as inf: no nan reaches the record
    rad = np.array([up(v) if math.isfinite(v) else math.inf for v in rad])
    xb_nonfinite = ~np.isfinite(xb_lo)            # x^H B x or its error overflowed: an overflow, not a small vector
    s_up = np.where(np.isfinite(s_up), s_up, np.inf)
    s_fl = np.where(np.isnan(s_fl), np.inf, s_fl)
    xb_val = np.where(np.isnan(xb_val), np.inf, xb_val)
    xb_lo = np.where(np.isnan(xb_lo), -np.inf, xb_lo)
    return {"residual_norm": s_fl, "residual_upper": s_up, "xBx": xb_val, "xBx_lower": xb_lo, "radius": rad,
            "xBx_nonfinite": xb_nonfinite}


# ------------------------------------------------------------------------------------------------ G3

def pencil_count_below(A: np.ndarray, B: np.ndarray, sigma: float, beta_c: float) -> Dict[str, Any]:
    """Inertia of M = fl(A - fl(sigma B)) with the collar for the pencil: with b = backward error of the LDL^H of M
    (inertia_dense) + ||Delta M||_2 (rounding of forming M, bounded by the infinity norm of u |sigma||B| + u |M|),
    #{lambda_pencil < sigma - b/beta_c} <= nu <= #{lambda_pencil <= sigma + b/beta_c} (Lemma 3 of the derivation)."""
    A, B = binary64(np.asarray(A), "A"), binary64(np.asarray(B), "B")    # exact (precision.py)
    M = A - sigma * B
    res = inertia_dense(M, 0.0)
    form = inf_norm_upper(U * abs(sigma) * abs_up(B) + U * abs_up(M))
    b_tot = float(up(res.beta + form)) if math.isfinite(res.beta) else float("inf")
    collar = float(up(b_tot / beta_c)) if (math.isfinite(b_tot) and beta_c > 0) else float("inf")
    return {"sigma": float(sigma), "inertia": res.to_dict(), "nu": res.negative, "conclusive": bool(res.conclusive),
            "backward_error_total": b_tot, "forming_error": float(form), "collar": collar,
            "lower_shift": float(sub_down(sigma, collar)) if math.isfinite(collar) else None,
            "upper_shift": float(add_up(sigma, collar)) if math.isfinite(collar) else None}


# ------------------------------------------------------------------------------------------------ driver

def _record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if not k.startswith("_") and k != "record_sha256"}
    return hashlib.sha256(canonical_json(_jsonable(body)).encode()).hexdigest()


def certify_generalized(A, B, eigenvalues, eigenvectors, kind: str = "lowest", *, lambda_min_B: Optional[float] = None,
                        lambda_min_B_source: str = "", max_dim: int = DEFAULT_MAX_DIM, gap_fractions=(0.9, 0.5, 0.2),
                        description: str = "") -> Dict[str, Any]:
    """Statements about the claimed eigenpairs (mu_i, x_i) of the pencil A x = lambda B x (E-03).

    kind: 'lowest' (the claimed eigenvalues are the k lowest of the pencil; G3 checks it by inertia) or 'subset' (no
    completeness claim). lambda_min_B: a DECLARED lower bound on lambda_min(B) with its source text, used only when the
    inertia of B - beta I is inconclusive; everything that rests on it is CERTIFIED_UNDER_HYPOTHESIS
    (`declared_lambda_min_B`). Returns a JSON-able record (schema GENERALIZED_SCHEMA)."""
    from .diagnostics import annotate_record
    if kind not in ("lowest", "subset"):
        raise ValueError("kind must be 'lowest' or 'subset' ('window' and 'all' pencils are not implemented)")
    widened: List[dict] = []
    A = _dense_hermitian(A, "A", widened)
    B = _dense_hermitian(B, "B", widened)
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, got {A.shape} and {B.shape}")
    n = A.shape[0]
    check_values(eigenvalues, "eigenvalues")   # the conversion below must be exact (precision.py)
    mu = claimed_reals(eigenvalues).ravel()
    X = binary64(np.asarray(eigenvectors), "eigenvectors", widened)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape != (n, mu.size):
        raise ValueError(f"eigenvectors must be n x k = {n} x {mu.size}, got {X.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(X)):
        raise ValueError("the claimed eigenvalues and eigenvectors must be finite")
    k = int(mu.size)
    assume = [STANDARD_MODEL]
    statements: List[dict] = []
    rec: Dict[str, Any] = {"schema": GENERALIZED_SCHEMA, "description": description or f"generalized {kind} claim, {k} pairs",
                           "operator": {"kind": "pencil", "dimension": n, "A_sha256": matrix_sha256(A), "B_sha256": matrix_sha256(B),
                                        "complex": bool(np.iscomplexobj(A) or np.iscomplexobj(B)),
                                        "refers_to": "the stored matrices A and B"},
                           "claim": {"kind": kind, "n_pairs": k, "eigenvalues": [float(m) for m in mu],
                                     "eigenvectors_sha256": array_sha256(X)},
                           "state": {"form": "pencil"}}
    widening = widening_record(widened)
    if widening is not None:                 # only when an input was converted: binary64 inputs keep their record
        rec["input_widening"] = widening
    if n > max_dim:
        for sid in GENERALIZED_STATEMENT_IDS:
            statements.append(_vstmt(sid, "NOT_CHECKED", f"dimension {n} exceeds max_dim = {max_dim} (dense path only)", {},
                                     reason_code="SIZE_LIMIT", assumptions=assume))
        rec["statements"] = statements
        annotate_record(rec)
        rec["record_sha256"] = _record_hash(rec)
        return rec

    # ---- G1
    pd = certify_B_positive_definite(B)
    hyps: List[dict] = []
    if pd.get("indefinite"):
        for sid in GENERALIZED_STATEMENT_IDS:
            statements.append(_vstmt(sid, "NOT_CHECKED", pd["reason"] + "; the pencil statements need B positive definite",
                                     {"inertia_at_zero": pd["inertia_at_zero"]}, reason_code="UNSUPPORTED_MODEL", assumptions=assume))
        rec["statements"] = statements
        annotate_record(rec)
        rec["record_sha256"] = _record_hash(rec)
        return rec
    if pd["ok"]:
        beta_c = float(pd["beta_c"])
        g1 = _vstmt("G1.B_positive_definite", "CERTIFIED",
                    f"lambda_min(B) >= {beta_c:.17g} > 0 (inertia of B - {pd['beta']:.17g} I: no negative eigenvalue, backward error "
                    f"{pd['inertia']['beta']:.3e})", {"beta_c": beta_c, "shift": pd["beta"], "inertia": pd["inertia"],
                                                      "inertia_at_zero": pd["inertia_at_zero"],
                                                      "B_inverse_norm_upper": float(up(1.0 / beta_c))}, assumptions=assume)
        if lambda_min_B is not None:
            g1["evidence"]["declared_lambda_min_B"] = {"value": float(lambda_min_B), "source": lambda_min_B_source,
                                                       "used": False}
    elif lambda_min_B is not None and float(lambda_min_B) > 0:
        beta_c = float(lambda_min_B)
        hyps = [{"name": "declared_lambda_min_B", "statement": f"lambda_min(B) >= {beta_c!r} (declared; the inertia of B - beta I "
                 "could not certify a positive lower bound)", "value": beta_c, "source": lambda_min_B_source or "(no source given)"}]
        g1 = _vstmt("G1.B_positive_definite", "CERTIFIED_UNDER_HYPOTHESIS",
                    f"lambda_min(B) >= {beta_c!r} UNDER THE DECLARED HYPOTHESIS (source: {lambda_min_B_source or '(no source given)'}); "
                    f"the inertia could not certify it: {pd['reason']}",
                    {"beta_c": beta_c, "inertia": pd.get("inertia"), "inertia_at_zero": pd["inertia_at_zero"],
                     "B_inverse_norm_upper": float(up(1.0 / beta_c))}, hypotheses=hyps, assumptions=assume)
    else:
        g1 = _vstmt("G1.B_positive_definite", "INCONCLUSIVE", f"no certified lower bound on lambda_min(B): {pd['reason']}",
                    {"inertia": pd.get("inertia"), "inertia_at_zero": pd["inertia_at_zero"]}, reason_code="NOT_APPLICABLE",
                    assumptions=assume)
        statements.append(g1)
        for sid in GENERALIZED_STATEMENT_IDS[1:]:
            statements.append(_vstmt(sid, "NOT_CHECKED", "needs a lower bound on lambda_min(B) (G1): declare one with its source "
                                     "(lambda_min_B, lambda_min_B_source) or make B better conditioned", {},
                                     reason_code="NEEDS_DEPENDENCY", assumptions=assume))
        rec["statements"] = statements
        annotate_record(rec)
        rec["record_sha256"] = _record_hash(rec)
        return rec
    statements.append(g1)
    under = "CERTIFIED_UNDER_HYPOTHESIS" if hyps else "CERTIFIED"

    # ---- G2
    rr = pencil_residual_radii(A, B, mu, X, beta_c)
    rad = rr["radius"]
    rec["enclosures"] = [{"index": i, "mu": float(mu[i]), "radius": float(rad[i]),
                          "lower": float(sub_down(mu[i], rad[i])) if math.isfinite(rad[i]) else None,
                          "upper": float(add_up(mu[i], rad[i])) if math.isfinite(rad[i]) else None,
                          "residual_norm": float(rr["residual_norm"][i]), "residual_upper": float(rr["residual_upper"][i]),
                          "xBx": float(rr["xBx"][i]), "xBx_lower": float(rr["xBx_lower"][i])} for i in range(k)]
    finite = bool(np.all(np.isfinite(rad)))
    if finite:
        statements.append(_vstmt("G2.eigenpairs", under,
                                 f"each of the {k} claimed values lies within its radius of an eigenvalue of the pencil (Theorem 1 "
                                 f"for L^-1 A L^-H; max radius {float(np.max(rad)):.3e})" +
                                 (" UNDER THE DECLARED HYPOTHESIS on lambda_min(B)" if hyps else ""),
                                 {"n_pairs": k, "max_radius": float(np.max(rad)), "median_radius": float(np.median(rad)),
                                  "beta_c": beta_c, "basis": "r_i = ||A x - mu B x||_2 / (sqrt(beta_c) ||x||_B); docs/proofs/E-03_generalized.md"},
                                 hypotheses=hyps, assumptions=assume))
    else:
        bad = [i for i in range(k) if not math.isfinite(rad[i])]
        unresolved = [i for i in bad if not rr["xBx_nonfinite"][i] and not rr["xBx_lower"][i] > 0]
        over = [i for i in bad if i not in unresolved]
        parts = []
        if over:
            parts.append(f"the radius of pairs {over[:20]} is not finite (a bound of the floating-point accounting "
                         f"overflowed)")
        if unresolved:
            parts.append(f"x^H B x could not be bounded away from zero for pairs {unresolved[:20]} (vectors too small "
                         f"or B-norm unresolved)")
        statements.append(_vstmt("G2.eigenpairs", "INCONCLUSIVE", "; ".join(parts),
                                 {"n_pairs": k, "unresolved": unresolved, "overflowed": over},
                                 reason_code="NONFINITE_BOUND" if over else "UNRESOLVED_VECTOR_NORM",
                                 assumptions=assume))

    # ---- G3
    if kind == "subset":
        statements.append(_vstmt("G3.completeness", "NOT_CHECKED", "claim kind 'subset' makes no completeness claim", {},
                                 reason_code="NO_COMPLETENESS_CLAIM", assumptions=assume))
    elif not finite:
        statements.append(_vstmt("G3.completeness", "NOT_CHECKED", "needs finite enclosures (G2)", {},
                                 reason_code="NEEDS_DEPENDENCY", assumptions=assume))
    else:
        order = np.argsort(mu, kind="stable")
        lo = np.array([e["lower"] for e in rec["enclosures"]])[order]
        hi = np.array([e["upper"] for e in rec["enclosures"]])[order]
        disjoint = bool(np.all(lo[1:] > hi[:-1])) if k > 1 else True
        top = float(np.max(hi))
        # estimate of the next eigenvalue for the shift: the pencil's (k+1)-th eigenvalue if it can be computed
        try:
            import scipy.linalg as la
            w_all = la.eigh(A, B, eigvals_only=True)
            nxt = float(w_all[k]) if k < n else None
        except Exception:
            nxt = None
        counts = []
        g3 = None
        if nxt is None or not nxt > top:
            cands = [top + abs(top) * 1e-6 + 1e-300]
        else:
            cands = [top + f * (nxt - top) for f in gap_fractions]
        for sig in cands:
            c = pencil_count_below(A, B, float(sig), beta_c)
            counts.append(c)
            if not c["conclusive"] or c["lower_shift"] is None:
                continue
            if c["lower_shift"] < top:
                continue            # the collar reaches into the claimed enclosures: try a higher shift
            nu = int(c["nu"])
            if nu == k and disjoint:
                g3 = _vstmt("G3.completeness", under,
                            f"exactly {k} eigenvalues of the pencil lie below {c['lower_shift']:.17g}: the claimed enclosures are the "
                            f"{k} lowest (inertia of A - sigma B at sigma = {sig:.17g}: nu = {nu}, collar {c['collar']:.3e}); "
                            f"lambda_{k + 1} >= {c['lower_shift']:.17g}" + (" UNDER THE DECLARED HYPOTHESIS on lambda_min(B)" if hyps else ""),
                            {"count_from_inertia": nu, "n_claimed": k, "complete_below": c["lower_shift"],
                             "next_eigenvalue_lower_bound": c["lower_shift"], "counts": counts, "pairing": "pairwise disjoint enclosures"},
                            hypotheses=hyps, assumptions=assume)
                break
            if nu < k:
                g3 = _vstmt("G3.completeness", "INCONCLUSIVE",
                            f"the inertia counts {nu} < {k} eigenvalues below {c['upper_shift']:.17g}: some claimed enclosures share an "
                            "eigenvalue (duplicate or non-orthogonal claimed vectors); the cluster bound for pencils is not implemented",
                            {"count_from_inertia": nu, "n_claimed": k, "counts": counts}, reason_code="NOT_IMPLEMENTED", assumptions=assume)
                break
            if nu > k:
                g3 = _vstmt("G3.completeness", "INCONCLUSIVE",
                            f"the inertia counts {nu} > {k} eigenvalues below {c['upper_shift']:.17g}: at least {nu - k} eigenvalue(s) of "
                            "the pencil below the shift are missing from the claim (or lie in the collar)",
                            {"count_from_inertia": nu, "n_claimed": k, "missing_at_least": nu - k, "counts": counts},
                            reason_code="NOT_APPLICABLE", assumptions=assume)
                g3["status"] = "INCONCLUSIVE"
                break
            if nu == k and not disjoint:
                g3 = _vstmt("G3.completeness", "INCONCLUSIVE",
                            f"the inertia counts {k} eigenvalues below {c['lower_shift']:.17g} but the claimed enclosures overlap: the "
                            "cluster bound (distinct eigenvalues per enclosure) for pencils is not implemented",
                            {"count_from_inertia": nu, "n_claimed": k, "counts": counts}, reason_code="NOT_IMPLEMENTED", assumptions=assume)
                break
        if g3 is None:
            g3 = _vstmt("G3.completeness", "INCONCLUSIVE",
                        "no conclusive inertia count with its collar above the claimed enclosures at the shifts tried",
                        {"counts": counts, "n_claimed": k}, reason_code="NOT_APPLICABLE", assumptions=assume)
        statements.append(g3)
    rec["statements"] = statements
    annotate_record(rec)
    rec["basis"] = ("B = L L^H, C = L^-1 A L^-H similar to the pencil; Theorem 1 for C with ||L^-1|| = lambda_min(B)^-1/2; "
                    "Sylvester's law for A - sigma B = L (C - sigma I) L^H; docs/proofs/E-03_generalized.md")
    rec["record_sha256"] = _record_hash(rec)
    return rec


def format_generalized(rec: dict) -> str:
    lines = [f"generalized eigenproblem record ({rec['schema']}): {rec.get('description', '')}",
             f"  dimension {rec['operator']['dimension']}, {rec['claim']['n_pairs']} claimed pairs ({rec['claim']['kind']})"]
    for s in rec["statements"]:
        lines.append(f"  {s['id']:24s} {s['status']:27s} {s['summary']}")
        for h in s.get("hypotheses", []):
            lines.append(f"      hypothesis {h['name']}: {h['statement']}; source: {h['source']}")
    return "\n".join(lines)
