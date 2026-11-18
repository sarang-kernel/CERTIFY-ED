"""Non-Hermitian eigenproblems (E-02, 3.3.0.dev0; dense path). Every statement is UNVERIFIED (D-12).

A is any square stored matrix, real or complex, not assumed Hermitian or normal. A claim lists values mu_i (complex)
with right vectors x_i (columns of V) and optionally left vectors y_i. Four statements
(docs/proofs/E-02_non_hermitian.md):

  N1.backward_errors      for each pair, mu_i is an eigenvalue of A + E_i for some E_i with ||E_i||_2 <= eps_i, where
                          eps_i >= ||A x_i - mu_i x_i|| / ||x_i|| (Lemma 1, right residual); with a left vector y_i also
                          of A + F_i with ||F_i||_2 <= eps'_i >= ||A^H y_i - conj(mu_i) y_i|| / ||y_i||. Equivalently
                          sigma_min(A - mu_i I) <= min(eps_i, eps'_i): mu_i lies in the eps-pseudospectrum of A. For a
                          matrix that is not normal this is NOT an enclosure of an eigenvalue of A.
  N2.eigenvector_inverse  for a complete claim (kind 'all', V square): V is invertible and ||V^-1||_2 <= beta =
                          c_W / (1 - theta), from theta >= ||I - W V||_2 < 1 for the computed inverse W and c_W >= ||W||_2
                          (Lemma 4); also cond_2(V) <= beta c_V with c_V >= ||V||_2
  N3.inclusion            every eigenvalue of A lies in the union of the closed discs D(mu_i, rho), rho >= beta ||R||_F >=
                          ||V^-1 R||_2 with R = A V - V diag(mu): A = V (diag(mu) + V^-1 R) V^-1, and Bauer-Fike for the
                          diagonal matrix diag(mu) (Theorem 3 (b))
  N4.counting             the discs are split into groups whose unions are pairwise disjoint (two discs are certified
                          disjoint when a lower bound on |mu_i - mu_j| exceeds 2 rho); the union of a group of m discs
                          contains exactly m eigenvalues of A counted with algebraic multiplicity (continuity of the
                          eigenvalues of diag(mu) + t V^-1 R, 0 <= t <= 1; Theorem 3 (c)); a disc that meets no other disc
                          contains exactly one eigenvalue, and it is simple; when every disc is isolated, A has n distinct
                          eigenvalues and is diagonalizable

The statements concern the stored matrix A. A need not be diagonalizable for N2 to N4: the similarity is by the claimed
V, whose invertibility N2 certifies. A defective A, or one close to a defective matrix, has an ill-conditioned or
numerically singular eigenvector matrix; then N2 is INCONCLUSIVE with reason ILL_CONDITIONED_EIGENVECTORS and N3, N4 are
not checked. A partial claim (kind 'subset', fewer than n pairs) gets N1 only: without the whole eigenvector matrix no
inclusion statement for a non-normal A follows from the residuals. Pseudospectral portraits, Lindbladian structure and
cluster statements with per-disc radii are not implemented.

Arithmetic (docs/proofs/E-02_non_hermitian.md, Propositions 5 to 7): binary64 round-to-nearest, the standard model
with the additive term of the underflow-tolerant model, BLAS inner products within gamma_n (gamma_dot for complex data),
the modulus of a complex number within 3u (hypothesis H-mod, as docs/proofs/D-01_sparse-residual.md), and the
no-underflow scan of the inputs (fp.underflow_scan; nonzero magnitudes >= 2^-900) unless underflow_check=False.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import _jsonable, array_sha256, matrix_sha256
from .fp import (U, ULP_MIN, gamma, gamma_dot, inf_norm_upper, norm2_lower, norm2_upper, safe_sqrt_upper, sub_down,
                 underflow_scan, up)
from .precision import binary64, check_values, widening_record
from .spec import canonical_json

NONHERMITIAN_SCHEMA = "certify-ed/nonhermitian/1.0"
NONHERMITIAN_STATEMENT_IDS = ("N1.backward_errors", "N2.eigenvector_inverse", "N3.inclusion", "N4.counting")
NONHERMITIAN_KINDS = ("all", "subset")
DEFAULT_MAX_DIM = 2000
UNVERIFIED = ("UNVERIFIED: derived in docs/proofs/E-02_non_hermitian.md and re-derived independently (not a human "
              "check); Sarang's check (D-12) pending")
STANDARD_MODEL = ("IEEE-754 binary64 round-to-nearest, standard model fl(a op b) = (a op b)(1 + delta), |delta| <= u, with "
                  "the additive term of the underflow-tolerant model (|eta| <= 2^-1075 per product); BLAS inner products "
                  "within gamma_n (complex: 2 gamma_{n+4}); modulus of a complex number within 3u (H-mod); no-underflow "
                  "scan of the stored inputs (nonzero magnitudes >= 2^-900)")
_G8 = 1.0 + 8.0 * U          # exact binary64 numbers (1 + 4 * 2^-52 and 1 + 6 * 2^-52)
_G12 = 1.0 + 12.0 * U
_ONE_MINUS_2U = 1.0 - 2.0 * U

__all__ = ["NONHERMITIAN_SCHEMA", "NONHERMITIAN_STATEMENT_IDS", "DEFAULT_MAX_DIM", "UNVERIFIED", "certify_nonhermitian",
           "residual_bounds", "inverse_bound", "matrix_norm2_upper", "disc_groups", "additive_term",
           "format_nonhermitian"]


# ------------------------------------------------------------------------------------------------ inputs

def _dense_square(M, name: str, widened: Optional[list] = None) -> np.ndarray:
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
    M = binary64(M, name, widened)        # exact for narrower formats (noted), a wider one refused (precision.py)
    M = M.astype(np.complex128 if np.iscomplexobj(M) else np.float64)     # a private copy
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} has non-finite entries")
    return M


def _vectors(X, n: int, k: int, name: str, widened: Optional[list] = None) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape != (n, k):
        raise ValueError(f"{name} must be n x k = {n} x {k}, got {X.shape}")
    X = binary64(X, name, widened)        # exact for narrower formats (noted), a wider one refused (precision.py)
    X = X.astype(np.complex128 if np.iscomplexobj(X) else np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must be finite")
    return X


# ------------------------------------------------------------------------------------------------ Proposition 5

def additive_term(n: int, w: int) -> float:
    """T_{n,w} >= sqrt(n) (10 w + 20) u_min, u_min = 2^-1075: the 2-norm of the per-entry shortfall that the additive
    terms of the underflow-tolerant model can cause in the computed residual and in its entrywise bound (at most
    4 w + 4 rounded products in an entry of the residual and w + 6 operations in an entry of the bound, each term
    propagated by a factor at most 2; Proposition 5 (iv)). Formed like fp.residual_additive_term: the guards cover
    the roundings of the normal-range factor, the ldexp rounds to the subnormal grid by at most u_min, and
    ULP_MIN = 2 u_min covers that."""
    t = math.sqrt(float(max(int(n), 1))) * (10.0 * max(int(w), 1) + 20.0) * _G8 * _G8
    return math.ldexp(t, -1075) + ULP_MIN


def _bound_constants(cplx: bool, w: int):
    """(G, g): G = gamma_dot(w + 2) bounds the rounding of one residual entry relative to (|A||x|)_i + |mu||x_i|;
    1/(1 - g) undoes the roundings of the computed fl(|A||x|) (w products and sums; for complex data the two moduli of
    each term, 3u each under H-mod: g = gamma_{w+6}; real data: g = gamma_{w+1})."""
    return gamma_dot(w + 2, cplx), (gamma(w + 6) if cplx else gamma(w + 1))


def residual_bounds(A: np.ndarray, mu: np.ndarray, X: np.ndarray, return_arrays: bool = False) -> Dict[str, Any]:
    """Proposition 5. For each column x_i of X and value mu_i: the computed residual s_i = fl(fl(A x_i) - fl(mu_i x_i)),
    an entrywise bound E_i >= |s_i - (A x_i - mu_i x_i)| (as computed), the upper bound
    S_i = up(norm2_upper(s_i) + norm2_upper(E_i) + T) >= ||A x_i - mu_i x_i||_2, the lower bound norm2_lower(x_i) on
    ||x_i||_2, and eps_i = up(S_i / ||x_i||_lower + 2 ULP_MIN) >= ||A x_i - mu_i x_i|| / ||x_i|| (inf when the lower
    bound is 0). return_arrays: also return the computed residual matrix 'S_hat' and the bound matrix 'E' (for the
    verification scripts)."""
    A, X = binary64(np.asarray(A), "A"), binary64(np.asarray(X), "eigenvectors")    # exact (precision.py)
    n, k = X.shape
    mu = np.asarray(mu).ravel()
    cplx = bool(np.iscomplexobj(A) or np.iscomplexobj(X) or np.iscomplexobj(mu))
    G, g = _bound_constants(cplx, n)
    AX = A @ X
    S_hat = AX - X * mu[None, :]
    del AX
    absAX = np.abs(A) @ np.abs(X)
    E = absAX / (1.0 - g)
    del absAX
    E += (np.abs(X) * np.abs(mu)[None, :]) * _G8
    E *= G
    E *= _G12
    T = additive_term(n, n)
    computed = np.empty(k)
    upper = np.empty(k)
    xl = np.empty(k)
    eps = np.empty(k)
    E_up = np.empty(k)
    for i in range(k):
        computed[i] = float(np.linalg.norm(S_hat[:, i]))
        E_up[i] = norm2_upper(E[:, i])
        upper[i] = up(norm2_upper(S_hat[:, i]) + E_up[i] + T)
        xl[i] = norm2_lower(X[:, i])
        eps[i] = up(upper[i] / xl[i] + 2.0 * ULP_MIN) if xl[i] > 0 and math.isfinite(upper[i]) else math.inf
    out = {"computed": computed, "upper": upper, "E_norm": E_up, "x_norm_lower": xl, "eps": eps, "complex": cplx}
    if return_arrays:
        out.update(S_hat=S_hat, E=E)
    return out


# ------------------------------------------------------------------------------------------------ Proposition 6

def matrix_norm2_upper(M: np.ndarray) -> float:
    """c >= ||M||_2: the smaller of the Frobenius bound and sqrt(||M||_1 ||M||_inf) (fp.inf_norm_upper on M^T and M;
    for complex M valid under H-mod, Proposition 6 (iii))."""
    M = binary64(np.asarray(M), "matrix")          # exact (precision.py)
    fro = norm2_upper(np.asarray(M).ravel())
    hoelder = safe_sqrt_upper(up(inf_norm_upper(np.asarray(M).T) * inf_norm_upper(np.asarray(M))))
    vals = [v for v in (fro, hoelder) if math.isfinite(v)]
    return min(vals) if vals else math.inf


def inverse_bound(V: np.ndarray, W: Optional[np.ndarray] = None, return_arrays: bool = False) -> Dict[str, Any]:
    """Lemma 4 and Proposition 6. With W a computed inverse of V (np.linalg.inv when not given; any matrix is allowed,
    its quality only decides whether theta < 1): P = fl(W V), G_hat = fl(I - P), the entrywise bound
    B = (gamma_dot(n) fl(|W||V|) / (1 - g) + 4u |diag G_hat|)(1 + 12u) >= |(I - W V) - G_hat|, and
    theta = up(norm2_upper(G_hat) + norm2_upper(B) + T_G) >= ||I - W V||_F >= ||I - W V||_2. When theta < 1: V is
    invertible, ||V^-1||_2 <= beta = up(c_W / sub_down(1, theta) + 2 ULP_MIN), and cond_2(V) <= kappa = up(beta c_V).
    Returns {'ok', 'theta', 'c_W', 'beta', 'c_V', 'kappa_upper', 'cond_estimate', 'reason'} (plus 'W', 'G_hat', 'B' with
    return_arrays)."""
    V = binary64(np.asarray(V), "eigenvectors")    # exact (precision.py)
    n = V.shape[0]
    out: Dict[str, Any] = {"ok": False, "theta": None, "c_W": None, "beta": None, "c_V": None, "kappa_upper": None,
                           "cond_estimate": None, "reason": ""}
    try:
        out["cond_estimate"] = float(np.linalg.cond(V))           # floating-point estimate, reported only
    except Exception:
        out["cond_estimate"] = math.inf
    if W is None:
        try:
            W = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            out["reason"] = "the claimed eigenvector matrix is singular in floating point (LAPACK found a zero pivot)"
            return out
    W = binary64(np.asarray(W), "W")
    if W.shape != V.shape or not np.all(np.isfinite(W)):
        out["reason"] = "the computed inverse of the eigenvector matrix is not finite (the matrix is numerically singular)"
        return out
    cplx = bool(np.iscomplexobj(V) or np.iscomplexobj(W))
    Gn = gamma_dot(n, cplx)
    g = gamma(n + 6) if cplx else gamma(n + 1)
    P = W @ V
    G_hat = np.eye(n, dtype=P.dtype) - P
    del P
    B = np.abs(W) @ np.abs(V)
    B /= (1.0 - g)
    B *= Gn
    B[np.diag_indices(n)] += (4.0 * U) * np.abs(np.diag(G_hat))
    B *= _G12
    T_G = math.ldexp(float(n) * (10.0 * n + 20.0) * _G8 * _G8, -1075) + ULP_MIN
    theta = up(norm2_upper(G_hat.ravel()) + norm2_upper(B.ravel()) + T_G)
    if return_arrays:
        out.update(W=W, G_hat=G_hat, B=B)
    del B, G_hat
    out["theta"] = float(theta)
    c_W = matrix_norm2_upper(W)
    c_V = matrix_norm2_upper(V)
    out["c_W"], out["c_V"] = float(c_W), float(c_V)
    if not (theta < 1.0):
        out["reason"] = (f"||I - W V||_2 could not be bounded below 1 (bound {theta:.3e}; floating-point estimate of cond_2(V) "
                         f"{out['cond_estimate']:.3e}): the eigenvector matrix is too ill-conditioned for binary64, or A is "
                         "defective or close to a defective matrix")
        return out
    one_minus = float(sub_down(1.0, theta))
    beta = up(c_W / one_minus + 2.0 * ULP_MIN) if one_minus > 0 else math.inf
    if not math.isfinite(beta):
        out["reason"] = "the bound on ||V^-1||_2 overflows"
        return out
    out.update(ok=True, beta=float(beta), kappa_upper=float(up(beta * c_V)) if math.isfinite(c_V) else math.inf)
    return out


# ------------------------------------------------------------------------------------------------ Proposition 7

def disc_groups(mu: np.ndarray, rho: float) -> Dict[str, Any]:
    """Groups of the discs D(mu_i, rho) whose unions are pairwise disjoint. Two discs are certified disjoint when
    delta_ij = fl(max(|fl(Re mu_i - Re mu_j)|, |fl(Im mu_i - Im mu_j)|) (1 - 2u)) > 2 rho; delta_ij <= |mu_i - mu_j|
    (Proposition 7 (i)), and 2 rho is exact. The groups are the connected components of the graph whose edges join
    discs that are not certified disjoint, so discs in different groups are disjoint (Proposition 7 (ii))."""
    from scipy.sparse.csgraph import connected_components
    mu = np.asarray(mu).ravel()
    n = mu.size
    re = np.ascontiguousarray(np.real(mu), dtype=np.float64)
    im = np.ascontiguousarray(np.imag(mu), dtype=np.float64)
    two_rho = 2.0 * float(rho)
    with np.errstate(over="ignore", invalid="ignore"):
        dx = np.abs(np.subtract.outer(re, re))
        dy = np.abs(np.subtract.outer(im, im))
        delta = np.maximum(dx, dy)
        del dx, dy
        delta *= _ONE_MINUS_2U
        joined = ~(delta > two_rho)
    del delta
    ncomp, labels = connected_components(sp.csr_matrix(joined), directed=False)
    groups: List[List[int]] = [[] for _ in range(ncomp)]
    for i, lab in enumerate(labels.tolist()):
        groups[lab].append(i)
    groups.sort(key=lambda gr: gr[0])
    group_of = [0] * n
    for gi, gr in enumerate(groups):
        for i in gr:
            group_of[i] = gi
    return {"groups": groups, "group_of": group_of, "isolated": [gr[0] for gr in groups if len(gr) == 1],
            "two_rho": two_rho}


# ------------------------------------------------------------------------------------------------ driver

def _stmt(sid, status, summary, evidence=None, reason_code=None, assumptions=None):
    ev = dict(evidence or {})
    if reason_code is not None:
        ev["reason_code"] = reason_code
    return {"id": sid, "status": status, "summary": summary, "evidence": ev, "hypotheses": [],
            "assumptions": list(assumptions or [])}


def _record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if not k.startswith("_") and k != "record_sha256"}
    return hashlib.sha256(canonical_json(_jsonable(body)).encode()).hexdigest()


def _finish(rec: dict, statements: List[dict]) -> dict:
    from .diagnostics import annotate_record
    rec["statements"] = statements
    annotate_record(rec)
    rec["record_sha256"] = _record_hash(rec)
    return rec


def _cpair(z) -> List[float]:
    z = complex(z)
    return [float(z.real), float(z.imag)]


def certify_nonhermitian(A, eigenvalues, eigenvectors, *, kind: str = "auto", left_eigenvectors=None,
                         W=None, max_dim: int = DEFAULT_MAX_DIM, underflow_check: bool = True,
                         description: str = "") -> Dict[str, Any]:
    """Statements about claimed eigenpairs (mu_i, x_i) of a square, not necessarily Hermitian matrix A (E-02).

    kind: 'all' (n pairs, the columns of a square eigenvector matrix V: N1 to N4), 'subset' (any k pairs: N1 only) or
    'auto' ('all' when k = n). left_eigenvectors: optional n x k left vectors y_i (y_i^H A = mu_i y_i^H) for the left
    backward errors of N1. W: optional computed inverse of V (default np.linalg.inv(V)); it is only a tool, the bound
    holds for any W. Returns a JSON-able record (schema NONHERMITIAN_SCHEMA). Every statement is UNVERIFIED."""
    widened: List[dict] = []
    A = _dense_square(A, "A", widened)
    n = A.shape[0]
    check_values(eigenvalues, "eigenvalues")   # the conversion below must be exact (precision.py)
    mu = np.asarray(eigenvalues).ravel()
    mu = mu.astype(np.complex128 if np.iscomplexobj(mu) else np.float64)
    if not np.all(np.isfinite(mu)):
        raise ValueError("the claimed eigenvalues must be finite")
    k = int(mu.size)
    if k < 1:
        raise ValueError("the claim must list at least one eigenpair")
    X = _vectors(eigenvectors, n, k, "eigenvectors", widened)
    Y = None if left_eigenvectors is None else _vectors(left_eigenvectors, n, k, "left_eigenvectors", widened)
    if kind == "auto":
        kind = "all" if k == n else "subset"
    if kind not in NONHERMITIAN_KINDS:
        raise ValueError(f"kind must be one of {NONHERMITIAN_KINDS} or 'auto', got {kind!r}")
    if kind == "all" and k != n:
        raise ValueError(f"kind 'all' needs n = {n} pairs (a square eigenvector matrix), got {k}")
    assume = [STANDARD_MODEL]
    rec: Dict[str, Any] = {
        "schema": NONHERMITIAN_SCHEMA, "description": description or f"non-Hermitian {kind} claim, {k} pairs",
        "operator": {"kind": "matrix", "dimension": n, "A_sha256": matrix_sha256(A), "complex": bool(np.iscomplexobj(A)),
                     "refers_to": "the stored matrix A (not assumed Hermitian or normal)"},
        "claim": {"kind": kind, "n_pairs": k, "eigenvalues": [_cpair(z) for z in mu],
                  "eigenvectors_sha256": array_sha256(X),
                  "left_eigenvectors_sha256": None if Y is None else array_sha256(Y)},
        "state": {"form": "non-Hermitian"}, "unverified": UNVERIFIED,
        "basis": ("Lemma 1 (backward error of a residual), Lemma 4 (Neumann bound on ||V^-1||), Theorem 3 (Bauer-Fike for "
                  "diag(mu) after the similarity V, counting by continuity); docs/proofs/E-02_non_hermitian.md"),
    }
    widening = widening_record(widened)
    if widening is not None:                 # only when an input was converted: binary64 inputs keep their record
        rec["input_widening"] = widening
    statements: List[dict] = []
    if n > max_dim:
        for sid in NONHERMITIAN_STATEMENT_IDS:
            statements.append(_stmt(sid, "NOT_CHECKED", f"dimension {n} exceeds max_dim = {max_dim} (dense path only)",
                                    {}, "SIZE_LIMIT", assume))
        return _finish(rec, statements)
    if underflow_check:
        parts = [("A", A), ("eigenvalues", mu), ("eigenvectors", X)] + ([("left_eigenvectors", Y)] if Y is not None else [])
        hits = underflow_scan(parts)
        if hits:
            rec["underflow_scan"] = hits
            for sid in NONHERMITIAN_STATEMENT_IDS:
                statements.append(_stmt(sid, "INCONCLUSIVE",
                                        "a nonzero stored magnitude is below 2^-900 (" + ", ".join(h["where"] for h in hits)
                                        + "): the no-underflow hypothesis of the floating-point model is not established",
                                        {"underflow_scan": hits}, "UNDERFLOW_RISK", assume))
            return _finish(rec, statements)
    else:
        assume = assume + ["no-underflow scan switched off by the caller (underflow_check=False)"]

    # ---- N1
    rr = residual_bounds(A, mu, X)
    lr = residual_bounds(A.conj().T, np.conj(mu), Y) if Y is not None else None
    rec["pairs"] = []
    for i in range(k):
        row = {"index": i, "mu": _cpair(mu[i]), "residual_norm": float(rr["computed"][i]),
               "residual_upper": float(rr["upper"][i]), "x_norm_lower": float(rr["x_norm_lower"][i]),
               "eps_right": float(rr["eps"][i])}
        if lr is not None:
            row.update(left_residual_upper=float(lr["upper"][i]), y_norm_lower=float(lr["x_norm_lower"][i]),
                       eps_left=float(lr["eps"][i]))
        row["backward_error_upper"] = float(min(row["eps_right"], row.get("eps_left", math.inf)))
        rec["pairs"].append(row)
    be = np.array([r["backward_error_upper"] for r in rec["pairs"]])
    if np.all(np.isfinite(be)):
        statements.append(_stmt(
            "N1.backward_errors", "CERTIFIED",
            f"each of the {k} claimed values mu_i is an eigenvalue of A + E_i for some E_i with ||E_i||_2 <= eps_i "
            f"(max eps_i {float(np.max(be)):.3e}), i.e. sigma_min(A - mu_i I) <= eps_i; for a matrix that is not normal this "
            f"is not an enclosure of an eigenvalue of A; {UNVERIFIED}",
            {"n_pairs": k, "max_backward_error": float(np.max(be)), "median_backward_error": float(np.median(be)),
             "left_vectors": Y is not None,
             "basis": "Lemma 1: E = -r x^H / (x^H x); eps_i = up(S_i / ||x_i||_lower); docs/proofs/E-02_non_hermitian.md"},
            None, assume))
    else:
        bad = [i for i in range(k) if not math.isfinite(be[i])]
        statements.append(_stmt("N1.backward_errors", "INCONCLUSIVE",
                                f"the norm of the claimed vector could not be bounded away from zero, or the residual bound "
                                f"overflowed, for pairs {bad[:20]}: no finite backward error", {"unresolved": bad[:200]},
                                "NOT_APPLICABLE", assume))

    # ---- N2 to N4
    if kind != "all":
        for sid in NONHERMITIAN_STATEMENT_IDS[1:]:
            statements.append(_stmt(sid, "NOT_CHECKED", "the claim is not complete (kind 'subset'): the inclusion and count "
                                    "statements need all n eigenpairs, a square eigenvector matrix", {},
                                    "NEEDS_WHOLE_SPECTRUM", assume))
        return _finish(rec, statements)
    inv = inverse_bound(X, W)
    rec["eigenvector_matrix"] = inv
    if not inv["ok"]:
        statements.append(_stmt("N2.eigenvector_inverse", "INCONCLUSIVE", inv["reason"],
                                {k_: inv[k_] for k_ in ("theta", "c_W", "cond_estimate")}, "ILL_CONDITIONED_EIGENVECTORS",
                                assume))
        for sid in NONHERMITIAN_STATEMENT_IDS[2:]:
            statements.append(_stmt(sid, "NOT_CHECKED", "needs a certified bound on ||V^-1||_2 (N2)", {},
                                    "NEEDS_DEPENDENCY", assume))
        return _finish(rec, statements)
    beta = inv["beta"]
    statements.append(_stmt(
        "N2.eigenvector_inverse", "CERTIFIED",
        f"the claimed eigenvector matrix V is invertible with ||V^-1||_2 <= {beta:.6e} (||I - W V||_2 <= {inv['theta']:.3e} < 1 "
        f"for its computed inverse W); cond_2(V) <= {inv['kappa_upper']:.6e}; {UNVERIFIED}",
        {"theta": inv["theta"], "c_W": inv["c_W"], "beta": beta, "c_V": inv["c_V"], "kappa_upper": inv["kappa_upper"],
         "cond_estimate": inv["cond_estimate"],
         "basis": "Lemma 4: V^-1 = (I - G)^-1 W with ||G||_2 <= theta < 1; docs/proofs/E-02_non_hermitian.md"},
        None, assume))
    rho_R = norm2_upper(rr["upper"])
    rho = up(beta * rho_R + 2.0 * ULP_MIN) if math.isfinite(rho_R) else math.inf
    rec["radius"] = {"rho": float(rho), "residual_frobenius_upper": float(rho_R), "beta": beta}
    if not math.isfinite(rho):
        statements.append(_stmt("N3.inclusion", "INCONCLUSIVE", "the disc radius beta ||R||_F overflows", {"beta": beta},
                                "NOT_APPLICABLE", assume))
        statements.append(_stmt("N4.counting", "NOT_CHECKED", "needs finite discs (N3)", {}, "NEEDS_DEPENDENCY", assume))
        return _finish(rec, statements)
    statements.append(_stmt(
        "N3.inclusion", "CERTIFIED",
        f"every eigenvalue of A lies in the union of the {n} closed discs |z - mu_i| <= {rho:.6e} around the claimed values "
        f"(Bauer-Fike for diag(mu) after the similarity V: rho >= ||V^-1||_2 ||A V - V diag(mu)||_F); {UNVERIFIED}",
        {"rho": float(rho), "residual_frobenius_upper": float(rho_R), "beta": beta,
         "basis": "Theorem 3 (b); docs/proofs/E-02_non_hermitian.md"}, None, assume))
    gr = disc_groups(mu, rho)
    groups = gr["groups"]
    rec["discs"] = [{"index": i, "center": _cpair(mu[i]), "radius": float(rho), "group": gr["group_of"][i],
                     "group_size": len(groups[gr["group_of"][i]]), "isolated": len(groups[gr["group_of"][i]]) == 1}
                    for i in range(n)]
    q = len(gr["isolated"])
    if q == n:
        iso = (f"every disc meets no other disc: A has {n} distinct eigenvalues, exactly one in each disc, each simple, and A is "
               "diagonalizable")
    elif q:
        iso = (f"{q} of the discs meet no other disc, and each of them contains exactly one eigenvalue of A, which is simple")
    else:
        iso = "no disc is separated from all the others"
    statements.append(_stmt(
        "N4.counting", "CERTIFIED",
        f"the {n} discs form {len(groups)} group(s) with pairwise disjoint unions; the union of the discs of a group of m discs "
        f"contains exactly m eigenvalues of A counted with algebraic multiplicity; {iso}; {UNVERIFIED}",
        {"n_groups": len(groups), "group_sizes": sorted((len(g_) for g_ in groups), reverse=True)[:50],
         "isolated": gr["isolated"][:2000], "n_isolated": q, "diagonalizable": q == n,
         "separation_test": "max(|d Re|, |d Im|) (1 - 2u) > 2 rho", "basis": "Theorem 3 (c); docs/proofs/E-02_non_hermitian.md"},
        None, assume))
    return _finish(rec, statements)


def format_nonhermitian(rec: dict) -> str:
    lines = [f"non-Hermitian eigenproblem record ({rec['schema']}): {rec.get('description', '')}",
             f"  dimension {rec['operator']['dimension']}, {rec['claim']['n_pairs']} claimed pairs ({rec['claim']['kind']})",
             f"  {rec.get('unverified', UNVERIFIED)}"]
    for s in rec["statements"]:
        lines.append(f"  {s['id']:24s} {s['status']:13s} {s['summary']}")
    return "\n".join(lines)
