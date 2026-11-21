"""Certified moments and sum rules on a certified eigenspace (C-07, first part; 3.2).

For a certified cluster (S4: g stored vectors X, projector P of the exact eigenspace, sin theta bound s with
||Qhat Qhat^H - P||_2 <= s, eta = ||X^H X - I||_2) and a Hermitian operator O, the moments

    m_k = tr(P O H^k O) / g,   k = 0..K,

are enclosed (docs/proofs/C-07_moments.md): tr(X^H O H^k O X) is computed by applying O once and H k times to the
stored vectors (never forming H^k), with the rounding of every application accounted (Theorem 1 machinery: entrywise
bounds gamma_dot |H||z|, propagated with ||H||), and |tr(P M) - tr(X X^H M)| <= (2 g s + g eta) ||M||_2 for the Hermitian
M = O H^k O (trace-norm bound on P - X X^H, Hoelder), ||M||_2 <= ||O||^2 ||H||^k.

Sum rules (consistency statements, not new theorems): the zeroth moment equals tr(P O^2)/g, which is also certified
by Theorem 6 on the matrix O^2 (formed in floating point with its rounding bound) -- the two intervals must
overlap; the first moment obeys m_1 - tr(P H P O^2)/g = tr(P [O, [H, O]]) / (2 g) (f-sum rule form), where
tr(P H P O^2)/g lies in [a, b] * m_0 for the level interval [a, b] of the cluster, and the right-hand side is
certified by Theorem 6 on the matrix [O, [H, O]] (formed with its rounding bound) -- the two intervals must overlap.

Not implemented (open part, docs/theory.md section 9): continued-fraction / Lanczos bounds on the spectral function
built from the moments (statement M4, NOT_IMPLEMENTED).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import _jsonable, array_sha256, matrix_sha256
from .eigenspace import EigenspaceCertificate, certify_observable
from .fp import (U, UNDERFLOW_THRESHOLD, abs_up, add_up, exact_dot, exact_sum, fro_upper, gamma, gamma_dot, inf_norm_upper,
                 max_nnz_per_row, modulus_up, norm2_upper, sub_down, sub_up, underflow_scan, up)
from .precision import binary64, summed, widening_record
from .spec import canonical_json

MOMENTS_SCHEMA = "certify-ed/moments/1.0"
MOMENTS_STATEMENT_IDS = ("M1.moments", "M2.sum_rule_zeroth", "M3.sum_rule_first", "M4.continued_fraction")
STANDARD_MODEL = ("IEEE-754 binary64 round-to-nearest, standard model fl(a op b) = (a op b)(1 + delta), |delta| <= u, "
                  "no-underflow hypothesis (nonzero operands >= 2^-900); the S4 cluster record (s, eta) of the certificate")

__all__ = ["MOMENTS_SCHEMA", "MOMENTS_STATEMENT_IDS", "certified_moments", "moments_from_certificate", "format_moments",
           "hermitian_matrix_with_bound"]


def _vstmt(sid, status, summary, evidence=None, hypotheses=None, assumptions=None, reason_code=None):
    from .variational import _vstmt as _v
    return _v(sid, status, summary, evidence, hypotheses, assumptions, reason_code)


def _as_matrix(M, name, widened: Optional[list] = None):
    # a narrower format (float32, complex64, integers) is converted exactly to binary64, a wider one refused
    # (precision.py): every bound below counts binary64 arithmetic
    M = summed(binary64(M if sp.issparse(M) else np.asarray(M), name, widened), name)
    if sp.issparse(M):
        M = M.tocsr()
        D = (M - M.conj().T)
        herm = float(abs(D).max()) if D.nnz else 0.0
    else:
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
        herm = float(np.max(np.abs(M - M.conj().T))) if M.size else 0.0
    if herm != 0.0:
        raise ValueError(f"{name} is not Hermitian (max |{name} - {name}^H| = {herm:.3e})")
    return M


def _abs(M):
    """|M| with the moduli of complex entries rounded up (fp.abs_up, docs/proofs/fp_modulus-up.md)."""
    return abs_up(M)


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _interval_mul(lo1, hi1, lo2, hi2):
    c = [lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2]
    lo, hi = min(c), max(c)
    return float(lo - abs(lo) * 4 * U), float(hi + abs(hi) * 4 * U)


def _verdict(overlap: bool, *endpoints) -> str:
    """CERTIFIED when the two intervals of a sum rule overlap, FAILED when they are disjoint, INCONCLUSIVE when an
    endpoint is not finite (an overflowed bound): an overlap with an unbounded interval, or a disjointness found on a
    nan, is not a finding."""
    if not all(math.isfinite(v) for v in endpoints):
        return "INCONCLUSIVE"
    return "CERTIFIED" if overlap else "FAILED"


def _rule_text(overlap: bool, verdict: str) -> str:
    if verdict == "INCONCLUSIVE":
        return "cannot be compared (an endpoint is not finite, inf or nan: an overflowed bound)"
    return "overlap (consistent)" if overlap else "are DISJOINT: inconsistent inputs or bug"


def _rec_interval(lo, hi):
    """An interval for the record: [lo, hi], or [None, None] when an end is not finite (no nan in the record)."""
    return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) else [None, None]


def _rec_bound(v):
    """A bound for the record: v, or inf when it is not finite."""
    return v if math.isfinite(v) else math.inf


def _overlap(i1, i2) -> bool:
    return max(i1[0], i2[0]) <= min(i1[1], i2[1])


# ------------------------------------------------------------------------------------------------ matrices with bounds

def hermitian_matrix_with_bound(factors: Sequence[Any], widths: Sequence[int], coeff: float = 1.0):
    """fl(coeff * F_1 F_2 ... F_m) for stored matrices (dense/sparse), with an entrywise bound on |fl(product) - product|
    (Lemma 4 of the derivation: |fl(F_1...F_m) - F_1...F_m| <= gamma_M |F_1|...|F_m| entrywise, M the sum over the
    successive products of the inner-product lengths: the nonzeros per row of F_1 for the first product, n after).
    Returns (dense product, dense bound matrix E) with |fl - exact| <= E entrywise; coeff is applied to both (exact when
    it is a power of two)."""
    factors = [summed(binary64(F if sp.issparse(F) else np.asarray(F), "factor"), "factor") for F in factors]   # exact
    n = factors[0].shape[0]
    cplx = any(np.iscomplexobj(F.data if sp.issparse(F) else F) for F in factors)
    T = factors[0]
    P = _abs(factors[0])
    total = 0
    for j in range(1, len(factors)):
        w = int(widths[j - 1]) if j == 1 else n            # terms per entry: nonzeros per row of the left factor
        total += (2 * (w + 4)) if cplx else w
        T = T @ factors[j]
        P = P @ _abs(factors[j])
    T = _dense(T)
    P = _dense(P).astype(float)
    g = gamma(max(total, 1))
    # the computed |F_1|...|F_m| underestimates the exact one by at most a relative gamma_total; the exact entrywise error
    # of the product chain is <= gamma_total |F_1|...|F_m| (Higham Lemma 3.3, applied as in docs/proofs/E-05 Lemma 4)
    E = (up(g) / (1 - g)) * P
    return coeff * T, coeff * E


# ------------------------------------------------------------------------------------------------ moments

def certified_moments(H, O, X, s: float, eta: float, K: int = 2, level: Optional[Sequence[float]] = None,
                      cluster: Optional[dict] = None, sum_rules: bool = True, underflow_check: bool = True,
                      description: str = "") -> Dict[str, Any]:
    """Certified moments m_k = tr(P O H^k O)/g, k = 0..K, for the certified cluster with stored vectors X (n x g),
    sin theta bound s and Gram defect eta (S4 record), and the two sum-rule consistency checks (C-07 first part).

    H, O: stored Hermitian matrices (dense or scipy.sparse). level: the level interval [a, b] of the cluster (union of its
    enclosures), needed for the first-moment sum rule. Returns a JSON-able record (schema MOMENTS_SCHEMA)."""
    from .diagnostics import annotate_record
    widened: List[dict] = []
    H = _as_matrix(H, "H", widened)
    O = _as_matrix(O, "O", widened)
    X = binary64(np.asarray(X), "cluster vectors", widened)
    if X.ndim == 1:
        X = X[:, None]
    n, g = X.shape
    if H.shape[0] != n or O.shape[0] != n:
        raise ValueError(f"H ({H.shape}), O ({O.shape}) and X ({X.shape}) must share the dimension")
    if not (0 <= s < 1 and 0 <= eta < 1):
        raise ValueError(f"the cluster record must have 0 <= s < 1 and 0 <= eta < 1 (got s = {s}, eta = {eta})")
    K = int(K)
    if K < 0:
        raise ValueError("K must be >= 0")
    assume = [STANDARD_MODEL]
    statements: List[dict] = []
    rec: Dict[str, Any] = {"schema": MOMENTS_SCHEMA, "description": description or f"moments k = 0..{K} on a certified cluster of dimension {g}",
                           "operator": {"kind": "matrix", "dimension": int(n), "H_sha256": matrix_sha256(H), "O_sha256": matrix_sha256(O),
                                        "H_norm_upper": float(inf_norm_upper(H)), "O_norm_upper": float(inf_norm_upper(O)),
                                        "refers_to": "the stored matrices H and O; the projector P of the certificate's cluster"},
                           "state": {"form": "cluster", "g": int(g), "vectors_sha256": array_sha256(X), "sin_theta_bound": float(s),
                                     "eta": float(eta), "level": None if level is None else [float(level[0]), float(level[1])],
                                     "cluster": cluster}, "K": K}
    widening = widening_record(widened)
    if widening is not None:                 # only when an input was converted: binary64 inputs keep their record
        rec["input_widening"] = widening
    cplx = np.iscomplexobj(X) or np.iscomplexobj(H.data if sp.issparse(H) else H) or np.iscomplexobj(O.data if sp.issparse(O) else O)
    wH, wO = max_nnz_per_row(H), max_nnz_per_row(O)
    Hn, On = rec["operator"]["H_norm_upper"], rec["operator"]["O_norm_upper"]
    findings = []
    if underflow_check:
        findings = underflow_scan([("cluster.vectors", X, UNDERFLOW_THRESHOLD),
                                   ("H.entries", H.data if sp.issparse(H) else H, UNDERFLOW_THRESHOLD),
                                   ("O.entries", O.data if sp.issparse(O) else O, UNDERFLOW_THRESHOLD)])
    rec["underflow_check"] = {"enabled": bool(underflow_check), "findings": findings}
    if findings:
        from .certify import _underflow_statement_note
        note = _underflow_statement_note(findings)
        for sid in MOMENTS_STATEMENT_IDS[:3]:
            statements.append(_vstmt(sid, "INCONCLUSIVE", note, {"findings": findings}, reason_code="UNDERFLOW_RISK", assumptions=assume))
        statements.append(_m4(assume))
        rec["statements"] = statements
        annotate_record(rec)
        rec["record_sha256"] = _record_hash(rec)
        return rec

    # ---- M1: moments by repeated application
    gO = gamma_dot(wO, cplx)
    gH = gamma_dot(wH, cplx)
    gwO = gamma(wO)
    gwH = gamma(wH)
    absO, absH = _abs(O), _abs(H)
    Y = O @ X                                             # y_i = fl(O x_i)
    e_y = np.array([float(norm2_upper(gO * (absO @ modulus_up(X[:, i])) / (1 - gwO))) for i in range(g)])
    ynorm_up = np.array([float(norm2_upper(Y[:, i])) for i in range(g)])
    Z = Y.copy()
    D = np.zeros(g)                                       # ||z_k - H^k y_hat||_2 bounds
    moments = []
    proj_norm1 = up(2 * g * s + g * eta)                  # trace-norm bound on P - X X^H
    for k in range(K + 1):
        if k > 0:
            eps_k = np.array([float(norm2_upper(gH * (absH @ modulus_up(Z[:, i])) / (1 - gwH))) for i in range(g)])
            Z = H @ Z
            D = np.array([float(up(Hn * D[i] + eps_k[i])) for i in range(g)])
        vals, dots_err, abs_vals = [], [], []
        for i in range(g):
            v, e = exact_dot(Y[:, i], Z[:, i])
            vals.append(float(np.real(v)))
            dots_err.append(float(e))
            abs_vals.append(abs(float(np.real(v))))
        try:
            Hk = float(Hn ** k)
        except OverflowError:                             # a Python float power raises where numpy gives inf
            Hk = math.inf
        # Step 1 (O x rounding): ||M|| (2 ||y_hat|| + e_y) e_y with ||M|| = ||H||^k; step 2-3: ||y_hat|| D_k; step 4: exact_dot error
        fp_terms = sum(up(Hk * (2 * ynorm_up[i] + e_y[i]) * e_y[i]) + up(ynorm_up[i] * D[i]) + dots_err[i] for i in range(g))
        total = exact_sum(vals)                           # math.fsum's value; inf or nan where it would raise
        sum_rnd = gamma(g + 1) * exact_sum(abs_vals) + U * abs(total)
        m_hat = total / g
        proj_term = up(proj_norm1 * On * On * Hk)
        bound = float(up((proj_term + fp_terms + sum_rnd) / g) + U * abs(m_hat) * 2)
        moments.append({"k": k, "value": float(m_hat), "bound": bound,
                        "interval": [float(sub_down(m_hat, bound)), float(add_up(m_hat, bound))],
                        "terms": {"projector": float(proj_term / g), "rounding": float((fp_terms + sum_rnd) / g)},
                        "M_norm_upper": float(up(On * On * Hk))})
    def _finite(m):
        return all(math.isfinite(v) for v in (m["bound"], m["value"], *m["interval"]))
    finite = all(_finite(m) for m in moments)
    for m in moments:
        if not _finite(m):
            # no nan in the record: a bound that is not finite is stored as inf (a finite one as computed, when only an
            # end of the interval overflowed), with no interval; a nan value as None
            m["bound"] = m["bound"] if math.isfinite(m["bound"]) else math.inf
            m["interval"] = [None, None]
            m["value"] = None if math.isnan(m["value"]) else m["value"]
            m["terms"] = {t: (v if math.isfinite(v) else math.inf) for t, v in m["terms"].items()}
            m["M_norm_upper"] = m["M_norm_upper"] if math.isfinite(m["M_norm_upper"]) else math.inf
    rec["moments"] = moments
    if not finite:
        # a bound of the accounting is inf or nan (a product, a sum or a modulus overflowed): no moment interval
        # and no sum-rule verdict rests on it
        note = "the floating-point accounting of the moments is not finite (a bound of the accounting overflowed)"
        for sid in ("M1.moments", "M2.sum_rule_zeroth", "M3.sum_rule_first"):
            statements.append(_vstmt(sid, "INCONCLUSIVE", note, {"moments": moments}, assumptions=assume,
                                     reason_code="NONFINITE_BOUND"))
    else:
        text = ", ".join(f"m_{m['k']} in [{m['interval'][0]:.12g}, {m['interval'][1]:.12g}]" for m in moments)
        basis = ("trace-norm bound (2 g s + g eta) ||O||^2 ||H||^k on the projector error, "
                 "Theorem 1 style rounding of O x and of each H application; docs/proofs/C-07_moments.md")
        statements.append(_vstmt("M1.moments", "CERTIFIED", f"m_k = tr(P O H^k O)/g for k = 0..{K}: " + text,
                                 {"moments": moments, "basis": basis}, assumptions=assume))

    # ---- sum rules through Theorem 6 on explicitly formed matrices
    if not finite:
        pass                                              # M2 and M3 are stated above
    elif not sum_rules:
        for sid in ("M2.sum_rule_zeroth", "M3.sum_rule_first"):
            statements.append(_vstmt(sid, "NOT_CHECKED", "sum rules not requested", {}, reason_code="NARROWED_BY_USER", assumptions=assume))
    else:
        ec = EigenspaceCertificate(list(range(g)), [float("nan")] * g, float("nan"), float("nan"), float(eta), float(s),
                                   float(up(s + eta)))
        # O^2
        O2, E2 = hermitian_matrix_with_bound([O, O], [wO])
        O2h = (O2 + O2.conj().T) / 2
        dev2 = float(up(fro_upper(E2) + U * fro_upper(O2h)))          # ||O2h - O^2||_2 <= ||E2||_F + averaging rounding
        ob2 = certify_observable(ec, X, O2h, "O^2", O_norm=float(up(On * On + dev2)))     # ||O2h|| <= ||O||^2 + deviation
        lo2 = float(sub_down(ob2["value"], up(ob2["bound"] + g * dev2))) / g
        hi2 = float(add_up(ob2["value"], up(ob2["bound"] + g * dev2))) / g
        lo2 -= abs(lo2) * 4 * U
        hi2 += abs(hi2) * 4 * U
        m0 = moments[0]["interval"]
        ok0 = _overlap(m0, (lo2, hi2))
        v0 = _verdict(ok0, lo2, hi2, *m0)
        statements.append(_vstmt("M2.sum_rule_zeroth", v0,
                                 (f"m_0 in [{m0[0]:.12g}, {m0[1]:.12g}] (moment route) and tr(P O^2)/g in [{lo2:.12g}, {hi2:.12g}] "
                                  f"(Theorem 6 on the formed O^2): the intervals {_rule_text(ok0, v0)}"),
                                 {"moment_interval": m0, "theorem6_interval": _rec_interval(lo2, hi2), "overlap": ok0,
                                  "matrix_deviation_bound": _rec_bound(dev2), "note": "a consistency statement between two certified routes, not a new theorem"},
                                 assumptions=assume,
                                 reason_code="NONFINITE_BOUND" if v0 == "INCONCLUSIVE" else None))
        # f-sum rule: m_1 - tr(P H P O^2)/g = tr(P [O,[H,O]])/(2 g)
        if K < 1:
            statements.append(_vstmt("M3.sum_rule_first", "NOT_CHECKED", "needs K >= 1 (the first moment)", {},
                                     reason_code="NOT_APPLICABLE", assumptions=assume))
        elif level is None:
            statements.append(_vstmt("M3.sum_rule_first", "NOT_CHECKED", "needs the level interval [a, b] of the cluster (level=...)",
                                     {}, reason_code="NEEDS_DEPENDENCY", assumptions=assume))
        else:
            a, b = float(level[0]), float(level[1])
            OHO, E_oho = hermitian_matrix_with_bound([O, H, O], [wO, n])
            OOH, E_ooh = hermitian_matrix_with_bound([O, O, H], [wO, n])
            HOO, E_hoo = hermitian_matrix_with_bound([H, O, O], [wH, n])
            S1 = 2.0 * OHO - OOH
            C = S1 - HOO
            E_C = 2.0 * E_oho + E_ooh + E_hoo + U * (modulus_up(S1) + modulus_up(C)) * (1 + 4 * U)
            Ch = (C + C.conj().T) / 2
            devC = float(up(fro_upper(E_C) + U * fro_upper(Ch)))
            Cn = float(up(4 * On * On * Hn))
            obC = certify_observable(ec, X, Ch, "[O,[H,O]]", O_norm=float(up(Cn + devC)))
            loC = float(sub_down(obC["value"], up(obC["bound"] + g * devC))) / (2 * g)
            hiC = float(add_up(obC["value"], up(obC["bound"] + g * devC))) / (2 * g)
            loC -= abs(loC) * 4 * U
            hiC += abs(hiC) * 4 * U
            m1 = moments[1]["interval"]
            m0lo, m0hi = max(0.0, m0[0]), max(0.0, m0[1])
            plo, phi = _interval_mul(a, b, m0lo, m0hi)            # tr(P H P O^2)/g in [a, b] * m_0
            lhs = (float(sub_down(m1[0], phi)), float(sub_up(m1[1], plo)))        # [m1] - [plo, phi]
            ok1 = _overlap(lhs, (loC, hiC))
            v1 = _verdict(ok1, loC, hiC, *lhs)
            statements.append(_vstmt("M3.sum_rule_first", v1,
                                     (f"m_1 - tr(P H P O^2)/g in [{lhs[0]:.12g}, {lhs[1]:.12g}] (moment route; level [a, b] = [{a:.12g}, {b:.12g}]) and "
                                      f"tr(P [O,[H,O]])/(2g) in [{loC:.12g}, {hiC:.12g}] (Theorem 6 on the formed double commutator): the intervals "
                                      f"{_rule_text(ok1, v1)}"),
                                     {"lhs_interval": _rec_interval(*lhs), "rhs_interval": _rec_interval(loC, hiC), "overlap": ok1,
                                      "level": [a, b], "PHPO2_interval": _rec_interval(plo, phi),
                                      "matrix_deviation_bound": _rec_bound(devC),
                                      "note": "f-sum rule form; a consistency statement between two certified routes, not a new theorem"},
                                     assumptions=assume,
                                     reason_code="NONFINITE_BOUND" if v1 == "INCONCLUSIVE" else None))
    statements.append(_m4(assume))
    rec["statements"] = statements
    annotate_record(rec)
    rec["basis"] = ("m_k from the stored vectors with Theorem 1 style rounding and the trace-norm projector bound; sum rules by "
                    "Theorem 6 on formed matrices; docs/proofs/C-07_moments.md")
    rec["record_sha256"] = _record_hash(rec)
    return rec


def _m4(assume):
    return _vstmt("M4.continued_fraction", "NOT_CHECKED",
                  "continued-fraction (Lanczos) bounds on the spectral function from the certified moments are not implemented "
                  "(open part of C-07; docs/theory.md section 9, docs/SCOPE_MATRIX.md)", {}, reason_code="NOT_IMPLEMENTED",
                  assumptions=assume)


def _record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if not k.startswith("_") and k != "record_sha256"}
    return hashlib.sha256(canonical_json(_jsonable(body)).encode()).hexdigest()


def moments_from_certificate(cert, H, O, K: int = 2, members: Sequence[int] = (0,), **kw) -> Dict[str, Any]:
    """C-07 from a saved certificate with stored eigenvectors (Options(store_eigenvectors=True)) and a CERTIFIED S4
    cluster with the given members (Options(eigenspaces='all') for isolated levels); H is the matrix the certificate
    is about (its hash is compared with subject.operator.matrix_sha256 when present)."""
    from .davis_kahan import _cluster_record
    X = cert.arrays.get("eigenvectors")
    if X is None:
        raise ValueError("the certificate needs stored eigenvectors (Options(store_eigenvectors=True))")
    cl = _cluster_record(cert, members)
    mu = np.asarray(cert.arrays["eigenvalues"], dtype=float)[list(members)]
    rad = np.asarray(cert.arrays["radius"], dtype=float)[list(members)]
    level = (float(np.min(sub_down(mu, rad))), float(np.max(add_up(mu, rad))))
    rec = certified_moments(H, O, np.asarray(X)[:, list(members)], cl["sin_theta_bound"], cl["eta"], K=K, level=level,
                            cluster={"members": list(members), "claimed": cl.get("claimed"), "status": cl.get("status")}, **kw)
    # the record's H_sha256 is matrix_sha256 of the binary64 H it worked on (one copy of a narrower H, precision.py)
    stored = (cert.body.get("subject", {}).get("operator") or {}).get("matrix_sha256")
    note = None
    if stored is not None and stored != rec["operator"]["H_sha256"]:
        note = "WARNING: the matrix H differs from the certificate's stored operator (hash mismatch); the projector P is the certified operator's"
    from .crossings import certificate_hash
    rec["certificate"] = {"hash": certificate_hash(cert), "members": list(members), "operator_hash_matches": stored is None or note is None}
    if note:
        rec["certificate"]["warning"] = note
    rec["record_sha256"] = _record_hash(rec)
    return rec


def format_moments(rec: dict) -> str:
    lines = [f"moments record ({rec['schema']}): {rec.get('description', '')}",
             f"  dimension {rec['operator']['dimension']}, cluster g = {rec['state']['g']}, s = {rec['state']['sin_theta_bound']:.3e}, "
             f"eta = {rec['state']['eta']:.3e}, K = {rec['K']}"]
    for m in rec.get("moments", []):
        val = "nan" if m["value"] is None else f"{m['value']:.15g}"
        lines.append(f"  m_{m['k']} = {val} +/- {m['bound']:.3e}")
    for s in rec["statements"]:
        lines.append(f"  {s['id']:24s} {s['status']:13s} {s['summary'][:160]}")
    return "\n".join(lines)
