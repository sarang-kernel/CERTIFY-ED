"""Certified Chebyshev moments and bounds on the spectral distribution of an operator on a certified level (C-07,
second part; 3.3.0.dev0; docs/proofs/C-07b_spectral_bounds.md, UNVERIFIED until Sarang's check, D-12).

For a certified cluster of the stored Hermitian H (stored vectors X, n x g; sin theta bound s; Gram defect eta; the
spectral projector P of the cluster) and a stored Hermitian O, the spectral measure of O on the cluster,

    mu(S) = tr(P O E(S) O) / g      (E the spectral measure of H),

is a positive measure of total mass m_0 = tr(P O^2)/g carried by the spectrum of H; for a ground state it is the
dynamical response of O (the structure factor when O is a density at a wave vector). The spectrum of H lies in an
interval [lo, hi] (Gershgorin discs, narrowed by proved bounds on the extreme eigenvalues, e.g. the lowest one from a
certificate), and binary64 numbers c and s with (lo - c) s >= -1 and (hi - c) s <= 1 (checked exactly) map it into
[-1, 1]: A = (H - c I) s has ||A||_2 <= 1. The Chebyshev moments

    tau_j = tr(P O T_j(A) O) / g,   j = 0..K,

are enclosed (M5): the three-term recurrence z_{j+1} = 2 A z_j - z_{j-1} is applied to the vectors O x_i with
A z formed as fl(s (fl(H z) - fl(c z))), the rounding of every step is propagated through the Chebyshev polynomials of
the second kind (||U_m(A)|| <= m + 1), and the projector term is (2 g s + g eta) ||O||^2, the same for every j because
||T_j(A)|| <= 1. For an energy E the distribution F(E) = mu((-inf, E]) is then bounded (M6) by the integrals of two
polynomials p and q of degree at most K with q <= 1_{(-inf, xi]} <= p on [-1, 1], xi = (E - c) s: F(E) <= sum_j d_j
tau_j for p = sum_j d_j T_j, and F(E) >= the same sum for q. p and q are chosen by a heuristic (the Hermite
interpolant of the step at the Gauss-Radau nodes of the moments through xi, else a linear program on a point set):
their sign conditions are proved in exact integer arithmetic (Bernstein coefficients on an adaptive subdivision,
rounded only downwards) before a bound is reported, and the sums over the moment intervals are evaluated exactly and
rounded outward. A polynomial whose proof fails gives the trivial bound (0 below, the upper end of tau_0 above). F is
nondecreasing, so the lower bounds at a list of energies are raised to their running maximum and the upper bounds
lowered to their running minimum from the right.
"""
from __future__ import annotations

import copy
import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .certificate_v2 import array_sha256, matrix_sha256
from .fp import (U, UNDERFLOW_THRESHOLD, abs_up, add_up, exact_dot, exact_sum, gamma, gamma_dot, inf_norm_upper,
                 max_nnz_per_row, modulus_up, norm2_upper, sub_down, underflow_scan, up)
from .moments import (CLUSTER_RECORD, MOMENTS_SCHEMA, STANDARD_MODEL, _as_matrix, _gram_defect_bounds, _rec_bound,
                      _record_hash, _sum_bounds, _vstmt)
from .precision import binary64, widening_record

# named in the assumptions of every M5 and M6 statement (rule D-12: new mathematics is marked until it is re-derived)
DERIVATION = ("derived in docs/proofs/C-07b_spectral_bounds.md; UNVERIFIED until an independent re-derivation "
              "(D-12)")

SPECTRAL_STATEMENT_IDS = ("M5.chebyshev_moments", "M6.distribution_bounds")
_DYADIC_BITS = 40              # the coefficients of p and q are rounded to multiples of 2^-40 before their proof
_MAX_DEPTH = 40                # bisection depth of the Bernstein proof
_PREC = 64                     # bits kept below the integer polynomial's unit in the Bernstein coefficients

__all__ = ["SPECTRAL_STATEMENT_IDS", "chebyshev_interval", "certified_chebyshev_moments", "distribution_bounds",
           "spectral_bounds_from_certificate", "format_spectral"]


# ------------------------------------------------------------------------------------------------ shift and scaling

def _gershgorin(H):
    """(lo, hi): every eigenvalue of the stored Hermitian H lies in [lo, hi] (Gershgorin discs, rounded outward).

    The off-diagonal part H - diag(H) is exact (a stored diagonal entry minus itself is 0); its row sums of upper
    moduli are computed in floating point, and for nonnegative terms the exact sum is at most fl(sum)/(1 - gamma_w)."""
    H = H.tocsr() if sp.issparse(H) else sp.csr_matrix(np.asarray(H))
    dg = np.asarray(H.diagonal())
    if np.iscomplexobj(dg) and np.any(np.imag(dg) != 0):
        raise ValueError("the diagonal of a Hermitian matrix must be real")
    d = np.real(dg).astype(float)
    off = (H - sp.diags(dg, format="csr")).tocsr()
    off.eliminate_zeros()
    w = max(max_nnz_per_row(off), 1)
    rows = np.asarray(abs_up(off).sum(axis=1)).ravel() if off.nnz else np.zeros(H.shape[0])
    R = np.array([up(x / (1.0 - gamma(w))) for x in rows]) if off.nnz else rows
    return float(np.min(sub_down(d, R))), float(np.max(add_up(d, R)))


def chebyshev_interval(H, lower: Optional[float] = None, upper: Optional[float] = None) -> Dict[str, Any]:
    """The interval [lo, hi] holding the spectrum of H and the map A = (H - c I) s onto [-1, 1].

    lo is the larger of the Gershgorin bound and `lower`, hi the smaller of the Gershgorin bound and `upper`; `lower`
    and `upper` must be proved bounds on the extreme eigenvalues (the caller's responsibility; from a certificate in
    spectral_bounds_from_certificate). c and s are binary64 numbers with (lo - c) s >= -1 and (hi - c) s <= 1, checked
    in exact rational arithmetic, so the spectrum of the exact A lies in [-1, 1]. A matrix in a narrower format is
    converted exactly to binary64 and its repeated entries are summed first, as certified_chebyshev_moments does."""
    glo, ghi = _gershgorin(_as_matrix(H, "H"))
    lo = glo if lower is None else max(glo, float(lower))
    hi = ghi if upper is None else min(ghi, float(upper))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"the spectral interval [{lo}, {hi}] is not finite")
    if hi < lo:
        raise ValueError(f"the bounds contradict each other: lower {lo!r} > upper {hi!r}")
    if hi - lo < max(abs(lo), abs(hi), 1.0) * 2.0 ** -40:      # (nearly) a multiple of the identity: widen
        pad = max(abs(lo), abs(hi), 1.0) * 2.0 ** -40
        lo, hi = float(lo - pad), float(hi + pad)
    c = 0.5 * lo + 0.5 * hi
    flo, fhi, fc = Fraction(lo), Fraction(hi), Fraction(c)
    R = max(fc - flo, fhi - fc)                         # > 0: the rounded midpoint lies in [lo, hi]
    S = 1 / R
    sc = float(S)
    if Fraction(sc) > S:                                # rounded up: one step down gives s <= 1/R
        sc = float(np.nextafter(sc, 0.0))
    fs = Fraction(sc)
    if not (sc > 0 and (flo - fc) * fs >= -1 and (fhi - fc) * fs <= 1):
        raise ValueError("no binary64 scale maps the spectral interval into [-1, 1]")
    return {"center": c, "scale": sc, "interval": [lo, hi], "gershgorin": [glo, ghi]}


# ------------------------------------------------------------------------------------------------ M5

def certified_chebyshev_moments(H, O, X, s: float, eta: float, K: int = 32, spectrum_lower: Optional[float] = None,
                                spectrum_upper: Optional[float] = None, spectrum_source: str = "",
                                cluster: Optional[dict] = None, underflow_check: bool = True,
                                description: str = "") -> Dict[str, Any]:
    """M5: enclosures of tau_j = tr(P O T_j((H - c I) s_c) O)/g, j = 0..K, for the certified cluster with stored
    vectors X (n x g), sin theta bound s and Gram defect eta. spectrum_lower/upper are optional proved bounds on the
    extreme eigenvalues of H (they narrow the Gershgorin interval; spectrum_source names where they come from).
    Returns a JSON-able record (schema certify-ed/moments/1.0) with the key 'chebyshev' (c, s_c, the interval and the
    moments) for distribution_bounds."""
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
    H = H.tocsr() if sp.issparse(H) else sp.csr_matrix(H)
    iv = chebyshev_interval(H, spectrum_lower, spectrum_upper)
    c, sc = iv["center"], iv["scale"]
    O_sha = matrix_sha256(O)
    O = O.tocsr() if sp.issparse(O) else sp.csr_matrix(O)
    declared = spectrum_lower is not None or spectrum_upper is not None
    if declared and not spectrum_source:
        spectrum_source = "bounds declared by the caller, not checked"
    assume = [STANDARD_MODEL, CLUSTER_RECORD, "the spectrum of H lies in the interval of the record (Gershgorin discs"
              + (f"; {spectrum_source}" if spectrum_source else "") + ")", DERIVATION]
    rec: Dict[str, Any] = {
        "schema": MOMENTS_SCHEMA,
        "description": description or f"Chebyshev moments j = 0..{K} on a certified cluster of dimension {g}",
        "operator": {"kind": "matrix", "dimension": int(n), "H_sha256": matrix_sha256(H), "O_sha256": O_sha,
                     "O_norm_upper": float(inf_norm_upper(O)),
                     "refers_to": "the stored matrices H and O; the projector P of the certificate's cluster"},
        "state": {"form": "cluster", "g": int(g), "vectors_sha256": array_sha256(X), "sin_theta_bound": float(s),
                  "eta": float(eta), "cluster": cluster},
        "chebyshev": {"center": c, "scale": sc, "K": K, "spectrum_interval": iv["interval"],
                      "gershgorin_interval": iv["gershgorin"],
                      "spectrum_source": spectrum_source or "Gershgorin discs of the stored H",
                      "declared_bounds": {"lower": None if spectrum_lower is None else float(spectrum_lower),
                                          "upper": None if spectrum_upper is None else float(spectrum_upper)}}}
    widening = widening_record(widened)
    if widening is not None:
        rec["input_widening"] = widening
    statements: List[dict] = []
    findings = []
    if underflow_check:
        findings = underflow_scan([("cluster.vectors", X, UNDERFLOW_THRESHOLD),
                                   ("H.entries", H.data, UNDERFLOW_THRESHOLD),
                                   ("O.entries", O.data, UNDERFLOW_THRESHOLD)])
    rec["underflow_check"] = {"enabled": bool(underflow_check), "findings": findings}
    if findings:
        from .certify import _underflow_statement_note
        note = _underflow_statement_note(findings)
        statements.append(_vstmt("M5.chebyshev_moments", "INCONCLUSIVE", note, {"findings": findings},
                                 reason_code="UNDERFLOW_RISK", assumptions=assume))
        return _finish(rec, statements, annotate_record)

    eta_lower, eta_upper = _gram_defect_bounds(X)
    consistent = float(eta) >= eta_lower
    rec["state"]["eta_check"] = {"lower_bound_from_vectors": _rec_bound(eta_lower),
                                 "upper_bound_from_vectors": _rec_bound(eta_upper), "consistent": bool(consistent)}
    if not consistent:
        if not eta_upper < 1.0:
            raise ValueError(f"the eta of the cluster record ({float(eta):.3e}) is below the Gram defect of the stored "
                             f"vectors (at least {eta_lower:.3e}), whose bound {eta_upper:.3e} is not below 1")
        eta = eta_upper
    rec["state"]["eta_used"] = float(eta)

    cplx = np.iscomplexobj(X) or np.iscomplexobj(H.data) or np.iscomplexobj(O.data)
    wH, wO = max_nnz_per_row(H), max_nnz_per_row(O)
    gO, gwO = gamma_dot(wO, cplx), gamma(wO)
    gA, gwA = gamma_dot(wH + 2, cplx), gamma(wH + 4)
    absO, absH = abs_up(O), abs_up(H)
    absc = abs(c)
    On = rec["operator"]["O_norm_upper"]

    def apply_A(Z):
        # fl(s_c (fl(H z) - fl(c z))): a sum of at most w + 1 products per component, then one multiplication
        return ((H @ Z) - c * Z) * sc

    def matvec_error(Z):
        # || fl(A z) - A z ||_2 for each column z, A = (H - c I) s_c (Lemma 7 of C-07b): per component at most
        # s_c gamma_dot(w + 2) (|H| |z| + |c| |z|); the bound vector is formed from nonnegative terms in at most w + 4
        # operations, covered by the division by 1 - gamma_{w+4}; the roundings of that division and of the two gammas
        # (at most four relative errors u) are covered by up() of the 2-norm (a factor 1 + 8u, rounded once)
        out = []
        for i in range(g):
            m = modulus_up(Z[:, i])
            out.append(float(up(norm2_upper(gA * sc * (absH @ m + absc * m) / (1 - gwA)))))
        return np.array(out)

    Y = O @ X
    # the divisor 1 - gamma_{w_O} covers the w_O roundings of fl(|O| m); the product by gamma_dot, the two rounded
    # gammas, 1 - gamma and the division (at most five relative errors u) are covered by up() (at least 1 + 6u)
    e_y = np.array([float(up(norm2_upper(gO * (absO @ modulus_up(X[:, i])) / (1 - gwO)))) for i in range(g)])
    ynorm = np.array([float(norm2_upper(Y[:, i])) for i in range(g)])
    proj = float(up(up(2 * g * s + g * eta) * On * On))      # (2gs + g eta) ||O||^2 ||T_j(Hs)||, ||T_j(Hs)|| <= 1
    step1 = np.array([float(up((2 * ynorm[i] + e_y[i]) * e_y[i])) for i in range(g)])
    S = np.zeros(g)                                           # sum of the local errors delta_1..delta_j
    E = np.zeros(g)                                           # E_j = sum_i (j - i + 1) delta_i >= ||z_j - T_j y||
    Zm, Z = None, Y.copy()
    moments = []
    for j in range(K + 1):
        if j == 1:
            Zn = apply_A(Z)
            delta = matvec_error(Z)
            Zm, Z = Z, Zn
        elif j >= 2:
            W = apply_A(Z)
            eps = matvec_error(Z)
            Zn = 2.0 * W - Zm                                 # 2 W exact; one rounding of the subtraction
            zn = np.array([float(norm2_upper(Zn[:, i])) for i in range(g)])
            delta = np.array([float(up(2.0 * eps[i] + (U / (1 - U)) * zn[i])) for i in range(g)])
            Zm, Z = Z, Zn
        if j >= 1:
            S = np.array([float(up(S[i] + delta[i])) for i in range(g)])
            E = np.array([float(up(E[i] + S[i])) for i in range(g)])
        vals, errs, absv = [], [], []
        for i in range(g):
            v, e = exact_dot(Y[:, i], Z[:, i])
            vals.append(float(np.real(v)))
            errs.append(float(e))
            absv.append(abs(float(np.real(v))))
        # the three terms of each member are added upward (add_up); their sum over the members is correctly rounded
        # (math.fsum: one relative error u, covered by the up() of the bound; inf when it overflows)
        fp_terms = _sum_bounds(float(add_up(add_up(step1[i], float(up(ynorm[i] * E[i]))), errs[i])) for i in range(g))
        total = exact_sum(vals)
        sum_rnd = gamma(g + 1) * exact_sum(absv) + U * abs(total)
        t_hat = total / g
        bound = float(add_up(up((proj + fp_terms + sum_rnd) / g), 2 * U * abs(t_hat)))   # the last addition upward
        lo, hi = float(np.nextafter(t_hat - bound, -np.inf)), float(np.nextafter(t_hat + bound, np.inf))
        moments.append({"j": j, "value": float(t_hat), "bound": bound, "interval": [lo, hi],
                        "terms": {"projector": float(proj / g), "rounding": float((fp_terms + sum_rnd) / g)}})
    finite = all(math.isfinite(v) for m in moments for v in (m["value"], m["bound"], *m["interval"]))
    if not finite:
        for m in moments:
            if not all(math.isfinite(v) for v in (m["value"], m["bound"], *m["interval"])):
                m["interval"] = [None, None]
                m["bound"] = m["bound"] if math.isfinite(m["bound"]) else math.inf
                m["value"] = m["value"] if math.isfinite(m["value"]) else None
        rec["chebyshev"]["moments"] = moments
        statements.append(_vstmt("M5.chebyshev_moments", "INCONCLUSIVE",
                                 "the floating-point accounting of the Chebyshev moments is not finite",
                                 {"moments": moments}, reason_code="NONFINITE_BOUND", assumptions=assume))
        return _finish(rec, statements, annotate_record)
    rec["chebyshev"]["moments"] = moments
    worst = max(m["bound"] for m in moments)
    statements.append(_vstmt(
        "M5.chebyshev_moments", "CERTIFIED",
        (f"tau_j = tr(P O T_j((H - c I) s) O)/g for j = 0..{K} (c = {c!r}, s = {sc!r}; spectrum in "
         f"[{iv['interval'][0]:.10g}, {iv['interval'][1]:.10g}]) enclosed; largest half-width {worst:.3e}; tau_0 in "
         f"[{moments[0]['interval'][0]:.12g}, {moments[0]['interval'][1]:.12g}]"),
        {"moments": moments, "center": c, "scale": sc, "spectrum_interval": iv["interval"],
         "basis": ("projector term (2 g s + g eta) ||O||^2 (||T_j(A)|| <= 1); rounding of the recurrence propagated by "
                   "the Chebyshev polynomials of the second kind; docs/proofs/C-07b_spectral_bounds.md")},
        assumptions=assume))
    return _finish(rec, statements, annotate_record)


def _finish(rec, statements, annotate_record):
    rec["statements"] = statements
    annotate_record(rec)
    rec["basis"] = "Chebyshev moments and distribution bounds; docs/proofs/C-07b_spectral_bounds.md"
    rec["record_sha256"] = _record_hash(rec)
    return rec


# ------------------------------------------------------------------------------------------------ exact polynomial checks

def _cheb_monomials(K: int) -> List[List[int]]:
    """Integer monomial coefficients of T_0..T_K (T[j][k] is the coefficient of u^k in T_j)."""
    T = [[1], [0, 1]]
    for j in range(2, K + 1):
        a = [0] + [2 * x for x in T[j - 1]]
        b = T[j - 2] + [0] * (len(a) - len(T[j - 2]))
        T.append([x - y for x, y in zip(a, b)])
    return T[:K + 1]


def _bernstein_int(A: Sequence[int]) -> List[int]:
    """Integers b_i <= 2^_PREC * beta_i, where beta_0..beta_m are the Bernstein coefficients on [0, 1] of
    t -> A(2t - 1) for the integer polynomial A of degree at most m = len(A) - 1 (ascending coefficients).

    Q(t) = A(2t - 1) has integer coefficients (Horner); beta_i = sum_{k <= i} C(i, k)/C(m, k) Q_k
    = N_i / C(m, i) with N_i = sum_{k <= i} C(m - k, i - k) Q_k an integer, and b_i = floor(2^_PREC N_i / C(m, i))."""
    m = len(A) - 1
    Q = [0] * (m + 1)
    deg = -1
    for coef in reversed(A):              # Q <- Q * (2t - 1) + coef
        R = [0] * (m + 1)
        for k in range(deg + 1):
            q = Q[k]
            if q:
                R[k] -= q
                R[k + 1] += 2 * q
        R[0] += coef
        Q = R
        deg += 1
    out = []
    for i in range(m + 1):
        N = sum(math.comb(m - k, i - k) * Q[k] for k in range(i + 1))
        out.append((N << _PREC) // math.comb(m, i))          # // floors, also for negative N
    return out


def _split(b: Sequence[int], p: int, q: int):
    """de Casteljau at t = p / 2^q (0 <= p <= 2^q): Bernstein coefficients of the two parts [0, t] and [t, 1], every
    intermediate value rounded down (>> floors). The weights are nonnegative, so every rounded value is at most the
    value from exact arithmetic on the same input, and the represented polynomials are lowered, never raised."""
    w = (1 << q) - p
    left, right = [b[0]], [b[-1]]
    cur = list(b)
    for _ in range(len(b) - 1):
        cur = [(w * cur[i] + p * cur[i + 1]) >> q for i in range(len(cur) - 1)]
        left.append(cur[0])
        right.append(cur[-1])
    right.reverse()
    return left, right


def _nonneg_bernstein(b: Sequence[int], max_depth: int = _MAX_DEPTH) -> bool:
    """Proof that the polynomial with (lowered) Bernstein coefficients b on [0, 1] is >= 0 there: all coefficients of
    a piece >= 0 proves the piece (the Bernstein basis is nonnegative and sums to 1); pieces are bisected."""
    stack = [(list(b), 0)]
    while stack:
        c, depth = stack.pop()
        if min(c) >= 0:
            continue
        if c[0] < 0 or c[-1] < 0 or depth >= max_depth:      # negative (lowered) value at an end, or too deep
            return False
        lo, hi = _split(c, 1, 1)
        stack.append((hi, depth + 1))
        stack.append((lo, depth + 1))
    return True


def _prove_majorant(D: Sequence[int], T: List[List[int]], xi: Fraction, upper: bool, hints=None) -> bool:
    """For p(u) = 2^-B sum_j D_j T_j(u) and a dyadic xi in [-1, 1): upper -> p >= 1 on [-1, xi] and p >= 0 on [xi, 1];
    lower -> p <= 1 on [-1, xi] and p <= 0 on [xi, 1]. Exact integer arithmetic (rounding only downwards)."""
    K = len(D) - 1
    A = [0] * (K + 1)
    for j, dj in enumerate(D):
        if dj:
            for k, t in enumerate(T[j]):
                A[k] += dj * t
    one = 1 << _DYADIC_BITS
    sign = 1 if upper else -1
    left = [sign * x for x in A]
    left[0] -= sign * one                 # upper: A - 2^B >= 0; lower: 2^B - A >= 0
    right = [sign * x for x in A]         # upper: A >= 0; lower: -A >= 0
    t = (xi + 1) / 2                      # the split point in [0, 1)
    q = t.denominator.bit_length() - 1
    if not (Fraction(-1) <= xi < 1) or t.denominator != 1 << q:
        raise ValueError(f"the energy maps to u = {xi} outside [-1, 1) or not dyadic")
    p = t.numerator
    xf = float(xi)
    h = np.sort(np.asarray(hints if hints is not None else [], dtype=float))
    # the touching points as positions within each part (s in (0, 1)); they only choose where to cut first
    cuts_l = [(v + 1.0) / (xf + 1.0) for v in h if -1.0 < v < xf] if xf > -1.0 else []
    cuts_r = [(v - xf) / (1.0 - xf) for v in h if xf < v < 1.0]
    ok_left = _nonneg_pieces(_split(_bernstein_int(left), p, q)[0], cuts_l)
    return ok_left and _nonneg_pieces(_split(_bernstein_int(right), p, q)[1], cuts_r)


def _nonneg_pieces(b: Sequence[int], cuts: Sequence[float], bits: int = 30) -> bool:
    """_nonneg_bernstein after cutting [0, 1] first near the given points (each cut rounded to a multiple of 2^-bits of
    the remaining part; any cut is valid, Lemma 8 (c)): a polynomial that touches its bound at the cuts then needs few
    bisections."""
    cur, done = list(b), 0.0
    for s in sorted(cuts):
        if not done < s < 1.0:
            continue
        loc = (s - done) / (1.0 - done)
        pk = int(round(loc * (1 << bits)))
        if pk <= 0 or pk >= (1 << bits):
            continue
        lo, cur = _split(cur, pk, bits)
        if not _nonneg_bernstein(lo):
            return False
        done = done + (pk / (1 << bits)) * (1.0 - done)
    return _nonneg_bernstein(cur)


# ------------------------------------------------------------------------------------------------ M6

_CAP = 1.0                     # the LP keeps p <= 1 + _CAP (upper) and q >= -_CAP (lower) on its points
_EXCHANGE_ROUNDS = 8
_GRID_FACTOR = 16              # initial LP points: max(512, _GRID_FACTOR (K + 1)) Chebyshev points
_LP_METHOD = "highs"
_ATTEMPTS = 12                 # proof attempts, the margin doubled after each failure


def _scan_points(xi: float, fine: int = 20001) -> np.ndarray:
    """Points of [-1, 1] for the floating-point scans: uniform in u, uniform in arccos u, and dense near xi."""
    th = np.linspace(0.0, np.pi, fine)
    near = np.concatenate([xi + np.logspace(-10, -1, 90), xi - np.logspace(-10, -1, 90), [xi]])
    pts = np.unique(np.concatenate([np.linspace(-1.0, 1.0, fine), np.cos(th), near]))
    return pts[(pts >= -1.0) & (pts <= 1.0)]


def _gap(d: np.ndarray, pts: np.ndarray, xi: float, upper: bool) -> np.ndarray:
    vals = np.polynomial.chebyshev.chebval(pts, d)
    target = (pts <= xi).astype(float)
    return (vals - target) if upper else (target - vals)


def _lp_polynomial(tau_hat: np.ndarray, xi: float, upper: bool, grid: Optional[int] = None):
    """Chebyshev coefficients of a near-optimal polynomial for the bound at xi (a heuristic: scipy linprog on a point
    set, enlarged by exchange rounds with the most violated local minima of a fine scan). The polynomial is kept
    between -_CAP and 1 + _CAP on the points, which bounds its coefficients when the moments leave directions free."""
    from scipy.optimize import linprog
    K = len(tau_hat) - 1
    m = int(grid) if grid else max(512, _GRID_FACTOR * (K + 1))
    k = np.arange(m)
    pts = np.cos(np.pi * (k + 0.5) / m)
    near = xi + np.concatenate([-np.logspace(-6, -1, 40), np.logspace(-6, -1, 40)])
    pts = np.unique(np.clip(np.concatenate([pts, near, [xi, -1.0, 1.0]]), -1.0, 1.0))
    scan = _scan_points(xi)
    tol = 1e-8
    d = None
    for _ in range(_EXCHANGE_ROUNDS):
        V = np.polynomial.chebyshev.chebvander(pts, K)
        target = (pts <= xi).astype(float)
        if upper:      # min tau.d  s.t.  target <= V d <= 1 + cap
            A = np.vstack([-V, V])
            b = np.concatenate([-target, np.full(len(pts), 1.0 + _CAP)])
            res = linprog(tau_hat, A_ub=A, b_ub=b, bounds=[(-1e3, 1e3)] * (K + 1), method=_LP_METHOD)
        else:          # max tau.d  s.t.  -cap <= V d <= target
            A = np.vstack([V, -V])
            b = np.concatenate([target, np.full(len(pts), _CAP)])
            res = linprog(-tau_hat, A_ub=A, b_ub=b, bounds=[(-1e3, 1e3)] * (K + 1), method=_LP_METHOD)
        if res.status != 0:
            return d
        d = res.x
        g = _gap(d, scan, xi, upper)
        if float(np.min(g)) >= -tol:
            break
        # the most violated local minima of the gap on the scan join the point set
        loc = np.where((g[1:-1] <= g[:-2]) & (g[1:-1] <= g[2:]) & (g[1:-1] < -tol))[0] + 1
        ends = [i for i in (0, len(g) - 1) if g[i] < -tol]
        idx = np.concatenate([loc, np.array(ends, dtype=int)])
        idx = idx[np.argsort(g[idx])][:4 * (K + 1)]
        pts = np.unique(np.concatenate([pts, scan[idx]]))
    return d


def _dyadic(d: np.ndarray) -> List[int]:
    return [int(round(float(x) * 2.0 ** _DYADIC_BITS)) for x in d]


def _float_violation(D: Sequence[int], xi: float, upper: bool) -> float:
    """Largest violation of the sign conditions, in floating point (to choose the margin; a heuristic): the scan
    points, then three nested refinements around the 64 lowest local minima of the gap on the scan."""
    d = np.asarray(D, dtype=float) / 2.0 ** _DYADIC_BITS
    scan = _scan_points(xi)
    g = _gap(d, scan, xi, upper)
    worst = float(np.min(g))
    loc = np.where((g[1:-1] <= g[:-2]) & (g[1:-1] <= g[2:]))[0] + 1
    for i in loc[np.argsort(g[loc])][:64]:
        a, b = scan[i - 1], scan[i + 1]
        for _ in range(3):
            pts = np.linspace(a, b, 65)
            gg = _gap(d, pts, xi, upper)
            j = int(np.argmin(gg))
            worst = min(worst, float(gg[j]))
            a, b = pts[max(j - 1, 0)], pts[min(j + 1, 64)]
    return max(0.0, -worst)


def _bound_from(D: Sequence[int], intervals, upper: bool) -> float:
    """Exact sum_j d_j tau_j over the moment intervals (the worst end for each j), rounded outward."""
    tot = Fraction(0)
    for dj, (lo, hi) in zip(D, intervals):
        if dj == 0:
            continue
        end = hi if (dj > 0) == upper else lo
        tot += Fraction(dj) * Fraction(end)
    tot /= (1 << _DYADIC_BITS)
    f = float(tot)
    if upper and Fraction(f) < tot:
        f = float(np.nextafter(f, np.inf))
    if not upper and Fraction(f) > tot:
        f = float(np.nextafter(f, -np.inf))
    return f


def _jacobi_from_moments(tau_hat: np.ndarray, tol: float = 1e-11):
    """Recurrence coefficients (alpha_k, beta_k) of the monic orthogonal polynomials of the measure with Chebyshev
    moments tau_hat (a heuristic in floating point: the modified Chebyshev algorithm, Gautschi 2004, Algorithm 2.1,
    with the monic Chebyshev polynomials p_k = T_k / 2^(k-1) as the reference, a_k = 0, b_1 = 1/2, b_k = 1/4).
    Stops at the first beta_k <= tol (a measure with few atoms) or a nonfinite value."""
    K = len(tau_hat) - 1
    N = (K + 1) // 2
    if N < 1 or not tau_hat[0] > 0:
        return [], []
    m = np.array([tau_hat[0]] + [tau_hat[k] / 2.0 ** (k - 1) for k in range(1, 2 * N)], dtype=float)
    b = np.full(2 * N, 0.25)
    b[0], b[1] = 0.0, 0.5
    alpha, beta = [m[1] / m[0]], [m[0]]
    sig_prev, sig = np.zeros(2 * N), m.copy()
    for k in range(1, N):
        new = np.zeros(2 * N)
        ls = np.arange(k, 2 * N - k)
        new[ls] = sig[ls + 1] - alpha[k - 1] * sig[ls] - beta[k - 1] * sig_prev[ls] + b[ls] * sig[ls - 1]
        if not (np.isfinite(new[k]) and new[k] > 0 and np.isfinite(new[k + 1])):
            break
        a_k = new[k + 1] / new[k] - sig[k] / sig[k - 1]
        b_k = new[k] / sig[k - 1]
        if not (np.isfinite(a_k) and b_k > tol):
            break
        alpha.append(a_k)
        beta.append(b_k)
        sig_prev, sig = sig, new
    return alpha, beta


def _radau_hermite(tau_hat: np.ndarray, xi: float, upper: bool, n: Optional[int] = None, jacobi=None,
                   return_nodes: bool = False):
    """Chebyshev coefficients of the Hermite interpolant of the step at the Gauss-Radau nodes through xi (a heuristic):
    p(x_k) = 1 left of xi, 0 right of it, p'(x_k) = 0 at the n free nodes, and p(xi) = 1 (upper) or 0 (lower).
    For any distinct nodes such a p of degree 2n lies above (below) the step on the whole line (C-07b, Remark); the
    Radau nodes make it the Chebyshev-Markov-Stieltjes bound. n defaults to the largest the moments allow. None when
    the construction breaks down."""
    from scipy.linalg import eigh_tridiagonal
    alpha, beta = jacobi if jacobi is not None else _jacobi_from_moments(tau_hat)
    n_all = len(alpha) - 1
    n = n_all if n is None else min(int(n), n_all)
    if n < 1:
        return None
    r = xi - alpha[0]                                   # r_k = pi_k(xi) / pi_{k-1}(xi)
    for k in range(1, n):
        if r == 0:
            return None
        r = (xi - alpha[k]) - beta[k] / r
    if not (np.isfinite(r) and abs(r) > 1e-300):
        return None
    a_t = xi - beta[n] / r
    d = np.array(alpha[:n] + [a_t], dtype=float)
    e = np.sqrt(np.array(beta[1:n + 1], dtype=float))
    try:
        nodes = eigh_tridiagonal(d, e, eigvals_only=True)
    except Exception:
        return None
    j = int(np.argmin(np.abs(nodes - xi)))
    if abs(nodes[j] - xi) > 1e-8 * max(1.0, abs(xi)):
        return None
    x = np.sort(np.delete(nodes, j))
    d = _hermite_step(x, xi, upper, len(tau_hat))
    return (d, x) if return_nodes else d


def _hermite_step(x: np.ndarray, xi: float, upper: bool, ncoef: int):
    """Chebyshev coefficients (length ncoef) of the polynomial p of degree 2n with p(x_k) = 1 for x_k < xi, 0 for
    x_k > xi, p'(x_k) = 0 at the n distinct free nodes x, and p(xi) = 1 (upper) or 0 (lower); a heuristic in floating
    point (rows equilibrated). By the Remark of C-07b it lies above (below) the step for any such nodes. None when the
    nodes are not distinct from each other and from xi, the degree exceeds ncoef - 1, or the solve fails."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    deg = 2 * n
    if n < 1 or deg + 1 > ncoef:
        return None
    if (n > 1 and np.min(np.diff(x)) < 1e-12) or np.min(np.abs(x - xi)) < 1e-12:
        return None
    pts = np.concatenate([x, [xi]])
    V = np.polynomial.chebyshev.chebvander(pts, deg)
    Ux = np.zeros((n, deg + 1))                         # T_j' = j U_{j-1}
    Um, Uc = np.zeros(n), np.ones(n)
    for jj in range(1, deg + 1):
        Ux[:, jj] = jj * Uc
        Um, Uc = Uc, 2.0 * x * Uc - Um
    M = np.vstack([V[:n], Ux, V[n:]])
    rhs = np.concatenate([(x < xi).astype(float), np.zeros(n), [1.0 if upper else 0.0]])
    scale = np.max(np.abs(M), axis=1)
    scale[scale == 0] = 1.0
    try:
        dcoef = np.linalg.solve(M / scale[:, None], rhs / scale)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(dcoef)):
        return None
    out = np.zeros(ncoef)
    out[:deg + 1] = dcoef
    return out


def _chebyshev_free_nodes(n: int, xi: float) -> np.ndarray:
    """The n Chebyshev points of the first kind, cos((2k - 1) pi / (2n)), without the one nearest to xi when it lies
    within a quarter of the local spacing (the free nodes of the Hermite-Fejer type step polynomial)."""
    k = np.arange(1, n + 1)
    x = np.cos((2 * k - 1) * np.pi / (2 * n))
    j = int(np.argmin(np.abs(x - xi)))
    spacing = np.pi / n * max(np.sqrt(max(1.0 - xi * xi, 0.0)), 1.0 / n)
    if abs(x[j] - xi) < 0.25 * spacing:
        x = np.delete(x, j)
    return np.sort(x)


def _prove_and_bound(d, intervals, T, xi_f: Fraction, upper: bool, hints=None):
    xi = float(xi_f)
    D = _dyadic(d)
    margin = 1.01 * _float_violation(D, xi, upper) + 1e-9 * (1.0 + float(np.sum(np.abs(d))))
    if not math.isfinite(margin):
        return None, {"reason": "the floating-point check of the candidate is not finite"}
    for attempt in range(_ATTEMPTS):
        Dm = list(D)
        scaled = margin * 2.0 ** _DYADIC_BITS
        if not math.isfinite(scaled):                 # the margin doubles per attempt; ceil(inf) would raise
            return None, {"reason": "the margin of the candidate overflows"}
        shift = int(math.ceil(scaled))
        Dm[0] += shift if upper else -shift
        if _prove_majorant(Dm, T, xi_f, upper, hints):
            return _bound_from(Dm, intervals, upper), {"margin": shift / 2.0 ** _DYADIC_BITS, "attempts": attempt + 1}
        margin *= 2.0
    return None, {"reason": "the sign conditions could not be proved (margin up to %.1e)" % (margin / 2.0)}


def _estimate(d, tau_hat, rho, xi, upper, tau0_hi):
    """Floating-point estimate of the bound a polynomial will give after its margin (to rank the candidates)."""
    D = _dyadic(d)
    margin = 1.01 * _float_violation(D, xi, upper) + 1e-9 * (1.0 + float(np.sum(np.abs(d))))
    spread = float(np.sum(np.abs(d) * rho)) + margin * tau0_hi
    centre = float(np.dot(d, tau_hat))
    return centre + spread if upper else centre - spread


_HEURISTICS = ("radau-hermite", "chebyshev-hermite", "linear program")


def _one_side(tau_hat, intervals, T, xi_f: Fraction, upper: bool, grid: Optional[int], jacobi=None):
    """The bound on one side at xi. Candidates: Hermite step polynomials at the Gauss-Radau nodes of the moments, of
    several orders (the full one can grow by orders of magnitude across a gap of the spectrum, where its coefficients
    and its rounding swamp the bound), and at Chebyshev points of the first kind (bounded, a resolution of one node
    spacing; useful when the moments determine few recurrence coefficients); ranked by an estimate and proved best
    first, with the free nodes as the first cuts of the proof; the linear program when none is proved."""
    xi = float(xi_f)
    tried = []
    K = len(tau_hat) - 1
    rho = np.array([(hi - lo) / 2.0 for lo, hi in intervals])
    tau0_hi = intervals[0][1]
    cands = []
    if "radau-hermite" in _HEURISTICS:
        jac = jacobi if jacobi is not None else _jacobi_from_moments(tau_hat)
        n_all = len(jac[0]) - 1
        for n in sorted({n_all, (3 * n_all) // 4, n_all // 2, n_all // 4, n_all // 8} - {0}, reverse=True):
            res = _radau_hermite(tau_hat, xi, upper, n=n, jacobi=jac, return_nodes=True)
            if res is not None and res[0] is not None and np.all(np.isfinite(res[0])):
                cands.append((_estimate(res[0], tau_hat, rho, xi, upper, tau0_hi), "radau-hermite", n, res[0], res[1]))
    if "chebyshev-hermite" in _HEURISTICS:
        for n in sorted({K // 2, K // 4} - {0}, reverse=True):
            x = _chebyshev_free_nodes(n, xi)
            d = _hermite_step(x, xi, upper, K + 1)
            if d is not None:
                cands.append((_estimate(d, tau_hat, rho, xi, upper, tau0_hi), "chebyshev-hermite", len(x), d, x))
    cands.sort(key=lambda c: c[0] if upper else -c[0])
    for _est, label, n, d, nodes in cands[:3]:
        val, info = _prove_and_bound(d, intervals, T, xi_f, upper, hints=nodes)
        if val is not None:
            info.update(method=label, order=int(n), candidates=len(cands))
            return val, info
        tried.append(f"{label} n = {n}: " + info["reason"])
    if not cands:
        tried.append("no Hermite polynomial")
    if "linear program" not in _HEURISTICS:
        return None, {"reason": "no candidate polynomial was proved", "tried": tried}
    d = _lp_polynomial(tau_hat, xi, upper, grid)
    if d is None:
        return None, {"reason": "the linear program did not return a polynomial", "tried": tried}
    val, info = _prove_and_bound(d, intervals, T, xi_f, upper)
    info["method"] = "linear program"
    info["tried"] = tried
    return val, info


def distribution_bounds(rec: Dict[str, Any], energies: Sequence[float], degree: Optional[int] = None,
                        grid: Optional[int] = None) -> Dict[str, Any]:
    """M6: bounds L(E) <= F(E) = tr(P O 1_{H <= E} O)/g <= U(E) at the given energies, from the M5 record of
    certified_chebyshev_moments. Returns a copy of the record with the statement M6 and the key 'distribution'.

    The moment intervals of the record are taken as given: its record_sha256 is not compared (the tests change
    intervals on purpose as negative controls), so a record changed after M5 gives bounds for the changed
    intervals. Comparing the hash on an entry for unmodified records is an open item."""
    from .diagnostics import annotate_record
    out = copy.deepcopy(rec)
    st = {s["id"]: s for s in out["statements"]}
    m5 = st.get("M5.chebyshev_moments")
    energies = [float(e) for e in energies]
    assume = list(m5.get("assumptions", [])) if m5 else [STANDARD_MODEL]
    if DERIVATION not in assume:
        assume.append(DERIVATION)
    statements = [s for s in out["statements"] if s["id"] != "M6.distribution_bounds"]
    if m5 is None or m5["status"] != "CERTIFIED":
        statements.append(_vstmt("M6.distribution_bounds", "NOT_CHECKED", "needs M5 CERTIFIED (the Chebyshev moments)",
                                 {}, reason_code="NEEDS_DEPENDENCY", assumptions=assume))
        out["statements"] = statements
        annotate_record(out)
        out["record_sha256"] = _record_hash(out)
        return out
    ch = out["chebyshev"]
    c, sc = Fraction(ch["center"]), Fraction(ch["scale"])
    K = ch["K"] if degree is None else min(int(degree), ch["K"])
    moms = ch["moments"][:K + 1]
    intervals = [tuple(m["interval"]) for m in moms]
    tau_hat = np.array([m["value"] for m in moms], dtype=float)
    T = _cheb_monomials(K)
    jac = _jacobi_from_moments(tau_hat)
    t0_lo, t0_hi = intervals[0]
    rows = []
    for E in energies:
        xi = (Fraction(E) - c) * sc                         # exact; a dyadic number (E, c and s are binary64)
        if xi < -1:
            rows.append({"energy": E, "u": float(xi), "lower": 0.0, "upper": 0.0, "rule": "below the spectrum"})
            continue
        if xi >= 1:
            rows.append({"energy": E, "u": float(xi), "lower": max(0.0, t0_lo), "upper": t0_hi,
                         "rule": "above the spectrum: the total mass tau_0"})
            continue
        U_, ui = _one_side(tau_hat, intervals, T, xi, True, grid, jacobi=jac)
        L_, li = _one_side(tau_hat, intervals, T, xi, False, grid, jacobi=jac)
        rows.append({"energy": E, "u": float(xi),
                     "lower": 0.0 if L_ is None else max(0.0, L_),
                     "upper": t0_hi if U_ is None else min(t0_hi, U_),
                     "rule": "polynomial majorants of degree %d" % K, "upper_proof": ui, "lower_proof": li})
    # F is nondecreasing: running max of the lower bounds, running min of the upper bounds from the right
    order = sorted(range(len(rows)), key=lambda i: rows[i]["energy"])
    best = 0.0
    for i in order:
        best = max(best, rows[i]["lower"])
        rows[i]["lower_monotone"] = best
    best = t0_hi
    for i in reversed(order):
        best = min(best, rows[i]["upper"])
        rows[i]["upper_monotone"] = best
    out["distribution"] = {"degree": K, "rows": rows, "total_mass_interval": [t0_lo, t0_hi]}
    failed = [row["energy"] for row in rows if (row.get("upper_proof") or {}).get("reason")
              or (row.get("lower_proof") or {}).get("reason")]
    text = "; ".join(f"F({row['energy']:.6g}) in [{row['lower_monotone']:.6g}, {row['upper_monotone']:.6g}]"
                     for row in sorted(rows, key=lambda x: x["energy"])[:12])
    summary = (f"F(E) = tr(P O 1_(H <= E) O)/g bounded at {len(rows)} energies by proved polynomial majorants of "
               f"degree {K}: {text}" + (" ..." if len(rows) > 12 else ""))
    if failed:
        summary += f"; at {len(failed)} energies a polynomial was not proved and the trivial bound is given"
    statements.append(_vstmt("M6.distribution_bounds", "CERTIFIED", summary,
                             {"rows": rows, "degree": K,
                              "basis": ("q <= 1_(-inf, E] <= p on the spectrum interval [c - r, c + r], signs proved "
                                        "in exact rationals (Bernstein); docs/proofs/C-07b_spectral_bounds.md")},
                             assumptions=assume))
    out["statements"] = statements
    annotate_record(out)
    out["record_sha256"] = _record_hash(out)
    return out


def spectral_bounds_from_certificate(cert, H, O, K: int = 32, energies: Sequence[float] = (), members=(0,),
                                     **kw) -> Dict[str, Any]:
    """M5 and M6 from a saved certificate with stored eigenvectors and a CERTIFIED S4 cluster with the given members
    (Options(store_eigenvectors=True, eigenspaces='all'))."""
    from .davis_kahan import _cluster_record
    X = cert.arrays.get("eigenvectors")
    if X is None:
        raise ValueError("the certificate needs stored eigenvectors (Options(store_eigenvectors=True))")
    op = (cert.body.get("subject") or {}).get("operator") or {}
    if not op.get("matrix_sha256"):
        raise ValueError("the certificate records no matrix hash (its operator is not a stored matrix), so H cannot "
                         "be compared with it; use certified_chebyshev_moments with the cluster record instead")
    if op["matrix_sha256"] != matrix_sha256(H):
        raise ValueError("H is not the matrix of the certificate (matrix_sha256 differs)")
    cl = _cluster_record(cert, members)
    lower, source = _lowest_lower_bound(cert)
    proved = lower is not None and "spectrum_lower" not in kw
    if proved:
        # one source per end: the certificate's lower bound, and what the caller says of an upper bound
        parts = [f"lower: {source}"]
        if kw.get("spectrum_upper") is not None:
            parts.append("upper: " + (kw.get("spectrum_source") or "declared by the caller, not checked"))
        elif kw.get("spectrum_source"):
            parts.append(kw["spectrum_source"])
        kw = dict(kw, spectrum_lower=lower, spectrum_source="; ".join(parts))
    rec = certified_chebyshev_moments(H, O, np.asarray(X)[:, list(members)], cl["sin_theta_bound"], cl["eta"], K=K,
                                      cluster={"members": list(members), "claimed": cl.get("claimed"),
                                               "status": cl.get("status")}, **kw)
    from .crossings import certificate_hash
    rec["certificate"] = {"hash": certificate_hash(cert), "members": list(members), "operator_hash_matches": True}
    if proved:                                      # the lower end was proved, not declared
        rec["chebyshev"]["declared_bounds"]["lower"] = None
        rec["chebyshev"]["proved_bounds"] = {"lower": float(lower), "source": source}
    rec["record_sha256"] = _record_hash(rec)
    return distribution_bounds(rec, energies) if len(energies) else rec


def _lowest_lower_bound(cert):
    """(lower bound on the lowest eigenvalue of the certificate's operator, its source) from a 'lowest' claim whose S2
    and S3 are CERTIFIED and which concerns the operator itself (no sector block); (None, '') otherwise. S3 states that
    the claimed intervals hold the lowest eigenvalues, so the smallest left end mu_i - r_i (rounded down, with the
    larger of the two stored radii) bounds the spectrum from below."""
    claim = cert.body.get("claim") or {}
    subject = cert.body.get("subject") or {}
    if claim.get("kind") != "lowest" or subject.get("sector") is not None:
        return None, ""
    if cert.status_of("S2.eigenpairs") != "CERTIFIED" or cert.status_of("S3.completeness") != "CERTIFIED":
        return None, ""
    mu = cert.arrays.get("eigenvalues")
    rad = cert.arrays.get("radius")
    if mu is None or rad is None or not len(mu):
        return None, ""
    rad = np.asarray(rad, dtype=float)
    ex = cert.arrays.get("existence_radius")
    if ex is not None:
        rad = np.maximum(rad, np.asarray(ex, dtype=float))
    lo = float(np.min(sub_down(np.asarray(mu, dtype=float), rad)))
    if not math.isfinite(lo):
        return None, ""
    return lo, "lowest eigenvalue from the certificate (S2 and S3 CERTIFIED, 'lowest' claim)"


def format_spectral(rec: Dict[str, Any]) -> str:
    ch = rec.get("chebyshev", {})
    lines = [f"spectral record ({rec['schema']}): {rec.get('description', '')}",
             f"  c = {ch.get('center')!r}, s = {ch.get('scale')!r}, spectrum in {ch.get('spectrum_interval')}, "
             f"K = {ch.get('K')}", f"  {DERIVATION}"]
    for s in rec["statements"]:
        lines.append(f"  {s['id']:24s} {s['status']:13s} {s['summary'][:200]}")
    for row in (rec.get("distribution") or {}).get("rows", []):
        lines.append(f"  E = {row['energy']:.6g}: F in [{row['lower_monotone']:.6g}, {row['upper_monotone']:.6g}]")
    return "\n".join(lines)
