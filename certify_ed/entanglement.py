"""S4 extension (roadmap C-03): entanglement entropy and entanglement spectrum of a certified eigenvector
or, for a cluster, of the normalised projector of the certified eigenspace.

Statement (docs/proofs/C-03_entanglement.md). Let P be the spectral projector of the certified cluster
(Theorem 6: ||Qhat Qhat^H - P||_2 <= s with Qhat the polar factor of the claimed vectors X, eta =
||X^H X - I||_2), A a subset of the sites, rho_A = tr_B(P)/g and rho_hat_A = tr_B(X X^H)/tr(X^H X). Then

  T := ||rho_A - rho_hat_A||_1 / 2 <= s + eta/(1 - eta)      (= s when g = 1: the second term vanishes),
  |S(rho_A) - S(rho_hat_A)| <= T ln(d_A - 1) + h(T)            (Fannes-Audenaert; needs T <= 1 - 1/d_A),
  |lambda_i(rho_A) - lambda_i(rho_hat_A)| <= 2 T                (Weyl).

Everything about rho_hat_A is computed without floating-point error analysis: the reduced matrix
M = tr_B(X X^H) is formed in exact rational arithmetic from the stored binary64 entries (which are dyadic
rationals), its eigenvalues are enclosed by Ostrowski's and Weyl's theorems around a floating-point
eigendecomposition whose defects (||V^H V - I||_F and ||V^H M V / t - diag(p)||_F) are computed exactly, and
the entropy, the binary entropy h and the logarithms are evaluated on dyadic intervals with outward rounding
(`ivl.py`, Lemma 17). The only floating-point step is the final conversion of the interval ends to
binary64 (outward). The bipartition is declared by the user (Options.entanglement = {name: {"sites": [...]}}).

Status of the derivation: docs/proofs/C-03_entanglement.md (derived here; independent re-derivation pending,
D-12).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import ivl
from .eigenspace import EigenspaceCertificate
from .precision import binary64

__all__ = ["bipartition_indices", "reduced_density_exact", "enclose_psd_eigenvalues", "entropy_interval",
           "fannes_audenaert_bound", "certify_entanglement", "entanglement_records", "DEFAULT_MAX_DIM_A"]

DEFAULT_MAX_DIM_A = 64        # dimension of the A space above which the exact stage is skipped (SIZE_LIMIT)


# ------------------------------------------------------------------------------------------------ exact helpers

def _frac_sqrt_upper(q: Fraction) -> Fraction:
    """Rational upper bound on sqrt(q) for q >= 0: (isqrt(a b) + 1) / b for q = a/b."""
    if q < 0:
        raise ValueError("sqrt of a negative rational")
    a, b = q.numerator, q.denominator
    r = math.isqrt(a * b)
    if r * r == a * b:                # exact (including q = 0)
        return Fraction(r, b)
    return Fraction(r + 1, b)


def _ints_common_scale(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Exact integer representation of a binary64 array: arr = ints / 2^K (K >= 0, common to all entries)."""
    flat = np.asarray(arr, dtype=float).ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("non-finite entry")
    K = 0
    mants: List[Tuple[int, int]] = []
    for v in flat.tolist():
        if v == 0.0:
            mants.append((0, 0))
            continue
        m, e = math.frexp(v)                      # v = m 2^e, 0.5 <= |m| < 1
        mi = int(m * (1 << 53))                   # exact: m has at most 53 significant bits
        ei = e - 53                               # v = mi 2^ei
        mants.append((mi, ei))
        K = max(K, -ei)
    out = np.empty(flat.size, dtype=object)
    for k, (mi, ei) in enumerate(mants):
        out[k] = mi << (ei + K) if mi else 0      # ei + K >= 0 by construction
    return out.reshape(np.asarray(arr).shape), K


def bipartition_indices(states_in_order: np.ndarray, n_sites: int, q: int,
                        sites_A: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """For every basis index: (index in A, index in B) of the state's digits, site 0 the most significant
    base-q digit (spec.site_order 'site0_msb'). Returns (a_idx, b_idx, d_A, d_B) with d_A = q^|A| and d_B the
    number of distinct B patterns present (<= q^(N - |A|))."""
    A = sorted({int(s) for s in sites_A})
    if not A or any(not (0 <= s < n_sites) for s in A):
        raise ValueError(f"sites of A must be a non-empty subset of range({n_sites}), got {list(sites_A)}")
    if len(A) == n_sites:
        raise ValueError("A must be a proper subset of the sites (B non-empty)")
    s = np.asarray(states_in_order, dtype=np.int64)
    Bs = [k for k in range(n_sites) if k not in set(A)]
    a_idx = np.zeros(s.size, dtype=np.int64)
    b_idx = np.zeros(s.size, dtype=np.int64)
    for k in A:
        a_idx = a_idx * q + (s // (q ** (n_sites - 1 - k))) % q
    for k in Bs:
        b_idx = b_idx * q + (s // (q ** (n_sites - 1 - k))) % q
    _, b_comp = np.unique(b_idx, return_inverse=True)
    d_A = q ** len(A)
    d_B = int(b_comp.max()) + 1 if s.size else 0
    return a_idx, b_comp.astype(np.int64), int(d_A), d_B


def reduced_density_exact(X: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray, d_A: int, d_B: int):
    """M = sum_j tr_B(x_j x_j^H) in exact rational arithmetic: (M_re, M_im) as object arrays of Fractions
    (d_A x d_A), and t = tr M = ||X||_F^2 as a Fraction. The vectors' binary64 entries are taken as the exact
    dyadic rationals they are; no rounding occurs (a narrower format is converted exactly, a wider one refused:
    precision.py)."""
    X = binary64(np.asarray(X), "eigenvectors")
    if X.ndim == 1:
        X = X[:, None]
    n, g = X.shape
    cplx = np.iscomplexobj(X)
    Mr = np.zeros((d_A, d_A), dtype=object)
    Mi = np.zeros((d_A, d_A), dtype=object)
    Mr[:] = 0
    Mi[:] = 0
    Kmax = 0
    parts = []
    for j in range(g):
        xr, Kr = _ints_common_scale(X[:, j].real)
        xi, Ki = (_ints_common_scale(X[:, j].imag) if cplx else (None, 0))
        parts.append((xr, Kr, xi, Ki))
        Kmax = max(Kmax, Kr, Ki)
    for xr, Kr, xi, Ki in parts:
        Yr = np.zeros((d_A, d_B), dtype=object)
        Yr[:] = 0
        sh = Kmax - Kr
        for p in range(n):
            if xr[p]:
                Yr[a_idx[p], b_idx[p]] = xr[p] << sh
        Mr += Yr @ Yr.T
        if cplx:
            Yi = np.zeros((d_A, d_B), dtype=object)
            Yi[:] = 0
            shi = Kmax - Ki
            for p in range(n):
                if xi[p]:
                    Yi[a_idx[p], b_idx[p]] = xi[p] << shi
            Mr += Yi @ Yi.T
            # Im (Y Y^H)_{aa'} = sum_b (Yi_ab Yr_a'b - Yr_ab Yi_a'b)
            Mi += Yi @ Yr.T - Yr @ Yi.T
    den = 1 << (2 * Kmax)
    fr = np.vectorize(lambda v: Fraction(int(v), den), otypes=[object])
    Mr_f = fr(Mr)
    Mi_f = fr(Mi) if cplx else None
    t = sum(Mr_f[i, i] for i in range(d_A))
    return Mr_f, Mi_f, t


def _frac_matrix(F: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    F = np.asarray(F)
    fr = np.vectorize(lambda v: Fraction(float(v)), otypes=[object])
    if np.iscomplexobj(F):
        return fr(F.real), fr(F.imag)
    return fr(F), None


def _cmatmul(A, B):
    """Product of complex matrices given as (re, im) object arrays (im may be None)."""
    Ar, Ai = A
    Br, Bi = B
    Rr = Ar @ Br
    if Ai is not None and Bi is not None:
        Rr = Rr - Ai @ Bi
    Ri = None
    if Ai is not None or Bi is not None:
        Ri = (Ar @ Bi if Bi is not None else 0) + (Ai @ Br if Ai is not None else 0)
    return Rr, Ri


def _cH(A):
    Ar, Ai = A
    return Ar.T, (None if Ai is None else -Ai.T)


def _fro2(A) -> Fraction:
    Ar, Ai = A
    s = sum(v * v for v in Ar.ravel())
    if Ai is not None:
        s += sum(v * v for v in Ai.ravel())
    return Fraction(s)


def enclose_psd_eigenvalues(Mr: np.ndarray, Mi: Optional[np.ndarray], t: Fraction) -> Dict[str, object]:
    """Enclosures [lo_k, hi_k] (Fractions, ascending) of the eigenvalues of the exact density matrix
    rho_hat = M / t, M positive semidefinite Hermitian with exact rational entries (Mr + i Mi), t = tr M > 0.

    Method (docs/proofs/C-03_entanglement.md, Lemma 3). A floating-point eigendecomposition rho_hat ~ V
    diag(p) V^H is computed; then, exactly in rationals, eta_V = ||V^H V - I||_F (upper bound on the 2-norm)
    and w = ||V^H rho_hat V - diag(p)||_F. Weyl's inequality gives |lambda_k(V^H rho_hat V) - p_(k)| <= w and
    Ostrowski's theorem lambda_k(V^H rho_hat V) = theta_k lambda_k(rho_hat) with theta_k in [1 - eta_V, 1 + eta_V],
    so for lambda_k(rho_hat) >= 0: lambda_k in [max(0, p_(k) - w) / (1 + eta_V), (p_(k) + w) / (1 - eta_V)].
    Returns None-free dict with 'lo', 'hi' (lists of Fractions), 'eta_V', 'w', 'p_float' (sorted floats)."""
    d = Mr.shape[0]
    rho_f = np.array([[float(Mr[i, j] / t) for j in range(d)] for i in range(d)], dtype=float)
    if Mi is not None:
        rho_f = rho_f + 1j * np.array([[float(Mi[i, j] / t) for j in range(d)] for i in range(d)], dtype=float)
    rho_f = (rho_f + rho_f.conj().T) / 2
    p, V = np.linalg.eigh(rho_f)
    order = np.argsort(p)
    p, V = p[order], V[:, order]
    Vf = _frac_matrix(V)
    VH = _cH(Vf)
    # eta_V = ||V^H V - I||_F exactly
    G = _cmatmul(VH, Vf)
    Gr, Gi = G
    for i in range(d):
        Gr[i, i] -= 1
    eta_V = _frac_sqrt_upper(_fro2((Gr, Gi)))
    if not eta_V < 1:
        raise ValueError("floating-point eigenvector matrix too far from unitary (eta_V >= 1)")
    # w = ||V^H (M/t) V - diag(p)||_F exactly
    rho_frac = (np.vectorize(lambda v: v / t, otypes=[object])(Mr),
                None if Mi is None else np.vectorize(lambda v: v / t, otypes=[object])(Mi))
    S_ = _cmatmul(_cmatmul(VH, rho_frac), Vf)
    Sr, Si = S_
    pf = [Fraction(float(v)) for v in p]
    for i in range(d):
        Sr[i, i] -= pf[i]
    w = _frac_sqrt_upper(_fro2((Sr, Si)))
    lo, hi = [], []
    for k in range(d):
        a = pf[k] - w
        lo.append(max(Fraction(0), a) / (1 + eta_V))
        hi.append(min(Fraction(1), (pf[k] + w) / (1 - eta_V)))
    return {"lo": lo, "hi": hi, "eta_V": eta_V, "w": w, "p_float": [float(v) for v in p]}


# ------------------------------------------------------------------------------------------------ interval functions

def _f_point(p: Fraction) -> ivl.Interval:
    """Enclosure of f(p) = -p ln p for a rational p in [0, 1]; f(0) = 0."""
    if p <= 0:
        return ivl.ZERO
    if p >= 1:
        return ivl.ZERO
    piv = ivl.from_fraction(p)
    return -(piv * ivl.ln_interval(piv))


def _f_interval(lo: Fraction, hi: Fraction) -> ivl.Interval:
    """Enclosure of {f(p): p in [lo, hi]} for 0 <= lo <= hi <= 1, f(p) = -p ln p: f increases on [0, 1/e] and
    decreases on [1/e, 1] (f' = -ln p - 1), so the range is [min(f(lo), f(hi)), max(f(lo), f(hi))] when 1/e is
    not inside, and the maximum is f(1/e) = 1/e when it may be."""
    lo = max(Fraction(0), lo)
    hi = min(Fraction(1), hi)
    if lo > hi:
        lo = hi
    fl, fh = _f_point(lo), _f_point(hi)
    low = min(fl.lo, fh.lo)
    high = max(fl.hi, fh.hi)
    inv_e = ivl.exp_interval(ivl.from_fraction(Fraction(-1)))       # enclosure of 1/e
    ilo, ihi = inv_e.fractions()
    if lo <= ihi and hi >= ilo:                                      # 1/e may lie in [lo, hi]
        high = max(high, inv_e.hi)
    return ivl.Interval(low, high)


def entropy_interval(lo: Sequence[Fraction], hi: Sequence[Fraction]) -> ivl.Interval:
    """Enclosure of S = sum_k f(p_k) over all p_k in [lo_k, hi_k] (nats)."""
    return ivl.sum_intervals(_f_interval(a, b) for a, b in zip(lo, hi))


def binary_entropy_upper(T: Fraction) -> int:
    """Upper end (scaled int) of an enclosure of h(T) = -T ln T - (1 - T) ln(1 - T), 0 <= T <= 1."""
    return (_f_point(T) + _f_point(1 - T)).hi


def fannes_audenaert_bound(T: Fraction, d_A: int) -> Optional[int]:
    """Upper end (scaled int) of T ln(d_A - 1) + h(T), valid for 0 <= T <= 1 - 1/d_A (Audenaert 2007); None
    outside that range. The function is increasing on [0, 1 - 1/d_A], so the value at an upper bound on T bounds
    the value at T (docs/proofs/C-03_entanglement.md, Lemma 4)."""
    if T < 0 or T > 1 - Fraction(1, d_A):
        return None
    if T == 0:
        return 0
    term = ivl.ZERO
    if d_A > 2:
        term = ivl.from_fraction(T) * ivl.ln_interval(ivl.from_fraction(Fraction(d_A - 1)))
    return (term + ivl.Interval(0, binary_entropy_upper(T))).hi


def _to_base(iv: ivl.Interval, base) -> ivl.Interval:
    """Convert an enclosure in nats to log base `base` (2 or 'e'): divide by the enclosure of ln(base)."""
    if base in ("e", None):
        return iv
    b = int(base)
    if b < 2:
        raise ValueError("log base must be 'e' or an integer >= 2")
    return iv / ivl.ln_interval(ivl.from_fraction(Fraction(b)))


# ------------------------------------------------------------------------------------------------ the S4 record

def certify_entanglement(ec: EigenspaceCertificate, X: np.ndarray, states_in_order: np.ndarray, n_sites: int,
                         q: int, sites_A: Sequence[int], name: str = "A", log_base="e",
                         max_dim_a: int = DEFAULT_MAX_DIM_A) -> dict:
    """Certified entanglement entropy and spectrum of the cluster's normalised projector across the cut A|B.

    ec carries s = ec.sin_theta (||Qhat Qhat^H - P||_2 <= s, Theorem 6) and eta = ec.eta. The record states
    the certified interval for S(rho_A), rho_A = tr_B(P)/g, the computed value (floating point, for display),
    the enclosure of S(rho_hat_A), the trace-distance bound T, the Fannes-Audenaert term, and the entanglement
    spectrum as intervals. Statuses: CERTIFIED, or NOT_CHECKED with a reason_code (the Fannes-Audenaert range
    T <= 1 - 1/d_A violated; the exact stage skipped above max_dim_a; a cluster whose eta makes T >= 1)."""
    g = len(ec.members)
    s = Fraction(float(ec.sin_theta))
    eta = Fraction(float(ec.eta))
    rec = {"name": name, "sites_A": sorted({int(x) for x in sites_A}), "cluster_size": g,
           "object": ("reduced state of the certified eigenvector" if g == 1 else
                      "reduced state of the normalised projector tr_B(P)/g of the certified eigenspace (basis independent)"),
           "sin_theta_bound": float(ec.sin_theta), "eta": float(ec.eta), "log_base": "e" if log_base in ("e", None) else int(log_base)}
    try:
        a_idx, b_idx, d_A, d_B = bipartition_indices(states_in_order, n_sites, q, sites_A)
    except ValueError as e:
        rec.update(status="NOT_CHECKED", reason=str(e), reason_code="NOT_APPLICABLE")
        return rec
    rec.update(d_A=int(d_A), d_B_present=int(d_B))
    if d_A > max_dim_a:
        rec.update(status="NOT_CHECKED", reason_code="SIZE_LIMIT",
                   reason=f"d_A = {d_A} exceeds the limit {max_dim_a} for the exact reduced-density stage (entanglement max_dim_a)")
        return rec
    # trace-distance bound T (Proposition 1): s for g = 1, s + eta/(1 - eta) for g > 1
    if g == 1:
        T = s
    else:
        if not eta < 1:
            rec.update(status="NOT_CHECKED", reason_code="NOT_APPLICABLE", reason="cluster eta >= 1: no trace-distance bound")
            return rec
        T = s + eta / (1 - eta)
    rec["trace_distance_bound"] = float(ivl.to_float_up(ivl.from_fraction(T).hi))
    rec["trace_distance_bound_formula"] = "s" if g == 1 else "s + eta/(1 - eta)"
    fa = fannes_audenaert_bound(T, d_A)
    if fa is None:
        rec.update(status="NOT_CHECKED", reason_code="NOT_APPLICABLE",
                   reason=(f"trace-distance bound T = {float(T):.3e} exceeds the Fannes-Audenaert range T <= 1 - 1/d_A = "
                           f"{1 - 1 / d_A:.6g}: no entropy interval is stated (the trivial bound ln d_A is not reported as "
                           f"a certified interval)"))
        return rec
    Xg = np.asarray(X)[:, ec.members]
    Mr, Mi, t = reduced_density_exact(Xg, a_idx, b_idx, d_A, d_B)
    if t <= 0:
        rec.update(status="NOT_CHECKED", reason_code="NOT_APPLICABLE", reason="zero vectors: tr(X^H X) = 0")
        return rec
    try:
        encl = enclose_psd_eigenvalues(Mr, Mi, t)
    except ValueError as e:
        rec.update(status="NOT_CHECKED", reason_code="NOT_APPLICABLE", reason=str(e))
        return rec
    lo, hi = encl["lo"], encl["hi"]
    S_hat = entropy_interval(lo, hi)
    fa_iv = ivl.Interval(0, fa)
    S_cert = S_hat + ivl.Interval(-fa_iv.hi, fa_iv.hi)
    ln_dA = ivl.ln_interval(ivl.from_fraction(Fraction(d_A)))
    S_cert = ivl.Interval(max(S_cert.lo, 0), min(S_cert.hi, ln_dA.hi))
    # conversion to the requested base (outward)
    S_hat_b, S_cert_b, fa_b = (_to_base(S_hat, log_base), _to_base(S_cert, log_base), _to_base(fa_iv, log_base))
    p_float = encl["p_float"]
    value = float(-sum(pk * math.log(pk) for pk in p_float if pk > 0))
    if log_base not in ("e", None):
        value /= math.log(int(log_base))
    two_T = 2 * T
    spectrum = []
    for k in range(d_A):
        l_hat, h_hat = lo[k], hi[k]
        l_c, h_c = max(Fraction(0), l_hat - two_T), min(Fraction(1), h_hat + two_T)
        spectrum.append({"index": k, "computed": float(p_float[k]),
                         "enclosure_computed_state": [ivl.to_float_down(ivl.from_fraction(l_hat).lo),
                                                      ivl.to_float_up(ivl.from_fraction(h_hat).hi)],
                         "certified_interval": [ivl.to_float_down(ivl.from_fraction(l_c).lo),
                                                ivl.to_float_up(ivl.from_fraction(h_c).hi)]})
    rec.update(status="CERTIFIED", value=value,
               entropy_enclosure_computed_state=list(S_hat_b.floats()),
               fannes_audenaert_term=float(ivl.to_float_up(fa_b.hi)),
               certified_interval=list(S_cert_b.floats()),
               entanglement_spectrum=spectrum,
               exact_stage={"eta_V": float(ivl.to_float_up(ivl.from_fraction(encl["eta_V"]).hi)),
                            "w": float(ivl.to_float_up(ivl.from_fraction(encl["w"]).hi)),
                            "trace": float(t), "arithmetic": "exact rationals (Fraction); dyadic intervals (ivl.py, 192 bits)"},
               statement=(f"S(rho_A) in [{S_cert_b.floats()[0]:.15g}, {S_cert_b.floats()[1]:.15g}] "
                          f"({'nats' if log_base in ('e', None) else 'log base ' + str(int(log_base))}) for A = {rec['sites_A']}; "
                          f"trace-distance bound T <= {rec['trace_distance_bound']:.3e}, Fannes-Audenaert term "
                          f"{float(ivl.to_float_up(fa_b.hi)):.3e}; spectrum intervals by Weyl (2T)"),
               basis=("rank-2g trace-norm bound and contractivity of the partial trace (Prop. 1), Fannes-Audenaert "
                      "inequality (Audenaert 2007), Weyl's inequality; exact reduced matrix, Ostrowski + Weyl enclosure of "
                      "its spectrum; dyadic interval evaluation of x ln x (Lemma 17)"))
    return rec


def entanglement_records(ec: EigenspaceCertificate, X, spec, claim, requests: Dict[str, dict]) -> Dict[str, dict]:
    """S4 hook: one record per requested cut (Options.entanglement). A spec is needed for the site encoding and a
    computational-basis claim (full space or a U(1) block, Claim.states) for the vectors; symmetry-adapted
    (sector_isometry) claims are not lifted (the lifted vectors carry the isometry and invariance defects, which
    this statement does not propagate). Errors become NOT_CHECKED records with a reason_code and a next_action,
    never exceptions."""
    from .diagnostics import entanglement_next_action
    out: Dict[str, dict] = {}
    for name, req in requests.items():
        req = dict(req or {})
        if spec is None:
            rec = {"name": name, "status": "NOT_CHECKED", "reason_code": "NEEDS_SPEC",
                   "reason": "the site encoding of the basis comes from the model specification"}
        elif getattr(claim, "sector_isometry", None) is not None:
            rec = {"name": name, "status": "NOT_CHECKED", "reason_code": "UNSUPPORTED_CLAIM",
                   "reason": "symmetry-adapted (sector_isometry) claim: the reduced state of the lifted vectors is not certified"}
        else:
            try:
                from .construct import full_state_set
                states = claim.states if getattr(claim, "states", None) is not None else full_state_set(spec)
                rec = certify_entanglement(ec, X, states.states_in_order(), spec.n_sites, spec.q, req.get("sites", ()),
                                           name=name, log_base=req.get("log_base", "e"),
                                           max_dim_a=int(req.get("max_dim_a", DEFAULT_MAX_DIM_A)))
            except (ValueError, TypeError) as e:
                rec = {"name": name, "status": "NOT_CHECKED", "reason_code": "NOT_APPLICABLE", "reason": str(e)}
        if rec.get("status") != "CERTIFIED":
            rec["next_action"] = entanglement_next_action(rec)
        out[name] = rec
    return out
