"""Davis-Kahan sin(Theta) statements (roadmap C-12, C-13, E-07): a residual-based alternative to the Theorem 6
eigenspace bound, approximate-symmetry labels, and overlaps (fidelities) between certified eigenspaces.

Derivations: docs/proofs/E-07_davis_kahan.md (the three statements share the sin(Theta) machinery). Notation as
in docs/theory.md: A Hermitian, X (n x g) the claimed vectors of a cluster, eta = ||X^H X - I||_2 < 1, X = Qhat M the
polar decomposition, Lambda = diag(mu), R = A X - X Lambda, P the spectral projector of A onto the g cluster
eigenvalues (certified distinct and separated by S2-S4), dist the distance from the claimed values [a, b] to
every other eigenvalue of A.

E-07 (sin(Theta) with the residual, Frobenius form).  H' = Qhat^H A Qhat is Hermitian with
    ||A Qhat - Qhat H'||_F <= ||R||_F / sqrt(1 - eta),
    ||H' - Lambda||_2 <= w' := ||X^H R||_2 / (1 - eta) + delta spread(mu) / sqrt(1 - eta),  delta = 1 - sqrt(1 - eta),
so the eigenvalues of H' lie within w' of [a, b] and the eigenvalues of A outside range P are at distance >= dist - w'
from them; the Davis-Kahan sin(Theta) theorem (Frobenius version, proved in the document) then gives
    ||Qhat Qhat^H - P||_2 = ||(I - P) Qhat||_2 <= ||(I - P) Qhat||_F <= ||R||_F / (sqrt(1 - eta) (dist - w')) =: s_DK.
Theorem 6's bound s_6 = rho / (dist - eps) with rho = ||R|| + (||A|| + ||Lambda||) delta is kept; S4 uses min(s_6, s_DK)
with both recorded (both are proven: Theorem 6 in the manuscript appendix, s_DK in the document).

C-12 (approximate symmetry).  For Hermitian S with ||[A, S]||_2 <= c and the cluster's level enclosure J = [J_L, J_U]
(width w, midpoint distance gamma_0 >= sep + w/2 to the other eigenvalues), every eigenvalue m of Q^H S Q (Q an
orthonormal basis of range P) has an eigenvalue of S within (c + ||S|| w/2) / gamma_0, and the sorted eigenvalues of
the computed X^H S X are within (2 s + 2 s^2 + eta) ||S|| + rounding of those of Q^H S Q. The label record states the
total deviation D and the hypothesis c (computed from the stored matrices with its rounding bound).

C-13 (fidelity).  For two certified clusters (P_1, X_1, s_1, eta_1) and (P_2, X_2, s_2, eta_2) of the same dimension g
(two couplings, two sizes of a model, two sectors):
    |tr(P_1 P_2) - tr(X_1 X_1^H X_2 X_2^H)| <= g (s_1 + eta_1) + g (1 + eta_1)(s_2 + eta_2),
and tr(X_1 X_1^H X_2 X_2^H) = ||X_1^H X_2||_F^2 carries its rounding bound; for g = 1 tr(P_1 P_2) = |<psi_1|psi_2>|^2.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .eigenspace import EigenspaceCertificate
from .fp import U, abs_up, add_up, fro_upper, gamma, gamma_dot, inf_norm_upper, sub_down, up
from .precision import binary64, summed

__all__ = ["sin_theta_davis_kahan", "commutator_bound", "approximate_symmetry_label", "certified_fidelity",
           "fidelity_from_certificates"]


# ------------------------------------------------------------------------------------------------ E-07

def sin_theta_davis_kahan(R_fro: float, XR: float, eta: float, delta: float, spread: float, dist: float,
                          model_shift: float = 0.0) -> Dict[str, object]:
    """s_DK = (||R||_F / sqrt(1 - eta)) / (dist - w') with w' = XR/(1 - eta) + delta spread / sqrt(1 - eta)
    (docs/proofs/E-07_davis_kahan.md, Proposition 1). All inputs are upper bounds (R_fro >= ||R||_F,
    XR >= ||X^H R||_2, delta >= 1 - sqrt(1 - eta), spread >= max mu - min mu) except dist, a lower bound on the
    distance from [min mu, max mu] to the other eigenvalues of A. A model shift sigma (S1) enlarges ||R|| by
    sqrt(1 + eta) sigma and ||X^H R|| by (1 + eta) sigma. Returns {'bound': s_DK or None, 'w_prime', 'gap', ...}."""
    if not (0 <= eta < 1) or not dist > 0:
        return {"bound": None, "reason": "eta >= 1 or no positive separation"}
    if model_shift > 0:
        R_fro = float(add_up(R_fro, up(math.sqrt(1 + eta) * model_shift)))
        XR = float(add_up(XR, up((1 + eta) * model_shift)))
    inv_sqrt = up(1.0 / math.sqrt(1.0 - eta))
    inv = up(1.0 / (1.0 - eta))
    w_prime = float(up(XR * inv + delta * spread * inv_sqrt))
    gap = float(sub_down(dist, w_prime))
    if not gap > 0:
        return {"bound": None, "w_prime": w_prime, "gap": gap, "reason": "Ritz values not separated from the rest (dist <= w')"}
    s = float(up(R_fro * inv_sqrt / gap))
    return {"bound": s, "w_prime": w_prime, "gap": gap, "residual_term": float(up(R_fro * inv_sqrt)),
            "statement": "||Qhat Qhat^H - P||_2 <= ||R||_F / (sqrt(1 - eta) (dist - w')) (Davis-Kahan sin Theta, Frobenius form)"}


# ------------------------------------------------------------------------------------------------ C-12

def commutator_bound(H, S) -> Dict[str, float]:
    """Upper bound c on ||[H, S]||_2 for stored Hermitian H and S (sparse or dense): the infinity norm (anti-Hermitian
    commutator: ||C||_2 <= ||C||_inf) of |fl(HS) - fl(SH)| plus the standard-model rounding gamma_{w_H + 1} |H||S| +
    gamma_{w_S + 1} |S||H| entrywise, rounded up. Returns {'bound', 'computed_inf_norm', 'rounding_entrywise_inf_norm'}."""
    H = summed(sp.csr_matrix(binary64(H, "operator")), "operator")   # exact, widened before the conversion; repeated
    S = summed(sp.csr_matrix(binary64(S, "symmetry")), "symmetry")   # entries summed on a copy (precision.py)
    cplx = np.iscomplexobj(H.data) or np.iscomplexobj(S.data)
    C = (H @ S - S @ H).tocsr()
    wH = int(np.diff(H.indptr).max()) if H.shape[0] else 1
    wS = int(np.diff(S.indptr).max()) if S.shape[0] else 1
    aH, aS = abs_up(H), abs_up(S)                            # upper moduli of complex data (fp.abs_up)
    absHS = (aH @ aS).tocsr()
    absSH = (aS @ aH).tocsr()
    # entrywise |fl(HS) - HS| <= gamma_{w_H} |H||S|, the subtraction adds one rounding (index + 1); the computed absolute
    # products underestimate the exact ones by at most a relative gamma_{w} entrywise
    gH, gS = gamma_dot(wH + 1, cplx), gamma_dot(wS + 1, cplx)
    E = (absHS * (gH * up(1.0 / (1.0 - gH))) + absSH * (gS * up(1.0 / (1.0 - gS)))).tocsr()
    # [H, S] is anti-Hermitian, so ||C||_2 <= (||C||_1 ||C||_inf)^(1/2) = ||C||_inf; the exact commutator is within E
    # entrywise of the computed one, and |C| + E has the row sums of a bound on |C_exact|
    comp = float(inf_norm_upper(C))
    total = float(inf_norm_upper((abs_up(C) + E).tocsr()))
    return {"bound": float(up(total)), "computed_inf_norm": comp, "rounding_entrywise_inf_norm": float(inf_norm_upper(E))}


def approximate_symmetry_label(ec: EigenspaceCertificate, X: np.ndarray, S, name: str, c: float,
                               J: Sequence[float], S_norm: Optional[float] = None, lattice: Optional[float] = None,
                               allowed_values: Optional[Sequence[float]] = None, c_record: Optional[dict] = None) -> dict:
    """Labels of an approximately conserved Hermitian S on a certified cluster (C-12).

    Hypothesis recorded: ||[A, S]||_2 <= c. With J = [J_L, J_U] the cluster's level enclosure (contains the g
    eigenvalues), w = J_U - J_L, gamma_0 = sep + w/2 (sep = ec.separation), s = ec.sin_theta, eta = ec.eta: every sorted
    eigenvalue of fl(X^H S X) is within
        D = (2 s + 2 s^2 + eta) ||S|| + rnd + (c + ||S|| w/2) / gamma_0
    of an eigenvalue of S (docs/proofs/E-07_davis_kahan.md, Proposition 2). When D < half the spacing of the
    declared lattice (or of the allowed values), the labels snap and the record is CERTIFIED (under the recorded
    hypothesis); otherwise CERTIFIED-INTERVALS (each computed value has an S-eigenvalue within D)."""
    Xg = binary64(np.asarray(X)[:, ec.members], "eigenvectors")    # exact for narrower formats (precision.py)
    S = summed(binary64(S if sp.issparse(S) else np.asarray(S), name), name)
    g = len(ec.members)
    n = Xg.shape[0]
    cplx = np.iscomplexobj(Xg) or (np.iscomplexobj(S.data) if sp.issparse(S) else np.iscomplexobj(S))
    M = Xg.conj().T @ (S @ Xg)
    M = (M + M.conj().T) / 2
    est = np.sort(np.linalg.eigvalsh(M))
    if S_norm is None:
        S_norm = inf_norm_upper(S)
    S_norm = float(S_norm)
    s, eta = float(ec.sin_theta), float(ec.eta)
    width_S = n if not sp.issparse(S) else int(np.diff(sp.csr_matrix(S).indptr).max())
    # rounding of fl(X^H S X) (two inner products, <= w_S + n terms) and of its Hermitian part, Frobenius norm bound
    # gamma ||X||_F^2 ||S||_inf with ||X||_F^2 <= g (1 + eta); eigvalsh backward error p(g) u ||M|| with p(g) <= 10 g^2 (LAPACK)
    rnd = gamma_dot(width_S + n + 4, cplx) * S_norm * (1 + eta) * g * (1 + 4 * U)
    rnd += 10 * g ** 2 * U * S_norm * (1 + eta)
    JL, JU = float(J[0]), float(J[1])
    w = float(np.nextafter(JU - JL, np.inf))
    gamma_0 = float(sub_down(ec.separation, -w / 2))           # sep + w/2, rounded down
    if not gamma_0 > 0:
        rec = {"status": "INCONCLUSIVE", "reason": "no positive distance from the level to the rest of the spectrum"}
        ec.quantum_numbers[name] = rec
        return rec
    geom = up((2 * s + 2 * s * s + eta) * S_norm)
    sym = up((c + S_norm * w / 2) / gamma_0)
    D = float(add_up(add_up(geom, up(rnd)), sym))
    rec = {"estimates": est.tolist(), "bound": D, "hypothesis": {"commutator_bound": float(c), "statement": "||[H, S]||_2 <= c",
                                                                  "record": c_record},
           "terms": {"geometric": float(geom), "rounding": float(up(rnd)), "symmetry": float(sym)},
           "S_norm_bound": S_norm, "level_enclosure": [JL, JU], "gamma_0": gamma_0,
           "basis": ("eigenvalues of X^H S X vs Q^H S Q: Ostrowski + alignment ||Qhat - Q U|| <= s + s^2; Q^H S Q eigenvalues "
                     "vs eigenvalues of S: Theorem 1 for S with residual (c + ||S|| w/2)/gamma_0 "
                     "(docs/proofs/E-07_davis_kahan.md, Proposition 2)")}
    av = None if allowed_values is None else np.sort(np.asarray(allowed_values, dtype=float))
    if av is not None:
        idx = np.argmin(np.abs(est[:, None] - av[None, :]), axis=1)
        snapped = av[idx]
        half_gap = np.min(np.diff(av)) / 2 if av.size > 1 else np.inf
        ok = D < half_gap and bool(np.all(np.abs(snapped - est) <= D))
    elif lattice:
        snapped = np.round(est / lattice) * lattice
        ok = D < lattice / 2 and bool(np.all(np.abs(snapped - est) <= D))
    else:
        snapped, ok = None, False
    if snapped is not None and ok:
        rec["certified_values"] = snapped.tolist()
        rec["status"] = "CERTIFIED"
        rec["statement"] = (f"under ||[H, S]|| <= {c:.3e}, the eigenvalues of S closest to the computed labels are "
                            f"{snapped.tolist()} (each computed label within {D:.3e} of an eigenvalue of S, below half the spacing)")
    elif snapped is not None:
        rec["status"] = "INCONCLUSIVE"
        rec["statement"] = f"deviation bound {D:.3e} not below half the spacing of the declared values"
    else:
        rec["status"] = "CERTIFIED-INTERVALS"
        rec["statement"] = f"under ||[H, S]|| <= {c:.3e}, each computed label has an eigenvalue of S within {D:.3e}"
    ec.quantum_numbers[name] = rec
    return rec


# ------------------------------------------------------------------------------------------------ C-13

def certified_fidelity(X1: np.ndarray, s1: float, eta1: float, X2: np.ndarray, s2: float, eta2: float) -> dict:
    """tr(P_1 P_2) from the stored vectors of two certified clusters of equal dimension g (C-13).

    F_hat = ||X_1^H X_2||_F^2 = tr(X_1 X_1^H X_2 X_2^H); |tr(P_1 P_2) - F_hat| <= g (s_1 + eta_1) + g (1 + eta_1)(s_2 + eta_2)
    + rounding (docs/proofs/E-07_davis_kahan.md, Proposition 3). For g = 1 this is the fidelity |<psi_1|psi_2>|^2; for
    g > 1 tr(P_1 P_2)/g is the mean overlap of the subspaces (in [0, 1]). The interval is clipped to [0, g]."""
    X1 = binary64(np.asarray(X1), "X1")             # exact for narrower formats (precision.py)
    X2 = binary64(np.asarray(X2), "X2")
    if X1.ndim == 1:
        X1 = X1[:, None]
    if X2.ndim == 1:
        X2 = X2[:, None]
    if X1.shape != X2.shape:
        raise ValueError(f"the two clusters must have the same dimension and live in the same space (shapes {X1.shape}, {X2.shape})")
    n, g = X1.shape
    cplx = np.iscomplexobj(X1) or np.iscomplexobj(X2)
    C = X1.conj().T @ X2
    # |C_ij|^2 from the components (two squares and a sum, inside the gamma_{g^2 + 2} below; no modulus is formed)
    F_hat = float(np.sum(C.real ** 2 + C.imag ** 2))
    # rounding: |fl(C) - C|_F <= gamma_n ||X1||_F ||X2||_F; | ||C_hat||^2 - ||C||^2 | <= (2 ||C_hat||_F + e) e; the sum of
    # squares of g^2 terms adds gamma_{g^2 + 2} ||C_hat||_F^2
    e = gamma_dot(n, cplx) * fro_upper(X1) * fro_upper(X2)
    cf = fro_upper(C)
    rnd = up((2 * cf + e) * e * (1 + 4 * U) + gamma(g * g + 2) * cf * cf * (1 + 4 * U))
    proj = up(g * (s1 + eta1) + g * (1 + eta1) * (s2 + eta2))
    bound = float(add_up(proj, rnd))
    lo = max(0.0, float(sub_down(F_hat, bound)))
    hi = min(float(g), float(add_up(F_hat, bound)))
    return {"value": F_hat, "bound": bound, "interval": [lo, hi], "cluster_size": int(g),
            "per_state_value": F_hat / g, "per_state_interval": [lo / g, hi / g],
            "terms": {"projector": float(proj), "rounding": float(rnd)},
            "inputs": {"sin_theta_1": float(s1), "eta_1": float(eta1), "sin_theta_2": float(s2), "eta_2": float(eta2)},
            "statement": (f"tr(P_1 P_2) in [{lo:.15g}, {hi:.15g}]" + (" (fidelity |<psi_1|psi_2>|^2)" if g == 1 else
                                                                      f" (subspace overlap; per state [{lo / g:.15g}, {hi / g:.15g}])")),
            "basis": "||X X^H - P||_2 <= s + eta (Theorem 6) for both clusters, Hoelder with ||P||_1 = g; docs/proofs/E-07_davis_kahan.md"}


def _cluster_record(cert, members: Sequence[int]) -> dict:
    s4 = cert.statement("S4.eigenspaces") or {}
    for cl in (s4.get("evidence") or {}).get("clusters", []):
        if list(cl.get("members", [])) == list(members) and cl.get("status") == "CERTIFIED":
            return cl
    raise ValueError(f"no CERTIFIED S4 cluster with members {list(members)} in the certificate (S4 is {cert.status_of('S4.eigenspaces')})")


def fidelity_from_certificates(cert_a, cert_b, members_a: Sequence[int] = (0,), members_b: Optional[Sequence[int]] = None) -> dict:
    """C-13 from two saved certificates with stored eigenvectors (Options(store_eigenvectors=True)) and CERTIFIED S4
    clusters (Options(eigenspaces='all') for isolated levels). members_* are the cluster member indices (default the
    lowest pair)."""
    members_b = members_a if members_b is None else members_b
    Xa = cert_a.arrays.get("eigenvectors")
    Xb = cert_b.arrays.get("eigenvectors")
    if Xa is None or Xb is None:
        raise ValueError("both certificates need stored eigenvectors (Options(store_eigenvectors=True))")
    ca, cb = _cluster_record(cert_a, members_a), _cluster_record(cert_b, members_b)
    rec = certified_fidelity(np.asarray(Xa)[:, list(members_a)], ca["sin_theta_bound"], ca["eta"],
                             np.asarray(Xb)[:, list(members_b)], cb["sin_theta_bound"], cb["eta"])
    from .crossings import certificate_hash
    rec.update(a={"hash": certificate_hash(cert_a), "members": list(members_a), "claimed": ca.get("claimed")},
               b={"hash": certificate_hash(cert_b), "members": list(members_b), "claimed": cb.get("claimed")},
               caveat=("the two certificates must describe operators on the same basis (same spec encoding and state set); "
                       "the record does not check that"))
    return rec


def labels_hook(ec: EigenspaceCertificate, X, requests: Dict[str, object], H_ref, J: Sequence[float]) -> List[str]:
    """S4 hook for Options.approximate_symmetries = {name: (S, lattice_or_None) | (S, lattice, allowed_values)}:
    computes the commutator bound against the reference operator H_ref and writes ec.quantum_numbers[name]. Returns
    the names processed. Without a stored reference matrix the records are NOT-CHECKED."""
    done = []
    for name, spec_ in requests.items():
        S = spec_[0]
        lattice = spec_[1] if len(spec_) > 1 else None
        allowed = spec_[2] if len(spec_) > 2 else None
        if H_ref is None or not (sp.issparse(H_ref) or isinstance(H_ref, np.ndarray)):
            ec.quantum_numbers[name] = {"status": "NOT-CHECKED", "reason_code": "NEEDS_MATRIX",
                                        "reason": "the commutator bound ||[H, S]|| needs a stored reference matrix"}
            continue
        S = summed(binary64(S if sp.issparse(S) else np.asarray(S), name), name)   # exact; summed (precision.py)
        Sm = sp.diags(S) if np.ndim(S) == 1 else sp.csr_matrix(S)
        cb = commutator_bound(H_ref, Sm)
        approximate_symmetry_label(ec, X, Sm, name, cb["bound"], J, lattice=lattice, allowed_values=allowed, c_record=cb)
        ec.quantum_numbers[name]["approximate_symmetry"] = True
        done.append(name)
    return done
