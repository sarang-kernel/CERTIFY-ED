"""Basis-independent certification of (quasi-)degenerate eigenspaces.

Inside a degenerate or nearly degenerate cluster, individual eigenvectors are
not well defined: any orthonormal rotation of a valid basis is equally valid,
so per-vector observables differ between solvers without either being wrong.
The well-defined objects are the spectral projector P onto the cluster and the
spectrum of conserved quantities restricted to it. Given

  * a group of claimed pairs certified as distinct (enclosure.Theorem 2),
  * a certified separation delta_sep > 0 between the group's claimed values and
    every other eigenvalue of A (from disjoint enclosures plus inertia-certified
    completeness of a window around the group),

we certify:

 (a) Davis-Kahan sin(Theta) theorem (Davis & Kahan 1970):
         || Qhat Qhat^* - P ||_2 <= rho / delta_sep =: s,
     where Qhat is the orthonormal polar factor of the claimed vectors X.
     Since X X^* = Qhat (X^*X) Qhat^*, || X X^* - P ||_2 <= s + eta.
 (b) For Hermitian O: | tr(P O) - tr(X^* O X) | <= 2 g ||O||_2 (s + eta).
 (c) For S with [A, S] = 0 EXACTLY (checked structurally), the multiset of
     S-eigenvalues on range(P) (quantum numbers of the multiplet) satisfies
         | q_i - eig_i(X^* S X) | <= 2 s^2 ||S|| + (2 kappa + kappa^2)(1+eta)||S|| + rounding,
     kappa = 1/sqrt(1-eta) - 1 (Ostrowski's theorem + Weyl; derivation in the
     paper's appendix). Quantum numbers known to lie on a lattice (e.g.
     half-integers) are certified exactly when the bound is below half the
     lattice spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .fp import U, gamma_dot, up, inf_norm_upper
from .precision import binary64


# ---------------------------------------------------------------------------
# exact commutation checks
# ---------------------------------------------------------------------------

def commutes_diagonal(A: sp.spmatrix, s_diag: np.ndarray) -> bool:
    """[A, diag(s)] = 0 exactly iff A_ij = 0 whenever s_i != s_j (structural, exact)."""
    A = sp.coo_matrix(A)
    nz = A.data != 0
    return bool(np.all(s_diag[A.row[nz]] == s_diag[A.col[nz]]))


def commutes_signed_permutation(A: sp.spmatrix, perm: np.ndarray, sign: np.ndarray) -> bool:
    """[A, P] = 0 for P|j> = sign_j |perm_j> with sign in {+1,-1}: checked exactly
    as A[perm_i, perm_j] * sign_i * sign_j == A[i, j] entrywise."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    P = sp.csr_matrix((sign.astype(A.dtype), (perm, np.arange(n))), shape=(n, n))
    B = (P.T @ A @ P).tocsr()
    D = (A - B).tocsr()
    D.eliminate_zeros()
    return D.nnz == 0


# ---------------------------------------------------------------------------
# eigenspace certificate
# ---------------------------------------------------------------------------

@dataclass
class EigenspaceCertificate:
    members: List[int]
    claimed: List[float]
    separation: float
    rho: float
    eta: float
    sin_theta: float                    # s
    projector_error: float              # || X X^* - P ||_2 bound
    observables: Dict[str, dict] = field(default_factory=dict)
    quantum_numbers: Dict[str, dict] = field(default_factory=dict)
    sin_theta_bounds: Dict[str, object] = field(default_factory=dict)   # E-07: 'theorem6', 'davis_kahan', 'used'

    def to_dict(self):
        return {"members": self.members, "claimed": self.claimed,
                "separation_lower_bound": self.separation, "rho": self.rho, "eta": self.eta,
                "sin_theta_bound": self.sin_theta, "projector_error_bound": self.projector_error,
                "sin_theta_bounds": self.sin_theta_bounds,
                "observables": self.observables, "quantum_numbers": self.quantum_numbers}


def certify_eigenspace(X: np.ndarray, members: List[int], claimed: np.ndarray,
                       rho: float, eta: float, separation: float,
                       sin_theta_alt: Optional[dict] = None) -> Optional[EigenspaceCertificate]:
    """s = rho / separation (Theorem 6: Davis-Kahan with the Theorem 2 residual bound rho) or, when `sin_theta_alt`
    carries a second proven bound (E-07, davis_kahan.sin_theta_davis_kahan: {'bound': s_dk, ...}), the smaller of
    the two; both are recorded in `sin_theta_bounds`. None when neither bound is below 1 or eta >= 1."""
    if not eta < 1:
        return None
    s6 = up(rho / separation) if separation > 0 else float("inf")
    s_dk = float("inf")
    if sin_theta_alt is not None and sin_theta_alt.get("bound") is not None:
        s_dk = float(sin_theta_alt["bound"])
    s = min(s6, s_dk)
    if not s < 1:
        return None
    used = "theorem6" if s6 <= s_dk else "davis_kahan"
    rec = {"theorem6": float(s6) if math.isfinite(s6) else None, "davis_kahan": float(s_dk) if math.isfinite(s_dk) else None,
           "used": used}
    if sin_theta_alt is not None:
        rec["davis_kahan_record"] = {k: v for k, v in sin_theta_alt.items() if k != "bound"}
    return EigenspaceCertificate(list(map(int, members)), [float(claimed[i]) for i in members],
                                 float(separation), float(rho), float(eta), float(s),
                                 float(up(s + eta)), sin_theta_bounds=rec)


def certify_observable(ec: EigenspaceCertificate, X: np.ndarray, O, name: str,
                       O_norm: Optional[float] = None) -> dict:
    """Certified tr(P O) for Hermitian O (dense or sparse). Data in a narrower format than binary64 are converted
    exactly first (precision.py)."""
    X = binary64(X, "eigenvectors")
    O = binary64(O if sp.issparse(O) else np.asarray(O), name)
    Xg = X[:, ec.members]
    OX = O @ Xg
    val = np.trace(Xg.conj().T @ OX).real
    g = len(ec.members)
    if O_norm is None:
        O_norm = inf_norm_upper(O)
    cplx = np.iscomplexobj(Xg) or np.iscomplexobj(O.data if sp.issparse(O) else O)
    n = Xg.shape[0]
    width = n if not sp.issparse(O) else int(np.diff(sp.csr_matrix(O).indptr).max())
    absOX = (abs(O) if sp.issparse(O) else np.abs(O)) @ np.abs(Xg)
    rnd = gamma_dot(width + n + 2, cplx) * float(np.sum(np.abs(Xg) * absOX)) * (1 + 8 * U)
    bound = up(2 * g * O_norm * ec.projector_error + rnd)
    rec = {"value": float(val), "bound": float(bound), "O_norm_bound": float(O_norm),
           "statement": f"tr(P {name}) in [{val - bound:.15g}, {val + bound:.15g}]"}
    ec.observables[name] = rec
    return rec


def certify_quantum_numbers(ec: EigenspaceCertificate, X: np.ndarray, S, name: str,
                            lattice: Optional[float] = None, S_norm: Optional[float] = None,
                            commutation_certified: bool = False, allowed_values=None) -> dict:
    """Certified multiset of S-eigenvalues on the cluster (requires exact [A,S]=0). Data in a narrower format than
    binary64 are converted exactly first (precision.py)."""
    X = binary64(X, "eigenvectors")
    S = binary64(S if sp.issparse(S) else np.asarray(S), name)
    Xg = X[:, ec.members]
    M = Xg.conj().T @ (S @ Xg)
    M = (M + M.conj().T) / 2
    est = np.sort(np.linalg.eigvalsh(M))
    if S_norm is None:
        S_norm = inf_norm_upper(S)
    eta = ec.eta
    kappa = 1.0 / math.sqrt(1.0 - eta) - 1.0 if eta < 1 else float("inf")
    n = Xg.shape[0]
    cplx = np.iscomplexobj(Xg)
    width = n if not sp.issparse(S) else int(np.diff(sp.csr_matrix(S).indptr).max())
    rnd = gamma_dot(width + n + 4, cplx) * S_norm * (1 + eta) * len(ec.members)
    # eigvalsh of the small matrix M: backward stable, error <= p(g) u ||M||; use generous 10 g^2 u
    rnd += 10 * len(ec.members) ** 2 * U * S_norm * (1 + eta)
    bound = up(2 * ec.sin_theta ** 2 * S_norm + (2 * kappa + kappa ** 2) * (1 + eta) * S_norm + rnd)
    rec = {"estimates": est.tolist(), "bound": float(bound),
           "commutation_certified": bool(commutation_certified)}
    if commutation_certified and allowed_values is not None:
        av = np.asarray(allowed_values, dtype=float)
        idx = np.argmin(np.abs(est[:, None] - av[None, :]), axis=1)
        snapped = av[idx]
        half_gap = np.min(np.diff(np.sort(av))) / 2 if av.size > 1 else np.inf
        if bound < half_gap and np.all(np.abs(snapped - est) <= bound):
            rec["certified_values"] = snapped.tolist()
            rec["status"] = "CERTIFIED"
        else:
            rec["status"] = "INCONCLUSIVE"
    elif commutation_certified and lattice:
        snapped = np.round(est / lattice) * lattice
        if bound < lattice / 2 and np.all(np.abs(snapped - est) <= bound):
            rec["certified_values"] = snapped.tolist()
            rec["status"] = "CERTIFIED"
        else:
            rec["status"] = "INCONCLUSIVE"
    else:
        rec["status"] = "CERTIFIED-INTERVALS" if commutation_certified else "NOT-CERTIFIED (commutation unverified)"
    ec.quantum_numbers[name] = rec
    return rec


def rotation_spread(X: np.ndarray, members: List[int], O, trials: int = 64, seed: int = 0) -> dict:
    """How much per-vector expectation values vary under random rotations inside the
    cluster (all equally valid eigenbases). Diagnostic, not a certified quantity."""
    rng = np.random.default_rng(seed)
    Xg = X[:, members]
    g = Xg.shape[1]
    vals = []
    for _ in range(trials):
        Z = rng.standard_normal((g, g)) + 1j * rng.standard_normal((g, g))
        Qr, _ = np.linalg.qr(Z)
        Y = Xg @ Qr
        vals.append(np.real(np.einsum("ij,ij->j", Y.conj(), O @ Y)))
    vals = np.array(vals)
    return {"min": float(vals.min()), "max": float(vals.max()),
            "spread": float(vals.max() - vals.min())}
