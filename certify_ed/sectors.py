"""Audit of a producer's symmetry-sector decomposition, and composition of
per-sector statements into statements about the full operator.

A producer that diagonalizes H block by block implicitly claims that its
sector basis V_s (d x d_s) is an isometry onto an H-invariant subspace and that
H_s = V_s^* H V_s. Given the producer's V_s and H_s and an independently built
full-space H (from the spec), we measure

  iso_s  = || V_s^* V_s - I ||          (orthonormal basis)
  inv_s  = || H V_s - V_s H_s ||        (invariance; the key condition: V^*HV = H_s
                                          alone holds for ANY subspace)
  orth   = || V_s^* V_t ||, s != t      (sectors mutually orthogonal)
  dim    = sum_s d_s vs dim of the covered space

If iso_s and inv_s are small, every certified eigenpair (mu, y) of H_s lifts to
(mu, V_s y) for H with residual <= ||H_s y - mu y|| + inv_s ||y||, and
per-sector inertia counts add up to counts for H on the covered subspace (with
the invariance defect added to the backward-error bound). Frobenius-norm
upper bounds are used throughout (computable in O(nnz)).

When the producer does not expose V_s, only necessary conditions are
available: dimension sums and trace moments sum_s tr H_s = tr H,
sum_s ||H_s||_F^2 = ||H||_F^2 (computed from the spec, see invariants.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from .fp import ULP_MIN, fro_upper, gamma_dot, norm2_upper, up, U


@dataclass
class SectorAudit:
    label: str
    dim: int
    isometry_defect: float
    invariance_defect: float
    compression_defect: float
    passed: bool
    note: str = ""

    def to_dict(self):
        return dict(self.__dict__)


def audit_sector(H: sp.spmatrix, V: sp.spmatrix, Hs: sp.spmatrix, label: str = "",
                 tol: float = 1e-10) -> SectorAudit:
    H = sp.csr_matrix(H)
    V = sp.csr_matrix(V)
    Hs = sp.csr_matrix(Hs)
    ds = V.shape[1]
    cplx = np.iscomplexobj(V.data) or np.iscomplexobj(H.data) or np.iscomplexobj(Hs.data)
    G = (V.conj().T @ V - sp.identity(ds, format="csr")).tocsr()
    wv = int(np.diff(sp.csc_matrix(V).indptr).max()) if ds else 1
    iso = up(fro_upper(G) + gamma_dot(wv, cplx) * fro_upper(V) ** 2)
    HV = H @ V
    VHs = V @ Hs
    D = (HV - VHs).tocsr()
    wh = int(np.diff(H.indptr).max()) if H.shape[0] else 1
    rnd = gamma_dot(wh + int(np.diff(Hs.indptr).max() if ds else 1) + 2, cplx) * (
        fro_upper(abs(H) @ abs(V)) + fro_upper(abs(V) @ abs(Hs)))
    inv = up(fro_upper(D) + rnd)
    C = (V.conj().T @ HV - Hs).tocsr()
    comp = up(fro_upper(C) + gamma_dot(wh + wv + 2, cplx) * fro_upper(abs(V).T @ abs(H) @ abs(V)))
    # a defect bound that is not finite (a bound of the accounting overflowed) passes nothing: inf <= tol * inf would
    # hold otherwise
    finite = bool(np.isfinite(iso) and np.isfinite(inv) and np.isfinite(comp))
    passed = finite and iso <= tol and inv <= tol * max(1.0, fro_upper(H) / max(1.0, np.sqrt(H.shape[0])))
    return SectorAudit(label, int(ds), float(iso), float(inv), float(comp), bool(passed))


def audit_orthogonality(Vs: List[sp.spmatrix]) -> float:
    """max_{s<t} || V_s^* V_t ||_F upper bound."""
    worst = 0.0
    for a in range(len(Vs)):
        for b in range(a + 1, len(Vs)):
            M = (Vs[a].conj().T @ Vs[b]).tocsr()
            if M.nnz:
                worst = max(worst, fro_upper(M))
    return up(worst)


@dataclass
class DecompositionAudit:
    sectors: List[SectorAudit]
    orthogonality_defect: float
    covered_dim: int
    expected_dim: int
    trace_check: Optional[dict] = None
    passed: bool = False

    def to_dict(self):
        return {"sectors": [s.to_dict() for s in self.sectors],
                "orthogonality_defect": self.orthogonality_defect,
                "covered_dim": self.covered_dim, "expected_dim": self.expected_dim,
                "trace_check": self.trace_check, "passed": self.passed}


def audit_decomposition(H: sp.spmatrix, Vs: List[sp.spmatrix], Hss: List[sp.spmatrix],
                        labels: Optional[List[str]] = None, tol: float = 1e-10) -> DecompositionAudit:
    labels = labels or [f"sector_{i}" for i in range(len(Vs))]
    audits = [audit_sector(H, V, Hs, lab, tol) for V, Hs, lab in zip(Vs, Hss, labels)]
    orth = audit_orthogonality(Vs)
    cov = sum(a.dim for a in audits)
    exp = H.shape[0]
    tr = trace_moments_check(H, Hss)
    passed = all(a.passed for a in audits) and orth <= tol and cov == exp and tr["consistent"]
    return DecompositionAudit(audits, float(orth), int(cov), int(exp), tr, bool(passed))


def trace_moments_check(H: sp.spmatrix, Hss: List[sp.spmatrix], rtol: float = 1e-10) -> dict:
    """Necessary conditions without isometries: tr and Frobenius norm additivity."""
    tH = complex(sp.csr_matrix(H).diagonal().sum())
    fH = float(sp.linalg.norm(H) ** 2)
    ts = complex(sum(sp.csr_matrix(M).diagonal().sum() for M in Hss))
    fs = float(sum(sp.linalg.norm(M) ** 2 for M in Hss))
    scale = max(1.0, fH)
    ok = abs(tH - ts) <= rtol * scale and abs(fH - fs) <= rtol * scale
    return {"trace_full": tH.real, "trace_sectors": ts.real, "fro2_full": fH, "fro2_sectors": fs,
            "consistent": bool(ok), "rtol": rtol}


def decomposition_shift(audit: "DecompositionAudit", H_norm: float, B_norm: float) -> Optional[float]:
    """Bound eps such that the union of sector spectra (block-diagonal B) and the
    spectrum of H match within eps in sorted order (Theorem 2 applied with the
    square, nearly unitary W = [V_1 ... V_m] and M = B).

    Requires the sectors to cover the space (covered_dim == expected_dim).
    Returns None if the audit cannot support the statement.
    """
    import math
    if audit.covered_dim != audit.expected_dim:
        return None
    m = len(audit.sectors)
    # ||W^H W - I||_2 <= ||.||_F: the m diagonal blocks V_s^H V_s - I and the m(m - 1) off-diagonal blocks V_s^H V_t,
    # each bounded by the audit's Frobenius bounds; norm2_upper covers the sum of squares and its underflow
    eta = norm2_upper(np.concatenate([np.array([s.isometry_defect for s in audit.sectors], dtype=float),
                                      np.full(m * (m - 1), audit.orthogonality_defect, dtype=float)]))
    if not eta < 1:
        return None
    delta = up(1 - math.sqrt(1 - eta)) + 4 * U
    D = norm2_upper(np.array([s.invariance_defect for s in audit.sectors], dtype=float))
    # Kahan's theorem on the polar factor of W pairs the eigenvalues of B with distinct eigenvalues of H
    # within rho (docs/proofs/thm2_polar-factor.md; for the square W Weyl gives the same constant,
    # docs/proofs/thm8_rho.md). Was (1 + sqrt 5)/2 rho before track/backlog-proofs.
    return up(D + (H_norm + B_norm) * delta) + ULP_MIN


def lift_radius(sector_rel_radius: float, audit: "SectorAudit") -> float:
    """Theorem-1 radius in the full space for (mu, V y) given the sector radius
    ||H_s y - mu y||/||y||: (sqrt(1+iso) r + inv) / sqrt(1-iso)."""
    import math
    iso = audit.isometry_defect
    if iso >= 1:
        return float("inf")
    return up((math.sqrt(1 + iso) * sector_rel_radius + audit.invariance_defect) / math.sqrt(1 - iso))


# ---------------------------------------------------------------------------
# Restricting sector audits to an exactly invariant U(1) block
# ---------------------------------------------------------------------------

def restrict_to_u1(spec, V, cache: Optional[dict] = None):
    """If every row in the support of V lies in one digit-sum (U(1)) sector and the
    spec conserves the digit sum exactly (zero leak), return (H_u, V_u, stateset):
    the spec operator on that sector and V with rows re-indexed into it.
    Returns None otherwise (the caller then needs the full-space operator)."""
    from .basis import digit_sum, u1_state_set
    from .construct import build_bitwise, full_state_set
    V = sp.csr_matrix(V)
    rows = np.unique(V.nonzero()[0])
    if rows.size == 0:
        return None
    full = full_state_set(spec)
    states = full.states_in_order()[rows]
    ds = digit_sum(states.astype(np.int64), spec.q, spec.n_sites)
    if not np.all(ds == ds[0]):
        return None
    total = int(ds[0])
    key = ("u1", total)
    if cache is not None and key in cache:
        Hu, st = cache[key]
    else:
        st = u1_state_set(spec, total)
        Hu, leak = build_bitwise(spec, st, return_leak=True)
        if leak > 0:
            return None
        if cache is not None:
            cache[key] = (Hu, st)
    idx, found = st.index_of(states)
    if not np.all(found):
        return None
    Vc = V.tocoo()
    pos = np.searchsorted(rows, Vc.row)
    Vu = sp.csr_matrix((Vc.data, (idx[pos], Vc.col)), shape=(st.size, V.shape[1]))
    return Hu, Vu, st


def symmetry_matrix(states, g) -> sp.csr_matrix:
    """Signed permutation matrix of a cyclic symmetry g on a closed state set:
    G e_s = phase(s) e_{g(s)}, rows/columns in states.states_in_order()."""
    s = states.states_in_order()
    t, ph = g.apply(s.astype(np.int64))
    idx, found = states.index_of(t)
    if not np.all(found):
        raise ValueError("state set is not closed under the symmetry")
    n = s.size
    return sp.csr_matrix((ph, (idx, np.arange(n))), shape=(n, n))


_SEP_NOTE = ("floating-point value of min |lambda_l - lambda_k| over the computed roots of unity, not a certified "
             "lower bound; the leakage bound divides by it")


def audit_label(spec, V, symmetry: str = "translation", cache=None) -> dict:
    """Which eigenvalue of a cyclic symmetry g a sector carries, with a certified leakage bound.

    For the n-th roots lambda_k of unity, picks the k minimising ||G V - lambda_k V||_F. Since G is
    unitary with spectrum in the n-th roots, any unit vector x in range(V) has a component outside
    the lambda_k-eigenspace of norm at most delta / ((1 - iso) min_{l != k} |lambda_l - lambda_k|),
    where delta >= ||G V - lambda_k V||_2 and iso = ||V^+ V - I|| (a bound, from computed
    quantities plus their rounding). Labels are reported as the eigenvalue exp(2 pi i j / n) of g (for
    translation, g maps site i to i+1) and as QuSpin's block index q.

    The separation min_{l != k} |lambda_l - lambda_k| is the floating-point value over the computed roots (np.exp, then
    np.abs of the differences), not a certified lower bound; the record says so ("separation_note"). A lower bound on
    it, and on the distance of the computed roots from the exact ones, is open (a D-12 item)."""
    from .basis import translation, spin_flip, reflection
    from .fp import fro_upper, gamma_dot, up
    gfun = {"translation": translation, "spin_flip": spin_flip, "reflection": reflection}[symmetry]
    g = gfun(spec)
    red = restrict_to_u1(spec, V, cache)
    if red is not None:
        _, Vu, st = red
    else:
        from .construct import full_state_set
        st, Vu = full_state_set(spec), sp.csr_matrix(V)
    G = symmetry_matrix(st, g)
    Vd = sp.csr_matrix(Vu)
    GV = G @ Vd
    n = g.order
    roots = np.exp(2j * np.pi * np.arange(n) / n)
    defects = [fro_upper((GV - lam * Vd).toarray() if Vd.shape[0] * Vd.shape[1] <= 4e7
                         else (GV - lam * Vd).data) for lam in roots]
    k = int(np.argmin(defects))
    lam = roots[k]
    width = 4
    rnd = gamma_dot(width, True) * 2 * fro_upper(np.abs(Vd.data))   # rounding in lam*V and the difference
    delta = up(defects[k] + rnd)
    gram = (Vd.conj().T @ Vd - sp.identity(Vd.shape[1])).tocsr()
    iso = up(fro_upper(gram.data) + gamma_dot(int(np.diff(Vd.tocsc().indptr).max()) + 1, True)
             * fro_upper(np.abs(Vd.data)) ** 2) if gram.nnz else 0.0
    sep = min(abs(lam - roots[j]) for j in range(n) if j != k) if n > 1 else np.inf
    leak = delta / ((1 - iso) * sep) if iso < 1 else np.inf
    # QuSpin's block index q: eigenvalue exp(-2 pi i q/n) of its translation, which maps site i to
    # i-1; with CERTIFY-ED's T (site i -> i+1) that is exp(+2 pi i q / n), i.e. q = k (checked against
    # QuSpin blocks in tests/test_label_audit.py)
    q_quspin = k
    return {"symmetry": symmetry, "order": n, "eigenvalue": [lam.real, lam.imag],
            "j_physics": k, "quspin_q": q_quspin, "defect_upper": float(delta),
            "isometry_defect_upper": float(iso), "leakage_bound": float(leak),
            "separation": float(sep), "separation_note": _SEP_NOTE,
            "certified": bool(leak < 0.5)}
