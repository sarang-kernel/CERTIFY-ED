"""Certified eigenvalue counts via Sylvester's law of inertia.

For a shift sigma we compute a symmetric-indefinite factorization of
A - sigma I, i.e. stored factors L (unit lower triangular up to a symmetric
permutation) and D (block diagonal, 1x1/2x2 blocks), and an upper bound beta on

    || L D L^H - P (A - sigma I) P^T ||_2 .

Sylvester's law makes nu = #negative eigenvalues of D EXACTLY the number of
negative eigenvalues of the congruent matrix L D L^H, which differs from
P(A - sigma I)P^T by at most beta in norm. By Weyl's inequality:

    #{lambda_j(A) < sigma - beta}  <=  nu  <=  #{lambda_j(A) <= sigma + beta}.

The sign of every 1x1 pivot is read exactly; the inertia of every 2x2 block is
decided from the exact rational value of its determinant. A zero pivot or a
singular 2x2 block makes the count INCONCLUSIVE for that shift.

Dense path : LAPACK ?sytrf/?hetrf (Bunch-Kaufman) via scipy.linalg.ldl; beta
             from the explicitly formed residual L D L^H - (A - sigma I) plus a
             bound on the rounding error of forming it.
Sparse path: SuperLU in symmetric mode with no pivoting and no equilibration
             (row permutation == column permutation, verified). beta from the
             a-priori LU bound |Delta| <= gamma_w |L||U| (Higham 2002, Thm 9.3)
             plus the explicitly computed asymmetry ||U - D L^H||. Only O(nnz)
             work beyond the factorization. Without pivoting the factorization
             can be unstable; instability shows up as a large beta and yields
             INCONCLUSIVE, never a wrong count.
Prior art: this is the inertia test used for completeness in spectrum-slicing
eigensolvers (Parlett 1998; Campos & Roman 2012, SLEPc). CERTIFY-ED applies it
a posteriori to results from arbitrary producers and records beta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fp import U, gamma_dot, up, fro_upper


@dataclass
class InertiaResult:
    sigma: float
    negative: Optional[int]     # nu; None if inconclusive
    beta: float                 # backward-error bound (2-norm)
    method: str
    seconds: float
    conclusive: bool
    note: str = ""
    fill_nnz: Optional[int] = None
    resource_failure: bool = False   # the factorization could not be completed (memory / library error)

    def to_dict(self):
        return asdict(self)


def _exact_sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _count_block_diagonal(D: np.ndarray):
    """Count negative eigenvalues of a block-diagonal D with 1x1/2x2 blocks.
    Returns (neg, ambiguous) with exact sign decisions."""
    n = D.shape[0]
    neg = 0
    i = 0
    while i < n:
        if i + 1 < n and (D[i + 1, i] != 0 or D[i, i + 1] != 0):
            a = Fraction(float(np.real(D[i, i])))
            c = Fraction(float(np.real(D[i + 1, i + 1])))
            b = D[i + 1, i]
            b2 = Fraction(float(np.real(b))) ** 2 + Fraction(float(np.imag(b))) ** 2
            det = a * c - b2
            if det < 0:
                neg += 1
            elif det > 0:
                if a < 0:
                    neg += 2
                elif a == 0:
                    return neg, True
            else:
                return neg, True
            i += 2
        else:
            s = _exact_sign(float(np.real(D[i, i])))
            if s == 0:
                return neg, True
            neg += s < 0
            i += 1
    return neg, False


def inertia_dense(A: np.ndarray, sigma: float) -> InertiaResult:
    t0 = time.time()
    A = np.asarray(A)
    n = A.shape[0]
    cplx = np.iscomplexobj(A)
    S = A - sigma * np.eye(n)
    lu, d, perm = la.ldl(S, lower=True, hermitian=True)
    neg, amb = _count_block_diagonal(d)
    # Sylvester's law needs an invertible factor: unit lower triangular after the row permutation
    # (unit diagonal AND zero strict upper triangle, as checker.inertia_exact checks; IMP-14)
    Lt = lu[perm]
    if not np.all(np.diag(Lt) == 1.0):
        return InertiaResult(float(sigma), None, float("inf"), "dense-bunch-kaufman",
                             time.time() - t0, False, "triangular factor lacks unit diagonal")
    if np.any(np.triu(Lt, 1) != 0):
        return InertiaResult(float(sigma), None, float("inf"), "dense-bunch-kaufman",
                             time.time() - t0, False, "factor not lower triangular after permutation")
    R = lu @ d @ lu.conj().T - S
    # rounding in forming lu d lu^H (two products, inner length <= n) and S
    absprod = np.abs(lu) @ np.abs(d) @ np.abs(lu).T
    g = gamma_dot(2 * n + 2, cplx)
    beta = fro_upper(R) + g * fro_upper(absprod) + U * float(np.max(np.abs(np.diag(S)))) * 2
    beta = up(beta)
    res = InertiaResult(float(sigma), None if amb else int(neg), float(beta), "dense-bunch-kaufman",
                        time.time() - t0, not amb, "zero pivot or singular 2x2 block" if amb else "")
    return res


def inertia_sparse(A: sp.spmatrix, sigma: float) -> InertiaResult:
    t0 = time.time()
    A = sp.csc_matrix(A)
    n = A.shape[0]
    cplx = np.iscomplexobj(A.data)
    S = (A - sigma * sp.identity(n, dtype=A.dtype, format="csc")).tocsc()
    try:
        f = spla.splu(S, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                      options={"SymmetricMode": True, "Equil": False})
    except RuntimeError as e:  # exactly singular pivot
        return InertiaResult(float(sigma), None, float("inf"), "sparse-superlu-symmetric",
                             time.time() - t0, False, f"factorization failed: {e}")
    if not np.array_equal(f.perm_r, f.perm_c):
        return InertiaResult(float(sigma), None, float("inf"), "sparse-superlu-symmetric",
                             time.time() - t0, False, "row pivoting occurred (perm_r != perm_c)")
    L = f.L.tocsr()
    Uf = f.U.tocsr()
    if not np.all(L.diagonal() == 1.0):
        return InertiaResult(float(sigma), None, float("inf"), "sparse-superlu-symmetric",
                             time.time() - t0, False, "L lacks unit diagonal")
    dU = Uf.diagonal()
    Dreal = np.real(dU)
    signs = np.sign(Dreal)
    if np.any(signs == 0):
        return InertiaResult(float(sigma), None, float("inf"), "sparse-superlu-symmetric",
                             time.time() - t0, False, "zero pivot")
    neg = int(np.sum(signs < 0))
    # beta <= gamma_w ||L||_F ||U||_F  (LU backward error, Higham Thm 9.3)
    #       + ||L||_F ||U - D L^H||_F   (symmetrisation)  + rounding of that difference
    w = int(np.diff(L.indptr).max()) if n else 1
    Lf, Uff = fro_upper(L), fro_upper(Uf)
    asym = (Uf - sp.diags(Dreal) @ L.conj().T).tocsr()
    asym_f = fro_upper(asym) if asym.nnz else 0.0
    g2 = gamma_dot(2, cplx)
    beta = (gamma_dot(w + 1, cplx) * Lf * Uff + Lf * asym_f
            + Lf * g2 * (Uff + float(np.max(np.abs(Dreal))) * Lf)
            + U * float(np.max(np.abs(S.diagonal()))) * 2)
    beta = up(beta)
    return InertiaResult(float(sigma), neg, float(beta), "sparse-superlu-symmetric",
                         time.time() - t0, True, "", fill_nnz=int(L.nnz + Uf.nnz))


def inertia(A, sigma: float, method: str = "auto", dense_max: int = 4096) -> InertiaResult:
    """Inertia count with backward-error bound. If the factorization cannot be completed for lack of
    memory (MemoryError) or SuperLU reports an internal failure (SystemError, which SciPy raises when
    SuperLU's allocation fails), the result is inconclusive with ``resource_failure=True``; the
    certificate then reports completeness as NOT_CHECKED rather than failing."""
    if method == "auto":
        method = "dense" if (not sp.issparse(A) or A.shape[0] <= dense_max and A.shape[0] <= 2500) else "sparse"
    t0 = time.time()
    try:
        if method == "dense":
            return inertia_dense(A.toarray() if sp.issparse(A) else np.asarray(A), sigma)
        return inertia_sparse(A if sp.issparse(A) else sp.csc_matrix(A), sigma)
    except (MemoryError, SystemError) as e:
        msg = " ".join(str(e).split())[:160]
        return InertiaResult(float(sigma), None, float("inf"),
                             "dense-bunch-kaufman" if method == "dense" else "sparse-superlu-symmetric",
                             time.time() - t0, False,
                             f"factorization could not be completed ({type(e).__name__}: {msg})",
                             resource_failure=True)


def count_bounds(res: InertiaResult):
    """Human-readable consequence of an inertia result."""
    if not res.conclusive:
        return None
    return {"at_least_below": f"#(lambda < {res.sigma - res.beta:.17g}) <= {res.negative}",
            "at_most_below": f"{res.negative} <= #(lambda <= {res.sigma + res.beta:.17g})"}
