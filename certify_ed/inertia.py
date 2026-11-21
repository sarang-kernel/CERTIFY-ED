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
             (row permutation == column permutation, verified). Since 3.2 (D-01,
             docs/proofs/D-01_sparse-residual.md) beta is A POSTERIORI: the residual
             L D L^H - P (A - sigma I) P^T of the stored factors is formed in floating
             point, row block by row block, and its rounding is bounded with the
             Theorem-1 accounting (ldl_residual.py). Nothing is assumed about how
             SuperLU computed the factors. The 3.1 a-priori bound (gamma_w ||L||_F
             ||U||_F, Higham 2002 Thm 9.3, plus the asymmetry ||U - D L^H||) is kept
             as method 'sparse-apriori'; it is conditional on SuperLU evaluating every
             entry of L and U as a standard inner-product update, and is not used by
             default. Without pivoting the factorization can be unstable; instability
             shows up as a large beta and yields INCONCLUSIVE, never a wrong count.
             The factorization is not attempted when the fill model (cost.py) predicts
             more memory than the budget (resource_failure -> NOT_CHECKED).
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

from .fp import U, GuardRangeError, gamma_dot, up, fro_upper, modulus_up
from .precision import binary64, require_binary64, summed


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
    residual: Optional[dict] = None  # a-posteriori residual record (ldl_residual.ResidualBound), sparse path

    def to_dict(self):
        return asdict(self)


def _no_count_on_nonfinite(res: InertiaResult) -> InertiaResult:
    """A backward-error bound beta that is not finite (a bound of the accounting overflowed; inf * 0 = nan) gives no
    count: beta = inf and conclusive False."""
    if np.isfinite(res.beta):
        return res
    res.beta, res.negative, res.conclusive = float("inf"), None, False
    res.note = (res.note + "; " if res.note else "") + "backward-error bound not finite (an overflowed bound)"
    return res


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
    A = binary64(np.asarray(A), "operator")     # exact; LAPACK in single precision would break the beta below
    require_binary64(A, "inertia_dense")        # fail closed (precision.py)
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
    # rounding in forming lu d lu^H (two products, inner length <= n) and S; complex moduli rounded up (fp.modulus_up)
    alu = modulus_up(lu)
    absprod = alu @ modulus_up(d) @ alu.T
    g = gamma_dot(2 * n + 2, cplx)
    beta = fro_upper(R) + g * fro_upper(absprod) + U * float(np.max(modulus_up(np.diag(S)))) * 2
    beta = up(beta)
    res = InertiaResult(float(sigma), None if amb else int(neg), float(beta), "dense-bunch-kaufman",
                        time.time() - t0, not amb, "zero pivot or singular 2x2 block" if amb else "")
    return _no_count_on_nonfinite(res)


def inertia_sparse(A: sp.spmatrix, sigma: float, residual: str = "posteriori", block_mb: Optional[float] = None,
                   threads: Optional[int] = None) -> InertiaResult:
    """Inertia of A - sigma I from SuperLU's symmetric-mode LU. residual='posteriori' (default, D-01): beta from
    the measured residual of the stored factors (ldl_residual.sparse_ldl_residual_bound); residual='apriori':
    the 3.1 bound gamma_{w+1} ||L||_F ||U||_F + ... (conditional on SuperLU's arithmetic, see module docstring)."""
    t0 = time.time()
    # exact, and before csc_matrix (which sums repeated entries in the narrower format); SuperLU in single precision
    # would break the beta below
    A = require_binary64(sp.csc_matrix(summed(binary64(A, "operator"), "operator")), "inertia_sparse")   # fail closed
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
    if not (np.all(np.isfinite(L.data)) and np.all(np.isfinite(Dreal))):
        return InertiaResult(float(sigma), None, float("inf"), "sparse-superlu-symmetric",
                             time.time() - t0, False, "non-finite entries in the stored factors")
    if residual == "apriori":
        # 3.1 bound: beta <= gamma_w ||L||_F ||U||_F  (LU backward error, Higham Thm 9.3)
        #            + ||L||_F ||U - D L^H||_F   (symmetrisation)  + rounding of that difference
        w = int(np.diff(L.indptr).max()) if n else 1
        Lf, Uff = fro_upper(L), fro_upper(Uf)
        asym = (Uf - sp.diags(Dreal) @ L.conj().T).tocsr()
        asym_f = fro_upper(asym) if asym.nnz else 0.0
        g2 = gamma_dot(2, cplx)
        beta = (gamma_dot(w + 1, cplx) * Lf * Uff + Lf * asym_f
                + Lf * g2 * (Uff + float(np.max(np.abs(Dreal))) * Lf)
                + U * float(np.max(modulus_up(S.diagonal()))) * 2)
        beta = up(beta)
        return _no_count_on_nonfinite(InertiaResult(float(sigma), neg, float(beta), "sparse-superlu-apriori",
                                                    time.time() - t0, True, "", fill_nnz=int(L.nnz + Uf.nnz)))
    from .ldl_residual import sparse_ldl_residual_bound
    del Uf
    rb = sparse_ldl_residual_bound(L, dU, S, f.perm_c, block_mb=block_mb, threads=threads)
    return _no_count_on_nonfinite(InertiaResult(float(sigma), neg, float(rb.beta), "sparse-superlu-residual",
                                                time.time() - t0, True, "", fill_nnz=int(2 * L.nnz - n),
                                                residual=rb.to_dict()))


def inertia(A, sigma: float, method: str = "auto", dense_max: int = 4096, budget_mb: Optional[float] = None) -> InertiaResult:
    """Inertia count with backward-error bound. If the factorization cannot be completed for lack of
    memory (MemoryError) or SuperLU reports an internal failure (SystemError, which SciPy raises when
    SuperLU's allocation fails), the result is inconclusive with ``resource_failure=True``; the
    certificate then reports completeness as NOT_CHECKED rather than failing.

    method='auto' (cost.recommend_inertia_method, D-08): dense for a dense matrix or a sparse one of dimension
    <= 2500 (the 3.1 rule), except that a sparse matrix whose dense LDL^T would exceed `budget_mb` (or, without
    a budget, the available physical memory) is factorized by SuperLU. The choice is recorded in the result's
    `method`; it decides which path is tried, never what a count says. method='sparse' uses the a-posteriori
    residual bound (D-01); method='sparse-apriori' the 3.1 bound.

    Memory gate (D-01): a sparse factorization whose predicted peak (cost.sparse_inertia_mb, fill model plus the
    residual's own arrays) exceeds `budget_mb` is not attempted and is reported as a resource failure. Without a
    budget (3.2.2, N-03) the limit is the memory left under the process memory cap (cost.process_memory_cap: job
    object or RLIMIT_AS), for the dense path (cost.dense_inertia_mb) as well; without a cap nothing is gated.

    A matrix in a narrower format is converted exactly to binary64 first (precision.py)."""
    A = summed(binary64(A if sp.issparse(A) else np.asarray(A), "operator"), "operator")
    if method == "auto":
        from .cost import recommend_inertia_method
        method, _why = recommend_inertia_method(int(A.shape[0]), int(A.nnz) if sp.issparse(A) else None,
                                                not sp.issparse(A), bool(np.iscomplexobj(A.data if sp.issparse(A) else A)),
                                                budget_mb, dense_max=min(dense_max, 2500))
    t0 = time.time()
    label = "dense-bunch-kaufman" if method == "dense" else ("sparse-superlu-apriori" if method == "sparse-apriori"
                                                             else "sparse-superlu-residual")
    if method != "dense" and budget_mb is not None:
        from .cost import sparse_inertia_mb
        n_ = int(A.shape[0])
        nnz_ = int(A.nnz) if sp.issparse(A) else n_ * n_
        need = sparse_inertia_mb(n_, nnz_, bool(np.iscomplexobj(A.data if sp.issparse(A) else A)))
        if need > budget_mb:
            return InertiaResult(float(sigma), None, float("inf"), label, time.time() - t0, False,
                                 f"factorization not attempted: predicted peak {need:.0f} MB exceeds the memory "
                                 f"budget {budget_mb:.0f} MB (cost.sparse_inertia_mb; out-of-core factorization "
                                 f"is not implemented)", resource_failure=True)
    if budget_mb is None:
        # 3.2.2 (N-03): without a budget the memory left under the process memory cap (job object / RLIMIT_AS) is the
        # limit, for the dense and the sparse path, so that a factorization that cannot fit is not started
        from .cost import dense_inertia_mb, memory_cap_note, memory_limit_mb, sparse_inertia_mb
        limit, cap = memory_limit_mb(None)
        if cap is not None:
            n_ = int(A.shape[0])
            nnz_ = int(A.nnz) if sp.issparse(A) else n_ * n_
            cplx_ = bool(np.iscomplexobj(A.data if sp.issparse(A) else A))
            need, est = ((dense_inertia_mb(n_, cplx_), "cost.dense_inertia_mb") if method == "dense"
                         else (sparse_inertia_mb(n_, nnz_, cplx_), "cost.sparse_inertia_mb"))
            if need > limit:
                return InertiaResult(float(sigma), None, float("inf"), label, time.time() - t0, False,
                                     f"factorization not attempted: predicted peak {need:.0f} MB exceeds "
                                     f"{memory_cap_note(cap)} ({est}; out-of-core factorization is not implemented)",
                                     resource_failure=True)
    try:
        if method == "dense":
            return inertia_dense(A.toarray() if sp.issparse(A) else np.asarray(A), sigma)
        return inertia_sparse(A if sp.issparse(A) else sp.csc_matrix(A), sigma,
                              residual="apriori" if method == "sparse-apriori" else "posteriori")
    except (MemoryError, SystemError) as e:
        msg = " ".join(str(e).split())[:160]
        return InertiaResult(float(sigma), None, float("inf"), label, time.time() - t0, False,
                             f"factorization could not be completed ({type(e).__name__}: {msg})",
                             resource_failure=True)
    except GuardRangeError as e:
        # 3.2.2 (finding L-3): the backward-error bound would use a floating-point guard outside the range in which
        # it is established (a norm of more than 2^26 entries, or a row width above 2^26); no count is reported
        return InertiaResult(float(sigma), None, float("inf"), label, time.time() - t0, False,
                             f"backward-error bound not formed: {e}")


def count_bounds(res: InertiaResult):
    """Human-readable consequence of an inertia result."""
    if not res.conclusive:
        return None
    return {"at_least_below": f"#(lambda < {res.sigma - res.beta:.17g}) <= {res.negative}",
            "at_most_below": f"{res.negative} <= #(lambda <= {res.sigma + res.beta:.17g})"}
