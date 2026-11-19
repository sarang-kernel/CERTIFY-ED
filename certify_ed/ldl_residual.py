"""A-posteriori backward-error bound of a sparse L D L^H factorization (D-01, docs/proofs/D-01_sparse-residual.md).

Given the stored factors L (unit lower triangular, sparse) and D (real diagonal, the pivots) of a
factorization P S P^T ~ L D L^H produced by SuperLU in symmetric mode, this module bounds

    beta >= || L D L^H - P S P^T ||_2        (exact arithmetic on the stored numbers),

by forming the residual in floating point, row block by row block, and adding a Theorem-1-type bound on
the rounding error of forming it. Nothing about the factorization algorithm is assumed: the bound rests
only on the stored L, D, S and on the standard model of binary64 arithmetic for the few operations this
module performs itself (products, sums, absolute values). This replaces the a-priori bound of 3.1
(gamma_w ||L||_F ||U||_F, Higham 2002 Thm 9.3), which assumed that SuperLU computes every entry of L and U
as a standard inner-product update.

Notation (the document uses the same): n the dimension; c_i the number of stored nonzeros in row i of L;
G := |L| |D| |L|^T (entrywise moduli; exact); F_c := fl(L (D L^H)) the computed product; S_P := P S P^T the
permuted shifted matrix (a re-indexing, exact); E_c := fl(F_c - S_P) the computed residual;
E := L D L^H - S_P the exact residual of the stored factors (Hermitian). Then, entry by entry,

    |E_ij| <= |E_c,ij| / (1 - u) + gamma_{t_ij + 1} G_ij,    t_ij <= min(c_i, c_j),

and ||E||_2 <= ||E||_inf = max_i sum_j |E_ij| (Hermitian). The row sums of the first term are formed from
the lower triangle of E_c (|E_ij| = |E_ji|), those of the second from one sparse matrix-vector product,
(G 1)_i = sum_k |L_ik| |D_k| (sum_j |L_jk|), weighted by gamma_{c_i + 1} (real) or 2 gamma_{c_i + 5} (complex).

Cost: the product L (D L^H) restricted to its lower triangle, about the flop count of the factorization
itself (sum_k nnz(L_{:,k})^2), done in row blocks (memory bounded by `block_mb`) on `threads` threads.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .fp import U, gamma, gamma_dot, up, _GUARD, check_guard_range, abs_up, modulus_up
from .precision import require_binary64

DEFAULT_BLOCK_MB = 64.0      # memory for the sparse temporaries of one row block (F block, |E| block, S block)
_MIN_BLOCK_ROWS = 64


@dataclass
class ResidualBound:
    """What the a-posteriori path measured (recorded in the certificate's inertia evidence)."""
    beta: float                 # upper bound on ||L D L^H - P S P^T||_2 plus the shift rounding term
    residual_inf: float         # upper bound on max_i sum_j |E_c,ij| (the computed residual itself)
    rounding_inf: float         # upper bound on max_i sum_j gamma_{t_ij+1} G_ij (rounding of forming the product)
    shift_term: float           # rounding of forming S = A - sigma I (2u max|S_ii|)
    nnz_L: int
    max_row_nnz_L: int
    nnz_product: int            # nonzeros of the computed lower triangle of L D L^H
    blocks: int
    threads: int
    seconds: float

    def to_dict(self):
        return asdict(self)


def _row_blocks(n: int, L: sp.csr_matrix, bytes_per_entry: int, block_mb: float, min_rows: int = _MIN_BLOCK_ROWS):
    """Row blocks whose products L[I, :] (D L^H)[:, :I_end] fit the block memory: the product's nonzeros lie in
    the filled pattern (docs/proofs/D-01_sparse-residual.md, Lemma R2), so a block has at most 2 nnz(L[I, :])
    entries; three such temporaries (product, difference, moduli) plus the S block are allowed."""
    if n == 0:
        return []
    budget = max(float(block_mb) * 2 ** 20, 1.0)
    per_row = np.diff(L.indptr).astype(np.int64)
    cost = (2 * per_row + 1) * 3 * (bytes_per_entry + 8) + (bytes_per_entry + 8)
    blocks = []
    start = 0
    acc = 0
    for i in range(n):
        acc += int(cost[i])
        if acc > budget and i + 1 - start >= max(1, int(min_rows)):
            blocks.append((start, i + 1))
            start, acc = i + 1, 0
    if start < n:
        blocks.append((start, n))
    return blocks


def sparse_ldl_residual_bound(L: sp.spmatrix, D: np.ndarray, S: sp.spmatrix, perm: np.ndarray,
                              block_mb: Optional[float] = None, threads: Optional[int] = None,
                              min_block_rows: int = _MIN_BLOCK_ROWS) -> ResidualBound:
    """beta >= ||L D L^H - P S P^T||_2 for the stored factors L (unit lower triangular), D (real pivots) and
    the stored Hermitian S, with (P S P^T)_ij = S[p_i, p_j] for p the inverse of `perm` (SuperLU's perm_c:
    L U = Pr S Pc with Pr[perm_r, :] = I, Pc[:, perm_c] = I, i.e. (Pr S Pc)_ij = S[ip_i, ip_j], ip = argsort(perm)).

    The shift rounding term 2u max_i |S_ii| (S = A - sigma I formed in floating point) is included in beta.
    """
    t0 = time.time()
    # fail closed (precision.py): the bound counts binary64 arithmetic on binary64 factors and matrix
    for M, name in ((L, "L"), (D, "D"), (S, "S")):
        require_binary64(M, f"ldl_residual.sparse_ldl_residual_bound: {name}")
    L = sp.csr_matrix(L)
    n = L.shape[0]
    # hypothesis (H3) of docs/proofs/D-01_sparse-residual.md: n <= 2^26 and w <= 2^26 (3.2.2, finding L-3)
    check_guard_range(n=n, w=int(np.diff(L.indptr).max()) if n else None, where="ldl_residual.sparse_ldl_residual_bound")
    D = np.asarray(D)
    cplx = np.iscomplexobj(L.data) or np.iscomplexobj(S.data if sp.issparse(S) else S)
    Dr = np.real(D).astype(float)
    if not (np.all(np.isfinite(L.data)) and np.all(np.isfinite(Dr))):
        raise FloatingPointError("non-finite entries in the stored factors")
    ip = np.argsort(np.asarray(perm))
    S = sp.csr_matrix(S)
    SP = S[ip][:, ip].tocsr()             # exact re-indexing
    DLt = (sp.diags(Dr) @ L.conj().T).tocsc()    # fl(D L^H): one rounding per entry
    # rounding of the product, by rows: gamma_{c_i+1} (|L| |D| |L|^T 1)_i  (one sparse matvec)
    absL = abs_up(L)                                        # complex moduli rounded up (fp.abs_up)
    colsum = np.asarray(absL.sum(axis=0)).ravel()          # sum_j |L_jk|, k-th column
    G1 = absL @ (np.abs(Dr) * colsum)                       # (G 1)_i
    c = np.diff(L.indptr).astype(np.int64)
    g_rows = np.array([gamma_dot(int(ci) + 1, cplx) for ci in np.unique(c)])
    gmap = dict(zip(np.unique(c).tolist(), g_rows.tolist()))
    g_i = np.array([gmap[int(ci)] for ci in c])
    rounding_rows = g_i * G1
    bytes_per = 16 if cplx else 8
    from .cost import residual_threads
    nth = max(1, int(threads)) if threads is not None else residual_threads()
    # the partition depends on block_mb and on L only (not on the thread count), so that the recorded bound is
    # the same on every machine: the blocks are reduced in block order
    blocks = _row_blocks(n, L, bytes_per, block_mb if block_mb is not None else DEFAULT_BLOCK_MB, min_block_rows)
    rows_abs = np.zeros(n)        # sum_j |E_c,ij| assembled from the lower triangle, mirrored
    nnz_prod = 0

    def one_block(blk):
        a, b = blk
        # columns < b of D L^H as a view on the CSC arrays (a slice DLt[:, :b] would copy nearly all of it per block)
        DLt_b = sp.csc_matrix((DLt.data[:DLt.indptr[b]], DLt.indices[:DLt.indptr[b]], DLt.indptr[:b + 1]),
                              shape=(n, b), copy=False)
        Fb = (L[a:b] @ DLt_b).tocsr()                # rows a:b, columns < b (the lower-triangular part and more)
        Eb = (Fb - SP[a:b, :b]).tocsr()
        # entry (r, j) of the block is (i, j) with i = a + r; keep the lower triangle j <= i, i.e. j - r <= a
        Eb = sp.tril(Eb, k=a).tocsr()
        Ab = abs_up(Eb)
        r_sum = np.asarray(Ab.sum(axis=1)).ravel()                      # row sums of the lower triangle
        strict = sp.tril(Ab, k=a - 1)
        c_sum = np.asarray(strict.sum(axis=0)).ravel()                 # column sums of the strict lower part
        return a, b, r_sum, c_sum, int(Eb.nnz)

    results = []
    if nth > 1 and len(blocks) > 1:
        with ThreadPoolExecutor(nth) as ex:
            results = list(ex.map(one_block, blocks))
    else:
        results = [one_block(blk) for blk in blocks]
    for a, b, r_sum, c_sum, nz in results:
        rows_abs[a:b] += r_sum
        rows_abs[:b] += c_sum
        nnz_prod += nz
    # residual part: every row sum has at most n terms plus len(blocks) partial-sum roundings (factor 1 - gamma_m
    # at most), the division by (1 - u) and the three roundings of the inflation are covered by (1 + 4u)(1 + 8u);
    # rounding part: G1 is below (G 1)_i by at most a factor (1 - gamma_{2n+6}) (column sums, scaling, matvec,
    # moduli), g_i carries two roundings (gamma itself) and the product one, so (1 + gamma_{2n+8})(1 + 8u)^2
    # restores an upper bound (docs/proofs/D-01_sparse-residual.md, Step 4, inequality Lambda-R2)
    m = n + len(blocks) + 1
    resid_inf = float(rows_abs.max()) if n else 0.0
    resid_inf = resid_inf * (1.0 + gamma(m)) * (1.0 + 4 * U) * _GUARD
    round_inf = float(rounding_rows.max()) if n else 0.0
    round_inf = round_inf * (1.0 + gamma(2 * n + 8)) * _GUARD * _GUARD
    shift_term = 2.0 * U * float(np.max(modulus_up(S.diagonal()))) if n else 0.0
    beta = up(resid_inf + round_inf + shift_term)
    return ResidualBound(float(beta), float(resid_inf), float(round_inf), float(shift_term), int(L.nnz),
                         int(c.max()) if n else 0, int(nnz_prod), len(blocks), nth, time.time() - t0)
