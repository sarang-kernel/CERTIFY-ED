"""Optional recomputation by algorithmically distinct eigensolvers (statement S5).

Recomputation is corroboration, not certification: the certified statements in
S2-S4 are a-posteriori bounds that do not depend on any eigensolver having run
correctly (they need only matrix-vector products and one factorization). S5 is
useful when (a) no factorization is affordable, so recomputing the lowest part
of the spectrum with an unrelated algorithm is the best available evidence of
completeness, or (b) a user wants to see agreement across implementations.

Each oracle declares the components it shares with the others, so a reader can
see what a disagreement could and could not reveal (Referee 1's point about the
shared Householder tridiagonalisation DSYTRD in LAPACK's ?SYEVD and ?SYEVR).
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._linalg import eigh

ORACLES = {
    "lapack_evd": {"algorithm": "Householder tridiagonalisation (?SYTRD/?HETRD) + divide and conquer",
                   "shares": ["tridiagonalisation with lapack_evr and lapack_ev", "BLAS"]},
    "lapack_evr": {"algorithm": "Householder tridiagonalisation (?SYTRD/?HETRD) + MRRR",
                   "shares": ["tridiagonalisation with lapack_evd and lapack_ev", "BLAS"]},
    "lapack_ev": {"algorithm": "Householder tridiagonalisation (?SYTRD/?HETRD) + implicit QL/QR",
                  "shares": ["tridiagonalisation with lapack_evd and lapack_evr", "BLAS"]},
    "jacobi": {"algorithm": "parallel-order cyclic Jacobi rotations (pure NumPy, no tridiagonalisation)",
               "shares": ["NumPy array arithmetic"]},
    "arpack": {"algorithm": "implicitly restarted Lanczos on matrix-vector products (ARPACK)",
               "shares": ["sparse matvec kernel"]},
    "lobpcg": {"algorithm": "block preconditioned conjugate gradient (SciPy LOBPCG)",
               "shares": ["sparse matvec kernel", "dense Rayleigh-Ritz via LAPACK on small blocks"]},
}


def jacobi_eigvalsh(A: np.ndarray, tol: float = 1e-15, max_sweeps: int = 40) -> np.ndarray:
    """Eigenvalues of a Hermitian matrix by parallel-order cyclic Jacobi.

    Complex Hermitian input is embedded as the real symmetric matrix
    [[Re, -Im], [Im, Re]], whose spectrum is that of A with every eigenvalue
    doubled; the duplicates are removed at the end.
    """
    A = np.asarray(A)
    cplx = np.iscomplexobj(A)
    if cplx:
        A = np.block([[A.real, -A.imag], [A.imag, A.real]])
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if n % 2:
        A = np.pad(A, ((0, 1), (0, 1)))
        A[-1, -1] = np.abs(A).sum() + 1.0          # isolated padding eigenvalue, removed later
        pad = True
    else:
        pad = False
    m = A.shape[0]
    scale = np.linalg.norm(A)
    players = list(range(m))
    for _ in range(max_sweeps):
        off = np.linalg.norm(A - np.diag(np.diag(A)))
        if off <= tol * scale:
            break
        for _r in range(m - 1):
            P = np.array(players[: m // 2])
            Q = np.array(players[m // 2:][::-1])
            apq = A[P, Q]
            app = A[P, P]
            aqq = A[Q, Q]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                tau = (aqq - app) / (2.0 * apq)
                t = np.where(apq == 0.0, 0.0,
                             np.sign(tau) / (np.abs(tau) + np.sqrt(1.0 + tau * tau)))
            t = np.where(np.isnan(t), 0.0, t)
            t = np.where((apq != 0.0) & (tau == 0.0), 1.0, t)
            c = 1.0 / np.sqrt(1.0 + t * t)
            s = t * c
            colP, colQ = A[:, P].copy(), A[:, Q].copy()
            A[:, P] = colP * c - colQ * s
            A[:, Q] = colP * s + colQ * c
            rowP, rowQ = A[P, :].copy(), A[Q, :].copy()
            A[P, :] = c[:, None] * rowP - s[:, None] * rowQ
            A[Q, :] = s[:, None] * rowP + c[:, None] * rowQ
            players = [players[0]] + [players[-1]] + players[1:-1]
    w = np.sort(np.diag(A))
    if pad:
        w = w[:-1]
    if cplx:
        w = w[::2]
    return w


def run_oracle(name: str, A, k: Optional[int] = None, which: str = "SA") -> np.ndarray:
    dense = A.toarray() if sp.issparse(A) else np.asarray(A)
    if name in ("lapack_evd", "lapack_evr", "lapack_ev"):
        # the 1 x 1 case from its entry (_linalg.py: scipy 1.8 to 1.10 refuse it); the record says so (n = 1)
        w = eigh(dense, eigvals_only=True, driver=name.split("_")[1])
    elif name == "jacobi":
        w = jacobi_eigvalsh(dense)
    elif name == "arpack":
        kk = k or 6
        v0 = np.random.default_rng(0).standard_normal(A.shape[0])   # reproducible start vector
        w = spla.eigsh(sp.csr_matrix(A), k=min(kk, A.shape[0] - 2), which=which,
                       return_eigenvectors=False, tol=0, v0=v0)
    elif name == "lobpcg":
        kk = k or 6
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((A.shape[0], kk))
        w, _ = spla.lobpcg(sp.csr_matrix(A), X0, largest=False, tol=1e-12, maxiter=2000)
    else:
        raise KeyError(name)
    return np.sort(np.real(w))


def corroborate(A, claimed: np.ndarray, enc=None, oracles: List[str] = ("lapack_evr", "jacobi")) -> dict:
    claimed = np.sort(np.asarray(claimed, dtype=float))
    k = claimed.size
    scale = float(np.max(np.abs(claimed))) if k else 1.0
    out = {"oracles": {}, "independence": {o: ORACLES[o] for o in oracles}}
    worst = 0.0
    for o in oracles:
        t0 = time.time()
        try:
            w = run_oracle(o, A, k=k)
            w = w[:k]
            d = float(np.max(np.abs(w - claimed[: w.size])))
            out["oracles"][o] = {"max_abs_diff_vs_claim": d, "seconds": time.time() - t0,
                                 "n_compared": int(w.size)}
            if o.startswith("lapack_") and A.shape == (1, 1):
                out["oracles"][o]["note"] = "1 x 1: the eigenvalue is the entry, no LAPACK call (_linalg.eigh)"
            if worst == worst and not d <= worst:       # nan-sticky maximum: a nan difference is kept once seen
                worst = d
        except Exception as e:  # pragma: no cover - reported, not raised
            out["oracles"][o] = {"error": f"{type(e).__name__}: {e}"}
    tol = max(1e-8 * max(scale, 1.0), float(enc.radius.max()) * 10 if enc is not None else 0.0)
    ok = all("error" not in v and v["max_abs_diff_vs_claim"] <= tol for v in out["oracles"].values()) and worst <= tol
    out["tolerance"] = tol
    # agreement is corroboration, not a certified statement: it gets its own status value
    out["status"] = "CORROBORATED" if ok else "INCONCLUSIVE"
    out["summary"] = (f"{len(oracles)} recomputation(s) agree with the claim to {worst:.1e}" if ok else
                      f"recomputation disagrees with the claim by {worst:.2e} (tolerance {tol:.1e})")
    out["note"] = "corroboration only: agreement is evidence, not a certified statement"
    return out
