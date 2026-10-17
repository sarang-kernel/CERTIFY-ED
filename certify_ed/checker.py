"""Independent checks of stored factorizations in ball arithmetic (Arb via python-flint).

The certifier bounds the backward error of an LDL^T factorization under the standard floating-point
model. The functions here re-check the same quantity without that model: every IEEE double is an exact
dyadic rational, so the residual E = L D L^T - A of computed factors against an exactly known matrix A
can be enclosed in ball arithmetic, whose rounding is controlled by the library rather than by an
assumption about the hardware. Together with an exact count of the negative eigenvalues of D
(1x1 pivots by sign, 2x2 pivots by the sign of a rational determinant and trace), this gives

    nu(A + E) = nu(D)   (Sylvester), and  lambda_i(A) >= lambda_i(A + E) - ||E||_2   (Weyl),

so in particular nu(D) = 0 implies lambda_min(A) >= -||E||_2 with ||E||_2 <= ||E||_inf (E symmetric).

Scope: the checker trusts python-flint/Arb and the exact representation of A that it is given.
"""
from fractions import Fraction

import numpy as np


def _arb():
    import flint
    return flint


def negative_count_exact(d: np.ndarray):
    """Number of negative eigenvalues of a real symmetric block-diagonal D (blocks 1x1 or 2x2),
    computed exactly from the stored doubles. Returns (count, ambiguous)."""
    n = d.shape[0]
    i, neg = 0, 0
    while i < n:
        if i + 1 < n and d[i + 1, i] != 0.0:
            a, b, c = Fraction(d[i, i]), Fraction(d[i + 1, i]), Fraction(d[i + 1, i + 1])
            det, tr = a * c - b * b, a + c
            if det == 0:
                return None, True
            neg += 1 if det < 0 else (2 if tr < 0 else 0)
            i += 2
        else:
            if d[i, i] == 0.0:
                return None, True
            neg += 1 if d[i, i] < 0 else 0
            i += 1
    return neg, False


def residual_inf_norm_upper(A: np.ndarray, lu: np.ndarray, d: np.ndarray, prec: int = 128) -> float:
    """Rigorous upper bound on ||lu d lu^T - A||_inf for real matrices; A, lu, d are taken as the
    exact rationals represented by their doubles."""
    flint = _arb()
    old = flint.ctx.prec
    flint.ctx.prec = prec
    try:
        L = flint.arb_mat(lu.tolist())
        D = flint.arb_mat(d.tolist())
        M = L * D * L.transpose() - flint.arb_mat(np.asarray(A, dtype=float).tolist())
        n = A.shape[0]
        worst = 0.0
        for i in range(n):
            s = flint.arb(0)
            for j in range(n):
                s += abs(M[i, j])
            worst = max(worst, float(s.upper()))
        return worst
    finally:
        flint.ctx.prec = old


def check_semidefinite_lower_bound(A: np.ndarray, prec: int = 128):
    """For an exactly represented real symmetric A: factorize in floating point, then check in ball
    arithmetic. Returns dict with nu (exact count for the computed D), bound_E (rigorous upper bound
    on ||E||_2) and, when nu == 0, lambda_min_lower = -bound_E."""
    import scipy.linalg as la
    lu, d, _ = la.ldl(A, lower=True, hermitian=True)
    nu, amb = negative_count_exact(d)
    out = {"nu": nu, "ambiguous": amb}
    if amb:
        return out
    e = residual_inf_norm_upper(A, lu, d, prec)
    out["bound_E"] = e
    if nu == 0:
        out["lambda_min_lower"] = -e
    return out


def residual_inf_norm_exact(A: np.ndarray, lu: np.ndarray, d: np.ndarray, pL: int = 52, pD=None,
                            rows_per_block: int = 256):
    """Exact (integer) version for large matrices with integer-valued A.

    The computed factors are first moved to dyadic grids, Lt = rint(lu 2^pL)/2^pL and
    Dt = rint(d 2^pD)/2^pD; Lt is still unit lower triangular after the row permutation (its unit
    diagonal and zero pattern are preserved), so Sylvester's law applies to Lt Dt Lt^T. The residual
    E = Lt Dt Lt^T - A is then computed exactly with FLINT integer matrices, block of rows by block of
    rows. Returns (upper bound on ||E||_inf as a float rounded up, Dt)."""
    import flint
    if not np.all(A == np.rint(A)):
        raise ValueError("A must be integer-valued")
    n = A.shape[0]
    if pD is None:      # round D at relative 2^-52 of its largest entry
        dmax = float(np.max(np.abs(d))) or 1.0
        pD = int(52 - np.ceil(np.log2(dmax)))
    if np.max(np.abs(lu)) >= 2.0 ** (62 - pL) or np.max(np.abs(d)) > 2.0 ** (52 - pD):
        raise ValueError("factor entries too large for the chosen grids")
    Li = np.rint(np.ldexp(lu, pL)).astype(np.int64)
    Di = np.rint(np.ldexp(d, pD)).astype(np.int64)
    Dt = np.ldexp(Di.astype(np.float64), -pD)            # exact (|Di| < 2^53 checked below)
    if np.max(np.abs(Di)) >= 2 ** 53:
        raise ValueError("D grid too fine")
    from fractions import Fraction
    LT = flint.fmpz_mat(Li.T.tolist())
    Df = flint.fmpz_mat(Di.tolist())
    scale = 2 ** (2 * pL + pD) if 2 * pL + pD >= 0 else Fraction(1, 2 ** -(2 * pL + pD))
    worst = 0
    for r0 in range(0, n, rows_per_block):
        r1 = min(n, r0 + rows_per_block)
        Lr = flint.fmpz_mat(Li[r0:r1].tolist())
        P = (Lr * Df) * LT                                # exact, scaled by 2^(2pL+pD)
        Ar = A[r0:r1]
        for i in range(r1 - r0):
            row_sum = 0
            for j in range(n):
                row_sum += abs(int(P[i, j]) - int(Ar[i, j]) * scale)
            worst = max(worst, row_sum)
    # worst / scale rounded up to a float
    from fractions import Fraction
    q = Fraction(worst, scale)
    f = float(q)
    if Fraction(f) < q:
        f = np.nextafter(f, np.inf)
    return float(f), Dt


def check_semidefinite_exact(A: np.ndarray):
    """Like check_semidefinite_lower_bound, with exact integer arithmetic (A integer-valued)."""
    import scipy.linalg as la
    lu, d, perm = la.ldl(A, lower=True, hermitian=True)
    if not np.all(np.diag(lu[perm]) == 1.0):
        return {"nu": None, "ambiguous": True, "note": "factor lacks unit diagonal"}
    e, Dt = residual_inf_norm_exact(A, lu, d)
    nu, amb = negative_count_exact(Dt)
    out = {"nu": nu, "ambiguous": amb, "bound_E": e}
    if nu == 0 and not amb:
        out["lambda_min_lower"] = -e
    return out
