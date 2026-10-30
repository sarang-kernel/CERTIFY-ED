"""C-16: exact eigenvalue counts and multiplicities in a rational bracket, with no floating-point assumption.

The declared model is taken as the exact rational matrix K = D^-1 H D of checker.exact_matrix (similar to H, so
the eigenvalues and their multiplicities are those of H), or as a symmetry-sector block (N, G) of exact_sectors
(generalized problem N v = lambda den G v with G diagonal positive, equivalent to the ordinary eigenproblem of
G^-1 N). The characteristic polynomial p of den K (resp. of den G^-1 N) is computed exactly (python-flint), so
that the eigenvalues of H are the roots of p(den x) with the same multiplicities (Hermitian H is
diagonalizable: algebraic multiplicity = geometric multiplicity = the number of equal eigenvalues).

What is decided for a rational bracket [a, b]: the number of eigenvalues in [a, b] counted with multiplicity,
the number of distinct eigenvalues there, and the multiplicity of each, from the squarefree factorization
p = c prod_j s_j^j (Yun) and Sturm sequences of the squarefree s_j evaluated at a and b in exact arithmetic.
Hence "exactly g eigenvalues in [a, b], all equal" (one root of multiplicity g) or "exactly g eigenvalues in
[a, b], not all equal (d distinct values)" are both decidable, and so is any count. A bracket end that is
itself a root is moved outward by 2^-200 (reported). Derivation: docs/proofs/C16_exact_bracket.md.

Cost: the characteristic polynomial of an n x n integer matrix is the expensive step (about 0.7 s at n = 200,
16 s at n = 400 for small entries); the dimension cap MAX_DIM guards it.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np

from .checker import ExactMatrix

__all__ = ["MAX_DIM", "exact_charpoly", "sector_charpoly", "sturm_count", "roots_in_bracket", "exact_bracket",
           "have_flint"]

MAX_DIM = 600            # charpoly dimension cap (real embedding counts double)
NUDGE = Fraction(1, 2 ** 200)


def have_flint() -> bool:
    try:
        import flint  # noqa: F401
        return True
    except Exception:
        return False


def _frac(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


# ------------------------------------------------------------------------------------ characteristic polynomials
def exact_charpoly(ex: ExactMatrix):
    """(p, den, factor): p in Z[x] (flint.fmpz_poly) with p(den * lambda) = 0 exactly for the eigenvalues lambda of
    the declared model, each with its multiplicity times `factor` (factor = 2 when K is complex and the real
    embedding [[Re, -Im], [Im, Re]], whose eigenvalues are those of K twice, had to be used)."""
    import flint
    n = int(ex.weights.size)
    cplx = bool(np.any(ex.im != 0))
    if cplx:
        if 2 * n > MAX_DIM:
            raise ValueError(f"dimension {n} (complex: embedding {2 * n}) above MAX_DIM {MAX_DIM}")
        M = np.zeros((2 * n, 2 * n), dtype=object); M[:] = 0
        for r, c, a, b in zip(ex.rows.tolist(), ex.cols.tolist(), ex.re.tolist(), ex.im.tolist()):
            M[r, c] += a; M[r + n, c + n] += a
            M[r, c + n] -= b; M[r + n, c] += b
        factor = 2
    else:
        if n > MAX_DIM:
            raise ValueError(f"dimension {n} above MAX_DIM {MAX_DIM}")
        M = np.zeros((n, n), dtype=object); M[:] = 0
        for r, c, a in zip(ex.rows.tolist(), ex.cols.tolist(), ex.re.tolist()):
            M[r, c] += a
        factor = 1
    p = flint.fmpz_mat([[int(x) for x in row] for row in M]).charpoly()
    return p, int(ex.den), factor


def sector_charpoly(Are, Aim, D, den: int):
    """(p, den, factor) for a sector block of exact_sectors.sector_integer_matrix: N = V^H (W K den) V and the Gram
    diagonal D = V^H W V. The eigenvalues mu of H restricted to the sector satisfy det(N - mu den D) = 0, i.e. they
    are the eigenvalues of (den) D^-1 N / den; p is the characteristic polynomial of D^-1 N (rational, cleared to
    Z[x] by a positive integer), so p(den * mu) = 0 with the sector multiplicities (times `factor`)."""
    import flint
    m = int(Are.shape[0])
    cplx = any(int(x) != 0 for x in np.asarray(Aim).ravel())
    if cplx:
        if 2 * m > MAX_DIM:
            raise ValueError(f"sector dimension {m} (complex: embedding {2 * m}) above MAX_DIM {MAX_DIM}")
        rows = []
        for i in range(m):
            rows.append([flint.fmpq(int(Are[i, j]), int(D[i])) for j in range(m)]
                        + [flint.fmpq(-int(Aim[i, j]), int(D[i])) for j in range(m)])
        for i in range(m):
            rows.append([flint.fmpq(int(Aim[i, j]), int(D[i])) for j in range(m)]
                        + [flint.fmpq(int(Are[i, j]), int(D[i])) for j in range(m)])
        factor = 2
    else:
        if m > MAX_DIM:
            raise ValueError(f"sector dimension {m} above MAX_DIM {MAX_DIM}")
        rows = [[flint.fmpq(int(Are[i, j]), int(D[i])) for j in range(m)] for i in range(m)]
        factor = 1
    q = flint.fmpq_mat(rows).charpoly()
    p = q.numer()                     # q.denom() > 0: a positive scaling leaves the roots and multiplicities
    return p, int(den), factor


# ------------------------------------------------------------------------------------ Sturm sequences
def _primitive(p):
    c = p.content()
    if int(c) in (0, 1):
        return p
    import flint
    return flint.fmpz_poly([int(a) // int(c) for a in p.coeffs()])


def sturm_sequence(s) -> list:
    """Canonical Sturm sequence of a squarefree s in Z[x], each remainder negated and made primitive (a positive
    scaling, which leaves every sign unchanged)."""
    import flint
    seq = [s, _primitive(s.derivative())]
    while not seq[-1].is_zero() and seq[-1].degree() > 0:
        r = divmod(flint.fmpq_poly(seq[-2]), flint.fmpq_poly(seq[-1]))[1]
        if r.is_zero():
            break
        seq.append(_primitive(-r.numer()))          # -rem, numerator (denominator > 0)
    return seq


def _sign_changes(seq, x: Fraction) -> int:
    import flint
    xq = flint.fmpq(x.numerator, x.denominator)
    signs = []
    for q in seq:
        v = q(xq)
        if v != 0:
            signs.append(1 if v > 0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def sturm_count(s, a: Fraction, b: Fraction) -> int:
    """Number of distinct real roots of the squarefree s in the open interval (a, b), a < b, s(a) s(b) != 0
    (Sturm's theorem)."""
    import flint
    aq, bq = flint.fmpq(a.numerator, a.denominator), flint.fmpq(b.numerator, b.denominator)
    if s(aq) == 0 or s(bq) == 0:
        raise ValueError("a bracket end is a root")
    seq = sturm_sequence(s)
    return _sign_changes(seq, a) - _sign_changes(seq, b)


def roots_in_bracket(p, a: Fraction, b: Fraction, factor: int = 1) -> Dict[str, object]:
    """Exact root statistics of p in the closed bracket [a, b]: the ends are moved outward by NUDGE while they are
    roots of a squarefree factor (so the final open bracket contains [a, b] and no root sits on an end), then for
    each squarefree part s_j (multiplicity j) the number of distinct roots is counted by Sturm. Returns
    {count (with multiplicity, divided by factor), distinct, multiplicities {j: distinct roots of multiplicity j},
    bracket_used [a', b'] (Fractions), nudged (bool)}."""
    import flint
    _c, parts = p.factor_squarefree()
    a2, b2 = _frac(a), _frac(b)
    nudged = False
    for _ in range(64):
        aq, bq = flint.fmpq(a2.numerator, a2.denominator), flint.fmpq(b2.numerator, b2.denominator)
        if all(s(aq) != 0 and s(bq) != 0 for s, _j in parts):
            break
        nudged = True
        a2, b2 = a2 - NUDGE, b2 + NUDGE
    else:
        raise ValueError("bracket ends keep hitting roots")
    mult: Dict[int, int] = {}
    count = 0
    distinct = 0
    for s, j in parts:
        d = sturm_count(s, a2, b2)
        if d:
            jj = int(j)
            if jj % factor:
                raise ValueError("embedding multiplicities are not even: not a real-eigenvalue problem")
            mult[jj // factor] = mult.get(jj // factor, 0) + d
            count += d * (jj // factor)
            distinct += d
    return {"count": count, "distinct": distinct, "multiplicities": mult, "bracket_used": (a2, b2), "nudged": nudged}


# ------------------------------------------------------------------------------------ the statement
_METHOD = ("exact characteristic polynomial (python-flint), squarefree factorization (Yun) and Sturm sequences at "
           "the rational bracket ends; C-16, docs/proofs/C16_exact_bracket.md")
_SRC_MATRIX = "exact scaled matrix den*K (checker.exact_matrix), similar to H"
_SRC_SECTOR = "symmetry-sector block (exact_sectors.sector_integer_matrix), generalized problem N v = mu den D v"


def _record(p, den_: int, factor: int, a_: Fraction, b_: Fraction, g: int, dim: int, src: str) -> dict:
    """The C-16 record for the bracket [a_, b_] (eigenvalue units) from the characteristic polynomial."""
    if not a_ < b_:
        return {"status": "NOT_CHECKED", "reason": "empty bracket", "g_claimed": int(g)}
    r = roots_in_bracket(p, a_ * den_, b_ * den_, factor)
    a_used, b_used = r["bracket_used"][0] / den_, r["bracket_used"][1] / den_
    rec = {"g_claimed": int(g), "count": int(r["count"]), "distinct_values": int(r["distinct"]),
           "multiplicities": {str(k): int(v) for k, v in sorted(r["multiplicities"].items())},
           "bracket": [float(np.nextafter(float(a_used), -np.inf)), float(np.nextafter(float(b_used), np.inf))],
           "bracket_exact": [f"{a_used.numerator}/{a_used.denominator}", f"{b_used.numerator}/{b_used.denominator}"],
           "bracket_nudged": bool(r["nudged"]), "charpoly_degree": int(p.degree()), "dimension": int(dim),
           "source": src, "no_floating_point_assumption": True, "method": _METHOD}
    w = float(b_used - a_used)
    lo, hi = rec["bracket"]
    if rec["count"] == g:
        rec["status"] = "CERTIFIED"
        rec["equal"] = rec["distinct_values"] == 1
        if rec["equal"]:
            rec["statement"] = (f"exactly {g} eigenvalues of the declared model in [{lo:.15g}, {hi:.15g}] (width {w:.2e}), "
                                f"all equal: one eigenvalue of multiplicity {g} (exact arithmetic, no floating-point "
                                f"assumption)")
        else:
            rec["statement"] = (f"exactly {g} eigenvalues of the declared model in [{lo:.15g}, {hi:.15g}] (width {w:.2e}), "
                                f"NOT all equal: {rec['distinct_values']} distinct eigenvalues with multiplicities "
                                f"{rec['multiplicities']} (exact arithmetic, no floating-point assumption)")
    else:
        rec["status"] = "FAILED"
        rec["equal"] = None
        rec["reason"] = (f"exact count {rec['count']} in [{lo:.15g}, {hi:.15g}] differs from the {g} claimed: the "
                         f"floating-point bracket statement is contradicted")
    return rec


def exact_bracket(source, a, b, g_claimed: int, den: Optional[int] = None) -> Dict[str, object]:
    """C-16 record for one cluster: `source` is an ExactMatrix (exact_charpoly) or a tuple (Are, Aim, D) of a sector
    block with `den` (sector_charpoly); a, b: the bracket in eigenvalue units (floats or Fractions, taken exactly);
    g_claimed: the number of claimed pairs in the cluster.

    status CERTIFIED when the exact count in [a, b] equals g_claimed, with `equal` True (one eigenvalue of
    multiplicity g: exact degeneracy) or False (d > 1 distinct eigenvalues: not degenerate); FAILED when the count
    differs from g_claimed (the floating-point statement 'exactly g in the bracket' is contradicted); NOT_CHECKED
    above the dimension cap."""
    try:
        if isinstance(source, ExactMatrix):
            p, den_, factor = exact_charpoly(source)
            src, dim = _SRC_MATRIX, int(source.weights.size)
        else:
            Are, Aim, D = source
            p, den_, factor = sector_charpoly(Are, Aim, D, int(den))
            src, dim = _SRC_SECTOR, int(Are.shape[0])
    except ValueError as e:
        return {"status": "NOT_CHECKED", "reason": str(e), "g_claimed": int(g_claimed)}
    return _record(p, den_, factor, _frac(a if isinstance(a, Fraction) else float(a)),
                   _frac(b if isinstance(b, Fraction) else float(b)), int(g_claimed), dim, src)


def exact_brackets_for_clusters(ex: ExactMatrix, recs: List[dict]) -> List[Optional[dict]]:
    """One exact_bracket record per S8 cluster record (None where S8 has no inner bracket), sharing one
    characteristic polynomial; the bracket is the S8 inner bracket (Proposition 14's I_in)."""
    try:
        p, den_, factor = exact_charpoly(ex)
    except ValueError as e:
        return [{"status": "NOT_CHECKED", "reason": str(e), "g_claimed": r.get("g_claimed")} for r in recs]
    out: List[Optional[dict]] = []
    for r in recs:
        if "bracket_inner" not in r:
            out.append(None)
            continue
        a, b = r["bracket_inner"]
        out.append(_record(p, den_, factor, Fraction(float(a)), Fraction(float(b)), int(r["g_claimed"]),
                           int(ex.weights.size), _SRC_MATRIX))
    return out
