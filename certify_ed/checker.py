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

from .precision import check_values, claimed_reals


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


# --------------------------------------------------------------------------- exact re-check layer
from collections import namedtuple

ExactMatrix = namedtuple("ExactMatrix", "rows cols re im den weights")
ExactMatrix.__doc__ = """The declared model in exact form. K = (re + i im) / den is the scaled matrix D^-1 H D with
D = diag(sqrt(weights)) (weights: positive integers, one per basis state, constant on symmetry orbits).
W K with W = diag(weights) equals D H D: Hermitian, integer after scaling by den. For q = 2 spaces all
weights are 1 and K = H."""


def exact_matrix(spec, states=None) -> ExactMatrix:
    """The declared model as an exact matrix (see ExactMatrix).

    Couplings are the binary64 numbers in the spec, taken exactly. Local operators are taken in the
    scaled basis of LocalSpace.exact_weights, where every matrix element is rational, so every local
    space (spin-S in all conventions, fermions, bosons) is covered. Entries are Python ints in object
    arrays, duplicates summed, sorted by (row, col)."""
    import itertools
    from math import lcm
    from .construct import full_state_set, _popcount
    st = states if states is not None else full_state_set(spec)
    s0 = st.states_in_order()
    n = st.size
    N, q, local = spec.n_sites, spec.q, spec.local
    lw = local.exact_weights()
    weights = np.ones(n, dtype=object)
    for site in range(N):
        digit = (s0 // q ** (N - 1 - site)) % q
        weights = weights * np.array(lw, dtype=object)[digit]
    col_idx = np.arange(n)
    groups = []
    for term in spec.terms:
        ex = []
        for letter in term.op:
            row = []
            for delta, vals in local.exact_scaled_components(letter):
                d = lcm(*[x.denominator for v in vals for x in v])
                row.append((delta, np.array([int(v[0] * d) for v in vals], dtype=object),
                            np.array([int(v[1] * d) for v in vals], dtype=object), d))
            ex.append(row)
        for c, sites in term.couplings:
            cr, ci = Fraction(float(np.real(c))), Fraction(float(np.imag(c)))
            for choice in itertools.product(*[range(len(r)) for r in ex]):
                s = s0.copy()
                are = np.ones(n, dtype=object); aim = np.zeros(n, dtype=object)
                ok = np.ones(n, dtype=bool)
                dprod = 1
                for pos in range(len(term.op) - 1, -1, -1):
                    letter, site = term.op[pos], sites[pos]
                    delta, nr, ni, d = ex[pos][choice[pos]]
                    place = q ** (N - 1 - site)
                    digit = (s // place) % q
                    new = digit + delta
                    valid = (new >= 0) & (new < q)
                    fr_, fi_ = nr[digit], ni[digit]
                    if local.jw_letter(letter):
                        side = s & ((1 << (N - 1 - site)) - 1) if local.jw_string == "higher" else s >> (N - site)
                        sgn = np.where(_popcount(side) % 2 == 1, -1, 1).astype(object)
                        fr_, fi_ = fr_ * sgn, fi_ * sgn
                    are, aim = are * fr_ - aim * fi_, are * fi_ + aim * fr_
                    dprod *= d
                    ok &= valid & ((fr_ != 0) | (fi_ != 0))
                    s = np.where(valid, s + delta * place, s)
                m = ok & ((are != 0) | (aim != 0))
                if not np.any(m):
                    continue
                r, found = st.index_of(s[m])
                groups.append((r[found], col_idx[m][found], are[m][found], aim[m][found], cr / dprod, ci / dprod))
    if not groups:
        z = np.zeros(0, dtype=object)
        return ExactMatrix(np.zeros(0, np.int64), np.zeros(0, np.int64), z, z, 1, weights)
    den = 1
    for g in groups:
        den = lcm(den, g[4].denominator, g[5].denominator)
    R, C, RE, IM = [], [], [], []
    for r, c, re, im, kr, ki in groups:
        ar, ai = int(kr * den), int(ki * den)
        R.append(r); C.append(c)
        RE.append(re * ar - im * ai); IM.append(re * ai + im * ar)
    R = np.concatenate(R).astype(np.int64); C = np.concatenate(C).astype(np.int64)
    RE = np.concatenate(RE); IM = np.concatenate(IM)
    order = np.lexsort((C, R))
    R, C, RE, IM = R[order], C[order], RE[order], IM[order]
    key = R * n + C
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    RE = np.add.reduceat(RE, starts); IM = np.add.reduceat(IM, starts)
    R, C = R[starts], C[starts]
    keep = (RE != 0) | (IM != 0)
    return ExactMatrix(R[keep], C[keep], RE[keep], IM[keep], den, weights)


def exact_dense(ex: ExactMatrix) -> np.ndarray:
    """Floating-point H = D K D^-1 (for comparisons and tests only)."""
    n = ex.weights.size
    d = np.sqrt(np.array([float(w) for w in ex.weights]))
    H = np.zeros((n, n), dtype=complex)
    H[ex.rows, ex.cols] = [complex(float(a) / ex.den, float(b) / ex.den) for a, b in zip(ex.re, ex.im)]
    return (d[:, None] * H) / d[None, :]


def is_hermitian_exact(ex: ExactMatrix) -> bool:
    """W K Hermitian, checked exactly: w_r K_rc == conj(w_c K_cr) for every stored entry."""
    w = ex.weights
    ent = {(int(r), int(c)): (a * w[r], b * w[r]) for r, c, a, b in zip(ex.rows, ex.cols, ex.re, ex.im)}
    return all(ent.get((c, r), (0, 0)) == (a, -b) for (r, c), (a, b) in ent.items())


def _float_vector_to_int(x):
    """x (binary64, real or complex) = (Xr + i Xi) 2^E exactly, Python-int object arrays."""
    x = np.asarray(x)
    parts = [np.real(x).astype(float), np.imag(x).astype(float) if np.iscomplexobj(x) else np.zeros(x.shape)]
    mants, exps = [], []
    for p in parts:
        m, e = np.frexp(p)
        mants.append((m * 2.0 ** 53).astype(np.int64)); exps.append(e.astype(np.int64) - 53)
    nz = [(mm != 0) for mm in mants]
    E = min([int(e[z].min()) for e, z in zip(exps, nz) if np.any(z)] + [0])
    outs = []
    for mm, e in zip(mants, exps):
        outs.append(np.array([int(a) << int(b - E) if a != 0 else 0 for a, b in zip(mm.tolist(), e.tolist())],
                             dtype=object))
    return outs[0], outs[1], E


def scaled_test_vector(ex: ExactMatrix, x):
    """y = fl(x / sqrt(w)): the test vector is then x' = D y, which equals x up to rounding. Any y gives
    a valid Krylov-Bogolyubov interval; this choice makes x' the claimed vector to about one ulp."""
    if all(int(w) == 1 for w in ex.weights):
        return np.asarray(x)
    d = np.sqrt(np.array([float(w) for w in ex.weights]))
    return np.asarray(x) / d


def radius_exact(ex: ExactMatrix, yr, yi, lam):
    """Exact upper bound (a dyadic Fraction) on ||H x' - lam x'|| / ||x'|| for x' = D y, y = (yr + i yi)
    in integer form: with K y = (KR + i KI)/den, ||D (K y - lam y)||^2 = sum_r w_r |(K y - lam y)_r|^2
    and ||x'||^2 = sum_r w_r |y_r|^2, all rational.

    Hypothesis (Krylov-Bogolyubov, Theorem 1): the test vector is nonzero, ||x'||^2 > 0. For a zero test
    vector (a zero claimed vector, or a vector whose exact projection onto the claimed sector vanishes) no
    interval exists: the function returns None and every caller reports INCONCLUSIVE (review of
    2026-10-21, IMP-18: the earlier code returned the point interval, radius 0, for such a vector)."""
    from math import isqrt
    n = yr.size
    w = ex.weights
    kr = np.zeros(n, dtype=object); ki = np.zeros(n, dtype=object)
    C, R = ex.cols, ex.rows
    np.add.at(kr, R, ex.re * yr[C] - ex.im * yi[C]); np.add.at(ki, R, ex.re * yi[C] + ex.im * yr[C])
    L = Fraction(float(lam)); a, b = L.numerator, L.denominator
    rr = kr * b - yr * (a * ex.den); ri = ki * b - yi * (a * ex.den)       # = b den (K y - lam y)
    num2 = int(np.sum(w * (rr * rr + ri * ri)))
    x2 = int(np.sum(w * (yr * yr + yi * yi)))
    if x2 == 0:
        return None
    if num2 == 0:
        return Fraction(0)
    D2 = (b * ex.den) ** 2 * x2
    k = 80
    return Fraction(isqrt((num2 << (2 * k)) // D2 + 1) + 1, 1 << k)


def _round_up_float(q: Fraction) -> float:
    f = float(q)
    if Fraction(f) < q:
        f = float(np.nextafter(f, np.inf))
    return f


def exact_eigenpair_radii(spec, lam, X, states=None, exact=None):
    """Residual radii of claimed eigenpairs against the declared model, in exact arithmetic.

    For each claimed (lam_i, x_i) the radius rho_i of the Krylov-Bogolyubov interval [lam_i - rho_i,
    lam_i + rho_i] of the test vector x_i' = D fl(x_i / d) (= x_i up to one rounding) is computed exactly
    and rounded up; for Hermitian H and x_i' != 0 the interval contains an eigenvalue. A zero test vector
    has no interval: its radius is reported as inf and its index is listed under 'zero_test_vectors'. A claimed
    value that is not finite (NaN, an infinity, or a non-finite imaginary part: precision.claimed_reals) has no
    interval either: radius inf, index under 'non_finite_values'. Hermiticity of the declared model is checked exactly.
    No floating-point assumption is involved."""
    check_values(lam, "lam")          # a value binary64 cannot hold is refused, as in certify() (precision.py)
    lam = np.atleast_1d(claimed_reals(lam))
    ex = exact if exact is not None else exact_matrix(spec, states)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    radii, zero, bad = [], [], []
    for i, lam_i in enumerate(lam):
        if not np.isfinite(lam_i):
            bad.append(int(i))
            radii.append(float("inf"))
            continue
        yr, yi, _ = _float_vector_to_int(scaled_test_vector(ex, X[:, i]))
        r = radius_exact(ex, yr, yi, lam_i)
        if r is None:
            zero.append(int(i))
        radii.append(float("inf") if r is None else _round_up_float(r))
    return {"radii": radii, "hermitian": is_hermitian_exact(ex), "den": int(ex.den), "nnz": int(len(ex.rows)),
            "zero_test_vectors": zero, "non_finite_values": bad}


def inertia_exact(M, rows_per_block: int = 256):
    """Inertia of an exact symmetric integer matrix M (object ndarray of Python ints, or flint.fmpz_mat).

    A floating-point LDL^T proposes factors; they are moved to dyadic grids (Lt = rint(L 2^52)/2^52,
    Dt with 52 bits relative to its largest entry) and the residual Lt Dt Lt^T - M is computed exactly
    with FLINT integers, block of rows by block of rows (memory O(n^2) small integers).
    Returns (n_negative, n_positive, beta) with beta >= ||Lt Dt Lt^T - M||_inf (a Fraction), or
    (None, None, None) when a pivot sign is ambiguous. Consequences (Sylvester + Weyl):
        #(eig M < -beta) <= n_negative,   #(eig M > beta) <= n_positive."""
    import flint
    import scipy.linalg as la
    if not isinstance(M, np.ndarray):
        n = M.nrows()
        M = np.array([int(x) for x in M.entries()], dtype=object).reshape(n, n)
    n = M.shape[0]
    maxbits = max([abs(int(x)).bit_length() for x in M.ravel()] + [1])
    s_down = max(0, maxbits - 900)
    if s_down:
        Af = np.array([[float(int(x) >> s_down) if x >= 0 else -float((-int(x)) >> s_down) for x in row]
                       for row in M])
    else:
        Af = M.astype(float)
    lu, d, perm = la.ldl(Af, lower=True, hermitian=True, overwrite_a=True)
    del Af
    if not np.all(np.diag(lu[perm]) == 1.0) or np.any(np.triu(lu[perm], 1)):
        # Sylvester's law needs an invertible factor: unit lower triangular after the row permutation
        # (then the rounded Lt is too, since rint keeps exact 0 and 1 * 2^52)
        raise ValueError("LDL factor not unit triangular after permutation")
    pL = 52
    dmax = float(np.max(np.abs(d))) or 1.0
    pD = int(52 - np.ceil(np.log2(dmax)))
    Li = np.rint(np.ldexp(lu, pL)).astype(np.int64)
    del lu
    Di = np.rint(np.ldexp(d, pD)).astype(np.int64)
    del d
    neg, amb = negative_count_exact(np.ldexp(Di.astype(np.float64), -pD))
    if amb:
        return None, None, None
    sh = 2 * pL + pD - s_down
    LT = flint.fmpz_mat(Li.T.tolist())
    Dm = flint.fmpz_mat(Di.tolist())
    del Di
    two = flint.fmpz(2)
    worst = 0
    for r0 in range(0, n, rows_per_block):
        r1 = min(n, r0 + rows_per_block)
        Pb = (flint.fmpz_mat(Li[r0:r1].tolist()) * Dm) * LT
        Ab = flint.fmpz_mat([[int(x) for x in row] for row in M[r0:r1]])
        Eb = Pb - Ab * (two ** sh) if sh >= 0 else Pb * (two ** (-sh)) - Ab
        ents = Eb.entries()
        for i in range(r1 - r0):
            worst = max(worst, sum(abs(int(x)) for x in ents[i * n:(i + 1) * n]))
    den = Fraction(2) ** sh if sh >= 0 else Fraction(1)
    return neg, n - neg, Fraction(worst) / den


def inertia_exact_int(A):
    """Backwards-compatible form of inertia_exact: returns (n_negative, beta)."""
    neg, pos, beta = inertia_exact(A)
    return neg, beta


def _weighted_shift_matrix(ex: ExactMatrix, g: Fraction, n: int):
    """Integer symmetric M = b (W K den) - a den W for g = a/b; M = den D (b H - a) D."""
    a, b = g.numerator, g.denominator
    M = np.zeros((n, n), dtype=object); M[:] = 0
    w = ex.weights
    for r, c, v in zip(ex.rows.tolist(), ex.cols.tolist(), ex.re.tolist()):
        M[r, c] += b * w[r] * v
    for i in range(n):
        M[i, i] -= a * ex.den * w[i]
    return M


def _components(lam, radii):
    order = np.argsort(lam)
    comps = []
    for i in order:
        lo, hi = Fraction(float(lam[i])) - radii[i], Fraction(float(lam[i])) + radii[i]
        if comps and lo <= comps[-1][1]:
            comps[-1][1] = max(comps[-1][1], hi); comps[-1][2] += 1
        else:
            comps.append([lo, hi, 1])
    return comps


def _float_out(x: Fraction, up: bool) -> float:
    """A float on the safe side of the rational x (rounded to nearest, then one step outward)."""
    return float(np.nextafter(float(x), np.inf if up else -np.inf))


def _plan_points(comps, next_hint):
    """Points at which eigenvalues are counted, each with the number of claimed values below it.

    Every gap between components gets a point, and so does the region above the last one. A component
    holding m >= 2 claimed values (a multiplet, or overlapping intervals) is bracketed tightly on both
    sides, including below the first component: the residual intervals show only that each holds *an*
    eigenvalue, not m distinct ones, so the m eigenvalues are located by the counts at the brackets."""
    top = comps[-1][1]
    if next_hint is None or not np.isfinite(next_hint) or Fraction(float(next_hint)) <= top:
        next_hint = float(top) + max(1e-6, 1e-6 * abs(float(top)))
    hint = Fraction(float(next_hint))
    scale = max(Fraction(1), max(max(abs(c[0]), abs(c[1])) for c in comps))

    def tight(lo, hi, gap):
        return min(max(64 * (hi - lo), scale / 10 ** 11), gap / 4)

    pts, below = [], 0
    if comps[0][2] >= 2:
        lo, hi, _ = comps[0]
        pts.append((lo - max(64 * (hi - lo), scale / 10 ** 11), 0))
    for j, (lo, hi, m) in enumerate(comps):
        below += m
        if j + 1 < len(comps):
            nlo, nhi, nm = comps[j + 1]
            gap = nlo - hi
            if m >= 2:
                pts.append((hi + tight(lo, hi, gap), below))
            if nm >= 2:
                pts.append((nlo - tight(nlo, nhi, gap), below))
            if m < 2 and nm < 2:
                pts.append(((hi + nlo) / 2, below))
        else:
            if m >= 2:
                pts.append((hi + tight(lo, hi, hint - hi), below))
            pts.append(((hi + hint) / 2, below))
    return [(g.limit_denominator(1 << 40) if g.denominator > (1 << 40) else g, k) for g, k in pts]


def _decide(comps, counts):
    """Status and certified enclosures from the counts (each {'point', 'count', 'delta', 'claimed_below'}).

    With #(eig < g - delta) <= nu <= #(eig < g + delta) at every point, nu equal to the number of claimed
    values below the point, and no component meeting any window [g - delta, g + delta]:
      * a component with one claimed value holds lambda_K (K = claimed values up to it) inside the claimed
        interval: at least K-1 eigenvalues lie below the window under it, at most K below the window
        above it, and the residual interval holds one;
      * the sorted eigenvalues lambda_{K'+1..K} of a component with m >= 2 claimed values lie in
        [a - delta_a, b + delta_b], a and b the bracketing points below and above it;
      * lambda_{k+1} >= (top point) - delta."""
    ok = True
    for c in counts:
        if c.get("count") is None:
            ok = False
            continue
        g, d = c["point_exact"], c["delta_exact"]
        c["separated"] = bool(all(hi < g - d or lo > g + d for lo, hi, _ in comps))
        ok &= c["count"] == c["claimed_below"] and c["separated"]
    if not ok:
        return _status(counts, False, comps), None, None
    enclosures, below = [], 0
    for lo, hi, m in comps:
        if m == 1:
            elo, ehi, kind = lo, hi, "residual interval"
        else:
            a = max((c for c in counts if c["point_exact"] < lo), key=lambda c: c["point_exact"])
            b = min((c for c in counts if c["point_exact"] > hi), key=lambda c: c["point_exact"])
            elo, ehi = a["point_exact"] - a["delta_exact"], b["point_exact"] + b["delta_exact"]
            kind = "level enclosure from counts"
        enclosures.append({"eigenvalue_indices": [below + 1, below + m], "lower": _float_out(elo, False),
                           "upper": _float_out(ehi, True), "kind": kind})
        below += m
    top = max(counts, key=lambda c: c["point_exact"])
    return "CERTIFIED", enclosures, _float_out(top["point_exact"] - top["delta_exact"], False)


def certify_by_counts(comps, next_hint, count_at):
    """Plan the count points for the components, count, and decide (see _plan_points and _decide).

    count_at(g) returns (nu, delta) with #(eig < g - delta) <= nu <= #(eig < g + delta), or (None, None).
    If the only excess is above the top component, one more count just below that component tells a
    level missing under the top claim (FAILED) from an unclaimed eigenvalue at or above it (INCONCLUSIVE),
    as the floating-point S3 does. Returns (status, enclosures, next_lower_bound, counts)."""
    def record(g, below):
        nu, delta = count_at(g)
        if nu is None:
            return {"point": float(g), "count": None, "claimed_below": below}
        return {"point": float(g), "count": nu, "claimed_below": below, "delta": _float_out(delta, True),
                "point_exact": g, "delta_exact": Fraction(delta)}

    counts = [record(g, below) for g, below in _plan_points(comps, next_hint)]
    status, enclosures, next_lower = _decide(comps, counts)
    lo_top = comps[-1][0]
    top_excess = any(c.get("count") is not None and c["count"] > c["claimed_below"] and c["point_exact"] > lo_top
                     for c in counts)
    if status == "INCONCLUSIVE" and top_excess and comps[-1][2] == 1:
        prev_hi = comps[-2][1] if len(comps) > 1 else None
        scale = max(Fraction(1), abs(lo_top))
        margin = max(64 * (comps[-1][1] - lo_top), scale / 10 ** 11)
        if prev_hi is not None:
            margin = min(margin, (lo_top - prev_hi) / 4)
        g = lo_top - margin
        g = g.limit_denominator(1 << 40) if g.denominator > (1 << 40) else g
        w = record(g, sum(m for _, _, m in comps[:-1]))
        w["witness_below_top"] = True
        if w.get("count") is not None:
            gd, dd = w["point_exact"], w["delta_exact"]
            w["separated"] = bool(all(hi < gd - dd or lo > gd + dd for lo, hi, _ in comps))
            if w["separated"] and w["count"] > w["claimed_below"]:
                status = "FAILED"
        counts.append(w)
    return status, enclosures, next_lower, counts


def _status(counts, ok, comps=None):
    """FAILED only when an unclaimed eigenvalue lies below a claimed one: an excess count at a separated
    point that has a claimed component above it. An excess only above the top component may be the rest
    of a multiplet that the claim cuts ('the k lowest' is then ill-posed, not false): INCONCLUSIVE."""
    if ok:
        return "CERTIFIED"
    top = max(c[0] for c in comps) if comps else None
    for c in counts:
        if c.get("count") is not None and c["count"] > c["claimed_below"] and c.get("separated"):
            if top is None or c["point_exact"] < top:
                return "FAILED"
    return "INCONCLUSIVE"


def exact_recheck_lowest(spec, lam, X, states=None, next_hint=None, max_dim=4000):
    """Independent re-check, in exact arithmetic only, of a claim 'these are the k lowest eigenpairs
    of the declared model' (any local space; see exact_matrix).

    1. Exact residual radii r_i (Krylov-Bogolyubov, hypothesis x_i' != 0): an eigenvalue lies in
       I_i = [lam_i - r_i, lam_i + r_i]. A zero test vector makes the result INCONCLUSIVE.
    2. The intervals are merged into connected components. At a dyadic point g between consecutive
       components and at a point sigma above the last one (placed using `next_hint`, a floating-point
       estimate of the next eigenvalue; only a hint), the number of eigenvalues below is counted by the
       exact inertia of M = den D (b H - a) D (g = a/b), a congruence of b H - a with D >= I:
       #(eig < g - delta) <= nu <= #(eig < g + delta), delta = beta / (b den).
    3. CERTIFIED when no component meets any window [g - delta, g + delta] and every count equals the
       number of claimed values below its point. Then (see _decide) a component with one claimed value
       holds lambda_K within the claimed radius, and the eigenvalues of a component with m >= 2 claimed
       values lie in its 'level enclosure' between the bracketing points: per-vector residuals do not
       show m *distinct* eigenvalues, so the counts locate them. 'enclosures' lists both kinds; the
       next eigenvalue is >= 'next_lower_bound'. No floating-point assumption is used anywhere."""
    check_values(lam, "lam")          # as in certify() (precision.py)
    lam = np.atleast_1d(claimed_reals(lam))
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(X))):
        return {"status": "INCONCLUSIVE", "reason": "claim contains non-finite values"}
    ex = exact_matrix(spec, states)
    if np.any(ex.im != 0):
        return {"status": "NOT_CHECKED", "reason": "complex model: exact inertia implemented for real matrices"}
    n = X.shape[0]
    if n > max_dim:
        return {"status": "NOT_CHECKED", "reason": f"dimension {n} above max_dim {max_dim}"}
    rad = exact_eigenpair_radii(spec, lam, X, states, ex)
    if not rad["hermitian"]:
        return {"status": "FAILED", "reason": "declared model is not Hermitian", **rad}
    if rad["zero_test_vectors"]:
        return {"status": "INCONCLUSIVE", "reason": f"zero test vector for claimed pair(s) {rad['zero_test_vectors']}: "
                "the Krylov-Bogolyubov interval needs x != 0 (IMP-18)", **rad}
    radii = [Fraction(r) for r in rad["radii"]]
    comps = _components(lam, radii)
    def count_at(g):
        nu, _pos, beta = inertia_exact(_weighted_shift_matrix(ex, g, n))
        return (None, None) if nu is None else (nu, Fraction(beta) / (g.denominator * ex.den))
    status, enclosures, next_lower, counts = certify_by_counts(comps, next_hint, count_at)
    return {"status": status, "radii": rad["radii"], "enclosures": enclosures, "next_lower_bound": next_lower,
            "components": [[_float_out(a, False), _float_out(b, True), m] for a, b, m in comps],
            "counts": [{k: v for k, v in c.items() if not k.endswith("_exact")} for c in counts],
            "den": int(ex.den), "no_floating_point_assumption": True}
