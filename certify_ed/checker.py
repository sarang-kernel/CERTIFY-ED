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


# --------------------------------------------------------------------------- exact S2 re-check
def _exact_rational(v, max_den=1 << 12):
    """Exact Fraction of a binary64 value if it is a rational with a small denominator, else None
    (irrational local matrix elements such as sqrt(2) are rejected)."""
    from fractions import Fraction
    f = Fraction(float(v))
    return f if f.denominator <= max_den else None


def exact_matrix(spec, states=None):
    """The declared model as an exact matrix: H = (Re + i Im) / den with integer entries.

    Couplings are the binary64 numbers in the spec, taken exactly. Local matrix elements must be
    rationals (true for every q = 2 space: spin-1/2 in all conventions, fermions; and for boson
    occupation operators n). Spaces with irrational elements (spin >= 1 raising operators,
    boson b^dagger) raise ValueError. Returns (rows, cols, re, im, den) with Python-int object arrays,
    duplicates summed, entries sorted by (row, col)."""
    import itertools
    from fractions import Fraction
    from math import lcm
    from .construct import full_state_set, _popcount
    st = states if states is not None else full_state_set(spec)
    s0 = st.states_in_order()
    n = st.size
    N, q, local = spec.n_sites, spec.q, spec.local
    col_idx = np.arange(n)
    groups = []                     # (rows, cols, re_int, im_int, Fraction scale)
    for term in spec.terms:
        comps = [local.shift_components(letter) for letter in term.op]
        # exact components: per component, (delta, num_re, num_im, den)
        ex = []
        for comp in comps:
            row = []
            for delta, ampvec in comp:
                fr = [_exact_rational(np.real(a)) for a in ampvec]
                fi = [_exact_rational(np.imag(a)) for a in ampvec]
                if any(x is None for x in fr + fi):
                    raise ValueError(f"local operator {term.op!r} has irrational matrix elements")
                d = lcm(*[x.denominator for x in fr + fi])
                row.append((delta, np.array([int(x * d) for x in fr], dtype=object),
                            np.array([int(x * d) for x in fi], dtype=object), d))
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
                        if local.jw_string == "higher":
                            side = s & ((1 << (N - 1 - site)) - 1)
                        else:
                            side = s >> (N - site)
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
                # coefficient (cr + i ci) / dprod
                re = are[m][found]; im = aim[m][found]
                groups.append((r[found], col_idx[m][found], re, im, cr / dprod, ci / dprod))
    if not groups:
        z = np.zeros(0, dtype=object)
        return np.zeros(0, np.int64), np.zeros(0, np.int64), z, z, 1
    den = 1
    for g in groups:
        den = lcm(den, g[4].denominator, g[5].denominator)
    R, C, RE, IM = [], [], [], []
    for r, c, re, im, kr, ki in groups:
        ar, ai = int(kr * den), int(ki * den)       # (ar + i ai) (re + i im)
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
    return R[keep], C[keep], RE[keep], IM[keep], den


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
        o = np.array([int(a) << int(b - E) if a != 0 else 0 for a, b in zip(mm.tolist(), e.tolist())], dtype=object)
        outs.append(o)
    return outs[0], outs[1], E


def exact_eigenpair_radii(spec, lam, X, states=None, exact=None):
    """Residual radii of claimed eigenpairs against the declared model, in exact arithmetic.

    For each claimed (lam_i, x_i): rho_i = ||H x_i - lam_i x_i|| / ||x_i|| is computed exactly and
    rounded up to a float; for Hermitian H there is an eigenvalue in [lam_i - rho_i, lam_i + rho_i]
    (Krylov-Bogolyubov). The Hermiticity of the exact matrix is checked exactly. No floating-point
    assumption is involved. Returns dict(radii, hermitian, den, nnz)."""
    from fractions import Fraction
    from math import isqrt
    R, C, RE, IM, den = exact if exact is not None else exact_matrix(spec, states)
    n = X.shape[0]
    # exact Hermiticity: entry (r, c) equals conj of entry (c, r)
    fwd = {(int(r), int(c)): (a, b) for r, c, a, b in zip(R, C, RE, IM)}
    herm = all(fwd.get((c, r), (0, 0)) == (a, -b) for (r, c), (a, b) in fwd.items())
    radii = []
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    for i, lam_i in enumerate(np.atleast_1d(lam)):
        xr, xi, E = _float_vector_to_int(X[:, i])
        # y = den * 2^-E * H x  (Gaussian integers)
        yr = np.zeros(n, dtype=object); yi = np.zeros(n, dtype=object)
        pr = RE * xr[C] - IM * xi[C]
        pi = RE * xi[C] + IM * xr[C]
        np.add.at(yr, R, pr); np.add.at(yi, R, pi)
        L = Fraction(float(lam_i))
        a, b = L.numerator, L.denominator
        # b den 2^-E r = b y - a den x
        rr = yr * b - xr * (a * den)
        ri = yi * b - xi * (a * den)
        num2 = int(np.sum(rr * rr + ri * ri))            # (b den)^2 2^-2E ||r||^2
        x2 = int(np.sum(xr * xr + xi * xi))              # 2^-2E ||x||^2
        # rho^2 = num2 / ((b den)^2 x2); upper bound via integer square roots
        if num2 == 0:
            radii.append(0.0)
            continue
        D2 = (b * den) ** 2 * x2
        # rho <= ceil(sqrt(num2 * 2^(2k) / D2)) / 2^k
        k = 80
        t = (num2 << (2 * k)) // D2 + 1
        s = isqrt(t) + 1
        q = Fraction(s, 1 << k)
        f = float(q)
        if Fraction(f) < q:
            f = float(np.nextafter(f, np.inf))
        radii.append(f)
    return {"radii": radii, "hermitian": bool(herm), "den": int(den), "nnz": int(len(R))}


def inertia_exact_int(A):
    """A: flint.fmpz_mat (symmetric, integer). Floating-point LDL^T of A, residual re-checked exactly.
    Returns (n_negative_pivots, bound on ||Lt Dt Lt^T - A||_inf as a Fraction).
    Consequence (Weyl): #(eig A < -beta) <= n_negative <= #(eig A < beta)."""
    import flint
    import scipy.linalg as la
    from fractions import Fraction
    n = A.nrows()
    ent = [int(x) for x in A.entries()]
    maxbits = max([abs(x).bit_length() for x in ent] + [1])
    s_down = max(0, maxbits - 900)
    Af = np.array([float(x >> s_down) if x >= 0 else -float((-x) >> s_down) for x in ent]).reshape(n, n)
    del ent
    lu, d, perm = la.ldl(Af, lower=True, hermitian=True)
    if not np.all(np.diag(lu[perm]) == 1.0):
        raise ValueError("LDL factor not unit triangular after permutation")
    pL = 52
    dmax = float(np.max(np.abs(d))) or 1.0
    pD = int(52 - np.ceil(np.log2(dmax)))
    Li = np.rint(np.ldexp(lu, pL)).astype(np.int64)
    Di = np.rint(np.ldexp(d, pD)).astype(np.int64)
    neg, amb = negative_count_exact(np.ldexp(Di.astype(np.float64), -pD))
    if amb:
        return None, None
    sh = 2 * pL + pD - s_down
    L = flint.fmpz_mat(Li.tolist()); Dm = flint.fmpz_mat(Di.tolist())
    P = (L * Dm) * L.transpose()
    if sh >= 0:
        E = P - A * (flint.fmpz(2) ** sh); den = Fraction(2) ** sh
    else:
        E = P * (flint.fmpz(2) ** (-sh)) - A; den = Fraction(1)
    ents = E.entries()
    worst = max(sum(abs(int(x)) for x in ents[i * n:(i + 1) * n]) for i in range(n))
    return neg, Fraction(worst) / den


def exact_recheck_lowest(spec, lam, X, states=None, next_hint=None, max_dim=4000):
    """Independent re-check, in exact arithmetic only, of a claim 'these are the k lowest eigenpairs
    of the declared model' (q = 2 spaces; see exact_matrix).

    1. Exact residual radii r_i (Krylov-Bogolyubov): an eigenvalue lies in I_i = [lam_i - r_i, lam_i + r_i].
    2. The intervals are merged into connected components. At a dyadic point g between consecutive
       components and at a point sigma above the last one (placed using `next_hint`, a floating-point
       estimate of the next eigenvalue; only a hint), the number of eigenvalues below is counted by the
       exact inertia of b den H - a den I (g = a/b), with residual bound beta:
       #(eig < g - delta) <= nu <= #(eig < g + delta), delta = beta / (b den).
    3. CERTIFIED when every component lies strictly between its separation points widened by delta and
       every count equals the number of claimed values below that point. Then component j contains
       exactly m_j eigenvalues, so the claim lists the k lowest eigenvalues with multiplicity, each within
       its radius. No floating-point assumption is used anywhere."""
    import flint
    from fractions import Fraction
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    ex = exact_matrix(spec, states)
    R, C, RE, IM, den = ex
    if np.any(IM != 0):
        return {"status": "NOT_CHECKED", "reason": "complex model: exact inertia implemented for real matrices"}
    n = X.shape[0]
    if n > max_dim:
        return {"status": "NOT_CHECKED", "reason": f"dimension {n} above max_dim {max_dim}"}
    rad = exact_eigenpair_radii(spec, lam, X, states, ex)
    if not rad["hermitian"]:
        return {"status": "FAILED", "reason": "declared model is not Hermitian", **rad}
    r = np.array(rad["radii"])
    order = np.argsort(lam)
    lo = [Fraction(float(lam[i])) - Fraction(float(r[i])) for i in order]
    hi = [Fraction(float(lam[i])) + Fraction(float(r[i])) for i in order]
    comps = []                                      # [lo, hi, count]
    for a, b in zip(lo, hi):
        if comps and a <= comps[-1][1]:
            comps[-1][1] = max(comps[-1][1], b); comps[-1][2] += 1
        else:
            comps.append([a, b, 1])
    top = comps[-1][1]
    if next_hint is None or not np.isfinite(next_hint) or Fraction(float(next_hint)) <= top:
        next_hint = float(top) + max(1e-6, 1e-6 * abs(float(top)))
    pts = [(comps[j][1] + comps[j + 1][0]) / 2 for j in range(len(comps) - 1)]
    pts.append((top + Fraction(float(next_hint))) / 2)
    Hint = flint.fmpz_mat(n, n)
    for i, j, v in zip(R.tolist(), C.tolist(), RE.tolist()):
        Hint[i, j] = v
    counts, ok, cum = [], True, 0
    for j, g in enumerate(pts):
        g = Fraction(g).limit_denominator(1 << 60) if Fraction(g).denominator > (1 << 60) else Fraction(g)
        a, b = g.numerator, g.denominator
        A = Hint * b - flint.fmpz_mat(n, n, [a * den if i == k else 0 for i in range(n) for k in range(n)])
        nu, beta = inertia_exact_int(A)
        cum += comps[j][2]
        if nu is None:
            counts.append({"point": float(g), "count": None}); ok = False; continue
        delta = beta / (b * den)
        sep_ok = comps[j][1] < g - delta and (j + 1 >= len(comps) or comps[j + 1][0] > g + delta)
        counts.append({"point": float(g), "count": nu, "claimed_below": cum, "delta": float(delta),
                       "separated": bool(sep_ok)})
        ok &= (nu == cum) and sep_ok
    status = "CERTIFIED" if ok else ("FAILED" if any(c.get("count") is not None and c["count"] > c["claimed_below"]
                                                     and c["separated"] for c in counts) else "INCONCLUSIVE")
    return {"status": status, "radii": rad["radii"], "components": [[float(a), float(b), m] for a, b, m in comps],
            "counts": counts, "den": int(den), "no_floating_point_assumption": True}
