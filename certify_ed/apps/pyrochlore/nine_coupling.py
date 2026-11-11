"""Nine-coupling spin-1/2 crown of Kranitz & Penc (arXiv:2601.07800, Appendix D): roadmap item N-11.

The paper's headline spin-1/2 bound e_GS >= -0.549832(8) (abstract, Table I) is lambda_min of the general crown
Hamiltonian

    H_0 = sum_{m=1..19} c_m sum_{(i,j) in o_m} S_i . S_j

over the 19 D3d orbits o_m of site pairs (the paper's Table V, ORBITS below), with the embedding constraints of the
nearest-neighbour model (J = 1, J2 = J3a = J3b = 0)

    J = 2 c1 + 4 c3 + 4 c12 + 2 c17,   J2 = 2 c2 + 2 c5 + c11 + c18,   J3a = 2 c4 + c19,   J3b = 2 c6 + 4 c13,
    0 = c7 + c15,   0 = c8 = c9 = c10 = c14 = c16,

and the nine free couplings printed in its Eq. (D4) (PRINTED). Unlike the reduced model of model.py (c3 = c12,
c5 = 0), the swaps of the two outer sites of a tetrahedron are not symmetries. The orbit-preserving site
permutations form the point group D3d of the cluster (order 12, automorphisms()); the exact layer needs
one-dimensional characters with values +-1, +-i, and uses the abelian subgroup {E, C2', i, sigma_d} together with
the global spin flip of the S^z = 0 space: 8 blocks.

The printed couplings carry one digit in parentheses, e.g. c1 = 0.037416(7). For the reduced model the paper
prints c1 = 0.038141(4) and c3 = 0.081795(7), which the project and the review read as 0.0381414 and 0.0817957
(n1/n3 = 381414/817957), and the printed bounds -0.550158(2) and -0.549832(8) agree with the computed values
-0.55015829... and -0.54983289... when the digit in parentheses is read as one more decimal. The default reading
here is therefore "appended" (c1 = 0.0374167); "truncated" drops the digit. Whatever the reading, H_0 at couplings
that satisfy the constraints exactly gives a valid Anderson-type bound e_GS >= lambda_min(H_0) by the paper's
argument (which is taken from the paper, not re-derived).

Rigour chain (as spin_half.certify_crown):
  1. Exact model: the printed decimals as exact rationals, the derived couplings c13, c15, c17, c18, c19 from the
     constraints in exact arithmetic (check_embedding), the spec of K H_0 with integer couplings.
  2. SU(2): 18 spins 1/2 have integer total spin, so every multiplet has a member with S^z = 0 and
     lambda_min(H_0) = lambda_min(H_0 on S^z = 0).
  3. Z2^3 (a C2' rotation, the inversion, the spin flip): orbit-sum bases with entries +-1, completeness checked
     by dimension, every generator checked exactly to commute with the integer matrix (exact_sectors).
  4. Exact inertia of each block at a dyadic shift g (checker.inertia_exact): if no block has an eigenvalue of
     K H_0 below g - delta, then lambda_min(K H_0) >= g - delta. The blocks are formed and counted one at a
     time, so memory is that of one block.
  5. Upper side: the Rayleigh quotient of a dyadic rounding of the Lanczos vector, evaluated in integers, is an
     upper bound on lambda_min (variational principle).
No floating-point assumption: the Lanczos estimate only places g and supplies the trial vector. Like the
reduced-model certificate, the lower bound rests on Lemmas 11 and 12 and Proposition 13 of the exact layer,
which are marked UNVERIFIED in docs/theory.md.
"""
import itertools
import math
import re
import time
from fractions import Fraction
from math import lcm

import numpy as np

from ...localops import LocalSpace
from ...spec import ModelSpec

N = 18

# Table V of Kranitz & Penc: the 19 D3d orbits of the 153 site pairs, sites 1-based as in the paper.
ORBITS = {
    1: ((1, 2), (3, 10), (6, 12), (7, 13), (9, 16), (17, 18)),
    2: ((1, 3), (1, 6), (2, 10), (2, 12), (3, 7), (6, 9), (7, 17), (9, 17), (10, 13), (12, 16), (13, 18),
         (16, 18)),
    3: ((1, 4), (1, 5), (4, 10), (5, 12), (7, 8), (7, 14), (8, 10), (9, 11), (9, 15), (11, 12), (14, 18),
         (15, 18)),
    4: ((1, 7), (1, 9), (7, 9), (10, 12), (10, 18), (12, 18)),
    5: ((1, 8), (1, 11), (4, 7), (4, 12), (5, 9), (5, 10), (7, 15), (8, 18), (9, 14), (10, 14), (11, 18),
         (12, 15)),
    6: ((1, 10), (1, 12), (7, 10), (7, 18), (9, 12), (9, 18)),
    7: ((1, 13), (1, 16), (2, 7), (2, 9), (3, 12), (3, 18), (6, 10), (6, 18), (7, 16), (9, 13), (10, 17),
         (12, 17)),
    8: ((1, 14), (1, 15), (4, 9), (4, 18), (5, 7), (5, 18), (7, 11), (8, 9), (8, 12), (10, 11), (10, 15),
         (12, 14)),
    9: ((1, 17), (2, 18), (3, 9), (6, 7), (10, 16), (12, 13)),
    10: ((1, 18), (7, 12), (9, 10)),
    11: ((2, 3), (2, 6), (3, 13), (6, 16), (13, 17), (16, 17)),
    12: ((2, 4), (2, 5), (3, 4), (3, 8), (5, 6), (6, 11), (8, 13), (11, 16), (13, 14), (14, 17), (15, 16),
         (15, 17)),
    13: ((2, 8), (2, 11), (3, 5), (3, 14), (4, 6), (4, 13), (5, 16), (6, 15), (8, 17), (11, 17), (13, 15),
         (14, 16)),
    14: ((2, 13), (2, 16), (3, 6), (3, 17), (6, 17), (13, 16)),
    15: ((2, 14), (2, 15), (3, 11), (3, 15), (4, 16), (4, 17), (5, 13), (5, 17), (6, 8), (6, 14), (8, 16),
         (11, 13)),
    16: ((2, 17), (3, 16), (6, 13)),
    17: ((4, 5), (4, 8), (5, 11), (8, 14), (11, 15), (14, 15)),
    18: ((4, 11), (4, 14), (5, 8), (5, 15), (8, 15), (11, 14)),
    19: ((4, 15), (5, 14), (8, 11)),
}
FREE = (1, 2, 3, 4, 5, 6, 7, 11, 12)                 # the nine free couplings of Eq. (D4)
ZERO = (8, 9, 10, 14, 16)                            # set to zero by the embedding
PRINTED = {1: "0.037416(7)", 2: "0.001151(3)", 3: "0.083729(9)", 4: "0.0000415(5)", 5: "-0.000546(9)",
           6: "-0.000210(9)", 7: "-0.000539(3)", 11: "0.001764(2)", 12: "0.080243(4)"}
PUBLISHED_BOUND = "-0.549832(8)"                     # e_LB of the nine-coupling crown, S = 1/2
REVIEW_FLOAT = -0.54983289                           # review/REVIEW_B_applications.md, finding B2-1 (floating point)


def printed_value(text, reading="appended"):
    """Exact rational of a printed value 'x.yyy(d)': 'appended' reads d as one more decimal, 'truncated' drops it."""
    m = re.fullmatch(r"(-?)(\d+)\.(\d+)\((\d)\)", text)
    if m is None:
        raise ValueError(f"not a printed value: {text!r}")
    sign, ip, fp, d = m.groups()
    if reading == "appended":
        fp = fp + d
    elif reading != "truncated":
        raise ValueError(f"unknown reading {reading!r}")
    v = Fraction(int(ip + fp), 10 ** len(fp))
    return -v if sign else v


def free_couplings(reading="appended"):
    return {k: printed_value(PRINTED[k], reading) for k in FREE}


def check_embedding(c, J=1, J2=0, J3a=0, J3b=0):
    """Raise ValueError unless the 19 couplings satisfy the matching conditions of Appendix D exactly."""
    conds = {"J": (2 * c[1] + 4 * c[3] + 4 * c[12] + 2 * c[17], J), "J2": (2 * c[2] + 2 * c[5] + c[11] + c[18], J2),
             "J3a": (2 * c[4] + c[19], J3a), "J3b": (2 * c[6] + 4 * c[13], J3b), "c7+c15": (c[7] + c[15], 0)}
    conds.update({f"c{k}": (c[k], 0) for k in ZERO})
    bad = {k: (str(a), str(b)) for k, (a, b) in conds.items() if Fraction(a) != Fraction(b)}
    if bad:
        raise ValueError(f"embedding constraints violated: {bad}")


def couplings(free=None, reading="appended", J=1, J2=0, J3a=0, J3b=0):
    """All 19 couplings (Fractions) from the nine free ones; the others follow from the embedding constraints."""
    f = free_couplings(reading) if free is None else {k: Fraction(v) for k, v in free.items()}
    c = {m: Fraction(0) for m in range(1, 20)}
    for k in FREE:
        c[k] = f[k]
    J, J2, J3a, J3b = (Fraction(x) for x in (J, J2, J3a, J3b))
    c[13] = (J3b - 2 * c[6]) / 4
    c[15] = -c[7]
    c[17] = (J - 2 * c[1] - 4 * c[3] - 4 * c[12]) / 2
    c[18] = J2 - 2 * c[2] - 2 * c[5] - c[11]
    c[19] = J3a - 2 * c[4]
    check_embedding(c, J, J2, J3a, J3b)
    return c


def check_orbit_table():
    """The table partitions the 153 pairs; its nearest-neighbour orbits o1, o3, o12, o17 are the 36 bonds of
    model.bonds with the same coupling names (c1, c3, c12, c17). Returns the orbit sizes."""
    from .model import bonds as nn_bonds
    pairs = [tuple(sorted(p)) for m in ORBITS for p in ORBITS[m]]
    if len(pairs) != 153 or set(pairs) != set(itertools.combinations(range(1, N + 1), 2)):
        raise ValueError("Table V does not partition the 153 site pairs")
    owner = {tuple(sorted(p)): m for m in ORBITS for p in ORBITS[m]}
    for name, i, j in nn_bonds("c1", "c3", "c12", "c17"):
        if owner[tuple(sorted((i, j)))] != int(name[1:]):
            raise ValueError(f"bond ({i},{j}) of model.py is in orbit {owner[tuple(sorted((i, j)))]}, not {name}")
    return {m: len(v) for m, v in ORBITS.items()}


def bonds(c):
    """(coupling, i, j) for every pair with a nonzero coupling, sites 1-based."""
    return [(c[m], i, j) for m in sorted(ORBITS) if c[m] != 0 for i, j in ORBITS[m]]


def scale_for(c):
    """K = 4 lcm(denominators): K c_m and K c_m / 2 are integers, so the spec couplings are exact in binary64."""
    return 4 * lcm(*[Fraction(x).denominator for x in c.values()])


def spec_for(c, scale=1, name="pyrochlore_crown_nine"):
    """ModelSpec of scale * H_0 for spin 1/2 (S.S = SzSz + (S+S- + S-S+)/2). With scale = scale_for(c) every
    coupling of the spec is an integer and is checked to be exact in binary64."""
    bl = bonds(c)
    zz = [[float(scale * x), i - 1, j - 1] for x, i, j in bl]
    pm = [[float(scale * x / 2), i - 1, j - 1] for x, i, j in bl]
    if scale != 1:
        for (x, _, _), a, b in zip(bl, zz, pm):
            if Fraction(a[0]) != scale * x or Fraction(b[0]) != scale * x / 2:
                raise ValueError("a scaled coupling is not exact in binary64")
    params = {"scale": int(scale), **{f"c{m}": str(c[m]) for m in sorted(c)}}
    spec = ModelSpec.from_static(N, LocalSpace.spin_space("1/2", "spin"), [["zz", zz], ["+-", pm], ["-+", pm]],
                                 name=name, parameters=params)
    return spec


def automorphisms():
    """All site permutations (1-based tuples, p[i-1] = image of site i) that map every orbit of Table V onto
    itself, i.e. the symmetries of H_0 for every choice of couplings. Backtracking search."""
    col = [[0] * (N + 1) for _ in range(N + 1)]
    for m, v in ORBITS.items():
        for a, b in v:
            col[a][b] = col[b][a] = m
    found = []

    def extend(p, used):
        i = len(p) + 1
        if i > N:
            found.append(tuple(p))
            return
        for x in range(1, N + 1):
            if x not in used and all(col[i][j] == col[x][p[j - 1]] for j in range(1, i)):
                p.append(x); used.add(x)
                extend(p, used)
                p.pop(); used.discard(x)
    extend([], set())
    return found


def _order(p):
    e, q, k = tuple(range(1, N + 1)), p, 1
    while q != e:
        q = tuple(p[q[i] - 1] for i in range(N)); k += 1
    return k


def c2h_generators():
    """(C2', inversion) as 1-based permutations: the inversion is the fixed-point-free involution (central in
    D3d), C2' the lexicographically smallest involution with 4 fixed points. They commute."""
    G = automorphisms()
    inv = [p for p in G if _order(p) == 2]
    inversion = [p for p in inv if all(p[i] != i + 1 for i in range(N))]
    rot = sorted(p for p in inv if sum(p[i] == i + 1 for i in range(N)) == 4)
    if len(G) != 12 or len(inversion) != 1 or not rot:
        raise ValueError(f"unexpected symmetry group of order {len(G)}")
    a, b = rot[0], inversion[0]
    if tuple(a[b[i] - 1] for i in range(N)) != tuple(b[a[i] - 1] for i in range(N)):
        raise ValueError("C2' and the inversion do not commute")
    return a, b


def symmetry_generators(spec):
    """C2', inversion and global spin flip as CyclicSymmetry objects (each checked exactly against the model by
    exact_sectors.check_commutes before it is used)."""
    from ...basis import spin_flip
    from ...exact_sectors import site_permutation
    a, b = c2h_generators()
    return [site_permutation(spec, [x - 1 for x in a], "C2prime"), site_permutation(spec, [x - 1 for x in b], "inversion"),
            spin_flip(spec)]


def estimate(c, seed=12345, k=1, tol=1e-12):
    """Lanczos estimate of the lowest eigenvalue(s) on S^z = 0 (floating point; places the shift only)."""
    import scipy.sparse.linalg as spla
    from ...basis import magnetization_state_set
    from ...construct import build_bitwise
    spec = spec_for(c)
    st = magnetization_state_set(spec, 0)
    H = build_bitwise(spec, st)
    v0 = np.random.default_rng(seed).standard_normal(H.shape[0])
    w, V = spla.eigsh(H, k=k, which="SA", tol=tol, v0=v0)
    order = np.argsort(w)
    return w[order], V[:, order], st


def count_below_by_block(spec, states, gens, g, log=None):
    """Exact count of eigenvalues of the spec's matrix below g on the states, block by block.

    The same construction as exact_sectors.exact_count_below (group closure, exact commutation check,
    orbit-sum bases of the 1D characters, completeness by dimension, integer congruence blocks, exact inertia),
    with each block formed, counted and released before the next, so that memory is that of one block. For
    characters with values +-1 and phases +-1 the blocks are real; the count is then that of
    exact_sectors._count_sector_below with factor 1. Returns {count, delta, sectors}: with delta the largest
    block delta, #(eig < g - delta) <= count <= #(eig < g + delta) on the states."""
    from ...checker import exact_matrix, inertia_exact
    from ...exact_sectors import FiniteGroup, _unit, check_commutes, orbit_columns, sector_integer_matrix
    g = Fraction(g)
    ex = exact_matrix(spec, states)
    group = FiniteGroup(gens, states)
    okc, bad = check_commutes(ex, group)
    if not okc:
        raise ValueError(f"{bad} is not a symmetry of the declared model")
    chars = []
    for js in itertools.product(*[range(x.order) for x in gens]):
        vals = [_unit(x.order, j) for x, j in zip(gens, js)]
        if any(v is None for v in vals):
            continue
        chi = group.character(vals)
        if chi is None:
            continue
        if any(ci != 0 for _, ci in chi):
            raise ValueError("count_below_by_block handles real characters only")
        cols = orbit_columns(group, chi)
        if cols:
            chars.append((dict(zip([x.name for x in gens], js)), chi, len(cols)))
    if sum(d for _, _, d in chars) != states.size:
        raise ValueError(f"1D irreps cover {sum(d for _, _, d in chars)} of {states.size} dimensions")
    a, b = g.numerator, g.denominator
    total, dmax, rows = 0, Fraction(0), []
    for qn, chi, dim in chars:
        t0 = time.time()
        cols = orbit_columns(group, chi)
        Are, Aim, D = sector_integer_matrix(ex, cols)
        del cols
        if any(int(x) != 0 for x in Aim.ravel()):
            raise ValueError("imaginary part in a real block")
        del Aim
        M = np.multiply(Are, b, out=Are)          # in place: one n x n object array per block
        del Are
        for i in range(dim):
            M[i, i] = M[i, i] - a * ex.den * D[i]
        nu, _, beta = inertia_exact(M)
        del M
        if nu is None:
            return {"count": None, "delta": None, "sectors": rows, "group_order": group.order, "den": int(ex.den)}
        delta = beta / (b * ex.den)
        total += nu
        dmax = max(dmax, delta)
        rows.append({"quantum_numbers": qn, "dim": dim, "count": int(nu), "delta": float(delta),
                     "seconds": time.time() - t0})
        if log:
            log(f"    block {qn} (dim {dim}): {nu} below {float(g):.12g}, delta {float(delta):.2e}, "
                f"{time.time() - t0:.0f}s")
    return {"count": total, "delta": dmax, "sectors": rows, "group_order": group.order, "den": int(ex.den)}


def rayleigh_upper(spec, states, v, bits=30):
    """Exact Rayleigh quotient x^T A x / x^T x of x = rint(v 2^bits / max|v|) for the spec's matrix A on the
    states (integer arithmetic). It is >= lambda_min(A) for any nonzero x."""
    from ...checker import exact_matrix
    ex = exact_matrix(spec, states)
    if any(int(w) != 1 for w in ex.weights):
        raise ValueError("rayleigh_upper expects unit weights (spin 1/2)")
    if any(int(x) != 0 for x in ex.im):
        raise ValueError("rayleigh_upper expects a real matrix")
    x = np.rint(np.ldexp(np.asarray(v, float) / np.max(np.abs(v)), bits)).astype(np.int64)
    xo = x.astype(object)
    num = int(np.sum(xo[ex.rows] * ex.re * xo[ex.cols]))
    den = int(np.dot(xo, xo)) * int(ex.den)
    if den == 0:
        raise ValueError("zero trial vector")
    return Fraction(num, den)


def certify_nine(reading="appended", free=None, margins_rel=(1e-10, 1e-9, 1e-8), log=print):
    """Certified enclosure lower <= lambda_min(H_0) <= upper for the nine-coupling crown at the printed couplings
    (or at the given free couplings), in exact arithmetic. Record fields follow spin_half.certify_crown."""
    t0 = time.time()
    c = couplings(free=free, reading=reading)
    K = scale_for(c)
    w, V, st = estimate(c)
    e0, v = float(w[0]), V[:, 0]
    spec_K = spec_for(c, scale=K)
    gens = symmetry_generators(spec_K)
    t1 = time.time()
    upper = rayleigh_upper(spec_K, st, v) / K
    t_upper = time.time() - t1
    grid = 2 ** 20
    tried, r, g = [], None, None
    for m in margins_rel:
        g = Fraction(int(np.floor((e0 - m * abs(e0)) * K * grid)), grid)           # in K H_0 units
        r = count_below_by_block(spec_K, st, gens, g, log=log)
        tried.append({"margin_rel": m, "shift": str(g / K), "count": r["count"]})
        if r["count"] == 0:
            break
        log(f"  nine-coupling crown: count {r['count']} below the shift at margin {m}; widening")
    if r["count"] != 0:
        raise RuntimeError(f"exact count below the shift is {r['count']} at every margin, expected 0")
    lb = (g - r["delta"]) / K
    if not lb <= upper:
        raise RuntimeError("lower bound above the Rayleigh quotient")
    rec = {"reading": reading, "couplings": {f"c{m}": str(c[m]) for m in sorted(c)},
           "printed": {f"c{k}": PRINTED[k] for k in FREE} if free is None else None, "K": int(K),
           "dim_Sz0": int(st.size), "group_order": r["group_order"], "n_sectors": len(r["sectors"]),
           "sectors": r["sectors"], "largest_sector": max(s["dim"] for s in r["sectors"]), "e0_estimate": e0,
           "shift": str(g / K), "shifts_tried": tried, "delta": float(r["delta"] / K),
           "lower_bound": float(np.nextafter(float(lb), -np.inf)),
           "lower_bound_rounded_down": float(Fraction(math.floor(lb * 10 ** 9), 10 ** 9)),
           "lower_bound_exact": f"{lb.numerator}/{lb.denominator}",
           "upper_bound": float(np.nextafter(float(upper), np.inf)),
           "upper_bound_rounded_up": float(Fraction(math.ceil(upper * 10 ** 9), 10 ** 9)),
           "upper_bound_exact": f"{upper.numerator}/{upper.denominator}", "upper_seconds": t_upper,
           "width": float(upper - lb), "no_floating_point_assumption": True, "seconds": time.time() - t0}
    log(f"  nine-coupling crown ({reading}): e0 estimate {e0:.12f}; certified {float(lb):.12f} <= lambda_min <= "
        f"{float(upper):.12f} ({rec['n_sectors']} blocks, largest {rec['largest_sector']}, {rec['seconds']:.0f}s)")
    return rec
