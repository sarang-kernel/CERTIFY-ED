"""Certified Pomata-Wei spectral-gap criteria for 2D AKLT models (arXiv:1911.01410).

For each lattice the published proof needs eta < 1/z (square-octagon: 4 eta_perp + 2 eta_par < 1),
where eta was computed in floating point. `certify_lattice` replaces each eta by an upper bound proved
in exact arithmetic (eta.py), after a floating-point reproduction of the published value
(`eta_float`) that confirms the reconstructed geometry. `kernel.kernel_check` confirms the kernel
lemma (ker H_Gamma = VBS range) exactly for the subgraphs small enough.

Requires python-flint.
"""
from fractions import Fraction
from math import isqrt

import numpy as np

from . import lattices
from .eta import _sqrt_up, certify_eta
from .graph import reduced_region, union_coords

# name: (builder, symmetries builder or None, published eta, sharp threshold t (just above eta^2),
#        criterion z (eta < 1/z) or None for the square-octagon pair)
LATTICES = {
    "star": (lattices.star, None, 0.1110430220, Fraction(12331, 1000000), 4),
    "square_octagon_par": (lambda: lattices.square_octagon("par"), None, 0.1061446858, Fraction(11267, 1000000), None),
    "square_octagon_perp": (lambda: lattices.square_octagon("perp"), None, 0.1589663310, Fraction(25271, 1000000), None),
    "cross": (lattices.cross, None, 0.1997384500, Fraction(4987, 125000), 3),
    "decorated_honeycomb": (lattices.decorated_honeycomb, None, 0.1530329085, Fraction(2342, 100000), 4),
    "decorated_square": (lattices.decorated_square, lambda g: [lattices.decorated_square_reflection(g)],
                         0.2203543174, Fraction(4856, 100000), 4),
}
QUICK = ("star", "square_octagon_par", "square_octagon_perp", "decorated_honeycomb")


def eta_float(g, L, C, R):
    """Floating-point eta in reduced V coordinates (geometry check only; not a certificate)."""
    import scipy.linalg as la
    rL, rC, rR = (reduced_region(g, X) for X in (L, C, R))
    Ti, di, si, _ = union_coords(g, [L, C], [rL, rC])
    Tj, dj, sj, _ = union_coords(g, [C, R], [rC, rR])
    f = lambda G: np.array([[float(x) for x in row] for row in G])
    GL, GC, GR = f(rL["Gn"]), f(rC["Gn"]), f(rR["Gn"])
    GL /= np.max(np.abs(GL)); GC /= np.max(np.abs(GC)); GR /= np.max(np.abs(GR))
    nL, nC, nR = rL["n"], rC["n"], rR["n"]

    def dense(T, shape):
        M = np.zeros(shape + (len(T),))
        for k, col in enumerate(T):
            for key, v in col.items():
                M[key + (k,)] += v
        return M
    Ti_d, Tj_d = dense(Ti, (nL, nC)), dense(Tj, (nC, nR))
    Gi = np.einsum("lca,lm,cd,mdb->ab", Ti_d, GL, GC, Ti_d, optimize=True)
    Gj = np.einsum("cra,cd,rs,dsb->ab", Tj_d, GC, GR, Tj_d, optimize=True)
    Z = np.einsum("lm,mcb->lcb", GL, Ti_d, optimize=True)
    Y = np.einsum("cd,rs,dsb->crb", GC, GR, Tj_d, optimize=True)
    labA, labB = {}, {}
    for b in range(len(Ti)):
        for r in range(nR):
            labA.setdefault(int(si + di[b] + rR["downs"][r] + rR["inner"]), []).append((b, r))
    for l in range(nL):
        for d in range(len(Tj)):
            labB.setdefault(int(sj + rL["downs"][l] + dj[d] + rL["inner"]), []).append((l, d))
    allw = []
    for k in sorted(labA):
        ra, cb = labA[k], labB.get(k, [])
        if not cb:
            continue
        bs = np.array([x[0] for x in ra]); rs = np.array([x[1] for x in ra])
        ls = np.array([x[0] for x in cb]); ds = np.array([x[1] for x in cb])
        GA = Gi[np.ix_(bs, bs)] * GR[np.ix_(rs, rs)]
        GB = GL[np.ix_(ls, ls)] * Gj[np.ix_(ds, ds)]
        Cb = sum(Z[ls[None, :], c, bs[:, None]] * Y[c, rs[:, None], ds[None, :]] for c in range(nC))
        LB = la.cholesky(GB, lower=True)
        Xm = la.solve_triangular(LB, Cb.T, lower=True)
        allw.extend(la.eigh(Xm.T @ Xm, GA, eigvals_only=True).tolist())
    w = np.sort(np.array(allw))[::-1]
    n1 = int(np.sum(w > 1 - 1e-8))
    return {"unit_cosines": n1, "eta": float(np.sqrt(max(w[n1], 0.0))) if n1 < len(w) else 0.0}


MARGINS = (1e-9, 1e-7, 1e-5)      # relative distance of the threshold above the floating-point eta^2


def _threshold_above(eta, margin):
    """Rational t >= eta^2 (1 + margin), rounded up on a 1e-14 grid."""
    from math import ceil
    return Fraction(ceil(eta * eta * (1 + margin) * 10 ** 14), 10 ** 14)


def certify_lattice(name, thresholds=None, check_geometry=True, log=print):
    """Exact eta certificate for one lattice of LATTICES. Returns a JSON-ready record including the
    floating-point reproduction, the certified bound sqrt(t') at the best certified threshold, and
    whether the criterion eta < 1/z holds (decided exactly on t' < 1/z^2).

    Thresholds: given explicitly, or (default) placed a relative 1e-9 above the floating-point eta^2,
    widening to 1e-7 and 1e-5 only if the tighter one cannot be certified; without the geometry check
    the fixed threshold of LATTICES is used. The floating-point value only *places* the threshold;
    the certificate does not depend on it being accurate."""
    build, sym, paper, t_sharp, z = LATTICES[name]
    g, L, C, R = build()
    rec = {"lattice": name, "eta_published": paper}
    if check_geometry:
        ef = eta_float(g, L, C, R)
        rec["eta_float"] = ef["eta"]
        rec["unit_cosines_float"] = ef["unit_cosines"]
        log(f"  {name}: floating-point eta {ef['eta']:.10f} (published {paper:.10f})")
    if thresholds is not None:
        ladder = [list(thresholds)]
    elif check_geometry:
        ladder = [[_threshold_above(rec["eta_float"], m)] for m in MARGINS]
    else:
        ladder = [[t_sharp]]
    out, tried = None, []
    for ths in ladder:
        out = certify_eta(g, L, C, R, ths, sigmas=sym(g) if sym else (), log=log)
        tried += [str(t) for t in ths]
        if any(v["certified"] for v in out["thresholds"].values()):
            break
        log(f"  {name}: not certified at {[str(t) for t in ths]}; widening the threshold")
    rec.update(out)
    rec["thresholds_tried"] = tried
    best = None
    for _key, v in out["thresholds"].items():
        if v["certified"]:
            num, den = (int(x) for x in v["t_prime_exact"].split("/"))
            tp = Fraction(num, den)
            if best is None or tp < best:
                best = tp
    rec["eta_upper"] = None if best is None else _sqrt_up(best)        # a float >= sqrt(t')
    rec["eta_upper_squared_exact"] = None if best is None else f"{best.numerator}/{best.denominator}"
    rec["criterion_z"] = z
    rec["criterion_holds"] = None if (z is None or best is None) else bool(best < Fraction(1, z * z))
    return rec


def square_octagon_criterion(rec_par, rec_perp):
    """4 eta_perp + 2 eta_par < 1, decided with rational upper bounds on the square roots."""
    def sqrt_up(s):
        num, den = (int(x) for x in s.split("/"))
        # sqrt(num/den) <= (isqrt(num * den) + 1) / den
        return Fraction(isqrt(num * den) + 1, den)
    ep, epar = sqrt_up(rec_perp["eta_upper_squared_exact"]), sqrt_up(rec_par["eta_upper_squared_exact"])
    total = 4 * ep + 2 * epar
    return {"sum_upper": float(np.nextafter(float(total), np.inf)), "criterion_holds": bool(total < 1)}
