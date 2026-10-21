"""S = 1/2 crown: certified lower bound on lambda_min(H_crown), hence on e_GS of the pyrochlore
Heisenberg antiferromagnet (Kranitz & Penc's Anderson-type argument, M = N).

Rigour chain:
  1. Exact model: spec of K H with integer couplings (model.spec_for, scale = K_SCALE).
  2. SU(2): 18 spins 1/2 have integer total spin, so every multiplet has an S^z = 0 member and
     lambda_min(H) = lambda_min(H on S^z = 0).
  3. Z2^7 symmetry (pair swaps and spin flip): 128 one-dimensional characters, orbit-sum bases with
     entries +-1; completeness checked by dimension; symmetry checked exactly against the model.
  4. Exact inertia of each block at g (checker.inertia_exact): if no eigenvalue of K H lies below
     g - delta in any block, lambda_min(K H) >= g - delta.
  5. e_GS >= lambda_min(H) >= (g - delta) / K.
No floating-point assumption: a Lanczos estimate only places g.
"""
import math
import time
from fractions import Fraction

import numpy as np
import scipy.sparse.linalg as spla

from ...basis import magnetization_state_set
from ...construct import build_bitwise
from ...exact_sectors import exact_count_below
from .model import K_SCALE, spec_for, symmetry_generators

PUBLISHED = {"n1": 381414, "n3": 817957}     # c1 = 0.0381414, c3 = 0.0817957 (Kranitz & Penc, Eq. 67)


def estimate_e0(n1, n3, seed=12345):
    spec, _ = spec_for(n1, n3)
    st = magnetization_state_set(spec, 0)
    H = build_bitwise(spec, st)
    v0 = np.random.default_rng(seed).standard_normal(H.shape[0])
    return float(spla.eigsh(H, k=1, which="SA", tol=0, v0=v0, return_eigenvectors=False)[0]), st


def certify_crown(n1=PUBLISHED["n1"], n3=PUBLISHED["n3"], margins_rel=(1e-10, 1e-9, 2e-8), log=print):
    """Exact lower bound on lambda_min(H_crown). The shift g is placed a relative margin below the
    Lanczos estimate on a fine dyadic grid (2^-20 in K H units); if an eigenvalue is counted below it,
    the next, larger margin is tried. The estimate only places g: the bound g - delta is certified by
    the exact counts whatever its accuracy."""
    t0 = time.time()
    e0, st = estimate_e0(n1, n3)
    spec_K, (c1, c3, c17) = spec_for(n1, n3, scale=K_SCALE)
    gens = symmetry_generators(spec_K)
    grid = 2 ** 20
    r = info = g = None
    tried = []
    for m in margins_rel:
        g = Fraction(int(np.floor((e0 - m * abs(e0)) * K_SCALE * grid)), grid)      # in K H units
        res, info = exact_count_below(spec_K, st, gens, {}, [g], log=None)
        r = res[0]
        tried.append({"margin_rel": m, "shift": str(g / K_SCALE), "count": r["count"]})
        if r["count"] == 0:
            break
        log(f"  crown S=1/2: {r['count']} eigenvalue(s) counted below the shift at margin {m}; widening")
    if r["count"] != 0:
        raise RuntimeError(f"exact count below the shift is {r['count']} at every margin, expected 0")
    lb = (g - r["delta"]) / K_SCALE
    rec = {"n1": n1, "n3": n3, "c1": str(c1), "c3": str(c3), "c17": str(c17), "K": K_SCALE,
           "dim_Sz0": int(st.size), "n_sectors": len(info["sectors"]),
           "largest_sector": max(s["dim"] for s in info["sectors"]), "e0_estimate": e0,
           "shift": str(g / K_SCALE), "shifts_tried": tried, "delta": float(r["delta"] / K_SCALE),
           # floats on the safe side of the exact rational bound: one step down after rounding to
           # nearest; the 9-decimal value is floor(lb * 1e9) / 1e9 computed exactly
           "lower_bound": float(np.nextafter(float(lb), -np.inf)),
           "lower_bound_rounded_down": float(Fraction(math.floor(lb * 10 ** 9), 10 ** 9)),
           "lower_bound_exact": f"{lb.numerator}/{lb.denominator}", "no_floating_point_assumption": True,
           "seconds": time.time() - t0}
    log(f"  crown S=1/2 n1={n1} n3={n3}: e0 estimate {e0:.12f}; certified lambda_min >= {float(lb):.12f} "
        f"({rec['n_sectors']} sectors, largest {rec['largest_sector']}, {rec['seconds']:.0f}s)")
    return rec
