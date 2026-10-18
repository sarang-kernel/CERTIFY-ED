"""Structural SU(2) invariance of a specification, and the total-spin Casimir on a basis.

A spin specification is SU(2) invariant if, for every pair of sites, the two-site bilinear part is
a multiple of S_i . S_j, there are no one-site terms other than the identity, and there are no terms
on three or more sites. The test reads the stored couplings and decides exactly (equalities between
the stored binary floating-point numbers); it is conservative: an invariant Hamiltonian written in
another form (for example a biquadratic (S_i.S_j)^2 term or a three-site scalar chirality written
in components) is reported as not established.

If the test passes, [H, S_tot^2] = 0 and [H, S_tot^z] = 0 hold exactly for the declared model, so
Proposition 1 of the paper applies to S_tot^2 without a numerical commutation check.
"""
from collections import defaultdict
from fractions import Fraction
from typing import Tuple

import numpy as np

from .spec import ModelSpec


def _letter_scales(convention: str):
    # factor that turns a stored letter into S-units: stored = f * S^a
    if convention == "spin":
        return 1.0, 1.0
    if convention == "pauli":
        return 2.0, 2.0
    if convention == "pauli_pm1":
        return 2.0, 1.0
    raise ValueError(convention)


def su2_structural(spec: ModelSpec) -> Tuple[bool, str]:
    loc = spec.local
    if loc.kind != "spin":
        return False, "local space is not a spin"
    fxyz, fpm = _letter_scales(loc.convention or "spin")
    pair = defaultdict(lambda: defaultdict(complex))
    for t in spec.terms:
        op = t.op
        if set(op) <= {"I"}:
            continue
        if len(op) != 2:
            return False, f"term '{op}' acts on {len(op)} sites"
        for c, sites in t.couplings:
            i, j = sites
            if i == j:
                return False, f"term '{op}' acts twice on site {i}"
            a, b = op
            if i > j:
                i, j, a, b = j, i, b, a
            f = (fxyz if a in "xyz" else fpm if a in "+-" else None)
            g = (fxyz if b in "xyz" else fpm if b in "+-" else None)
            if f is None or g is None:
                return False, f"letter in '{op}' is not a spin component"
            pair[(i, j)][a + b] += complex(c) * f * g   # coefficient of the product in S-units
    for (i, j), co in pair.items():
        allowed = {"xx", "yy", "zz", "+-", "-+"}
        bad = set(k for k, v in co.items() if v != 0) - allowed
        if bad:
            return False, f"pair ({i},{j}) has non-invariant components {sorted(bad)}"
        pm, mp = co.get("+-", 0), co.get("-+", 0)
        if pm != mp:
            return False, f"pair ({i},{j}): S+S- and S-S+ coefficients differ (antisymmetric exchange)"
        # S+_i S-_j + S-_i S+_j = 2 (SxSx + SySy)
        cxx = co.get("xx", 0) + 2 * pm
        cyy = co.get("yy", 0) + 2 * pm
        czz = co.get("zz", 0)
        if not (cxx == cyy == czz):
            return False, f"pair ({i},{j}): xx, yy, zz coefficients {cxx}, {cyy}, {czz} are not equal"
        if czz.imag != 0:
            return False, f"pair ({i},{j}): complex exchange coupling"
    return True, "every two-site term is a real multiple of S_i.S_j; no one-site or longer terms"


def casimir_spec(spec: ModelSpec) -> ModelSpec:
    """S_tot^2 = sum_{i != j} S_i.S_j + N S(S+1), in the local convention 'spin'."""
    from .localops import LocalSpace
    loc = spec.local
    S = Fraction(loc.spin)
    n = spec.n_sites
    zz = [[2.0, i, j] for i in range(n) for j in range(i + 1, n)]
    pm = [[1.0, i, j] for i in range(n) for j in range(i + 1, n)]
    const = float(n * S * (S + 1))
    return ModelSpec.from_static(n, LocalSpace.spin_space(str(S), "spin"),
                                 [["zz", zz], ["+-", pm], ["-+", pm], ["I", [[const, 0]]]],
                                 name="S_total_squared")


def allowed_casimir_values(spec: ModelSpec):
    S = Fraction(spec.local.spin)
    smax = spec.n_sites * S
    s0 = smax - int(smax)                       # 0 or 1/2
    vals, s = [], s0
    while s <= smax:
        vals.append(float(s * (s + 1)))
        s += 1
    return np.array(vals)
