"""Computational verification of Proposition 16 (docs/proofs/P16_thermo_intervals.md).

For random complete enclosures ([l_k, h_k], c_k) and random spectra consistent with them (each eigenvalue in
its interval, ends included with positive probability), the exact Z, F, U, C computed with mpmath at 200
digits must lie in the intervals returned by certify_ed.thermo.thermodynamics. Adversarial cases: point
spectra, extreme spectra (all eigenvalues at the lower or upper ends), a single interval (C = 0), very low
and very high temperatures, large energies, negative energies, touching intervals.

Usage: python docs/proofs/verify_P16.py [--random N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import mpmath as mp  # noqa: E402

from certify_ed.thermo import thermodynamics  # noqa: E402

mp.mp.dps = 200


def exact_thermo(E, T):
    """(Z, F, U, C) as mpmath interval numbers (mpmath.iv, 200 digits) for the spectrum E (floats or
    Fractions) and temperature T: each contains the exact value, with width of order 1e-195 relative. The
    interval form is needed because C = beta^2 (<E^2> - <E>^2) cancels exactly for a single level and a
    point value computed at 200 digits can come out as -1e-200."""
    iv = mp.iv
    iv.dps = 200
    b = 1 / iv.mpf(T)
    Em = [iv.mpf(e.numerator) / e.denominator if isinstance(e, Fraction) else iv.mpf(e) for e in E]
    w = [iv.exp(-b * e) for e in Em]
    Z = sum(w[1:], w[0])
    U = sum((e * x for e, x in zip(Em[1:], w[1:])), Em[0] * w[0]) / Z
    E2 = sum((e * e * x for e, x in zip(Em[1:], w[1:])), Em[0] * Em[0] * w[0]) / Z
    C = b * b * (E2 - U * U)
    F = -iv.log(Z) / b
    return Z, F, U, C


def _meets(bounds, ref) -> bool:
    """The implementation interval [lo, hi] (floats) meets the reference interval (which contains the exact
    value): lo <= ref.b and ref.a <= hi."""
    lo, hi = bounds
    return bool(mp.mpf(lo) <= ref.b and ref.a <= mp.mpf(hi))


REF_MAX_WIDTH = [mp.mpf(0)]      # widest reference interval seen (absolute), to show the check is sharp


def _inside(row, Z, F, U, C):
    REF_MAX_WIDTH[0] = max(REF_MAX_WIDTH[0], *(mp.mpf(x.delta.b) for x in (F, U, C)))
    ok = _meets(row["F"], F) and _meets(row["U"], U) and _meets(row["C"], C)
    ok &= _meets(row["lnZ"], mp.iv.log(Z))
    if row["Z"] is not None:
        ok &= _meets(row["Z"], Z)
    ok &= row["C"][0] >= 0 and row["variance_E"][0] >= 0
    return bool(ok)


def _spectrum(rng: random.Random, ivs, mode="random"):
    E = []
    for r in ivs:
        lo, hi, c = r["lo"], r["hi"], r["count"]
        for _ in range(c):
            if mode == "lo":
                E.append(lo)
            elif mode == "hi":
                E.append(hi)
            elif mode == "alt":
                E.append(lo if rng.random() < 0.5 else hi)
            else:
                t = rng.random()
                E.append(lo if t < 0.1 else hi if t < 0.2 else lo + (hi - lo) * rng.random())
    return E


def _random_enclosure(rng: random.Random):
    m = rng.randrange(1, 7)
    scale = 10.0 ** rng.uniform(-2, 3)
    centers = sorted(rng.uniform(-1, 1) * scale for _ in range(m))
    ivs = []
    for c in centers:
        w = 0.0 if rng.random() < 0.3 else 10.0 ** rng.uniform(-14, -1) * scale
        ivs.append({"lo": c - w, "hi": c + w, "count": rng.randrange(1, 4)})
    ivs.sort(key=lambda r: r["lo"])
    return ivs


def check_random(rng: random.Random, n: int):
    cases = viol = 0
    for _ in range(n):
        ivs = _random_enclosure(rng)
        scale = max(abs(r["lo"]) for r in ivs) + max(abs(r["hi"]) for r in ivs) + 1e-3
        temps = [10.0 ** rng.uniform(-2, 2) * scale for _ in range(2)]
        res = thermodynamics(ivs, temps)
        for row in res["rows"]:
            for mode in ("random", "random", "lo", "hi", "alt"):
                E = _spectrum(rng, ivs, mode)
                cases += 1
                viol += not _inside(row, *exact_thermo(E, row["T"]))
    return cases, viol


def adversarial():
    """Enclosures at the hypothesis boundaries; every case is paired with its extreme spectra."""
    cases = []
    cases.append(([{"lo": 0.0, "hi": 0.0, "count": 1}, {"lo": 1.0, "hi": 1.0, "count": 1}], [1e-3, 1.0, 1e6]))
    cases.append(([{"lo": -5.0, "hi": -5.0, "count": 4}], [0.01, 1.0, 100.0]))           # one level: C = 0
    cases.append(([{"lo": 0.0, "hi": 1.0, "count": 2}, {"lo": 1.0, "hi": 2.0, "count": 1}], [0.5, 3.0]))   # touching
    cases.append(([{"lo": -1e6, "hi": -1e6 + 1e-9, "count": 1}, {"lo": 1e6, "hi": 1e6, "count": 2}], [1e3, 1e7]))
    cases.append(([{"lo": 0.0, "hi": 0.0, "count": 1}, {"lo": 2000.0, "hi": 2000.0, "count": 1}], [1.0]))   # weight 1e-869
    cases.append(([{"lo": 0.1, "hi": 0.1, "count": 1}, {"lo": 0.2, "hi": 0.3, "count": 3},
                   {"lo": 0.3, "hi": 0.3, "count": 1}], [1e-2, 0.25, 10.0]))
    cases.append(([{"lo": 1e-300, "hi": 2e-300, "count": 2}, {"lo": 1.0, "hi": 1.0, "count": 1}], [2.0 ** -190, 1.0]))
    cases.append(([{"lo": -1.0, "hi": 1.0, "count": 5}], [0.1, 1.0, 10.0]))                     # wide single interval
    return cases


def check_adversarial(rng: random.Random):
    cases = viol = 0
    for ivs, temps in adversarial():
        res = thermodynamics(ivs, temps)
        for row in res["rows"]:
            for mode in ("lo", "hi", "alt", "random", "random"):
                E = _spectrum(rng, ivs, mode)
                cases += 1
                viol += not _inside(row, *exact_thermo(E, row["T"]))
    return cases, viol


def check_tightness():
    """Point spectra: the interval widths are of the order of the grid conversion, not of the method (the
    only sources of width are the outward roundings of Lemma 17 and the final binary64 conversion)."""
    res = thermodynamics([{"lo": 0.0, "hi": 0.0, "count": 1}, {"lo": 1.0, "hi": 1.0, "count": 1}], [0.5, 1.0, 2.0])
    return max(max(r["U"][1] - r["U"][0], r["C"][1] - r["C"][0], r["F"][1] - r["F"][0]) for r in res["rows"])


def run(n_random: int = 400, seed: int = 0) -> dict:
    rng = random.Random(seed)
    cr, vr = check_random(rng, n_random)
    ca, va = check_adversarial(rng)
    return {"random_enclosures": n_random, "random_cases": cr, "random_violations": vr,
            "adversarial_cases": ca, "adversarial_violations": va, "point_spectrum_max_width": check_tightness(),
            "reference_max_width": float(REF_MAX_WIDTH[0]),
            "violations": vr + va}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.random, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
