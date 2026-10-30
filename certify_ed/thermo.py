"""S7: thermodynamic quantities as intervals from a complete whole-spectrum enclosure (Proposition 16).

Input: intervals [l_k, h_k] with counts c_k whose union contains every eigenvalue, interval k holding
exactly c_k of them (Proposition 9, `enclose_spectrum(...).complete == True`). Output per temperature T > 0
(k_B = 1): intervals for Z, F = -T ln Z, U = <E> and C = beta^2 (<E^2> - <E>^2), each certified to contain
the exact value for the matrix whose spectrum is enclosed. All arithmetic is dyadic interval arithmetic
(`ivl.py`, Lemma 17); the only floating-point steps are reading the input floats exactly and the final
outward conversion of each end to binary64.

The hypothesis and the known looseness (C and, to a lesser extent, U are quotients of intervals that ignore
the correlation between numerator and denominator) are stated in every record. Derivation and machine check:
docs/proofs/P16_thermo_intervals.md and docs/proofs/L17_dyadic_exp_ln.md; independent re-derivation pending (D-12).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence

from .ivl import Interval, exp_interval, from_float, from_fraction, ln_interval, sum_intervals

HYPOTHESIS = ("every eigenvalue of the operator lies in one of the listed intervals and interval k contains "
              "exactly count_k eigenvalues (complete whole-spectrum enclosure, Prop 9); k_B = 1; T > 0")
LOOSENESS = ("U and C are quotients of interval sums: the bound ignores the correlation between numerator and "
             "denominator, so the C interval can be much wider than the eigenvalue intervals, especially at low T")

__all__ = ["thermodynamics", "thermo_intervals_from_enclosure", "HYPOTHESIS", "LOOSENESS"]


def thermo_intervals_from_enclosure(enc) -> List[dict]:
    """[{lo, hi, count}] from a SpectrumEnclosure (requires enc.complete)."""
    if not getattr(enc, "complete", False):
        raise ValueError("S7 needs a complete enclosure (every interval counted, counts summing to the dimension)")
    return [{"lo": float(iv.lo), "hi": float(iv.hi), "count": int(iv.count)} for iv in enc.intervals]


def _as_interval(lo: float, hi: float) -> Interval:
    a, b = from_float(float(lo)), from_float(float(hi))
    return Interval(a.lo, b.hi)


def thermodynamics(intervals: Sequence[dict], temperatures: Iterable[float], dimension: Optional[int] = None,
                   quantities: Sequence[str] = ("Z", "F", "U", "C")) -> Dict[str, object]:
    """Interval bounds on Z, F, U, C at each temperature (Proposition 16).

    intervals: [{"lo", "hi", "count"}], counts summing to the dimension (checked when `dimension` is given).
    Returns {"temperatures": [...], "rows": [{"T", "beta", "Z": [lo, hi], "F": [...], "U": [...], "C": [...]}],
    "hypothesis", "looseness", "energy_offset", "dimension"}; endpoints are binary64 rounded outward.
    Temperatures are read as exact binary64 values; beta = 1/T is an enclosure."""
    ivs = []
    counts = []
    for r in intervals:
        lo, hi, c = float(r["lo"]), float(r["hi"]), int(r["count"])
        if not (lo <= hi) or c <= 0:
            raise ValueError(f"bad interval {r}")
        ivs.append((lo, hi))
        counts.append(c)
    if not ivs:
        raise ValueError("no intervals")
    n = sum(counts)
    if dimension is not None and n != int(dimension):
        raise ValueError(f"counts sum to {n}, dimension is {dimension}: the enclosure is not complete")
    E0 = min(lo for lo, _ in ivs)                      # exact binary64; shift so that all energies are >= 0
    E0q = Fraction(E0)
    shifted = [(from_fraction(Fraction(lo) - E0q), from_fraction(Fraction(hi) - E0q)) for lo, hi in ivs]
    e_ivs = [Interval(a.lo, b.hi) for a, b in shifted]   # e in [a_k, b_k], a_k >= 0
    E0_iv = from_float(E0)
    rows = []
    temps = [float(t) for t in temperatures]
    for T in temps:
        if not (T > 0):
            raise ValueError("temperatures must be > 0")
        T_iv = from_float(T)
        beta = Interval(1 << 0, 1 << 0)                 # placeholder, replaced below
        one = from_fraction(Fraction(1))
        beta = one / T_iv
        W = []
        for e_iv, c in zip(e_ivs, counts):
            # weight interval c * [exp(-beta b), exp(-beta a)]: exp of the interval -beta*[a, b]
            arg = -(beta * e_iv)
            W.append(exp_interval(arg).scale_int(c))
        Zt = sum_intervals(W)                              # Z e^{beta E0}
        row = {"T": T, "beta": list(beta.floats())}
        if "Z" in quantities:
            # Z = e^{-beta E0} Zt; report Z and ln Z (Z itself may over/underflow binary64)
            lnZ = ln_interval(Zt) - beta * E0_iv
            row["lnZ"] = list(lnZ.floats())
            try:
                row["Z"] = list(exp_interval(lnZ).floats())
            except (OverflowError, ValueError):
                row["Z"] = None
        if "F" in quantities or "Z" in quantities:
            lnZt = ln_interval(Zt)
            F = E0_iv - T_iv * lnZt
            row["F"] = list(F.floats())
        if "U" in quantities or "C" in quantities:
            num1 = sum_intervals(w * e for w, e in zip(W, e_ivs))
            e_mean = num1 / Zt
            U = E0_iv + e_mean
            row["U"] = list(U.floats())
        if "C" in quantities:
            num2 = sum_intervals(w * e.square() for w, e in zip(W, e_ivs))
            e2_mean = num2 / Zt
            var = (e2_mean - e_mean.square()).intersect_nonneg()
            C = beta.square() * var
            row["C"] = list(C.floats())
            row["variance_E"] = list(var.floats())
        rows.append(row)
    return {"temperatures": temps, "rows": rows, "hypothesis": HYPOTHESIS, "looseness": LOOSENESS,
            "energy_offset": E0, "dimension": n, "n_intervals": len(ivs),
            "method": "dyadic interval arithmetic, 192-bit grid, outward rounding (ivl.py, Lemma 17); "
                      "Proposition 16 [derived: docs/proofs/P16_thermo_intervals.md; UNVERIFIED by independent "
                      "re-derivation, D-12]"}
