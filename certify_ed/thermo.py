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

import math
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence

from .ivl import Interval, exp_interval, from_float, from_fraction, ln_interval, sum_intervals, to_float_up

HYPOTHESIS = ("every eigenvalue of the operator lies in one of the listed intervals and interval k contains "
              "exactly count_k eigenvalues (complete whole-spectrum enclosure, Prop 9); k_B = 1; T > 0")
LOOSENESS = ("U and C are quotients of interval sums: the bound ignores the correlation between numerator and "
             "denominator, so the C interval can be much wider than the eigenvalue intervals, especially at low T")

TAIL_HYPOTHESIS = ("the k claimed eigenvalues lie in the listed intervals, interval k containing exactly count_k of them "
                   "(S2 radii, S3 count); the remaining n - k eigenvalues are >= next_lower_bound (S3){upper}; "
                   "n is the dimension of the operator; k_B = 1; T > 0")
TAIL_LOOSENESS = ("without an upper bound on the spectrum the tail's first and second moments are bounded by the envelopes "
                  "sup_{e >= l} e exp(-beta e) and sup_{e >= l} e^2 exp(-beta e), which are informative only when the "
                  "temperature is well below the gap l = next_lower_bound - E_0 (T < l/2); U and C can then be wide")

__all__ = ["thermodynamics", "thermodynamics_truncated", "thermo_intervals_from_enclosure", "HYPOTHESIS", "LOOSENESS",
           "TAIL_HYPOTHESIS", "TAIL_LOOSENESS"]


def thermo_intervals_from_enclosure(enc) -> List[dict]:
    """[{lo, hi, count}] from a SpectrumEnclosure (requires enc.complete)."""
    if not getattr(enc, "complete", False):
        raise ValueError("S7 needs a complete enclosure (every interval counted, counts summing to the dimension)")
    return [{"lo": float(iv.lo), "hi": float(iv.hi), "count": int(iv.count)} for iv in enc.intervals]


def _as_interval(lo: float, hi: float) -> Interval:
    a, b = from_float(float(lo)), from_float(float(hi))
    return Interval(a.lo, b.hi)


def _read_intervals(intervals):
    ivs, counts = [], []
    for r in intervals:
        lo, hi, c = float(r["lo"]), float(r["hi"]), int(r["count"])
        if not (lo <= hi) or c <= 0:
            raise ValueError(f"bad interval {r}")
        ivs.append((lo, hi))
        counts.append(c)
    if not ivs:
        raise ValueError("no intervals")
    return ivs, counts


def _rows(ivs, counts, temperatures, quantities, tail=None):
    """The interval computation of Proposition 16 (tail=None) and of its truncated-spectrum extension
    (docs/proofs/C06_truncated_thermodynamics.md): tail = (m, L) adds m eigenvalues known only to be >= L."""
    E0 = min(lo for lo, _ in ivs)                      # exact binary64; shift so that all energies are >= 0
    E0q = Fraction(E0)
    shifted = [(from_fraction(Fraction(lo) - E0q), from_fraction(Fraction(hi) - E0q)) for lo, hi in ivs]
    e_ivs = [Interval(a.lo, b.hi) for a, b in shifted]   # e in [a_k, b_k], a_k >= 0
    E0_iv = from_float(E0)
    one = from_fraction(Fraction(1))
    if tail is not None:
        m_tail, L, M = tail
        ell = from_fraction(Fraction(float(L)) - E0q)       # l = L - E_0 > 0 (checked by the caller)
        if ell.lo <= 0 or m_tail <= 0:
            raise ValueError("tail needs next_lower_bound > every claimed interval and a positive omitted count")
        em = None if M is None else from_fraction(Fraction(float(M)) - E0q)   # m = M - E_0 >= l
        e_m1 = exp_interval(from_fraction(Fraction(-1)))     # e^-1
        e_m2 = exp_interval(from_fraction(Fraction(-2)))     # e^-2
    rows = []
    temps = [float(t) for t in temperatures]
    for T in temps:
        if not (T > 0):
            raise ValueError("temperatures must be > 0")
        T_iv = from_float(T)
        beta = one / T_iv
        W = []
        for e_iv, c in zip(e_ivs, counts):
            # weight interval c * [exp(-beta b), exp(-beta a)]: exp of the interval -beta*[a, b]
            arg = -(beta * e_iv)
            W.append(exp_interval(arg).scale_int(c))
        E_list = list(e_ivs)
        row = {"T": T, "beta": list(beta.floats())}
        if tail is not None:
            # omitted weight: m members of [exp(-beta m), exp(-beta l)] (lower end 0 without M); first and second
            # moments: m members of [inf, sup] of phi(e) = e exp(-beta e) and psi(e) = e^2 exp(-beta e) over [l, m]
            # (C06, Lemma C: phi increases up to 1/beta = T and decreases after, psi up to 2T)
            w_l = exp_interval(-(beta * ell))
            w_m = exp_interval(-(beta * em)) if em is not None else None
            phi_l, psi_l = ell * w_l, ell.square() * w_l
            if em is not None:
                phi_m, psi_m = em * w_m, em.square() * w_m
            # suprema: at the clamp of T (resp. 2T) into [l, m]
            if ell.lo >= T_iv.hi:
                phi_sup, regime = phi_l.hi, "T <= l: phi(l)"
            elif em is not None and em.hi <= T_iv.lo:
                phi_sup, regime = phi_m.hi, "T >= m: phi(m)"
            else:
                phi_sup, regime = (T_iv * e_m1).hi, "l < T < m: e^-1 / beta"
            two_T = T_iv.scale_int(2)
            if ell.lo >= two_T.hi:
                psi_sup = psi_l.hi
            elif em is not None and em.hi <= two_T.lo:
                psi_sup = psi_m.hi
            else:
                psi_sup = (T_iv.square() * e_m2).scale_int(4).hi
            # infima: 0 without M (phi, psi -> 0 at infinity); else at an end of [l, m] (unimodal functions)
            phi_inf = 0 if em is None else min(phi_l.lo, phi_m.lo)
            psi_inf = 0 if em is None else min(psi_l.lo, psi_m.lo)
            W_tail = Interval(0 if w_m is None else w_m.lo, w_l.hi).scale_int(m_tail)
            T1_tail = Interval(phi_inf, phi_sup).scale_int(m_tail)
            T2_tail = Interval(psi_inf, psi_sup).scale_int(m_tail)
            row["tail"] = {"omitted_count": int(m_tail), "weight_bounds": list(W_tail.floats()),
                           "weight_upper": to_float_up(W_tail.hi),
                           "first_moment_bounds": list(T1_tail.floats()), "second_moment_bounds": list(T2_tail.floats()),
                           "envelope_regime": regime}
            Zt = sum_intervals(W + [W_tail])
            Zt_claimed = sum_intervals(W)
            if Zt_claimed.lo > 0:
                row["tail"]["weight_fraction_upper"] = to_float_up((W_tail / Zt_claimed).hi)
        else:
            Zt = sum_intervals(W)                              # Z e^{beta E0}
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
            num1 = sum_intervals([w * e for w, e in zip(W, E_list)] + ([T1_tail] if tail is not None else []))
            e_mean = num1 / Zt
            U = E0_iv + e_mean
            row["U"] = list(U.floats())
        if "C" in quantities:
            num2 = sum_intervals([w * e.square() for w, e in zip(W, E_list)] + ([T2_tail] if tail is not None else []))
            e2_mean = num2 / Zt
            var = (e2_mean - e_mean.square()).intersect_nonneg()
            C = beta.square() * var
            row["C"] = list(C.floats())
            row["variance_E"] = list(var.floats())
        rows.append(row)
    return E0, temps, rows


def thermodynamics(intervals: Sequence[dict], temperatures: Iterable[float], dimension: Optional[int] = None,
                   quantities: Sequence[str] = ("Z", "F", "U", "C")) -> Dict[str, object]:
    """Interval bounds on Z, F, U, C at each temperature (Proposition 16).

    intervals: [{"lo", "hi", "count"}], counts summing to the dimension (checked when `dimension` is given).
    Returns {"temperatures": [...], "rows": [{"T", "beta", "Z": [lo, hi], "F": [...], "U": [...], "C": [...]}],
    "hypothesis", "looseness", "energy_offset", "dimension"}; endpoints are binary64 rounded outward.
    Temperatures are read as exact binary64 values; beta = 1/T is an enclosure."""
    ivs, counts = _read_intervals(intervals)
    n = sum(counts)
    if dimension is not None and n != int(dimension):
        raise ValueError(f"counts sum to {n}, dimension is {dimension}: the enclosure is not complete")
    E0, temps, rows = _rows(ivs, counts, temperatures, quantities)
    return {"temperatures": temps, "rows": rows, "hypothesis": HYPOTHESIS, "looseness": LOOSENESS,
            "energy_offset": E0, "dimension": n, "n_intervals": len(ivs),
            "method": "dyadic interval arithmetic, 192-bit grid, outward rounding (ivl.py, Lemma 17); "
                      "Proposition 16 [derived: docs/proofs/P16_thermo_intervals.md; UNVERIFIED by independent "
                      "re-derivation, D-12]"}


def thermodynamics_truncated(intervals: Sequence[dict], next_lower_bound: float, dimension: int,
                             temperatures: Iterable[float], upper_bound: Optional[float] = None,
                             quantities: Sequence[str] = ("Z", "F", "U", "C")) -> Dict[str, object]:
    """Interval bounds on Z, F, U, C from the k lowest eigenvalues only (C-06; docs/proofs/C06_truncated_thermodynamics.md).

    intervals: the k certified intervals (counts summing to k < n); next_lower_bound: L <= lambda_{k+1} (the S3
    bound, declared model); dimension: n; upper_bound: M >= lambda_n when known (the operator norm bound), in which
    case the omitted weight and moments are bounded on both sides by the envelopes of Lemma C over [L, M];
    otherwise the omitted weight lies in [0, (n - k) exp(-beta L)] and the omitted moments in [0, (n - k) sup] of
    the envelopes over [L, inf). Two-sided intervals for Z, F, U, C in both cases; the record carries `tail_bound`."""
    ivs, counts = _read_intervals(intervals)
    k = sum(counts)
    n = int(dimension)
    L = float(next_lower_bound)
    if not (k <= n):
        raise ValueError(f"claimed counts {k} exceed the dimension {n}")
    hi_k = max(hi for _, hi in ivs)
    if not (L > hi_k):
        raise ValueError(f"next_lower_bound {L} does not exceed the highest claimed interval end {hi_k}")
    m_tail = n - k
    if m_tail == 0:
        res = thermodynamics(intervals, temperatures, dimension=n, quantities=quantities)
        res["tail_bound"] = {"omitted_count": 0, "method": "none: the claim is the whole spectrum"}
        return res
    M = None
    if upper_bound is not None and math.isfinite(float(upper_bound)):
        M = float(upper_bound)
        if not (M >= L):
            raise ValueError(f"upper_bound {M} below next_lower_bound {L}")
    E0, temps, rows = _rows(ivs, counts, temperatures, quantities, tail=(m_tail, L, M))
    method = "envelope with upper bound" if M is not None else "envelope"
    hyp = TAIL_HYPOTHESIS.format(upper="" if M is None else f" and <= upper_bound {M!r} (operator norm bound)")
    loose = LOOSENESS + ("" if M is not None else "; " + TAIL_LOOSENESS)
    return {"temperatures": temps, "rows": rows, "hypothesis": hyp, "looseness": loose,
            "energy_offset": E0, "dimension": n, "n_intervals": len(ivs), "claimed_count": k,
            "tail_bound": {"omitted_count": m_tail, "next_lower_bound": L, "upper_bound": M, "method": method,
                           "weight_bound": "sum_{j>k} exp(-beta lambda_j) <= (n - k) exp(-beta L)",
                           "statement": "docs/proofs/C06_truncated_thermodynamics.md (Proposition 16')"},
            "method": "dyadic interval arithmetic, 192-bit grid, outward rounding (ivl.py, Lemma 17); truncated spectrum "
                      f"via the tail bound, method {method!r} (C-06, Proposition 16'; derived: "
                      "docs/proofs/C06_truncated_thermodynamics.md; UNVERIFIED by independent re-derivation, D-12)"}
