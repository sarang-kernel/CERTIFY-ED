"""Independent adversarial search for Proposition 16 (docs/proofs/P16_thermo_intervals.md, thermo.thermodynamics).

Ground truth: eigenvalues placed anywhere in the certified intervals (ends, midpoints, random members, all at
one end), the exact Z, F, U, Var, C computed with mpmath at 120 digits from those members; the reported binary64
intervals must contain them. Targets: degenerate intervals (l = h, Var = 0 exactly so C's interval must contain
0), one level (n = 1), two levels far apart, counts up to 10^6, energies at scale 1e+-150, huge offset with tiny
spread (E_0 ~ 1e200, cancellation in a_k), negative energies, T at 2^-192 (grid boundary; below it the code must
raise), T = 1e300, T = 1e-300 (ZeroDivisionError expected), intervals whose ends are adjacent binary64 numbers,
inconsistent counts (dimension mismatch must raise).

Usage: python docs/proofs/independent/adversarial_P16_thermo_intervals.py [--cases 200] [--seed 31]
"""
import argparse
import math
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, report  # noqa: E402
from certify_ed.thermo import thermodynamics  # noqa: E402

mp.mp.dps = 120


def mpq(x):
    f = Fraction(x) if not isinstance(x, Fraction) else x
    return mp.mpf(f.numerator) / mp.mpf(f.denominator)


def exact_quantities(members, T):
    """members: list of (energy as Fraction, multiplicity). Returns Z, F, U, Var, C, lnZ as mp."""
    beta = 1 / mpq(T)
    E0 = min(e for e, _ in members)
    ws = [(mpq(e), m * mp.exp(-beta * (mpq(e) - mpq(E0)))) for e, m in members]
    Zt = sum(w for _, w in ws)
    mean = sum(e * w for e, w in ws) / Zt
    var = sum(e * e * w for e, w in ws) / Zt - mean ** 2
    lnZ = mp.log(Zt) - beta * mpq(E0)
    return dict(Z=mp.exp(lnZ), F=-mpq(T) * lnZ, U=mean, variance_E=var, C=beta ** 2 * var, lnZ=lnZ)


def contains(iv, val):
    if iv is None:
        return True       # the code reported no value (over/underflow of Z): nothing to check
    lo, hi = iv
    return mp.mpf(lo) <= val <= mp.mpf(hi)


def make(rng, i):
    kind = i % 10
    if kind == 0:
        m = int(rng.integers(1, 6)); los = np.sort(rng.standard_normal(m)); ws = np.abs(rng.standard_normal(m)) * 1e-6
        ivs = [{"lo": float(l), "hi": float(l + w), "count": int(rng.integers(1, 4))} for l, w in zip(los, ws)]
        return ivs, [float(t) for t in rng.choice([0.01, 0.1, 1.0, 10.0], size=2)], "random", False
    if kind == 1:   # degenerate intervals (exact levels), Var exactly 0 when one level
        return [{"lo": 0.5, "hi": 0.5, "count": 3}], [0.3, 2.0], "single_degenerate_level", True
    if kind == 2:   # two levels far apart, low and high T
        return [{"lo": -1.0, "hi": -1.0, "count": 1}, {"lo": 1e3, "hi": 1e3 + 1e-9, "count": 2}], [1e-3, 1.0, 1e4], "two_levels_far", True
    if kind == 3:   # huge counts
        return [{"lo": -2.0, "hi": -1.999, "count": 10 ** 6}, {"lo": 0.0, "hi": 0.001, "count": 10 ** 6 - 1}], [0.5, 50.0], "huge_counts", True
    if kind == 4:   # scale 1e+-150
        s = 1e150 if i % 20 < 10 else 1e-150
        return [{"lo": -s, "hi": -s * (1 - 1e-12), "count": 2}, {"lo": s * 0.5, "hi": s, "count": 1}], [s * 0.1, s * 10], f"scale({s:g})", True
    if kind == 5:   # huge offset, tiny spread
        return [{"lo": 1e200, "hi": 1e200 * (1 + 2 ** -52), "count": 2}, {"lo": 1e200 * (1 + 2 ** -50), "hi": 1e200 * (1 + 2 ** -49), "count": 1}], [1e184, 1e190], "huge_offset", True
    if kind == 6:   # T at the grid boundary 2^-192 and T = 1e300
        return [{"lo": -1.0, "hi": -0.5, "count": 1}, {"lo": 0.0, "hi": 0.25, "count": 2}], [2.0 ** -192, 1e300, 2.0 ** -100], "T_boundaries", True
    if kind == 7:   # adjacent binary64 ends
        x = float(rng.standard_normal())
        return [{"lo": x, "hi": math.nextafter(x, math.inf), "count": 2}, {"lo": x + 1, "hi": math.nextafter(x + 1, math.inf), "count": 1}], [0.7], "adjacent_floats", True
    if kind == 8:   # n = 1
        return [{"lo": float(rng.standard_normal()), "hi": float(abs(rng.standard_normal())) + 3, "count": 1}], [0.2, 5.0], "n_eq_1", True
    # 9: many narrow intervals (spectrum-like), moderate T
    los = np.sort(rng.standard_normal(40)) * 5
    return [{"lo": float(l), "hi": float(l + 1e-10), "count": 1} for l in los], [0.05, 0.5, 5.0], "spectrum_like", True


def members_for(ivs, rng, mode):
    out = []
    for iv in ivs:
        lo, hi, c = Fraction(iv["lo"]), Fraction(iv["hi"]), iv["count"]
        if mode == "lo":
            out.append((lo, c))
        elif mode == "hi":
            out.append((hi, c))
        elif mode == "mid":
            out.append(((lo + hi) / 2, c))
        else:   # spread the members across the interval
            for j in range(min(c, 5)):
                t = Fraction(int(rng.integers(0, 1001)), 1000)
                out.append((lo + t * (hi - lo), c // min(c, 5) + (1 if j < c % min(c, 5) else 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_in = Tally("P16: reported intervals contain the exact Z, lnZ, F, U, Var, C for members of the intervals")
    T_neg = Tally("negative controls: dimension mismatch, T <= 0, T below the grid raise")
    T_nov = Tally("no-value cases: ln of a Z~ with lower end 0 raises only when beta * b_k > 130 for every interval")
    for i in range(a.cases):
        ivs, temps, tag, adv = make(rng, i)
        n = sum(iv["count"] for iv in ivs)
        rows = []
        for T in temps:
            try:
                rows += thermodynamics(ivs, [T], dimension=n)["rows"]
            except ZeroDivisionError:
                T_neg.add(T < 2.0 ** -192, True, f"{tag}: ZeroDivisionError at T={T}")
            except ValueError as e:
                # no value: the lower end of Z~ is 0 on the grid (beta * b_k > ~133 for every interval); legitimate,
                # certify() reports INCONCLUSIVE. Check that this is indeed the situation.
                # beta on the grid is at most max(1/T, 2^-192) rounded up by one unit; the lower end of W_k is
                # exp(-beta_hi * b_k), which is 0 on the grid when beta_hi * b_k > ~133
                E0 = min(iv["lo"] for iv in ivs)
                beta_hi = max(1.0 / T, 2.0 ** -192) + 2.0 ** -192
                T_nov.add(all((iv["hi"] - E0) * beta_hi > 130 for iv in ivs), True, f"{tag}: ValueError at T={T} ({e}) although some beta_hi*b_k <= 130")
        res = {"rows": rows}
        for row in res["rows"]:
            T = row["T"]
            for mode in ("lo", "hi", "mid", "spread"):
                mem = members_for(ivs, rng, mode)
                ex = exact_quantities(mem, T)
                ok = all(contains(row.get(key), ex[key]) for key in ("Z", "lnZ", "F", "U", "variance_E", "C") if key in row)
                T_in.add(ok, adv, f"{tag} T={T} mode={mode}: " + "; ".join(f"{k}: {row.get(k)} vs {mp.nstr(ex[k], 20)}" for k in ex if k in row and not contains(row.get(k), ex[k])))
    # negative controls
    for i in range(20):
        ivs = [{"lo": -1.0, "hi": -0.5, "count": 1}, {"lo": 0.0, "hi": 0.25, "count": 2}]
        try:
            thermodynamics(ivs, [1.0], dimension=4); T_neg.add(False, True, "dimension mismatch not raised")
        except ValueError:
            T_neg.add(True, True, "")
        try:
            thermodynamics(ivs, [-1.0 if i % 2 else 0.0], dimension=3); T_neg.add(False, True, "T <= 0 not raised")
        except ValueError:
            T_neg.add(True, True, "")
        try:
            thermodynamics(ivs, [1e-300], dimension=3); T_neg.add(False, True, "T below the grid not raised")
        except ZeroDivisionError:
            T_neg.add(True, True, "")
    sys.exit(1 if report([T_in, T_neg, T_nov]) else 0)


if __name__ == "__main__":
    main()
