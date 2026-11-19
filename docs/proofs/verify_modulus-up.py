"""Computational verification of docs/proofs/fp_modulus-up.md (Lemma M: fp.modulus_up), in exact rational arithmetic.

For z = x + iy with binary64 components (fractions.Fraction, no rounding anywhere in the checks):
  (i)   m >= |z|                     checked as m^2 >= x^2 + y^2 (m = inf passes);
  (ii)  m <= |z| (1 + 12u)           when |z| >= 2^-1022, checked as m^2 <= (x^2 + y^2)(1 + 12u)^2;
  (ii') m <= |z| (1 + 12u) + 2^-1073 for every finite m, checked as (m - 2^-1073)^2 <= (x^2 + y^2)(1 + 12u)^2 or m <= 2^-1073;
  (iii) m = inf only if |z| (1 + 12u) > the largest binary64 number Omega, checked as (x^2 + y^2)(1 + 12u)^2 > Omega^2;
  (iv)  a zero component gives m = |z| exactly, checked as m == |other component| on every such case;
  specials: inf and nan as np.abs gives them; real floating input returned as np.abs (exact).
Random cases: components with random significands and exponents over the whole binary64 range (including subnormals),
random signs, and, for half of them, a second component |x| f 2^s with f in [0.5, 1) and s in [-4, 4], so that the
exponents differ by at most 5 (both components matter); a non-finite z arises where that scaling overflows, and is
dropped. Adversarial cases: the three
arguments on which NumPy's complex absolute value errs by more than 2u on the Windows reference host; values near
2^-1074, 2^-1022, 2^-537, 2^-511, 2^-500 and 2^500 (the ends of the unscaled branch), 2^511, 2^1023 and Omega with their
binary64 neighbours; zero components; equal
components; components differing by 2^60 and 2^-60; pairs whose result lies just below 2^-1022; 2 000 cases with both
components nonzero and |z| log-uniform over the subnormal band [2^-1074, 2^-1022] (fixed seed 1022).
Negative controls: (a) np.abs on the three arguments (platform dependent: reported, not counted); (b) three broken
variants of modulus_up, without the factor (1 + 8u) (on a second random draw of 20 000, seed + 1), without the step
below 2^-1022 and with the step only below 2^-1030 (on the adversarial cases), which (i) must detect: a variant that
is never detected fails the run.
For comparison, correlations.modulus_upper (P15 Lemma A) is checked on the adversarial cases (its infinite results
are split by cause), and both functions must agree bit for bit when both components lie in [2^-500, 2^500] (the same
five operations; checked on the second random draw). The largest relative excess (m/|z| - 1)/u is computed to 40
significant digits from the exact ratio m^2/|z|^2.

Usage: python docs/proofs/verify_modulus-up.py [--cases 200000] [--seed 7]
"""
from __future__ import annotations

import argparse
import decimal
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from certify_ed.correlations import modulus_upper  # noqa: E402
from certify_ed.fp import U, modulus_up  # noqa: E402

u = Fraction(U)
C12 = 1 + 12 * u
T1073 = Fraction(2) ** -1073
OMEGA = Fraction(float(np.finfo(float).max))
TINY = 2.0 ** -1074
MIN_NORMAL_SQ = Fraction(2) ** -2044

# NumPy 2.4.4 on Windows (AVX512F): relative errors of np.abs -2.466u, -2.462u, +2.269u (exact, docs/proofs/fp_modulus-up.md)
EXAMPLES = [0.02019294356747192 - 0.5012539518584045j, 1.9986893956853402 + 0.07375207458660175j,
            1.010573342221019 + 0.3330018761835207j]


def sq(z):
    return Fraction(float(z.real)) ** 2 + Fraction(float(z.imag)) ** 2


def check_one(z, m):
    """(ok_i, ok_ii, ok_ii', ok_iii) for a finite z and the computed m."""
    r2 = sq(z)
    if m == np.inf:
        return True, True, True, r2 * C12 * C12 > OMEGA * OMEGA
    M = Fraction(float(m))
    ok_i = M * M >= r2
    ok_ii = (r2 < MIN_NORMAL_SQ) or M * M <= r2 * C12 * C12
    t = M - T1073
    ok_iip = t <= 0 or t * t <= r2 * C12 * C12
    return ok_i, ok_ii, ok_iip, True


def random_components(rng, n):
    sig = rng.integers(0, 2 ** 52, n, dtype=np.uint64)
    ex = rng.integers(0, 2047, n, dtype=np.uint64)            # 0: subnormal or zero; 2046: largest binade
    sign = rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
    return (sign | (ex << np.uint64(52)) | sig).view(np.float64)


def random_cases(rng, n):
    x = random_components(rng, n)
    y = random_components(rng, n)
    # half of them with exponents within 4 of each other (both components matter)
    k = n // 2
    shift = rng.integers(-4, 5, k)                            # exponents within 5 of each other
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        y[:k] = np.ldexp(np.abs(x[:k]) * rng.uniform(0.5, 1.0, k), shift) * np.sign(rng.standard_normal(k))
        z = x + 1j * y
    return z[np.isfinite(z)]


def adversarial_cases():
    big = float(np.finfo(float).max)
    base = [TINY, 2 * TINY, 3 * TINY, 2.0 ** -1060, 2.0 ** -1023, 2.0 ** -1022, 2.0 ** -1000, 2.0 ** -537,
            2.0 ** -512, 2.0 ** -511, 2.0 ** -501, 2.0 ** -500, 2.0 ** -60, 0.5, 1.0, 1.5, 2.0 ** 60, 2.0 ** 500,
            2.0 ** 501, 2.0 ** 511, 2.0 ** 512, 2.0 ** 1022, 2.0 ** 1023, big]
    vals = set([0.0])
    with np.errstate(over="ignore", under="ignore"):
        for b in base:
            v, w = b, b
            for _ in range(2):
                v, w = float(np.nextafter(v, np.inf)), float(np.nextafter(w, 0.0))
            vals.update([b, float(np.nextafter(b, np.inf)), float(np.nextafter(b, 0.0)), v, w])
    vals = sorted(v for v in vals if np.isfinite(v))
    out = list(EXAMPLES)
    for a in vals:
        for b in vals:
            out += [complex(a, b), complex(-a, -b)]
    for a in vals:                                            # equal components, ratios 2^60 and 2^-60
        with np.errstate(over="ignore", under="ignore"):
            out += [complex(a, a), complex(a, -a), complex(a, a * 2.0 ** 60), complex(a * 2.0 ** -60, a)]
    for k in range(1, 200):                                   # results just below the smallest normal number
        a = k * 2.0 ** -1032
        out += [complex(a, a), complex(a, 2.0 ** -1023 - a), complex(2.0 ** -1023, a)]
    for z in EXAMPLES:                                        # the hard arguments, scaled by powers of two
        for e in (-1074, -1060, -1022, -600, -511, -60, 60, 511, 1000, 1022):
            with np.errstate(over="ignore", under="ignore"):
                out.append(complex(np.ldexp(z.real, e), np.ldexp(z.imag, e)))
    band = np.random.default_rng(1022)                        # |z| log-uniform over the subnormal band
    r, t = np.exp2(band.uniform(-1074.0, -1022.0, 4000)), band.uniform(0.0, np.pi / 2, 4000)
    with np.errstate(under="ignore"):
        zb = r * np.cos(t) + 1j * (r * np.sin(t))
    zb = zb[(zb.real != 0) & (zb.imag != 0) & (np.abs(zb) <= 2.0 ** -1022)]
    out += zb[:2000].tolist()
    z = np.array(out, dtype=complex)
    return z[np.isfinite(z)]


def _broken(z, guard=True, step=True, step_below=2.0 ** -1022):
    """modulus_up with the factor (1 + 8u) or the step below 2^-1022 removed, or the step taken only below step_below
    (negative controls)."""
    z = np.asarray(z, dtype=complex)
    x, y = np.abs(z.real), np.abs(z.imag)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        e = np.frexp(np.maximum(x, y))[1]
        a, b = np.ldexp(x, -e), np.ldexp(y, -e)
        s = np.sqrt(a * a + b * b)
        if guard:
            s = s * (1.0 + 8 * U)
        m = np.ldexp(s, e)
    if step:
        small = m < step_below
        m[small] = np.nextafter(m[small], np.inf)
    m = np.where(x == 0, y, np.where(y == 0, x, m))
    return m


def run(n_random: int = 200000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    viol = 0
    worst = Fraction(1)
    tallies = {}
    for name, Z in (("random", random_cases(rng, n_random)), ("adversarial", adversarial_cases())):
        M = modulus_up(Z)
        c = [0, 0, 0, 0, 0, 0, 0, 0]    # cases, (i), (ii), (ii'), (iii) failures, inf results, zero cases, (iv) failures
        for z, m in zip(Z.tolist(), M.tolist()):
            ok = check_one(z, m)
            c[0] += 1
            for j in range(4):
                c[1 + j] += not ok[j]
            c[5] += m == np.inf
            if z.real == 0 or z.imag == 0:
                c[6] += 1
                c[7] += Fraction(float(m)) != Fraction(abs(z.imag if z.real == 0 else z.real))
            r2 = sq(z)
            if m != np.inf and r2 >= MIN_NORMAL_SQ:
                worst = max(worst, Fraction(float(m)) ** 2 / r2)
        tallies[name] = c
        viol += sum(c[1:5]) + c[7]
    out["random_cases"] = tallies["random"][0]
    out["adversarial_cases"] = tallies["adversarial"][0]
    out["failures_i_ii_iip_iii_random"] = tallies["random"][1:5]
    out["failures_i_ii_iip_iii_adversarial"] = tallies["adversarial"][1:5]
    out["inf_results_random_adversarial"] = [tallies["random"][5], tallies["adversarial"][5]]
    out["zero_component_cases_and_iv_failures"] = [tallies["random"][6] + tallies["adversarial"][6],
                                                  tallies["random"][7] + tallies["adversarial"][7]]
    with decimal.localcontext() as ctx:
        ctx.prec = 40
        q = decimal.Decimal(worst.numerator) / decimal.Decimal(worst.denominator)
        out["largest_relative_excess_over_u"] = round(float((q.sqrt() - 1) / decimal.Decimal(U)), 3)
    # specials: inf, nan and real input
    specials = [0.0, 1.0, -TINY, np.inf, -np.inf, np.nan]
    Zs = np.array([complex(a, b) for a in specials for b in specials])
    Ms, Rs = modulus_up(Zs), np.abs(Zs)
    ok4 = bool(np.array_equal(np.isinf(Ms), np.isinf(Rs)) and np.array_equal(np.isnan(Ms), np.isnan(Rs)))
    real = rng.standard_normal(1000) * np.exp2(rng.integers(-1074, 1000, 1000))
    ok4 &= bool(np.array_equal(modulus_up(real), np.abs(real)))
    out["inf_nan_real_input_ok"] = ok4
    viol += not ok4
    # negative control (a): np.abs on the three arguments (platform dependent; reported only)
    below, outside = 0, 0
    signed = []
    for z in EXAMPLES:
        a = Fraction(float(np.abs(np.complex128(z))))
        r2 = sq(z)
        below += a * a < r2
        e_lo, e_hi = (1 - 2 * u) ** 2 * r2, (1 + 2 * u) ** 2 * r2
        outside += not (e_lo <= a * a <= e_hi)
        signed.append(round(float((a * a / r2 - 1) / (2 * u)), 3))
    out["negative_control_np_abs"] = {"below_|z|": f"{below}/3", "outside_2u": f"{outside}/3",
                                      "relative_error_over_u": signed}
    # negative control (b): broken variants must fail (i) somewhere
    Zr = random_cases(np.random.default_rng(seed + 1), 20000)
    Za = adversarial_cases()
    det = {}
    for label, kw, Z in (("no_guard", {"guard": False}, Zr), ("no_subnormal_step", {"step": False}, Za),
                         ("step_only_below_2^-1030", {"step_below": 2.0 ** -1030}, Za)):
        Mb = _broken(Z, **kw)
        det[label] = f"{sum(not check_one(z, m)[0] for z, m in zip(Z.tolist(), Mb.tolist()))}/{Z.size}"
    out["negative_control_broken_variants_detected"] = det
    viol += sum(d.startswith("0/") for d in det.values())          # an undetected broken variant fails the run
    # informational: P15 Lemma A (correlations.modulus_upper) on the adversarial cases
    p15_bad, p15_inf, p15_worst = 0, 0, 0.0
    causes = {"square": 0, "sum_of_squares": 0, "guard_of_root": 0, "abs_sum_branch": 0}
    for z in Za.tolist():
        m = modulus_upper(z)
        if m == np.inf:
            p15_inf += 1
            x, y = np.float64(abs(z.real)), np.float64(abs(z.imag))
            with np.errstate(over="ignore"):
                if min(x, y) < 2.0 ** -500:
                    causes["abs_sum_branch"] += 1                  # up(|x| + |y|) overflows
                elif not (np.isfinite(x * x) and np.isfinite(y * y)):
                    causes["square"] += 1
                elif not np.isfinite(x * x + y * y):
                    causes["sum_of_squares"] += 1
                else:
                    causes["guard_of_root"] += 1
            continue
        r2 = sq(z)
        p15_bad += Fraction(float(m)) ** 2 < r2
        if r2 > 0:
            p15_worst = max(p15_worst, float(Fraction(float(m)) ** 2 / r2))
    out["p15_modulus_upper_on_adversarial"] = {"cases": int(Za.size), "below_|z|": p15_bad, "inf": p15_inf,
                                               "inf_by_cause": causes, "largest_ratio": round(p15_worst ** 0.5, 6)}
    viol += p15_bad
    # cross-check: with both components in [2^-500, 2^500] the two functions perform the same five operations
    Zc = Zr[(np.minimum(np.abs(Zr.real), np.abs(Zr.imag)) >= 2.0 ** -500)
            & (np.maximum(np.abs(Zr.real), np.abs(Zr.imag)) <= 2.0 ** 500)]
    Mc = modulus_up(Zc)
    same = sum(float(m) == modulus_upper(z) for z, m in zip(Zc.tolist(), Mc.tolist()))
    out["p15_bitwise_agreement_common_range"] = f"{same}/{Zc.size}"           # second draw (seed + 1)
    viol += Zc.size - same
    out["violations"] = viol
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    r = run(a.cases, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    print("RESULT:", "0 violations" if r["violations"] == 0 else "VIOLATIONS FOUND")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
