"""Independent adversarial search for Proposition 15 (docs/proofs/P15_summed_bounds.md, correlations.py).

Exact ground truth: fractions.Fraction components for a_k, v_k (binary64 pairs), exact sum a_k v_k, exact
|a_k| b_k via mpmath (60 digits) for the square roots. Checks: Lemma A (modulus_upper >= |z|) on components at
the 2^-500 threshold, subnormal components, huge components (inf allowed), one zero component; Lemma B (the
actual rounding of numpy's fl(sum a v) <= gamma_{m+2} sum |a||v|) for m from 1 to 20000 with cancellation, random
scales and complex v; (2): B >= sum |a_k| b_k + |val - exact sum| for the same data (with tr(P O_k) replaced by
any point within b_k of v_k: the worst case is the triangle inequality, checked at the extreme points).

Usage: python docs/proofs/independent/adversarial_P15_summed_bounds.py [--cases 300] [--seed 41]
"""
import argparse
import math
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, Tally, mpf, report  # noqa: E402
from certify_ed.correlations import combine_certified, modulus_upper  # noqa: E402
from certify_ed.fp import U, gamma  # noqa: E402

mp.mp.dps = 60


def mabs(z):
    z = complex(z)
    return mp.sqrt(mpf(z.real) ** 2 + mpf(z.imag) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_A = Tally("Lemma A: modulus_upper(z) >= |z| (threshold 2^-500, subnormal, huge, zero components)")
    T_B = Tally("Lemma B: |fl(sum a v) - exact| <= gamma_{m+2} sum |a||v|")
    T_2 = Tally("(2): B >= sum |a_k| b_k + |val - exact sum| (extreme points of the Theorem 6 boxes)")
    # Lemma A
    for i in range(a.cases * 3):
        kind = i % 6
        if kind == 0:
            z = complex(rng.standard_normal(), rng.standard_normal()) * 10.0 ** float(rng.integers(-300, 300))
        elif kind == 1:
            z = complex(2.0 ** -500 * float(rng.choice([0.999, 1.0, 1.001])), rng.standard_normal())
        elif kind == 2:
            z = complex(rng.standard_normal() * 2.0 ** -1060, rng.standard_normal() * 2.0 ** -1060)
        elif kind == 3:
            z = complex(rng.standard_normal() * 1e200, rng.standard_normal() * 1e200)
        elif kind == 4:
            z = complex(rng.standard_normal(), 0.0) if i % 2 else complex(0.0, rng.standard_normal())
        else:
            z = complex(1.0 + 2 * U * int(rng.integers(0, 2 ** 40)), 2.0 ** -26)       # modulus just below a rounding midpoint
        m_ = modulus_upper(z)
        T_A.add(math.isinf(m_) or mpf(m_) >= mabs(z), kind != 0, f"modulus_upper({z!r}) = {m_} < |z|")
    # Lemma B and (2)
    for i in range(a.cases):
        kind = i % 6
        m = {0: int(rng.integers(1, 6)), 1: 1, 2: 2000, 3: 20000, 4: int(rng.integers(2, 50)), 5: int(rng.integers(2, 50))}[kind]
        scale = 10.0 ** float(rng.integers(-100, 100)) if kind == 4 else 1.0
        av = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        vv = (rng.standard_normal(m) + (1j * rng.standard_normal(m) if i % 2 else 0.0)) * scale
        if kind == 5:   # cancellation: pairs of opposite terms plus one small term
            half = m // 2
            av[half:2 * half] = av[:half]; vv[half:2 * half] = -vv[:half]
        bb = np.abs(rng.standard_normal(m)) * scale * 1e-3
        val, B = combine_certified(av, vv, bb)
        # exact sum and the sum of moduli
        ex_re = sum(F(x.real) * F(y.real) - F(x.imag) * F(y.imag) for x, y in zip(av, vv))
        ex_im = sum(F(x.real) * F(y.imag) + F(x.imag) * F(y.real) for x, y in zip(av, vv))
        exact = mp.mpc(mp.mpf(ex_re.numerator) / ex_re.denominator, mp.mpf(ex_im.numerator) / ex_im.denominator)
        S = sum(mabs(x) * mabs(y) for x, y in zip(av, vv))
        err = abs(mp.mpc(mpf(val.real), mpf(val.imag)) - exact)
        T_B.add(err <= mpf(gamma(m + 2)) * S, kind != 0, f"m={m} kind={kind}: err={mp.nstr(err, 8)} bound={mp.nstr(mpf(gamma(m + 2)) * S, 8)}")
        sum_ab = sum(mabs(x) * mpf(b_) for x, b_ in zip(av, bb))
        T_2.add(mpf(B) >= sum_ab + err, kind != 0, f"m={m} kind={kind}: B={B} needed={mp.nstr(sum_ab + err, 12)}")
    # m = 0
    val0, B0 = combine_certified([], [], [])
    T_2.add(val0 == 0 and 0.0 <= B0 <= 2.0 ** -1073, True, f"m = 0: {val0}, {B0}")   # add_up(0, 0) = one ulp above 0
    sys.exit(1 if report([T_A, T_B, T_2]) else 0)


if __name__ == "__main__":
    main()
