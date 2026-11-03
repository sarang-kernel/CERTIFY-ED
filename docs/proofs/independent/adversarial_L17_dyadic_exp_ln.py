"""Independent adversarial search for Lemma 17 (docs/proofs/L17_dyadic_exp_ln.md, certify_ed/ivl.py).

Ground truth: fractions.Fraction for the rational operations and mpmath at 150 digits (well beyond the 192-bit
grid, 2^-192 ~ 1.6e-58) for exp and ln. Every output interval is checked to contain the exact value (exp, ln via
mp with a certified rounding margin: the mp value is compared with the grid ends shifted by 10^-120).
Targets: exp at 0, at the halving boundaries (t = ONE/2, ONE/2 + 1, odd arguments through several halvings),
at 1 unit, at the maximal argument 2^16, just below/above -1000, at huge negative arguments, random arguments
up to 2^16 with random bit patterns; ln at 1, at exact powers of two (positive and negative k), at
v = ((2 ONE - 1) << k) + 1 (m_hi = 2 ONE, u_hi = ceil(ONE/3): the series-argument boundary), at the smallest
grid point 1, at random v spanning 2^-192 .. 2^600; arithmetic with mixed signs, division by intervals touching
but not containing zero, squares straddling zero; float conversions at binade boundaries and halfway points.

Usage: python docs/proofs/independent/adversarial_L17_dyadic_exp_ln.py [--cases 300] [--seed 29]
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
from certify_ed.ivl import ONE, PREC, Interval, exp_interval, from_fraction, ln_interval, to_float_down, to_float_up  # noqa: E402

mp.mp.dps = 150
sys.set_int_max_str_digits(0)
MARGIN = mp.mpf("1e-120")


def q(v):
    return Fraction(v, ONE)


def mpq(fr):
    return mp.mpf(fr.numerator) / mp.mpf(fr.denominator)


def contains_mp(iv, value):
    return mpq(q(iv.lo)) - MARGIN <= value <= mpq(q(iv.hi)) + MARGIN


def check_exp(x, T, tag):
    iv = exp_interval(Interval(x, x))
    val = mp.exp(mpq(q(x)))
    ok = contains_mp(iv, val) and iv.lo <= iv.hi
    width = mpq(q(iv.hi - iv.lo)) / val
    T.add(ok, True, f"exp {tag}: x~2^{x.bit_length()} iv.lo~2^{iv.lo.bit_length()} relwidth={mp.nstr(width, 5)}")
    return width


def check_ln(v, T, tag):
    iv = ln_interval(Interval(v, v))
    val = mp.log(mpq(q(v)))
    ok = contains_mp(iv, val) and iv.lo <= iv.hi
    T.add(ok, True, f"ln {tag}: v~2^{v.bit_length()} exact={mp.nstr(val, 30)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=29)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_ar = Tally("(i)-(ii) arithmetic encloses the exact rational result (corners, signs, division, square)")
    T_exp = Tally("(iii) exp encloses the exact value (mp 150 digits)")
    T_ln = Tally("(iv) ln encloses the exact value (mp 150 digits)")
    T_fl = Tally("(v) to_float_down/up are the adjacent binary64 numbers")
    T_w = Tally("tightness: relative width of exp/ln enclosures below 2^-150 for |x| <= 1")
    # (i)-(ii)
    def rand_iv():
        c = Fraction(int(rng.integers(-10 ** 6, 10 ** 6)), int(rng.integers(1, 10 ** 4))) * Fraction(2) ** int(rng.integers(-200, 60))
        w = Fraction(int(rng.integers(0, 10 ** 3)), int(rng.integers(1, 10 ** 3))) * Fraction(2) ** int(rng.integers(-200, 10))
        X = from_fraction(c) + from_fraction(c + w) - from_fraction(c)
        return X
    for i in range(a.cases):
        X, Y = rand_iv(), rand_iv()
        if i % 5 == 0:
            Y = Interval(1, Y.hi - Y.lo + 1)            # touches zero from above but excludes it
        if i % 7 == 0:
            X = Interval(-abs(X.lo), abs(X.hi))        # straddles zero
        xs = [q(X.lo), q(X.hi), (q(X.lo) + q(X.hi)) / 2, q(X.lo) + Fraction(1, 7) * (q(X.hi) - q(X.lo))]
        ys = [q(Y.lo), q(Y.hi), (q(Y.lo) * 2 + q(Y.hi)) / 3]
        k_ = int(rng.integers(-5, 6))
        P = X * Y; S = X.square(); A_ = X + Y; D_ = X - Y; N_ = -X; C = X.scale_int(k_)
        ok = all(P.contains(x * y) for x in xs for y in ys) and all(S.contains(x * x) for x in xs)
        ok = ok and all(A_.contains(x + y) and D_.contains(x - y) for x in xs for y in ys) and all(N_.contains(-x) for x in xs)
        ok = ok and all(C.contains(x * k_) for x in xs)
        if not (Y.lo <= 0 <= Y.hi):
            Q = X / Y
            ok = ok and all(Q.contains(x / y) for x in xs for y in ys)
        if X.hi >= 0:
            ok = ok and all(X.intersect_nonneg().contains(x) for x in xs if x >= 0)
        T_ar.add(ok, i % 5 == 0 or i % 7 == 0, f"arith case {i}")
    # (iii) exp
    half = ONE >> 1
    specials = [0, 1, 2, half - 1, half, half + 1, half + 2, ONE - 1, ONE, ONE + 1, 2 * ONE + 1, 3 * ONE - 1, 7 * ONE + 12345,
                ONE << 16, (ONE << 16) - 1, -1, -half, -ONE, -(999 * ONE + ONE - 1), -1000 * ONE + 1, -1000 * ONE, -1000 * ONE - 1,
                -(ONE << 16), -(ONE << 40)]
    for x in specials:
        w = check_exp(x, T_exp, "special")
        if abs(x) <= ONE and x != 0:
            T_w.add(w < mp.mpf(2) ** -150, True, f"exp width x={x} relwidth={mp.nstr(w, 5)}")
    for i in range(a.cases):
        kind = i % 4
        if kind == 0:
            x = int(rng.integers(0, 2 ** 62)) * (1 << int(rng.integers(0, PREC - 62)))      # random bit pattern <= ONE-ish
            x = x % (ONE << 16)
        elif kind == 1:
            x = (ONE >> int(rng.integers(0, 60))) + int(rng.integers(-3, 4))               # near halving boundaries
        elif kind == 2:
            x = -((int(rng.integers(0, 2 ** 62)) << int(rng.integers(0, PREC - 52))) % (1000 * ONE))   # negative, series branch
        else:
            x = int(rng.integers(0, 2 ** 60)) << int(rng.integers(0, 30))                  # odd-ish, several halvings
            x = x % (ONE << 16)
        w = check_exp(x, T_exp, f"kind{kind}")
        if 0 < abs(x) <= ONE:
            T_w.add(w < mp.mpf(2) ** -150, False, f"exp width x={x} relwidth={mp.nstr(w, 5)}")
    try:
        exp_interval(Interval((ONE << 16) + 1, (ONE << 16) + 1)); T_exp.add(False, True, "no OverflowError above 2^16")
    except OverflowError:
        T_exp.add(True, True, "")
    # (iv) ln
    for v in [1, 2, 3, ONE - 1, ONE, ONE + 1, 2 * ONE - 1, 2 * ONE, 2 * ONE + 1, 3 * ONE, ONE << 100, (ONE << 100) + 1, (ONE << 100) - 1,
              ONE >> 100, (ONE >> 100) + 1, ((2 * ONE - 1) << 1) + 1, ((2 * ONE - 1) << 37) + 1, ((2 * ONE - 1) << 600) + 12345,
              (ONE << 600) - 1]:
        check_ln(v, T_ln, "special")
    for i in range(a.cases):
        kind = i % 3
        if kind == 0:
            v = int(rng.integers(1, 2 ** 62)) << int(rng.integers(0, 800))
        elif kind == 1:
            v = ((2 * ONE - 1) << int(rng.integers(1, 400))) + int(rng.integers(1, 2 ** 40))   # m_hi = 2 ONE family
        else:
            v = int(rng.integers(1, 2 ** 62))                                                   # tiny (k very negative)
        check_ln(v, T_ln, f"kind{kind}")
        iv = ln_interval(Interval(v, v)); val = mp.log(mpq(q(v)))
        if abs(val) >= 1 and kind == 0:
            T_w.add(mpq(q(iv.hi - iv.lo)) / abs(val) < mp.mpf(2) ** -150, False, f"ln width v={v}")
    # (v) float conversion
    for i in range(a.cases):
        kind = i % 4
        if kind == 0:
            f = float(rng.standard_normal()) * 2.0 ** int(rng.integers(-1000, 1000)); fr = Fraction(f)
        elif kind == 1:   # halfway between two binary64 numbers
            f = float(rng.standard_normal()); fr = (Fraction(f) + Fraction(math.nextafter(f, math.inf))) / 2
        elif kind == 2:   # just above a power of two
            fr = Fraction(2) ** int(rng.integers(-1000, 1000)) + Fraction(1, 2 ** 1100)
        else:
            fr = Fraction(int(rng.integers(1, 10 ** 9)), int(rng.integers(1, 10 ** 9)))
        iv = from_fraction(fr)
        vlo, vhi = iv.lo, iv.hi
        for v in (vlo, vhi):
            qv = q(v)
            d, u_ = to_float_down(v), to_float_up(v)
            ok = Fraction(d) <= qv <= Fraction(u_) and (d == u_ or math.nextafter(d, math.inf) == u_ or Fraction(d) == qv or Fraction(u_) == qv)
            ok = ok and (Fraction(d) == qv or Fraction(math.nextafter(d, math.inf)) > qv) and (Fraction(u_) == qv or Fraction(math.nextafter(u_, -math.inf)) < qv)
            T_fl.add(ok, kind in (1, 2), f"float conv v={v} d={d} u={u_}")
    sys.exit(1 if report([T_ar, T_exp, T_ln, T_fl, T_w]) else 0)


if __name__ == "__main__":
    main()
