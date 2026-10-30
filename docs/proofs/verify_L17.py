"""Computational verification of Lemma 17 (docs/proofs/L17_dyadic_exp_ln.md).

Ground truth: exact rational arithmetic (fractions.Fraction) for the grid operations and mpmath at 200
digits for exp and ln. Every enclosure produced by certify_ed.ivl must contain the exact value; the script
counts cases and violations.

Usage: python docs/proofs/verify_L17.py [--random N] [--seed S]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import mpmath as mp  # noqa: E402

from certify_ed import ivl  # noqa: E402

ONE = ivl.ONE
PREC = ivl.PREC


def _mpf(q: Fraction):
    return mp.mpf(q.numerator) / mp.mpf(q.denominator)


def _contains_frac(iv: ivl.Interval, q: Fraction) -> bool:
    lo, hi = iv.fractions()
    return lo <= q <= hi


def _rand_fraction(rng: random.Random, mag: int = 60) -> Fraction:
    d = rng.randrange(1, 1 << rng.randrange(1, mag))
    n = rng.randrange(-(1 << mag), 1 << mag)
    return Fraction(n, d)


def _rand_interval(rng: random.Random) -> ivl.Interval:
    a, b = sorted([_rand_fraction(rng), _rand_fraction(rng)])
    ia, ib = ivl.from_fraction(a), ivl.from_fraction(b)
    return ivl.Interval(ia.lo, ib.hi)


def _member(rng: random.Random, iv: ivl.Interval) -> Fraction:
    """A rational point of the interval (ends included with positive probability)."""
    lo, hi = iv.fractions()
    r = rng.random()
    if r < 0.1:
        return lo
    if r < 0.2:
        return hi
    t = Fraction(rng.randrange(0, 1 << 40), 1 << 40)
    return lo + (hi - lo) * t


def check_grid_ops(rng: random.Random, n: int):
    """(i)-(ii): +, -, neg, *, scale_int, /, square, intersect_nonneg, from_fraction, to_float_down/up,
    sum_intervals: the result contains the exact operation on members of the operands."""
    cases = viol = 0
    for _ in range(n):
        X, Y = _rand_interval(rng), _rand_interval(rng)
        x, y = _member(rng, X), _member(rng, Y)
        cases += 1
        ok = _contains_frac(X + Y, x + y) and _contains_frac(X - Y, x - y) and _contains_frac(-X, -x)
        ok &= _contains_frac(X * Y, x * y) and _contains_frac(X.square(), x * x)
        c = rng.randrange(-1000, 1000)
        ok &= _contains_frac(X.scale_int(c), x * c)
        if not (Y.lo <= 0 <= Y.hi):
            ok &= _contains_frac(X / Y, x / y)
        q = _rand_fraction(rng)
        fq = ivl.from_fraction(q)
        ok &= _contains_frac(fq, q) and (fq.hi - fq.lo) <= 1
        lo, hi = X.fractions()
        fd, fu = ivl.to_float_down(X.lo), ivl.to_float_up(X.hi)
        ok &= Fraction(fd) <= lo and Fraction(fu) >= hi
        ok &= math.nextafter(fd, math.inf) > lo and math.nextafter(fu, -math.inf) < hi    # tightest floats
        ok &= _contains_frac(ivl.sum_intervals([X, Y, X]), 2 * x + y)
        if X.hi >= 0:
            ok &= _contains_frac(X.intersect_nonneg(), max(x, Fraction(0)))
        viol += not ok
    return cases, viol


def check_exp(rng: random.Random, n: int, xs=None):
    mp.mp.dps = 200
    cases = viol = 0
    min_slack = mp.inf
    if xs is None:
        xs = []
        for _ in range(n):
            r = rng.random()
            if r < 0.3:
                xs.append(Fraction(rng.uniform(-700, 700)))
            elif r < 0.6:
                xs.append(Fraction(rng.uniform(-2, 2)))
            elif r < 0.8:
                xs.append(Fraction(rng.randrange(-(1 << 70), 1 << 70), 1 << rng.randrange(60, 80)))
            else:
                xs.append(Fraction(rng.randrange(-(1 << 200), 1 << 200), ONE))      # arbitrary grid points
    for x in xs:
        cases += 1
        ex = mp.exp(_mpf(x))
        try:
            iv = ivl.exp_interval(ivl.from_fraction(x))
        except OverflowError:
            viol += not (x > 1 << 16)          # refused only above the documented argument limit
            continue
        lo, hi = iv.fractions()
        ok = _mpf(lo) <= ex <= _mpf(hi) and iv.lo >= 0
        if ok and ex > 0:
            min_slack = min(min_slack, (ex - _mpf(lo)) / ex, (_mpf(hi) - ex) / ex)
        # monotone interval version: a random interval containing x
        w = Fraction(rng.randrange(0, 1 << 30), 1 << 60)
        J = ivl.Interval(ivl.from_fraction(x - w).lo, ivl.from_fraction(x + w).hi)
        try:
            ej = ivl.exp_interval(J)
            ok &= _mpf(ej.fractions()[0]) <= ex <= _mpf(ej.fractions()[1])
        except OverflowError:
            ok &= x + w > 1 << 16
        viol += not ok
    return cases, viol, min_slack


def check_ln(rng: random.Random, n: int, zs=None):
    mp.mp.dps = 200
    cases = viol = 0
    max_width = mp.mpf(0)
    if zs is None:
        zs = []
        for _ in range(n):
            r = rng.random()
            if r < 0.3:
                zs.append(Fraction(math.exp(rng.uniform(-30, 300))))
            elif r < 0.6:
                zs.append(Fraction(rng.uniform(0.5, 4.0)))
            elif r < 0.8:
                zs.append(Fraction(rng.randrange(1, 1 << 90), 1 << rng.randrange(0, 90)))
            else:
                zs.append(Fraction(rng.randrange(1, 1 << 600), ONE))
    for z in zs:
        cases += 1
        Z = ivl.from_fraction(z)
        iv = ivl.ln_interval(Z)
        lz = mp.log(_mpf(z))
        lo, hi = iv.fractions()
        ok = _mpf(lo) <= lz <= _mpf(hi)
        if ok:
            max_width = max(max_width, _mpf(hi - lo))
        viol += not ok
    return cases, viol, max_width


def adversarial_cases():
    """Points at the hypothesis boundaries of Lemma 17 (iii) and (iv): x = 0, |x| = 1/2 and its grid
    neighbours (halving count changes), odd grid points with many inexact halvings, x near the binary64
    overflow threshold, |x| = 1000 and its grid neighbours (the underflow shortcut), x = 2^16 and one unit above
    (the argument limit); z = 1, powers of two and their grid neighbours (m = 1, m -> 2, u -> 1/3), inexact
    shifts (k > 0 with low bits set), very large and very small z."""
    xs = [Fraction(0), Fraction(1, 2), Fraction(ONE // 2 + 1, ONE), Fraction(ONE // 2 - 1, ONE), Fraction(1),
          Fraction(1, ONE), Fraction(-1, ONE), Fraction(-1, 2), Fraction(-1), Fraction(700), Fraction(-700),
          Fraction(745), Fraction(-745), Fraction((1 << 10) + 1, 1), Fraction(3 * ONE // 4 + 1, ONE),
          Fraction(1000), Fraction(-1000), Fraction(-1000 * ONE + 1, ONE), Fraction(-1000 * ONE - 1, ONE),
          Fraction(-5000), Fraction(-10 ** 6), Fraction(1 << 16), Fraction((1 << 16) * ONE + 1, ONE)]
    for j in range(0, 12):
        xs += [Fraction((1 << j) * ONE + 1, ONE), Fraction((1 << j) * ONE - 1, ONE), Fraction(1, 1 << j)]
    for j in range(1, 8):
        xs.append(Fraction((3 ** j) * (ONE >> 1) + 1, ONE))
    zs = [Fraction(1), Fraction(2), Fraction(1, 2), Fraction(2 * ONE - 1, ONE), Fraction(2 * ONE + 1, ONE),
          Fraction(ONE + 1, ONE), Fraction(ONE - 1, ONE), Fraction(1, ONE), Fraction(3), Fraction(1, 3),
          Fraction(2 ** 2000), Fraction(1, 2 ** 150), Fraction((1 << (PREC + 5)) + 1, ONE),
          Fraction((1 << (PREC + 5)) - 1, ONE), Fraction((1 << (PREC + 700)) + 12345, ONE),
          Fraction(4 * ONE - 1, ONE), Fraction(4 * ONE - 2, ONE)]
    for j in range(1, 20):
        zs += [Fraction((2 ** j) * ONE - 1, ONE), Fraction(2 ** j), Fraction(2 ** j * ONE + 1, ONE)]
    return xs, zs


def check_remainder_constants():
    """The two remainder constants of the proof, in exact arithmetic: 1/(1 - t/(N+2)) <= 2 for t <= 1/2 and
    1/(1 - u^2) <= 8/7 for u <= 0.35; and that the series lengths put the remainders below one grid unit at
    the largest arguments (tightness, not needed for correctness)."""
    N = ivl._EXP_TERMS
    t = Fraction(1, 2)
    c1 = 1 / (1 - t / (N + 2)) <= 2
    u = Fraction(7, 20)
    c2 = 1 / (1 - u * u) <= Fraction(8, 7)
    c3 = ivl._U_MAX >= (ONE + 2) // 3           # ceil(ONE/3) <= _U_MAX: the upper argument is always admitted
    rem_exp = t ** (N + 1) / math.factorial(N + 1) * 2
    M = ivl._LN_TERMS
    rem_ln = 2 * u ** (2 * M + 3) / ((2 * M + 3) * (1 - u * u))
    return c1 and c2 and c3, rem_exp < Fraction(1, ONE), rem_ln < Fraction(1, ONE)


def run(n_random: int = 2000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = {}
    out["grid_ops"] = check_grid_ops(rng, n_random)
    c, v, s = check_exp(rng, n_random)
    out["exp_random"] = (c, v)
    out["exp_min_rel_slack"] = float(s)
    c, v, s = check_ln(rng, n_random)
    out["ln_random"] = (c, v)
    out["ln_max_width"] = float(s)
    xs, zs = adversarial_cases()
    c1, v1, _ = check_exp(rng, 0, xs)
    c2, v2, _ = check_ln(rng, 0, zs)
    out["exp_adversarial"] = (c1, v1)
    out["ln_adversarial"] = (c2, v2)
    out["constants_ok"], out["exp_remainder_below_grid"], out["ln_remainder_below_grid"] = check_remainder_constants()
    out["random_cases"] = out["grid_ops"][0] + out["exp_random"][0] + out["ln_random"][0]
    out["adversarial_cases"] = c1 + c2
    out["violations"] = (out["grid_ops"][1] + out["exp_random"][1] + out["ln_random"][1] + v1 + v2
                         + (0 if out["constants_ok"] else 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.random, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
