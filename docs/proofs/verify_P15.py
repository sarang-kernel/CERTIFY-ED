"""Computational verification of Proposition 15 (docs/proofs/P15_summed_bounds.md).

Part A (the statement with its floating-point evaluation): exact Gaussian-rational coefficients a_k and
certified values v_k with bounds b_k; exact 'true' traces t_k = v_k + d_k with |d_k| <= b_k (rational, the
adversary saturates |d_k| = b_k in the direction of conj(a_k) where that is rational). The output of
certify_ed.correlations.combine_certified, (val, bound), must satisfy |sum a_k t_k - val| <= bound in exact
arithmetic (compared through squared moduli).
Part B (Lemma A): modulus_upper(z) >= |z| exactly, for random and boundary z.
Part C (Lemma B): |fl(sum a_k v_k) - sum a_k v_k| <= gamma_{m+2} sum |a_k||v_k| exactly, including
cancellation-heavy sums and m up to 20000.

Usage: python docs/proofs/verify_P15.py [--random N] [--seed S]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from certify_ed.correlations import combine_certified, modulus_upper  # noqa: E402

PYTHAGOREAN = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (1, 0, 1), (0, 1, 1)]


class Q2:
    """Exact complex rational."""
    __slots__ = ("re", "im")

    def __init__(self, re, im=0):
        self.re, self.im = Fraction(re), Fraction(im)

    @classmethod
    def of(cls, z: complex):
        return cls(Fraction(float(z.real)), Fraction(float(z.imag)))

    def __add__(self, o):
        return Q2(self.re + o.re, self.im + o.im)

    def __sub__(self, o):
        return Q2(self.re - o.re, self.im - o.im)

    def __mul__(self, o):
        return Q2(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    def abs2(self) -> Fraction:
        return self.re * self.re + self.im * self.im


def _gamma_exact(n: int) -> Fraction:
    u = Fraction(1, 2 ** 53)
    return n * u / (1 - n * u)


def _rand_coeff(rng: random.Random, kind: str) -> Q2:
    if kind == "pythagorean":
        p, q, r = rng.choice(PYTHAGOREAN)
        s = Fraction(rng.choice([1, -1]) * rng.randrange(1, 100), rng.randrange(1, 100))
        return Q2(s * p / r * rng.choice([1, -1]), s * q / r * rng.choice([1, -1]))
    if kind == "tiny":
        return Q2(Fraction(rng.randrange(-9, 10), 2 ** 520), Fraction(rng.randrange(-9, 10), 2 ** 520))
    return Q2(Fraction(rng.uniform(-1, 1)), Fraction(rng.uniform(-1, 1)))


def _to_complex(q: Q2) -> complex:
    return complex(float(q.re), float(q.im))


def case_A(rng: random.Random, m: int, kind: str, saturate: bool):
    a = [_rand_coeff(rng, kind) for _ in range(m)]
    v = [Q2(Fraction(rng.uniform(-3, 3)), Fraction(rng.uniform(-3, 3)) if rng.random() < 0.5 else 0) for _ in range(m)]
    b = [Fraction(abs(rng.gauss(0, 1)) * 10.0 ** rng.randrange(-16, 0)) for _ in range(m)]
    # the floating-point inputs are exactly these rationals (every float is a dyadic rational)
    af = [_to_complex(x) for x in a]
    vf = [_to_complex(x) for x in v]
    bf = [float(x) for x in b]
    a = [Q2.of(x) for x in af]
    v = [Q2.of(x) for x in vf]
    b = [Fraction(x) for x in bf]
    # adversarial deviations: |d_k| <= b_k; saturating in the direction of conj(a_k) when |a_k| is rational
    d = []
    for ak, bk in zip(a, b):
        r2 = ak.abs2()
        rt = Fraction(math.isqrt(r2.numerator), math.isqrt(r2.denominator))
        if saturate and rt * rt == r2 and rt > 0:
            d.append(Q2(bk * ak.re / rt, -bk * ak.im / rt))        # |d| = b exactly, phase of conj(a)
        else:
            p, q, r = rng.choice(PYTHAGOREAN)
            t = Fraction(rng.randrange(0, 1001), 1000)
            d.append(Q2(bk * t * p / r, bk * t * q / r))              # |d| = t b <= b exactly
    val, bound = combine_certified(af, vf, bf)
    exact = Q2(0)
    for ak, vk, dk in zip(a, v, d):
        exact = exact + ak * (vk + dk)
    delta = exact - Q2.of(val)
    return delta.abs2() <= Fraction(bound) ** 2


def check_A(rng: random.Random, n: int):
    cases = viol = 0
    for _ in range(n):
        m = rng.choice([1, 2, 3, 8, 16, 64, 256, 1000])
        kind = rng.choice(["uniform", "uniform", "pythagorean", "tiny"])
        cases += 1
        viol += not case_A(rng, m, kind, saturate=rng.random() < 0.5)
    return cases, viol


def adversarial_A(rng: random.Random):
    """m = 1 with saturation; cancellation (sum a_k v_k = 0 exactly while sum |a_k||v_k| is large); all-tiny
    coefficients; large m; bounds 0; a value exactly 0."""
    ok = []
    ok.append(case_A(rng, 1, "pythagorean", True))
    # cancellation: a = (1, -1), v = (x, x)
    for x in (1.0, 1e10, 1e-10, 3.7):
        val, bound = combine_certified([1.0, -1.0], [x, x], [0.0, 0.0])
        ok.append(val == 0 and bound >= 0)
        val, bound = combine_certified([1.0 + 1j, 1.0 - 1j, -2.0], [x, x, x], [0.0, 0.0, 0.0])
        ok.append(abs(val) <= bound)
    ok.append(case_A(rng, 5000, "uniform", True))
    ok.append(case_A(rng, 64, "tiny", True))
    val, bound = combine_certified([], [], [])
    ok.append(val == 0 and 0 <= bound <= 5e-324)        # add_up(0, 0) is one ulp above 0: a valid upper bound
    # the per-state division b/g rounded up (certify_correlations): up(b/g) >= b/g exactly
    from certify_ed.fp import up
    for b_ in (1.0, 1e-17, 0.3, 7.0):
        for g in (1, 2, 3, 7, 1000):
            ok.append(Fraction(float(up(b_ / g))) >= Fraction(b_) / g)
    return len(ok), sum(1 for x in ok if not x)


def check_B(rng: random.Random, n: int):
    cases = viol = 0
    worst_ratio = 1.0
    zs = []
    for _ in range(n):
        e = rng.randrange(-520, 300)
        zs.append(complex(rng.uniform(-1, 1) * 2.0 ** e, rng.uniform(-1, 1) * 2.0 ** rng.randrange(-520, 300)))
    zs += [complex(2.0 ** -500, 2.0 ** -500), complex(2.0 ** -501, 1.0), complex(1.0, 2.0 ** -501), complex(3.0, 4.0),
           complex(1.0, 1.0), complex(0.0, 1e-300), complex(1e-300, 0.0), complex(1e153, 1e153), complex(5e-324, 1.0),
           complex(0.6, 0.8), complex(1 / 3, 1 / 7)]
    for z in zs:
        cases += 1
        mz = modulus_upper(z)
        exact2 = Fraction(z.real) ** 2 + Fraction(z.imag) ** 2
        ok = math.isfinite(mz) and Fraction(mz) ** 2 >= exact2
        viol += not ok
        if ok and exact2 > 0 and min(abs(z.real), abs(z.imag)) >= 2.0 ** -500:
            worst_ratio = max(worst_ratio, float(Fraction(mz) ** 2 / exact2))
    return cases, viol, worst_ratio


def check_C(rng: random.Random, n: int):
    """Lemma B: the rounding of numpy's complex dot product against gamma_{m+2} sum |a||v| (exact)."""
    cases = viol = 0
    worst = 0.0
    for _ in range(n):
        m = rng.choice([1, 2, 7, 100, 1000, 20000])
        a = np.array([complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(m)])
        v = np.array([complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(m)])
        if rng.random() < 0.3 and m % 2 == 0:
            v[m // 2:] = -v[:m // 2]            # heavy cancellation
            a[m // 2:] = a[:m // 2]
        val = complex(np.sum(a * v))
        exact = Q2(0)
        s_av = Fraction(0)
        for ak, vk in zip(a.tolist(), v.tolist()):
            qa, qv = Q2.of(ak), Q2.of(vk)
            exact = exact + qa * qv
            s_av += Fraction(math.sqrt(float(qa.abs2() * qv.abs2())))     # only for the ratio report
        err2 = (exact - Q2.of(val)).abs2()
        # exact upper bound on sum |a||v| through the modulus_upper route (rigorous), used for the check
        s_up = sum(Fraction(modulus_upper(ak)) * Fraction(modulus_upper(vk)) for ak, vk in zip(a.tolist(), v.tolist()))
        gm = _gamma_exact(m + 2)
        cases += 1
        ok = err2 <= (gm * s_up) ** 2
        viol += not ok
        if s_av > 0:
            worst = max(worst, math.sqrt(float(err2)) / float(gm * s_av))
    return cases, viol, worst


def run(n_random: int = 500, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = {}
    out["A_random"] = check_A(rng, n_random)
    out["A_adversarial"] = adversarial_A(rng)
    cb, vb, ratio = check_B(rng, n_random)
    out["B_modulus"] = (cb, vb)
    out["B_worst_ratio_minus_1"] = ratio - 1.0
    cc, vc, worst = check_C(rng, max(50, n_random // 5))
    out["C_rounding"] = (cc, vc)
    out["C_worst_error_over_bound"] = worst
    out["random_cases"] = out["A_random"][0] + cb + cc
    out["adversarial_cases"] = out["A_adversarial"][0]
    out["violations"] = out["A_random"][1] + out["A_adversarial"][1] + vb + vc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.random, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
