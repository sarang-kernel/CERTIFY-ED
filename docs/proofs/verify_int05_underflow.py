"""Computational verification of docs/proofs/int05_underflow.md (the underflow-tolerant model and the additive terms),
in exact rational arithmetic (fractions.Fraction represents every binary64 number, subnormals included, exactly).

M   the model fl(z) = z(1 + delta) + eta on random operands over the whole exponent range (+, -, x, /, sqrt), and
    exactness of sums below 2^-1022
U1  dot products with tiny entries against the exact sum and the bound gamma_k sum|a b| + k (1 + gamma_{k-1}) u_min
U2  exact_dot on tiny, subnormal and mixed-scale vectors (|s - exact| <= err), including the rescaling branch
U2b exact_dot at the top of the range (factors >= 2^995, products >= 2^996, real and complex) and exact_sum with
    overflowing partial sums: |s - exact| <= err and the nearest value, or not finite when the exact value is beyond
U3  norm2_upper >= ||v|| >= norm2_lower for vectors whose squares underflow (incl. 10^4 entries of 2^-540); the scalar inequality of T_N on a grid
U5  residuals on scaled operators and LAPACK-like vectors: rel^2 ||x||^2 >= ||r||^2, upper >= ||r||, E_i + c_w u_min >= |Rc_i - r_i|
U6  radii never shrink against the pre-branch formula, difference <= D, bit identity for normal radii
Usage: python docs/proofs/verify_int05_underflow.py [--cases 20000] [--seed 7]
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from decimal import Decimal, getcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter  # noqa: E402

from certify_ed.enclosure import OperatorAdapter, residuals  # noqa: E402
from certify_ed.fp import (HUGE_FACTOR, HUGE_PRODUCT, U, ULP_MIN, _GUARD, exact_dot, exact_sum, gamma,  # noqa: E402
                           norm2_lower, norm2_upper, norm_underflow_term, residual_additive_term,
                           residual_entry_count)

getcontext().prec = 150
u = Fraction(U)
u_min = Fraction(1, 2 ** 1075)
MIN_NORMAL = Fraction(1, 2 ** 1022)


def F(a):
    return Fraction(float(a))


def gam(m):
    return m * u / (1 - m * u)


def sqrt_up(q):
    if q <= 0:
        return Fraction(0)
    getcontext().rounding = ROUND_CEILING
    return Fraction((Decimal(q.numerator) / Decimal(q.denominator)).sqrt()) * (1 + Fraction(1, 10 ** 120))


def sqrt_down(q):
    if q <= 0:
        return Fraction(0)
    getcontext().rounding = ROUND_FLOOR
    return Fraction((Decimal(q.numerator) / Decimal(q.denominator)).sqrt()) * (1 - Fraction(1, 10 ** 120))


def rand_float(rng, lo=-1080, hi=1020):
    e = int(rng.integers(lo, hi))
    m = float(rng.uniform(1.0, 2.0)) * (1 if rng.random() < 0.5 else -1)
    return math.ldexp(m, e)


def check_model(rng, counters):
    for _ in range(1):
        x, y = rand_float(rng), rand_float(rng)
        op = rng.choice(["+", "-", "*", "/", "sqrt"])
        if op == "+":
            z, f = F(x) + F(y), x + y
        elif op == "-":
            z, f = F(x) - F(y), x - y
        elif op == "*":
            z, f = F(x) * F(y), x * y
        elif op == "/":
            if y == 0:
                return
            z, f = F(x) / F(y), x / y
        else:
            x = abs(x)
            z2 = F(x)
            f = math.sqrt(x)
            # sqrt is irrational in general: check the bracket |f - sqrt(z2)| <= max(u sqrt(z2), u_min) through squares
            fz = F(f)
            lo, hi = sqrt_down(z2), sqrt_up(z2)
            ok = (fz - hi <= max(u * hi, u_min)) and (lo - fz <= max(u * hi, u_min))
            counters["M"].add(False, ok, f"sqrt {x}")
            return
        if not math.isfinite(f):
            return
        err = abs(F(f) - z)
        ok = err <= max(u * abs(z), u_min)
        if op in "+-" and abs(z) < MIN_NORMAL:
            ok &= F(f) == z                                  # an underflowing sum is exact
        if abs(z) >= MIN_NORMAL:
            ok &= err <= u * abs(z)                          # a normal result: relative only
        else:
            ok &= err <= u_min
        counters["M"].add(False, ok, f"{op} {x} {y}: err={float(err)}")


def check_dot(rng, counters, adversarial=False):
    k = int(rng.integers(1, 12))
    scale_a = 10.0 ** rng.integers(-320, 0)
    scale_b = 10.0 ** rng.integers(-320, 0)
    a = rng.standard_normal(k) * scale_a
    b = rng.standard_normal(k) * scale_b
    f = float(np.dot(a, b))
    exact = sum(F(x) * F(y) for x, y in zip(a, b))
    bound = gam(k) * sum(abs(F(x) * F(y)) for x, y in zip(a, b)) + k * (1 + gam(max(k - 1, 0))) * u_min
    counters["U1"].add(adversarial, abs(F(f) - exact) <= bound, f"dot k={k} scales {scale_a} {scale_b}")


def check_exact_dot(rng, counters, adversarial=False, kind=None):
    k = int(rng.integers(1, 9))
    kind = kind or rng.choice(["tiny", "subnormal", "mixed", "normal"])
    if kind == "tiny":
        a = rng.standard_normal(k) * 10.0 ** rng.integers(-300, -150)
        b = rng.standard_normal(k) * 10.0 ** rng.integers(-300, -150)
    elif kind == "subnormal":
        a = np.array([math.ldexp(float(rng.integers(1, 2 ** 40)), -1074) for _ in range(k)])
        b = rng.standard_normal(k) * 10.0 ** rng.integers(10, 300)        # the rescaling branch (|b| >= 2^54)
    elif kind == "mixed":
        a = np.array([rand_float(rng, -1074, 0) for _ in range(k)])
        b = np.array([rand_float(rng, -1074, 0) for _ in range(k)])
    else:
        a, b = rng.standard_normal(k), rng.standard_normal(k)
    if rng.random() < 0.3:
        a = a + 1j * rng.standard_normal(k) * (np.abs(a).max() if np.abs(a).max() > 0 else 1.0)
    s, err = exact_dot(a, b)
    if np.iscomplexobj(a):
        exact_re = sum(F(x.real) * F(y) for x, y in zip(a, b))
        exact_im = sum(-F(x.imag) * F(y) for x, y in zip(a, b))          # conj(a) . b
        d2 = (F(s.real) - exact_re) ** 2 + (F(s.imag) - exact_im) ** 2
        ok = d2 <= F(err) ** 2
    else:
        exact = sum(F(x) * F(y) for x, y in zip(a, b))
        ok = abs(F(s) - exact) <= F(err)
    counters["U2"].add(adversarial, ok, f"exact_dot {kind} k={k}")


# Lemma U2b: the top of the range. An exact value rounds beyond the range iff its magnitude is at least
# 2^1024 - 2^970 (the midpoint above the largest binary64 number, a tie that rounds away from its odd significand).
BEYOND = Fraction(2 ** 1024 - 2 ** 970)
BIG = float(np.finfo(float).max)


def is_nearest(f, q):
    """f is the binary64 number nearest to the rational q (ties to even), checked against both neighbours in exact
    arithmetic (independent of Python's correctly rounded int / int division used by fp._wide_sum)."""
    if not math.isfinite(f) or abs(q) >= BEYOND:
        return False
    d = abs(F(f) - q)
    with np.errstate(over="ignore"):
        nb = (np.nextafter(f, math.inf), np.nextafter(f, -math.inf))
    for g in nb:
        G = F(g) if math.isfinite(g) else Fraction(2 ** 1024) * (1 if g > 0 else -1)
        dg = abs(G - q)
        if dg < d:
            return False
        if dg == d and math.isfinite(g) and (struct.unpack("<q", struct.pack("<d", f))[0] & 1):
            return False                                       # a tie rounds to the even significand
    return True


def top_vector(rng, k, kind):
    if kind == "product":                 # products with exponents 990 ... 1023 (both factors normal)
        ea = rng.integers(400, 1023, k)
        eb = rng.integers(990, 1024, k) - ea
    elif kind == "factor":                # one factor at or above 2^995, the product anywhere
        ea = rng.integers(995, 1024, k)
        eb = rng.integers(-1060, 1024 - ea + 1)
    else:                                 # mixed: top pairs among tiny, subnormal and ordinary ones
        ea = rng.integers(-1074, 1024, k)
        eb = np.array([int(rng.integers(max(-1074, -1074 - e), 1024 - max(e, 0) + 2)) for e in ea])
    eb = np.minimum(eb, 1024)            # finite entries only (the lemma concerns finite inputs)
    a = rng.uniform(1, 2, k) * np.ldexp(1.0, (ea - 1).astype(np.int32)) * rng.choice([-1.0, 1.0], k)
    b = rng.uniform(1, 2, k) * np.ldexp(1.0, (eb - 1).astype(np.int32)) * rng.choice([-1.0, 1.0], k)
    a[~np.isfinite(a)] = BIG
    b[~np.isfinite(b)] = -BIG
    sw = rng.random(k) < 0.5
    a[sw], b[sw] = b[sw], a[sw]
    return a, b


def check_top_pair(a, b, counters, adversarial, tag):
    """The pair is finite exactly when every product rounds to a finite value and the retained sum (the products not
    left out as tiny, |fl(ab)| >= 2^-967: Lemma U2; each real component) rounds into the range; then |s - x^H y| <= err
    with the exact inner product (real: also nearest for one pair). Not finite otherwise."""
    a, b = np.asarray(a), np.asarray(b)
    cplx = np.iscomplexobj(a) or np.iscomplexobj(b)

    def kept(u, v):              # exact sum of the products exact_dot keeps, and of all products
        with np.errstate(over="ignore", under="ignore"):
            k = [abs(float(x) * float(y)) >= 2.0 ** -967 for x, y in zip(u, v)]
        return (sum((F(x) * F(y) for x, y, kk in zip(u, v, k) if kk), Fraction(0)),
                sum((F(x) * F(y) for x, y in zip(u, v)), Fraction(0)))
    with np.errstate(over="ignore", invalid="ignore"):
        s, err = exact_dot(a, b)
        if cplx:
            ac, bc = a.astype(complex), b.astype(complex)
            prods = np.concatenate([ac.real * bc.real, ac.imag * bc.imag, ac.real * bc.imag, ac.imag * bc.real])
        else:
            prods = a * b
    if cplx:
        (k1, e1), (k2, e2) = kept(ac.real, bc.real), kept(ac.imag, bc.imag)
        (k3, e3), (k4, e4) = kept(ac.real, bc.imag), kept(ac.imag, bc.real)
        re_k, im_k, re, im = k1 + k2, k3 - k4, e1 + e2, e3 - e4
        if np.isfinite(prods).all() and abs(re_k) < BEYOND and abs(im_k) < BEYOND:
            ok = (math.isfinite(s.real) and math.isfinite(s.imag) and math.isfinite(err)
                  and abs(F(s.real) - re) + abs(F(s.imag) - im) <= F(err))
        else:
            ok = not (math.isfinite(s.real) and math.isfinite(s.imag) and math.isfinite(err))
    else:
        q_k, q = kept(a, b)
        if np.isfinite(prods).all() and abs(q_k) < BEYOND:
            ok = math.isfinite(s) and math.isfinite(err) and abs(F(s) - q) <= F(err)
            if a.size == 1 and abs(prods[0]) >= 2.0 ** -967:      # a pair below TINY_PRODUCT is left out (Lemma U2)
                ok = ok and is_nearest(s, q)
        else:
            ok = not (math.isfinite(s) and math.isfinite(err))
    counters["U2b"].add(adversarial, ok, f"exact_dot top {tag}: {a.tolist()} {b.tolist()} -> {s} {err}")


def check_exact_dot_top(rng, counters):
    k = int(rng.integers(1, 9))
    kind = str(rng.choice(["product", "factor", "mixed"]))
    a, b = top_vector(rng, k, kind)
    if rng.random() < 0.3:
        a2, b2 = top_vector(rng, k, kind)
        ac, bc = np.empty(k, complex), np.empty(k, complex)    # set the parts directly: 1j * BIG would give a nan part
        ac.real, ac.imag, bc.real, bc.imag = a, a2, b, b2[::-1]
        a, b = ac, bc
    check_top_pair(a, b, counters, False, kind)


def check_sum_top(t, counters, adversarial, tag):
    """exact_sum (math.fsum path and, padded beyond EXACT_SUM_MIN_TERMS, the cascade) of finite terms: the nearest
    binary64 number to the exact sum, nan when the exact sum rounds beyond the range."""
    t = np.asarray(t, dtype=float)
    q = sum((F(x) for x in t.tolist()), Fraction(0))
    for pad in (0, 5000):
        tt = np.concatenate([t, np.tile([1.0, -1.0], pad // 2)]) if pad else t
        got = exact_sum(tt)
        ok = math.isnan(got) if abs(q) >= BEYOND else is_nearest(got, q)
        counters["U2b"].add(adversarial, ok, f"exact_sum top {tag} pad={pad}: {t.tolist()[:6]} -> {got}")


def check_wide_sum(rng, counters):
    n = int(rng.integers(2, 12))
    t = rng.uniform(1, 2, n) * np.ldexp(1.0, (rng.integers(-1074, 1024, n) - 1).astype(np.int32)) * rng.choice([-1.0, 1.0], n)
    big = rng.choice([BIG, -BIG, 2.0 ** 1023, -2.0 ** 1023], int(rng.integers(1, 4)))
    t = rng.permutation(np.concatenate([t[np.isfinite(t)], big]))
    check_sum_top(t, counters, False, "random")


def check_norm(rng, counters, adversarial=False, v=None):
    if v is None:
        n = int(rng.integers(1, 40))
        v = rng.standard_normal(n) * 10.0 ** rng.integers(-330, -140)
        if rng.random() < 0.3:
            v = v + 1j * rng.standard_normal(n) * 10.0 ** rng.integers(-330, -140)
    comps = []
    for t in np.asarray(v).ravel():
        z = complex(t)
        comps += [F(z.real), F(z.imag)]
    s2 = sum(c ** 2 for c in comps)
    hi, lo = sqrt_up(s2), sqrt_down(s2)
    ok = F(norm2_upper(v)) >= hi and F(norm2_lower(v)) <= lo
    counters["U3"].add(adversarial, ok, f"norm n={np.asarray(v).size}: upper={norm2_upper(v)} exact~{float(hi)} lower={norm2_lower(v)}")


def check_norm_scalar(counters):
    for N in list(range(1, 100_001)) + [2 ** j for j in range(17, 28)]:
        gN, g2 = gam(N), gam(N + 2)
        counters["U3"].add(N > 100_000, (1 + 2 * g2) ** 2 >= (1 + gN) / (1 - gN), f"T_N inequality N={N}")
        # the computed term is at least sqrt(N u_min) sqrt((1 + gN)/(1 - gN))
        T = F(norm_underflow_term(N))
        counters["U3"].add(N > 100_000, T ** 2 >= N * u_min * (1 + gN) / (1 - gN), f"T_N value N={N}")


def residual_check(A, x, mu, counters, adversarial, tag):
    ad = OperatorAdapter(A)
    n = x.size
    res = residuals(ad, np.array([mu]), x[:, None])
    Ad = np.asarray(A)
    r = []
    for i in range(n):
        re = sum(F(Ad[i, j]) * F(x[j]) for j in range(n)) - F(mu) * F(x[i])
        r.append(re)
    r2 = sum(t ** 2 for t in r)
    x2 = sum(F(t) ** 2 for t in x)
    rel = float(res.rel_bound[0])
    if x2 == 0 or not np.isfinite(rel):
        counters["U5"].add(adversarial, rel == float("inf") or x2 == 0, f"inf radius {tag}")
        return
    ok = F(rel) ** 2 * x2 >= r2 and F(res.upper[0]) ** 2 >= r2
    Rc, Ec = res.R[:, 0], res.E[:, 0]
    cw = F(residual_entry_count(ad.width, False))
    ok_e = all(F(Ec[i]) + cw * u_min >= abs(F(Rc[i]) - r[i]) for i in range(n))
    counters["U5"].add(adversarial, ok and ok_e, f"U5 {tag}: rel={rel} ok={ok} entrywise={ok_e}")


def check_residuals(rng, counters):
    n = int(rng.integers(1, 6))
    M = rng.standard_normal((n, n))
    A = (M + M.T) / 2
    kind = rng.choice(["scaled_operator", "lapack_zeros", "tiny_x", "normal"])
    w, V = np.linalg.eigh(A)
    j = int(rng.integers(0, n))
    if kind == "scaled_operator":
        s = 10.0 ** rng.integers(-300, -140)
        A2, x, mu = A * s, V[:, j], w[j] * s
    elif kind == "lapack_zeros":
        x = V[:, j].copy()
        x[int(rng.integers(0, n))] = 10.0 ** rng.integers(-300, -250) * rng.choice([-1, 1])
        A2, mu = A, w[j]
    elif kind == "tiny_x":
        A2, x, mu = A, V[:, j] * 10.0 ** rng.integers(-170, -150), w[j]
    else:
        A2, x, mu = A, V[:, j] + 1e-9 * rng.standard_normal(n), w[j]
    residual_check(A2, x, mu, counters, False, kind)


def check_monotone(rng, counters):
    n = int(rng.integers(2, 7))
    M = rng.standard_normal((n, n))
    A = (M + M.T) / 2
    w, V = np.linalg.eigh(A)
    j = int(rng.integers(0, n))
    scale = 10.0 ** rng.choice([0, -50, -120, -150, -200])
    x, mu = V[:, j] * scale, w[j]
    ad = OperatorAdapter(A)
    res = residuals(ad, np.array([mu]), x[:, None])
    Rc, Ec = res.R[:, 0], res.E[:, 0]
    R_old = float(np.linalg.norm(Rc)) * (1 + gamma(n + 2)) * _GUARD
    E_old = float(np.linalg.norm(Ec)) * (1 + gamma(n + 2)) * _GUARD
    xl_old = max(0.0, float(np.linalg.norm(x)) * (1 - gamma(n + 2)) / _GUARD)
    rel_old = (R_old + E_old) / xl_old * (1 + 4 * U) if xl_old > 0 else float("inf")
    rel_new = float(res.rel_bound[0])
    if not (np.isfinite(rel_old) and np.isfinite(rel_new)):
        counters["U6"].add(False, rel_new >= rel_old or not np.isfinite(rel_new), f"monotone inf scale {scale}")
        return
    T = norm_underflow_term(n)
    A_ = residual_additive_term(n, ad.width, False)
    xl_new = float(res.xnorm_lower[0])
    D = 2 * _GUARD * ((2 * T + A_) / xl_new + (R_old + E_old) * T / (xl_new * xl_old)) + 4 * ULP_MIN
    ok = rel_old <= rel_new <= rel_old + D
    if rel_old > 1e-145:
        ok &= rel_new == rel_old
    counters["U6"].add(False, ok, f"monotone scale {scale}: old={rel_old} new={rel_new} D={D}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("M", "U1", "U2", "U2b", "U3", "U5", "U6")}
    for i in range(args.cases):
        check_model(rng, counters)
        if i % 4 == 0:
            check_dot(rng, counters)
        if i % 4 == 1:
            check_exact_dot(rng, counters)
            if i % 8 == 1:
                check_exact_dot_top(rng, counters)
            else:
                check_wide_sum(rng, counters)
        if i % 4 == 2:
            check_norm(rng, counters)
        if i % 8 == 3:
            check_residuals(rng, counters)
        if i % 8 == 7:
            check_monotone(rng, counters)
        if i % 5000 == 0:
            print(f"  ... {i} cases", flush=True)
    check_norm_scalar(counters)
    # adversarial
    A3 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    for _ in range(50):
        residual_check(A3, np.array([1e-160, 0.0, -1e-160]), 2.0, counters, True, "zero residual, tiny x")
        xs = np.array([1.0, 2.0 ** -1060, 0.5])
        residual_check(A3, xs, 0.7, counters, True, "one subnormal entry")
        residual_check(A3 * 2.0 ** -1040, np.array([1.0, 0.3, -0.2]), 2.0 ** -1041, counters, True, "subnormal A and mu")
        if _ == 0:
            check_norm(rng, counters, True, np.full(10 ** 4, 2.0 ** -540))        # every square flushes to 0
        for e in (-968, -967, -966):
            a = np.array([math.ldexp(1.0, e // 2), 1.0])
            b = np.array([math.ldexp(1.0, e - e // 2) * (1 + 1e-9), 1.0])
            s, err = exact_dot(a, b)
            exact = sum(F(x) * F(y) for x, y in zip(a, b))
            counters["U2"].add(True, abs(F(s) - exact) <= F(err), f"threshold 2^{e}")
        check_exact_dot(rng, counters, True, "subnormal")
    # adversarial, Lemma U2b: the split threshold 2^1024 / (2^27 + 1), HUGE_FACTOR and HUGE_PRODUCT from both sides,
    # products at the overflow threshold, factors at the top with subnormal partners, overflowing partial sums
    # the retained sum decides finiteness at the midpoint above BIG (a left-out tiny product on either side), and a TwoSum
    # step whose sum is finite while an intermediate overflows (-3 * 2^970 then BIG)
    check_top_pair([BIG, 2.0 ** 970, 2.0 ** -500], [1.0, 1.0, -2.0 ** -500], counters, True, "retained sum at midpoint")
    check_top_pair([BIG, 2.0 ** 970, -2.0 ** -966] + [0.9 * 2.0 ** -467] * 3, [1.0, 1.0, 1.0] + [2.0 ** -500] * 3,
                   counters, True, "retained sum below midpoint")
    check_sum_top([-3.0 * 2.0 ** 970, BIG], counters, True, "TwoSum intermediate overflow")
    s0 = math.ldexp(1.0 / (2 ** 27 + 1), 1024)
    for f in (s0, np.nextafter(s0, 0), np.nextafter(s0, math.inf), HUGE_FACTOR, np.nextafter(HUGE_FACTOR, 0), BIG):
        for g in (2.0 ** -30, 1.0, 5e-324, 2.0 ** -1060, 0.75):
            check_top_pair([f], [g], counters, True, "factor edge")
            check_top_pair([g], [-f], counters, True, "factor edge, mirrored")
    root = math.sqrt(BIG)
    for p_ in (HUGE_PRODUCT, np.nextafter(HUGE_PRODUCT, 0), np.nextafter(HUGE_PRODUCT, math.inf)):
        check_top_pair([p_], [1.0], counters, True, "product edge")
        check_top_pair([math.sqrt(p_)], [math.sqrt(p_)], counters, True, "product edge, square")
    for a_, b_ in ((root, root), (np.nextafter(root, math.inf), root), (np.nextafter(root, math.inf),) * 2,
                   (BIG, 1.0 - 2.0 ** -53), (BIG, np.nextafter(1.0, 2.0)), (2.0 ** 1023, 1.9999999999999998),
                   (2.0 ** 512, 2.0 ** 511 * 1.9999999999999998)):
        check_top_pair([a_], [b_], counters, True, "overflow threshold")
    check_top_pair(np.array([BIG + 0j, complex(0.0, root)]), np.array([1.0 - 1e-3j, -root + 0j]), counters, True,
                   "complex, both parts at the top")
    check_top_pair(np.array([BIG + 0j, complex(0.0, root)]), np.array([1.0 - 1e-3j, root + 0j]), counters, True,
                   "complex, Im beyond the range")
    for t in ([BIG, BIG, -BIG], [-BIG, -BIG, BIG, 1.0], [BIG, BIG, -BIG, 5e-324], [BIG, 2.0 ** 970],
              [BIG, 2.0 ** 970, -5e-324], [BIG, 2.0 ** 970 - 2.0 ** 918], [BIG, BIG], [2.0 ** 1023, 2.0 ** 1023],
              [2.0 ** 1023, 2.0 ** 1023, -5e-324], [BIG, 2.0 ** 1023, -2.0 ** 1023, -BIG, 2.0 ** -1074]):
        check_sum_top(t, counters, True, "edge")
    ok = True
    for key, name in (("M", "model (M-U)"), ("U1", "dot products (Lemma U1)"), ("U2", "exact_dot (Lemma U2)"),
                      ("U2b", "exact_dot and exact_sum at the top of the range (Lemma U2b)"),
                      ("U3", "norms and T_N (Lemma U3)"), ("U5", "Theorem 1 radius under (M-U) (Theorem U5)"),
                      ("U6", "radii never shrink (Theorem U6)")):
        ok &= counters[key].report(name)
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
