#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""adversarial_fp_modulus-up.py

Independent adversarial check (project rule D-12, the independent re-derivation) of Lemma M and its corollary for
certify_ed/fp.py: modulus_up, _modulus_up_scaled, abs_up (docs/proofs/fp_modulus-up.md; independent derivation in
docs/proofs/independent/fp_modulus-up.md). Written on 2026-11-18 before the original proof was read, against an excerpt of
fp.py at branch fix/complex-moduli; in the repository it tests certify_ed.fp itself (or an excerpt given by --code).
UNVERIFIED (D-12): this script is one of the three legs of D-12; it is not a check by Sarang.

What is exact here.  Every pass/fail decision on Lemma M is a comparison of rational numbers (fractions.Fraction on
the binary64 values):
  (i)   m^2 >= x^2 + y^2
  (ii)  (m - 2^-1073)^2 <= (x^2 + y^2)(1 + 12u)^2 when m > 2^-1073, and m^2 <= (x^2 + y^2)(1 + 12u)^2 when
        x^2 + y^2 >= 2^-2044
  (iii) m = inf  =>  (x^2 + y^2)(1 + 12u)^2 > Omega^2
  (iv)  x = 0 or y = 0  =>  m = max(x, y)
and, beyond the statement, the lower margin m^2 >= (x^2 + y^2)(1 + 4u)^2 for two nonzero components.
The code under test is certify_ed.fp of this repository (with --code FILE: an excerpt, exec of the file with
U = 2^-53 supplied, as in the first run).  An exact reference model of "the computation" of the statement, written from the statement
alone with integer arithmetic (every rounding done on integers, no libm), is compared with the code bit for bit and
is also run with a truncating and an upward np.ldexp (both allowed by FP-M) and with negative controls.
The only inexact parts are labelled: the double-double screen that selects hard cases (a selector, the selected
cases are then checked exactly) and the reported extreme ratios (floats evaluated from exact rationals).

Usage: python docs/proofs/independent/adversarial_fp_modulus-up.py [--workers 6] [--scale 1.0] [--out FILE] [--json FILE]
       [--code EXCERPT.py]
       (--workers N runs the exact checks in N child processes that exchange files in --work, default a temporary
       directory)
"""
import argparse
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction as F

import numpy as np

INF = math.inf
OMEGA_F = sys.float_info.max
UF = F(1, 2 ** 53)                 # u
T = F(1, 2 ** 1074)                # 2^-1074
SN_F = 2.0 ** -1022
SN2 = F(1, 2 ** 2044)              # (2^-1022)^2
OM2 = F(OMEGA_F) ** 2
CK = {k: (1 + k * UF) ** 2 for k in (8, 9, 10, 11, 12)}
C12 = CK[12]
CL = {k: (1 + k * UF) ** 2 for k in (4, 5, 6)}
MASK52 = (1 << 52) - 1
LO, HI = 2.0 ** -500, 2.0 ** 500
G8 = 1.0 + 2.0 ** -50              # 1 + 8u
G2, G4, G6 = 1.0 + 2.0 ** -52, 1.0 + 2.0 ** -51, 1.0 + 3 * 2.0 ** -52


# ----------------------------------------------------------------------------------------------------------------
# exact reference arithmetic on integers: a float is n * 2^s with integers n, s (read from the IEEE bits)
# ----------------------------------------------------------------------------------------------------------------
def f2d(x):
    """non-negative finite binary64 x -> (n, s) with x = n * 2^s exactly"""
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    e = (b >> 52) & 0x7FF
    f = b & MASK52
    return (f, -1074) if e == 0 else (f | (1 << 52), e - 1075)


def d2f(k, q):
    """k * 2^q as a binary64 number (k < 2^53, and k >= 2^52 or q = -1074); inf when it is >= 2^1024"""
    if k == 0:
        return 0.0
    if k < (1 << 52):
        assert q == -1074, (k, q)
        return struct.unpack("<d", struct.pack("<Q", k))[0]
    e = q + 1075
    if e >= 2047:
        return INF
    return struct.unpack("<d", struct.pack("<Q", (e << 52) | (k & MASK52)))[0]


def rnd(n, s, mode=0, ftz=False):
    """(binary64 neighbour of n * 2^s, inexact flag); mode 0 = nearest, ties to even; 1 = toward zero; 2 = away
    from zero; ftz: a result below 2^-1022 is flushed to zero (a control, not IEEE default)."""
    if n == 0:
        return 0.0, False
    E = n.bit_length() - 1 + s          # 2^E <= value < 2^(E+1)
    if ftz and E < -1022:
        return 0.0, True
    q = E - 52 if E - 52 > -1074 else -1074
    sh = q - s
    if sh <= 0:
        return d2f(n << (-sh), q), False
    k = n >> sh
    rem = n & ((1 << sh) - 1)
    if rem:
        if mode == 0:
            half = 1 << (sh - 1)
            if rem > half or (rem == half and (k & 1)):
                k += 1
        elif mode == 2:
            k += 1
    if k == (1 << 53):
        k >>= 1
        q += 1
    return d2f(k, q), rem != 0


def mul_ref(a, b, ftz=False):
    if a == INF or b == INF:
        return INF
    na, sa = f2d(a)
    nb, sb = f2d(b)
    return rnd(na * nb, sa + sb, 0, ftz)[0]


def add_ref(a, b):
    if a == INF or b == INF:
        return INF
    na, sa = f2d(a)
    nb, sb = f2d(b)
    s = sa if sa < sb else sb
    return rnd((na << (sa - s)) + (nb << (sb - s)), s)[0]


def sqrt_ref(x):
    """correctly rounded square root by integer square root with a sticky bit"""
    if x == INF:
        return INF
    if x == 0.0:
        return 0.0
    n, s = f2d(x)
    sh = 112 + (s & 1)
    N = n << sh
    k = math.isqrt(N)
    sticky = 1 if k * k != N else 0
    return rnd((k << 1) | sticky, (s - sh) // 2 - 1)[0]


def next_up(t):
    return struct.unpack("<d", struct.pack("<Q", struct.unpack("<Q", struct.pack("<d", t))[0] + 1))[0]


def ref_modulus(x, y, guard=G8, step=True, mode=0, ftz=False, noscale=False):
    """The computation of the statement for components x, y >= 0 (finite), every rounding exact.
    Returns (m, tag); tag names the case of the proof."""
    if x == 0.0:
        return y, "zero"
    if y == 0.0:
        return x, "zero"
    big, sml = (x, y) if x >= y else (y, x)
    nb, sb = f2d(big)
    ns, ss = f2d(sml)
    if noscale or (LO <= big <= HI):
        p = rnd(nb * nb, 2 * sb, 0, ftz)[0]
        q = rnd(ns * ns, 2 * ss, 0, ftz)[0]
        tag = "U-B(square of the smaller underflows)" if (ns * ns).bit_length() - 1 + 2 * ss < -1022 else "U-A"
        return mul_ref(sqrt_ref(add_ref(p, q)), guard), tag
    e = nb.bit_length() + sb                        # big = f 2^e, f in [1/2, 1)
    a, ainex = rnd(nb, sb - e, mode)
    assert not ainex and 0.5 <= a < 1.0
    b, binex = rnd(ns, ss - e, mode)
    na, sa = f2d(a)
    n2, s2 = f2d(b)
    p = rnd(na * na, 2 * sa, 0, ftz)[0]
    q = rnd(n2 * n2, 2 * s2, 0, ftz)[0]
    if binex:
        tag = "S-3(scaled smaller rounded to the subnormal grid)"
    elif (n2 * n2).bit_length() - 1 + 2 * s2 < -1022:
        tag = "S-2(square of the scaled smaller underflows)"
    else:
        tag = "S-1"
    ms = mul_ref(sqrt_ref(add_ref(p, q)), guard)
    nm, sm = f2d(ms)
    mp, minex = rnd(nm, sm + e, mode)
    if mp == INF:
        return mp, tag + "/R-inf"
    if mp < SN_F:
        tag += "/R-subnormal-inexact" if minex else "/R-subnormal-exact"
        if step:
            mp = next_up(mp)
    elif minex:
        tag += "/R-rounded-up-to-2^-1022"
    else:
        tag += "/R-normal"
    return mp, tag


VARIANTS = [
    # name, guard, subnormal step, ldexp mode, flush products to zero, no scaling
    ("ref_RN", G8, True, 0, False, False),
    ("ref_ldexp_trunc", G8, True, 1, False, False),
    ("ref_ldexp_up", G8, True, 2, False, False),
    ("NC_guard_1+2u", G2, True, 0, False, False),
    ("NC_no_guard", 1.0, True, 0, False, False),
    ("NC_no_subnormal_step", G8, False, 0, False, False),
    ("NC_no_subnormal_step_ldexp_trunc", G8, False, 1, False, False),
    ("NC_no_scaling", G8, True, 0, False, True),
    ("NC_flush_to_zero_products", G8, True, 0, True, False),
    ("INFO_guard_1+4u", G4, True, 0, False, False),
    ("INFO_guard_1+6u", G6, True, 0, False, False),
]


def hx(v):
    return v.hex() if v == v else "nan"


def check_chunk(job):
    """exact checks of one chunk; returns (set name, Counter, examples, extremes)"""
    name, xs, ys, ms, ams = job
    C = Counter()
    EX = {}
    mx_rel = (-INF, "")
    mn_rel = (INF, "")
    mx_add = (-INF, "")

    def ex(key, x, y, m):
        lst = EX.setdefault(key, [])
        if len(lst) < 3:
            lst.append("x=%s y=%s m=%s" % (hx(x), hx(y), hx(m)))

    for xv, yv, m, am in zip(xs, ys, ms, ams):
        x = abs(xv)
        y = abs(yv)
        X = F(x)
        Y = F(y)
        s2 = X * X + Y * Y
        s2c = s2 * C12
        zsub = s2 < SN2
        cache = {}

        def sq(v):
            r = cache.get(v)
            if r is None:
                fv = F(v)
                r = (fv, fv * fv)
                cache[v] = r
            return r

        C["n"] += 1
        if m != m:
            C["code_nan"] += 1
            ex("code_nan", x, y, m)
            continue
        # ---- the code under test ----
        if m == INF:
            C["code_inf"] += 1
            C["iii_checked"] += 1
            if not s2c > OM2:
                C["iii_viol"] += 1
                ex("iii_viol", x, y, m)
        else:
            Mf, M2 = sq(m)
            if M2 < s2:
                C["i_viol"] += 1
                ex("i_viol", x, y, m)
            t = Mf - 2 * T
            if t > 0 and t * t > s2c:
                C["iia_viol"] += 1
                ex("iia_viol", x, y, m)
            if m < SN_F:
                C["code_m_below_2^-1022"] += 1
            if x != 0.0 and y != 0.0:
                # lower margin (not claimed by the statement; proved in the independent derivation): m >= |z|(1 + 4u)
                C["LOW_checked"] += 1
                for k in (4, 5, 6):
                    if M2 < s2 * CL[k]:
                        C["LOW_m<|z|(1+%du)" % k] += 1
            if not zsub:
                C["iib_checked"] += 1
                if M2 > s2c:
                    C["iib_viol"] += 1
                    ex("iib_viol", x, y, m)
                for k in (8, 9, 10, 11):
                    if M2 > s2 * CK[k]:
                        C["TIGHT_m>|z|(1+%du)" % k] += 1
                if s2 > 0:
                    e = float(M2 / s2 - 1)
                    d = (e / 2 - e * e / 8) * 2.0 ** 53
                    if d > mx_rel[0]:
                        mx_rel = (d, "x=%s y=%s m=%s" % (hx(x), hx(y), hx(m)))
                    if d < mn_rel[0] and x != 0.0 and y != 0.0:
                        mn_rel = (d, "x=%s y=%s m=%s" % (hx(x), hx(y), hx(m)))
            else:
                C["z_below_2^-1022"] += 1
                for addk in (0, 1):
                    t = Mf - addk * T
                    if t > 0 and t * t > s2c:
                        C["ADD_term_%d*2^-1074_viol" % addk] += 1
                        ex("ADD_term_%d*2^-1074_viol" % addk, x, y, m)
                if x != 0.0 and y != 0.0:
                    Xi, Yi, Mi = int(X / T), int(Y / T), int(Mf / T)
                    R = math.isqrt((Xi * Xi + Yi * Yi) << 80)
                    exc = ((Mi << 40) - R) / 2.0 ** 40 - 12 * 2.0 ** -53 * (R / 2.0 ** 40)
                    if exc > mx_add[0]:
                        mx_add = (exc, "x=%s y=%s m=%s" % (hx(x), hx(y), hx(m)))
        if x == 0.0 or y == 0.0:
            C["iv_checked"] += 1
            if not m == (x if x >= y else y):
                C["iv_viol"] += 1
                ex("iv_viol", x, y, m)
        # ---- numpy.abs on the same input (host baseline, a control) ----
        if am == am and am != INF:
            if sq(am)[1] < s2:
                C["np.abs:i_viol"] += 1
                ex("np.abs:i_viol", x, y, am)
        # ---- exact reference model and its variants ----
        got_rn = None
        for vn, guard, step, mode, ftz, noscale in VARIANTS:
            mv, tag = ref_modulus(x, y, guard, step, mode, ftz, noscale)
            if vn == "ref_RN":
                got_rn = mv
                C["tag:" + tag] += 1
                if mv != m:
                    C["code_ne_ref_RN"] += 1
                    ex("code_ne_ref_RN", x, y, m)
            elif vn in ("ref_ldexp_trunc", "ref_ldexp_up"):
                if mv != got_rn:
                    C[vn + ":differs_from_ref_RN"] += 1
                if m != got_rn and mv == m:
                    C["code_eq_" + vn] += 1
            if mv == INF:
                if (vn.startswith("ref_") or vn == "NC_no_scaling") and not s2c > OM2:
                    C[vn + ":iii_viol"] += 1
                    ex(vn + ":iii_viol", x, y, mv)
                continue
            Vf, V2 = sq(mv)
            if V2 < s2:
                C[vn + ":i_viol"] += 1
                ex(vn + ":i_viol", x, y, mv)
            if vn.startswith("ref_"):
                t = Vf - 2 * T
                if t > 0 and t * t > s2c:
                    C[vn + ":iia_viol"] += 1
                    ex(vn + ":iia_viol", x, y, mv)
                if not zsub and V2 > s2c:
                    C[vn + ":iib_viol"] += 1
                    ex(vn + ":iib_viol", x, y, mv)
                if zsub:
                    t = Vf - T
                    if t > 0 and t * t > s2c:
                        C[vn + ":ADD_term_1*2^-1074_viol"] += 1
    return name, C, EX, (mx_rel, mn_rel, mx_add)


# ----------------------------------------------------------------------------------------------------------------
# test sets
# ----------------------------------------------------------------------------------------------------------------
NUMPY_COUNTEREXAMPLES = [complex(0.02019294356747192, -0.5012539518584045),
                         complex(1.9986893956853402, 0.07375207458660175),
                         complex(1.010573342221019, 0.3330018761835207)]


def stepf(v, k):
    for _ in range(abs(k)):
        v = math.nextafter(v, INF if k > 0 else 0.0)
    return v


def mk(E, mant):
    with np.errstate(all="ignore"):
        return np.ldexp(mant, np.asarray(E).astype(np.int32))


def rmant(rng, n):
    return 1.0 + rng.integers(0, 1 << 52, n).astype(np.float64) * 2.0 ** -52


def signs(rng, n):
    return rng.choice(np.array([-1.0, 1.0]), n)


def finish(rng, name, xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    return name, x * signs(rng, x.size), y * signs(rng, y.size)


def random_pair(rng, n, EM, d):
    """larger component mant 2^EM, smaller mant' 2^(EM-d) (clipped at 2^-1074), random order and signs"""
    EN = np.maximum(EM - d, -1074)
    M = mk(EM, rmant(rng, n))
    N = mk(EN, rmant(rng, n))
    sw = rng.random(n) < 0.5
    return np.where(sw, N, M), np.where(sw, M, N)


def build_sets(scale, seed=20261118):
    rng = np.random.default_rng(seed)
    sets = []
    mants = (1.25, 1.5, 1.75, math.sqrt(2.0), 4.0 / 3.0, 1.0 + 2.0 ** -52, 2.0 - 2.0 ** -52)

    # A: boundary magnitudes, all ordered pairs
    B = set()
    for k in range(1, 6):
        B.add(k * 5e-324)
    for base in (2.0 ** -1022, 2.0 ** -500, 2.0 ** 500):
        for k in range(-3, 4):
            B.add(stepf(base, k))
    for base in (2.0 ** -537, 2.0 ** -511, 2.0 ** 511, 2.0 ** 1023, 1.0, 2.0 ** -1021):
        for k in (-1, 0, 1):
            B.add(stepf(base, k))
    for k in range(0, 4):
        B.add(stepf(OMEGA_F, -k))
    for p in (-1060, -1000, -538, -536, -512, -510, -501, -499, -53, -26, -1, 1, 26, 53, 499, 501, 510, 512, 1000,
              1022):
        B.add(2.0 ** p)
    for E in (-1073, -1060, -1023, -1022, -1021, -538, -537, -512, -511, -501, -500, -499, 0, 499, 500, 501, 511,
              512, 1022, 1023):
        for mt in mants:
            v = math.ldexp(mt, E)
            if 0.0 < v < INF:
                B.add(v)
    B = sorted(B)
    sets.append(finish(rng, "A boundary magnitudes, all ordered pairs (%d magnitudes)" % len(B),
                       [a for a in B for b in B], [b for a in B for b in B]))

    # B: ratios 2^-k and 2^+k of the components, k = 0..1100
    xs, ys = [], []
    bases_dn = [OMEGA_F, 2.0 ** 1023, 1.5 * 2.0 ** 1023, math.sqrt(2.0) * 2.0 ** 1000, 2.0 ** 511, 2.0 ** 501,
                stepf(2.0 ** 500, 1), 2.0 ** 500, stepf(2.0 ** 500, -1), 1.75 * 2.0 ** 499, 2.0 ** 26, 1.0,
                math.sqrt(2.0), stepf(2.0, -1), 2.0 ** -499, stepf(2.0 ** -500, 1), 2.0 ** -500,
                stepf(2.0 ** -500, -1), 1.3 * 2.0 ** -501, 2.0 ** -511, 1.5 * 2.0 ** -1000, 2.0 ** -1021,
                2.0 ** -1022]
    rm = rmant(rng, len(bases_dn) * 1101).tolist()
    j = 0
    for M in bases_dn:
        eM = math.frexp(M)[1]
        for k in range(0, 1101):
            N = math.ldexp(M, -k)
            if N > 0.0:
                xs += [M, N]
                ys += [N, M]
            N2 = math.ldexp(rm[j], eM - 1 - k)
            j += 1
            if N2 > 0.0:
                xs += [M, N2]
                ys += [N2, M]
    bases_up = [5e-324, 3 * 5e-324, 2.0 ** -1073, 1.5 * 2.0 ** -1060, stepf(2.0 ** -1022, -1), 2.0 ** -1022,
                math.sqrt(2.0) * 2.0 ** -1000, 2.0 ** -600, 2.0 ** -511, 2.0 ** -500]
    for N in bases_up:
        for k in range(0, 1101):
            M = math.ldexp(N, k) if k + math.frexp(N)[1] <= 1024 else INF
            if M < INF:
                xs += [M, N]
                ys += [N, M]
    sets.append(finish(rng, "B component ratios 2^-k and 2^+k, k = 0..1100 (%d + %d bases)"
                       % (len(bases_dn), len(bases_up)), xs, ys))

    # C: equal components at every exponent
    vals = set()
    for E in range(-1074, 1024):
        for mt in (1.0,) + mants:
            v = math.ldexp(mt, E)
            if 0.0 < v < INF:
                vals.add(v)
    vals = sorted(vals)
    sets.append(finish(rng, "C equal components, every exponent -1074..1023", vals, vals))

    # D: the three NumPy counterexamples, all signs and both orders, and scaled by 2^j, j = -1080..1030
    xs, ys = [], []
    for z in NUMPY_COUNTEREXAMPLES:
        for sr in (1.0, -1.0):
            for si in (1.0, -1.0):
                xs += [sr * z.real, si * z.imag]
                ys += [si * z.imag, sr * z.real]
        for jj in range(-1080, 1031):
            try:
                a, b = math.ldexp(z.real, jj), math.ldexp(z.imag, jj)
            except OverflowError:
                continue
            if a != 0.0 and b != 0.0:
                xs += [a, b]
                ys += [b, a]
    sets.append(finish(rng, "D the three NumPy counterexamples, signs, orders, scaled by 2^j", xs, ys))

    # F: Pythagorean triples scaled by 2^j (the exact modulus is a binary64 number)
    xs, ys = [], []
    for (a, b) in ((3.0, 4.0), (5.0, 12.0), (8.0, 15.0), (20.0, 21.0)):
        for jj in range(-1078, 1021):
            try:
                p, q = math.ldexp(a, jj), math.ldexp(b, jj)
            except OverflowError:
                continue
            if p != 0.0 and q != 0.0:
                xs += [p, q]
                ys += [q, p]
    sets.append(finish(rng, "F Pythagorean triples scaled by 2^j", xs, ys))

    # G: results at the border of the subnormal range: larger component within 200 ulps of 2^-1022
    xs, ys = [], []
    smalls = [5e-324, 1e-323, 2.0 ** -1060, 2.0 ** -1049, 2.0 ** -1048, 1.5 * 2.0 ** -1048, 2.0 ** -1047,
              2.0 ** -1040, 2.0 ** -1030, 2.0 ** -1023]
    for jj in range(-200, 201):
        M = stepf(2.0 ** -1022, jj)
        for N in smalls + [M, stepf(M, -1)]:
            xs += [M, N]
            ys += [N, M]
    # the tie of the rescaling: scaled root times guard = 1 - u, e = -1022, so that m_s 2^e = 2^-1022 - 2^-1075
    xs += [stepf(2.0 ** -1022, -5), 2.0 ** -1048]
    ys += [2.0 ** -1048, stepf(2.0 ** -1022, -5)]
    sets.append(finish(rng, "G larger component within 200 ulps of 2^-1022 (border of the subnormal step)", xs, ys))

    # Z: one component zero (+0 or -0), the other over the boundary magnitudes and random over the whole range
    nz = max(200, int(20000 * scale))
    v = np.concatenate([np.array(B), mk(rng.integers(-1074, 1024, nz), rmant(rng, nz)), np.array([0.0, 0.0])])
    zero = np.where(rng.random(v.size) < 0.5, 0.0, -0.0)
    sets.append(finish(rng, "Z one component zero (+0 or -0), the other any magnitude; and 0 + 0i",
                       np.concatenate([v, zero]), np.concatenate([zero, v])))

    # E: random
    def rset(name, n, EM, d):
        n = max(1, int(n * scale))
        x, y = random_pair(rng, n, EM(n), d(n))
        sets.append(finish(rng, name, x, y))

    ri = rng.integers
    rset("E1 random, larger component in [1/4, 4), ratio 2^0..2^-3", 150000, lambda n: ri(-2, 2, n),
         lambda n: ri(0, 4, n))
    n2 = max(1, int(60000 * scale))
    sets.append(finish(rng, "E2 random, independent exponents -1074..1023",
                       mk(ri(-1074, 1024, n2), rmant(rng, n2)), mk(ri(-1074, 1024, n2), rmant(rng, n2))))
    rset("E3 random, exponent of the larger -1074..1023, ratio 2^0..2^-60", 150000,
         lambda n: ri(-1074, 1024, n), lambda n: ri(0, 61, n))
    rset("E4 random, larger component in [2^-1074, 2^-1014), ratio 2^0..2^-56", 100000,
         lambda n: ri(-1074, -1014, n), lambda n: ri(0, 57, n))
    rset("E5 random, larger component in [2^1017, Omega], ratio 2^0..2^-56", 60000,
         lambda n: ri(1017, 1024, n), lambda n: ri(0, 57, n))
    rset("E6 random, larger component within 3 binades of 2^-500 or 2^500", 100000,
         lambda n: np.where(rng.random(n) < 0.5, ri(-503, -496, n), ri(497, 504, n)), lambda n: ri(0, 57, n))
    n7 = max(1, int(60000 * scale))
    M7 = mk(ri(-500, -486, n7), rmant(rng, n7))
    N7 = mk(ri(-536, -510, n7), rmant(rng, n7))
    sw = rng.random(n7) < 0.5
    sets.append(finish(rng, "E7 random, unscaled low end: larger in [2^-500, 2^-486), smaller in [2^-536, 2^-510)",
                       np.where(sw, N7, M7), np.where(sw, M7, N7)))
    rset("E8a random, scaled high: ratio 2^-505..2^-520 or 2^-1016..2^-1080", 60000,
         lambda n: ri(501, 1024, n), lambda n: np.where(rng.random(n) < 0.5, ri(505, 521, n), ri(1016, 1081, n)))
    rset("E8b random, scaled low: larger in [2^-560, 2^-500), ratio 2^-505..2^-520", 30000,
         lambda n: ri(-560, -500, n), lambda n: ri(505, 521, n))
    return sets


# double-double screen (a selector only; the selected cases are then checked exactly)
def two_prod(a, b):
    p = a * b
    c = 134217729.0
    t = c * a
    ah = t - (t - a)
    al = a - ah
    t = c * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def dd_rel(v, x, y):
    """(v^2 - (x^2 + y^2)) / (2 (x^2 + y^2)), about v/|z| - 1, evaluated in double-double"""
    p1, e1 = two_prod(x, x)
    p2, e2 = two_prod(y, y)
    sh, sl = two_sum(p1, p2)
    sl = sl + (e1 + e2)
    vh, vl = two_prod(v, v)
    return ((vh - sh) + (vl - sl)) / (2.0 * sh)


def screened_set(mu, scale, seed=977, keep=3000):
    rng = np.random.default_rng(seed)
    total = max(20000, int(5000000 * scale))
    keep = max(50, int(keep * min(1.0, scale * 4)))
    batch = 1000000
    pools = {"r_low": [], "r_high": [], "m_low": [], "m_high": []}
    done = 0
    ext = [INF, -INF, INF, -INF]
    while done < total:
        n = min(batch, total - done)
        done += n
        x = rmant(rng, n)
        y = mk(-rng.integers(0, 5, n), rmant(rng, n))
        z = np.empty(n, np.complex128)
        z.real = x
        z.imag = y
        m = mu(z)
        r = np.sqrt(x * x + y * y)
        dr = dd_rel(r, x, y) * 2.0 ** 53
        dm = dd_rel(m, x, y) * 2.0 ** 53
        ext = [min(ext[0], dr.min()), max(ext[1], dr.max()), min(ext[2], dm.min()), max(ext[3], dm.max())]
        kk = min(keep, n - 1)
        for key, arr, low in (("r_low", dr, True), ("r_high", dr, False), ("m_low", dm, True),
                              ("m_high", dm, False)):
            idx = np.argpartition(arr, kk)[:kk] if low else np.argpartition(-arr, kk)[:kk]
            pools[key].append((arr[idx], x[idx], y[idx]))
    xs, ys = [], []
    for key, parts in pools.items():
        a = np.concatenate([p[0] for p in parts])
        xx = np.concatenate([p[1] for p in parts])
        yy = np.concatenate([p[2] for p in parts])
        order = np.argsort(a if key.endswith("low") else -a)[:keep]
        xs.append(xx[order])
        ys.append(yy[order])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    name = ("E9 hard cases selected by a double-double screen of %d random z in the unscaled branch "
            "(%d kept: lowest/highest error of the unguarded root and of m)" % (total, x.size))
    info = ("screen estimates (double-double, a selector, not exact): unguarded root rel. error in [%.4f u, %.4f u];"
            " m/|z| - 1 in [%.4f u, %.4f u]" % tuple(ext))
    return (name, x, y), info, total


# ----------------------------------------------------------------------------------------------------------------
# the hypotheses FP-M on this host, checked exactly on samples
# ----------------------------------------------------------------------------------------------------------------
def host_checks(scale, P):
    rng = np.random.default_rng(7)
    n = max(2000, int(200000 * scale))
    res = {}
    with np.errstate(all="ignore"):
        # multiplication (general, including subnormal and zero results, and overflow)
        a = mk(rng.integers(-600, 513, n), rmant(rng, n))
        b = mk(rng.integers(-600, 513, n), rmant(rng, n))
        got = np.multiply(a, b).tolist()
        res["np.multiply(a, b) != correctly rounded a b"] = sum(
            g != mul_ref(x, y) for g, x, y in zip(got, a.tolist(), b.tolist()))
        sub = sum(0.0 < g < SN_F for g in got)
        got = np.multiply(a, a).tolist()
        res["np.multiply(a, a) != correctly rounded a^2"] = sum(g != mul_ref(x, x) for g, x in zip(got, a.tolist()))
        # addition
        c = mk(rng.integers(-1074, 1024, n), rmant(rng, n))
        d = c * mk(-rng.integers(0, 61, n), rmant(rng, n))
        got = np.add(c, d).tolist()
        res["np.add(a, b) != correctly rounded a + b"] = sum(
            g != add_ref(x, y) for g, x, y in zip(got, c.tolist(), d.tolist()))
        # square root
        got = np.sqrt(c).tolist()
        res["np.sqrt(a) != correctly rounded sqrt(a)"] = sum(g != sqrt_ref(x) for g, x in zip(got, c.tolist()))
        res["math.sqrt(a) != correctly rounded sqrt(a)"] = sum(math.sqrt(x) != sqrt_ref(x) for x in c.tolist())
        # frexp
        f, e = np.frexp(c)
        bad = 0
        for fv, ev, v in zip(f.tolist(), e.tolist(), c.tolist()):
            nv, sv = f2d(v)
            nf, sf = f2d(fv)
            ok = (0.5 <= fv < 1.0) and ev == nv.bit_length() + sv and \
                nf << max(0, sf + ev - sv) == nv << max(0, sv - sf - ev)
            bad += not ok
        res["np.frexp(v): f not in [1/2, 1) or f 2^e != v (v > 0, subnormals included)"] = bad
        # ldexp: exact results, overflow, subnormal results
        v = mk(rng.integers(-1074, 1024, n), rmant(rng, n))
        ev = np.frexp(v)[1]
        k1 = (rng.integers(-1021, 1025, n) - ev).astype(np.int32)       # result in the normal range: exact
        got = np.ldexp(v, k1).tolist()
        bad = 0
        for g, x, kk in zip(got, v.tolist(), k1.tolist()):
            nx, sx = f2d(x)
            w, inex = rnd(nx, sx + kk)
            bad += (g != w) or inex
        res["np.ldexp(v, k) != v 2^k when v 2^k is a normal binary64 number"] = bad
        k2 = (rng.integers(1025, 1200, n) - ev).astype(np.int32)        # v 2^k >= 2^1024
        res["np.ldexp(v, k) != inf when v 2^k >= 2^1024"] = int(np.sum(np.ldexp(v, k2) != np.inf))
        k3 = (rng.integers(-1130, -1021, n) - ev).astype(np.int32)      # result below 2^-1022
        got = np.ldexp(v, k3).tolist()
        cnt = Counter()
        for g, x, kk in zip(got, v.tolist(), k3.tolist()):
            nx, sx = f2d(x)
            w0, inex = rnd(nx, sx + kk, 0)
            if not inex:
                cnt["exactly representable"] += 1
                cnt["exactly representable but not returned"] += g != w0
                continue
            cnt["inexact"] += 1
            w1 = rnd(nx, sx + kk, 1)[0]
            w2 = rnd(nx, sx + kk, 2)[0]
            cnt["inexact: == round to nearest even"] += g == w0
            cnt["inexact: == truncation"] += g == w1
            cnt["inexact: == rounded up"] += g == w2
            cnt["inexact: not a neighbour (error >= 2^-1074)"] += g not in (w1, w2)
        res["np.ldexp subnormal results"] = dict(cnt)
        # nextafter
        t = np.concatenate([np.array([0.0, 5e-324, SN_F - 5e-324, SN_F - 1e-323]),
                            rng.integers(0, 1 << 52, n).astype(np.float64) * 5e-324])
        got = np.nextafter(t, np.inf).tolist()
        res["np.nextafter(t, inf) != t + 2^-1074 for 0 <= t < 2^-1022"] = sum(
            g != next_up(x) for g, x in zip(got, t.tolist()))
    P("n per check = %d (products with a subnormal nonzero result: %d)" % (n, sub))
    fails = 0
    for k, vv in res.items():
        if isinstance(vv, dict):
            P("  %s:" % k)
            for kk, v2 in vv.items():
                P("      %-52s %d" % (kk, v2))
            fails += vv.get("inexact: not a neighbour (error >= 2^-1074)", 0)
            fails += vv.get("exactly representable but not returned", 0)
        else:
            P("  %-78s %d" % (k, vv))
            if not k.startswith("math.sqrt"):
                fails += vv
    return res, fails


# ----------------------------------------------------------------------------------------------------------------
# special inputs (task 2): zeros, non-finite, dtypes, shapes, scalars, large arrays, abs_up
# ----------------------------------------------------------------------------------------------------------------
def bits_equal(a, b):
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.all(a.view(np.uint64) == b.view(np.uint64)))


def cplx(re, im):
    z = np.empty(len(re), np.complex128)
    z.real = re
    z.imag = im
    return z


def exact_i_fail(re, im, m):
    """number of entries with m^2 < re^2 + im^2 (exact); nan and inf entries of m are not counted"""
    bad = 0
    for a, b, v in zip(np.asarray(re, dtype=np.float64).tolist(), np.asarray(im, dtype=np.float64).tolist(),
                       np.asarray(m, dtype=np.float64).tolist()):
        if v != v or v == INF:
            continue
        bad += F(v) ** 2 < F(a) ** 2 + F(b) ** 2
    return bad


def special_tests(mod, scale, P):
    import scipy.sparse as sp
    mu, au = mod["modulus_up"], mod["abs_up"]
    rows = []
    notes = []

    def rec(desc, n, nfail):
        rows.append((desc, n, nfail))

    mags = [5e-324, 1e-323, 2.0 ** -1022, stepf(2.0 ** -1022, -1), 2.0 ** -600, 2.0 ** -500, stepf(2.0 ** -500, -1),
            1e-10, 0.1, 1.0, 3.0, 2.0 ** 500, stepf(2.0 ** 500, 1), 2.0 ** 600, 2.0 ** 1023, OMEGA_F]
    # S1 zeros (both signs of the zero and of the other component); (iv) and the sign of the result
    re, im, want = [], [], []
    for v in mags + [0.0]:
        for zr in (0.0, -0.0):
            for sg in (1.0, -1.0):
                re += [zr, sg * v]
                im += [sg * v, zr]
                want += [v, v]
    with np.errstate(all="raise"):
        m = mu(cplx(re, im))
    nf = sum(not (g == w and math.copysign(1.0, g) == 1.0) for g, w in zip(m.tolist(), want))
    nf += sum(not (mu(complex(a, b)) == w) for a, b, w in zip(re, im, want))
    rec("S1 zero component, +0 and -0, array and scalar calls: m = |other component| exactly, sign +", 2 * len(re), nf)
    # S2 non-finite components
    vals = [0.0, -0.0, 1.0, -3.0, 5e-324, 2.0 ** 600, OMEGA_F, INF, -INF, math.nan]
    re = [a for a in vals for b in vals]
    im = [b for a in vals for b in vals]
    with np.errstate(all="raise"):
        m = mu(cplx(re, im)).tolist()
    with np.errstate(all="ignore"):
        ab = np.abs(cplx(re, im)).tolist()
    nf = n = 0
    for a, b, g, h in zip(re, im, m, ab):
        if math.isinf(a) or math.isinf(b):
            n += 1
            nf += not (g == INF and h == INF)
            nf += not (mu(complex(a, b)) == INF)
        elif math.isnan(a) or math.isnan(b):
            n += 1
            nf += not (g != g and h != h)
            nf += not math.isnan(mu(complex(a, b)))
    rec("S2 inf or nan component (inf wins over nan and over 0; nan otherwise; same as np.abs); no FP error raised "
        "under np.errstate(all='raise')", n, nf)
    # S3 complex64
    rng = np.random.default_rng(31)
    n = max(2000, int(50000 * scale))
    with np.errstate(all="ignore"):
        r32 = (rmant(rng, n) * 2.0 ** rng.integers(-149, 128, n)).astype(np.float32)
        i32 = (rmant(rng, n) * 2.0 ** rng.integers(-149, 128, n)).astype(np.float32)
    ext32 = np.array([np.finfo(np.float32).max, np.finfo(np.float32).tiny, 1e-45, 1.0, 0.0], dtype=np.float32)
    r32 = np.concatenate([r32, np.repeat(ext32, 5)])
    i32 = np.concatenate([i32, np.tile(ext32, 5)])
    z64 = np.empty(r32.size, np.complex64)
    z64.real = r32
    z64.imag = i32
    m = mu(z64)
    nf = (m.dtype != np.float64) + (not bits_equal(m, mu(z64.astype(np.complex128)))) + exact_i_fail(r32, i32, m)
    rec("S3 complex64 input: float64 result, bit-identical with the complex128 call, (i) exact", r32.size, int(nf))
    # S4 scalars and 0-d arrays
    nf = n = 0
    for z in NUMPY_COUNTEREXAMPLES + [3 + 4j, complex(5e-324, 5e-324), complex(OMEGA_F, 1.0), complex(0.0, -2.0)]:
        ref = mu(np.array([z]))[0]
        for form in (z, np.complex128(z), np.array(z)):
            g = mu(form)
            n += 1
            nf += not (type(g) is float and (g == ref))
    g = mu(np.complex64(3 + 4j))
    n += 1
    nf += not (type(g) is float and g == mu(3 + 4j))
    rec("S4 scalar input (complex, np.complex128, np.complex64, 0-d array): a Python float equal to the array result",
        n, nf)
    # S5 shapes and memory layouts; the input is not modified
    n = 3 * 4 * 5
    z = cplx(rmant(rng, n) * 2.0 ** rng.integers(-1074, 1023, n), rmant(rng, n) * 2.0 ** rng.integers(-1074, 1023, n))
    flat = mu(z)
    nf = 0
    zc = z.reshape(3, 4, 5)
    zf = np.asfortranarray(zc)
    nf += not bits_equal(mu(zc), flat.reshape(3, 4, 5))
    nf += not bits_equal(mu(zf), flat.reshape(3, 4, 5))
    nf += not bits_equal(mu(zc[:, ::2, ::-1]), flat.reshape(3, 4, 5)[:, ::2, ::-1])
    ro = z.copy()
    ro.setflags(write=False)
    before = ro.tobytes()
    nf += not bits_equal(mu(ro), flat)
    nf += before != ro.tobytes()
    for shp in ((0,), (0, 3), (2, 0, 4)):
        g = mu(np.zeros(shp, np.complex128))
        nf += not (g.shape == shp and g.dtype == np.float64)
    be = z.astype(">c16")
    nf += not bits_equal(mu(be), flat)
    rec("S5 shapes (3-d C and Fortran order, strided view, read-only, empty, big-endian): same values, input "
        "unchanged", 9, nf)
    # S6 real input
    xr = np.array([-1.5, 0.0, -0.0, 5e-324, -OMEGA_F, INF, -INF])
    nf = not bits_equal(mu(xr), np.abs(xr))
    nf += mu(np.float32(-2.5)) != 2.5
    rec("S6 real floating input: np.abs(z), exact", 2, int(nf))
    imin = np.array([np.iinfo(np.int64).min, -5], dtype=np.int64)
    with np.errstate(all="ignore"):
        gi = mu(imin)
    notes.append("real integer input: modulus_up(int64 [-2^63, -5]) = %s (np.abs wraps at the most negative integer; "
                 "the statement's 'np.abs(z), which is exact' holds for floating dtypes)" % gi.tolist())
    zo = np.array(NUMPY_COUNTEREXAMPLES, dtype=object)
    try:
        go = mu(zo)
    except TypeError as exc:                       # the repository code rejects object arrays (since the review)
        go = None
        notes.append("object-dtype array of Python complex numbers: rejected with TypeError (%s)" % exc)
    if go is not None:
        nobj = max(2000, int(100000 * scale))
        xo = rmant(rng, nobj)
        yo = rmant(rng, nobj) * 2.0 ** -rng.integers(0, 4, nobj)
        zobj = np.array([complex(a, b) for a, b in zip(xo.tolist(), yo.tolist())], dtype=object)
        gobj = np.array(mu(zobj).tolist(), dtype=np.float64)
        bad_obj = exact_i_fail(xo, yo, gobj)
        notes.append("object-dtype array of Python complex numbers (dtype.kind 'O', not 'c'): modulus_up returns "
                     "np.abs(z), dtype %s; m < |z| in %d of %d random entries and in %d of the 3 NumPy counterexamples "
                     "(outside the statement; the code does not reject it)"
                     % (go.dtype, bad_obj, nobj,
                        exact_i_fail([z.real for z in NUMPY_COUNTEREXAMPLES], [z.imag for z in NUMPY_COUNTEREXAMPLES],
                                     np.array(go.tolist(), dtype=np.float64))))
    notes.append("np.clongdouble on this host: itemsize %d bytes, %d mantissa bits (binary64 here: the path that "
                 "rounds wider components to float64 with astype cannot be exercised on this host)"
                 % (np.dtype(np.clongdouble).itemsize, np.finfo(np.longdouble).nmant))
    # S7 large array: position and alignment independence
    n = max(20003, int(2000003 * scale))
    EM = np.where(rng.random(n) < 0.7, rng.integers(-3, 4, n), rng.integers(-1074, 1024, n))
    x, y = random_pair(rng, n, EM, rng.integers(0, 40, n))
    z = cplx(x, y)
    full = mu(z)
    nf = 0
    nf += not bits_equal(mu(z[1:]), full[1:])
    nf += not bits_equal(mu(z[3:-2]), full[3:-2])
    nf += not bits_equal(mu(z[::-1])[::-1], full)
    nf += not bits_equal(mu(z[::3]), full[::3])
    idx = rng.integers(0, n, 20000)
    single = np.array([mu(z[i:i + 1])[0] for i in idx.tolist()])
    nf += not bits_equal(single, full[idx])
    rec("S7 large array (%d entries, all branches): results independent of offset, order, stride; 20000 entries "
        "equal to one-entry calls" % n, 5, nf)
    # S8 abs_up
    nf = n = 0
    dz = z[:5000].reshape(50, 100)
    n += 1
    nf += not bits_equal(au(dz), mu(dz))
    ii = rng.integers(0, 40, 600)
    jj = rng.integers(0, 40, 600)
    dat = cplx(rmant(rng, 600) * 2.0 ** rng.integers(-3, 4, 600), -rmant(rng, 600) * 2.0 ** rng.integers(-3, 4, 600))
    for maker in (sp.coo_matrix, sp.coo_array):
        A = maker((dat.copy(), (ii.copy(), jj.copy())), shape=(40, 40))
        for fmt in ("coo", "csr", "csc", "bsr", "dia", "lil", "dok"):
            Af = A.asformat(fmt) if fmt != "coo" else A.copy()
            R = au(Af)
            n += 1
            Ac = Af.tocsr()
            Ac.sum_duplicates()
            Rc = sp.csr_matrix(R)
            Rc.sum_duplicates()
            D = Ac.toarray()
            Rd = Rc.toarray()
            ok = (R.shape == Af.shape) and not np.iscomplexobj(Rd) and exact_i_fail(D.real.ravel(), D.imag.ravel(),
                                                                                    Rd.ravel()) == 0
            ok = ok and bool(np.all((Rd != 0) == (D != 0)))
            nf += not ok
    Ar = sp.csr_matrix(np.array([[-1.5, 0.0], [2.0, -3.0]]))
    n += 1
    nf += not bits_equal(au(Ar).toarray(), np.abs(Ar.toarray()))
    Adup = sp.coo_matrix((dat.copy(), (ii.copy(), jj.copy())), shape=(40, 40))
    nnz_before = Adup.nnz
    au(Adup)
    notes.append("abs_up on a COO matrix with duplicate entries sums the duplicates of the argument in place "
                 "(stored entries %d -> %d), as its docstring says; the moduli bounded are those of the summed "
                 "(rounded) entries" % (nnz_before, Adup.nnz))
    rec("S8 abs_up: dense = modulus_up; sparse (coo, csr, csc, bsr, dia, lil, dok; matrix and array): same pattern, "
        "real, entries >= |a_ij| exactly; real sparse = abs", n, nf)
    tot = 0
    for desc, n, nfail in rows:
        P("  %-6s checks %-7d failures %-4d %s" % (desc.split()[0], n, nfail, desc[len(desc.split()[0]) + 1:]))
        tot += nfail
    for nt in notes:
        P("  note: " + nt)
    return rows, notes, tot


# ----------------------------------------------------------------------------------------------------------------
# corollary
# ----------------------------------------------------------------------------------------------------------------
def sqrt_upper(q, K=200):
    """a rational >= sqrt(q) for a Fraction q >= 0 (relative excess below 2^-190)"""
    p, d = q.numerator, q.denominator
    return F(math.isqrt((p * d) << (2 * K)) + 1, d << K)


def corollary_tests(mod, scale, P):
    mu = mod["modulus_up"]
    rng = np.random.default_rng(99)
    out = Counter()
    first = {}

    def family(tag, elo, ehi, trials_scale):
        for w, trials in ((1, 200), (2, 200), (3, 200), (5, 100), (10, 100), (100, 40), (1000, 8)):
            trials = max(2, int(trials * scale * trials_scale))
            gam = F(w, 2 ** 53) / (1 - F(w, 2 ** 53))
            for _ in range(trials):
                a = cplx(rmant(rng, w) * 2.0 ** rng.integers(elo, ehi, w), rmant(rng, w) * 2.0 ** rng.integers(elo, ehi, w))
                x = cplx(rmant(rng, w) * 2.0 ** rng.integers(elo, ehi, w), rmant(rng, w) * 2.0 ** rng.integers(elo, ehi, w))
                ma, mx = mu(a), mu(x)
                if w == 1:
                    ma, mx = np.atleast_1d(ma), np.atleast_1d(mx)
                prods_exact = [F(p) * F(q) for p, q in zip(ma.tolist(), mx.tolist())]
                s_mm = sum(prods_exact)
                s_true_up = sum(sqrt_upper((F(p.real) ** 2 + F(p.imag) ** 2) * (F(q.real) ** 2 + F(q.imag) ** 2))
                                for p, q in zip(a.tolist(), x.tolist()))
                with np.errstate(all="ignore"):
                    pr = (ma * mx).tolist()
                    acc1 = 0.0
                    for v in pr:
                        acc1 += v
                    acc2 = 0.0
                    for i in rng.permutation(w).tolist():
                        acc2 += pr[i]
                    acc3 = float(np.sum(ma * mx))
                    acc4 = float(np.dot(ma, mx))
                    na, nx = np.atleast_1d(np.abs(a)), np.atleast_1d(np.abs(x))
                    acc5 = 0.0
                    for v in (na * nx).tolist():
                        acc5 += v
                if not tag.startswith("CONTROL"):
                    out["CONTROL numpy.abs in place of modulus_up, " + tag + ": sums checked"] += 1
                    if F(acc5) < (1 - gam) * s_true_up:
                        out["CONTROL numpy.abs in place of modulus_up, " + tag
                            + ": fl(sum abs abs) < (1 - gamma_w) sum |a||x|"] += 1
                for nm, acc in (("left to right", acc1), ("random order", acc2), ("np.sum (pairwise)", acc3),
                                ("np.dot (BLAS)", acc4)):
                    out[tag + ": sums checked"] += 1
                    if acc == INF:
                        continue
                    if F(acc) < (1 - gam) * s_mm:
                        out[tag + ": fl(sum m m) < (1 - gamma_w) sum m m"] += 1
                    if F(acc) < (1 - gam) * s_true_up:
                        out[tag + ": fl(sum m m) < (1 - gamma_w) sum |a||x|"] += 1
                        first.setdefault(tag, "w=%d, %s: fl=%s" % (w, nm, acc.hex()))
                    if F(acc) < (1 - w * UF) * s_true_up:
                        out[tag + ": fl(sum m m) < (1 - w u) sum |a||x|  (stronger than claimed)"] += 1

    family("no underflow, exponents -3..3", -3, 4, 1.0)
    family("no underflow, exponents -300..300", -300, 301, 1.0)
    family("CONTROL products underflow, exponents -620..-480", -620, -479, 0.5)
    for k in sorted(out):
        P("  %-108s %d" % (k, out[k]))
    for k, v in first.items():
        P("  first violation in '%s': %s" % (k, v))
    # explicit counterexamples to the corollary as stated (no hypothesis on underflow, finite moduli not required)
    a = complex(2.0 ** -600, 0.0)
    x = complex(2.0 ** -600, 0.0)
    ma, mx = mu(a), mu(x)
    prod = ma * mx
    ex1 = (ma == 2.0 ** -600 and mx == 2.0 ** -600 and prod == 0.0)
    P("  explicit: a = x = 2^-600: m_a = %s, m_x = %s (exact moduli), fl(m_a m_x) = %s, |a||x| = 2^-1200 > 0; "
      "fl < (1 - gamma_1)|a||x|: %s" % (ma.hex(), mx.hex(), prod, ex1))
    a = complex(2.0 ** -600, 2.0 ** -600)
    ma = mu(a)
    prod = ma * ma
    ex2 = F(prod) < (1 - F(1, 2 ** 53) / (1 - F(1, 2 ** 53))) * 2 * F(2.0 ** -600) ** 2
    P("  explicit: a = x = 2^-600 (1 + i): m_a = %s, fl(m_a m_a) = %s, |a||x| = 2^-1199; fl < (1 - gamma_1)|a||x|: %s"
      % (ma.hex(), prod, ex2))
    a = complex(OMEGA_F, OMEGA_F)
    ma = mu(a)
    mx = mu(complex(0.0, 0.0))
    with np.errstate(all="ignore"):
        prod = np.float64(ma) * np.float64(mx)
    ex3 = bool(ma == INF and mx == 0.0 and prod != prod)
    P("  explicit: a = Omega (1 + i), x = 0: m_a = %s, m_x = %s, fl(m_a m_x) = %s (inf * 0), |a||x| = 0; "
      "the comparison fl >= ... is false for nan: %s" % (ma, mx, prod, ex3))
    return out, (ex1, ex2, ex3)


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def load_code(path):
    """The names of the code under test: certify_ed.fp of this repository (path None), or an excerpt executed with
    U = 2^-53 supplied (the first run of 2026-11-18 used the excerpt fp_modulus_up_code.py)."""
    if path is None:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        import certify_ed.fp as fp
        return vars(fp)
    ns = {"__name__": "fp_excerpt", "U": 2.0 ** -53}
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, path, "exec"), ns)
    return ns


def shard_work(work, i, nsh):
    """exact checks of the chunks c with c % nsh == i of every set stored in work/sets.npz"""
    data = np.load(os.path.join(work, "sets.npz"))
    with open(os.path.join(work, "names.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    out = {}
    ch = 2000
    for si, name in enumerate(names):
        x, y, m, am = (data["%s%d" % (k, si)] for k in "xyma")
        tot = Counter()
        EXs = {}
        e3 = [(-INF, ""), (INF, ""), (-INF, "")]
        for ci, st in enumerate(range(0, x.size, ch)):
            if ci % nsh != i:
                continue
            _, C, EX, (a, b, c) = check_chunk((name, x[st:st + ch].tolist(), y[st:st + ch].tolist(),
                                              m[st:st + ch].tolist(), am[st:st + ch].tolist()))
            tot.update(C)
            for k, v in EX.items():
                lst = EXs.setdefault(k, [])
                lst.extend(v[:max(0, 3 - len(lst))])
            if a[0] > e3[0][0]:
                e3[0] = a
            if b[0] < e3[1][0]:
                e3[1] = b
            if c[0] > e3[2][0]:
                e3[2] = c
        out[name] = {"C": dict(tot), "EX": EXs, "ext": [list(v) for v in e3]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="internal: i/N")
    ap.add_argument("--work", default=None, help="directory for the intermediate files")
    ap.add_argument("--code", default=None, help="an excerpt to test instead of certify_ed.fp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.shard:
        i, nsh = (int(v) for v in args.shard.split("/"))
        res = shard_work(args.work, i, nsh)
        with open(os.path.join(args.work, "part_%d.json" % i), "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        return 0
    lines = []

    def P(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    mod = load_code(args.code)
    mu = mod["modulus_up"]
    assert mod["_GUARD"] == G8 and mod["_SAFE_LO"] == LO and mod["_SAFE_HI"] == HI and mod["_SMALLEST_NORMAL"] == SN_F
    P("adversarial_modulus_up.py -- independent exact check of Lemma M and its corollary (D-12; UNVERIFIED (D-12))")
    P("code under test: %s (_GUARD == 1 + 2^-50: %s)"
      % ("certify_ed.fp of " + REPO if args.code is None else os.path.basename(args.code) + " (exec of the excerpt, "
         "U = 2^-53 supplied)", mod["_GUARD"] == G8))
    try:
        feats = np._core._multiarray_umath.__cpu_features__
        simd = ", ".join(k for k in ("AVX2", "FMA3", "AVX512F", "AVX512_SKX") if feats.get(k))
    except Exception:
        simd = "?"
    import scipy
    P("host: Python %s, NumPy %s, SciPy %s, %s, CPU features in use: %s; scale %.3g, workers %d"
      % (platform.python_version(), np.__version__, scipy.__version__, platform.platform(), simd, args.scale,
         args.workers))
    P()
    P("== 0. The observations of the statement's 'Why' paragraph (numpy.abs on this host) ==")
    import mpmath
    mpmath.mp.prec = 400
    stated = (-2.466, -2.462, 2.269)
    why_ok = True
    for z, st in zip(NUMPY_COUNTEREXAMPLES, stated):
        za = mpmath.sqrt(mpmath.mpf(z.real) ** 2 + mpmath.mpf(z.imag) ** 2)
        ra = float((mpmath.mpf(float(np.abs(np.complex128(z)))) / za - 1) * 2 ** 53)
        rm = float((mpmath.mpf(mu(z)) / za - 1) * 2 ** 53)
        ok = abs(ra - st) < 0.0015
        why_ok &= ok
        P("  z = %r: (np.abs(z)/|z| - 1)/u = %+.4f (statement: %+.3f; %s); (modulus_up(z)/|z| - 1)/u = %+.4f"
          % (z, ra, st, "reproduced" if ok else "NOT reproduced", rm))
    tiny = float(np.abs(np.complex128(complex(5e-324, 5e-324))))
    why_ok &= tiny == 5e-324
    P("  np.abs(2^-1074 (1 + i)) = %s 2^-1074 (statement: 1; |z| = 1.41421... 2^-1074); modulus_up gives %s 2^-1074"
      % (tiny / 5e-324, mu(complex(5e-324, 5e-324)) / 5e-324))
    P()
    P("== 1. Hypotheses FP-M on this host (exact comparison with integer reference arithmetic) ==")
    hres, hfail = host_checks(args.scale, P)
    P()

    # ---- main sets ----
    sets = build_sets(args.scale)
    e9, e9info, e9total = screened_set(mu, args.scale)
    sets.append(e9)
    work = args.work or tempfile.mkdtemp(prefix="adv_modulus_up_")
    os.makedirs(work, exist_ok=True)
    order = [nm for nm, _, _ in sets]
    arrs = {}
    for si, (name, x, y) in enumerate(sets):
        z = cplx(x, y)
        m = mu(z)                                   # the code under test, one array call per set
        with np.errstate(all="ignore"):
            am = np.abs(z)
        arrs["x%d" % si], arrs["y%d" % si], arrs["m%d" % si], arrs["a%d" % si] = x, y, m, am
    np.savez(os.path.join(work, "sets.npz"), **arrs)
    with open(os.path.join(work, "names.json"), "w", encoding="utf-8") as fh:
        json.dump(order, fh)
    nsh = max(1, args.workers)
    parts = []
    if nsh > 1:
        procs = []
        for i in range(nsh):
            log = open(os.path.join(work, "shard_%d.log" % i), "w")
            procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "--shard", "%d/%d" % (i, nsh),
                                            "--work", work], stdout=log, stderr=subprocess.STDOUT), log))
        for i, (pr, log) in enumerate(procs):
            rc = pr.wait()
            log.close()
            if rc != 0:
                raise RuntimeError("shard %d failed, see %s" % (i, os.path.join(work, "shard_%d.log" % i)))
        for i in range(nsh):
            with open(os.path.join(work, "part_%d.json" % i), encoding="utf-8") as fh:
                parts.append(json.load(fh))
    else:
        parts.append(shard_work(work, 0, 1))
    per_set = {nm: Counter() for nm in order}
    examples = {}
    ext = {"max_rel": (-INF, ""), "min_rel": (INF, ""), "max_add": (-INF, "")}
    for part in parts:
        for nm in order:
            d = part[nm]
            per_set[nm].update(d["C"])
            for k, v in d["EX"].items():
                lst = examples.setdefault(k, [])
                for sx in v:
                    if len(lst) < 4:
                        lst.append("[%s] %s" % (nm.split()[0], sx))
            a, b, c = d["ext"]
            if a[0] > ext["max_rel"][0]:
                ext["max_rel"] = tuple(a)
            if b[0] < ext["min_rel"][0]:
                ext["min_rel"] = tuple(b)
            if c[0] > ext["max_add"][0]:
                ext["max_add"] = tuple(c)
    print("  ... exact checks done, %.0f s" % (time.time() - t0), flush=True)
    total = Counter()
    for c in per_set.values():
        total.update(c)

    P("== 2. Lemma M on the code, exact (Fractions) ==")
    P("sets:")
    for nm in order:
        P("  %s" % nm)
    P("  " + e9info)
    P()
    hdr = ("set", "cases", "(i)", "(ii)+2^-1073", "(ii)|z|>=2^-1022", "(iii)", "(iv)", "m=inf", "m<2^-1022", "code!=ref")
    P("violations / checked, per set (code under test):")
    P("  %-4s %8s %6s %13s %18s %13s %13s %7s %10s %9s" % hdr)

    def row(tag, c):
        P("  %-4s %8d %6d %13d %11d/%-7d %6d/%-6d %6d/%-6d %7d %10d %9d"
          % (tag, c["n"], c["i_viol"], c["iia_viol"], c["iib_viol"], c["iib_checked"], c["iii_viol"],
             c["iii_checked"], c["iv_viol"], c["iv_checked"], c["code_inf"], c["code_m_below_2^-1022"],
             c["code_ne_ref_RN"]))

    for nm in order:
        row(nm.split()[0], per_set[nm])
    row("ALL", total)
    P("  (code!=ref: entries where the code differs bit for bit from the exact reference model with a correctly"
      " rounded ldexp; code m = nan: %d)" % total["code_nan"])
    P("lower margin, not claimed by the statement (finite m, both components nonzero, %d cases): m < |z|(1 + 4u): %d;"
      " m < |z|(1 + 5u): %d; m < |z|(1 + 6u) [control]: %d"
      % (total["LOW_checked"], total["LOW_m<|z|(1+4u)"], total["LOW_m<|z|(1+5u)"], total["LOW_m<|z|(1+6u)"]))
    P()
    P("cases of the proof reached (reference model, all sets):")
    for k in sorted(k for k in total if k.startswith("tag:")):
        P("  %-75s %d" % (k[4:], total[k]))
    P("  |z| < 2^-1022: %d cases" % total["z_below_2^-1022"])
    P()
    P("reference model with other np.ldexp roundings allowed by FP-M (violations, all sets):")
    for vn in ("ref_RN", "ref_ldexp_trunc", "ref_ldexp_up"):
        P("  %-18s (i) %d   (ii)+2^-1073 %d   (ii)|z|>=2^-1022 %d   (iii) %d   [differs from ref_RN in %d cases; "
          "additive term 2^-1074 instead of 2^-1073 violated in %d cases]"
          % (vn, total[vn + ":i_viol"], total[vn + ":iia_viol"], total[vn + ":iib_viol"], total[vn + ":iii_viol"],
             total[vn + ":differs_from_ref_RN"], total[vn + ":ADD_term_1*2^-1074_viol"]))
    P()
    P("== 3. Negative controls (the checks can fail) and sensitivity, per set: violations of (i) ==")
    ctl = ["NC_guard_1+2u", "NC_no_guard", "NC_no_subnormal_step", "NC_no_subnormal_step_ldexp_trunc",
           "NC_no_scaling", "NC_flush_to_zero_products", "np.abs", "INFO_guard_1+4u", "INFO_guard_1+6u"]
    P("  %-4s %8s " % ("set", "cases") + " ".join("%s" % c for c in ctl))
    for nm in order + ["ALL"]:
        c = total if nm == "ALL" else per_set[nm]
        P("  %-4s %8d " % (nm.split()[0], c["n"]) + " ".join("%*d" % (len(cn), c[cn + ":i_viol"]) for cn in ctl))
    P("  NC_no_scaling also returns inf where (iii) forbids it: %d cases" % total["NC_no_scaling:iii_viol"])
    P("  tightness controls on the code's m (cases with |z| >= 2^-1022, %d checked): m > |z|(1 + 8u): %d;"
      " (1 + 9u): %d; (1 + 10u): %d; (1 + 11u): %d; (1 + 12u) [the claim]: %d"
      % (total["iib_checked"], total["TIGHT_m>|z|(1+8u)"], total["TIGHT_m>|z|(1+9u)"], total["TIGHT_m>|z|(1+10u)"],
         total["TIGHT_m>|z|(1+11u)"], total["iib_viol"]))
    P("  additive-term controls on the code's m (cases with |z| < 2^-1022, %d): m > |z|(1 + 12u) + 0: %d;"
      " + 2^-1074: %d; + 2^-1073 [the claim]: counted in (ii) above"
      % (total["z_below_2^-1022"], total["ADD_term_0*2^-1074_viol"], total["ADD_term_1*2^-1074_viol"]))
    P()
    P("extremes on the code's m (floats evaluated from exact rationals; not pass/fail):")
    P("  max (m/|z| - 1)/u over |z| >= 2^-1022: %.4f at %s" % ext["max_rel"])
    P("  min (m/|z| - 1)/u over |z| >= 2^-1022, both components nonzero: %.4f at %s" % ext["min_rel"])
    P("  max (m - |z|(1 + 12u))/2^-1074 over |z| < 2^-1022, both components nonzero: %.4f at %s" % ext["max_add"])
    P()
    P("examples (first few per kind):")
    for k in sorted(examples):
        for s in examples[k][:2]:
            P("  %-36s %s" % (k, s))
    P()
    P("== 4. Special inputs (code against statement) ==")
    srows, snotes, sfail = special_tests(mod, args.scale, P)
    P()
    P("== 5. Corollary: fl(sum_j m_ij m_j) >= (1 - gamma_w) sum_j |a_ij||x_j|, exact rational comparison ==")
    cres, cex = corollary_tests(mod, args.scale, P)
    P()

    # ---- expectations ----
    P("== 6. Expectations ==")
    exp = []

    def E(desc, ok):
        exp.append((desc, bool(ok)))
        P("  [%s] %s" % ("ok" if ok else "FAIL", desc))

    E("statement's numpy.abs observations reproduced on this host", why_ok)
    E("FP-M primitives on this host: 0 deviations", hfail == 0)
    E("(i) m >= |z|: 0 violations in %d cases" % total["n"], total["i_viol"] == 0 and total["code_nan"] == 0)
    E("(ii) m <= |z|(1+12u) + 2^-1073: 0 violations; m <= |z|(1+12u) for |z| >= 2^-1022: 0 violations in %d"
      % total["iib_checked"], total["iia_viol"] == 0 and total["iib_viol"] == 0)
    E("(iii) m = inf only if |z|(1+12u) > Omega: 0 violations in %d infinite results" % total["iii_checked"],
      total["iii_viol"] == 0 and total["iii_checked"] > 0)
    E("lower margin m >= |z|(1+4u) for two nonzero components: 0 violations in %d; control 1+6u violated (%d > 0)"
      % (total["LOW_checked"], total["LOW_m<|z|(1+6u)"]),
      total["LOW_m<|z|(1+4u)"] == 0 and total["LOW_m<|z|(1+6u)"] > 0)
    E("(iv) zero component: m = |z| exactly: 0 violations in %d (+ S1)" % total["iv_checked"],
      total["iv_viol"] == 0 and total["iv_checked"] > 0)
    E("code bit-identical with the exact reference model (ldexp to nearest): 0 differences",
      total["code_ne_ref_RN"] == 0)
    E("reference model with truncating / upward ldexp: 0 violations of (i), (ii), (iii)",
      sum(total[v + k] for v in ("ref_ldexp_trunc", "ref_ldexp_up", "ref_RN")
          for k in (":i_viol", ":iia_viol", ":iib_viol", ":iii_viol")) == 0)
    for cn in ("NC_guard_1+2u", "NC_no_guard", "NC_no_subnormal_step", "NC_no_subnormal_step_ldexp_trunc",
               "NC_no_scaling", "NC_flush_to_zero_products", "np.abs"):
        E("negative control %s fails (i): %d violations (> 0 expected)" % (cn, total[cn + ":i_viol"]),
          total[cn + ":i_viol"] > 0)
    E("negative control: tightness constant 1 + 8u is violated (%d > 0)" % total["TIGHT_m>|z|(1+8u)"],
      total["TIGHT_m>|z|(1+8u)"] > 0)
    E("negative control: additive term 0 is violated (%d > 0)" % total["ADD_term_0*2^-1074_viol"],
      total["ADD_term_0*2^-1074_viol"] > 0)
    E("special inputs: 0 failures", sfail == 0)
    nou = [k for k in cres if k.startswith("no underflow") and "<" in k]
    nab = sum(v for k, v in cres.items() if k.startswith("CONTROL numpy.abs") and "<" in k)
    E("corollary control: with numpy.abs in place of modulus_up the inequality fails (%d > 0)" % nab, nab > 0)
    E("corollary, no underflow: 0 violations in %d sums"
      % sum(v for k, v in cres.items() if k.startswith("no underflow") and k.endswith("sums checked")),
      sum(cres[k] for k in nou) == 0)
    E("corollary control: the inequality fails when products underflow (%d > 0) and for inf * 0 (explicit cases: %s)"
      % (cres["CONTROL products underflow, exponents -620..-480: fl(sum m m) < (1 - gamma_w) sum |a||x|"], cex),
      cres["CONTROL products underflow, exponents -620..-480: fl(sum m m) < (1 - gamma_w) sum |a||x|"] > 0
      and all(cex))
    nbad = sum(not ok for _, ok in exp)
    P()
    P("total exact Lemma-M cases: %d (of which %d selected by the screen of %d); expectations not met: %d; "
      "wall time %.0f s" % (total["n"], per_set[order[-1]]["n"], e9total, nbad, time.time() - t0))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"total": dict(total), "per_set": {k: dict(v) for k, v in per_set.items()},
                       "host": {k: v for k, v in hres.items()}, "corollary": dict(cres),
                       "special": [list(r) for r in srows], "special_notes": snotes,
                       "extremes": {k: list(v) for k, v in ext.items()},
                       "expectations": [[d, ok] for d, ok in exp]}, fh, indent=1)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
