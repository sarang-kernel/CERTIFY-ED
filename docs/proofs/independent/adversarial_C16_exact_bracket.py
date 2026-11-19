"""Independent adversarial search for C-16 (docs/proofs/C16_exact_bracket.md, exact_degeneracy.py).

Ground truth: rational symmetric / Hermitian matrices with KNOWN eigenvalues and multiplicities, built as
H = Q diag(d) Q^H with Q a rational orthogonal (Householder with a Pythagorean vector) or rational unitary (Gaussian
integer Householder vector) matrix; the characteristic polynomial of den*H (or of its real embedding) is formed
with python-flint exactly as the code does, and roots_in_bracket is checked against the known multiset for
brackets whose ends are roots (nudging), lie between roots 2^-60 apart, contain nothing, contain everything,
or sit at the extremes; multiplicities up to n = 8 (all equal), factor = 2 cases; sturm_count against the known
roots on random rational brackets; the decision rules of _record through exact_bracket on an ExactMatrix of a
small Heisenberg chain (numpy eigenvalues as the oracle only where the spectrum is well separated).

Usage: python docs/proofs/independent/adversarial_C16_exact_bracket.py [--cases 200] [--seed 47]
"""
import argparse
import os
import sys
from fractions import Fraction
from math import lcm

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, report  # noqa: E402
import flint  # noqa: E402
from certify_ed.exact_degeneracy import roots_in_bracket, sturm_count  # noqa: E402


def householder_rational(rng, n, cplx):
    """Q = I - 2 v v^H / (v^H v) with integer (Gaussian integer) v: exactly unitary with rational entries."""
    while True:
        v = [complex(int(rng.integers(-3, 4)), int(rng.integers(-3, 4)) if cplx else 0) for _ in range(n)]
        if any(v):
            break
    vv = sum(Fraction(z.real) ** 2 + Fraction(z.imag) ** 2 for z in v)
    Q = [[(Fraction(1 if i == j else 0), Fraction(0)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            # v_i conj(v_j)
            re = Fraction(v[i].real) * Fraction(v[j].real) + Fraction(v[i].imag) * Fraction(v[j].imag)
            im = Fraction(v[i].imag) * Fraction(v[j].real) - Fraction(v[i].real) * Fraction(v[j].imag)
            Q[i][j] = (Q[i][j][0] - 2 * re / vv, Q[i][j][1] - 2 * im / vv)
    return Q


def build(rng, d, cplx):
    """H = Q diag(d) Q^H as (re, im) Fractions; returns H entries."""
    n = len(d)
    Q = householder_rational(rng, n, cplx)
    H = [[(Fraction(0), Fraction(0)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            re = im = Fraction(0)
            for k in range(n):
                # Q_ik d_k conj(Q_jk)
                a, b = Q[i][k]; c, e = Q[j][k]
                re += d[k] * (a * c + b * e)
                im += d[k] * (b * c - a * e)
            H[i][j] = (re, im)
    return H


def charpoly_of(H, cplx):
    n = len(H)
    den = 1
    for row in H:
        for re, im in row:
            den = lcm(den, re.denominator, im.denominator)
    if cplx:
        M = [[0] * (2 * n) for _ in range(2 * n)]
        for i in range(n):
            for j in range(n):
                re, im = H[i][j]
                a, b = int(re * den), int(im * den)
                M[i][j] += a; M[i + n][j + n] += a; M[i][j + n] -= b; M[i + n][j] += b
        return flint.fmpz_mat(M).charpoly(), den, 2
    M = [[int(H[i][j][0] * den) for j in range(n)] for i in range(n)]
    return flint.fmpz_mat(M).charpoly(), den, 1


def expected(d, a, b):
    """count, distinct, multiplicities of the values d in the closed [a, b]."""
    inside = [v for v in d if a <= v <= b]
    vals = sorted(set(inside))
    mult = {}
    for v in vals:
        m = inside.count(v); mult[m] = mult.get(m, 0) + 1
    return len(inside), len(vals), mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=47)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_r = Tally("roots_in_bracket: count / distinct / multiplicities equal the known spectrum in the used bracket")
    T_s = Tally("sturm_count equals the number of distinct known roots in (a, b)")
    T_n = Tally("nudging: ends on roots move outward by 2^-200 and the used bracket contains the requested one")
    for i in range(a.cases):
        kind = i % 6
        n = int(rng.integers(1, 7))
        cplx = i % 3 == 0
        # eigenvalues: rationals with planted multiplicities, some 2^-60 apart
        base = [Fraction(int(rng.integers(-6, 7)), 2) for _ in range(n)]
        if kind == 1:
            base = [Fraction(1)] * n                                  # all equal: one root of multiplicity n
        if kind == 2 and n >= 2:
            base[1] = base[0] + Fraction(1, 2 ** 60)                  # two roots 2^-60 apart
        if kind == 3 and n >= 3:
            base[1] = base[0]; base[2] = base[0] + Fraction(1, 2 ** 60)
        d = sorted(base)
        H = build(rng, d, cplx)
        p, den, factor = charpoly_of(H, cplx)
        # brackets
        if kind == 4:
            lo, hi = d[0] - 1, d[-1] + 1                              # everything
        elif kind == 5:
            lo, hi = d[-1] + 1, d[-1] + 2                             # nothing
        elif kind in (2, 3) and n >= 2:
            lo, hi = d[0] - Fraction(1, 2 ** 70), d[0] + Fraction(1, 2 ** 61)   # between the two close roots
        else:
            j0 = int(rng.integers(0, n)); j1 = int(rng.integers(j0, n))
            lo, hi = d[j0] - Fraction(int(rng.integers(0, 2)), 4), d[j1] + Fraction(int(rng.integers(0, 2)), 4)
        r = roots_in_bracket(p, lo * den, hi * den, factor)
        a_used, b_used = r["bracket_used"][0] / den, r["bracket_used"][1] / den
        cnt, dist, mult = expected(d, a_used, b_used)
        ok = r["count"] == cnt and r["distinct"] == dist and {int(k): v for k, v in r["multiplicities"].items()} == mult
        T_r.add(ok, kind != 0, f"kind={kind} n={n} cplx={cplx} d={[str(v) for v in d]} bracket=[{lo},{hi}] got={r} expected={(cnt, dist, mult)}")
        T_n.add(a_used <= lo and b_used >= hi and (r["nudged"] == (lo in d or hi in d)), kind != 0, f"nudge kind={kind}: used=({a_used},{b_used}) asked=({lo},{hi}) nudged={r['nudged']}")
        # sturm_count on the squarefree parts
        _c, parts = p.factor_squarefree()
        aa, bb = lo * den - Fraction(1, 3), hi * den + Fraction(1, 3)
        for s, j in parts:
            roots_j = sorted({v for v in d if d.count(v) * factor == int(j)})
            if s(flint.fmpq(aa.numerator, aa.denominator)) == 0 or s(flint.fmpq(bb.numerator, bb.denominator)) == 0:
                continue
            T_s.add(sturm_count(s, aa, bb) == sum(1 for v in roots_j if aa < v * den < bb), True, f"sturm j={j} n={n}")
    sys.exit(1 if report([T_r, T_s, T_n]) else 0)


if __name__ == "__main__":
    main()
