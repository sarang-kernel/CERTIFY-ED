"""Computational verification of the C-16 statement (docs/proofs/C16_exact_bracket.md).

Part A (exact ground truth): integer matrices U diag(d) U^-1 with U unimodular and planted eigenvalues d with
multiplicities; exact_charpoly + roots_in_bracket must return the planted count, number of distinct values and
multiplicities for random rational brackets and for adversarial brackets whose ends are eigenvalues (nudging).
Part B (Sturm on known polynomials): products of linear factors with random rational roots and multiplicities,
degree up to 150; the count of distinct roots and multiplicities in random brackets against the known roots.
Part C (sector polynomial identity): for random integer N and positive diagonal D, the polynomial of
sector_charpoly satisfies p(x) = c det(N - x D) at deg + 1 random rational points (exact determinants), with c a
positive constant; and for a Hermitian Gaussian-integer matrix the embedding's multiplicities are all even.
Part D (models): Heisenberg chains N = 6 (periodic and open) and the spin-1 chain N = 4: the exact multiplicity structure in
brackets around every level against mpmath eigenvalues at 60 digits (a cross-check, not a proof).

Usage: python docs/proofs/verify_C16.py [--random N] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402

from certify_ed.checker import ExactMatrix, exact_matrix  # noqa: E402
from certify_ed.exact_degeneracy import (exact_charpoly, roots_in_bracket, sector_charpoly,  # noqa: E402
                                         sturm_count)


def _unimodular(rng: random.Random, n: int, steps: int = 20):
    U = np.eye(n, dtype=object)
    Ui = np.eye(n, dtype=object)
    for _ in range(steps):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        c = rng.randrange(-2, 3)
        U[i] = U[i] + c * U[j]             # row op on U  <->  inverse column op on U^-1
        Ui[:, j] = Ui[:, j] - c * Ui[:, i]
    return U, Ui


def _ex_from_int_matrix(M, im=None, den=1):
    n = M.shape[0]
    if im is None:
        rows, cols = np.nonzero(M != 0)
        return ExactMatrix(rows.astype(np.int64), cols.astype(np.int64), M[rows, cols], np.zeros(rows.size, dtype=object),
                           den, np.ones(n, dtype=object))
    nz = (M != 0) | (im != 0)
    rows, cols = np.nonzero(nz)
    return ExactMatrix(rows.astype(np.int64), cols.astype(np.int64), M[rows, cols], im[rows, cols], den,
                       np.ones(n, dtype=object))


def _truth(d, a: Fraction, b: Fraction):
    vals = {}
    for v in d:
        if a <= v <= b:
            vals[v] = vals.get(v, 0) + 1
    mult = {}
    for m in vals.values():
        mult[m] = mult.get(m, 0) + 1
    return sum(vals.values()), len(vals), mult


def check_A(rng: random.Random, n: int):
    cases = viol = 0
    for _ in range(n):
        dim = rng.randrange(2, 13)
        levels = sorted(set(Fraction(rng.randrange(-20, 21), rng.choice([1, 1, 2, 4])) for _ in range(rng.randrange(1, 5))))
        d = []
        for lv in levels:
            d += [lv] * rng.randrange(1, 4)
        d = d[:dim] if len(d) > dim else d
        dim = len(d)
        # integer matrix: scale by the lcm of denominators (den), similar to diag(den d)
        den = 1
        for v in d:
            den = den * v.denominator // np.gcd(den, v.denominator)
        U, Ui = _unimodular(rng, dim)
        M = U.dot(np.diag(np.array([int(v * den) for v in d], dtype=object))).dot(Ui)
        ex = _ex_from_int_matrix(M, den=den)
        p, den_, factor = exact_charpoly(ex)
        # three brackets: random, ends at eigenvalues (adversarial), and the whole range
        brackets = [(Fraction(rng.randrange(-25, 25), 2), Fraction(rng.randrange(-25, 25), 2))]
        brackets.append((rng.choice(d), rng.choice(d)))
        brackets.append((min(d) - 1, max(d) + 1))
        for a, b in brackets:
            if a > b:
                a, b = b, a
            if a == b:
                b = a + Fraction(1, 3)
            cases += 1
            r = roots_in_bracket(p, a * den_, b * den_, factor)
            tc, td, tm = _truth(d, a, b)
            ok = r["count"] == tc and r["distinct"] == td and r["multiplicities"] == tm
            viol += not ok
    return cases, viol


def check_B(rng: random.Random, n: int):
    import flint
    cases = viol = 0
    x = flint.fmpz_poly([0, 1])
    for _ in range(n):
        roots = {}
        for _ in range(rng.randrange(1, 8)):
            r = Fraction(rng.randrange(-50, 51), rng.randrange(1, 7))
            roots[r] = rng.randrange(1, 5)
        p = flint.fmpz_poly([rng.choice([1, 2, 3, -1])])
        for r, m in roots.items():
            p = p * (r.denominator * x - r.numerator) ** m
        d = [r for r, m in roots.items() for _ in range(m)]
        for _ in range(3):
            a, b = sorted([Fraction(rng.randrange(-60, 61), rng.randrange(1, 5)) for _ in range(2)])
            if a == b:
                b = a + 1
            cases += 1
            res = roots_in_bracket(p, a, b)
            tc, td, tm = _truth(d, a, b)
            viol += not (res["count"] == tc and res["distinct"] == td and res["multiplicities"] == tm)
        # Sturm on the squarefree part directly, with a bracket end at a root rejected
        sq = flint.fmpz_poly([1])
        for r in roots:
            sq = sq * (r.denominator * x - r.numerator)
        r0 = next(iter(roots))
        cases += 1
        try:
            sturm_count(sq, r0, r0 + 1)
            viol += 1
        except ValueError:
            pass
    # a high-degree case
    p = flint.fmpz_poly([1])
    big = [Fraction(k, 3) for k in range(-70, 71)]
    for r in big:
        p = p * (r.denominator * x - r.numerator)           # roots k/3 (reduced fractions: k/3 or integers)
    cases += 1
    res = roots_in_bracket(p, Fraction(-10), Fraction(10))
    viol += not (res["count"] == 61 and res["distinct"] == 61)
    return cases, viol


def check_C(rng: random.Random, n: int):
    import flint
    cases = viol = 0
    for _ in range(n):
        m = rng.randrange(1, 7)
        N = np.zeros((m, m), dtype=object); N[:] = 0
        for i in range(m):
            for j in range(i, m):
                v = rng.randrange(-5, 6)
                N[i, j] = v; N[j, i] = v
        D = np.array([rng.randrange(1, 6) for _ in range(m)], dtype=object)
        den = rng.randrange(1, 5)
        p, den_, factor = sector_charpoly(N, np.zeros((m, m), dtype=object), D, den)
        cases += 1
        ok = factor == 1 and den_ == den and p.degree() == m
        # p(x) = c * det(N - x D) at m + 1 points, c = (-1)^m lead / prod(D): p is lead * det(x I - D^-1 N) and
        # det(x I - D^-1 N) = (-1)^m det(N - x D) / det(D)
        lead = int(p.leading_coefficient())
        prodD = 1
        for v in D:
            prodD *= int(v)
        c = Fraction((-1) ** m * lead, prodD)
        ok &= lead > 0
        for _ in range(m + 1):
            xq = Fraction(rng.randrange(-30, 31), rng.randrange(1, 4))
            xf = flint.fmpq(xq.numerator, xq.denominator)
            Mx = flint.fmpq_mat([[flint.fmpq(int(N[i, j])) - (int(D[i]) * xf if i == j else 0)
                                  for j in range(m)] for i in range(m)])
            det = Mx.det()
            pv = p(flint.fmpq(xq.numerator, xq.denominator))
            ok &= pv == det * flint.fmpq(c.numerator, c.denominator)
        viol += not ok
        # Hermitian Gaussian-integer matrix: the embedding's multiplicities are even and the count is m
        Are = N.copy()
        Aim = np.zeros((m, m), dtype=object); Aim[:] = 0
        for i in range(m):
            for j in range(i + 1, m):
                v = rng.randrange(-3, 4)
                Aim[i, j] = v; Aim[j, i] = -v
        p2, den2, factor2 = sector_charpoly(Are, Aim, D, den)
        cases += 1
        try:
            r = roots_in_bracket(p2, Fraction(-10 ** 6), Fraction(10 ** 6), factor2)
            cplx = any(int(x) != 0 for x in Aim.ravel())
            viol += not (factor2 == (2 if cplx else 1) and r["count"] == m)
        except ValueError:
            viol += 1
    return cases, viol


def check_D():
    """Models against mpmath eigenvalues (cross-check)."""
    import certify_ed as ce
    import mpmath as mp
    from certify_ed.checker import exact_dense
    mp.mp.dps = 60
    cases = viol = 0
    for spec in (ce.heisenberg_chain(6), ce.heisenberg_chain(6, boundary="open"), ce.spin1_heisenberg_chain(4)):
        ex = exact_matrix(spec)
        Hd = exact_dense(ex)
        E, _Q = mp.eigsy(mp.matrix(np.real(Hd).tolist()))
        ev = sorted(E[i] for i in range(E.rows))
        p, den, factor = exact_charpoly(ex)
        # levels by clustering at 1e-30
        levels = []
        for e in ev:
            if levels and abs(e - levels[-1][0]) < mp.mpf(10) ** -30:
                levels[-1][1] += 1
            else:
                levels.append([e, 1])
        for e, m in levels:
            a = Fraction(str(mp.nstr(e - mp.mpf(10) ** -20, 40)))
            b = Fraction(str(mp.nstr(e + mp.mpf(10) ** -20, 40)))
            cases += 1
            r = roots_in_bracket(p, a * den, b * den, factor)
            viol += not (r["count"] == m and r["distinct"] == 1 and r["multiplicities"] == {m: 1})
    return cases, viol


def run(n_random: int = 200, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = {}
    out["A_planted_matrices"] = check_A(rng, n_random)
    out["B_known_polynomials"] = check_B(rng, n_random)
    out["C_sector_identity"] = check_C(rng, max(20, n_random // 4))
    out["D_models"] = check_D()
    out["random_cases"] = out["A_planted_matrices"][0] + out["B_known_polynomials"][0] + out["C_sector_identity"][0]
    out["adversarial_cases"] = out["D_models"][0]
    out["violations"] = sum(v[1] for k, v in out.items() if isinstance(v, tuple))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = run(a.random, a.seed)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0 if r["violations"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
