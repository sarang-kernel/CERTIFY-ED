"""Computational verification of docs/proofs/thm1_fp-accounting.md (Theorem 1-fl: the floating-point evaluation of
the residual radius), in exact rational arithmetic (fractions.Fraction; square roots bracketed with decimal).

Lambda  the scalar lemmas (L2), (L4), (L5), (L6) on the grid n, w <= 2e4 and 2^j (15 <= j <= 26)
R       random dense/sparse, real/complex cases: the stage inequalities (S2)-(S5) and rel >= ||A x - mu x|| / ||x||
Adv     the adversarial families of the document (zero residual, scaling, single nonzero entry, A = 0, diagonal A,
        1 x 1, huge mu, cancellation, mixed real/complex, a spec operator against its dense build)
Usage: python docs/proofs/verify_thm1_fp-accounting.py [--cases 20000] [--seed 6]
"""
from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal, getcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter  # noqa: E402

import certify_ed as ce  # noqa: E402
from certify_ed.construct import SpecOperator  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, residuals  # noqa: E402
from certify_ed.fp import U, gamma, gamma_dot, norm2_lower, norm2_upper  # noqa: E402

getcontext().prec = 120
u = Fraction(U)


def F(a):
    return Fraction(float(a))


def gam(m):
    return m * u / (1 - m * u)


def sqrt_up(q):
    getcontext().rounding = ROUND_CEILING
    d = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
    return Fraction(d) * (1 + Fraction(1, 10 ** 100))


def sqrt_down(q):
    if q <= 0:
        return Fraction(0)
    getcontext().rounding = ROUND_FLOOR
    d = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
    return Fraction(d) * (1 - Fraction(1, 10 ** 100))


def cabs2(z):
    """|z|^2 as an exact Fraction for a binary64 real or complex number."""
    z = complex(z)
    return F(z.real) ** 2 + F(z.imag) ** 2


def norm_bounds(v):
    s = sum(cabs2(t) for t in np.asarray(v).ravel())
    return sqrt_down(s), sqrt_up(s)


def exact_residual(A, x, mu):
    """Entries of r = A x - mu x as exact complex Fractions (re, im)."""
    Ad = A.toarray() if sp.issparse(A) else np.asarray(A)
    n = Ad.shape[0]
    out = []
    for i in range(n):
        re = Fraction(0)
        im = Fraction(0)
        for j in range(n):
            a, b = complex(Ad[i, j]), complex(x[j])
            re += F(a.real) * F(b.real) - F(a.imag) * F(b.imag)
            im += F(a.real) * F(b.imag) + F(a.imag) * F(b.real)
        xr, xi = complex(x[i]).real, complex(x[i]).imag
        re -= F(mu) * F(xr)
        im -= F(mu) * F(xi)
        out.append((re, im))
    return out


def check_scalar_lemmas(counters):
    grid = list(range(1, 20_001)) + [2 ** j for j in range(15, 27)]
    for k in grid:
        g0, g1, g2 = gam(k), gam(k + 1), gam(k + 2)
        ok = g1 >= g0 + u * (1 + g0)                                                            # (L2)
        # (L4) with N = k and N = 2k
        for N in (k, 2 * k):
            gN = gam(N)
            lo = sqrt_down(1 - gN) * (1 - u) * (1 + g2 * (1 - u) ** 2) * (1 + 8 * u) * (1 - u) ** 4
            hi = sqrt_up(1 + gN) * (1 + u) * (1 - g2 * (1 + u) ** 2) * (1 + u) ** 3 / (1 + 8 * u)
            ok &= lo >= 1 + 3 * u and hi <= 1 - 3 * u
        ok &= (1 - g0) * (1 - u) ** 3 * (1 + 4 * u) / (1 - g1 * (1 - u) ** 2) >= 1 + u           # (L5)
        ok &= (1 - u) ** 6 * (1 + 4 * u) * (1 + u / g2) >= 1                                    # (L6)
        counters["L"].add(k > 20_000, ok, f"scalar lemma at k={k}")


def check_case(A, X, mu, counters, adversarial, tag):
    ad = OperatorAdapter(A)
    w = ad.width
    cplx = ad.complex or np.iscomplexobj(X)
    res = residuals(ad, np.asarray([mu], dtype=float), X)
    x = np.asarray(X)[:, 0]
    n = x.size
    Rc, Ec, rel = res.R[:, 0], res.E[:, 0], float(res.rel_bound[0])
    r = exact_residual(A, x, mu)
    r2 = sum(re ** 2 + im ** 2 for re, im in r)
    x2 = sum(cabs2(t) for t in x)
    if x2 == 0:
        counters["R"].add(adversarial, rel == float("inf"), f"zero x {tag}")
        return
    # the final statement: rel^2 ||x||^2 >= ||r||^2
    ok = F(rel) ** 2 * x2 >= r2 if np.isfinite(rel) else True
    counters["R"].add(adversarial, ok, f"rel {tag}: rel={rel}")
    # (S2) absHX_up >= |A||x|
    Ad = A.toarray() if sp.issparse(A) else np.asarray(A)
    absAx = [sum(sqrt_up(cabs2(Ad[i, j]) * cabs2(x[j])) for j in range(n)) for i in range(n)]
    absHX_c = ad.abs_matvec(X)[:, 0]
    gw = gamma(w + 1)
    absHX_up = absHX_c / (1.0 - gw) * (1 + 4 * U)
    ok2 = all(F(absHX_up[i]) >= absAx[i] for i in range(n))
    # (S3) E_i >= |Rc_i - r_i| (exact), using the exact squares
    ok3 = True
    for i in range(n):
        d2 = (F(complex(Rc[i]).real) - r[i][0]) ** 2 + (F(complex(Rc[i]).imag) - r[i][1]) ** 2
        ok3 &= F(Ec[i]) ** 2 >= d2
    counters["S"].add(adversarial, ok2 and ok3, f"S2/S3 {tag}: S2={ok2} S3={ok3}")
    # (S4) norms with the 3u slack
    ok4 = True
    for v in (Rc, Ec, x):
        lo, hi = norm_bounds(v)
        ok4 &= F(norm2_upper(v)) >= hi * (1 + 3 * u) if hi > 0 else F(norm2_upper(v)) >= 0
        ok4 &= F(norm2_lower(v)) <= lo * (1 - 3 * u)
    # (S5) upper >= ||Rc|| + ||E||, rel >= upper / ||x|| (xl <= ||x||)
    nR_lo, nR_hi = norm_bounds(Rc)
    nE_lo, nE_hi = norm_bounds(Ec)
    nx_lo, nx_hi = norm_bounds(x)
    ok5 = F(res.upper[0]) >= nR_hi + nE_hi and F(res.xnorm_lower[0]) <= nx_lo
    if np.isfinite(rel):
        ok5 &= F(rel) * nx_hi >= F(res.upper[0])
    counters["S"].add(adversarial, ok4 and ok5, f"S4/S5 {tag}: S4={ok4} S5={ok5}")
    g = gamma_dot(w + 2, cplx)
    assert g > 0


def random_case(rng):
    n = int(rng.integers(1, 7))
    cplx = rng.random() < 0.3
    M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    A = (M + M.conj().T) / 2
    w, V = np.linalg.eigh(A)
    j = int(rng.integers(0, n))
    kind = rng.choice(["exact", "perturbed", "random", "far"])
    if kind == "exact":
        x, mu = V[:, j], w[j]
    elif kind == "perturbed":
        x, mu = V[:, j] + 1e-9 * rng.standard_normal(n), w[j] + 1e-7
    elif kind == "random":
        x, mu = rng.standard_normal(n), float(rng.standard_normal())
    else:
        x, mu = V[:, j], w[j] + 1e3
    if rng.random() < 0.3:
        A = sp.csr_matrix(A * (np.abs(rng.standard_normal((n, n))) > 0.5) if n > 1 else A)
        A = (A + A.conj().T) / 2 if sp.issparse(A) else A
    return A, np.asarray(x)[:, None], float(mu), f"{kind} n={n} cplx={cplx}"


def adversarial_cases(rng):
    out = []
    A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    out.append((A, np.array([[1.0], [0.0], [-1.0]]), 2.0, "exact eigenpair, zero residual"))         # A x = 2 x exactly
    out.append((A, np.array([[1e150], [1.5e150], [1e150]]), 2.0 - 1.4142135623730951, "huge x"))
    out.append((A, np.array([[1e-150], [1.5e-150], [1e-150]]), 0.5857864376269049, "tiny x (normal products)"))
    out.append((A, np.array([[0.0], [1.0], [0.0]]), 2.0, "single nonzero entry"))
    out.append((np.zeros((3, 3)), np.array([[1.0], [2.0], [3.0]]), 0.3, "A = 0"))
    out.append((np.diag([1.0, 2.0, 3.0]), np.array([[1.0], [1.0], [1.0]]), 2.0, "diagonal A"))
    out.append((np.array([[1.5]]), np.array([[0.7]]), 1.5, "1 x 1"))
    out.append((A, np.array([[1.0], [1.0], [1.0]]), 1e10, "huge mu"))
    out.append((A, np.array([[1.0], [1.0 + 1e-15], [1.0]]), 0.0, "cancellation"))
    Ac = np.array([[2.0, 1j], [-1j, 2.0]])
    out.append((Ac, np.array([[1.0], [1.0]]), 1.0, "complex A, real x"))
    out.append((A, np.array([[1.0 + 1j], [0.5], [-1j]]), 1.0, "real A, complex x"))
    out.append((sp.csr_matrix(A), np.array([[1.0], [1.4142135623730951], [1.0]]), 2.0 - 1.4142135623730951, "sparse"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("L", "R", "S")}
    check_scalar_lemmas(counters)
    for i in range(args.cases):
        A, X, mu, tag = random_case(rng)
        check_case(A, X, mu, counters, False, tag)
        if i % 5000 == 0:
            print(f"  ... {i} random cases", flush=True)
    for A, X, mu, tag in adversarial_cases(rng):
        for _ in range(20):
            check_case(A, X, mu, counters, True, tag)
            # jitter the vector by one ulp in a random entry to vary the roundings
            X = X.copy()
            k = int(rng.integers(0, X.shape[0]))
            direction = np.inf if rng.random() < 0.5 else -np.inf
            if np.iscomplexobj(X):
                X[k, 0] = complex(np.nextafter(X[k, 0].real, direction), X[k, 0].imag)
            else:
                X[k, 0] = np.nextafter(X[k, 0], direction)
    # spec operator (Remark 3): matrix-free application of a spin-1 chain, claim from the dense build
    spec = ce.spin1_heisenberg_chain(3)
    H = ce.build_bitwise(spec).toarray()
    w, V = np.linalg.eigh(H)
    op = SpecOperator(spec)
    for j in range(0, 27, 5):
        ad = OperatorAdapter(op)
        res = residuals(ad, np.asarray([w[j]]), V[:, j][:, None])
        # rel must dominate the residual of the exact spec operator; the dense build differs from it by the
        # builder error (<= gamma_W |H|_abs), so a sufficient check is rel * ||x|| >= ||H_ref x - mu x|| - _builder_shift ||x||
        from certify_ed.certify import _builder_shift
        r_ref = H @ V[:, j] - w[j] * V[:, j]
        need = F(float(np.linalg.norm(r_ref))) * (1 - Fraction(1, 10 ** 12)) - F(_builder_shift(spec)) * F(float(np.linalg.norm(V[:, j])))
        counters["R"].add(True, F(float(res.rel_bound[0])) * F(float(np.linalg.norm(V[:, j]))) * (1 + Fraction(1, 10 ** 12)) >= need,
                          f"spec operator j={j}")
    ok = True
    for key, name in (("L", "scalar lemmas (L2), (L4), (L5), (L6)"), ("R", "rel >= exact radius"),
                      ("S", "stage inequalities (S2)-(S5)")):
        ok &= counters[key].report(name)
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
