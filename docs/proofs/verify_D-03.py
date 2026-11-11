"""Machine check of docs/proofs/D-03_ozaki_draft.md (Theorems G and R, Lemmas S1 to S3, P and A) against exact rational
arithmetic (python-flint fmpq / fmpq_mat).

Run from the repository root: `python docs/proofs/verify_D-03.py [--cases N] [--seed S]`. Prints the case counts and
the number of violations, writes nothing else, and exits with status 1 when a violation was found (the negative
controls must FAIL their check; a control that passes is counted as a violation of the instrument).
"""
import argparse
import os
import sys
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import flint                                        # noqa: E402

from certify_ed import ozaki as oz                  # noqa: E402


def q(x):
    return flint.fmpq(*float(x).as_integer_ratio())


def qmat(A):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return flint.fmpq_mat(A.shape[0], A.shape[1], [q(v) for v in A.ravel()])


def ent(M):
    return [[M[i, j] for j in range(M.ncols())] for i in range(M.nrows())]


def exact_gram(X, Y):
    Xr, Xi, Yr, Yi = (np.real(X).astype(float), np.imag(X).astype(float), np.real(Y).astype(float), np.imag(Y).astype(float))
    re = qmat(Xr).transpose() * qmat(Yr) + qmat(Xi).transpose() * qmat(Yi)
    im = qmat(Xr).transpose() * qmat(Yi) - qmat(Xi).transpose() * qmat(Yr)
    return ent(re), ent(im)


def exact_residual(H, mu, X):
    Hd = H.toarray() if sp.issparse(H) else np.asarray(H)
    M = qmat(np.diag(mu))
    Hr, Hi = qmat(np.real(Hd)), qmat(np.imag(Hd))
    Xr, Xi = qmat(np.real(X)), qmat(np.imag(X))
    return ent(Hr * Xr - Hi * Xi - Xr * M), ent(Hr * Xi + Hi * Xr - Xi * M)


def bad_entries(V, E, ex_re, ex_im):
    bad = 0
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            dr = q(np.real(V[i, j])) - ex_re[i][j]
            di = q(np.imag(V[i, j])) - ex_im[i][j]
            bad += dr * dr + di * di > q(E[i, j]) ** 2
    return int(bad)


def bad_norms(upper, ex_re, ex_im):
    bad = 0
    for c in range(len(upper)):
        s = sum((ex_re[i][c] ** 2 + ex_im[i][c] ** 2 for i in range(len(ex_re))), flint.fmpq(0))
        bad += q(upper[c]) ** 2 < s
    return int(bad)


def columns(rng, n, g, cplx, kind):
    if kind == "mixed":
        X = rng.standard_normal((n, g)) * np.logspace(0, -float(rng.integers(0, 40)), n)[:, None]
        X[rng.integers(0, n, 2), 0] *= 2.0 ** -int(rng.integers(100, 400))
    elif kind == "unit":
        X = np.linalg.qr(rng.standard_normal((max(n, g), g)))[0][:n]
    else:
        X = rng.standard_normal((n, g)) * 2.0 ** int(rng.integers(-300, 300))
    if cplx:
        X = X + 1j * rng.standard_normal((n, g)) * 10.0 ** -float(rng.integers(0, 8))
    return X


def hermitian(rng, n, density, cplx, dyadic):
    A = sp.random(n, n, density=density, random_state=int(rng.integers(1 << 30)), format="csr")
    if dyadic:
        A.data = np.round(A.data * 16) / 16
    if cplx:
        A = A + 1j * sp.random(n, n, density=density, random_state=int(rng.integers(1 << 30)), format="csr")
    A = (A + A.conj().T) * 0.5 + sp.diags(rng.standard_normal(n) * (0.25 if dyadic else 1.0))
    return sp.csr_matrix(A)


def check_split(A, beta, K):
    S = oz.split_columns(A, beta, K)
    tot = qmat(S.remainder)
    ok = True
    for xp, s in zip(S.slices, S.units):
        tot = tot + qmat(xp)
        m = np.ldexp(xp, -s[None, :].astype(np.int32))
        ok &= bool(np.all(m == np.rint(m)) and np.all(np.abs(m) <= 2.0 ** beta))
        for yq in S.slices:
            ok &= qmat(xp).transpose() * qmat(yq) == qmat(xp.T @ yq) if A.shape[0] * 4 ** beta <= 2 ** 53 else True
    return ok and tot == qmat(A)


def scalar_checks():
    """Lemma P's width rule and the Sum2 coefficient of `_Sum2.result`, in exact arithmetic."""
    bad = 0
    ns = list(range(1, 5000)) + [2 ** j + d for j in range(12, 27) for d in (-1, 0, 1) if 2 ** j + d <= oz.MAX_TERMS]
    for n in ns:
        b = oz.split_beta(n)
        bad += not (n * 4 ** b <= 2 ** 53 < n * 4 ** (b + 1))
    try:                                                     # above 2^26 terms the split is refused
        oz.split_beta(oz.MAX_TERMS + 1)
        bad += 1
    except oz.NotApplicable:
        pass
    u = Fraction(1, 2 ** 53)
    for T in list(range(2, 3000)) + [10 ** 4, 10 ** 5, 10 ** 6]:
        g = Fraction(T - 1) * u / (1 - Fraction(T - 1) * u)
        need = g * g / ((1 - g) * (1 - u))
        bad += Fraction(oz.sum2_coefficient(T)) < need
    return bad, len(ns) + 3001


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    viol = {"split": 0, "gram": 0, "coldots": 0, "residual_entries": 0, "residual_norms": 0, "scalar": 0}
    counts = {"random": 0, "adversarial": 0, "refused": 0, "negative_controls": 0}
    # random cases
    for case in range(a.cases):
        kind = ["mixed", "unit", "scaled"][case % 3]
        cplx = case % 4 == 3
        K = int(rng.integers(1, 9))
        try:
            if case % 3 == 0:
                n, g = int(rng.integers(1, 80)), int(rng.integers(1, 5))
                viol["split"] += not check_split(np.real(columns(rng, n, g, False, kind)), int(rng.integers(2, 27)), K)
                X = columns(rng, n, g, cplx, kind)
                Y = None if case % 2 else columns(rng, n, int(rng.integers(1, 4)), cplx, kind)
                r = oz.ozaki_gram(X, Y, max_slices=K)
                viol["gram"] += bad_entries(r.G, r.E, *exact_gram(X, X if Y is None else Y))
            elif case % 3 == 1:
                n, g = int(rng.integers(1, 80)), int(rng.integers(1, 5))
                X, Y = columns(rng, n, g, cplx, kind), columns(rng, n, g, cplx, kind)
                d, e = oz.ozaki_column_dots(X, Y, max_slices=K)
                er, ei = exact_gram(X, Y)
                viol["coldots"] += bad_entries(d[None, :], e[None, :], [[er[c][c] for c in range(g)]],
                                               [[ei[c][c] for c in range(g)]])
            else:
                n = int(rng.integers(2, 60))
                H = hermitian(rng, n, float(rng.choice([0.05, 0.2, 0.6])), cplx, case % 5 == 0)
                w, V = np.linalg.eigh(H.toarray())
                k = min(n, int(rng.integers(1, 4)))
                mu, X = w[:k], V[:, :k] + (1e-8 * rng.standard_normal((n, k)) if case % 2 else 0.0)
                r = oz.ozaki_residual(H if case % 7 else H.toarray(), mu, X, max_slices=K)
                ex = exact_residual(H, mu, X)
                viol["residual_entries"] += bad_entries(r.R, r.E, *ex)
                viol["residual_norms"] += bad_norms(r.upper, *ex)
            counts["random"] += 1
        except oz.NotApplicable:
            counts["refused"] += 1
    # adversarial cases
    adv = []
    n = 2000
    b = oz.split_beta(n)
    adv.append(("worst-case slice magnitudes", np.full((n, 2), 2.0 ** b - 1.0), None))
    adv.append(("entries just below 2^480", np.array([[2.0 ** 479], [-2.0 ** 478]]), None))
    adv.append(("exactly orthogonal columns", np.array([[1.0, 1.0], [1.0, -1.0], [2.0 ** -60, 2.0 ** -60]]), None))
    adv.append(("subnormal entries", np.array([[1.0, 5e-324], [2.0 ** -1060, 2.0 ** -1070], [3.0, 1.0]]), None))
    adv.append(("zero column", np.array([[0.0, 1.0], [0.0, 2.0]]), None))
    adv.append(("rint ties (half units)", np.ldexp(np.array([[1.5], [2.5], [-0.5], [3.0]]), -30) + 1.0, None))
    adv.append(("complex with zero real part", 1j * np.array([[1.0], [1e-9], [3.0]]), None))
    adv.append(("53-bit significands", np.array([[1.0 + 2.0 ** -52], [1.0 - 2.0 ** -53], [np.pi]]), None))
    for _name, X, Y in adv:
        try:
            r = oz.ozaki_gram(X, Y)
            viol["gram"] += bad_entries(r.G, r.E, *exact_gram(X, X if Y is None else Y))
            counts["adversarial"] += 1
        except oz.NotApplicable:
            counts["refused"] += 1
    # adversarial residuals: exact eigenvectors (zero residual), 53-bit mu, dense worst-case row width
    Hd = sp.diags([0.5, -0.25, 3.0, 1.0 + 2.0 ** -52]).tocsr()
    for mu, X in ((np.array([0.5, 3.0]), np.eye(4)[:, [0, 2]]), (np.array([np.pi, -np.e]), np.eye(4)[:, [1, 3]])):
        r = oz.ozaki_residual(Hd, mu, X)
        ex = exact_residual(Hd, mu, X)
        viol["residual_entries"] += bad_entries(r.R, r.E, *ex)
        viol["residual_norms"] += bad_norms(r.upper, *ex)
        counts["adversarial"] += 1
    Hw = np.full((40, 40), 2.0 ** -3 * (1 + 2.0 ** -40))
    Hw = (Hw + Hw.T) / 2
    w, V = np.linalg.eigh(Hw)
    r = oz.ozaki_residual(Hw, w[-2:], V[:, -2:])
    ex = exact_residual(Hw, w[-2:], V[:, -2:])
    viol["residual_entries"] += bad_entries(r.R, r.E, *ex)
    viol["residual_norms"] += bad_norms(r.upper, *ex)
    counts["adversarial"] += 1
    # refusal paths (hypotheses not met): counted, not checked
    for X in (np.array([[2.0 ** 481], [1.0]]), np.array([[1e-300], [1e-301]]), np.array([[np.inf], [1.0]])):
        try:
            oz.ozaki_gram(X)
            viol["gram"] += 1                                   # must refuse
        except oz.NotApplicable:
            counts["refused"] += 1
    sb, sn = scalar_checks()
    viol["scalar"] += sb
    counts["adversarial"] += 2
    # negative controls: each must fail its check
    neg_fail = 0
    bb = oz.split_beta(n) + 4                                   # (1) slices too wide: the BLAS product rounds
    x = rng.integers(2 ** (bb - 1), 2 ** bb, (n, 1)).astype(float)      # integers of bb bits: products of 2 bb bits
    y = rng.integers(2 ** (bb - 1), 2 ** bb, (n, 1)).astype(float)
    neg_fail += qmat(x).transpose() * qmat(y) == qmat(x.T @ y)
    H = hermitian(rng, 30, 0.3, False, False)                   # (2) the plain floating-point residual is not exact
    w, V = np.linalg.eigh(H.toarray())
    ex = exact_residual(H, w[:2], V[:, :2])
    Rfp = H @ V[:, :2] - V[:, :2] * w[None, :2]
    neg_fail += bad_entries(Rfp, np.zeros(Rfp.shape), *ex) == 0
    r = oz.ozaki_residual(H, w[:2], V[:, :2], max_slices=1)     # (3) the remainder bound shrunk 1000-fold
    neg_fail += bad_entries(r.R, r.E * 1e-3, *ex) == 0
    Xg = columns(rng, 50, 3, False, "mixed")                     # (4) the Gram bound shrunk 1000-fold (truncated split)
    rg = oz.ozaki_gram(Xg, max_slices=1)
    neg_fail += bad_entries(rg.G, rg.E * 1e-3, *exact_gram(Xg, Xg)) == 0
    counts["negative_controls"] = 4
    total = sum(viol.values()) + neg_fail
    print(f"D-03 verification: {counts['random']} random + {counts['adversarial']} adversarial cases "
          f"({counts['refused']} refused as hypothesis not met), scalar grid {sn} values; "
          f"{counts['negative_controls']} negative controls, {neg_fail} of them failed to detect")
    print("violations:", viol, "total", total, f"({time.time() - t0:.0f} s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
