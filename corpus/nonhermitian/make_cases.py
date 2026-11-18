"""Generator of the non-Hermitian planted-error cases (E-02): python corpus/nonhermitian/make_cases.py

Every case is A = S J S^-1 with S a unimodular integer matrix and J a dyadic Jordan form (or a 2 x 2 example), so the
matrix entries are exact binary64 numbers and the eigenvalues with their algebraic multiplicities are known exactly; the
claimed eigenpairs come from numpy.linalg.eig (LAPACK geev) and then, for the planted cases, one stated change. The
generator has no random element beyond a fixed seed; it writes cases/<name>.npz (A, eigenvalues, eigenvectors and, where
present, left_eigenvectors) and cases/<name>.json (the expectation and the exact spectrum as fractions).
"""
from __future__ import annotations

import json
import os
from fractions import Fraction

import numpy as np
import scipy.linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")
ALL = ["N1.backward_errors", "N2.eigenvector_inverse", "N3.inclusion", "N4.counting"]


def _unimodular(rng, n, m):
    L = np.tril(rng.integers(-m, m + 1, (n, n)), -1) + np.eye(n, dtype=np.int64)
    Uu = np.triu(rng.integers(-m, m + 1, (n, n)), 1) + np.eye(n, dtype=np.int64)
    Li = np.round(np.linalg.inv(L)).astype(np.int64)
    Ui = np.round(np.linalg.inv(Uu)).astype(np.int64)
    assert np.array_equal(Li @ L, np.eye(n, dtype=np.int64)) and np.array_equal(Ui @ Uu, np.eye(n, dtype=np.int64))
    return L @ Uu, Ui @ Li


def _exact_product(S, J, Si):
    """S J S^-1 with J a list of lists of complex Fractions (pairs); S, Si integer arrays."""
    n = len(J)

    def mul(X, Y):
        return [[(sum(X[i][l][0] * Y[l][j][0] - X[i][l][1] * Y[l][j][1] for l in range(n)),
                  sum(X[i][l][0] * Y[l][j][1] + X[i][l][1] * Y[l][j][0] for l in range(n))) for j in range(n)]
                for i in range(n)]
    Sc = [[(Fraction(int(S[i, j])), Fraction(0)) for j in range(n)] for i in range(n)]
    Sic = [[(Fraction(int(Si[i, j])), Fraction(0)) for j in range(n)] for i in range(n)]
    P = mul(mul(Sc, J), Sic)
    cplx = any(P[i][j][1] != 0 for i in range(n) for j in range(n))
    A = np.empty((n, n), dtype=complex if cplx else float)
    for i in range(n):
        for j in range(n):
            re, im = P[i][j]
            assert Fraction(float(re)) == re and Fraction(float(im)) == im
            A[i, j] = complex(float(re), float(im)) if cplx else float(re)
    return A


def _jordan(diag, superdiag=()):
    n = len(diag)
    J = [[(Fraction(0), Fraction(0)) for _ in range(n)] for _ in range(n)]
    for i, d in enumerate(diag):
        J[i][i] = d
    for i in superdiag:
        J[i][i + 1] = (Fraction(1), Fraction(0))
    return J


def _q(v):
    return [str(v[0]), str(v[1])]


def build():
    rng = np.random.default_rng(20261111)
    cases = []
    F = Fraction
    real5 = [(F(-7, 4), F(0)), (F(-1, 2), F(0)), (F(1, 4), F(0)), (F(3, 2), F(0)), (F(13, 4), F(0))]
    S, Si = _unimodular(rng, 5, 2)
    A_real = _exact_product(S, _jordan(real5), Si)
    cpx = [(F(-1), F(1, 2)), (F(1, 4), F(-3, 4)), (F(2), F(0)), (F(-5, 4), F(-1)), (F(3, 4), F(5, 4))]
    S2, Si2 = _unimodular(rng, 5, 2)
    A_cpx = _exact_product(S2, _jordan(cpx), Si2)
    # real matrix with conjugate pairs: rotation blocks [[a, -c], [c, a]]
    J = _jordan([(F(1, 2), F(0)), (F(1, 2), F(0)), (F(-3, 4), F(0)), (F(-3, 4), F(0))])
    J[0][1], J[1][0] = (F(-2), F(0)), (F(2), F(0))
    J[2][3], J[3][2] = (F(-1, 4), F(0)), (F(1, 4), F(0))
    S3, Si3 = _unimodular(rng, 4, 3)
    A_pairs = _exact_product(S3, J, Si3)
    spec_pairs = [(F(1, 2), F(2)), (F(1, 2), F(-2)), (F(-3, 4), F(1, 4)), (F(-3, 4), F(-1, 4))]
    S4, Si4 = _unimodular(rng, 5, 6)
    A_nn = _exact_product(S4, _jordan(real5), Si4)
    defect = [(F(1, 2), F(0))] * 3 + [(F(-2), F(0)), (F(5, 4), F(0))]
    S5, Si5 = _unimodular(rng, 5, 2)
    A_def = _exact_product(S5, _jordan(defect, superdiag=(0, 1)), Si5)

    def claim(A):
        w, Y, V = la.eig(A, left=True, right=True)
        return w, V, Y

    w, V, Y = claim(A_real)
    cases.append(("nh_correct_real", A_real, w, V, Y, real5, "correct", ALL, [], "control: distinct real eigenvalues"))
    w, V, Y = claim(A_cpx)
    cases.append(("nh_correct_complex", A_cpx, w, V, Y, cpx, "correct", ALL, [], "control: complex matrix"))
    w, V, Y = claim(A_pairs)
    cases.append(("nh_correct_conjugate_pairs", A_pairs, w, V, Y, spec_pairs, "correct", ALL, [],
                  "control: real matrix with two complex conjugate pairs"))
    w, V, Y = claim(A_nn)
    cases.append(("nh_correct_nonnormal", A_nn, w, V, Y, real5, "correct", ALL, [],
                  "control: strongly non-normal similarity (|S| entries up to 6)"))
    a, b, tau = 1.0, 1.0, 2.0 ** -10
    A2 = np.array([[a, b], [0.0, a]])
    cases.append(("nh_bauer_fike_tight", A2, np.array([a + b * tau, a - b * tau]), np.array([[1.0, 1.0], [tau, -tau]]), None,
                  [(F(1), F(0))] * 2, "correct", ALL, [],
                  "control: defective 2 x 2 matrix with two pseudo-eigenpairs at distance b tau = ||V^-1 R||_2 (Bauer-Fike is "
                  "an equality)"))
    w, V, Y = claim(A_real)
    V2, w2 = V.copy(), w.copy()
    V2[:, 1], w2[1] = V2[:, 0], w2[0]
    cases.append(("nh_duplicated_vector", A_real, w2, V2, None, real5, "duplicated_vector", ["N1.backward_errors"],
                  ALL[1:], "the first pair listed twice, the second dropped: V is singular, an eigenvalue is missing"))
    w, V, Y = claim(A_def)
    cases.append(("nh_defective", A_def, w, V, None, defect, "defective", ["N1.backward_errors"], [],
                  "a Jordan block of size 3: no claim of three simple eigenvalues near 1/2 may be certified"))
    w, V, Y = claim(A_real)
    w3 = w.copy()
    w3[0] = w3[0] + 0.5
    cases.append(("nh_wrong_eigenvalue", A_real, w3, V, None, real5, "wrong_eigenvalue", ["N1.backward_errors"], [],
                  "the first eigenvalue shifted by 0.5 with its vector unchanged"))
    w, V, Y = claim(A_real)
    V4 = V.copy()
    V4[:, [0, 3]] = V4[:, [3, 0]]
    cases.append(("nh_swapped_vectors", A_real, w, V4, None, real5, "swapped_vectors", ["N1.backward_errors"], [],
                  "the eigenvectors of two levels interchanged"))
    w, V, Y = claim(A_real)
    V5 = V.copy()
    i0 = np.unravel_index(np.argmin(np.abs(V5)), V5.shape)
    V5[i0] = 1e-310
    cases.append(("nh_subnormal_entry", A_real, w, V5, None, real5, "subnormal_entry", [], ALL,
                  "the smallest eigenvector entry replaced by 1e-310 (no-underflow hypothesis not established; a model "
                  "with gradual underflow may certify)"))
    w, V, Y = claim(A_cpx)
    cases.append(("nh_partial_claim", A_cpx, w[:2], V[:, :2], Y[:, :2], cpx, "partial_claim", ["N1.backward_errors"], [],
                  "two of five pairs (kind subset): backward errors only"))
    return cases


def main():
    os.makedirs(CASES, exist_ok=True)
    index = {}
    for name, A, w, V, Y, spec, cls, must, must_not, note in build():
        arrays = {"A": A, "eigenvalues": np.asarray(w), "eigenvectors": np.asarray(V)}
        if Y is not None:
            arrays["left_eigenvectors"] = np.asarray(Y)
        np.savez(os.path.join(CASES, name + ".npz"), **arrays)
        exp = {"schema": "certify-ed/corpus-case-nonhermitian/1.0", "case": name, "error_class": cls, "note": note,
               "dimension": int(A.shape[0]), "n_pairs": int(np.asarray(w).size), "must_certify": must,
               "must_not_certify": must_not, "conditional": cls == "subnormal_entry",
               "exact_eigenvalues": [_q(v) for v in spec]}
        with open(os.path.join(CASES, name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(exp, f, indent=1)
            f.write("\n")
        index[name] = cls
    with open(os.path.join(CASES, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    print(f"{len(index)} cases written to {CASES}")


if __name__ == "__main__":
    main()
