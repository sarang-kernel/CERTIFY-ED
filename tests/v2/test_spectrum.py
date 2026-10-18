"""Whole-spectrum enclosure (certify_ed.spectrum), the symveig functionality on the CERTIFY-ED engine.
Includes the five synthetic classes of symveig's test suite and the two counterexamples found in the
2026-10-18 audit of symveig 1.0.0."""
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.spectrum import enclose_spectrum


def _contained(enc, w, tol=None):
    """Containment of floating-point reference eigenvalues, allowing for the reference's own error
    (a LAPACK eigenvalue is only accurate to about n u ||A||, which can exceed a tight enclosure)."""
    if tol is None:
        tol = 64 * len(w) * 2.0 ** -53 * max(1.0, float(np.max(np.abs(w))))
    return all(any(iv.lo - tol <= x <= iv.hi + tol for iv in enc.intervals) for x in w)


def _sym(rng, n):
    B = rng.standard_normal((n, n))
    return (B + B.T) / 2


@pytest.mark.parametrize("case", ["isolated", "cluster", "ill_conditioned", "near_degenerate"])
def test_synthetic_classes_complete_and_contained(case):
    rng = np.random.default_rng(1)
    n = 40
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    if case == "isolated":
        lam = np.arange(n, dtype=float)
    elif case == "cluster":
        lam = np.r_[np.full(8, 2.0), np.arange(10, 10 + n - 8, dtype=float)]
    elif case == "ill_conditioned":
        lam = np.logspace(-16, 0, n)
    else:
        lam = np.r_[1.0, 1.0 + 1e-10, np.arange(3, 3 + n - 2, dtype=float)]
    A = (Q * lam) @ Q.T
    A = (A + A.T) / 2                          # exactly symmetric
    enc = enclose_spectrum(A)
    w = la.eigvalsh(A)
    assert _contained(enc, w)
    assert enc.complete
    assert sum(iv.count for iv in enc.intervals) == n


def test_block_structured_matrix_with_sectors():
    rng = np.random.default_rng(2)
    blocks = [_sym(rng, m) for m in (5, 9, 3)]
    A = la.block_diag(*blocks)
    S = np.repeat([0.0, 1.0, 2.0], [5, 9, 3])
    perm = rng.permutation(A.shape[0])
    A, S = A[np.ix_(perm, perm)], S[perm]
    enc = enclose_spectrum(A, S)
    assert enc.complete and _contained(enc, la.eigvalsh(A))
    assert sorted(enc.sector_dims.values()) == [3, 5, 9]


def test_heisenberg_sectors_complete_and_tight():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    S = np.array([bin(s).count("1") for s in range(H.shape[0])], dtype=float)
    enc = enclose_spectrum(H, S)
    w = la.eigvalsh(H.toarray())
    assert enc.complete and _contained(enc, w)
    assert enc.max_halfwidth < 2e-12            # symveig 1.0.0: 3.6e-12 (sector path, same model)


def test_duplicated_eigenvector_is_not_counted():
    # symveig 1.0.0 reported [-1.4e-15, 1.4e-15] with multiplicity 2 here, missing the eigenvalue 1
    A = np.diag([0.0, 1.0])
    X = np.array([[1.0, 1.0], [0.0, 0.0]])
    enc = enclose_spectrum(A, eigh_result=(np.array([0.0, 0.0]), X))
    assert not enc.complete
    assert not any(iv.counted and iv.count == 2 for iv in enc.intervals)


def test_non_conserved_quantity_rejected():
    # symveig 1.0.0 enclosed the wrong spectrum here (58 of 64 eigenvalues outside its enclosures)
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).toarray()
    n = H.shape[0]
    X = np.zeros((n, n))
    for s in range(n):
        for i in range(6):
            X[s ^ (1 << i), s] += 0.05 * (-1) ** i
    S = np.array([bin(s).count("1") for s in range(n)], dtype=float)
    with pytest.raises(ValueError, match="not conserved"):
        enclose_spectrum(H + X, S)


def test_non_hermitian_input_rejected():
    A = np.array([[0.0, 1.0], [1.0 + 1e-15, 0.0]])
    with pytest.raises(ValueError, match="Hermitian"):
        enclose_spectrum(A)
