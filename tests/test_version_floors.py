"""The declared floors (requires-python >= 3.9, numpy >= 1.22, scipy >= 1.8) and the summation of signed zeros.

Run on Python 3.9.23, numpy 1.22.4, scipy 1.8.1 (OpenBLAS), the suite of main bf567f9f showed three defects of the package:
scipy 1.8 cannot size the LAPACK workspace of a 1 x 1 eigh (every driver), which the dense corroboration oracles, the
enclosure of a one-dimensional block in spectrum and the default sweep solver met; scipy.sparse.sparray does not exist
before scipy 1.11, so precision.summed returned a coo_matrix for a coo_array; and np.add.reduceat gives +0.0 for two
negative zeros there, where numpy 2 gives -0.0 (and the vectorized passes and math.fsum gave +0.0 for longer groups on
any version). Each test here holds on both builds.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed.corroborate import run_oracle
from certify_ed.precision import _is_sparse_array, summed
from certify_ed.spectrum import _enclose_block
from certify_ed.sweep import _default_solver


@pytest.mark.parametrize("name", ["lapack_evr", "lapack_evd", "lapack_ev"])
@pytest.mark.parametrize("value", [1.5, -2.0 + 0j])
def test_dense_oracles_on_a_1x1_matrix(name, value):
    """A 1 x 1 Hermitian matrix: its entry, for every dense oracle (scipy 1.8 raised from LAPACK's workspace query)."""
    assert run_oracle(name, np.array([[value]])).tolist() == [np.real(value)]
    assert run_oracle(name, sp.csr_matrix(np.array([[value]]))).tolist() == [np.real(value)]


def test_one_dimensional_block_and_sweep_solver():
    """The enclosure of a 1 x 1 block in spectrum and the default solver of certified_sweep."""
    iv, enc = _enclose_block(sp.csr_matrix([[1.5]]))
    assert enc.mu.tolist() == [1.5] and enc.radius[0] < 1e-14
    w, Y = _default_solver(sp.csr_matrix([[3.0]]), 1)
    assert w.tolist() == [3.0] and Y.tolist() == [[1.0]]
    w, Y = _default_solver(np.array([[2.0 + 0j]]), 1)
    assert w.tolist() == [2.0] and Y.shape == (1, 1)


@pytest.mark.parametrize("k", [1, 2, 3, 70, 700])
def test_sum_of_negative_zeros_is_negative_zero(k):
    """k stored -0.0 at one position: -0.0 on every route (one value, one addition, the vectorized passes, math.fsum);
    a +0.0 among them, or values that cancel, give +0.0, as IEEE addition does in any order."""
    def one(vals):
        vals = np.asarray(vals, dtype=float)
        C = sp.coo_matrix((vals, (np.zeros(vals.size, int), np.zeros(vals.size, int))), shape=(1, 1))
        return summed(C).data[0]
    z = one(np.full(k, -0.0))
    assert z == 0.0 and np.signbit(z)
    if k > 1:
        mixed = np.full(k, -0.0)
        mixed[k // 2] = 0.0
        assert not np.signbit(one(mixed))
        cancel = np.full(k, -0.0)
        cancel[0], cancel[-1] = 1.0, -1.0
        assert one(cancel) == 0.0 and not np.signbit(one(cancel))


def test_complex_part_sign_and_many_positions():
    """Each part of a complex sum on its own, over many positions at once (the vectorized blocks): the real part of
    600 positions holding (-0.0 + 1j) twice, and of 600 holding it three times, is -0.0."""
    for reps in (2, 3):
        n = 600
        rows = np.repeat(np.arange(n), reps)
        vals = np.full(rows.size, complex(-0.0, 1.0))
        S = summed(sp.coo_matrix((vals, (rows, np.zeros(rows.size, int))), shape=(n, 1)))
        assert np.all(np.signbit(S.data.real)) and np.all(S.data.imag == reps)


def test_sparse_array_class_kept():
    """summed keeps a sparse array a sparse array on every scipy (scipy.sparse.sparray exists from scipy 1.11; before,
    the arrays derive from the matrix classes through a private mixin)."""
    r, c = np.array([0, 0, 1]), np.array([1, 1, 0])
    A = sp.coo_array((np.array([1.0, 2.0, 3.0]), (r, c)), shape=(2, 2))
    M = sp.coo_matrix((np.array([1.0, 2.0, 3.0]), (r, c)), shape=(2, 2))
    assert _is_sparse_array(A) and not _is_sparse_array(M)
    assert type(summed(A)) is type(A) and type(summed(M)) is type(M)
    assert summed(A).toarray()[0, 1] == 3.0
