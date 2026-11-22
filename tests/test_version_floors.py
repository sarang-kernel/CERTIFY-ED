"""The declared floors (requires-python >= 3.9, numpy >= 1.22, scipy >= 1.8) and the summation of signed zeros.

Run on Python 3.9.23, numpy 1.22.4, scipy 1.8.1 (OpenBLAS), the suite of main bf567f9f showed three defects of the package:
scipy 1.8 (and 1.10) refuse a 1 x 1 eigh with the drivers evr (the default) and evd, which the dense corroboration
oracles, the enclosure of a one-dimensional block in spectrum and the default sweep solver met (now _compat.eigh); scipy.sparse.sparray does not exist
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128, np.complex64, np.int64])
def test_compat_eigh_matches_scipy_on_a_1x1_matrix(dtype):
    """_compat.eigh of a 1 x 1 matrix: scipy's values, vectors and dtypes where scipy runs (the drivers ev and evx run
    on every scipy), and scipy's refusal of a non-finite entry."""
    import scipy.linalg as la

    from certify_ed._compat import eigh
    a = np.array([[3]], dtype=dtype) if dtype is np.int64 else np.array([[-2.5]], dtype=dtype)
    w, X = eigh(a)
    w_ref, X_ref = la.eigh(a, driver="ev")
    assert w.tolist() == w_ref.tolist() and abs(X.tolist()[0][0]) == abs(X_ref.tolist()[0][0]) == 1
    assert w.dtype == w_ref.dtype and X.dtype == X_ref.dtype
    assert eigh(a, eigvals_only=True, subset_by_index=[0, 0]).tolist() == w_ref.tolist()
    with pytest.raises(ValueError):
        eigh(a, subset_by_index=[0, 1])
    if dtype is not np.int64:
        with pytest.raises(ValueError):
            eigh(np.array([[np.nan]], dtype=dtype))


def test_no_1x1_standard_eigh_reaches_scipy(monkeypatch):
    """Tripwire: with scipy's eigh and eigvalsh refusing every 1 x 1 matrix (as scipy 1.8 to 1.10 do for the default
    driver), the package's callers still handle one: the dense oracles, a one-dimensional spectrum block, the sweep
    solver and the next-eigenvalue estimate of certify."""
    import scipy.linalg as la

    from certify_ed.certify import _estimate_next
    real_eigh, real_eigvalsh = la.eigh, la.eigvalsh

    def refuse(f):
        def g(a, *args, **kw):
            if np.shape(a) == (1, 1) and not args and "b" not in kw:
                raise RuntimeError("1 x 1 standard problem reached scipy")
            return f(a, *args, **kw)
        return g
    monkeypatch.setattr(la, "eigh", refuse(real_eigh))
    monkeypatch.setattr(la, "eigvalsh", refuse(real_eigvalsh))
    for name in ("lapack_evr", "lapack_evd", "lapack_ev"):
        assert run_oracle(name, np.array([[1.5]])).tolist() == [1.5]
    assert _enclose_block(sp.csr_matrix([[1.5]]))[1].mu.tolist() == [1.5]
    assert _default_solver(sp.csr_matrix([[3.0]]), 1)[0].tolist() == [3.0]
    assert _estimate_next(np.array([[4.0]]), 0) == 4.0
    with pytest.raises(RuntimeError):                      # the tripwire itself fires on a direct call
        la.eigh(np.array([[1.0]]))


def test_every_standard_eigh_call_goes_through_compat():
    """Audit: a call of scipy.linalg.eigh or eigvalsh with one matrix (a standard problem) appears only in _compat.py,
    or in an application whose matrix has a fixed dimension above 1 (listed here with that dimension)."""
    import ast
    import os

    import certify_ed
    root = os.path.dirname(certify_ed.__file__)
    allowed = {("apps/pyrochlore/spin_one.py", "la.eigh"): "the spin-1 ring of 6 sites, dimension 729"}
    found = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("eigh", "eigvalsh") and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in ("la", "_la", "scipy_linalg", "linalg")):
                    continue
                generalized = len(node.args) > 1 or any(k.arg == "b" for k in node.keywords)
                if not generalized:
                    found.setdefault((rel, f"{node.func.value.id}.{node.func.attr}"), []).append(node.lineno)
    stray = {k: v for k, v in found.items() if k[0] != "_compat.py" and k not in allowed}
    assert not stray, stray
    assert set(allowed) <= set(found), "an allowlisted call is gone: remove it from the list"


@pytest.mark.parametrize("reps", [2, 3, 70])
def test_hash_of_repeated_negative_zero_parts_equals_the_summed_form(reps):
    """matrix_sha256 of a complex matrix whose repeated entries have real parts -0.0 equals that of the matrix with the
    summed entry (-0.0 + reps j) stored once, on every numpy and route. At main this held for two repeats under numpy 2
    only: for three or more the real part summed to +0.0, so the hash of such a matrix changes from main's."""
    from certify_ed.certificate_v2 import matrix_sha256
    n = 5
    rows = np.repeat(np.arange(n), reps)
    dup = sp.coo_matrix((np.full(rows.size, complex(-0.0, 1.0)), (rows, rows)), shape=(n, n))
    once = sp.coo_matrix((np.full(n, complex(-0.0, float(reps))), (np.arange(n), np.arange(n))), shape=(n, n))
    assert matrix_sha256(dup) == matrix_sha256(once)
    assert matrix_sha256(dup.tocsr()) == matrix_sha256(once.tocsr())
