"""The declared floors (requires-python >= 3.9, numpy >= 1.22, scipy >= 1.8) and the summation of signed zeros.

Run on Python 3.9.23, numpy 1.22.4, scipy 1.8.1 (OpenBLAS), the suite of main bf567f9f showed three defects of the package:
scipy 1.8 (and 1.10) refuse a 1 x 1 eigh with the drivers evr (the default) and evd, which the dense corroboration
oracles, the enclosure of a one-dimensional block in spectrum and the default sweep solver met (now _linalg.eigh); scipy.sparse.sparray does not exist
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16, np.complex128, np.complex64, np.int64,
                                   np.uint64, np.int32, np.int16, np.uint16, np.int8, np.uint8, np.bool_])
def test_linalg_eigh_matches_scipy_on_a_1x1_matrix(dtype):
    """_linalg.eigh of a 1 x 1 matrix: scipy's values, vectors and dtypes where scipy runs (the drivers ev and evx run
    on every scipy; boolean, 8- and 16-bit integer and float16 data are computed in single precision there), and
    scipy's refusal of a non-finite entry."""
    import scipy.linalg as la

    from certify_ed._linalg import eigh
    kind = np.dtype(dtype).kind
    a = np.array([[True]]) if kind == "b" else np.array([[3]], dtype=dtype) if kind in "iu" else np.array([[-2.5]], dtype=dtype)
    w, X = eigh(a)
    w_ref, X_ref = la.eigh(a, driver="ev")
    assert w.tolist() == w_ref.tolist() and abs(X.tolist()[0][0]) == abs(X_ref.tolist()[0][0]) == 1
    assert w.dtype == w_ref.dtype and X.dtype == X_ref.dtype
    assert eigh(a, eigvals_only=True, subset_by_index=[0, 0]).tolist() == w_ref.tolist()
    with pytest.raises(ValueError):
        eigh(a, subset_by_index=[0, 1])
    if kind in "fc":
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


def _scipy_eigh_calls(tree):
    """(name, lineno, generalized) of every call of scipy.linalg.eigh or eigvalsh in a module: through any name the
    module binds to scipy.linalg (import scipy.linalg as X, from scipy import linalg [as X], or the dotted
    scipy.linalg after import scipy...), and through the names it binds to the functions themselves (from scipy.linalg
    import eigh [as Y]). Imports anywhere in the module count, also those inside functions."""
    import ast
    mods, funcs, dotted = set(), {}, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "scipy.linalg" and a.asname:
                    mods.add(a.asname)
                elif a.name in ("scipy", "scipy.linalg") and not a.asname:
                    dotted = True
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for a in node.names:
                if node.module == "scipy" and a.name == "linalg":
                    mods.add(a.asname or "linalg")
                elif node.module == "scipy.linalg" and a.name in ("eigh", "eigvalsh"):
                    funcs[a.asname or a.name] = a.name
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f, name = node.func, None
        if isinstance(f, ast.Attribute) and f.attr in ("eigh", "eigvalsh"):
            v = f.value
            if isinstance(v, ast.Name) and v.id in mods:
                name = f"{v.id}.{f.attr}"
            elif (dotted and isinstance(v, ast.Attribute) and v.attr == "linalg" and isinstance(v.value, ast.Name)
                  and v.value.id == "scipy"):
                name = f"scipy.linalg.{f.attr}"
        elif isinstance(f, ast.Name) and f.id in funcs:
            name = f.id
        if name:
            generalized = len(node.args) > 1 or any(k.arg == "b" for k in node.keywords)
            out.append((name, node.lineno, generalized))
    return out


def test_the_audit_resolves_import_aliases():
    """The audit below sees a call through every binding form of scipy.linalg and of its functions, and not a call of
    numpy.linalg.eigh (which handles a 1 x 1 matrix at every supported version)."""
    import ast
    src = """
import scipy.linalg as sla
import scipy
import numpy as np
from scipy import linalg as L2
from scipy.linalg import eigh as E, eigvalsh
from numpy import linalg
def f(a, b):
    import scipy.linalg as inner
    sla.eigh(a); scipy.linalg.eigvalsh(a); L2.eigh(a); E(a); eigvalsh(a); inner.eigh(a, b)
    np.linalg.eigh(a); linalg.eigh(a)
"""
    got = sorted((n, g) for n, _, g in _scipy_eigh_calls(ast.parse(src)))
    assert got == sorted([("sla.eigh", False), ("scipy.linalg.eigvalsh", False), ("L2.eigh", False), ("E", False),
                          ("eigvalsh", False), ("inner.eigh", True)]), got


def test_every_standard_eigh_call_goes_through_linalg():
    """Audit: a call of scipy.linalg.eigh or eigvalsh with one matrix (a standard problem), through any import alias,
    appears only in _linalg.py, or in an application whose matrix has a fixed dimension above 1 (listed here with that
    dimension). Generalized problems (a second matrix: generalized.py and apps/aklt) are left to scipy, which solves a
    1 x 1 pencil at the floors (test_a_1x1_generalized_problem_runs)."""
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
            for name, line, gen in _scipy_eigh_calls(ast.parse(open(path, encoding="utf-8").read())):
                if not gen:
                    found.setdefault((rel, name), []).append(line)
    stray = {k: v for k, v in found.items() if k[0] != "_linalg.py" and k not in allowed}
    assert not stray, stray
    assert set(allowed) <= set(found), "an allowlisted call is gone: remove it from the list"


@pytest.mark.parametrize("driver", [None, "gv", "gvd", "gvx"])
def test_a_1x1_generalized_problem_runs(driver):
    """scipy refuses a 1 x 1 standard problem with the drivers evr and evd at 1.8 and 1.10, but solves a 1 x 1 pencil
    with every driver (run at scipy 1.8.1, 1.10.1 and the current version), so the generalized calls of the package
    (generalized.py, apps/aklt) are left to scipy."""
    import scipy.linalg as la
    kw = {} if driver is None else {"driver": driver}
    assert la.eigh(np.array([[2.0]]), np.array([[4.0]]), eigvals_only=True, **kw).tolist() == [0.5]
    w, v = la.eigh(np.array([[2.0]]), np.array([[4.0]]), **kw)
    assert w.tolist() == [0.5] and abs(abs(float(v[0, 0])) - 0.5) < 1e-15


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


@pytest.mark.parametrize("oracles", [("lapack_evr", "jacobi"), ("lapack_evd",), ("jacobi",)])
def test_corroborate_does_not_agree_on_a_nan_matrix(oracles):
    """A matrix with a nan entry: the dense oracles refuse it (scipy's check_finite, kept by the 1 x 1 helper) and the
    Jacobi oracle returns nan, so corroborate is INCONCLUSIVE; a nan difference was dropped by max() before, so a 1 x 1
    nan matrix with the shortcut, or any nan matrix with the Jacobi oracle alone, came out CORROBORATED."""
    from certify_ed.corroborate import corroborate
    for A in (np.array([[np.nan]]), np.array([[1.0, np.nan], [np.nan, 2.0]])):
        cor = corroborate(A, np.array([1.0]), oracles=list(oracles))
        assert cor["status"] == "INCONCLUSIVE", cor


def test_corroborate_keeps_a_nan_difference_after_a_finite_one(monkeypatch):
    """The worst difference is a nan-sticky maximum: an oracle that returns nan, then one that agrees, gives an
    INCONCLUSIVE record whose summary names the nan (the per-oracle condition alone kept the status right before)."""
    import certify_ed.corroborate as cm
    outs = iter([np.array([np.nan]), np.array([1.0])])
    monkeypatch.setattr(cm, "run_oracle", lambda name, A, k=None: next(outs))
    cor = cm.corroborate(np.array([[1.0]]), np.array([1.0]), oracles=["lapack_evr", "jacobi"])
    assert cor["status"] == "INCONCLUSIVE" and "nan" in cor["summary"], cor
