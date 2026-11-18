"""fp.modulus_up and fp.abs_up: upper moduli from IEEE *, +, sqrt and exact power-of-two scaling
(docs/proofs/fp_modulus-up.md, Lemma M; UNVERIFIED until Sarang's check, D-12). Exact rational checks:
m^2 >= re^2 + im^2 for every finite z, and the tightness m <= |z| (1 + 12u) + 2^-1073."""
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed.fp import U, abs_up, modulus_up

# relative errors of NumPy's complex absolute value measured on the Windows reference host (numpy 2.4.4, AVX512F):
# 2.466u, 2.462u and 2.269u (docs/proofs/fp_modulus-up.md, section 1)
EXAMPLES = [0.02019294356747192 - 0.5012539518584045j, 1.9986893956853402 + 0.07375207458660175j,
            1.010573342221019 + 0.3330018761835207j]
_T = Fraction(2) ** -1073
_C = 1 + 12 * Fraction(U)


def _upper_ok(z, m):
    if m == np.inf:
        return True
    x, y = Fraction(float(np.real(z))), Fraction(float(np.imag(z)))
    return Fraction(float(m)) ** 2 >= x * x + y * y


def _tight_ok(z, m):
    """Lemma M (ii) for a finite m, and (iii) for m = inf: inf only when |z| (1 + 12u) exceeds the largest float
    (z with an infinite component: test_inf_and_nan_follow_numpy)."""
    if not np.isfinite(z):
        return m == np.inf
    x, y = Fraction(float(np.real(z))), Fraction(float(np.imag(z)))
    if m == np.inf:
        return (x * x + y * y) * _C * _C > Fraction(float(np.finfo(float).max)) ** 2
    t = Fraction(float(m)) - _T
    return t <= 0 or t * t <= (x * x + y * y) * _C * _C


def _boundary_values():
    tiny = 2.0 ** -1074
    v = [0.0, tiny, 3 * tiny, 2.0 ** -1022 - tiny, 2.0 ** -1022, 2.0 ** -1022 + tiny, 2.0 ** -537, 2.0 ** -512,
         2.0 ** -511, 2.0 ** -511 * 1.5, 2.0 ** -500, np.nextafter(2.0 ** -500, 0), 2.0 ** -60, 0.1, 1.0,
         1.0 - 2.0 ** -53, 2.0 ** 60, 2.0 ** 500, np.nextafter(2.0 ** 500, np.inf), 2.0 ** 511, 2.0 ** 512,
         2.0 ** 1022, 2.0 ** 1023, np.finfo(float).max, np.nextafter(np.finfo(float).max, 0)]
    return v


def _adversarial():
    v = _boundary_values()
    out = list(EXAMPLES)
    for a in v:
        for b in v:
            out += [complex(a, b), complex(-a, b), complex(a, -b)]
    for a in (1.0, 0.75, 2.0 ** -1000, 2.0 ** 1000):
        out += [complex(a, a), complex(a, a * 2.0 ** 60), complex(a, a * 2.0 ** -60), complex(a * 2.0 ** -60, a)]
    return np.array(out, dtype=complex)


def test_upper_modulus_is_an_upper_bound_and_tight_on_adversarial_cases():
    Z = _adversarial()
    M = modulus_up(Z)
    bad = [(z, m) for z, m in zip(Z, M) if not (_upper_ok(z, m) and _tight_ok(z, m))]
    assert not bad, bad[:5]
    for z in EXAMPLES:
        assert _upper_ok(z, modulus_up(z)) and _tight_ok(z, modulus_up(z))


def test_the_guard_leaves_a_margin_above_3u():
    """Pins the guard factor (1 + 8u) of both branches. In the proof of Lemma M (i), (1 - u)^3 (1 + 8u) >= 1 + 4u; so
    when both components are nonzero and m is a normal float, m >= |z| (1 + 4u)(1 - 2^-73) > |z| (1 + 3u). A guard
    of (1 + 2u) would leave m below |z| (1 + 3u) for many z."""
    rng = np.random.default_rng(20261119)
    n = 4000
    scaled = [z * 2.0 ** k for z in EXAMPLES for k in (-1000, -600, 600, 1000)]
    with np.errstate(over="ignore", invalid="ignore"):
        R = (rng.standard_normal(n) * np.exp2(rng.integers(-1074, 1024, n))
             + 1j * rng.standard_normal(n) * np.exp2(rng.integers(-1074, 1024, n)))
        Z = np.concatenate([_adversarial(), R, np.array(scaled)])
        Z = Z[np.isfinite(Z) & (Z.real != 0) & (Z.imag != 0)]
    M = modulus_up(Z)
    keep = np.isfinite(M) & (M >= 2.0 ** -1021)
    assert keep.sum() > 2000
    c = 1 + 3 * Fraction(U)
    bad = [(z, m) for z, m in zip(Z[keep], M[keep])
           if Fraction(float(m)) ** 2 < (Fraction(z.real) ** 2 + Fraction(z.imag) ** 2) * c * c]
    assert not bad, bad[:5]


def test_upper_modulus_on_random_magnitudes():
    rng = np.random.default_rng(20261118)
    n = 4000
    Z = (rng.standard_normal(n) * np.exp2(rng.integers(-1074, 1024, n))
         + 1j * rng.standard_normal(n) * np.exp2(rng.integers(-1074, 1024, n)))
    with np.errstate(over="ignore", invalid="ignore"):
        Z = Z[np.isfinite(Z)]
    M = modulus_up(Z)
    assert all(_upper_ok(z, m) and _tight_ok(z, m) for z, m in zip(Z, M))


def test_the_step_up_covers_the_whole_subnormal_band():
    """Lemma M, Step 4 (c): the step up below 2^-1022, on 2 400 z with both components nonzero and |z| log-uniform in
    [2^-1074, 2^-1022], and 400 more just below 2^-1022 (exact check m^2 >= x^2 + y^2, and (ii)). A step taken only
    below a smaller threshold (2^-1030, 2^-1023), or no step, gives m < |z| on some of them."""
    rng = np.random.default_rng(20261120)
    e = np.concatenate([rng.uniform(-1074.0, -1022.0, 2400), rng.uniform(-1023.0, -1022.0, 400)])
    t = rng.uniform(0.0, np.pi / 2, e.size)
    r = np.exp2(e)
    Z = r * np.cos(t) + 1j * (r * np.sin(t))
    Z = Z[(Z.real != 0) & (Z.imag != 0) & (np.abs(Z) <= 2.0 ** -1022)]
    assert Z.size > 2000
    M = modulus_up(Z)
    bad = [(z, m) for z, m in zip(Z, M) if not (_upper_ok(z, m) and _tight_ok(z, m))]
    assert not bad, bad[:5]


def test_the_exact_check_rejects_a_value_one_step_below():
    """Negative control of the check itself: one binary64 step below fl(sqrt(re^2 + im^2)) rounded down is never an
    upper bound (whatever NumPy's complex absolute value returns on a given platform)."""
    for z in EXAMPLES:
        x, y = Fraction(z.real), Fraction(z.imag)
        s = float(np.sqrt(np.float64(z.real) ** 2 + np.float64(z.imag) ** 2))
        while Fraction(s) ** 2 >= x * x + y * y:
            s = float(np.nextafter(s, 0.0))
        assert not _upper_ok(z, s)


def test_real_input_is_returned_exactly():
    rng = np.random.default_rng(3)
    for a in (rng.standard_normal(17), rng.standard_normal((4, 5)).astype(np.float32)):
        m = modulus_up(a)
        assert m.dtype == np.abs(a).dtype and np.array_equal(m, np.abs(a))
    # a scalar gives a float, whatever its type (exact: every type here is at most binary64)
    for a in (-2.5, np.float64(-0.0), np.float32(-1.5), np.float16(-3.0), -7, np.int32(-4), True, 3 + 4j,
              np.complex64(1j), np.asarray(-0.25), np.asarray(1 - 1j)):
        m = modulus_up(a)
        assert type(m) is float and m >= abs(complex(a)), a
        if not np.iscomplexobj(a):
            assert m == abs(float(a))
    w = np.longdouble(-1.5)
    assert modulus_up(w) == 1.5 and (type(modulus_up(w)) is float) == (np.finfo(np.longdouble).nmant <= 52)
    # complex data with a zero component: the other's absolute value, exact
    a = rng.standard_normal(9)
    assert np.array_equal(modulus_up(a + 0j), np.abs(a)) and np.array_equal(modulus_up(1j * a), np.abs(a))


def test_dtypes_outside_binary64_components():
    """Integers and booleans: float64 absolute values, exact below 2^53 and one step up from there (the conversion may
    round; np.abs itself wraps at the most negative integer). Rejected: scipy sparse input (abs_up takes it), object
    arrays and complex types wider than complex128. complex64 is widened exactly and gives the complex128 result."""
    ints = np.array([-3, 0, 7, -2 ** 63, 2 ** 63 - 1, 2 ** 53 + 1, -(2 ** 53) - 1], dtype=np.int64)
    m = modulus_up(ints)
    assert m.dtype == np.float64
    assert all(Fraction(float(v)) >= abs(int(n)) for v, n in zip(m, ints))
    assert np.array_equal(m[:3], [3.0, 0.0, 7.0])
    assert modulus_up(np.array([2 ** 64 - 1], dtype=np.uint64))[0] >= 2 ** 64 - 1
    assert np.array_equal(modulus_up(np.array([True, False])), [1.0, 0.0])
    assert modulus_up(np.int64(-5)) == 5.0
    for bad in (np.array([1 + 1j, 2], dtype=object), sp.csr_matrix(np.eye(2) * (1 + 1j)), np.array(["a"])):
        with pytest.raises(TypeError):
            modulus_up(bad)
    with pytest.raises(TypeError, match="abs_up"):                 # the message names the function for matrices
        modulus_up(sp.csr_matrix(np.eye(2) * (1 + 1j)))
    Z = np.array(EXAMPLES) * np.array([1.0, 2.0 ** -600, 2.0 ** 600])
    if np.finfo(np.clongdouble).nmant > 52:
        with pytest.raises(TypeError):
            modulus_up(Z.astype(np.clongdouble))
    else:
        assert np.array_equal(modulus_up(Z.astype(np.clongdouble)), modulus_up(Z))
    z64 = (np.array(EXAMPLES) * 2.0 ** -100).astype(np.complex64)
    assert np.array_equal(modulus_up(z64), modulus_up(z64.astype(np.complex128)))


def test_chunks_do_not_change_the_values():
    """modulus_up processes the input in passes of at most fp._CHUNK entries; every entry is computed by itself, so the
    result equals the one-entry calls, at the pass boundaries too."""
    from certify_ed.fp import _CHUNK
    rng = np.random.default_rng(8)
    n = 2 * _CHUNK + 5
    with np.errstate(over="ignore"):
        Z = rng.standard_normal(n) * np.exp2(rng.integers(-1074, 1000, n)) + 1j * rng.standard_normal(n) * np.exp2(
            rng.integers(-1074, 1000, n))
    M = modulus_up(Z)
    for i in (0, _CHUNK - 1, _CHUNK, _CHUNK + 1, 2 * _CHUNK - 1, 2 * _CHUNK, n - 1):
        assert M[i] == modulus_up(Z[i:i + 1])[0] == modulus_up(complex(Z[i]))
    assert np.array_equal(M[_CHUNK - 3:_CHUNK + 3], modulus_up(Z[_CHUNK - 3:_CHUNK + 3]))


def test_layouts_give_the_same_values_and_no_copy_of_the_input():
    """modulus_up walks the input in passes of at most fp._CHUNK entries along its leading axes and makes no copy of it:
    Fortran-ordered, transposed and sliced arrays larger than a pass give the values of the C-ordered array, and the
    temporaries beyond the float64 result stay below 12 MiB (tracemalloc; a copy of the input would be 32 MiB)."""
    import tracemalloc

    from certify_ed.fp import _CHUNK
    rng = np.random.default_rng(13)
    X = rng.standard_normal((_CHUNK, 8)) + 1j * rng.standard_normal((_CHUNK, 8))
    X[::5] = 0.0
    M = modulus_up(X)
    F = np.asfortranarray(X)
    for z, ref in ((F, M), (X.T, M.T), (X[:, 2:5], M[:, 2:5]), (F[:, 3], M[:, 3]), (X[::-3, ::2], M[::-3, ::2]),
                   (np.ascontiguousarray(X.T)[None], M.T[None])):
        got = modulus_up(z)
        assert got.shape == z.shape and np.array_equal(got, ref)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        got = modulus_up(z)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak - base - got.nbytes < 12 * 2 ** 20, (z.shape, z.strides, peak - base - got.nbytes)


def _nearest(t, p):
    """Whether the binary64 number p is a nearest binary64 number to the rational t (exact comparison)."""
    if not np.isfinite(p):
        return False
    d = abs(Fraction(float(p)) - t)
    return all(abs(Fraction(float(q)) - t) >= d for q in (np.nextafter(p, -np.inf), np.nextafter(p, np.inf))
               if np.isfinite(q))


def test_fp_m_hypotheses_on_this_host():
    """Known answers for the items of FP-M that fp.probe_float_environment does not probe at run time (Lemma M uses
    them; fp_modulus-up.md, section 1): correctly rounded np.multiply (subnormal results included), np.add and
    np.sqrt; np.frexp exact (subnormal arguments included); np.ldexp exact when the result is a binary64 number, inf
    above the largest float, within 2^-1074 below 2^-1022; np.nextafter(t, inf) = t + 2^-1074 for t < 2^-1022. Each
    is one NumPy call per operation, as in modulus_up. This runs with the test suite on the host that runs it."""
    rng = np.random.default_rng(20261118)
    n = 3000
    tiny = 2.0 ** -1074
    a = rng.uniform(0.5, 1.0, n) * np.exp2(rng.integers(-560, 520, n))
    b = rng.uniform(0.5, 1.0, n) * np.exp2(rng.integers(-560, 520, n))
    with np.errstate(over="ignore", under="ignore"):
        P, S, R = np.multiply(a, b), np.add(a, b), np.sqrt(a)
    for i in range(n):
        x, y = Fraction(float(a[i])), Fraction(float(b[i]))
        if abs(x * y) <= Fraction(float(np.finfo(float).max)):
            assert P[i] == 0.0 and x * y <= Fraction(tiny) / 2 or _nearest(x * y, P[i]), (a[i], b[i])
        assert _nearest(x + y, S[i]), (a[i], b[i])
        r = Fraction(float(R[i]))
        h = (Fraction(float(np.nextafter(R[i], np.inf))) - r) / 2
        assert (r - h) ** 2 <= x <= (r + h) ** 2, a[i]
    v = np.concatenate([a, np.array([tiny, 3 * tiny, 2.0 ** -1022, 2.0 ** -1023 * 3])])
    f, e = np.frexp(v)
    assert np.all((f >= 0.5) & (f < 1.0))
    assert all(Fraction(float(f[i])) * Fraction(2) ** int(e[i]) == Fraction(float(v[i])) for i in range(v.size))
    k = rng.integers(-1200, 1100, v.size)
    with np.errstate(over="ignore", under="ignore"):
        L = np.ldexp(v, k)
    big = Fraction(float(np.finfo(float).max))
    for i in range(v.size):
        t = Fraction(float(v[i])) * Fraction(2) ** int(k[i])
        if t > big:
            assert L[i] == np.inf
        elif t >= Fraction(2) ** -1022:
            assert Fraction(float(L[i])) == t
        else:
            assert abs(Fraction(float(L[i])) - t) < Fraction(tiny)
    t = np.array([0.0, tiny, 2.0 ** -1030, 2.0 ** -1022 - tiny])
    assert np.array_equal(np.nextafter(t, np.inf), t + tiny)


def test_inf_and_nan_follow_numpy():
    vals = [0.0, 1.0, -2.0 ** -1074, np.inf, -np.inf, np.nan]
    Z = np.array([complex(a, b) for a in vals for b in vals])
    M, R = modulus_up(Z), np.abs(Z)
    assert np.array_equal(np.isinf(M), np.isinf(R)) and np.array_equal(np.isnan(M), np.isnan(R))
    # huge components: inf rather than a finite value below |z|
    big = np.finfo(float).max
    assert modulus_up(complex(big, big)) == np.inf and modulus_up(complex(big, 1.0)) == np.inf
    assert modulus_up(complex(big, 0.0)) == big


def test_zero_entries_take_the_unscaled_branch(monkeypatch):
    """An entry 0 + 0i is exact on the unscaled branch (0, then the zero override), so it does not pay for the scaled
    one: arrays of Pauli type never call _modulus_up_scaled, and the values are unchanged."""
    import certify_ed.fp as fp
    calls = []
    real_scaled = fp._modulus_up_scaled

    def spy(x, y):
        calls.append(x.size)
        return real_scaled(x, y)
    monkeypatch.setattr(fp, "_modulus_up_scaled", spy)
    for a in (np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], dtype=complex), np.zeros(5, dtype=complex)):
        assert np.array_equal(fp.modulus_up(a), np.abs(a)) and not calls
    assert fp.modulus_up(complex(2.0 ** -600, 0.0)) == 2.0 ** -600 and calls == [1]


def test_shapes_dtypes_and_elementwise_identity():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((30, 7)) + 1j * rng.standard_normal((30, 7))
    M = modulus_up(X)
    assert M.shape == X.shape and M.dtype == np.float64
    assert np.array_equal(modulus_up(np.asfortranarray(X)), M)
    assert np.array_equal(modulus_up(X[::3, 1:5]), M[::3, 1:5])          # strided view
    rows = np.array([3, 0, 17, 29])
    assert np.array_equal(modulus_up(X[rows]), M[rows])                  # what the GPU spot check relies on
    assert modulus_up(np.zeros((0, 3), dtype=complex)).shape == (0, 3)
    assert isinstance(modulus_up(3 + 4j), float) and modulus_up(3 + 4j) >= 5.0
    z64 = (X[:, 0] * 0.3).astype(np.complex64)
    m64 = modulus_up(z64)
    assert m64.dtype == np.float64 and all(_upper_ok(complex(z), m) for z, m in zip(z64, m64))


def test_abs_up_of_sparse_and_dense_matrices():
    rng = np.random.default_rng(8)
    A = sp.random(40, 40, density=0.2, format="csr", random_state=np.random.RandomState(8))
    B = A.copy()
    B.data = rng.standard_normal(B.nnz)
    C = (A + 1j * B).tocsr()
    for M in (A, C, C.tocoo(), C.tocsc()):
        got, ref = abs_up(M), abs(M.copy())
        assert got.format == ref.format and got.shape == ref.shape
        G, R = got.tocsr(), ref.tocsr()
        assert np.array_equal(G.indptr, R.indptr) and np.array_equal(G.indices, R.indices)
        assert np.array_equal(G.data, modulus_up(M.tocsr().data)) and G.dtype == np.float64
    assert abs_up(A).data is not A.data and np.array_equal(abs_up(A).data, abs(A).data)
    D = C.toarray()
    assert np.array_equal(abs_up(D), modulus_up(D))
    # duplicate entries are summed first, as abs() does
    Dup = sp.csr_matrix((np.array([1 + 1j, 2 - 1j]), np.array([0, 0]), np.array([0, 2, 2])), shape=(2, 2))
    got = abs_up(Dup)
    assert got.nnz == 1 and _upper_ok(3 + 0j, got.data[0]) and got.data[0] == 3.0


def test_abs_up_of_integer_sparse_data_is_stepped_up():
    """Integer and boolean sparse data take modulus_up (float64, exact below 2^53, one step up from there), not abs(A),
    which wraps at the most negative integer; real floating data keep abs(A)."""
    lo = np.iinfo(np.int64).min
    for mk in (sp.csr_matrix, sp.csc_array, sp.coo_matrix):
        A = mk(np.array([[lo, 1], [0, -(2 ** 53) - 1]], dtype=np.int64))
        got = abs_up(A)
        assert got.dtype == np.float64 and type(got) is type(A)
        D = got.toarray()
        assert Fraction(float(D[0, 0])) >= 2 ** 63 and D[0, 1] == 1.0 and Fraction(float(D[1, 1])) >= 2 ** 53 + 1
    B = sp.csr_matrix(np.array([[True, False], [False, True]]))
    assert abs_up(B).dtype == np.float64 and np.array_equal(abs_up(B).toarray(), np.eye(2))
    R = sp.csr_matrix(np.array([[-1.5, 0.0], [2.0, -0.25]], dtype=np.float32))
    assert abs_up(R).dtype == np.float32 and np.array_equal(abs_up(R).toarray(), abs(R).toarray())


def test_abs_up_sums_duplicates_in_place_and_keeps_the_format():
    """For CSR, CSC, BSR and COO input with duplicate entries, abs_up sums them in place (A.sum_duplicates(), binary64,
    as abs(A) does), so the moduli and the products formed with A afterwards refer to the same summed matrix. DIA (no
    duplicates) keeps its diagonals; LIL is converted (A unchanged). The result has A's format and class (sparse
    matrix or sparse array)."""
    rng = np.random.default_rng(12)
    idx, indptr = np.array([2, 0, 2, 1, 1, 0, 0, 2, 0]), np.array([0, 3, 5, 9])
    for mk in (sp.csr_matrix, sp.csc_matrix, sp.csr_array, sp.csc_array):
        for cplx in (False, True):
            v = rng.standard_normal(9) + (1j * rng.standard_normal(9) if cplx else 0)
            A = mk((v.copy(), idx.copy(), indptr.copy()), shape=(3, 3))
            ref = A.copy()
            ref.sum_duplicates()
            got = abs_up(A)
            assert A.has_canonical_format and A.nnz == ref.nnz == 5 and np.array_equal(A.data, ref.data)
            assert type(got) is type(A) and np.array_equal(got.data, modulus_up(ref.data))
            assert np.array_equal(got.indices, ref.indices) and np.array_equal(got.indptr, ref.indptr)
    r, c = rng.integers(0, 4, 30), rng.integers(0, 4, 30)
    v = rng.standard_normal(30) + 1j * rng.standard_normal(30)
    for mk in (sp.coo_matrix, sp.coo_array):
        A = mk((v, (r, c)), shape=(4, 4))
        n0 = A.nnz
        got = abs_up(A)
        assert A.nnz < n0 and type(got) is type(A) and np.array_equal(got.data, modulus_up(A.data))
    B = sp.coo_matrix((v, (r, c)), shape=(4, 4)).tobsr(blocksize=(2, 2))
    assert np.array_equal(abs_up(B).toarray(), modulus_up(B.toarray()))
    Dg = sp.dia_matrix((np.array([[1 + 1j, 2, 3j], [4, 5 - 1j, 6]]), np.array([0, 1])), shape=(3, 3))
    assert np.array_equal(abs_up(Dg).data, modulus_up(Dg.data)) and Dg.data.shape == (2, 3)
    L = sp.lil_matrix(np.diag([1 + 1j, 2j, 3.0]))
    assert abs_up(L).format == "csr" and L.format == "lil"
