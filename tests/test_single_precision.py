"""Inputs stored in a narrower format than binary64 (3.3.0.dev0, certify_ed/precision.py).

Every bound counts binary64 arithmetic (u = 2^-53), while numpy forms the products of float32 or complex64 data in
single precision (u = 2^-24). 3.0.0 to 3.2.3 therefore certified A = [[1, 2^-30], [2^-30, 1]] (entries exact in float32)
with x = (1, 1) and the claimed eigenvalue 1 at radius 8.9e-16 (float32) or 3.6e-15 (complex64), although the stored
matrix has the eigenvalues 1 -+ 2^-30: a correct radius is at least 2^-30. Every entry point now widens such data
exactly to binary64 first, so the statements concern the values as stored and agree with those for a binary64 copy;
float64 and complex128 inputs pass through unchanged. Integers are widened when every value is a binary64 number;
wider formats and non-numeric data are refused.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.certify import Claim, Options, certify

E = 2.0 ** -30                              # the off-diagonal entry; eigenvalues 1 - E and 1 + E
SINGLE = {"float32": "float64", "complex64": "complex128"}
WIDE = {"float32": np.float64, "complex64": np.complex128}


def _A(dtype, fmt="dense"):
    A = np.array([[1, E], [E, 1]], dtype=dtype)
    return sp.csr_matrix(A) if fmt == "sparse" else A


def _x(dtype):
    return np.array([[1], [1]], dtype=dtype)


def _wide(M):
    """The binary64 copy of a single-precision matrix or array."""
    if M is None:
        return None
    dt = WIDE.get(np.dtype(M.dtype).name, M.dtype)
    return M.astype(dt)


def _drop(obj, keys=("environment", "timing", "input_widening", "record_sha256")):
    if isinstance(obj, dict):
        return {k: _drop(v, keys) for k, v in obj.items()
                if k not in keys and not any(w in k for w in ("second", "time", "wall"))}
    if isinstance(obj, (list, tuple)):
        return [_drop(v, keys) for v in obj]
    return obj


def _same_certificate(a, b):
    """Equal bodies (without the environment, the times and arithmetic.input_widening) and equal arrays, bit for bit."""
    assert json.dumps(_drop(a.body), sort_keys=True, default=repr) == json.dumps(_drop(b.body), sort_keys=True, default=repr)
    assert sorted(a.arrays) == sorted(b.arrays)
    for k in a.arrays:
        x, y = np.asarray(a.arrays[k]), np.asarray(b.arrays[k])
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), k


def _same_record(a, b):
    assert json.dumps(_drop(a), sort_keys=True, default=repr) == json.dumps(_drop(b), sort_keys=True, default=repr)


def _encloses_an_eigenvalue(lower, upper):
    return lower <= 1 - E <= upper or lower <= 1 + E <= upper


def _s2(cert):
    return cert.statement("S2.eigenpairs")


def _widened(cert):
    return {(w["input"], w["dtype"], w["converted_to"])
            for w in (cert.body["arithmetic"].get("input_widening") or {}).get("converted", [])}


# ------------------------------------------------------------------------------------------- certify(): operator, vectors
@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_single_precision_operator_and_vectors(dtype, fmt):
    c = certify(Claim([1.0], _x(dtype), _A(dtype, fmt), kind="subset"))
    s2 = _s2(c)
    assert s2["status"] == "CERTIFIED"
    assert c.arrays["radius"][0] >= E and s2["evidence"]["max_radius"] >= E
    enc = c.body["enclosures"][0]
    assert _encloses_an_eigenvalue(enc["lower"], enc["upper"])
    assert _widened(c) == {("claim.operator", dtype, SINGLE[dtype]), ("claim.eigenvectors", dtype, SINGLE[dtype])}
    assert c.body["arithmetic"]["input_widening"]["exact"] is True
    # the same certificate as for the binary64 copy of the data (the widening is exact), hashes included
    _same_certificate(c, certify(Claim([1.0], _wide(_x(dtype)), _wide(_A(dtype, fmt)), kind="subset")))


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_single_precision_operator_with_binary64_vectors_and_back(dtype, fmt):
    """The mixed cases: a single-precision operator with binary64 vectors, and the reverse (whose x-norm bound was
    formed in single precision before)."""
    for op_dt, x_dt in ((dtype, SINGLE[dtype]), (SINGLE[dtype], dtype)):
        c = certify(Claim([1.0], _x(x_dt), _A(op_dt, fmt), kind="subset"))
        ref = certify(Claim([1.0], _x(SINGLE[dtype]), _A(SINGLE[dtype], fmt), kind="subset"))
        assert _s2(c)["status"] == "CERTIFIED" and c.arrays["radius"][0] >= E
        _same_certificate(c, ref)


def _pauli_spec():
    return ce.ModelSpec.from_pauli_strings({"I": 1.0, "X": E})     # H = I + 2^-30 sigma_x = A, exactly


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("sector", [False, True])
def test_certify_with_spec_and_sector_isometry(dtype, fmt, sector):
    spec = _pauli_spec()
    V = sp.identity(2, format="csr", dtype=dtype) if sector else None
    c = certify(Claim([1.0], _x(dtype), _A(dtype, fmt), kind="subset", sector_isometry=V), spec)
    assert c.statement("S1.model_fidelity")["status"] == "CERTIFIED"
    assert _s2(c)["status"] == "CERTIFIED" and _s2(c)["evidence"]["max_radius"] >= E and c.arrays["radius"][0] >= E
    want = {("claim.operator", dtype, SINGLE[dtype]), ("claim.eigenvectors", dtype, SINGLE[dtype])}
    if sector:
        want.add(("claim.sector_isometry", dtype, SINGLE[dtype]))
    assert _widened(c) == want
    ref = certify(Claim([1.0], _wide(_x(dtype)), _wide(_A(dtype, fmt)), kind="subset", sector_isometry=_wide(V)), spec)
    _same_certificate(c, ref)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_streamed_single_precision_vectors(tmp_path, dtype):
    """Vectors in a numpy memmap (and an HDF5 dataset when h5py is installed) are widened block by block; the hash and
    the stored vectors are those of the binary64 copy."""
    x = _x(dtype)
    mm = np.lib.format.open_memmap(str(tmp_path / "x.npy"), mode="w+", dtype=dtype, shape=x.shape)
    mm[:] = x
    mm.flush()
    sources = [np.load(str(tmp_path / "x.npy"), mmap_mode="r")]
    try:
        import h5py
        f = h5py.File(str(tmp_path / "x.h5"), "w")
        f.create_dataset("x", data=x)
        sources.append(f["x"])
    except ImportError:
        f = None
    try:
        ref = certify(Claim([1.0], _wide(x), _A(SINGLE[dtype], "sparse"), kind="subset"), options=Options(store_eigenvectors=True))
        for X in sources:
            c = certify(Claim([1.0], X, _A(dtype, "sparse"), kind="subset"), options=Options(store_eigenvectors=True))
            assert _s2(c)["status"] == "CERTIFIED" and c.arrays["radius"][0] >= E
            assert c.arrays["eigenvectors"].dtype == np.dtype(SINGLE[dtype])
            assert c.body["claim"]["eigenvectors_sha256"] == ref.body["claim"]["eigenvectors_sha256"]
            _same_certificate(c, ref)
    finally:
        if f is not None:
            f.close()


def _claim_all(dtype=np.float64):
    """Both eigenpairs of the binary64 A (isolated levels 1 -+ 2^-30), for the S4 records."""
    s = np.sqrt(0.5)
    X = np.array([[s, s], [-s, s]], dtype=dtype)
    return Claim([1 - E, 1 + E], X, _A(np.float64, "sparse"), kind="all")


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_single_precision_observable_and_symmetries(dtype, fmt):
    """Options.observables, approximate_symmetries and symmetries in single precision. ||A||_inf = 1 + 2^-30 was
    computed as 1 in single precision, below the true norm, in the observable's bound."""
    O = _A(dtype, fmt)
    d = np.array([0.5, -0.5], dtype=np.float32 if dtype == "float32" else np.complex64).real.astype(dtype)
    opts = Options(eigenspaces="all", observables={"O": O}, approximate_symmetries={"S": (O, None)},
                   symmetries={"d": (np.real(d), None)})
    c = certify(_claim_all(), options=opts)
    ref = certify(_claim_all(), options=Options(eigenspaces="all", observables={"O": _wide(O)},
                                                 approximate_symmetries={"S": (_wide(O), None)},
                                                 symmetries={"d": (_wide(np.real(d)), None)}))
    s4 = c.statement("S4.eigenspaces")
    assert s4["status"] == "CERTIFIED"
    recs = [r for r in s4["evidence"]["clusters"] if r.get("status") == "CERTIFIED"]
    assert len(recs) == 2 and all(r["observables"]["O"]["O_norm_bound"] >= 1 + E for r in recs)
    assert {w[0] for w in _widened(c)} == {"options.observables[O]", "options.approximate_symmetries[S]",
                                           "options.symmetries[d]"}
    _same_certificate(c, ref)


# ------------------------------------------------------------------------------------------- the module-level entry points
@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_operator_adapter_and_enclose(dtype, fmt):
    from certify_ed.enclosure import OperatorAdapter, enclose
    ad = OperatorAdapter(_A(dtype, fmt))
    assert (ad.A.dtype if not sp.issparse(ad.A) else ad.A.data.dtype) == np.dtype(SINGLE[dtype])
    enc = enclose(ad, [1.0], _x(dtype))
    assert enc.rigorous and enc.radius[0] >= E and _encloses_an_eigenvalue(*enc.interval(0))
    ref = enclose(OperatorAdapter(_wide(_A(dtype, fmt))), [1.0], _wide(_x(dtype)))
    assert enc.radius.tobytes() == ref.radius.tobytes()
    assert enc.residual.upper.tobytes() == ref.residual.upper.tobytes()


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_kato_temple_and_rayleigh_interval(dtype, fmt):
    from certify_ed.enclosure import OperatorAdapter, rayleigh_interval
    x = _x(dtype).ravel()
    lo, hi, th, r = rayleigh_interval(OperatorAdapter(_A(dtype, fmt)), x)
    assert lo <= 1 + E <= hi                               # theta(x) = 1 + 2^-30 exactly for x = (1, 1)
    assert (lo, hi, th, r) == rayleigh_interval(OperatorAdapter(_wide(_A(dtype, fmt))), _wide(x))


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_enclose_spectrum(dtype, fmt):
    from certify_ed.spectrum import enclose_spectrum
    res = enclose_spectrum(_A(dtype, fmt))
    ref = enclose_spectrum(_wide(_A(dtype, fmt)))
    got = [(iv.lo, iv.hi, iv.count) for iv in res.intervals]
    assert got == [(iv.lo, iv.hi, iv.count) for iv in ref.intervals]
    for lam in (1 - E, 1 + E):
        assert any(lo <= lam <= hi for lo, hi, _ in got)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_inertia_counts(dtype):
    """sigma = 1 - 2^-29 rounds to 1 in single precision, so A - sigma I was factorized as A - I (one negative
    eigenvalue) although both eigenvalues of A lie above sigma."""
    from certify_ed.inertia import inertia, inertia_dense, inertia_sparse
    sigma = 1 - 2 * E
    for fn, A in ((inertia_sparse, _A(dtype, "sparse")), (inertia_dense, _A(dtype)), (inertia, _A(dtype, "sparse")),
                  (inertia, _A(dtype))):
        r = fn(A, sigma)
        assert r.conclusive and r.negative == 0 and r.beta < E, (fn.__name__, r)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_variational(dtype):
    from certify_ed.variational import TrialState, certify_variational
    def trial(v):
        return TrialState(v, "dense", {"form": "dense"})
    for make in (lambda v: v, trial):
        rec = certify_variational(_A(dtype, "sparse"), make(_x(dtype).ravel()))
        v2 = next(s for s in rec["statements"] if s["id"] == "V2.existence")
        assert v2["status"] == "CERTIFIED" and _encloses_an_eigenvalue(*v2["evidence"]["interval"])
        assert {w["input"] for w in rec["input_widening"]["converted"]} == {"operator", "trial state"}
        _same_record(rec, certify_variational(_wide(_A(dtype, "sparse")), make(_wide(_x(dtype).ravel()))))
    # MPS tensors and sparse CI amplitudes
    mps = [np.array([[[1.0], [0.0]]], dtype=dtype), np.array([[[1.0], [1.0]]], dtype=dtype)]   # e_0 (x) (1, 1)
    rec = certify_variational(np.kron(np.eye(2), _A(dtype)), {"mps": mps})
    assert any(w["input"].startswith("MPS tensor") for w in rec["input_widening"]["converted"])
    _same_record(rec, certify_variational(np.kron(np.eye(2), _wide(_A(dtype))), {"mps": [_wide(t) for t in mps]}))
    ci = {"indices": [0, 1], "amplitudes": _x(dtype).ravel(), "n": 2}
    rec = certify_variational(_A(dtype), ci)
    _same_record(rec, certify_variational(_wide(_A(dtype)), dict(ci, amplitudes=_wide(_x(dtype).ravel()))))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_generalized(dtype):
    from certify_ed.generalized import certify_generalized
    B = np.eye(2, dtype=dtype)
    rec = certify_generalized(_A(dtype), B, [1.0], _x(dtype), kind="subset")
    g2 = next(s for s in rec["statements"] if s["id"].startswith("G2"))
    assert g2["status"] == "CERTIFIED", g2
    radius = g2["evidence"]["pairs"][0]["radius"] if "pairs" in g2["evidence"] else g2["evidence"]["max_radius"]
    assert radius >= E
    assert {w["input"] for w in rec["input_widening"]["converted"]} == {"A", "B", "eigenvectors"}
    _same_record(rec, certify_generalized(_wide(_A(dtype)), _wide(B), [1.0], _wide(_x(dtype)), kind="subset"))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_nonhermitian(dtype):
    """The non-Hermitian record converted its inputs to binary64 already (exactly for single precision); it now does
    so through precision.binary64 and records it."""
    from certify_ed.nonhermitian import certify_nonhermitian
    rec = certify_nonhermitian(_A(dtype, "sparse"), [1.0], _x(dtype), kind="subset")
    n1 = next(s for s in rec["statements"] if s["id"].startswith("N1"))
    assert n1["status"] == "CERTIFIED" and rec["pairs"][0]["eps_right"] >= E
    assert {w["input"] for w in rec["input_widening"]["converted"]} == {"A", "eigenvectors"}
    _same_record(rec, certify_nonhermitian(_wide(_A(dtype, "sparse")), [1.0], _wide(_x(dtype)), kind="subset"))


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certified_moments(dtype, fmt):
    from certify_ed.moments import certified_moments
    X = (_x(dtype) * np.sqrt(0.5)).astype(dtype)
    rec = certified_moments(_A(dtype, fmt), _A(dtype, fmt), X, 1e-6, 1e-6, K=2)
    assert rec["operator"]["H_norm_upper"] >= 1 + E and rec["operator"]["O_norm_upper"] >= 1 + E
    assert {w["input"] for w in rec["input_widening"]["converted"]} == {"H", "O", "cluster vectors"}
    _same_record(rec, certified_moments(_wide(_A(dtype, fmt)), _wide(_A(dtype, fmt)), _wide(X), 1e-6, 1e-6, K=2))


def _ec():
    from certify_ed.eigenspace import certify_eigenspace
    return certify_eigenspace(np.ones((2, 1)), [0], np.array([1.0]), 1e-12, 1e-12, 1.0)


@pytest.mark.parametrize("fmt", ["dense", "sparse"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_eigenspace_correlation_and_label_records(dtype, fmt):
    from certify_ed.correlations import certify_observable_general, compress_operator
    from certify_ed.davis_kahan import approximate_symmetry_label, certified_fidelity, commutator_bound
    from certify_ed.eigenspace import certify_observable, certify_quantum_numbers
    X, O = _x(dtype) * 1, _A(dtype, fmt)
    Xw, Ow = _wide(X), _wide(O)
    r = certify_observable(_ec(), X, O, "O")
    assert r["O_norm_bound"] >= 1 + E and r == certify_observable(_ec(), Xw, Ow, "O")
    assert certify_observable_general(_ec(), X, O, "O") == certify_observable_general(_ec(), Xw, Ow, "O")
    Sd = sp.diags(np.array([1.0, E], dtype=dtype))
    assert certify_quantum_numbers(_ec(), X, Sd, "S") == certify_quantum_numbers(_ec(), Xw, _wide(Sd), "S")
    cb = commutator_bound(O, Sd)
    assert cb == commutator_bound(Ow, _wide(Sd))
    assert approximate_symmetry_label(_ec(), X, sp.csr_matrix(O), "S", 0.0, [1.0, 1.0]) == \
        approximate_symmetry_label(_ec(), Xw, sp.csr_matrix(Ow), "S", 0.0, [1.0, 1.0])
    assert certified_fidelity(X, 0.0, 0.0, X, 0.0, 0.0) == certified_fidelity(Xw, 0.0, 0.0, Xw, 0.0, 0.0)
    V = sp.identity(2, format="csr", dtype=dtype)
    a, b = compress_operator(V, O), compress_operator(_wide(V), Ow)
    assert (a[0] != b[0]).nnz == 0 and a[1:] == b[1:] and a[2] >= 1 + E


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_sector_audits(dtype):
    from certify_ed.sectors import audit_decomposition, audit_orthogonality, audit_sector
    H, V = _A(dtype, "sparse"), sp.identity(2, format="csr", dtype=dtype)
    assert audit_sector(H, V, H).to_dict() == audit_sector(_wide(H), _wide(V), _wide(H)).to_dict()
    V1 = sp.csr_matrix(np.array([[1.0], [E]], dtype=dtype))
    V2 = sp.csr_matrix(np.array([[-E], [1.0]], dtype=dtype))
    assert audit_orthogonality([V1, V2]) == audit_orthogonality([_wide(V1), _wide(V2)])
    d = audit_decomposition(H, [V1, V2], [H[:1, :1], H[1:, 1:]])
    w = audit_decomposition(_wide(H), [_wide(V1), _wide(V2)], [_wide(H[:1, :1]), _wide(H[1:, 1:])])
    assert d.to_dict() == w.to_dict()


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_ozaki_and_arb_paths(dtype):
    from certify_ed.ozaki import ozaki_gram, ozaki_residual
    X = np.array([[1.0, E], [E, 1.0]], dtype=dtype)
    a, b = ozaki_gram(X), ozaki_gram(_wide(X))
    assert a.G.tobytes() == b.G.tobytes() and a.E.tobytes() == b.E.tobytes()
    r, s = ozaki_residual(_A(dtype), [1.0], _x(dtype)), ozaki_residual(_wide(_A(dtype)), [1.0], _wide(_x(dtype)))
    assert r.upper.tobytes() == s.upper.tobytes() and r.upper[0] >= E
    pytest.importorskip("flint")
    from certify_ed.rigorous import arb_residual_radii
    assert arb_residual_radii(_A(dtype), [1.0], _x(dtype))[0] >= E


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_fp_norm_bounds(dtype):
    """1 + 2^-26 rounds to 1 in single precision: the norm bounds were below the exact norms."""
    from certify_ed.fp import exact_dot, fro_upper, inf_norm_upper, norm2_lower, norm2_upper
    v = np.array([1.0, 2.0 ** -13], dtype=dtype)
    exact = np.sqrt(1 + 2.0 ** -26)
    assert norm2_lower(v) <= exact <= norm2_upper(v) and fro_upper(v[None, :]) >= exact
    assert inf_norm_upper(_A(dtype)) >= 1 + E and inf_norm_upper(_A(dtype, "sparse")) >= 1 + E
    s, err = exact_dot(v, v)
    assert abs(s - (1 + 2.0 ** -26)) <= err


# ------------------------------------------------------------------------------------------- binary64 inputs, integers, refusals
def test_binary64_inputs_pass_through_unchanged(tmp_path):
    from certify_ed.precision import binary64
    widened = []
    for a in (np.ones(3), np.ones(3, dtype=complex), sp.random(4, 4, density=0.5, format="csr"),
              np.zeros((2, 2), dtype=np.longdouble) if np.finfo(np.longdouble).nmant == 52 else np.ones(1)):
        assert binary64(a, "a", widened) is a
    mm = np.lib.format.open_memmap(str(tmp_path / "m.npy"), mode="w+", dtype=np.float64, shape=(3, 2))
    assert binary64(mm, "m", widened) is mm
    assert widened == []
    c = certify(Claim([1.0], _x(np.float64), _A(np.float64), kind="subset"))
    assert "input_widening" not in c.body["arithmetic"]


def test_widened_sparse_matrix_keeps_its_structure():
    from certify_ed.precision import binary64
    D = sp.csr_matrix((np.array([0.5, 0.5, 1.0], dtype=np.float32), np.array([1, 1, 0]), np.array([0, 2, 3])),
                      shape=(2, 2))                             # a repeated entry in row 0
    D.indices, D.indptr = D.indices.astype(np.int64), D.indptr.astype(np.int64)
    W = binary64(D)
    assert W is not D and W.data.dtype == np.float64 and W.indices.dtype == W.indptr.dtype == np.int64
    assert np.array_equal(W.indices, D.indices) and np.array_equal(W.indptr, D.indptr) and W.nnz == 3
    assert W.data.tolist() == [0.5, 0.5, 1.0] and D.data.dtype == np.float32


def test_float16_is_widened_exactly():
    from certify_ed.precision import binary64
    h = np.array([1, 2.0 ** -24, 65504], dtype=np.float16)
    w = []
    out = binary64(h, "h", w)
    assert out.dtype == np.float64 and out.tolist() == [1.0, 2.0 ** -24, 65504.0]
    assert w == [{"input": "h", "dtype": "float16", "converted_to": "float64"}]


def test_integer_input_is_widened_when_exact():
    A = np.array([[2, 1], [1, 2]], dtype=np.int64)              # eigenvalues 1 and 3
    c = certify(Claim([3.0], np.array([[1], [1]], dtype=np.int32), A, kind="subset"))
    assert _s2(c)["status"] == "CERTIFIED"
    assert _widened(c) == {("claim.operator", "int64", "float64"), ("claim.eigenvectors", "int32", "float64")}
    _same_certificate(c, certify(Claim([3.0], np.array([[1.0], [1.0]]), A.astype(float), kind="subset")))
    from certify_ed.precision import binary64
    assert binary64(np.array([2 ** 60, -(2 ** 60)], dtype=np.int64)).tolist() == [2.0 ** 60, -2.0 ** 60]


def test_integers_binary64_cannot_hold_are_refused():
    from certify_ed.precision import PrecisionError, binary64
    big = np.array([[2 ** 53 + 1, 0], [0, 1]], dtype=np.int64)
    with pytest.raises(PrecisionError, match="not exactly representable"):
        certify(Claim([1.0], _x(np.float64), big, kind="subset"))
    with pytest.raises(PrecisionError, match="not exactly representable"):
        certify(Claim([2 ** 53 + 1], _x(np.float64), _A(np.float64), kind="subset"))
    with pytest.raises(PrecisionError):
        binary64(np.array([2 ** 64 - 1], dtype=np.uint64))
    assert issubclass(PrecisionError, ValueError)


def test_non_numeric_and_wider_types_are_refused():
    from certify_ed.precision import PrecisionError, binary64, target_dtype
    obj = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=object)
    with pytest.raises(PrecisionError, match="not numeric"):
        certify(Claim([1.0], _x(np.float64), obj, kind="subset"))
    with pytest.raises(PrecisionError, match="not numeric"):
        binary64(np.array(["a"]))
    if np.finfo(np.longdouble).nmant > 52:                     # x87 extended or quad long double
        with pytest.raises(PrecisionError, match="wider than binary64"):
            certify(Claim([1.0], _x(np.float64), _A(np.longdouble), kind="subset"))
        with pytest.raises(PrecisionError, match="wider than binary64"):
            certify(Claim(np.array([1.0], dtype=np.longdouble), _x(np.float64), _A(np.float64), kind="subset"))
    else:                                                      # long double is binary64 here (MSVC, some ARM ABIs)
        assert target_dtype(np.longdouble) is None and target_dtype(np.clongdouble) is None
        c = certify(Claim([1.0], _x(np.longdouble), _A(np.longdouble), kind="subset"))
        _same_certificate(c, certify(Claim([1.0], _x(np.float64), _A(np.float64), kind="subset")))


def test_widened_streamed_columns_hash_like_the_binary64_array(tmp_path, monkeypatch):
    from certify_ed import certificate_v2
    from certify_ed.certificate_v2 import array_sha256
    from certify_ed.enclosure import StreamedColumns, read_columns
    x = (np.arange(12, dtype=np.float32) / 7).reshape(6, 2).astype(np.float32)
    mm = np.lib.format.open_memmap(str(tmp_path / "x.npy"), mode="w+", dtype=np.float32, shape=x.shape)
    mm[:] = x
    w = StreamedColumns(mm, dtype=np.float64)
    assert w.dtype == np.float64 and w[:, 0].dtype == np.float64 and read_columns(w, [1, 0]).dtype == np.float64
    assert np.array_equal(read_columns(w, [1, 0]), x.astype(np.float64)[:, [1, 0]])
    monkeypatch.setattr(certificate_v2, "_HASH_CHUNK_BYTES", 16)       # several row chunks
    assert array_sha256(w) == array_sha256(x.astype(np.float64)) != array_sha256(mm)
