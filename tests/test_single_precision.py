"""Inputs stored in a narrower format than binary64 (3.3.0.dev0, certify_ed/precision.py).

Every bound counts binary64 arithmetic (u = 2^-53), while numpy forms the products of float32 or complex64 data in
single precision (u = 2^-24). 3.0.0 to 3.2.3 therefore certified A = [[1, 2^-30], [2^-30, 1]] (entries exact in float32)
with x = (1, 1) and the claimed eigenvalue 1 at radius 8.9e-16 (float32) or 3.6e-15 (complex64), although the stored
matrix has the eigenvalues 1 -+ 2^-30: a correct radius is at least 2^-30. Every entry point now widens such data
exactly to binary64 first, so the statements concern the values as stored and agree with those for a binary64 copy;
float64 and complex128 inputs pass through unchanged. Integers are widened when every value is a binary64 number;
wider formats and non-numeric data are refused.
"""
import importlib.util
import json

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.certify import Claim, Options, certify
from conftest import eigh_dense

E = 2.0 ** -30                              # the off-diagonal entry; eigenvalues 1 - E and 1 + E
SINGLE = {"float32": "float64", "complex64": "complex128"}
WIDE = {"float32": np.float64, "complex64": np.complex128}


def _A(dtype, fmt="dense"):
    A = np.array([[1, E], [E, 1]], dtype=dtype)
    return sp.csr_matrix(A) if fmt == "sparse" else A


def _x(dtype):
    return np.array([[1], [1]], dtype=dtype)


def _wide(M):
    """The binary64 copy of a single-precision matrix or array; a sparse matrix keeps its stored entries as given
    (scipy's astype would sum repeated entries)."""
    if M is None:
        return None
    dt = WIDE.get(np.dtype(M.dtype).name, M.dtype)
    if sp.issparse(M):
        W = M.copy()
        W.data = M.data.astype(dt)
        return W
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
    enc = enclose(ad, [1.0], _x(dtype))
    assert enc.rigorous and enc.radius[0] >= E and _encloses_an_eigenvalue(*enc.interval(0))
    assert (ad.A.dtype if not sp.issparse(ad.A) else ad.A.data.dtype) == np.dtype(SINGLE[dtype])
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
    from certify_ed.certificate_v2 import array_sha256
    assert w == [{"input": "h", "dtype": "float16", "converted_to": "float64", "source_sha256": array_sha256(h)}]


def test_integer_inputs_are_checked_and_not_recorded():
    """Integer inputs were not affected: they are checked (every value a binary64 number), used as given and not
    recorded, so their certificates are those of 3.x (tests compare with cd0ca97 in the methods record). Integer
    vectors with an integer operator, which 3.x refused with a numpy casting error, now certify."""
    from certify_ed.certificate_v2 import array_sha256
    A = np.array([[2, 1], [1, 2]], dtype=np.int64)              # eigenvalues 1 and 3
    xi = np.array([[1], [1]], dtype=np.int32)
    c = certify(Claim([3.0], xi, A, kind="subset"), options=Options(store_eigenvectors=True))
    assert _s2(c)["status"] == "CERTIFIED" and "input_widening" not in c.body["arithmetic"]
    assert c.arrays["eigenvectors"].dtype == np.int32
    assert c.body["claim"]["eigenvectors_sha256"] == array_sha256(xi)
    ref = certify(Claim([3.0], xi.astype(float), A.astype(float), kind="subset"))
    assert c.arrays["radius"].tobytes() == ref.arrays["radius"].tobytes()
    assert c.body["subject"]["operator"]["matrix_sha256"] == ref.body["subject"]["operator"]["matrix_sha256"]
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
    from fractions import Fraction
    obj = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=object)       # exact values: accepted (one rule, precision.py)
    _same_certificate(certify(Claim([1.0], _x(np.float64), obj, kind="subset")),
                      certify(Claim([1.0], _x(np.float64), obj.astype(float), kind="subset")))
    with pytest.raises(PrecisionError, match="not exactly a binary64 number"):
        certify(Claim([1.0], _x(np.float64), np.array([[Fraction(1, 3), 0], [0, 1]], dtype=object), kind="subset"))
    with pytest.raises(PrecisionError, match="not numeric"):
        certify(Claim([1.0], _x(np.float64), np.array([["a", 0], [0, 1]], dtype=object), kind="subset"))
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


# ------------------------------------------------------------------------------------------- round 1 of the review
T = 2.0 ** -25                              # 1 + T rounds to 1 in single precision


def _repeated(fmt, dtype):
    """diag(1 + 2^-25, 3), entry (0, 0) stored as 1 and 2^-25: summed in single precision they round to 1."""
    data = np.array([1, T, 3], dtype=dtype)
    if fmt == "coo":
        return sp.coo_matrix((data, (np.array([0, 0, 1]), np.array([0, 0, 1]))), shape=(2, 2))
    M = sp.csr_matrix((data, np.array([0, 0, 1]), np.array([0, 2, 3])), shape=(2, 2))       # repeated index in row 0
    assert not M.has_canonical_format
    return M


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_repeated_entries_are_summed_in_binary64(dtype, fmt):
    """Every entry point widens the caller's matrix before a format conversion (tocsr, csr_matrix, csc_matrix,
    toarray), which sums repeated entries in the matrix's own precision."""
    from certify_ed.certificate_v2 import matrix_sha256
    from certify_ed.correlations import certify_observable_general, certify_observable_lifted, compress_operator
    from certify_ed.davis_kahan import commutator_bound
    from certify_ed.enclosure import OperatorAdapter, enclose
    from certify_ed.generalized import certify_generalized
    from certify_ed.inertia import inertia_sparse
    from certify_ed.nonhermitian import certify_nonhermitian
    from certify_ed.sectors import audit_sector
    from certify_ed.spectrum import enclose_spectrum
    from certify_ed.variational import certify_variational
    M = _repeated(fmt, dtype)

    def W():               # a fresh binary64 copy for each comparison: OperatorAdapter sums a binary64 caller's matrix
        return _repeated(fmt, SINGLE[dtype])         # in place (CHANGELOG, "The caller's matrices stay as given")
    lam = 1 + T                                             # the eigenvalue at e_0 (the other one is 3)
    x = np.array([[1.0], [0.0]])
    enc = enclose(OperatorAdapter(M), [1.0], x)
    assert enc.radius[0] >= T and enc.interval(0)[0] <= lam <= enc.interval(0)[1]
    assert enc.radius.tobytes() == enclose(OperatorAdapter(W()), [1.0], x).radius.tobytes()
    v2 = next(s for s in certify_variational(M, x.ravel())["statements"] if s["id"] == "V2.existence")
    assert v2["evidence"]["interval"][0] <= lam <= v2["evidence"]["interval"][1]
    r = inertia_sparse(M, 1 + T / 2)                        # both eigenvalues lie above the shift
    assert r.conclusive and r.negative == 0
    a = audit_sector(M, sp.identity(2), sp.diags([lam, 3.0]))
    assert a.invariance_defect < 1e-12 and a.to_dict() == audit_sector(W(), sp.identity(2), sp.diags([lam, 3.0])).to_dict()
    S = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert commutator_bound(M, S) == commutator_bound(W(), S)
    ob = certify_observable_general(_ec_at(x), x, M, "M")
    assert abs(ob["value"] - lam) <= ob["bound"]
    Oh, rnd, _ = compress_operator(sp.identity(2), M)
    assert abs(Oh[0, 0] - lam) <= rnd and (Oh != compress_operator(sp.identity(2), W())[0]).nnz == 0
    li = certify_observable_lifted(_ec_at(x), x, sp.identity(2), M, "M", 0.0)
    assert abs(li["value"] - lam) <= li["bound"]
    nh = certify_nonhermitian(M, [1.0], x, kind="subset")
    assert nh["pairs"][0]["eps_right"] >= T
    g2 = next(s for s in certify_generalized(M, np.eye(2), [1.0], x, kind="subset")["statements"] if s["id"].startswith("G2"))
    assert (g2["evidence"]["pairs"][0]["radius"] if "pairs" in g2["evidence"] else g2["evidence"]["max_radius"]) >= T
    ivs = enclose_spectrum(M).intervals
    assert all(any(iv.lo <= lam_ <= iv.hi for iv in ivs) for lam_ in (lam, 3.0))
    canonical = sp.diags([lam, 3.0]).astype(SINGLE[dtype]).tocsr()
    c = certify(Claim([1.0], x, M, kind="subset"))
    assert c.arrays["radius"][0] >= T
    assert matrix_sha256(M) == matrix_sha256(canonical) == c.body["subject"]["operator"]["matrix_sha256"]
    flint = pytest.importorskip("flint")       # noqa: F841
    from certify_ed.rigorous import arb_eigenvalues, arb_residual_radii
    assert arb_residual_radii(M, [1.0], x)[0] >= T
    assert any(lo <= lam <= hi for lo, hi in arb_eigenvalues(M))


def _ec_at(x):
    from certify_ed.eigenspace import certify_eigenspace
    return certify_eigenspace(x, [0], np.array([1.0]), 1e-12, 1e-12, 1.0)


def _memmap(tmp_path, name, a):
    m = np.lib.format.open_memmap(str(tmp_path / f"{name}.npy"), mode="w+", dtype=a.dtype, shape=a.shape)
    m[:] = a
    m.flush()
    return np.load(str(tmp_path / f"{name}.npy"), mmap_mode="r")


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_single_precision_memmap_matrices(tmp_path, dtype):
    """A memmap operator, observable, symmetry diagonal or sector isometry becomes an exact binary64 array in memory;
    only claimed eigenvectors are read block by block."""
    A = _A(dtype)
    c = certify(Claim([1.0], _x(SINGLE[dtype]), _memmap(tmp_path, "A", A), kind="subset"))
    assert _s2(c)["status"] == "CERTIFIED" and c.arrays["radius"][0] >= E
    assert ("claim.operator", dtype, SINGLE[dtype]) in _widened(c)
    _same_certificate(c, certify(Claim([1.0], _x(SINGLE[dtype]), _wide(A), kind="subset")))
    d = np.array([0.5, -0.5], dtype=np.float32)
    opts = Options(eigenspaces="all", observables={"O": _memmap(tmp_path, "O", A)},
                   symmetries={"d": (_memmap(tmp_path, "d", d), None)})
    c = certify(_claim_all(), options=opts)
    assert {"options.observables[O]", "options.symmetries[d]"} <= {w[0] for w in _widened(c)}
    _same_certificate(c, certify(_claim_all(), options=Options(eigenspaces="all", observables={"O": _wide(A)},
                                                               symmetries={"d": (_wide(d), None)})))
    spec = _pauli_spec()
    V = np.eye(2, dtype=dtype)
    c = certify(Claim([1.0], _x(dtype), A, kind="subset", sector_isometry=_memmap(tmp_path, "V", V)), spec)
    assert _s2(c)["status"] == "CERTIFIED" and c.arrays["radius"][0] >= E
    _same_certificate(c, certify(Claim([1.0], _wide(_x(dtype)), _wide(A), kind="subset", sector_isometry=_wide(V)), spec))


class _ArrayLike:
    """An object that only exposes __array__ and a dtype (a data frame, a tensor on the CPU, a zarr array)."""

    def __init__(self, a):
        self._a, self.dtype, self.shape = a, a.dtype, a.shape

    def __array__(self, dtype=None, copy=None):
        return self._a if dtype is None else self._a.astype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_array_like_vectors_are_widened_and_recorded(dtype):
    from certify_ed.enclosure import OperatorAdapter, enclose
    opts = Options(store_eigenvectors=True)
    c = certify(Claim([1.0], _ArrayLike(_x(dtype)), _A(SINGLE[dtype]), kind="subset"), options=opts)
    ref = certify(Claim([1.0], _wide(_x(dtype)), _A(SINGLE[dtype]), kind="subset"), options=opts)
    assert c.arrays["radius"][0] == ref.arrays["radius"][0]
    assert c.arrays["eigenvectors"].dtype == np.dtype(SINGLE[dtype])
    assert ("claim.eigenvectors", dtype, SINGLE[dtype]) in _widened(c)
    _same_certificate(c, ref)
    a = enclose(OperatorAdapter(_A(dtype)), [1.0], _ArrayLike(_x(dtype)))
    assert a.radius.tobytes() == enclose(OperatorAdapter(_wide(_A(dtype))), [1.0], _wide(_x(dtype))).radius.tobytes()


def test_widening_entries_carry_the_source_hash_and_lists_are_treated_like_arrays():
    from certify_ed.certificate_v2 import array_sha256
    from certify_ed.precision import source_sha256
    A, x = _A("float32", "sparse"), _x("float32")
    c = certify(Claim([1.0], x, A, kind="subset"))
    rows = {w["input"]: w for w in c.body["arithmetic"]["input_widening"]["converted"]}
    assert rows["claim.eigenvectors"]["source_sha256"] == array_sha256(x)
    assert rows["claim.operator"]["source_sha256"] == source_sha256(A) != source_sha256(_wide(A))
    # integers: an int64 array and the same Python list give the same certificate, without a widening entry
    cs = [certify(_claim_all(), options=Options(eigenspaces="all", symmetries={"d": (d, 1.0)}))
          for d in (np.array([1, -1], dtype=np.int64), [1, -1])]
    assert _widened(cs[0]) == _widened(cs[1]) == set()
    _same_certificate(*cs)


@pytest.mark.parametrize("n", [2, 8, 100, 1000])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_fp_norm_bounds_against_exact_sums(dtype, n):
    """norm2_lower(v)^2 <= ||v||^2 <= norm2_upper(v)^2 with the exact sum of squares (Fraction)."""
    from fractions import Fraction
    from certify_ed.fp import norm2_lower, norm2_upper
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n) + (1j * rng.standard_normal(n) if dtype == "complex64" else 0)
    v = v.astype(dtype)
    comps = np.concatenate([v.real, v.imag]) if dtype == "complex64" else v
    exact = sum(Fraction(float(c)) ** 2 for c in comps)
    assert Fraction(norm2_lower(v)) ** 2 <= exact <= Fraction(norm2_upper(v)) ** 2


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_hermitian_matrix_with_bound(dtype):
    from fractions import Fraction
    from certify_ed.moments import hermitian_matrix_with_bound
    F = [_A(dtype, "sparse"), _A(dtype)]
    T1, E1 = hermitian_matrix_with_bound(F, [2])
    T2, E2 = hermitian_matrix_with_bound([_wide(f) for f in F], [2])
    assert T1.dtype == T2.dtype and T1.tobytes() == T2.tobytes() and E1.tobytes() == E2.tobytes()
    exact = 1 + Fraction(E) ** 2                             # (A^2)_00
    assert abs(Fraction(float(np.real(T1[0, 0]))) - exact) <= Fraction(float(E1[0, 0]))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_pencil_count_below(dtype):
    """sigma = 1 - 2^-29 rounds to 1 in single precision; the pencil (A, I) has no eigenvalue below it."""
    from certify_ed.generalized import pencil_count_below
    B = np.eye(2, dtype=dtype)
    r = pencil_count_below(_A(dtype), B, 1 - 2 * E, 1.0)
    assert r["conclusive"] and r["nu"] == 0
    _same_record(r, pencil_count_below(_wide(_A(dtype)), _wide(B), 1 - 2 * E, 1.0))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_certify_low_energy_single_precision_blocks(dtype):
    import scipy.linalg as la
    from certify_ed.compose import SectorResult, certify_low_energy
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    full = ce.full_state_set(spec)
    wf = la.eigvalsh(H.toarray())
    E_cut = float((wf[2] + wf[3]) / 2)
    secs, ref = [], []
    for nup in range(spec.n_sites + 1):
        idx, found = full.index_of(ce.u1_state_set(spec, nup).states_in_order())
        assert np.all(found)
        V = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (V.T @ H @ V).tocsr()
        ws, Ys = eigh_dense(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        V32, H32, Y32 = V.astype(dtype), Hs.astype(dtype), Ys[:, sel].astype(dtype)   # entries exact in float32
        secs.append(SectorResult(f"nup={nup}", V32, H32, ws[sel], Y32))
        ref.append(SectorResult(f"nup={nup}", _wide(V32), _wide(H32), ws[sel], _wide(Y32)))
    a = certify_low_energy(spec, secs, E_cut, H_full=H.astype(dtype))
    b = certify_low_energy(spec, ref, E_cut, H_full=_wide(H.astype(dtype)))
    _same_record(a, b)


def test_cli_refuses_with_a_message(tmp_path, capsys):
    from certify_ed.cli import main
    import certify_ed.precision as precision
    assert ce.PrecisionError is precision.PrecisionError
    npz = tmp_path / "claim.npz"
    np.savez(str(npz), eigenvalues=np.array([2 ** 53 + 1], dtype=np.int64), eigenvectors=_x(np.float64),
             H=_A(np.float64))
    assert main(["certify", str(npz), "-o", str(tmp_path / "c")]) == 2
    assert "refused" in capsys.readouterr().out
    npz32 = tmp_path / "claim32.npz"
    np.savez(str(npz32), eigenvalues=np.array([1.0]), eigenvectors=_x(np.float32), H=_A(np.float32))
    assert main(["certify", str(npz32), "-o", str(tmp_path / "c32")]) == 0
    cert = ce.Certificate.load(str(tmp_path / "c32") + ".json")
    assert cert.arrays["radius"][0] >= E and cert.body["arithmetic"]["input_widening"]["exact"] is True


def test_plan_counts_the_callers_narrower_arrays():
    from certify_ed.cost import narrow_input_mb, plan_for_claim
    n = 400
    A32, x32 = np.eye(n, dtype=np.float32), np.ones((n, 2), dtype=np.float32)
    c32 = Claim([1.0, 1.0], x32, A32, kind="subset")
    c64 = Claim([1.0, 1.0], x32.astype(np.float64), A32.astype(np.float64), kind="subset")
    extra = (A32.nbytes + x32.nbytes) / 2 ** 20
    assert narrow_input_mb(c32, Options()) == extra and narrow_input_mb(c64, Options()) == 0.0
    p32, p64 = plan_for_claim(c32, None, Options()), plan_for_claim(c64, None, Options())
    assert p32.peak_mb == p64.peak_mb + extra and p32.baseline_mb == p64.baseline_mb + extra
    assert len(p32.notes) == len(p64.notes) + 1
    lines = []
    c = certify(c32, options=Options(plan=True, progress=lines.append))
    assert c.body["plan"]["peak_mb"] == round(p32.peak_mb, 1)



# ------------------------------------------------------------------------------------------- round 2 of the review
_FORMATS = ["csr", "csc", "bsr", "coo", "dia", "lil", "dok"]


def _int_A():
    return np.array([[2, 1], [1, 2]], dtype=np.int64)                # eigenvalues 1 and 3


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("dtype", ["float32", "complex64", "int64"])
def test_every_sparse_format_is_widened(dtype, fmt):
    """Every scipy format, LIL and DOK included (one value per position, converted by their own astype), in single
    precision and as integers: binary64 keeps the format, certify() gives the certificate of the binary64 copy, and
    matrix_sha256 is the hash in that certificate."""
    from certify_ed.certificate_v2 import matrix_sha256
    from certify_ed.precision import PrecisionError, binary64
    dense = _int_A() if dtype == "int64" else _A(dtype)
    ref = dense.astype(np.complex128 if dtype == "complex64" else np.float64)
    M, M64 = (getattr(sp, f"{fmt}_matrix")(a) for a in (dense, ref))
    W = binary64(M, "m")
    assert W.format == fmt and W.dtype == M64.dtype and np.array_equal(W.toarray(), ref) and M.dtype == dense.dtype
    lam, x = (3.0, np.array([[1.0], [1.0]])) if dtype == "int64" else (1.0, _x(dtype))
    c = certify(Claim([lam], x, M, kind="subset"))
    assert _s2(c)["status"] == "CERTIFIED"
    if dtype == "int64":                                     # checked, used as given, not recorded
        assert _widened(c) == set()
    else:
        assert ("claim.operator", dtype, M64.dtype.name) in _widened(c) and c.arrays["radius"][0] >= E
    _same_certificate(c, certify(Claim([lam], x.astype(ref.dtype), M64, kind="subset")))
    assert matrix_sha256(M) == matrix_sha256(M64) == c.body["subject"]["operator"]["matrix_sha256"]
    if fmt in ("lil", "dok"):
        if dtype == "int64":
            big = getattr(sp, f"{fmt}_matrix")(np.array([[2 ** 53 + 1, 0], [0, 1]], dtype=np.int64))
            with pytest.raises(PrecisionError, match="not exactly representable"):
                binary64(big, "m")
        else:
            O, O64 = (getattr(sp, f"{fmt}_matrix")(_A(t)) for t in (dtype, SINGLE[dtype]))
            co = certify(_claim_all(), options=Options(eigenspaces="all", observables={"O": O}))
            assert ("options.observables[O]", dtype, SINGLE[dtype]) in _widened(co)
            _same_certificate(co, certify(_claim_all(), options=Options(eigenspaces="all", observables={"O": O64})))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_repeated_entries_in_sector_blocks_and_isometries(dtype):
    """compose.certify_low_energy, sectors.restrict_to_u1 and sectors.audit_label widen a COO block or isometry with
    repeated entries before they convert it. (i) The block of the all-up sector stored as h and t, where h + t is not a
    single-precision number, with h + t in the full matrix: summed in single precision the block would be h and the
    sector not invariant. (ii) The all-down isometry entry stored as 1 and 2^-25: an isometry defect of about 2^-24,
    which a single-precision sum (to 1) would hide in the audit, the label audit and the decomposition's eta."""
    from certify_ed.compose import SectorResult, certify_low_energy
    from certify_ed.sectors import audit_label, restrict_to_u1
    wide = SINGLE[dtype]
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec).tocsr()
    full = ce.full_state_set(spec)
    idx = {nup: full.index_of(ce.u1_state_set(spec, nup).states_in_order())[0] for nup in range(spec.n_sites + 1)}
    k_up, k_down = int(idx[4][0]), int(idx[0][0])
    h = float(H[k_up, k_up].real)
    t = float(np.spacing(np.float32(h))) / 2
    assert np.float32(h) + np.float32(t) == np.float32(h) and h + t != h
    Hf = (H + sp.csr_matrix(([t], ([k_up], [k_up])), shape=H.shape)).tocsr()
    wf = np.linalg.eigvalsh(Hf.toarray())
    E_cut = float((wf[2] + wf[3]) / 2)

    def coo1(n, i, a, b, dt):        # n x 1 (or 1 x 1) COO with the entry (i, 0) stored twice, as a and b
        return sp.coo_matrix((np.array([a, b], dtype=dt), (np.array([i, i]), np.array([0, 0]))), shape=(n, 1))

    secs, ref = [], []
    for nup, ix in idx.items():
        V = sp.csr_matrix((np.ones(ix.size), (ix, np.arange(ix.size))), shape=(H.shape[0], ix.size))
        Hs = (V.T @ Hf @ V).toarray()
        ws, Ys = np.linalg.eigh(Hs)
        sel = np.where(ws < E_cut)[0]
        if nup == 4:
            blk = lambda dt: coo1(1, 0, h, t, dt)                           # noqa: E731
        else:
            blk = lambda dt, Hs=Hs: sp.coo_matrix(Hs.astype(dt))           # noqa: E731
        if nup == 0:
            iso = lambda dt: coo1(H.shape[0], k_down, 1.0, T, dt)           # noqa: E731
        else:
            iso = lambda dt, V=V: V.astype(dt)                              # noqa: E731
        Y = Ys[:, sel].astype(dtype)
        secs.append(SectorResult(f"nup={nup}", iso(dtype), blk(dtype), ws[sel], Y))
        ref.append(SectorResult(f"nup={nup}", iso(wide), blk(wide), ws[sel], _wide(Y)))
    a = certify_low_energy(spec, secs, E_cut, H_full=Hf, label_symmetries=("translation",))
    b = certify_low_energy(spec, ref, E_cut, H_full=Hf, label_symmetries=("translation",))
    _same_record(a, b)
    assert "input_widening" not in b
    assert {w["input"] for w in a["input_widening"]["converted"]} == {f"sector nup={k}: {p}" for k in range(5)
                                                                       for p in ("isometry", "block", "eigenvectors")}
    rec = {s["label"]: s for s in a["sectors"]}
    assert rec["nup=4"]["invariance_defect"] < 1e-12                          # the block is h + t, as in Hf
    assert rec["nup=0"]["isometry_defect"] >= 2 ** -24 and a["eta"] >= 2 ** -24
    assert rec["nup=0"]["label_audit"]["translation"]["isometry_defect_upper"] >= 2 ** -24
    V0 = coo1(H.shape[0], k_down, 1.0, T, dtype)
    Vu = restrict_to_u1(spec, V0)[1]
    assert Vu.dtype == np.dtype(wide) and Vu[0, 0] == 1 + T
    la = audit_label(spec, V0, "translation")
    assert la == audit_label(spec, coo1(H.shape[0], k_down, 1.0, T, wide), "translation")
    assert la["isometry_defect_upper"] >= 2 ** -24

    def mixed(dt):                   # support in two U(1) sectors: audit_label converts V itself (no restriction)
        return sp.coo_matrix((np.array([1.0, T, 2.0 ** -20], dtype=dt), (np.array([k_down, k_down, k_up]),
                                                                         np.zeros(3, dtype=int))), shape=(H.shape[0], 1))
    assert restrict_to_u1(spec, mixed(dtype)) is None
    lm = audit_label(spec, mixed(dtype), "translation")
    assert lm == audit_label(spec, mixed(wide), "translation") and lm["isometry_defect_upper"] >= 2 ** -24


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64"])
def test_array_like_and_list_operators_are_stored_matrices(dtype):
    """An operator given as a list of lists or as an array-like (an object with __array__ and no matvec) is a stored
    matrix whatever its dtype; it is recorded only when a narrower floating format was converted (a list of Python
    floats is binary64 already; integers are not recorded)."""
    dense = _int_A() if dtype == "int64" else _A(dtype)
    lam, x = (3.0, np.array([[1.0], [1.0]])) if dtype == "int64" else (1.0, _x(np.float64))
    ref = certify(Claim([lam], x, dense.astype(np.float64), kind="subset"))
    for op, rec in ((_ArrayLike(dense), dtype == "float32"), (dense.tolist(), False)):
        c = certify(Claim([lam], x, op, kind="subset"))
        assert c.body["subject"]["operator"]["kind"] == "matrix" and _s2(c)["status"] == "CERTIFIED"
        assert _widened(c) == ({("claim.operator", dtype, "float64")} if rec else set())
        _same_certificate(c, ref)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_compose_decomposition_audit_in_binary64(dtype):
    """The decomposition audit of certify_low_energy (eta >= ||W^* W - I||_2, W = [V_1 ... V_m]) forms W^* W from the
    widened isometries: with V entries float32(1/sqrt 2) it was formed in single precision before (eta 8.43e-08 against
    4.84e-08 for the binary64 copy, both above the exact max |G_pq| = 3.42e-08)."""
    from fractions import Fraction
    from certify_ed.compose import SectorResult, certify_low_energy
    spec = ce.heisenberg_chain(2)
    H = ce.build_bitwise(spec).toarray()
    st = list(ce.full_state_set(spec).states_in_order())
    c = float(np.float32(1 / np.sqrt(2)))
    cols = [[(st.index(0), 1.0)], [(st.index(3), 1.0)], [(st.index(1), c), (st.index(2), c)],
            [(st.index(1), c), (st.index(2), -c)]]

    def sectors(dt):
        out = []
        for j, entries in enumerate(cols):
            v = np.zeros((H.shape[0], 1))
            for i, a in entries:
                v[i, 0] = a
            Hs = v.T @ H @ v
            w, Y = np.linalg.eigh(Hs)
            out.append(SectorResult(f"s{j}", sp.csr_matrix(v.astype(dt)), sp.csr_matrix(Hs), w, Y))
        return out
    a = certify_low_energy(spec, sectors(dtype), 10.0, H_full=sp.csr_matrix(H))
    b = certify_low_energy(spec, sectors(SINGLE[dtype]), 10.0, H_full=sp.csr_matrix(H))
    _same_record(a, b)
    Wx = [[Fraction(0)] * len(cols) for _ in range(H.shape[0])]
    for j, entries in enumerate(cols):
        for i, v in entries:
            Wx[i][j] = Fraction(v)
    G = [[sum(Wx[i][p] * Wx[i][q] for i in range(H.shape[0])) - (p == q) for q in range(len(cols))]
         for p in range(len(cols))]
    assert a["eta"] >= max(abs(float(x)) for row in G for x in row)



# ------------------------------------------------------------------------------------------- review round 1
def _coo_files(tmp_path, dtype):
    """The repeated-entry matrix diag(1 + 2^-25, 3) of _repeated as files: scipy COO npz, npz row/col/data, HDF5
    row/col/data group; eigenvalues (1, 3) and vectors I."""
    h5py = pytest.importorskip("h5py")
    M = _repeated("coo", dtype)
    sp.save_npz(str(tmp_path / "H_scipy.npz"), M, compressed=False)
    np.savez(str(tmp_path / "H_rcd.npz"), row=M.row, col=M.col, data=M.data, shape=np.array(M.shape))
    with h5py.File(str(tmp_path / "H.h5"), "w") as f:
        g = f.create_group("H")
        for k, v in (("row", M.row), ("col", M.col), ("data", M.data), ("shape", np.array(M.shape))):
            g.create_dataset(k, data=v)
    np.save(str(tmp_path / "ev.npy"), np.array([1.0, 3.0]))
    np.save(str(tmp_path / "X.npy"), np.eye(2))
    return M, ["H_scipy.npz", "H_rcd.npz", "H.h5"]


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_file_readers_keep_single_precision_coo_files_as_read(tmp_path, dtype):
    """read_operator (claim_from_file, claim_from_bundle, the CLI) returns a single-precision COO file in its stored
    layout, so certify() converts it exactly and sums the repeated entry in binary64: the certificate is that of the
    in-memory matrix, its hash is matrix_sha256 of the file's data, and source_sha256 is that of the arrays as read."""
    from certify_ed.certificate_v2 import matrix_sha256
    from certify_ed.cli import main
    from certify_ed.importers import claim_from_file
    from certify_ed.io.formats import read_operator
    from certify_ed.precision import source_sha256
    M, names = _coo_files(tmp_path, dtype)
    ref = certify(Claim([1.0, 3.0], np.eye(2), M, kind="subset"))
    assert ref.arrays["radius"][0] >= T
    for name in names:
        path = str(tmp_path / name)
        A = read_operator(path)
        assert A.format == "coo" and A.nnz == 3 and A.dtype == np.dtype(dtype)
        c = certify(claim_from_file(str(tmp_path / "ev.npy"), str(tmp_path / "X.npy"), path))
        assert c.arrays["radius"].tobytes() == ref.arrays["radius"].tobytes()
        assert c.body["subject"]["operator"]["matrix_sha256"] == matrix_sha256(M)
        row = {w["input"]: w for w in c.body["arithmetic"]["input_widening"]["converted"]}["claim.operator"]
        assert row["dtype"] == dtype and row["source_sha256"] == source_sha256(A)
        out = str(tmp_path / f"c_{name}")
        assert main(["certify", str(tmp_path / "ev.npy"), "--eigenvectors", str(tmp_path / "X.npy"), "--operator", path,
                     "--kind", "subset", "-o", out]) == 0
        assert ce.Certificate.load(out + ".json").arrays["radius"].tobytes() == ref.arrays["radius"].tobytes()


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_writers_store_the_binary64_copy_of_a_single_precision_sparse_matrix(tmp_path, dtype):
    """write_operator, write_bundle and export_npz would sum a single-precision COO matrix's repeated entries in single
    precision when they convert it to CSR; they write its exact binary64 copy instead, summed in binary64."""
    from certify_ed.importers import claim_from_bundle, claim_from_npz, export_npz
    from certify_ed.io.bundle import write_bundle
    from certify_ed.io.formats import read_operator, write_operator
    M = _repeated("coo", dtype)
    want = sp.csr_matrix(_wide(M)).toarray()                   # (0, 0) = 1 + 2^-25
    have_h5py = importlib.util.find_spec("h5py") is not None       # HDF5 is the optional extra 'hdf5'
    for ext in [e for e in (".npz", ".mtx", ".h5", ".json", ".txt") if e != ".h5" or have_h5py]:
        write_operator(str(tmp_path / f"op{ext}"), M)
        assert np.array_equal(read_operator(str(tmp_path / f"op{ext}")).toarray(), want), ext
    write_bundle(str(tmp_path / "b"), [1.0, 3.0], np.eye(2), operator=M, basis=sp.identity(2, dtype=dtype, format="coo"))
    claim = claim_from_bundle(str(tmp_path / "b"))[0]
    assert np.array_equal(claim.operator.toarray(), want)
    export_npz(str(tmp_path / "e.npz"), [1.0, 3.0], np.eye(2), M)
    assert np.array_equal(claim_from_npz(str(tmp_path / "e.npz")).operator.toarray(), want)
    assert certify(claim).arrays["radius"][0] >= T


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_matrix_affinity_defect_and_sweep_isometry(dtype):
    from certify_ed.sweep import certified_sweep, matrix_affinity_defect
    one, zero = (np.array([[v]], dtype=dtype) for v in (1, 0))
    dc = 1 + 2.0 ** -30                                         # exact defect |1 - dc| = 2^-30
    d = matrix_affinity_defect(zero, one, dc, one)
    assert d >= 2.0 ** -30 and d == matrix_affinity_defect(_wide(zero), _wide(one), dc, _wide(one))
    # a single-precision COO sector isometry with a repeated entry, as sector_fn returns it

    def sector(dt):
        def fn(spec, states):
            return sp.coo_matrix((np.array([1.0, T, 1.0, 1.0, 1.0], dtype=dt), ([0, 0, 1, 2, 3], [0, 0, 1, 2, 3])),
                                 shape=(4, 4))
        return fn

    def solver(H, k):
        w, V = np.linalg.eigh(H.toarray() if sp.issparse(H) else H)
        return w[:k], V[:, :k]
    recs = [certified_sweep(lambda h: ce.tfim_chain(2, 1.0, h), [0.5, 0.6], k=1, solver=solver, sector_fn=sector(dt))
            for dt in (dtype, SINGLE[dtype])]
    certs = [r.pop("_certificates") for r in recs]
    for a, b in zip(*certs):
        assert a.arrays["radius"].tobytes() == b.arrays["radius"].tobytes()
    for r in recs:                                              # certificate hashes carry the time of the run
        r.pop("record_sha256")
        for p in r["points"]:
            p.pop("hash")
    _same_record(*recs)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_symveig_symmetrized_part_and_exact_helpers(dtype):
    from certify_ed.compat.symveig import enclose_global
    from certify_ed.eigenspace import commutes_signed_permutation
    from certify_ed.sectors import trace_moments_check
    H = np.array([[0, 1], [2.0 ** -25, 0]], dtype=dtype)       # Hermitian part: eigenvalues -+(0.5 + 2^-26)
    e = enclose_global(H, symmetrize=True)
    lam = 0.5 + 2.0 ** -26
    assert any(x.lo <= -lam <= x.hi for x in e) and any(x.lo <= lam <= x.hi for x in e)
    assert [(x.lo, x.hi) for x in e] == [(x.lo, x.hi) for x in enclose_global(_wide(H), symmetrize=True)]
    A = sp.coo_matrix((np.array([1.0, T, 0.5, 0.5, 1.0], dtype=dtype), ([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])), shape=(2, 2))
    swap, sign = np.array([1, 0]), np.array([1, 1])           # (0, 0) = 1 + 2^-25 != (1, 1): no commutation
    assert commutes_signed_permutation(A, swap, sign) is False
    assert commutes_signed_permutation(_wide(A), swap, sign) is False
    rng = np.random.default_rng(3)
    B = sp.csr_matrix(rng.standard_normal((12, 12)).astype(dtype))
    assert trace_moments_check(B, [B]) == trace_moments_check(_wide(B), [_wide(B)])


@pytest.mark.parametrize("vdtype", ["float32", "complex64"])
def test_binary64_operator_with_single_precision_vectors(vdtype):
    """3.x formed fp.norm2_lower of single-precision vectors in single precision; it can exceed the exact norm by a
    relative 6e-8, so the radius fell below the true distance: [[1, 2^-20], [2^-20, 1]] with x = (7, 7) in float32,
    accuracy_target 1e-5, gave S2 CERTIFIED with radius 9.536742991635583e-07 < 2^-20."""
    A = sp.csr_matrix(np.array([[1, 2.0 ** -20], [2.0 ** -20, 1]]))
    x = np.array([[7], [7]], dtype=vdtype)
    c = certify(Claim([1.0], x, A, kind="subset", accuracy_target=1e-5))
    assert _s2(c)["status"] == "CERTIFIED" and c.arrays["radius"][0] >= 2.0 ** -20
    ref = certify(Claim([1.0], _wide(x), A, kind="subset", accuracy_target=1e-5))
    assert c.arrays["radius"].tobytes() == ref.arrays["radius"].tobytes()


def test_one_refusal_rule(tmp_path, capsys):
    """Inexact values are refused wherever they enter: integers beyond 2^53 in a list next to floats (Python
    integers, numpy integer scalars, integer arrays), beyond 2^64 or of 2^1024 or more, object values v with
    float(v) != v, an array-like that cannot be converted, and the eigenvalue file of claim_from_file before its
    conversion. Exact object values are accepted, NaN goes on to the finite check, and a complex eigenvalue array of
    a Hermitian claim is converted as in 3.x (the real part, with numpy's ComplexWarning)."""
    from fractions import Fraction
    from certify_ed.cli import main
    from certify_ed.importers import claim_from_file
    from certify_ed.precision import PrecisionError, binary64, check_values
    from certify_ed.variational import certify_variational
    for bad in ([[2 ** 53 + 1, 0.5], [0.5, 1.0]], [[2 ** 70 + 1, 0], [0, 1]]):
        with pytest.raises(PrecisionError, match="not exactly"):
            binary64(bad)
        with pytest.raises(PrecisionError, match="not exactly"):
            certify(Claim([1.0], _x(np.float64), bad, kind="subset"))
    for bad in ([[1.0, np.int64(2 ** 60 + 1)], [np.int64(2 ** 60 + 1), 1.0]],
                [np.array([1.0, 0.0]), np.array([2 ** 60 + 1, 1], dtype=np.int64)], [[1.0, 10 ** 400]]):
        with pytest.raises(PrecisionError, match="not exactly|not a binary64 number"):
            binary64(bad)
    for bad in ([2 ** 53 + 1, 1.5], [2 ** 70 + 1], np.array([Fraction(1, 3)], dtype=object), [0.5, np.int64(2 ** 60 + 1)],
                [np.uint64(2 ** 64 - 1), 1.5], [10 ** 400], [1.0, 10 ** 400]):
        with pytest.raises(PrecisionError, match="not exactly|not a binary64 number"):
            check_values(bad)
    check_values([0.5, np.int64(2 ** 60)])
    nan = binary64(np.array([float("nan"), Fraction(1, 2)], dtype=object))
    assert np.isnan(nan[0]) and nan[1] == 0.5 and nan.dtype == np.float64
    cw = getattr(np, "exceptions", np).ComplexWarning
    with pytest.warns(cw):                                      # as in 3.x: the real part, numpy's warning
        c = certify(Claim(np.array([1 + 2.0 ** -30 + 0.5j]), _x(np.float64), _A(np.float64), kind="subset"))
    ref = certify(Claim([1 + 2.0 ** -30], _x(np.float64), _A(np.float64), kind="subset"))
    assert c.arrays["radius"].tobytes() == ref.arrays["radius"].tobytes()
    assert c.body["claim"]["eigenvalues"] == ref.body["claim"]["eigenvalues"]
    assert "input_widening" not in c.body["arithmetic"]
    A2 = np.diag([1.0, 2.0])
    r = certify_variational(A2, np.array([Fraction(1), Fraction(0)], dtype=object))
    assert r == certify_variational(A2, np.array([1.0, 0.0]))
    with pytest.raises(PrecisionError, match="not exactly"):
        certify_variational(A2, np.array([Fraction(1, 3), Fraction(2, 3)], dtype=object))

    class Device:
        dtype, shape = np.dtype(np.float32), (2, 1)

        def __array__(self, dtype=None, copy=None):
            raise RuntimeError("the data live on a device")
    with pytest.raises(PrecisionError, match="could not be converted"):
        certify(Claim([1.0], Device(), _A(np.float64), kind="subset"))
    np.save(str(tmp_path / "ev.npy"), np.array([2 ** 53 + 1], dtype=np.int64))
    np.save(str(tmp_path / "X.npy"), _x(np.float64))
    np.save(str(tmp_path / "H.npy"), _A(np.float64))
    with pytest.raises(PrecisionError, match="not exactly representable"):
        claim_from_file(str(tmp_path / "ev.npy"), str(tmp_path / "X.npy"), str(tmp_path / "H.npy"))
    assert main(["certify", str(tmp_path / "ev.npy"), "--eigenvectors", str(tmp_path / "X.npy"), "--operator",
                 str(tmp_path / "H.npy"), "-o", str(tmp_path / "c")]) == 2
    assert "certify: refused" in capsys.readouterr().out


def test_every_cli_command_refuses_with_exit_2(tmp_path, capsys):
    from certify_ed.cli import main
    big = np.array([[2 ** 53 + 1, 0], [0, 1]], dtype=np.int64)
    np.save(str(tmp_path / "big.npy"), big)
    np.savez(str(tmp_path / "big.npz"), H=big)
    assert main(["spectrum", str(tmp_path / "big.npz")]) == 2
    assert "spectrum: refused" in capsys.readouterr().out
    np.save(str(tmp_path / "psi.npy"), np.array([1.0, 0.0]))
    assert main(["variational", "--operator", str(tmp_path / "big.npy"), "--state", str(tmp_path / "psi.npy")]) == 2
    assert "variational: refused" in capsys.readouterr().out


def test_binary64_hdf5_dataset_is_a_stored_matrix_like_a_float32_one(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(tmp_path / "A.h5"), "w") as f:
        f.create_dataset("A64", data=_A(np.float64))
        f.create_dataset("A32", data=_A(np.float32))
    with h5py.File(str(tmp_path / "A.h5"), "r") as f:
        c64 = certify(Claim([1.0], _x(np.float64), f["A64"], kind="subset"))
        c32 = certify(Claim([1.0], _x(np.float64), f["A32"], kind="subset"))
    assert c64.body["subject"]["operator"]["kind"] == c32.body["subject"]["operator"]["kind"] == "matrix"
    assert c64.arrays["radius"].tobytes() == c32.arrays["radius"].tobytes()
    assert "input_widening" not in c64.body["arithmetic"] and "input_widening" in c32.body["arithmetic"]


def test_integer_checks_are_vectorized_and_skip_dia_padding(monkeypatch):
    import builtins
    from certify_ed import precision
    from certify_ed.precision import PrecisionError, binary64
    a = np.arange(2_000_000, dtype=np.int64)
    a[7] = 2 ** 60
    calls = []

    def counting_int(*args):
        calls.append(1)
        return builtins.int(*args)
    monkeypatch.setattr(precision, "int", counting_int, raising=False)
    assert binary64(a)[7] == 2.0 ** 60
    assert len(calls) < 100                                     # one comparison, then only the large values
    monkeypatch.undo()
    a[8] = 2 ** 60 + 1
    with pytest.raises(PrecisionError, match="not exactly representable"):
        binary64(a)
    D = sp.dia_matrix((np.array([[2 ** 53 + 1, 2, 3]], dtype=np.int64), [1]), shape=(3, 3))   # first entry: padding
    assert np.array_equal(binary64(D).toarray(), D.toarray().astype(float))


def test_long_double_decision_on_every_platform(monkeypatch):
    """The format decision is classify_format (significand bits, exponent range, item size), so the refusal of the
    x87 80-bit and IEEE 128-bit long double is tested here too, where long double is binary64."""
    from certify_ed import precision
    from certify_ed.precision import PrecisionError, classify_format, target_dtype
    assert classify_format(52, 1024, -1022, 8) == "binary64" and classify_format(52, 1024, -1022, 16, "c") == "binary64"
    assert classify_format(63, 16384, -16381, 16) == "wider" and classify_format(63, 16384, -16381, 12) == "wider"
    assert classify_format(112, 16384, -16381, 16) == "wider"
    assert classify_format(23, 128, -126, 4) == "narrower" and classify_format(10, 16, -14, 2) == "narrower"
    assert classify_format(52, 1024, -1022, 16) == "narrower"   # binary64 padded to 16 bytes: converted exactly
    real = precision._format_of
    monkeypatch.setattr(precision, "_format_of", lambda dt: "wider" if dt == np.dtype(np.float32) else real(dt))
    with pytest.raises(PrecisionError, match="wider than binary64"):
        target_dtype(np.float32, "x")
    with pytest.raises(PrecisionError, match="wider than binary64"):
        certify(Claim([1.0], _x(np.float64), _A(np.float32), kind="subset"))


def test_small_branches(tmp_path):
    """bool, float16 and big-endian inputs, integer vectors on disk, ragged lists and numpy scalars."""
    from certify_ed.precision import PrecisionError, binary64
    for A, X in ((np.eye(2, dtype=bool), _x(np.float64)), (np.eye(2), np.array([[True], [True]]))):
        c = certify(Claim([1.0], X, A, kind="subset"))               # 3.x: TypeError
        assert _s2(c)["status"] == "CERTIFIED" and "input_widening" not in c.body["arithmetic"]
    c16 = certify(Claim([1.0], _x(np.float16), _A(np.float16), kind="subset"))
    assert _s2(c16)["status"] == "CERTIFIED" and ("claim.operator", "float16", "float64") in _widened(c16)
    be = sp.csr_matrix(_A(np.float64))
    be.data = be.data.astype(">f8")
    _same_certificate(certify(Claim([1.0], _x(np.float64), be, kind="subset")),
                      certify(Claim([1.0], _x(np.float64), _A(np.float64, "sparse"), kind="subset")))
    for vals, ok in (([[1], [1]], True), ([[2 ** 53 + 1], [1]], False)):
        mm = np.lib.format.open_memmap(str(tmp_path / f"x{ok}.npy"), mode="w+", dtype=np.int64, shape=(2, 1))
        mm[:] = np.array(vals, dtype=np.int64)
        mm.flush()
        X = np.load(str(tmp_path / f"x{ok}.npy"), mmap_mode="r")
        if ok:
            assert _s2(certify(Claim([1.0], X, _A(np.float64), kind="subset")))["status"] == "CERTIFIED"
        else:
            with pytest.raises(PrecisionError, match="not exactly representable"):
                certify(Claim([1.0], X, _A(np.float64), kind="subset"))
    ragged = [[1.0], [1.0, 2.0]]
    assert binary64(ragged) is ragged                           # not array data: left to the caller
    s = np.float64(1.5)
    assert binary64(s) is s and binary64(np.float32(1.5)).dtype == np.float64



def test_exact_layer_checks_the_claimed_values():
    """The exported entry points of the exact layer apply the one rule to lam: a Fraction 1/3 was rounded silently."""
    from fractions import Fraction
    from certify_ed.checker import exact_eigenpair_radii, exact_recheck_lowest
    from certify_ed.exact_sectors import exact_recheck_sector_lowest
    from certify_ed.precision import PrecisionError
    spec = ce.heisenberg_chain(2)
    x = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2)
    for call in (lambda lam: exact_eigenpair_radii(spec, lam, x), lambda lam: exact_recheck_lowest(spec, lam, x),
                 lambda lam: exact_recheck_sector_lowest(spec, None, [], {}, lam, x)):
        with pytest.raises(PrecisionError, match="not exactly"):
            call([Fraction(1, 3)])
    assert exact_eigenpair_radii(spec, [Fraction(-3, 4)], x) == exact_eigenpair_radii(spec, [-0.75], x)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_comparison_and_invariant_helpers_sum_in_binary64(dtype):
    """construct.matrix_difference (the S1 comparator), construct.hermitian_part, invariants.matrix_invariants (which
    carries a rounding bound) and basis.sector_matrix convert a narrower matrix before they sum its entries."""
    from certify_ed.basis import sector_matrix
    from certify_ed.construct import hermitian_part, matrix_difference
    from certify_ed.invariants import matrix_invariants
    M = _repeated("coo", dtype)                                 # diag(1 + 2^-25, 3), (0, 0) stored as 1 and 2^-25
    W = sp.csr_matrix(_wide(M))
    assert matrix_difference(M, W)["max_abs_diff"] == 0.0
    assert hermitian_part(M)[0, 0] == 1 + T and hermitian_part(M).dtype == np.dtype(SINGLE[dtype])
    assert matrix_invariants(M)["trace"] == 4 + T and matrix_invariants(M) == matrix_invariants(W)
    V = sp.identity(2, dtype=dtype, format="coo")
    assert sector_matrix(M, V).toarray().tolist() == sector_matrix(W, _wide(V)).toarray().tolist()


@pytest.mark.parametrize("fmt", ["dense", "csr"])
def test_integer_matrices_with_large_row_sums(fmt):
    """3.x summed |a_ij| of an int64 matrix in int64: beyond 2^63 the norm bound was negative and S2 FAILED
    spuriously (norm_upper_bound -9.22e18 at cd0ca97); now the bounds are those of the float64 copy."""
    A = np.full((2, 2), 2 ** 62, dtype=np.int64)
    A = A if fmt == "dense" else sp.csr_matrix(A)
    x = np.array([[1.0], [-1.0]])
    c, cf = (certify(Claim([0.0], x, M, kind="subset")) for M in (A, A.astype(float)))
    assert c.body["subject"]["operator"]["norm_upper_bound"] > 0 and _s2(c)["status"] == "CERTIFIED"
    assert c.body["subject"]["operator"]["norm_upper_bound"] == cf.body["subject"]["operator"]["norm_upper_bound"]
    assert c.arrays["radius"].tobytes() == cf.arrays["radius"].tobytes()



@pytest.mark.parametrize("dtype", ["float16", "float32", "complex64"])
def test_matrix_market_writes_narrower_dense_arrays_as_binary64(tmp_path, dtype):
    """With 17 significant digits scipy 1.17's mmwrite stops the process on float32 data and writes an unreadable file
    for complex64 data; write_array (and write_operator and write_basis, which use it for dense arrays) converts a
    narrower dense array exactly first, so the file reads back as its binary64 copy."""
    from certify_ed.io.formats import read_array, read_basis, read_operator, write_array, write_basis, write_operator
    rng = np.random.default_rng(7)
    cplx = dtype == "complex64"
    wide = np.complex128 if cplx else np.float64

    def rnd(*shape):
        return (rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)).astype(dtype)
    X, v, A = rnd(6, 2), rnd(7), rnd(4, 4)
    for name, a in (("X", X), ("v", v)):
        p = str(tmp_path / f"{name}.mtx")
        write_array(p, a)
        b = np.asarray(read_array(p))
        assert b.dtype == wide and np.array_equal(b.reshape(a.shape), a.astype(wide))
    p = str(tmp_path / "A.mtx")
    write_operator(p, A)
    B = read_operator(p)
    assert np.array_equal(B.toarray() if sp.issparse(B) else np.asarray(B), A.astype(wide))
    p = str(tmp_path / "V.mtx")
    write_basis(p, A[:, :2])
    kind, V = read_basis(p)
    assert kind == "isometry" and np.array_equal(V.toarray(), A[:, :2].astype(wide))
