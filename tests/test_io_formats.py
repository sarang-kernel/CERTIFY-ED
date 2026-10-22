"""Universal producer import: round trips for every format, the bundle convention, and rejection of bad input."""
import json
import os

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.basis import magnetization_state_set, translation, cyclic_isometry, sector_matrix
from certify_ed.importers import claim_from_bundle, claim_from_file
from certify_ed.io import (FormatError, read_array, read_basis, read_bundle, read_operator, write_array,
                           write_basis, write_bundle, write_operator)

ARRAY_FMTS = ["npy", "npz", "h5", "csv", "tsv", "txt", "dat", "json", "mtx"]
SPARSE_FMTS = ["npz", "mtx", "h5", "json", "csv", "txt"]


@pytest.fixture(scope="module")
def heis6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


@pytest.mark.parametrize("fmt", ARRAY_FMTS)
def test_real_array_round_trip(tmp_path, fmt):
    rng = np.random.default_rng(1)
    v = rng.standard_normal(7)
    M = rng.standard_normal((5, 3))
    p = str(tmp_path / f"v.{fmt}")
    write_array(p, v)
    assert np.array_equal(read_array(p), v)
    q = str(tmp_path / f"m.{fmt}")
    write_array(q, M, role="eigenvectors")
    assert np.array_equal(read_array(q, role="eigenvectors"), M)


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5", "csv", "txt", "json", "mtx"])
def test_complex_array_round_trip(tmp_path, fmt):
    rng = np.random.default_rng(2)
    M = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    p = str(tmp_path / f"c.{fmt}")
    write_array(p, M, role="eigenvectors")
    assert np.array_equal(read_array(p, role="eigenvectors"), M)


def test_text_header_and_comments_and_complex_pairs(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("# produced by some code\nenergy\n-1.5\n-0.25\n2.0\n")
    assert np.array_equal(read_array(str(p)), [-1.5, -0.25, 2.0])
    q = tmp_path / "pairs.txt"
    q.write_text("1.0 2.0\n3.0 -4.0\n")
    assert np.array_equal(read_array(str(q), complex_pairs=True), [1 + 2j, 3 - 4j])


def test_npz_and_hdf5_keys(tmp_path):
    p = str(tmp_path / "multi.npz")
    np.savez(p, energies=np.arange(3.0), other=np.ones(2))
    assert np.array_equal(read_array(p), np.arange(3.0))              # conventional name
    with pytest.raises(FormatError):
        read_array(p, key="missing")
    h5py = pytest.importorskip("h5py")
    q = str(tmp_path / "m.h5")
    with h5py.File(q, "w") as f:
        f.create_dataset("spectrum/values", data=np.arange(4.0))
        f.create_dataset("eigenvalues", data=np.arange(2.0))
    assert np.array_equal(read_array(q), np.arange(2.0))
    assert np.array_equal(read_array(q, key="spectrum/values"), np.arange(4.0))


@pytest.mark.parametrize("fmt", SPARSE_FMTS)
def test_sparse_operator_round_trip(tmp_path, fmt, heis6):
    _spec, H, _w, _V = heis6
    p = str(tmp_path / f"H.{fmt}")
    write_operator(p, H)
    A = read_operator(p)
    assert sp.issparse(A)
    D = (sp.csr_matrix(A) - H).tocsr()
    D.eliminate_zeros()
    assert D.nnz == 0


@pytest.mark.parametrize("fmt", ["npy", "npz", "h5", "csv", "json", "mtx", "txt"])
def test_dense_operator_round_trip(tmp_path, fmt):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5))
    A = A + A.T
    p = str(tmp_path / f"A.{fmt}")
    write_operator(p, A)
    B = read_operator(p)
    assert not sp.issparse(B) and np.array_equal(B, A)


def test_complex_sparse_operator_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    A = sp.random(8, 8, density=0.3, random_state=5, dtype=complex)
    A = (A + A.conj().T).tocsr()
    A.data += 1j * rng.standard_normal(A.nnz) * 0   # keep complex dtype
    for fmt in ("npz", "mtx", "h5", "json", "csv"):
        if fmt == "h5":
            pytest.importorskip("h5py")
        p = str(tmp_path / f"C.{fmt}")
        write_operator(p, A)
        B = read_operator(p)
        D = (sp.csr_matrix(B) - A).tocsr()
        D.eliminate_zeros()
        assert D.nnz == 0, fmt


def test_legacy_npz_triplets_and_coo_npz(tmp_path, heis6):
    _spec, H, _w, _V = heis6
    p = str(tmp_path / "legacy.npz")
    ce.export_npz(p, np.arange(2.0), H=H)
    assert abs(read_operator(p) - H).max() == 0
    C = H.tocoo()
    q = str(tmp_path / "coo.npz")
    np.savez(q, row=C.row, col=C.col, data=C.data, shape=np.array(C.shape))
    assert abs(read_operator(q) - H).max() == 0


def test_coo_text_layout_explicit(tmp_path):
    p = tmp_path / "op.txt"
    p.write_text("0 0 1.0\n0 1 0.5\n1 0 0.5\n1 1 -1.0\n")
    A = read_operator(str(p), layout="coo", shape=(2, 2))
    assert np.array_equal(A.toarray(), [[1.0, 0.5], [0.5, -1.0]])
    q = tmp_path / "one.csv"
    q.write_text("i,j,v\n1,1,2.0\n2,2,3.0\n")
    A = read_operator(str(q), layout="coo", shape=(2, 2), one_based=True)
    assert np.array_equal(A.toarray(), [[2.0, 0.0], [0.0, 3.0]])


def test_matrix_free_operator_requires_allow_code(tmp_path):
    p = tmp_path / "operator.py"
    p.write_text("import numpy as np\ndim = 4\n_d = np.array([1.0, 2.0, 3.0, 4.0])\n"
                 "def matvec(x):\n    return _d * x\n")
    with pytest.raises(FormatError):
        read_operator(str(p))
    op = read_operator(str(p), allow_code=True)
    assert op.shape == (4, 4)
    assert np.array_equal(op.matvec(np.ones(4)), [1.0, 2.0, 3.0, 4.0])
    # certify a matrix-free claim through it
    w = np.array([1.0, 2.0])
    X = np.eye(4)[:, :2]
    c = ce.certify(ce.Claim(w, X, operator=op, kind="subset"))
    assert c.status_of("S2.eigenpairs") in ("CERTIFIED", "INCONCLUSIVE")


def test_non_finite_rejected(tmp_path):
    p = str(tmp_path / "bad.npy")
    np.save(p, np.array([1.0, np.nan]))
    with pytest.raises(FormatError):
        read_array(p)
    q = str(tmp_path / "badop.npy")
    np.save(q, np.array([[1.0, np.inf], [np.inf, 1.0]]))
    with pytest.raises(FormatError):
        read_operator(q)


def test_non_square_operator_rejected(tmp_path):
    p = str(tmp_path / "rect.npy")
    np.save(p, np.ones((2, 3)))
    with pytest.raises(FormatError):
        read_operator(p)


@pytest.mark.parametrize("fmt", ["npz", "npy", "csv", "json", "mtx"])
def test_basis_round_trip(tmp_path, fmt):
    states = np.array([3, 5, 6, 9, 10, 12], dtype=np.int64)
    p = str(tmp_path / f"basis.{fmt}")
    write_basis(p, states)
    kind, B = read_basis(p)
    assert kind == "states" and np.array_equal(B, states) and B.dtype == np.int64
    rng = np.random.default_rng(6)
    V, _ = np.linalg.qr(rng.standard_normal((6, 3)))
    q = str(tmp_path / f"iso.{fmt}")
    write_basis(q, V)
    kind, W = read_basis(q)
    assert kind == "isometry" and np.array_equal(W.toarray(), V)


@pytest.mark.parametrize("fmt", ARRAY_FMTS)
def test_bundle_round_trip_full_space(tmp_path, fmt, heis6):
    spec, H, w, V = heis6
    d = str(tmp_path / f"bundle_{fmt}")
    write_bundle(d, w[:4], V[:, :4], H, spec, fmt=fmt, kind="lowest",
                 meta={"producer": {"name": "test code", "version": "0"}, "description": "ground state + triplet"})
    b = read_bundle(d)
    assert np.array_equal(b.claim.eigenvalues, w[:4])
    assert np.array_equal(np.asarray(b.claim.eigenvectors), V[:, :4])
    assert abs(sp.csr_matrix(b.claim.operator) - H).max() == 0
    assert b.spec.sha256() == spec.sha256()
    assert b.claim.kind == "lowest" and b.claim.producer["name"] == "test code"
    assert set(b.files) >= {"eigenvalues", "eigenvectors", "operator", "spec", "meta"}
    c = ce.certify(b.claim, b.spec)
    for sid in ("S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces"):
        assert c.status_of(sid) == "CERTIFIED", c.summary()


def test_bundle_with_u1_state_list(tmp_path):
    spec = ce.heisenberg_chain(8)
    states = magnetization_state_set(spec, 0)
    H = ce.build_bitwise(spec, states)
    w, V = la.eigh(H.toarray())
    d = str(tmp_path / "u1")
    write_bundle(d, w[:1], V[:, :1], H, spec, basis=states.states_in_order(), fmt="npz", kind="lowest",
                 meta={"state_order": "descending"})
    claim, spec2, meta = claim_from_bundle(d)
    assert claim.states is not None and claim.states.size == states.size
    assert claim.states.order == "descending"
    c = ce.certify(claim, spec2)
    assert c.status_of("S1.model_fidelity") == "CERTIFIED", c.summary()
    assert c.status_of("S3.completeness") == "CERTIFIED", c.summary()


def test_bundle_with_sector_isometry(tmp_path):
    spec = ce.heisenberg_chain(6)
    st = magnetization_state_set(spec, 0)
    V0 = cyclic_isometry(st, translation(spec), 0)
    full_idx, _ = ce.full_state_set(spec).index_of(st.states_in_order())
    P = sp.csr_matrix((np.ones(st.size), (full_idx, np.arange(st.size))), shape=(spec.dim, st.size))
    V = (P @ V0).tocsr()                     # rows in the spec's full-space encoding (Claim docstring)
    Hs = sector_matrix(ce.build_bitwise(spec, st), V0)
    w, X = la.eigh(Hs.toarray())
    d = str(tmp_path / "k0")
    write_bundle(d, w[:1], X[:, :1], Hs, spec, basis=V, fmt="npz", kind="lowest")
    b = read_bundle(d)
    assert b.claim.sector_isometry is not None and b.claim.sector_isometry.shape == V.shape
    c = ce.certify(b.claim, b.spec)
    assert c.status_of("S1.model_fidelity") == "CERTIFIED", c.summary()


def test_bundle_eigenvectors_as_rows_and_meta_overrides(tmp_path, heis6):
    spec, H, w, V = heis6
    d = tmp_path / "rows"
    d.mkdir()
    np.save(d / "eigenvalues.npy", w[:3])
    np.save(d / "eigenvectors.npy", V[:, :3].T.copy())      # 3 x 64: rows are the vectors
    (d / "meta.json").write_text(json.dumps({"kind": "subset", "eigenvectors_layout": "rows",
                                             "accuracy_target": 1e-6}))
    b = read_bundle(str(d))
    assert np.asarray(b.claim.eigenvectors).shape == (64, 3)
    assert b.claim.accuracy_target == 1e-6 and b.claim.kind == "subset"


def test_bundle_requires_eigenvalues(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FormatError):
        read_bundle(str(d))
    with pytest.raises(FormatError):
        read_bundle(str(tmp_path / "nonexistent"))


def test_bundle_matrix_free_needs_allow_code(tmp_path):
    d = tmp_path / "mf"
    d.mkdir()
    np.save(d / "eigenvalues.npy", np.array([1.0]))
    (d / "operator.py").write_text("import numpy as np\ndim = 3\ndef matvec(x):\n    return np.arange(1.0, 4.0) * x\n")
    with pytest.raises(FormatError):
        read_bundle(str(d))
    b = read_bundle(str(d), allow_code=True)
    assert b.claim.operator.shape == (3, 3)


def test_claim_from_file_mixed_formats(tmp_path, heis6):
    spec, H, w, V = heis6
    pe = str(tmp_path / "e.csv")
    pv = str(tmp_path / "v.npy")
    po = str(tmp_path / "H.mtx")
    write_array(pe, w[:2])
    write_array(pv, V[:, :2], role="eigenvectors")
    write_operator(po, H)
    claim = claim_from_file(pe, pv, po, kind="lowest")
    c = ce.certify(claim, spec)
    assert c.status_of("S2.eigenpairs") == "CERTIFIED"
    assert c.body["claim"]["producer"]["name"] == "files"


def test_bundle_format_preference(tmp_path):
    """When two files share a stem the binary one wins; the choice is recorded in the producer block."""
    d = tmp_path / "two"
    d.mkdir()
    np.save(d / "eigenvalues.npy", np.array([1.0, 2.0]))
    (d / "eigenvalues.csv").write_text("9.0\n")
    b = read_bundle(str(d))
    assert np.array_equal(b.claim.eigenvalues, [1.0, 2.0])
    assert b.claim.producer["files"]["eigenvalues"] == "eigenvalues.npy"
    assert os.path.basename(b.files["eigenvalues"]) == "eigenvalues.npy"


def test_matrix_free_matvec_is_called_with_vectors_and_checked(tmp_path):
    # a producer function written for 1-D vectors must not be broadcast by (n, 1) columns (scipy 1.17
    # LinearOperator.matmat), and a wrong output length is a FormatError naming the file
    p = tmp_path / "operator.py"
    p.write_text("import numpy as np\ndim = 3\n_d = np.array([1.0, 2.0, 3.0])\n"
                 "def matvec(x):\n    assert x.shape == (3,), x.shape\n    return _d * x\n")
    op = read_operator(str(p), allow_code=True)
    assert np.array_equal(op.matmat(np.eye(3)), np.diag([1.0, 2.0, 3.0]))
    q = tmp_path / "bad.py"
    q.write_text("import numpy as np\ndim = 3\ndef matvec(x):\n    return np.ones(4)\n")
    bad = read_operator(str(q), allow_code=True)
    with pytest.raises(FormatError, match="returned shape"):
        bad.matvec(np.ones(3))
