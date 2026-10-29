"""F-02: readers for the output formats of HPhi, DiagHam, SPINPACK and ALPS.

Each reader is tested on a synthetic fixture written by tests/fixtures/ed_codes.py in the documented (or, for
SPINPACK, the assumed) layout, round-tripped through certify() on a small Heisenberg chain, and shown to reject
files whose layout check fails. Negative controls: a planted wrong eigenvalue written into the producer's file is
not certified; a swapped vector is not certified."""
import os

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.io import (FormatError, claim_from_alps, claim_from_diagham, claim_from_hphi, claim_from_spinpack,
                           read_alps_spectrum, read_diagham_spectrum, read_diagham_vector, read_hphi_eigenvalue_file,
                           read_hphi_eigenvec, read_hphi_energy, read_spinpack_def, read_spinpack_energies)
from fixtures import ed_codes as fx

N = 6


def _status(cert, sid):
    return {s["id"].split(".")[0]: s["status"] for s in cert.statements}[sid]


@pytest.fixture(scope="module")
def full_chain():
    """Open Heisenberg chain, full space (dimension 64), all eigenpairs by LAPACK."""
    spec = ce.heisenberg_chain(N, boundary="open")
    H = ce.build_bitwise(spec)
    w, V = np.linalg.eigh(H.toarray())
    return spec, H.tocsr(), w, V


@pytest.fixture(scope="module")
def sz0_block():
    """Sz = 0 block (dimension 20) of the periodic chain, with the state list and the block matrix."""
    spec = ce.heisenberg_chain(N)
    states = ce.u1_state_set(spec, N // 2)
    H = ce.build_bitwise(spec, states).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    return spec, states, H, w, V


# ============================================================================================ HPhi
@pytest.mark.parametrize("form", ["block", "single", "oneline"])
def test_hphi_energy_forms(tmp_path, form):
    e = [-2.5, -1.25, 0.75][: (1 if form == "single" else 3)]
    p = fx.write_hphi_energy(str(tmp_path / "zvo_energy.dat"), e, sz=[0.0] * len(e), form=form)
    r = read_hphi_energy(p)
    np.testing.assert_allclose(r["Energy"], e, rtol=0, atol=1e-6 if form == "oneline" else 0)
    assert r["state"].tolist() == list(range(len(e)))


def test_hphi_eigenvalue_file(tmp_path, full_chain):
    spec, H, w, V = full_chain
    p = fx.write_hphi_eigenvalue_file(str(tmp_path / "Eigenvalue.dat"), w)
    np.testing.assert_array_equal(read_hphi_eigenvalue_file(p), w)
    claim = claim_from_hphi(p, kind="all", operator=H)
    assert claim.kind == "all" and claim.eigenvectors is None
    with open(p, "a") as f:
        f.write("7 1.0\n")                                      # eigennumber out of sequence
    with pytest.raises(FormatError, match="eigennumbers"):
        read_hphi_eigenvalue_file(p)


@pytest.mark.parametrize("size_t_bytes", [8, 4])
def test_hphi_eigenvec_binary_layout(tmp_path, size_t_bytes):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(37) + 1j * rng.standard_normal(37)
    p = fx.write_hphi_eigenvec(str(tmp_path / "zvo_eigenvec_0_rank_0.dat"), v, iterations=57,
                               size_t_bytes=size_t_bytes, unused_first=123.0)
    r = read_hphi_eigenvec(p)
    assert r["iterations"] == 57 and r["local_size"] == 37 and r["size_t_bytes"] == size_t_bytes
    np.testing.assert_array_equal(r["vector"], v)                 # eigen_vector[0] dropped
    # layout checks: a truncated file and a header that disagrees with the size
    data = open(p, "rb").read()
    open(str(tmp_path / "trunc.dat"), "wb").write(data[:-8])
    with pytest.raises(FormatError, match="matches neither"):
        read_hphi_eigenvec(str(tmp_path / "trunc.dat"))
    with pytest.raises(FormatError, match="differs from the expected"):
        read_hphi_eigenvec(p, expected_size=36)


@pytest.mark.parametrize("ranks", [1, 3])
def test_hphi_round_trip_certifies(tmp_path, full_chain, ranks):
    spec, H, w, V = full_chain
    k = next(i for i in range(2, 12) if w[i] - w[i - 1] > 1e-3)      # cut at a gap (no split multiplet)
    run = fx.write_hphi_run(str(tmp_path / "output"), w[:k], V[:, :k], ranks=ranks)
    claim = claim_from_hphi(run["energy"], run["pattern"], operator=H, kind="lowest")
    assert claim.producer["name"] == "HPhi"
    assert claim.producer["eigenvectors"]["dimension"] == H.shape[0]
    assert len(claim.producer["eigenvectors"]["files"][0]) == ranks
    np.testing.assert_array_equal(np.asarray(claim.eigenvectors), V[:, :k])     # real parts kept, order kept
    cert = ce.certify(claim, spec)
    assert _status(cert, "S1") == "CERTIFIED", cert.summary()
    assert _status(cert, "S2") == "CERTIFIED"
    assert _status(cert, "S3") == "CERTIFIED"
    direct = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec)
    assert [(s["id"], s["status"]) for s in cert.statements] == [(s["id"], s["status"]) for s in direct.statements]


def test_hphi_negative_control_planted_energy(tmp_path, full_chain):
    spec, H, w, V = full_chain
    bad = w[:3].copy()
    bad[2] += 2e-3
    run = fx.write_hphi_run(str(tmp_path / "output"), bad, V[:, :3])
    cert = ce.certify(claim_from_hphi(run["energy"], run["pattern"], operator=H), spec)
    assert _status(cert, "S2") != "CERTIFIED"


def test_hphi_negative_control_swapped_vector(tmp_path, full_chain):
    spec, H, w, V = full_chain
    X = V[:, :3].copy()
    X[:, [0, 1]] = X[:, [1, 0]]
    run = fx.write_hphi_run(str(tmp_path / "output"), w[:3], X)
    cert = ce.certify(claim_from_hphi(run["energy"], run["pattern"], operator=H), spec)
    assert _status(cert, "S2") != "CERTIFIED"


def test_hphi_state_count_mismatch_and_n_states(tmp_path, full_chain):
    spec, H, w, V = full_chain
    run = fx.write_hphi_run(str(tmp_path / "output"), w[:4], V[:, :2])
    with pytest.raises(FormatError, match="eigenvector states for"):
        claim_from_hphi(run["energy"], run["pattern"], operator=H)
    claim = claim_from_hphi(run["energy"], run["pattern"], operator=H, n_states=2)
    assert claim.eigenvalues.size == 2 if hasattr(claim.eigenvalues, "size") else len(claim.eigenvalues) == 2
    open(str(tmp_path / "junk.dat"), "w").write("hello world\n")
    with pytest.raises(FormatError, match="not an HPhi energy.dat line"):
        read_hphi_energy(str(tmp_path / "junk.dat"))


# ========================================================================================= DiagHam
def test_diagham_vector_layouts(tmp_path):
    rng = np.random.default_rng(1)
    v = rng.standard_normal(50)
    p = fx.write_diagham_vector(str(tmp_path / "r.vec"), v)
    assert os.path.getsize(p) == 4 + 8 * 50
    np.testing.assert_array_equal(read_diagham_vector(p), v)
    c = v + 1j * rng.standard_normal(50)
    pc = fx.write_diagham_vector(str(tmp_path / "c.vec"), c)
    assert os.path.getsize(pc) == 4 + 16 * 50
    np.testing.assert_array_equal(read_diagham_vector(pc), c)
    p8 = fx.write_diagham_vector(str(tmp_path / "r8.vec"), v, header_bytes=8)
    np.testing.assert_array_equal(read_diagham_vector(p8, header_bytes=8), v)
    # size checks
    with pytest.raises(FormatError, match="neither"):
        read_diagham_vector(p8)                                   # 8-byte header read as 4-byte: size mismatch
    with pytest.raises(FormatError, match="expected"):
        read_diagham_vector(p, complex=True)
    data = open(pc, "rb").read()
    open(str(tmp_path / "t.vec"), "wb").write(data[:-4])
    with pytest.raises(FormatError, match="neither"):
        read_diagham_vector(str(tmp_path / "t.vec"))


def test_diagham_spectrum_and_round_trip(tmp_path, sz0_block):
    spec, states, H, w, V = sz0_block
    # a spectrum file with two Sz sectors; the Sz=0 lines are the block's spectrum
    other = np.sort(np.linalg.eigvalsh(ce.build_bitwise(spec, ce.u1_state_set(spec, N // 2 + 1)).toarray()))
    table = np.vstack([np.column_stack([np.zeros(w.size), w]), np.column_stack([np.ones(other.size), other])])
    p = fx.write_diagham_spectrum(str(tmp_path / "chain.dat"), table, columns=["Sz", "E"])
    r = read_diagham_spectrum(p)
    assert r["columns"] == ["Sz", "E"] and r["table"].shape == table.shape
    k = 3
    vecs = [fx.write_diagham_vector(str(tmp_path / f"chain_sz_0.{i}.vec"), V[:, i]) for i in range(k)]
    claim = claim_from_diagham(p, vecs, select={"Sz": 0}, n_states=k, operator=H, states=states, kind="lowest")
    assert claim.producer["name"] == "DiagHam" and claim.producer["select"] == {"Sz": 0}
    np.testing.assert_array_equal(claim.eigenvalues, w[:k])
    cert = ce.certify(claim, spec)
    assert _status(cert, "S1") == "CERTIFIED", cert.summary()
    assert _status(cert, "S2") == "CERTIFIED"
    assert _status(cert, "S3") == "CERTIFIED"
    # selection by index and errors
    claim2 = claim_from_diagham(p, select={0: 1}, kind="subset")
    np.testing.assert_array_equal(claim2.eigenvalues, other)
    with pytest.raises(FormatError, match="no column named"):
        claim_from_diagham(p, select={"Lz": 0})
    with pytest.raises(FormatError, match="no lines match"):
        claim_from_diagham(p, select={"Sz": 7})
    with pytest.raises(FormatError, match="vector files for"):
        claim_from_diagham(p, vecs[:2], select={"Sz": 0}, n_states=3)


def test_diagham_negative_controls(tmp_path, sz0_block):
    spec, states, H, w, V = sz0_block
    bad = w.copy()
    bad[0] -= 1e-3
    p = fx.write_diagham_spectrum(str(tmp_path / "bad.dat"), np.column_stack([np.zeros(w.size), bad]), ["Sz", "E"])
    vecs = [fx.write_diagham_vector(str(tmp_path / f"v{i}.vec"), V[:, i]) for i in range(2)]
    cert = ce.certify(claim_from_diagham(p, vecs, select={"Sz": 0}, operator=H, states=states, n_states=2), spec)
    assert _status(cert, "S2") != "CERTIFIED"
    # ragged table
    with open(p, "a") as f:
        f.write("0 1 2 3\n")
    with pytest.raises(FormatError, match="different numbers of columns"):
        read_diagham_spectrum(p)


# ======================================================================================== SPINPACK
def test_spinpack_assumed_layouts(tmp_path, sz0_block):
    spec, states, H, w, V = sz0_block
    labels = np.column_stack([np.zeros(w.size), np.arange(w.size) % 2])
    p = fx.write_spinpack_table(str(tmp_path / "ev.dat"), w, labels)
    r = read_spinpack_energies(p)
    np.testing.assert_array_equal(r["energies"], w)
    assert r["labels"].shape == (w.size, 2) and r["layout"] == "table"
    pk = fx.write_spinpack_keyvalue(str(tmp_path / "out.log"), w[:4])
    np.testing.assert_array_equal(read_spinpack_energies(pk, layout="keyvalue")["energies"], w[:4])
    pd = fx.write_spinpack_def(str(tmp_path / "daten.def"), N, couplings=["1.0"])
    assert read_spinpack_def(pd)["N"] == str(N)
    claim = claim_from_spinpack(p, select={1: 0}, definition=pd, operator=H, states=states, kind="subset")
    np.testing.assert_array_equal(claim.eigenvalues, w[0::2])
    assert claim.producer["layout_status"].startswith("assumed")
    assert claim.producer["definition"]["N"] == str(N)
    # eigenvalue-only claim: S2 cannot run; a spec-level round trip still gives S1 against the operator
    cert = ce.certify(claim, spec)
    assert _status(cert, "S1") == "CERTIFIED", cert.summary()
    assert _status(cert, "S2") in ("NOT_CHECKED", "INCONCLUSIVE")
    open(str(tmp_path / "x.dat"), "w").write("a b c\n")
    with pytest.raises(FormatError, match="non-numeric"):
        read_spinpack_energies(str(tmp_path / "x.dat"))
    with pytest.raises(FormatError, match="no 'E=<value>'"):
        read_spinpack_energies(p, layout="keyvalue")


def test_spinpack_negative_control_vectors_from_elsewhere(tmp_path, sz0_block):
    """The energies come from the file; the vectors from a bundle route. A planted energy must not certify."""
    spec, states, H, w, V = sz0_block
    bad = w[:2].copy()
    bad[1] += 5e-3
    p = fx.write_spinpack_table(str(tmp_path / "ev.dat"), bad)
    claim = claim_from_spinpack(p, operator=H, states=states, kind="lowest")
    claim.eigenvectors = V[:, :2]
    cert = ce.certify(claim, spec)
    assert _status(cert, "S2") != "CERTIFIED"


# ============================================================================================ ALPS
def test_alps_round_trip_with_basis_table(tmp_path, sz0_block):
    pytest.importorskip("h5py")
    spec, states, H, w, V = sz0_block
    # ALPS lists the basis in its own order: permute the block randomly and write local-state indices
    rng = np.random.default_rng(3)
    perm = rng.permutation(states.size)
    ints = states.states_in_order()[perm]
    digits = np.array([[(s >> (N - 1 - i)) & 1 for i in range(N)] for s in ints])    # site 0 most significant
    k = next(i for i in range(2, 12) if w[i] - w[i - 1] > 1e-3)      # cut at a gap
    secs = [{"energies": w[:k], "quantumnumbers": {"Sz": 0.0}, "eigenvectors": V[perm, :k], "basis": digits},
            {"energies": np.array([1.0, 2.0]), "quantumnumbers": {"Sz": 1.0}}]
    p = fx.write_alps_h5(str(tmp_path / "parm.task1.out.h5"), secs)
    r = read_alps_spectrum(p)
    assert len(r) == 2 and r[1]["quantumnumbers"] == {"Sz": 1.0} and r[1]["eigenvectors"] is None
    claim = claim_from_alps(p, sector=0, spec=spec, kind="lowest")
    assert claim.states is not None and claim.states.size == states.size
    np.testing.assert_array_equal(claim.states.states_asc, states.states_asc)
    np.testing.assert_array_equal(np.asarray(claim.eigenvectors), V[:, :k])       # reordered into the spec encoding
    cert = ce.certify(claim, spec)                                                   # operator: the spec's own block
    assert _status(cert, "S2") == "CERTIFIED", cert.summary()
    assert _status(cert, "S3") == "CERTIFIED"
    # with the producer's matrix in ALPS order, S1 compares it after the same reordering
    Halps = H[perm][:, perm]
    cert2 = ce.certify(claim_from_alps(p, sector=0, spec=spec, operator=Halps), spec)
    assert _status(cert2, "S1") == "CERTIFIED", cert2.summary()
    assert _status(cert2, "S2") == "CERTIFIED"


def test_alps_without_basis_and_sector_selection(tmp_path, sz0_block):
    pytest.importorskip("h5py")
    spec, states, H, w, V = sz0_block
    secs = [{"energies": np.array([5.0]), "quantumnumbers": {"Sz": 2.0}},
            {"energies": w[:2], "quantumnumbers": {"Sz": 0.0}, "eigenvectors": V[:, :2]}]
    p = fx.write_alps_h5(str(tmp_path / "a.out.h5"), secs)
    claim = claim_from_alps(p, sector=1, operator=H, states=states)
    assert claim.producer["quantumnumbers"] == {"Sz": 0.0} and claim.producer["n_sectors"] == 2
    cert = ce.certify(claim, spec)
    assert _status(cert, "S1") == "CERTIFIED" and _status(cert, "S2") == "CERTIFIED", cert.summary()
    with pytest.raises(FormatError, match="out of range"):
        claim_from_alps(p, sector=2)


def test_alps_complex_vectors_and_layout_checks(tmp_path, sz0_block):
    h5py = pytest.importorskip("h5py")
    spec, states, H, w, V = sz0_block
    phase = np.exp(1j * 0.3)
    p = fx.write_alps_h5(str(tmp_path / "c.out.h5"), [{"energies": w[:2], "quantumnumbers": {"Sz": 0.0},
                                                       "eigenvectors": V[:, :2] * phase}])
    claim = claim_from_alps(p, operator=H, states=states)
    assert np.iscomplexobj(claim.eigenvectors)
    assert _status(ce.certify(claim, spec), "S2") == "CERTIFIED"
    with h5py.File(p, "a") as f:
        f["spectrum/sectors/0/eigenvectors/shape"][...] = np.array([3, V.shape[0]])
    with pytest.raises(FormatError, match="entries, shape"):
        read_alps_spectrum(p)
    with h5py.File(str(tmp_path / "empty.h5"), "w") as f:
        f.create_group("results")
    with pytest.raises(FormatError, match="/spectrum/sectors"):
        read_alps_spectrum(str(tmp_path / "empty.h5"))


def test_alps_negative_control_planted_energy(tmp_path, sz0_block):
    pytest.importorskip("h5py")
    spec, states, H, w, V = sz0_block
    bad = w[:3].copy()
    bad[1] -= 1e-3
    p = fx.write_alps_h5(str(tmp_path / "bad.out.h5"), [{"energies": bad, "quantumnumbers": {"Sz": 0.0},
                                                         "eigenvectors": V[:, :3]}])
    cert = ce.certify(claim_from_alps(p, operator=H, states=states), spec)
    assert _status(cert, "S2") != "CERTIFIED"


def test_alps_basis_table_checks(sz0_block):
    from certify_ed.io import states_from_alps_basis
    spec, states, H, w, V = sz0_block
    with pytest.raises(FormatError, match="outside"):
        states_from_alps_basis(np.array([[0, 2]]), q=2)
    with pytest.raises(FormatError, match="repeated"):
        states_from_alps_basis(np.array([[0, 1], [0, 1]]), q=2)
    st, perm = states_from_alps_basis(np.array([[0, 1], [1, 0]]), q=2, state_order="descending")
    assert st.states_in_order().tolist() == [2, 1] and perm.tolist() == [1, 0]


def test_importer_names_exported():
    for name in ("claim_from_hphi", "claim_from_diagham", "claim_from_spinpack", "claim_from_alps"):
        assert callable(getattr(ce, name))
        assert callable(getattr(ce.importers, name))
    assert sp.issparse(ce.build_bitwise(ce.heisenberg_chain(4)))
