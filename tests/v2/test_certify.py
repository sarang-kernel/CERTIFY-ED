"""End-to-end statements on realistic claims, including injected failures."""
import json
import os

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import certify_ed as ce
from certify_ed.basis import magnetization_state_set, translation, cyclic_isometry, sector_matrix
from certify_ed.certificate_v2 import Certificate
from conftest import needs_quspin


def status(cert, sid):
    return cert.status_of(sid)


@pytest.fixture(scope="module")
def heis10():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def test_correct_lowest_claim(heis10):
    spec, H, w, V = heis10
    c = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec)
    for sid in ("S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces"):
        assert status(c, sid) == "CERTIFIED", c.summary()
    s3 = [s for s in c.statements if s["id"] == "S3.completeness"][0]["evidence"]
    assert w[3] < s3["next_eigenvalue_lower_bound"] <= w[4]


def test_missing_eigenvalue_is_detected(heis10):
    spec, H, w, V = heis10
    idx = [0, 2, 3, 4]      # skip the first excited level's first member
    c = ce.certify(ce.Claim(w[idx], V[:, idx], operator=H, kind="lowest"), spec)
    assert status(c, "S2.eigenpairs") == "CERTIFIED"
    assert status(c, "S3.completeness") in ("FAILED", "INCONCLUSIVE")


def test_missing_ground_state_fails(heis10):
    spec, H, w, V = heis10
    c = ce.certify(ce.Claim(w[1:4], V[:, 1:4], operator=H, kind="lowest"), spec)
    assert status(c, "S3.completeness") == "FAILED"


def test_truncated_multiplet_is_inconclusive_with_explanation(heis10):
    spec, H, w, V = heis10
    k = int(np.argmax(np.abs(np.diff(w)) < 1e-10)) + 1   # cut inside the first degenerate level
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec)
    s3 = [s for s in c.statements if s["id"] == "S3.completeness"][0]
    assert s3["status"] == "INCONCLUSIVE"
    assert "top_multiplet" in s3["evidence"]


def test_wrong_bond_sign_is_localized(heis10):
    spec, H, w, V = heis10
    static = spec.to_quspin_static()
    static[2][1][3][0] *= -1          # flip the zz coupling on bond 3
    bad = ce.ModelSpec.from_static(spec.n_sites, spec.local, static)
    Hb = ce.build_bitwise(bad)
    wb, Vb = la.eigh(Hb.toarray())
    c = ce.certify(ce.Claim(wb[:3], Vb[:, :3], operator=Hb, kind="lowest"), spec)
    assert status(c, "S1.model_fidelity") == "FAILED"
    assert status(c, "S2.eigenpairs") == "CERTIFIED"     # residuals cannot see the wrong model
    fit = c.statements[0]["evidence"]["diagnosis"]["coupling_fit"]
    top = fit["couplings"][0]
    assert top["op"] == "zz" and top["sites"] == [3, 4]
    assert abs(top["producer_minus_spec"][0] + 2 * spec.terms[2].couplings[3][0].real) < 1e-12


def test_convention_mismatch_is_not_explained_by_couplings():
    spec = ce.heisenberg_chain(8)
    H_asc = ce.build_bitwise(spec.with_state_order("ascending"))
    w, V = la.eigh(H_asc.toarray())
    # producer used ascending order; spec says descending
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H_asc, kind="lowest"),
                   spec.with_state_order("descending"))
    # Heisenberg is invariant under global flip, so the orders give the SAME matrix: must certify
    assert status(c, "S1.model_fidelity") == "CERTIFIED"
    tf = ce.ModelSpec.from_static(6, ce.LocalSpace.spin_space("1/2", "pauli"),
                                  [["zz", [[-1.0, i, i + 1] for i in range(5)]], ["z", [[0.3, 0]]],
                                   ["x", [[-0.7, i] for i in range(6)]]])
    Ha = ce.build_bitwise(tf.with_state_order("ascending"))
    w, V = la.eigh(Ha.toarray())
    c2 = ce.certify(ce.Claim(w[:2], V[:, :2], operator=Ha, kind="lowest"), tf)
    assert status(c2, "S1.model_fidelity") == "FAILED"


def test_quantum_numbers_of_triplet(heis10):
    spec, H, w, V = heis10
    N = spec.n_sites
    states = np.arange(spec.dim)[::-1]
    Sz = np.array([bin(int(s)).count("1") - N / 2 for s in states])
    lev = np.where(np.abs(w - w[1]) < 1e-9)[0]
    k = int(lev[-1]) + 1
    opts = ce.Options(symmetries={"Sz": (Sz, 1.0)})
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, opts)
    s4 = [s for s in c.statements if s["id"] == "S4.eigenspaces"][0]
    assert s4["status"] == "CERTIFIED"
    qn = s4["evidence"]["clusters"][0]["quantum_numbers"]["Sz"]
    assert qn["status"] == "CERTIFIED" and qn["certified_values"] == [-1.0, 0.0, 1.0]


def test_rotated_multiplet_gives_same_projector_statements(heis10):
    spec, H, w, V = heis10
    lev = np.where(np.abs(w - w[1]) < 1e-9)[0]
    k = int(lev[-1]) + 1
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((len(lev), len(lev))))
    V2 = V[:, :k].copy()
    V2[:, lev] = V2[:, lev] @ Q
    O = sp.diags(np.linspace(-1, 1, spec.dim))
    opts = ce.Options(observables={"O": O})
    c1 = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, opts)
    c2 = ce.certify(ce.Claim(w[:k], V2, operator=H, kind="lowest"), spec, opts)
    o1 = [s for s in c1.statements if s["id"] == "S4.eigenspaces"][0]["evidence"]["clusters"][0]["observables"]["O"]
    o2 = [s for s in c2.statements if s["id"] == "S4.eigenspaces"][0]["evidence"]["clusters"][0]["observables"]["O"]
    assert abs(o1["value"] - o2["value"]) <= o1["bound"] + o2["bound"]


def test_window_claim(heis10):
    spec, H, w, V = heis10
    a, b = (w[5] + w[4]) / 2 if w[5] - w[4] > 1e-6 else w[4] - 1e-3, w[20] + 1e-3
    a = float((w[3] + w[4]) / 2) if w[4] - w[3] > 1e-6 else a
    sel = np.where((w > a) & (w < b))[0]
    c = ce.certify(ce.Claim(w[sel], V[:, sel], operator=H, kind="window", window=(a, b)), spec)
    assert status(c, "S3.completeness") == "CERTIFIED", c.summary()
    drop = sel[1:]
    c2 = ce.certify(ce.Claim(w[drop], V[:, drop], operator=H, kind="window", window=(a, b)), spec)
    assert status(c2, "S3.completeness") == "FAILED"


def test_sector_isometry_audit_and_broken_isometry():
    spec = ce.heisenberg_chain(10)
    st = magnetization_state_set(spec, 0)
    Hs0 = ce.build_bitwise(spec, st)
    V0 = cyclic_isometry(st, translation(spec), 2)
    # embed rows into the full space (descending order)
    full_idx, _ = ce.full_state_set(spec).index_of(st.states_in_order())
    P = sp.csr_matrix((np.ones(st.size), (full_idx, np.arange(st.size))), shape=(spec.dim, st.size))
    V = (P @ V0).tocsr()
    Hk = sector_matrix(Hs0, V0)
    w, Y = la.eigh(Hk.toarray())
    c = ce.certify(ce.Claim(w[:3], Y[:, :3], operator=Hk, kind="lowest", sector_isometry=V), spec)
    assert status(c, "S1.model_fidelity") == "CERTIFIED", c.summary()
    assert status(c, "S3.completeness") == "CERTIFIED"
    # break the isometry: drop the phase of one orbit (wrong representative phase)
    Vb = V.tolil()
    col = 3
    rows = Vb[:, col].nonzero()[0]
    Vb[rows[0], col] = -Vb[rows[0], col]
    Vb = Vb.tocsr()
    Hkb = sector_matrix(ce.build_bitwise(spec), Vb)
    wb, Yb = la.eigh(Hkb.toarray())
    c2 = ce.certify(ce.Claim(wb[:3], Yb[:, :3], operator=Hkb, kind="lowest", sector_isometry=Vb), spec)
    assert status(c2, "S1.model_fidelity") == "FAILED"


def test_matrix_free_producer_with_spec():
    spec = ce.heisenberg_chain(12)
    op = ce.SpecOperator(spec).as_linear_operator()
    w, V = spla.eigsh(op, k=3, which="SA", tol=1e-12)
    c = ce.certify(ce.Claim(w, V, operator=op, kind="lowest"), spec)
    assert status(c, "S1.model_fidelity") == "CERTIFIED"
    assert status(c, "S2.eigenpairs") == "CERTIFIED"


def test_certificate_roundtrip_and_tamper(tmp_path, heis10):
    spec, H, w, V = heis10
    c = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec,
                   ce.Options(store_eigenvectors=True))
    p = c.save(str(tmp_path / "cert"))
    c2 = Certificate.load(p)
    assert c2.verify_integrity()["ok"]
    body = json.load(open(p))
    body["claim"]["eigenvalues"][0] += 1e-9
    open(p, "w").write(json.dumps(body))
    with pytest.raises(ValueError):
        Certificate.load(p)
    # sidecar tamper
    c.save(str(tmp_path / "cert2"))
    z = dict(np.load(str(tmp_path / "cert2.arrays.npz")))
    z["eigenvectors"][0, 0] += 1e-12
    np.savez(str(tmp_path / "cert2.arrays.npz"), **z)
    with pytest.raises(ValueError):
        Certificate.load(str(tmp_path / "cert2.json"))


@needs_quspin
def test_quspin_momentum_sector_end_to_end():
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian
    spec = ce.heisenberg_chain(10)
    static = spec.to_quspin_static()
    b = spin_basis_1d(10, Nup=5, kblock=3, pauli=0)
    H = hamiltonian(static, [], basis=b, dtype=np.complex128, check_herm=False, check_symm=False,
                    check_pcon=False)
    w, V = H.eigh()
    claim, sp_ = ce.claim_from_quspin(H, b, static, w[:3], V[:, :3], kind="lowest")
    c = ce.certify(claim, sp_)
    assert status(c, "S1.model_fidelity") == "CERTIFIED", c.summary()
    assert status(c, "S2.eigenpairs") == "CERTIFIED"
    assert status(c, "S3.completeness") == "CERTIFIED"


@needs_quspin
def test_quspin_u1_sector_end_to_end():
    from quspin.basis import spinless_fermion_basis_1d
    from quspin.operators import hamiltonian
    spec = ce.spinless_fermion_chain(10, V=1.2, boundary="periodic")
    static = spec.to_quspin_static()
    b = spinless_fermion_basis_1d(10, Nf=4)
    H = hamiltonian(static, [], basis=b, dtype=np.float64, check_herm=False, check_symm=False,
                    check_pcon=False)
    w, V = la.eigh(H.toarray())
    claim, sp_ = ce.claim_from_quspin(H, b, static, w[:5], V[:, :5], kind="lowest")
    c = ce.certify(claim, sp_)
    assert status(c, "S1.model_fidelity") == "CERTIFIED", c.summary()
    assert status(c, "S2.eigenpairs") == "CERTIFIED"


@needs_quspin
@pytest.mark.parametrize("bug", [None, "drop", "phase", "miss"])
def test_global_spectrum_from_sectors(bug):
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian
    from certify_ed.compose import SectorResult, certify_low_energy
    N = 8
    spec = ce.heisenberg_chain(N)
    static = spec.to_quspin_static()
    Hfull = ce.build_bitwise(spec)
    wfull = np.sort(la.eigvalsh(Hfull.toarray()))
    gaps = np.where(np.diff(wfull[:20]) > 1e-3)[0]
    E_cut = float((wfull[gaps[2]] + wfull[gaps[2] + 1]) / 2)
    secs = []
    for nup in range(N + 1):
        for k in range(N):
            if bug == "drop" and (nup, k) == (4, 1):
                continue
            b = spin_basis_1d(N, Nup=nup, kblock=k, pauli=0)
            if b.Ns == 0:
                continue
            V = sp.csr_matrix(b.get_proj(np.complex128))
            Hs = hamiltonian(static, [], basis=b, dtype=np.complex128, check_herm=False,
                             check_symm=False, check_pcon=False).tocsr()
            if bug == "phase" and (nup, k) == (4, 2):
                V = V.tolil()
                r0 = V[:, 0].nonzero()[0][0]
                V[r0, 0] *= -1
                V = V.tocsr()
                Hs = (V.conj().T @ Hfull @ V).tocsr()
            w, Y = la.eigh(Hs.toarray())
            sel = np.where(w < E_cut)[0]
            if bug == "miss" and (nup, k) == (4, 0) and sel.size:
                sel = sel[1:]
            secs.append(SectorResult(f"{nup},{k}", V, Hs, w[sel], Y[:, sel]))
    res = certify_low_energy(spec, secs, E_cut, H_full=Hfull)
    if bug is None:
        assert res["status"] == "CERTIFIED", res["summary"]
        assert np.max(np.abs(np.array(res["lowest"]) - wfull[:res["K"]])) <= res["radius"]
        assert wfull[res["K"]] >= res["next_lower_bound"]
    else:
        assert res["status"] == "FAILED", (bug, res["summary"])


def test_radius_includes_model_shift():
    """S2 radii refer to the declared model: a producer matrix that differs from the spec
    within rounding tolerance is certified, and the difference is added to the radius."""
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec).tolil()
    d = 4e-16
    H[3, 5] += d
    H[5, 3] += d
    H = H.tocsr()
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest"), spec)
    assert status(c, "S1.model_fidelity") == "CERTIFIED"
    s2 = [s for s in c.statements if s["id"] == "S2.eigenpairs"][0]
    assert s2["evidence"]["model_shift"] >= d
    assert s2["evidence"]["radius_refers_to"] == "declared model (spec)"
    assert np.all(c.arrays["radius"] >= d)
