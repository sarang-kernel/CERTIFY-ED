"""C-12, C-13, E-07 (3.2): Davis-Kahan sin(Theta) alternative for S4, approximate-symmetry labels, fidelities
(certify_ed.davis_kahan; docs/proofs/E-07_davis_kahan.md)."""
import math

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed import davis_kahan as dk
from certify_ed.correlations import site_operator
from certify_ed.eigenspace import EigenspaceCertificate


@pytest.fixture(scope="module")
def ring8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _s4(cert):
    s4 = [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]
    assert s4["status"] == "CERTIFIED", s4["summary"]
    return s4


# ------------------------------------------------------------------------------------------------ E-07
def test_davis_kahan_bound_is_recorded_and_not_looser_than_theorem6(ring8):
    spec, H, w, V = ring8
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, ce.Options(eigenspaces="all"))
    for cl in _s4(cert)["evidence"]["clusters"]:
        b = cl["sin_theta_bounds"]
        assert b["theorem6"] is not None and b["davis_kahan"] is not None
        assert cl["sin_theta_bound"] == min(b["theorem6"], b["davis_kahan"])
        assert b["used"] == "davis_kahan" and b["davis_kahan"] < b["theorem6"]
        assert b["davis_kahan_record"]["gap"] > 0 and b["davis_kahan_record"]["w_prime"] < 1e-12
        assert cl["projector_error_bound"] >= cl["sin_theta_bound"] + cl["eta"] * (1 - 1e-12)


def _planted(rng, n, g, gap, eps, eta_scale, cplx):
    """A with a planted cluster (eigenvalues near 0, the rest beyond +-gap); X = (Q_c + eps W) M."""
    Qf, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
    lam = np.concatenate([rng.uniform(-0.1, 0.1, g), rng.choice([-1, 1], n - g) * (gap + rng.uniform(0, 1, n - g))])
    A = (Qf * lam) @ Qf.conj().T
    A = (A + A.conj().T) / 2
    Q = Qf[:, :g]
    W = Qf[:, g:] @ (rng.standard_normal((n - g, g)) + (1j * rng.standard_normal((n - g, g)) if cplx else 0))
    W /= np.linalg.norm(W, 2)
    M = np.eye(g) + eta_scale * rng.standard_normal((g, g))
    X = (Q + eps * W) @ M
    P = Q @ Q.conj().T
    return A, X, P, lam


def _ingredients(A, X, dist_true):
    mu = np.real(np.einsum("ij,ij->j", X.conj(), A @ X) / np.einsum("ij,ij->j", X.conj(), X))
    R = A @ X - X * mu
    G = X.conj().T @ X
    eta = np.linalg.norm(G - np.eye(X.shape[1]), 2) * (1 + 1e-12)
    delta = (1 - math.sqrt(1 - eta)) * (1 + 1e-12) + 1e-300
    return dict(R_fro=np.linalg.norm(R, "fro") * (1 + 1e-12), XR=np.linalg.norm(X.conj().T @ R, 2) * (1 + 1e-12), eta=eta,
                delta=delta, spread=(mu.max() - mu.min()) * (1 + 1e-12) + 1e-300, dist=dist_true, mu=mu)


def test_davis_kahan_function_bounds_the_exact_sin_theta_on_planted_matrices():
    rng = np.random.default_rng(0)
    n_ok = 0
    for trial in range(40):
        n, g = int(rng.integers(6, 16)), int(rng.integers(1, 4))
        cplx = bool(rng.integers(0, 2))
        gap = rng.uniform(0.5, 3.0)
        A, X, P, lam = _planted(rng, n, g, gap, 10.0 ** rng.uniform(-6, -1.5), 10.0 ** rng.uniform(-6, -1.5), cplx)
        ing = _ingredients(A, X, None)
        mu = ing.pop("mu")
        others = lam[g:]
        dist = float(np.min(np.abs(others - np.clip(others, mu.min(), mu.max())))) * (1 - 1e-12)
        ing["dist"] = dist
        rec = dk.sin_theta_davis_kahan(**ing)
        Qh, _ = la.polar(X)
        sin_exact = np.linalg.norm(Qh @ Qh.conj().T - P, 2)
        assert rec["bound"] is not None and sin_exact <= rec["bound"], (trial, sin_exact, rec)
        n_ok += 1
    assert n_ok == 40


def test_negative_control_planted_gap_violation_breaks_the_bound():
    """If `dist` overstates the true separation (an eigenvalue sits inside the claimed gap), the Davis-Kahan quotient can
    undercut the exact sin(theta): the hypothesis must come from S3, as the S4 hook does."""
    rng = np.random.default_rng(1)
    broken = 0
    for _ in range(20):
        n, g = 10, 1
        A, X, P, lam = _planted(rng, n, g, 1.0, 0.3, 0.0, False)      # a large rotation towards the other eigenvectors
        ing = _ingredients(A, X, None)
        ing.pop("mu")
        ing["dist"] = 50.0                                             # false hypothesis: no eigenvalue within 50
        rec = dk.sin_theta_davis_kahan(**ing)
        Qh, _ = la.polar(X)
        sin_exact = np.linalg.norm(Qh @ Qh.conj().T - P, 2)
        broken += int(rec["bound"] is not None and rec["bound"] < sin_exact)
    assert broken > 0
    # eta >= 1 or no separation: no bound, with a reason
    assert dk.sin_theta_davis_kahan(1e-3, 1e-3, 1.0, 1.0, 0.0, 1.0)["bound"] is None
    assert dk.sin_theta_davis_kahan(1e-3, 1.0, 0.1, 0.05, 0.0, 0.5)["bound"] is None       # w' > dist


# ------------------------------------------------------------------------------------------------ C-12
def test_approximate_symmetry_labels_exact_and_perturbed(ring8):
    spec, H, w, V = ring8
    Sz = sum(site_operator(spec, "z", (i,)) for i in range(8)).tocsr()
    Sx0 = site_operator(spec, "x", (0,))
    S_pert = (Sz + 1e-3 * Sx0).tocsr()
    opts = ce.Options(eigenspaces="all", approximate_symmetries={"Sz": (Sz.diagonal(), 1.0), "Sz_pert": (S_pert, 1.0),
                                                                 "Sz_vals": (Sz, None, [-4, -3, -2, -1, 0, 1, 2, 3, 4])})
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, opts)
    comm_true = np.linalg.norm((H @ S_pert - S_pert @ H).toarray(), 2)
    for cl in _s4(cert)["evidence"]["clusters"]:
        g = len(cl["members"])
        expect = [0.0] if g == 1 else [-1.0, 0.0, 1.0]
        for name in ("Sz", "Sz_pert", "Sz_vals"):
            q = cl["quantum_numbers"][name]
            assert q["approximate_symmetry"] and q["status"] == "CERTIFIED", (name, q)
            assert q["certified_values"] == expect
            assert q["hypothesis"]["statement"] == "||[H, S]||_2 <= c"
        assert cl["quantum_numbers"]["Sz"]["hypothesis"]["commutator_bound"] < 1e-12          # exact symmetry: rounding only
        c = cl["quantum_numbers"]["Sz_pert"]["hypothesis"]["commutator_bound"]
        assert comm_true <= c <= comm_true * 1.5 + 1e-10
        assert cl["quantum_numbers"]["Sz_pert"]["bound"] > cl["quantum_numbers"]["Sz"]["bound"]


def test_approximate_symmetry_bound_contains_an_eigenvalue_of_S_and_fails_safely_when_loose(ring8):
    spec, H, w, V = ring8
    Sz = sum(site_operator(spec, "z", (i,)) for i in range(8)).tocsr()
    Sx0 = site_operator(spec, "x", (0,))
    for amp, expect_status in ((0.05, "CERTIFIED"), (0.6, "INCONCLUSIVE")):
        S = (Sz + amp * Sx0).tocsr()
        cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec,
                          ce.Options(eigenspaces="all", approximate_symmetries={"S": (S, 1.0)}))
        q = _s4(cert)["evidence"]["clusters"][0]["quantum_numbers"]["S"]
        assert q["status"] == expect_status, q
        eig_S = la.eigvalsh(S.toarray())
        for m in q["estimates"]:
            assert np.min(np.abs(eig_S - m)) <= q["bound"]          # an eigenvalue of S within D of each label
        if expect_status == "INCONCLUSIVE":
            assert "certified_values" not in q and q["bound"] >= 0.5
    # without a stored reference matrix the label is not checked
    ec = EigenspaceCertificate([0], [0.0], 1.0, 0.0, 0.0, 1e-12, 1e-12)
    dk.labels_hook(ec, V[:, :1], {"S": (Sz, 1.0)}, None, (0.0, 0.0))
    assert ec.quantum_numbers["S"]["status"] == "NOT-CHECKED" and ec.quantum_numbers["S"]["reason_code"] == "NEEDS_MATRIX"


# ------------------------------------------------------------------------------------------------ C-13
def test_fidelity_between_ground_states_at_two_couplings(ring8):
    spec, H, w, V = ring8
    spec2 = ce.heisenberg_chain(8, J2=0.1)
    H2 = ce.build_bitwise(spec2)
    w2, V2 = la.eigh(H2.toarray())
    o = ce.Options(eigenspaces="all", store_eigenvectors=True)
    ca = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, o)
    cb = ce.certify(ce.Claim(w2[:4], V2[:, :4], operator=H2, kind="lowest"), spec2, o)
    f = dk.fidelity_from_certificates(ca, cb, [0], [0])
    exact = abs(V[:, 0] @ V2[:, 0]) ** 2
    assert f["interval"][0] <= exact <= f["interval"][1] and f["bound"] < 1e-11 and f["cluster_size"] == 1
    assert "fidelity" in f["statement"] and f["a"]["hash"] != f["b"]["hash"]
    # the triplet subspaces (g = 3): tr(P_1 P_2) exact from the eigenvectors, basis independent
    f3 = dk.fidelity_from_certificates(ca, cb, [1, 2, 3], [1, 2, 3])
    P1, P2 = V[:, 1:4] @ V[:, 1:4].T, V2[:, 1:4] @ V2[:, 1:4].T
    exact3 = float(np.trace(P1 @ P2))
    assert f3["interval"][0] <= exact3 <= f3["interval"][1] and f3["cluster_size"] == 3 and f3["per_state_interval"][1] <= 1.0
    with pytest.raises(ValueError, match="same dimension"):
        dk.certified_fidelity(V[:, :1], 0.0, 0.0, V2[:, :3], 0.0, 0.0)
    cc = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, ce.Options(eigenspaces="all"))
    with pytest.raises(ValueError, match="stored eigenvectors"):
        dk.fidelity_from_certificates(cc, cb, [0], [0])


def test_fidelity_interval_contains_the_exact_value_for_planted_rotations():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n, g = int(rng.integers(5, 12)), int(rng.integers(1, 3))
        Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        P1, P2 = Q1[:, :g] @ Q1[:, :g].T, Q2[:, :g] @ Q2[:, :g].T
        exact = float(np.trace(P1 @ P2))
        # claimed vectors rotated by known angles into the complements, scaled (eta > 0)
        a1, a2 = rng.uniform(0, 0.3), rng.uniform(0, 0.3)
        X1 = (math.cos(a1) * Q1[:, :g] + math.sin(a1) * Q1[:, g:2 * g]) * rng.uniform(0.9, 1.1)
        X2 = (math.cos(a2) * Q2[:, :g] + math.sin(a2) * Q2[:, g:2 * g]) * rng.uniform(0.9, 1.1)
        eta1 = np.linalg.norm(X1.T @ X1 - np.eye(g), 2) * (1 + 1e-12)
        eta2 = np.linalg.norm(X2.T @ X2 - np.eye(g), 2) * (1 + 1e-12)
        rec = dk.certified_fidelity(X1, math.sin(a1) * (1 + 1e-12), eta1, X2, math.sin(a2) * (1 + 1e-12), eta2)
        assert rec["interval"][0] <= exact <= rec["interval"][1]
