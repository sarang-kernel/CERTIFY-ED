"""S4 extensions (Proposition 15, UNVERIFIED): batches of observables, two-point functions and structure factors,
checked on small exactly solvable models."""
import math

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.correlations import (certify_observables_batch, combine_certified, site_operator,
                                     structure_factor)
from certify_ed.eigenspace import EigenspaceCertificate


def _s4_cluster(cert, which=0):
    s4 = [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]
    assert s4["status"] == "CERTIFIED", s4["summary"]
    return s4["evidence"]["clusters"][which]


def test_combine_certified_sums_absolute_coefficients_times_bounds():
    val, bound = combine_certified([1.0, -2.0, 0.5j], [1.0, 2.0, 4.0], [1e-3, 1e-3, 2e-3])
    assert val == pytest.approx(1.0 - 4.0 + 2.0j)
    assert bound >= 1e-3 + 2e-3 + 1e-3 and bound < 4.0e-3 * (1 + 1e-9)


def test_site_operator_matches_bitwise_builder_and_restricts_to_states():
    spec = ce.heisenberg_chain(4)
    Sz1 = site_operator(spec, "z", (1,))
    # S^z on site 1 in the site0_msb, descending state order: diagonal of +-1/2
    d = Sz1.diagonal()
    assert set(np.round(d, 12)) == {0.5, -0.5} and Sz1.shape == (16, 16)
    st = ce.magnetization_state_set(spec, 0)
    Sz1_block = site_operator(spec, "z", (1,), states=st)
    assert Sz1_block.shape == (6, 6)
    with pytest.raises(ValueError):
        site_operator(spec, "+", (1,), states=st)         # S+ leaves the S^z = 0 block


def test_heisenberg_ring_n4_correlations_exact():
    """N = 4 Heisenberg ring, singlet ground state: <S^z_i S^z_j> = -1/6 (nn), +1/12 (nnn), 1/4 (i = j)."""
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    opts = ce.Options(eigenspaces="all",
                      correlations={"zz": {"letter": "z", "sites": [0, 1, 2, 3],
                                           "positions": [[0.0], [1.0], [2.0], [3.0]],
                                           "q": [[0.0], [math.pi / 2], [math.pi]]}})
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, opts)
    cl = _s4_cluster(cert)
    corr = cl["correlations"]["zz"]
    assert corr["status"] == "CERTIFIED" and corr["cluster_size"] == 1
    exact = {0: 0.25, 1: -1.0 / 6, 2: 1.0 / 12, 3: -1.0 / 6}
    assert len(corr["pairs"]) == 10
    for p in corr["pairs"]:
        d = (p["j"] - p["i"]) % 4
        assert abs(p["value"] - exact[d]) <= p["bound"], p
        assert p["bound"] < 1e-9
    # S(q) = (1/N) sum_ij cos(q (i - j)) C_ij from the exact values
    for rec in corr["structure_factor"]:
        q = rec["q"][0]
        exact_S = sum(math.cos(q * (i - j)) * exact[(j - i) % 4] for i in range(4) for j in range(4)) / 4
        assert abs(rec["value"] - exact_S) <= rec["bound"], (q, rec, exact_S)
        assert abs(rec["imag_part"]) <= rec["bound"]
        assert rec["bound"] < 1e-8 and rec["n_terms"] == 16
    # S(pi) is the antiferromagnetic peak: largest of the three
    vals = {round(r["q"][0], 6): r["value"] for r in corr["structure_factor"]}
    assert vals[round(math.pi, 6)] > vals[0.0] and vals[round(math.pi, 6)] > vals[round(math.pi / 2, 6)]
    # the structure factor is a linear combination with summed bounds (Prop 15): bound >= (1/N) sum of pair bounds / g
    pair_bound_sum = sum(p["bound"] for p in corr["pairs"]) + sum(p["bound"] for p in corr["pairs"] if p["i"] != p["j"])
    for rec in corr["structure_factor"]:
        assert rec["bound"] >= pair_bound_sum / 4 * (1 - 1e-12)


def test_triplet_cluster_correlations_are_basis_independent():
    """For the N = 4 triplet (g = 3) tr(P Sz_i Sz_j) does not depend on the basis chosen inside the cluster."""
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    lev = [i for i in range(16) if abs(w[i] - w[1]) < 1e-9]
    assert len(lev) == 3
    k = lev[-1] + 1
    opts = ce.Options(correlations={"zz": {"letter": "z", "sites": [0, 1]}})
    c1 = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, opts)
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    V2 = V[:, :k].copy()
    V2[:, lev] = V2[:, lev] @ Q
    c2 = ce.certify(ce.Claim(w[:k], V2, operator=H, kind="lowest"), spec, opts)
    p1 = {(p["i"], p["j"]): p for p in _s4_cluster(c1)["correlations"]["zz"]["pairs"]}
    p2 = {(p["i"], p["j"]): p for p in _s4_cluster(c2)["correlations"]["zz"]["pairs"]}
    for key in p1:
        assert abs(p1[key]["value"] - p2[key]["value"]) <= p1[key]["bound"] + p2[key]["bound"]
        assert p1[key]["per_state_value"] == pytest.approx(p1[key]["value"] / 3)
    # tr(P Sz_0 Sz_0) = g/4 for spin-1/2
    assert abs(p1[(0, 0)]["value"] - 3 / 4) <= p1[(0, 0)]["bound"]


def test_free_fermion_density_correlations():
    """Two-site hopping, one-particle ground state (-t): <n_0 n_1> = 0, <n_0> = <n_1> = 1/2, <n_0 n_0> = 1/2."""
    spec = ce.spinless_fermion_chain(2, t=1.0, boundary="open")
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    assert w[0] == pytest.approx(-1.0)
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec,
                      ce.Options(eigenspaces="all", correlations={"nn": {"letter": "n", "sites": [0, 1]}},
                                 observables={"n0": site_operator(spec, "n", (0,))}))
    cl = _s4_cluster(cert)
    pairs = {(p["i"], p["j"]): p for p in cl["correlations"]["nn"]["pairs"]}
    assert abs(pairs[(0, 1)]["value"]) <= pairs[(0, 1)]["bound"] and pairs[(0, 1)]["bound"] < 1e-12
    assert abs(pairs[(0, 0)]["value"] - 0.5) <= pairs[(0, 0)]["bound"]
    assert abs(cl["observables"]["n0"]["value"] - 0.5) <= cl["observables"]["n0"]["bound"]


def test_batch_observables_and_non_hermitian_rejected():
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    ec = EigenspaceCertificate([0], [float(w[0])], 1.0, 1e-15, 1e-16, 1e-15, 2e-15)
    ops = {f"z{i}": site_operator(spec, "z", (i,)) for i in range(4)}
    out = certify_observables_batch(ec, V, ops)
    assert set(out) == set(ops) and all(abs(r["value"]) <= r["bound"] + 1e-12 for r in out.values())   # singlet: <Sz_i> = 0
    with pytest.raises(ValueError):
        certify_observables_batch(ec, V, {"bad": sp.csr_matrix(site_operator(spec, "+", (0,)))})


def test_correlation_request_without_spec_is_reported_not_raised():
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), None,
                      ce.Options(eigenspaces="all", correlations={"zz": {"letter": "z", "sites": [0, 1]}}))
    cl = _s4_cluster(cert)
    assert cl["correlations"]["zz"]["status"] == "NOT_CHECKED"
    assert cl["correlations"]["zz"]["reason_code"] == "NEEDS_SPEC"


def test_structure_factor_helper_direct():
    corr = {"sites": [0, 1], "cluster_size": 1,
            "pairs": [{"i": 0, "j": 0, "value": 0.25, "bound": 1e-6}, {"i": 1, "j": 1, "value": 0.25, "bound": 1e-6},
                      {"i": 0, "j": 1, "value": -0.25, "bound": 1e-6}]}
    sf = structure_factor(corr, [[0.0], [1.0]], [[0.0], [math.pi]])
    assert sf[0]["value"] == pytest.approx(0.0) and sf[1]["value"] == pytest.approx(0.5)
    assert sf[0]["bound"] >= 2e-6 and sf[0]["bound"] < 2.1e-6
    assert all(r["n_terms"] == 4 for r in sf)
