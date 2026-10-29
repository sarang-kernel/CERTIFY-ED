"""C-05: non-Hermitian observables and correlation products (S+_i S-_j) through the Hermitian decomposition
O = H + iK, each part a Theorem 6 observable, recombined with Proposition 15. Checked against exact values of
the N = 4 Heisenberg ring (singlet ground state: <S^a_i S^a_j> = -1/6 nn, +1/12 nnn, so <S+_i S-_j> = -1/3, +1/6,
and S+_i S-_i = 1/2 + S^z_i has expectation 1/2)."""
import math

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.correlations import (certify_observable_general, certify_observables_batch, hermitian_parts,
                                     site_operator)


@pytest.fixture(scope="module")
def ring4():
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _cluster(cert):
    s4 = cert.statement("S4.eigenspaces")
    assert s4["status"] == "CERTIFIED", s4["summary"]
    return s4["evidence"]["clusters"][0]


EXACT_PM = {0: 0.5, 1: -1.0 / 3, 2: 1.0 / 6, 3: -1.0 / 3}


def test_hermitian_parts_are_hermitian_and_recombine(ring4):
    spec, H, w, V = ring4
    O = site_operator(spec, "+-", (0, 1))
    assert abs(O - O.conj().T).max() > 0
    Hp, Kp, exact = hermitian_parts(O)
    assert exact
    assert abs(Hp - Hp.conj().T).max() == 0 and abs(Kp - Kp.conj().T).max() == 0
    assert abs(Hp + 1j * Kp - O).max() == 0
    # a Hermitian input has K = 0
    Hz, Kz, ex2 = hermitian_parts(site_operator(spec, "z", (0,)))
    assert ex2 and Kz.nnz == 0 and abs(Hz - site_operator(spec, "z", (0,))).max() == 0
    # rounded parts are detected
    rng = np.random.default_rng(1)
    R = sp.csr_matrix(rng.uniform(0.1, 0.9, (16, 16)))
    _, _, ex3 = hermitian_parts(R)
    assert ex3 is False


def test_non_hermitian_observable_in_options(ring4):
    spec, H, w, V = ring4
    O = site_operator(spec, "+-", (0, 1))
    Oc = O * (1.0 + 1.0j)                 # complex coefficient: both parts have nonzero expectation
    opts = ce.Options(eigenspaces="all", observables={"SpSm": O, "SpSm_c": Oc, "Sz0": site_operator(spec, "z", (0,))})
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, opts)
    obs = _cluster(cert)["observables"]
    rec = obs["SpSm"]
    assert rec["hermitian_decomposition"] and rec["parts_formed_exactly"] and rec["hermitian"] is False
    assert set(rec["parts"]) == {"hermitian", "anti_hermitian"} and "recombine" in rec and "definition" in rec
    re, im = rec["value"]
    assert abs(re - EXACT_PM[1]) <= rec["bound"] and abs(im) <= rec["bound"] and rec["bound"] < 1e-9
    assert rec["bound"] >= rec["parts"]["hermitian"]["bound"] + rec["parts"]["anti_hermitian"]["bound"] * (1 - 1e-12)
    rc = obs["SpSm_c"]
    assert abs(rc["value"][0] - EXACT_PM[1]) <= rc["bound"] and abs(rc["value"][1] - EXACT_PM[1]) <= rc["bound"]
    assert obs["Sz0"]["hermitian"] is True and "parts" not in obs["Sz0"]
    assert "per_vector_spread_under_rotation" in rec
    assert cert.validate_schema() in (True, None)


def test_correlations_with_two_letters_match_exact_values(ring4):
    spec, H, w, V = ring4
    opts = ce.Options(eigenspaces="all",
                      correlations={"pm": {"letter": "+-", "sites": [0, 1, 2, 3], "positions": [[0.0], [1.0], [2.0], [3.0]],
                                           "q": [[0.0], [math.pi]]}})
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, opts)
    corr = _cluster(cert)["correlations"]["pm"]
    assert corr["status"] == "CERTIFIED" and corr["conjugate_symmetry"] == "C_ji = conj(C_ij)"
    assert len(corr["pairs"]) == 10
    for p in corr["pairs"]:
        d = (p["j"] - p["i"]) % 4
        assert abs(p["value"] - EXACT_PM[d]) <= p["bound"], p
        assert abs(p["imag_value"]) <= p["bound"] and p["bound"] < 1e-9
        if p["i"] != p["j"]:
            assert p["hermitian_decomposition"]["parts_formed_exactly"]
            assert "parts" in p["hermitian_decomposition"] and "recombine" in p["hermitian_decomposition"]
        else:
            assert "hermitian_decomposition" not in p           # S+_i S-_i is Hermitian
    # S(q) = (1/N) sum_ij e^{iq(i-j)} <S+_i S-_j>: 0 at q = 0 for a singlet, 4/3 at q = pi
    sf = {round(r["q"][0], 6): r for r in corr["structure_factor"]}
    assert abs(sf[0.0]["value"]) <= sf[0.0]["bound"]
    assert abs(sf[round(math.pi, 6)]["value"] - 4.0 / 3) <= sf[round(math.pi, 6)]["bound"]
    assert abs(sf[round(math.pi, 6)]["imag_part"]) <= sf[round(math.pi, 6)]["bound"]
    # zz correlations unchanged (single letter path)
    opts2 = ce.Options(eigenspaces="all", correlations={"zz": {"letter": "z", "sites": [0, 1]}})
    c2 = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, opts2)
    zz = _cluster(c2)["correlations"]["zz"]
    assert zz["conjugate_symmetry"] == "C_ji = C_ij" and all("imag_value" not in p for p in zz["pairs"])


def test_rounded_parts_are_not_certified(ring4):
    """Negative control: an observable whose Hermitian parts cannot be formed exactly gets an INCONCLUSIVE record."""
    spec, H, w, V = ring4
    rng = np.random.default_rng(3)
    R = sp.csr_matrix(rng.uniform(0.1, 0.9, (16, 16)))
    opts = ce.Options(eigenspaces="all", observables={"R": R})
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, opts)
    rec = _cluster(cert)["observables"]["R"]
    assert rec["hermitian_decomposition"] and rec["parts_formed_exactly"] is False
    assert rec["status"] == "INCONCLUSIVE" and "exactly" in rec["reason"]
    # the exact-dyadic case is certified without a status word in the record (records of Theorem 6 carry none)
    good = certify_observables_batch
    assert callable(good)


def test_certify_observable_general_direct(ring4):
    from certify_ed.eigenspace import EigenspaceCertificate
    spec, H, w, V = ring4
    ec = EigenspaceCertificate([0], [float(w[0])], 1.0, 1e-15, 1e-15, 1e-15, 1e-14)
    O = site_operator(spec, "+-", (1, 2))
    rec = certify_observable_general(ec, V, O, "pm12")
    assert abs(rec["value"][0] - EXACT_PM[1]) <= rec["bound"]
    assert "pm12" in ec.observables and "pm12.hermitian_part" not in ec.observables
    batch = certify_observables_batch(ec, V, {"h": site_operator(spec, "z", (1,)), "nh": O})
    assert batch["h"]["hermitian"] is True and batch["nh"]["hermitian_decomposition"]
