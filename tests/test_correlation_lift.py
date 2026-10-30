"""C-04 (3.2): correlation functions and observables for symmetry-adapted claims, lifted through the sector isometry
(certify_ed.correlations.certify_observable_lifted / certify_correlations_lifted; docs/proofs/C-04_correlation_lift.md).

Positive cases compare the lifted values with full-space values on 8-site chains (k = 0 and parity sectors);
negative controls: a planted isometry defect (S1 FAILED -> INCONCLUSIVE records), S1 skipped (NOT_CHECKED), and a
planted defect at function level where the isometry term is what makes the interval contain tr(Pi_Q O)."""
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.basis import cyclic_isometry, reflection, sector_matrix, translation
from certify_ed.correlations import (certify_observable_lifted, compress_operator, site_operator)
from certify_ed.eigenspace import EigenspaceCertificate


@pytest.fixture(scope="module")
def ring8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    st = ce.magnetization_state_set(spec, 0)
    Hs0 = ce.build_bitwise(spec, st)
    full_idx, _ = ce.full_state_set(spec).index_of(st.states_in_order())
    P = sp.csr_matrix((np.ones(st.size), (full_idx, np.arange(st.size))), shape=(spec.dim, st.size))
    return spec, H, w, V, st, Hs0, P


def _sector(ring, sym, j):
    spec, H, w, V, st, Hs0, P = ring
    V0 = cyclic_isometry(st, sym(spec), j)
    Vs = (P @ V0).tocsr()
    Hk = sector_matrix(Hs0, V0)
    wk, Y = la.eigh(Hk.toarray())
    return Vs, Hk, wk, Y


def _full_zz(spec, psi):
    out = {}
    for i in range(spec.n_sites):
        for j in range(i, spec.n_sites):
            O = site_operator(spec, "z", (i,)) @ site_operator(spec, "z", (j,))
            out[(i, j)] = float(np.real(psi.conj() @ (O @ psi)))
    return out


def _s4_cluster(cert):
    s4 = [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]
    assert s4["status"] == "CERTIFIED", s4["summary"]
    return s4["evidence"]["clusters"][0]


@pytest.mark.parametrize("sym", [translation, reflection])
def test_lifted_correlations_match_full_space_values(ring8, sym):
    spec, H, w, V, st, Hs0, P = ring8
    Vs, Hk, wk, Y = _sector(ring8, sym, 0)
    assert abs(wk[0] - w[0]) < 1e-12                        # the ground state (singlet) lies in k = 0 / even parity
    full = _full_zz(spec, V[:, 0])
    opts = ce.Options(eigenspaces="all",
                      correlations={"zz": {"letter": "z", "sites": list(range(8)), "q": [[0.0], [math.pi]]},
                                    "pm": {"letter": "+-", "sites": [0, 1, 3]}})
    cert = ce.certify(ce.Claim(wk[:1], Y[:, :1], operator=Hk, kind="lowest", sector_isometry=Vs), spec, opts)
    assert cert.status_of("S1.model_fidelity") == "CERTIFIED"
    cl = _s4_cluster(cert)
    zz = cl["correlations"]["zz"]
    assert zz["status"] == "CERTIFIED" and zz["lifted"] and len(zz["pairs"]) == 36
    for pr in zz["pairs"]:
        assert abs(pr["value"] - full[(pr["i"], pr["j"])]) <= pr["bound"], pr
        assert pr["bound_V"] <= pr["bound"] < 1e-11
    # S(pi) of the 8-site singlet from the full-space values
    S_pi = sum(math.cos(math.pi * (i - j)) * full[(min(i, j), max(i, j))] for i in range(8) for j in range(8)) / 8
    sf = {tuple(r["q"]): r for r in zz["structure_factor"]}
    assert abs(sf[(math.pi,)]["value"] - S_pi) <= sf[(math.pi,)]["bound"]
    assert abs(sf[(0.0,)]["value"]) <= sf[(0.0,)]["bound"]           # S^z_total = 0 in the singlet
    # S+_i S-_j: SU(2) gives <S+_i S-_j> = 2 <S^z_i S^z_j> for i != j in a singlet, real
    pm = cl["correlations"]["pm"]
    assert pm["status"] == "CERTIFIED"
    for pr in pm["pairs"]:
        i, j = pr["i"], pr["j"]
        if i != j:
            assert abs(pr["value"] - 2 * full[(i, j)]) <= pr["bound"] and abs(pr["imag_value"]) <= pr["bound"]
            assert pr["hermitian_decomposition"]["parts_formed_exactly"]
        else:
            assert abs(pr["value"] - 0.5) <= pr["bound"]            # S+ S- = 1/2 + S^z on a spin-1/2 site, <S^z> = 0
    assert "Pi_Q" in zz["refers_to"] and "isometry_defect" in zz


def test_full_space_observable_is_lifted_for_a_sector_claim(ring8):
    spec, H, w, V, st, Hs0, P = ring8
    Vs, Hk, wk, Y = _sector(ring8, translation, 0)
    O = (site_operator(spec, "z", (0,)) @ site_operator(spec, "z", (4,))).tocsr()      # full space, 256 x 256
    ref = float(V[:, 0] @ (O @ V[:, 0]))
    cert = ce.certify(ce.Claim(wk[:1], Y[:, :1], operator=Hk, kind="lowest", sector_isometry=Vs), spec,
                      ce.Options(eigenspaces="all", observables={"zz04": O}))
    r = _s4_cluster(cert)["observables"]["zz04"]
    assert r["lifted"] and abs(r["value"] - ref) <= r["bound"] < 1e-11


def test_planted_isometry_defect_fails_S1_and_leaves_records_inconclusive(ring8):
    spec, H, w, V, st, Hs0, P = ring8
    Vs, Hk, wk, Y = _sector(ring8, translation, 0)
    Vb = Vs.tolil()
    Vb[:, 2] = Vb[:, 2] * 1.001                      # one basis vector 0.1 % too long: not an isometry
    Vb = Vb.tocsr()
    Hkb = sector_matrix(H, Vb)
    wb, Yb = la.eigh(Hkb.toarray())
    cert = ce.certify(ce.Claim(wb[:1], Yb[:, :1], operator=Hkb, kind="lowest", sector_isometry=Vb,
                               accuracy_target=float("inf")), spec,
                      ce.Options(eigenspaces="all", correlations={"zz": {"letter": "z", "sites": [0, 1]}}))
    assert cert.status_of("S1.model_fidelity") == "FAILED"
    cl = _s4_cluster(cert)                           # S2-S4 in the block are still certified: the lift is what fails
    r = cl["correlations"]["zz"]
    assert r["status"] == "INCONCLUSIVE" and "audit failed" in r["reason"] and "pairs" not in r
    assert r["isometry_defect"] > 1e-3


def test_without_S1_the_lift_is_not_checked(ring8):
    spec, H, w, V, st, Hs0, P = ring8
    Vs, Hk, wk, Y = _sector(ring8, translation, 0)
    cert = ce.certify(ce.Claim(wk[:1], Y[:, :1], operator=Hk, kind="lowest", sector_isometry=Vs), spec,
                      ce.Options(eigenspaces="all", model_fidelity=False, correlations={"zz": {"letter": "z", "sites": [0, 1]}}))
    cl = _s4_cluster(cert)
    r = cl["correlations"]["zz"]
    assert r["status"] == "NOT_CHECKED" and r["reason_code"] == "DEPENDENCY_NARROWED" and "pairs" not in r


def test_isometry_term_is_needed_and_sufficient_for_a_planted_defect():
    """V = Q M with a planted non-unit singular value: tr(Pi_Q O) differs from the computed value by more than the
    block bound alone (bound_V) but lies within the full bound (the isometry term is what closes the gap)."""
    rng = np.random.default_rng(5)
    n, d = 12, 5
    Q, _ = np.linalg.qr(rng.standard_normal((n, d)))
    iota = 0.05
    Mdiag = np.array([math.sqrt(1 + iota), 1.0, 1.0, 1.0, math.sqrt(1 - iota)])
    Vm = Q @ np.diag(Mdiag)
    # block eigenvector along the stretched direction, so that Pi_V and Pi_Q differ maximally
    x = np.zeros(d)
    x[0] = 1.0
    O = rng.standard_normal((n, n))
    O = (O + O.T) / 2
    ec = EigenspaceCertificate([0], [0.0], 1.0, 0.0, 0.0, 1e-15, 1e-15)
    iota_up = float(np.linalg.norm(Vm.T @ Vm - np.eye(d), 2)) * (1 + 1e-12)
    r = certify_observable_lifted(ec, x[:, None], sp.csr_matrix(Vm), sp.csr_matrix(O), "O", iota_up)
    PiQ = Q[:, :1] @ Q[:, :1].T                      # exact: Q P_s Q^H with P_s = e_0 e_0^T
    PiV = Vm[:, :1] @ Vm[:, :1].T
    tQ, tV = float(np.trace(PiQ @ O)), float(np.trace(PiV @ O))
    assert abs(r["value"] - tV) <= r["bound_V"]                       # exact identity tr(P_s V^H O V) = tr(Pi_V O)
    assert abs(r["value"] - tQ) > r["bound_V"]                         # the block bound alone would be false
    assert abs(r["value"] - tQ) <= r["bound"]                          # the isometry term makes it true
    assert r["isometry_term"] >= abs(tV - tQ)


def test_compression_rounding_bound_in_exact_arithmetic():
    """||fl(V^H O V)_h - V^H O V||_F in exact rationals is below the bound returned by compress_operator."""
    rng = np.random.default_rng(9)
    for cplx in (False, True):
        n, d = 30, 7
        V = sp.random(n, d, density=0.3, random_state=11, format="csr")
        if cplx:
            V = V.multiply(np.exp(2j * np.pi * rng.random((n, 1))))
        O = sp.random(n, n, density=0.2, random_state=12, format="csr")
        if cplx:
            O = O.multiply(np.exp(2j * np.pi * rng.random((n, 1))))
        O = (O + O.conj().T).tocsr()
        Oh, rnd, _ = compress_operator(V, O)
        Vd, Od, Ohd = V.toarray(), O.toarray(), Oh.toarray()
        F = lambda z: (Fraction(float(np.real(z))), Fraction(float(np.imag(z))))
        err2 = Fraction(0)
        for i in range(d):
            for j in range(d):
                re = im = Fraction(0)
                for a in range(n):
                    if Vd[a, i] == 0:
                        continue
                    va_r, va_i = F(np.conj(Vd[a, i]))
                    for b in range(n):
                        if Od[a, b] == 0 or Vd[b, j] == 0:
                            continue
                        o_r, o_i = F(Od[a, b])
                        vb_r, vb_i = F(Vd[b, j])
                        p_r, p_i = va_r * o_r - va_i * o_i, va_r * o_i + va_i * o_r
                        re += p_r * vb_r - p_i * vb_i
                        im += p_r * vb_i + p_i * vb_r
                c_r, c_i = F(Ohd[i, j])
                err2 += (re - c_r) ** 2 + (im - c_i) ** 2
        assert err2 <= Fraction(rnd) ** 2
