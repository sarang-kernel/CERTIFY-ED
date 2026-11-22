"""Tests closing gaps found by the survivors of the 2026-10-18 full-run mutation analysis
(results/run_full_final/e6_mutation_20261018_192859, 12 survivors inspected; see inspection.json)."""
import sys

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from conftest import needs_flint

CERT = sys.modules["certify_ed.certify"]


def _st(c, sid):
    return [s for s in c.statements if s["id"] == sid][0]


# ------------------------------------------------ certify.py:255: the internal builder cross-check runs for full-space claims
def test_internal_builder_disagreement_makes_s1_inconclusive(monkeypatch):
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    monkeypatch.setattr(CERT, "cross_construct", lambda *a, **k: {"agree": False, "max_abs_diff": 1.0, "tolerance": 0.0})
    c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec)
    s1 = _st(c, "S1.model_fidelity")
    assert s1["status"] == "INCONCLUSIVE" and "disagree" in s1["summary"]


# ------------------------------------------------ certify.py:565: one-dimensional eigenvector input
def test_single_eigenvector_given_as_1d_array():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim([w[0]], V[:, 0], operator=H, kind="lowest"), spec)
    assert _st(c, "S2.eigenpairs")["status"] == "CERTIFIED"
    assert _st(c, "S3.completeness")["status"] == "CERTIFIED"


# ------------------------------------------------ certify.py:615: a dense, slightly non-Hermitian producer matrix
def test_dense_nonhermitian_producer_is_symmetrised_and_reported():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).toarray()
    rng = np.random.default_rng(8)
    A = rng.standard_normal(H.shape) * (np.abs(H) > 0)
    Hp = H + 1e-15 * (A - A.T)                      # dense ndarray, tiny anti-Hermitian part
    w, V = la.eigh((Hp + Hp.T) / 2)
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=Hp, kind="lowest"), spec)
    hp = c.body["hermitian_part"]
    assert hp is not None and hp["max_abs_asymmetry"] > 0
    assert _st(c, "S2.eigenpairs")["status"] == "CERTIFIED"


# ------------------------------------------------ certify.py:876: the float-model caveat is listed unless S6 certified
@needs_flint
def test_float_model_caveat_listed_exactly_when_no_exact_recheck():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    claim = ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest")
    caveat = "interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')"
    c1 = ce.certify(claim, spec)
    assert caveat in c1.body["not_verified"]
    c2 = ce.certify(claim, spec, ce.Options(exact_recheck=True))
    assert _st(c2, "S6.exact_recheck")["status"] == "CERTIFIED"
    assert caveat not in c2.body["not_verified"]


# ------------------------------------------------ basis.py:91: fermionic translation sign
def test_fermion_translation_commutes_with_periodic_hopping():
    from certify_ed.basis import translation
    from certify_ed.sectors import symmetry_matrix
    N = 6
    spec = ce.spinless_fermion_chain(N, t=1.0, V=0.7, boundary="periodic")
    T = translation(spec)
    for nf in (1, 2, 3, 4):                         # the boundary sign depends on the parity of N_f
        st = ce.u1_state_set(spec, nf)
        H = sp.csr_matrix(ce.build_bitwise(spec, st))
        G = symmetry_matrix(st, T)
        assert abs(G @ H - H @ G).max() < 1e-14, nf


# ------------------------------------------------ spec.py:251: default J2 of the square lattice
def test_heisenberg_square_default_has_no_next_nearest_coupling():
    a = ce.build_bitwise(ce.heisenberg_square(3, 2)).toarray()
    b = ce.build_bitwise(ce.heisenberg_square(3, 2, J2=0.0)).toarray()
    assert np.array_equal(a, b)


# ------------------------------------------------ exact_sectors.py:347: separation of consecutive components
@needs_flint
def test_exact_sector_count_requires_separation_from_the_next_component():
    """The next claimed interval ends just above the previous one (gap far below delta). The count
    at the point between them cannot be attributed, so the claim must not be CERTIFIED, whatever the
    width of the next interval."""
    from certify_ed.basis import spin_flip
    from certify_ed.exact_sectors import exact_recheck_sector_lowest
    spec = ce.heisenberg_chain(8, J2=0.3)
    st = ce.u1_state_set(spec, 4)
    H = ce.build_bitwise(spec, st).toarray()
    w, V = la.eigh(H)
    i = next(k for k in range(1, len(w)) if w[k] - w[k - 1] > 1e-3)     # a gap above level 0
    X = V[:, [0, i]]
    # claimed value w[i] - d, radius ~ d: interval [w0 + 2 off, w_i]. The first radius depends on the LAPACK that
    # computed V (about 1.4e-15 with one build, 2.4e-15 with another): the offset grows until the two intervals are
    # apart, so that the case below is constructed on any build
    for k in range(10):
        off = 1e-15 * 2 ** k
        d = 0.5 * (w[i] - w[0]) - off
        lam = np.array([w[0], w[i] - d])
        r = exact_recheck_sector_lowest(spec, st, [spin_flip(spec)], {}, lam, X, next_hint=w[i + 1])
        comps = r["components"]
        if len(comps) == 2:
            break
    first = r["counts"][0]
    assert len(comps) == 2, r
    margin = comps[1][0] - first["point"]
    assert len(comps) == 2 and margin <= first["delta"], r          # the case this test is about
    assert not first["separated"] and r["status"] != "CERTIFIED", r
