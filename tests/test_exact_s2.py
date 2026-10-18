"""Exact (floating-point-free) re-check of S2: the declared model as an exact rational matrix and
residual radii computed in integer arithmetic (certify_ed.checker.exact_matrix / exact_eigenpair_radii)."""
import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.checker import exact_matrix, exact_eigenpair_radii
from certify_ed.spec import ModelSpec


def _dense(ex, n):
    from certify_ed.checker import exact_dense
    return exact_dense(ex)


@pytest.mark.parametrize("make", [lambda: ce.heisenberg_chain(8, J2=0.3, Delta=0.7),
                                  lambda: ce.spinless_fermion_chain(7, t=1.0, V=0.37, mu=0.11),
                                  lambda: ce.heisenberg_chain(6, convention="pauli")])
def test_exact_matrix_matches_float_builder(make):
    spec = make()
    H = ce.build_bitwise(spec).toarray()
    E = _dense(exact_matrix(spec), H.shape[0])
    assert np.abs(E - H).max() <= 4e-16 * max(1.0, np.abs(H).max())


def test_exact_radius_zero_for_exact_eigenvector():
    spec = ce.heisenberg_chain(6)            # ferromagnetic state |up...up>: eigenvalue N/4 exactly
    n = spec.dim
    x = np.zeros((n, 1)); x[0, 0] = 1.0
    r = exact_eigenpair_radii(spec, [6 / 4], x)
    assert r["hermitian"] and r["radii"] == [0.0]


def test_exact_radius_bounds_eigenvalue_distance():
    spec = ce.heisenberg_chain(8, J2=0.3)
    st = ce.u1_state_set(spec, 4)
    H = ce.build_bitwise(spec, st).toarray()
    w, V = la.eigh(H)
    shift = 1e-3
    r = exact_eigenpair_radii(spec, w[:3] + shift, V[:, :3], st)
    for k in range(3):
        # Krylov-Bogolyubov: some eigenvalue within the radius; the true distance is ~shift
        assert r["radii"][k] >= np.min(np.abs(w - (w[k] + shift))) * (1 - 1e-9)
        assert r["radii"][k] <= shift * (1 + 1e-6)


def test_exact_radii_below_floating_point_radii():
    spec = ce.heisenberg_chain(10, J2=0.3)
    st = ce.u1_state_set(spec, 5)
    Hs = ce.build_bitwise(spec, st)
    w, V = la.eigh(Hs.toarray())
    r = exact_eigenpair_radii(spec, w[:4], V[:, :4], st)
    c = ce.certify(ce.Claim(w[:4], V[:, :4], operator=Hs, kind="lowest", states=st), spec)
    s2 = [s for s in c.statements if s["id"] == "S2.eigenpairs"][0]
    assert s2["status"] == "CERTIFIED"
    assert max(r["radii"]) <= s2["evidence"]["max_radius"] + s2["evidence"]["model_shift"]


def test_complex_hermitian_model():
    base = ce.heisenberg_chain(5)
    static = [[t.op, [[c] + list(s) for c, s in t.couplings]] for t in base.terms]
    D = 0.3
    static += [["xy", [[D, i, (i + 1) % 5] for i in range(5)]], ["yx", [[-D, i, (i + 1) % 5] for i in range(5)]]]
    spec = ModelSpec.from_static(5, base.local, static, name="dm_chain")
    H = ce.build_bitwise(spec).toarray()
    assert np.iscomplexobj(H) and np.abs(H.imag).max() > 0
    E = _dense(exact_matrix(spec), H.shape[0])
    assert np.abs(E - H).max() <= 1e-15
    w, V = la.eigh(H)
    r = exact_eigenpair_radii(spec, w[:2], V[:, :2])
    assert r["hermitian"] and max(r["radii"]) < 1e-13


def test_non_hermitian_spec_detected():
    base = ce.heisenberg_chain(4)
    spec = ModelSpec.from_static(4, base.local, [["+-", [[1.0, 0, 1]]]], name="one_way")
    r = exact_eigenpair_radii(spec, [0.0], np.eye(spec.dim)[:, :1])
    assert r["hermitian"] is False


@pytest.mark.parametrize("make", [lambda: ce.spin1_heisenberg_chain(5),
                                  lambda: ce.bose_hubbard_chain(4, n_max=3),
                                  lambda: ModelSpec.from_static(4, __import__("certify_ed").LocalSpace.spin_space("3/2"),
                                      [["+-", [[0.5, i, (i + 1) % 4] for i in range(4)]],
                                       ["-+", [[0.5, i, (i + 1) % 4] for i in range(4)]],
                                       ["zz", [[1.0, i, (i + 1) % 4] for i in range(4)]],
                                       ["x", [[0.3, i] for i in range(4)]]])])
def test_higher_spin_and_bosons_exact(make):
    """Spin >= 1 and bosons have irrational matrix elements; in the scaled basis D^-1 H D they are
    rational, and the exact layer certifies their spectra."""
    spec = make()
    H = ce.build_bitwise(spec).toarray()
    ex = exact_matrix(spec)
    assert np.abs(_dense(ex, H.shape[0]) - H).max() <= 1e-13 * max(1.0, np.abs(H).max())
    from certify_ed.checker import is_hermitian_exact
    assert is_hermitian_exact(ex)
    w, V = la.eigh(H)
    k = 1
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    from certify_ed.checker import exact_recheck_lowest
    r = exact_recheck_lowest(spec, w[:k], V[:, :k], next_hint=w[k])
    assert r["status"] == "CERTIFIED", r
    assert max(r["radii"]) < 1e-12


def _gap_k(w, kmin):
    k = kmin
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    return k


@pytest.mark.parametrize("make,sector", [(lambda: ce.heisenberg_chain(10, J2=0.3), 5),
                                         (lambda: ce.spinless_fermion_chain(9, t=1.0, V=0.37), None)])
def test_exact_recheck_lowest_certifies_complete_claim(make, sector):
    spec = make()
    st = ce.u1_state_set(spec, sector) if sector is not None else None
    H = ce.build_bitwise(spec, st).toarray()
    w, V = la.eigh(H)
    k = _gap_k(w, 6)
    from certify_ed.checker import exact_recheck_lowest
    r = exact_recheck_lowest(spec, w[:k], V[:, :k], st, next_hint=w[k])
    assert r["status"] == "CERTIFIED", r
    assert r["no_floating_point_assumption"]
    assert sum(m for _, _, m in r["components"]) == k


def test_exact_recheck_lowest_detects_missing_level():
    spec = ce.heisenberg_chain(10, J2=0.3)
    st = ce.u1_state_set(spec, 5)
    w, V = la.eigh(ce.build_bitwise(spec, st).toarray())
    sel = [0, 1, 3, 4]                                # w[2] left out
    from certify_ed.checker import exact_recheck_lowest
    r = exact_recheck_lowest(spec, w[sel], V[:, sel], st, next_hint=w[5])
    assert r["status"] == "FAILED", r


def test_exact_recheck_lowest_detects_truncated_multiplet():
    spec = ce.heisenberg_chain(10, J2=0.3)
    st = ce.u1_state_set(spec, 5)
    w, V = la.eigh(ce.build_bitwise(spec, st).toarray())
    k = next(i for i in range(1, len(w)) if w[i] - w[i - 1] < 1e-8)   # cut inside a degenerate level
    from certify_ed.checker import exact_recheck_lowest
    r = exact_recheck_lowest(spec, w[:k], V[:, :k], st, next_hint=w[k])
    assert r["status"] != "CERTIFIED", r


def test_certify_with_s6_exact_recheck():
    spec = ce.spin1_heisenberg_chain(6)
    st = ce.u1_state_set(spec, 6)
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    k = 1
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest", states=st), spec,
                   ce.Options(exact_recheck=True))
    s6 = [s for s in c.statements if s["id"] == "S6.exact_recheck"][0]
    assert s6["status"] == "CERTIFIED", s6
    assert not any("standard floating-point model" in x for x in c.body["not_verified"])
    # a claim that skips a level fails S6 as well
    sel = [0, 2] if w[1] - w[0] > 1e-8 else [0, 3]
    c2 = ce.certify(ce.Claim(w[sel], V[:, sel], operator=H, kind="lowest", states=st), spec,
                    ce.Options(exact_recheck=True))
    s6b = [s for s in c2.statements if s["id"] == "S6.exact_recheck"][0]
    assert s6b["status"] in ("FAILED", "INCONCLUSIVE")
