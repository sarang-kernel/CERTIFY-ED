"""Metamorphic relations: transformations of a specification with a known effect on the spectrum.

Each relation is checked on random operator-string models for every supported local space, through
the reference builder and the certifier. A violation exposes an error in operator construction
(signs, site indexing, Jordan-Wigner strings, conventions) without needing the correct answer.
"""
import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.spec import ModelSpec

from conftest import LOCALS, random_spec


def _spec_like(spec, static):
    return ModelSpec.from_static(spec.n_sites, spec.local, static)


def _static(spec):
    return [t.to_json() for t in spec.terms]


def _relabel(spec, perm):
    st = []
    for op, cps in _static(spec):
        st.append([op, [[c[0]] + [perm[s] for s in c[1:]] for c in cps]])
    return _spec_like(spec, st)


def _scaled(spec, a):
    st = []
    for op, cps in _static(spec):
        st.append([op, [[_mul(c[0], a)] + c[1:] for c in cps]])
    return _spec_like(spec, st)


def _mul(c, a):
    if isinstance(c, (list, tuple)):
        return [c[0] * a, c[1] * a]
    return c * a


def _spectrum(spec):
    return la.eigvalsh(ce.build_bitwise(spec).toarray())


@pytest.mark.parametrize("name,mk,N", LOCALS, ids=[l[0] for l in LOCALS])
@pytest.mark.parametrize("seed", range(5))
def test_site_relabelling_preserves_spectrum(name, mk, N, seed):
    rng = np.random.default_rng(seed)
    local = mk()
    spec = random_spec(local, N, rng, allow_complex=not local.is_fermionic)
    perm = list(rng.permutation(N))
    if local.is_fermionic:
        # a site permutation of fermions is unitary (with Jordan-Wigner signs) only for
        # number-conserving models; keep the number-conserving strings
        spec = random_spec(local, N, rng, allow_complex=False)
        st = [t for t in _static(spec) if t[0].count("+") == t[0].count("-")]
        if not st:
            pytest.skip("no number-conserving strings drawn")
        spec = _spec_like(spec, st)
    w0 = _spectrum(spec)
    w1 = _spectrum(_relabel(spec, perm))
    assert np.allclose(w0, w1, atol=1e-10 * max(1, np.abs(w0).max()))


@pytest.mark.parametrize("name,mk,N", LOCALS, ids=[l[0] for l in LOCALS])
@pytest.mark.parametrize("a", [3.0, -0.5])
def test_scaling(name, mk, N, a):
    rng = np.random.default_rng(7)
    spec = random_spec(mk(), N, rng)
    w0 = _spectrum(spec)
    w1 = _spectrum(_scaled(spec, a))
    assert np.allclose(np.sort(a * w0), w1, atol=1e-10 * max(1, abs(a) * np.abs(w0).max()))


def test_certified_values_follow_a_shift_and_relabelling():
    """The certificate's statements are invariant under a relabelling applied to both the producer
    and the specification, and detect it when applied to one of them only (error F3 of the paper)."""
    spec = ce.heisenberg_chain(10, J2=0.3)
    st = [t.to_json() for t in spec.terms] + [["z", [[0.37, 0]]]]       # breaks reflection symmetry
    spec = ModelSpec.from_static(10, spec.local, st)
    perm = [3, 0, 7, 1, 9, 2, 8, 5, 6, 4]
    spec_p = _relabel(spec, perm)
    H, Hp = ce.build_bitwise(spec), ce.build_bitwise(spec_p)
    w, V = la.eigh(H.toarray())
    wp, Vp = la.eigh(Hp.toarray())
    assert np.allclose(w, wp, atol=1e-11)
    c_both = ce.certify(ce.Claim(wp[:3], Vp[:, :3], operator=Hp, kind="lowest"), spec_p)
    c_one = ce.certify(ce.Claim(wp[:3], Vp[:, :3], operator=Hp, kind="lowest"), spec)
    s = lambda c, i: [x for x in c.statements if x["id"] == i][0]["status"]
    assert s(c_both, "S1.model_fidelity") == "CERTIFIED"
    assert s(c_one, "S1.model_fidelity") == "FAILED"


def test_sublattice_rotation_on_bipartite_ring():
    """U = prod_{odd i} exp(i pi S^z_i) maps sum J (S+S- + h.c.)/2 to its negative: H(J, Delta) and
    H(-J flip, Delta) with the transverse part negated are unitarily equivalent on an even ring."""
    N = 10
    spec = ce.heisenberg_chain(N, Delta=0.6)
    st = []
    for op, cps in _static(spec):
        if op in ("+-", "-+"):
            st.append([op, [[_mul(c[0], -1.0)] + c[1:] for c in cps]])
        else:
            st.append([op, cps])
    w0, w1 = _spectrum(spec), _spectrum(_spec_like(spec, st))
    assert np.allclose(w0, w1, atol=1e-11)


def test_decoupled_sum():
    a = ce.heisenberg_chain(4, boundary="open")
    b = ce.heisenberg_chain(3, J=0.8, Delta=0.4, boundary="open")
    # place a on sites 0-3 and b (written in spin units) on sites 4-6 of a 7-site spin-1/2 system
    loc = ce.LocalSpace.spin_space("1/2", "spin")
    st = [t.to_json() for t in a.terms]
    for op, cps in [t.to_json() for t in b.terms]:
        st.append([op, [[c[0]] + [s + 4 for s in c[1:]] for c in cps]])
    # b may use a different convention; rebuild its spectrum in the same local space for the oracle
    b_same = ModelSpec.from_static(3, loc, [t.to_json() for t in b.terms]) if b.local.convention == "spin" else None
    if b_same is None:
        pytest.skip("tfim factory uses another convention")
    w = _spectrum(ModelSpec.from_static(7, loc, st))
    wa, wb = _spectrum(a), _spectrum(b_same)
    assert np.allclose(np.sort(np.add.outer(wa, wb).ravel()), w, atol=1e-11)
