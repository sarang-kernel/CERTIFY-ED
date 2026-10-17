import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

import certify_ed as ce
from certify_ed.su2 import su2_structural


def _qn(cert, name):
    s4 = [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]
    return [c["quantum_numbers"][name] for c in s4["evidence"]["clusters"] if name in c.get("quantum_numbers", {})]


def test_structural_check():
    assert su2_structural(ce.heisenberg_chain(6, J2=0.3))[0]
    assert su2_structural(ce.heisenberg_chain(6, convention="pauli"))[0]
    assert not su2_structural(ce.heisenberg_chain(6, Delta=0.5))[0]
    assert not su2_structural(ce.tfim_chain(6, h=1.0))[0]


def test_spin1_triplet_and_singlet_labels():
    spec = ce.spin1_heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    k = 4      # singlet + triplet
    opts = ce.Options(symmetries={"S": ("su2", None)}, eigenspaces="all")
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, opts)
    q = _qn(c, "S")
    spins = sorted(x for r in q for x in r.get("total_spin", []))
    assert all(r["status"] == "CERTIFIED" for r in q), q
    assert spins == [0.0, 1.0, 1.0, 1.0]


def test_spin_half_ring_in_sz_block():
    spec = ce.heisenberg_chain(10)
    st = ce.u1_state_set(spec, 5)
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    k = 4
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    opts = ce.Options(symmetries={"S": ("su2", None)}, eigenspaces="all")
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest", states=st), spec, opts)
    q = _qn(c, "S")
    assert q and all(r["status"] == "CERTIFIED" for r in q)
    assert q[0]["total_spin"] == [0.0]          # Lieb-Mattis: singlet ground state


def test_not_invariant_is_not_checked():
    spec = ce.heisenberg_chain(8, Delta=0.7)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    opts = ce.Options(symmetries={"S": ("su2", None)}, eigenspaces="all")
    c = ce.certify(ce.Claim(w[:3], V[:, :3], operator=H, kind="lowest"), spec, opts)
    q = _qn(c, "S")
    assert q and all(r["status"] == "NOT-CHECKED" for r in q)
