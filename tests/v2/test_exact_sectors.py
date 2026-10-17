"""Exact sector re-check (certify_ed.exact_sectors)."""
import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.basis import translation, reflection, spin_flip, cyclic_isometry
from certify_ed.exact_sectors import exact_recheck_sector_lowest, site_permutation


def _k0(N, J2):
    spec = ce.heisenberg_chain(N, J2=J2)
    st = ce.u1_state_set(spec, N // 2)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    V0 = cyclic_isometry(st, T, 0)
    w, Y = la.eigh((V0.conj().T @ H @ V0).toarray())
    return spec, st, T, V0, w, Y


def test_chain_k0_lowest_certified_and_dimensions_add_up():
    spec, st, T, V0, w, Y = _k0(12, 0.3)
    r = exact_recheck_sector_lowest(spec, st, [T, reflection(spec), spin_flip(spec)], {0: 0}, w[:2],
                                    (V0 @ Y[:, :2]).real, next_hint=w[2])
    assert r["status"] == "CERTIFIED", r
    assert sum(s["dim"] for s in r["sectors"]) == r["dim_fixed"] == V0.shape[1]
    assert r["group_order"] == 48


def test_chain_k0_missing_level_fails():
    spec, st, T, V0, w, Y = _k0(12, 0.3)
    r = exact_recheck_sector_lowest(spec, st, [T, reflection(spec), spin_flip(spec)], {0: 0}, w[[0, 2]],
                                    (V0 @ Y[:, [0, 2]]).real, next_hint=w[3])
    assert r["status"] == "FAILED", r


def test_wrong_sector_vector_is_not_certified():
    # a k = pi eigenvector claimed as the k = 0 ground state: its exact projection onto k = 0 vanishes
    # or has a large residual, so the claim cannot be certified
    spec = ce.heisenberg_chain(10)
    st = ce.u1_state_set(spec, 5)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    Vpi = cyclic_isometry(st, T, 5)
    w, Y = la.eigh((Vpi.conj().T @ H @ Vpi).toarray())
    x = np.real_if_close(Vpi @ Y[:, :1])
    with np.errstate(all="ignore"):
        try:
            r = exact_recheck_sector_lowest(spec, st, [T], {0: 0}, w[:1], x, next_hint=w[1])
            assert r["status"] != "CERTIFIED"
        except ZeroDivisionError:
            pass


def test_complex_momenta_square_lattice_full_block():
    Lx = Ly = 2
    spec = ce.heisenberg_square(4, 2)              # 8 sites; translations of order 4 and 2
    n = spec.n_sites
    st = ce.u1_state_set(spec, n // 2)
    H = ce.build_bitwise(spec, st).toarray()
    w, V = la.eigh(H)
    site = lambda x, y: (y % 2) * 4 + (x % 4)
    px = [0] * n; py = [0] * n
    for y in range(2):
        for x in range(4):
            px[site(x, y)] = site(x + 1, y); py[site(x, y)] = site(x, y + 1)
    gens = [site_permutation(spec, px, "Tx"), site_permutation(spec, py, "Ty"), spin_flip(spec)]
    k = 1
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    r = exact_recheck_sector_lowest(spec, st, gens, {}, w[:k], V[:, :k], next_hint=w[k])
    assert r["status"] == "CERTIFIED", r
    assert sum(s["dim"] for s in r["sectors"]) == st.size


def test_non_symmetry_rejected():
    spec = ce.heisenberg_chain(8, boundary="open")   # translation is not a symmetry
    st = ce.u1_state_set(spec, 4)
    w, V = la.eigh(ce.build_bitwise(spec, st).toarray())
    r = exact_recheck_sector_lowest(spec, st, [translation(spec)], {0: 0}, w[:1], V[:, :1], next_hint=w[1])
    assert r["status"] == "FAILED" and "not a symmetry" in r["reason"]
