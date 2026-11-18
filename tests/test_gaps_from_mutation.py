"""Tests written to close gaps found by mutation analysis (experiments/e6_mutation.py, first run
2026-10-17): code paths that no test exercised, or exercised without checking the result."""
import itertools
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.basis import u1_state_set
from certify_ed.checker import (check_semidefinite_exact, negative_count_exact, residual_inf_norm_exact,
                                residual_inf_norm_upper)
from certify_ed.spec import ModelSpec
from certify_ed.su2 import su2_structural

from conftest import needs_flint, needs_quspin


# ---------------------------------------------------------------- exact checker
def test_negative_count_exact_blocks():
    d = np.zeros((5, 5))
    d[0, 0] = -1.0                           # 1x1 negative
    d[1:3, 1:3] = [[1.0, 2.0], [2.0, 1.0]]   # det < 0: one negative
    d[3:5, 3:5] = [[-2.0, 0.5], [0.5, -1.0]] # det > 0, trace < 0: two negative
    assert negative_count_exact(d) == (4, False)
    d2 = np.diag([1.0, 0.0, 2.0])
    assert negative_count_exact(d2) == (None, True)


@needs_flint
def test_residual_norms_match_rational_arithmetic():
    rng = np.random.default_rng(3)
    n = 12
    B = rng.integers(-5, 6, (n, n)).astype(float)
    A = B + B.T + 40 * np.eye(n)
    lu, d, _ = la.ldl(A, lower=True, hermitian=True)
    # exact value of ||lu d lu^T - A||_inf in rationals
    F = lambda M: [[Fraction(x) for x in row] for row in M]
    L, D, Aq = F(lu), F(d), F(A)
    LD = [[sum(L[i][k] * D[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    E = [[sum(LD[i][k] * L[j][k] for k in range(n)) - Aq[i][j] for j in range(n)] for i in range(n)]
    exact = max(sum(abs(x) for x in row) for row in E)
    ub = residual_inf_norm_upper(A, lu, d)
    assert Fraction(ub) >= exact and ub <= 2 * float(exact) + 1e-300
    ue, Dt = residual_inf_norm_exact(A, lu, d)
    assert ue >= 0 and ue < 1e-9 * np.abs(A).max()


def test_check_semidefinite_exact_definite_and_indefinite():
    rng = np.random.default_rng(4)
    n = 30
    B = rng.integers(-9, 10, (n, n)).astype(float)
    A = B + B.T
    lam = la.eigvalsh(A)
    Apd = A - np.floor(lam[0] - 2) * np.eye(n)          # integer, positive definite
    r = check_semidefinite_exact(Apd)
    assert r["nu"] == 0 and r["bound_E"] < 1e-8 and r["lambda_min_lower"] == -r["bound_E"]
    Ain = A - np.floor(lam[3]) * np.eye(n)              # at least 3 negative eigenvalues
    r2 = check_semidefinite_exact(Ain)
    assert r2["nu"] == int(np.sum(la.eigvalsh(Ain) < 0)) and "lambda_min_lower" not in r2


# ---------------------------------------------------------------- SU(2) structural test
def test_su2_rejects_off_diagonal_and_antisymmetric_exchange():
    loc = ce.LocalSpace.spin_space("1/2", "spin")
    heis = [["zz", [[1.0, 0, 1]]], ["+-", [[0.5, 0, 1]]], ["-+", [[0.5, 0, 1]]]]
    assert su2_structural(ModelSpec.from_static(3, loc, heis))[0]
    assert not su2_structural(ModelSpec.from_static(3, loc, heis + [["xy", [[0.3, 1, 2]]]]))[0]
    dm = [["zz", [[1.0, 0, 1]]], ["+-", [[0.5 + 0.1j, 0, 1]]], ["-+", [[0.5 - 0.1j, 0, 1]]]]
    assert not su2_structural(ModelSpec.from_static(3, loc, dm))[0]
    assert not su2_structural(ModelSpec.from_static(3, loc, heis + [["z", [[0.2, 2]]]]))[0]


# ---------------------------------------------------------------- bases and factories
@pytest.mark.parametrize("N,total", [(6, 0), (6, 6), (7, 3), (5, 0)])
def test_u1_state_set_edges(N, total):
    spec = ce.heisenberg_chain(N)
    st = u1_state_set(spec, total)
    brute = sorted(sum(1 << (N - 1 - i) for i in c) for c in itertools.combinations(range(N), total))
    assert list(st.states_asc) == brute


def test_heisenberg_square_open_bond_count():
    Lx, Ly = 4, 3
    spec = ce.heisenberg_square(Lx, Ly, boundary="open")
    zz = [t for t in spec.terms if t.op == "zz"][0]
    assert len(zz.couplings) == 2 * Lx * Ly - Lx - Ly
    spec_p = ce.heisenberg_square(Lx, Ly)
    assert len([t for t in spec_p.terms if t.op == "zz"][0].couplings) == 2 * Lx * Ly


# ---------------------------------------------------------------- statements
def test_twofold_cluster_certified_by_default():
    spec = ce.heisenberg_chain(8)
    st = ce.u1_state_set(spec, 4)
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    k = 2
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    while k < len(w) and w[k] - w[k - 1] < 1e-8:
        k += 1
    # find a k that ends after a twofold level
    sizes = []
    i = 0
    while i < k:
        j = i
        while j + 1 < len(w) and w[j + 1] - w[i] < 1e-8:
            j += 1
        sizes.append(j - i + 1)
        i = j + 1
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest", states=st), spec)
    s4 = [s for s in c.statements if s["id"] == "S4.eigenspaces"][0]
    if max(sizes) >= 2:
        assert s4["status"] == "CERTIFIED"
        assert any(len(cl["members"]) == 2 for cl in s4["evidence"]["clusters"]) or max(sizes) > 2


def test_window_claim_certified_window_inside_requested():
    spec = ce.heisenberg_chain(8, J2=0.2)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    gaps = [i for i in range(1, len(w)) if w[i] - w[i - 1] > 1e-6]
    i0 = next(i for i in gaps if i >= 4)
    i1 = next(i for i in gaps if i >= i0 + 6)
    a, b = 0.5 * (w[i0 - 1] + w[i0]), 0.5 * (w[i1 - 1] + w[i1])
    sel = (w >= a) & (w <= b)
    c = ce.certify(ce.Claim(w[sel], V[:, sel], operator=H, kind="window", window=(a, b)), spec)
    s3 = [s for s in c.statements if s["id"] == "S3.completeness"][0]
    assert s3["status"] == "CERTIFIED"
    lo, hi = s3["evidence"]["certified_window"]
    assert a <= lo <= w[sel].min() and w[sel].max() <= hi <= b


def test_next_eigenvalue_lower_bound_is_below_shift_and_truth():
    spec = ce.heisenberg_chain(10)
    st = ce.u1_state_set(spec, 5)
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest", states=st), spec)
    ev = [s for s in c.statements if s["id"] == "S3.completeness"][0]["evidence"]
    assert ev["next_eigenvalue_lower_bound"] < ev["top_shift"]
    # sigma - beta rounded downward (one step below the round-to-nearest difference)
    from fractions import Fraction
    from certify_ed.fp import sub_down
    assert ev["next_eigenvalue_lower_bound"] == float(sub_down(ev["top_shift"], ev["beta_effective"]))
    assert Fraction(ev["next_eigenvalue_lower_bound"]) <= Fraction(ev["top_shift"]) - Fraction(ev["beta_effective"])
    assert ev["next_eigenvalue_lower_bound"] <= w[1]


def test_lift_radius_refuses_non_isometry():
    from certify_ed.sectors import audit_sector, lift_radius
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    V = sp.csr_matrix(np.eye(H.shape[0])[:, :1] * 1.5)       # not an isometry: defect 1.25
    aud = audit_sector(H, V, (V.T @ H @ V).tocsr())
    assert 1 <= aud.isometry_defect < 2
    assert lift_radius(1e-12, aud) == float("inf")


# ---------------------------------------------------------------- QuSpin importer
@needs_quspin
def test_importer_reads_pauli_pm1_and_boson_bases():
    from quspin.basis import boson_basis_1d, spin_basis_1d
    from certify_ed.spec import local_space_from_quspin_basis
    assert local_space_from_quspin_basis(spin_basis_1d(4, pauli=-1)).convention == "pauli_pm1"
    assert local_space_from_quspin_basis(spin_basis_1d(4, pauli=1)).convention == "pauli"
    assert local_space_from_quspin_basis(spin_basis_1d(4, pauli=0)).convention == "spin"
    lb = local_space_from_quspin_basis(boson_basis_1d(3, sps=4))
    assert lb.kind == "boson" and lb.q == 4
