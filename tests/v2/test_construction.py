"""The two reference builders must agree exactly, and must agree with QuSpin."""
import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed.construct import (build_bitwise, build_kron, cross_construct, matrix_difference,
                                  SpecOperator)
from certify_ed.invariants import spec_invariants, matrix_invariants, compare_invariants
from certify_ed.localops import LocalSpace
from certify_ed import spec as S

from conftest import LOCALS, random_spec, needs_quspin


@pytest.mark.parametrize("name,mk,N", LOCALS, ids=[l[0] for l in LOCALS])
def test_shift_decomposition_matches_matrix(name, mk, N):
    local = mk()
    for letter in local.letters:
        M = local.matrix(letter)
        R = np.zeros_like(M)
        for delta, amp in local.shift_components(letter):
            for k in range(local.q):
                if 0 <= k + delta < local.q:
                    R[k + delta, k] += amp[k]
        assert np.array_equal(M, R) or np.allclose(M, R, atol=1e-15, rtol=0), letter


@pytest.mark.parametrize("name,mk,N", LOCALS, ids=[l[0] for l in LOCALS])
@pytest.mark.parametrize("seed", range(6))
def test_kron_equals_bitwise_random(name, mk, N, seed):
    rng = np.random.default_rng(1000 * seed + len(name))
    spec = random_spec(mk(), N, rng)
    rep = cross_construct(spec)
    assert rep["agree"], rep
    H = build_bitwise(spec)
    assert abs(H - H.conj().T).max() <= 1e-14 * max(1.0, abs(H).max())


@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_state_order(order):
    spec = S.heisenberg_chain(5).with_state_order(order)
    assert cross_construct(spec)["agree"]


def test_matrix_free_operator_matches_matrix():
    rng = np.random.default_rng(3)
    spec = random_spec(LocalSpace.fermion_space(), 6, rng)
    H = build_bitwise(spec)
    op = SpecOperator(spec)
    x = rng.standard_normal(spec.dim) + 1j * rng.standard_normal(spec.dim)
    assert np.allclose(op.matvec(x), H @ x, atol=1e-13)
    # |H||x| is dominated by abs_matvec
    assert np.all(op.abs_matvec(np.abs(x)) >= abs(H) @ np.abs(x) - 1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_spec_invariants_match_matrix(seed):
    rng = np.random.default_rng(seed)
    for mk in (lambda: LocalSpace.spin_space("1/2", "pauli"), LocalSpace.fermion_space,
               lambda: LocalSpace.boson_space(2)):
        spec = random_spec(mk(), 4, rng)
        c = compare_invariants(spec_invariants(spec), matrix_invariants(build_bitwise(spec)))
        assert c["consistent"], c


@needs_quspin
@pytest.mark.parametrize("case", ["heis_spin", "heis_pauli", "heis_pm1", "tfim", "fermion", "boson",
                                  "spin1", "complex"])
def test_equals_quspin(case):
    from quspin.basis import spin_basis_1d, spinless_fermion_basis_1d, boson_basis_1d
    from quspin.operators import hamiltonian
    mk = {
        "heis_spin": (S.heisenberg_chain(7, J2=0.3, convention="spin"), lambda: spin_basis_1d(7, pauli=0)),
        "heis_pauli": (S.heisenberg_chain(7, convention="pauli"), lambda: spin_basis_1d(7, pauli=1)),
        "heis_pm1": (S.heisenberg_chain(7, convention="pauli_pm1"), lambda: spin_basis_1d(7, pauli=-1)),
        "tfim": (S.tfim_chain(8, h=0.7), lambda: spin_basis_1d(8, pauli=1)),
        "fermion": (S.spinless_fermion_chain(7, V=0.8, mu=0.2, boundary="periodic"),
                    lambda: spinless_fermion_basis_1d(7)),
        "boson": (S.bose_hubbard_chain(4, 3, U=2.5), lambda: boson_basis_1d(4, sps=4)),
        "spin1": (S.spin1_heisenberg_chain(5, D=0.4), lambda: spin_basis_1d(5, S="1")),
        "complex": (S.ModelSpec.from_static(5, LocalSpace.spin_space("1/2", "pauli"),
                                            [["xy", [[0.3, 0, 1], [0.3, 1, 2]]],
                                             ["yx", [[-0.3, 0, 1], [-0.3, 1, 2]]],
                                             ["y", [[0.5, 3]]]]), lambda: spin_basis_1d(5, pauli=1)),
    }
    spec, basis = mk[case]
    dtype = np.float64 if spec.is_real() else np.complex128
    Hq = hamiltonian(spec.to_quspin_static(), [], basis=basis(), dtype=dtype,
                     check_herm=False, check_symm=False, check_pcon=False).tocsr()
    d = matrix_difference(Hq, build_bitwise(spec))
    assert d["pattern_match"] and d["max_abs_diff"] <= 1e-15 * max(1.0, d["max_abs_entry"]) * 8
