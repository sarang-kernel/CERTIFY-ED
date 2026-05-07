"""Tests for symmetries module."""
import numpy as np
import pytest
from certify_ed import (
    build_model,
    total_sz_operator, parity_operator, fermion_number_operator,
    fermion_parity_operator, check_conservation, commutator_norm,
)


def test_sz_eigenvalues():
    """Total Sz on N=4 should have eigenvalues -2, -1, 0, 1, 2 (with multiplicities)."""
    Sz = total_sz_operator(4)
    evals = np.linalg.eigvalsh(Sz)
    # Expected eigenvalues with multiplicities: -2 (1), -1 (4), 0 (6), 1 (4), 2 (1)
    expected = sorted([-2.0]*1 + [-1.0]*4 + [0.0]*6 + [1.0]*4 + [2.0]*1)
    assert np.allclose(np.sort(evals), expected)


def test_parity_squared_is_identity():
    """P^2 = I for both X-parity and Z-parity."""
    P = parity_operator(4)
    assert np.allclose(P @ P, np.eye(P.shape[0]))


def test_heisenberg_conserves_sz():
    """[H_Heisenberg, S_z] = 0."""
    H = build_model('heisenberg', n_sites=4)
    Sz = total_sz_operator(4)
    result = check_conservation(H, Sz)
    assert result['is_conserved']
    assert result['commutator_norm'] < 1e-12


def test_tfim_conserves_parity():
    """TFIM conserves spin-flip parity P = prod X."""
    H = build_model('tfim', n_sites=4)
    P = parity_operator(4)
    assert commutator_norm(H, P) < 1e-12


def test_tfim_does_not_conserve_sz():
    """TFIM does NOT conserve total Sz (transverse field breaks U(1))."""
    H = build_model('tfim', n_sites=4)
    Sz = total_sz_operator(4)
    assert commutator_norm(H, Sz) > 0.1


def test_free_fermion_conserves_number():
    """Free fermion chain conserves total particle number."""
    H = build_model('free_fermion', n_sites=4)
    N_op = fermion_number_operator(4)
    assert commutator_norm(H, N_op) < 1e-12


def test_kitaev_conserves_fermion_parity():
    """Kitaev chain conserves fermion parity (but not number)."""
    H = build_model('kitaev_chain', n_sites=4, t=1.0, mu=0.5, Delta=0.7)
    P_F = fermion_parity_operator(4)
    assert commutator_norm(H, P_F) < 1e-12
    # Check it does NOT conserve number
    N_op = fermion_number_operator(4)
    assert commutator_norm(H, N_op) > 0.1
