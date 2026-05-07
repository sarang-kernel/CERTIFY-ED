"""Tests for hamiltonian module."""
import numpy as np
import pytest
from certify_ed import build_model, list_models


def test_pauli_matrices_hermitian():
    from certify_ed import pauli_matrices
    I, X, Y, Z = pauli_matrices()
    for op in (I, X, Y, Z):
        assert np.allclose(op, op.conj().T)


def test_pauli_anticommutation():
    from certify_ed import pauli_matrices
    _, X, Y, Z = pauli_matrices()
    # {X, Y} = 0, {Y, Z} = 0, {Z, X} = 0
    assert np.allclose(X @ Y + Y @ X, 0, atol=1e-15)
    assert np.allclose(Y @ Z + Z @ Y, 0, atol=1e-15)
    assert np.allclose(Z @ X + X @ Z, 0, atol=1e-15)


@pytest.mark.parametrize("model_name,kwargs", [
    ('tfim', {'n_sites': 4}),
    ('heisenberg', {'n_sites': 4}),
    ('xxz', {'n_sites': 4, 'Delta': 0.5}),
    ('xyz', {'n_sites': 4}),
    ('ssh', {'n_sites': 4}),
    ('j1j2', {'n_sites': 4}),
    ('majumdar_ghosh', {'n_sites': 4}),
    ('cluster', {'n_sites': 4}),
    ('free_fermion', {'n_sites': 4}),
    ('kitaev_chain', {'n_sites': 4}),
    ('hubbard', {'n_sites': 3}),
    ('aklt', {'n_sites': 3}),
    ('haldane', {'n_sites': 3}),
    ('tfim_2d', {'Lx': 2, 'Ly': 2}),
    ('heisenberg_2d', {'Lx': 2, 'Ly': 2}),
    ('kitaev_honeycomb', {'Lx': 2, 'Ly': 2}),
])
def test_model_is_hermitian(model_name, kwargs):
    """Every model in the registry must produce a Hermitian matrix."""
    H = build_model(model_name, **kwargs)
    assert np.allclose(H, H.conj().T, atol=1e-12), f"{model_name} not Hermitian"


def test_list_models_complete():
    models = list_models()
    expected = {
        'tfim', 'heisenberg', 'xxz', 'xyz', 'ssh', 'j1j2',
        'majumdar_ghosh', 'cluster', 'free_fermion', 'kitaev_chain',
        'hubbard', 'aklt', 'haldane',
        'tfim_2d', 'heisenberg_2d', 'kitaev_honeycomb',
    }
    assert set(models) == expected


def test_tfim_classical_limit_ground_state():
    """TFIM h=0: E_0 = -J(N-1) for OBC."""
    N = 4
    H = build_model('tfim', n_sites=N, J=1.0, h=0.0, boundary='open')
    evals = np.linalg.eigvalsh(H)
    assert abs(evals[0] - (-(N - 1))) < 1e-12


def test_2site_heisenberg_singlet():
    """2-site Heisenberg ground state is singlet at -3J/4."""
    H = build_model('heisenberg', n_sites=2, J=1.0)
    evals = np.linalg.eigvalsh(H)
    assert abs(evals[0] - (-0.75)) < 1e-12


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        build_model('nonexistent_model', n_sites=4)
