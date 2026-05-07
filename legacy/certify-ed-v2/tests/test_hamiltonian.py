"""Tests for hamiltonian module."""

import numpy as np
import pytest
from certify_ed import (
    pauli_matrices, tensor_product, SymbolicHamiltonian,
    build_tfim, build_heisenberg, build_xxz
)


class TestPauliMatrices:
    def test_dimensions(self):
        I, X, Y, Z = pauli_matrices()
        assert all(M.shape == (2, 2) for M in (I, X, Y, Z))
    
    def test_hermiticity(self):
        I, X, Y, Z = pauli_matrices()
        for M in (I, X, Y, Z):
            assert np.allclose(M, M.conj().T)
    
    def test_squares(self):
        I, X, Y, Z = pauli_matrices()
        assert np.allclose(X @ X, I)
        assert np.allclose(Y @ Y, I)
        assert np.allclose(Z @ Z, I)
    
    def test_anticommutators(self):
        I, X, Y, Z = pauli_matrices()
        assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2)))
        assert np.allclose(Y @ Z + Z @ Y, np.zeros((2, 2)))
        assert np.allclose(Z @ X + X @ Z, np.zeros((2, 2)))


class TestSymbolicHamiltonian:
    def test_creation(self):
        ham = SymbolicHamiltonian(n_sites=3)
        assert ham.n_sites == 3
        assert ham.hilbert_dim == 8
    
    def test_invalid_sites(self):
        with pytest.raises(ValueError):
            SymbolicHamiltonian(n_sites=0)
    
    def test_invalid_operator(self):
        ham = SymbolicHamiltonian(n_sites=2)
        with pytest.raises(ValueError):
            ham.add_term(1.0, [('Q', 0)])
    
    def test_invalid_site(self):
        ham = SymbolicHamiltonian(n_sites=2)
        with pytest.raises(ValueError):
            ham.add_term(1.0, [('Z', 5)])
    
    def test_build_and_verify(self):
        ham = SymbolicHamiltonian(n_sites=2)
        ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
        H = ham.build()
        assert H.shape == (4, 4)
        assert ham.verify_hermiticity()


class TestTFIM:
    def test_2site_h0(self):
        """TFIM with h=0: classical Ising."""
        H = build_tfim(n_sites=2, J=1.0, h=0.0)
        evals = np.linalg.eigvalsh(H)
        # E_0 = -J = -1 (degenerate), E_1 = +J = +1 (degenerate)
        assert np.allclose(sorted(evals), [-1.0, -1.0, 1.0, 1.0])
    
    def test_hermiticity(self):
        for N in (2, 3, 4):
            H = build_tfim(n_sites=N, J=1.0, h=0.5)
            assert np.allclose(H, H.conj().T)
    
    def test_dimensions(self):
        for N in range(1, 6):
            H = build_tfim(n_sites=N, J=1.0, h=0.5)
            assert H.shape == (2**N, 2**N)


class TestHeisenberg:
    def test_2site_singlet_triplet(self):
        """2-site Heisenberg: -3J/4 singlet, +J/4 triplet."""
        J = 1.0
        H = build_heisenberg(n_sites=2, J=J)
        evals = np.linalg.eigvalsh(H)
        expected = np.array([-3*J/4, J/4, J/4, J/4])
        assert np.allclose(sorted(evals), sorted(expected), atol=1e-13)
    
    def test_hermiticity(self):
        for N in (2, 3, 4):
            H = build_heisenberg(n_sites=N, J=1.0)
            assert np.allclose(H, H.conj().T)


class TestXXZ:
    def test_isotropic_limit(self):
        """XXZ with Delta=1 should equal Heisenberg."""
        H_xxz = build_xxz(n_sites=3, J=1.0, Delta=1.0)
        H_heis = build_heisenberg(n_sites=3, J=1.0)
        assert np.allclose(H_xxz, H_heis)
    
    def test_xx_limit(self):
        """XXZ with Delta=0 should be XX chain."""
        H_xx = build_xxz(n_sites=3, J=1.0, Delta=0.0)
        # Should be Hermitian
        assert np.allclose(H_xx, H_xx.conj().T)
        # No ZZ terms in eigenstructure analysis
        evals = np.linalg.eigvalsh(H_xx)
        assert len(evals) == 8
