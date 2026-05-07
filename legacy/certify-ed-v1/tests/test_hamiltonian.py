"""
Tests for hamiltonian module
===========================

This test suite validates:
- Pauli matrix properties
- Tensor product correctness
- Hermiticity verification
- TFIM and Heisenberg model construction
"""

import numpy as np
import pytest
from certify_ed.hamiltonian import (
    pauli_matrices,
    tensor_product,
    SymbolicHamiltonian,
    build_tfim,
    build_heisenberg,
)


class TestPauliMatrices:
    """Test Pauli matrix properties."""
    
    def test_pauli_dimensions(self):
        """All Pauli matrices should be 2x2."""
        I, X, Y, Z = pauli_matrices()
        assert I.shape == (2, 2)
        assert X.shape == (2, 2)
        assert Y.shape == (2, 2)
        assert Z.shape == (2, 2)
    
    def test_pauli_commutations(self):
        """Test [X,Y] = 2iZ, etc."""
        I, X, Y, Z = pauli_matrices()
        
        # [X, Y] = 2iZ
        assert np.allclose(X @ Y - Y @ X, 2j * Z)
        
        # [Y, Z] = 2iX
        assert np.allclose(Y @ Z - Z @ Y, 2j * X)
        
        # [Z, X] = 2iY
        assert np.allclose(Z @ X - X @ Z, 2j * Y)
    
    def test_pauli_squares(self):
        """X² = Y² = Z² = I."""
        I, X, Y, Z = pauli_matrices()
        assert np.allclose(X @ X, I)
        assert np.allclose(Y @ Y, I)
        assert np.allclose(Z @ Z, I)
    
    def test_pauli_hermiticity(self):
        """All Pauli matrices are Hermitian."""
        I, X, Y, Z = pauli_matrices()
        assert np.allclose(I, I.conj().T)
        assert np.allclose(X, X.conj().T)
        assert np.allclose(Y, Y.conj().T)
        assert np.allclose(Z, Z.conj().T)


class TestTensorProduct:
    """Test tensor product functionality."""
    
    def test_tensor_dimensions(self):
        """Tensor product of 2x2 matrices gives 4x4."""
        I, X, Y, Z = pauli_matrices()
        ZZ = tensor_product(Z, Z)
        assert ZZ.shape == (4, 4)
    
    def test_three_site_tensor(self):
        """Three-site tensor product."""
        I, X, Y, Z = pauli_matrices()
        ZZZ = tensor_product(Z, Z, Z)
        assert ZZZ.shape == (8, 8)
    
    def test_identity_tensor(self):
        """I ⊗ I = I."""
        I, X, Y, Z = pauli_matrices()
        II = tensor_product(I, I)
        assert np.allclose(II, np.eye(4))


class TestSymbolicHamiltonian:
    """Test symbolic Hamiltonian construction."""
    
    def test_initialization(self):
        """Test Hamiltonian initialization."""
        ham = SymbolicHamiltonian(n_sites=3)
        assert ham.n_sites == 3
        assert ham.hilbert_dim == 8
        assert ham.matrix is None
    
    def test_invalid_sites(self):
        """Test invalid number of sites."""
        with pytest.raises(ValueError):
            SymbolicHamiltonian(n_sites=0)
        
        with pytest.raises(ValueError):
            SymbolicHamiltonian(n_sites=-1)
    
    def test_add_term(self):
        """Test adding terms."""
        ham = SymbolicHamiltonian(n_sites=2)
        ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
        assert len(ham.terms) == 1
    
    def test_invalid_operator(self):
        """Test invalid operator name."""
        ham = SymbolicHamiltonian(n_sites=2)
        with pytest.raises(ValueError):
            ham.add_term(1.0, [('W', 0)])  # Invalid operator
    
    def test_invalid_site_index(self):
        """Test invalid site index."""
        ham = SymbolicHamiltonian(n_sites=2)
        with pytest.raises(ValueError):
            ham.add_term(1.0, [('Z', 5)])  # Out of range
    
    def test_build_matrix(self):
        """Test matrix construction."""
        ham = SymbolicHamiltonian(n_sites=2)
        ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
        H = ham.build()
        assert H.shape == (4, 4)
        assert ham.matrix is not None
    
    def test_hermiticity_verification(self):
        """Test Hermiticity verification."""
        ham = SymbolicHamiltonian(n_sites=2)
        ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
        ham.build()
        assert ham.verify_hermiticity() is True


class TestTFIM:
    """Test transverse-field Ising model."""
    
    def test_tfim_dimensions(self):
        """Test TFIM matrix dimensions."""
        H = build_tfim(n_sites=3, J=1.0, h=0.5)
        assert H.shape == (8, 8)
    
    def test_tfim_hermiticity(self):
        """TFIM should be Hermitian."""
        H = build_tfim(n_sites=4, J=1.0, h=0.5)
        assert np.allclose(H, H.conj().T)
    
    def test_tfim_two_site_exact(self):
        """Test 2-site TFIM against known exact result."""
        # For 2 sites with J=1, h=0: eigenvalues are {-1, -1, 1, 1}
        H = build_tfim(n_sites=2, J=1.0, h=0.0)
        evals = np.linalg.eigvalsh(H)
        expected = np.array([-1.0, -1.0, 1.0, 1.0])
        assert np.allclose(sorted(evals), sorted(expected))
    
    def test_tfim_periodic_vs_open(self):
        """Periodic and open boundary conditions should differ."""
        H_open = build_tfim(n_sites=3, J=1.0, h=0.5, boundary='open')
        H_periodic = build_tfim(n_sites=3, J=1.0, h=0.5, boundary='periodic')
        
        # Matrices should be different
        assert not np.allclose(H_open, H_periodic)


class TestHeisenberg:
    """Test Heisenberg model."""
    
    def test_heisenberg_dimensions(self):
        """Test Heisenberg matrix dimensions."""
        H = build_heisenberg(n_sites=3, J=1.0)
        assert H.shape == (8, 8)
    
    def test_heisenberg_hermiticity(self):
        """Heisenberg should be Hermitian."""
        H = build_heisenberg(n_sites=4, J=1.0)
        assert np.allclose(H, H.conj().T)
    
    def test_heisenberg_two_site_exact(self):
        """Test 2-site Heisenberg against known result."""
        # For 2 sites: ground state is singlet with E = -3J/4
        H = build_heisenberg(n_sites=2, J=1.0)
        evals = np.linalg.eigvalsh(H)
        
        # Singlet energy: -3/4
        # Triplet energies: 1/4, 1/4, 1/4
        expected = np.array([-0.75, 0.25, 0.25, 0.25])
        assert np.allclose(sorted(evals), sorted(expected), atol=1e-10)
    
    def test_heisenberg_anisotropy(self):
        """Test anisotropic Heisenberg (XXZ model)."""
        H_xxx = build_heisenberg(n_sites=2, J=1.0)
        H_xxz = build_heisenberg(
            n_sites=2, 
            J=1.0, 
            anisotropy={'Jx': 1.0, 'Jy': 1.0, 'Jz': 2.0}
        )
        
        # XXX and XXZ should differ
        assert not np.allclose(H_xxx, H_xxz)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
