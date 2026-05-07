"""
Tests for oracles module
=======================

Test multi-oracle consensus validation.
"""

import numpy as np
import pytest
from certify_ed.oracles import (
    NumPyOracle,
    ScipyOracle,
    MultiOracle,
)
from certify_ed.hamiltonian import build_tfim


class TestNumPyOracle:
    """Test NumPy oracle."""
    
    def test_diagonalize_identity(self):
        """Diagonalize identity matrix."""
        oracle = NumPyOracle()
        H = np.eye(4, dtype=complex)
        evals, evecs = oracle.diagonalize(H)
        
        assert np.allclose(evals, np.ones(4))
    
    def test_diagonalize_diagonal(self):
        """Diagonalize diagonal matrix."""
        oracle = NumPyOracle()
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        evals, evecs = oracle.diagonalize(H)
        
        assert np.allclose(evals, [1.0, 2.0, 3.0, 4.0])


class TestScipyOracle:
    """Test SciPy oracle."""
    
    def test_diagonalize_identity(self):
        """Diagonalize identity matrix."""
        oracle = ScipyOracle()
        H = np.eye(4, dtype=complex)
        evals, evecs = oracle.diagonalize(H)
        
        assert np.allclose(evals, np.ones(4))
    
    def test_different_drivers(self):
        """Test different LAPACK drivers."""
        H = np.diag([1.0, 2.0, 3.0]).astype(complex)
        
        for driver in ['evd', 'evr', 'ev']:
            oracle = ScipyOracle(driver=driver)
            evals, evecs = oracle.diagonalize(H)
            assert np.allclose(evals, [1.0, 2.0, 3.0])


class TestMultiOracle:
    """Test multi-oracle consensus."""
    
    def test_consensus_on_simple_matrix(self):
        """Consensus should be achieved for simple matrices."""
        oracle = MultiOracle()
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        
        evals, evecs, report = oracle.diagonalize_with_consensus(H)
        
        assert report['consensus'] is True
        assert report['max_disagreement'] < 1e-12
    
    def test_consensus_on_tfim(self):
        """Test consensus on physical Hamiltonian."""
        oracle = MultiOracle()
        H = build_tfim(n_sites=4, J=1.0, h=0.5)
        
        evals, evecs, report = oracle.diagonalize_with_consensus(H)
        
        assert report['consensus'] is True
        assert len(evals) == 16
    
    def test_residuals_computation(self):
        """Test residual computation."""
        oracle = MultiOracle()
        H = build_tfim(n_sites=3, J=1.0, h=0.5)
        
        evals, evecs, _ = oracle.diagonalize_with_consensus(H)
        residuals = oracle.compute_residuals(H, evals, evecs)
        
        # All residuals should be very small
        assert np.all(residuals < 1e-10)
    
    def test_insufficient_oracles(self):
        """Test error with too few oracles."""
        with pytest.raises(ValueError):
            MultiOracle(oracles=[NumPyOracle()])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
