"""
Tests for Phases 2-3: Diagonalization

Author: Sarang Vehale
"""

import numpy as np
import pytest
from certifyEd import CertifiedHamiltonian, MultiOracleDiagonalizer


class TestMultiOracleDiagonalizer:
    """Test MultiOracleDiagonalizer class"""

    def test_diagonalization(self):
        """Test basic diagonalization"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()

        assert results.eigenvalues is not None
        assert results.eigenvectors is not None
        assert results.agreement_validated is not None

    def test_oracle_results(self):
        """Test that all 3 oracles ran"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()

        assert len(results.oracle_results) == 3
        oracle_names = [r.name for r in results.oracle_results]
        assert "NumPy" in oracle_names
        assert "HighPrecision" in oracle_names
        assert "Iterative" in oracle_names

    def test_consensus_computation(self):
        """Test consensus eigenvalue computation"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()

        assert np.all(np.diff(results.eigenvalues) >= 0)

    def test_agreement_validation(self):
        """Test oracle agreement validation"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H, epsilon=1e-12)
        results = diag.diagonalize()

        metrics = diag.get_agreement_metrics()
        assert metrics["validated"] == True

    def test_ground_state_energy(self):
        """Test ground state energy extraction"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()

        E0 = results.ground_state_energy()
        assert isinstance(E0, float)
        assert E0 < 0

    def test_spectral_gap(self):
        """Test spectral gap computation"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()

        gap = results.spectral_gap()
        assert isinstance(gap, float)
        assert gap > 0

    def test_residual_computation(self):
        """Test residual computation"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()

        residuals = results.compute_residuals()
        assert np.all(residuals > 0)
        assert np.max(residuals) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
