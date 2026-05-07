"""
Tests for Phase 5: Observables

Author: Sarang Vehale
"""

import numpy as np
import pytest
from certifyEd import (
    CertifiedHamiltonian,
    MultiOracleDiagonalizer,
    ObservableCalculator
)


class TestObservableCalculator:
    """Test ObservableCalculator class"""

    @pytest.fixture
    def setup_system(self):
        """Setup a simple 2-site TFIM for testing"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        diag = MultiOracleDiagonalizer(H)
        results = diag.diagonalize()
        return ObservableCalculator(results)

    def test_expectation_value(self, setup_system):
        """Test expectation value computation"""
        obs_calc = setup_system
        identity = np.eye(4, dtype=complex)
        result = obs_calc.expectation_value(identity)
        assert abs(result.value - 1.0) < 1e-10

    def test_observable_result(self, setup_system):
        """Test ObservableResult object"""
        obs_calc = setup_system
        identity = np.eye(4, dtype=complex)
        result = obs_calc.expectation_value(identity)

        assert hasattr(result, 'value')
        assert hasattr(result, 'error')
        assert hasattr(result, 'certified')
        assert isinstance(result.value, float)
        assert isinstance(result.error, float)
        assert isinstance(result.certified, bool)

    def test_summary(self, setup_system):
        """Test summary statistics"""
        obs_calc = setup_system
        summary = obs_calc.summary()

        assert "ground_state_energy" in summary
        assert "energy_error_bound" in summary
        assert "spectral_gap" in summary
        assert "agreement_validated" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
