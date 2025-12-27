"""
Tests for Phase 1: Hamiltonian Construction

Author: Sarang Vehale
"""

import numpy as np
import pytest
from certifyEd import CertifiedHamiltonian


class TestCertifiedHamiltonian:
    """Test CertifiedHamiltonian class"""

    def test_tfim_creation(self):
        """Test TFIM Hamiltonian creation"""
        H = CertifiedHamiltonian.TFIM(L=4, J=1.0, h=0.5)

        assert H.model_name == "TFIM"
        assert H.dimension == 2**4
        assert H.parameters == {"L": 4, "J": 1.0, "h": 0.5}
        assert H.H.shape == (16, 16)

    def test_hermiticity_verification(self):
        """Test Hermiticity verification (Phase 1)"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        assert H.verify_hermiticity() == True
        assert H.is_hermitian == True

    def test_non_hermitian_fails(self):
        """Test that non-Hermitian matrix fails verification"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        H.H[0, 1] += 1.0

        with pytest.raises(ValueError):
            H.verify_hermiticity()

    def test_heisenberg_creation(self):
        """Test Heisenberg model creation"""
        H = CertifiedHamiltonian.Heisenberg(L=3, J=1.0, Jz=1.0)

        assert H.model_name == "Heisenberg_XXZ"
        assert H.dimension == 2**3
        assert H.verify_hermiticity() == True

    def test_tensor_product_safe(self):
        """Test safe tensor product computation"""
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[0, 1], [1, 0]])

        result = CertifiedHamiltonian.tensor_product_safe(A, B)

        assert result.shape == (4, 4)
        assert result.dtype == complex
        assert np.allclose(result, np.kron(A, B))

    def test_export_specification(self):
        """Test specification export"""
        H = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)
        H.verify_hermiticity()

        spec = H.export_specification()

        assert "model" in spec
        assert "dimension" in spec
        assert "specification_hash" in spec
        assert spec["hermitian"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
