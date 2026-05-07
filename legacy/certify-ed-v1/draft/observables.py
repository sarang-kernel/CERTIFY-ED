"""
Observable Calculation and Error Propagation

Computes physical observables from certified eigenpairs with error bounds.
"""

import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ObservableCalculator:
    """Computes observables from certified eigenpairs."""

    def __init__(self):
        """Initialize observable calculator."""
        self.results = {}

    def ground_state_energy(
        self,
        H: np.ndarray,
        psi_0: np.ndarray,
        E_0: float,
        residual: float,
        gap: float
    ) -> Dict:
        """
        Compute ground state energy with error bound.

        Args:
            H: Hamiltonian matrix
            psi_0: Ground state eigenvector
            E_0: Computed ground state energy
            residual: Eigenvalue equation residual
            gap: Spectral gap (E_1 - E_0)

        Returns:
            Dictionary with energy and error bound
        """
        # Compute H |ψ₀⟩
        H_psi = H @ psi_0

        # Expectation value
        E_computed = np.real(psi_0.conj() @ H_psi)

        # Error bound from Davis-Kahan theorem
        H_norm = np.linalg.norm(H, ord=2)
        if gap > 1e-14:
            error_bound = 2 * H_norm * residual / gap
        else:
            error_bound = residual

        result = {
            'observable': 'E_0 (ground state energy)',
            'value': E_computed,
            'error_bound': error_bound,
            'residual': residual,
            'gap': gap
        }

        logger.info(f"Ground state energy: {E_computed:.12f} ± {error_bound:.2e}")

        return result

    def spectral_gap(
        self,
        E_0: float,
        E_1: float,
        residual_0: float,
        residual_1: float,
        H_norm: float
    ) -> Dict:
        """
        Compute spectral gap with error bound.

        Args:
            E_0: Ground state energy
            E_1: First excited state energy
            residual_0: Ground state residual
            residual_1: Excited state residual
            H_norm: Spectral norm of Hamiltonian

        Returns:
            Dictionary with gap and error bound
        """
        gap = E_1 - E_0

        # Error bound: add residual contributions
        error_bound = 2 * H_norm * (residual_0 + residual_1)

        result = {
            'observable': 'Δ (spectral gap)',
            'value': gap,
            'error_bound': error_bound,
            'lower_bound': max(0, gap - error_bound),
            'upper_bound': gap + error_bound
        }

        logger.info(f"Spectral gap: {gap:.12f} ± {error_bound:.2e}")

        return result

    def correlation_function(
        self,
        psi: np.ndarray,
        O_i: np.ndarray,
        O_j: np.ndarray,
        residual: float,
        H_norm: float,
        gap: float
    ) -> Dict:
        """
        Compute correlation function ⟨ψ|Oᵢ Oⱼ|ψ⟩ with error bound.

        Args:
            psi: Eigenstate
            O_i: First operator at site i
            O_j: Second operator at site j
            residual: Eigenvalue residual
            H_norm: Spectral norm
            gap: Energy gap

        Returns:
            Dictionary with correlation and error bound
        """
        # Compute ⟨ψ|Oᵢ Oⱼ|ψ⟩
        O_psi = O_j @ psi
        corr = np.real(psi.conj() @ O_i @ O_psi)

        # Error bound
        O_norm = max(np.linalg.norm(O_i, ord=2), np.linalg.norm(O_j, ord=2))
        if gap > 1e-14:
            error_bound = 2 * H_norm * O_norm * residual / gap
        else:
            error_bound = O_norm * residual

        result = {
            'observable': '⟨Oᵢ Oⱼ⟩ (correlation)',
            'value': corr,
            'error_bound': error_bound
        }

        logger.info(f"Correlation: {corr:.6f} ± {error_bound:.2e}")

        return result

    def magnetization(
        self,
        psi: np.ndarray,
        S_z: np.ndarray,
        residual: float,
        H_norm: float,
        gap: float
    ) -> Dict:
        """
        Compute magnetization ⟨ψ|Sᶻ|ψ⟩ with error bound.

        Args:
            psi: Eigenstate
            S_z: z-component spin operator
            residual: Eigenvalue residual
            H_norm: Spectral norm
            gap: Energy gap

        Returns:
            Dictionary with magnetization and error bound
        """
        # Compute ⟨Sᶻ⟩
        mag = np.real(psi.conj() @ S_z @ psi)

        # For conserved quantities, error is minimal
        S_norm = np.linalg.norm(S_z, ord=2)
        error_bound = S_norm * residual

        result = {
            'observable': '⟨Sᶻ⟩ (magnetization)',
            'value': mag,
            'error_bound': error_bound
        }

        logger.info(f"Magnetization: {mag:.6f} ± {error_bound:.2e}")

        return result

    def compute_all_standard(
        self,
        H: np.ndarray,
        eigenpairs,
        H_norm: float = None
    ) -> Dict:
        """
        Compute all standard observables.

        Args:
            H: Hamiltonian
            eigenpairs: List of (eigenvalue, eigenvector, residual) tuples
            H_norm: Spectral norm (computed if None)

        Returns:
            Dictionary of all computed observables
        """
        if H_norm is None:
            H_norm = np.linalg.norm(H, ord=2)

        results = {}

        if len(eigenpairs) >= 2:
            E_0, psi_0, res_0 = eigenpairs[0]
            E_1, psi_1, res_1 = eigenpairs[1]
            gap = E_1 - E_0

            results['ground_state'] = self.ground_state_energy(
                H, psi_0, E_0, res_0, gap
            )

            results['gap'] = self.spectral_gap(
                E_0, E_1, res_0, res_1, H_norm
            )

        logger.info(f"✓ Computed {len(results)} observables")

        return results


if __name__ == "__main__":
    # Test: Simple 2×2 system
    H_test = np.array([[1.0, 0.5], [0.5, -1.0]])
    evals, evecs = np.linalg.eigh(H_test)

    calc = ObservableCalculator()
    result = calc.ground_state_energy(H_test, evecs[:, 0], evals[0], 1e-14, evals[1] - evals[0])
    print(f"Ground state: {result['value']:.6f}")
