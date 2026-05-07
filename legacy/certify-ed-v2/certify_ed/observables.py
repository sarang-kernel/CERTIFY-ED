"""
Physical Observables Module
==========================

Computes expectation values and correlations from certified eigendecompositions.
"""

import numpy as np
from typing import Optional


class ObservableCalculator:
    """Compute expectation values from eigendecomposition."""
    
    def __init__(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                 residuals: Optional[np.ndarray] = None):
        self.eigenvalues = np.asarray(eigenvalues)
        self.eigenvectors = np.asarray(eigenvectors)
        self.residuals = residuals
    
    def expectation_value(self, operator: np.ndarray, state_index: int = 0) -> float:
        """Compute <psi_n|O|psi_n>."""
        psi = self.eigenvectors[:, state_index]
        return float(np.vdot(psi, operator @ psi).real)
    
    def all_expectation_values(self, operator: np.ndarray) -> np.ndarray:
        """Compute <psi_n|O|psi_n> for all n."""
        return np.array([self.expectation_value(operator, i)
                          for i in range(len(self.eigenvalues))])
    
    def correlation(self, op_a: np.ndarray, op_b: np.ndarray,
                   state_index: int = 0) -> float:
        """Connected correlation <AB> - <A><B>."""
        psi = self.eigenvectors[:, state_index]
        ab = np.vdot(psi, op_a @ op_b @ psi).real
        a = np.vdot(psi, op_a @ psi).real
        b = np.vdot(psi, op_b @ psi).real
        return float(ab - a * b)
    
    def thermal_average(self, operator: np.ndarray, beta: float) -> float:
        """Thermal expectation <O> = sum_n exp(-beta E_n) <n|O|n> / Z."""
        E = self.eigenvalues - self.eigenvalues[0]  # shift for stability
        weights = np.exp(-beta * E)
        Z = np.sum(weights)
        
        avg = 0.0
        for n in range(len(self.eigenvalues)):
            avg += weights[n] * self.expectation_value(operator, n)
        return float(avg / Z)
