"""Physical Observables Module."""

import numpy as np
from typing import Optional


class ObservableCalculator:
    """Compute expectation values from eigendecomposition."""
    
    def __init__(self, eigenvalues, eigenvectors, residuals=None):
        self.eigenvalues = np.asarray(eigenvalues)
        self.eigenvectors = np.asarray(eigenvectors)
        self.residuals = residuals
    
    def expectation_value(self, operator, state_index=0):
        psi = self.eigenvectors[:, state_index]
        return float(np.vdot(psi, operator @ psi).real)
    
    def all_expectation_values(self, operator):
        return np.array([self.expectation_value(operator, i)
                          for i in range(len(self.eigenvalues))])
    
    def correlation(self, op_a, op_b, state_index=0):
        psi = self.eigenvectors[:, state_index]
        ab = np.vdot(psi, op_a @ op_b @ psi).real
        a = np.vdot(psi, op_a @ psi).real
        b = np.vdot(psi, op_b @ psi).real
        return float(ab - a * b)
    
    def thermal_average(self, operator, beta):
        E = self.eigenvalues - self.eigenvalues[0]
        weights = np.exp(-beta * E)
        Z = np.sum(weights)
        avg = sum(weights[n] * self.expectation_value(operator, n)
                  for n in range(len(self.eigenvalues)))
        return float(avg / Z)
    
    def partition_function(self, beta):
        return float(np.sum(np.exp(-beta * self.eigenvalues)))
    
    def free_energy(self, beta):
        # F = -T * ln(Z), use shifted form for stability
        E0 = self.eigenvalues[0]
        Z_shifted = np.sum(np.exp(-beta * (self.eigenvalues - E0)))
        return float(E0 - np.log(Z_shifted) / beta)
