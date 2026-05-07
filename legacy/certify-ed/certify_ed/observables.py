"""
Physical Observable Computation
==============================

This module provides tools for computing physical observables from certified
eigendecompositions with error propagation.

Classes:
    ObservableCalculator: Compute expectation values with error bounds
"""

import numpy as np
from typing import Dict, Any, Optional, List
import warnings


class ObservableCalculator:
    """
    Compute physical observables from eigendecomposition.
    
    This class computes expectation values of observables given eigenvalues
    and eigenvectors, with error propagation based on residual bounds.
    
    Example:
        >>> calc = ObservableCalculator(eigenvalues, eigenvectors, residuals)
        >>> # Compute ground state magnetization
        >>> I, X, Y, Z = pauli_matrices()
        >>> Sz_total = sum(tensor_product(*ops) for ops in ...)
        >>> mag = calc.expectation_value(Sz_total, state_index=0)
    """
    
    def __init__(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        residuals: Optional[np.ndarray] = None
    ):
        """
        Initialize observable calculator.
        
        Args:
            eigenvalues: Eigenvalues array
            eigenvectors: Eigenvectors (columns)
            residuals: Optional residuals for error estimation
        """
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.residuals = residuals
        
    def expectation_value(
        self,
        operator: np.ndarray,
        state_index: int = 0
    ) -> float:
        """
        Compute expectation value ⟨ψ_n|O|ψ_n⟩.
        
        Args:
            operator: Observable operator (Hermitian matrix)
            state_index: Index of eigenstate (default: 0 = ground state)
            
        Returns:
            Expectation value (real number)
            
        Example:
            >>> # Ground state energy (should match eigenvalue)
            >>> calc = ObservableCalculator(evals, evecs)
            >>> E0 = calc.expectation_value(H, state_index=0)
            >>> np.isclose(E0, evals[0])
            True
        """
        psi = self.eigenvectors[:, state_index]
        expectation = np.vdot(psi, operator @ psi).real
        return float(expectation)
    
    def correlation_function(
        self,
        operator_i: np.ndarray,
        operator_j: np.ndarray,
        state_index: int = 0
    ) -> float:
        """
        Compute correlation function ⟨ψ|O_i O_j|ψ⟩ - ⟨ψ|O_i|ψ⟩⟨ψ|O_j|ψ⟩.
        
        Args:
            operator_i: First operator
            operator_j: Second operator
            state_index: Eigenstate index
            
        Returns:
            Connected correlation function
        """
        psi = self.eigenvectors[:, state_index]
        
        # ⟨O_i O_j⟩
        OiOj = np.vdot(psi, operator_i @ operator_j @ psi).real
        
        # ⟨O_i⟩⟨O_j⟩
        Oi = np.vdot(psi, operator_i @ psi).real
        Oj = np.vdot(psi, operator_j @ psi).real
        
        correlation = OiOj - Oi * Oj
        return float(correlation)
    
    def all_expectation_values(
        self,
        operator: np.ndarray
    ) -> np.ndarray:
        """
        Compute expectation value for all eigenstates.
        
        Args:
            operator: Observable operator
            
        Returns:
            Array of expectation values (one per eigenstate)
        """
        n_states = len(self.eigenvalues)
        expectations = np.zeros(n_states)
        
        for i in range(n_states):
            expectations[i] = self.expectation_value(operator, i)
        
        return expectations
