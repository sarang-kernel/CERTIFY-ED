"""
Multi-Oracle Eigendecomposition and Consensus Validation
=======================================================

This module provides multiple independent eigensolvers and consensus validation
to detect numerical errors and implementation bugs.

Classes:
    Oracle: Base class for eigendecomposition oracles
    NumPyOracle: Uses numpy.linalg.eigh (LAPACK)
    ScipyOracle: Uses scipy.linalg.eigh (different LAPACK driver)
    MultiOracle: Orchestrates consensus validation across oracles
"""

import numpy as np
from scipy import linalg
from typing import Tuple, List, Dict, Optional, Any
import warnings
from abc import ABC, abstractmethod


class Oracle(ABC):
    """
    Base class for eigendecomposition oracles.
    
    Each oracle represents an independent implementation of Hermitian
    eigendecomposition. Oracles are used in multi-oracle consensus validation
    to detect numerical errors.
    """
    
    @abstractmethod
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize Hermitian matrix.
        
        Args:
            H: Hermitian matrix (n x n complex array)
            
        Returns:
            eigenvalues: Real eigenvalues sorted in ascending order (n,)
            eigenvectors: Corresponding eigenvectors as columns (n x n)
            
        Raises:
            ValueError: If H is not Hermitian
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return oracle name for reporting."""
        pass


class NumPyOracle(Oracle):
    """
    NumPy-based oracle using numpy.linalg.eigh.
    
    This oracle uses NumPy's LAPACK wrapper for Hermitian eigendecomposition.
    The underlying LAPACK routine is typically DSYEVD (divide-and-conquer).
    
    Example:
        >>> oracle = NumPyOracle()
        >>> H = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> evals, evecs = oracle.diagonalize(H)
        >>> evals
        array([-1.,  1.])
    """
    
    def __init__(self):
        pass
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize using numpy.linalg.eigh.
        
        Args:
            H: Hermitian matrix
            
        Returns:
            eigenvalues: Sorted eigenvalues
            eigenvectors: Corresponding eigenvectors
            
        Raises:
            np.linalg.LinAlgError: If eigendecomposition fails
        """
        # Verify Hermiticity
        if not np.allclose(H, H.conj().T, rtol=1e-10, atol=1e-12):
            warnings.warn("Input matrix is not Hermitian within tolerance")
        
        # Diagonalize (returns sorted eigenvalues)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        return eigenvalues, eigenvectors
    
    def name(self) -> str:
        return "NumPy/LAPACK"


class ScipyOracle(Oracle):
    """
    SciPy-based oracle using scipy.linalg.eigh.
    
    This oracle uses SciPy's LAPACK wrapper, which may use different
    drivers or options compared to NumPy. Provides independent validation.
    
    Example:
        >>> oracle = ScipyOracle()
        >>> H = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> evals, evecs = oracle.diagonalize(H)
        >>> evals
        array([-1.,  1.])
    """
    
    def __init__(self, driver: str = 'evd'):
        """
        Initialize SciPy oracle.
        
        Args:
            driver: LAPACK driver to use ('evd', 'evr', or 'ev')
                   'evd': DSYEVD (divide-and-conquer, fastest)
                   'evr': DSYEVR (relatively robust, more accurate)
                   'ev':  DSYEV (classical QR, most stable)
        """
        valid_drivers = {'evd', 'evr', 'ev'}
        if driver not in valid_drivers:
            raise ValueError(f"driver must be one of {valid_drivers}")
        self.driver = driver
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize using scipy.linalg.eigh.
        
        Args:
            H: Hermitian matrix
            
        Returns:
            eigenvalues: Sorted eigenvalues
            eigenvectors: Corresponding eigenvectors
        """
        # Verify Hermiticity
        if not np.allclose(H, H.conj().T, rtol=1e-10, atol=1e-12):
            warnings.warn("Input matrix is not Hermitian within tolerance")
        
        # Diagonalize with specified driver
        eigenvalues, eigenvectors = linalg.eigh(H, driver=self.driver)
        
        return eigenvalues, eigenvectors
    
    def name(self) -> str:
        return f"SciPy/LAPACK/{self.driver.upper()}"


class MultiOracle:
    """
    Multi-oracle consensus validation.
    
    This class orchestrates eigendecomposition across multiple independent
    oracles and validates consensus. Disagreement indicates numerical issues
    or implementation bugs.
    
    Attributes:
        oracles: List of oracle instances
        tolerance: Tolerance for consensus agreement
        
    Example:
        >>> oracle = MultiOracle(tolerance=1e-10)
        >>> H = build_tfim(n_sites=4, J=1.0, h=0.5)
        >>> evals, evecs, consensus = oracle.diagonalize_with_consensus(H)
        >>> consensus['consensus']
        True
        >>> consensus['max_disagreement'] < 1e-10
        True
    """
    
    def __init__(
        self,
        oracles: Optional[List[Oracle]] = None,
        tolerance: float = 1e-10
    ):
        """
        Initialize multi-oracle validator.
        
        Args:
            oracles: List of oracle instances (default: NumPy + SciPy)
            tolerance: Maximum allowed disagreement for consensus
        """
        if oracles is None:
            oracles = [
                NumPyOracle(),
                ScipyOracle(driver='evd'),
                ScipyOracle(driver='evr'),
            ]
        
        if len(oracles) < 2:
            raise ValueError("At least 2 oracles required for consensus validation")
        
        self.oracles = oracles
        self.tolerance = tolerance
        
    def diagonalize_with_consensus(
        self,
        H: np.ndarray,
        return_all_results: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Diagonalize with multi-oracle consensus validation.
        
        Args:
            H: Hermitian matrix
            return_all_results: If True, include all oracle results in report
            
        Returns:
            eigenvalues: Consensus eigenvalues (from primary oracle)
            eigenvectors: Eigenvectors from primary oracle
            consensus_report: Dictionary with validation metrics
            
        The consensus_report contains:
            - consensus: bool, True if all oracles agree
            - max_disagreement: Maximum eigenvalue difference
            - oracle_names: Names of all oracles
            - disagreement_oracles: Indices of disagreeing oracles (if any)
            - tolerance: Tolerance used
            - eigenvalue_differences: Pairwise differences (if return_all_results=True)
            
        Example:
            >>> oracle = MultiOracle()
            >>> H = np.diag([1.0, 2.0, 3.0])
            >>> evals, evecs, report = oracle.diagonalize_with_consensus(H)
            >>> report['consensus']
            True
        """
        # Run all oracles
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        oracle_names: List[str] = []
        
        for oracle in self.oracles:
            eigenvalues, eigenvectors = oracle.diagonalize(H)
            results.append((eigenvalues, eigenvectors))
            oracle_names.append(oracle.name())
        
        # Primary result (from first oracle)
        eigenvalues_primary, eigenvectors_primary = results[0]
        
        # Compute pairwise eigenvalue differences
        n_oracles = len(self.oracles)
        n_eigenvalues = len(eigenvalues_primary)
        
        eigenvalue_diffs = np.zeros((n_oracles, n_oracles, n_eigenvalues))
        
        for i in range(n_oracles):
            for j in range(i + 1, n_oracles):
                diff = np.abs(results[i][0] - results[j][0])
                eigenvalue_diffs[i, j, :] = diff
                eigenvalue_diffs[j, i, :] = diff  # Symmetric
        
        # Find maximum disagreement
        max_disagreement = np.max(eigenvalue_diffs)
        
        # Check consensus
        consensus = max_disagreement < self.tolerance
        
        # Identify disagreeing oracles
        disagreement_oracles = []
        if not consensus:
            # Find which oracles disagree with primary
            for i in range(1, n_oracles):
                if np.max(eigenvalue_diffs[0, i, :]) >= self.tolerance:
                    disagreement_oracles.append(i)
        
        # Build consensus report
        consensus_report = {
            'consensus': consensus,
            'max_disagreement': float(max_disagreement),
            'oracle_names': oracle_names,
            'disagreement_oracles': disagreement_oracles,
            'tolerance': self.tolerance,
            'n_oracles': n_oracles,
        }
        
        if return_all_results:
            consensus_report['all_eigenvalues'] = [r[0] for r in results]
            consensus_report['eigenvalue_differences'] = eigenvalue_diffs
        
        # Warn if disagreement
        if not consensus:
            warnings.warn(
                f"Oracle disagreement detected!\n"
                f"Max disagreement: {max_disagreement:.2e} > tolerance {self.tolerance:.2e}\n"
                f"Disagreeing oracles: {[oracle_names[i] for i in disagreement_oracles]}\n"
                f"This may indicate:\n"
                f"  - Numerical instability in eigendecomposition\n"
                f"  - Near-degenerate eigenvalues\n"
                f"  - Implementation bugs in one oracle\n"
                f"Recommendation: Investigate eigenvalue spectrum and increase precision if needed."
            )
        
        return eigenvalues_primary, eigenvectors_primary, consensus_report
    
    def compute_residuals(
        self,
        H: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute residuals ||H|ψ_n⟩ - E_n|ψ_n⟩|| for each eigenpair.
        
        This provides independent validation of eigendecomposition quality
        beyond multi-oracle consensus.
        
        Args:
            H: Hamiltonian matrix
            eigenvalues: Eigenvalues
            eigenvectors: Eigenvectors (columns)
            
        Returns:
            residuals: Array of residual norms (one per eigenvalue)
            
        Example:
            >>> oracle = MultiOracle()
            >>> H = np.diag([1.0, 2.0, 3.0])
            >>> evals, evecs, _ = oracle.diagonalize_with_consensus(H)
            >>> residuals = oracle.compute_residuals(H, evals, evecs)
            >>> np.all(residuals < 1e-14)
            True
        """
        n_eigenvalues = len(eigenvalues)
        residuals = np.zeros(n_eigenvalues)
        
        for i in range(n_eigenvalues):
            E_n = eigenvalues[i]
            psi_n = eigenvectors[:, i]
            
            # Compute ||H|ψ⟩ - E|ψ⟩||
            residual_vector = H @ psi_n - E_n * psi_n
            residuals[i] = np.linalg.norm(residual_vector)
        
        return residuals
    
    def __repr__(self) -> str:
        oracle_names = ', '.join([o.name() for o in self.oracles])
        return (
            f"MultiOracle(n_oracles={len(self.oracles)}, "
            f"tolerance={self.tolerance:.2e}, "
            f"oracles=[{oracle_names}])"
        )
