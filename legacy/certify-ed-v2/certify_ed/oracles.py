"""
Multi-Oracle Eigendecomposition Module
=====================================

Provides multiple independent eigensolvers for consensus validation.
"""

import numpy as np
from scipy import linalg as sla
from typing import Tuple, List, Dict, Any, Optional
from abc import ABC, abstractmethod
import warnings


class Oracle(ABC):
    """Base class for eigendecomposition oracles."""
    
    @abstractmethod
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize Hermitian matrix H.
        
        Returns
        -------
        eigenvalues : np.ndarray
            Sorted eigenvalues (ascending).
        eigenvectors : np.ndarray
            Corresponding eigenvectors as columns.
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Oracle identifier."""
        pass


class NumPyOracle(Oracle):
    """NumPy/LAPACK eigendecomposition (DSYEVD via numpy.linalg.eigh)."""
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not np.allclose(H, H.conj().T, atol=1e-12):
            warnings.warn("Input matrix not Hermitian within 1e-12")
        return np.linalg.eigh(H)
    
    def name(self) -> str:
        return "NumPy/LAPACK"


class ScipyOracle(Oracle):
    """SciPy/LAPACK eigendecomposition with selectable driver."""
    
    VALID_DRIVERS = {'evd', 'evr', 'ev'}
    
    def __init__(self, driver: str = 'evd'):
        if driver not in self.VALID_DRIVERS:
            raise ValueError(f"driver must be in {self.VALID_DRIVERS}")
        self.driver = driver
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not np.allclose(H, H.conj().T, atol=1e-12):
            warnings.warn("Input matrix not Hermitian within 1e-12")
        return sla.eigh(H, driver=self.driver)
    
    def name(self) -> str:
        return f"SciPy/{self.driver.upper()}"


class MultiOracle:
    """
    Multi-oracle consensus validation.
    
    Runs multiple independent eigensolvers and validates consensus.
    Default oracles: NumPy/DSYEVD, SciPy/EVD, SciPy/EVR.
    """
    
    def __init__(self, oracles: Optional[List[Oracle]] = None,
                 tolerance: float = 1e-10):
        if oracles is None:
            oracles = [
                NumPyOracle(),
                ScipyOracle(driver='evd'),
                ScipyOracle(driver='evr'),
            ]
        if len(oracles) < 2:
            raise ValueError("Need >= 2 oracles for consensus")
        
        self.oracles = oracles
        self.tolerance = tolerance
    
    def diagonalize_with_consensus(self, H: np.ndarray,
                                   return_all: bool = False
                                   ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run all oracles and validate consensus.
        
        Parameters
        ----------
        H : np.ndarray
            Hermitian matrix.
        return_all : bool
            If True, include all oracle results in report.
        
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues from primary oracle.
        eigenvectors : np.ndarray
            Eigenvectors from primary oracle.
        consensus_report : dict
            Validation report with consensus status and metrics.
        """
        results = []
        names = []
        
        for oracle in self.oracles:
            evals, evecs = oracle.diagonalize(H)
            results.append((evals, evecs))
            names.append(oracle.name())
        
        evals_primary, evecs_primary = results[0]
        n_oracles = len(self.oracles)
        n_evals = len(evals_primary)
        
        # Pairwise eigenvalue differences
        max_diff_pair = 0.0
        max_diff_eigenvalue = 0.0
        diff_matrix = np.zeros((n_oracles, n_oracles))
        
        for i in range(n_oracles):
            for j in range(i + 1, n_oracles):
                diff = np.max(np.abs(results[i][0] - results[j][0]))
                diff_matrix[i, j] = diff
                diff_matrix[j, i] = diff
                if diff > max_diff_pair:
                    max_diff_pair = diff
        
        consensus = max_diff_pair < self.tolerance
        
        report = {
            'consensus': consensus,
            'max_disagreement': float(max_diff_pair),
            'oracle_names': names,
            'n_oracles': n_oracles,
            'tolerance': self.tolerance,
            'pairwise_max_diffs': diff_matrix.tolist(),
        }
        
        if return_all:
            report['all_eigenvalues'] = [r[0].tolist() for r in results]
        
        if not consensus:
            warnings.warn(
                f"Oracle disagreement: max diff = {max_diff_pair:.2e} "
                f"> tolerance {self.tolerance:.2e}"
            )
        
        return evals_primary, evecs_primary, report
    
    def compute_residuals(self, H: np.ndarray, evals: np.ndarray,
                          evecs: np.ndarray) -> np.ndarray:
        """
        Compute residuals ||H|psi_n> - E_n|psi_n>|| for each eigenpair.
        
        Returns
        -------
        np.ndarray
            Array of residual norms.
        """
        n = len(evals)
        residuals = np.zeros(n)
        for i in range(n):
            psi = evecs[:, i]
            residuals[i] = np.linalg.norm(H @ psi - evals[i] * psi)
        return residuals
    
    def __repr__(self) -> str:
        names = [o.name() for o in self.oracles]
        return f"MultiOracle(oracles={names}, tol={self.tolerance:.0e})"
