"""
Multi-Oracle Eigendecomposition Module
======================================

Independent eigensolvers for consensus validation. Each oracle uses a
different code path through LAPACK/ARPACK to detect both implementation
bugs and algorithm-specific numerical instabilities.

Oracles provided:
    - NumPyOracle: numpy.linalg.eigh (LAPACK DSYEVD wrapper)
    - ScipyOracle: scipy.linalg.eigh with selectable driver (DSYEVD/DSYEVR/DSYEV)
    - SparseOracle: scipy.sparse.linalg.eigsh (ARPACK Lanczos, partial spectrum)
"""

import numpy as np
from scipy import linalg as sla
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
import warnings


class Oracle(ABC):
    """Base class for eigendecomposition oracles."""
    
    @abstractmethod
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize Hermitian H. Returns (eigenvalues, eigenvectors)."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Oracle identifier."""
        pass


class NumPyOracle(Oracle):
    """numpy.linalg.eigh - LAPACK DSYEVD via NumPy wrapper."""
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not np.allclose(H, H.conj().T, atol=1e-12):
            warnings.warn("Input not Hermitian within 1e-12")
        return np.linalg.eigh(H)
    
    def name(self) -> str:
        return "NumPy/LAPACK_DSYEVD"


class ScipyOracle(Oracle):
    """scipy.linalg.eigh with selectable LAPACK driver."""
    
    VALID_DRIVERS = {'evd', 'evr', 'ev'}
    
    def __init__(self, driver: str = 'evd'):
        if driver not in self.VALID_DRIVERS:
            raise ValueError(f"driver must be in {self.VALID_DRIVERS}")
        self.driver = driver
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not np.allclose(H, H.conj().T, atol=1e-12):
            warnings.warn("Input not Hermitian within 1e-12")
        return sla.eigh(H, driver=self.driver)
    
    def name(self) -> str:
        return f"SciPy/LAPACK_{self.driver.upper()}"


class SparseOracle(Oracle):
    """
    scipy.sparse.linalg.eigsh - ARPACK Lanczos.
    
    Returns only k lowest eigenvalues (default: all if d <= 50, else k=10).
    """
    
    def __init__(self, k: Optional[int] = None, max_dim_full: int = 50):
        self.k = k
        self.max_dim_full = max_dim_full
    
    def diagonalize(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = H.shape[0]
        # ARPACK can compute at most d-1 eigenvalues
        if self.k is None:
            k_use = min(d - 1, self.max_dim_full) if d <= self.max_dim_full else 10
        else:
            k_use = min(self.k, d - 1)
        
        if k_use < 1:
            # Fall back to dense for tiny matrices
            return np.linalg.eigh(H)
        
        H_sparse = csr_matrix(H)
        # Use 'SA' = smallest algebraic for Hermitian operators
        evals, evecs = eigsh(H_sparse, k=k_use, which='SA')
        # Sort
        order = np.argsort(evals)
        return evals[order], evecs[:, order]
    
    def name(self) -> str:
        return "SciPy/ARPACK_Lanczos"


class MultiOracle:
    """
    Multi-oracle consensus validation.
    
    Default oracles:
        1. NumPy/LAPACK_DSYEVD (divide-and-conquer)
        2. SciPy/LAPACK_DSYEVD (different wrapper, same algorithm)
        3. SciPy/LAPACK_DSYEVR (different algorithm: relatively robust)
    
    Optional sparse oracle for cross-checking lowest eigenvalues only.
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
    
    def diagonalize_with_consensus(self, H: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Run all dense oracles, validate consensus."""
        # Filter sparse oracles - they get separate handling
        dense_oracles = [o for o in self.oracles if not isinstance(o, SparseOracle)]
        sparse_oracles = [o for o in self.oracles if isinstance(o, SparseOracle)]
        
        if not dense_oracles:
            raise ValueError("Need at least one dense oracle")
        
        results = []
        names = []
        for oracle in dense_oracles:
            evals, evecs = oracle.diagonalize(H)
            results.append((evals, evecs))
            names.append(oracle.name())
        
        # Consensus over dense oracles (full spectra)
        max_diff = 0.0
        n_oracles = len(dense_oracles)
        diff_matrix = np.zeros((n_oracles, n_oracles))
        for i in range(n_oracles):
            for j in range(i + 1, n_oracles):
                diff = np.max(np.abs(results[i][0] - results[j][0]))
                diff_matrix[i, j] = diff_matrix[j, i] = diff
                max_diff = max(max_diff, diff)
        
        consensus = max_diff < self.tolerance
        
        # Sparse oracle cross-check (lowest eigenvalues only)
        sparse_check = None
        if sparse_oracles:
            try:
                evals_sparse, _ = sparse_oracles[0].diagonalize(H)
                k = len(evals_sparse)
                evals_dense_low = results[0][0][:k]
                sparse_diff = np.max(np.abs(evals_sparse - evals_dense_low))
                sparse_check = {
                    'oracle': sparse_oracles[0].name(),
                    'k_eigenvalues': k,
                    'max_diff_lowest_k': float(sparse_diff),
                    'agrees': bool(sparse_diff < self.tolerance * 100)  # sparse is less precise
                }
            except Exception as e:
                sparse_check = {'oracle': sparse_oracles[0].name(), 'error': str(e)}
        
        report = {
            'consensus': consensus,
            'max_disagreement': float(max_diff),
            'oracle_names': names,
            'n_oracles': n_oracles,
            'tolerance': self.tolerance,
            'pairwise_max_diffs': diff_matrix.tolist(),
        }
        if sparse_check is not None:
            report['sparse_cross_check'] = sparse_check
        
        if not consensus:
            warnings.warn(
                f"Oracle disagreement: {max_diff:.2e} > tol {self.tolerance:.2e}"
            )
        
        return results[0][0], results[0][1], report
    
    def compute_residuals(self, H: np.ndarray, evals: np.ndarray,
                          evecs: np.ndarray) -> np.ndarray:
        """Compute ||H|psi_n> - E_n|psi_n>|| for each eigenpair."""
        n = len(evals)
        residuals = np.zeros(n)
        for i in range(n):
            psi = evecs[:, i]
            residuals[i] = np.linalg.norm(H @ psi - evals[i] * psi)
        return residuals
    
    def __repr__(self) -> str:
        return f"MultiOracle({len(self.oracles)} oracles, tol={self.tolerance:.0e})"
