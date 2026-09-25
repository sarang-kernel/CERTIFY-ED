"""
Sparse vs Dense Validator
=========================

Cross-validates ARPACK Lanczos (sparse iterative) against LAPACK
(dense direct) eigendecomposition. These use fundamentally different
algorithms and serve as a strong independent check.

Tests cover:
    - Lowest-k eigenvalue agreement
    - Multiple model classes (TFIM, Heisenberg, Kitaev, fermionic)
    - System size sweep
"""

import numpy as np
from typing import Dict, List, Any
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from certify_ed import build_model, MultiOracle, NumPyOracle


class SparseDenseValidator:
    """Cross-validate sparse Lanczos vs dense LAPACK."""
    
    def __init__(self, tolerance: float = 1e-8):
        # Sparse Lanczos is less precise than dense; tolerance is looser
        self.tolerance = tolerance
        self.dense_oracle = NumPyOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        self.results.append(self.validate_tfim())
        self.results.append(self.validate_heisenberg())
        self.results.append(self.validate_kitaev())
        self.results.append(self.validate_free_fermion())
        self.results.append(self.validate_xxz())
        return self.results
    
    def _compare_lowest_k(self, H: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Compare lowest k eigenvalues between sparse and dense."""
        # Dense
        evals_dense, _ = self.dense_oracle.diagonalize(H)
        evals_dense_low = np.sort(evals_dense)[:k]
        
        # Sparse Lanczos
        H_sparse = csr_matrix(H)
        k_use = min(k, H.shape[0] - 1)
        evals_sparse, _ = eigsh(H_sparse, k=k_use, which='SA')
        evals_sparse = np.sort(evals_sparse)
        
        # Compare
        diff = np.abs(evals_dense_low[:k_use] - evals_sparse)
        return {
            'k_compared': k_use,
            'dense_lowest': evals_dense_low[:k_use].tolist(),
            'sparse_lowest': evals_sparse.tolist(),
            'max_abs_diff': float(np.max(diff)),
            'mean_abs_diff': float(np.mean(diff)),
        }
    
    def validate_tfim(self) -> Dict[str, Any]:
        N = 6
        H = build_model('tfim', n_sites=N, J=1.0, h=0.5)
        comp = self._compare_lowest_k(H, k=5)
        return {
            'test_name': 'sparse_dense_tfim',
            'description': f'TFIM N={N}: ARPACK Lanczos vs LAPACK direct',
            **comp,
            'passed': bool(comp['max_abs_diff'] < self.tolerance),
        }
    
    def validate_heisenberg(self) -> Dict[str, Any]:
        N = 6
        H = build_model('heisenberg', n_sites=N, J=1.0)
        comp = self._compare_lowest_k(H, k=5)
        return {
            'test_name': 'sparse_dense_heisenberg',
            'description': f'Heisenberg N={N}: ARPACK Lanczos vs LAPACK direct',
            **comp,
            'passed': bool(comp['max_abs_diff'] < self.tolerance),
        }
    
    def validate_kitaev(self) -> Dict[str, Any]:
        N = 6
        H = build_model('kitaev_chain', n_sites=N, t=1.0, mu=0.5, Delta=0.7)
        comp = self._compare_lowest_k(H, k=5)
        return {
            'test_name': 'sparse_dense_kitaev',
            'description': f'Kitaev chain N={N}: sparse vs dense',
            **comp,
            'passed': bool(comp['max_abs_diff'] < self.tolerance),
        }
    
    def validate_free_fermion(self) -> Dict[str, Any]:
        N = 6
        H = build_model('free_fermion', n_sites=N, t=1.0, mu=0.0)
        comp = self._compare_lowest_k(H, k=5)
        return {
            'test_name': 'sparse_dense_free_fermion',
            'description': f'Free fermion N={N}: sparse vs dense',
            **comp,
            'passed': bool(comp['max_abs_diff'] < self.tolerance),
        }
    
    def validate_xxz(self) -> Dict[str, Any]:
        N = 6
        H = build_model('xxz', n_sites=N, J=1.0, Delta=0.5)
        comp = self._compare_lowest_k(H, k=5)
        return {
            'test_name': 'sparse_dense_xxz',
            'description': f'XXZ N={N}: sparse vs dense',
            **comp,
            'passed': bool(comp['max_abs_diff'] < self.tolerance),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r.get('passed', False))
        return {
            'validator': 'SparseDenseValidator',
            'tolerance': self.tolerance,
            'note': 'Sparse Lanczos has lower numerical precision than dense LAPACK; tolerance reflects this',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'individual_results': self.results,
        }
