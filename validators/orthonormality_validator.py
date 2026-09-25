"""
Orthonormality and Completeness Validator
=========================================

Eigenvectors of a Hermitian matrix must be orthonormal and complete:
    1. Orthonormality: <psi_i|psi_j> = delta_ij  (V^dag V = I)
    2. Completeness: sum_n |psi_n><psi_n| = I    (V V^dag = I)
    3. Spectral decomposition: H = sum_n E_n |psi_n><psi_n| = V D V^dag

These are independent of the eigenvalues themselves. Failure indicates
loss of orthogonality (typical in iterative methods, less so in LAPACK).
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import build_model, MultiOracle


class OrthonormalityValidator:
    """Validate eigenvector orthonormality and completeness."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        models = [
            ('tfim', {'n_sites': 5, 'J': 1.0, 'h': 0.5}),
            ('heisenberg', {'n_sites': 4}),
            ('xxz', {'n_sites': 4, 'Delta': 0.5}),
            ('ssh', {'n_sites': 5}),
            ('j1j2', {'n_sites': 4}),
            ('cluster', {'n_sites': 5}),
            ('free_fermion', {'n_sites': 5}),
            ('kitaev_chain', {'n_sites': 5, 't': 1.0, 'mu': 0.5, 'Delta': 0.7}),
            ('aklt', {'n_sites': 3}),
        ]
        for name, kwargs in models:
            self.results.append(self.test_orthonormality(name, kwargs))
        return self.results
    
    def test_orthonormality(self, model_name: str, model_kwargs: Dict) -> Dict[str, Any]:
        H = build_model(model_name, **model_kwargs)
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        d = H.shape[0]
        
        # 1. Orthonormality: V^dag V = I
        orth_matrix = evecs.conj().T @ evecs
        orth_error = float(np.max(np.abs(orth_matrix - np.eye(d))))
        
        # 2. Completeness: V V^dag = I
        comp_matrix = evecs @ evecs.conj().T
        comp_error = float(np.max(np.abs(comp_matrix - np.eye(d))))
        
        # 3. Spectral decomposition: H = V D V^dag
        D = np.diag(evals).astype(complex)
        H_reconstructed = evecs @ D @ evecs.conj().T
        spectral_error = float(np.max(np.abs(H - H_reconstructed)))
        
        # 4. Individual eigenvector norms
        norms = np.array([np.linalg.norm(evecs[:, i]) for i in range(d)])
        max_norm_error = float(np.max(np.abs(norms - 1.0)))
        
        # 5. Pairwise orthogonality
        max_overlap = 0.0
        for i in range(min(d, 30)):  # check first 30 for speed
            for j in range(i + 1, min(d, 30)):
                overlap = abs(np.vdot(evecs[:, i], evecs[:, j]))
                max_overlap = max(max_overlap, overlap)
        
        max_err = max(orth_error, comp_error, spectral_error, max_norm_error)
        
        return {
            'test_name': f'orthonormality_{model_name}',
            'description': f'Orthonormality + completeness for {model_name}',
            'dimension': d,
            'orthonormality_error': orth_error,
            'completeness_error': comp_error,
            'spectral_decomposition_error': spectral_error,
            'max_norm_error': max_norm_error,
            'max_pairwise_overlap': float(max_overlap),
            'max_error': max_err,
            'passed': bool(max_err < self.tolerance),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        max_err = max(r['max_error'] for r in self.results)
        return {
            'validator': 'OrthonormalityValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'max_orthonormality_error_overall': max_err,
            'individual_results': self.results,
        }
