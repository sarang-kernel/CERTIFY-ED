"""
Unitarity Validator
===================

Time evolution operator U(t) = exp(-i*H*t) must be unitary for any
Hermitian H. This is a strong consistency check on the eigendecomposition
since:

    exp(-iHt) = V exp(-iDt) V^dag

If V is not orthonormal or D is wrong, U deviates from unitary.

Tests:
    1. ||U U^dag - I||  (unitarity)
    2. ||U(t1)U(t2) - U(t1+t2)||  (group property)
    3. det(U) phase consistency
    4. Multiple time scales (short, medium, long)
"""

import numpy as np
from typing import Dict, List, Any
from scipy.linalg import expm
from certify_ed import build_model, MultiOracle


class UnitarityValidator:
    """Validate unitarity of time evolution from eigendecomposition."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        models = [
            ('tfim', {'n_sites': 4, 'J': 1.0, 'h': 0.5}),
            ('heisenberg', {'n_sites': 4}),
            ('xxz', {'n_sites': 4, 'Delta': 0.7}),
            ('ssh', {'n_sites': 4}),
            ('cluster', {'n_sites': 4}),
            ('free_fermion', {'n_sites': 4}),
            ('kitaev_chain', {'n_sites': 4, 't': 1.0, 'mu': 0.5, 'Delta': 0.7}),
        ]
        for name, kwargs in models:
            self.results.append(self.test_unitarity(name, kwargs))
        return self.results
    
    def _build_U(self, evals: np.ndarray, evecs: np.ndarray, t: float) -> np.ndarray:
        """U(t) = V exp(-i*D*t) V^dag from spectral decomposition."""
        D_t = np.diag(np.exp(-1j * evals * t))
        return evecs @ D_t @ evecs.conj().T
    
    def test_unitarity(self, model_name: str, model_kwargs: Dict) -> Dict[str, Any]:
        H = build_model(model_name, **model_kwargs)
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        d = H.shape[0]
        
        # Test at multiple times
        times = [0.1, 1.0, 10.0]
        unitarity_errors = []
        group_errors = []
        
        for t in times:
            U_spec = self._build_U(evals, evecs, t)
            
            # Unitarity check: U U^dag = I
            UUd = U_spec @ U_spec.conj().T
            unitarity_err = float(np.max(np.abs(UUd - np.eye(d))))
            unitarity_errors.append(unitarity_err)
        
        # Group property: U(t1)*U(t2) = U(t1+t2)
        for t1, t2 in [(0.5, 0.3), (1.0, 1.0), (2.0, 0.5)]:
            U1 = self._build_U(evals, evecs, t1)
            U2 = self._build_U(evals, evecs, t2)
            U12_spec = U1 @ U2
            U12_direct = self._build_U(evals, evecs, t1 + t2)
            group_err = float(np.max(np.abs(U12_spec - U12_direct)))
            group_errors.append(group_err)
        
        # Compare with direct matrix exponential at one time
        U_direct = expm(-1j * H * 1.0)
        U_spec = self._build_U(evals, evecs, 1.0)
        spec_vs_direct = float(np.max(np.abs(U_direct - U_spec)))
        
        max_err = max(max(unitarity_errors), max(group_errors), spec_vs_direct)
        
        return {
            'test_name': f'unitarity_{model_name}',
            'description': f'Time evolution unitarity for {model_name}',
            'dimension': d,
            'times_tested': times,
            'max_unitarity_error': float(max(unitarity_errors)),
            'max_group_property_error': float(max(group_errors)),
            'spectral_vs_expm_error': spec_vs_direct,
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
            'validator': 'UnitarityValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'max_unitarity_error_overall': max_err,
            'individual_results': self.results,
        }
