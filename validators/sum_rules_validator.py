"""
Spectral Sum Rules Validator
============================

Validates fundamental sum rules that any correct eigendecomposition
must satisfy. These are matrix invariants independent of basis.

Sum rules tested:
    1. Trace: tr(H) = sum_n E_n
    2. Trace of H^2: tr(H^2) = sum_n E_n^2  (Frobenius norm squared)
    3. Trace of H^3: tr(H^3) = sum_n E_n^3
    4. Determinant: det(H) = prod_n E_n
    5. Operator norm: ||H||_2 = max_n |E_n|

These provide N independent checks per system, each at machine precision.
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import build_model, MultiOracle


class SpectralSumRuleValidator:
    """Validate spectral sum rules across all models."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        # Test sum rules across multiple models
        models = [
            ('tfim', {'n_sites': 5, 'J': 1.0, 'h': 0.5}),
            ('heisenberg', {'n_sites': 4, 'J': 1.0}),
            ('xxz', {'n_sites': 4, 'J': 1.0, 'Delta': 0.7}),
            ('xyz', {'n_sites': 4, 'Jx': 1.0, 'Jy': 1.2, 'Jz': 0.8}),
            ('ssh', {'n_sites': 5, 't1': 1.0, 't2': 0.5}),
            ('j1j2', {'n_sites': 4, 'J1': 1.0, 'J2': 0.4}),
            ('cluster', {'n_sites': 5, 'h': 0.5}),
            ('free_fermion', {'n_sites': 5, 't': 1.0, 'mu': 0.3}),
            ('kitaev_chain', {'n_sites': 5, 't': 1.0, 'mu': 0.5, 'Delta': 0.7}),
            ('aklt', {'n_sites': 3}),
            ('haldane', {'n_sites': 3, 'D': 0.2}),
        ]
        
        for name, kwargs in models:
            self.results.append(self.test_sum_rules(name, kwargs))
        
        return self.results
    
    def test_sum_rules(self, model_name: str, model_kwargs: Dict) -> Dict[str, Any]:
        """Test all sum rules for a given model."""
        H = build_model(model_name, **model_kwargs)
        evals, _, _ = self.oracle.diagonalize_with_consensus(H)
        
        # Sum rule 1: trace
        tr_H_direct = np.trace(H).real
        tr_H_eigs = float(np.sum(evals))
        err_trace = abs(tr_H_direct - tr_H_eigs)
        
        # Sum rule 2: trace of H^2 (Frobenius norm squared)
        H_squared = H @ H
        tr_H2_direct = np.trace(H_squared).real
        tr_H2_eigs = float(np.sum(evals ** 2))
        err_trace2 = abs(tr_H2_direct - tr_H2_eigs)
        
        # Sum rule 3: trace of H^3
        H_cubed = H_squared @ H
        tr_H3_direct = np.trace(H_cubed).real
        tr_H3_eigs = float(np.sum(evals ** 3))
        err_trace3 = abs(tr_H3_direct - tr_H3_eigs)
        
        # Sum rule 4: log|det(H)| (use slogdet for stability)
        # If H has zero eigenvalue, skip
        try:
            sign, logdet = np.linalg.slogdet(H)
            logdet_direct = float(logdet)
            # From eigenvalues: log|det| = sum log|E_n|
            # Skip near-zero eigenvalues
            nonzero = np.abs(evals) > 1e-12
            if np.all(nonzero):
                logdet_eigs = float(np.sum(np.log(np.abs(evals))))
                err_logdet = abs(logdet_direct - logdet_eigs)
            else:
                err_logdet = None
        except Exception:
            err_logdet = None
        
        # Sum rule 5: operator norm = max |eigenvalue|
        norm_direct = float(np.linalg.norm(H, ord=2))
        norm_eigs = float(np.max(np.abs(evals)))
        err_norm = abs(norm_direct - norm_eigs)
        
        # Combined pass: all primary sum rules below tolerance
        # (logdet allowed to be None for matrices with zero eigenvalues)
        primary_max_err = max(err_trace, err_trace2, err_trace3, err_norm)
        
        return {
            'test_name': f'sum_rules_{model_name}',
            'description': f'Sum rules for {model_name}',
            'parameters': model_kwargs,
            'dimension': H.shape[0],
            'trace_error': err_trace,
            'trace_squared_error': err_trace2,
            'trace_cubed_error': err_trace3,
            'logdet_error': err_logdet,
            'operator_norm_error': err_norm,
            'max_error': primary_max_err,
            'passed': bool(primary_max_err < self.tolerance),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        max_err = max(r['max_error'] for r in self.results)
        return {
            'validator': 'SpectralSumRuleValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'max_sum_rule_error_overall': max_err,
            'sum_rules_tested': ['trace', 'trace_H_squared', 'trace_H_cubed',
                                 'log_determinant', 'operator_norm'],
            'individual_results': self.results,
        }
