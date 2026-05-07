"""
Thermal Limit Validator
=======================

In the high-temperature limit (beta -> 0), thermal averages have
known analytic forms:
    
    <O>_T = tr(O) / d  (infinite temperature)
    
    Z(T -> inf) = d  (Hilbert space dimension)
    
    F(T -> inf) -> -T*ln(d)
    
    <H>_T = tr(H) / d at beta=0
    
    <H^2>_T = tr(H^2) / d at beta=0

Tests these in the asymptotic regime as beta -> 0.

Also tests low-temperature limit:
    <O>_T -> <0|O|0> as T -> 0 (ground state expectation)
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import (
    build_model, MultiOracle, ObservableCalculator,
    total_sz_operator,
)


class ThermalLimitValidator:
    """Validate thermal observables against asymptotic limits."""
    
    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        models = [
            ('tfim', {'n_sites': 4, 'J': 1.0, 'h': 0.5}),
            ('heisenberg', {'n_sites': 4}),
            ('xxz', {'n_sites': 4, 'Delta': 0.5}),
            ('cluster', {'n_sites': 4}),
        ]
        for name, kwargs in models:
            self.results.append(self.test_high_T_limit(name, kwargs))
            self.results.append(self.test_low_T_limit(name, kwargs))
        return self.results
    
    def test_high_T_limit(self, model_name: str, model_kwargs: Dict
                           ) -> Dict[str, Any]:
        """High-T limit: <H>_T -> tr(H)/d as beta -> 0."""
        H = build_model(model_name, **model_kwargs)
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        d = H.shape[0]
        calc = ObservableCalculator(evals, evecs)
        
        # Analytic high-T limits
        tr_H_over_d = float(np.trace(H).real / d)
        tr_H2_over_d = float(np.trace(H @ H).real / d)
        Z_at_zero = float(d)  # Z(beta=0) = d
        
        # Compute thermal averages at very small beta
        beta_small = 1e-8  # near beta=0
        H_avg_thermal = calc.thermal_average(H, beta_small)
        H2_avg_thermal = calc.thermal_average(H @ H, beta_small)
        Z_thermal = calc.partition_function(beta_small)
        
        # Compare
        err_H = abs(H_avg_thermal - tr_H_over_d)
        err_H2 = abs(H2_avg_thermal - tr_H2_over_d)
        # Z at small beta: Z ~ d - beta*tr(H), so use relative
        err_Z = abs(Z_thermal - Z_at_zero) / Z_at_zero
        
        max_err = max(err_H, err_H2, err_Z)
        
        return {
            'test_name': f'high_T_limit_{model_name}',
            'description': f'{model_name}: high-T thermal limits',
            'beta_used': beta_small,
            'analytic_H_avg': tr_H_over_d,
            'computed_H_avg': float(H_avg_thermal),
            'analytic_H2_avg': tr_H2_over_d,
            'computed_H2_avg': float(H2_avg_thermal),
            'analytic_Z': Z_at_zero,
            'computed_Z': float(Z_thermal),
            'max_relative_error': float(max_err),
            'passed': bool(max_err < self.tolerance * 1e3),  # allow some slack at finite beta
        }
    
    def test_low_T_limit(self, model_name: str, model_kwargs: Dict
                          ) -> Dict[str, Any]:
        """Low-T limit: thermal averages -> ground state expectation."""
        H = build_model(model_name, **model_kwargs)
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        calc = ObservableCalculator(evals, evecs)
        
        # Use Sz total as test observable (defined for all spin models)
        n_sites = model_kwargs.get('n_sites', 4)
        Sz = total_sz_operator(n_sites)
        
        # Ground state expectation
        gs_Sz = calc.expectation_value(Sz, 0)
        # Ground state energy
        E0 = float(evals[0])
        
        # Thermal average at very low T (large beta)
        beta_large = 1000.0
        thermal_Sz = calc.thermal_average(Sz, beta_large)
        thermal_E = calc.thermal_average(H, beta_large)
        
        # Note: if ground state is degenerate, thermal average -> avg over GS manifold
        gap = float(evals[1] - evals[0])
        is_degenerate = gap < 1e-10
        
        err_Sz = abs(thermal_Sz - gs_Sz) if not is_degenerate else None
        err_E = abs(thermal_E - E0)
        
        passed = err_E < self.tolerance * 100  # finite beta has some error
        if err_Sz is not None:
            passed = passed and err_Sz < self.tolerance * 100
        
        return {
            'test_name': f'low_T_limit_{model_name}',
            'description': f'{model_name}: low-T thermal -> ground state',
            'beta_used': beta_large,
            'is_degenerate_GS': is_degenerate,
            'gap': gap,
            'gs_energy': E0,
            'thermal_E_at_low_T': float(thermal_E),
            'energy_error': float(err_E),
            'gs_Sz': float(gs_Sz),
            'thermal_Sz_at_low_T': float(thermal_Sz),
            'Sz_error': float(err_Sz) if err_Sz is not None else None,
            'passed': bool(passed),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        return {
            'validator': 'ThermalLimitValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'individual_results': self.results,
        }
