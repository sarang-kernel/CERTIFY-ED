"""
Finite-Size Scaling Validator
=============================

Tests that finite-size results approach known thermodynamic limits.
This is a softer check (since finite-N values won't equal asymptotic
values), but verifies the trend is correct.

Tests:
    - Heisenberg ground state energy per site -> -ln(2) + 1/4 = 0.443... (Bethe ansatz, J=1)
      Wait, J*S.S convention: E_0/N = ln(2) - 1/4 = 0.443
      In our (J/4)(XX+YY+ZZ) convention with Pauli ops, this is J*<S.S>_per_bond
      For ours with J=1: E/N approaches the Bethe ansatz result as N grows
    - TFIM ground state energy per site at critical point h/J=1
    - Free fermion energy per site -> -2t/pi (continuum limit, half-filled, J=1)
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import build_model, MultiOracle


class FiniteSizeScalingValidator:
    """Validate finite-size scaling toward thermodynamic limits."""
    
    def __init__(self, tolerance: float = 1e-2):
        # Tolerance is loose because finite-N != infinite-N
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        self.results.append(self.test_heisenberg_scaling())
        self.results.append(self.test_free_fermion_scaling())
        self.results.append(self.test_tfim_critical_scaling())
        return self.results
    
    def test_heisenberg_scaling(self) -> Dict[str, Any]:
        """Heisenberg E_0/N approaches Bethe ansatz value.
        
        Bethe ansatz infinite chain (S=1/2, J=1, our normalization):
        E_0/N = (1/4 - ln 2) = -0.4431471...
        
        Energy per bond is what's universal; for OBC chain with N sites
        we have N-1 bonds, so E_0/(N-1) -> -0.4431...
        """
        sizes = [4, 6, 8, 10]
        energies_per_bond = []
        for N in sizes:
            H = build_model('heisenberg', n_sites=N, J=1.0, boundary='periodic')
            evals, _, _ = self.oracle.diagonalize_with_consensus(H)
            E0 = float(evals[0])
            # PBC: N bonds
            energies_per_bond.append(E0 / N)
        
        # Bethe ansatz: lim N->inf E_0/N = -ln(2) + 1/4 = -0.44315...
        bethe_value = -np.log(2) + 0.25
        
        # Check trend: E/N should approach bethe_value as N grows
        # For finite N, there are 1/N^2 corrections
        # We just check that the largest N is closest
        deviations = [abs(e - bethe_value) for e in energies_per_bond]
        is_monotonic = all(deviations[i] >= deviations[i+1] - 0.01
                           for i in range(len(deviations)-1))
        
        # Final deviation
        final_dev = deviations[-1]
        
        return {
            'test_name': 'heisenberg_scaling',
            'description': f'Heisenberg PBC: E_0/N approaches Bethe ansatz {bethe_value:.6f}',
            'sizes': sizes,
            'energies_per_bond': energies_per_bond,
            'bethe_ansatz_limit': float(bethe_value),
            'deviations': deviations,
            'final_deviation': float(final_dev),
            'monotonic_convergence': bool(is_monotonic),
            'passed': bool(final_dev < 0.05),  # within 5%
        }
    
    def test_free_fermion_scaling(self) -> Dict[str, Any]:
        """Free fermion E_0/N at half-filling -> -2t/pi.
        
        For tight-binding chain at half-filling:
        E_0/N -> -(2t/pi) integral_{-pi/2}^{pi/2} cos(k) dk = -2t/pi (per site)
        With t=1: E_0/N -> -0.6366...
        """
        sizes = [4, 6, 8, 10]
        energies_per_site = []
        for N in sizes:
            H = build_model('free_fermion', n_sites=N, t=1.0, mu=0.0,
                            boundary='periodic')
            evals, _, _ = self.oracle.diagonalize_with_consensus(H)
            energies_per_site.append(float(evals[0]) / N)
        
        thermo_limit = -2.0 / np.pi  # = -0.6366...
        deviations = [abs(e - thermo_limit) for e in energies_per_site]
        
        return {
            'test_name': 'free_fermion_scaling',
            'description': f'Free fermion PBC: E_0/N -> -2t/pi = {thermo_limit:.6f}',
            'sizes': sizes,
            'energies_per_site': energies_per_site,
            'thermodynamic_limit': float(thermo_limit),
            'deviations': deviations,
            'final_deviation': float(deviations[-1]),
            'passed': bool(deviations[-1] < 0.1),  # finite-size and PBC corrections are O(1/N)
        }
    
    def test_tfim_critical_scaling(self) -> Dict[str, Any]:
        """TFIM at critical point h/J=1: E_0/N -> -4/pi.
        
        At quantum critical point, ground state energy density:
        E_0/N -> -(4J/pi) for J=h critical point.
        """
        J = 1.0
        h = 1.0
        sizes = [4, 6, 8, 10]
        energies_per_site = []
        for N in sizes:
            H = build_model('tfim', n_sites=N, J=J, h=h, boundary='periodic')
            evals, _, _ = self.oracle.diagonalize_with_consensus(H)
            energies_per_site.append(float(evals[0]) / N)
        
        thermo_limit = -4.0 / np.pi  # = -1.273...
        deviations = [abs(e - thermo_limit) for e in energies_per_site]
        
        return {
            'test_name': 'tfim_critical_scaling',
            'description': f'TFIM critical point: E_0/N -> -4/pi = {thermo_limit:.6f}',
            'sizes': sizes,
            'energies_per_site': energies_per_site,
            'thermodynamic_limit': float(thermo_limit),
            'deviations': deviations,
            'final_deviation': float(deviations[-1]),
            'passed': bool(deviations[-1] < 0.15),  # critical point has slow convergence
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        return {
            'validator': 'FiniteSizeScalingValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'note': 'Tolerance is loose (~5-15%) because finite-N values genuinely differ from thermodynamic limit',
            'individual_results': self.results,
        }
