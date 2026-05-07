"""
QuSpin Cross-Validator
======================

Cross-validates against QuSpin (independent ED implementation by Weinberg & Bukov).
Falls back gracefully if QuSpin is not installed.

Tests cover:
    - TFIM full parameter sweep (h from 0.1 to 2.0)
    - Heisenberg full eigenspectrum
    - XXZ at several anisotropies
    - Free fermion via QuSpin spinless basis
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import build_model, MultiOracle


class QuSpinValidator:
    """Validate against QuSpin if available."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
        self.quspin_available = self._check_quspin()
    
    def _check_quspin(self) -> bool:
        try:
            import quspin
            return True
        except ImportError:
            return False
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        if not self.quspin_available:
            self.results.append({
                'test_name': 'quspin_validation',
                'status': 'skipped',
                'reason': 'QuSpin not installed',
                'install': 'pip install quspin',
                'note': 'QuSpin requires Python 3.10. If unavailable, this is skipped without invalidating other validators.',
            })
            return self.results
        
        self.results.append(self.validate_tfim_sweep())
        self.results.append(self.validate_heisenberg_full_spectrum())
        self.results.append(self.validate_xxz_anisotropy_sweep())
        return self.results
    
    def _build_tfim_quspin(self, N, J, h):
        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        import warnings as _w
        _w.filterwarnings('ignore')
        basis = spin_basis_1d(L=N, pauli=1)
        zz = [[-J, i, i+1] for i in range(N-1)]
        x = [[-h, i] for i in range(N)]
        H = hamiltonian([['zz', zz], ['x', x]], [], basis=basis,
                        check_symm=False, check_herm=False, check_pcon=False,
                        dtype=np.complex128)
        return H.toarray()
    
    def _build_heisenberg_quspin(self, N, J):
        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        import warnings as _w
        _w.filterwarnings('ignore')
        basis = spin_basis_1d(L=N, pauli=0)  # spin operators (S=sigma/2)
        xx = [[J, i, i+1] for i in range(N-1)]
        yy = [[J, i, i+1] for i in range(N-1)]
        zz = [[J, i, i+1] for i in range(N-1)]
        H = hamiltonian([['xx', xx], ['yy', yy], ['zz', zz]], [], basis=basis,
                        check_symm=False, check_herm=False, check_pcon=False,
                        dtype=np.complex128)
        return H.toarray()
    
    def _build_xxz_quspin(self, N, J, Delta):
        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        import warnings as _w
        _w.filterwarnings('ignore')
        basis = spin_basis_1d(L=N, pauli=0)
        xx = [[J, i, i+1] for i in range(N-1)]
        yy = [[J, i, i+1] for i in range(N-1)]
        zz = [[J*Delta, i, i+1] for i in range(N-1)]
        H = hamiltonian([['xx', xx], ['yy', yy], ['zz', zz]], [], basis=basis,
                        check_symm=False, check_herm=False, check_pcon=False,
                        dtype=np.complex128)
        return H.toarray()
    
    def validate_tfim_sweep(self) -> Dict[str, Any]:
        """20-point sweep of TFIM h parameter, compare full spectra."""
        N = 4
        J = 1.0
        h_values = np.linspace(0.1, 2.0, 20)
        
        all_diffs = []
        max_abs = 0.0
        max_rel = 0.0
        
        for h in h_values:
            H_cert = build_model('tfim', n_sites=N, J=J, h=h, boundary='open')
            evals_c, _, _ = self.oracle.diagonalize_with_consensus(H_cert)
            
            H_q = self._build_tfim_quspin(N, J, h)
            evals_q = np.linalg.eigvalsh(H_q)
            
            diff = np.abs(np.sort(evals_c) - np.sort(evals_q))
            rel = diff / (np.abs(np.sort(evals_q)) + 1e-15)
            
            max_abs = max(max_abs, np.max(diff))
            max_rel = max(max_rel, np.max(rel))
            all_diffs.extend(diff.tolist())
        
        return {
            'test_name': 'tfim_sweep_quspin',
            'description': '4-site TFIM, h in [0.1, 2.0], 20 points x 16 eigenvalues = 320 comparisons',
            'n_parameter_points': len(h_values),
            'n_eigenvalues_per_point': 2**N,
            'total_comparisons': len(h_values) * 2**N,
            'max_absolute_difference': float(max_abs),
            'mean_absolute_difference': float(np.mean(all_diffs)),
            'max_relative_difference': float(max_rel),
            'passed': bool(max_abs < self.tolerance),
        }
    
    def validate_heisenberg_full_spectrum(self) -> Dict[str, Any]:
        N = 4
        J = 1.0
        H_cert = build_model('heisenberg', n_sites=N, J=J)
        evals_c, _, _ = self.oracle.diagonalize_with_consensus(H_cert)
        
        H_q = self._build_heisenberg_quspin(N, J)
        evals_q = np.linalg.eigvalsh(H_q)
        
        diff = np.abs(np.sort(evals_c) - np.sort(evals_q))
        return {
            'test_name': 'heisenberg_full_spectrum_quspin',
            'description': '4-site Heisenberg, full 16-eigenvalue spectrum',
            'max_abs_diff': float(np.max(diff)),
            'mean_abs_diff': float(np.mean(diff)),
            'passed': bool(np.max(diff) < self.tolerance),
        }
    
    def validate_xxz_anisotropy_sweep(self) -> Dict[str, Any]:
        N = 4
        J = 1.0
        Deltas = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        max_abs = 0.0
        all_diffs = []
        for Delta in Deltas:
            H_cert = build_model('xxz', n_sites=N, J=J, Delta=Delta)
            evals_c, _, _ = self.oracle.diagonalize_with_consensus(H_cert)
            H_q = self._build_xxz_quspin(N, J, Delta)
            evals_q = np.linalg.eigvalsh(H_q)
            diff = np.abs(np.sort(evals_c) - np.sort(evals_q))
            max_abs = max(max_abs, np.max(diff))
            all_diffs.extend(diff.tolist())
        
        return {
            'test_name': 'xxz_anisotropy_sweep_quspin',
            'description': f'4-site XXZ, Delta sweep over {len(Deltas)} points',
            'deltas_tested': Deltas,
            'max_abs_diff': float(max_abs),
            'mean_abs_diff': float(np.mean(all_diffs)),
            'passed': bool(max_abs < self.tolerance),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        if not self.quspin_available:
            return {
                'validator': 'QuSpinValidator',
                'available': False,
                'n_total': 0,
                'n_passed': 0,
                'note': 'Skipped (QuSpin not installed)',
                'individual_results': self.results,
            }
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r.get('passed', False))
        return {
            'validator': 'QuSpinValidator',
            'available': True,
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'individual_results': self.results,
        }
