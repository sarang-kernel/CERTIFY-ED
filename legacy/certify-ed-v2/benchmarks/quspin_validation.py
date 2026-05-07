"""
QuSpin Cross-Validation Module
==============================

Cross-validates CERTIFY-ED against QuSpin (if installed).
Falls back gracefully if QuSpin is not available.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from certify_ed import build_tfim, build_heisenberg, MultiOracle


class QuSpinValidator:
    """
    Cross-validate CERTIFY-ED against QuSpin.
    
    QuSpin is an independent ED implementation widely used in the community.
    Agreement provides cross-validation evidence.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.quspin_available = self._check_quspin()
        self.results: List[Dict[str, Any]] = []
    
    def _check_quspin(self) -> bool:
        """Check if QuSpin is available."""
        try:
            import quspin
            return True
        except ImportError:
            return False
    
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all QuSpin validation tests."""
        self.results = []
        
        if not self.quspin_available:
            self.results.append({
                'test_name': 'quspin_validation',
                'status': 'skipped',
                'reason': 'QuSpin not installed',
                'install_command': 'pip install quspin',
                'note': 'QuSpin requires Python 3.10. If unavailable, this test is skipped but does not invalidate other benchmarks.'
            })
            return self.results
        
        self.results.append(self.validate_tfim_sweep())
        self.results.append(self.validate_heisenberg())
        return self.results
    
    def _build_tfim_quspin(self, N: int, J: float, h: float):
        """Build TFIM in QuSpin."""
        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        import warnings as _warnings
        _warnings.filterwarnings('ignore')
        
        basis = spin_basis_1d(L=N, pauli=1)  # Use Pauli convention
        
        # ZZ terms: -J sum Z_i Z_{i+1}
        zz_list = [[-J, i, i+1] for i in range(N-1)]
        # X terms: -h sum X_i
        x_list = [[-h, i] for i in range(N)]
        
        static = [['zz', zz_list], ['x', x_list]]
        H = hamiltonian(static, [], basis=basis,
                       check_symm=False, check_herm=False, check_pcon=False,
                       dtype=np.complex128)
        return H.toarray()
    
    def validate_tfim_sweep(self) -> Dict[str, Any]:
        """Sweep TFIM parameters and compare with QuSpin."""
        N = 4
        J = 1.0
        h_values = np.linspace(0.1, 2.0, 20)
        
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        all_diffs = []
        
        for h in h_values:
            # CERTIFY-ED
            H_cert = build_tfim(n_sites=N, J=J, h=h, boundary='open')
            evals_cert, _, _ = self.oracle.diagonalize_with_consensus(H_cert)
            
            # QuSpin
            H_quspin = self._build_tfim_quspin(N, J, h)
            evals_quspin = np.linalg.eigvalsh(H_quspin)
            
            # Compare (sort both)
            evals_cert_sorted = np.sort(evals_cert)
            evals_quspin_sorted = np.sort(evals_quspin)
            
            abs_diff = np.abs(evals_cert_sorted - evals_quspin_sorted)
            rel_diff = abs_diff / (np.abs(evals_quspin_sorted) + 1e-15)
            
            max_abs_diff = max(max_abs_diff, np.max(abs_diff))
            max_rel_diff = max(max_rel_diff, np.max(rel_diff))
            all_diffs.extend(abs_diff.tolist())
        
        return {
            'test_name': 'tfim_sweep_quspin_validation',
            'description': '4-site TFIM, h sweep [0.1, 2.0], 20 points',
            'system_size': N,
            'n_parameter_points': len(h_values),
            'n_eigenvalues_per_point': 2**N,
            'total_comparisons': len(h_values) * 2**N,
            'max_absolute_difference': float(max_abs_diff),
            'mean_absolute_difference': float(np.mean(all_diffs)),
            'max_relative_difference': float(max_rel_diff),
            'passed': bool(max_abs_diff < self.tolerance)
        }
    
    def validate_heisenberg(self) -> Dict[str, Any]:
        """Validate Heisenberg model against QuSpin."""
        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        import warnings as _warnings
        _warnings.filterwarnings('ignore')
        
        N = 4
        J = 1.0
        
        # CERTIFY-ED uses (J/4) * sum (XX+YY+ZZ) [Pauli operators]
        # QuSpin with pauli=1 uses Pauli operators directly
        # So in QuSpin: H = J*sum (S^x S^x + S^y S^y + S^z S^z) where S = sigma/2
        # = (J/4) sum (XX+YY+ZZ) ✓
        
        basis = spin_basis_1d(L=N, pauli=0)  # spin operators (S = sigma/2)
        
        xx_list = [[J, i, i+1] for i in range(N-1)]
        yy_list = [[J, i, i+1] for i in range(N-1)]
        zz_list = [[J, i, i+1] for i in range(N-1)]
        
        static = [['xx', xx_list], ['yy', yy_list], ['zz', zz_list]]
        H_quspin = hamiltonian(static, [], basis=basis,
                               check_symm=False, check_herm=False, check_pcon=False,
                               dtype=np.complex128)
        evals_quspin = np.linalg.eigvalsh(H_quspin.toarray())
        
        # CERTIFY-ED
        H_cert = build_heisenberg(n_sites=N, J=J, boundary='open')
        evals_cert, _, _ = self.oracle.diagonalize_with_consensus(H_cert)
        
        # Compare
        evals_cert_sorted = np.sort(evals_cert)
        evals_quspin_sorted = np.sort(evals_quspin)
        
        abs_diff = np.abs(evals_cert_sorted - evals_quspin_sorted)
        max_abs = np.max(abs_diff)
        
        return {
            'test_name': 'heisenberg_quspin_validation',
            'description': '4-site Heisenberg, comparison with QuSpin',
            'system_size': N,
            'n_eigenvalues': 2**N,
            'max_absolute_difference': float(max_abs),
            'mean_absolute_difference': float(np.mean(abs_diff)),
            'ground_state_certify': float(evals_cert_sorted[0]),
            'ground_state_quspin': float(evals_quspin_sorted[0]),
            'passed': bool(max_abs < self.tolerance)
        }
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary."""
        if not self.results:
            self.run_all()
        
        if not self.quspin_available:
            return {
                'quspin_available': False,
                'n_total': 0,
                'n_passed': 0,
                'note': 'QuSpin validation skipped - QuSpin not installed',
                'individual_results': self.results
            }
        
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r.get('passed', False))
        
        return {
            'quspin_available': True,
            'n_total': n_total,
            'n_passed': n_passed,
            'all_passed': n_passed == n_total,
            'individual_results': self.results
        }
