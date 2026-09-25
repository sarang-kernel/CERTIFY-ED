"""
High-Precision Validator
========================

Cross-validates against arbitrary-precision computation using mpmath.

mpmath uses arbitrary-precision floating-point arithmetic. Eigenvalues
computed at e.g. 50-digit precision serve as gold standard for
comparing standard double-precision results.

This is independent of LAPACK and provides a different numerical pathway
through mpmath's matrix algorithms (which use arbitrary precision QR
or characteristic polynomial methods).

Falls back gracefully if mpmath is not installed.
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import build_model, MultiOracle


class HighPrecisionValidator:
    """Cross-validate against mpmath arbitrary precision."""
    
    def __init__(self, precision_dps: int = 50, tolerance: float = 1e-10):
        """
        Parameters
        ----------
        precision_dps : int
            Decimal digits of precision (default 50).
        tolerance : float
            Tolerance for double-precision agreement.
        """
        self.precision_dps = precision_dps
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
        self.mpmath_available = self._check_mpmath()
    
    def _check_mpmath(self) -> bool:
        try:
            import mpmath
            return True
        except ImportError:
            return False
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        if not self.mpmath_available:
            self.results.append({
                'test_name': 'high_precision_validation',
                'status': 'skipped',
                'reason': 'mpmath not installed',
                'install': 'pip install mpmath',
            })
            return self.results
        
        # Run small-system tests where mpmath is fast
        self.results.append(self.validate_tfim_small())
        self.results.append(self.validate_heisenberg_small())
        self.results.append(self.validate_kitaev_small())
        return self.results
    
    def _diagonalize_mpmath(self, H_np: np.ndarray) -> np.ndarray:
        """Diagonalize using mpmath at high precision."""
        import mpmath
        mpmath.mp.dps = self.precision_dps
        # Convert to mpmath matrix (real part only - all our Hamiltonians are real Hermitian
        # in the basis we use; symmetrize first if needed)
        H_real = H_np.real
        if not np.allclose(H_np.imag, 0, atol=1e-12):
            # Has imaginary parts; mpmath handles via mpc but slower
            # For our test set, Hamiltonians are real symmetric
            raise NotImplementedError("mpmath validator currently real Hermitian only")
        
        d = H_np.shape[0]
        H_mp = mpmath.matrix([[mpmath.mpf(float(H_real[i, j])) for j in range(d)]
                               for i in range(d)])
        # eigsy returns eigenvalues + eigenvectors for symmetric matrices
        eigvals_mp, _ = mpmath.eigsy(H_mp)
        # Convert back to numpy floats
        evals_np = np.array([float(e) for e in eigvals_mp])
        return np.sort(evals_np)
    
    def validate_tfim_small(self) -> Dict[str, Any]:
        """TFIM N=3: compare double precision vs mpmath at 50 digits."""
        N = 3
        J = 1.0
        h = 0.5
        H = build_model('tfim', n_sites=N, J=J, h=h)
        
        # Standard double precision
        evals_dp, _, _ = self.oracle.diagonalize_with_consensus(H)
        evals_dp = np.sort(evals_dp)
        
        # High precision reference
        evals_hp = self._diagonalize_mpmath(H)
        
        diff = np.abs(evals_dp - evals_hp)
        max_diff = float(np.max(diff))
        
        return {
            'test_name': 'tfim_high_precision',
            'description': f'TFIM N={N}, double vs mpmath {self.precision_dps}-digit',
            'precision_dps': self.precision_dps,
            'n_eigenvalues': len(evals_dp),
            'max_abs_diff': max_diff,
            'mean_abs_diff': float(np.mean(diff)),
            'eigenvalues_dp_first_few': evals_dp[:4].tolist(),
            'eigenvalues_hp_first_few': evals_hp[:4].tolist(),
            'passed': bool(max_diff < self.tolerance),
        }
    
    def validate_heisenberg_small(self) -> Dict[str, Any]:
        """Heisenberg N=3: high-precision check."""
        N = 3
        J = 1.0
        H = build_model('heisenberg', n_sites=N, J=J, boundary='periodic')
        
        evals_dp, _, _ = self.oracle.diagonalize_with_consensus(H)
        evals_dp = np.sort(evals_dp)
        evals_hp = self._diagonalize_mpmath(H)
        
        diff = np.abs(evals_dp - evals_hp)
        return {
            'test_name': 'heisenberg_high_precision',
            'description': f'Heisenberg N={N} PBC, double vs mpmath {self.precision_dps}-digit',
            'precision_dps': self.precision_dps,
            'max_abs_diff': float(np.max(diff)),
            'mean_abs_diff': float(np.mean(diff)),
            'passed': bool(np.max(diff) < self.tolerance),
        }
    
    def validate_kitaev_small(self) -> Dict[str, Any]:
        """Kitaev chain N=4: high-precision check."""
        N = 4
        H = build_model('kitaev_chain', n_sites=N, t=1.0, mu=0.5, Delta=0.7)
        
        evals_dp, _, _ = self.oracle.diagonalize_with_consensus(H)
        evals_dp = np.sort(evals_dp)
        evals_hp = self._diagonalize_mpmath(H)
        
        diff = np.abs(evals_dp - evals_hp)
        return {
            'test_name': 'kitaev_high_precision',
            'description': f'Kitaev chain N={N}, double vs mpmath {self.precision_dps}-digit',
            'precision_dps': self.precision_dps,
            'max_abs_diff': float(np.max(diff)),
            'mean_abs_diff': float(np.mean(diff)),
            'passed': bool(np.max(diff) < self.tolerance),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        if not self.mpmath_available:
            return {
                'validator': 'HighPrecisionValidator',
                'available': False,
                'n_total': 0,
                'n_passed': 0,
                'note': 'Skipped (mpmath not installed)',
                'individual_results': self.results,
            }
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r.get('passed', False))
        return {
            'validator': 'HighPrecisionValidator',
            'available': True,
            'precision_dps': self.precision_dps,
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'individual_results': self.results,
        }
