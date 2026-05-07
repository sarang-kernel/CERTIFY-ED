"""
Platform Reproducibility Benchmarks
==================================

Tests numerical consistency on the current platform.
For multi-platform testing, run this on each platform and compare results.
"""

import numpy as np
import platform
import hashlib
from typing import Dict, List, Any
from certify_ed import build_tfim, build_heisenberg, MultiOracle


class PlatformBenchmarks:
    """
    Generate platform-specific reproducibility data.
    
    Computes reference results on current platform with full metadata.
    Run on multiple platforms to compare.
    """
    
    def __init__(self):
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Collect detailed platform information."""
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'platform': platform.platform(),
            'processor': platform.processor() or 'unknown',
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'numpy_version': np.__version__,
        }
        
        try:
            import scipy
            info['scipy_version'] = scipy.__version__
        except ImportError:
            info['scipy_version'] = 'not installed'
        
        # BLAS info
        try:
            blas_info = np.show_config(mode='dicts')
            info['blas_info'] = blas_info if isinstance(blas_info, dict) else str(blas_info)
        except Exception:
            try:
                # Older numpy
                import io
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    np.show_config()
                info['numpy_config'] = buf.getvalue()
            except Exception:
                info['numpy_config'] = 'unavailable'
        
        return info
    
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all platform benchmarks."""
        self.results = []
        self.results.append(self.compute_reference_tfim())
        self.results.append(self.compute_reference_heisenberg())
        return self.results
    
    def compute_reference_tfim(self) -> Dict[str, Any]:
        """Compute reference TFIM result with full precision."""
        N = 4
        J = 1.0
        h = 0.5
        H = build_tfim(n_sites=N, J=J, h=h, boundary='open')
        
        evals, evecs, report = self.oracle.diagonalize_with_consensus(H)
        residuals = self.oracle.compute_residuals(H, evals, evecs)
        
        # Hash of eigenvalues for quick comparison
        evals_bytes = evals.tobytes()
        evals_hash = hashlib.sha256(evals_bytes).hexdigest()
        
        return {
            'test_name': 'platform_reference_tfim',
            'description': '4-site TFIM reference for platform comparison',
            'parameters': {'N': N, 'J': J, 'h': h},
            'eigenvalues_full_precision': [f"{e:.20e}" for e in evals],
            'eigenvalues_float': evals.tolist(),
            'eigenvalues_hash_sha256': evals_hash,
            'max_residual': float(np.max(residuals)),
            'consensus': report['consensus'],
            'max_disagreement': report['max_disagreement'],
        }
    
    def compute_reference_heisenberg(self) -> Dict[str, Any]:
        """Compute reference Heisenberg result."""
        N = 4
        J = 1.0
        H = build_heisenberg(n_sites=N, J=J, boundary='open')
        
        evals, evecs, report = self.oracle.diagonalize_with_consensus(H)
        residuals = self.oracle.compute_residuals(H, evals, evecs)
        
        evals_hash = hashlib.sha256(evals.tobytes()).hexdigest()
        
        return {
            'test_name': 'platform_reference_heisenberg',
            'description': '4-site Heisenberg reference for platform comparison',
            'parameters': {'N': N, 'J': J},
            'eigenvalues_full_precision': [f"{e:.20e}" for e in evals],
            'eigenvalues_float': evals.tolist(),
            'eigenvalues_hash_sha256': evals_hash,
            'max_residual': float(np.max(residuals)),
            'consensus': report['consensus'],
            'max_disagreement': report['max_disagreement'],
        }
    
    def summary(self) -> Dict[str, Any]:
        """Summary with platform info."""
        if not self.results:
            self.run_all()
        
        return {
            'platform_info': self.get_platform_info(),
            'reference_computations': self.results,
            'note': (
                'For multi-platform validation, run this benchmark on each '
                'target platform and compare eigenvalue hashes and float values.'
            )
        }
