"""
Performance Benchmarks Module
============================

Measures timing and scaling of CERTIFY-ED across system sizes.
"""

import numpy as np
import time
from typing import Dict, List, Any
from certify_ed import (
    build_tfim, build_heisenberg,
    NumPyOracle, ScipyOracle, MultiOracle
)


class PerformanceBenchmarks:
    """
    Benchmark performance characteristics.
    
    Tests:
    - Single-oracle timing
    - Multi-oracle timing
    - Scaling with system size
    - Memory usage
    """
    
    def __init__(self, n_runs: int = 5, max_n_sites: int = 10):
        self.n_runs = n_runs
        self.max_n_sites = max_n_sites
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all performance benchmarks."""
        self.results = []
        self.results.append(self.benchmark_scaling())
        self.results.append(self.benchmark_oracle_comparison())
        return self.results
    
    def _time_function(self, func, *args, n_runs: int = None) -> Dict[str, float]:
        """Time a function over multiple runs."""
        if n_runs is None:
            n_runs = self.n_runs
        
        # Warm-up
        func(*args)
        
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            func(*args)
            times.append(time.perf_counter() - t0)
        
        return {
            'mean_seconds': float(np.mean(times)),
            'std_seconds': float(np.std(times)),
            'min_seconds': float(np.min(times)),
            'max_seconds': float(np.max(times)),
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
        }
    
    def benchmark_scaling(self) -> Dict[str, Any]:
        """Measure performance scaling with system size."""
        sizes_data = []
        
        single_oracle = NumPyOracle()
        multi_oracle = MultiOracle()
        
        for N in range(2, self.max_n_sites + 1):
            d = 2 ** N
            
            # Build Hamiltonian
            t_build = self._time_function(
                build_tfim, N, 1.0, 0.5, 'open', n_runs=3
            )
            
            H = build_tfim(n_sites=N, J=1.0, h=0.5)
            
            # Single oracle
            t_single = self._time_function(
                single_oracle.diagonalize, H, n_runs=self.n_runs
            )
            
            # Multi-oracle
            t_multi = self._time_function(
                multi_oracle.diagonalize_with_consensus, H, n_runs=max(3, self.n_runs)
            )
            
            overhead = t_multi['mean_seconds'] / t_single['mean_seconds']
            
            sizes_data.append({
                'n_sites': N,
                'dimension': d,
                'build_time': t_build,
                'single_oracle_time': t_single,
                'multi_oracle_time': t_multi,
                'consensus_overhead': float(overhead),
                'matrix_memory_mb': (d * d * 16) / (1024 * 1024),  # complex128 = 16 bytes
            })
        
        return {
            'test_name': 'performance_scaling',
            'description': f'Scaling from N=2 to N={self.max_n_sites}',
            'n_runs_per_size': self.n_runs,
            'sizes': sizes_data
        }
    
    def benchmark_oracle_comparison(self) -> Dict[str, Any]:
        """Compare individual oracle performance."""
        N = 8
        H = build_tfim(n_sites=N, J=1.0, h=0.5)
        
        oracles = {
            'numpy_dsyevd': NumPyOracle(),
            'scipy_evd': ScipyOracle(driver='evd'),
            'scipy_evr': ScipyOracle(driver='evr'),
            'scipy_ev': ScipyOracle(driver='ev'),
        }
        
        oracle_data = {}
        for name, oracle in oracles.items():
            t = self._time_function(oracle.diagonalize, H, n_runs=self.n_runs)
            oracle_data[name] = t
        
        return {
            'test_name': 'oracle_comparison',
            'description': f'Individual oracle performance at N={N}',
            'n_sites': N,
            'dimension': 2**N,
            'n_runs': self.n_runs,
            'oracles': oracle_data
        }
    
    def summary(self) -> Dict[str, Any]:
        """Summary of performance results."""
        if not self.results:
            self.run_all()
        return {
            'individual_results': self.results
        }
