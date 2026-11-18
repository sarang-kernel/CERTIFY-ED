"""
Symmetry-Resolved Spectrum Validator
====================================

When a Hamiltonian has a conserved quantity S, the spectrum splits into
sectors labeled by S eigenvalues. This validator:
    1. Constructs symmetry sector subspaces
    2. Diagonalizes H within each sector independently
    3. Verifies sector spectra union equals full spectrum

This is a strong consistency check because:
    - Sector diagonalization uses smaller matrices (different rounding)
    - Sectors must be disjoint and cover the full spectrum
    - Each sector eigenvalue must appear in the full spectrum

Tested for: Heisenberg (Sz sectors), TFIM (parity sectors).
"""

import numpy as np
from typing import Dict, List, Any
from scipy.linalg import eigh
from certify_ed import (
    build_model, MultiOracle,
    total_sz_operator, parity_operator,
)


class SymmetrySectorValidator:
    """Validate symmetry-resolved spectrum decomposition."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        self.results.append(self.test_heisenberg_sz_sectors())
        self.results.append(self.test_tfim_parity_sectors())
        self.results.append(self.test_xxz_sz_sectors())
        return self.results
    
    def _split_into_sectors(self, S: np.ndarray, tolerance: float = 1e-10
                            ) -> List[np.ndarray]:
        """Diagonalize S to get sector projectors."""
        s_evals, s_evecs = eigh(S)
        # Group eigenvalues that are equal within tolerance
        sectors = []  # list of (eigenvalue, basis_columns)
        for i, ev in enumerate(s_evals):
            placed = False
            for sec in sectors:
                if abs(sec[0] - ev) < tolerance:
                    sec[1].append(s_evecs[:, i])
                    placed = True
                    break
            if not placed:
                sectors.append([ev, [s_evecs[:, i]]])
        return [(ev, np.column_stack(basis)) for ev, basis in sectors]
    
    def _diagonalize_in_sector(self, H: np.ndarray, basis: np.ndarray
                                ) -> np.ndarray:
        """Project H into sector basis and diagonalize."""
        H_sector = basis.conj().T @ H @ basis
        # Symmetrize for floating point
        H_sector = (H_sector + H_sector.conj().T) / 2
        return np.linalg.eigvalsh(H_sector)
    
    def test_heisenberg_sz_sectors(self) -> Dict[str, Any]:
        """Heisenberg: split into Sz sectors, verify spectrum decomposition."""
        N = 4
        H = build_model('heisenberg', n_sites=N)
        Sz = total_sz_operator(N)
        
        # Full spectrum
        evals_full, _, _ = self.oracle.diagonalize_with_consensus(H)
        evals_full = np.sort(evals_full)
        
        # Sector decomposition
        sectors = self._split_into_sectors(Sz)
        all_sector_evals = []
        sector_info = []
        for sz_value, basis in sectors:
            evals_sec = self._diagonalize_in_sector(H, basis)
            all_sector_evals.extend(evals_sec.tolist())
            sector_info.append({
                'Sz': float(sz_value),
                'sector_dimension': basis.shape[1],
                'n_eigenvalues': len(evals_sec),
            })
        
        all_sector_evals = np.sort(np.array(all_sector_evals))
        
        # Compare full spectrum with union of sectors
        diff = np.abs(evals_full - all_sector_evals)
        max_diff = float(np.max(diff))
        
        return {
            'test_name': 'heisenberg_sz_sectors',
            'description': f'Heisenberg N={N}: full spectrum vs union of Sz sectors',
            'n_sectors': len(sectors),
            'sector_info': sector_info,
            'max_abs_diff': max_diff,
            'mean_abs_diff': float(np.mean(diff)),
            'passed': bool(max_diff < self.tolerance),
        }
    
    def test_tfim_parity_sectors(self) -> Dict[str, Any]:
        """TFIM: split into parity (+/-) sectors."""
        N = 4
        H = build_model('tfim', n_sites=N, J=1.0, h=0.5)
        P = parity_operator(N)
        
        evals_full, _, _ = self.oracle.diagonalize_with_consensus(H)
        evals_full = np.sort(evals_full)
        
        sectors = self._split_into_sectors(P)
        all_sector_evals = []
        sector_info = []
        for p_value, basis in sectors:
            evals_sec = self._diagonalize_in_sector(H, basis)
            all_sector_evals.extend(evals_sec.tolist())
            sector_info.append({
                'parity': float(p_value),
                'sector_dimension': basis.shape[1],
                'n_eigenvalues': len(evals_sec),
            })
        
        all_sector_evals = np.sort(np.array(all_sector_evals))
        diff = np.abs(evals_full - all_sector_evals)
        
        return {
            'test_name': 'tfim_parity_sectors',
            'description': f'TFIM N={N}: full spectrum vs union of parity sectors',
            'n_sectors': len(sectors),
            'sector_info': sector_info,
            'max_abs_diff': float(np.max(diff)),
            'mean_abs_diff': float(np.mean(diff)),
            'passed': bool(np.max(diff) < self.tolerance),
        }
    
    def test_xxz_sz_sectors(self) -> Dict[str, Any]:
        """XXZ: split into Sz sectors."""
        N = 4
        H = build_model('xxz', n_sites=N, J=1.0, Delta=0.7)
        Sz = total_sz_operator(N)
        
        evals_full, _, _ = self.oracle.diagonalize_with_consensus(H)
        evals_full = np.sort(evals_full)
        
        sectors = self._split_into_sectors(Sz)
        all_sector_evals = []
        sector_info = []
        for sz_value, basis in sectors:
            evals_sec = self._diagonalize_in_sector(H, basis)
            all_sector_evals.extend(evals_sec.tolist())
            sector_info.append({
                'Sz': float(sz_value),
                'sector_dimension': basis.shape[1],
                'n_eigenvalues': len(evals_sec),
            })
        
        all_sector_evals = np.sort(np.array(all_sector_evals))
        diff = np.abs(evals_full - all_sector_evals)
        
        return {
            'test_name': 'xxz_sz_sectors',
            'description': f'XXZ N={N}, Delta=0.7: full vs Sz-sector spectrum',
            'n_sectors': len(sectors),
            'sector_info': sector_info,
            'max_abs_diff': float(np.max(diff)),
            'mean_abs_diff': float(np.mean(diff)),
            'passed': bool(np.max(diff) < self.tolerance),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        return {
            'validator': 'SymmetrySectorValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'individual_results': self.results,
        }
