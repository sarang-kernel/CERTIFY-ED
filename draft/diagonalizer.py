"""
Multi-Oracle Diagonalization and Consensus

Implements parallel eigendecomposition using multiple independent oracles
(NumPy, optionally Lanczos) with consensus validation.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class MultiOracleDiagonalizer:
    """Manages multi-oracle eigendecomposition and consensus."""

    def __init__(self, tolerance: float = 1e-10, use_lanczos: bool = False):
        """
        Initialize diagonalizer with tolerance parameters.

        Args:
            tolerance: Agreement tolerance between oracles
            use_lanczos: Whether to use Lanczos for large sparse systems
        """
        self.tolerance = tolerance
        self.use_lanczos = use_lanczos
        self.results = {
            'numpy': None,
            'numpy_alt': None,
            'lanczos': None
        }
        self.consensus_eigenvalues = None
        self.agreement_validation = None

    def _oracle_numpy(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition via NumPy LAPACK wrapper (primary).

        Args:
            H: Hamiltonian matrix

        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            logger.info("✓ NumPy (primary) diagonalization complete")
            return eigenvalues, eigenvectors
        except Exception as e:
            logger.error(f"NumPy (primary) oracle failed: {e}")
            return None, None

    def _oracle_numpy_alt(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition via NumPy with double-pass for validation.

        Args:
            H: Hamiltonian matrix

        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        try:
            # Ensure Hermitian
            H_herm = (H + H.conj().T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(H_herm)
            logger.info("✓ NumPy (alt) diagonalization complete")
            return eigenvalues, eigenvectors
        except Exception as e:
            logger.error(f"NumPy (alt) oracle failed: {e}")
            return None, None

    def _oracle_lanczos(self, H: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition via Lanczos iteration (for sparse systems).

        Args:
            H: Hamiltonian matrix
            k: Number of eigenvalues to compute (default: 20% of dimension)

        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        try:
            d = H.shape[0]
            if k is None:
                k = max(10, min(d-2, d // 5))

            eigenvalues, eigenvectors = eigsh(H, k=k, which='SM')
            logger.info(f"✓ Lanczos diagonalization complete ({k} eigenvalues)")
            return eigenvalues, eigenvectors
        except Exception as e:
            logger.error(f"Lanczos oracle failed: {e}")
            return None, None

    def diagonalize(self, H: np.ndarray, full_spectrum: bool = True) -> Dict:
        """
        Execute multi-oracle diagonalization with consensus.

        Args:
            H: Hamiltonian matrix
            full_spectrum: Whether to compute all eigenvalues

        Returns:
            Dictionary with consensus results and diagnostics
        """
        logger.info(f"Starting multi-oracle diagonalization ({H.shape[0]}×{H.shape[1]})")

        # Run oracles
        ev_np, ec_np = self._oracle_numpy(H)
        ev_alt, ec_alt = self._oracle_numpy_alt(H)

        if self.use_lanczos:
            ev_lz, ec_lz = self._oracle_lanczos(H)
        else:
            ev_lz, ec_lz = None, None

        # Store results
        self.results['numpy'] = (ev_np, ec_np)
        self.results['numpy_alt'] = (ev_alt, ec_alt)
        self.results['lanczos'] = (ev_lz, ec_lz)

        # Consensus protocol
        self._compute_consensus(ev_np, ev_alt, ev_lz)
        self._validate_agreement(ev_np, ev_alt)

        return {
            'consensus_eigenvalues': self.consensus_eigenvalues,
            'eigenvectors': ec_np,  # Use NumPy eigenvectors as canonical
            'agreement_validation': self.agreement_validation,
            'oracle_results': self.results
        }

    def _compute_consensus(self, ev_np, ev_alt, ev_lz):
        """
        Compute consensus eigenvalues via median aggregation.
        """
        eigenvalue_sets = [ev_np, ev_alt]
        if ev_lz is not None:
            eigenvalue_sets.append(ev_lz)

        eigenvalue_sets = [e for e in eigenvalue_sets if e is not None]

        if not eigenvalue_sets:
            logger.error("No valid oracle results")
            return

        # Median aggregation
        min_len = min(len(e) for e in eigenvalue_sets)
        consensus = np.median([e[:min_len] for e in eigenvalue_sets], axis=0)

        self.consensus_eigenvalues = consensus
        logger.info(f"✓ Consensus computed for {len(consensus)} eigenvalues")

    def _validate_agreement(self, ev_np, ev_alt):
        """
        Validate agreement between oracles.
        """
        if ev_np is None or ev_alt is None:
            self.agreement_validation = {'status': 'incomplete'}
            return

        # Pairwise differences
        min_len = min(len(ev_np), len(ev_alt))
        differences = np.abs(ev_np[:min_len] - ev_alt[:min_len])

        max_diff = np.max(differences)
        mean_diff = np.mean(differences)

        passed = max_diff < self.tolerance

        self.agreement_validation = {
            'status': 'passed' if passed else 'warning',
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': self.tolerance,
            'num_disagreements': int(np.sum(differences > self.tolerance))
        }

        if passed:
            logger.info(f"✓ Agreement validation passed (max diff: {max_diff:.2e})")
        else:
            logger.warning(f"⚠ Agreement warning (max diff: {max_diff:.2e})")


if __name__ == "__main__":
    # Test: Simple 2×2 system
    H_test = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=np.complex128)
    diag = MultiOracleDiagonalizer(tolerance=1e-10)
    result = diag.diagonalize(H_test)
    print("Diagonalization complete")
    print(f"Consensus eigenvalues: {result['consensus_eigenvalues']}")
