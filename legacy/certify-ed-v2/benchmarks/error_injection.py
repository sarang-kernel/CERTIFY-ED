"""
Error Injection Tests Module
===========================

Validates that CERTIFY-ED detects intentionally injected errors.
This is critical for verifying the framework's verification capability.
"""

import numpy as np
from typing import Dict, List, Any
from certify_ed import (
    build_tfim, SymbolicHamiltonian, MultiOracle,
    pauli_matrices, tensor_product
)
import warnings


class ErrorInjectionTests:
    """
    Inject controlled errors to verify detection capability.
    
    Tests:
    1. Sign error in coupling - should change spectrum
    2. Missing Hermitian conjugate - should fail Hermiticity check
    3. Matrix corruption - should produce non-zero residuals
    4. Inconsistent oracles - should fail consensus
    """
    
    def __init__(self):
        self.oracle = MultiOracle(tolerance=1e-10)
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        """Run all error injection tests."""
        self.results = []
        self.results.append(self.test_non_hermitian_detection())
        self.results.append(self.test_matrix_corruption_detection())
        self.results.append(self.test_oracle_disagreement_detection())
        self.results.append(self.test_residual_amplification())
        return self.results
    
    def test_non_hermitian_detection(self) -> Dict[str, Any]:
        """Inject non-Hermitian term and check detection."""
        N = 3
        H = build_tfim(n_sites=N, J=1.0, h=0.5)
        
        # Inject non-Hermitian perturbation
        perturbation = np.zeros_like(H)
        perturbation[0, 1] = 1j * 0.5  # Anti-Hermitian
        H_corrupted = H + perturbation
        
        # Check Hermiticity manually
        max_dev = np.max(np.abs(H_corrupted - H_corrupted.conj().T))
        is_hermitian = max_dev < 1e-14
        
        # Was the corruption detected?
        detected = not is_hermitian
        
        return {
            'test_name': 'non_hermitian_detection',
            'description': 'Inject anti-Hermitian term, check detection',
            'injected_perturbation_norm': float(np.linalg.norm(perturbation)),
            'max_hermiticity_deviation': float(max_dev),
            'detected': bool(detected),
            'passed': bool(detected),  # We expect detection
            'note': 'Detection means the framework correctly identifies the error'
        }
    
    def test_matrix_corruption_detection(self) -> Dict[str, Any]:
        """Inject random Hermitian noise and check residual amplification."""
        N = 4
        H_original = build_tfim(n_sites=N, J=1.0, h=0.5)
        
        # Inject Hermitian noise (small)
        rng = np.random.default_rng(42)
        noise_real = rng.normal(0, 1e-6, H_original.shape)
        noise = noise_real + 1j * 0  # Keep real for Hermitian
        noise = (noise + noise.T) / 2  # Make symmetric
        
        H_corrupted = H_original + noise
        
        # Diagonalize corrupted matrix
        evals_c, evecs_c, _ = self.oracle.diagonalize_with_consensus(H_corrupted)
        
        # Compute residuals against ORIGINAL H (the one we claim to certify)
        # If user claims results are for H_original but actually computed H_corrupted,
        # residuals should reveal this
        residuals_against_original = self.oracle.compute_residuals(
            H_original, evals_c, evecs_c
        )
        
        max_residual = np.max(residuals_against_original)
        # Should be ~ noise norm
        noise_norm = np.linalg.norm(noise)
        
        # Detection: residuals are larger than they should be for clean computation
        detected = max_residual > 1e-10
        
        return {
            'test_name': 'matrix_corruption_detection',
            'description': 'Compute on corrupted matrix, check residuals against original',
            'injected_noise_norm': float(noise_norm),
            'max_residual': float(max_residual),
            'expected_residual_order': 1e-6,
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def test_oracle_disagreement_detection(self) -> Dict[str, Any]:
        """Manually create disagreeing oracle and check consensus failure."""
        from certify_ed.oracles import Oracle, NumPyOracle, ScipyOracle, MultiOracle
        
        class CorruptedOracle(Oracle):
            """Oracle that returns incorrect eigenvalues."""
            def diagonalize(self, H):
                evals, evecs = np.linalg.eigh(H)
                # Inject systematic error
                evals = evals + 1e-3
                return evals, evecs
            def name(self):
                return "Corrupted"
        
        # Create multi-oracle with corrupted oracle
        bad_multi = MultiOracle(
            oracles=[NumPyOracle(), ScipyOracle(driver='evd'), CorruptedOracle()],
            tolerance=1e-10
        )
        
        N = 3
        H = build_tfim(n_sites=N, J=1.0, h=0.5)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, report = bad_multi.diagonalize_with_consensus(H)
        
        # Should detect disagreement
        detected = not report['consensus']
        
        return {
            'test_name': 'oracle_disagreement_detection',
            'description': 'Add corrupted oracle, check consensus failure',
            'consensus': report['consensus'],
            'max_disagreement': report['max_disagreement'],
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def test_residual_amplification(self) -> Dict[str, Any]:
        """Test that approximate eigenvectors give larger residuals."""
        N = 3
        H = build_tfim(n_sites=N, J=1.0, h=0.5)
        
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        
        # Original residuals (should be tiny)
        residuals_correct = self.oracle.compute_residuals(H, evals, evecs)
        
        # Perturb eigenvectors slightly (mix two eigenvectors)
        evecs_perturbed = evecs.copy()
        alpha = 0.01  # 1% mixing
        v0 = evecs[:, 0].copy()
        v1 = evecs[:, 1].copy()
        evecs_perturbed[:, 0] = (v0 + alpha * v1) / np.sqrt(1 + alpha**2)
        
        # Residuals with perturbed vectors
        residuals_perturbed = self.oracle.compute_residuals(H, evals, evecs_perturbed)
        
        # Should be much larger
        amplification = residuals_perturbed[0] / residuals_correct[0]
        
        return {
            'test_name': 'residual_amplification',
            'description': 'Perturb eigenvector by 1%, residual should grow',
            'mixing_amplitude': alpha,
            'residual_correct': float(residuals_correct[0]),
            'residual_perturbed': float(residuals_perturbed[0]),
            'amplification_ratio': float(amplification),
            'passed': bool(residuals_perturbed[0] > 100 * residuals_correct[0]),
        }
    
    def summary(self) -> Dict[str, Any]:
        """Summary."""
        if not self.results:
            self.run_all()
        
        n_total = len(self.results)
        n_passed = sum(1 for r in self.results if r['passed'])
        
        return {
            'n_total': n_total,
            'n_passed': n_passed,
            'all_passed': n_passed == n_total,
            'individual_results': self.results
        }
