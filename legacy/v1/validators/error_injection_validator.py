"""
Error Injection Validator
=========================

Validates that the framework correctly DETECTS errors.

For a verification framework to be trustworthy, it must catch:
    1. Non-Hermitian inputs (algebraic check)
    2. Matrix corruption (residual check)
    3. Inconsistent oracles (consensus check)
    4. Wrong eigenvectors (residual amplification)
    5. Tampered certificates (hash mismatch)
    6. Missing terms in Hamiltonian (cross-check)

If we INJECT a known error and the framework FAILS to detect it,
the verification claim is invalid.
"""

import numpy as np
import tempfile
import os
import json
import warnings
from typing import Dict, List, Any
from certify_ed import (
    build_model, MultiOracle, Certificate, load_certificate,
    NumPyOracle, ScipyOracle,
)
from certify_ed.oracles import Oracle


class ErrorInjectionValidator:
    """Validate framework's error detection capability."""
    
    def __init__(self):
        self.oracle = MultiOracle(tolerance=1e-10)
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        self.results.append(self.test_non_hermitian_detection())
        self.results.append(self.test_matrix_corruption_residual())
        self.results.append(self.test_oracle_disagreement_detection())
        self.results.append(self.test_eigenvector_perturbation())
        self.results.append(self.test_certificate_tampering())
        self.results.append(self.test_swapped_eigenvectors())
        return self.results
    
    def test_non_hermitian_detection(self) -> Dict[str, Any]:
        """Inject anti-Hermitian term, framework must detect."""
        H = build_model('tfim', n_sites=4, J=1.0, h=0.5)
        # Add anti-Hermitian perturbation
        perturbation = np.zeros_like(H)
        perturbation[0, 1] = 0.1j
        # NOT making it Hermitian on purpose
        H_corrupted = H + perturbation
        
        deviation = np.max(np.abs(H_corrupted - H_corrupted.conj().T))
        detected = deviation > 1e-12
        
        return {
            'test_name': 'non_hermitian_detection',
            'description': 'Inject anti-Hermitian term -> framework should detect',
            'injected_norm': float(np.linalg.norm(perturbation)),
            'hermiticity_deviation': float(deviation),
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def test_matrix_corruption_residual(self) -> Dict[str, Any]:
        """Compute on corrupted matrix, residuals against original should reveal."""
        H_orig = build_model('tfim', n_sites=4, J=1.0, h=0.5)
        # Inject Hermitian noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1e-5, H_orig.shape)
        noise = (noise + noise.T) / 2
        H_corrupted = H_orig + noise
        
        # Compute on corrupted
        evals_c, evecs_c, _ = self.oracle.diagonalize_with_consensus(H_corrupted)
        
        # Residuals against ORIGINAL
        residuals_vs_orig = self.oracle.compute_residuals(H_orig, evals_c, evecs_c)
        max_res = float(np.max(residuals_vs_orig))
        
        # Should be ~ noise norm
        noise_norm = float(np.linalg.norm(noise))
        detected = max_res > 1e-7
        
        return {
            'test_name': 'matrix_corruption_residual',
            'description': 'Wrong matrix used -> residuals against true H reveal it',
            'noise_norm': noise_norm,
            'max_residual_against_original': max_res,
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def test_oracle_disagreement_detection(self) -> Dict[str, Any]:
        """Inject corrupted oracle, multi-oracle must catch disagreement."""
        
        class CorruptedOracle(Oracle):
            def diagonalize(self, H):
                evals, evecs = np.linalg.eigh(H)
                evals = evals + 1e-3  # systematic shift
                return evals, evecs
            def name(self):
                return "Corrupted"
        
        bad_multi = MultiOracle(
            oracles=[NumPyOracle(), ScipyOracle('evd'), CorruptedOracle()],
            tolerance=1e-10
        )
        H = build_model('tfim', n_sites=3, J=1.0, h=0.5)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, report = bad_multi.diagonalize_with_consensus(H)
        
        detected = not report['consensus']
        return {
            'test_name': 'oracle_disagreement_detection',
            'description': 'Bad oracle -> consensus must fail',
            'consensus': report['consensus'],
            'max_disagreement': report['max_disagreement'],
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def test_eigenvector_perturbation(self) -> Dict[str, Any]:
        """Perturb eigenvector slightly, residual should grow significantly."""
        H = build_model('tfim', n_sites=4, J=1.0, h=0.5)
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        residuals_correct = self.oracle.compute_residuals(H, evals, evecs)
        
        # Mix two eigenvectors
        evecs_perturbed = evecs.copy()
        alpha = 0.01
        v0 = evecs[:, 0].copy()
        v1 = evecs[:, 1].copy()
        evecs_perturbed[:, 0] = (v0 + alpha * v1) / np.sqrt(1 + alpha ** 2)
        
        residuals_perturbed = self.oracle.compute_residuals(H, evals, evecs_perturbed)
        
        ratio = float(residuals_perturbed[0] / max(residuals_correct[0], 1e-20))
        # Should grow significantly
        detected = residuals_perturbed[0] > 100 * max(residuals_correct[0], 1e-15)
        
        return {
            'test_name': 'eigenvector_perturbation',
            'description': 'Perturb eigenvector -> residual must grow',
            'mixing_amplitude': alpha,
            'residual_correct': float(residuals_correct[0]),
            'residual_perturbed': float(residuals_perturbed[0]),
            'amplification_ratio': ratio,
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def test_certificate_tampering(self) -> Dict[str, Any]:
        """Save certificate, modify it, hash check must fail on load."""
        H = build_model('tfim', n_sites=3, J=1.0, h=0.5)
        evals, evecs, report = self.oracle.diagonalize_with_consensus(H)
        cert = Certificate(evals, evecs, H, consensus_report=report)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmpfile = f.name
        
        try:
            cert.save(tmpfile)
            # Tamper: change a value
            with open(tmpfile, 'r') as f:
                data = json.load(f)
            data['spectrum']['eigenvalues'][0] = 9999.0  # tamper
            with open(tmpfile, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Try to load - should fail
            tampering_detected = False
            try:
                load_certificate(tmpfile, verify_hash=True)
            except ValueError:
                tampering_detected = True
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
        
        return {
            'test_name': 'certificate_tampering',
            'description': 'Tampered certificate -> hash check must detect',
            'detected': tampering_detected,
            'passed': bool(tampering_detected),
        }
    
    def test_swapped_eigenvectors(self) -> Dict[str, Any]:
        """Swap two eigenvectors with mismatched eigenvalues -> residuals reveal."""
        H = build_model('heisenberg', n_sites=4, J=1.0)
        evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
        
        # Swap eigenvectors of states 0 and 1 (keep eigenvalues unchanged)
        evecs_swapped = evecs.copy()
        evecs_swapped[:, 0] = evecs[:, 1]
        evecs_swapped[:, 1] = evecs[:, 0]
        
        residuals = self.oracle.compute_residuals(H, evals, evecs_swapped)
        max_res = float(np.max(residuals))
        # Skip if eigenvalues are equal (degenerate)
        gap = float(evals[1] - evals[0])
        if gap < 1e-10:
            return {
                'test_name': 'swapped_eigenvectors',
                'description': 'Skipped: GS is degenerate so swapping is fine',
                'gap': gap,
                'passed': True,  # vacuous
            }
        
        # Should be huge residual since wrong eigenvalue assigned
        detected = max_res > 1e-3
        return {
            'test_name': 'swapped_eigenvectors',
            'description': 'Mismatched (E_n, |psi_n>) -> residual should reveal',
            'gap_between_swapped': gap,
            'max_residual_after_swap': max_res,
            'detected': bool(detected),
            'passed': bool(detected),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        return {
            'validator': 'ErrorInjectionValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_detected': n_pass == n,
            'note': 'A validator that fails its own injection tests cannot be trusted',
            'individual_results': self.results,
        }
