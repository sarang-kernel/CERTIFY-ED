"""
Verification Certificates
========================

This module provides tools for generating and validating verification certificates
for exact diagonalization computations.

Classes:
    Certificate: Verification certificate with residuals and metadata

Functions:
    load_certificate: Load certificate from JSON file
"""

import json
import hashlib
import numpy as np
from datetime import datetime
import platform
from typing import Dict, Any, Optional
import warnings


class Certificate:
    """
    Verification certificate for eigendecomposition.
    
    A certificate contains eigenvalues, eigenvectors, verification metrics
    (residuals, normalization errors), and computational metadata. Certificates
    can be saved to JSON for independent verification.
    
    Attributes:
        eigenvalues: Computed eigenvalues
        eigenvectors: Computed eigenvectors
        residuals: ||H|ψ_n⟩ - E_n|ψ_n⟩|| for each eigenpair
        normalization_errors: |⟨ψ_n|ψ_n⟩ - 1| for each eigenvector
        metadata: User-provided metadata (model description, parameters, etc.)
        
    Example:
        >>> H = build_tfim(n_sites=3, J=1.0, h=0.5)
        >>> oracle = MultiOracle()
        >>> evals, evecs, consensus = oracle.diagonalize_with_consensus(H)
        >>> cert = Certificate(evals, evecs, H, metadata={'model': 'TFIM'})
        >>> cert.save('tfim_certificate.json')
    """
    
    def __init__(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        hamiltonian_matrix: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        consensus_report: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize certificate.
        
        Args:
            eigenvalues: Eigenvalues array
            eigenvectors: Eigenvectors matrix (columns are eigenvectors)
            hamiltonian_matrix: Hamiltonian matrix H
            metadata: Optional metadata (model name, parameters, etc.)
            consensus_report: Optional multi-oracle consensus report
        """
        self.eigenvalues = eigenvalues.copy()
        self.eigenvectors = eigenvectors.copy()
        self.H = hamiltonian_matrix.copy()
        self.metadata = metadata or {}
        self.consensus_report = consensus_report
        
        # Compute verification metrics
        self.residuals = self._compute_residuals()
        self.normalization_errors = self._compute_normalization_errors()
        
        # Certification status
        self.is_certified = self._check_certification()
        
    def _compute_residuals(self) -> np.ndarray:
        """
        Compute ||H|ψ_n⟩ - E_n|ψ_n⟩|| for each eigenpair.
        
        Returns:
            Array of residual norms
        """
        n_eigenvalues = len(self.eigenvalues)
        residuals = np.zeros(n_eigenvalues)
        
        for i in range(n_eigenvalues):
            E_n = self.eigenvalues[i]
            psi_n = self.eigenvectors[:, i]
            
            residual_vector = self.H @ psi_n - E_n * psi_n
            residuals[i] = np.linalg.norm(residual_vector)
        
        return residuals
    
    def _compute_normalization_errors(self) -> np.ndarray:
        """
        Compute |⟨ψ_n|ψ_n⟩ - 1| for each eigenvector.
        
        Returns:
            Array of normalization errors
        """
        n_eigenvectors = self.eigenvectors.shape[1]
        errors = np.zeros(n_eigenvectors)
        
        for i in range(n_eigenvectors):
            psi_n = self.eigenvectors[:, i]
            norm_squared = np.vdot(psi_n, psi_n).real
            errors[i] = np.abs(norm_squared - 1.0)
        
        return errors
    
    def _check_certification(
        self,
        residual_threshold: float = 1e-10,
        normalization_threshold: float = 1e-12
    ) -> bool:
        """
        Check if eigendecomposition meets certification criteria.
        
        Args:
            residual_threshold: Maximum allowed residual
            normalization_threshold: Maximum allowed normalization error
            
        Returns:
            True if certified, False otherwise
        """
        max_residual = np.max(self.residuals)
        max_norm_error = np.max(self.normalization_errors)
        
        certified = (
            max_residual < residual_threshold and
            max_norm_error < normalization_threshold
        )
        
        if not certified:
            warnings.warn(
                f"Certification criteria not met:\n"
                f"  Max residual: {max_residual:.2e} "
                f"(threshold: {residual_threshold:.2e})\n"
                f"  Max normalization error: {max_norm_error:.2e} "
                f"(threshold: {normalization_threshold:.2e})"
            )
        
        return certified
    
    def to_dict(self, include_eigenvectors: bool = True) -> Dict[str, Any]:
        """
        Convert certificate to dictionary for serialization.
        
        Args:
            include_eigenvectors: If True, include full eigenvectors.
                                 If False, only include eigenvalues and metrics
                                 (reduces file size significantly)
                                 
        Returns:
            Dictionary representation of certificate
        """
        cert_dict = {
            'format_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': platform.system(),
                'machine': platform.machine(),
                'python_version': platform.python_version(),
                'numpy_version': np.__version__,
            },
            'spectrum': {
                'eigenvalues': self.eigenvalues.tolist(),
                'ground_state_energy': float(self.eigenvalues[0]),
                'first_excitation_energy': float(self.eigenvalues[1]) if len(self.eigenvalues) > 1 else None,
                'spectral_gap': float(self.eigenvalues[1] - self.eigenvalues[0]) if len(self.eigenvalues) > 1 else None,
            },
            'verification_metrics': {
                'residuals': self.residuals.tolist(),
                'normalization_errors': self.normalization_errors.tolist(),
                'max_residual': float(np.max(self.residuals)),
                'max_normalization_error': float(np.max(self.normalization_errors)),
                'mean_residual': float(np.mean(self.residuals)),
            },
            'certification': {
                'is_certified': self.is_certified,
                'certification_criteria': {
                    'residual_threshold': 1e-10,
                    'normalization_threshold': 1e-12,
                }
            },
            'metadata': self.metadata,
        }
        
        if include_eigenvectors:
            # Store eigenvectors (warning: large for big systems)
            cert_dict['eigenvectors'] = {
                'data': self.eigenvectors.tolist(),
                'shape': list(self.eigenvectors.shape),
                'note': 'Eigenvectors stored as columns'
            }
        
        if self.consensus_report is not None:
            # Remove large arrays from consensus report for JSON serialization
            consensus_summary = {
                k: v for k, v in self.consensus_report.items()
                if k not in ['all_eigenvalues', 'eigenvalue_differences']
            }
            cert_dict['consensus'] = consensus_summary
        
        return cert_dict
    
    def save(self, filename: str, include_eigenvectors: bool = True) -> None:
        """
        Save certificate to JSON file.
        
        Args:
            filename: Output filename (should end in .json)
            include_eigenvectors: If True, include eigenvectors (large files)
            
        Example:
            >>> cert.save('results.json', include_eigenvectors=False)
        """
        cert_dict = self.to_dict(include_eigenvectors=include_eigenvectors)
        
        # Compute SHA-256 hash for tamper detection
        cert_json = json.dumps(cert_dict, sort_keys=True, indent=2)
        sha256_hash = hashlib.sha256(cert_json.encode()).hexdigest()
        
        cert_dict['sha256'] = sha256_hash
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(cert_dict, f, indent=2)
    
    def summary(self) -> str:
        """
        Generate human-readable summary of certificate.
        
        Returns:
            Multi-line string with certificate summary
            
        Example:
            >>> print(cert.summary())
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CERTIFY-ED Verification Certificate")
        lines.append("=" * 60)
        
        # Spectrum
        lines.append("\nSpectrum:")
        lines.append(f"  Ground state energy: {self.eigenvalues[0]:.12f}")
        if len(self.eigenvalues) > 1:
            lines.append(f"  First excited state:  {self.eigenvalues[1]:.12f}")
            gap = self.eigenvalues[1] - self.eigenvalues[0]
            lines.append(f"  Spectral gap:         {gap:.12f}")
        lines.append(f"  Hilbert space dim:    {len(self.eigenvalues)}")
        
        # Verification metrics
        lines.append("\nVerification Metrics:")
        lines.append(f"  Max residual:         {np.max(self.residuals):.2e}")
        lines.append(f"  Mean residual:        {np.mean(self.residuals):.2e}")
        lines.append(f"  Max norm error:       {np.max(self.normalization_errors):.2e}")
        
        # Certification status
        lines.append("\nCertification Status:")
        status = "✓ CERTIFIED" if self.is_certified else "✗ NOT CERTIFIED"
        lines.append(f"  {status}")
        
        # Consensus (if available)
        if self.consensus_report is not None:
            lines.append("\nMulti-Oracle Consensus:")
            consensus = "✓ CONSENSUS" if self.consensus_report['consensus'] else "✗ DISAGREEMENT"
            lines.append(f"  {consensus}")
            lines.append(f"  Max disagreement:     {self.consensus_report['max_disagreement']:.2e}")
            lines.append(f"  Oracles:              {', '.join(self.consensus_report['oracle_names'])}")
        
        # Metadata
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return (
            f"Certificate(n_eigenvalues={len(self.eigenvalues)}, "
            f"certified={self.is_certified}, "
            f"max_residual={np.max(self.residuals):.2e})"
        )


def load_certificate(filename: str, verify_hash: bool = True) -> Dict[str, Any]:
    """
    Load certificate from JSON file.
    
    Args:
        filename: Certificate filename
        verify_hash: If True, verify SHA-256 hash for tamper detection
        
    Returns:
        Certificate data as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If hash verification fails
        
    Example:
        >>> cert_data = load_certificate('results.json')
        >>> eigenvalues = cert_data['spectrum']['eigenvalues']
    """
    with open(filename, 'r') as f:
        cert_dict = json.load(f)
    
    if verify_hash and 'sha256' in cert_dict:
        stored_hash = cert_dict.pop('sha256')
        
        # Recompute hash
        cert_json = json.dumps(cert_dict, sort_keys=True, indent=2)
        recomputed_hash = hashlib.sha256(cert_json.encode()).hexdigest()
        
        if stored_hash != recomputed_hash:
            raise ValueError(
                "Certificate hash mismatch - file may have been tampered with!\n"
                f"  Stored hash:     {stored_hash}\n"
                f"  Recomputed hash: {recomputed_hash}"
            )
    
    return cert_dict
