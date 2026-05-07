"""
Verification Certificates Module
================================

Generates exportable verification certificates with cryptographic integrity.
"""

import json
import hashlib
import numpy as np
import platform
from datetime import datetime
from typing import Dict, Any, Optional
import warnings


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Certificate:
    """
    Verification certificate for eigendecomposition results.
    
    Contains eigenvalues, eigenvectors, residuals, and metadata.
    Includes SHA-256 hash for tamper detection.
    """
    
    def __init__(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                 hamiltonian: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None,
                 consensus_report: Optional[Dict[str, Any]] = None):
        self.eigenvalues = np.asarray(eigenvalues)
        self.eigenvectors = np.asarray(eigenvectors)
        self.H = np.asarray(hamiltonian)
        self.metadata = metadata or {}
        self.consensus_report = consensus_report
        
        self.residuals = self._compute_residuals()
        self.normalization_errors = self._compute_norm_errors()
        self.is_certified = self._check_certification()
    
    def _compute_residuals(self) -> np.ndarray:
        n = len(self.eigenvalues)
        residuals = np.zeros(n)
        for i in range(n):
            psi = self.eigenvectors[:, i]
            residuals[i] = np.linalg.norm(self.H @ psi - self.eigenvalues[i] * psi)
        return residuals
    
    def _compute_norm_errors(self) -> np.ndarray:
        n = self.eigenvectors.shape[1]
        errors = np.zeros(n)
        for i in range(n):
            psi = self.eigenvectors[:, i]
            errors[i] = abs(np.vdot(psi, psi).real - 1.0)
        return errors
    
    def _check_certification(self, residual_threshold: float = 1e-10,
                              norm_threshold: float = 1e-12) -> bool:
        max_res = np.max(self.residuals)
        max_norm = np.max(self.normalization_errors)
        return max_res < residual_threshold and max_norm < norm_threshold
    
    def to_dict(self, include_eigenvectors: bool = True) -> Dict[str, Any]:
        """Convert certificate to dict for serialization."""
        d = {
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
                'n_eigenvalues': len(self.eigenvalues),
                'ground_state_energy': float(self.eigenvalues[0]),
                'spectral_gap': (float(self.eigenvalues[1] - self.eigenvalues[0])
                                if len(self.eigenvalues) > 1 else None),
            },
            'verification': {
                'residuals': self.residuals.tolist(),
                'max_residual': float(np.max(self.residuals)),
                'mean_residual': float(np.mean(self.residuals)),
                'normalization_errors': self.normalization_errors.tolist(),
                'max_norm_error': float(np.max(self.normalization_errors)),
            },
            'certification': {
                'is_certified': bool(self.is_certified),
                'residual_threshold': 1e-10,
                'norm_threshold': 1e-12,
            },
            'metadata': self.metadata,
        }
        
        if include_eigenvectors:
            d['eigenvectors_real'] = self.eigenvectors.real.tolist()
            d['eigenvectors_imag'] = self.eigenvectors.imag.tolist()
        
        if self.consensus_report:
            d['consensus'] = {k: v for k, v in self.consensus_report.items()
                              if k != 'all_eigenvalues'}
        
        return d
    
    def save(self, filename: str, include_eigenvectors: bool = True) -> None:
        """Save certificate to JSON file with SHA-256 hash."""
        d = self.to_dict(include_eigenvectors=include_eigenvectors)
        
        # Compute hash over sorted JSON
        cert_json = json.dumps(d, sort_keys=True, indent=2, cls=_NumpyJSONEncoder)
        d['sha256'] = hashlib.sha256(cert_json.encode()).hexdigest()
        
        with open(filename, 'w') as f:
            json.dump(d, f, indent=2, cls=_NumpyJSONEncoder)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "CERTIFY-ED Verification Certificate",
            "=" * 60,
            f"Eigenvalues:      {len(self.eigenvalues)}",
            f"Ground state:     {self.eigenvalues[0]:.12f}",
        ]
        if len(self.eigenvalues) > 1:
            lines.append(f"First excited:    {self.eigenvalues[1]:.12f}")
            lines.append(f"Spectral gap:     {self.eigenvalues[1] - self.eigenvalues[0]:.12f}")
        
        lines.extend([
            f"Max residual:     {np.max(self.residuals):.2e}",
            f"Max norm error:   {np.max(self.normalization_errors):.2e}",
            f"Certified:        {'YES' if self.is_certified else 'NO'}",
        ])
        
        if self.consensus_report:
            lines.append(f"Consensus:        {'YES' if self.consensus_report['consensus'] else 'NO'}")
            lines.append(f"Max disagreement: {self.consensus_report['max_disagreement']:.2e}")
        
        if self.metadata:
            lines.append("Metadata:")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"Certificate(n={len(self.eigenvalues)}, "
                f"certified={self.is_certified}, "
                f"max_res={np.max(self.residuals):.2e})")


def load_certificate(filename: str, verify_hash: bool = True) -> Dict[str, Any]:
    """
    Load certificate from JSON file with optional hash verification.
    
    Parameters
    ----------
    filename : str
        Path to certificate file.
    verify_hash : bool
        If True, verify SHA-256 hash and raise if invalid.
    
    Returns
    -------
    dict
        Certificate data.
    """
    with open(filename, 'r') as f:
        d = json.load(f)
    
    if verify_hash and 'sha256' in d:
        stored = d.pop('sha256')
        cert_json = json.dumps(d, sort_keys=True, indent=2, cls=_NumpyJSONEncoder)
        computed = hashlib.sha256(cert_json.encode()).hexdigest()
        
        if stored != computed:
            raise ValueError(
                f"Certificate hash mismatch!\n"
                f"  Stored:   {stored}\n"
                f"  Computed: {computed}"
            )
        d['sha256'] = stored  # restore
    
    return d
