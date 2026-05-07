"""Verification Certificates Module."""

import json
import hashlib
import numpy as np
import platform
from datetime import datetime
from typing import Dict, Any, Optional


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder handling numpy types."""
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
    """Verification certificate with SHA-256 integrity."""
    
    def __init__(self, eigenvalues, eigenvectors, hamiltonian,
                 metadata: Optional[Dict] = None,
                 consensus_report: Optional[Dict] = None):
        self.eigenvalues = np.asarray(eigenvalues)
        self.eigenvectors = np.asarray(eigenvectors)
        self.H = np.asarray(hamiltonian)
        self.metadata = metadata or {}
        self.consensus_report = consensus_report
        self.residuals = self._compute_residuals()
        self.normalization_errors = self._compute_norm_errors()
        self.is_certified = self._check_certification()
    
    def _compute_residuals(self):
        n = len(self.eigenvalues)
        r = np.zeros(n)
        for i in range(n):
            psi = self.eigenvectors[:, i]
            r[i] = np.linalg.norm(self.H @ psi - self.eigenvalues[i] * psi)
        return r
    
    def _compute_norm_errors(self):
        n = self.eigenvectors.shape[1]
        e = np.zeros(n)
        for i in range(n):
            psi = self.eigenvectors[:, i]
            e[i] = abs(np.vdot(psi, psi).real - 1.0)
        return e
    
    def _check_certification(self, residual_threshold=1e-10, norm_threshold=1e-12):
        return (np.max(self.residuals) < residual_threshold and
                np.max(self.normalization_errors) < norm_threshold)
    
    def to_dict(self, include_eigenvectors=True):
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
                              if k not in ('all_eigenvalues',)}
        return d
    
    def save(self, filename, include_eigenvectors=True):
        d = self.to_dict(include_eigenvectors=include_eigenvectors)
        cert_json = json.dumps(d, sort_keys=True, indent=2, cls=_NumpyJSONEncoder)
        d['sha256'] = hashlib.sha256(cert_json.encode()).hexdigest()
        with open(filename, 'w') as f:
            json.dump(d, f, indent=2, cls=_NumpyJSONEncoder)
    
    def summary(self):
        lines = [
            "=" * 60,
            "CERTIFY-ED Verification Certificate",
            "=" * 60,
            f"Eigenvalues:      {len(self.eigenvalues)}",
            f"Ground state:     {self.eigenvalues[0]:.12f}",
        ]
        if len(self.eigenvalues) > 1:
            lines.append(f"First excited:    {self.eigenvalues[1]:.12f}")
            lines.append(f"Spectral gap:     {self.eigenvalues[1]-self.eigenvalues[0]:.12f}")
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


def load_certificate(filename, verify_hash=True):
    with open(filename, 'r') as f:
        d = json.load(f)
    if verify_hash and 'sha256' in d:
        stored = d.pop('sha256')
        cert_json = json.dumps(d, sort_keys=True, indent=2, cls=_NumpyJSONEncoder)
        computed = hashlib.sha256(cert_json.encode()).hexdigest()
        if stored != computed:
            raise ValueError(f"Hash mismatch: {stored} vs {computed}")
        d['sha256'] = stored
    return d
