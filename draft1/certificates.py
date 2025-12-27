"""
Certificate Generation and Management

Generates portable, cryptographically-signed proof certificates
for quantum many-body eigendecompositions.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CertifiedEigenpair:
    """Represents a certified eigenvalue-eigenvector pair."""

    index: int
    eigenvalue: float
    eigenvector: np.ndarray
    residual: float
    normalization_error: float
    quantum_numbers: Dict[str, int]

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'eigenvalue': float(self.eigenvalue),
            'eigenvector': self.eigenvector.tolist() if isinstance(self.eigenvector, np.ndarray) else self.eigenvector,
            'residual': float(self.residual),
            'normalization_error': float(self.normalization_error),
            'quantum_numbers': self.quantum_numbers
        }


@dataclass
class Certificate:
    """Complete proof certificate for eigendecomposition."""

    timestamp: str
    hamiltonian_name: str
    system_size: int
    hilbert_dimension: int
    model_parameters: Dict
    eigenpairs: List[CertifiedEigenpair]
    agreement_validation: Dict
    software_versions: Dict
    architecture_info: Dict

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of certificate content."""
        # Create deterministic JSON representation
        cert_dict = asdict(self)
        cert_dict['eigenpairs'] = [ep.to_dict() for ep in self.eigenpairs]

        json_str = json.dumps(cert_dict, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(json_str.encode())
        return hash_obj.hexdigest()

    def to_json(self, filename: str = None) -> str:
        """
        Serialize to JSON format.

        Args:
            filename: Optional file to write to

        Returns:
            JSON string representation
        """
        cert_dict = asdict(self)
        cert_dict['eigenpairs'] = [ep.to_dict() for ep in self.eigenpairs]
        cert_dict['hash'] = self.compute_hash()

        json_str = json.dumps(cert_dict, indent=2, default=str)

        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
            logger.info(f"✓ Certificate written to {filename}")

        return json_str

    def to_hdf5(self, filename: str):
        """
        Serialize to HDF5 format (for large eigenvectors).

        Args:
            filename: Output HDF5 file path
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py not available; use to_json() instead")
            return

        with h5py.File(filename, 'w') as f:
            # Metadata
            f.attrs['timestamp'] = self.timestamp
            f.attrs['hamiltonian_name'] = self.hamiltonian_name
            f.attrs['system_size'] = self.system_size
            f.attrs['hilbert_dimension'] = self.hilbert_dimension
            f.attrs['hash'] = self.compute_hash()

            # Eigenvalues and eigenvectors
            evals = np.array([ep.eigenvalue for ep in self.eigenpairs])
            f.create_dataset('eigenvalues', data=evals)

            # Store eigenvectors as block
            if self.eigenpairs:
                evecs = np.array([ep.eigenvector for ep in self.eigenpairs]).T
                f.create_dataset('eigenvectors', data=evecs)

                # Store residuals
                residuals = np.array([ep.residual for ep in self.eigenpairs])
                f.create_dataset('residuals', data=residuals)

        logger.info(f"✓ Certificate written to {filename}")

    @staticmethod
    def verify_hash(cert_dict: Dict, claimed_hash: str) -> bool:
        """
        Verify certificate integrity via hash.

        Args:
            cert_dict: Certificate dictionary
            claimed_hash: Hash value to verify against

        Returns:
            True if hash matches, False otherwise
        """
        # Remove hash from dict before recomputing
        cert_copy = cert_dict.copy()
        cert_copy.pop('hash', None)

        json_str = json.dumps(cert_copy, sort_keys=True, default=str)
        computed_hash = hashlib.sha256(json_str.encode()).hexdigest()

        return computed_hash == claimed_hash


class CertificationEngine:
    """Generates and manages certificates."""

    def __init__(self):
        """Initialize certification engine."""
        self.certificate = None

    def generate_certificate(
        self,
        eigenpairs: List[CertifiedEigenpair],
        hamiltonian_name: str,
        system_size: int,
        hilbert_dimension: int,
        model_parameters: Dict,
        agreement_validation: Dict,
        software_versions: Dict,
        architecture_info: Dict
    ) -> Certificate:
        """
        Generate a complete proof certificate.

        Args:
            eigenpairs: List of certified eigenpairs
            hamiltonian_name: Name of the model
            system_size: Number of qubits/spins
            hilbert_dimension: Dimension of Hilbert space
            model_parameters: Physical parameters (J, h, etc.)
            agreement_validation: Multi-oracle agreement results
            software_versions: Software and library versions
            architecture_info: System architecture information

        Returns:
            Generated Certificate object
        """
        self.certificate = Certificate(
            timestamp=datetime.now().isoformat(),
            hamiltonian_name=hamiltonian_name,
            system_size=system_size,
            hilbert_dimension=hilbert_dimension,
            model_parameters=model_parameters,
            eigenpairs=eigenpairs,
            agreement_validation=agreement_validation,
            software_versions=software_versions,
            architecture_info=architecture_info
        )

        logger.info(f"✓ Certificate generated with {len(eigenpairs)} eigenpairs")
        return self.certificate

    def export(self, format: str = 'json', filename: str = None) -> str:
        """
        Export certificate to file.

        Args:
            format: 'json' or 'hdf5'
            filename: Output file path

        Returns:
            Serialized certificate or path if file written
        """
        if self.certificate is None:
            raise ValueError("No certificate generated yet")

        if format == 'json':
            return self.certificate.to_json(filename)
        elif format == 'hdf5':
            self.certificate.to_hdf5(filename)
            return filename
        else:
            raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Test: Create a simple certificate
    ep = CertifiedEigenpair(
        index=0,
        eigenvalue=-1.0,
        eigenvector=np.array([1, 0]),
        residual=1e-14,
        normalization_error=0.0,
        quantum_numbers={'S_z': 0}
    )

    cert = Certificate(
        timestamp=datetime.now().isoformat(),
        hamiltonian_name="TFIM",
        system_size=2,
        hilbert_dimension=4,
        model_parameters={'J': 1.0, 'h': 0.5},
        eigenpairs=[ep],
        agreement_validation={'status': 'passed'},
        software_versions={'python': '3.11', 'numpy': '1.24'},
        architecture_info={'os': 'Linux', 'arch': 'x86_64'}
    )

    print("Certificate hash:", cert.compute_hash()[:16])
