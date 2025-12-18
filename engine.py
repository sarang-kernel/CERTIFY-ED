"""
Main CERTIFY-ED Certification Pipeline

Orchestrates the complete certification workflow from Hamiltonian
construction through certificate generation.
"""

import numpy as np
import logging
import platform
import sys

from .hamiltonian import CertifiedHamiltonian
from .diagonalizer import MultiOracleDiagonalizer
from .certificates import Certificate, CertifiedEigenpair, CertificationEngine as CertEngine
from .observables import ObservableCalculator

logger = logging.getLogger(__name__)


class CertificationEngine:
    """Complete certification pipeline orchestrator."""

    def __init__(self, verbose: bool = True):
        """
        Initialize certification engine.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.hamiltonian = None
        self.diagonalizer = None
        self.diag_result = None
        self.cert_engine = CertEngine()
        self.observable_calc = ObservableCalculator()

        if verbose:
            logging.getLogger('certify_ed').setLevel(logging.INFO)

    def certify_hamiltonian(
        self,
        ham: CertifiedHamiltonian,
        tolerance: float = 1e-10,
        use_lanczos: bool = False
    ) -> Certificate:
        """
        Run complete certification pipeline.

        Args:
            ham: CertifiedHamiltonian instance
            tolerance: Agreement tolerance between oracles
            use_lanczos: Use Lanczos for large sparse systems

        Returns:
            Generated Certificate
        """
        logger.info("="*60)
        logger.info("CERTIFY-ED Certification Pipeline")
        logger.info("="*60)

        self.hamiltonian = ham

        # Step 1: Verify Hamiltonian
        logger.info("
[1] Hamiltonian Verification")
        logger.info(f"    Model: {ham.name}")
        logger.info(f"    System size: {ham.system_size}")
        logger.info(f"    Hilbert dimension: {ham.hilbert_dim}")

        if not ham.verification_results.get('hermiticity', False):
            logger.warning("⚠ Hermiticity verification failed!")
            return None

        logger.info("    ✓ Hermiticity verified")

        # Step 2: Convert to numeric
        logger.info("
[2] Numeric Conversion")
        H_numeric = ham.to_numeric()
        logger.info(f"    Shape: {H_numeric.shape}")
        logger.info(f"    Condition number: {np.linalg.cond(H_numeric):.2e}")

        # Step 3: Multi-oracle diagonalization
        logger.info("
[3] Multi-Oracle Diagonalization")
        self.diagonalizer = MultiOracleDiagonalizer(
            tolerance=tolerance,
            use_lanczos=use_lanczos
        )
        self.diag_result = self.diagonalizer.diagonalize(H_numeric)

        # Step 4: Validate agreement
        logger.info("
[4] Consensus Validation")
        agreement = self.diag_result['agreement_validation']
        logger.info(f"    Status: {agreement['status']}")
        logger.info(f"    Max difference: {agreement['max_difference']:.2e}")
        logger.info(f"    Mean difference: {agreement['mean_difference']:.2e}")

        if agreement['status'] != 'passed':
            logger.warning("⚠ Agreement validation failed")

        # Step 5: Generate certificates
        logger.info("
[5] Certificate Generation")
        eigenpairs = self._generate_eigenpairs(
            H_numeric,
            self.diag_result['consensus_eigenvalues'],
            self.diag_result['eigenvectors']
        )

        # Step 6: Get software/architecture info
        software_versions = {
            'python': f"{sys.version_info.major}.{sys.version_info.minor}",
            'numpy': np.__version__,
        }

        architecture_info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }

        # Step 7: Generate certificate
        certificate = self.cert_engine.generate_certificate(
            eigenpairs=eigenpairs,
            hamiltonian_name=ham.name,
            system_size=ham.system_size,
            hilbert_dimension=ham.hilbert_dim,
            model_parameters={},  # Fill from ham
            agreement_validation=agreement,
            software_versions=software_versions,
            architecture_info=architecture_info
        )

        logger.info(f"    ✓ Generated {len(eigenpairs)} certified eigenpairs")
        logger.info(f"    Certificate hash: {certificate.compute_hash()[:16]}...")

        # Step 8: Observable validation
        logger.info("
[6] Observable Validation")
        if len(eigenpairs) >= 2:
            obs_results = self.observable_calc.compute_all_standard(
                H_numeric,
                [(ep.eigenvalue, ep.eigenvector, ep.residual) for ep in eigenpairs],
                H_norm=np.linalg.norm(H_numeric, ord=2)
            )
            logger.info(f"    ✓ Computed {len(obs_results)} observables with error bounds")

        logger.info("
" + "="*60)
        logger.info("Certification Complete")
        logger.info("="*60)

        return certificate

    def _generate_eigenpairs(
        self,
        H: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray
    ) -> list:
        """
        Generate CertifiedEigenpair objects with residuals.

        Args:
            H: Hamiltonian matrix
            eigenvalues: Eigenvalues
            eigenvectors: Eigenvectors (columns)

        Returns:
            List of CertifiedEigenpair objects
        """
        eigenpairs = []

        for i, E_i in enumerate(eigenvalues):
            psi_i = eigenvectors[:, i]

            # Compute residual: ||H|ψ⟩ - E|ψ⟩||
            H_psi = H @ psi_i
            residual_vec = H_psi - E_i * psi_i
            residual = np.linalg.norm(residual_vec)

            # Compute normalization error
            norm = np.linalg.norm(psi_i)
            norm_error = abs(norm - 1.0)

            ep = CertifiedEigenpair(
                index=i,
                eigenvalue=float(E_i),
                eigenvector=psi_i,
                residual=float(residual),
                normalization_error=float(norm_error),
                quantum_numbers={}  # Can be extended
            )

            eigenpairs.append(ep)

        return eigenpairs

    def export_certificate(
        self,
        format: str = 'json',
        filename: str = None
    ) -> str:
        """
        Export generated certificate.

        Args:
            format: 'json' or 'hdf5'
            filename: Output file path

        Returns:
            Serialized certificate or filename
        """
        if self.cert_engine.certificate is None:
            raise ValueError("No certificate generated yet")

        return self.cert_engine.export(format=format, filename=filename)


if __name__ == "__main__":
    # Test: Certify 2-site TFIM
    logger.info("Testing CERTIFY-ED certification pipeline")

    engine = CertificationEngine(verbose=True)
    ham = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)

    cert = engine.certify_hamiltonian(ham)

    if cert:
        print("
✓ Certification successful!")
        print(f"Ground state energy: {cert.eigenpairs[0].eigenvalue:.10f}")
