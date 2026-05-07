"""
Example: Certifying a Two-Site Transverse-Field Ising Model

This example demonstrates the complete CERTIFY-ED workflow.
"""

import numpy as np
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import CERTIFY-ED
try:
    from certify_ed import (
        CertifiedHamiltonian,
        CertificationEngine
    )
except ImportError:
    print("Error: certify_ed package not found")
    print("Install with: pip install -e .")
    sys.exit(1)


def main():
    """Run certification example."""

    print("\n" + "="*70)
    print("CERTIFY-ED: Certifying Quantum Many-Body Systems")
    print("="*70)

    # 1. Create Hamiltonian
    print("\n[1] Creating Hamiltonian")
    print("    Model: Transverse-Field Ising Model")
    print("    System size: 2 qubits")
    print("    Parameters: J=1.0, h=0.5")

    ham = CertifiedHamiltonian.TFIM(L=2, J=1.0, h=0.5)

    # 2. Initialize certification engine
    print("\n[2] Initializing Certification Engine")
    engine = CertificationEngine(verbose=True)

    # 3. Run certification
    print("\n[3] Running Certification Pipeline")
    certificate = engine.certify_hamiltonian(
        ham,
        tolerance=1e-10,
        use_lanczos=False
    )

    if certificate is None:
        print("\n✗ Certification failed!")
        return

    # 4. Display results
    print("\n" + "="*70)
    print("CERTIFICATION RESULTS")
    print("="*70)

    print(f"\nSystem: {certificate.hamiltonian_name}")
    print(f"System size: {certificate.system_size} qubits")
    print(f"Hilbert dimension: {certificate.hilbert_dimension}")
    print(f"Number of certified eigenpairs: {len(certificate.eigenpairs)}")

    print("\nEigenvalue Spectrum:")
    print("-" * 50)
    print(f"{'Index':<6} {'Eigenvalue':<20} {'Residual':<15}")
    print("-" * 50)

    for ep in certificate.eigenpairs:
        print(f"{ep.index:<6} {ep.eigenvalue:<20.12f} {ep.residual:<15.2e}")

    # 5. Spectral properties
    if len(certificate.eigenpairs) >= 2:
        E_0 = certificate.eigenpairs[0].eigenvalue
        E_1 = certificate.eigenpairs[1].eigenvalue
        gap = E_1 - E_0

        print("\nSpectral Properties:")
        print("-" * 50)
        print(f"Ground state energy: {E_0:.12f}")
        print(f"First excited energy: {E_1:.12f}")
        print(f"Spectral gap: {gap:.12f}")

    # 6. Agreement validation
    print("\nMulti-Oracle Agreement:")
    print("-" * 50)
    agreement = certificate.agreement_validation
    print(f"Status: {agreement['status']}")
    print(f"Max difference: {agreement['max_difference']:.2e}")
    print(f"Mean difference: {agreement['mean_difference']:.2e}")
    print(f"Number disagreements: {agreement['num_disagreements']}")

    # 7. Export certificate
    print("\n[4] Exporting Certificate")

    json_str = engine.export_certificate(format='json')
    with open('certificate.json', 'w') as f:
        f.write(json_str)
    print("    ✓ Exported to certificate.json")

    print("\n" + "="*70)
    print("Certification Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
