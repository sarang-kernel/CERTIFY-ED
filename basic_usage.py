"""
Basic Usage Example for CERTIFY-ED

Demonstrates the complete 5-phase workflow for a simple system.

Author: Sarang Vehale
"""

from certifyEd import (
    CertifiedHamiltonian,
    MultiOracleDiagonalizer,
    ObservableCalculator,
)


def main():
    print("\n" + "=" * 80)
    print("CERTIFY-ED: Basic Usage Example")
    print("=" * 80)

    # Phase 1: Create and verify Hamiltonian
    print("\nPhase 1: Symbolic Construction...")
    H = CertifiedHamiltonian.TFIM(L=4, J=1.0, h=0.5)
    H.verify_hermiticity()
    print("✓ Hermiticity verified symbolically")

    # Phase 2-3: Diagonalize with multi-oracle consensus
    print("\nPhases 2-3: Multi-Oracle Diagonalization + Consensus...")
    diag = MultiOracleDiagonalizer(H, epsilon=1e-12)
    results = diag.diagonalize()
    print(f"✓ Agreement validated: {results.agreement_validated}")

    # Get agreement metrics
    metrics = diag.get_agreement_metrics()
    print(f"✓ Max disagreement: {metrics['max_disagreement']:.2e}")

    # Phase 4: Generate certificate
    print("\nPhase 4: Certificate Generation...")
    cert = results.generate_certificate()
    cert.export_json('example_certificate.json')
    print("✓ Certificate exported to example_certificate.json")

    # Phase 5: Compute observables
    print("\nPhase 5: Observable Validation...")
    obs_calc = ObservableCalculator(results)
    summary = obs_calc.summary()

    print(f"\nResults:")
    print(f"  Ground state energy: {summary['ground_state_energy']:.15f}")
    print(f"  Error bound:         {summary['energy_error_bound']:.2e}")
    print(f"  Spectral gap:        {summary['spectral_gap']:.15f}")

    print("\n" + "=" * 80)
    print("✓ All 5 phases completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
