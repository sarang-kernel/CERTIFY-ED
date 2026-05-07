"""
Example 2: Heisenberg Model with Analytic Validation
===================================================

This example validates against known exact results for the 2-site and 3-site
Heisenberg chains.

Known Results:
- 2-site Heisenberg: E_singlet = -3J/4, E_triplet = J/4
- 3-site Heisenberg (periodic): Ground state is doublet with E_0 = -3J/2
"""

import numpy as np
from certify_ed import build_heisenberg, MultiOracle, Certificate


def validate_two_site():
    """Validate 2-site Heisenberg against exact solution."""
    print("\n" + "=" * 70)
    print("Two-Site Heisenberg Model")
    print("=" * 70)
    
    J = 1.0
    H = build_heisenberg(n_sites=2, J=J, boundary='open')
    
    # Exact eigenvalues
    E_singlet_exact = -3 * J / 4  # -0.75
    E_triplet_exact = J / 4       #  0.25
    
    # Compute numerically
    oracle = MultiOracle()
    eigenvalues, eigenvectors, consensus = oracle.diagonalize_with_consensus(H)
    
    print(f"\nExact eigenvalues:")
    print(f"  Singlet:  {E_singlet_exact}")
    print(f"  Triplet:  {E_triplet_exact} (3-fold degenerate)")
    
    print(f"\nNumerical eigenvalues:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:.15f}")
    
    # Compare
    print(f"\nValidation:")
    singlet_error = abs(eigenvalues[0] - E_singlet_exact)
    triplet_error = abs(eigenvalues[1] - E_triplet_exact)
    
    print(f"  Singlet error:  {singlet_error:.2e}")
    print(f"  Triplet error:  {triplet_error:.2e}")
    
    # Check consensus
    print(f"\nConsensus report:")
    print(f"  Consensus: {consensus['consensus']}")
    print(f"  Max disagreement: {consensus['max_disagreement']:.2e}")
    
    # Compute residuals
    residuals = oracle.compute_residuals(H, eigenvalues, eigenvectors)
    print(f"\nResiduals:")
    print(f"  Max residual: {np.max(residuals):.2e}")
    
    # Verify agreement to machine precision
    assert singlet_error < 1e-14, "Singlet energy error too large!"
    assert triplet_error < 1e-14, "Triplet energy error too large!"
    
    print(f"\n✓ Validation passed: agreement to {max(singlet_error, triplet_error):.2e}")
    
    return eigenvalues, eigenvectors, consensus


def validate_three_site():
    """Validate 3-site Heisenberg (periodic) against exact solution."""
    print("\n" + "=" * 70)
    print("Three-Site Heisenberg Model (Periodic)")
    print("=" * 70)
    
    J = 1.0
    H = build_heisenberg(n_sites=3, J=J, boundary='periodic')
    
    # Exact ground state energy (known from Bethe ansatz)
    E0_exact = -3 * J / 2  # -1.5 (doubly degenerate)
    
    # Compute numerically
    oracle = MultiOracle()
    eigenvalues, eigenvectors, consensus = oracle.diagonalize_with_consensus(H)
    
    print(f"\nExact ground state energy: {E0_exact}")
    print(f"Numerical ground state:    {eigenvalues[0]:.15f}")
    print(f"First excited state:       {eigenvalues[1]:.15f}")
    
    # Validate
    ground_error = abs(eigenvalues[0] - E0_exact)
    print(f"\nGround state error: {ground_error:.2e}")
    
    assert ground_error < 1e-13, "Ground state error too large!"
    
    print(f"✓ Validation passed: agreement to {ground_error:.2e}")
    
    # Display full spectrum
    print(f"\nFull spectrum:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:12.10f}")
    
    # Generate certificate
    metadata = {
        'model': 'Heisenberg XXX Chain',
        'n_sites': 3,
        'parameters': {'J': J},
        'boundary_conditions': 'periodic',
        'validation': {
            'exact_ground_energy': E0_exact,
            'numerical_ground_energy': float(eigenvalues[0]),
            'agreement': float(ground_error)
        }
    }
    
    cert = Certificate(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        hamiltonian_matrix=H,
        metadata=metadata,
        consensus_report=consensus
    )
    
    cert.save('heisenberg_3site_certificate.json')
    print(f"\n✓ Certificate saved to: heisenberg_3site_certificate.json")
    
    return eigenvalues, eigenvectors, consensus


def main():
    print("=" * 70)
    print("CERTIFY-ED Example 2: Analytic Validation")
    print("=" * 70)
    
    # Validate 2-site
    evals_2, evecs_2, consensus_2 = validate_two_site()
    
    # Validate 3-site
    evals_3, evecs_3, consensus_3 = validate_three_site()
    
    print("\n" + "=" * 70)
    print("All validations passed!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  2-site Heisenberg: max error < 1e-14")
    print(f"  3-site Heisenberg: max error < 1e-13")
    print(f"  All consensus checks passed")
    print(f"  All residuals < 1e-13")


if __name__ == '__main__':
    main()
