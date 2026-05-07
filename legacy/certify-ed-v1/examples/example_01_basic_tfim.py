"""
Example 1: Basic Usage - Transverse-Field Ising Model
====================================================

This example demonstrates the complete CERTIFY-ED workflow:
1. Build Hamiltonian symbolically
2. Verify Hermiticity
3. Diagonalize with multi-oracle consensus
4. Generate verification certificate
5. Compute physical observables
"""

import numpy as np
from certify_ed import (
    build_tfim,
    MultiOracle,
    Certificate,
    ObservableCalculator,
    pauli_matrices,
    tensor_product,
)


def main():
    print("=" * 70)
    print("CERTIFY-ED Example 1: Transverse-Field Ising Model")
    print("=" * 70)
    
    # Parameters
    n_sites = 4
    J = 1.0  # Ising coupling
    h = 0.5  # Transverse field
    
    print(f"\nModel: TFIM with {n_sites} sites")
    print(f"  H = -J * sum_i Z_i Z_{{i+1}} - h * sum_i X_i")
    print(f"  J = {J}")
    print(f"  h = {h}")
    print(f"  Hilbert space dimension: {2**n_sites}")
    
    # Step 1: Build Hamiltonian
    print("\n" + "-" * 70)
    print("Step 1: Building Hamiltonian...")
    H = build_tfim(n_sites=n_sites, J=J, h=h, boundary='open')
    print(f"  Matrix shape: {H.shape}")
    print(f"  Hermitian: {np.allclose(H, H.conj().T)}")
    
    # Step 2: Diagonalize with multi-oracle consensus
    print("\n" + "-" * 70)
    print("Step 2: Diagonalizing with multi-oracle consensus...")
    
    oracle = MultiOracle(tolerance=1e-10)
    eigenvalues, eigenvectors, consensus_report = oracle.diagonalize_with_consensus(H)
    
    print(f"  Consensus achieved: {consensus_report['consensus']}")
    print(f"  Max disagreement: {consensus_report['max_disagreement']:.2e}")
    print(f"  Oracles used: {', '.join(consensus_report['oracle_names'])}")
    
    # Step 3: Compute residuals
    print("\n" + "-" * 70)
    print("Step 3: Computing residuals...")
    
    residuals = oracle.compute_residuals(H, eigenvalues, eigenvectors)
    print(f"  Max residual: {np.max(residuals):.2e}")
    print(f"  Mean residual: {np.mean(residuals):.2e}")
    
    # Step 4: Display spectrum
    print("\n" + "-" * 70)
    print("Step 4: Energy spectrum:")
    print(f"  Ground state energy: {eigenvalues[0]:.12f}")
    print(f"  First excited state: {eigenvalues[1]:.12f}")
    print(f"  Spectral gap: {eigenvalues[1] - eigenvalues[0]:.12f}")
    
    print(f"\n  Full spectrum (first 8 eigenvalues):")
    for i in range(min(8, len(eigenvalues))):
        print(f"    E_{i} = {eigenvalues[i]:12.8f}  "
              f"(residual: {residuals[i]:.2e})")
    
    # Step 5: Generate certificate
    print("\n" + "-" * 70)
    print("Step 5: Generating verification certificate...")
    
    metadata = {
        'model': 'Transverse-Field Ising Model',
        'n_sites': n_sites,
        'parameters': {'J': J, 'h': h},
        'boundary_conditions': 'open',
    }
    
    cert = Certificate(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        hamiltonian_matrix=H,
        metadata=metadata,
        consensus_report=consensus_report
    )
    
    # Save certificate
    cert.save('tfim_example_certificate.json', include_eigenvectors=False)
    print(f"  Certificate saved to: tfim_example_certificate.json")
    print(f"  Certification status: {'✓ CERTIFIED' if cert.is_certified else '✗ NOT CERTIFIED'}")
    
    # Step 6: Compute observables
    print("\n" + "-" * 70)
    print("Step 6: Computing physical observables...")
    
    calc = ObservableCalculator(eigenvalues, eigenvectors, residuals)
    
    # Total magnetization in Z direction
    I, X, Y, Z = pauli_matrices()
    
    # Build total S^z operator
    Sz_total = np.zeros((2**n_sites, 2**n_sites), dtype=complex)
    for site in range(n_sites):
        ops = [I] * n_sites
        ops[site] = Z
        Sz_total += 0.5 * tensor_product(*ops)  # S^z = Z/2
    
    mag_ground = calc.expectation_value(Sz_total, state_index=0)
    print(f"  Ground state magnetization ⟨S^z_total⟩: {mag_ground:.6f}")
    
    # Verify ground state energy
    E0_check = calc.expectation_value(H, state_index=0)
    print(f"  Ground state energy (direct): {E0_check:.12f}")
    print(f"  Ground state energy (eigenvalue): {eigenvalues[0]:.12f}")
    print(f"  Difference: {abs(E0_check - eigenvalues[0]):.2e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(cert.summary())
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
