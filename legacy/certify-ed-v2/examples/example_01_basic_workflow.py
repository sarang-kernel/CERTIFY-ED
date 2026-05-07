"""
Example 1: Basic Workflow
=========================

Demonstrates the complete CERTIFY-ED workflow on the transverse-field Ising model.
"""

import numpy as np
from certify_ed import (
    build_tfim, MultiOracle, Certificate,
    ObservableCalculator, pauli_matrices, tensor_product
)


def main():
    print("=" * 70)
    print("CERTIFY-ED Example 1: Transverse-Field Ising Model")
    print("=" * 70)
    
    # Parameters
    N = 4
    J = 1.0
    h = 0.5
    
    print(f"\nModel: {N}-site TFIM, J={J}, h={h}")
    
    # 1. Build Hamiltonian
    print("\n[1] Building Hamiltonian...")
    H = build_tfim(n_sites=N, J=J, h=h, boundary='open')
    print(f"    Shape: {H.shape}, Hermitian: {np.allclose(H, H.conj().T)}")
    
    # 2. Diagonalize with consensus
    print("\n[2] Multi-oracle diagonalization...")
    oracle = MultiOracle()
    evals, evecs, report = oracle.diagonalize_with_consensus(H)
    print(f"    Consensus: {report['consensus']}")
    print(f"    Max disagreement: {report['max_disagreement']:.2e}")
    print(f"    Oracles: {', '.join(report['oracle_names'])}")
    
    # 3. Compute residuals
    print("\n[3] Residual verification...")
    residuals = oracle.compute_residuals(H, evals, evecs)
    print(f"    Max residual: {np.max(residuals):.2e}")
    print(f"    Mean residual: {np.mean(residuals):.2e}")
    
    # 4. Spectrum
    print("\n[4] Spectrum:")
    print(f"    Ground state:    E_0 = {evals[0]:.10f}")
    print(f"    First excited:   E_1 = {evals[1]:.10f}")
    print(f"    Spectral gap:    Δ   = {evals[1] - evals[0]:.10f}")
    
    # 5. Generate certificate
    print("\n[5] Generating certificate...")
    cert = Certificate(
        eigenvalues=evals,
        eigenvectors=evecs,
        hamiltonian=H,
        metadata={'model': 'TFIM', 'N': N, 'J': J, 'h': h},
        consensus_report=report
    )
    cert.save('tfim_example.json', include_eigenvectors=False)
    print(f"    Certificate saved: tfim_example.json")
    print(f"    Certified: {cert.is_certified}")
    
    # 6. Compute observables
    print("\n[6] Observables:")
    calc = ObservableCalculator(evals, evecs, residuals)
    
    # Total Z magnetization
    I, X, Y, Z = pauli_matrices()
    Mz = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        ops = [I] * N
        ops[i] = Z
        Mz += tensor_product(*ops)
    
    mag_gs = calc.expectation_value(Mz, 0)
    print(f"    Ground state <M_z>: {mag_gs:.6f}")
    
    print("\n" + "=" * 70)
    print(cert.summary())


if __name__ == '__main__':
    main()
