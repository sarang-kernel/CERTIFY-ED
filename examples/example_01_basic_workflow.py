"""
Example 1: Basic CERTIFY-ED workflow
====================================

Build a Hamiltonian, diagonalize with multi-oracle consensus,
verify, and save a certificate.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certify_ed import build_model, MultiOracle, Certificate


def main():
    print('=' * 60)
    print('Example 1: Basic CERTIFY-ED workflow')
    print('=' * 60)

    # Step 1: Build a Hamiltonian
    print('\nStep 1: Build TFIM Hamiltonian (N=4, J=1, h=0.5)')
    H = build_model('tfim', n_sites=4, J=1.0, h=0.5)
    print(f'   Hilbert dim: {H.shape[0]}')

    # Step 2: Diagonalize with multi-oracle consensus
    print('\nStep 2: Multi-oracle consensus diagonalization')
    oracle = MultiOracle()
    print(f'   {oracle}')
    eigenvalues, eigenvectors, report = oracle.diagonalize_with_consensus(H)
    print(f'   Consensus: {report["consensus"]}')
    print(f'   Max disagreement between oracles: {report["max_disagreement"]:.2e}')
    print(f'   Ground state energy: {eigenvalues[0]:.10f}')
    print(f'   Spectral gap:        {eigenvalues[1] - eigenvalues[0]:.10f}')

    # Step 3: Build a certificate
    print('\nStep 3: Build verification certificate')
    cert = Certificate(eigenvalues, eigenvectors, H,
                       metadata={'model': 'tfim', 'N': 4, 'J': 1.0, 'h': 0.5},
                       consensus_report=report)
    print(cert.summary())

    # Step 4: Save certificate to disk
    output_path = 'tfim_n4_certificate.json'
    cert.save(output_path)
    print(f'\nCertificate saved to {output_path}')

    # Step 5: Verify the saved certificate by loading it
    from certify_ed import load_certificate
    loaded = load_certificate(output_path, verify_hash=True)
    print(f'Hash verified: {"sha256" in loaded}')


if __name__ == '__main__':
    main()
