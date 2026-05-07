"""
Example 3: Symmetry-resolved spectrum analysis
==============================================

For models with conserved quantities, decompose the Hilbert space
into symmetry sectors and verify spectrum decomposition.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certify_ed import (
    build_model, MultiOracle,
    total_sz_operator, parity_operator, check_conservation,
)


def main():
    print('=' * 60)
    print('Example 3: Symmetry-resolved spectra')
    print('=' * 60)

    # Heisenberg model conserves total Sz
    N = 4
    H = build_model('heisenberg', n_sites=N, J=1.0)
    Sz = total_sz_operator(N)

    # Check conservation
    cons = check_conservation(H, Sz)
    print(f'\nHeisenberg N={N}:')
    print(f'   |[H, Sz]| = {cons["commutator_norm"]:.2e}  '
          f'(conserved: {cons["is_conserved"]})')

    # Diagonalize Sz to get sectors
    sz_evals, sz_evecs = np.linalg.eigh(Sz)
    distinct_sz = np.unique(np.round(sz_evals, 8))
    print(f'   Sz sectors found: {distinct_sz.tolist()}')

    # For each Sz sector, project H and diagonalize
    oracle = MultiOracle()
    print(f'\n{"Sz":>5}  {"dim":>4}  {"E_min":>14}  {"E_max":>14}')
    print('-' * 42)
    all_sector_eigs = []
    for sz in distinct_sz:
        mask = np.abs(sz_evals - sz) < 1e-9
        basis = sz_evecs[:, mask]
        if basis.shape[1] == 0:
            continue
        H_proj = basis.conj().T @ H @ basis
        H_proj = (H_proj + H_proj.conj().T) / 2
        eigs = np.linalg.eigvalsh(H_proj)
        all_sector_eigs.extend(eigs.tolist())
        print(f'{sz:>5.1f}  {basis.shape[1]:>4d}  '
              f'{float(eigs[0]):>14.10f}  {float(eigs[-1]):>14.10f}')

    # Check union of sectors equals full spectrum
    full_eigs, _, _ = oracle.diagonalize_with_consensus(H)
    diff = np.max(np.abs(np.sort(full_eigs) - np.sort(all_sector_eigs)))
    print(f'\nUnion-of-sectors vs full spectrum: max diff = {diff:.2e}')

    # TFIM conserves spin-flip parity
    print(f'\n{"-"*60}')
    print(f'TFIM N={N} parity sector decomposition:')
    H = build_model('tfim', n_sites=N)
    P = parity_operator(N)
    cons = check_conservation(H, P)
    print(f'   |[H, P]| = {cons["commutator_norm"]:.2e}  '
          f'(conserved: {cons["is_conserved"]})')

    p_evals, p_evecs = np.linalg.eigh(P)
    for parity in (1.0, -1.0):
        mask = np.abs(p_evals - parity) < 1e-9
        basis = p_evecs[:, mask]
        if basis.shape[1] == 0:
            continue
        H_proj = basis.conj().T @ H @ basis
        H_proj = (H_proj + H_proj.conj().T) / 2
        eigs = np.linalg.eigvalsh(H_proj)
        sym_label = '+' if parity > 0 else '-'
        print(f'   Parity {sym_label}: {basis.shape[1]} states, '
              f'E_0 = {float(eigs[0]):.10f}')


if __name__ == '__main__':
    main()
