"""
Conservation Law Validator
==========================

Tests that conservation laws [H, S] = 0 are satisfied numerically and
that eigenvectors are simultaneous eigenstates of H and S.

For each (model, symmetry) pair where conservation should hold:
    1. Verify [H, S] = 0 numerically
    2. Verify each eigenvector has well-defined S eigenvalue
    3. Verify spectrum is organized by S quantum number

Conservation pairs tested:
    - Heisenberg / XXZ + total Sz (U(1))
    - TFIM + parity (Z2)
    - Cluster model + parity (Z2)
    - Free fermion + fermion number (U(1))
    - Kitaev chain + fermion parity (Z2)
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from certify_ed import (
    build_model, MultiOracle,
    total_sz_operator, parity_operator, z_parity_operator,
    fermion_number_operator, fermion_parity_operator,
    check_conservation,
)


class ConservationLawValidator:
    """Validate conservation laws and quantum number assignments."""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.oracle = MultiOracle()
        self.results: List[Dict[str, Any]] = []
    
    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        # (model_name, model_kwargs, symmetry_name, symmetry_op_func, expected_to_conserve)
        # NOTE: cluster model with X-field does NOT conserve spin-flip parity P=prod X_i,
        # because XZX anticommutes with prod Z (Z-parity), but commutes with X (X-parity).
        # Wait: P = prod X_i. Under P, each X_i -> X_i (commutes), Z_i -> -Z_i.
        # So XZX -> X(-Z)X = -XZX. So [H_cluster, P_X] != 0.
        # The cluster Hamiltonian commutes with Z-parity (prod Z_i):
        # under prod Z, X_i -> -X_i, Z_i -> Z_i. So XZX -> (-X)Z(-X) = XZX. Good.
        # But the X-field term: -h*X_i -> -h*(-X_i) = +h*X_i. So [H_field, prod Z_i] != 0.
        # So cluster + X-field doesn't conserve either standard parity!
        # Switch test to cluster at h=0 with Z-parity instead.
        tests = [
            ('heisenberg', {'n_sites': 4}, 'total_Sz',
             lambda N: total_sz_operator(N), True),
            ('xxz', {'n_sites': 4, 'Delta': 0.7}, 'total_Sz',
             lambda N: total_sz_operator(N), True),
            ('tfim', {'n_sites': 4}, 'spin_flip_parity',
             lambda N: parity_operator(N), True),
            ('tfim', {'n_sites': 4}, 'total_Sz',
             lambda N: total_sz_operator(N), False),  # NOT conserved
            ('cluster', {'n_sites': 5, 'h': 0.0}, 'z_parity',
             lambda N: z_parity_operator(N), True),  # at h=0, conserves Z-parity
            ('free_fermion', {'n_sites': 4}, 'fermion_number',
             lambda N: fermion_number_operator(N), True),
            ('kitaev_chain', {'n_sites': 4, 't': 1.0, 'mu': 0.5, 'Delta': 0.7},
             'fermion_parity', lambda N: fermion_parity_operator(N), True),
        ]
        
        for spec in tests:
            self.results.append(self.test_conservation(*spec))
        return self.results
    
    def test_conservation(self, model_name: str, model_kwargs: Dict,
                          sym_name: str, sym_op_func, expected_conserved: bool
                          ) -> Dict[str, Any]:
        # Determine N from kwargs
        n_sites = model_kwargs.get('n_sites', 4)
        H = build_model(model_name, **model_kwargs)
        S = sym_op_func(n_sites)
        
        # Check commutator
        cons = check_conservation(H, S, tolerance=self.tolerance)
        
        # If conserved, verify that S is block-diagonal in H's eigenbasis,
        # blocked by H's degenerate subspaces. This is the correct test:
        # within a degenerate subspace of H, LAPACK returns an arbitrary
        # orthonormal basis, so individual eigenvectors need NOT be S-eigenstates.
        # But [H,S]=0 implies S is block-diagonal in any eigenbasis of H,
        # with blocks within each degenerate H-eigenspace.
        sector_check = None
        if cons['is_conserved']:
            evals, evecs, _ = self.oracle.diagonalize_with_consensus(H)
            d = len(evals)
            # Compute S in H-eigenbasis: V^dag S V
            S_in_H_basis = evecs.conj().T @ S @ evecs
            
            # Group eigenvalue indices by degenerate subspaces
            degen_groups = []
            current_group = [0]
            for i in range(1, d):
                if abs(evals[i] - evals[i-1]) < 1e-9:
                    current_group.append(i)
                else:
                    degen_groups.append(current_group)
                    current_group = [i]
            degen_groups.append(current_group)
            
            # Check that off-block entries of S_in_H_basis are zero
            max_off_block = 0.0
            for gi, group_i in enumerate(degen_groups):
                for gj, group_j in enumerate(degen_groups):
                    if gi == gj:
                        continue
                    for i in group_i:
                        for j in group_j:
                            max_off_block = max(max_off_block, abs(S_in_H_basis[i, j]))
            
            sector_check = {
                'eigenbasis_dimension': d,
                'n_distinct_energies': len(degen_groups),
                'n_degenerate_subspaces': sum(1 for g in degen_groups if len(g) > 1),
                'max_off_block_S': float(max_off_block),
                'S_block_diagonal_in_H_eigenbasis': bool(max_off_block < 1e-9),
            }
        
        # Pass if behavior matches expectation
        passed = (cons['is_conserved'] == expected_conserved)
        if expected_conserved and sector_check is not None:
            passed = passed and sector_check['S_block_diagonal_in_H_eigenbasis']
        
        return {
            'test_name': f'conservation_{model_name}_{sym_name}',
            'description': f'{model_name} with {sym_name} (expect conserved={expected_conserved})',
            'expected_conserved': expected_conserved,
            'commutator_norm': cons['commutator_norm'],
            'is_conserved': cons['is_conserved'],
            'sector_check': sector_check,
            'passed': bool(passed),
        }
    
    def summary(self) -> Dict[str, Any]:
        if not self.results:
            self.run_all()
        n = len(self.results)
        n_pass = sum(1 for r in self.results if r['passed'])
        return {
            'validator': 'ConservationLawValidator',
            'n_total': n,
            'n_passed': n_pass,
            'all_passed': n_pass == n,
            'individual_results': self.results,
        }
