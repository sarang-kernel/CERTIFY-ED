"""
Example 2: Validation Against Exact Solutions
=============================================

Validates CERTIFY-ED against systems with known analytic eigenvalues.
"""

import numpy as np
from certify_ed import build_heisenberg, build_tfim, MultiOracle


def validate_2site_heisenberg():
    print("\n--- 2-site Heisenberg Chain ---")
    print("Exact: E_singlet = -3J/4, E_triplet = J/4 (3-fold)")
    
    J = 1.0
    H = build_heisenberg(n_sites=2, J=J)
    
    oracle = MultiOracle()
    evals, evecs, report = oracle.diagonalize_with_consensus(H)
    
    expected = np.array([-3*J/4, J/4, J/4, J/4])
    errors = np.abs(np.sort(evals) - np.sort(expected))
    
    print(f"  Eigenvalues: {sorted(evals)}")
    print(f"  Max error:   {np.max(errors):.2e}")
    print(f"  Consensus:   {report['consensus']}")
    
    assert np.max(errors) < 1e-13, f"Test failed: error = {np.max(errors)}"
    print("  STATUS: PASS")


def validate_3site_heisenberg_pbc():
    print("\n--- 3-site Heisenberg PBC ---")
    print("Exact: E_0 = -3J/4 (S_total=1/2 sector)")
    
    J = 1.0
    H = build_heisenberg(n_sites=3, J=J, boundary='periodic')
    
    oracle = MultiOracle()
    evals, evecs, report = oracle.diagonalize_with_consensus(H)
    
    expected_ground = -3 * J / 4
    error = abs(evals[0] - expected_ground)
    
    print(f"  Ground state: {evals[0]:.15f}")
    print(f"  Expected:     {expected_ground}")
    print(f"  Error:        {error:.2e}")
    
    assert error < 1e-13, f"Test failed: error = {error}"
    print("  STATUS: PASS")


def validate_classical_ising():
    print("\n--- TFIM at h=0 (classical Ising) ---")
    print("Exact: E_0 = -J*(N-1) for open BC")
    
    N = 4
    J = 1.0
    H = build_tfim(n_sites=N, J=J, h=0.0, boundary='open')
    
    oracle = MultiOracle()
    evals, evecs, report = oracle.diagonalize_with_consensus(H)
    
    expected_ground = -J * (N - 1)
    error = abs(evals[0] - expected_ground)
    
    print(f"  Ground state: {evals[0]:.10f}")
    print(f"  Expected:     {expected_ground}")
    print(f"  Error:        {error:.2e}")
    
    # Also check ground state degeneracy
    gap = evals[1] - evals[0]
    print(f"  E_1 - E_0:    {gap:.2e} (should be ~0 for degenerate GS)")
    
    assert error < 1e-13, f"Test failed: error = {error}"
    print("  STATUS: PASS")


def main():
    print("=" * 70)
    print("CERTIFY-ED Example 2: Analytic Validation")
    print("=" * 70)
    
    validate_2site_heisenberg()
    validate_3site_heisenberg_pbc()
    validate_classical_ising()
    
    print("\n" + "=" * 70)
    print("All validations passed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
