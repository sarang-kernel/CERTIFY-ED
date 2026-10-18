"""
Symmetries Module
=================

Build symmetry operators for conservation law verification and
symmetry-resolved spectrum analysis.

Provides:
    - Total Sz operator (U(1) spin symmetry)
    - Parity (Z2) operator
    - Translation operator (cyclic systems)
    - Fermion number operator
    - Spin parity (rotation by pi around X)
"""

import numpy as np
from typing import Tuple, List
from .hamiltonian import (
    pauli_matrices, tensor_product, site_operator,
    fermion_operators, jordan_wigner_operator,
)


# ============================================================================
# Spin-1/2 symmetry operators
# ============================================================================

def total_sz_operator(n_sites: int) -> np.ndarray:
    """
    Total S^z operator for n spin-1/2 sites.
    
    S^z_total = sum_i (Z_i / 2)
    
    Eigenvalues: integer multiples of 1/2 from -N/2 to +N/2.
    """
    I, _, _, Z = pauli_matrices()
    d = 2 ** n_sites
    Sz = np.zeros((d, d), dtype=complex)
    for i in range(n_sites):
        Sz += 0.5 * site_operator(Z, i, n_sites)
    return Sz


def total_sx_operator(n_sites: int) -> np.ndarray:
    """Total S^x operator."""
    I, X, _, _ = pauli_matrices()
    d = 2 ** n_sites
    Sx = np.zeros((d, d), dtype=complex)
    for i in range(n_sites):
        Sx += 0.5 * site_operator(X, i, n_sites)
    return Sx


def parity_operator(n_sites: int) -> np.ndarray:
    """
    Spin-flip parity P = prod_i X_i.
    
    Generator of Z_2 symmetry of TFIM and related models.
    Eigenvalues: ±1.
    """
    I, X, _, _ = pauli_matrices()
    return tensor_product(*[X] * n_sites)


def z_parity_operator(n_sites: int) -> np.ndarray:
    """
    Z-parity operator P_z = prod_i Z_i.
    
    For systems where Z is the conserved/measured direction.
    Eigenvalues: ±1.
    """
    I, _, _, Z = pauli_matrices()
    return tensor_product(*[Z] * n_sites)


def translation_operator(n_sites: int) -> np.ndarray:
    """
    Cyclic translation operator T: site i -> site i+1 (mod N).
    
    For a system with periodic boundary conditions.
    T |s_0, s_1, ..., s_{N-1}> = |s_{N-1}, s_0, s_1, ..., s_{N-2}>
    
    Eigenvalues: e^(2*pi*i*k/N) for k = 0, 1, ..., N-1.
    """
    d = 2 ** n_sites
    T = np.zeros((d, d), dtype=complex)
    
    for state in range(d):
        # Get bits [s_{N-1}, ..., s_1, s_0] using big-endian convention
        bits = [(state >> i) & 1 for i in range(n_sites)]
        # Translate: new state has site i + 1 = old site i (cyclic)
        # That is, new bits[i+1] = old bits[i]
        # Or equivalently: new_state shifts left by one position with wrap
        # We'll use the convention: T |s_0 s_1 ... s_{N-1}> = |s_{N-1} s_0 s_1 ... s_{N-2}>
        new_bits = [bits[(i - 1) % n_sites] for i in range(n_sites)]
        new_state = sum(new_bits[i] << i for i in range(n_sites))
        T[new_state, state] = 1.0
    
    return T


# ============================================================================
# Fermionic symmetry operators
# ============================================================================

def fermion_number_operator(n_sites: int) -> np.ndarray:
    """
    Total fermion number N = sum_i n_i = sum_i c^dag_i c_i.
    
    For Jordan-Wigner mapped fermions, N = sum_i (1 - Z_i)/2.
    """
    # n_i = (1 - Z_i)/2 for JW with |0>=spin up, |1>=spin down convention
    I, _, _, Z = pauli_matrices()
    d = 2 ** n_sites
    N_op = np.zeros((d, d), dtype=complex)
    half_I = 0.5 * np.eye(d, dtype=complex)
    for i in range(n_sites):
        N_op += half_I - 0.5 * site_operator(Z, i, n_sites)
    return N_op


def fermion_parity_operator(n_sites: int) -> np.ndarray:
    """
    Fermion parity (-1)^N. Useful for Kitaev chain analysis.
    """
    I, _, _, Z = pauli_matrices()
    return tensor_product(*[Z] * n_sites)


# ============================================================================
# Sector projection
# ============================================================================

def project_onto_sector(eigenvectors: np.ndarray, eigenvalues: np.ndarray,
                         symmetry_op: np.ndarray, target_value: float,
                         tolerance: float = 1e-8
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project eigenstates onto symmetry sector with given quantum number.
    
    For each eigenstate |psi_n>, computes <psi_n|S|psi_n> and selects
    those within tolerance of target_value.
    
    Returns (filtered_eigenvalues, filtered_eigenvectors).
    """
    n = len(eigenvalues)
    keep = []
    for i in range(n):
        psi = eigenvectors[:, i]
        s_val = np.vdot(psi, symmetry_op @ psi).real
        if abs(s_val - target_value) < tolerance:
            keep.append(i)
    
    if not keep:
        return np.array([]), np.zeros((eigenvectors.shape[0], 0), dtype=complex)
    
    return eigenvalues[keep], eigenvectors[:, keep]


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[A, B] = AB - BA."""
    return A @ B - B @ A


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Operator norm of [A, B], measures non-commutativity."""
    return float(np.linalg.norm(commutator(A, B), ord=2))


def check_conservation(H: np.ndarray, S: np.ndarray,
                       tolerance: float = 1e-12) -> dict:
    """
    Check if [H, S] = 0 (S is conserved).
    
    Returns dict with commutator norm and verdict.
    """
    comm = commutator(H, S)
    norm = float(np.linalg.norm(comm, ord=2))
    return {
        'commutator_norm': norm,
        'is_conserved': norm < tolerance,
        'tolerance': tolerance,
    }
