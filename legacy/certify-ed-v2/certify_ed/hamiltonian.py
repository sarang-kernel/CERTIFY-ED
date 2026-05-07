"""
Hamiltonian Construction Module
==============================

Provides symbolic Hamiltonian construction with Hermiticity verification
for quantum many-body systems.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return Pauli matrices (I, X, Y, Z) as 2x2 complex arrays.
    
    Returns
    -------
    I, X, Y, Z : np.ndarray
        The four 2x2 Pauli matrices.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def tensor_product(*operators: np.ndarray) -> np.ndarray:
    """
    Compute tensor (Kronecker) product of operators.
    
    Parameters
    ----------
    *operators : np.ndarray
        Operators to tensor together.
    
    Returns
    -------
    np.ndarray
        Tensor product result.
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


class SymbolicHamiltonian:
    """
    Symbolic Hamiltonian with Hermiticity verification.
    
    Builds Hamiltonians from operator term specifications and verifies
    Hermiticity before returning the matrix.
    
    Parameters
    ----------
    n_sites : int
        Number of sites (qubits).
    
    Attributes
    ----------
    n_sites : int
        Number of sites.
    hilbert_dim : int
        Hilbert space dimension (2^n_sites).
    matrix : np.ndarray or None
        Hamiltonian matrix (after build()).
    is_hermitian : bool or None
        Whether matrix is Hermitian (after verify_hermiticity()).
    """
    
    VALID_OPERATORS = {'I', 'X', 'Y', 'Z'}
    
    def __init__(self, n_sites: int):
        if n_sites < 1:
            raise ValueError(f"n_sites must be >= 1, got {n_sites}")
        if n_sites > 16:
            warnings.warn(
                f"Large system: n_sites={n_sites} gives "
                f"d=2^{n_sites}={2**n_sites}. Memory usage may be high."
            )
        
        self.n_sites = n_sites
        self.hilbert_dim = 2 ** n_sites
        self.matrix: Optional[np.ndarray] = None
        self.is_hermitian: Optional[bool] = None
        self.terms: List[Tuple[complex, List[Tuple[str, int]]]] = []
    
    def add_term(self, coefficient: complex, 
                 operators: List[Tuple[str, int]]) -> None:
        """
        Add a term to the Hamiltonian.
        
        Parameters
        ----------
        coefficient : complex
            Coupling constant.
        operators : list of tuple
            List of (operator_name, site_index) tuples.
            operator_name is 'I', 'X', 'Y', or 'Z'.
        """
        for op_name, site_idx in operators:
            if op_name not in self.VALID_OPERATORS:
                raise ValueError(
                    f"Invalid operator '{op_name}'. "
                    f"Must be one of {self.VALID_OPERATORS}"
                )
            if not (0 <= site_idx < self.n_sites):
                raise ValueError(
                    f"Site index {site_idx} out of range [0, {self.n_sites})"
                )
        
        self.terms.append((coefficient, operators))
    
    def build(self) -> np.ndarray:
        """
        Build the Hamiltonian matrix.
        
        Returns
        -------
        np.ndarray
            Hamiltonian matrix of shape (2^n_sites, 2^n_sites).
        """
        if not self.terms:
            raise ValueError("No terms added to Hamiltonian")
        
        I, X, Y, Z = pauli_matrices()
        op_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        H = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
        
        for coeff, op_list in self.terms:
            site_ops = [I] * self.n_sites
            for op_name, site_idx in op_list:
                site_ops[site_idx] = op_map[op_name]
            
            H += coeff * tensor_product(*site_ops)
        
        self.matrix = H
        return H
    
    def verify_hermiticity(self, tolerance: float = 1e-14) -> bool:
        """
        Verify that H = H^dagger.
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed deviation.
        
        Returns
        -------
        bool
            True if Hermitian within tolerance.
        """
        if self.matrix is None:
            raise RuntimeError("Must call build() before verify_hermiticity()")
        
        deviation = np.max(np.abs(self.matrix - self.matrix.conj().T))
        self.is_hermitian = deviation < tolerance
        
        if not self.is_hermitian:
            warnings.warn(
                f"Hamiltonian not Hermitian: max|H - H^dag| = {deviation:.2e}"
            )
        
        return self.is_hermitian
    
    def __repr__(self) -> str:
        return (f"SymbolicHamiltonian(n_sites={self.n_sites}, "
                f"d={self.hilbert_dim}, terms={len(self.terms)})")


def build_tfim(n_sites: int, J: float = 1.0, h: float = 0.5,
               boundary: str = "open") -> np.ndarray:
    """
    Build transverse-field Ising model Hamiltonian.
    
    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    
    Parameters
    ----------
    n_sites : int
        Number of sites.
    J : float
        Ising coupling strength.
    h : float
        Transverse field strength.
    boundary : str
        'open' or 'periodic'.
    
    Returns
    -------
    np.ndarray
        TFIM Hamiltonian matrix.
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got '{boundary}'")
    
    ham = SymbolicHamiltonian(n_sites)
    
    # ZZ interactions
    for i in range(n_sites - 1):
        ham.add_term(-J, [('Z', i), ('Z', i + 1)])
    
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(-J, [('Z', n_sites - 1), ('Z', 0)])
    
    # Transverse field
    for i in range(n_sites):
        ham.add_term(-h, [('X', i)])
    
    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_heisenberg(n_sites: int, J: float = 1.0,
                     boundary: str = "open") -> np.ndarray:
    """
    Build isotropic Heisenberg (XXX) Hamiltonian.
    
    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) / 4
    
    Note: Factor of 1/4 makes this S=1/2 spin operators (S = sigma/2).
    
    Parameters
    ----------
    n_sites : int
        Number of sites.
    J : float
        Coupling strength.
    boundary : str
        'open' or 'periodic'.
    
    Returns
    -------
    np.ndarray
        Heisenberg Hamiltonian matrix.
    """
    return build_xxz(n_sites, J=J, Delta=1.0, boundary=boundary)


def build_xxz(n_sites: int, J: float = 1.0, Delta: float = 1.0,
              boundary: str = "open") -> np.ndarray:
    """
    Build XXZ Hamiltonian.
    
    H = (J/4) * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1})
    
    Parameters
    ----------
    n_sites : int
        Number of sites.
    J : float
        XY coupling strength.
    Delta : float
        Anisotropy parameter (Delta=1 gives isotropic Heisenberg).
    boundary : str
        'open' or 'periodic'.
    
    Returns
    -------
    np.ndarray
        XXZ Hamiltonian matrix.
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic'")
    
    ham = SymbolicHamiltonian(n_sites)
    
    # Factor of 1/4 from S = sigma/2
    Jq = J / 4.0
    
    for i in range(n_sites - 1):
        ham.add_term(Jq, [('X', i), ('X', i + 1)])
        ham.add_term(Jq, [('Y', i), ('Y', i + 1)])
        ham.add_term(Jq * Delta, [('Z', i), ('Z', i + 1)])
    
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(Jq, [('X', n_sites - 1), ('X', 0)])
        ham.add_term(Jq, [('Y', n_sites - 1), ('Y', 0)])
        ham.add_term(Jq * Delta, [('Z', n_sites - 1), ('Z', 0)])
    
    H = ham.build()
    ham.verify_hermiticity()
    return H
