"""
Hamiltonian Construction and Symbolic Verification
=================================================

This module provides tools for constructing quantum Hamiltonians symbolically
and verifying their Hermiticity before numerical computation.

Classes:
    SymbolicHamiltonian: Symbolic Hamiltonian with verification

Functions:
    build_tfim: Construct transverse-field Ising model
    build_heisenberg: Construct Heisenberg model
    pauli_matrices: Get Pauli matrices in various formats
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings


def pauli_matrices(as_complex: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Return Pauli matrices (I, X, Y, Z).
    
    Args:
        as_complex: If True, return complex arrays. If False, return real where possible.
        
    Returns:
        Tuple of (I, X, Y, Z) as 2x2 numpy arrays
        
    Example:
        >>> I, X, Y, Z = pauli_matrices()
        >>> np.allclose(X @ X, I)
        True
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex if as_complex else float)
    X = np.array([[0, 1], [1, 0]], dtype=complex if as_complex else float)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex if as_complex else float)
    
    return I, X, Y, Z


def tensor_product(*operators: np.ndarray) -> np.ndarray:
    """
    Compute tensor product of operators.
    
    Args:
        *operators: Variable number of operators to tensor together
        
    Returns:
        Tensor product as numpy array
        
    Example:
        >>> I, X, Y, Z = pauli_matrices()
        >>> ZZ = tensor_product(Z, Z)
        >>> ZZ.shape
        (4, 4)
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


class SymbolicHamiltonian:
    """
    Symbolic Hamiltonian construction with Hermiticity verification.
    
    This class constructs quantum Hamiltonians from symbolic operator specifications
    and verifies Hermiticity before numerical diagonalization.
    
    Attributes:
        n_sites: Number of sites/qubits
        hilbert_dim: Dimension of Hilbert space (2^n_sites)
        matrix: Numerical Hamiltonian matrix
        is_hermitian: Whether H = H†
        terms: List of (coefficient, operators) defining the Hamiltonian
        
    Example:
        >>> # Construct 3-site TFIM: H = -J*sum(Z_i Z_{i+1}) - h*sum(X_i)
        >>> ham = SymbolicHamiltonian(n_sites=3)
        >>> ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
        >>> ham.add_term(-1.0, [('Z', 1), ('Z', 2)])
        >>> ham.add_term(-0.5, [('X', 0)])
        >>> ham.add_term(-0.5, [('X', 1)])
        >>> ham.add_term(-0.5, [('X', 2)])
        >>> ham.build()
        >>> ham.verify_hermiticity()
        True
    """
    
    def __init__(self, n_sites: int):
        """
        Initialize Hamiltonian.
        
        Args:
            n_sites: Number of sites (qubits)
            
        Raises:
            ValueError: If n_sites < 1 or too large (>20)
        """
        if n_sites < 1:
            raise ValueError("Number of sites must be at least 1")
        if n_sites > 20:
            warnings.warn(
                f"Large Hilbert space (2^{n_sites} = {2**n_sites}). "
                "This may consume significant memory."
            )
            
        self.n_sites = n_sites
        self.hilbert_dim = 2 ** n_sites
        self.matrix: Optional[np.ndarray] = None
        self.is_hermitian: Optional[bool] = None
        self.terms: List[Tuple[complex, List[Tuple[str, int]]]] = []
        
    def add_term(
        self,
        coefficient: complex,
        operators: List[Tuple[str, int]]
    ) -> None:
        """
        Add a term to the Hamiltonian.
        
        Args:
            coefficient: Coupling constant (can be complex)
            operators: List of (operator_name, site_index) tuples
                      operator_name in {'I', 'X', 'Y', 'Z'}
                      site_index in range(n_sites)
                      
        Raises:
            ValueError: If operator name invalid or site index out of range
            
        Example:
            >>> ham = SymbolicHamiltonian(n_sites=2)
            >>> ham.add_term(-1.0, [('Z', 0), ('Z', 1)])  # -Z_0 Z_1
            >>> ham.add_term(-0.5, [('X', 0)])             # -0.5 X_0
        """
        # Validate operators
        valid_ops = {'I', 'X', 'Y', 'Z'}
        for op_name, site_idx in operators:
            if op_name not in valid_ops:
                raise ValueError(
                    f"Invalid operator '{op_name}'. Must be one of {valid_ops}"
                )
            if not (0 <= site_idx < self.n_sites):
                raise ValueError(
                    f"Site index {site_idx} out of range [0, {self.n_sites})"
                )
        
        self.terms.append((coefficient, operators))
        
    def build(self) -> np.ndarray:
        """
        Build the Hamiltonian matrix from terms.
        
        Returns:
            Hamiltonian matrix (complex array)
            
        Raises:
            ValueError: If no terms have been added
            
        Example:
            >>> ham = SymbolicHamiltonian(n_sites=2)
            >>> ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
            >>> H = ham.build()
            >>> H.shape
            (4, 4)
        """
        if not self.terms:
            raise ValueError("No terms added to Hamiltonian")
        
        I, X, Y, Z = pauli_matrices()
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Initialize Hamiltonian matrix
        H = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
        
        # Build each term
        for coeff, operators in self.terms:
            # Start with identity on all sites
            op_list = [I for _ in range(self.n_sites)]
            
            # Replace with specified operators
            for op_name, site_idx in operators:
                op_list[site_idx] = pauli_dict[op_name]
            
            # Tensor product
            term_matrix = tensor_product(*op_list)
            
            # Add to Hamiltonian
            H += coeff * term_matrix
        
        self.matrix = H
        return H
    
    def verify_hermiticity(self, tolerance: float = 1e-14) -> bool:
        """
        Verify that H = H†.
        
        Args:
            tolerance: Numerical tolerance for Hermiticity check
            
        Returns:
            True if Hermitian, False otherwise
            
        Raises:
            ValueError: If matrix has not been built yet
            
        Example:
            >>> ham = SymbolicHamiltonian(n_sites=2)
            >>> ham.add_term(-1.0, [('Z', 0), ('Z', 1)])
            >>> ham.build()
            >>> ham.verify_hermiticity()
            True
        """
        if self.matrix is None:
            raise ValueError("Must call build() before verifying Hermiticity")
        
        # Compute H - H†
        diff = self.matrix - self.matrix.conj().T
        max_deviation = np.max(np.abs(diff))
        
        self.is_hermitian = max_deviation < tolerance
        
        if not self.is_hermitian:
            warnings.warn(
                f"Hamiltonian is not Hermitian! "
                f"Max |H - H†| = {max_deviation:.2e} > tolerance {tolerance:.2e}"
            )
        
        return self.is_hermitian
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the Hamiltonian matrix.
        
        Returns:
            Hamiltonian matrix
            
        Raises:
            ValueError: If matrix has not been built
        """
        if self.matrix is None:
            raise ValueError("Must call build() before getting matrix")
        return self.matrix.copy()
    
    def __repr__(self) -> str:
        return (
            f"SymbolicHamiltonian(n_sites={self.n_sites}, "
            f"hilbert_dim={self.hilbert_dim}, "
            f"n_terms={len(self.terms)}, "
            f"hermitian={self.is_hermitian})"
        )


def build_tfim(
    n_sites: int,
    J: float = 1.0,
    h: float = 0.5,
    boundary: str = "open"
) -> np.ndarray:
    """
    Build transverse-field Ising model Hamiltonian.
    
    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    
    Args:
        n_sites: Number of sites
        J: Ising coupling constant
        h: Transverse field strength
        boundary: 'open' or 'periodic'
        
    Returns:
        Hamiltonian matrix (complex array)
        
    Example:
        >>> H = build_tfim(n_sites=3, J=1.0, h=0.5)
        >>> H.shape
        (8, 8)
        >>> # Verify it's Hermitian
        >>> np.allclose(H, H.conj().T)
        True
    """
    ham = SymbolicHamiltonian(n_sites)
    
    # ZZ interactions
    for i in range(n_sites - 1):
        ham.add_term(-J, [('Z', i), ('Z', i + 1)])
    
    # Periodic boundary condition
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(-J, [('Z', n_sites - 1), ('Z', 0)])
    
    # Transverse field
    for i in range(n_sites):
        ham.add_term(-h, [('X', i)])
    
    H = ham.build()
    
    # Verify Hermiticity
    if not ham.verify_hermiticity():
        raise ValueError("TFIM Hamiltonian is not Hermitian (this should never happen)")
    
    return H


def build_heisenberg(
    n_sites: int,
    J: float = 1.0,
    boundary: str = "open",
    anisotropy: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Build Heisenberg model Hamiltonian.
    
    H = J * sum_i (J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1})
    
    Args:
        n_sites: Number of sites
        J: Overall coupling constant
        boundary: 'open' or 'periodic'
        anisotropy: Dict with keys 'Jx', 'Jy', 'Jz' (default: all 1.0 for XXX model)
        
    Returns:
        Hamiltonian matrix (complex array)
        
    Example:
        >>> # Isotropic XXX model
        >>> H = build_heisenberg(n_sites=3, J=1.0)
        >>> H.shape
        (8, 8)
        
        >>> # Anisotropic XXZ model
        >>> H = build_heisenberg(n_sites=3, J=1.0, anisotropy={'Jx': 1.0, 'Jy': 1.0, 'Jz': 2.0})
    """
    if anisotropy is None:
        anisotropy = {'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0}
    
    Jx = anisotropy.get('Jx', 1.0)
    Jy = anisotropy.get('Jy', 1.0)
    Jz = anisotropy.get('Jz', 1.0)
    
    ham = SymbolicHamiltonian(n_sites)
    
    # Nearest-neighbor interactions
    for i in range(n_sites - 1):
        ham.add_term(J * Jx, [('X', i), ('X', i + 1)])
        ham.add_term(J * Jy, [('Y', i), ('Y', i + 1)])
        ham.add_term(J * Jz, [('Z', i), ('Z', i + 1)])
    
    # Periodic boundary condition
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(J * Jx, [('X', n_sites - 1), ('X', 0)])
        ham.add_term(J * Jy, [('Y', n_sites - 1), ('Y', 0)])
        ham.add_term(J * Jz, [('Z', n_sites - 1), ('Z', 0)])
    
    H = ham.build()
    
    # Verify Hermiticity
    if not ham.verify_hermiticity():
        raise ValueError("Heisenberg Hamiltonian is not Hermitian (this should never happen)")
    
    return H
