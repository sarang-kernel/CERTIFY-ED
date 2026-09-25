"""
Hamiltonian Construction Module
===============================

Symbolic Hamiltonian construction with Hermiticity verification for
quantum many-body systems. Includes a comprehensive library of physical
models commonly used as benchmarks in computational physics.

Models provided:
    Spin systems:
        - TFIM (Transverse-Field Ising Model)
        - Heisenberg (XXX, XXZ, XYZ)
        - SSH chain (dimerized hopping)
        - J1-J2 chain (frustrated Heisenberg)
        - AKLT chain (spin-1)
        - Haldane chain (spin-1)
    Fermionic systems:
        - Hubbard model (1D and 2D)
        - t-J model
        - Free fermions (tight-binding)
    Bosonic systems:
        - Bose-Hubbard model (constrained)
    Topological:
        - Kitaev chain (p-wave superconductor)
        - Kitaev honeycomb (small clusters)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


# ============================================================================
# Single-site operators
# ============================================================================

def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Pauli matrices (I, X, Y, Z) as 2x2 complex arrays."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def spin_half_operators() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return spin-1/2 operators (I, Sx, Sy, Sz, S+, S-)."""
    I = np.eye(2, dtype=complex)
    Sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    Sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    Sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    return I, Sx, Sy, Sz, Sp, Sm


def spin_one_operators() -> Tuple[np.ndarray, ...]:
    """Return spin-1 operators (I, Sx, Sy, Sz, S+, S-)."""
    I = np.eye(3, dtype=complex)
    Sz = np.diag([1, 0, -1]).astype(complex)
    Sp = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=complex)
    Sm = Sp.conj().T
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2j)
    return I, Sx, Sy, Sz, Sp, Sm


def fermion_operators() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fermionic operators (c, c_dag, n, F) for one mode.
    
    F is the Jordan-Wigner string operator (-1)^n.
    Uses the convention |0> = empty, |1> = occupied.
    """
    c = np.array([[0, 1], [0, 0]], dtype=complex)      # annihilation
    c_dag = np.array([[0, 0], [1, 0]], dtype=complex)  # creation
    n = c_dag @ c                                        # number operator
    F = np.diag([1, -1]).astype(complex)                 # JW string (-1)^n
    return c, c_dag, n, F


# ============================================================================
# Tensor product utilities
# ============================================================================

def tensor_product(*operators: np.ndarray) -> np.ndarray:
    """Compute tensor (Kronecker) product of operators."""
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


def site_operator(op: np.ndarray, site: int, n_sites: int,
                  local_dim: int = 2) -> np.ndarray:
    """Embed single-site operator at given site in n_sites system."""
    I = np.eye(local_dim, dtype=complex)
    factors = [I] * n_sites
    factors[site] = op
    return tensor_product(*factors)


def jordan_wigner_operator(c_or_cdag: np.ndarray, F: np.ndarray,
                            site: int, n_sites: int) -> np.ndarray:
    """Apply Jordan-Wigner transformation: c_j -> F_0 F_1 ... F_{j-1} c_j."""
    factors = []
    for k in range(n_sites):
        if k < site:
            factors.append(F)
        elif k == site:
            factors.append(c_or_cdag)
        else:
            factors.append(np.eye(2, dtype=complex))
    return tensor_product(*factors)


# ============================================================================
# Symbolic Hamiltonian Class
# ============================================================================

class SymbolicHamiltonian:
    """
    Build Hamiltonians symbolically with Hermiticity verification.
    
    Supports spin-1/2 systems with Pauli operators (I, X, Y, Z).
    """
    
    VALID_OPERATORS = {'I', 'X', 'Y', 'Z'}
    
    def __init__(self, n_sites: int):
        if n_sites < 1:
            raise ValueError(f"n_sites must be >= 1, got {n_sites}")
        if n_sites > 16:
            warnings.warn(f"Large system: 2^{n_sites} = {2**n_sites} dim")
        
        self.n_sites = n_sites
        self.hilbert_dim = 2 ** n_sites
        self.matrix: Optional[np.ndarray] = None
        self.is_hermitian: Optional[bool] = None
        self.terms: List[Tuple[complex, List[Tuple[str, int]]]] = []
    
    def add_term(self, coefficient: complex,
                 operators: List[Tuple[str, int]]) -> None:
        for op_name, site_idx in operators:
            if op_name not in self.VALID_OPERATORS:
                raise ValueError(f"Invalid operator '{op_name}'")
            if not (0 <= site_idx < self.n_sites):
                raise ValueError(f"Site index {site_idx} out of range")
        self.terms.append((coefficient, operators))
    
    def build(self) -> np.ndarray:
        if not self.terms:
            raise ValueError("No terms added")
        
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
        if self.matrix is None:
            raise RuntimeError("Must call build() first")
        deviation = np.max(np.abs(self.matrix - self.matrix.conj().T))
        self.is_hermitian = deviation < tolerance
        if not self.is_hermitian:
            warnings.warn(f"Not Hermitian: dev={deviation:.2e}")
        return self.is_hermitian
    
    def __repr__(self) -> str:
        return (f"SymbolicHamiltonian(n_sites={self.n_sites}, "
                f"d={self.hilbert_dim}, terms={len(self.terms)})")


# ============================================================================
# Spin-1/2 Models
# ============================================================================

def build_tfim(n_sites: int, J: float = 1.0, h: float = 0.5,
               boundary: str = "open") -> np.ndarray:
    """
    Transverse-Field Ising Model.
    
    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    
    Parameters
    ----------
    n_sites : int
    J : float
        Ising coupling.
    h : float
        Transverse field.
    boundary : str
        'open' or 'periodic'.
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic'")
    
    ham = SymbolicHamiltonian(n_sites)
    for i in range(n_sites - 1):
        ham.add_term(-J, [('Z', i), ('Z', i + 1)])
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(-J, [('Z', n_sites - 1), ('Z', 0)])
    for i in range(n_sites):
        ham.add_term(-h, [('X', i)])
    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_heisenberg(n_sites: int, J: float = 1.0,
                     boundary: str = "open") -> np.ndarray:
    """
    Isotropic Heisenberg (XXX) model.
    
    H = J sum_<ij> S_i . S_j  where S = sigma/2.
    
    In Pauli operators: H = (J/4) sum (X_iX_j + Y_iY_j + Z_iZ_j).
    """
    return build_xxz(n_sites, J=J, Delta=1.0, boundary=boundary)


def build_xxz(n_sites: int, J: float = 1.0, Delta: float = 1.0,
              boundary: str = "open") -> np.ndarray:
    """
    XXZ model.
    
    H = (J/4) sum_<ij> (X_i X_j + Y_i Y_j + Delta * Z_i Z_j)
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic'")
    
    ham = SymbolicHamiltonian(n_sites)
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


def build_xyz(n_sites: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0,
              boundary: str = "open") -> np.ndarray:
    """
    XYZ model with three independent couplings.
    
    H = (1/4) sum_<ij> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j)
    """
    ham = SymbolicHamiltonian(n_sites)
    
    for i in range(n_sites - 1):
        ham.add_term(Jx / 4, [('X', i), ('X', i + 1)])
        ham.add_term(Jy / 4, [('Y', i), ('Y', i + 1)])
        ham.add_term(Jz / 4, [('Z', i), ('Z', i + 1)])
    
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(Jx / 4, [('X', n_sites - 1), ('X', 0)])
        ham.add_term(Jy / 4, [('Y', n_sites - 1), ('Y', 0)])
        ham.add_term(Jz / 4, [('Z', n_sites - 1), ('Z', 0)])
    
    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_ssh(n_sites: int, t1: float = 1.0, t2: float = 0.5,
              boundary: str = "open") -> np.ndarray:
    """
    SSH (Su-Schrieffer-Heeger) model in spin-XY language.
    
    H = sum_i t_i (S+_i S-_{i+1} + h.c.)
    
    Alternating hoppings t1, t2, t1, t2, ...
    Topological for t1 < t2 (open chain), trivial for t1 > t2.
    """
    ham = SymbolicHamiltonian(n_sites)
    
    for i in range(n_sites - 1):
        t = t1 if i % 2 == 0 else t2
        # S+S- + h.c. = (1/2)(XX + YY)
        ham.add_term(t / 2, [('X', i), ('X', i + 1)])
        ham.add_term(t / 2, [('Y', i), ('Y', i + 1)])
    
    if boundary == "periodic" and n_sites > 2:
        t = t1 if (n_sites - 1) % 2 == 0 else t2
        ham.add_term(t / 2, [('X', n_sites - 1), ('X', 0)])
        ham.add_term(t / 2, [('Y', n_sites - 1), ('Y', 0)])
    
    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_j1j2_chain(n_sites: int, J1: float = 1.0, J2: float = 0.5,
                     boundary: str = "open") -> np.ndarray:
    """
    J1-J2 frustrated Heisenberg chain.
    
    H = J1 sum_i S_i.S_{i+1} + J2 sum_i S_i.S_{i+2}
    
    Famous Majumdar-Ghosh point at J2/J1 = 1/2.
    """
    ham = SymbolicHamiltonian(n_sites)
    
    # Nearest-neighbor
    for i in range(n_sites - 1):
        ham.add_term(J1 / 4, [('X', i), ('X', i + 1)])
        ham.add_term(J1 / 4, [('Y', i), ('Y', i + 1)])
        ham.add_term(J1 / 4, [('Z', i), ('Z', i + 1)])
    if boundary == "periodic" and n_sites > 2:
        ham.add_term(J1 / 4, [('X', n_sites - 1), ('X', 0)])
        ham.add_term(J1 / 4, [('Y', n_sites - 1), ('Y', 0)])
        ham.add_term(J1 / 4, [('Z', n_sites - 1), ('Z', 0)])
    
    # Next-nearest-neighbor
    for i in range(n_sites - 2):
        ham.add_term(J2 / 4, [('X', i), ('X', i + 2)])
        ham.add_term(J2 / 4, [('Y', i), ('Y', i + 2)])
        ham.add_term(J2 / 4, [('Z', i), ('Z', i + 2)])
    if boundary == "periodic" and n_sites > 3:
        ham.add_term(J2 / 4, [('X', n_sites - 2), ('X', 0)])
        ham.add_term(J2 / 4, [('Y', n_sites - 2), ('Y', 0)])
        ham.add_term(J2 / 4, [('Z', n_sites - 2), ('Z', 0)])
        ham.add_term(J2 / 4, [('X', n_sites - 1), ('X', 1)])
        ham.add_term(J2 / 4, [('Y', n_sites - 1), ('Y', 1)])
        ham.add_term(J2 / 4, [('Z', n_sites - 1), ('Z', 1)])
    
    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_majumdar_ghosh(n_sites: int) -> np.ndarray:
    """
    Majumdar-Ghosh point of J1-J2 chain (J2/J1 = 1/2).
    
    Famous for exact dimer ground state on PBC chains with even N.
    """
    return build_j1j2_chain(n_sites, J1=1.0, J2=0.5, boundary="periodic")


def build_cluster_model(n_sites: int, h: float = 1.0,
                        boundary: str = "open") -> np.ndarray:
    """
    Cluster model with three-body interactions.
    
    H = -sum_i X_{i-1} Z_i X_{i+1} - h sum_i X_i
    
    Symmetry-protected topological phase (string order).
    """
    if n_sites < 3:
        raise ValueError("Cluster model needs n_sites >= 3")
    
    ham = SymbolicHamiltonian(n_sites)
    
    for i in range(1, n_sites - 1):
        ham.add_term(-1.0, [('X', i - 1), ('Z', i), ('X', i + 1)])
    
    if boundary == "periodic":
        # Wrap around terms
        ham.add_term(-1.0, [('X', n_sites - 2), ('Z', n_sites - 1), ('X', 0)])
        ham.add_term(-1.0, [('X', n_sites - 1), ('Z', 0), ('X', 1)])
    
    for i in range(n_sites):
        ham.add_term(-h, [('X', i)])
    
    H = ham.build()
    ham.verify_hermiticity()
    return H


# ============================================================================
# Fermionic Models (via Jordan-Wigner)
# ============================================================================

def build_free_fermion_chain(n_sites: int, t: float = 1.0,
                              mu: float = 0.0,
                              boundary: str = "open") -> np.ndarray:
    """
    Free fermion (tight-binding) chain via Jordan-Wigner.
    
    H = -t sum_i (c^dag_i c_{i+1} + h.c.) - mu sum_i n_i
    
    Solvable analytically: epsilon_k = -2t cos(k) - mu.
    """
    c, cdag, num, F = fermion_operators()
    d = 2 ** n_sites
    H = np.zeros((d, d), dtype=complex)
    
    for i in range(n_sites - 1):
        cdag_i = jordan_wigner_operator(cdag, F, i, n_sites)
        c_j = jordan_wigner_operator(c, F, i + 1, n_sites)
        H += -t * (cdag_i @ c_j + (cdag_i @ c_j).conj().T)
    
    if boundary == "periodic" and n_sites > 2:
        cdag_i = jordan_wigner_operator(cdag, F, n_sites - 1, n_sites)
        c_j = jordan_wigner_operator(c, F, 0, n_sites)
        # Sign factor for fermion parity (anti-periodic for half-filled)
        # We use periodic boundary: H_pbc = H_open + extra hopping
        # With JW string this is correct as written
        H += -t * (cdag_i @ c_j + (cdag_i @ c_j).conj().T)
    
    if mu != 0:
        for i in range(n_sites):
            n_i = jordan_wigner_operator(num, F, i, n_sites)
            H += -mu * n_i
    
    # Symmetrize for floating point cleanliness
    H = (H + H.conj().T) / 2
    return H


def build_kitaev_chain(n_sites: int, t: float = 1.0, mu: float = 0.0,
                       Delta: float = 1.0,
                       boundary: str = "open") -> np.ndarray:
    """
    Kitaev p-wave superconducting chain.
    
    H = -t sum (c^dag_i c_{i+1} + h.c.)
        - mu sum n_i
        + Delta sum (c_i c_{i+1} + c^dag_{i+1} c^dag_i)
    
    Topological for |mu| < 2|t|, has Majorana edge modes.
    """
    c, cdag, num, F = fermion_operators()
    d = 2 ** n_sites
    H = np.zeros((d, d), dtype=complex)
    
    for i in range(n_sites - 1):
        cdag_i = jordan_wigner_operator(cdag, F, i, n_sites)
        c_i = jordan_wigner_operator(c, F, i, n_sites)
        cdag_j = jordan_wigner_operator(cdag, F, i + 1, n_sites)
        c_j = jordan_wigner_operator(c, F, i + 1, n_sites)
        
        # Hopping
        H += -t * (cdag_i @ c_j + cdag_j @ c_i)
        # Pairing
        H += Delta * (c_i @ c_j + cdag_j @ cdag_i)
    
    if boundary == "periodic" and n_sites > 2:
        cdag_i = jordan_wigner_operator(cdag, F, n_sites - 1, n_sites)
        c_i = jordan_wigner_operator(c, F, n_sites - 1, n_sites)
        cdag_j = jordan_wigner_operator(cdag, F, 0, n_sites)
        c_j = jordan_wigner_operator(c, F, 0, n_sites)
        H += -t * (cdag_i @ c_j + cdag_j @ c_i)
        H += Delta * (c_i @ c_j + cdag_j @ cdag_i)
    
    if mu != 0:
        for i in range(n_sites):
            n_i = jordan_wigner_operator(num, F, i, n_sites)
            H += -mu * n_i
    
    H = (H + H.conj().T) / 2
    return H


def build_hubbard_chain(n_sites: int, t: float = 1.0, U: float = 4.0,
                        boundary: str = "open") -> np.ndarray:
    """
    Single-band Hubbard model on a 1D chain.
    
    H = -t sum_<ij>,sigma (c^dag_{i,sigma} c_{j,sigma} + h.c.)
        + U sum_i n_{i,up} n_{i,down}
    
    Uses two spin species: state space = 4^N (each site: 0, up, down, both).
    Limited to small n_sites (4^N grows fast).
    """
    if n_sites > 6:
        warnings.warn(f"Hubbard with n={n_sites}: dim = 4^{n_sites} = {4**n_sites}")
    
    # Local basis: |0>, |up>, |down>, |up,down>
    # Operators: c_up, c_down, n_up, n_down
    
    # Single-site fermion operators
    c, cdag, num, F = fermion_operators()
    I2 = np.eye(2, dtype=complex)
    
    # Two-fermion site operators (4-dim local basis)
    # We use |0>, |down>, |up>, |up,down> ordering
    # c_up = c (x) F  (anticommute with down)
    # c_down = I (x) c
    c_up = np.kron(c, F)
    cdag_up = c_up.conj().T
    c_down = np.kron(I2, c)
    cdag_down = c_down.conj().T
    n_up = cdag_up @ c_up
    n_down = cdag_down @ c_down
    
    # JW string for site (4-dim)
    F_site = np.diag([1, -1, -1, 1]).astype(complex)
    
    def jw_op(local_op: np.ndarray, site: int, n_sites: int) -> np.ndarray:
        I4 = np.eye(4, dtype=complex)
        factors = []
        for k in range(n_sites):
            if k < site:
                factors.append(F_site)
            elif k == site:
                factors.append(local_op)
            else:
                factors.append(I4)
        return tensor_product(*factors)
    
    d = 4 ** n_sites
    H = np.zeros((d, d), dtype=complex)
    
    # Hopping terms
    for i in range(n_sites - 1):
        for op_pair in [(c_up, cdag_up), (c_down, cdag_down)]:
            c_op, cd_op = op_pair
            cdag_i = jw_op(cd_op, i, n_sites)
            c_j = jw_op(c_op, i + 1, n_sites)
            H += -t * (cdag_i @ c_j + (cdag_i @ c_j).conj().T)
    
    if boundary == "periodic" and n_sites > 2:
        for op_pair in [(c_up, cdag_up), (c_down, cdag_down)]:
            c_op, cd_op = op_pair
            cdag_i = jw_op(cd_op, n_sites - 1, n_sites)
            c_j = jw_op(c_op, 0, n_sites)
            H += -t * (cdag_i @ c_j + (cdag_i @ c_j).conj().T)
    
    # On-site interaction
    for i in range(n_sites):
        n_up_i = jw_op(n_up, i, n_sites)
        n_down_i = jw_op(n_down, i, n_sites)
        H += U * (n_up_i @ n_down_i)
    
    H = (H + H.conj().T) / 2
    return H


# ============================================================================
# Spin-1 Models
# ============================================================================

def build_aklt_chain(n_sites: int, boundary: str = "open") -> np.ndarray:
    """
    AKLT (Affleck-Kennedy-Lieb-Tasaki) spin-1 chain.
    
    H = sum_i [S_i.S_{i+1} + (1/3)(S_i.S_{i+1})^2]
    
    Famous for: gapped, hidden Z2xZ2 symmetry, exact VBS ground state.
    """
    I, Sx, Sy, Sz, Sp, Sm = spin_one_operators()
    d_local = 3
    d = d_local ** n_sites
    H = np.zeros((d, d), dtype=complex)
    
    def site_op(op, site):
        factors = [np.eye(d_local, dtype=complex)] * n_sites
        factors[site] = op
        return tensor_product(*factors)
    
    pairs = list(range(n_sites - 1))
    if boundary == "periodic" and n_sites > 2:
        pairs.append(n_sites - 1)
    
    for i in pairs:
        j = (i + 1) % n_sites
        SiSj = (site_op(Sx, i) @ site_op(Sx, j) +
                site_op(Sy, i) @ site_op(Sy, j) +
                site_op(Sz, i) @ site_op(Sz, j))
        H += SiSj + (1.0 / 3.0) * (SiSj @ SiSj)
    
    H = (H + H.conj().T) / 2
    return H


def build_haldane_chain(n_sites: int, J: float = 1.0,
                        D: float = 0.0, E: float = 0.0,
                        boundary: str = "open") -> np.ndarray:
    """
    Spin-1 Heisenberg chain with single-ion anisotropy.
    
    H = J sum_<ij> S_i.S_j + D sum_i (S^z_i)^2 + E sum_i ((S^x_i)^2 - (S^y_i)^2)
    
    Pure J=1, D=0: Haldane gap ~ 0.41 J for thermodynamic limit.
    """
    I, Sx, Sy, Sz, Sp, Sm = spin_one_operators()
    d_local = 3
    d = d_local ** n_sites
    H = np.zeros((d, d), dtype=complex)
    
    def site_op(op, site):
        factors = [np.eye(d_local, dtype=complex)] * n_sites
        factors[site] = op
        return tensor_product(*factors)
    
    # Heisenberg
    pairs = list(range(n_sites - 1))
    if boundary == "periodic" and n_sites > 2:
        pairs.append(n_sites - 1)
    
    for i in pairs:
        j = (i + 1) % n_sites
        H += J * (site_op(Sx, i) @ site_op(Sx, j) +
                  site_op(Sy, i) @ site_op(Sy, j) +
                  site_op(Sz, i) @ site_op(Sz, j))
    
    # Anisotropy
    for i in range(n_sites):
        if D != 0:
            H += D * (site_op(Sz, i) @ site_op(Sz, i))
        if E != 0:
            H += E * (site_op(Sx, i) @ site_op(Sx, i) - site_op(Sy, i) @ site_op(Sy, i))
    
    H = (H + H.conj().T) / 2
    return H


# ============================================================================
# 2D Models (Small Clusters)
# ============================================================================

def _index_2d(i: int, j: int, Lx: int, Ly: int) -> int:
    """Map (i,j) to flat index for 2D lattice."""
    return i * Ly + j


def build_tfim_2d(Lx: int, Ly: int, J: float = 1.0, h: float = 0.5,
                  boundary: str = "open") -> np.ndarray:
    """
    2D Transverse-Field Ising Model on Lx x Ly lattice.

    H = -J sum_<ij> Z_i Z_j - h sum_i X_i

    Each site has nearest neighbors in x and y directions.
    """
    n_sites = Lx * Ly
    if n_sites > 12:
        warnings.warn(f"2D TFIM with {n_sites} sites: dim = 2^{n_sites}")

    ham = SymbolicHamiltonian(n_sites)

    # x-direction bonds
    for i in range(Lx - 1):
        for j in range(Ly):
            site_a = _index_2d(i, j, Lx, Ly)
            site_b = _index_2d(i + 1, j, Lx, Ly)
            ham.add_term(-J, [('Z', site_a), ('Z', site_b)])

    # y-direction bonds
    for i in range(Lx):
        for j in range(Ly - 1):
            site_a = _index_2d(i, j, Lx, Ly)
            site_b = _index_2d(i, j + 1, Lx, Ly)
            ham.add_term(-J, [('Z', site_a), ('Z', site_b)])

    # Periodic boundary
    if boundary == "periodic":
        if Lx > 2:
            for j in range(Ly):
                site_a = _index_2d(Lx - 1, j, Lx, Ly)
                site_b = _index_2d(0, j, Lx, Ly)
                ham.add_term(-J, [('Z', site_a), ('Z', site_b)])
        if Ly > 2:
            for i in range(Lx):
                site_a = _index_2d(i, Ly - 1, Lx, Ly)
                site_b = _index_2d(i, 0, Lx, Ly)
                ham.add_term(-J, [('Z', site_a), ('Z', site_b)])

    # Transverse field
    for site in range(n_sites):
        ham.add_term(-h, [('X', site)])

    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_heisenberg_2d(Lx: int, Ly: int, J: float = 1.0,
                        boundary: str = "open") -> np.ndarray:
    """
    2D Heisenberg model on Lx x Ly lattice.

    H = J sum_<ij> S_i . S_j
    """
    n_sites = Lx * Ly
    if n_sites > 12:
        warnings.warn(f"2D Heisenberg with {n_sites} sites: dim = 2^{n_sites}")

    ham = SymbolicHamiltonian(n_sites)
    Jq = J / 4.0

    bonds = []
    # x-direction
    for i in range(Lx - 1):
        for j in range(Ly):
            bonds.append((_index_2d(i, j, Lx, Ly),
                          _index_2d(i + 1, j, Lx, Ly)))
    # y-direction
    for i in range(Lx):
        for j in range(Ly - 1):
            bonds.append((_index_2d(i, j, Lx, Ly),
                          _index_2d(i, j + 1, Lx, Ly)))

    # Periodic
    if boundary == "periodic":
        if Lx > 2:
            for j in range(Ly):
                bonds.append((_index_2d(Lx - 1, j, Lx, Ly),
                              _index_2d(0, j, Lx, Ly)))
        if Ly > 2:
            for i in range(Lx):
                bonds.append((_index_2d(i, Ly - 1, Lx, Ly),
                              _index_2d(i, 0, Lx, Ly)))

    for (a, b) in bonds:
        ham.add_term(Jq, [('X', a), ('X', b)])
        ham.add_term(Jq, [('Y', a), ('Y', b)])
        ham.add_term(Jq, [('Z', a), ('Z', b)])

    H = ham.build()
    ham.verify_hermiticity()
    return H


def build_kitaev_honeycomb(Lx: int = 2, Ly: int = 2,
                           Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0
                           ) -> np.ndarray:
    """
    Kitaev honeycomb model on small cluster.

    H = -Jx sum_x XX - Jy sum_y YY - Jz sum_z ZZ

    Bond directions alternate: each plaquette has one bond of each type.
    Uses a small honeycomb cluster (2 sites per unit cell).
    """
    n_sites = 2 * Lx * Ly
    if n_sites > 12:
        warnings.warn(f"Kitaev honeycomb with {n_sites} sites: large dim")

    ham = SymbolicHamiltonian(n_sites)

    # Honeycomb has 2 sublattices A,B per unit cell at (i,j)
    # A = 2*(i*Ly + j), B = A+1
    # Bonds: A-B same cell (z bond), A-(B left) (x bond), A-(B down) (y bond)
    for i in range(Lx):
        for j in range(Ly):
            A = 2 * (i * Ly + j)
            B = A + 1
            # z-bond (within unit cell)
            ham.add_term(-Jz, [('Z', A), ('Z', B)])
            # x-bond: A connects to B of (i-1, j) cell
            if i > 0:
                B_left = 2 * ((i - 1) * Ly + j) + 1
                ham.add_term(-Jx, [('X', A), ('X', B_left)])
            # y-bond: A connects to B of (i, j-1) cell
            if j > 0:
                B_down = 2 * (i * Ly + (j - 1)) + 1
                ham.add_term(-Jy, [('Y', A), ('Y', B_down)])

    H = ham.build()
    ham.verify_hermiticity()
    return H


# ============================================================================
# Dispatcher: model registry
# ============================================================================

MODEL_REGISTRY = {
    # 1D spin-1/2 systems
    'tfim': build_tfim,
    'heisenberg': build_heisenberg,
    'xxz': build_xxz,
    'xyz': build_xyz,
    'ssh': build_ssh,
    'j1j2': build_j1j2_chain,
    'majumdar_ghosh': build_majumdar_ghosh,
    'cluster': build_cluster_model,
    # Fermionic
    'free_fermion': build_free_fermion_chain,
    'kitaev_chain': build_kitaev_chain,
    'hubbard': build_hubbard_chain,
    # Spin-1
    'aklt': build_aklt_chain,
    'haldane': build_haldane_chain,
    # 2D
    'tfim_2d': build_tfim_2d,
    'heisenberg_2d': build_heisenberg_2d,
    'kitaev_honeycomb': build_kitaev_honeycomb,
}


def list_models() -> List[str]:
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> np.ndarray:
    """Build a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)
