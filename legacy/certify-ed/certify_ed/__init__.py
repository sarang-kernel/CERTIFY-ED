"""
CERTIFY-ED: Certified Exact Diagonalization
==========================================

A framework for verified exact diagonalization of quantum many-body systems.

Main modules:
- hamiltonian: Symbolic Hamiltonian construction and verification
- oracles: Multi-oracle eigendecomposition with consensus validation
- certificates: Verification certificate generation and validation
- observables: Physical observable computation with error propagation

Example usage:
    >>> from certify_ed import build_tfim, MultiOracle, Certificate
    >>> H = build_tfim(n_sites=4, J=1.0, h=0.5)
    >>> oracle = MultiOracle()
    >>> evals, evecs, consensus = oracle.diagonalize_with_consensus(H)
    >>> cert = Certificate(evals, evecs, H)
    >>> cert.save('results.json')
"""

__version__ = "1.0.0"
__author__ = "Sarang Vehale"
__license__ = "MIT"

from .hamiltonian import (
    SymbolicHamiltonian,
    build_tfim,
    build_heisenberg,
    pauli_matrices,
)
from .oracles import (
    Oracle,
    NumPyOracle,
    ScipyOracle,
    MultiOracle,
)
from .certificates import Certificate, load_certificate
from .observables import ObservableCalculator

__all__ = [
    # Hamiltonian construction
    "SymbolicHamiltonian",
    "build_tfim",
    "build_heisenberg",
    "pauli_matrices",
    # Oracles
    "Oracle",
    "NumPyOracle",
    "ScipyOracle",
    "MultiOracle",
    # Certificates
    "Certificate",
    "load_certificate",
    # Observables
    "ObservableCalculator",
]
