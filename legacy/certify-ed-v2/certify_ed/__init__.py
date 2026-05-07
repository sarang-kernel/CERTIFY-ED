"""
CERTIFY-ED: Certified Exact Diagonalization
==========================================

A verification framework for exact diagonalization of quantum many-body systems.
"""

__version__ = "1.0.0"
__author__ = "Sarang Vehale"

from .hamiltonian import (
    SymbolicHamiltonian,
    build_tfim,
    build_heisenberg,
    build_xxz,
    pauli_matrices,
    tensor_product,
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
    "SymbolicHamiltonian",
    "build_tfim",
    "build_heisenberg",
    "build_xxz",
    "pauli_matrices",
    "tensor_product",
    "Oracle",
    "NumPyOracle",
    "ScipyOracle",
    "MultiOracle",
    "Certificate",
    "load_certificate",
    "ObservableCalculator",
]
