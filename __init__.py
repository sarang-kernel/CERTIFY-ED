"""
CERTIFY-ED: Certified Exact Diagonalization Framework
======================================================

Production-ready framework for symbolic validation and numerical certification
of exact diagonalization computations in quantum many-body systems.

Author: Sarang Vehale
License: MIT
Version: 1.0.0
"""

from .core import (
    CertifiedHamiltonian,
    MultiOracleDiagonalizer,
    DiagonalizationResults,
    Certificate,
    ObservableCalculator,
)

__version__ = "1.0.0"
__author__ = "Sarang Vehale"
__license__ = "MIT"

__all__ = [
    "CertifiedHamiltonian",
    "MultiOracleDiagonalizer",
    "DiagonalizationResults",
    "Certificate",
    "ObservableCalculator",
]
