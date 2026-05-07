"""
CERTIFY-ED: Certified Exact Diagonalization Framework

A comprehensive formal verification framework for quantum many-body systems
using multi-oracle consensus, symbolic verification, and exportable proofs.

Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__all__ = [
    'CertifiedHamiltonian',
    'MultiOracleDiagonalizer', 
    'CertificationEngine',
    'ObservableCalculator',
    'Certificate',
    'CertifiedEigenpair'
]

from .hamiltonian import CertifiedHamiltonian
from .diagonalizer import MultiOracleDiagonalizer
from .engine import CertificationEngine
from .observables import ObservableCalculator
from .certificates import Certificate, CertifiedEigenpair

# Logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
