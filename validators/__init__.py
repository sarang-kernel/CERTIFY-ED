"""
Independent Validators
======================

Each validator implements a different independent check that goes
beyond simple eigenvalue computation. These are the multiple points
of validation that together establish trustworthiness.

Validators:
    AnalyticValidator      - Compare against closed-form solutions
    QuSpinValidator        - Cross-check against QuSpin (if installed)
    HighPrecisionValidator - Cross-check against mpmath arbitrary precision
    SparseDenseValidator   - Cross-check sparse Lanczos vs dense LAPACK
    JordanWignerValidator  - Free-fermion solvable models via JW transform
    SpectralSumRuleValidator - Trace, Frobenius norm consistency
    OrthonormalityValidator - Eigenvector orthonormality + completeness
    UnitarityValidator     - exp(-iHt) is unitary
    ConservationLawValidator - [H, S] = 0 implies block diagonal
    SymmetrySectorValidator - Symmetry-resolved spectrum match
    ThermalLimitValidator  - High-T thermal observables match analytic
    FiniteSizeScalingValidator - Trends approach known thermodynamic limits
    ErrorInjectionValidator - Framework correctly detects injected errors
"""

from .analytic_validator import AnalyticValidator
from .quspin_validator import QuSpinValidator
from .high_precision_validator import HighPrecisionValidator
from .sparse_dense_validator import SparseDenseValidator
from .jordan_wigner_validator import JordanWignerValidator
from .sum_rules_validator import SpectralSumRuleValidator
from .orthonormality_validator import OrthonormalityValidator
from .unitarity_validator import UnitarityValidator
from .conservation_validator import ConservationLawValidator
from .symmetry_sector_validator import SymmetrySectorValidator
from .thermal_validator import ThermalLimitValidator
from .scaling_validator import FiniteSizeScalingValidator
from .error_injection_validator import ErrorInjectionValidator

__all__ = [
    "AnalyticValidator",
    "QuSpinValidator",
    "HighPrecisionValidator",
    "SparseDenseValidator",
    "JordanWignerValidator",
    "SpectralSumRuleValidator",
    "OrthonormalityValidator",
    "UnitarityValidator",
    "ConservationLawValidator",
    "SymmetrySectorValidator",
    "ThermalLimitValidator",
    "FiniteSizeScalingValidator",
    "ErrorInjectionValidator",
]
