"""Smoke tests for all validators - verify they at least run."""
import pytest


def test_analytic_validator_runs():
    from validators import AnalyticValidator
    v = AnalyticValidator()
    s = v.summary()
    assert s['n_total'] > 0
    # All analytic tests should pass (they're machine-precision checks)
    assert s['n_passed'] == s['n_total'], f"Failed: {[r['test_name'] for r in s['individual_results'] if not r['passed']]}"


def test_sum_rules_validator_runs():
    from validators import SpectralSumRuleValidator
    v = SpectralSumRuleValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_orthonormality_validator_runs():
    from validators import OrthonormalityValidator
    v = OrthonormalityValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_unitarity_validator_runs():
    from validators import UnitarityValidator
    v = UnitarityValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_conservation_validator_runs():
    from validators import ConservationLawValidator
    v = ConservationLawValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_symmetry_sector_validator_runs():
    from validators import SymmetrySectorValidator
    v = SymmetrySectorValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_thermal_validator_runs():
    from validators import ThermalLimitValidator
    v = ThermalLimitValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_jordan_wigner_validator_runs():
    from validators import JordanWignerValidator
    v = JordanWignerValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_sparse_dense_validator_runs():
    from validators import SparseDenseValidator
    v = SparseDenseValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_scaling_validator_runs():
    from validators import FiniteSizeScalingValidator
    v = FiniteSizeScalingValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_passed']


def test_error_injection_validator_detects_all():
    from validators import ErrorInjectionValidator
    v = ErrorInjectionValidator()
    s = v.summary()
    assert s['n_total'] > 0
    assert s['all_detected'], "Error injection validator must detect ALL injected errors"


def test_high_precision_if_available():
    from validators import HighPrecisionValidator
    v = HighPrecisionValidator(precision_dps=30)
    s = v.summary()
    if s.get('available'):
        assert s['all_passed']


def test_quspin_optional():
    """QuSpin is optional. If present, must pass; if not, validator skips gracefully."""
    from validators import QuSpinValidator
    v = QuSpinValidator()
    s = v.summary()
    if s.get('available'):
        assert s['all_passed']
    # If unavailable, no assertion needed; validator returns 0/0 cleanly
