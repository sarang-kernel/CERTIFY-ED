"""Tests for oracles module."""

import numpy as np
import pytest
from certify_ed import (
    NumPyOracle, ScipyOracle, MultiOracle,
    build_tfim, build_heisenberg
)


class TestNumPyOracle:
    def test_diagonal(self):
        oracle = NumPyOracle()
        H = np.diag([1.0, 2.0, 3.0]).astype(complex)
        evals, _ = oracle.diagonalize(H)
        assert np.allclose(evals, [1.0, 2.0, 3.0])


class TestScipyOracle:
    def test_drivers(self):
        H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        for driver in ['evd', 'evr', 'ev']:
            oracle = ScipyOracle(driver=driver)
            evals, _ = oracle.diagonalize(H)
            assert np.allclose(evals, [1.0, 2.0, 3.0, 4.0])
    
    def test_invalid_driver(self):
        with pytest.raises(ValueError):
            ScipyOracle(driver='invalid')


class TestMultiOracle:
    def test_consensus_simple(self):
        oracle = MultiOracle()
        H = np.diag([1.0, 2.0, 3.0]).astype(complex)
        evals, _, report = oracle.diagonalize_with_consensus(H)
        assert report['consensus']
        assert report['max_disagreement'] < 1e-12
    
    def test_consensus_tfim(self):
        oracle = MultiOracle()
        H = build_tfim(n_sites=4, J=1.0, h=0.5)
        evals, evecs, report = oracle.diagonalize_with_consensus(H)
        assert report['consensus']
    
    def test_residuals(self):
        oracle = MultiOracle()
        H = build_tfim(n_sites=3, J=1.0, h=0.5)
        evals, evecs, _ = oracle.diagonalize_with_consensus(H)
        residuals = oracle.compute_residuals(H, evals, evecs)
        assert np.all(residuals < 1e-10)
    
    def test_too_few_oracles(self):
        with pytest.raises(ValueError):
            MultiOracle(oracles=[NumPyOracle()])
