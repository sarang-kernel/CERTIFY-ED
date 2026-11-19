"""Tests for multi-oracle module."""
import numpy as np
import pytest
from certify_ed import build_model, MultiOracle, NumPyOracle, ScipyOracle, SparseOracle


def test_oracle_consensus_basic():
    H = build_model('tfim', n_sites=4)
    oracle = MultiOracle()
    evals, evecs, report = oracle.diagonalize_with_consensus(H)
    assert report['consensus']
    assert report['max_disagreement'] < 1e-10


def test_residuals_small():
    H = build_model('heisenberg', n_sites=4)
    oracle = MultiOracle()
    evals, evecs, _ = oracle.diagonalize_with_consensus(H)
    residuals = oracle.compute_residuals(H, evals, evecs)
    assert np.max(residuals) < 1e-10


def test_eigenvectors_orthonormal():
    H = build_model('xxz', n_sites=4, Delta=0.5)
    oracle = MultiOracle()
    _, evecs, _ = oracle.diagonalize_with_consensus(H)
    overlap = evecs.conj().T @ evecs
    assert np.allclose(overlap, np.eye(overlap.shape[0]), atol=1e-12)


def test_spectral_decomposition():
    H = build_model('cluster', n_sites=4, h=0.5)
    oracle = MultiOracle()
    evals, evecs, _ = oracle.diagonalize_with_consensus(H)
    H_reconstructed = evecs @ np.diag(evals).astype(complex) @ evecs.conj().T
    assert np.allclose(H, H_reconstructed, atol=1e-12)


def test_sparse_lanczos_lowest_k():
    H = build_model('tfim', n_sites=6)
    oracle = SparseOracle(k=3)
    evals, evecs = oracle.diagonalize(H)
    assert len(evals) == 3
    # Compare against dense
    evals_dense = np.linalg.eigvalsh(H)
    assert np.allclose(np.sort(evals), np.sort(evals_dense)[:3], atol=1e-8)


def test_invalid_driver_raises():
    with pytest.raises(ValueError):
        ScipyOracle(driver='invalid')


def test_multi_oracle_minimum_oracles():
    with pytest.raises(ValueError):
        MultiOracle(oracles=[NumPyOracle()])
