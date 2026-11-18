"""Tests for certificates module."""
import json
import os
import tempfile
import numpy as np
import pytest
from certify_ed import build_model, MultiOracle, Certificate, load_certificate


def _make_certificate():
    H = build_model('tfim', n_sites=3)
    oracle = MultiOracle()
    evals, evecs, report = oracle.diagonalize_with_consensus(H)
    return Certificate(evals, evecs, H, consensus_report=report,
                       metadata={'test': 'certificate'})


def test_certificate_creation():
    cert = _make_certificate()
    assert cert.is_certified
    assert np.max(cert.residuals) < 1e-10
    assert np.max(cert.normalization_errors) < 1e-12


def test_certificate_save_and_load():
    cert = _make_certificate()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        cert.save(path)
        data = load_certificate(path, verify_hash=True)
        assert 'sha256' in data
        assert data['certification']['is_certified']
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_certificate_tampering_detected():
    cert = _make_certificate()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        cert.save(path)
        # Tamper
        with open(path, 'r') as f:
            data = json.load(f)
        data['spectrum']['eigenvalues'][0] = 99999.0
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        # Loading must raise
        with pytest.raises(ValueError):
            load_certificate(path, verify_hash=True)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_certificate_summary_runs():
    cert = _make_certificate()
    summary = cert.summary()
    assert isinstance(summary, str)
    assert 'Certified' in summary
