"""Tests for certificates module."""

import numpy as np
import pytest
import tempfile
import os
from certify_ed import (
    Certificate, load_certificate,
    build_tfim, MultiOracle
)


class TestCertificate:
    def setup_method(self):
        self.H = build_tfim(n_sites=3, J=1.0, h=0.5)
        oracle = MultiOracle()
        self.evals, self.evecs, self.report = oracle.diagonalize_with_consensus(self.H)
    
    def test_creation(self):
        cert = Certificate(self.evals, self.evecs, self.H)
        assert cert.is_certified
        assert np.max(cert.residuals) < 1e-10
    
    def test_save_load(self):
        cert = Certificate(self.evals, self.evecs, self.H,
                          metadata={'model': 'TFIM'},
                          consensus_report=self.report)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmpname = f.name
        
        try:
            cert.save(tmpname)
            assert os.path.exists(tmpname)
            
            # Load and verify
            data = load_certificate(tmpname, verify_hash=True)
            assert data['certification']['is_certified']
            assert data['metadata']['model'] == 'TFIM'
        finally:
            if os.path.exists(tmpname):
                os.remove(tmpname)
    
    def test_summary(self):
        cert = Certificate(self.evals, self.evecs, self.H)
        s = cert.summary()
        assert 'CERTIFY-ED' in s
        assert 'Ground state' in s
