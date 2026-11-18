import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.sectors import audit_label

quspin = pytest.importorskip("quspin")
from quspin.basis import spin_basis_1d


@pytest.mark.parametrize("k", [0, 1, 3, 5, 9])
def test_quspin_momentum_labels(k):
    N = 10
    spec = ce.heisenberg_chain(N)
    b = spin_basis_1d(N, Nup=N // 2, kblock=k, pauli=0)
    V = sp.csr_matrix(b.get_proj(np.complex128))
    r = audit_label(spec, V, "translation")
    assert r["certified"] and r["quspin_q"] == k, r
    assert r["leakage_bound"] < 1e-12


def test_mislabelled_block_is_detected():
    N = 8
    spec = ce.heisenberg_chain(N)
    V2 = sp.csr_matrix(spin_basis_1d(N, Nup=4, kblock=2, pauli=0).get_proj(np.complex128))
    V3 = sp.csr_matrix(spin_basis_1d(N, Nup=4, kblock=3, pauli=0).get_proj(np.complex128))
    mixed = sp.hstack([V2, V3]).tocsr()          # a "sector" that mixes two momenta
    r = audit_label(spec, mixed, "translation")
    assert not r["certified"]
