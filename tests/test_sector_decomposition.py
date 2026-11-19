"""Audits of a full symmetry decomposition (certify_ed.sectors): coverage, orthogonality, trace
moments and the certified shift between the union of sector spectra and the full spectrum."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.basis import translation, cyclic_isometry
from certify_ed.fp import inf_norm_upper
from certify_ed.sectors import audit_decomposition, decomposition_shift, trace_moments_check


def _momentum_blocks(N=8, J2=0.3):
    spec = ce.heisenberg_chain(N, J2=J2)
    st = ce.u1_state_set(spec, N // 2)
    H = sp.csr_matrix(ce.build_bitwise(spec, st))
    T = translation(spec)
    Vs = [cyclic_isometry(st, T, k) for k in range(N)]
    Vs = [V for V in Vs if V.shape[1]]
    Hss = [sp.csr_matrix(V.conj().T @ H @ V) for V in Vs]
    return H, Vs, Hss


def test_full_momentum_decomposition_passes_and_spectra_match():
    H, Vs, Hss = _momentum_blocks()
    audit = audit_decomposition(H, Vs, Hss)
    assert audit.passed and audit.covered_dim == audit.expected_dim
    Hn = inf_norm_upper(H)
    Bn = max(inf_norm_upper(M) for M in Hss)
    eps = decomposition_shift(audit, Hn, Bn)
    assert eps is not None and eps < 1e-10
    full = la.eigvalsh(H.toarray())
    union = np.sort(np.concatenate([la.eigvalsh(M.toarray()) for M in Hss]))
    assert np.max(np.abs(full - union)) <= eps + 1e-12


def test_missing_sector_is_not_a_decomposition():
    H, Vs, Hss = _momentum_blocks()
    audit = audit_decomposition(H, Vs[:-1], Hss[:-1])
    assert not audit.passed and audit.covered_dim < audit.expected_dim
    assert decomposition_shift(audit, 10.0, 10.0) is None


def test_corrupted_sector_block_fails_the_audit():
    H, Vs, Hss = _momentum_blocks()
    bad = Hss[0].tolil(copy=True)
    bad[0, 0] += 1e-3
    audit = audit_decomposition(H, Vs, [bad.tocsr()] + Hss[1:])
    assert not audit.passed
    assert not trace_moments_check(H, [bad.tocsr()] + Hss[1:])["consistent"]
