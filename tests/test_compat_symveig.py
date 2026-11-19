"""symveig 1.0.0's test suite, ported to certify_ed.compat.symveig (same inputs and containment checks),
plus model checks. Containment is checked against the known exact spectra of the synthetic matrices,
allowing for the rounding of Q diag(lam) Q^H itself (a few ulps of ||H||)."""
import warnings

import numpy as np
import pytest

from certify_ed.compat.symveig import (enclose_global, enclose_sectors, summary_stats, heisenberg_1d,
                                       j1j2_heisenberg_1d, heisenberg_2d)


def _check_contains(enclosures, true_eigs, slack):
    encs = sorted(enclosures, key=lambda e: e.midpoint)
    remaining = [e.multiplicity for e in encs]
    for t in np.sort(true_eigs):
        for i, e in enumerate(encs):
            if remaining[i] > 0 and e.lo - slack <= t <= e.hi + slack:
                remaining[i] -= 1
                break
        else:
            return False, t
    return True, None


def _random_hermitian(true_eigs, seed):
    n = len(true_eigs)
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    H = Q @ np.diag(true_eigs.astype(np.complex128)) @ Q.conj().T
    return 0.5 * (H + H.conj().T)          # exactly Hermitian


def _slack(H):
    # the matrix as stored differs from Q diag(lam) Q^H by rounding of the products
    return 64 * H.shape[0] * 2.0 ** -53 * np.abs(H).sum(axis=1).max()


def test_synthetic_isolated():
    true = np.arange(128) + 0.5
    H = _random_hermitian(true, 42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encs = enclose_global(H, tier2=True)
    ok, bad = _check_contains(encs, true, _slack(H))
    assert ok, bad
    s = summary_stats(encs)
    assert s["total_eigs"] == 128 and s["all_counted"]


def test_synthetic_clustered():
    true = np.arange(128) + 0.5
    true[60:70] = 50.0
    H = _random_hermitian(true, 43)
    encs = enclose_global(H)
    ok, bad = _check_contains(encs, true, _slack(H))
    assert ok, bad
    assert max(e.multiplicity for e in encs) >= 10
    assert summary_stats(encs)["all_counted"]


def test_synthetic_sectored():
    sizes = [30, 50, 48]
    rng = np.random.default_rng(44)
    eigs = [rng.uniform(-2, -1, sizes[0]), rng.uniform(-1.5, 1.5, sizes[1]), rng.uniform(1, 2, sizes[2])]
    n = sum(sizes)
    H = np.zeros((n, n), dtype=complex)
    S = np.concatenate([np.full(m, float(k)) for k, m in enumerate(sizes)])
    off = 0
    for k, m in enumerate(sizes):
        H[off:off + m, off:off + m] = _random_hermitian(eigs[k], 100 + k)
        off += m
    true = np.sort(np.concatenate(eigs))
    for encs in (enclose_global(H), enclose_sectors(H, S)):
        ok, bad = _check_contains(encs, true, _slack(H))
        assert ok, bad
        assert summary_stats(encs)["all_counted"]


def test_ill_conditioned():
    true = 10.0 ** np.linspace(-8, 8, 64)
    H = _random_hermitian(true, 45)
    ok, bad = _check_contains(enclose_global(H), true, _slack(H))
    assert ok, bad


def test_near_degenerate_cluster():
    true = np.arange(64).astype(float)
    true[10], true[11] = 5.0, 5.0 + 1e-10
    H = _random_hermitian(true, 46)
    ok, bad = _check_contains(enclose_global(H), true, _slack(H))
    assert ok, bad


def test_non_hermitian_needs_explicit_symmetrize():
    H = np.array([[0.0, 1.0], [1.0 + 1e-15, 0.0]])
    with pytest.raises(ValueError):
        enclose_global(H)
    encs = enclose_global(H, symmetrize=True)
    assert summary_stats(encs)["total_eigs"] == 2


@pytest.mark.parametrize("build", [lambda: heisenberg_1d(8), lambda: heisenberg_1d(8, periodic=False),
                                   lambda: j1j2_heisenberg_1d(8), lambda: heisenberg_2d(3, 2)])
def test_models_hermitian_conserving_and_sector_sizes(build):
    from math import comb
    H, sec = build()
    S = sec["Sz"]
    assert np.array_equal(H, H.conj().T)
    assert np.abs(H * (S[:, None] - S[None, :])).max() == 0          # [H, S_z] = 0 exactly
    L = int(round(np.log2(H.shape[0])))
    vals, counts = np.unique(S, return_counts=True)
    for v, c in zip(vals, counts):
        assert c == comb(L, int(round(L / 2 + v)))
    encs = enclose_sectors(H, S)
    assert summary_stats(encs)["total_eigs"] == H.shape[0] and summary_stats(encs)["all_counted"]
