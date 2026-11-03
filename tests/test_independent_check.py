"""Negative-control tests added by the independent re-derivation (Phase 2, docs/proofs/VERDICTS.md, section
"Required corrections"). Each test fails on the code before the recorded correction and passes after it."""
import numpy as np

from certify_ed.enclosure import OperatorAdapter, enclose, residuals
from certify_ed.fp import norm2_lower, norm2_upper


def test_norm2_lower_is_zero_when_the_squared_norm_overflows():
    """int05_underflow, Theorem U5 'no magnitude hypothesis': for ||v|| > ~1.3e154 the BLAS sum of squares is inf
    and the old norm2_lower returned inf, which is not a lower bound on ||v||."""
    v = np.array([1e160, 0.0])
    assert norm2_lower(v) == 0.0
    assert norm2_upper(v) == float("inf")
    # just below the overflow threshold the bound is finite and valid
    w = np.array([1e150, 1e150])
    assert 0.0 < norm2_lower(w) <= np.sqrt(2) * 1e150 <= norm2_upper(w) < float("inf")


def test_huge_claimed_vector_gives_no_finite_enclosure():
    """Counterexample of the re-derivation: A = diag(1, 2), mu = 1 + 1e-8, x = (1e160, 0). Before the correction
    the Theorem 1 radius was 2 ULP_MIN = 1e-323 (upper / inf = 0), a false enclosure (the nearest eigenvalue is
    1e-8 away). Now the radius is infinite and the pair is not a distinct match."""
    A = np.diag([1.0, 2.0])
    x = np.array([[1e160], [0.0]])
    mu = np.array([1.0 + 1e-8])
    res = residuals(OperatorAdapter(A), mu, x)
    assert res.xnorm_lower[0] == 0.0 and res.rel_bound[0] == float("inf")
    enc = enclose(OperatorAdapter(A), mu, x)
    assert enc.radius[0] == float("inf") and not enc.all_distinct
