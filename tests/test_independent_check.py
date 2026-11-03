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


def test_exact_bracket_statement_names_the_exact_bracket():
    """C16_exact_bracket: the certified sentence used to print the outward-rounded binary64 bracket; an eigenvalue
    within one ulp below the counted bracket's end lies in the printed interval but not in the count, so the
    sentence 'exactly 1 eigenvalue in [1, 2]' was false here (the model has two eigenvalues in [1, 2])."""
    import flint
    from fractions import Fraction
    from certify_ed.exact_degeneracy import _record
    lam = [Fraction(1) + Fraction(1, 2 ** 60), Fraction(3, 2)]
    den = 2 ** 60
    p = flint.fmpz_mat([[int(lam[0] * den), 0], [0, int(lam[1] * den)]]).charpoly()
    rec = _record(p, den, 1, Fraction(1.0 + 2.0 ** -52), Fraction(2.0), 1, 2, "demo")
    assert rec["status"] == "CERTIFIED" and rec["count"] == 1
    assert rec["bracket"][0] <= float(lam[0]) <= rec["bracket"][1]          # the display interval holds a 2nd eigenvalue
    assert "exact bracket [4503599627370497/4503599627370496, 2/1]" in rec["statement"]
    assert "in [1, 2]" not in rec["statement"].split("(about")[0]


# --- E-04 (docs/proofs/E-04_parameter_intervals.md, Lemma 3 applied at interior grid points) ----------------------
# Two scalar scans: a: F_0(c) = -1.9951 + c on [0, 1] (B = 1) with H_a(1) = -1.0951, i.e. an affinity defect
# d_0 = 0.1; b: H_b(c) = -c exactly (B = 1, d = 0). The segment-0 branches cross at c* = 0.99755, inside the last
# sample interval [0.995, 1] before the grid point. Before the fix the criterion used the right-segment tube at c = 1
# (H_a(1), not F_0(1)) and certified '<' on [0, 1.045]; with the defect added to the slack the interval is not
# separated.
def _scalar_scan(cs, B, defects, values):
    pts = [{"index": i, "c": c, "lower": [v], "upper": [v], "rank_certified": True, "next_eigenvalue_lower_bound": None}
           for i, (c, v) in enumerate(zip(cs, values))]
    return {"parameter": "c", "couplings": cs, "B": B, "affinity_defects": defects, "points": pts,
            "record_sha256": "synthetic"}


def test_crossing_criterion_accounts_for_the_affinity_defect_at_interior_grid_points():
    from certify_ed.sweep import certified_crossing_interval
    ra = _scalar_scan([0.0, 1.0, 2.0], 1.0, [0.1, 0.0], [-1.9951, -1.0951, -0.0951])
    rb = _scalar_scan([0.0, 1.0, 2.0], 1.0, [0.0, 0.0], [0.0, -1.0, -2.0])
    out = certified_crossing_interval(ra, 0, rb, 0, samples_per_segment=200)
    c_star = 1.9951 / 2                                   # a(c) - b(c) = -1.9951 + 2c = 0 on segment 0
    for run in out["runs"]:
        lo, hi = run["interval"]
        if run["relation"] == "<" and lo < 1.0:          # segment 0: nothing certified '<' beyond its crossing
            assert hi <= c_star + 1e-12, run
    bad = [run for run in out["runs"] if run["relation"] == "<" and run["interval"][0] < 1.0 and run["interval"][1] > 0.995 + 1e-12]
    assert not bad, bad                                   # before the fix: [{'relation': '<', 'interval': [0.0, 1.045]}]
    # the same scans with d_0 = 0 are a consistent affine family for which '<' on [0, 0.995] is right
    ra0 = _scalar_scan([0.0, 1.0, 2.0], 1.0, [0.0, 0.0], [-1.9951, -0.9951, 0.0049])
    out0 = certified_crossing_interval(ra0, 0, rb, 0, samples_per_segment=200)
    assert out0["runs"][0]["relation"] == "<" and out0["runs"][0]["interval"][1] >= 0.99


def test_rounding_bound_uses_width_plus_three_for_complex_data():
    """thm1 correction (VERDICTS.md): for complex data the division constant is gamma_{w+3}, so the entrywise bound
    E is never smaller than the real-data formula would give; a real operator with real vectors is unchanged."""
    import numpy as np
    from certify_ed.enclosure import OperatorAdapter, _rounding_bound
    from certify_ed.fp import U, gamma, gamma_dot
    rng = np.random.default_rng(7)
    A = rng.standard_normal((6, 6)); A = A + A.T
    x = rng.standard_normal((6, 2))
    ad = OperatorAdapter(A)
    E_real = _rounding_bound(ad, np.array([0.3, -1.2]), x)
    # the real formula, reproduced
    absHX = np.abs(A) @ np.abs(x)
    ref = gamma_dot(ad.width + 2, False) * (absHX / (1 - gamma(ad.width + 1)) * (1 + 4 * U) + np.abs(x) * np.abs(np.array([0.3, -1.2]))[None, :]) * (1 + 4 * U)
    assert np.allclose(E_real, ref, rtol=1e-12, atol=0)
    xc = x + 1j * rng.standard_normal((6, 2))
    E_cplx = _rounding_bound(ad, np.array([0.3, -1.2]), xc)
    absHXc = np.abs(A) @ np.abs(xc)
    ref_old = gamma_dot(ad.width + 2, True) * (absHXc / (1 - gamma(ad.width + 1)) * (1 + 4 * U) + np.abs(xc) * np.abs(np.array([0.3, -1.2]))[None, :]) * (1 + 4 * U)
    ref_new = gamma_dot(ad.width + 2, True) * (absHXc / (1 - gamma(ad.width + 3)) * (1 + 4 * U) + np.abs(xc) * np.abs(np.array([0.3, -1.2]))[None, :]) * (1 + 4 * U)
    assert np.all(E_cplx >= ref_old) and np.allclose(E_cplx, ref_new, rtol=1e-12, atol=0)
