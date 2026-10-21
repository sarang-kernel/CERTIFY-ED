"""S7 thermodynamic intervals (Proposition 16) and the dyadic enclosures of exp/ln (Lemma 17). UNVERIFIED math:
the tests check the implementation against the statements of docs/theory.md Section 9.3."""
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed import ivl
from certify_ed.thermo import thermodynamics


# ------------------------------------------------------------------------------------ ivl
def test_exp_and_ln_exact_points():
    one = ivl.from_fraction(Fraction(1))
    e0 = ivl.exp_interval(ivl.Interval(0, 0))
    assert e0.lo == e0.hi == ivl.ONE
    l1 = ivl.ln_interval(one)
    assert l1.lo <= 0 <= l1.hi and l1.width() < 1e-50
    assert ivl.from_float(0.1).contains(Fraction(0.1))
    assert ivl.to_float_down(ivl.from_float(0.1).lo) == 0.1 and ivl.to_float_up(ivl.from_float(0.1).hi) == 0.1


def test_exp_ln_enclose_high_precision_reference():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 70
    rng = np.random.default_rng(7)
    for x in np.concatenate([rng.uniform(-100, 100, 60), [0.5, -0.5, 1.0, -1.0, 37.25, -37.25]]):
        iv = ivl.exp_interval(ivl.from_float(float(x)))
        lo, hi = iv.fractions()
        ex = mp.exp(mp.mpf(float(x)))
        assert mp.mpf(lo.numerator) / lo.denominator <= ex <= mp.mpf(hi.numerator) / hi.denominator
        if x > -100:
            assert float((hi - lo) / hi) < 1e-40 or hi - lo < Fraction(1, 2 ** 150)
    for z in np.concatenate([np.exp(rng.uniform(-30, 200, 60)), [1.0, 2.0, 0.5, 3.0, 1e100]]):
        iv = ivl.ln_interval(ivl.from_float(float(z)))
        lo, hi = iv.fractions()
        lz = mp.log(mp.mpf(float(z)))
        assert mp.mpf(lo.numerator) / lo.denominator <= lz <= mp.mpf(hi.numerator) / hi.denominator
        assert float(hi - lo) < 1e-50


def test_interval_arithmetic_outward():
    a = ivl.from_float(0.1)
    b = ivl.from_float(0.2)
    s = a + b
    assert s.contains(Fraction(0.1) + Fraction(0.2))
    p = a * b
    assert p.contains(Fraction(0.1) * Fraction(0.2))
    q = a / b
    assert q.contains(Fraction(0.1) / Fraction(0.2))
    with pytest.raises(ZeroDivisionError):
        a / ivl.Interval(-1, 1)
    m = ivl.Interval(-3 * ivl.ONE, 2 * ivl.ONE)
    assert m.square().lo == 0 and m.square().hi == 9 * ivl.ONE
    assert m.intersect_nonneg().lo == 0


# ------------------------------------------------------------------------------------ Prop 16
def _check_rows(res, ref):
    """ref(T) -> (Z, F, U, C) in mpmath or float; every reference inside the interval."""
    for r in res["rows"]:
        Z, F, U, C = ref(r["T"])
        assert r["F"][0] <= F <= r["F"][1], ("F", r["T"], r["F"], F)
        assert r["U"][0] <= U <= r["U"][1], ("U", r["T"], r["U"], U)
        assert r["C"][0] <= C <= r["C"][1], ("C", r["T"], r["C"], C)
        if r["Z"] is not None:
            assert r["Z"][0] <= Z <= r["Z"][1], ("Z", r["T"], r["Z"], Z)
        assert r["C"][0] >= 0


def test_two_level_system_against_closed_form():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    res = thermodynamics([{"lo": 0.0, "hi": 0.0, "count": 1}, {"lo": 1.0, "hi": 1.0, "count": 1}],
                         [0.05, 0.5, 1.0, 2.0, 100.0], dimension=2)

    def ref(T):
        T = mp.mpf(T)
        b = 1 / T
        Z = 1 + mp.exp(-b)
        return Z, -T * mp.log(Z), mp.exp(-b) / Z, b * b * mp.exp(-b) / Z ** 2
    _check_rows(res, ref)
    for r in res["rows"]:
        assert r["U"][1] - r["U"][0] < 1e-15 and r["C"][1] - r["C"][0] < 1e-15
    assert "every eigenvalue" in res["hypothesis"] and "loose" in res["looseness"].lower() or "ignore" in res["looseness"]


def test_degenerate_counts_and_wide_intervals_are_handled():
    # three-fold level at 1 and a wide interval: the enclosure must contain every spectrum consistent with it
    ivs = [{"lo": 0.0, "hi": 0.0, "count": 1}, {"lo": 1.0, "hi": 1.0, "count": 3}, {"lo": 2.0, "hi": 2.5, "count": 2}]
    res = thermodynamics(ivs, [0.3, 1.0, 3.0])
    rng = np.random.default_rng(1)
    for r in res["rows"]:
        T = r["T"]
        for _ in range(20):
            E = np.array([0.0, 1.0, 1.0, 1.0, *rng.uniform(2.0, 2.5, 2)])
            b = 1 / T
            w = np.exp(-b * E)
            Z = w.sum()
            U = (E * w).sum() / Z
            C = b * b * ((E ** 2 * w).sum() / Z - U ** 2)
            F = -T * np.log(Z)
            assert r["Z"][0] <= Z <= r["Z"][1] and r["F"][0] <= F <= r["F"][1]
            assert r["U"][0] <= U <= r["U"][1] and r["C"][0] <= C <= r["C"][1]
        assert r["U"][1] - r["U"][0] > 1e-4           # the width reflects the wide interval, as it must


def test_input_validation():
    with pytest.raises(ValueError):
        thermodynamics([{"lo": 0.0, "hi": 0.0, "count": 1}], [1.0], dimension=2)        # not complete
    with pytest.raises(ValueError):
        thermodynamics([{"lo": 0.0, "hi": 0.0, "count": 1}], [-1.0])
    with pytest.raises(ValueError):
        thermodynamics([{"lo": 1.0, "hi": 0.0, "count": 1}], [1.0])


def test_free_fermions_full_spectrum_through_certify():
    """Spinless free fermions, N=5 open chain: Z, U and C follow from the single-particle energies
    (grand canonical over the Fock space at mu = 0). S7 through certify() must enclose them."""
    N = 5
    spec = ce.spinless_fermion_chain(N, t=1.0, V=0.0, mu=0.0, boundary="open")
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    temps = [0.2, 0.5, 1.0, 3.0]
    cert = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec, ce.Options.from_preset("whole-spectrum", temperatures=temps))
    s7 = [s for s in cert.statements if s["id"] == "S7.thermodynamics"][0]
    assert s7["status"] == "CERTIFIED", s7["summary"]
    assert cert.body["spectrum_enclosure"]["complete"]
    # single-particle energies from the one-particle block of the same matrix
    h1 = np.zeros((N, N))
    for i in range(N - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    eps = la.eigvalsh(h1)
    # sign convention: the many-body spectrum must be reproduced by sums of subsets of eps (or of -eps)
    for sgn in (1.0, -1.0):
        subset_sums = sorted(sum(sgn * eps[k] for k in range(N) if (m >> k) & 1) for m in range(2 ** N))
        if np.allclose(subset_sums, np.sort(w), atol=1e-9):
            eps = sgn * eps
            break
    else:
        pytest.fail("many-body spectrum is not the subset sums of the single-particle energies")
    for r in s7["evidence"]["rows"]:
        b = 1 / r["T"]
        f = 1 / (np.exp(b * eps) + 1)
        lnZ = np.sum(np.log1p(np.exp(-b * eps)))
        U = np.sum(eps * f)
        C = b * b * np.sum(eps ** 2 * f * (1 - f))
        F = -r["T"] * lnZ
        assert r["lnZ"][0] <= lnZ <= r["lnZ"][1]
        assert r["F"][0] <= F <= r["F"][1]
        assert r["U"][0] <= U <= r["U"][1]
        assert r["C"][0] <= C <= r["C"][1]
        assert r["C"][1] - r["C"][0] < 1e-8
    assert any("UNVERIFIED" in a for a in s7["assumptions"])


def test_s7_not_checked_reasons():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec, ce.Options.from_preset("whole-spectrum"))
    s7 = [s for s in c.statements if s["id"] == "S7.thermodynamics"][0]
    assert s7["status"] == "NOT_CHECKED" and s7["evidence"]["reason_code"] == "NEEDS_TEMPERATURES"
    # an incomplete claim (one vector duplicated): S3 cannot certify completeness, S7 stays out
    V2 = V.copy()
    V2[:, 1] = V2[:, 0]
    c = ce.certify(ce.Claim(w, V2, operator=H, kind="all"), spec, ce.Options.from_preset("whole-spectrum", temperatures=[1.0]))
    s7 = [s for s in c.statements if s["id"] == "S7.thermodynamics"][0]
    assert s7["status"] == "NOT_CHECKED" and s7["evidence"]["reason_code"] == "NEEDS_WHOLE_SPECTRUM"
    assert math.isfinite(s7["seconds"])
