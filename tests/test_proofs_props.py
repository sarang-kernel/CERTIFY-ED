"""Pinning tests for the derivations in docs/proofs (D-59): Lemma 17, Propositions 14, 15, 16.

Each verification script exposes run(n_random, seed); the tests run it at reduced size (the documents record
the full runs) and pin the code facts the proofs rely on: the series argument range and the exp argument
guards of ivl.py, the rigorous modulus bound of correlations.py, the open-interval wording of the S8
statement. Negative controls show that the verification harnesses detect a false enclosure.
"""
import importlib.util
import math
import os
from fractions import Fraction

import numpy as np
import pytest

from certify_ed import ivl
from certify_ed.correlations import combine_certified, modulus_upper
from certify_ed.thermo import thermodynamics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROOFS = os.path.join(ROOT, "docs", "proofs")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROOFS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script,n", [("verify_L17", 60), ("verify_P16", 4), ("verify_P14", 200), ("verify_P15", 40)])
def test_verification_scripts_report_zero_violations(script, n):
    pytest.importorskip("mpmath")
    r = _load(script).run(n, seed=1)
    assert r["violations"] == 0, r
    assert r["random_cases"] > 0 and r["adversarial_cases"] > 0


def test_proof_documents_exist_with_the_six_sections():
    for name in ("L17_dyadic_exp_ln", "P14_degeneracy_bracketing", "P15_summed_bounds", "P16_thermo_intervals"):
        text = open(os.path.join(PROOFS, name + ".md"), encoding="utf-8").read()
        for sec in ("## 1. Statement", "## 2. Proof", "## 3. Where the code implements it",
                    "## 4. Computational verification", "## 5. Analytic cross-check", "## 6. Status"):
            assert sec in text, (name, sec)
        assert "Derived by Claude on" in text and "machine-checked (" in text
        assert "independent re-derivation: pending (Phase 2); Sarang's check (D-12): pending." in text
        assert "verified" not in text.lower().replace("unverified", "").replace("machine-checked", "")


# ------------------------------------------------------------------------------------------- Lemma 17 pins
def test_atanh_series_argument_range_is_enforced():
    with pytest.raises(ValueError):
        ivl._atanh2_series(0, ivl._U_MAX + 1)
    assert ivl._U_MAX >= -((-ivl.ONE) // 3)          # ceil(ONE/3) is admitted (u_hi of ln at m -> 2)
    assert Fraction(ivl._U_MAX, ivl.ONE) <= Fraction(7, 20)


def test_exp_argument_guards():
    lo, hi = ivl.exp_interval(ivl.from_fraction(Fraction(-1001))).lo, ivl.exp_interval(ivl.from_fraction(Fraction(-1001))).hi
    assert lo == 0 and hi == 1                          # [0, 2^-PREC] encloses exp(-1001) ~ 1e-435
    with pytest.raises(OverflowError):
        ivl.exp_interval(ivl.from_fraction(Fraction((1 << 16) + 1)))
    e = ivl.exp_interval(ivl.from_fraction(Fraction(1 << 16)))      # the limit itself is formed
    assert e.lo > 0


def test_exp_of_ln_contains_the_argument():
    for z in (Fraction(1), Fraction(2), Fraction(7, 3), Fraction(2 * ivl.ONE - 1, ivl.ONE), Fraction(10) ** 30,
              Fraction(1, 10 ** 30)):
        Z = ivl.from_fraction(z)
        back = ivl.exp_interval(ivl.ln_interval(Z))
        lo, hi = back.fractions()
        assert lo <= z <= hi


# ------------------------------------------------------------------------------------------- Prop 15 pins
def test_modulus_upper_is_an_upper_bound_without_libm():
    for z in (3 + 4j, 0.6 + 0.8j, 1e-160 + 1j, 2.0 ** -501 + 1j, 1 + 0j, 0 + 1j, 1e153 + 1e153j, 5e-324 + 1j):
        m = modulus_upper(z)
        assert Fraction(m) ** 2 >= Fraction(z.real) ** 2 + Fraction(z.imag) ** 2
    assert modulus_upper(3 + 4j) <= 5 * (1 + 1e-14)


def test_combine_certified_bound_grows_with_m_and_covers_cancellation():
    val, b1 = combine_certified([1.0], [1.0], [0.0])
    val2, b2 = combine_certified([1.0] * 100, [1.0] * 100, [0.0] * 100)
    assert b2 > b1 > 0 and val == 1 and val2 == 100
    val, bound = combine_certified([1 + 1j, 1 - 1j, -2.0], [3.7, 3.7, 3.7], [0.0, 0.0, 0.0])
    assert abs(val) <= bound                           # exact cancellation: val is rounding noise inside the bound
    # the certified deviation dominates the rounding term for ordinary inputs
    val, bound = combine_certified([0.5, 0.5], [1.0, 2.0], [1e-10, 1e-10])
    assert 1e-10 <= bound <= 1e-10 * (1 + 1e-4)        # rounding term 2 gamma_4 * 1.5 ~ 1.3e-15


# ------------------------------------------------------------------------------------------- Prop 16 pins
def test_thermo_refuses_sub_grid_temperature():
    with pytest.raises(ZeroDivisionError):
        thermodynamics([{"lo": 0.0, "hi": 0.0, "count": 1}], [1e-300])


def test_thermo_harness_detects_a_false_enclosure():
    """Negative control of verify_P16: a spectrum with an eigenvalue outside the enclosure is outside the
    thermodynamic intervals at some temperature."""
    mp = pytest.importorskip("mpmath")
    v16 = _load("verify_P16")
    ivs = [{"lo": 0.0, "hi": 0.0, "count": 1}, {"lo": 1.0, "hi": 1.0, "count": 1}]
    res = thermodynamics(ivs, [0.3, 1.0])
    wrong = [0.0, 1.001]                               # the second eigenvalue is NOT in its interval
    hits = [v16._inside(row, *v16.exact_thermo(wrong, row["T"])) for row in res["rows"]]
    assert not all(hits)
    good = [0.0, 1.0]
    assert all(v16._inside(row, *v16.exact_thermo(good, row["T"])) for row in res["rows"])
    del mp


# ------------------------------------------------------------------------------------------- Prop 14 pins
def test_s8_statement_names_the_open_interval_and_the_proof():
    import scipy.sparse as sp
    from certify_ed.degeneracy import degeneracy_counts
    from certify_ed.enclosure import OperatorAdapter, enclose
    d = np.array([0.0, 1.0, 1.0, 2.5])
    A = sp.diags(d).tocsr()
    enc = enclose(OperatorAdapter(A), d, np.eye(4))
    r = degeneracy_counts(A, enc, [[1, 2]], method="dense")[0]
    assert r["status"] == "CERTIFIED" and "open interval (" in r["statement"]
    assert "docs/proofs/P14_degeneracy_bracketing.md" in r["statement"] and "UNVERIFIED" in r["statement"]
    lo, hi = r["bracket_inner"]
    assert int(np.sum((d > lo) & (d < hi))) == 2


def test_p14_harness_detects_a_wrong_count():
    """Negative control of verify_P14 part A: a count that is off by one violates (b) or (c)."""
    v14 = _load("verify_P14")
    lam = [Fraction(0), Fraction(1), Fraction(1), Fraction(3)]
    s_lo, s_hi, b = Fraction(1, 2), Fraction(2), Fraction(1, 100)
    nu_lo, nu_hi = 1, 3                                 # correct
    n_in = v14._in_open(lam, s_lo + b, s_hi - b)
    assert n_in <= nu_hi - nu_lo <= v14._in_closed(lam, s_lo - b, s_hi + b)
    assert not (n_in <= (nu_hi - 1) - nu_lo)            # an undercount breaks (b)
    assert math.isfinite(float(b))
