"""Chebyshev moments (M5) and distribution bounds (M6) on a certified level: docs/proofs/C-07b_spectral_bounds.md."""
import json
import math
import random
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed import spectral_bounds as sb
from certify_ed.correlations import site_operator
from certify_ed.schema import validate_record

ONE = 1 << sb._DYADIC_BITS


@pytest.fixture(scope="module")
def chain6():
    """Heisenberg chain N = 6, certified ground state, O = staggered S^z (the structure factor at q = pi)."""
    N = 6
    spec = ce.heisenberg_chain(N)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    O = sp.csr_matrix(sum(np.cos(np.pi * j) * site_operator(spec, "z", [j]) for j in range(N)) * np.sqrt(2.0 / N))
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec,
                      ce.Options(store_eigenvectors=True, eigenspaces="all"))
    weights = (V.T @ (O @ V[:, 0])) ** 2
    return H, O, cert, w, weights


def _exact_bernstein(A, a, b):
    m = len(A) - 1
    Q = [Fraction(0)] * (m + 1)
    for k, ak in enumerate(A):
        for i in range(k + 1):
            Q[i] += ak * math.comb(k, i) * a ** (k - i) * (b - a) ** i
    return [sum((Fraction(math.comb(i, k), math.comb(m, k)) * Q[k] for k in range(i + 1)), Fraction(0))
            for i in range(m + 1)]


def test_integer_bernstein_and_split_are_lowered_exact_values():
    rng = random.Random(3)
    for _ in range(40):
        m = rng.randint(1, 9)
        A = [rng.randint(-2 ** 18, 2 ** 18) for _ in range(m + 1)]
        b = sb._bernstein_int(A)
        for got, want in zip(b, _exact_bernstein(A, Fraction(-1), Fraction(1))):
            assert Fraction(got) <= want * 2 ** sb._PREC < got + 1
        q = rng.randint(0, 20)
        p = rng.randint(0, 2 ** q)
        t = Fraction(p, 2 ** q)
        L, R = sb._split(b, p, q)
        for part, (lo, hi) in ((L, (Fraction(-1), 2 * t - 1)), (R, (2 * t - 1, Fraction(1)))):
            for got, want in zip(part, _exact_bernstein(A, lo, hi)):
                assert Fraction(got) <= want * 2 ** sb._PREC <= got + m + 1


def test_majorant_with_a_dip_of_one_unit_is_rejected():
    # p(u) = (u + 1/2)^2 + 1 + e = 1.75 + e + T_1 + T_2/2; p >= 1 on [-1, 0] iff e >= 0 (minimum at u = -1/2)
    T = sb._cheb_monomials(2)
    base = [int(1.75 * ONE), ONE, ONE // 2]
    assert sb._prove_majorant([base[0] + 1] + base[1:], T, Fraction(0), True)
    assert not sb._prove_majorant([base[0] - 1] + base[1:], T, Fraction(0), True)
    # the same polynomial is no minorant (q <= 1 on [-1, 0] fails at u = -1)
    assert not sb._prove_majorant(base, T, Fraction(0), False)


def test_interval_holds_the_spectrum_and_maps_it_into_the_unit_interval():
    rng = np.random.default_rng(0)
    M = sp.random(40, 40, density=0.1, random_state=1)
    H = ((M + M.T) + sp.diags(rng.standard_normal(40))).tocsr()
    w = np.linalg.eigvalsh(H.toarray())
    iv = sb.chebyshev_interval(H)
    lo, hi = iv["interval"]
    assert lo <= w[0] and w[-1] <= hi
    c, s = Fraction(iv["center"]), Fraction(iv["scale"])
    assert (Fraction(lo) - c) * s >= -1 and (Fraction(hi) - c) * s <= 1
    assert (Fraction(lo) - c) * s < Fraction(-999, 1000) and (Fraction(hi) - c) * s > Fraction(999, 1000)
    # a proved lower bound narrows the interval; contradictory bounds are refused
    iv2 = sb.chebyshev_interval(H, lower=float(w[0]) - 1e-9)
    assert iv2["interval"][0] == float(w[0]) - 1e-9 and iv2["interval"][1] == hi
    with pytest.raises(ValueError, match="contradict"):
        sb.chebyshev_interval(H, lower=hi + 1.0)
    # the Gershgorin bound of the Heisenberg chain is the ferromagnetic energy N/4 exactly
    spec = ce.heisenberg_chain(6)
    assert sb.chebyshev_interval(ce.build_bitwise(spec).tocsr())["gershgorin"][1] >= 1.5
    assert sb.chebyshev_interval(ce.build_bitwise(spec).tocsr())["gershgorin"][1] < 1.5 + 1e-12


def test_moments_enclose_the_exact_chebyshev_moments(chain6):
    H, O, cert, w, weights = chain6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=20)
    st = {s["id"]: s for s in rec["statements"]}
    assert st["M5.chebyshev_moments"]["status"] == "CERTIFIED"
    c, s = rec["chebyshev"]["center"], rec["chebyshev"]["scale"]
    lo = rec["chebyshev"]["spectrum_interval"][0]
    assert w[0] - 1e-8 < lo <= w[0]                     # the lower end comes from the certificate (S2, S3)
    assert "certificate" in rec["chebyshev"]["spectrum_source"]
    x = (w - c) * s
    assert np.all(np.abs(x) <= 1.0 + 1e-15)
    for m in rec["chebyshev"]["moments"]:
        exact = float(np.sum(weights * np.cos(m["j"] * np.arccos(np.clip(x, -1, 1)))))
        lo, hi = m["interval"]
        assert lo - 1e-14 <= exact <= hi + 1e-14          # reference from eigh, error far below the half-widths
        assert m["bound"] < 1e-10


def test_distribution_bounds_contain_the_exact_distribution(chain6):
    H, O, cert, w, weights = chain6
    energies = [w[0] - 0.3, w[0] + 0.05, -1.5, -0.5, 0.75]
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=24, energies=energies)
    st = {s["id"]: s for s in rec["statements"]}
    assert st["M6.distribution_bounds"]["status"] == "CERTIFIED"
    rows = sorted(rec["distribution"]["rows"], key=lambda x: x["energy"])
    tau0_hi = rec["distribution"]["total_mass_interval"][1]
    for row in rows:
        F = float(weights[w <= row["energy"]].sum())
        assert row["lower_monotone"] <= F + 1e-15 and F - 1e-15 <= row["upper_monotone"]
    lows = [row["lower_monotone"] for row in rows]
    ups = [row["upper_monotone"] for row in rows]
    assert lows == sorted(lows) and ups == sorted(ups)
    assert min(u - l for u, l in zip(ups, lows)) < 0.25 * tau0_hi       # not the trivial bound everywhere
    assert validate_record(rec) in (True, None)
    json.dumps(rec)
    assert ": F in [" in sb.format_spectral(rec)


def test_unproved_polynomial_gives_the_trivial_bound(chain6, monkeypatch):
    H, O, cert, w, weights = chain6
    monkeypatch.setattr(sb, "_prove_majorant", lambda *a, **k: False)
    monkeypatch.setattr(sb, "_ATTEMPTS", 2)
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=8, energies=[-1.0])
    row = rec["distribution"]["rows"][0]
    assert row["lower"] == 0.0 and row["upper"] == rec["distribution"]["total_mass_interval"][1]
    st = {s["id"]: s for s in rec["statements"]}
    assert "not proved" in st["M6.distribution_bounds"]["summary"]


def test_another_matrix_than_the_certificates_is_refused(chain6):
    H, O, cert, w, weights = chain6
    with pytest.raises(ValueError, match="not the matrix of the certificate"):
        sb.spectral_bounds_from_certificate(cert, (H * 1.5).tocsr(), O, K=4)


def test_distribution_needs_certified_moments(chain6):
    H, O, cert, w, weights = chain6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=4)
    for s in rec["statements"]:
        if s["id"] == "M5.chebyshev_moments":
            s["status"] = "INCONCLUSIVE"
    out = sb.distribution_bounds(rec, [-1.0])
    st = {s["id"]: s for s in out["statements"]}
    assert st["M6.distribution_bounds"]["status"] == "NOT_CHECKED"
    assert st["M6.distribution_bounds"]["evidence"]["reason_code"] == "NEEDS_DEPENDENCY"
