"""Chebyshev moments (M5) and distribution bounds (M6) on a certified level: docs/proofs/C-07b_spectral_bounds.md."""
import copy
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
    # a spectrum far from 0 (the rounded midpoint is off-centre by many ulps of the scale; found by verify_C-07b.py)
    Hf = (H + sp.identity(40) * 1.0e5).tocsr()
    ivf = sb.chebyshev_interval(Hf)
    cf, sf = Fraction(ivf["center"]), Fraction(ivf["scale"])
    assert (Fraction(ivf["interval"][0]) - cf) * sf >= -1 and (Fraction(ivf["interval"][1]) - cf) * sf <= 1
    assert (Fraction(ivf["interval"][1]) - cf) * sf > Fraction(999, 1000)
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


def test_hermite_step_polynomials_lie_on_the_side_of_the_step():
    # the Remark of C-07b, in floating point: any distinct nodes give a majorant (p(xi) = 1) and a minorant (p(xi) = 0)
    rng = np.random.default_rng(5)
    u = np.linspace(-1.0, 1.0, 4001)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(1, 10))
        x = np.sort(rng.uniform(-1, 1, n))
        xi = float(rng.uniform(-1, 1))
        if np.min(np.abs(x - xi)) < 0.05 or (n > 1 and np.min(np.diff(x)) < 0.05):
            continue
        for upper in (True, False):
            d = sb._hermite_step(x, xi, upper, 2 * n + 1)
            step = (u <= xi).astype(float)
            p = np.polynomial.chebyshev.chebval(u, d)
            gap = (p - step) if upper else (step - p)
            assert gap.min() > -1e-8 * (1.0 + np.abs(d).sum())
            checked += 1
    assert checked >= 20


def test_chebyshev_hermite_bounds_alone_contain_the_exact_distribution(chain6, monkeypatch):
    H, O, cert, w, weights = chain6
    monkeypatch.setattr(sb, "_HEURISTICS", ("chebyshev-hermite",))
    energies = [w[0] + 0.05, -1.5, -0.5]
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=24, energies=energies)
    for row in rec["distribution"]["rows"]:
        F = float(weights[w <= row["energy"]].sum())
        assert row["lower_monotone"] <= F + 1e-15 and F - 1e-15 <= row["upper_monotone"]
        assert (row.get("upper_proof") or {}).get("method") in ("chebyshev-hermite", None)


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


def test_statements_name_the_unverified_derivation(chain6):
    H, O, cert, w, weights = chain6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=8, energies=[-1.0])
    for s in rec["statements"]:
        assert sb.DERIVATION in s["assumptions"], s["id"]
    assert "UNVERIFIED" in sb.DERIVATION and "D-12" in sb.DERIVATION
    assert sb.DERIVATION in sb.format_spectral(rec)
    assert rec["certificate"]["operator_hash_matches"] and rec["certificate"]["members"] == [0]


def test_certificate_without_a_matrix_hash_is_refused(chain6):
    H, O, cert, w, weights = chain6
    c2 = copy.deepcopy(cert)
    c2.body["subject"]["operator"]["matrix_sha256"] = None
    with pytest.raises(ValueError, match="no matrix hash"):
        sb.spectral_bounds_from_certificate(c2, H, O, K=4)


def test_lowest_bound_is_taken_only_from_a_certified_lowest_claim(chain6):
    H, O, cert, w, weights = chain6
    lo, src = sb._lowest_lower_bound(cert)
    assert lo is not None and lo <= w[0] and "certificate" in src
    other = copy.deepcopy(cert)
    other.body["claim"]["kind"] = "window"
    assert sb._lowest_lower_bound(other) == (None, "")
    sector = copy.deepcopy(cert)
    sector.body["subject"]["sector"] = {"label": "test"}
    assert sb._lowest_lower_bound(sector) == (None, "")
    no_s3 = copy.deepcopy(cert)
    real = no_s3.status_of
    no_s3.status_of = lambda sid: "INCONCLUSIVE" if sid == "S3.completeness" else real(sid)
    assert sb._lowest_lower_bound(no_s3) == (None, "")
    rec = sb.spectral_bounds_from_certificate(no_s3, H, O, K=4)
    assert rec["chebyshev"]["spectrum_source"] == "Gershgorin discs of the stored H"
    assert rec["chebyshev"]["spectrum_interval"][0] == rec["chebyshev"]["gershgorin_interval"][0]


def test_declared_spectrum_bounds_are_recorded_as_declared(chain6):
    H, O, cert, w, weights = chain6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=4, spectrum_lower=float(w[0]) - 1e-9)
    assert rec["chebyshev"]["spectrum_source"] == ("lower: declared by the caller, not checked; "
                                                  "upper: Gershgorin discs of the stored H")
    assert rec["chebyshev"]["declared_bounds"]["lower"] == float(w[0]) - 1e-9
    st = {s["id"]: s for s in rec["statements"]}
    assert any("declared by the caller" in a for a in st["M5.chebyshev_moments"]["assumptions"])


def test_cluster_record_and_one_source_per_end(chain6):
    """M5 and M6 name the cluster record taken as given; with the certificate's lower bound and a caller's upper bound
    the record names one source per end and keeps the proved end apart from the declared one."""
    from certify_ed.moments import CLUSTER_RECORD
    H, O, cert, w, weights = chain6
    alone = sb.spectral_bounds_from_certificate(cert, H, O, K=4)["chebyshev"]
    assert alone["spectrum_source"].startswith("lower: ")
    assert alone["spectrum_source"].endswith("; upper: Gershgorin discs of the stored H")
    hi = float(np.nextafter(alone["gershgorin_interval"][1], -np.inf))   # one step below the discs, still above
    assert hi >= float(w[-1]) + 1e-15                                     # the largest eigenvalue here
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=4, energies=[-1.0], spectrum_upper=hi)
    for s in rec["statements"]:
        assert CLUSTER_RECORD in s["assumptions"], s["id"]
    ch = rec["chebyshev"]
    assert ch["spectrum_source"].startswith("lower: ")
    assert "upper: declared by the caller, not checked" in ch["spectrum_source"]
    assert ch["declared_bounds"] == {"lower": None, "upper": hi}
    assert ch["proved_bounds"]["lower"] <= w[0] and "certificate" in ch["proved_bounds"]["source"]
    assert "larger of this bound and the Gershgorin bound" in ch["proved_bounds"]["use"]
    assert validate_record(rec) in (True, None)
    named = sb.spectral_bounds_from_certificate(cert, H, O, K=4, spectrum_upper=hi, spectrum_source="a test bound")
    assert named["chebyshev"]["spectrum_source"].endswith("; upper: declared by the caller, not checked (a test bound)")
    with pytest.raises(ValueError, match="give it with spectrum_upper"):
        sb.spectral_bounds_from_certificate(cert, H, O, K=4, spectrum_source="a source without a bound")


def test_the_source_names_the_gershgorin_end_where_it_is_larger(chain6, monkeypatch):
    """A certificate bound below the Gershgorin bound leaves the interval's lower end to the discs, and the record
    says so."""
    H, O, cert, w, weights = chain6
    glo = sb._gershgorin(sb._as_matrix(H, "H"))[0]
    monkeypatch.setattr(sb, "_lowest_lower_bound", lambda c: (glo - 1.0, "a bound below the discs"))
    ch = sb.spectral_bounds_from_certificate(cert, H, O, K=4)["chebyshev"]
    assert ch["spectrum_interval"][0] == glo
    assert ch["spectrum_source"].startswith("lower: Gershgorin discs of the stored H (not below the bound of the "
                                            "certificate: a bound below the discs)")
    assert ch["proved_bounds"]["lower"] == glo - 1.0


def test_a_margin_that_overflows_skips_the_candidate(monkeypatch):
    monkeypatch.setattr(sb, "_float_violation", lambda D, xi, upper: 1e300)
    val, info = sb._prove_and_bound(np.array([1.0, 0.0]), None, None, Fraction(0), True)
    assert val is None and "overflows" in info["reason"]


def test_each_end_is_named_from_the_bound_in_use(chain6):
    """certified_chebyshev_moments names each end from the bound that sets it: a caller's text stays behind "declared
    by the caller, not checked", a declared bound that the discs beat is named as such, and a text without a declared
    bound is refused."""
    H, O, cert, w, weights = chain6
    X = np.asarray(cert.arrays["eigenvectors"])[:, :1]
    ghi = sb.chebyshev_interval(H)["gershgorin"][1]
    rec = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=2, spectrum_lower=float(w[0]) - 1e-9,
                                         spectrum_source="from a run")
    assert rec["chebyshev"]["spectrum_source"] == ("lower: declared by the caller, not checked (from a run); "
                                                   "upper: Gershgorin discs of the stored H")
    rec = sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=2, spectrum_upper=ghi + 5.0)
    assert rec["chebyshev"]["spectrum_source"].endswith("upper: Gershgorin discs of the stored H (not above the "
                                                        "declared bound)")
    assert rec["chebyshev"]["spectrum_interval"][1] == ghi
    with pytest.raises(ValueError, match="names the source of a declared bound"):
        sb.certified_chebyshev_moments(H, O, X, 0.0, 0.0, K=2, spectrum_source="a text")
