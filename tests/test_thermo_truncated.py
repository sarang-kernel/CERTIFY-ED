"""C-06: S7 thermodynamic intervals from a 'lowest' claim with S3 CERTIFIED (Proposition 16', the tail bound
(n - k) exp(-beta L)). Checked against the full spectrum of chains; negative controls: a claim without S3, a
planted wrong L, a dimension that is too small."""
import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.thermo import thermodynamics, thermodynamics_truncated


def _s7(cert):
    return [s for s in cert.statements if s["id"] == "S7.thermodynamics"][0]


def _contains(outer, inner):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def test_truncated_encloses_full_spectrum_on_chains():
    """Open Heisenberg N = 6, 8 and free fermions N = 5: for every k that ends at a gap, the truncated intervals
    (with and without the norm bound) contain the full-spectrum intervals at low and moderate temperatures."""
    temps = [0.05, 0.1, 0.25, 0.5, 1.0]
    for spec in (ce.heisenberg_chain(6), ce.heisenberg_chain(8), ce.spinless_fermion_chain(5, t=1.0, V=0.5, boundary="open")):
        H = ce.build_bitwise(spec).toarray()
        w = la.eigvalsh(H)
        full = thermodynamics([{"lo": float(x), "hi": float(x), "count": 1} for x in w], temps)
        M = float(np.abs(H).sum(axis=1).max())
        for k in [k for k in range(1, 7) if w[k] - w[k - 1] > 1e-6]:
            ivs = [{"lo": float(x), "hi": float(x), "count": 1} for x in w[:k]]
            L = float(w[k]) - 1e-9
            for upper in (M, None):
                res = thermodynamics_truncated(ivs, L, w.size, temps, upper_bound=upper)
                assert res["tail_bound"]["omitted_count"] == w.size - k
                for row, fr in zip(res["rows"], full["rows"]):
                    for q in ("F", "U", "C", "lnZ"):
                        assert _contains(row[q], fr[q]), (spec.name, k, upper, q, row["T"], row[q], fr[q])
                    assert row["tail"]["weight_upper"] >= 0


def test_low_temperature_widths_shrink_with_the_gap():
    """At T far below the gap the truncated intervals are nearly as tight as the full-spectrum ones."""
    spec = ce.heisenberg_chain(8)
    w = la.eigvalsh(ce.build_bitwise(spec).toarray())
    k = 4
    ivs = [{"lo": float(x), "hi": float(x), "count": 1} for x in w[:k]]
    res = thermodynamics_truncated(ivs, float(w[k]) - 1e-9, w.size, [0.02, 0.05])
    for row in res["rows"]:
        assert row["F"][1] - row["F"][0] < 1e-6 and row["U"][1] - row["U"][0] < 1e-4
        assert row["tail"]["weight_fraction_upper"] < 1e-3


def test_certify_lowest_claim_gets_s7_with_tail_bound(heis8):
    spec, H, w, V = heis8
    k = 4
    cert = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec,
                      ce.Options(temperatures=[0.05, 0.2], eigenspaces="none"))
    s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]
    assert s3["status"] == "CERTIFIED"
    s7 = _s7(cert)
    assert s7["status"] == "CERTIFIED", s7["summary"]
    tb = s7["evidence"]["tail_bound"]
    assert tb["omitted_count"] == 2 ** 8 - k and tb["method"].startswith("envelope")
    assert tb["next_lower_bound"] == s3["evidence"]["next_eigenvalue_lower_bound"]
    assert tb["upper_bound"] is not None and tb["upper_bound"] >= w[-1]
    assert "tail bound" in s7["summary"] and any("C06" in a for a in s7["assumptions"])
    full = thermodynamics([{"lo": float(x), "hi": float(x), "count": 1} for x in w], [0.05, 0.2])
    for row, fr in zip(s7["evidence"]["rows"], full["rows"]):
        for q in ("F", "U", "C"):
            assert _contains(row[q], fr[q])
    # the certificate body validates against the schema (new evidence keys are declared)
    from certify_ed.schema import structural_check
    assert not structural_check(cert.body)


def test_negative_control_claim_without_s3_is_not_certified(heis8):
    spec, H, w, V = heis8
    # subset claim: no completeness statement, hence no L
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="subset"), spec,
                      ce.Options(temperatures=[0.1], eigenspaces="none"))
    s7 = _s7(cert)
    assert s7["status"] == "NOT_CHECKED" and s7["evidence"]["reason_code"] == "NEEDS_WHOLE_SPECTRUM"
    # lowest claim with S3 skipped: the tail bound is unavailable
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec,
                      ce.Options(temperatures=[0.1], eigenspaces="none", skip=["S3"], narrowing_reason="test"))
    assert not [s for s in cert.statements if s["id"] == "S7.thermodynamics"]      # narrowed with its dependency
    assert any(x["id"] == "S7.thermodynamics" for x in cert.body["scope"]["narrowed"])
    # lowest claim whose S3 fails (a member of the triplet omitted: k = 2 cuts the multiplet)
    cert = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest"), spec,
                      ce.Options(temperatures=[0.1], eigenspaces="none"))
    s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]
    assert s3["status"] != "CERTIFIED"
    assert _s7(cert)["status"] == "NOT_CHECKED"


def test_negative_control_planted_wrong_lower_bound_is_exposed():
    """A wrong L (above lambda_{k+1}) produces intervals that miss the true values; a correct L does not."""
    spec = ce.heisenberg_chain(6)
    w = la.eigvalsh(ce.build_bitwise(spec).toarray())
    k = 4
    ivs = [{"lo": float(x), "hi": float(x), "count": 1} for x in w[:k]]
    temps = [0.3, 1.0]
    full = thermodynamics([{"lo": float(x), "hi": float(x), "count": 1} for x in w], temps)
    good = thermodynamics_truncated(ivs, float(w[k]) - 1e-9, w.size, temps)
    assert all(_contains(r[q], f[q]) for r, f in zip(good["rows"], full["rows"]) for q in ("F", "U"))
    wrong = thermodynamics_truncated(ivs, float(w[k]) + 1.0, w.size, temps)       # planted: L above lambda_{k+1}
    assert not all(_contains(r[q], f[q]) for r, f in zip(wrong["rows"], full["rows"]) for q in ("F", "U"))
    small = thermodynamics_truncated(ivs, float(w[k]) - 1e-9, k + 1, temps)          # planted: dimension too small
    assert not all(_contains(r["F"], f["F"]) for r, f in zip(small["rows"], full["rows"]))
    with pytest.raises(ValueError):
        thermodynamics_truncated(ivs, float(w[k - 1]), w.size, temps)                 # L not above the claim
    with pytest.raises(ValueError):
        thermodynamics_truncated(ivs, float(w[k]), w.size, temps, upper_bound=float(w[k]) - 1.0)
