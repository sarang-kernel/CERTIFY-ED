"""C-01 gap records and C-11 certified crossings (Wave 1, track/statements).

Soundness: every gap interval and every crossing is read off S2/S3 data that is already certified; the
negative controls check that a planted error removes the record rather than producing a wrong one.
"""
import json
import math
import os

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import Certificate
from certify_ed.crossings import certificate_hash, certified_crossing, format_crossing, level_interval
from certify_ed.gaps import gap_records, merge_enclosures


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _cert(spec, H, w, V, k, **opts):
    return ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, ce.Options(**opts))


# ----------------------------------------------------------------------------- merge_enclosures

def test_merge_enclosures_chains_and_counts():
    mu = np.array([0.0, 1.0, 1.0 + 1e-12, 3.0])
    rad = np.array([1e-10, 1e-10, 1e-10, 1e-10])
    lv = merge_enclosures(mu, rad)
    assert [x["count"] for x in lv] == [1, 2, 1]
    assert lv[1]["members"] == [1, 2]
    assert lv[0]["upper"] >= 1e-10 and lv[0]["lower"] <= -1e-10          # outward rounding
    assert lv[1]["lower"] <= 1.0 - 1e-10 and lv[1]["upper"] >= 1.0 + 1e-12 + 1e-10


# ----------------------------------------------------------------------------- gap records

def test_gap_record_contains_true_gaps_and_has_ranks(heis8):
    spec, H, w, V = heis8
    cert = _cert(spec, H, w, V, 5)          # singlet, triplet, singlet: 3 levels, ranks 1 | 2-4 | 5
    assert cert.status_of("S2.eigenpairs") == "CERTIFIED" and cert.status_of("S3.completeness") == "CERTIFIED"
    g = cert.body["gaps"]
    assert g is not None and g["ranked"] and g["n_levels"] == 3
    assert [lv["ranks"] for lv in g["levels"]] == [[1, 1], [2, 4], [5, 5]]
    assert g["refers_to"] == "declared model (spec)"
    by_ranks = {tuple(x["between_ranks"]): x for x in g["gaps"]}
    # between levels: both ends from S2, interval contains the true gap
    for (r0, r1), true in (((1, 2), w[1] - w[0]), ((4, 5), w[4] - w[3])):
        rec = by_ranks[(r0, r1)]
        assert rec["lower_from"] == "S2" and rec["upper_from"] == "S2" and not rec["within_level"]
        assert rec["interval"][0] <= true <= rec["interval"][1]
        assert rec["interval"][1] - rec["interval"][0] < 1e-11
    # inside the triplet: [0, width]
    rec = by_ranks[(2, 4)]
    assert rec["within_level"] and rec["interval"][0] == 0.0 and rec["interval"][1] >= w[3] - w[1]
    # the gap above the highest claimed level: lower bound from S3, no upper bound
    top = g["top_gap"]
    assert top is not None and top["lower_from"] == "S3" and top["upper_from"] is None
    assert top["between_ranks"] == [5, 6] and math.isinf(top["interval"][1])
    assert top["interval"][0] <= w[5] - w[4]
    s3 = cert.statement("S3.completeness")["evidence"]
    assert s3["gap_interval"] == top["interval"] and s3["gap_interval"][0] == s3["gap_lower_bound"]


def test_gap_record_next_pair_missing_has_no_upper_bound(heis8):
    """A k-pair claim: the gap to the (k+1)-th eigenvalue is bounded below by S3 only."""
    spec, H, w, V = heis8
    cert = _cert(spec, H, w, V, 4)
    g = cert.body["gaps"]
    assert g["top_gap"]["between_ranks"] == [4, 5]
    assert math.isinf(g["top_gap"]["interval"][1]) and g["top_gap"]["interval"][0] <= w[4] - w[3]
    # claiming the next pair as well bounds that gap on both sides
    c5 = _cert(spec, H, w, V, 5)
    rec = next(x for x in c5.body["gaps"]["gaps"] if x["between_ranks"] == [4, 5])
    assert rec["upper_from"] == "S2" and rec["interval"][0] <= w[4] - w[3] <= rec["interval"][1] < math.inf
    assert rec["interval"][0] >= g["top_gap"]["interval"][0]       # the S2 bound is at least as tight as S3's


def test_gap_record_negative_control_wrong_next_eigenvalue(heis8):
    """A planted wrong (k+1)-th eigenvalue: S2 FAILED, no gap record, no gap_interval in S3."""
    spec, H, w, V = heis8
    mu = np.r_[w[:4], w[4] + 0.2]
    cert = ce.certify(ce.Claim(mu, V[:, :5], operator=H, kind="lowest"), spec, ce.Options())
    assert cert.status_of("S2.eigenpairs") == "FAILED"
    assert cert.body["gaps"] is None
    assert "gap_interval" not in cert.statement("S3.completeness")["evidence"]


def test_gap_record_absent_without_s3(heis8):
    spec, H, w, V = heis8
    cert = _cert(spec, H, w, V, 4, skip=["S3"], narrowing_reason="test")
    assert cert.body["gaps"] is None
    sub = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="subset"), spec, ce.Options())
    assert sub.body["gaps"] is None


def test_gap_record_window_claim_is_not_ranked(heis8):
    spec, H, w, V = heis8
    # levels of the N = 8 ring: singlet, triplet, singlet (index 4), sextet (5-10), ...: window around ranks 5-11
    claim = ce.Claim(w[4:11], V[:, 4:11], operator=H, kind="window", window=(float(w[4]) - 1e-3, float(w[10]) + 1e-3))
    cert = ce.certify(claim, spec, ce.Options())
    assert cert.status_of("S3.completeness") == "CERTIFIED", cert.statement("S3.completeness")["summary"]
    g = cert.body["gaps"]
    assert g is not None and not g["ranked"] and g["top_gap"] is None
    assert all("ranks" not in lv for lv in g["levels"]) and [lv["count"] for lv in g["levels"]] == [1, 6]
    assert g["gaps"][0]["interval"][0] <= w[5] - w[4] <= g["gaps"][0]["interval"][1]
    assert "gap_interval" not in cert.statement("S3.completeness")["evidence"]
    for rec in g["gaps"]:
        assert rec["interval"][0] <= rec["interval"][1]


def test_gap_record_survives_save_load_and_schema(heis8, tmp_path):
    spec, H, w, V = heis8
    cert = _cert(spec, H, w, V, 5)
    path = cert.save(str(tmp_path / "c5"))
    loaded = Certificate.load(path)
    assert loaded.validate_schema() in (True, None)
    with open(path) as f:
        body = json.load(f)
    assert body["gaps"]["top_gap"]["interval"][1] == "inf"
    assert body["statements"][2]["evidence"]["gap_interval"][1] == "inf"


def test_gap_records_function_rounds_outward():
    mu = np.array([0.0, 1.0])
    rad = np.array([0.1, 0.1])
    g = gap_records(mu, rad, "lowest", {"gap_lower_bound": 0.5}, "producer operator")
    rec = g["gaps"][0]
    assert rec["interval"][0] <= 0.8 and rec["interval"][1] >= 1.2
    assert g["top_gap"]["interval"] == [0.5, math.inf]
    g_all = gap_records(mu, rad, "all", {}, "producer operator")
    assert g_all["top_gap"] is None and len(g_all["gaps"]) == 1


# ----------------------------------------------------------------------------- crossings

@pytest.fixture(scope="module")
def two_couplings():
    out = {}
    # J2 = 0: singlet + triplet (k = 4); J2 = 0.3: singlet, singlet, triplet (k = 5); both complete below a gap
    for J2, k in ((0.0, 4), (0.3, 5)):
        spec = ce.heisenberg_chain(8, J2=J2)
        H = ce.build_bitwise(spec)
        w, V = la.eigh(H.toarray())
        out[J2] = (ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, ce.Options()), w)
    return out


def test_crossing_orders_ground_states_at_two_couplings(two_couplings):
    ca, wa = two_couplings[0.0]
    cb, wb = two_couplings[0.3]
    assert ca.status_of("S3.completeness") == "CERTIFIED" and cb.status_of("S3.completeness") == "CERTIFIED"
    rec = certified_crossing(ca, cb, level=0)
    assert rec["relation"] == "<" and rec["rank_certified"] and not rec["caveats"]
    assert rec["margin"] > 0 and rec["margin"] <= wb[0] - wa[0]
    assert rec["a"]["interval"][0] <= wa[0] <= rec["a"]["interval"][1]
    assert rec["b"]["interval"][0] <= wb[0] <= rec["b"]["interval"][1]
    assert rec["a"]["hash"] == certificate_hash(ca) and rec["b"]["hash"] == certificate_hash(cb)
    assert rec["statement"].startswith("lambda_0(A) < lambda_0(B)")
    assert not rec["same_spec_sha256"]
    # reversed arguments reverse the relation
    rev = certified_crossing(cb, ca, level=0)
    assert rev["relation"] == ">" and rev["margin"] == rec["margin"]
    # level 1: the triplet of J2 = 0 (-3.128) against the second singlet of J2 = 0.3 (-2.851)
    r1 = certified_crossing(ca, cb, level=1)
    assert r1["a"]["level_count"] == 3 and r1["b"]["level_count"] == 1 and r1["relation"] == "<"
    assert r1["margin"] <= wb[1] - wa[1]
    assert r1["a"]["interval"][0] <= wa[1] <= r1["a"]["interval"][1]
    text = format_crossing(rec)
    assert "relation: <" in text and rec["a"]["hash"] in text


def test_crossing_overlapping_intervals_not_separated(two_couplings):
    ca, _ = two_couplings[0.0]
    rec = certified_crossing(ca, ca, level=0)
    assert rec["relation"] == "not separated" and rec["margin"] <= 0
    assert "not separated" in rec["statement"]


def test_crossing_without_s3_states_interval_of_claimed_value(two_couplings, heis8):
    spec, H, w, V = heis8
    cb, _ = two_couplings[0.3]
    cs = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec,
                    ce.Options(skip=["S3"], narrowing_reason="test"))
    assert cs.status_of("S3.completeness") == "NOT_CHECKED"
    rec = certified_crossing(cs, cb, level=0)
    assert rec["relation"] == "<" and not rec["rank_certified"]
    assert len(rec["caveats"]) == 1 and rec["caveats"][0].startswith("A: S3.completeness is NOT_CHECKED")
    assert "certified interval of the claimed value" in rec["statement"] and "lambda_0(A)" not in rec["statement"]
    assert rec["a"]["caveat"] and rec["b"]["caveat"] is None


def test_crossing_refuses_without_certified_s2(heis8):
    spec, H, w, V = heis8
    bad = ce.certify(ce.Claim(w[:2] + 0.3, V[:, :2], operator=H, kind="lowest"), spec, ce.Options())
    assert bad.status_of("S2.eigenpairs") == "FAILED"
    with pytest.raises(ValueError, match="S2.eigenpairs is FAILED"):
        level_interval(bad, 0)
    good = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest"), spec, ce.Options())
    with pytest.raises(ValueError, match="out of range"):
        level_interval(good, 2)


def test_crossing_cli_and_json(two_couplings, tmp_path):
    from certify_ed.cli import main
    ca, _ = two_couplings[0.0]
    cb, _ = two_couplings[0.3]
    pa = ca.save(str(tmp_path / "a"))
    pb = cb.save(str(tmp_path / "b"))
    out = str(tmp_path / "x.json")
    assert main(["crossing", pa, pb, "--level", "0", "--json", out]) == 0
    with open(out) as f:
        rec = json.load(f)
    assert rec["relation"] == "<" and rec["a"]["hash_source"].startswith("saved certificate")
    with open(pa) as f:
        assert rec["a"]["hash"] == json.load(f)["integrity"]["body_sha256"]
    assert main(["crossing", pa, pa]) == 2        # not separated
    assert os.path.exists(out)
