"""C-09 level statistics (interval r-ratios, spacing histogram, declared unfolding) and C-10 extrapolation report
(per-size certified intervals by hash, declared model, extrapolated value never carries a status)."""
import json
import re

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import Certificate
from certify_ed.extrapolation import NOT_CERTIFIED_SENTENCE, extrapolation_report
from certify_ed.levelstats import LevelStatisticsError, format_level_statistics, level_statistics

STATUS_WORDS = ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED", "CORROBORATED")


@pytest.fixture(scope="module")
def goe60():
    """A dense symmetric matrix without symmetries: a complete, counted, nondegenerate spectrum."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 60))
    A = (A + A.T) / 2
    w, V = la.eigh(A)
    cert = ce.certify(ce.Claim(w, V, operator=A, kind="all"), None, ce.Options(eigenspaces="none"))
    assert cert.status_of("S2.eigenpairs") == "CERTIFIED" and cert.status_of("S3.completeness") == "CERTIFIED"
    return A, w, cert


@pytest.fixture(scope="module")
def heis_all():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec, ce.Options.from_preset("whole-spectrum"))
    assert cert.status_of("S3.completeness") == "CERTIFIED"
    return spec, w, cert


# ----------------------------------------------------------------------------- level statistics

def _r_float(w):
    s = np.diff(np.sort(w))
    return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])


def test_levelstats_intervals_contain_float_values(goe60):
    A, w, cert = goe60
    rec = level_statistics(cert)
    assert rec["dimension"] == 60 and rec["n_levels"] == 60 and rec["n_ratios_undefined"] == 0
    r_true = _r_float(w)
    for iv, r, rf in zip(rec["r_ratios"], r_true, rec["r_ratios_float"]):
        assert iv[0] <= r <= iv[1] and iv[0] <= rf <= iv[1]
        assert 0 <= iv[0] and iv[1] <= 1 and iv[1] - iv[0] < 1e-9
    for iv, s in zip(rec["spacings"], np.diff(np.sort(w))):
        assert iv[0] <= s <= iv[1] and iv[0] >= 0
    lo, hi = rec["mean_r"]
    assert lo <= float(np.mean(r_true)) <= hi and lo <= rec["mean_r_float"] <= hi
    h = rec["histogram"]
    assert sum(h["at_least"]) + h["above_range_at_least"] <= len(rec["spacings"]) and all(
        a <= b for a, b in zip(h["at_least"], h["at_most"]))
    mean_s = (w[-1] - w[0]) / 59
    assert h["mean_spacing"][0] <= mean_s <= h["mean_spacing"][1]
    counts, _ = np.histogram(np.diff(np.sort(w)) / mean_s, bins=np.array(h["edges"]))
    assert all(a <= c <= b for a, b, c in zip(h["at_least"], h["at_most"], counts))
    assert rec["unfolding"]["kind"] == "none" and rec["unfolding"]["status"] == "declared, not certified"
    text = format_level_statistics(rec)
    assert "declared, not certified" in text and "mean r in" in text


def test_levelstats_polynomial_unfolding_is_declared(goe60):
    A, w, cert = goe60
    rec = level_statistics(cert, unfolding=5)
    u = rec["unfolding"]
    assert u["kind"] == "polynomial" and u["degree"] == 5 and len(u["coefficients"]) == 6
    assert u["status"] == "declared, not certified" and "modelling choice" in u["note"]
    assert "increasing_certified" in u and "derivative_on_hull" in u
    # the unfolded float values lie in the unfolded intervals
    p = np.polyval(u["coefficients"], np.sort(w))
    for iv, s in zip(rec["spacings"], np.diff(p)):
        assert iv[0] <= s <= iv[1]
    for iv, rf in zip(rec["r_ratios"], rec["r_ratios_float"]):
        if iv is not None:
            assert iv[0] <= rf <= iv[1]
    if u["increasing_certified"]:
        assert rec["histogram"] is not None
    with pytest.raises(LevelStatisticsError):
        level_statistics(cert, unfolding=0)


def test_levelstats_degenerate_spectrum_has_undefined_ratios(heis_all):
    spec, w, cert = heis_all
    rec = level_statistics(cert)
    assert rec["dimension"] == 256 and rec["n_levels"] < 256
    assert rec["n_ratios_undefined"] > 0                      # inside a multiplet both spacings may be zero
    for iv, rf in zip(rec["r_ratios"], rec["r_ratios_float"]):
        assert (iv is None) == (rf is None)
        if iv is not None:
            assert iv[0] <= rf <= iv[1]
    # every spacing interval contains the exact spacing of the dense spectrum
    for iv, s in zip(rec["spacings"], np.diff(w)):
        assert iv[0] <= s <= iv[1] + 1e-15


def test_levelstats_refuses_without_counted_spectrum(goe60):
    A, w, cert = goe60
    low = ce.certify(ce.Claim(w[:5], la.eigh(A)[1][:, :5], operator=A, kind="lowest"), None, ce.Options())
    assert low.status_of("S3.completeness") == "CERTIFIED"
    with pytest.raises(LevelStatisticsError, match="kind 'all'"):
        level_statistics(low)
    V = la.eigh(A)[1]
    skipped = ce.certify(ce.Claim(w, V, operator=A, kind="all"), None, ce.Options(skip=["S3"], narrowing_reason="t"))
    with pytest.raises(LevelStatisticsError, match="S3.completeness is NOT_CHECKED"):
        level_statistics(skipped)
    incomplete = ce.certify(ce.Claim(w[:59], V[:, :59], operator=A, kind="all"), None, ce.Options())
    assert incomplete.status_of("S3.completeness") == "INCONCLUSIVE"
    with pytest.raises(LevelStatisticsError, match="not CERTIFIED"):
        level_statistics(incomplete)


def test_levelstats_cli(goe60, tmp_path):
    from certify_ed.cli import main
    A, w, cert = goe60
    path = cert.save(str(tmp_path / "all"))
    out = str(tmp_path / "ls.json")
    assert main(["levelstats", path, "--unfold", "poly:3", "--bins", "6", "--json", out]) == 0
    with open(out) as f:
        rec = json.load(f)
    assert rec["unfolding"]["degree"] == 3 and len(rec["histogram"]["edges"]) == 7
    with open(path) as f:
        assert rec["certificate_hash"] == json.load(f)["integrity"]["body_sha256"]
    low = ce.certify(ce.Claim(w[:3], la.eigh(A)[1][:, :3], operator=A, kind="lowest"), None, ce.Options())
    plow = low.save(str(tmp_path / "low"))
    assert main(["levelstats", plow]) == 2                    # refused with a message, not a traceback


# ----------------------------------------------------------------------------- extrapolation report

@pytest.fixture(scope="module")
def chain_certs(tmp_path_factory):
    d = tmp_path_factory.mktemp("sizes")
    certs, paths, energies = [], [], {}
    for N in (4, 6, 8):
        spec = ce.heisenberg_chain(N)
        H = ce.build_bitwise(spec)
        w, V = la.eigh(H.toarray())
        c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec, ce.Options(eigenspaces="none"))
        assert c.status_of("S3.completeness") == "CERTIFIED"
        paths.append(c.save(str(d / f"n{N}")))
        certs.append(Certificate.load(paths[-1]))
        energies[N] = float(w[0])
    return certs, paths, energies


def test_extrapolation_report_lists_certificate_intervals_and_declares_the_model(chain_certs):
    certs, paths, energies = chain_certs
    rep = extrapolation_report(certs, model="1/N", per_site=True)
    assert [r["N"] for r in rep["rows"]] == [4, 6, 8]
    for r, path in zip(rep["rows"], paths):
        with open(path) as f:
            body = json.load(f)
        assert r["hash"] == body["integrity"]["body_sha256"] and r["hash_source"].startswith("saved")
        e = energies[r["N"]] / r["N"]
        assert r["interval"][0] <= e <= r["interval"][1] and r["rank_certified"] and r["caveat"] is None
        # the per-size interval is the certificate's S2 interval divided by N (rounded outward)
        enc = body["enclosures"][0]
        assert r["interval"][0] <= enc["lower"] / r["N"] and r["interval"][1] >= enc["upper"] / r["N"]
    fit = rep["fit"]
    assert fit["model"] == "1/N" and len(fit["coefficients"]) == 2 and fit["n_points"] == 3
    ex = rep["extrapolated"]
    assert ex["value"] is not None and "status" not in ex and ex["sentence"] == NOT_CERTIFIED_SENTENCE
    assert ex["depends_on"]["model"] == "1/N" and ex["depends_on"]["certificates"] == [r["hash"] for r in rep["rows"]]
    md = rep["markdown"]
    assert "NOT certified" in md and "depends" in NOT_CERTIFIED_SENTENCE
    # the extrapolation section carries no status word
    section = md.split("## Extrapolated value")[1]
    assert not any(wd in section for wd in STATUS_WORDS), section
    assert "not certified" in section.lower()
    for r in rep["rows"]:
        assert r["hash"] in md
    # the 1/N^2 model gives a different value: the number depends on the declared model
    rep2 = extrapolation_report(certs, model="1/N^2", per_site=True)
    assert rep2["extrapolated"]["value"] != ex["value"] and rep2["fit"]["model"] == "1/N^2"
    # custom callable
    rep3 = extrapolation_report(certs, model=lambda N: [1.0 / N, 1.0 / N ** 2], per_site=True)
    assert len(rep3["fit"]["coefficients"]) == 3 and rep3["extrapolated"]["model"].startswith("custom:")
    with pytest.raises(ValueError):
        extrapolation_report(certs, model="exp")
    with pytest.raises(ValueError):
        extrapolation_report(certs, sizes=[4, 6])


def test_extrapolation_report_without_s3_carries_caveat(chain_certs):
    certs, paths, energies = chain_certs
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec,
                   ce.Options(skip=["S3"], narrowing_reason="t", eigenspaces="none"))
    rep = extrapolation_report([certs[0], c, certs[2]], model="1/N")
    row = rep["rows"][1]
    assert row["rank_certified"] is False and "NOT_CHECKED" in row["caveat"]
    assert "Caveats:" in rep["markdown"] and "rank not certified" in rep["markdown"]
    section = rep["markdown"].split("## Extrapolated value")[1]
    assert not any(wd in section for wd in STATUS_WORDS)
    # too few sizes for the model: no fitted value, still no status word
    one = extrapolation_report([certs[0]], model="1/N")
    assert one["extrapolated"]["value"] is None and "No fitted limit" in one["markdown"]


def test_extrapolation_cli(chain_certs, tmp_path):
    from certify_ed.cli import main
    certs, paths, energies = chain_certs
    md = str(tmp_path / "rep.md")
    js = str(tmp_path / "rep.json")
    assert main(["extrapolate", *paths, "--model", "1/N^2", "--per-site", "-o", md, "--json", js]) == 0
    text = open(md, encoding="utf-8").read()
    assert "# Finite-size report" in text and "NOT certified" in text
    with open(js) as f:
        rec = json.load(f)
    assert rec["fit"]["model"] == "1/N^2" and "markdown" not in rec and "status" not in rec["extrapolated"]
    assert re.search(r"\| 8 \|", text)
