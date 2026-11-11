"""Performance regression tests from E2 points (roadmap H-04). Skipped unless pytest runs with `--perf` (or the
environment variable CERTIFY_ED_PERF=1); they need QuSpin (the E2 producer).

Each small E2 point (experiments/perf_points.SMALL: dimensions 810 to 3432, a few seconds each) is measured as
experiments/e2_cost.py measures it and compared with the baseline recorded on the release machine
(tests/fixtures/perf_baseline.json, written by `python experiments/perf_points.py --record ...`):

* every statement status equals the baseline's (a statement that silently stops running is also "faster");
* certify() total seconds <= TIME_FACTOR x baseline + TIME_SLACK_S;
* peak resident memory <= MEMORY_FACTOR x baseline + MEMORY_SLACK_MB.

The factors are generous because the times of one machine vary with its load; they can be set with the
environment variables CERTIFY_ED_PERF_TIME_FACTOR and CERTIFY_ED_PERF_MEMORY_FACTOR. A baseline from another machine
says little: re-record it there first (the test prints the recorded machine).
"""
import json
import os
import sys

import pytest

from conftest import needs_quspin

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "fixtures", "perf_baseline.json")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import perf_points  # noqa: E402  (the E2 methodology; QuSpin is needed only inside its producer process)

TIME_FACTOR = float(os.environ.get("CERTIFY_ED_PERF_TIME_FACTOR", "2.0"))
TIME_SLACK_S = 2.0
MEMORY_FACTOR = float(os.environ.get("CERTIFY_ED_PERF_MEMORY_FACTOR", "1.5"))
MEMORY_SLACK_MB = 64.0


@pytest.fixture
def perf(request):
    if not (request.config.getoption("--perf", default=False) or os.environ.get("CERTIFY_ED_PERF") == "1"):
        pytest.skip("performance regression tests run with --perf (or CERTIFY_ED_PERF=1)")
    with open(BASELINE, encoding="utf-8") as f:
        return json.load(f)


@needs_quspin
@pytest.mark.parametrize("label", perf_points.SMALL)
def test_e2_point_within_baseline(perf, label, tmp_path):
    base = perf["points"][label]["certifier"]
    cur = perf_points.measure(label, repeat=1, workdir=str(tmp_path))
    c = cur["certifier"]
    assert "error" not in c, c
    statuses = {k: v["status"] for k, v in c["statements"].items()}
    assert statuses == {k: v["status"] for k, v in base["statements"].items()}
    limit_s = TIME_FACTOR * base["total"] + TIME_SLACK_S
    limit_mb = MEMORY_FACTOR * base["maxrss_mb"] + MEMORY_SLACK_MB
    msg = (f"{label}: {c['total']:.2f} s (limit {limit_s:.2f}), {c['maxrss_mb']:.0f} MB (limit {limit_mb:.0f}); "
           f"baseline machine {perf['machine'].get('cpu_model')} on {perf['machine'].get('date')}")
    print(msg)
    assert c["total"] <= limit_s, msg
    assert c["maxrss_mb"] <= limit_mb, msg


def test_baseline_file_is_well_formed():
    with open(BASELINE, encoding="utf-8") as f:
        d = json.load(f)
    assert d["schema"] == "certify-ed/perf-points/1.0"
    for lab in perf_points.SMALL:
        c = d["points"][lab]["certifier"]
        assert c["total"] > 0 and c["maxrss_mb"] > 0 and c["statements"]
