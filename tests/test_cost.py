"""A-05: the advisory cost model, budget_seconds='auto' and Options.progress."""
import json
import math

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.cost import (AUTO_FACTOR, AUTO_FLOOR_SECONDS, COEFFICIENTS, E2_RECORDS, auto_budget, estimate_seconds,
                             resolve_budget_seconds)


@pytest.fixture(scope="module")
def heis10():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def test_estimate_is_monotone_in_dimension():
    dims = [1, 2, 10, 100, 810, 924, 2704, 3432, 9252, 12870, 50000, 10 ** 6]
    for model in list(COEFFICIENTS) + ["auto"]:
        est = [estimate_seconds(d, model) for d in dims]
        assert all(math.isfinite(e) and e > 0 for e in est)
        assert all(b >= a for a, b in zip(est, est[1:])), (model, est)
    # every fitted exponent is positive and the fits pass through the measured range
    for name, (a, b) in COEFFICIENTS.items():
        assert b > 0
        for d, s in E2_RECORDS[name]:
            assert 0.2 * s <= math.exp(a) * d ** b <= 5 * s, (name, d, s)
    # the envelope dominates every measured point
    for pts in E2_RECORDS.values():
        for d, s in pts:
            assert estimate_seconds(d) >= 0.5 * s


def test_auto_budget_is_finite_and_floored():
    for d in (0, 1, 1024, 12870, 10 ** 5):
        b = auto_budget(d)
        assert math.isfinite(b) and b >= AUTO_FLOOR_SECONDS
        assert b >= AUTO_FACTOR * estimate_seconds(d) - 1e-9
    sec, rec = resolve_budget_seconds("auto", 12870)
    assert math.isfinite(sec) and rec["mode"] == "auto" and rec["dimension"] == 12870
    assert resolve_budget_seconds(None, 5) == (None, None)
    assert resolve_budget_seconds(12, 5) == (12.0, None)
    with pytest.raises(ValueError):
        resolve_budget_seconds("soon", 5)
    with pytest.raises(ValueError):
        resolve_budget_seconds(float("inf"), 5)


def test_auto_budget_gives_the_same_statuses_as_no_budget(heis10):
    spec, H, w, V = heis10
    plain = ce.quick(H, w[:4], V[:, :4], spec)
    auto = ce.quick(H, w[:4], V[:, :4], spec, budget_seconds="auto")
    assert [(s["id"], s["status"]) for s in auto.statements] == [(s["id"], s["status"]) for s in plain.statements]
    assert [s["summary"] for s in auto.statements] == [s["summary"] for s in plain.statements]
    assert np.array_equal(auto.arrays["radius"], plain.arrays["radius"])
    sc = auto.body["scope"]
    assert math.isfinite(sc["options"]["budget_seconds"]) and sc["budgets"]["seconds"] == sc["options"]["budget_seconds"]
    assert sc["options"]["budget_auto"]["dimension"] == 1024
    assert plain.body["scope"]["options"]["budget_auto"] is None
    # an eigenvalue-only claim, a preset and the exact statement behave the same way
    for kw in ({"V": None}, {"preset": "proof-lemma"}):
        v = kw.pop("V", V[:, :4])
        a = ce.quick(H, w[:4], v, spec, budget_seconds="auto", **kw)
        b = ce.quick(H, w[:4], v, spec, **kw)
        assert [(s["id"], s["status"]) for s in a.statements] == [(s["id"], s["status"]) for s in b.statements]


def test_auto_budget_is_still_enforced_when_spent(heis10, monkeypatch):
    """Negative control: 'auto' is a budget like any other. With the estimate forced to zero and the floor removed,
    statements after S1 are NOT_CHECKED with BUDGET_TIME, never CERTIFIED by default."""
    import certify_ed.cost as cost
    spec, H, w, V = heis10
    monkeypatch.setattr(cost, "AUTO_FLOOR_SECONDS", 0.0)
    monkeypatch.setattr(cost, "COEFFICIENTS", {"zero": (-700.0, 0.0)})
    c = ce.quick(H, w[:4], V[:, :4], spec, budget_seconds="auto")
    assert c.body["scope"]["options"]["budget_seconds"] < 1e-300
    assert all(s["status"] == "NOT_CHECKED" and s["evidence"]["reason_code"] == "BUDGET_TIME" for s in c.statements[1:])


def test_progress_prints_one_line_per_statement(heis10, capsys):
    spec, H, w, V = heis10
    c = ce.quick(H, w[:4], V[:, :4], spec, progress=True)
    err = capsys.readouterr().err
    lines = [l for l in err.splitlines() if l.startswith("[")]
    assert len(lines) == len(c.statements)
    for line, s in zip(lines, c.statements):
        assert s["id"] in line and s["status"] in line and line.rstrip().endswith(s["summary"]) and " s]" in line
    sink = []
    c2 = ce.quick(H, w[:4], V[:, :4], spec, progress=sink.append)
    assert len(sink) == len(c2.statements) and all(s["id"] in l for l, s in zip(sink, c2.statements))
    assert set(c.body["timing"]["per_statement"]) == {s["id"] for s in c.statements}
    assert [(s["id"], s["status"]) for s in c.statements] == \
        [(s["id"], s["status"]) for s in ce.quick(H, w[:4], V[:, :4], spec).statements]


def test_cli_budget_auto_and_progress(heis10, tmp_path, capsys):
    from certify_ed.cli import main
    spec, H, w, V = heis10
    npz = tmp_path / "claim.npz"
    ce.export_npz(str(npz), w[:4], V[:, :4], H)
    specf = tmp_path / "spec.json"
    specf.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    out = tmp_path / "cert"
    code = main(["certify", str(npz), "--spec", str(specf), "--kind", "lowest", "--budget", "auto", "--progress",
                 "-o", str(out)])
    captured = capsys.readouterr()
    assert code == 0, captured.out
    cert = ce.Certificate.load(str(out) + ".json")
    assert cert.body["scope"]["options"]["budget_auto"]["mode"] == "auto"
    assert math.isfinite(cert.body["scope"]["options"]["budget_seconds"])
    assert captured.err.count(" s] ") == len(cert.statements)
    assert main(["certify", str(npz), "--spec", str(specf), "--kind", "lowest", "--budget", "300", "-o", str(out)]) == 0
    assert ce.Certificate.load(str(out) + ".json").body["scope"]["options"]["budget_seconds"] == 300.0
