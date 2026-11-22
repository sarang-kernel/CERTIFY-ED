"""Planner (roadmap D-08): predicted memory against measured peaks, the dense/sparse choice of
inertia_method='auto', and the rule that the plan never changes a status."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.cost import (DENSE_LDL_ARRAYS, Plan, dense_inertia_mb, plan_certification, plan_for_claim,
                             recommend_inertia_method, sparse_fill_estimate, sparse_inertia_mb)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# peak resident memory of whole runs recorded on the development machine (experiments: bench_whole.py of the
# C-17 record, whole-spectrum preset, two temperatures; peak_rss_mb minus the process baseline before certify):
# (n, nnz, k, dense operator, measured peak MB above baseline)
RECORDED_PEAKS = [
    (3432, 29304, 3432, True, 699 - 183),
    (12870, 117794, 12870, False, 2974 - 200),
    (12870, 117794, 12870, True, 8715 - 200),
]

CHILD = r'''
import sys, json, time, numpy as np
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/experiments")
import portable
import certify_ed as ce
from certify_ed.inertia import inertia_dense, inertia_sparse
N, mode = int(sys.argv[1]), sys.argv[2]
spec = ce.heisenberg_chain(N); st = ce.u1_state_set(spec, N // 2); H = ce.build_bitwise(spec, st)
sigma = float(-0.4 * N)
base = portable.peak_rss_mb()
r = inertia_dense(H.toarray(), sigma) if mode == "dense" else inertia_sparse(H, sigma)
print("RESULT" + json.dumps({"n": int(H.shape[0]), "nnz": int(H.nnz), "peak_mb": portable.peak_rss_mb() - base,
                             "conclusive": r.conclusive, "fill": r.fill_nnz}))
'''.replace("ROOT", repr(ROOT.replace("\\", "/")))


def _measure(N, mode):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    p = subprocess.run([sys.executable, "-c", CHILD, str(N), mode], capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    assert line, (p.stdout + p.stderr)[-800:]
    return json.loads(line[0][6:])


PARENT = r'''
import subprocess, sys
import numpy as np
big = np.ones(300 * 2 ** 20 // 8)        # 300 MiB resident in the parent
child = "import sys; sys.path.insert(0, ROOT + '/experiments'); import portable; print(portable.peak_rss_mb())"
print(subprocess.run([sys.executable, "-c", child], capture_output=True, text=True).stdout.strip(), float(big[-1]))
'''.replace("ROOT", repr(ROOT.replace("\\", "/")))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the inherited ru_maxrss is a Linux behaviour")
def test_peak_of_a_child_does_not_include_the_parents_peak():
    """After fork (or vfork) and exec, ru_maxrss of the child starts at the parent's peak resident size; the planner
    measurements read VmHWM instead, so a child of a large process (as inside a long pytest run) measures its own."""
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    p = subprocess.run([sys.executable, "-c", PARENT], capture_output=True, text=True, env=env)
    peak = float(p.stdout.split()[0])
    assert peak < 200, (peak, p.stderr[-500:])


@pytest.mark.parametrize("N,mode", [(14, "dense"), (14, "sparse"), (16, "sparse")])
def test_predicted_inertia_memory_within_factor_two_of_measured_peak(N, mode):
    """Dense Bunch-Kaufman at n = 3432 and SuperLU at n = 3432 and 12870, measured in a fresh process (the
    dense factorization at 12870 would need about 12 GB and is exactly what the planner steers away from)."""
    m = _measure(N, mode)
    assert m["conclusive"]
    pred = dense_inertia_mb(m["n"]) if mode == "dense" else sparse_inertia_mb(m["n"], m["nnz"])
    assert 0.5 * m["peak_mb"] <= pred <= 2 * m["peak_mb"], (mode, m, pred)
    if mode == "sparse":
        assert 0.5 * m["fill"] <= sparse_fill_estimate(m["n"], m["nnz"]) <= 2 * m["fill"]


def test_predicted_s2_peak_within_factor_two_of_recorded_runs():
    for n, nnz, k, dense, measured in RECORDED_PEAKS:
        plan = plan_certification(n, nnz, k, "all", dense, statements=("S1", "S2", "S3", "S7"), temperatures=2)
        assert 0.5 * measured <= plan.peak_mb <= 2 * measured, (n, dense, plan.peak_mb, measured)


def test_recommendation_keeps_the_3_1_rule_and_adds_the_memory_guard():
    assert recommend_inertia_method(100, 500, dense_operator=True)[0] == "dense"
    assert recommend_inertia_method(2500, 10 ** 4, dense_operator=False, budget_mb=10 ** 6)[0] == "dense"
    assert recommend_inertia_method(2501, 10 ** 4, dense_operator=False, budget_mb=10 ** 6)[0] == "sparse"
    m, why = recommend_inertia_method(2000, 10 ** 4, dense_operator=False, budget_mb=100)
    assert m == "sparse" and "budget" in why and f"{dense_inertia_mb(2000):.0f}" in why
    assert DENSE_LDL_ARRAYS * 2000 ** 2 * 8 / 2 ** 20 == dense_inertia_mb(2000)
    assert recommend_inertia_method(2000, 10 ** 4, dense_operator=True, budget_mb=100)[0] == "dense"   # no fallback


@pytest.fixture(scope="module")
def heis10():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _sig(c):
    return [(s["id"], s["status"], s["summary"]) for s in c.statements]


def test_auto_matches_the_explicit_method_and_the_guard_changes_no_status(heis10):
    spec, H, w, V = heis10
    k = 5
    auto = ce.quick(H, w[:k], V[:, :k], spec)                                       # n = 1024: the 3.1 rule picks dense
    dense = ce.quick(H, w[:k], V[:, :k], spec, inertia_method="dense")
    assert _sig(auto) == _sig(dense)
    assert all(c["method"] == "dense-bunch-kaufman" for s in auto.statements if s["id"] == "S3.completeness"
               for c in s["evidence"]["counts"])
    # with a memory budget below the dense workspace (76 MB at n = 1024) but above the sparse one the guard sends the
    # counts to SuperLU: same statuses (in 3.1 this budget made S3 NOT_CHECKED, since only the dense path was estimated)
    guarded = ce.quick(H, w[:k], V[:, :k], spec, budget_mb=50.0)
    sparse = ce.quick(H, w[:k], V[:, :k], spec, inertia_method="sparse")
    assert [(s["id"], s["status"]) for s in guarded.statements] == [(s["id"], s["status"]) for s in sparse.statements]
    assert all(c["method"] == "sparse-superlu-residual" for s in guarded.statements if s["id"] == "S3.completeness"
               for c in s["evidence"]["counts"])
    assert [(s["id"], s["status"]) for s in guarded.statements] == [(s["id"], s["status"]) for s in auto.statements]
    # a window claim and the degeneracy counts go through the same switch
    a, b = 0.5 * (w[3] + w[4]), 0.5 * (w[8] + w[9])
    ga = ce.quick(H, w[4:9], V[:, 4:9], spec, kind="window", window=(a, b), budget_mb=50.0)
    gs = ce.quick(H, w[4:9], V[:, 4:9], spec, kind="window", window=(a, b), inertia_method="sparse")
    assert [(s["id"], s["status"]) for s in ga.statements] == [(s["id"], s["status"]) for s in gs.statements]


def test_plan_option_records_the_plan_and_changes_nothing(heis10, capsys):
    spec, H, w, V = heis10
    plain = ce.quick(H, w[:4], V[:, :4], spec)
    planned = ce.quick(H, w[:4], V[:, :4], spec, plan=True)
    assert _sig(plain) == _sig(planned)
    assert np.array_equal(plain.arrays["radius"], planned.arrays["radius"])
    assert plain.body["plan"] is None
    p = planned.body["plan"]
    assert p["schema"] == "certify-ed/plan/1.0" and p["n"] == 1024 and p["k"] == 4 and p["inertia_method"] == "dense"
    from certify_ed.certificate_v2 import STATEMENT_IDS
    assert [r["statement"] for r in p["rows"]] == list(STATEMENT_IDS)
    assert "plan:" in capsys.readouterr().err
    from certify_ed.schema import validate_certificate_body
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = planned.save(os.path.join(d, "planned"))
        validate_certificate_body(ce.Certificate.load(path).body)


def test_plan_for_claim_reads_claim_shapes(heis10):
    spec, H, w, V = heis10
    claim = ce.Claim(w, V, operator=H, kind="all")
    plan = plan_for_claim(claim, spec, ce.Options.from_preset("whole-spectrum", temperatures=[1.0]))
    assert isinstance(plan, Plan) and plan.n == 1024 and plan.k == 1024 and plan.nnz == H.nnz and not plan.dense_operator
    runs = {r.statement: r.runs for r in plan.rows}
    assert runs["S7.thermodynamics"] and runs["S2.eigenpairs"] and not runs["S4.eigenspaces"]
    assert plan.total_seconds > 0 and plan.peak_mb > 0
    text = plan.format()
    assert "S2.eigenpairs" in text and "advisory" in text
    dense_plan = plan_for_claim(ce.Claim(w, V, operator=H.toarray(), kind="all"), spec, ce.Options())
    assert dense_plan.dense_operator and dense_plan.peak_mb > plan.peak_mb


def test_cli_plan_command(heis10, tmp_path, capsys):
    from certify_ed.cli import main
    spec, H, w, V = heis10
    npz = tmp_path / "claim.npz"
    ce.export_npz(str(npz), w[:4], V[:, :4], H)
    specf = tmp_path / "spec.json"
    specf.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert main(["plan", str(npz), "--spec", str(specf), "--kind", "lowest", "-o", str(tmp_path / "plan.json")]) == 0
    out = capsys.readouterr().out
    assert "S3.completeness" in out and "dense" in out
    rec = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert rec["kind"] == "lowest" and rec["n"] == 1024
    assert main(["certify", str(npz), "--spec", str(specf), "--kind", "lowest", "--plan", "-o", str(tmp_path / "c")]) == 0
    assert ce.Certificate.load(str(tmp_path / "c.json")).body["plan"]["n"] == 1024
