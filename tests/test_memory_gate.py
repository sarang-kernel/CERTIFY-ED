"""3.2.2 (roadmap N-03): without a memory budget, the process memory cap decides whether a factorization is attempted.

The evidence runs cap each experiment's memory (experiments/portable.py: a Windows job object with a per-process
memory limit, RLIMIT_AS on POSIX). In the 3.2.0 run an allocation beyond such a cap ended the process (E4 W4 under
5 GiB). The test starts a child process, caps it a few hundred MB above what it already uses, and certifies a
diagonal operator of dimension 10000 (20000 on POSIX), whose sparse factorization the fill model of certify_ed.cost
predicts at more than the headroom although the real factorization is tiny. Expected: S3 NOT_CHECKED with
BUDGET_MEMORY before the factorization starts, while S2 is certified; with an explicit budget the cap is not
read and S3 is certified; without a cap nothing is gated (checked when the test process itself has no cap).
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = sys.platform == "win32"

CHILD = r'''
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[2], "experiments"))
import numpy as np
import scipy.sparse as sp
from certify_ed import Claim, Options, certify
from certify_ed import cost
from certify_ed.inertia import inertia
from certify_ed.scope import Budget

mode, n, headroom_mb = sys.argv[1], int(sys.argv[3]), float(sys.argv[4])
A = sp.diags(np.arange(1.0, n + 1.0)).tocsr()
X = np.zeros((n, 2))
X[0, 0] = X[1, 1] = 1.0
claim = Claim(eigenvalues=[1.0, 2.0], eigenvectors=X, operator=A, kind="lowest")
certify(Claim(eigenvalues=[1.0], eigenvectors=np.eye(4)[:, :1], operator=sp.diags([1.0, 2.0, 3.0, 4.0]).tocsr(),
              kind="lowest"), None, Options())                      # warm-up: every lazy import happens before the cap
out = {"estimate_mb": cost.sparse_inertia_mb(n, n, False)}
if mode == "capped":
    from portable import cap_memory
    if sys.platform == "win32":
        used = cost._windows_private_bytes()
    else:
        with open("/proc/self/statm") as f:
            used = float(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    out["mechanism_set"] = cap_memory(int(used + headroom_mb * 2 ** 20))
out["cap"] = cost.process_memory_cap()
r = inertia(A, 1.5, method="sparse")
out["inertia"] = {"conclusive": r.conclusive, "resource_failure": r.resource_failure, "negative": r.negative,
                  "note": r.note}
out["budget_none"] = list(Budget().memory_ok("S3.completeness", out["estimate_mb"], "test"))
out["budget_given"] = list(Budget(mb=1e7).memory_ok("S3.completeness", out["estimate_mb"], "test"))
for key, opts in (("statuses", Options()), ("statuses_budget", Options(budget_mb=1e7))):
    cert = certify(claim, None, opts)
    out[key] = {s["id"]: [s["status"], s["evidence"].get("reason_code"), s["summary"]] for s in cert.statements}
    if key == "statuses":
        out["scope_events"] = cert.body["scope"]["budgets"]["events"] if "budgets" in cert.body["scope"] else None
print("RESULT " + json.dumps(out))
'''


def _run_child(tmp_path, mode, n, headroom_mb):
    script = tmp_path / "gate_child.py"
    script.write_text(CHILD, encoding="utf-8")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("PYTHONSAFEPATH", None)
    p = subprocess.run([sys.executable, str(script), mode, ROOT, str(n), str(headroom_mb)], capture_output=True,
                       text=True, env=env, timeout=1800, cwd=str(tmp_path))
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    return json.loads(line[len("RESULT "):])


N, HEADROOM = (10000, 250.0) if WINDOWS else (20000, 1024.0)


def test_capped_child_does_not_start_a_factorization_that_cannot_fit(tmp_path):
    out = _run_child(tmp_path, "capped", N, HEADROOM)
    if str(out.get("mechanism_set", "")).startswith("not enforced"):
        pytest.skip(f"memory cap not enforceable here: {out['mechanism_set']}")
    cap = out["cap"]
    assert cap is not None and cap["mechanism"].startswith("job object" if WINDOWS else "RLIMIT_AS")
    assert cap["headroom_mb"] < out["estimate_mb"]             # the planted situation: prediction above headroom
    # the factorization gate (inertia) and the statement gate (scope.Budget) both refuse before starting
    inr = out["inertia"]
    assert inr["resource_failure"] and not inr["conclusive"] and inr["negative"] is None
    assert "process memory cap" in inr["note"] and "not attempted" in inr["note"]
    assert out["budget_none"][0] is False and "process memory cap" in out["budget_none"][1]
    assert out["budget_given"] == [True, ""]                   # a given budget is used; the cap is not read
    st = out["statuses"]
    assert st["S2.eigenpairs"][0] == "CERTIFIED"
    assert st["S3.completeness"][:2] == ["NOT_CHECKED", "BUDGET_MEMORY"]
    assert "process memory cap" in st["S3.completeness"][2]
    # with an explicit budget above the estimate the factorization runs (it is tiny) and S3 is certified
    assert out["statuses_budget"]["S3.completeness"][:2] == ["CERTIFIED", None]


def test_without_a_cap_nothing_is_gated(tmp_path):
    out = _run_child(tmp_path, "uncapped", N, HEADROOM)
    if out["cap"] is not None:
        pytest.skip(f"the test process itself runs under a memory cap: {out['cap']}")
    assert out["inertia"]["conclusive"] and out["inertia"]["negative"] == 1
    assert out["budget_none"] == [True, ""]
    assert out["statuses"]["S3.completeness"][:2] == ["CERTIFIED", None]


def test_memory_limit_prefers_the_budget():
    from certify_ed.cost import memory_limit_mb
    assert memory_limit_mb(123.0) == (123.0, None)
    lim, cap = memory_limit_mb(None)
    assert (lim is None) == (cap is None)
    if cap is not None:
        assert lim == cap["headroom_mb"] and cap["cap_mb"] >= cap["in_use_mb"] >= 0


def test_cap_does_not_gate_s1_but_still_gates_factorizations(monkeypatch):
    """3.2.3: the S1 estimate (dim x terms x 2 entries of 32 bytes) is a coarse upper bound, and the 3.2.2 evidence run
    refused S1 under the experiments' cap for blocks that 3.2.1 certified (E2 chain N=22 k=0, E4 W4). Without a
    budget the process memory cap no longer gates S1; it still gates the factorizations, and a given budget still
    gates S1."""
    import scipy.linalg as la

    import certify_ed as ce
    from certify_ed import cost
    from certify_ed.scope import Budget

    fake = {"mechanism": "job object (per-process memory limit)", "cap_mb": 1.0, "in_use_mb": 0.999,
            "headroom_mb": 0.001}
    monkeypatch.setattr(cost, "process_memory_cap", lambda: dict(fake))
    assert Budget().memory_ok("S1.model_fidelity", 10.0, "reference", use_cap=False) == (True, "")
    ok, note = Budget().memory_ok("S3.completeness", 10.0, "factorization")
    assert ok is False and "process memory cap" in note
    ok, note = Budget(mb=1.0).memory_ok("S1.model_fidelity", 10.0, "reference", use_cap=False)
    assert ok is False and "memory budget" in note
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    cert = ce.certify(ce.Claim(eigenvalues=w[:1], eigenvectors=V[:, :1], operator=H, kind="lowest"), spec,
                      ce.Options())
    st = {s["id"]: (s["status"], s["evidence"].get("reason_code")) for s in cert.statements}
    assert st["S1.model_fidelity"][0] == "CERTIFIED", st
    assert st["S3.completeness"] == ("NOT_CHECKED", "BUDGET_MEMORY"), st
