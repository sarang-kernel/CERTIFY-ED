"""D-04 (certify_ed/parallel.py, compose.certify_low_energy(workers=...)): every bound and status is the same for 1, 2
and 4 workers, and the scheduler's rules (budget per worker from the process cap, deferral, serial retry after a
worker failure) hold. The 3.2.0 evidence run lost E4 W4 to concurrent work under a memory cap (cost.residual_threads);
these tests pin that parallel work never changes a certified number."""
import os

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed import parallel as par
from certify_ed.compose import SectorResult, certify_low_energy
from certify_ed.construct import full_state_set
from conftest import eigh_dense

TIMING_KEYS = ("second", "time", "elapsed", "parallel")


def strip(x):
    """Drop timing fields (and the scheduling record) recursively; everything else is compared exactly."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not any(t in str(k).lower() for t in TIMING_KEYS)}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    if isinstance(x, np.ndarray):
        return ("ndarray", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, float):
        return ("float", x.hex()) if x == x else ("float", "nan")
    return x


def sz_sectors(spec, E_cut, merge=None):
    H = ce.build_bitwise(spec)
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites + 1):
        st = ce.u1_state_set(spec, nup)
        idx, found = full.index_of(st.states_in_order())
        assert np.all(found)
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        secs.append((nup, Vs))
    if merge is not None:                    # one "sector" spanning two U(1) sectors: the full-matrix path
        a, b = merge
        Va, Vb = dict(secs)[a], dict(secs)[b]
        secs = [s for s in secs if s[0] not in (a, b)] + [(f"{a}+{b}", sp.hstack([Va, Vb]).tocsr())]
    out = []
    for label, Vs in secs:
        Hs = (Vs.T @ H @ Vs).tocsr()
        ws, Ys = eigh_dense(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        out.append(SectorResult(f"nup={label}", Vs, Hs, ws[sel], Ys[:, sel]))
    return out


@pytest.mark.parametrize("merge", [(3, 4)])
def test_low_energy_identical_for_1_2_4_workers(merge):
    # sectors 0..2 take their U(1) restriction, the merged sector and every later one the full matrix: both paths
    spec = ce.heisenberg_chain(8)
    wf = la.eigvalsh(ce.build_bitwise(spec).toarray())
    E_cut = float((wf[10] + wf[11]) / 2)        # between levels, above the six-fold level wf[5] to wf[10]
    secs = sz_sectors(spec, E_cut, merge)
    ref = certify_low_energy(spec, secs, E_cut)
    assert ref["status"] == "CERTIFIED"
    for w in (2, 4):
        out = certify_low_energy(spec, secs, E_cut, workers=w, cap_mb=8192)
        assert strip(out) == strip(ref), w
        where = [t["where"] for t in out["parallel"]["tasks"]]
        assert where.count("worker") >= len(secs) // 2, out["parallel"]      # a retried task is redone here


def u1_claims(N=10, nups=(4, 5, 6), k=3):
    spec = ce.heisenberg_chain(N)
    jobs = []
    for nup in nups:
        st = ce.u1_state_set(spec, nup)
        Hs = ce.build_bitwise(spec, st)
        w, V = eigh_dense(Hs.toarray())
        claim = ce.Claim(w[:k], V[:, :k], operator=Hs, kind="lowest", states=st)
        jobs.append((claim, spec, ce.Options()))
    return jobs


def test_certify_many_identical_for_1_2_4_workers():
    jobs = u1_claims()
    ref = par.certify_many(jobs, workers=1)
    assert [s["status"] for s in ref[0].statements][:3] == ["CERTIFIED"] * 3
    for w in (2, 4):
        certs, rep = par.certify_many(jobs, workers=w, cap_mb=8192, report=True)
        for a, b in zip(ref, certs):
            assert strip(a.statements) == strip(b.statements)
            assert strip(dict(a.arrays)) == strip(dict(b.arrays))
        assert [t.where for t in rep.tasks].count("worker") >= 1, rep.to_dict()


# ------------------------------------------------------------------ the scheduler's rules

def test_one_worker_runs_in_process_in_order():
    res, rep = par.run_tasks(_mixed, [("pid", 0)] * 3, workers=1)
    assert res == [os.getpid()] * 3 and [t.where for t in rep.tasks] == ["serial"] * 3


def test_budget_is_the_cap_divided_among_workers():
    b = par.worker_budget(4, cap_mb=4096.0, used_mb=96.0)
    assert b["budget_mb"] == pytest.approx((4096.0 - 96.0) * par.SAFETY / 4)
    assert b["cap_source"] == "argument"
    assert par.worker_budget(1, cap_mb=100.0, used_mb=200.0)["budget_mb"] == 0.0


def _mixed(arg):
    """One function for the rules: arg = (kind, parent_pid)."""
    kind, parent = arg
    if kind == "pid":
        return os.getpid()
    if kind == "cap":
        return par.process_memory_cap_mb()
    if kind == "fail" and os.getpid() != parent:
        raise MemoryError("simulated allocation failure in a worker")
    if kind == "die" and os.getpid() != parent:
        os._exit(3)                          # a worker that dies (as after a native access violation)
    if kind == "flag":
        return "worker" if os.getpid() != parent else "parent"
    return "serial result"


def test_scheduler_rules_deferral_retry_death_cap():
    me = os.getpid()
    args = [("pid", me), ("pid", me), ("fail", me), ("die", me), ("flag", me), ("cap", me), ("pid", me)]
    res, rep = par.run_tasks(_mixed, args, workers=2, predicted_mb=[1.0, 1e9, 1.0, 1.0, 1.0, 1.0, 1.0], cap_mb=6000.0,
                             needs_serial=lambda r: "resource" if r == "worker" else "")
    where = [t.where for t in rep.tasks]
    assert where == ["worker", "deferred", "retried", "retried", "retried", "worker", "worker"], where
    assert res[0] != me and res[1] == me and res[6] != me                 # rule 3: the large task ran here
    assert res[2] == res[3] == "serial result" and res[4] == "parent"     # rule 4: redone here
    B = rep.budget["budget_mb"]
    assert res[5] is not None and res[5] == pytest.approx(B, rel=1e-6, abs=1.0), (res[5], B)   # rule 2


def test_resource_reasons_of_a_certificate():
    class C:
        statements = [{"id": "S3.completeness", "status": "NOT_CHECKED",
                       "summary": "inertia count not attempted to completion: factorization could not be completed "
                                  "(MemoryError: ...)"}]
    assert par.certificate_resource_reason(C())
    C.statements = [{"id": "S3.completeness", "status": "NOT_CHECKED", "summary": "no completeness claim",
                     "evidence": {"reason_code": "NO_COMPLETENESS_CLAIM"}}]
    assert par.certificate_resource_reason(C()) == ""
