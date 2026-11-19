"""Parallel certification across sectors and claims with a memory budget per worker (roadmap D-04, 3.3.0.dev0).

Not a new statement and not a new bound: these functions decide only WHERE a certification runs. The rules make
every bound and status independent of the number of workers:

1. `workers=1` (the default everywhere) runs the tasks in this process, in input order: exactly the calls a loop
   would make, no pool is created.
2. With W > 1 workers the memory of the certification is divided: the per-worker budget is B = (C - M0) / W, where C
   is the memory cap of this process (`process_memory_cap_mb`: the job-object limit on Windows, RLIMIT_AS on POSIX;
   the available physical memory when no cap is set) and M0 the memory this process already holds. Each worker
   caps itself at B (job object / RLIMIT_AS), so the W workers together stay within the cap. This is the lesson of
   the 3.2.0 evidence run: eight concurrent sparse products under a committed-memory cap ended the process with an
   access violation instead of a MemoryError (RESULT_evidence_run_3.2.1.md, `cost.residual_threads`).
3. A task whose predicted peak (`cost.plan_for_claim`, or the caller's estimate) exceeds B is not sent to a worker;
   it runs after the pool, in this process, with the whole cap, as it would with one worker.
4. A task that fails in a worker (an exception, a worker process that dies, as a native allocation failure does),
   or whose result shows a resource limit (a statement NOT_CHECKED for memory or time, an inertia count with
   `resource_failure`), is run again in this process after the pool; the serial result is the one returned.

The workers are processes started with this interpreter; tasks and results travel as pickles through files (no
pipe or queue is shared, so a worker that dies loses only its current task, which rule 4 redoes here).
A worker computes with the same code and inputs as this process and inherits its environment variables (the BLAS
and OpenMP thread counts), so a task completed in a worker returns the numbers the serial call returns;
tests/test_parallel.py checks every status and every recorded number (timings excluded) for 1, 2 and 4 workers.
MPI (mpi4py) is not used: the tasks of one certification fit one machine, and a distributed run is a sequence of
independent `certify-ed certify` calls.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

WINDOWS = sys.platform == "win32"
SAFETY = 0.9          # fraction of the cap the workers may use together (the rest: interpreter, pickles, the pool)


# ---------------------------------------------------------------------------------------------------------------
# the memory cap of this process and its current use
# ---------------------------------------------------------------------------------------------------------------

def _win_extended_limits():
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    return ctypes, wintypes, EXTENDED


def process_memory_cap_mb() -> Optional[float]:
    """The memory limit of this process in MiB, or None when there is none.

    The cap is read by `cost.process_memory_cap`, the reader of the memory gate (3.2.2, N-03), so that the gate and the
    worker budgets see the same limit. Windows: the job object the process belongs to (innermost),
    JOB_OBJECT_LIMIT_PROCESS_MEMORY (committed private memory of one process) and JOB_OBJECT_LIMIT_JOB_MEMORY (of all
    processes of the job); the smaller when both are set. POSIX: the soft RLIMIT_AS (address space). The two caps
    measure different things (D-07); both make an allocation beyond them fail."""
    from .cost import process_memory_cap
    cap = process_memory_cap()
    if cap is None or not cap.get("cap_mb"):
        return None
    return float(cap["cap_mb"])


def process_memory_mb() -> Optional[float]:
    """Memory this process holds in the sense of its cap: committed private memory on Windows (PagefileUsage), the
    address-space size on Linux (/proc/self/statm), the peak resident size elsewhere; None when unknown."""
    try:
        if WINDOWS:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            c = PMC()
            c.cb = ctypes.sizeof(c)
            if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return c.PagefileUsage / 2 ** 20
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[0])
            return pages * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 1024
    except Exception:
        return None


def cap_process_memory(nbytes: int) -> str:
    """Cap the calling process at nbytes: a job object with JOB_OBJECT_LIMIT_PROCESS_MEMORY on Windows (nested in the
    parent's job when there is one), RLIMIT_AS on POSIX. Returns the mechanism or 'not enforced: <reason>'."""
    try:
        if not WINDOWS:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            lim = int(nbytes) if hard in (resource.RLIM_INFINITY, -1) else min(int(nbytes), hard)
            resource.setrlimit(resource.RLIMIT_AS, (lim, hard))
            return "RLIMIT_AS"
        ctypes, wintypes, EXTENDED = _win_extended_limits()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x00000100        # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = int(nbytes)
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
        _JOBS.append(job)
        return "job object"
    except Exception as e:                       # e.g. a parent job that forbids nesting
        return f"not enforced: {type(e).__name__}: {e}"


_JOBS: list = []          # keep job handles alive for the life of the worker


def worker_budget(workers: int, cap_mb: Optional[float] = None, used_mb: Optional[float] = None) -> dict:
    """The per-worker memory budget of rule 2: {'workers', 'cap_mb', 'cap_source', 'used_mb', 'budget_mb'}.

    cap_mb: overrides the detected cap (a test, or a user who knows the machine); used_mb: overrides M0."""
    workers = max(1, int(workers))
    source = "argument"
    if cap_mb is None:
        cap_mb = process_memory_cap_mb()
        source = "process cap"
        if cap_mb is None:
            from .cost import available_memory_mb
            cap_mb = available_memory_mb()
            source = "available physical memory"
            used_mb = 0.0 if used_mb is None else used_mb         # available memory already excludes this process
    if used_mb is None:
        used_mb = process_memory_mb() or 0.0
    budget = None if cap_mb is None else max(0.0, (float(cap_mb) - float(used_mb)) * SAFETY / workers)
    return {"workers": workers, "cap_mb": None if cap_mb is None else float(cap_mb), "cap_source": source,
            "used_mb": float(used_mb), "budget_mb": budget}


# ---------------------------------------------------------------------------------------------------------------
# the scheduler
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class TaskLog:
    index: int
    where: str                      # 'serial' (workers = 1), 'worker', 'deferred' (rule 3) or 'retried' (rule 4)
    predicted_mb: Optional[float]
    seconds: float
    reason: str = ""


@dataclass
class ParallelReport:
    budget: dict
    tasks: List[TaskLog] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"budget": dict(self.budget), "tasks": [t.__dict__.copy() for t in self.tasks]}


def _timed_call(fn, arg):
    t0 = time.perf_counter()
    return fn(arg), time.perf_counter() - t0


def _io_retry(op, attempts: int = 100, wait: float = 0.02):
    """op() with retries on PermissionError: on Windows another process (a scanner, the reader) can hold a file
    for a moment, and a rename or open then fails with a sharing violation."""
    for k in range(attempts):
        try:
            return op()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(wait)


def _atomic_pickle(obj, path: str):
    import pickle
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def write():
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    _io_retry(write)


def _read_pickle(path: str, remove: bool = True):
    import pickle

    def read():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if remove:
            os.remove(path)
        return obj
    return _io_retry(read)


def _worker_main(argv=None):
    """Entry point of one worker process: `python -c "from certify_ed.parallel import _worker_main; _worker_main()"
    WORKDIR ID BUDGET_BYTES`. Caps itself at BUDGET_BYTES (rule 2), then serves tasks through files: it waits for
    WORKDIR/w<ID>.task (a pickled (index, fn, arg)), removes it, runs fn(arg) and writes WORKDIR/w<ID>.out<index> =
    (ok, value or error text, seconds); WORKDIR/w<ID>.stop ends it."""
    wd, wid, budget = (argv if argv is not None else sys.argv[1:])[:3]
    if int(budget) > 0:
        cap_process_memory(int(budget))
    task, stop = os.path.join(wd, f"w{wid}.task"), os.path.join(wd, f"w{wid}.stop")
    while not os.path.exists(stop):
        if not os.path.exists(task):
            time.sleep(0.01)
            continue
        i, fn, arg = _read_pickle(task)
        try:
            value, sec = _timed_call(fn, arg)
            payload = (True, value, sec)
        except BaseException as e:                          # MemoryError and the like: reported, not raised
            payload = (False, f"{type(e).__name__}: {str(e)[:300]}", 0.0)
        try:
            _atomic_pickle(payload, os.path.join(wd, f"w{wid}.out{i}"))
        except Exception as e:                              # an unpicklable result: reported as a failure
            _atomic_pickle((False, f"result not picklable: {type(e).__name__}", 0.0), os.path.join(wd, f"w{wid}.out{i}"))


def _worker_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    paths = [p for p in sys.path if p and os.path.isdir(p)]
    old = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + old))
    return env


def run_tasks(fn: Callable[[Any], Any], args: Sequence[Any], workers: int = 1,
              predicted_mb: Optional[Sequence[Optional[float]]] = None, cap_mb: Optional[float] = None,
              needs_serial: Optional[Callable[[Any], str]] = None, cap_workers: bool = True,
              workdir: Optional[str] = None, poll_seconds: float = 0.01):
    """[fn(a) for a in args] under rules 1 to 4 of the module docstring -> (results, ParallelReport).

    The workers are separate processes started with this interpreter (`_worker_main`), at most `workers` of them,
    each capped at the per-worker budget; tasks and results travel as pickles through files in `workdir` (default:
    a new temporary directory), one task per worker at a time, in input order. No pipe or queue is shared, so a
    worker that dies loses only its current task (it is redone here, and a new worker takes its place). fn must be
    a module-level function; needs_serial(result) returns a non-empty reason when a worker's result shows a resource
    limit (rule 4). cap_workers=False leaves the workers uncapped (the budget still decides rule 3)."""
    import shutil
    import subprocess
    import tempfile
    args = list(args)
    n = len(args)
    pred = list(predicted_mb) if predicted_mb is not None else [None] * n
    results: List[Any] = [None] * n
    if workers <= 1 or n <= 1:
        rep = ParallelReport({"workers": 1, "budget_mb": None, "cap_mb": cap_mb, "cap_source": "one worker"})
        for i, a in enumerate(args):
            results[i], sec = _timed_call(fn, a)
            rep.tasks.append(TaskLog(i, "serial", pred[i], sec))
        return results, rep
    budget = worker_budget(workers, cap_mb)
    rep = ParallelReport(budget)
    B = budget["budget_mb"]
    logs: dict = {}
    pooled = [i for i in range(n) if B is None or pred[i] is None or pred[i] <= B]
    later = [i for i in range(n) if i not in set(pooled)]
    for i in later:
        logs[i] = TaskLog(i, "deferred", pred[i], 0.0, f"predicted {pred[i]:.0f} MB > worker budget {B:.0f} MB")
    retry: List[int] = []
    own_dir = workdir is None
    wd = tempfile.mkdtemp(prefix="certify_ed_workers_") if own_dir else workdir
    os.makedirs(wd, exist_ok=True)
    cap_bytes = int(B * 2 ** 20) if (B and cap_workers) else 0
    env = _worker_env()
    code = "from certify_ed.parallel import _worker_main; _worker_main()"
    queue = list(pooled)
    procs: dict = {}          # worker id -> Popen
    busy: dict = {}           # worker id -> task index
    next_id = 0

    def start_worker():
        nonlocal next_id
        wid = next_id
        next_id += 1
        procs[wid] = subprocess.Popen([sys.executable, "-c", code, wd, str(wid), str(cap_bytes)], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wid

    try:
        for _ in range(min(int(workers), len(queue))):
            start_worker()
        while queue or busy:
            for wid in list(procs):                       # hand out tasks to idle workers
                if wid in busy or not queue:
                    continue
                i = queue.pop(0)
                try:
                    _atomic_pickle((i, fn, args[i]), os.path.join(wd, f"w{wid}.task"))
                except Exception as e:                     # cannot pickle: do it here
                    logs[i] = TaskLog(i, "retried", pred[i], 0.0, f"pickle: {type(e).__name__}: {str(e)[:160]}")
                    retry.append(i)
                    continue
                busy[wid] = i
            for wid in list(busy):
                i = busy[wid]
                out = os.path.join(wd, f"w{wid}.out{i}")
                if os.path.exists(out):
                    del busy[wid]
                    try:
                        ok, value, sec = _read_pickle(out)
                    except Exception as e:
                        ok, value, sec = False, f"result unreadable: {type(e).__name__}", 0.0
                    why = value if not ok else (needs_serial(value) if needs_serial is not None else "")
                    if why:
                        logs[i] = TaskLog(i, "retried", pred[i], sec, str(why))
                        retry.append(i)
                    else:
                        results[i] = value
                        logs[i] = TaskLog(i, "worker", pred[i], sec)
                elif procs[wid].poll() is not None:        # the worker died with its task
                    del busy[wid]
                    logs[i] = TaskLog(i, "retried", pred[i], 0.0, f"worker exit status {procs[wid].returncode}")
                    retry.append(i)
                    del procs[wid]
                    if queue:
                        start_worker()
            for wid in list(procs):                        # a worker that died while idle
                if wid not in busy and procs[wid].poll() is not None:
                    del procs[wid]
                    if queue:
                        start_worker()
            if busy:
                time.sleep(poll_seconds)
    finally:
        for wid in list(procs):
            try:
                open(os.path.join(wd, f"w{wid}.stop"), "w").close()
            except OSError:
                pass
        for p in procs.values():
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
        if own_dir:
            shutil.rmtree(wd, ignore_errors=True)
    for i in sorted(later + retry):                          # rules 3 and 4: in this process, after the workers
        results[i], sec = _timed_call(fn, args[i])
        logs[i].seconds = sec
    rep.tasks = [logs[i] for i in range(n)]
    return results, rep


# ---------------------------------------------------------------------------------------------------------------
# certificates: many claims (one per sector, or any independent claim/spec pairs)
# ---------------------------------------------------------------------------------------------------------------

_RESOURCE_TEXT = ("memory", "could not be completed", "budget", "timed out", "time limit")


def certificate_resource_reason(cert) -> str:
    """Non-empty when a statement of `cert` is NOT_CHECKED for a reason that depends on the resources of the process
    that ran it (memory or time budget, an allocation failure)."""
    from .scope import classify_not_checked
    for s in getattr(cert, "statements", []):
        if s.get("status") != "NOT_CHECKED":
            continue
        code = classify_not_checked(s)
        text = str(s.get("summary", "")).lower()
        if code in ("BUDGET_MEMORY", "BUDGET_TIME") or any(t in text for t in _RESOURCE_TEXT):
            return f"{s.get('id')}: {code}: {text[:120]}"
    return ""


def _certify_job(job):
    from .certify import certify
    claim, spec, options = job
    return certify(claim, spec, options)


def predicted_peak_mb(claim, spec, options) -> Optional[float]:
    """Predicted peak memory of one certification (cost.plan_for_claim), None when the planner cannot say."""
    try:
        from .certify import Options
        from .cost import plan_for_claim
        return float(plan_for_claim(claim, spec, options if options is not None else Options()).peak_mb)
    except Exception:
        return None


def certify_many(jobs: Sequence[tuple], workers: int = 1, cap_mb: Optional[float] = None, report: bool = False):
    """certify(claim, spec, options) for every (claim, spec, options) in `jobs`, in input order, on `workers`
    processes under rules 1 to 4. Returns the list of certificates, and the ParallelReport when report=True."""
    jobs = [tuple(j) + (None,) * (3 - len(j)) for j in jobs]
    pred = [predicted_peak_mb(*j) for j in jobs] if workers > 1 else None
    certs, rep = run_tasks(_certify_job, jobs, workers=workers, predicted_mb=pred, cap_mb=cap_mb,
                           needs_serial=certificate_resource_reason)
    return (certs, rep) if report else certs


__all__ = ["process_memory_cap_mb", "process_memory_mb", "cap_process_memory", "worker_budget", "run_tasks",
           "certify_many", "certificate_resource_reason", "predicted_peak_mb", "TaskLog", "ParallelReport", "SAFETY"]
