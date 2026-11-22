"""Advisory cost model: wall seconds of S1+S2+S3 against the dimension of the claim.

The model is a table of coefficients fitted to the E2 records of the 3.1.0 evidence run
(results/run_full_20261022, Windows, 8 cores, one BLAS thread per experiment; the records themselves are
not part of the repository because results/ is git-ignored). Measured certify seconds for S1+S2+S3:

    Heisenberg chain, U(1) block (Sz = 0)      dim   924: 0.7 s     3432: 2.9 s     12870: 65.5 s
    momentum k = 0 blocks of the chain          dim   810: 0.9 s     2704: 2.7 s      9252: 17.8 s
    square lattice 4x4, Sz = 0                  dim 12870: 175.4 s

Each row is fitted as log(seconds) = a + b log(dim) by least squares (two parameters, three points for the
chain and the block rows; the square row is the chain row shifted to pass through its single point). The
estimate for an unknown model is the maximum over the rows, so it is an upper envelope of what was
measured, not a prediction for a particular machine.

The estimate is advisory. It never changes a status: `budget_seconds="auto"` only turns it into the
default time budget (`auto_budget`), which is then enforced like any other budget (statements not started
within it are NOT_CHECKED with reason BUDGET_TIME, scope.py). A user who wants a statement to run
regardless passes a number or None.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Dict, Optional, Tuple  # noqa: F401

# (dimension, seconds) of the 3.1.0 evidence run (E2), per model family
E2_RECORDS: Dict[str, Tuple[Tuple[int, float], ...]] = {
    "chain_u1": ((924, 0.7), (3432, 2.9), (12870, 65.5)),
    "k0_blocks": ((810, 0.9), (2704, 2.7), (9252, 17.8)),
    "square_u1": ((12870, 175.4),),
}

# the auto budget is this factor times the envelope estimate, never below the floor
AUTO_FACTOR = 5.0
AUTO_FLOOR_SECONDS = 60.0


def _fit_loglog(points) -> Tuple[float, float]:
    """Least-squares a, b in log(s) = a + b log(d)."""
    xs = [math.log(d) for d, _ in points]
    ys = [math.log(s) for _, s in points]
    n = len(points)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def _coefficients() -> Dict[str, Tuple[float, float]]:
    a_c, b_c = _fit_loglog(E2_RECORDS["chain_u1"])
    a_k, b_k = _fit_loglog(E2_RECORDS["k0_blocks"])
    d_sq, s_sq = E2_RECORDS["square_u1"][0]
    a_s = math.log(s_sq) - b_c * math.log(d_sq)          # chain exponent through the square point
    return {"chain_u1": (a_c, b_c), "k0_blocks": (a_k, b_k), "square_u1": (a_s, b_c)}


COEFFICIENTS: Dict[str, Tuple[float, float]] = _coefficients()
"""model family -> (a, b) with  seconds = exp(a) * dim ** b  (fitted at import time from E2_RECORDS)."""


def estimate_seconds(dimension: int, model: str = "auto") -> float:
    """Estimated wall seconds of S1+S2+S3 for a claim of this dimension.

    `model` is a key of COEFFICIENTS or "auto" (the maximum over all rows). Monotone increasing in the
    dimension (every fitted exponent is positive). Dimensions below 1 are treated as 1."""
    d = max(1.0, float(dimension))
    rows = COEFFICIENTS.values() if model == "auto" else (COEFFICIENTS[model],)
    return max(math.exp(a) * d ** b for a, b in rows)


def auto_budget(dimension: int, model: str = "auto", factor: Optional[float] = None,
                floor: Optional[float] = None) -> float:
    """Default time budget: max(floor, factor * estimate_seconds(dimension, model)); always finite.
    `factor` and `floor` default to AUTO_FACTOR and AUTO_FLOOR_SECONDS (read at call time)."""
    est = estimate_seconds(dimension, model)
    f = AUTO_FACTOR if factor is None else factor
    fl = AUTO_FLOOR_SECONDS if floor is None else floor
    return float(max(fl, f * est))


def resolve_budget_seconds(value, dimension: Optional[int]) -> Tuple[Optional[float], Optional[dict]]:
    """Turn Options.budget_seconds (None, a number, or "auto") into (seconds or None, record or None).

    The record says how an "auto" budget was derived, for the certificate's scope block."""
    if value is None:
        return None, None
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(f"budget_seconds must be a number, None or 'auto' (got {value!r})")
        dim = int(dimension or 0)
        est = estimate_seconds(dim)
        sec = auto_budget(dim)
        return sec, {"mode": "auto", "dimension": dim, "estimate_seconds": est, "factor": AUTO_FACTOR,
                     "floor_seconds": AUTO_FLOOR_SECONDS, "budget_seconds": sec,
                     "model": "upper envelope of the 3.1.0 E2 rows (certify_ed.cost)"}
    sec = float(value)
    if not math.isfinite(sec) or sec < 0:
        raise ValueError(f"budget_seconds must be finite and non-negative (got {value!r})")
    return sec, None


# ---------------------------------------------------------------------------------------------------------------
# Planner (roadmap D-08, 3.2.0.dev0): predicted wall seconds and peak memory per statement from n, nnz and k.
#
# Memory is counted array by array (what the code allocates, in bytes per entry of the claim's dtype); the
# constants below that are not arithmetic facts were measured by experiments/planner_calibration.py on the
# 3.2.0.dev0 development machine (Windows, 8 cores, OpenBLAS 0.3.31, 2026-10-29) on Heisenberg chain Sz = 0
# blocks: dense LDL^T 9.28 / 8.75 n^2 entries at n = 924 / 3432; SuperLU fill 97156 / 1669784 / 19164106 at
# n = 924 / 3432 / 12870 (0.117 n^2.006), working storage 4.2-4.4 x fill x 12 bytes; gemm 22-41 GFlop/s,
# sparse H X 0.07-0.2 x 10^9 entries/s, one exact inner product 110 ns per term at n = 3432. Time predictions
# are rough (a factor 2-3 on those blocks; another machine needs its own RATES); the memory predictions are
# checked against measured peaks in tests/test_planner.py. Everything here is advisory:
# the plan is printed and recorded, and it chooses between dense and sparse inertia only through
# `recommend_inertia_method`, which never changes a status (a factorization that does not fit is NOT_CHECKED
# either way; the choice only decides which path is tried).
# ---------------------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import List, Sequence

# machine rates measured by experiments/planner_calibration.py (2026-10-29, see above); replace to recalibrate
RATES = {
    "gemm_gflops": 40.0,            # dense H @ X and |H| |X| (one call each; 2 n^2 k flops per call): 41 at n = 3432
    "spmm_gentries_per_s": 0.2,     # sparse H @ X: nnz x k multiply-adds per second (10^9): 0.20 at n = 3432
    "elementwise_gentries_per_s": 0.4,   # n x k elementwise passes (rounding bound E, |X|, R): entries per second (10^9)
    "column_us": 20.0,              # per claimed column: three 2-norms of length n (ravel + ddot), fixed overhead
    "column_per_entry_ns": 4.0,     # ... plus this per entry (35 us measured at n = 3432)
    "gram_dot_ns_per_term": 110.0,  # one accurately rounded inner product of a cluster: ns per term (2 n terms)
    "ldl_gflops": 4.0,              # dense Bunch-Kaufman + explicit residual, n^3/3 + 4 n^3 flops: 2.2 / 5.0 at 924 / 3432
    "splu_ns_per_fill_op": 1.0,     # sparse LU: ns per (nnz(L+U))^2 / n operation: 1.2 / 0.78 at 3432 / 12870
    "thermo_us_per_interval_T": 400.0,   # S7: per interval and temperature (interval exp in Python)
}
# dense LDL peak memory: la.ldl copies S, lu, d; the explicit residual lu d lu^H - S and |lu||d||lu|^T hold
# about six more n x n arrays (measured 9.28 n^2 entries at 924, 8.75 at 3432)
DENSE_LDL_ARRAYS = 9.5
# sparse LU fill of the Heisenberg chain blocks (SuperLU, MMD_AT_PLUS_A, no pivoting): nnz(L+U) fitted as
# a * n^b to the measured fills at n = 924, 3432, 12870; the model is the one family the evidence run covers
SPARSE_FILL_FIT = (0.1172, 2.0061)
SPARSE_FILL_FROM_NNZ = 40.0         # fallback when the fit is switched off: nnz(L+U) ~ this x nnz(H)
SPARSE_LU_WORKING = 4.5             # peak memory relative to fill x 12 bytes (measured 4.2-4.4)


def _bytes(complex_: bool) -> int:
    return 16 if complex_ else 8


def sparse_fill_estimate(n: int, nnz: int) -> float:
    """Predicted nnz(L) + nnz(U) of the symmetric-mode SuperLU factorization (the estimate certify() uses for
    the memory budget of sparse inertia)."""
    a, b = SPARSE_FILL_FIT
    if a > 0 and b > 0:
        return float(a * max(1, n) ** b)
    return float(SPARSE_FILL_FROM_NNZ * max(nnz, n))


def dense_inertia_mb(n: int, complex_: bool = False) -> float:
    return DENSE_LDL_ARRAYS * float(n) * n * _bytes(complex_) / 2 ** 20


# a-posteriori residual of the sparse factors (D-01, ldl_residual.py): D L^H and |L| (nnz(L) ~ fill / 2 each) and
# the permuted S stay resident; each worker thread holds the temporaries of one row block, measured at
# RESIDUAL_BLOCK_FACTOR x the block budget (75 MB per thread at block_mb = 64 on the n = 12870 chain block, 2026-10-31)
RESIDUAL_BLOCK_FACTOR = 1.2


def residual_threads() -> int:
    """Threads for the block products of ldl_residual.sparse_ldl_residual_bound. Default 1 (3.2.1): the bound does not
    depend on the thread count, and under a committed-memory cap (job object / RLIMIT_AS) several concurrent sparse
    products can exhaust memory inside SciPy's C++ kernels, which ends the process (access violation) instead of
    raising MemoryError (3.2.0 evidence run, E4 W4 4x4 J2 = 0.5 under 5 GiB). Opt in with the environment variable
    CERTIFY_ED_RESIDUAL_THREADS=<n> (capped at the CPU count and 8)."""
    import os
    try:
        want = int(os.environ.get("CERTIFY_ED_RESIDUAL_THREADS", "1"))
    except ValueError:
        want = 1
    return max(1, min(8, want, os.cpu_count() or 1))


def sparse_residual_mb(n: int, nnz: int, complex_: bool = False, threads: Optional[int] = None,
                       block_mb: Optional[float] = None) -> float:
    """Memory of ldl_residual.sparse_ldl_residual_bound beyond the factors: resident arrays plus the per-thread block
    temporaries (at most ceil(total / block_mb) blocks are live at once, each at most RESIDUAL_BLOCK_FACTOR x its size)."""
    from .ldl_residual import DEFAULT_BLOCK_MB
    fill = sparse_fill_estimate(n, nnz)
    b = _bytes(complex_) + 4
    resident = (fill * b + nnz * b) / 2 ** 20
    bmb = block_mb if block_mb is not None else DEFAULT_BLOCK_MB
    total = (fill + n) * 3 * (b + 4) / 2 ** 20            # the partition's cost function summed over all rows
    blocks = max(1, int(-(-total // bmb)))
    live = min(threads if threads is not None else residual_threads(), blocks)
    return resident + live * RESIDUAL_BLOCK_FACTOR * min(bmb, total / blocks)


def sparse_inertia_mb(n: int, nnz: int, complex_: bool = False, threads: Optional[int] = None) -> float:
    """Predicted peak of one sparse inertia count: SuperLU's working storage (fill model) plus the a-posteriori
    residual (D-01). Checked against measured peaks in tests/test_planner.py."""
    fill = sparse_fill_estimate(n, nnz)
    return (SPARSE_LU_WORKING * fill * (_bytes(complex_) + 4) / 2 ** 20 + (nnz * (_bytes(complex_) + 4) + 8 * n) / 2 ** 20
            + sparse_residual_mb(n, nnz, complex_, threads))


def dense_inertia_seconds(n: int) -> float:
    return (n ** 3 / 3 + 4 * n ** 3) / (RATES["ldl_gflops"] * 1e9)


def sparse_inertia_seconds(n: int, nnz: int) -> float:
    fill = sparse_fill_estimate(n, nnz)
    return RATES["splu_ns_per_fill_op"] * 1e-9 * fill * fill / max(1, n)


def available_memory_mb() -> Optional[float]:
    """Physical memory available to this process in MiB, or None when the platform does not say."""
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            from ctypes import wintypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(m)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            return m.ullAvailPhys / 2 ** 20
        import os
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------------------------------------------
# Memory gate without a budget (3.2.2, roadmap N-03). When Options.budget_mb is None, the memory cap of the process
# decides whether a statement or a factorization is attempted: the cap the evidence runs set (experiments/portable.py:
# a Windows job object with JOB_OBJECT_LIMIT_PROCESS_MEMORY, which counts committed private memory, or RLIMIT_AS on
# POSIX, which counts address space). The 3.2.0 run showed that an allocation beyond such a cap can end the process
# instead of raising MemoryError (E4 W4 under 5 GiB); with the gate, a factorization whose estimate exceeds the memory
# left under the cap is NOT_CHECKED before it starts. Advisory like the rest of this module: it never makes a
# statement CERTIFIED or FAILED, and without a cap (or with a budget) nothing changes.
# ---------------------------------------------------------------------------------------------------------------

def _windows_job_memory_limit() -> Optional[Tuple[float, str]]:
    """(limit in bytes, kind) of the job object the process belongs to (the innermost one when jobs are nested), or None
    when the process is in no job or the job sets no memory limit."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                       "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.DWORD)]
    k32.QueryInformationJobObject.restype = wintypes.BOOL
    info = EXTENDED()
    if not k32.QueryInformationJobObject(None, 9, ctypes.byref(info), ctypes.sizeof(info), None):   # Extended
        return None
    flags = int(info.BasicLimitInformation.LimitFlags)
    limits = []
    if flags & 0x00000100:                                   # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        limits.append((float(info.ProcessMemoryLimit), "job object (per-process memory limit)"))
    if flags & 0x00000200:                                   # JOB_OBJECT_LIMIT_JOB_MEMORY
        limits.append((float(info.JobMemoryLimit), "job object (job memory limit)"))
    return min(limits) if limits else None


def _windows_private_bytes() -> Optional[float]:
    """Committed private memory of this process (what a job object's memory limit counts), in bytes."""
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
    return float(c.PagefileUsage)


def process_memory_cap() -> Optional[dict]:
    """The memory cap of this process, or None when there is none (or the platform does not say).

    Returns {'mechanism', 'cap_mb', 'in_use_mb', 'headroom_mb'}: the job object's memory limit and the committed private
    memory of the process (Windows), or the soft RLIMIT_AS and the address space in use, VmSize from /proc/self/statm
    (POSIX; 0 when /proc is missing). headroom_mb = cap_mb - in_use_mb is what a new allocation may still use. A job
    memory limit is shared by every process of the job; the headroom then counts this process only (an upper bound)."""
    import sys
    try:
        if sys.platform == "win32":
            lim = _windows_job_memory_limit()
            if lim is None:
                return None
            cap, mech = lim
            used = _windows_private_bytes() or 0.0
        else:
            import resource
            soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft == resource.RLIM_INFINITY or soft <= 0:
                return None
            cap, mech, used = float(soft), "RLIMIT_AS", 0.0
            try:
                import os
                with open("/proc/self/statm") as f:
                    used = float(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                used = 0.0
    except Exception:
        return None
    mb = 2.0 ** 20
    return {"mechanism": mech, "cap_mb": cap / mb, "in_use_mb": used / mb, "headroom_mb": max(0.0, (cap - used) / mb)}


def memory_limit_mb(budget_mb: Optional[float]) -> Tuple[Optional[float], Optional[dict]]:
    """(limit in MB, cap record) for the memory gates: budget_mb when given (record None); else the headroom under the
    process memory cap (process_memory_cap), or (None, None) when there is no cap."""
    if budget_mb is not None:
        return float(budget_mb), None
    cap = process_memory_cap()
    if cap is None:
        return None, None
    return float(cap["headroom_mb"]), cap


def memory_cap_note(cap: dict) -> str:
    return (f"the memory left under the process memory cap ({cap['mechanism']}: cap {cap['cap_mb']:.0f} MB, "
            f"{cap['in_use_mb']:.0f} MB in use, {cap['headroom_mb']:.0f} MB left; no memory budget was given)")


def recommend_inertia_method(n: int, nnz: Optional[int], dense_operator: bool, complex_: bool = False,
                             budget_mb: Optional[float] = None, dense_max: int = 2500) -> Tuple[str, str]:
    """(method, reason) for Options.inertia_method='auto'.

    The rule of 3.1 (inertia.inertia): dense for a dense operator or a sparse one of dimension <= dense_max,
    else sparse. New in 3.2: when that rule picks dense for a SPARSE operator and the predicted dense peak
    (dense_inertia_mb) exceeds the memory limit -- budget_mb when given, else the available physical memory or, when
    smaller, the memory left under the process memory cap (3.2.2) -- the sparse path is taken instead. A dense
    operator has no sparse factorization to fall back to."""
    if dense_operator:
        return "dense", "dense operator: Bunch-Kaufman on the stored matrix"
    if n > dense_max:
        return "sparse", f"dimension {n} > {dense_max}: SuperLU in symmetric mode"
    need = dense_inertia_mb(n, complex_)
    limit = budget_mb if budget_mb is not None else available_memory_mb()
    what = "budget" if budget_mb is not None else "available"
    if budget_mb is None:
        # 3.2.2 (N-03): under a process memory cap the headroom below it is a limit too
        cap_limit, _cap = memory_limit_mb(None)
        if cap_limit is not None and (limit is None or cap_limit < limit):
            limit, what = cap_limit, "left under the process memory cap"
    if limit is not None and need > limit:
        return "sparse", (f"dense LDL^T would need about {need:.0f} MB > {what} "
                          f"{limit:.0f} MB: SuperLU in symmetric mode")
    return "dense", f"dimension {n} <= {dense_max}: Bunch-Kaufman (about {need:.0f} MB)"


@dataclass
class PlanRow:
    statement: str
    runs: bool
    method: str
    seconds: float
    peak_mb: float
    basis: str

    def to_dict(self) -> dict:
        return {"statement": self.statement, "runs": self.runs, "method": self.method,
                "seconds": round(float(self.seconds), 3), "peak_mb": round(float(self.peak_mb), 1), "basis": self.basis}


@dataclass
class Plan:
    n: int
    nnz: int
    k: int
    kind: str
    dense_operator: bool
    complex: bool
    streamed: bool
    inertia_method: str
    inertia_reason: str
    baseline_mb: float
    rows: List[PlanRow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return float(sum(r.seconds for r in self.rows if r.runs))

    @property
    def peak_mb(self) -> float:
        return float(max([self.baseline_mb] + [r.peak_mb for r in self.rows if r.runs]))

    def to_dict(self) -> dict:
        return {"schema": "certify-ed/plan/1.0", "n": self.n, "nnz": self.nnz, "k": self.k, "kind": self.kind,
                "dense_operator": self.dense_operator, "complex": self.complex, "streamed": self.streamed,
                "inertia_method": self.inertia_method, "inertia_reason": self.inertia_reason,
                "baseline_mb": round(self.baseline_mb, 1), "total_seconds": round(self.total_seconds, 3),
                "peak_mb": round(self.peak_mb, 1), "rows": [r.to_dict() for r in self.rows], "notes": list(self.notes),
                "advisory": "predictions from certify_ed.cost (measured rates of one machine); never change a status"}

    def format(self) -> str:
        head = (f"plan: n = {self.n}, nnz = {self.nnz}, k = {self.k}, kind = {self.kind}, "
                f"{'dense' if self.dense_operator else 'sparse'} operator, {'streamed' if self.streamed else 'in-memory'} "
                f"vectors; baseline {self.baseline_mb:.0f} MB")
        lines = [head, f"  {'statement':20s} {'runs':5s} {'method':28s} {'seconds':>9s} {'peak MB':>9s}  basis"]
        for r in self.rows:
            lines.append(f"  {r.statement:20s} {'yes' if r.runs else 'no':5s} {r.method:28s} {r.seconds:9.2f} {r.peak_mb:9.0f}  {r.basis}")
        lines.append(f"  total {self.total_seconds:.1f} s predicted, peak {self.peak_mb:.0f} MB; inertia: {self.inertia_method} "
                     f"({self.inertia_reason})")
        for n_ in self.notes:
            lines.append(f"  note: {n_}")
        lines.append("  advisory: measured rates of one machine (certify_ed.cost.RATES); the plan never changes a status")
        return "\n".join(lines)


def plan_certification(n: int, nnz: Optional[int], k: int, kind: str = "lowest", dense_operator: bool = False,
                       complex_: bool = False, streamed: bool = False, statements: Sequence[str] = ("S1", "S2", "S3", "S4", "S8"),
                       inertia_method: str = "auto", budget_mb: Optional[float] = None, block_mb: Optional[float] = None,
                       n_clusters: Optional[int] = None, temperatures: int = 0, has_spec: bool = True,
                       has_vectors: bool = True) -> Plan:
    """Predicted seconds and peak memory per statement. nnz: of the stored operator (for a dense operator its
    nonzeros if known, else n^2). n_clusters: (quasi-)degenerate groups among the claimed values if known
    (default: k/2, the Heisenberg-block experience)."""
    n, k = int(n), int(k)
    b = _bytes(complex_)
    nnz = int(nnz if nnz is not None else n * n)
    kb = k if not streamed else 0
    op_mb = (n * n * b if dense_operator else nnz * (b + 4) + 8 * n) / 2 ** 20
    x_mb = n * kb * b / 2 ** 20
    baseline = op_mb + x_mb
    if inertia_method == "auto":
        method, why = recommend_inertia_method(n, nnz, dense_operator, complex_, budget_mb)
    else:
        method, why = inertia_method, "Options.inertia_method"
    ncl = int(n_clusters if n_clusters is not None else max(1, k // 2))
    rows: List[PlanRow] = []
    wanted = set(statements)
    from .enclosure import DEFAULT_BLOCK_MB
    blk_mb = DEFAULT_BLOCK_MB if block_mb is None else float(block_mb)
    blk_cols = max(2, min(k, int(blk_mb * 2 ** 20 // max(1, n * b * 4)))) if k else 0

    def row(sid, runs, meth, sec, add_mb, basis):
        rows.append(PlanRow(sid, runs, meth, sec, baseline + add_mb, basis))

    # S1: reference CSR from the spec plus the entrywise comparison (three sparse temporaries of nnz entries)
    s1_mb = (3 * nnz * (b + 8) + (n * n if dense_operator else 0)) / 2 ** 20
    s1_s = 1.2e-7 * nnz + 1e-6 * n
    row("S1.model_fidelity", "S1" in wanted and has_spec, "spec reference, entrywise", s1_s, s1_mb,
        "3 nnz (b+8) bytes; ~0.1 us per entry")
    # S2: residuals (gemm or spmm, rounding bound, per-column norms) and the cluster bounds (exact Gram dots)
    if dense_operator:
        prod_s = 2 * (2.0 * n * n * k) / (RATES["gemm_gflops"] * 1e9)
        s2_mb = (n * n * b + 3 * n * k * b) / 2 ** 20
        prod_basis = "|H| transient + H X, |X|, |H||X| (n^2 + 3 n k entries)"
    else:
        prod_s = 2 * nnz * k / (RATES["spmm_gentries_per_s"] * 1e9)
        s2_mb = (n * k * b + 2 * n * blk_cols * b) / 2 ** 20 if not streamed else (4 * n * blk_cols * b) / 2 ** 20
        prod_basis = (f"R (n k) + 2 blocks of {blk_cols} columns" if not streamed else f"4 blocks of {blk_cols} columns")
    elem_s = 8.0 * n * k / (RATES["elementwise_gentries_per_s"] * 1e9)
    col_s = k * (RATES["column_us"] * 1e-6 + RATES["column_per_entry_ns"] * 1e-9 * n)
    gram_s = ncl * 3 * RATES["gram_dot_ns_per_term"] * 1e-9 * 2 * n
    row("S2.eigenpairs", "S2" in wanted and has_vectors, "residual enclosure (Thm 1-2)", prod_s + elem_s + col_s + gram_s,
        s2_mb, prod_basis + f"; {ncl} clusters x 3 exact dots")
    # S3: inertia counts (lowest: 1-2 factorizations; window: 2; all: none)
    n_fact = {"lowest": 1.5, "window": 2.0}.get(kind, 0.0)
    if method == "dense":
        s3_s, s3_mb, s3_basis = n_fact * dense_inertia_seconds(n), dense_inertia_mb(n, complex_), f"{DENSE_LDL_ARRAYS:g} n^2 entries, 13/3 n^3 flops"
    else:
        s3_s, s3_mb = n_fact * sparse_inertia_seconds(n, nnz), sparse_inertia_mb(n, nnz, complex_)
        s3_basis = f"fill ~ {sparse_fill_estimate(n, nnz):.3g} x {SPARSE_LU_WORKING:g} working"
    row("S3.completeness", "S3" in wanted and has_vectors and kind in ("lowest", "window", "all"),
        f"{method} inertia x {n_fact:g}" if n_fact else "count = dimension (kind 'all')", s3_s if n_fact else 0.0,
        s3_mb if n_fact else 0.0, s3_basis if n_fact else "no factorization")
    # S4: per cluster, a Gram and a few products of g columns; observables not counted
    row("S4.eigenspaces", "S4" in wanted and has_vectors, "projector bounds per cluster", ncl * 3 * RATES["gram_dot_ns_per_term"] * 1e-9 * 2 * n,
        2 * n * 8 * b / 2 ** 20, "cluster columns only")
    row("S5.corroboration", "S5" in wanted, "second solver", 0.0, 0.0, "not modelled")
    row("S6.exact_recheck", "S6" in wanted and has_spec and has_vectors, "integer arithmetic", 0.0, n * n * 48 / 2 ** 20,
        "n^2 x 48 bytes (the 3.1 estimate); time not modelled")
    row("S7.thermodynamics", "S7" in wanted and kind == "all" and temperatures > 0, "interval sums",
        RATES["thermo_us_per_interval_T"] * 1e-6 * ncl * 2 * max(1, temperatures), 0.0, "per interval and temperature")
    n8 = 2 * ncl
    row("S8.degeneracy", "S8" in wanted and has_vectors and kind != "subset", f"{method} inertia x {n8}",
        n8 * (dense_inertia_seconds(n) if method == "dense" else sparse_inertia_seconds(n, nnz)),
        dense_inertia_mb(n, complex_) if method == "dense" else sparse_inertia_mb(n, nnz, complex_),
        "two factorizations per cluster")
    plan = Plan(n, nnz, k, kind, dense_operator, complex_, streamed, method, why, baseline, rows)
    if kind == "all" and "S8" in wanted and ncl > 50:
        plan.notes.append(f"S8 needs {n8} factorizations for {ncl} clusters; --preset whole-spectrum leaves it out")
    if streamed and dense_operator:
        plan.notes.append("dense operator: the vectors are read into memory (a dense n x n operator is larger than n x k vectors)")
    return plan


def narrow_input_mb(claim, options) -> float:
    """Megabytes of the caller's inputs stored in a narrower floating format than binary64 (float16, float32,
    complex64): certify() works on exact binary64 copies of them (precision.py), which live next to the caller's arrays
    (integer inputs are used as given, as in 3.x, and add nothing here). The plan's baseline counts
    the binary64 arrays; this is the memory of the caller's arrays on top (operator, sector isometry, in-memory vectors,
    the matrices of Options.observables, approximate_symmetries and symmetries). Vectors read from disk add none."""
    import numpy as np
    import scipy.sparse as sp
    from .enclosure import is_streamed
    from .precision import PrecisionError, is_narrow_float

    def mb(a) -> float:
        try:
            if sp.issparse(a):
                if not is_narrow_float(a.dtype):
                    return 0.0
                if a.format in ("lil", "dok"):    # one value per position: the values (not their Python containers)
                    return float(a.nnz * a.dtype.itemsize) / 2 ** 20
                return sum(int(getattr(a, k).nbytes) for k in ("data", "indices", "indptr", "row", "col", "offsets")
                           if isinstance(getattr(a, k, None), np.ndarray)) / 2 ** 20
            if isinstance(a, np.ndarray) and not is_streamed(a) and not is_narrow_float(a.dtype):
                return 0.0
            if isinstance(a, np.ndarray) and not is_streamed(a):
                return float(a.nbytes) / 2 ** 20
        except (PrecisionError, TypeError, ValueError):
            return 0.0
        return 0.0
    def first(v):
        # S of an entry (S, lattice[, allowed]) as certify reads it: a sequence, a mapping {0: S, ...}, or a collection
        # that can be iterated again but not indexed (a dict.values() view, a class with __iter__ only); an iterator is
        # not read here (it would be spent before certify reads it)
        if v is None or sp.issparse(v) or isinstance(v, (str, bytes, Iterator)):
            return None
        if isinstance(v, Mapping):
            return v.get(0)
        if hasattr(v, "__getitem__"):
            try:
                return v[0] if len(v) else None
            except (TypeError, IndexError, KeyError):
                return None
        return next(iter(v), None) if isinstance(v, Iterable) else None
    parts = [claim.operator, claim.sector_isometry, claim.eigenvectors]
    parts += list(options.observables.values())
    parts += [first(v) for v in options.approximate_symmetries.values()] + [first(v) for v in options.symmetries.values()]
    seen, total = set(), 0.0
    for a in parts:
        if a is not None and id(a) not in seen:
            seen.add(id(a))
            total += mb(a)
    return total


def plan_for_claim(claim, spec, options, narrow_mb: Optional[float] = None) -> Plan:
    """The plan for a Claim / ModelSpec / Options triple (what Options(plan=True) and `certify-ed plan` print).
    narrow_mb: memory of the caller's inputs in a narrower format than binary64 (narrow_input_mb; computed from the
    claim when None), added to the baseline and to every statement's peak."""
    import numpy as np
    import scipy.sparse as sp
    from .enclosure import is_streamed
    from .presets import resolve_selection
    X = claim.eigenvectors
    k = int(np.asarray(claim.eigenvalues).size)
    op = claim.operator
    if op is not None and (sp.issparse(op) or isinstance(op, np.ndarray)):
        n = int(op.shape[0])
        dense = isinstance(op, np.ndarray)
        nnz = int(op.nnz) if sp.issparse(op) else int(np.count_nonzero(op)) if n * n <= 2 ** 26 else n * n
        complex_ = bool(np.iscomplexobj(op.data if sp.issparse(op) else op))
    else:
        n = int(claim.states.size if claim.states is not None else (claim.sector_isometry.shape[1] if claim.sector_isometry is not None
                                                                    else (spec.dim if spec is not None else (op.shape[0] if op is not None else 0))))
        dense = False
        nnz = int(n * max(1, spec.n_terms) * 2) if spec is not None else n * 8
        complex_ = bool(spec is not None and not spec.is_real())
    if X is not None:
        complex_ = complex_ or np.dtype(X.dtype).kind == "c"
    sel = resolve_selection(options.preset, options.only, options.skip, options.narrowing_reason,
                            corroborate=bool(options.corroborate), exact_recheck=bool(options.exact_recheck),
                            temperatures=bool(len(options.temperatures)), degeneracy=bool(options.degeneracy))
    plan = plan_certification(n, nnz, k, claim.kind, dense, complex_, X is not None and is_streamed(X), sel.wanted,
                              options.inertia_method, options.budget_mb, options.block_mb, temperatures=len(options.temperatures),
                              has_spec=spec is not None, has_vectors=X is not None)
    extra = narrow_input_mb(claim, options) if narrow_mb is None else float(narrow_mb)
    if extra > 0:          # only then: a binary64 claim keeps its plan
        plan.baseline_mb += extra
        for r in plan.rows:
            r.peak_mb += extra
        plan.notes.append(f"inputs in a narrower format than binary64: certify() works on exact binary64 copies, which "
                          f"live next to the caller's arrays ({extra:.1f} MB of the caller's arrays added to every peak)")
    return plan


__all__ = ["E2_RECORDS", "COEFFICIENTS", "AUTO_FACTOR", "AUTO_FLOOR_SECONDS", "estimate_seconds", "auto_budget",
           "resolve_budget_seconds", "RATES", "DENSE_LDL_ARRAYS", "SPARSE_FILL_FIT", "sparse_fill_estimate",
           "dense_inertia_mb", "sparse_inertia_mb", "sparse_residual_mb", "RESIDUAL_BLOCK_FACTOR", "dense_inertia_seconds", "sparse_inertia_seconds",
           "available_memory_mb", "process_memory_cap", "memory_limit_mb", "memory_cap_note", "recommend_inertia_method", "Plan", "PlanRow", "plan_certification", "plan_for_claim"]
