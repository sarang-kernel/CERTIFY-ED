"""Entrywise comparison of the full E7 chain matrices (20 sites, S^z = 0, 184756 states) between the Lean
rebuild (`ceapps export`) and the Python builder (`certify_ed.checker.exact_matrix`), with a lower memory
footprint than compare_models.py: entries are held in sorted int64 arrays instead of dictionaries of
Fractions, and equality of the rational entries a / (den_py * scale) and v / den_lean is tested by the exact
cross products a * den_lean == v * den_py * scale (magnitudes are checked to stay below 2^62 first).

Measured for every model: wall time and peak working set of the Lean export process and of this Python
process, so that the cost of the 20-site chains can be extrapolated from the smaller models first.

usage: python compare_big.py --exe <ceapps.exe> --code <certify-ed clone> --work <dir> --out <json> KEY [KEY ...]
       python compare_big.py --merge <out json> <part json> [<part json> ...]

The keys are run one per process (chain12_J2 and square4_J2 first, to measure; then chain20 and chain20_J2), so
that the peak memory of each process belongs to one model. --merge joins the part files into one file with the
estimate for the 20-site chain with next-nearest bonds (4048556 nonzero entries are not known in advance; the
estimate uses 21 entries per state, one diagonal and about one per bond of each of the two bond families),
scaled from the largest measured model by the ratio of nonzero entries.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time

import numpy as np


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.OpenProcess.restype = wt.HANDLE
_k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_k32.CloseHandle.argtypes = [wt.HANDLE]
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]


def peak_ws(handle=None):
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    h = handle if handle is not None else _k32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return int(pmc.PeakWorkingSetSize)


def run_measured(cmd):
    """Run a child process; return (exit code, seconds, peak working set in bytes)."""
    t0 = time.time()
    p = subprocess.Popen(cmd)
    h = _k32.OpenProcess(0x0400 | 0x0010, False, p.pid)
    peak = 0
    while p.poll() is None:
        peak = max(peak, peak_ws(h))
        time.sleep(0.2)
    peak = max(peak, peak_ws(h))
    _k32.CloseHandle(h)
    return p.returncode, time.time() - t0, peak


def read_lean(path):
    with open(path) as f:
        N, n, den = (int(x) for x in f.readline().split())
        states = np.array([int(f.readline()) for _ in range(n)], dtype=np.int64)
        body = np.array(f.read().split(), dtype=np.int64).reshape(-1, 3)
    return N, states, den, body


def py_model(key):
    import certify_ed as ce
    table = {
        "chain12_J2": (lambda: ce.heisenberg_chain(12, J2=0.5), 6),
        "square4": (lambda: ce.heisenberg_square(4, 4, J2=0.0), 8),
        "square4_J2": (lambda: ce.heisenberg_square(4, 4, J2=0.5), 8),
        "chain20": (lambda: ce.heisenberg_chain(20, J2=0.0), 10),
        "chain20_J2": (lambda: ce.heisenberg_chain(20, J2=0.5), 10),
    }
    build, up = table[key]
    spec = build()
    return spec, ce.u1_state_set(spec, up), 1


def compare(exe, key, work):
    from certify_ed.checker import exact_matrix
    path = os.path.join(work, f"big_{key}.txt")
    rc, t_lean, peak_lean = run_measured([exe, "export", key, path])
    assert rc == 0, f"ceapps export {key} failed"
    t1 = time.time()
    N, states_l, den_l, body = read_lean(path)
    t_read = time.time() - t1
    t2 = time.time()
    spec, st, scale = py_model(key)
    ex = exact_matrix(spec, st)
    t_py = time.time() - t2
    t3 = time.time()
    order = np.array([int(s) for s in st.states_in_order()], dtype=np.int64)
    states_equal = bool(np.array_equal(np.sort(order), states_l))
    imag_zero = bool(np.all(np.asarray(ex.im) == 0))
    weights_one = bool(np.all(np.asarray(ex.weights) == 1))
    den_py = int(ex.den)
    # Python entries keyed by (row state, column state), Lean entries likewise
    pr = order[np.asarray(ex.rows, dtype=np.int64)]
    pc = order[np.asarray(ex.cols, dtype=np.int64)]
    pv = np.array([int(a) for a in ex.re], dtype=np.int64)
    keep = pv != 0
    pr, pc, pv = pr[keep], pc[keep], pv[keep]
    lr = states_l[body[:, 0]]
    lc = states_l[body[:, 1]]
    lv = body[:, 2]
    bound_ok = (int(np.abs(pv).max()) * den_l < 2 ** 62) and (int(np.abs(lv).max()) * den_py * scale < 2 ** 62)
    nbits = int(N)
    pk = pr * (1 << nbits) + pc
    lk = lr * (1 << nbits) + lc
    po = np.argsort(pk, kind="stable")
    lo = np.argsort(lk, kind="stable")
    keys_equal = bool(pk.size == lk.size and np.array_equal(pk[po], lk[lo]))
    unique = bool(np.all(np.diff(pk[po]) > 0)) if pk.size > 1 else True
    vals_equal = bool(keys_equal and bound_ok and np.array_equal(pv[po] * den_l, lv[lo] * (den_py * scale)))
    t_cmp = time.time() - t3
    return {"model": key, "sites": int(N), "states": int(states_l.size), "nonzeros_lean": int(lv.size),
            "nonzeros_python": int(pv.size), "state_sets_equal": states_equal, "python_imaginary_part_zero": imag_zero,
            "python_weights_one": weights_one, "python_keys_unique": unique, "keys_equal": keys_equal,
            "magnitudes_below_2_62": bound_ok, "entries_equal": vals_equal, "den_lean": int(den_l), "den_python": den_py,
            "seconds_lean_export": round(t_lean, 1), "peak_bytes_lean_export": peak_lean,
            "seconds_read_lean_file": round(t_read, 1), "seconds_python_build": round(t_py, 1),
            "seconds_compare": round(t_cmp, 1), "peak_bytes_python_process_so_far": peak_ws()}


def merge(out, parts):
    models = []
    for p in parts:
        models += json.load(open(p))["models"]
    small = [m for m in models if m["states"] < 184756]
    est = None
    if small:
        m = max(small, key=lambda x: x["nonzeros_lean"])
        nnz = 184756 * 21
        ratio = nnz / m["nonzeros_lean"]
        est = {"basis": m["model"], "nonzeros_estimate": nnz, "ratio": ratio,
               "peak_bytes_estimate": int(m["peak_bytes_python_process_so_far"] * ratio),
               "seconds_estimate": round((m["seconds_lean_export"] + m["seconds_read_lean_file"] +
                                          m["seconds_python_build"] + m["seconds_compare"]) * ratio)}
    big = [m for m in models if m["states"] >= 184756]
    rec = {"description": "Entrywise comparison of the full E7 chain matrices (20 sites, S^z = 0) between ceapps export "
                          "and certify_ed.checker.exact_matrix (apps_export/compare_big.py); measured first on smaller "
                          "models, then run",
           "limits": {"memory_bytes": 6 * 2 ** 30, "seconds": 7200},
           "measurements": {m["model"]: m for m in small}, "estimate_chain20_J2": est, "models": big,
           "all_equal": bool(big) and all(m["entries_equal"] and m["state_sets_equal"] for m in big)}
    with open(out, "w", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("all equal:", rec["all_equal"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--merge":
        merge(sys.argv[2], sys.argv[3:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("keys", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    os.makedirs(a.work, exist_ok=True)
    res = []
    for k in a.keys:
        r = compare(a.exe, k, a.work)
        print(json.dumps(r), flush=True)
        res.append(r)
        os.remove(os.path.join(a.work, f"big_{k}.txt"))
    with open(a.out, "w", newline="\n") as f:
        json.dump({"models": res, "all_equal": all(r["entries_equal"] and r["state_sets_equal"] for r in res)},
                  f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
