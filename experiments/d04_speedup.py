"""D-04: wall time of parallel.certify_many with 1, 2, 4 and 8 workers on the U(1) sectors of a Heisenberg chain.

The jobs are the sectors with nup up spins of the periodic chain of N sites, for nup in NUPS (default N = 16 and
nup = 3 to 13, dimensions 560 to 12870: the configuration of the first measurement, 22 November 2026; the chain of 18
sites did not finish within a job's clock). Each sector claims its lowest eigenpairs from scipy's eigsh (the producer),
cut at a gap so that the claim does not split a level, and certify() runs with the default Options. For every worker
count the record holds the wall time of certify_many, where each task ran (a worker or this process) and its seconds,
the id, status and reason code of every statement, and whether the body and the arrays of every certificate equal
those of the one-worker run apart from the fields that record time (_dropped: matched by exact path, and listed in the
record as fields_not_compared); rules 1 to 4 of certify_ed/parallel.py make them independent of the worker count.
The record of 22 November 2026 (docs/results/d04_speedup_n16_20261122.json) was made by the version of 57607a70,
which compared the statements and arrays only, with keys dropped by substring. The speed-up is the one-worker wall
time over the W-worker wall time. It depends on the free cores of the machine, so the record also holds the cores
this process may use (os.sched_getaffinity where it exists), the load average before and after each run (POSIX), and
the CPU seconds of each run: this process (time.process_time) and its finished child processes, the workers (os.times;
POSIX only, None elsewhere).

usage: python experiments/d04_speedup.py [--N 16] [--nups 3,4,...,13] [--k 4] [--workers 1,2,4,8] [--cap-gib 32]
"""
import os
import sys
import time

import numpy as np

from harness import Run
from portable import cap_memory, peak_rss_mb
import certify_ed as ce
from certify_ed import parallel as par

def _arg(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def _dropped(path):
    """The fields of a certificate body that record time and so differ between runs by construction: the block
    'timing', 'environment/created_utc', and every key 'seconds' below 'statements' (of a statement, or of a count in
    its evidence). They are matched by their exact path; a list index is written '[]'."""
    return (path == ("timing",) or path == ("environment", "created_utc")
            or (len(path) > 1 and path[0] == "statements" and path[-1] == "seconds"))


def strip(x, path=(), dropped=None):
    """x without the fields of _dropped (their paths are added to the set dropped); everything else is kept for an
    exact comparison (floats by their bits, arrays by dtype, shape and bytes)."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            p = path + (str(k),)
            if _dropped(p):
                if dropped is not None:
                    dropped.add("/".join(p))
                continue
            out[k] = strip(v, p, dropped)
        return out
    if isinstance(x, (list, tuple)):
        return [strip(v, path + ("[]",), dropped) for v in x]
    if isinstance(x, np.ndarray):
        return ("ndarray", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, float):
        return ("float", x.hex()) if x == x else ("float", "nan")
    return x


def _status(s):
    """(id, status, reason code or None) of a statement record."""
    return [s["id"], s["status"], s.get("reason_code") or (s.get("evidence") or {}).get("reason_code")]


def _load():
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _child_cpu():
    """CPU seconds of the finished child processes (user + system); None where the OS does not report them."""
    if os.name != "posix":
        return None
    t = os.times()
    return t.children_user + t.children_system


def _cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def sector_claim(spec, nup, k_max):
    """The claim of sector nup: its lowest eigenpairs from eigsh, at most k_max, cut where the next level is apart."""
    from scipy.sparse.linalg import eigsh
    st = ce.u1_state_set(spec, nup)
    Hs = ce.build_bitwise(spec, st).tocsr()
    n = Hs.shape[0]
    t0 = time.perf_counter()
    if n <= 400:
        from certify_ed._linalg import eigh
        w, V = eigh(Hs.toarray())
    else:
        w, V = eigsh(Hs, k=min(k_max + 4, n - 2), which="SA", tol=0)
        order = np.argsort(w)
        w, V = w[order], V[:, order]
    k = next((j for j in range(min(k_max, len(w) - 1), 0, -1) if w[j] - w[j - 1] > 1e-6), 1)
    claim = ce.Claim(w[:k], V[:, :k], operator=Hs, kind="lowest", states=st)
    return claim, {"nup": nup, "dim": int(n), "k": int(k), "producer_seconds": round(time.perf_counter() - t0, 3)}


def main(argv):
    N = int(_arg(argv, "--N", "16"))
    nups = [int(x) for x in _arg(argv, "--nups", ",".join(str(i) for i in range(3, 14))).split(",") if x.strip()]
    k_max = int(_arg(argv, "--k", "4"))
    workers = [int(x) for x in _arg(argv, "--workers", "1,2,4,8").split(",") if x.strip()]
    if workers[0] != 1:
        workers = [1] + workers                 # the one-worker run is the reference
    cap_gib = float(_arg(argv, "--cap-gib", "32"))
    mech = cap_memory(int(cap_gib * 2 ** 30))
    with Run("d04_speedup") as run:
        run.log(f"N {N}, sectors nup {nups}, k <= {k_max}, workers {workers}; memory cap {cap_gib} GiB ({mech}); "
                f"cores available {_cores()}")
        spec = ce.heisenberg_chain(N)
        jobs, rows = [], []
        for nup in nups:
            claim, row = sector_claim(spec, nup, k_max)
            jobs.append((claim, spec, ce.Options()))
            rows.append(row)
            run.log(f"  sector nup={nup}: dim {row['dim']}, k {row['k']}, producer {row['producer_seconds']} s")
        out = {"N": N, "sectors": rows, "cap_gib": cap_gib, "cap_mechanism": mech, "cores_available": _cores(),
               "cpu_count": os.cpu_count(), "runs": []}
        ref = None
        for W in workers:
            before = _load()
            c0, k0 = time.process_time(), _child_cpu()
            t0 = time.perf_counter()
            certs, rep = par.certify_many(jobs, workers=W, cap_mb=cap_gib * 1024, report=True)
            wall = time.perf_counter() - t0
            c1, k1 = time.process_time(), _child_cpu()
            after = _load()
            dropped = set()
            body = [(strip(c.body, (), dropped), strip(dict(c.arrays), ("arrays",), dropped)) for c in certs]
            if ref is None:
                ref = body
                ref_wall = wall
            same = body == ref
            statuses = [[_status(s) for s in c.statements] for c in certs]
            r = {"workers": W, "wall_seconds": round(wall, 3), "speedup": round(ref_wall / wall, 3),
                 "cpu_seconds_parent": round(c1 - c0, 3),
                 "cpu_seconds_workers": None if k0 is None else round(k1 - k0, 3),
                 "identical_to_one_worker": bool(same), "fields_not_compared": sorted(dropped),
                 "load_before": before, "load_after": after,
                 "tasks": [{"where": t.where, "seconds": round(float(t.seconds), 3), "reason": t.reason}
                           for t in rep.tasks],
                 "statuses": statuses}
            out["runs"].append(r)
            run.log(f"workers {W}: {wall:.1f} s, speed-up {r['speedup']}, identical {same}, load {before} -> {after}")
            run.record("d04_speedup", out)          # rewritten after every worker count
        out["peak_rss_mb"] = peak_rss_mb()
        run.record("d04_speedup", out)


if __name__ == "__main__":
    main(sys.argv[1:])
