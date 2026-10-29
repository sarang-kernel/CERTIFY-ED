"""Measure the machine rates and the memory constants of the planner (certify_ed.cost, roadmap D-08).

    python experiments/planner_calibration.py [N ...]        default N = 12 14 (Sz = 0 blocks of dimension 924, 3432)

For each Heisenberg chain Sz = 0 block it measures, in fresh processes where memory is concerned:
  * dense Bunch-Kaufman inertia (certify_ed.inertia.inertia_dense): seconds and peak resident memory above the
    process baseline, in units of n^2 entries  -> cost.DENSE_LDL_ARRAYS, RATES['ldl_gflops']
  * sparse SuperLU inertia in symmetric mode: seconds, fill nnz(L)+nnz(U), peak memory -> cost.SPARSE_FILL_FIT,
    RATES['splu_ns_per_fill_op'], cost.SPARSE_LU_WORKING
  * dense gemm (H X and |H||X| on k = n columns): GFlop/s -> RATES['gemm_gflops']
  * sparse H X: entries per second -> RATES['spmm_gentries_per_s']
  * one accurately rounded inner product (fp.exact_dot) of two columns: ns per term -> RATES['gram_dot_ns_per_term']
  * three 2-norms of a column (the per-column work of enclosure.residuals) -> RATES['column_us'], ['column_per_entry_ns']
The numbers are printed as JSON and as the lines to paste into cost.py. Advisory: the planner never changes a status.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import sys, json, time, numpy as np, scipy.sparse as sp
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/experiments")
import portable
import certify_ed as ce
from certify_ed.inertia import inertia_dense, inertia_sparse
N, mode = int(sys.argv[1]), sys.argv[2]
spec = ce.heisenberg_chain(N); st = ce.u1_state_set(spec, N // 2); H = ce.build_bitwise(spec, st)
n = H.shape[0]; sigma = float(-0.4 * N)
if mode == "dense":
    A = H.toarray()
    base = portable.peak_rss_mb()
    t = time.time(); r = inertia_dense(A, sigma); dt = time.time() - t
    out = {"n": n, "seconds": dt, "peak_above_base_mb": portable.peak_rss_mb() - base, "entries_per_n2": (portable.peak_rss_mb() - base) * 2 ** 20 / 8 / n / n,
           "conclusive": r.conclusive, "negative": r.negative}
else:
    base = portable.peak_rss_mb()
    t = time.time(); r = inertia_sparse(H, sigma); dt = time.time() - t
    out = {"n": n, "nnz": int(H.nnz), "seconds": dt, "fill_nnz": r.fill_nnz, "peak_above_base_mb": portable.peak_rss_mb() - base,
           "conclusive": r.conclusive, "negative": r.negative}
print("RESULT" + json.dumps(out))
'''.replace("ROOT", repr(ROOT.replace("\\", "/")))


def child(N, mode):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    p = subprocess.run([sys.executable, "-c", CHILD, str(N), mode], capture_output=True, text=True, env=env)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    return json.loads(line[0][6:]) if line else {"error": (p.stdout + p.stderr)[-800:]}


def in_process_rates(N):
    import numpy as np
    import certify_ed as ce
    from certify_ed.fp import exact_dot, norm2_lower, norm2_upper
    spec = ce.heisenberg_chain(N)
    st = ce.u1_state_set(spec, N // 2)
    H = ce.build_bitwise(spec, st)
    n = H.shape[0]
    rng = np.random.default_rng(0)
    X = np.asfortranarray(rng.standard_normal((n, n)))
    out = {"n": n}
    A = H.toarray()
    t = time.time()
    HX = A @ X
    AX = np.abs(A) @ np.abs(X)
    dt = time.time() - t
    out["gemm_gflops"] = 2 * 2.0 * n * n * n / dt / 1e9
    del HX, AX, A
    t = time.time()
    HX = H @ X
    dt = time.time() - t
    out["spmm_gentries_per_s"] = H.nnz * n / dt / 1e9
    del HX
    t = time.time()
    for _ in range(20):
        exact_dot(X[:, 0], X[:, 1])
    out["gram_dot_ns_per_term"] = (time.time() - t) / 20 / (2 * n) * 1e9
    t = time.time()
    for i in range(200):
        norm2_lower(X[:, i])
        norm2_upper(X[:, i])
        norm2_upper(X[:, i + 1])
    out["column_seconds"] = (time.time() - t) / 200
    return out


if __name__ == "__main__":
    Ns = [int(a) for a in sys.argv[1:]] or [12, 14]
    res = {"dense": [], "sparse": [], "rates": []}
    for N in Ns:
        if 2 ** N <= 2 ** 14:
            res["dense"].append(child(N, "dense"))
        res["sparse"].append(child(N, "sparse"))
        if 2 ** N <= 2 ** 14:
            res["rates"].append(in_process_rates(N))
        print(json.dumps({k: v[-1] for k, v in res.items() if v}), flush=True)
    import math
    fills = [(r["n"], r["fill_nnz"]) for r in res["sparse"] if "fill_nnz" in r]
    if len(fills) >= 2:
        xs = [math.log(n) for n, _ in fills]
        ys = [math.log(f) for _, f in fills]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = math.exp(my - b * mx)
        res["SPARSE_FILL_FIT"] = (a, b)
    if res["dense"]:
        res["DENSE_LDL_ARRAYS"] = max(r["entries_per_n2"] for r in res["dense"] if "entries_per_n2" in r)
        res["ldl_gflops"] = [(r["n"] ** 3 / 3 + 4 * r["n"] ** 3) / r["seconds"] / 1e9 for r in res["dense"]]
    if res["sparse"]:
        res["splu_ns_per_fill_op"] = [r["seconds"] * 1e9 / (r["fill_nnz"] ** 2 / r["n"]) for r in res["sparse"] if "fill_nnz" in r]
        res["SPARSE_LU_WORKING"] = [r["peak_above_base_mb"] * 2 ** 20 / (r["fill_nnz"] * 12) for r in res["sparse"] if "fill_nnz" in r]
    print(json.dumps(res, indent=1))
