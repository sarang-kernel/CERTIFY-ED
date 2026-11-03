"""E2 reproduction for D-01: reach of the sparse S3 path with the a-posteriori residual bound.

For the Heisenberg chain S^z = 0 blocks (N = 12, 14, 16, 18: d = 924, 3432, 12870, 48620) the script builds the
block (certify_ed.build_bitwise), computes the k lowest eigenpairs with ARPACK (scipy eigsh, seeded start vector),
and certifies S1-S3 with Options(inertia_method='sparse', budget_mb=BUDGET) in a fresh process per point. It
records, per point: dimension, nnz, S3 status, the per-statement seconds, the inertia records (beta, residual,
rounding and shift terms, blocks, threads, fill), peak RSS and the planner's memory prediction. The 48620 block is
attempted only through the memory gate: cost.sparse_inertia_mb predicts about 19 GB for it, so S3 is expected to be
NOT_CHECKED (BUDGET_MEMORY) without any allocation. A second pass with inertia_method='sparse-apriori' records the
3.1 bound for comparison on the points that fit.

Usage: python experiments/e2_s3_reach.py [--out results/e2_s3_reach.json] [--budget-mb 7000] [--max-N 18]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse.linalg as spla
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import portable
import certify_ed as ce
cfg = json.loads(sys.argv[1])
N, k, method, budget = cfg["N"], cfg["k"], cfg["method"], cfg["budget_mb"]
t0 = time.time()
spec = ce.heisenberg_chain(N)
st = ce.u1_state_set(spec, N // 2)
H = ce.build_bitwise(spec, st)
t_build = time.time() - t0
t1 = time.time()
v0 = np.random.default_rng(cfg["seed"]).standard_normal(H.shape[0])
w, V = spla.eigsh(H, k=k + 4, which="SA", tol=0, v0=v0)
order = np.argsort(w); w, V = w[order], V[:, order]
t_solve = time.time() - t1
kk = k
while kk < len(w) - 1 and w[kk] - w[kk - 1] < 1e-6:
    kk += 1
claim = ce.Claim(w[:kk], V[:, :kk], operator=H, kind="lowest", states=st)
opts = ce.Options(inertia_max_dim=10 ** 7, inertia_method=method, budget_mb=budget, only=("S1", "S2", "S3"),
                  narrowing_reason="E2 reach reproduction: S1-S3 only")
t2 = time.time()
cert = ce.certify(claim, spec, opts)
tot = time.time() - t2
st_ = {s["id"]: {"status": s["status"], "seconds": s["seconds"], "summary": s["summary"][:240],
                 "reason_code": s.get("reason_code")} for s in cert.statements}
s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]["evidence"]
counts = [{kk_: c.get(kk_) for kk_ in ("sigma", "negative", "beta", "method", "seconds", "fill_nnz", "note", "residual")}
          for c in s3.get("counts", [])]
from certify_ed.cost import sparse_inertia_mb
out = {"N": N, "dim": int(H.shape[0]), "nnz": int(H.nnz), "k_claimed": int(kk), "t_build": t_build, "t_solve": t_solve,
       "t_certify": tot, "statements": st_, "counts": counts, "gap_lower_bound": s3.get("gap_lower_bound"),
       "predicted_s3_mb": sparse_inertia_mb(int(H.shape[0]), int(H.nnz)), "maxrss_mb": portable.peak_rss_mb(),
       "method": method}
print("RESULT" + json.dumps(out))
'''.replace("ROOT", repr(ROOT.replace("\\", "/")))


def run_child(cfg, timeout):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONPATH"] = ROOT
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}, time.time() - t
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:]), time.time() - t
    return {"error": (p.stderr or p.stdout)[-800:]}, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e2_s3_reach.json"))
    ap.add_argument("--budget-mb", type=float, default=7000.0)
    ap.add_argument("--max-N", type=int, default=18)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for N in (12, 14, 16, 18, 20):
        if N > args.max_N:
            break
        for method in ("sparse", "sparse-apriori"):
            if method == "sparse-apriori" and N >= 18:
                continue
            cfg = {"N": N, "k": args.k, "method": method, "budget_mb": args.budget_mb, "seed": 1000 + N}
            res, wall = run_child(cfg, timeout=3600)
            res["wall"] = wall
            rows.append(res)
            if "error" in res:
                print(f"N={N} {method}: error {res['error'][-300:]}")
            else:
                s3 = res["statements"]["S3.completeness"]
                betas = [c["beta"] for c in res["counts"]]
                print(f"N={N} d={res['dim']:>6d} {method:14s} S3={s3['status']:12s} certify {res['t_certify']:.1f}s "
                      f"(S3 {s3['seconds']:.1f}s) beta={betas} mem {res['maxrss_mb']:.0f}MB pred {res['predicted_s3_mb']:.0f}MB "
                      f"| {s3['summary'][:90]}")
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"settings": vars(args), "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
