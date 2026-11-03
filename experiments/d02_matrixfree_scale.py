"""D-02 scale run: S2 radii for a claim whose matrix is never stored, d = 2^20 .. 2^24.

Model: Heisenberg chain of N = 20, 22, 24 sites (periodic, J = 1), full Hilbert space d = 2^N. Claim: the k = pi
one-magnon state x_j = (-1)^j / sqrt(N) on the basis state with the spin at site j flipped from the fully polarised
state (an exact eigenvector with eigenvalue J (N/4 - 2)); mu is set to the exact value. The vector is written to a
float64 .npy file and passed to certify() as a memmap (A-06), so that neither the matrix nor a second copy of the
vector is held in memory. Options(matrix_free=True, only=("S1", "S2")). Each point runs in a fresh process; the
script records S1/S2 status and seconds, the certified radius, the adapter's rounding width W and norm bound,
one matvec's time, and the peak RSS.

Usage: python experiments/d02_matrixfree_scale.py [--max-N 24] [--out results/d02_matrixfree_scale.json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import portable
import certify_ed as ce
from certify_ed.matrixfree import MatrixFreeOperator
from certify_ed.enclosure import OperatorAdapter
cfg = json.loads(sys.argv[1])
N = cfg["N"]
spec = ce.heisenberg_chain(N)
n = 2 ** N
t0 = time.time()
# site 0 is the most significant bit; state order descending: index = n - 1 - state. Fully polarised "up" = all
# digits q-1 = 1 -> state n - 1 (index 0); flipping site j clears bit N-1-j.
path = cfg["vec"]
x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(n, 1))
x[:] = 0.0
up = n - 1
for j in range(N):
    s = up & ~(1 << (N - 1 - j))
    idx = (n - 1 - s) if spec.state_order == "descending" else s
    x[idx, 0] = (-1.0) ** j / np.sqrt(N)
x.flush(); del x
t_vec = time.time() - t0
mu = float(N) / 4.0 - 2.0
X = np.load(path, mmap_mode="r")
op = MatrixFreeOperator(spec)
t1 = time.time()
y = op.matvec(np.asarray(X[:, 0]))
t_matvec = time.time() - t1
ad = OperatorAdapter(op)
t_adapter = time.time() - t1 - t_matvec
del y
claim = ce.Claim(np.array([mu]), X, kind="subset")
opts = ce.Options(matrix_free=True, only=("S1", "S2"), narrowing_reason="D-02 scale run: S1-S2 only",
                  block_mb=256.0)
t2 = time.time()
cert = ce.certify(claim, spec, opts)
t_cert = time.time() - t2
st = {s["id"]: {"status": s["status"], "seconds": s["seconds"], "summary": s["summary"][:300],
                "reason_code": s.get("reason_code")} for s in cert.statements}
out = {"N": N, "dim": n, "mu": mu, "t_vector": t_vec, "t_matvec": t_matvec, "t_adapter_norm": t_adapter,
       "t_certify": t_cert, "statements": st, "radius": float(cert.arrays["radius"][0]),
       "W": int(op.W), "terms_per_row": int(op.terms_per_row), "width_used": int(ad.width),
       "norm_bound_spec": float(op.norm_bound()), "norm_upper_adapter": float(ad.norm_upper()),
       "maxrss_mb": portable.peak_rss_mb()}
print("RESULT" + json.dumps(out))
'''.replace("ROOT", repr(ROOT.replace("\\", "/")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-N", type=int, default=24)
    ap.add_argument("--min-N", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "d02_matrixfree_scale.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["PYTHONPATH"] = ROOT
    for N in (16, 20, 22, 24):
        if N < args.min_N:
            continue
        if N > args.max_N:
            break
        with tempfile.TemporaryDirectory() as d:
            cfg = {"N": N, "vec": os.path.join(d, f"magnon_{N}.npy").replace("\\", "/")}
            t = time.time()
            p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], capture_output=True, text=True, env=env)
            wall = time.time() - t
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
            if line:
                res = json.loads(line[0][6:])
                res["wall"] = wall
                print(f"N={N} d=2^{N} S2={res['statements']['S2.eigenpairs']['status']} radius {res['radius']:.3e} "
                      f"matvec {res['t_matvec']:.1f}s certify {res['t_certify']:.1f}s (S2 {res['statements']['S2.eigenpairs']['seconds']:.1f}s) "
                      f"W={res['W']} width={res['width_used']} norm {res['norm_upper_adapter']:.4g} mem {res['maxrss_mb']:.0f}MB")
            else:
                res = {"N": N, "error": (p.stderr or p.stdout)[-1200:], "wall": wall}
                print(f"N={N}: error {res['error'][-400:]}")
            rows.append(res)
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
