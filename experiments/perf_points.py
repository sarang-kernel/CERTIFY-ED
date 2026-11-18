"""Performance points of the E2 experiment (roadmap H-04; groundwork for the benchmark page G-10).

A point is one row of experiments/e2_cost.py, measured exactly as that experiment measures it: a producer process
(QuSpin basis and H, ARPACK eigsh with the start vector of seed 12345 + the row's index, as in every evidence run),
then a certifier process (certify() with S1 to S4 and S8, `inertia_method="sparse"`), each capped at 4.5 GiB with one
thread, reporting the certify() wall seconds per statement and the peak resident memory (`portable.peak_rss_mb`).
The numbers are therefore comparable row by row with the E2 records of the evidence runs (results/run_full_*/
e2_cost_*/data.json); import time of the processes is not included.

    python experiments/perf_points.py --record FILE [--points small] [--repeat 3]
    python experiments/perf_points.py --record tests/fixtures/perf_baseline.json --from-run results/run_full_20261105

The first measures now (per point the certifier run with the smallest total of `--repeat` runs, the statuses and the
machine); the second copies the rows of an evidence run's E2 record unchanged (the committed baseline is the 3.2.1
run's own measurement, made on the release machine with two experiments in parallel).
tests/test_perf_regression.py (run with `pytest --perf`) compares the small points against it;
experiments/bench_table.py tabulates any measurement against a run record.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e2_cost            # noqa: E402
import portable           # noqa: E402
from harness import ROOT  # noqa: E402

E2_POINTS = ([dict(model="chain", block="u1", N=N) for N in (12, 14, 16, 18)]
             + [dict(model="chain", block="k0", N=N) for N in (16, 18, 20, 22, 24)]
             + [dict(model="square", block="u1", Lx=4, Ly=4), dict(model="square", block="u1", Lx=4, Ly=4, J2=0.5)])
RUN_SEED = 12345          # harness.Run default: point i of e2_cost uses seed 12345 + i
SMALL = ("chain N=12 u1 J2=0.0", "chain N=14 u1 J2=0.0", "chain N=16 k0 J2=0.0", "chain N=18 k0 J2=0.0")


def label(pt) -> str:
    """The row label of experiments/e2_cost.py."""
    return (f"{pt['model']} N={pt.get('N', pt.get('Lx', 0) * pt.get('Ly', 0))} {pt['block']} J2={pt.get('J2', 0.0)}")


LABELS = [label(p) for p in E2_POINTS]


def measure(lab: str, repeat: int = 1, workdir: str = None, timeout: int = 900) -> dict:
    """One E2 row: the producer once, the certifier `repeat` times; the certifier run with the smallest total."""
    i = LABELS.index(lab)
    pt = E2_POINTS[i]
    wd = workdir or tempfile.mkdtemp(prefix="certify_ed_perf_")
    cfg = dict(pt, k=8, seed=RUN_SEED + i, npz=os.path.join(wd, f"point{i}.npz"))
    prod, _ = e2_cost.run_child(e2_cost.PRODUCER, cfg, 1800)
    if "error" in prod:
        return {"label": lab, "cfg": pt, "producer": prod}
    runs = []
    for _ in range(max(1, int(repeat))):
        cert, _ = e2_cost.run_child(e2_cost.CERTIFIER, cfg, timeout)
        runs.append(cert)
    try:
        os.remove(cfg["npz"])
    except OSError:
        pass
    ok = [r for r in runs if "error" not in r]
    best = min(ok, key=lambda r: r["total"]) if ok else runs[-1]
    return {"label": lab, "cfg": pt, "dim": prod["dim"], "nnz": prod["nnz"], "producer": prod, "certifier": best,
            "totals": [r.get("total") for r in runs], "maxrss_mb_all": [r.get("maxrss_mb") for r in runs]}


def machine() -> dict:
    import certify_ed
    from certify_ed.provenance import source_fingerprint
    try:
        src = source_fingerprint(ROOT)
    except Exception as e:                   # no git checkout
        src = {"error": f"{type(e).__name__}: {e}"}
    return {"date": datetime.date.today().isoformat(), "certify_ed": certify_ed.__version__, "source": src,
            "cpu_model": portable.cpu_model(), "cpu_count": os.cpu_count(), "memory_GiB": portable.memory_gib(),
            "platform": sys.platform, "threads": e2_cost.THREADS, "memory_cap_GiB": e2_cost.MEM_CAP_BYTES / 2 ** 30}


def from_run(run_dir: str, labs) -> dict:
    """The rows `labs` copied unchanged from the E2 record of an evidence run (its e2_cost_*/data.json), with the
    machine of its environment.json: a baseline that is the release's own measurement."""
    import glob
    found = sorted(glob.glob(os.path.join(run_dir, "e2_cost_*", "data.json")))
    if not found:
        raise SystemExit(f"no e2_cost_*/data.json under {run_dir}")
    with open(found[-1], encoding="utf-8") as f:
        rows = {r["label"]: r for r in json.load(f)["rows"]}
    env_path = os.path.join(os.path.dirname(found[-1]), "environment.json")
    env = json.load(open(env_path, encoding="utf-8")) if os.path.exists(env_path) else {}
    m = {k: env.get(k) for k in ("certify_ed", "git_commit", "source_hash", "cpu_model", "cpu_count", "memory_GiB",
                                 "platform", "started", "threads_env")}
    m["date"] = (env.get("started") or "")[:10]
    m["record"] = os.path.relpath(found[-1], run_dir).replace(os.sep, "/")
    points = {}
    for lab in labs:
        r = dict(rows[lab])
        r["dim"] = r.get("producer", {}).get("dim")
        r["nnz"] = r.get("producer", {}).get("nnz")
        points[lab] = r
    return {"schema": "certify-ed/perf-points/1.0", "machine": m, "repeat": 1, "source": "evidence run record",
            "points": points}


def main(argv=None):
    ap = argparse.ArgumentParser(description="measure E2 points as experiments/e2_cost.py does")
    ap.add_argument("--record", required=True, help="output JSON (for example tests/fixtures/perf_baseline.json)")
    ap.add_argument("--points", default="small", help="'small', 'all' or comma-separated row labels")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--from-run", help="copy the rows from this evidence-run folder instead of measuring")
    a = ap.parse_args(argv)
    labs = SMALL if a.points == "small" else (LABELS if a.points == "all" else [s.strip() for s in a.points.split(",")])
    if a.from_run:
        out = from_run(a.from_run, labs)
        with open(a.record, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print("wrote", a.record, "from", a.from_run)
        return
    out = {"schema": "certify-ed/perf-points/1.0", "machine": machine(), "repeat": a.repeat, "points": {}}
    for lab in labs:
        r = measure(lab, a.repeat)
        out["points"][lab] = r
        c = r.get("certifier", {})
        print(f"{lab}: total {c.get('total', float('nan')):.2f} s (runs {r.get('totals')}), peak {c.get('maxrss_mb', float('nan')):.0f} MB",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
    with open(a.record, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", a.record)


if __name__ == "__main__":
    main()
