"""Benchmark table (groundwork for the benchmark page, roadmap G-10): certify() time and peak memory per E2 point,
a measurement against the E2 record of an evidence run.

    python experiments/bench_table.py --run results/run_full_20261105 --current tests/fixtures/perf_baseline.json
    python experiments/bench_table.py --run results/run_full_20261105 --measure small [--repeat 3]

--run: an evidence-run folder (its e2_cost_*/data.json is read) or an e2_cost data.json. --current: a file written by
experiments/perf_points.py, or another e2_cost data.json; --measure: measure those points now (perf_points.measure,
the methodology of experiments/e2_cost.py). Writes a Markdown table to stdout (and --md / --csv / --json files):
total certify() seconds, the S3 and S8 seconds, peak resident MB, the ratios current/run, and whether every
statement status is the same. A row whose certifier did not finish in the run (the experiment's 900 s limit) shows
the run's time as '>900' and its measurement without S3.
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STATEMENTS = ("S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces", "S8.degeneracy")


def load_rows(path: str) -> dict:
    """label -> row with 'certifier' (and maybe 'certifier_without_S3'), from a run folder, an e2_cost data.json or
    a perf_points file."""
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "e2_cost_*", "data.json")))
        if not found:
            raise SystemExit(f"no e2_cost_*/data.json under {path}")
        path = found[-1]
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "points" in d:
        return {k: v for k, v in d["points"].items()}
    return {r["label"]: r for r in d["rows"]}


def _cert(row):
    c = (row or {}).get("certifier") or {}
    return c if "error" not in c else None


def summarize(row) -> dict:
    """total, S3, S8 seconds, peak MB, statuses, and a note for a certifier that did not finish."""
    if row is None:
        return {"note": "absent"}
    c = _cert(row)
    if c is None:
        w = row.get("certifier_without_S3") or {}
        err = ((row.get("certifier") or {}).get("error") or "error")[:40]
        return {"total": None, "note": err, "without_S3_total": w.get("total"), "peak": w.get("maxrss_mb"),
                "statuses": {k: v["status"] for k, v in (w.get("statements") or {}).items()}}
    st = c.get("statements") or {}
    return {"total": c.get("total"), "S3": (st.get("S3.completeness") or {}).get("seconds"),
            "S8": (st.get("S8.degeneracy") or {}).get("seconds"), "peak": c.get("maxrss_mb"),
            "statuses": {k: v["status"] for k, v in st.items()}, "note": ""}


def fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def ratio(a, b):
    return None if (a is None or b in (None, 0)) else a / b


def table(run: dict, cur: dict, labels):
    out = []
    for lab in labels:
        r, c = summarize(run.get(lab)), summarize(cur.get(lab))
        same = (r.get("statuses") == c.get("statuses")) if (r.get("statuses") and c.get("statuses")) else None
        dim = (cur.get(lab) or {}).get("dim") or ((run.get(lab) or {}).get("producer") or {}).get("dim")
        out.append({"label": lab, "dim": dim,
                    "run_total_s": r.get("total"), "cur_total_s": c.get("total"),
                    "time_ratio": ratio(c.get("total"), r.get("total")),
                    "run_S3_s": r.get("S3"), "cur_S3_s": c.get("S3"), "run_S8_s": r.get("S8"), "cur_S8_s": c.get("S8"),
                    "run_peak_MB": r.get("peak"), "cur_peak_MB": c.get("peak"),
                    "memory_ratio": ratio(c.get("peak"), r.get("peak")),
                    "statuses_same": same, "run_note": r.get("note", ""), "cur_note": c.get("note", "")})
    return out


def markdown(rows, run_name, cur_name) -> str:
    head = (f"| E2 point | dim | total s, {run_name} | total s, {cur_name} | ratio | S3 s | S8 s | peak MB, {run_name} | "
            f"peak MB, {cur_name} | ratio | statuses same |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rt = fmt(r["run_total_s"], 2) if r["run_total_s"] is not None else (">900" if "timeout" in r["run_note"] else r["run_note"])
        ct = fmt(r["cur_total_s"], 2) if r["cur_total_s"] is not None else r["cur_note"]
        s3 = f"{fmt(r['run_S3_s'])} / {fmt(r['cur_S3_s'])}"
        s8 = f"{fmt(r['run_S8_s'])} / {fmt(r['cur_S8_s'])}"
        same = {True: "yes", False: "NO", None: ""}[r["statuses_same"]]
        lines.append(f"| {r['label']} | {r['dim'] or ''} | {rt} | {ct} | {fmt(r['time_ratio'], 2)} | {s3} | {s8} | "
                     f"{fmt(r['run_peak_MB'], 0)} | {fmt(r['cur_peak_MB'], 0)} | {fmt(r['memory_ratio'], 2)} | {same} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--current")
    ap.add_argument("--measure", help="'small', 'all' or comma-separated labels: measure now")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--current-name", default="now")
    ap.add_argument("--md")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    run = load_rows(a.run)
    if a.current:
        cur = load_rows(a.current)
        labels = [k for k in run if k in cur] or list(cur)
    elif a.measure:
        import perf_points
        labels = (list(perf_points.SMALL) if a.measure == "small" else
                  (list(perf_points.LABELS) if a.measure == "all" else [s.strip() for s in a.measure.split(",")]))
        cur = {lab: perf_points.measure(lab, a.repeat) for lab in labels}
    else:
        raise SystemExit("give --current FILE or --measure POINTS")
    rows = table(run, cur, labels)
    md = markdown(rows, a.run_name, a.current_name)
    print(md)
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md + "\n")
    if a.csv:
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"run": a.run, "current": a.current or a.measure, "rows": rows}, f, indent=1)
    return rows


if __name__ == "__main__":
    main()
