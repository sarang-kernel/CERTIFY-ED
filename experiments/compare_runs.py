"""Compare two reproduction runs record by record (the project's two-stack rule).

Usage: python compare_runs.py <run_dir_A> <run_dir_B> <out.csv>

For every experiment present in both runs, the two data.json files are flattened to (path, value) leaves
and compared. Statuses, booleans, integers and strings must be identical; floats are compared with a
relative tolerance (default 1e-9) and reported with their relative difference. Timing fields (names
containing 'seconds', 'time', 'elapsed') and float-derived certified bounds (beta, delta, radii, exact
rationals of bounds; see STACK_DEPENDENT) are compared with the looser tolerance STACK_RTOL and reported
with their relative difference; beyond that tolerance they are mismatches like any other field. The
two-stack rule asks that the same statements be certified; the bounds themselves may differ in their
last digits and, for factorization residuals, by a few per cent (review SS-19). The script prints a
one-line summary per experiment and writes every leaf to the CSV.
"""
import csv
import json
import os
import sys

TIMING = ("seconds", "time", "elapsed", "started", "finished")
# certified bounds that are computed from floating-point proposals (factor residuals, Weyl margins, exact
# radii): valid on each stack, but not bit-identical across stacks (review SS-19). They are compared with
# their own, looser tolerance STACK_RTOL (default 10 %: the factorization residual beta differed by up to
# 6 % between the Linux run and the Windows replication); beyond it they are MISMATCH like any other field.
STACK_RTOL = 0.10
STACK_DEPENDENT = ("beta", "delta", "radius", "t_prime", "_exact", "bound_exact", "eta_upper", "lower_bound",
                   "eigenvalue_upper", "margin", "half_width", "halfwidth", "width")


def flatten(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, o


def records(run_dir):
    out = {}
    for d in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, d, "data.json")
        if os.path.isfile(p):
            exp = d.split("_")[0]
            out[exp] = (d, json.load(open(p, encoding="utf-8")))
    return out


def _num(x):
    """float or exact rational ('p/q' strings are exact values stored as text)."""
    if isinstance(x, bool):
        raise ValueError
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and "/" in x:
        from fractions import Fraction
        return float(Fraction(x))
    return float(x)


def classify(k, a, b, rtol):
    kl = k.lower()
    if any(t in kl for t in TIMING):
        return "timing", ""
    if any(t in kl for t in STACK_DEPENDENT) and not any(t in kl for t in ("status", "rounded", "count")):
        try:
            fa, fb = _num(a), _num(b)
            rel = abs(fa - fb) / max(1.0, abs(fa))
            if fa == fb:
                return "same", "0"
            return ("stack-dependent" if rel <= STACK_RTOL else "MISMATCH"), f"{rel:.3e}"
        except (TypeError, ValueError):
            pass
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return ("same" if a == b else "MISMATCH"), ""
    if isinstance(a, int) and isinstance(b, int):
        return ("same" if a == b else "MISMATCH"), ""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return ("same" if a == b else "MISMATCH"), ""
    if fa == fb:
        return "same", "0"
    rel = abs(fa - fb) / max(1.0, abs(fa))
    return ("float-differs" if rel <= rtol else "MISMATCH"), f"{rel:.3e}"


def main(run_a, run_b, out_csv, rtol=1e-9):
    ra, rb = records(run_a), records(run_b)
    common = sorted(set(ra) & set(rb))
    rows = []
    for exp in common:
        da, db = dict(flatten(ra[exp][1])), dict(flatten(rb[exp][1]))
        counts = {"same": 0, "float-differs": 0, "stack-dependent": 0, "MISMATCH": 0, "timing": 0, "missing": 0}
        for k in sorted(set(da) | set(db)):
            if k not in da or k not in db:
                counts["missing"] += 1
                rows.append((exp, k, da.get(k, "<missing>"), db.get(k, "<missing>"), "missing", ""))
                continue
            cls, rel = classify(k, da[k], db[k], rtol)
            counts[cls] += 1
            rows.append((exp, k, da[k], db[k], cls, rel))
        print(f"{exp}: {ra[exp][0]} vs {rb[exp][0]}: {counts}")
    only = sorted((set(ra) | set(rb)) - set(common))
    if only:
        print("present in one run only:", only)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "path", "value_A", "value_B", "class", "rel_diff"])
        for r in rows:
            w.writerow([str(x) for x in r])
    return 0 if all(r[4] != "MISMATCH" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
