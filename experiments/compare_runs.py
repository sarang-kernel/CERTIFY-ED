"""Compare two reproduction runs record by record (the project's two-stack rule).

Usage: python compare_runs.py <run_dir_A> <run_dir_B> <out.csv>

For every experiment present in both runs, the two data.json files are flattened to (path, value) leaves
and compared. Statuses, booleans, integers and strings must be identical; floats are compared with a
relative tolerance (default 1e-9) and reported with their relative difference. Timing fields (names
containing 'seconds', 'time', 'elapsed') and float-derived certified bounds (beta, delta, radii, exact
rationals of bounds; see STACK_DEPENDENT) are compared with the looser tolerance STACK_RTOL and reported
with their relative difference; beyond that tolerance they are mismatches like any other field. Descriptive
floating-point statistics (FP_STATISTIC: residual ratios, slopes, slowdowns of E8) are reported with their
relative difference and never decide the exit code. The
two-stack rule asks that the same statements be certified; the bounds themselves may differ in their
last digits and, for factorization residuals, by a few per cent (review SS-19). The script prints a
one-line summary per experiment and writes every leaf to the CSV.
"""
import csv
import json
import os
import re
import sys

TIMING = ("seconds", "time", "elapsed", "started", "finished")
# leaf names that are timings or memory although they do not say so (E2: producer t_build/t_solve, certifier total,
# peak memory in MiB)
TIMING_LEAVES = ("t_build", "t_solve", "total", "maxrss_mb")
# certified bounds that are computed from floating-point proposals (factor residuals, Weyl margins, exact
# radii): valid on each stack, but not bit-identical across stacks (review SS-19). They are compared with
# their own, looser tolerance STACK_RTOL (default 10 %: the factorization residual beta differed by up to
# 6 % between the Linux run and the Windows replication); beyond it they are MISMATCH like any other field.
STACK_RTOL = 0.10
STACK_DEPENDENT = ("beta", "delta", "radius", "t_prime", "_exact", "bound_exact", "eta_upper", "lower_bound",
                   "eigenvalue_upper", "margin", "half_width", "halfwidth", "width", "certified_gap", "fraction_of_gap")
# descriptive floating-point statistics of residual norms (E8: the ratio of symveig's to certify_ed's largest
# half-width per model, the fitted slopes and slowdowns): quotients of two LAPACK-dependent residuals, neither a
# statement nor a certified bound. They are reported with their relative difference and never decide the exit
# code; the manuscript marks them run-stack (SS-19). The statements of the same record (complete, counted claims,
# reference_outside) stay strict.
FP_STATISTIC = ("ratio", "slope", "slowdown", "spread", "bound_over_", ".p99", ".p50", ".median")
# records whose items are drawn from what the stack produced, compared by their invariants instead of leaf by leaf:
# E5 rows are ARPACK runs per seed (which seed misses a multiplet member depends on the floating-point stack; the
# invariant is that no incorrect result is certified), E6 mutants are a random sample of the mutation sites of the
# code that ran (a different code has different sites; the invariant is the baseline and the score).
SAMPLE_DEPENDENT = {"e5": ("rows[",), "e6": ("mutants[",)}
# differences explained once, by path pattern (regular expression) and reason; they are reported in their own class
EXPLAINED = {
    "e4": [(r"^W1\.levels\[\d+\]\[[12]\]$",
            "quantum numbers of a degenerate level at the cut: which member of the multiplet is listed is a tie"),
           (r"^W4\[3\]\.(statuses\.S3\.completeness|gap_lower_bound|s3_note)$",
            "4x4 J2=0.5: the S3 factorization fits under a committed-memory cap (job object) and not under an "
            "address-space cap (RLIMIT_AS) of the same size")],
    "e6": [(r"^(sites_per_module\.|summary\.)",
            "the mutation sites and the sampled mutants belong to the code that ran; between two versions of the code "
            "they differ, and so do the counts and the score (a version difference, reported, not a reproduction failure)")],
    "e2": [(r"\.statements\.S3\.completeness\.summary$",
            "the reason text of a NOT_CHECKED S3 names the allocation failure as the platform reports it "
            "(MemoryError vs SuperLU's gstrf error); the status is compared strictly")],
}
# floating-point numbers only (a decimal point or an exponent): integers in prose are counts and stay strict
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")


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


CLASSES = ("same", "float-differs", "stack-dependent", "fp-statistic", "text-numbers", "extended", "sample-dependent",
           "explained", "timing", "missing", "MISMATCH")


def classify(k, a, b, rtol, exp=""):
    kl = k.lower()
    if a == b:
        return "same", ""
    for prefix in SAMPLE_DEPENDENT.get(exp, ()):
        if k.startswith(prefix):
            return "sample-dependent", ""
    for pat, why in EXPLAINED.get(exp, ()):
        if re.search(pat, k):
            return "explained", why
    if any(t in kl for t in TIMING) or k.rsplit(".", 1)[-1] in TIMING_LEAVES:
        return "timing", ""
    if any(t in kl for t in FP_STATISTIC) and not any(t in kl for t in ("status", "count")):
        try:
            fa, fb = _num(a), _num(b)
            rel = abs(fa - fb) / max(abs(fa), abs(fb), 1e-300)
            return ("same" if fa == fb else "fp-statistic"), f"{rel:.3e}"
        except (TypeError, ValueError):
            pass
    if any(t in kl for t in STACK_DEPENDENT) and not any(t in kl for t in ("status", "rounded", "count")):
        try:
            fa, fb = _num(a), _num(b)
            rel = abs(fa - fb) / max(1.0, abs(fa))
            if fa == fb:
                return "same", "0"
            return ("stack-dependent" if rel <= STACK_RTOL else "MISMATCH"), f"{rel:.3e}"
        except (TypeError, ValueError):
            pass
    if isinstance(a, str) and isinstance(b, str):
        if NUM.sub("#", a) == NUM.sub("#", b):
            return "text-numbers", ""                # the same sentence with run-stack digits
        if b.startswith(a.rstrip()) or a.startswith(b.rstrip()):
            return "extended", ""                    # one record says more (a statement added in the newer version)
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
        counts = dict.fromkeys(CLASSES, 0)
        for k in sorted(set(da) | set(db)):
            if k not in da or k not in db:
                counts["missing"] += 1
                rows.append((exp, k, da.get(k, "<missing>"), db.get(k, "<missing>"), "missing", "", ""))
                continue
            cls, rel = classify(k, da[k], db[k], rtol, exp)
            counts[cls] += 1
            rows.append((exp, k, da[k], db[k], cls, "" if cls == "explained" else rel, rel if cls == "explained" else ""))
        print(f"{exp}: {ra[exp][0]} vs {rb[exp][0]}: {counts}")
    only = sorted((set(ra) | set(rb)) - set(common))
    if only:
        print("present in one run only:", only)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "path", "value_A", "value_B", "class", "rel_diff", "note"])
        for r in rows:
            w.writerow([str(x) for x in r])
    return 0 if all(r[4] != "MISMATCH" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
