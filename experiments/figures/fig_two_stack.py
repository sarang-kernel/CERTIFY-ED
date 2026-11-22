"""E3 fig_two_stack: record-by-record comparison classes of two runs, per experiment.

Reads compare_vs_<other run>.csv at the root of the evidence run (written by experiments/compare_runs.py);
with --results-b the CSV named after that run is preferred, else the first in sorted order.
Each leaf of each experiment's data.json is in exactly one class: same; float-differs (relative difference
<= 1e-9); stack-dependent (a named floating-point statistic or a limit within 10 %); fp-statistic
(a statistic of floating-point residuals, compared by value only); text-numbers (the same sentence with
run-stack digits); extended (the newer version says more); sample-dependent (E5 draws, E6 mutants);
explained (a difference explained once, by path pattern, with its reason in the CSV); timing; missing;
MISMATCH. The figure counts leaves per class and experiment; the title states the MISMATCH count.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figcommon as C  # noqa: E402

NAME = "fig_two_stack"
CLASSES = ["same", "float-differs", "stack-dependent", "fp-statistic", "text-numbers", "extended", "sample-dependent",
           "explained", "timing", "missing", "MISMATCH"]
LABELS = {"same": "same", "float-differs": "float differs, rel ≤ 1e-9", "stack-dependent": "stack-dependent (≤ 10 %)",
          "fp-statistic": "FP statistic", "text-numbers": "same text, run-stack digits", "extended": "extended (newer says more)",
          "sample-dependent": "sample-dependent (random draws)", "explained": "explained (reason in CSV)",
          "timing": "timing / memory", "missing": "missing on one side", "MISMATCH": "MISMATCH"}
COLORS = {"same": "#2a78d6", "float-differs": "#7fb0e8", "stack-dependent": "#bcd3ef", "fp-statistic": "#1b9e8a",
          "text-numbers": "#8fd1c6", "extended": "#7b5ea7", "sample-dependent": "#c9c7c2", "explained": "#e0a21b",
          "timing": "#f1e3b4", "missing": "#8a8984", "MISMATCH": "#c8372d"}


def read(csv_path):
    counts = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts.setdefault(r["experiment"], {}).setdefault(r["class"], 0)
            counts[r["experiment"]][r["class"]] += 1
    return counts


def draw(counts, title_runs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    C.style()
    exps = sorted(counts, key=lambda e: int(e[1:]))
    fig, ax = plt.subplots(figsize=(7.1, 3.3), gridspec_kw=dict(left=0.07, right=0.72, bottom=0.14, top=0.88))
    total = sum(sum(v.values()) for v in counts.values())
    n_mis = sum(v.get("MISMATCH", 0) for v in counts.values())
    for i, e in enumerate(exps):
        left = 0
        for cls in CLASSES:
            n = counts[e].get(cls, 0)
            if n == 0:
                continue
            ax.barh(i, n, left=left, color=COLORS[cls], edgecolor="white", lw=0.4, height=0.72)
            if n >= 60:
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=C.SIZES[2],
                        color="white" if cls in ("same", "fp-statistic", "extended", "missing") else C.INK)
            left += n
        ax.text(left + 8, i, str(left), va="center", fontsize=C.SIZES[2], color=C.INK2)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels([e.upper() for e in exps])
    ax.invert_yaxis()
    ax.set_xlabel("leaves of data.json compared")
    ax.set_title(f"{total} leaves in 11 experiments, {n_mis} MISMATCH  ({title_runs})")
    ax.margins(x=0.1)
    handles = [Patch(fc=COLORS[c], ec="none", label=LABELS[c]) for c in CLASSES
               if any(v.get(c, 0) for v in counts.values()) or c == "MISMATCH"]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=C.SIZES[2], handlelength=1.2,
              handletextpad=0.5, borderaxespad=0)
    per_class = {c: sum(v.get(c, 0) for v in counts.values()) for c in CLASSES}
    numbers = {"leaves": total, "experiments": len(exps), "MISMATCH": n_mis, "per_class": per_class,
               "per_experiment": {e: sum(counts[e].values()) for e in exps}}
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    files = sorted(glob.glob(os.path.join(a.results, "compare_vs_*.csv")))
    if a.results_b:  # prefer the comparison against the run named by --results-b / CERTIFY_RESULTS_B
        pref = os.path.join(a.results, "compare_vs_" + os.path.basename(os.path.normpath(a.results_b)) + ".csv")
        if pref in files:
            files = [f for f in files if f != pref] + [pref]
    if not files:
        C.absent(f"compare_vs_*.csv under {a.results}")
    csv_path = files[-1]
    other = os.path.basename(csv_path)[len("compare_vs_"):-4]
    counts = read(csv_path)
    fig, numbers = draw(counts, f"{os.path.basename(os.path.normpath(a.results))} vs {other}")
    numbers["csv"] = os.path.basename(csv_path)
    numbers["run_A"], numbers["run_B"] = os.path.basename(os.path.normpath(a.results)), other
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
