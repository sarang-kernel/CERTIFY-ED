"""Shared conventions of the figure scripts (experiments/figures/*.py).

Every script draws one figure and writes it as PDF and PNG into the output directory (default
docs/figures/). The scripts read records and documents; they never compute a certified number
(experiments/figures is listed in provenance.NON_COMPUTATIONAL).

Locating the data
-----------------
  records of the evidence run        $CERTIFY_RESULTS, else experiments/results   (run_full_20261022 layout:
                                     <experiment>_<timestamp>/data.json; the compare CSV at the run root)
  records of the second-stack run    $CERTIFY_RESULTS_B (the Linux 3.0.0 run, run_full_20261021)
  documents in the repository        corpus/results/certify-ed/report.json, experiments/survey/survey_for_scope.json,
                                     RESULT_integration_3.2.md, experiments/figures/data/*.csv

`require(path)` exits with code 3 ("data absent") when a file a figure needs is missing; the smoke test
(tests/test_figures.py) treats that exit as a skip. Every number that a caption prints is written by the
script to <figure>.numbers.json next to the figure, so that INDEX.md can be checked against the data.
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_ABSENT = 3

# The three font sizes (titles and axis labels / legend and annotations / ticks), a palette that never
# opposes red and green, and one alarm hue reserved for failures.
SIZES = (8, 7, 6)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, TEAL, PURPLE, GREY = "#2a78d6", "#eb6834", "#1b9e8a", "#7b5ea7", "#9a9994"
STATUS_COLORS = {"CERTIFIED": "#2a78d6", "INCONCLUSIVE": "#e0a21b", "NOT_CHECKED": "#b8b7b3",
                 "FAILED": "#c8372d", "not run": "#ffffff"}
STATUS_ORDER = ("CERTIFIED", "INCONCLUSIVE", "NOT_CHECKED", "FAILED")


def style():
    """Publication rcParams: role-mapped font ladder, open frame, outward ticks, Type-42 fonts, 300 dpi."""
    import matplotlib as mpl
    mpl.use("Agg")
    base, secondary, tick = SIZES
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelweight": "normal",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "mathtext.fontset": "dejavusans",
    })
    return mpl


def panel_letter(ax, letter, dx=-0.12, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=SIZES[0] + 1,
            va="bottom", ha="left")


def parse_args(description, needs=()):
    """Common command line: --out DIR, --results DIR, --results-b DIR.  `needs` lists the record prefixes
    (e.g. 'e1_failure_matrix') the figure reads from --results; a missing one exits with DATA_ABSENT."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"), help="output directory")
    p.add_argument("--results", default=os.environ.get("CERTIFY_RESULTS", os.path.join(ROOT, "experiments", "results")),
                   help="directory of the evidence run (records <experiment>_<timestamp>/data.json)")
    p.add_argument("--results-b", default=os.environ.get("CERTIFY_RESULTS_B"),
                   help="directory of the second-stack run (fig_cost_memory)")
    p.add_argument("--png-only", action="store_true", help="skip the PDF (quicker smoke test)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for prefix in needs:
        if latest_record(a.results, prefix) is None:
            absent(f"no record {prefix}_* under {a.results}")
    return a


def absent(msg):
    print(f"data absent: {msg}", file=sys.stderr)
    sys.exit(DATA_ABSENT)


def require(path):
    if path is None or not os.path.exists(path):
        absent(str(path))
    return path


def latest_record(results, prefix):
    """Most recent results/<prefix>_<timestamp>/data.json, or None."""
    if not results:
        return None
    c = sorted(d for d in glob.glob(os.path.join(results, prefix + "_2*")) if os.path.isdir(d))
    if not c or not os.path.exists(os.path.join(c[-1], "data.json")):
        return None
    return os.path.join(c[-1], "data.json")


def load_record(results, prefix):
    """(data, record path relative to the run directory)."""
    path = require(latest_record(results, prefix))
    with open(path, encoding="utf-8") as f:
        return json.load(f), os.path.relpath(path, results).replace(os.sep, "/")


def save(fig, out_dir, name, png_only=False):
    """Write <name>.png (and .pdf) and return the written paths."""
    paths = []
    png = os.path.join(out_dir, name + ".png")
    fig.savefig(png)
    paths.append(png)
    if not png_only:
        pdf = os.path.join(out_dir, name + ".pdf")
        fig.savefig(pdf, metadata={"CreationDate": None})   # no timestamp: identical input, identical file
        paths.append(pdf)
    return paths


def write_numbers(out_dir, name, numbers):
    """Record the numbers a caption prints (<name>.numbers.json), so the caption can be checked against them."""
    with open(os.path.join(out_dir, name + ".numbers.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(numbers, f, indent=1, sort_keys=True)
        f.write("\n")


def check_layout(fig):
    """Geometric self-check: no two visible text boxes overlap, every text box lies inside the figure.
    Returns a list of findings (empty when clean)."""
    import matplotlib as mpl
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t.get_window_extent(r).width > 0]
    findings = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                findings.append(f"text overlap: {a.get_text()[:30]!r} / {b.get_text()[:30]!r}")
    fb = fig.bbox
    for t, bt in texts:
        if bt.x0 < fb.x0 - 1 or bt.y0 < fb.y0 - 1 or bt.x1 > fb.x1 + 1 or bt.y1 > fb.y1 + 1:
            findings.append(f"text outside figure: {t.get_text()[:30]!r}")
    return findings


def wilson(k, n, z=1.959963984540054):
    """Wilson 95 % interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))
