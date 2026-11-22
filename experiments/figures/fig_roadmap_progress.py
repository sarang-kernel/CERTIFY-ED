"""E7 fig_roadmap_progress: the 78 roadmap items by wave and theme, with status and points.

Reads experiments/figures/data/ROADMAP_expansion_items_2026-11-03.csv (see data/README.md: the roadmap
items with their wave). (a) points (score = adoption + rigor) per wave, stacked by status on 2026-11-03;
(b) item counts per theme, stacked by status. Status words are those of the CSV: done, partial, blocked,
not started.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figcommon as C  # noqa: E402

NAME = "fig_roadmap_progress"
DATA = os.path.join(C.HERE, "data", "ROADMAP_expansion_items_2026-11-03.csv")
STATUS = ["done", "partial", "blocked", "not started"]
COLORS = {"done": C.BLUE, "partial": "#7fb0e8", "blocked": C.ORANGE, "not started": "#e4e3df"}
WAVE_TITLE = {"0": "0  ship what exists", "1": "1  cover what labs report", "2": "2  trust and scale",
              "3": "3  new statements (D-12)", "4": "4  the field"}


def load():
    with open(DATA, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def draw(rows):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    C.style()
    status_col = sorted(c for c in rows[0] if c.startswith("status_2"))[-1]   # the latest dated status column
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.9,
                                                                              left=0.19, right=0.99, bottom=0.16, top=0.86))
    waves = sorted({r["wave"] for r in rows})
    pts = {w: {s: sum(int(r["score"]) for r in rows if r["wave"] == w and r[status_col] == s) for s in STATUS} for w in waves}
    cnt = {w: {s: sum(1 for r in rows if r["wave"] == w and r[status_col] == s) for s in STATUS} for w in waves}
    for i, w in enumerate(waves):
        left = 0
        for s in STATUS:
            v = pts[w][s]
            if v:
                axa.barh(i, v, left=left, color=COLORS[s], edgecolor="white", lw=0.4, height=0.7)
                if v >= 12:
                    axa.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=C.SIZES[2],
                             color="white" if s in ("done", "blocked") else C.INK)
                left += v
        n_items = sum(cnt[w].values())
        axa.text(left + 2, i, f"{left} pts, {n_items} items", va="center", fontsize=C.SIZES[2], color=C.INK2)
    axa.set_yticks(range(len(waves)))
    axa.set_yticklabels([WAVE_TITLE.get(w, w) for w in waves])
    axa.invert_yaxis()
    axa.set_xlabel("points (adoption + rigor)")
    axa.margins(x=0.3)
    done_pts = sum(pts[w]["done"] for w in waves)
    all_pts = sum(sum(pts[w].values()) for w in waves)
    axa.set_title(f"{done_pts} of {all_pts} points done, by wave", loc="left")
    C.panel_letter(axa, "a", dx=-0.62)
    themes = sorted({r["theme"] for r in rows}, key=lambda t: -sum(1 for r in rows if r["theme"] == t))
    tc = {t: {s: sum(1 for r in rows if r["theme"] == t and r[status_col] == s) for s in STATUS} for t in themes}
    for i, t in enumerate(themes):
        left = 0
        for s in STATUS:
            v = tc[t][s]
            if v:
                axb.barh(i, v, left=left, color=COLORS[s], edgecolor="white", lw=0.4, height=0.7)
                if v >= 2:
                    axb.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=C.SIZES[2],
                             color="white" if s in ("done", "blocked") else C.INK)
                left += v
    axb.set_yticks(range(len(themes)))
    axb.set_yticklabels(themes)
    axb.invert_yaxis()
    axb.set_xlabel("items")
    axb.margins(x=0.1)
    n_done = sum(1 for r in rows if r[status_col] == "done")
    axb.set_title(f"{n_done} of {len(rows)} items done, by theme", loc="left")
    C.panel_letter(axb, "b", dx=-0.5)
    handles = [Patch(fc=COLORS[s], ec="none" if s != "not started" else C.INK2, lw=0.4, label=s) for s in STATUS]
    axb.legend(handles=handles, loc="lower right", fontsize=C.SIZES[2], handlelength=1.2)
    numbers = {"items": len(rows), "status_column": status_col,
               "status_counts": {s: sum(1 for r in rows if r[status_col] == s) for s in STATUS},
               "points_total": all_pts, "points_done": done_pts,
               "points_per_wave": {w: sum(pts[w].values()) for w in waves},
               "done_points_per_wave": {w: pts[w]["done"] for w in waves},
               "items_per_wave": {w: sum(cnt[w].values()) for w in waves},
               "partial_ids": [r["id"] for r in rows if r[status_col] == "partial"],
               "blocked_ids": [r["id"] for r in rows if r[status_col] == "blocked"]}
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0])
    C.require(DATA)
    fig, numbers = draw(load())
    numbers["source"] = os.path.relpath(DATA, C.ROOT).replace(os.sep, "/")
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
