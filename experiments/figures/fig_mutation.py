"""E4 fig_mutation: mutation-testing outcomes of the evidence run (--results) and the latest dev/3.2 sample.

(a) e6_mutation_*/data.json of the evidence run (64 sampled mutants): killed and survived per module;
survivors are split by the category assigned in classification.json (experiments/e6_classify.py's rules,
the "needs inspection" class awaiting a human verdict). (b) the most recent 16-mutant sample recorded in
RESULT_integration_3.2.md (the last section whose heading says "sample"; the 3.2.0 release sample): killed count
and the surviving mutants named there. The title of (a) names the run's certify-ed version from its manifest. The two samples are drawn from different code (different mutation
sites) and different sizes; the figure places them side by side, it does not compare the scores.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

NAME = "fig_mutation"
CAT_SHORT = {
    "needs inspection": "survived: needs inspection",
    "heuristic or performance parameter (shift placement, method choice, caching)": "survived: heuristic / performance parameter",
    "rounding allowance in a worst-case bound (looser: sound; tighter: below test resolution)": "survived: rounding allowance",
    "no effect on statements (timing, messages)": "survived: no effect on statements",
    "strict vs non-strict comparison (differs only at exact equality)": "survived: strict vs non-strict comparison",
}
CAT_COLORS = ["#c8372d", "#e0a21b", "#7b5ea7", "#b8b7b3", "#8fd1c6"]


def parse_integration(md_path):
    """(killed, n, [mutant labels], heading) of the LAST mutation sample recorded in RESULT_integration_3.2.md.

    The file accumulates one section per sample (the 2026-10-29 integration sample, the 3.2.0 release sample);
    the figure shows the most recent one."""
    text = open(md_path, encoding="utf-8").read()
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"^## (.*(?:[Ss]ample).*)$", text, flags=re.M)]
    start, heading = heads[-1] if heads else (0, "dev/3.2 sample")
    sec = text[start:]
    nxt = re.search(r"^## ", sec[3:], flags=re.M)
    if nxt:
        sec = sec[:nxt.start() + 3]
    m = re.search(r"\*\*(\d+) of (\d+) killed\*\*", sec) or re.search(r"(\d+) of (\d+) killed", sec)
    killed, n = int(m.group(1)), int(m.group(2))
    labels = re.findall(r"^\| `([a-z_]+\.py:\d+)`", sec, flags=re.M)
    short = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$", "", heading)
    return killed, n, labels, short


def run_version(results):
    """certify-ed version of the run from its manifest (environment.certify_ed), else the directory name."""
    try:
        mf = json.load(open(os.path.join(results, "manifest.json"), encoding="utf-8"))
        env = mf.get("environment") or {}
        for k in ("certify_ed", "certify-ed", "version"):
            if env.get(k):
                return str(env[k])
    except (OSError, ValueError):
        pass
    return os.path.basename(os.path.normpath(results))


def draw(data, classification, integ, version):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    C.style()
    mutants = data["mutants"]
    cats = list(CAT_SHORT)
    surv_cat = {(s["module"], s["line"]): s["category"] for s in classification["survivors"]}
    per_mod = {}
    for m in mutants:
        d = per_mod.setdefault(m["module"], {"killed": 0, **{c: 0 for c in cats}})
        if m["status"] == "killed":
            d["killed"] += 1
        else:
            d[surv_cat.get((m["module"], m["line"]), "needs inspection")] += 1
    mods = sorted(per_mod, key=lambda k: -sum(per_mod[k].values()))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 3.4), gridspec_kw=dict(width_ratios=[3.2, 1], wspace=0.5,
                                                                              left=0.13, right=0.98, bottom=0.14, top=0.86))
    for i, mod in enumerate(mods):
        d = per_mod[mod]
        left = 0
        axa.barh(i, d["killed"], color=C.BLUE, height=0.7, edgecolor="white", lw=0.4)
        left += d["killed"]
        for c, col in zip(cats, CAT_COLORS):
            if d[c]:
                axa.barh(i, d[c], left=left, color=col, height=0.7, edgecolor="white", lw=0.4)
                left += d[c]
        axa.text(left + 0.15, i, str(left), va="center", fontsize=C.SIZES[2], color=C.INK2)
    axa.set_yticks(range(len(mods)))
    axa.set_yticklabels(mods)
    axa.invert_yaxis()
    axa.set_xlabel("sampled mutants")
    s = data["summary"]
    axa.set_title(f"{version} evidence run: {s['killed']} of {s['n']} mutants killed, {s['survived']} survived")
    axa.margins(x=0.08)
    handles = [Patch(fc=C.BLUE, label="killed (a test failed)")] + [Patch(fc=col, label=CAT_SHORT[c])
                                                                      for c, col in zip(cats, CAT_COLORS)]
    axa.legend(handles=handles, loc="lower right", fontsize=C.SIZES[2], handlelength=1.2)
    C.panel_letter(axa, "a", dx=-0.3)
    # (b) the integration sample
    killed, n, labels, heading = integ
    axb.bar(0, killed, color=C.BLUE, width=0.6)
    axb.bar(0, n - killed, bottom=killed, color="#c8372d", width=0.6)
    axb.text(0, killed / 2, f"{killed}\nkilled", ha="center", va="center", color="white", fontsize=C.SIZES[2])
    axb.text(0, killed + (n - killed) / 2, f"{n - killed}", ha="center", va="center", color="white", fontsize=C.SIZES[2])
    for j, lab in enumerate(labels):
        axb.text(0.36, n - 0.5 - j * 1.1, lab, ha="left", va="center", fontsize=C.SIZES[2], color=C.INK2)
    axb.set_xlim(-0.45, 1.6)
    axb.set_xticks([])
    axb.set_ylabel("mutants")
    axb.set_title(f"{heading}: {n} mutants")
    axb.spines["bottom"].set_visible(False)
    C.panel_letter(axb, "b", dx=-0.45)
    numbers = {"evidence_run": {"version": version, "n": s["n"], "killed": s["killed"], "survived": s["survived"], "timeouts": s["timeouts"],
                             "score": s["score"], "modules": len(mods),
                             "survivor_categories": classification["survivor_categories"],
                             "baseline_seconds": data["baseline"]["seconds"]},
               "dev_3_2_sample": {"n": n, "killed": killed, "survivors": labels}}
    return fig, numbers


def main():
    a = C.parse_args(__doc__.splitlines()[0], needs=("e6_mutation",))
    data, rec = C.load_record(a.results, "e6_mutation")
    cls_path = C.require(os.path.join(os.path.dirname(os.path.join(a.results, rec)), "classification.json"))
    classification = json.load(open(cls_path, encoding="utf-8"))
    md = C.require(os.path.join(C.ROOT, "RESULT_integration_3.2.md"))
    integ = parse_integration(md)
    fig, numbers = draw(data, classification, integ, run_version(a.results))
    numbers["record"] = rec
    numbers["classification"] = os.path.relpath(cls_path, a.results).replace(os.sep, "/")
    numbers["integration_result"] = "RESULT_integration_3.2.md"
    findings = C.check_layout(fig)
    paths = C.save(fig, a.out, NAME, a.png_only)
    C.write_numbers(a.out, NAME, numbers)
    for f in findings:
        print("layout:", f)
    print(numbers)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
