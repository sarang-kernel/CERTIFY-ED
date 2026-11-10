"""build.py: assemble CHANGELOG_PROJECT.md from the tables in tools/changelog/ and check it.

Usage:  python tools/changelog/build.py [--archive sources/archive_v5] [--out CHANGELOG_PROJECT.md]

Inputs (all in this folder): entries.tsv (date, area, change, sources), months.tsv (month, paragraph),
versions.tsv (version, date, commit, contents, evidence), intro.md (text before the first month), and the tables
written by collect.py (commits.tsv, decisions.tsv, runs.tsv, tags.tsv). Run collect.py first.

Checks (the script exits with status 1 if one fails):
  every row has a date, an area from the fixed list and at least one source;
  every source is a path that exists under the archive, a commit of commits.tsv, a decision D-01 to D-61, or a note
  marked with a leading "~";
  every commit of commits.tsv, every decision, every run record and every handover is cited by at least one row;
  dates lie between 18 December 2025 and 6 November 2026 and do not go backwards within a month;
  no table cell contains a pipe character, no text contains an em or en dash.
Also writes changelog_sources_bundle.txt (the text of every cited file, the member lists of cited zip and tar files
and the tables of this folder), for `python tools/lint_docs.py --bundle tools/changelog/changelog_sources_bundle.txt`.
"""
import csv
import datetime as dt
import os
import re
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ARCHIVE = arg("--archive", os.path.join(ROOT, "sources", "archive_v5"))
OUT = arg("--out", os.path.join(ROOT, "CHANGELOG_PROJECT.md"))
BUNDLE = arg("--bundle-out", os.path.join(HERE, "changelog_sources_bundle.txt"))
AREAS = ["code", "mathematics", "manuscript", "runs", "decisions", "applications", "documents", "archive"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
FIRST, LAST = "2025-12-18", "2026-11-06"


def tsv(name):
    with open(os.path.join(HERE, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def show_date(d):
    """2026-05-13 -> 13 May 2026; 2026-05 -> May 2026 (day not recorded); A/B -> a range."""
    if "/" in d:
        a, b = d.split("/")
        if len(a) == 7 and len(b) == 7:
            ya, ma = int(a[:4]), int(a[5:7])
            yb, mb = int(b[:4]), int(b[5:7])
            left = MONTHS[ma - 1] + ("" if ya == yb else " " + str(ya))
            return f"{left} to {MONTHS[mb - 1]} {yb} (days not recorded)"
        da, db = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        if da.month == db.month:
            return f"{da.day} to {db.day} {MONTHS[da.month - 1]} {da.year}"
        return f"{da.day} {MONTHS[da.month - 1]} to {db.day} {MONTHS[db.month - 1]} {db.year}"
    if len(d) == 7:
        return f"{MONTHS[int(d[5:7]) - 1]} {d[:4]} (day not recorded)"
    x = dt.date.fromisoformat(d)
    return f"{x.day} {MONTHS[x.month - 1]} {x.year}"


def month_key(d):
    return d.split("/")[0][:7]


def main():
    probs = []
    commits = {c["commit"]: c for c in tsv("commits.tsv")}
    decisions = {d["id"] for d in tsv("decisions.tsv")}
    runs = [r["run"] for r in tsv("runs.tsv")]
    entries = tsv("entries.tsv")
    months = {m["month"]: m["paragraph"] for m in tsv("months.tsv")}
    versions = tsv("versions.tsv")
    cited_commits, cited_dec, cited_paths = set(), set(), set()

    def render_sources(field, where):
        out, hashes = [], []
        items = [s.strip() for s in field.split(";") if s.strip()]
        if not items:
            probs.append((where, "row without a source"))
        for s in items:
            if re.fullmatch(r"[0-9a-f]{7}", s):
                if s not in commits:
                    probs.append((where, "unknown commit " + s))
                cited_commits.add(s)
                hashes.append("`" + s + "`")
                continue
            if hashes:
                out.append(", ".join(hashes))
                hashes = []
            if re.fullmatch(r"D-\d\d", s):
                if s not in decisions:
                    probs.append((where, "unknown decision " + s))
                cited_dec.add(s)
                out.append(s)
            elif s.startswith("~"):
                out.append(s[1:].strip())
            else:
                if not os.path.exists(os.path.join(ARCHIVE, *s.rstrip("/").split("/"))):
                    probs.append((where, "missing path " + s))
                cited_paths.add(s.rstrip("/"))
                out.append("`" + s + "`")
        if hashes:
            out.append(", ".join(hashes))
        return "; ".join(out)

    lines = [open(os.path.join(HERE, "intro.md"), encoding="utf-8").read().rstrip("\n"), ""]
    by_month = {}
    for i, e in enumerate(entries, 2):
        where = f"entries.tsv line {i}"
        d = e["date"].strip()
        if not re.fullmatch(r"\d{4}-\d\d(-\d\d)?(/\d{4}-\d\d(-\d\d)?)?", d):
            probs.append((where, "bad date " + d))
            continue
        for part in d.split("/"):
            if part[:len(FIRST[:len(part)])] < FIRST[:len(part)] or part > LAST:
                probs.append((where, "date outside the period " + d))
        if e["area"] not in AREAS:
            probs.append((where, "unknown area " + e["area"]))
        for cell in (e["change"], e["sources"]):
            if "|" in cell:
                probs.append((where, "pipe in a cell"))
        by_month.setdefault(month_key(d), []).append((where, d, e))
    ym = (2025, 12)
    while ym <= (2026, 11):
        key = f"{ym[0]}-{ym[1]:02d}"
        lines.append(f"## {MONTHS[ym[1] - 1]} {ym[0]}")
        lines.append("")
        if key not in months:
            probs.append(("months.tsv", "no paragraph for " + key))
        lines.append(months.get(key, ""))
        lines.append("")
        rows = by_month.get(key, [])
        if rows:
            lines.append("| Date | Area | Change | Source |")
            lines.append("|---|---|---|---|")
            last = ""
            for where, d, e in rows:
                first = d.split("/")[0]
                if len(first) == 10:
                    if first < last:
                        probs.append((where, "date goes backwards " + d))
                    last = first
                lines.append(f"| {show_date(d)} | {e['area']} | {e['change'].strip()} | "
                             f"{render_sources(e['sources'], where)} |")
            lines.append("")
        ym = (ym[0] + (ym[1] == 12), 1 if ym[1] == 12 else ym[1] + 1)
    lines.append("## Version index")
    lines.append("")
    lines.append(months.get("versions", ""))
    lines.append("")
    lines.append("| Version or tag | Date | Commit | What it contained | Evidence run |")
    lines.append("|---|---|---|---|---|")
    for i, v in enumerate(versions, 2):
        where = f"versions.tsv line {i}"
        for c in re.findall(r"`([0-9a-f]{7})`", v["commit"]):
            if c not in commits and "public repository" not in v["commit"]:
                probs.append((where, "unknown commit " + c))
        for cell in v.values():
            if "|" in cell:
                probs.append((where, "pipe in a cell"))
        lines.append(f"| {v['version']} | {v['date']} | {v['commit']} | {v['contents']} | {v['evidence']} |")
    lines.append("")
    text = "\n".join(lines)
    for ch in ("\u2014", "\u2013", "\u2012", "\u2015"):
        if ch in text:
            probs.append(("output", "dash character U+%04X" % ord(ch)))
    missing = sorted(set(commits) - cited_commits, key=lambda h: (commits[h]["date_ist"], commits[h]["time_ist"]))
    for h in missing:
        probs.append(("coverage", "commit not cited " + h + " " + commits[h]["date_ist"]))
    for dname in sorted(decisions - cited_dec):
        probs.append(("coverage", "decision not cited " + dname))
    for r in runs:
        if "code/certify-ed/results/" + r not in cited_paths:
            probs.append(("coverage", "run record not cited " + r))
    for f in sorted(os.listdir(os.path.join(ARCHIVE, "project_docs"))):
        if f.startswith("SESSION_HANDOVER") and "project_docs/" + f not in cited_paths:
            probs.append(("coverage", "handover not cited " + f))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    parts = []
    for pth in sorted(cited_paths):
        full = os.path.join(ARCHIVE, *pth.split("/"))
        files = [full] if os.path.isfile(full) else [os.path.join(dp, n) for dp, _, ns in os.walk(full) for n in ns]
        for fp in files:
            try:
                if fp.endswith(".zip"):
                    z = zipfile.ZipFile(fp)
                    parts.append("\n".join(f"{i.filename} {i.date_time}" for i in z.infolist()))
                    if z.comment:
                        parts.append(z.comment.decode("ascii", "replace"))
                elif fp.endswith((".tar.gz", "_tar.gz", ".tgz")):
                    with tarfile.open(fp) as t:
                        parts.append("\n".join(f"{m.name} {dt.datetime.fromtimestamp(m.mtime, dt.timezone.utc)}"
                                               for m in t.getmembers()))
                elif os.path.getsize(fp) < 3_000_000 and not fp.endswith((".pdf", ".png", ".docx", ".bundle")):
                    parts.append(open(fp, encoding="utf-8", errors="replace").read())
            except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
                probs.append((pth, "unreadable source: " + str(exc)[:60]))
    for name in ("commits.tsv", "decisions.tsv", "runs.tsv", "tags.tsv"):
        parts.append(open(os.path.join(HERE, name), encoding="utf-8").read())
    with open(BUNDLE, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))
    n_rows = sum(len(v) for v in by_month.values())
    print(f"{n_rows} rows, {len(versions)} versions, {len(cited_commits)} of {len(commits)} commits cited, "
          f"{len(cited_dec)} of {len(decisions)} decisions cited")
    for p in probs:
        print(*p, sep="\t")
    print(f"{len(probs)} build problems")
    sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
