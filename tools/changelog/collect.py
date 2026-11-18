"""collect.py: machine-readable tables of the dated sources of CERTIFY-ED, for docs/project/CHANGELOG_PROJECT.md.

Usage:  python tools/changelog/collect.py [--archive archive_v5] [--git <repository or bundle clone>]

Writes, next to this script:
  commits.tsv    every commit of every ref of the code repository (git log --all), author date converted to India
                 Standard Time (UTC+05:30), short and full id, number of parents, branch heads and tags at the commit,
                 the id before the history rewrite of 21 October 2026 (code/certify-ed/HASH_MAP.csv), subject
  tags.tsv       every tag: name, commit, tag date (IST), first line of the tag message
  decisions.tsv  D-01 to D-61 of project_docs/DECISIONS_LOG.md: id, date as recorded, title (docs/project/decisions/README.md)
  runs.tsv       every run record code/certify-ed/results/*/manifest.json: started, finished (clock of the machine),
                 commit (rewritten id where the record names an old one), source and tests fingerprints, platform,
                 versions, steps that did not end ok
Only the standard library is used. git must be on PATH (or set GIT to its path).
"""
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ARCHIVE = arg("--archive", os.environ.get("CERTIFYED_ARCHIVE") or os.path.normpath(os.path.join(ROOT, "..", "archive", "archive_v5")))
if not os.path.isdir(ARCHIVE):
    sys.exit("archive v5 not found at " + ARCHIVE + ": unpack it there or set CERTIFYED_ARCHIVE (docs/project/sources/SOURCES.md)")
GITDIR = arg("--git", os.path.join(ARCHIVE, "code", "certify-ed"))
GIT = os.environ.get("GIT", "git")


def git(*a):
    return subprocess.run([GIT, "-C", GITDIR, *a], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def write(name, header, rows):
    with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main():
    hashmap = {}
    with open(os.path.join(ARCHIVE, "code", "certify-ed", "HASH_MAP.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hashmap[r["new_commit"]] = r["old_commit"]
    heads, tags = {}, {}
    for line in git("for-each-ref", "--format=%(refname)\t%(objectname)\t%(*objectname)").splitlines():
        ref, obj, peeled = line.split("\t")
        target = peeled or obj
        if ref.startswith("refs/heads/"):
            heads.setdefault(target, []).append(ref[len("refs/heads/"):])
        elif ref.startswith("refs/tags/"):
            tags.setdefault(target, []).append(ref[len("refs/tags/"):])
    rows = []
    fmt = "%H%x09%h%x09%aI%x09%P%x09%s"
    for line in git("log", "--all", "--reverse", "--topo-order", "--format=" + fmt).splitlines():
        H, h, ad, par, subj = line.split("\t", 4)
        t = dt.datetime.fromisoformat(ad).astimezone(IST)
        rows.append([t.strftime("%Y-%m-%d"), t.strftime("%H:%M"), h, H, len(par.split()),
                     " ".join(sorted(heads.get(H, []))), " ".join(sorted(tags.get(H, []))),
                     hashmap.get(H, "")[:7], subj])
    rows.sort(key=lambda r: (r[0], r[1]))
    write("commits.tsv", ["date_ist", "time_ist", "commit", "commit_full", "parents", "branch_heads", "tags",
                          "old_id", "subject"], rows)
    trows = []
    for line in git("for-each-ref", "refs/tags", "--format=%(refname:short)\t%(objecttype)\t%(*objectname:short)"
                    "\t%(objectname:short)\t%(creatordate:iso-strict)\t%(contents:subject)").splitlines():
        name, typ, peeled, obj, date, subj = line.split("\t", 5)
        t = dt.datetime.fromisoformat(date).astimezone(IST)
        trows.append([name, peeled or obj, typ, t.strftime("%Y-%m-%d %H:%M"), subj])
    write("tags.tsv", ["tag", "commit", "type", "date_ist", "message"], trows)
    # ids and dates from the decision log of the archive; the titles are the neutral ones of
    # docs/project/decisions/README.md (the headlines of the log itself are working notes and are not published)
    log = open(os.path.join(ARCHIVE, "project_docs", "DECISIONS_LOG.md"), encoding="utf-8").read()
    drows = re.findall(r"^\*\*(D-\d+)\s*\(([^)]*)\)\s*\u2014\s*(.*?)\*\*", log, re.M)
    readme = open(os.path.join(ROOT, "docs", "project", "decisions", "README.md"), encoding="utf-8").read()
    titles = dict(re.findall(r"^\| (D-\d+) \| [^|]+ \| ([^|]+?) \|", readme, re.M))
    write("decisions.tsv", ["id", "date_recorded", "title"], [[d, date, titles.get(d, "")] for d, date, _ in drows])
    res = os.path.join(ARCHIVE, "code", "certify-ed", "results")
    rrows = []
    short = {k[:7]: k for k in hashmap}
    old2new = {v[:7]: k[:7] for k, v in hashmap.items()}
    for d in sorted(os.listdir(res)):
        mf = os.path.join(res, d, "manifest.json")
        if not os.path.exists(mf):
            continue
        m = json.load(open(mf, encoding="utf-8"))
        env = m.get("environment", {})
        src = m.get("source") or {}
        c = m.get("git_commit", "")[:7]
        steps = m.get("steps", {})
        bad = [k for k, v in steps.items() if isinstance(v, dict) and v.get("status") not in (None, "ok")]
        rrows.append([d, m.get("mode"), m.get("started"), m.get("finished"), c, old2new.get(c, c),
                      src.get("source_hash", ""), src.get("tests_hash", ""), env.get("platform"), env.get("python"),
                      env.get("certify_ed"), " ".join(f"{k}:{steps[k].get('status')}" for k in bad)])
    write("runs.tsv", ["run", "mode", "started", "finished", "commit_recorded", "commit_now", "source_hash",
                       "tests_hash", "platform", "python", "certify_ed", "steps_not_ok"], rrows)
    print(len(rows), "commits;", len(trows), "tags;", len(drows), "decisions;", len(rrows), "runs")


if __name__ == "__main__":
    main()
