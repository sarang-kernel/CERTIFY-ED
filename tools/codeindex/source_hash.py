"""source_hash.py: recompute the computational source hash and the test-suite hash of certify-ed at any commit.

Usage:  python tools/codeindex/source_hash.py [REPO] [REV ...]
        (defaults: REPO = archive_v5/code/certify-ed, REV = v3.1.0 v3.2.0 v3.2.1 HEAD)

certify-ed fingerprints the code that can change a recorded number (certify_ed/provenance.py,
`source_fingerprint`): the SHA-256 of the output of `git ls-files -s -- certify_ed experiments` with the files
of NON_COMPUTATIONAL excluded, truncated to 16 hex digits, and the same over `tests/` for `tests_hash`.
That function reads the index of a checkout. This script reproduces the same hashes from the commit alone:
it reads NON_COMPUTATIONAL from `certify_ed/provenance.py` as committed at REV, lists the blobs with
`git ls-tree -r REV`, writes each entry in the `ls-files -s` form "<mode> <blob> 0<TAB><path>", sorts by
path bytes (the order of the index) and hashes. It never writes to the repository: no checkout, no index
refresh, no `git status`. The printed values can be compared with `source_hash` and `tests_hash` in
`manifest.json` of a run directory.
"""
import ast
import hashlib
import os
import re
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(["git", "-c", "core.filemode=false", *args], cwd=repo, capture_output=True,
                          check=True).stdout.decode("utf-8")


def non_computational(repo, rev):
    src = git(repo, "show", f"{rev}:certify_ed/provenance.py")
    m = re.search(r"NON_COMPUTATIONAL = (\(.*?\))\n\n", src, re.S)
    return ast.literal_eval(m.group(1)) if m else ()


def ls_files_form(repo, rev, paths, exclude=()):
    out = []
    for line in git(repo, "ls-tree", "-r", rev, "--", *paths).splitlines():
        meta, path = line.split("\t", 1)
        mode, _typ, blob = meta.split()
        if any(path == x or path.startswith(x.rstrip("/") + "/") for x in exclude):
            continue
        out.append((path.encode("utf-8"), f"{mode} {blob} 0\t{path}"))
    out.sort(key=lambda t: t[0])
    return "\n".join(s for _, s in out)


def hashes(repo, rev):
    nc = non_computational(repo, rev)
    src = ls_files_form(repo, rev, ["certify_ed", "experiments"], nc)
    tst = ls_files_form(repo, rev, ["tests"])
    commit = git(repo, "rev-parse", "--short", rev + "^{commit}").strip()
    return {"rev": rev, "commit": commit, "non_computational_entries": len(nc),
            "source_hash": hashlib.sha256(src.encode()).hexdigest()[:16],
            "tests_hash": hashlib.sha256(tst.encode()).hexdigest()[:16]}


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = sys.argv[1:]
    repo = args.pop(0) if args and os.path.isdir(args[0]) else os.path.join(os.environ.get("CERTIFYED_ARCHIVE") or os.path.join(here, "..", "archive", "archive_v5"), "code",
                                                                             "certify-ed")
    if not os.path.isdir(repo):
        sys.exit("code checkout not found at " + repo + ": set CERTIFYED_ARCHIVE to the unpacked archive v5, or pass a "
                 "checkout of the code as the first argument (this repository works)")
    revs = args or ["v3.1.0", "v3.2.0", "v3.2.1", "HEAD"]
    for rev in revs:
        h = hashes(repo, rev)
        print(f"{h['rev']}\t{h['commit']}\tsource_hash {h['source_hash']}\ttests_hash {h['tests_hash']}\t"
              f"({h['non_computational_entries']} non-computational entries)")
