"""Provenance of a computation: which code produced a record.

`source_fingerprint` identifies the code that can change a recorded number: the git blobs of
certify_ed/ and experiments/ without the files listed in NON_COMPUTATIONAL (the command line, test
helpers, this module, and the post-processing scripts that only read records), plus a flag for
uncommitted changes to those paths (content only: the executable bit is ignored). The numbers step
records its own commit and the blob hashes of the number scripts (`numbers_fingerprint`). Two records
with the same `source_hash` and `source_dirty = False` ran the same computational code, even if the
commits differ. The test suite is fingerprinted separately (`tests_hash`, `tests_dirty`): it never
changes a computed number, but the mutation analysis (E6) measures it."""
import hashlib
import os
import subprocess

NON_COMPUTATIONAL = ("certify_ed/cli.py", "certify_ed/testing.py", "certify_ed/provenance.py",
                     # post-processing: they read records and write tables, they never produce a record
                     "experiments/make_numbers.py", "experiments/make_numbers_apps.py",
                     "experiments/e6_classify.py", "experiments/e6_recheck.py",
                     # manuscript tables from documents, not from records: the survey dataset and its
                     # generators, and the scope-matrix generator (3.1.0)
                     "experiments/make_survey_table.py", "experiments/make_scope_table.py",
                     "experiments/compare_runs.py", "experiments/make_result_candidate.py", "experiments/survey")


def repo_root():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return here if os.path.isdir(os.path.join(here, ".git")) or os.path.isdir(os.path.join(here, "experiments")) else None


def _git(root, *args):
    """git with core.filemode=false: a lost executable bit (Windows checkouts, some copies) is not a content
    change and must not mark the computational source as dirty (review IMP-05 / SS-02)."""
    try:
        return subprocess.check_output(["git", "-c", "core.filemode=false", *args], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def source_fingerprint(root=None):
    """Fingerprint of the computational source: {'git_commit', 'source_hash', 'source_dirty'} (all None
    outside a git checkout). Two records with equal source_hash and source_dirty False ran the same code."""
    root = root or repo_root()
    if root is None:
        return {"git_commit": None, "source_hash": None, "source_dirty": None}
    excl = [f":(exclude){f}" for f in NON_COMPUTATIONAL]
    files = _git(root, "ls-files", "-s", "--", "certify_ed", "experiments", *excl)
    dirty = _git(root, "status", "--porcelain", "--", "certify_ed", "experiments", *excl)
    # the test suite cannot change a computed number, but it decides the mutation score (E6), so it is
    # fingerprinted separately
    tfiles = _git(root, "ls-files", "-s", "--", "tests")
    tdirty = _git(root, "status", "--porcelain", "--", "tests")
    return {"git_commit": _git(root, "rev-parse", "HEAD"),
            "source_hash": hashlib.sha256(files.encode()).hexdigest()[:16] if files is not None else None,
            "source_dirty": bool(dirty) if dirty is not None else None,
            "tests_hash": hashlib.sha256(tfiles.encode()).hexdigest()[:16] if tfiles is not None else None,
            "tests_dirty": bool(tdirty) if tdirty is not None else None}


NUMBER_SCRIPTS = ("experiments/make_numbers.py", "experiments/make_numbers_apps.py")


def numbers_fingerprint(root=None):
    """Provenance of the post-processing step (review IMP-06): the commit, the git blob hash of each number
    script as committed, the sha256 of its working-tree content (line endings normalised), whether it is
    modified, and the fingerprint of the computational source and the tests at that time."""
    root = root or repo_root()
    out = {"git_commit": None, "scripts": {}, **({} if root is None else source_fingerprint(root))}
    if root is None:
        return out
    out["git_commit"] = _git(root, "rev-parse", "HEAD")
    for rel in NUMBER_SCRIPTS:
        ls = _git(root, "ls-files", "-s", "--", rel)
        blob = ls.split()[1] if ls else None
        path = os.path.join(root, *rel.split("/"))
        try:
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
        except OSError:
            sha = None
        st = _git(root, "status", "--porcelain", "--", rel)
        out["scripts"][rel] = {"blob": blob, "sha256": sha, "modified": bool(st) if st is not None else None}
    return out
