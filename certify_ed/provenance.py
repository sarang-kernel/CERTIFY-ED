"""Provenance of a computation: which code produced a record.

`source_fingerprint` identifies the code that can change a recorded number: the git blobs of
certify_ed/ and experiments/ without the files listed in NON_COMPUTATIONAL (the command line, test
helpers, this module, and the post-processing scripts that only read records), plus a flag for
uncommitted changes to those paths. The numbers step records its own commit. Two records with the same `source_hash`
and `source_dirty = False` ran the same computational code, even if the commits differ. The test suite
is fingerprinted separately (`tests_hash`, `tests_dirty`): it never changes a computed number, but the
mutation analysis (E6) measures it."""
import hashlib
import os
import subprocess

NON_COMPUTATIONAL = ("certify_ed/cli.py", "certify_ed/testing.py", "certify_ed/provenance.py",
                     # post-processing: they read records and write tables, they never produce a record
                     "experiments/make_numbers.py", "experiments/make_numbers_apps.py",
                     "experiments/e6_classify.py", "experiments/e6_recheck.py")


def repo_root():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return here if os.path.isdir(os.path.join(here, ".git")) or os.path.isdir(os.path.join(here, "experiments")) else None


def _git(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
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
