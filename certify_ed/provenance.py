"""Provenance of a computation: which code produced a record.

`source_fingerprint` identifies the code that can change a number: the git blobs of certify_ed/ and
experiments/ without the files listed in NON_COMPUTATIONAL (the command line, test helpers and this
module), plus a flag for uncommitted changes to those paths. Two records with the same `source_hash`
and `source_dirty = False` ran the same computational code, even if the commits differ."""
import hashlib
import os
import subprocess

NON_COMPUTATIONAL = ("certify_ed/cli.py", "certify_ed/testing.py", "certify_ed/provenance.py")


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
    return {"git_commit": _git(root, "rev-parse", "HEAD"),
            "source_hash": hashlib.sha256(files.encode()).hexdigest()[:16] if files is not None else None,
            "source_dirty": bool(dirty) if dirty is not None else None}
