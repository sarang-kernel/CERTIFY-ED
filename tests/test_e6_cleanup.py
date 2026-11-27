"""E6 removes its repository copies whatever happens (experiments/e6_mutation.py).

Up to 3.2.3 the per-worker copies (mut<w>_* next to the repository) were deleted only after a complete run, with
shutil.rmtree(ignore_errors=True). A run whose baseline failed (SystemExit) kept them, and on Windows the read-only
object files of the git repository that a test creates inside each copy were silently kept even after a complete run.
"""
import os
import stat
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, EXP)

import e6_mutation as e6  # noqa: E402


def _read_only_tree(base):
    """A small tree shaped like a git repository whose object file is read-only, as git leaves it."""
    obj = os.path.join(base, ".git", "objects", "ab")
    os.makedirs(obj)
    path = os.path.join(obj, "cdef0123")
    with open(path, "wb") as fh:
        fh.write(b"x")
    os.chmod(path, stat.S_IREAD)
    os.chmod(obj, stat.S_IREAD | stat.S_IEXEC | stat.S_IWRITE)
    return path


def test_remove_tree_deletes_read_only_files(tmp_path):
    d = tmp_path / "mut0_abc"
    _read_only_tree(str(d))
    assert e6._remove_tree(str(d))
    assert not d.exists()


def test_handler_retries_removals_only(tmp_path, monkeypatch):
    """The rmtree handler retries a failed removal after setting the permission bits; any other failed call is left
    alone, and an error of the retry does not propagate."""
    f = tmp_path / "f"
    f.write_bytes(b"x")
    os.chmod(f, stat.S_IREAD)
    seen = []
    e6._force_writable(lambda p: seen.append(p), str(f), None)        # e.g. os.scandir: not retried
    assert seen == [] and f.exists() and not stat.S_IMODE(os.stat(f).st_mode) & stat.S_IWRITE
    e6._force_writable(os.unlink, str(f), None)
    assert not f.exists()
    d = tmp_path / "d"
    d.mkdir()
    e6._force_writable(os.rmdir, str(d), None)
    assert not d.exists()
    g = tmp_path / "g"
    g.write_bytes(b"x")

    def refuse(*_args):
        raise ValueError("refused")

    monkeypatch.setattr(e6.os, "chmod", refuse)
    e6._force_writable(os.unlink, str(g), None)                         # does not raise
    assert g.exists()


def test_remove_tree_of_a_missing_path_is_true(tmp_path):
    assert e6._remove_tree(str(tmp_path / "absent"))


def test_remove_copies_reports_what_is_left(tmp_path, monkeypatch):
    a, b = tmp_path / "mut0_a", tmp_path / "mut1_b"
    a.mkdir()
    b.mkdir()
    real = e6._remove_tree
    monkeypatch.setattr(e6, "_remove_tree", lambda p: False if p == str(b) else real(p))
    assert e6._remove_copies([str(a), str(b)]) == [str(b)]
    assert not a.exists()


def test_copies_are_removed_when_the_baseline_fails(tmp_path, monkeypatch):
    copy_dir = tmp_path / "copies"
    copy_dir.mkdir()
    monkeypatch.setenv("CERTIFY_ED_E6_COPY_DIR", str(copy_dir))
    monkeypatch.setenv("CERTIFY_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(e6, "_tracked_files", lambda: ["pyproject.toml"])
    seen = []

    def failing_baseline(workdir, timeout):
        seen.append(workdir)
        _read_only_tree(workdir)   # a test inside the copy made a git repository
        return "killed", "tests/test_x.py::test_y", 0.0

    monkeypatch.setattr(e6, "run_tests", failing_baseline)
    monkeypatch.setattr(sys, "argv", ["e6_mutation.py", "1", "2"])
    with pytest.raises(SystemExit):
        e6.main()
    assert len(seen) == 1
    assert os.path.basename(seen[0]).startswith("mut0_")
    assert list(copy_dir.iterdir()) == []
