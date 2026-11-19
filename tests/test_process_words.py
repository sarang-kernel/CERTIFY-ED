"""No tracked text file names a product or tool of the writing process: the digests of tools/lint_docs.py are applied
to every tracked text file, apart from the third-party survey data and the commit table that mirrors the history."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lint_docs  # noqa: E402

EXCLUDED = {"experiments/survey/survey_sample.json", "experiments/survey/survey_coded.csv",
            "experiments/survey/survey_handcheck.csv", "tools/changelog/commits.tsv"}


def _tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    return [p for p in out.decode("utf-8").split("\0") if p]


def test_no_tracked_text_names_a_tool_of_the_writing_process():
    hits = []
    for rel in _tracked():
        if rel in EXCLUDED:
            continue
        try:
            with open(os.path.join(ROOT, rel), "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue                                   # binary file
        for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if lint_docs._process_digest_hit(line.lower(), line):
                hits.append(f"{rel}:{i}")
    assert not hits, hits[:20]


def test_the_digest_check_finds_a_listed_word():
    """Negative control: a line with a listed word in upper case, as in a disclosure sentence, is found."""
    assert lint_docs._process_digest_hit("x", "with " + "".join(map(chr, (65, 73))) + " assistance")
    assert not lint_docs._process_digest_hit("plain text about eigenvalues", "plain text about eigenvalues")
