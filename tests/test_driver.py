"""The reproduction driver (certify-ed reproduce / run.py): provenance, resume, parallel jobs, numbers."""
import json
import os

import pytest

from certify_ed.cli import main
from certify_ed.provenance import repo_root, source_fingerprint
from conftest import needs_flint, needs_quspin

ROOT = repo_root()
needs_repo = pytest.mark.skipif(ROOT is None or not os.path.isdir(os.path.join(ROOT or "", "experiments")),
                                reason="needs a repository checkout")


@needs_repo
def test_source_fingerprint_ignores_the_command_line():
    fp = source_fingerprint(ROOT)
    assert fp["source_hash"] and len(fp["source_hash"]) == 16
    from certify_ed.provenance import NON_COMPUTATIONAL
    assert "certify_ed/cli.py" in NON_COMPUTATIONAL


@needs_repo
def test_unknown_experiment_is_rejected(tmp_path):
    assert main(["reproduce", "--only", "E99", "--no-tests", "--no-numbers", "--out", str(tmp_path / "r")]) == 2


@needs_repo
@needs_quspin
@needs_flint
def test_reproduce_parallel_resume_and_numbers(tmp_path):
    out = tmp_path / "run"
    # two quick experiments on two workers
    assert main(["reproduce", "--only", "E1,E3", "--jobs", "2", "--no-tests", "--no-numbers", "--out", str(out)]) == 0
    m = json.loads((out / "manifest.json").read_text())
    for name in ("E1", "E3"):
        step = m["steps"][name]
        assert step["status"] == "ok" and len(step["records"]) == 1
        assert step["records"][0].startswith(name.lower() + "_")       # records attributed by prefix
        assert step["source_hash"] == m["source"]["source_hash"]
    first = {k: v["records"] for k, v in m["steps"].items()}
    # resume keeps both steps (same computational source) unless the source changed in between
    rc = main(["reproduce", "--only", "E1,E3", "--resume", "--no-tests", "--no-numbers", "--out", str(out)])
    assert rc == 0
    m2 = json.loads((out / "manifest.json").read_text())
    if not m["source"]["source_dirty"]:
        assert {k: v["records"] for k, v in m2["steps"].items()} == first
    assert "extended" in m2
    # numbers on a partial run: application numbers skip missing records, manuscript numbers are skipped
    assert main(["numbers", "--run", str(out)]) == 0
    assert (out / "SUMMARY.md").exists()
