"""Smoke test of the figure scripts (experiments/figures): every script runs and writes its PNG.

Concept figures and the figures drawn from documents in the repository always run. Figures drawn from
evidence-run records need $CERTIFY_RESULTS (and $CERTIFY_RESULTS_B for the two-stack cost figure); when a
script exits with _figcommon.DATA_ABSENT the test is skipped, not failed. A tree without experiments/figures (the
source distribution) skips the module.
"""
import glob
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "experiments", "figures")
# the figure scripts run in child processes only (their helper is experiments/figures/_figcommon.py)
pytestmark = pytest.mark.skipif(not os.path.isdir(FIG_DIR), reason="experiments/figures is not in this tree (sdist)")
SCRIPTS = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(FIG_DIR, "fig_*.py")))
DATA_ABSENT = 3
# the scripts draw with matplotlib (the 'plots' extra, not the 'test' one): without it they are skipped, not failed
needs_matplotlib = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib not installed")


def _run(name, out_dir):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    return subprocess.run([sys.executable, os.path.join(FIG_DIR, name + ".py"), "--out", out_dir, "--png-only"],
                          capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)


def test_every_figure_has_a_script_and_index_entry():
    assert len(SCRIPTS) >= 13
    index = open(os.path.join(ROOT, "docs", "figures", "INDEX.md"), encoding="utf-8").read()
    for name in SCRIPTS:
        assert f"`{name}.png`" in index or f"{name}.pdf" in index, f"{name} missing from docs/figures/INDEX.md"
        assert f"experiments/figures/{name}.py" in index


@needs_matplotlib
@pytest.mark.parametrize("name", SCRIPTS)
def test_figure_script_runs(name, tmp_path):
    out = str(tmp_path / "figs")
    r = _run(name, out)
    if r.returncode == DATA_ABSENT:
        pytest.skip(f"{name}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'data absent'}")
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(os.path.join(out, name + ".png"))
    numbers = json.load(open(os.path.join(out, name + ".numbers.json"), encoding="utf-8"))
    assert isinstance(numbers, dict) and numbers


def test_data_absent_exit_code_is_a_skip(tmp_path):
    """Negative control: a record-driven script pointed at an empty results directory exits with DATA_ABSENT and
    writes nothing (the smoke test above turns that into a skip, never a pass)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["CERTIFY_RESULTS"] = str(empty)
    env.pop("CERTIFY_RESULTS_B", None)
    out = tmp_path / "figs"
    r = subprocess.run([sys.executable, os.path.join(FIG_DIR, "fig_failure_matrix.py"), "--out", str(out), "--png-only"],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert r.returncode == DATA_ABSENT
    assert "data absent" in r.stderr
    assert not glob.glob(str(out / "*.png"))


def test_index_is_in_sync_with_numbers_files():
    """docs/figures/INDEX.md is rendered from the .numbers.json files; a stale index fails."""
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    r = subprocess.run([sys.executable, os.path.join(FIG_DIR, "make_index.py"), "--check"], capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr


@needs_matplotlib
def test_concept_figures_need_no_data(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["CERTIFY_RESULTS"] = str(tmp_path / "nowhere")
    env.pop("CERTIFY_RESULTS_B", None)
    out = tmp_path / "figs"
    r = subprocess.run([sys.executable, os.path.join(FIG_DIR, "fig_certificate_anatomy.py"), "--out", str(out), "--png-only"],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "fig_certificate_anatomy.png").exists()


def test_mutation_figure_reads_the_integration_sample_from_docs_results():
    """fig_mutation.py reads the 3.2 mutation sample from docs/results/ (moved from the root on 20 November 2026);
    the record-driven smoke test above skips without run records, so the path and the parse are checked here. The
    script is imported in a separate process, so that no figure module enters the test session."""
    path = os.path.join(ROOT, "docs", "results", "RESULT_integration_3.2.md")
    assert os.path.exists(path) and not os.path.exists(os.path.join(ROOT, "RESULT_integration_3.2.md"))
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import fig_mutation; "
            "print(json.dumps(fig_mutation.parse_integration(sys.argv[2])))")
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    r = subprocess.run([sys.executable, "-c", code, FIG_DIR, path], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    killed, n, labels, heading = json.loads(r.stdout.strip().splitlines()[-1])
    assert (killed, n) == (10, 16) and len(labels) == 6 and "sample" in heading
