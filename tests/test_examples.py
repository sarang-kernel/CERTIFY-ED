"""The example scripts run and finish without error."""
import os
import subprocess
import sys


from conftest import needs_flint, needs_quspin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")


def _run(name, tmp_path):
    p = subprocess.run([sys.executable, os.path.join(EX, name)], cwd=tmp_path, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=ROOT), timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout


@needs_quspin
def test_example_01(tmp_path):
    out = _run("01_certify_quspin.py", tmp_path)
    assert "CERTIFIED" in out and (tmp_path / "example_01_certificate.json").exists()


@needs_flint
def test_example_02(tmp_path):
    out = _run("02_exact_recheck.py", tmp_path)
    assert "S6.exact_recheck       CERTIFIED" in out and "status CERTIFIED" in out


def test_example_03(tmp_path):
    out = _run("03_whole_spectrum.py", tmp_path)
    assert "complete: True" in out and "'all_counted': True" in out


@needs_flint
def test_example_04(tmp_path):
    out = _run("04_applications.py", tmp_path)
    assert "eta < 1/4: True" in out and ": True" in out.splitlines()[-1]
