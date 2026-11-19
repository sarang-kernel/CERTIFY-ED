"""Packaging metadata (A-03), CI workflows (H-01) and container recipes (H-05): the files parse and agree with
each other and with the documented evidence-run stack. None of this touches a computed number."""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = {"numpy": "2.4.4", "scipy": "1.17.1", "python-flint": "0.9.0", "quspin": "1.0.1"}
QUSPIN_TRICK = "pip install quspin==1.0.1 --no-deps"


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_pyproject_extras():
    tomllib = pytest.importorskip("tomllib")
    pp = tomllib.loads(_read("pyproject.toml"))
    extras = pp["project"]["optional-dependencies"]
    for name in ("exact", "quspin", "hdf5", "schema", "all", "dev"):
        assert name in extras, name
    assert any(d.startswith("python-flint") for d in extras["exact"])
    assert any(d.startswith("quspin") for d in extras["quspin"])
    assert any(d.startswith("jsonschema") for d in extras["schema"])
    # `all` is the union of the optional layers (plus pytest)
    for name in ("exact", "quspin", "hdf5", "plots", "schema"):
        for dep in extras[name]:
            assert dep in extras["all"], (name, dep)
    assert pp["project"]["scripts"]["certify-ed"] == "certify_ed.cli:main"
    assert pp["project"]["version"] == __import__("certify_ed").__version__


def test_manifest_ships_schema_and_reference_bundles():
    m = _read("MANIFEST.in")
    assert "recursive-include schema *.json" in m
    assert "recursive-include conformance" in m and "*.npz" in m


def test_workflows_parse_and_use_the_documented_quspin_install():
    yaml = pytest.importorskip("yaml")
    ci = yaml.safe_load(_read(".github/workflows/ci.yml"))
    m = ci["jobs"]["test"]["strategy"]["matrix"]
    assert set(m["os"]) == {"ubuntu-latest", "macos-latest", "windows-latest"}
    assert set(m["python-version"]) == {"3.11", "3.12"}
    runs = "\n".join(str(s.get("run", "")) for s in ci["jobs"]["test"]["steps"])
    assert QUSPIN_TRICK in runs and "quspin-extensions" in runs and "parallel-sparse-tools" in runs
    assert "pytest -q" in runs and "ruff check certify_ed tests experiments conformance" in \
        "\n".join(str(s.get("run", "")) for s in ci["jobs"]["lint"]["steps"])
    assert any("upload-artifact" in str(s.get("uses", "")) for s in ci["jobs"]["test"]["steps"])
    nightly = yaml.safe_load(_read(".github/workflows/nightly-mutation.yml"))
    assert "schedule" in nightly[True] or "schedule" in nightly.get("on", {})
    runs = "\n".join(str(s.get("run", "")) for s in nightly["jobs"]["mutation-sample"]["steps"])
    assert "experiments/e6_mutation.py" in runs and "32" in runs


@pytest.mark.parametrize("rel", ["containers/Dockerfile", "containers/certify-ed.def"])
def test_container_recipes_pin_the_evidence_stack(rel):
    txt = _read(rel)
    for pkg, ver in STACK.items():
        assert f"{pkg}=={ver}" in txt, (rel, pkg)
    assert QUSPIN_TRICK in txt
    assert "python:3.11" in txt
    for pkg in ("h5py", "matplotlib", "mpmath"):
        assert re.search(r"\b" + pkg + r"\b", txt), (rel, pkg)


GOVERNANCE_FILES = ("CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "GOVERNANCE.md", "ROADMAP.md", "docs/SCHEMA_RFC_PROCESS.md",
                    ".github/ISSUE_TEMPLATE/bug.md", ".github/ISSUE_TEMPLATE/feature.md",
                    ".github/ISSUE_TEMPLATE/could-not-certify.md", ".github/ISSUE_TEMPLATE/config.yml")


def test_governance_files_exist_and_say_what_they_must():
    for rel in GOVERNANCE_FILES:
        assert os.path.exists(os.path.join(ROOT, rel)), rel
    assert "Contributor Covenant" in _read("CODE_OF_CONDUCT.md") and "sarangvehale2@gmail.com" in _read("CODE_OF_CONDUCT.md")
    c = _read("CONTRIBUTING.md")
    assert "negative control" in c and "D-12" in c and "co-author lines" in c
    g = _read("GOVERNANCE.md")
    assert "never removed" in g and "Deprecation" in g
    icc = _read(".github/ISSUE_TEMPLATE/could-not-certify.md")
    assert icc.startswith("---\nname: I could not certify X") and "SCOPE_MATRIX" in icc and "reason code" in icc
    yaml = pytest.importorskip("yaml")
    for rel in GOVERNANCE_FILES:
        if rel.endswith(".md") and ".github" in rel:
            front = _read(rel).split("---")[1]
            assert {"name", "about"} <= set(yaml.safe_load(front)), rel
    rm = _read("ROADMAP.md")
    for item in ("B-02", "F-09", "A-03", "H-01", "H-05", "B-09", "G-05", "G-12", "B-01", "D-01"):
        assert item in rm, item


def test_reproducibility_doc_describes_the_quspin_install():
    assert QUSPIN_TRICK in _read("docs/reproducibility.md")
    assert QUSPIN_TRICK in _read("containers/README.md")
