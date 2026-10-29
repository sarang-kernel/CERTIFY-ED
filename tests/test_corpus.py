"""B-07: the planted-error corpus (corpus/make_corpus.py, corpus/run_corpus.py, corpus/cases).

What is checked: the committed cases are well-formed producer bundles whose expected.json is consistent (valid
statement ids, disjoint lists covering every statement, a declared accuracy); the index hashes match the committed
arrays; the generator is deterministic in-process; the scorer's verdicts (detected / unsound / silent / refused /
conditional / error / false alarm / control) on synthetic statuses; the runner end to end with a checker that
certifies everything (every planted case unsound, exit 1) and with one that refuses (NaN cases refused, the rest
error); and one real case pair through certify-ed (a planted error detected, the control clean).

Soundness note: nothing here changes a status; the corpus is a battery of negative controls for any checker.
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from certify_ed.certificate_v2 import STATEMENT_IDS
from certify_ed.io.bundle import read_bundle
from certify_ed.io.formats import FormatError
from certify_ed.schema import validate_bundle_meta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
CASES = os.path.join(CORPUS, "cases")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CORPUS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def make_corpus():
    return _load("make_corpus")


@pytest.fixture(scope="module")
def run_corpus():
    return _load("run_corpus")


def _case_dirs():
    return sorted(d for d in os.listdir(CASES) if os.path.isfile(os.path.join(CASES, d, "expected.json")))


def test_committed_corpus_is_small_and_indexed():
    dirs = _case_dirs()
    assert len(dirs) >= 50
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(CASES) for fn in fns)
    assert total < 5_000_000
    with open(os.path.join(CASES, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    assert sorted(c["case"] for c in index["cases"]) == dirs


@pytest.mark.parametrize("case", _case_dirs())
def test_every_case_is_a_readable_bundle_with_a_consistent_expectation(case):
    d = os.path.join(CASES, case)
    with open(os.path.join(d, "expected.json"), encoding="utf-8") as f:
        exp = json.load(f)
    assert exp["schema"] == "certify-ed/corpus-case/1.0" and exp["case"] == case
    mn, mc, may = exp["must_not_certify"], exp["must_certify"], exp["may_certify"]
    assert set(mn) <= set(STATEMENT_IDS) and set(mc) <= set(STATEMENT_IDS)
    assert not (set(mn) & set(mc)) and not (set(mn) & set(may)) and not (set(mc) & set(may))
    assert set(mn) | set(mc) | set(may) == set(STATEMENT_IDS)
    assert exp["error_class"] == "correct" or mn or exp["error_class"] == "unnormalised_isolated"
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(d, "spec.json"), encoding="utf-8") as f:
        spec_d = json.load(f)
    assert meta["accuracy_target"] == 1e-8 and meta["kind"] == exp["claim_kind"]
    validate_bundle_meta(meta, spec_d)
    if exp["error_class"].startswith("nan_"):
        with pytest.raises(FormatError):               # non-finite entries are rejected at the door
            read_bundle(d)
        return
    b = read_bundle(d)
    assert b.spec is not None and b.claim.operator is not None
    assert b.claim.eigenvalues.size == exp["parameters"].get("k", b.claim.eigenvalues.size)
    assert b.claim.eigenvectors.shape[1] == b.claim.eigenvalues.size


def test_index_hashes_match_the_committed_arrays(make_corpus):
    with open(os.path.join(CASES, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    for rec in index["cases"]:
        d = os.path.join(CASES, rec["case"])
        with np.load(os.path.join(d, "eigenvalues.npz")) as z:
            lam = z[z.files[0]]
        assert make_corpus.array_sha256(lam) == rec["sha256"]["eigenvalues"], rec["case"]
        with np.load(os.path.join(d, "eigenvectors.npz")) as z:
            X = z[z.files[0]]
        assert make_corpus.array_sha256(X) == rec["sha256"]["eigenvectors"], rec["case"]


def test_generator_is_deterministic_in_process(make_corpus):
    a, _ = make_corpus.make_cases_for("heis8")
    b, _ = make_corpus.make_cases_for("heis8")
    assert [c.cid for c in a] == [c.cid for c in b]
    for ca, cb in zip(a, b):
        assert make_corpus.array_sha256(ca.lam) == make_corpus.array_sha256(cb.lam)
        assert make_corpus.array_sha256(ca.X) == make_corpus.array_sha256(cb.X)
        assert ca.exp == cb.exp
    classes = {c.exp["error_class"] for c in a}
    assert {"correct", "wrong_eigenvalue", "swapped_pairs", "duplicated_vector", "wrong_coupling", "non_hermitian",
            "nan_eigenvalue", "mislabeled_sector", "subnormal_entries", "wrong_window_count",
            "missing_multiplet_member"} <= classes


def test_planted_errors_are_what_they_say(make_corpus):
    """The arrays of a few cases carry the stated change (the generator did not silently produce a correct claim)."""
    cases, _ = make_corpus.make_cases_for("heis8")
    by = {c.cid: c for c in cases}
    good = by["heis8_correct"]
    H = good.H
    res = lambda c: np.linalg.norm(H @ c.X - c.X * c.lam[None, :], axis=0) / np.linalg.norm(c.X, axis=0)
    assert np.max(res(good)) < 1e-12
    assert abs(np.max(res(by["heis8_wrong_eigenvalue_1e-05"])) - 1e-5) < 1e-9
    assert np.max(res(by["heis8_swapped_pairs"])) > 0.1
    dup = by["heis8_duplicated_vector"]
    assert np.allclose(dup.X[:, 0], dup.X[:, 1]) and dup.lam[0] == dup.lam[1]
    nh = by["heis8_non_hermitian"].H
    assert abs(nh - nh.T).max() > 0.1
    assert np.isnan(by["heis8_nan_eigenvalue"].lam).sum() == 1
    sub = by["heis8_subnormal_entries"]
    i, j = sub.exp["parameters"]["entry"]
    assert 0 < sub.X[i, j] < 2.0 ** -1022
    wc = by["heis8_wrong_coupling"]
    assert abs(wc.H - H).max() > 0.4                         # a flipped bond changes entries by 2|J|/4 at least


# --------------------------------------------------------------------------------------------------- the scorer
def _exp(must_not, must, cls="x", conditional=False, refusal_ok=False):
    return {"error_class": cls, "must_not_certify": must_not, "must_certify": must, "conditional": conditional,
            "refusal_ok": refusal_ok}


def test_scorer_verdicts(run_corpus):
    sc = run_corpus.score
    S1, S2, S3 = "S1.model_fidelity", "S2.eigenpairs", "S3.completeness"
    assert sc(_exp([S2], [S1]), {S1: "CERTIFIED", S2: "FAILED"}, False, False)["verdict"] == "detected"
    assert sc(_exp([S2], [S1]), {S1: "CERTIFIED", S2: "INCONCLUSIVE"}, False, False)["verdict"] == "detected"
    r = sc(_exp([S2], [S1]), {S1: "CERTIFIED", S2: "CERTIFIED"}, False, False)
    assert r["verdict"] == "unsound" and r["unsound"] == [S2]
    assert sc(_exp([S2], [S1]), {S1: "CERTIFIED", S2: "NOT_CHECKED"}, False, False)["verdict"] == "silent"
    assert sc(_exp([S2], [S1]), {S1: "CERTIFIED"}, False, False)["verdict"] == "silent"      # absent = not checked
    r = sc(_exp([S2], [S1]), {S1: "FAILED", S2: "FAILED"}, False, False)
    assert r["verdict"] == "detected" and r["false_alarms"] == [S1]
    r = sc(_exp([S2], [S1]), {S1: "INCONCLUSIVE", S2: "FAILED"}, False, False)
    assert r["weak_alarms"] == [S1] and not r["false_alarms"]
    # conditional case: CERTIFIED is reported, not counted as unsound
    r = sc(_exp([S2], [S1], conditional=True), {S1: "CERTIFIED", S2: "CERTIFIED"}, False, False)
    assert r["verdict"] == "conditional" and r["conditional"] == [S2] and not r["unsound"]
    # two statements: one certified (unsound) wins over the other flagged
    r = sc(_exp([S2, S3], [S1]), {S1: "CERTIFIED", S2: "CERTIFIED", S3: "FAILED"}, False, False)
    assert r["verdict"] == "unsound"
    # refusal
    assert sc(_exp([S2], [], refusal_ok=True), None, True, False)["verdict"] == "refused"
    assert sc(_exp([S2], [], refusal_ok=False), None, True, False)["verdict"] == "error"
    assert sc(_exp([S2], []), None, False, True)["verdict"] == "error"
    # control
    r = sc(_exp([], [S1, S2, S3], cls="correct"), {S1: "CERTIFIED", S2: "CERTIFIED", S3: "FAILED"}, False, False)
    assert r["verdict"] == "control" and r["false_alarms"] == [S3]


def test_summary_counts(run_corpus):
    rows = [{"verdict": v, "false_alarms": fa, "weak_alarms": wa} for v, fa, wa in (
        ("detected", [], []), ("detected", ["S1.model_fidelity"], []), ("unsound", [], []), ("silent", [], []),
        ("refused", [], []), ("error", [], []), ("conditional", [], []), ("control", [], []),
        ("control", [], ["S3.completeness"]))]
    s = run_corpus.summarize(rows)
    assert s["planted"] == 7 and s["controls"] == 2 and s["controls_clean"] == 1
    assert s["detected"] == 3 and s["detected_by_refusal"] == 1 and s["missed"] == 3
    assert s["unsound"] == 1 and s["silent"] == 1 and s["error"] == 1 and s["conditional"] == 1
    assert s["false_alarms"] == 1 and s["weak_alarms"] == 1
    assert s["line"].startswith("detected 3 / missed 3 (unsound 1, silent 1, error 1) / false alarms 1 (weak 1) of 7")


def test_result_reader_accepts_certificates_and_flat_objects(run_corpus, tmp_path):
    p = tmp_path / "a"
    p.with_suffix(".json").write_text(json.dumps({"statements": [{"id": "S1.model_fidelity", "status": "CERTIFIED"},
                                                                  {"id": "S5.corroboration", "status": "CORROBORATED"}]}))
    r = run_corpus.read_result(str(p))
    assert r == {"S1.model_fidelity": "CERTIFIED", "S5.corroboration": "NOT_CHECKED"}
    q = tmp_path / "b"
    q.write_text(json.dumps({"S2": "failed", "S3": "Not Checked", "extra": 1}))
    assert run_corpus.read_result(str(q)) == {"S2.eigenpairs": "FAILED", "S3.completeness": "NOT_CHECKED"}
    assert run_corpus.read_result(str(tmp_path / "missing")) is None


# ------------------------------------------------------------------------------------------- the runner end to end
_CERTIFY_ALL = """import json, sys
out = sys.argv[2]
json.dump({s: "CERTIFIED" for s in ["S1", "S2", "S3", "S4", "S8"]}, open(out + ".json", "w"))
"""
_REFUSE = "import sys\nsys.exit(2)\n"


def _run(args, tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("PYTHONSAFEPATH", None)
    return subprocess.run([sys.executable, os.path.join(CORPUS, "run_corpus.py"), "--results", str(tmp_path / "res"),
                           *args], cwd=ROOT, capture_output=True, text=True, env=env, timeout=1200)


def test_runner_scores_a_checker_that_certifies_everything_as_unsound(tmp_path):
    chk = tmp_path / "certify_all.py"
    chk.write_text(_CERTIFY_ALL)
    only = "heis8_correct,heis8_swapped_pairs,heis8_wrong_coupling,heis8_subnormal_entries,heis8_nan_eigenvalue"
    r = _run(["--checker", f'"{sys.executable}" "{chk}" {{case}} {{out}}', "--name", "certify_all", "--only", only],
             tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    rep = json.load(open(tmp_path / "res" / "report.json"))
    by = {row["case"]: row for row in rep["rows"]}
    assert by["heis8_correct"]["verdict"] == "control" and not by["heis8_correct"]["false_alarms"]
    assert by["heis8_swapped_pairs"]["verdict"] == "unsound" and by["heis8_wrong_coupling"]["verdict"] == "unsound"
    assert by["heis8_subnormal_entries"]["verdict"] == "conditional"          # allowed under a stronger model
    assert by["heis8_nan_eigenvalue"]["verdict"] == "unsound"                  # NaN certified is never right
    assert rep["summary"]["unsound"] == 3 and rep["summary"]["detected"] == 0
    md = (tmp_path / "res" / "report.md").read_text(encoding="utf-8")
    assert "UNSOUND: S2" in md and "heis8_correct" in md


def test_runner_scores_refusals(tmp_path):
    chk = tmp_path / "refuse.py"
    chk.write_text(_REFUSE)
    r = _run(["--checker", f'"{sys.executable}" "{chk}" {{case}} {{out}}', "--name", "refuse",
              "--only", "heis8_nan_eigenvalue,heis8_swapped_pairs"], tmp_path)
    assert r.returncode == 1
    rep = json.load(open(tmp_path / "res" / "report.json"))
    by = {row["case"]: row for row in rep["rows"]}
    assert by["heis8_nan_eigenvalue"]["verdict"] == "refused" and by["heis8_swapped_pairs"]["verdict"] == "error"
    assert rep["summary"]["detected"] == 1 and rep["summary"]["missed"] == 1
    assert os.path.exists(tmp_path / "res" / "heis8_swapped_pairs" / "checker_output.txt")


def test_certify_ed_detects_a_planted_error_and_passes_the_control(tmp_path):
    """The default checker on one planted case and its control (the full run is corpus/results/certify-ed)."""
    r = _run(["--only", "heis8_swapped_pairs,heis8_correct"], tmp_path)
    rep = json.load(open(tmp_path / "res" / "report.json"))
    by = {row["case"]: row for row in rep["rows"]}
    assert by["heis8_swapped_pairs"]["verdict"] == "detected", by["heis8_swapped_pairs"]
    assert by["heis8_swapped_pairs"]["statuses"]["S2.eigenpairs"] == "FAILED"
    assert by["heis8_correct"]["verdict"] == "control" and not by["heis8_correct"]["false_alarms"]
    assert by["heis8_correct"]["statuses"]["S3.completeness"] == "CERTIFIED"
    assert r.returncode == 0, r.stdout + r.stderr


def test_committed_report_of_certify_ed_has_no_unsound_case():
    """The committed run (corpus/results/certify-ed/report.json) is the record of the RESULT; a regression that
    makes a planted case CERTIFIED must be re-recorded there and explained."""
    p = os.path.join(CORPUS, "results", "certify-ed", "report.json")
    if not os.path.exists(p):
        pytest.skip("no committed run")
    rep = json.load(open(p, encoding="utf-8"))
    assert rep["summary"]["unsound"] == 0 and rep["summary"]["false_alarms"] == 0
    assert rep["summary"]["planted"] + rep["summary"]["controls"] == len(_case_dirs())
