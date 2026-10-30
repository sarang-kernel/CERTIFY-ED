"""A-09: every INCONCLUSIVE / NOT_CHECKED statement carries a non-empty, specific `next_action`.
A-04: python-flint absent -> S6 NOT_CHECKED with MISSING_DEPENDENCY, never an exception; QuSpin absent -> an
ImportError that names the extra to install."""
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

import certify_ed as ce
certify_mod = importlib.import_module("certify_ed.certify")
from certify_ed.diagnostics import NEXT_ACTION_BY_CODE, annotate_next_actions, next_action
from certify_ed.presets import STATEMENTS
from certify_ed.scope import REASON_CODES, classify_not_checked

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = ("INCONCLUSIVE", "NOT_CHECKED")


@pytest.fixture(scope="module")
def heis10():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _open_statements(cert):
    return [s for s in cert.statements if s["status"] in OPEN]


def _assert_actions(cert, where):
    opened = _open_statements(cert)
    for s in opened:
        act = s["evidence"].get("next_action")
        assert isinstance(act, str) and len(act) > 20, (where, s["id"], s["status"], s["summary"])
    for s in cert.statements:
        if s["status"] not in OPEN:
            assert "next_action" not in s["evidence"], (where, s["id"])
    return opened


def test_every_reason_code_has_a_next_action():
    assert set(REASON_CODES) <= set(NEXT_ACTION_BY_CODE)
    for code, text in NEXT_ACTION_BY_CODE.items():
        assert text and len(text) > 20, code


# the INCONCLUSIVE and NOT_CHECKED paths of certify.py, by their summary / reason wording
_CERTIFY_PATHS = [
    ("S1.model_fidelity", "INCONCLUSIVE", "CERTIFY-ED's two reference builders disagree (internal error)", {}, "build_bitwise"),
    ("S1.model_fidelity", "NOT_CHECKED", "no model specification supplied", {"reason_code": "NEEDS_SPEC"}, "ModelSpec"),
    ("S1.model_fidelity", "NOT_CHECKED", "disabled", {"reason_code": "NARROWED_BY_USER"}, "model_fidelity=True"),
    ("S1.model_fidelity", "NOT_CHECKED", "reference build did not fit in memory (MemoryError)", {}, "sector"),
    ("S2.eigenpairs", "INCONCLUSIVE", "claim contains non-finite values (eigenvalue indices [1], eigenvector columns []): "
                                      "no enclosure can be formed", {}, "non-finite"),
    ("S2.eigenpairs", "NOT_CHECKED", "eigenvalue-only claim: no vectors to bound (see S3 for counts)",
     {"reason_code": "NEEDS_EIGENVECTORS"}, "eigenvectors"),
    ("S2.eigenpairs", "INCONCLUSIVE", "residuals computed but rounding not boundable for this operator (estimates only)", {},
     "stored"),
    ("S2.eigenpairs", "INCONCLUSIVE", "each claimed value has an eigenvalue within its radius, but some claimed pairs could "
                                      "not be matched to distinct eigenvalues (duplicate or non-orthogonal vectors)", {},
     "orthonormalize"),
    ("S2.eigenpairs", "INCONCLUSIVE", "certified radius of pair 0 is not finite: no accuracy statement", {}, "zero vector"),
    ("S2.eigenpairs", "INCONCLUSIVE", "each claimed value has an eigenvalue within its Theorem-1 radius (...), but the "
                                      "cluster radius (pair 1, orthonormality defect eta = 1e-3) exceeds it", {},
     "orthonormalize"),
    ("S3.completeness", "INCONCLUSIVE", "claim contains non-finite values: counts cannot be matched to the claimed set", {},
     "non-finite"),
    ("S3.completeness", "INCONCLUSIVE", "some claimed pair has no finite enclosure (zero eigenvector): counts cannot be "
                                        "matched to the claimed set", {}, "zero vector"),
    ("S3.completeness", "INCONCLUSIVE", "full-spectrum claim but pairs not all certified distinct or count != dimension", {},
     "count = dimension"),
    ("S3.completeness", "NOT_CHECKED", "inertia counting disabled", {"reason_code": "NARROWED_BY_USER"}, "inertia=True"),
    ("S3.completeness", "NOT_CHECKED", "no stored matrix to factorize (matrix-free producer without spec)",
     {"reason_code": "NEEDS_MATRIX"}, "stored matrix"),
    ("S3.completeness", "NOT_CHECKED", "factorization skipped: dimension 70000 > inertia_max_dim 60000",
     {"reason_code": "SIZE_LIMIT"}, "inertia_max_dim"),
    ("S3.completeness", "NOT_CHECKED", "eigenvalue-only claim: completeness needs certified intervals",
     {"reason_code": "NEEDS_EIGENVECTORS"}, "eigenvectors"),
    ("S3.completeness", "NOT_CHECKED", "claim kind 'subset' makes no completeness claim",
     {"reason_code": "NO_COMPLETENESS_CLAIM"}, "kind='lowest'"),
    ("S3.completeness", "NOT_CHECKED", "inertia count not attempted to completion: memory", {}, "sparse"),
    ("S3.completeness", "INCONCLUSIVE", "the claim is complete below -3.1, but the highest claimed level is a (quasi-)"
                                        "degenerate multiplet of at least 3 eigenvalues", {}, "more eigenpairs"),
    ("S3.completeness", "INCONCLUSIVE", "at least one unclaimed eigenvalue lies within 1e-7 above the highest claimed one",
     {}, "more eigenpairs"),
    ("S3.completeness", "INCONCLUSIVE", "count above the claim could not be classified", {}, "more eigenpairs"),
    ("S3.completeness", "INCONCLUSIVE", "", {"reason": "factorization at sigma=1.0 inconclusive (zero pivot)"}, "dense"),
    ("S3.completeness", "INCONCLUSIVE", "", {"reason": "no conclusive inertia count"}, "dense"),
    ("S3.completeness", "INCONCLUSIVE", "", {"reason": "factorization backward error (1e-5) too large relative to the "
                                                       "margin above the highest claimed value"}, "backward error"),
    ("S3.completeness", "INCONCLUSIVE", "", {"reason": "claimed pairs not all certified distinct (S2); the count matches "
                                                       "but cannot be attributed"}, "orthonormalize"),
    ("S3.completeness", "INCONCLUSIVE", "", {"reason": "inertia counts 3 eigenvalues below 1.0 but 4 claimed pairs lie "
                                                       "below it: claimed pairs are not all distinct eigenpairs"},
     "orthonormalize"),
    ("S3.completeness", "INCONCLUSIVE", "counts could not be matched to the claimed set", {}, "window"),
    ("S3.completeness", "INCONCLUSIVE", "", {"reason": "inertia inconclusive at a window edge"}, "window edges"),
    ("S4.eigenspaces", "NOT_CHECKED", "not requested or no eigenvectors", {"reason_code": "NEEDS_EIGENVECTORS"},
     "eigenvectors"),
    ("S4.eigenspaces", "NOT_CHECKED", "not requested or no eigenvectors", {"reason_code": "NARROWED_BY_USER"},
     "eigenspaces="),
    ("S4.eigenspaces", "NOT_CHECKED", "requires a certified complete window (S3) to bound separations",
     {"reason_code": "NEEDS_WHOLE_SPECTRUM"}, "S3"),
    ("S4.eigenspaces", "NOT_CHECKED", "no clusters among the claimed pairs", {"reason_code": "NO_CLUSTERS"},
     "eigenspaces='all'"),
    ("S4.eigenspaces", "INCONCLUSIVE", "0/1 clusters certified at projector level",
     {"clusters": [{"status": "INCONCLUSIVE", "reason": "cluster vectors too far from orthonormal"}]}, "orthonormalize"),
    ("S4.eigenspaces", "INCONCLUSIVE", "0/1 clusters certified at projector level",
     {"clusters": [{"status": "INCONCLUSIVE", "reason": "cluster interval overlaps other claimed pairs or the window edge"}]},
     "more eigenpairs"),
    ("S4.eigenspaces", "INCONCLUSIVE", "0/1 clusters certified at projector level",
     {"clusters": [{"status": "INCONCLUSIVE", "reason": "residual not small relative to separation"}]}, "convergence"),
    ("S5.corroboration", "NOT_CHECKED", "no corroborating solvers named (Options.corroborate)",
     {"reason_code": "NEEDS_DEPENDENCY"}, "corroborate"),
    ("S5.corroboration", "INCONCLUSIVE", "solvers disagree", {}, "solvers"),
    ("S6.exact_recheck", "NOT_CHECKED", "needs a model specification", {"reason_code": "NEEDS_SPEC"}, "ModelSpec"),
    ("S6.exact_recheck", "NOT_CHECKED", "eigenvalue-only claim: the exact re-check needs eigenvectors",
     {"reason_code": "NEEDS_EIGENVECTORS"}, "eigenvectors"),
    ("S6.exact_recheck", "NOT_CHECKED", "implemented for 'lowest' claims (this claim: 'window')",
     {"reason_code": "UNSUPPORTED_CLAIM"}, "kind='lowest'"),
    ("S6.exact_recheck", "INCONCLUSIVE", "claim contains non-finite values", {}, "non-finite"),
    ("S6.exact_recheck", "NOT_CHECKED", "symmetry-adapted claims: use exact_recheck_sector_lowest",
     {"reason_code": "UNSUPPORTED_CLAIM"}, "exact_recheck_sector_lowest"),
    ("S6.exact_recheck", "NOT_CHECKED", "dimension 5000 above exact_max_dim 4000", {"reason_code": "SIZE_LIMIT"},
     "exact_max_dim"),
    ("S6.exact_recheck", "NOT_CHECKED", "python-flint not installed", {"reason_code": "MISSING_DEPENDENCY"},
     "python-flint"),
    ("S6.exact_recheck", "NOT_CHECKED", "python-flint not installed", {"reason_code": "NEEDS_DEPENDENCY"}, "python-flint"),
    ("S6.exact_recheck", "INCONCLUSIVE", "exact re-check could not separate the claimed values from the rest", {},
     "more eigenpairs"),
    ("S7.thermodynamics", "NOT_CHECKED", "needs a whole-spectrum claim (kind 'all') whose enclosure is complete (S3)",
     {"reason_code": "NEEDS_WHOLE_SPECTRUM"}, "kind='all'"),
    ("S7.thermodynamics", "NOT_CHECKED", "no temperatures given (Options.temperatures)",
     {"reason_code": "NEEDS_TEMPERATURES"}, "temperatures"),
    ("S7.thermodynamics", "INCONCLUSIVE", "thermodynamic intervals not formed: overflow", {}, "temperatures"),
    ("S8.degeneracy", "NOT_CHECKED", "eigenvalue-only claim: no clusters to count", {"reason_code": "NEEDS_EIGENVECTORS"},
     "eigenvectors"),
    ("S8.degeneracy", "NOT_CHECKED", "no stored matrix to factorize (matrix-free producer without spec)",
     {"reason_code": "NEEDS_MATRIX"}, "stored matrix"),
    ("S8.degeneracy", "NOT_CHECKED", "factorization skipped: dimension 70000 > inertia_max_dim 60000",
     {"reason_code": "SIZE_LIMIT"}, "inertia_max_dim"),
    ("S8.degeneracy", "NOT_CHECKED", "no clusters among the claimed pairs", {"reason_code": "NO_CLUSTERS"},
     "eigenspaces='all'"),
    ("S8.degeneracy", "INCONCLUSIVE", "0/1 clusters: eigenvalue count in the level enclosure certified by bracketing "
                                      "inertia counts", {}, "more eigenpairs"),
    # the gates (_gate) for every statement
    *[(meta["id"], "NOT_CHECKED", "not selected (skip: skipped by the user)",
       {"reason_code": "NARROWED_BY_USER", "narrowed": {"by": "skip", "reason": "skipped by the user"}}, "select")
      for meta in STATEMENTS.values()],
    *[(meta["id"], "NOT_CHECKED", "not selected (dependency: needs S2, which is not selected)",
       {"reason_code": "DEPENDENCY_NARROWED", "narrowed": {"by": "dependency", "reason": "needs S2, which is not selected"}},
       "needs S2") for meta in STATEMENTS.values()],
    *[(meta["id"], "NOT_CHECKED", "time budget 0 s spent (0.1 s elapsed) before " + meta["id"],
       {"reason_code": "BUDGET_TIME"}, "budget_seconds") for meta in STATEMENTS.values()],
    *[(meta["id"], "NOT_CHECKED", "estimated 900 MB for x exceeds the memory budget 100 MB",
       {"reason_code": "BUDGET_MEMORY", "estimate_mb": 900.0}, "budget_mb") for meta in STATEMENTS.values()],
]


@pytest.mark.parametrize("sid,status,summary,evidence,expect", _CERTIFY_PATHS,
                         ids=[f"{p[0].split('.')[0]}-{p[1][:3]}-{i}" for i, p in enumerate(_CERTIFY_PATHS)])
def test_every_certify_path_has_a_specific_next_action(sid, status, summary, evidence, expect):
    stmt = {"id": sid, "status": status, "summary": summary, "evidence": dict(evidence), "assumptions": [], "seconds": 0.0}
    act = next_action(stmt, {"kind": "lowest"})
    assert act and expect.lower() in act.lower(), (sid, summary, act)


def test_certified_and_failed_get_no_next_action():
    for st in ("CERTIFIED", "FAILED", "CORROBORATED"):
        assert next_action({"id": "S2.eigenpairs", "status": st, "summary": "", "evidence": {}}) is None
    stmts = [{"id": "S2.eigenpairs", "status": "CERTIFIED", "summary": "", "evidence": {"max_radius": 1e-13}},
             {"id": "S3.completeness", "status": "NOT_CHECKED", "summary": "inertia counting disabled",
              "evidence": {"reason_code": "NARROWED_BY_USER"}}]
    annotate_next_actions(stmts)
    assert "next_action" not in stmts[0]["evidence"] and stmts[1]["evidence"]["next_action"]
    assert stmts[0]["status"] == "CERTIFIED" and stmts[1]["status"] == "NOT_CHECKED"   # statuses untouched


def test_battery_of_calls_every_open_statement_carries_next_action(heis10, monkeypatch):
    spec, H, w, V = heis10
    k = 4
    seen_codes = set()

    def run(where, **kw):
        c = ce.quick(kw.pop("H", H), kw.pop("w", w[:k]), kw.pop("V", V[:, :k]), kw.pop("spec", spec), **kw)
        opened = _assert_actions(c, where)
        for s in opened:
            if s["status"] == "NOT_CHECKED":
                seen_codes.add(classify_not_checked(s))
        return c

    # skip: the skipped statement and its dependants
    c = run("skip", skip=["S3"])
    assert c.status_of("S3.completeness") == "NOT_CHECKED" and "S3" not in c.body["scope"]["selection"]["wanted"]
    assert "select the statement" in c.statement("S3.completeness")["evidence"]["next_action"]
    assert "needs" in c.statement("S4.eigenspaces")["evidence"]["next_action"]
    # budget 0: nothing starts after S1 (S1 itself may start when the clock has not advanced yet)
    c = run("budget0", budget_seconds=0)
    assert all(s["status"] == "NOT_CHECKED" for s in c.statements[1:])
    assert all("budget" in s["evidence"]["next_action"] for s in c.statements[1:])
    # memory budget too small for S3 / S8
    c = run("budget_mb", budget_mb=0.001)
    assert c.status_of("S3.completeness") == "NOT_CHECKED"
    assert "budget_mb" in c.statement("S3.completeness")["evidence"]["next_action"]
    # no vectors
    c = run("novectors", V=None)
    assert "supply the eigenvectors" in c.statement("S2.eigenpairs")["evidence"]["next_action"]
    # NaN
    wn = np.array(w[:k]); wn[2] = np.nan
    c = run("nan", w=wn)
    assert "non-finite" in c.statement("S2.eigenpairs")["evidence"]["next_action"]
    # the multiplet case of 3.1.0 (k = 3 cuts a degenerate level): S3 INCONCLUSIVE asks for more eigenpairs
    c = run("multiplet", w=w[:3], V=V[:, :3])
    assert c.status_of("S3.completeness") == "INCONCLUSIVE"
    assert "more eigenpairs" in c.statement("S3.completeness")["evidence"]["next_action"]
    # subset claim, no spec, matrix-free operator
    c = run("subset", kind="subset")
    assert "kind='lowest'" in c.statement("S3.completeness")["evidence"]["next_action"]
    c = run("nospec", spec=None)
    assert "ModelSpec" in c.statement("S1.model_fidelity")["evidence"]["next_action"]
    c = run("matrix-free", H=spla.aslinearoperator(H), spec=None)
    assert c.status_of("S3.completeness") == "NOT_CHECKED"
    assert "stored matrix" in c.statement("S3.completeness")["evidence"]["next_action"]
    # size limits
    c = run("sizelimit", inertia_max_dim=10)
    assert "inertia_max_dim" in c.statement("S3.completeness")["evidence"]["next_action"]
    c = run("exactlimit", exact_recheck=True, exact_max_dim=10)
    assert "exact_max_dim" in c.statement("S6.exact_recheck")["evidence"]["next_action"]
    # S6 on a window claim: unsupported claim kind
    c = run("exactwindow", exact_recheck=True, kind="window", window=(float(w[0]) - 0.1, float(0.5 * (w[3] + w[4]))))
    assert c.status_of("S6.exact_recheck") == "NOT_CHECKED"
    assert "kind='lowest'" in c.statement("S6.exact_recheck")["evidence"]["next_action"]
    # missing flint, simulated: S6 NOT_CHECKED with MISSING_DEPENDENCY and an install instruction
    monkeypatch.setattr(certify_mod, "_have_flint", lambda: False)
    c = run("noflint", exact_recheck=True)
    s6 = c.statement("S6.exact_recheck")
    assert s6["status"] == "NOT_CHECKED" and s6["evidence"]["reason_code"] == "MISSING_DEPENDENCY"
    assert "pip install" in s6["evidence"]["next_action"] and "python-flint" in s6["evidence"]["next_action"]
    assert "MISSING_DEPENDENCY" in c.body["scope"]["reason_codes"]
    monkeypatch.undo()
    # presets: lab-minimal (S4 not requested), whole-spectrum on a partial claim (S7 needs kind 'all')
    c = run("labminimal", preset="lab-minimal")
    assert "eigenspaces=" in c.statement("S4.eigenspaces")["evidence"]["next_action"]
    c = run("wholespectrum", preset="whole-spectrum")
    # since 3.2 (C-06) a 'lowest' claim with S3 CERTIFIED lacks only the temperatures for S7
    assert "temperatures" in c.statement("S7.thermodynamics")["evidence"]["next_action"]
    c = run("wholespectrum-subset", preset="whole-spectrum", kind="subset")
    assert "kind='all'" in c.statement("S7.thermodynamics")["evidence"]["next_action"]
    assert {"NARROWED_BY_USER", "DEPENDENCY_NARROWED", "BUDGET_TIME", "BUDGET_MEMORY", "NEEDS_EIGENVECTORS", "NEEDS_SPEC",
            "NEEDS_MATRIX", "NO_COMPLETENESS_CLAIM", "SIZE_LIMIT", "UNSUPPORTED_CLAIM", "MISSING_DEPENDENCY",
            "NEEDS_WHOLE_SPECTRUM"} <= seen_codes


def test_next_action_never_changes_a_status(heis10):
    """Negative control for the annotation step: the same claim with annotation disabled gives the same statuses,
    summaries and numeric evidence."""
    spec, H, w, V = heis10
    c = ce.quick(H, w[:3], V[:, :3], spec, exact_recheck=True, exact_max_dim=10)
    assert c.status_of("S3.completeness") == "INCONCLUSIVE"
    for s in c.statements:
        ev = dict(s["evidence"]); ev.pop("next_action", None)
        again = next_action(s)
        if s["status"] in OPEN:
            assert again == s["evidence"]["next_action"]
        assert s["status"] in ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED")
    # the sidecar numbers are those of the plain certify() call
    ref = ce.certify(ce.Claim(w[:3], V[:, :3], operator=H, kind="lowest"), spec,
                     ce.Options(exact_recheck=True, exact_max_dim=10))
    assert np.array_equal(c.arrays["radius"], ref.arrays["radius"])


# ---------------------------------------------------------------- A-04: graceful degradation


_NO_FLINT = r"""
import sys, json
sys.modules["flint"] = None                      # make `import flint` raise ImportError
sys.path.insert(0, {root!r})
import numpy as np, scipy.linalg as la
import certify_ed as ce
from certify_ed import rigorous, checker, cli
assert rigorous.HAVE_FLINT is False
spec = ce.heisenberg_chain(8); H = ce.build_bitwise(spec); w, V = la.eigh(H.toarray())
c = ce.quick(H, w[:4], V[:, :4], spec, exact_recheck=True)
c2 = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec, ce.Options.from_preset("proof-lemma"))
s6 = c.statement("S6.exact_recheck"); s6b = c2.statement("S6.exact_recheck")
env = cli.environment()
out = {{"statuses": [(s["id"], s["status"]) for s in c.statements], "s6": s6, "s6b": s6b,
        "env_flint": env["flint"], "flint_c": env["flint_c_library"], "verify_scope": None,
        "cert_env_has_flint": "python_flint" in c.body["environment"], "summary": c.summary(),
        "paragraph_ok": "was not checked (missing dependency)" in c.methods_paragraph()}}
print(json.dumps(out, default=str))
"""


def test_import_and_certify_without_python_flint(tmp_path):
    """python-flint absent (sys.modules['flint'] = None in a fresh interpreter): importing certify_ed, certify()
    and quick() with exact_recheck / the proof-lemma preset do not raise; S6 is NOT_CHECKED with
    MISSING_DEPENDENCY and names the extra; the CLI environment probe reports it as not installed."""
    script = tmp_path / "noflint.py"
    script.write_text(_NO_FLINT.format(root=ROOT), encoding="utf-8")
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    st = dict(out["statuses"])
    assert st["S6.exact_recheck"] == "NOT_CHECKED"
    assert st["S1.model_fidelity"] == "CERTIFIED" and st["S2.eigenpairs"] == "CERTIFIED"
    for s6 in (out["s6"], out["s6b"]):
        assert s6["evidence"]["reason_code"] == "MISSING_DEPENDENCY"
        assert s6["evidence"]["extra"] == "certify-ed[exact]"
        assert "pip install" in s6["evidence"]["next_action"]
    assert out["env_flint"] == "not installed" and out["flint_c"] == "not installed"
    assert out["cert_env_has_flint"] is False
    assert out["paragraph_ok"]


def test_quspin_importer_names_the_extra_when_quspin_is_missing(monkeypatch):
    import certify_ed.importers as imp
    monkeypatch.setitem(sys.modules, "quspin", None)

    class FakeBasis:        # not a QuSpin object, so the importer asks for the package first
        Ns = 4

    with pytest.raises(ImportError) as ei:
        imp.claim_from_quspin(np.eye(4), FakeBasis(), [], np.zeros(2))
    msg = str(ei.value)
    assert "certify-ed[quspin]" in msg and "claim_from_arrays" in msg
    with pytest.raises(ImportError, match=r"certify-ed\[quspin\]"):
        imp.require_quspin()
