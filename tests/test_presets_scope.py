"""Presets, statement selection, budgets and the scope block (docs/presets.md, docs/SCOPE_MATRIX.md)."""
import json

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import Certificate, STATEMENT_IDS
from certify_ed.presets import (DEFAULT_STATEMENTS, PRESETS, STATEMENTS, format_presets, parse_statement_list,
                                resolve_selection)
from certify_ed.scope import REASON_CODES, SCOPE_SCHEMA, classify_not_checked
from conftest import HAVE_FLINT


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _claim(heis8, kind="lowest", k=4):
    spec, H, w, V = heis8
    if kind == "all":
        return ce.Claim(w, V, operator=H, kind="all")
    return ce.Claim(w[:k], V[:, :k], operator=H, kind=kind)


def _by_id(cert):
    return {s["id"]: s for s in cert.statements}


# ------------------------------------------------------------------------------------ selection
def test_resolve_default_selection():
    sel = resolve_selection()
    assert sel.wanted == list(DEFAULT_STATEMENTS)
    assert set(sel.narrowed) == {"S5", "S6", "S7"}
    assert all(v["by"] == "default" for v in sel.narrowed.values())
    sel = resolve_selection(corroborate=True, exact_recheck=True, temperatures=True)
    assert sel.wanted == list(STATEMENTS)


def test_resolve_only_adds_dependencies_and_records_them():
    sel = resolve_selection(only=["S4"], reason="just labels")
    assert set(sel.wanted) == {"S2", "S3", "S4"}
    assert sel.implied == {"S2": "required by S4", "S3": "required by S4"} or \
        set(sel.implied) == {"S2", "S3"}
    assert sel.narrowed["S1"]["by"] == "only" and sel.narrowed["S1"]["reason"] == "just labels"


def test_resolve_skip_propagates_to_dependents():
    sel = resolve_selection(skip="S2", reason="no vectors stored")
    assert "S2" not in sel.wanted and sel.narrowed["S2"]["by"] == "skip"
    for s in ("S3", "S4", "S8"):
        assert s not in sel.wanted and sel.narrowed[s]["by"] == "dependency", s
    assert "S1" in sel.wanted


def test_resolve_presets_and_errors():
    for name, p in PRESETS.items():
        sel = resolve_selection(preset=name)
        assert set(sel.wanted) == set(p["statements"]), name
        assert sel.preset == name
    with pytest.raises(ValueError):
        resolve_selection(preset="nonexistent")
    with pytest.raises(ValueError):
        parse_statement_list("S1,S9")
    assert parse_statement_list("s1, S2.eigenpairs S3") == ["S1", "S2", "S3"]


def test_presets_table_is_data_and_printable():
    txt = format_presets()
    md = format_presets(markdown=True)
    for name in PRESETS:
        assert name in txt and f"`{name}`" in md
    for meta in STATEMENTS.values():
        assert meta["id"] in txt and meta["id"] in md
        for d in meta["needs"]:
            assert d in STATEMENTS
    assert set(REASON_CODES) >= {"NARROWED_BY_USER", "BUDGET_TIME", "BUDGET_MEMORY", "NEEDS_EIGENVECTORS",
                                 "NOT_IMPLEMENTED", "OUT_OF_SCOPE_DYNAMICS", "UNSUPPORTED_MODEL"}


# ------------------------------------------------------------------------------------ scope block
def _check_scope_block(cert, preset):
    sc = cert.body["scope"]
    assert sc["schema"] == SCOPE_SCHEMA and sc["preset"] == preset
    ids_in = {x["id"] for x in sc["in_scope"]}
    ids_nar = {x["id"] for x in sc["narrowed"]}
    ids_out = {x["quantity"] for x in sc["out_of_scope"] if x["quantity"] in STATEMENT_IDS}
    # every statement id is accounted for exactly once
    assert ids_in | ids_nar | ids_out == set(STATEMENT_IDS)
    assert not (ids_in & ids_nar) and not (ids_in & ids_out) and not (ids_nar & ids_out)
    for x in sc["out_of_scope"] + sc["narrowed"]:
        assert x["reason_code"] in REASON_CODES
    # the standing out-of-scope quantities are always declared, with reason codes
    quantities = {x["quantity"] for x in sc["out_of_scope"]}
    assert {"entanglement entropy and entanglement spectrum", "time evolution", "finite-size extrapolation"} <= quantities
    # a statement in scope actually ran
    by = _by_id(cert)
    for x in sc["in_scope"]:
        assert by[x["id"]]["status"] != "NOT_CHECKED"
    # a narrowed statement, if it appears at all, is NOT_CHECKED
    for x in sc["narrowed"]:
        assert x["id"] not in by or by[x["id"]]["status"] == "NOT_CHECKED"
    return sc


@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_preset_yields_a_valid_certificate_with_scope(tmp_path, heis8, preset):
    kind = "all" if preset in ("whole-spectrum", "full") else "lowest"
    opts = ce.Options.from_preset(preset, temperatures=[0.5, 2.0])
    if preset in ("proof-lemma", "full") and not HAVE_FLINT:
        pytest.skip("python-flint")
    cert = ce.certify(_claim(heis8, kind), heis8[0], opts)
    sc = _check_scope_block(cert, preset)
    wanted = set(PRESETS[preset]["statements"])
    for x in sc["in_scope"]:
        assert x["id"].split(".")[0] in wanted
    for x in sc["narrowed"]:
        assert x["id"].split(".")[0] not in wanted and x["by"] == "preset"
    # the preset's statements all CERTIFIED (or CORROBORATED) on a correct claim
    by = _by_id(cert)
    for short in wanted:
        sid = STATEMENTS[short]["id"]
        if preset == "full" and short == "S6":
            # S6 is implemented for 'lowest' claims only; on a whole-spectrum claim it says so with a code
            assert by[sid]["status"] == "NOT_CHECKED" and classify_not_checked(by[sid]) == "UNSUPPORTED_CLAIM"
            continue
        assert by[sid]["status"] in ("CERTIFIED", "CORROBORATED"), (preset, sid, by[sid]["summary"])
    # save / load keeps the block and the integrity hash
    path = cert.save(str(tmp_path / f"{preset}.certificate"))
    loaded = Certificate.load(path)
    assert loaded.body["scope"]["preset"] == preset
    assert "scope" in loaded.summary()


def test_narrowing_never_changes_the_status_of_statements_still_in_scope(heis8):
    spec = heis8[0]
    claim = _claim(heis8)
    ref = _by_id(ce.certify(claim, spec, ce.Options(eigenspaces="all")))
    cases = [ce.Options(skip=["S3"], narrowing_reason="factorization too costly here"),
             ce.Options(only=["S2"]),
             ce.Options(only=["S1", "S2"]),
             ce.Options(skip=["S8"]),
             ce.Options(skip=["S1"]),
             ce.Options.from_preset("labels"),
             ce.Options.from_preset("lab-minimal")]
    for opts in cases:
        cert = ce.certify(claim, spec, opts)
        sc = cert.body["scope"]
        by = _by_id(cert)
        for x in sc["in_scope"]:
            sid = x["id"]
            if sid == "S4.eigenspaces" and opts.eigenspaces != "all":
                continue             # S4 content depends on the eigenspaces option, not on narrowing
            assert by[sid]["status"] == ref[sid]["status"], (opts, sid)
            if sid in ("S2.eigenpairs",):
                assert by[sid]["evidence"]["max_radius"] == ref[sid]["evidence"]["max_radius"]
        for x in sc["narrowed"]:
            assert by.get(x["id"], {"status": "NOT_CHECKED"})["status"] == "NOT_CHECKED"


def test_skip_records_reason_and_dependency_narrowing(heis8):
    cert = ce.certify(_claim(heis8), heis8[0], ce.Options(skip=["S3"], narrowing_reason="no time for LDL"))
    by = _by_id(cert)
    assert by["S3.completeness"]["status"] == "NOT_CHECKED"
    assert by["S3.completeness"]["evidence"]["reason_code"] == "NARROWED_BY_USER"
    assert by["S4.eigenspaces"]["evidence"]["reason_code"] == "DEPENDENCY_NARROWED"
    sc = cert.body["scope"]
    nar = {x["id"]: x for x in sc["narrowed"]}
    assert nar["S3.completeness"]["reason"] == "no time for LDL" and nar["S3.completeness"]["by"] == "skip"
    assert nar["S4.eigenspaces"]["by"] == "dependency"
    assert sc["selection"]["narrowing_reason"] == "no time for LDL"
    assert by["S2.eigenpairs"]["status"] == "CERTIFIED"


def test_budget_seconds_zero_turns_statements_into_not_checked_without_exception(heis8):
    cert = ce.certify(_claim(heis8), heis8[0], ce.Options(budget_seconds=0.0))
    by = _by_id(cert)
    codes = {sid: s["evidence"].get("reason_code") for sid, s in by.items() if s["status"] == "NOT_CHECKED"}
    assert "BUDGET_TIME" in codes.values()
    for sid in ("S2.eigenpairs", "S3.completeness", "S4.eigenspaces"):
        assert by[sid]["status"] == "NOT_CHECKED" and by[sid]["evidence"]["reason_code"] == "BUDGET_TIME", sid
    assert cert.body["scope"]["budgets"]["seconds"] == 0.0
    assert any(e["code"] == "BUDGET_TIME" for e in cert.body["scope"]["budgets"]["events"])
    out = {x["quantity"]: x["reason_code"] for x in cert.body["scope"]["out_of_scope"]}
    assert out["S2.eigenpairs"] == "BUDGET_TIME"


def test_budget_mb_turns_the_factorization_into_not_checked(heis8):
    cert = ce.certify(_claim(heis8), heis8[0], ce.Options(budget_mb=0.001))
    by = _by_id(cert)
    assert by["S3.completeness"]["status"] == "NOT_CHECKED"
    assert by["S3.completeness"]["evidence"]["reason_code"] == "BUDGET_MEMORY"
    assert by["S2.eigenpairs"]["status"] == "CERTIFIED"          # no memory estimate attached to S2
    assert by["S1.model_fidelity"]["status"] == "NOT_CHECKED"     # the reference build has one
    ok = ce.certify(_claim(heis8), heis8[0], ce.Options(budget_mb=10_000))
    assert _by_id(ok)["S3.completeness"]["status"] == "CERTIFIED"


def test_reason_codes_for_missing_inputs(heis8):
    spec, H, w, V = heis8
    c = ce.certify(ce.Claim(w[:3], None, operator=H, kind="lowest"), spec)
    by = _by_id(c)
    assert by["S2.eigenpairs"]["evidence"]["reason_code"] == "NEEDS_EIGENVECTORS"
    assert classify_not_checked(by["S3.completeness"]) == "NEEDS_EIGENVECTORS"
    c = ce.certify(ce.Claim(w[:3], V[:, :3], operator=H, kind="subset"))
    by = _by_id(c)
    assert by["S1.model_fidelity"]["evidence"]["reason_code"] == "NEEDS_SPEC"
    assert by["S3.completeness"]["evidence"]["reason_code"] == "NO_COMPLETENESS_CLAIM"
    c = ce.certify(ce.Claim(w[:3], V[:, :3], operator=H, kind="lowest"), spec, ce.Options(temperatures=[1.0]))
    assert _by_id(c)["S7.thermodynamics"]["evidence"]["reason_code"] == "NEEDS_WHOLE_SPECTRUM"


def test_non_finite_claim_is_never_certified(heis8):
    """Merge decision (release/3.1.0): certify() keeps processing a non-finite claim and reports S2/S3/S6 as
    INCONCLUSIVE with a diagnosis (fix track, IMP-21/22); file importers still refuse non-finite data at the door."""
    spec, H, w, V = heis8
    bad = w[:3].copy()
    bad[1] = np.nan
    c = ce.certify(ce.Claim(bad, V[:, :3], operator=H, kind="lowest"), spec)
    st = {s["id"]: s["status"] for s in c.statements}
    assert st["S2.eigenpairs"] == "INCONCLUSIVE" and st["S3.completeness"] == "INCONCLUSIVE"
    assert all(v != "CERTIFIED" for k, v in st.items() if k != "S1.model_fidelity")


def test_cli_certify_from_bundle_and_presets(tmp_path, heis8, capsys):
    from certify_ed.cli import main
    spec, H, w, V = heis8
    d = str(tmp_path / "bundle")
    ce.write_bundle(d, w[:4], V[:, :4], H, spec, kind="lowest", meta={"preset": "lab-minimal"})
    rc = main(["certify", "--from", d])
    out = capsys.readouterr().out
    assert rc == 0 and "preset lab-minimal" in out
    cert = Certificate.load(str(tmp_path / "bundle" / "certificate.json"))
    assert cert.body["scope"]["preset"] == "lab-minimal"
    rc = main(["import", d, "--only", "S2", "--reason", "quick", "-o", str(tmp_path / "c2")])
    out = capsys.readouterr().out
    assert rc == 0 and "S1 (only)" in out
    body = json.load(open(str(tmp_path / "c2.json")))
    assert body["scope"]["selection"]["narrowing_reason"] == "quick"
    rc = main(["presets"])
    assert rc == 0 and "lab-minimal" in capsys.readouterr().out
