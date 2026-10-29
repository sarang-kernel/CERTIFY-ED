"""A-07: the producer record hook `with certify_ed.record(spec, path) as rec:` and the decorator `recorded`.

A bundle written by the hook must pass `certify-ed conform`, certify (through claim_from_bundle and through the
CLI `certify --from`) to the same statuses as the direct Claim, and carry the planted errors of the producer
unchanged (negative controls: a wrong eigenvalue written through the hook is not certified)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse.linalg as sla

import certify_ed as ce
from certify_ed.conformance import check_bundle
from certify_ed.io.formats import FormatError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def chain():
    spec = ce.heisenberg_chain(8, boundary="open")
    H = ce.build_bitwise(spec)
    w, V = sla.eigsh(H, k=4, which="SA", v0=np.ones(H.shape[0]))
    o = np.argsort(w)
    return spec, H, w[o], V[:, o]


def _statuses(cert):
    return {s["id"]: s["status"] for s in cert.statements}


def test_context_manager_writes_conformant_bundle(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "run.bundle")
    with ce.record(spec, path, producer={"name": "scipy eigsh", "version": "test"}) as rec:
        rec.capture(w, V, H)
    assert rec.written and rec.path == path
    assert sorted(os.listdir(path)) == ["eigenvalues.npz", "eigenvectors.npz", "meta.json", "operator.npz", "spec.json"]
    meta = json.load(open(os.path.join(path, "meta.json")))
    assert meta["schema"] == "certify-ed/producer-bundle/1.0"
    assert meta["kind"] == "lowest"
    assert meta["producer"] == {"name": "scipy eigsh", "version": "test"}
    assert set(meta["environment"]) >= {"python", "numpy", "scipy", "certify_ed", "platform"}
    assert meta["environment"]["certify_ed"] == ce.__version__
    assert meta["timing"]["wall_seconds"] >= 0 and meta["timing"]["started_utc"] <= meta["timing"]["finished_utc"]
    rep = check_bundle(path)
    assert rep["ok"], rep["steps"]
    direct = ce.certify(ce.Claim(w, V, operator=H, kind="lowest"), spec)
    assert rep["statuses"] == _statuses(direct)
    assert rep["statuses"]["S2.eigenpairs"] == "CERTIFIED"


def test_attribute_form_and_claim_from_bundle(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "attr.bundle")
    with ce.record(spec, path) as rec:
        rec.eigenvalues = w
        rec.eigenvectors = V
        rec.operator = H
    claim, bspec, meta = ce.claim_from_bundle(path)
    assert bspec.sha256() == spec.sha256()
    assert claim.kind == "lowest"
    np.testing.assert_array_equal(claim.eigenvalues, w)
    np.testing.assert_array_equal(np.asarray(claim.eigenvectors), V)
    cert = ce.certify(claim, bspec)
    direct = ce.certify(ce.Claim(w, V, operator=H, kind="lowest"), spec)
    assert _statuses(cert) == _statuses(direct)


def test_cli_certify_from_recorded_bundle(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "cli.bundle")
    with ce.record(spec, path) as rec:
        rec.capture(w, V, H)
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "certify", "--from", path, "--only", "S1,S2",
                        "--reason", "test"], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "S2.eigenpairs" in r.stdout and "CERTIFIED" in r.stdout
    assert os.path.exists(os.path.join(path, "certificate.json"))
    # the conform subcommand accepts it too
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "conform", path], cwd=ROOT, capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr


def test_decorator_form(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "deco.bundle")

    @ce.recorded(spec, path, operator=H, producer={"name": "decorated"})
    def solve(k):
        ww, VV = sla.eigsh(H, k=k, which="SA", v0=np.ones(H.shape[0]))
        o = np.argsort(ww)
        return ww[o], VV[:, o]

    ww, VV = solve(4)
    assert solve.last_bundle == path
    claim, _, _ = ce.claim_from_bundle(path)
    np.testing.assert_array_equal(claim.eigenvalues, ww)          # written unchanged
    np.testing.assert_array_equal(np.asarray(claim.eigenvectors), VV)
    rep = check_bundle(path)
    assert rep["ok"], rep["steps"]
    assert rep["statuses"]["S2.eigenpairs"] == "CERTIFIED"
    assert json.load(open(os.path.join(path, "meta.json")))["producer"]["name"] == "decorated"


def test_decorator_three_tuple_and_bare_array(tmp_path, chain):
    spec, H, w, V = chain

    @ce.recorded(spec, str(tmp_path / "three.bundle"))
    def solve3():
        return w, V, H

    solve3()
    rep = check_bundle(str(tmp_path / "three.bundle"))
    assert rep["ok"] and rep["statuses"]["S2.eigenpairs"] == "CERTIFIED"

    @ce.recorded(spec, str(tmp_path / "bare.bundle"), kind="subset")
    def values_only():
        return w

    values_only()
    claim, _, _ = ce.claim_from_bundle(str(tmp_path / "bare.bundle"))
    assert claim.eigenvectors is None and claim.kind == "subset"


def test_exception_in_block_writes_nothing(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "failed.bundle")
    with pytest.raises(RuntimeError, match="solver"):
        with ce.record(spec, path) as rec:
            rec.capture(w, V, H)
            raise RuntimeError("solver diverged")
    assert not os.path.exists(path)
    assert not rec.written


def test_missing_eigenvalues_raise(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "empty.bundle")
    with pytest.raises(ValueError, match="no eigenvalues"):
        with ce.record(spec, path) as rec:
            rec.operator = H
    assert not os.path.exists(path)


def test_shape_and_kind_errors(tmp_path, chain):
    spec, H, w, V = chain
    with pytest.raises(ValueError, match="one column per"):
        with ce.record(spec, str(tmp_path / "t.bundle")) as rec:
            rec.capture(w, V.T, H)
    with pytest.raises(ValueError, match="window"):
        ce.record(spec, str(tmp_path / "w.bundle"), kind="window")
    with pytest.raises(ValueError, match="kind"):
        ce.record(spec, str(tmp_path / "k.bundle"), kind="best")
    with pytest.raises(TypeError, match="spec must be"):
        ce.record("heisenberg", str(tmp_path / "s.bundle"))


def test_window_kind_and_text_format(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "win.bundle")
    lo, hi = float(w[0]) - 0.1, float(w[2]) + 1e-6
    with ce.record(spec, path, kind="window", window=(lo, hi), fmt="csv", accuracy_target=1e-6,
                   description="window run") as rec:
        rec.capture(w[:3], V[:, :3], H)
    meta = json.load(open(os.path.join(path, "meta.json")))
    assert meta["kind"] == "window" and meta["window"] == [lo, hi]
    assert meta["accuracy_target"] == 1e-6 and meta["description"] == "window run"
    assert os.path.exists(os.path.join(path, "eigenvalues.csv"))
    claim, _, _ = ce.claim_from_bundle(path)
    assert claim.kind == "window" and claim.window == (lo, hi)
    rep = check_bundle(path)
    assert rep["ok"], rep["steps"]


def test_u1_basis_state_list_is_recorded(tmp_path):
    spec = ce.heisenberg_chain(6)
    states = ce.u1_state_set(spec, 3)
    H = ce.build_bitwise(spec, states)
    w, V = np.linalg.eigh(H.toarray())
    path = str(tmp_path / "u1.bundle")
    with ce.record(spec, path, kind="lowest") as rec:
        rec.capture(w[:2], V[:, :2], H, basis=states.states_asc if hasattr(states, "states_asc") else states.states)
    claim, bspec, meta = ce.claim_from_bundle(path)
    assert claim.states is not None and claim.states.size == H.shape[0]
    assert meta["state_order"] == spec.state_order
    cert = ce.certify(claim, bspec)
    assert _statuses(cert)["S1.model_fidelity"] == "CERTIFIED", cert.summary()
    assert _statuses(cert)["S2.eigenpairs"] == "CERTIFIED"


def test_spec_none_then_spec_on_the_command_line(tmp_path, chain):
    spec, H, w, V = chain
    path = str(tmp_path / "nospec.bundle")
    with ce.record(None, path) as rec:
        rec.capture(w, V, H)
    assert not os.path.exists(os.path.join(path, "spec.json"))
    claim, bspec, _ = ce.claim_from_bundle(path)
    assert bspec is None
    cert = ce.certify(claim, spec)
    assert _statuses(cert)["S2.eigenpairs"] == "CERTIFIED"


def test_negative_control_planted_eigenvalue_through_the_hook(tmp_path, chain):
    spec, H, w, V = chain
    bad = w.copy()
    bad[0] -= 1e-4
    path = str(tmp_path / "bad.bundle")
    with ce.record(spec, path) as rec:
        rec.capture(bad, V, H)
    rep = check_bundle(path)
    assert rep["ok"], "a wrong value is a property of the data, not of conformance"
    assert rep["statuses"]["S2.eigenpairs"] != "CERTIFIED"
    assert rep["statuses"]["S2.eigenpairs"] == _statuses(ce.certify(ce.Claim(bad, V, operator=H, kind="lowest"),
                                                                      spec))["S2.eigenpairs"]


def test_negative_control_planted_operator_through_the_hook(tmp_path, chain):
    spec, H, w, V = chain
    Hbad = H.tolil()
    Hbad[0, 1] += 1e-3
    Hbad[1, 0] += 1e-3
    path = str(tmp_path / "badop.bundle")
    with ce.record(spec, path) as rec:
        rec.capture(w, V, Hbad.tocsr())
    rep = check_bundle(path)
    assert rep["statuses"]["S1.model_fidelity"] != "CERTIFIED"


def test_meta_validates_against_bundle_schema(tmp_path, chain):
    pytest.importorskip("jsonschema")
    from certify_ed.schema import validate_bundle_meta
    spec, H, w, V = chain
    path = str(tmp_path / "schema.bundle")
    with ce.record(spec, path) as rec:
        rec.capture(w, V, H)
    meta = json.load(open(os.path.join(path, "meta.json")))
    spec_d = json.load(open(os.path.join(path, "spec.json")))
    assert validate_bundle_meta(meta, spec_d, require=True) is True


def test_bundle_is_a_directory_check(tmp_path, chain):
    spec, H, w, V = chain
    with pytest.raises(FormatError):
        ce.claim_from_bundle(str(tmp_path / "does-not-exist"))
