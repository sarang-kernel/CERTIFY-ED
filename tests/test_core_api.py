"""A-01: the one-call API `quick`, the one-sentence summary and the Jupyter representations."""
import html as _html

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce


def _statuses(cert):
    return [(s["id"], s["status"]) for s in cert.statements]


@pytest.fixture(scope="module")
def heis10():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def test_quick_is_certify_on_the_same_claim(heis10):
    spec, H, w, V = heis10
    c = ce.quick(H, w[:4], V[:, :4], spec)
    ref = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec)
    assert _statuses(c) == _statuses(ref)
    for sid in ("S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces"):
        assert c.status_of(sid) == "CERTIFIED", c.report()
    # the statements and their numbers are the same records
    for a, b in zip(c.statements, ref.statements):
        assert a["summary"] == b["summary"]
        ea = {k: v for k, v in a["evidence"].items() if k != "next_action"}
        eb = {k: v for k, v in b["evidence"].items() if k != "next_action"}
        assert ea.keys() == eb.keys()


def test_summary_is_one_sentence(heis10):
    spec, H, w, V = heis10
    c = ce.quick(H, w[:4], V[:, :4], spec)
    s = c.summary()
    assert "\n" not in s and s.endswith(".") and s.count(". ") == 0
    assert "4 of 5 statements CERTIFIED" in s or "5 of 5 statements CERTIFIED" in s
    assert "max certified radius" in s and "completeness holds" in s and "gap >=" in s
    # the radius in the sentence is never displayed below the certified value
    tok = s.split("max certified radius ")[1].split(" ")[0]
    assert float(tok) >= c.max_radius()
    # the full text is still available
    rep = c.report()
    assert "CERTIFY-ED certificate" in rep and "scope" in rep and all(sid in rep for sid, _ in _statuses(c))


def test_summary_without_vectors_and_with_preset(heis10):
    spec, H, w, V = heis10
    c = ce.quick(H, w[:4], None, spec, preset="lab-minimal")
    s = c.summary()
    assert "NOT_CHECKED" in s and "preset lab-minimal" in s and s.endswith(".")
    assert c.status_of("S3.completeness") == "NOT_CHECKED"
    assert c.body["scope"]["preset"] == "lab-minimal"


def test_jupyter_representations(heis10):
    spec, H, w, V = heis10
    c = ce.quick(H, w[:4], V[:, :4], spec)
    md = c._repr_markdown_()
    html = c._repr_html_()
    assert md.startswith("**") and "| statement | status | evidence |" in md
    assert "<table" in html and "</table>" in html and c.summary() in _html.unescape(html)
    for sid, st in _statuses(c):
        assert f"`{sid}`" in md and st in md
        assert f"<code>{sid}</code>" in html
    # open statements show their next action in the table
    c2 = ce.quick(H, w[:4], None, spec)
    assert "next:" in c2._repr_markdown_() and "next:" in c2._repr_html_()


def test_quick_nan_input_does_not_raise_and_is_inconclusive(heis10):
    spec, H, w, V = heis10
    wn = np.array(w[:4], dtype=float)
    wn[1] = np.nan
    c = ce.quick(H, wn, V[:, :4], spec)
    assert c.status_of("S2.eigenpairs") == "INCONCLUSIVE"
    assert c.status_of("S3.completeness") == "INCONCLUSIVE"
    assert all(s["status"] != "CERTIFIED" for s in c.statements if s["id"] != "S1.model_fidelity")
    for s in c.statements:
        if s["status"] in ("INCONCLUSIVE", "NOT_CHECKED"):
            assert s["evidence"]["next_action"]
    assert "INCONCLUSIVE" in c.summary()
    Vn = V[:, :4].copy()
    Vn[0, 2] = np.inf
    c2 = ce.quick(H, w[:4], Vn, spec)
    assert c2.status_of("S2.eigenpairs") == "INCONCLUSIVE"


def test_quick_non_hermitian_input_does_not_raise(heis8):
    """A non-Hermitian producer matrix: the statements concern its Hermitian part (recorded), S1 fails against
    the spec, and quick() says exactly what certify() says."""
    spec, H, w, V = heis8
    Hn = H.toarray().copy()
    Hn[0, 5] += 0.3
    c = ce.quick(Hn, w[:4], V[:, :4], spec)
    ref = ce.certify(ce.Claim(w[:4], V[:, :4], operator=Hn, kind="lowest"), spec)
    assert _statuses(c) == _statuses(ref)
    assert c.body["hermitian_part"]["max_abs_asymmetry"] == pytest.approx(0.3)
    assert c.status_of("S1.model_fidelity") == "FAILED"
    # without a spec nothing ties the matrix to a model: S1 is NOT_CHECKED with a next action
    c3 = ce.quick(Hn, w[:4], V[:, :4])
    assert c3.status_of("S1.model_fidelity") == "NOT_CHECKED"
    assert "ModelSpec" in c3.statement("S1.model_fidelity")["evidence"]["next_action"]


def test_quick_keyword_routing(heis10):
    spec, H, w, V = heis10
    # Claim fields and Options fields through the same keyword list; spec as a dictionary
    c = ce.quick(H, w[:4], V[:, :4], spec.to_dict(), description="unit test", accuracy_target=1e-6,
                 inertia_method="dense", eigenspaces="none")
    assert c.body["claim"]["description"] == "unit test"
    assert c.body["claim"]["accuracy_target"] == 1e-6
    assert c.body["scope"]["options"]["eigenspaces"] == "none"
    assert c.status_of("S2.eigenpairs") == "CERTIFIED"
    with pytest.raises(TypeError, match="unknown keyword"):
        ce.quick(H, w[:4], V[:, :4], spec, not_an_option=1)
    with pytest.raises(TypeError, match="spec must be"):
        ce.quick(H, w[:4], V[:, :4], spec="heisenberg")
    # a window claim
    cw = ce.quick(H, w[:4], V[:, :4], spec, kind="window", window=(float(w[0]) - 0.1, float(0.5 * (w[3] + w[4]))))
    assert cw.body["claim"]["kind"] == "window"
    assert cw.status_of("S3.completeness") == "CERTIFIED", cw.report()


def test_quick_without_operator_uses_the_spec(heis8):
    spec, H, w, V = heis8
    c = ce.quick(None, w[:4], V[:, :4], spec)
    assert c.status_of("S2.eigenpairs") == "CERTIFIED"
    with pytest.raises(ValueError, match="no operator"):
        ce.quick(None, w[:4], V[:, :4])
