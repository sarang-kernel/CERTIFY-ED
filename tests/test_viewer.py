"""A-08: the offline HTML certificate viewer (certify_ed.viewer, `certify-ed view`).

What is checked: every fixture certificate of the schema tests renders; the page names every statement id and
its status; every number on the page is the certificate's value rounded in the declared direction (the
`data-exact` attribute carries the exact value); a tampered certificate (body edited, or sidecar missing) shows a
visible integrity-failure banner; user strings are HTML-escaped; the page is self-contained (no scripts, no
external resources); the module is outside the computational fingerprint.
"""
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from decimal import Decimal

import numpy as np
import pytest

import certify_ed as ce
from certify_ed.certificate_v2 import STATEMENT_IDS, Certificate
from certify_ed.provenance import NON_COMPUTATIONAL
from certify_ed.spec import canonical_json
from certify_ed.viewer import integrity_report, render_html, schema_report, view_file
from test_schema_conformance import fixture_certificates, heis8  # noqa: F401  (pytest fixtures)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# every number element the viewer emits: <td|span class="num MODE" data-exact="EXACT">TEXT</...>
_NUM = re.compile(r'<(?:td|span) class="num (up|down|exact)" data-exact="([^"]*)">([^<]*)</(?:td|span)>')


def _check_numbers(html):
    """Every displayed number is >= (up), <= (down) or == (exact) the certificate's value; returns the count."""
    n = 0
    for mode, exact, text in _NUM.findall(html):
        n += 1
        if exact in ("inf", "-inf", "nan"):
            assert text == exact
            continue
        ex = Decimal(float(exact)) if ("." in exact or "e" in exact) else Decimal(int(exact))
        tx = Decimal(text)
        if mode == "up":
            assert tx >= ex, (mode, exact, text)
        elif mode == "down":
            assert tx <= ex, (mode, exact, text)
        else:
            assert float(text) == float(exact), (mode, exact, text)
    return n


def _statuses_in_page(html):
    """statement id -> status as shown in the statements table."""
    out = {}
    for m in re.finditer(r'<tr id="stmt-([^"]+)">.*?<td class="status ([A-Z_]+)">', html, flags=re.S):
        out[m.group(1)] = m.group(2)
    return out


def test_every_fixture_certificate_renders(fixture_certificates, tmp_path):  # noqa: F811
    for name, path in fixture_certificates.items():
        rep = view_file(path, str(tmp_path / f"{name}.html"))
        assert os.path.exists(rep["html"])
        assert rep["integrity"]["ok"] is True, (name, rep["integrity"])
        assert rep["schema"]["ok"] in (True, None), (name, rep["schema"])
        html = open(rep["html"], encoding="utf-8").read()
        cert = Certificate.load(path, validate_schema=False)
        shown = _statuses_in_page(html)
        for s in cert.statements:
            assert shown.get(s["id"]) == s["status"], (name, s["id"], shown)
        assert len(shown) == len(cert.statements)
        # every statement id is accounted for somewhere on the page (statements table or scope block)
        for sid in STATEMENT_IDS:
            assert sid in html, (name, sid)
        assert "INTEGRITY VERIFIED" in html and "INTEGRITY CHECK FAILED" not in html
        assert "methods paragraph unavailable" not in html
        assert _check_numbers(html) > 0
        # self-contained: no scripts, no external resources
        assert "<script" not in html.lower()
        assert not re.search(r"<(?:link|img|iframe|object|embed)\b", html, flags=re.I)
        assert 'src="' not in html


def test_enclosure_table_has_one_row_per_pair_with_radii_rounded_up(fixture_certificates):  # noqa: F811
    path = fixture_certificates["lowest_spec"]
    cert = Certificate.load(path, validate_schema=False)
    html = render_html(cert, integrity_report(cert), schema_report(cert))
    enc = cert.body["enclosures"]
    sec = html.split('id="enclosures"')[1].split("</details>")[0]
    rows = re.findall(r"<tr>(.*?)</tr>", sec.split("<tbody>")[1], flags=re.S)
    assert len(rows) == len(enc) == cert.body["claim"]["n_eigenvalues"]
    for row, e in zip(rows, enc):
        cells = re.findall(r'<td class="num (up|down|exact)" data-exact="([^"]*)">([^<]*)</td>', row)
        by_pos = {i: c for i, c in enumerate(cells)}
        assert float(by_pos[0][1]) == e["index"] and by_pos[1][0] == "exact" and float(by_pos[1][2]) == e["claimed"]
        assert by_pos[2][0] == "up" and Decimal(by_pos[2][2]) >= Decimal(e["radius"])
        assert by_pos[3][0] == "up" and Decimal(by_pos[3][2]) >= Decimal(e["existence_radius"])
        # sidecar radii (declared model: producer radius plus the certified model shift, INT-03)
        assert by_pos[4][0] == "up" and float(by_pos[4][1]) == float(cert.arrays["radius"][e["index"]])
        assert Decimal(by_pos[4][2]) >= Decimal(float(cert.arrays["radius"][e["index"]]))
        assert by_pos[5][0] == "up" and Decimal(by_pos[5][2]) >= Decimal(float(cert.arrays["existence_radius"][e["index"]]))
        assert by_pos[6][0] == "down" and Decimal(by_pos[6][2]) <= Decimal(e["lower"])
        assert by_pos[7][0] == "up" and Decimal(by_pos[7][2]) >= Decimal(e["upper"])
    # the sidecar radius is never below the producer-operator radius (the shift is added upward)
    assert all(cert.arrays["radius"][e["index"]] >= e["radius"] for e in enc)
    assert "declared model" in sec


def test_gap_and_window_numbers_rounded_in_the_safe_direction(fixture_certificates):  # noqa: F811
    cert = Certificate.load(fixture_certificates["lowest_spec"], validate_schema=False)
    s3 = cert.statement("S3.completeness")
    assert s3["status"] == "CERTIFIED"
    html = render_html(cert, integrity_report(cert), schema_report(cert))
    sec = html.split('id="completeness"')[1].split("</details>")[0]
    gap = re.search(r'Gap to the next eigenvalue, at least</th><td><span class="num down" data-exact="([^"]*)">([^<]*)<', sec)
    assert gap and float(gap.group(1)) == s3["evidence"]["gap_lower_bound"]
    assert Decimal(gap.group(2)) <= Decimal(s3["evidence"]["gap_lower_bound"])
    lo, hi = s3["evidence"]["certified_window"]
    if isinstance(hi, (int, float)):
        win = re.search(r'Certified window \(rounded inward\)</th><td>\[<span class="num exact" data-exact="-inf">-inf</span>, '
                        r'<span class="num down" data-exact="([^"]*)">([^<]*)</span>\]', sec)
        assert win and float(win.group(1)) == hi and Decimal(win.group(2)) <= Decimal(hi)


def _retarget_hash(body):
    b = copy.deepcopy(body)
    b.pop("integrity", None)
    digest = hashlib.sha256(canonical_json(b).encode()).hexdigest()
    b["integrity"] = {"algorithm": "sha256", "canonicalization": "certify-ed-canonical-json-1", "body_sha256": digest}
    return b


def test_tampered_body_shows_a_visible_integrity_failure_banner(fixture_certificates, tmp_path):  # noqa: F811
    """Negative control: a FAILED statement edited to CERTIFIED in the JSON must not render as a clean page."""
    src = fixture_certificates["planted_fail"]
    with open(src) as f:
        body = json.load(f)
    side = body["arrays"]["sidecar"]
    shutil.copy(os.path.join(os.path.dirname(src), side), tmp_path / side)
    s2 = [s for s in body["statements"] if s["id"] == "S2.eigenpairs"][0]
    assert s2["status"] == "FAILED"
    s2["status"] = "CERTIFIED"
    p = tmp_path / "tampered.json"
    p.write_text(canonical_json(body))
    rep = view_file(str(p))
    assert rep["integrity"]["ok"] is False and rep["integrity"]["body_hash_ok"] is False
    html = open(rep["html"], encoding="utf-8").read()
    assert "INTEGRITY CHECK FAILED" in html and "INTEGRITY VERIFIED" not in html
    assert "Do not rely on any number on this page" in html
    assert rep["integrity"]["recomputed_body_sha256"] != rep["integrity"]["stored_body_sha256"]
    assert rep["integrity"]["recomputed_body_sha256"] in html and rep["integrity"]["stored_body_sha256"] in html
    # the CLI writes the page and exits 1
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = tmp_path / "cli.html"
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "view", str(p), "-o", str(out)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert out.exists() and "INTEGRITY CHECK FAILED" in out.read_text(encoding="utf-8")
    assert "integrity: FAILED" in r.stdout


def test_tampered_sidecar_and_missing_sidecar_show_the_banner(fixture_certificates, tmp_path):  # noqa: F811
    src = fixture_certificates["lowest_spec"]
    with open(src) as f:
        body = json.load(f)
    side = body["arrays"]["sidecar"]
    # (a) sidecar with a radius made smaller: the array hash catches it
    with np.load(os.path.join(os.path.dirname(src), side)) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["radius"] = arrays["radius"] * 0.5
    d1 = tmp_path / "a"
    d1.mkdir()
    np.savez(d1 / side, **arrays)
    (d1 / "c.json").write_text(canonical_json(body))
    rep = view_file(str(d1 / "c.json"))
    assert rep["integrity"]["ok"] is False and rep["integrity"]["bad_arrays"] == ["radius"]
    html = open(rep["html"], encoding="utf-8").read()
    assert "INTEGRITY CHECK FAILED" in html and "radius" in html.split("INTEGRITY CHECK FAILED")[1][:300]
    # (b) sidecar missing: the page still renders, with the banner
    d2 = tmp_path / "b"
    d2.mkdir()
    (d2 / "c.json").write_text(canonical_json(body))
    rep = view_file(str(d2 / "c.json"))
    assert rep["integrity"]["ok"] is False and rep["notes"]
    assert "INTEGRITY CHECK FAILED" in open(rep["html"], encoding="utf-8").read()


def test_scope_block_tampering_shows_the_banner(fixture_certificates, tmp_path):  # noqa: F811
    """A narrowed statement silently removed from the scope block (hash retargeted): verify_scope objects."""
    src = fixture_certificates["skip_S3"]
    with open(src) as f:
        body = json.load(f)
    shutil.copy(os.path.join(os.path.dirname(src), body["arrays"]["sidecar"]), tmp_path / body["arrays"]["sidecar"])
    body["scope"]["narrowed"] = [x for x in body["scope"]["narrowed"] if x["id"] != "S3.completeness"]
    p = tmp_path / "c.json"
    p.write_text(canonical_json(_retarget_hash(body)))
    rep = view_file(str(p))
    assert rep["integrity"]["body_hash_ok"] is True and rep["integrity"]["scope_ok"] is False
    html = open(rep["html"], encoding="utf-8").read()
    assert "INTEGRITY CHECK FAILED" in html and "S3.completeness" in html


def test_user_strings_are_escaped(heis8, tmp_path):  # noqa: F811
    spec, H, w, V = heis8
    reason = "<script>alert('x')</script> & <b>bold</b>"
    cert = ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest", description="<img src=x onerror=alert(1)>"),
                      spec, ce.Options(skip=["S3"], narrowing_reason=reason))
    path = cert.save(str(tmp_path / "esc"))
    rep = view_file(path)
    html = open(rep["html"], encoding="utf-8").read()
    assert "<script" not in html.lower() and "<img" not in html.lower() and "onerror" not in html.lower().replace("onerror=alert(1)&gt;", "")
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert rep["integrity"]["ok"] is True


def test_certificate_without_scope_block_renders(fixture_certificates, tmp_path):  # noqa: F811
    """Pre-2.1 shape: no scope block. verify_scope is None (not applicable), the page says so and still renders."""
    src = fixture_certificates["lowest_spec"]
    with open(src) as f:
        body = json.load(f)
    shutil.copy(os.path.join(os.path.dirname(src), body["arrays"]["sidecar"]), tmp_path / body["arrays"]["sidecar"])
    body.pop("scope")
    p = tmp_path / "old.json"
    p.write_text(canonical_json(_retarget_hash(body)))
    rep = view_file(str(p))
    assert rep["integrity"]["scope_ok"] is None and rep["integrity"]["ok"] is True
    html = open(rep["html"], encoding="utf-8").read()
    assert "no scope block" in html and "INTEGRITY VERIFIED" in html


def test_render_without_integrity_report_says_so(fixture_certificates):  # noqa: F811
    cert = Certificate.load(fixture_certificates["lowest_spec"], validate_schema=False)
    html = render_html(cert)
    assert "INTEGRITY VERIFIED" not in html and "not checked (render_html called without an integrity report)" in html


def test_max_pairs_caps_the_enclosure_table(fixture_certificates):  # noqa: F811
    cert = Certificate.load(fixture_certificates["preset_whole-spectrum"], validate_schema=False)
    html = render_html(cert, integrity_report(cert), schema_report(cert), max_pairs=10)
    sec = html.split('id="enclosures"')[1].split("</details>")[0]
    assert len(re.findall(r"<tr>", sec.split("<tbody>")[1])) == 10
    assert "showing the first" in sec


def test_cli_view_default_output_and_exit_code(fixture_certificates, tmp_path):  # noqa: F811
    src = fixture_certificates["lowest_spec"]
    with open(src) as f:
        body = json.load(f)
    shutil.copy(src, tmp_path / "c.json")
    shutil.copy(os.path.join(os.path.dirname(src), body["arrays"]["sidecar"]), tmp_path / body["arrays"]["sidecar"])
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "view", str(tmp_path / "c.json")], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "c.html").exists() and "integrity: ok" in r.stdout


def test_viewer_is_outside_the_computational_fingerprint():
    assert "certify_ed/viewer.py" in NON_COMPUTATIONAL


@pytest.mark.parametrize("sid", STATEMENT_IDS)
def test_every_statement_id_has_a_plain_words_title(sid):
    from certify_ed.viewer import STATEMENT_TITLES
    assert sid in STATEMENT_TITLES and len(STATEMENT_TITLES[sid]) > 10
