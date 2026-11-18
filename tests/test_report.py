"""A-02: the methods paragraph. Every number in the text is a field of the certificate (through the number
registry), rounded in the safe direction; the LaTeX style carries the same numbers as macros."""
import json
import re
from decimal import Decimal

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import Certificate
from certify_ed.report import (CONCEPT_DOI, VERSION_DOI_PLACEHOLDER, methods_numbers, methods_paragraph,
                               round_decimal)

# a number token: optional sign, digits with optional fraction, optional exponent
_NUM = re.compile(r"(?<![\w.-])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.]*[A-Za-z])")


@pytest.fixture(scope="module")
def heis10():
    spec = ce.heisenberg_chain(10)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def _strip_non_numeric_parts(text):
    """Remove the fragments where digits are identifiers, not numbers: statement ids, version strings, hashes,
    the BibTeX entry, the platform string, the DOI."""
    text = text.split("@software")[0]
    text = re.sub(r"\bS[1-8]\b", "", text)                               # statement ids
    text = re.sub(r"\b[0-9a-f]{64}\b", "", text)                        # SHA-256
    text = re.sub(r"IEEE 754 binary64", "", text)
    text = re.sub(r"Run stack: [^.]*(?:\.[^.]*)*?\. Certification", "Certification", text)   # versions, platform
    text = re.sub(r"Run stack: .*?\.(?= The run stack)", "", text)
    text = re.sub(r"CERTIFY-ED \S+", "", text)                           # package version
    text = re.sub(r"schema \S+", "", text)
    text = re.sub(r"\(1e-8 \* max\(1, \|\|H\|\|\)\)", "", text)          # the default-target formula (words)
    text = re.sub(r" Not verified by the certificate: .*$", "", text, flags=re.S)   # verbatim body.not_verified
    text = re.sub(r"\\num\{([^}]*)\}", r"\1", text)                      # latex wrapper
    return text


def _numbers_in(text):
    return [m.group(0) for m in _NUM.finditer(_strip_non_numeric_parts(text))]


def _check_every_number(cert, style):
    text = methods_paragraph(cert, style=style)
    reg = methods_numbers(cert, style=style)
    displayed = {e["text"] for e in reg if e["rounding"] != "label"}
    head = text.split("% numbers of this certificate")[0]
    for e in reg:
        if e["rounding"] == "label":                       # quoted verbatim from the certificate, not a number
            assert _by_path(json.loads(json.dumps(cert.body, default=str)), e["path"]) == e["text"], e
            head = head.replace(e["text"], " ")
    # the not-verified list is quoted verbatim from the certificate body (checked here, then not scanned)
    for item in cert.body.get("not_verified", []):
        assert style == "latex" or " ".join(str(item).split()) in head
    found = _numbers_in(head)
    assert found, "no numbers found in the paragraph"
    for tok in found:
        assert tok in displayed, (tok, sorted(displayed))
    # every registry entry refers to a value in the certificate and is rounded the safe way
    body = json.loads(json.dumps(cert.body, default=str))
    for e in reg:
        if e["rounding"] == "label":
            continue
        val, txt, mode = float(e["value"]), e["text"], e["rounding"]
        shown = Decimal(txt)
        if mode == "exact":
            assert float(shown) == val
        elif mode == "up":
            assert shown >= Decimal(val), e
        elif mode == "down":
            assert shown <= Decimal(val), e
        else:
            assert abs(float(shown) - val) <= 1e-2 * abs(val) + 1e-300, e
        assert _value_in_certificate(body, cert, e["path"], val), e
    return text, reg


def _value_in_certificate(body, cert, path, val):
    if path.startswith("len:"):
        return len(_by_path(body, path[4:])) == val
    if path.startswith("count:"):
        return sum(1 for s in body["statements"] if s["status"] == path[6:]) == val
    if path.startswith("count_clusters:"):
        return sum(1 for c in _by_path(body, "statements[S4.eigenspaces].evidence.clusters")
                   if c.get("status") == "CERTIFIED") == val
    if path == "arrays.radius.max":
        return float(np.max(cert.arrays["radius"])) == val
    got = _by_path(body, path)
    return float(got) == val


def _by_path(body, path):
    cur = body
    for part in re.findall(r"statements\[[^\]]+\]|[^.]+", path):
        m = re.match(r"statements\[(.+)\]$", part)
        if m:
            cur = next(s for s in cur["statements"] if s["id"] == m.group(1))
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def test_every_number_in_plain_paragraph_is_a_certificate_field(heis10):
    spec, H, w, V = heis10
    cert = ce.quick(H, w[:4], V[:, :4], spec)
    text, reg = _check_every_number(cert, "plain")
    for needle in ("heisenberg_chain", "4 lowest eigenvalues", "IEEE 754 binary64", "underflow or overflow",
                   "run-time probe", "rounded upward", "rounded downward", CONCEPT_DOI, VERSION_DOI_PLACEHOLDER,
                   "@software", "was certified"):
        assert needle in text, needle
    names = {e["name"] for e in reg}
    assert {"certNsites", "certK", "certDim", "certMaxRadius", "certGap", "certNextLower"} <= names


def test_latex_style_emits_every_number_as_a_macro(heis10):
    spec, H, w, V = heis10
    cert = ce.quick(H, w[:4], V[:, :4], spec)
    text, reg = _check_every_number(cert, "latex")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{\\num\{([^}]*)\}\}", text))
    numeric = [e for e in reg if e["rounding"] != "label"]
    assert set(macros) == {e["name"] for e in numeric}
    for e in numeric:
        assert macros[e["name"]] == e["text"]
        assert f"\\num{{{e['text']}}}" in text.split("% numbers")[0]       # also inline
    assert "\\_" in text or "_" not in text.split("@software")[0].split("% numbers")[0]   # underscores escaped


def test_saved_certificate_paragraph_quotes_the_body_hash(heis10, tmp_path):
    spec, H, w, V = heis10
    cert = ce.quick(H, w[:4], V[:, :4], spec)
    path = cert.save(str(tmp_path / "c.certificate"))
    loaded = Certificate.load(path)
    text = loaded.methods_paragraph()
    h = loaded.body["integrity"]["body_sha256"]
    assert h in text and "body SHA-256" in text
    _check_every_number(loaded, "plain")
    _check_every_number(loaded, "latex")


def test_paragraph_for_open_and_failed_statements(heis10):
    spec, H, w, V = heis10
    # eigenvalue-only claim: NOT_CHECKED statements are stated in words with their reason code
    c = ce.quick(H, w[:4], None, spec)
    text = c.methods_paragraph()
    assert "was not checked (needs eigenvectors)" in text
    _check_every_number(c, "plain")
    # a missing ground state: S3 refuted, the count of missing eigenvalues is in the certificate
    c2 = ce.quick(H, w[1:4], V[:, 1:4], spec)
    assert c2.status_of("S3.completeness") == "FAILED"
    text2, reg2 = _check_every_number(c2, "plain")
    assert "was refuted" in text2 and "certMissing" in {e["name"] for e in reg2}
    # a window claim carries the window edges exactly
    cw = ce.quick(H, w[:4], V[:, :4], spec, kind="window", window=(float(w[0]) - 0.1, float(0.5 * (w[3] + w[4]))))
    _check_every_number(cw, "plain")


def test_round_decimal_directions():
    x = 1.234567e-13
    assert Decimal(round_decimal(x, 3, "up")) >= Decimal(x)
    assert Decimal(round_decimal(x, 3, "down")) <= Decimal(x)
    assert round_decimal(0.0) == "0"
    assert round_decimal(9.999, 3, "up") == "10"
    assert round_decimal(-3.802758426593845, 10, "down").startswith("-3.80275842")
    assert Decimal(round_decimal(-3.802758426593845, 10, "down")) <= Decimal(-3.802758426593845)
    assert round_decimal(float("inf")) == "inf"
    with pytest.raises(KeyError):
        round_decimal(1.0, 3, "sideways")


def test_cli_statement_command(heis10, tmp_path, capsys):
    from certify_ed.cli import main
    spec, H, w, V = heis10
    cert = ce.quick(H, w[:4], V[:, :4], spec)
    path = cert.save(str(tmp_path / "c.certificate"))
    assert main(["statement", path]) == 0
    out = capsys.readouterr().out
    assert "We certified" in out and "@software" in out
    target = tmp_path / "methods.tex"
    assert main(["statement", path, "--style", "latex", "-o", str(target)]) == 0
    assert "\\newcommand{\\certMaxRadius}" in target.read_text(encoding="utf-8")
