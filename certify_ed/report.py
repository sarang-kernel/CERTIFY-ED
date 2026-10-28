"""Paper statements from a certificate: a methods-section paragraph whose every number is a field of the
certificate, plus a BibTeX entry.

`methods_paragraph(cert, style)` writes, in prose, the model (from the spec), the claim kind and k, the status
of every statement in words, the largest certified radius (rounded upward in decimal, which is noted), the
completeness statement with its gap bound (rounded downward), the hypotheses (IEEE 754 binary64, no
underflow/overflow, the run-time probe), the run stack, the certificate's body hash and the package version,
and a BibTeX entry. In style "latex" every number is also emitted as a `\\newcommand` block the user can paste
(numbers are wrapped in siunitx `\\num{}`).

Every number passes through a registry (`methods_numbers`) that records its value, its displayed text, its
rounding direction and the certificate path it came from; tests/test_report.py parses every number in the
paragraph and matches it against that registry and the certificate. Rounding for display uses decimal
arithmetic on the exact binary value of the float (`decimal.Decimal(float)` is exact), so a value marked
"up" is never displayed smaller than the certificate's value and one marked "down" never larger.

This module reads a certificate and writes text; it cannot change a number in one
(provenance.NON_COMPUTATIONAL).
"""
from __future__ import annotations

import re
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from typing import Any, Dict, List

__all__ = ["methods_paragraph", "methods_numbers", "bibtex_entry", "CONCEPT_DOI", "VERSION_DOI_PLACEHOLDER",
           "round_decimal"]

CONCEPT_DOI = "10.5281/zenodo.20066565"
VERSION_DOI_PLACEHOLDER = "VERSION-DOI-NOT-YET-ASSIGNED"
AUTHORS = ("Vehale, Sarang", "Goel, Ritu")
TITLE = "CERTIFY-ED: certificates for exact-diagonalization results"

_STATUS_WORDS = {"CERTIFIED": "was certified", "FAILED": "was refuted", "INCONCLUSIVE": "could not be decided",
                 "NOT_CHECKED": "was not checked", "CORROBORATED": "was corroborated"}
_WHAT = {
    "S1.model_fidelity": "that the producer's operator is the declared model (S1)",
    "S2.eigenpairs": "that each claimed value lies within a certified radius of a distinct eigenvalue (S2)",
    "S3.completeness": "that no eigenvalue is missing from the claimed set (S3)",
    "S4.eigenspaces": "the eigenspace statement for the claimed clusters (S4)",
    "S5.corroboration": "the recomputation by independent solvers (S5)",
    "S6.exact_recheck": "the re-derivation in exact integer arithmetic (S6)",
    "S7.thermodynamics": "the thermodynamic intervals (S7)",
    "S8.degeneracy": "the degeneracy count of the clusters (S8)",
}


def round_decimal(x: float, sig: int = 3, mode: str = "nearest") -> str:
    """Decimal string of x with `sig` significant digits, rounded up ("up"), down ("down") or to nearest,
    applied to the exact binary value of x. Non-finite values are returned as 'inf' / '-inf' / 'nan'."""
    xf = float(x)
    if xf != xf or xf in (float("inf"), float("-inf")):
        return "nan" if xf != xf else ("inf" if xf > 0 else "-inf")
    if xf == 0.0:
        return "0"
    d = Decimal(xf)
    exp = d.adjusted()                       # position of the leading digit
    quant = Decimal(1).scaleb(exp - sig + 1)
    rounding = {"up": ROUND_CEILING, "down": ROUND_FLOOR, "nearest": ROUND_HALF_EVEN}[mode]
    q = d.quantize(quant, rounding=rounding)
    if q.adjusted() != exp:                  # carried into a new leading digit (e.g. 9.99 -> 10.0)
        q = q.quantize(Decimal(1).scaleb(q.adjusted() - sig + 1), rounding=rounding)
    s = format(q, "e") if abs(q.adjusted()) >= 5 else format(q, "f")
    if "e" in s:
        m, e = s.split("e")
        m = m.rstrip("0").rstrip(".") if "." in m else m
        s = f"{m}e{int(e)}"
    elif "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


class _Numbers:
    """Registry of every number emitted into the paragraph."""

    def __init__(self, style: str):
        self.style = style
        self.entries: List[Dict[str, Any]] = []

    def add(self, name: str, value, path: str, rounding: str = "exact", sig: int = 3, note: str = "") -> str:
        if rounding == "exact":
            if isinstance(value, bool):
                raise TypeError("booleans are words, not numbers")
            text = str(int(value)) if float(value) == int(value) and abs(float(value)) < 1e15 else repr(float(value))
        else:
            text = round_decimal(value, sig, rounding)
        self.entries.append({"name": name, "value": value, "text": text, "rounding": rounding, "path": path,
                             "note": note})
        return self._wrap(text)

    def _wrap(self, text: str) -> str:
        return f"\\num{{{text}}}" if self.style == "latex" else text

    def macros(self) -> str:
        return "\n".join(f"\\newcommand{{\\{e['name']}}}{{{self._wrap(e['text'])}}}" for e in self.entries)


def _esc(text: str, style: str) -> str:
    if style != "latex":
        return text
    return re.sub(r"([#$%&_{}])", r"\\\1", text).replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                "8": "Eight", "9": "Nine"}


def _macro_name(key: str) -> str:
    """LaTeX macro names allow letters only: 'J2' -> 'JTwo', 'h_x' -> 'HX'."""
    return "".join(_DIGIT_WORDS.get(ch, ch) for ch in key.title() if ch.isalnum()) or "X"


def _model_sentence(cert, nums: _Numbers, style: str) -> str:
    sub = cert.body.get("subject") or {}
    spec = sub.get("spec")
    if not spec:
        return _esc(sub.get("description") or "an operator supplied without a model specification", style)
    name = _esc(str(spec.get("name") or "the declared model"), style)
    n = nums.add("certNsites", spec["n_sites"], "subject.spec.n_sites")
    params = spec.get("parameters") or {}
    pieces = []
    for key in sorted(params):
        val = params[key]
        if isinstance(val, bool) or val is None:
            continue
        if isinstance(val, (int, float)):
            pieces.append(f"{_esc(key, style)} = " + nums.add(f"certParam{_macro_name(key)}", val,
                                                                f"subject.spec.parameters.{key}"))
        elif isinstance(val, str) and key in ("boundary", "convention"):
            pieces.append(f"{_esc(key, style)} {_esc(val, style)}")
    local = spec.get("local_space") or {}
    kind = local.get("kind")
    kind_txt = f", {_esc(str(local.get('spin') or ''), style)} spins" if kind == "spin" and local.get("spin") else ""
    return f"the model `{name}' with {n} sites{kind_txt}" + (f" ({', '.join(pieces)})" if pieces else "")


def _claim_sentence(cert, nums: _Numbers, style: str) -> str:
    claim = cert.body.get("claim") or {}
    k = nums.add("certK", claim.get("n_eigenvalues", 0), "claim.n_eigenvalues")
    kind = claim.get("kind")
    vec = "with eigenvectors" if claim.get("has_eigenvectors") else "without eigenvectors"
    if kind == "lowest":
        return f"the claim that the {k} lowest eigenvalues are the ones given ({vec})"
    if kind == "window":
        w = claim.get("window") or [None, None]
        a = nums.add("certWindowLo", w[0], "claim.window.0", rounding="exact")
        b = nums.add("certWindowHi", w[1], "claim.window.1", rounding="exact")
        return f"the claim that the {k} given eigenvalues are all eigenvalues in the window ({a}, {b}) ({vec})"
    if kind == "all":
        return f"the claim that the {k} given eigenpairs are the whole spectrum ({vec})"
    return f"the claim that the {k} given values are eigenvalues (a `subset' claim: no completeness is asserted; {vec})"


def _statements_text(cert, nums: _Numbers, style: str) -> List[str]:
    out = []
    for s in cert.statements:
        sid, st = s["id"], s["status"]
        what = _WHAT.get(sid, sid)
        ev = s.get("evidence") or {}
        sent = f"{what[0].upper()}{what[1:]} {_STATUS_WORDS.get(st, st.lower())}"
        if st == "NOT_CHECKED":
            code = ev.get("reason_code")
            sent += f" ({_esc(str(code).replace('_', ' ').lower(), style)})" if code else ""
        if sid == "S2.eigenpairs" and st == "CERTIFIED":
            r = cert.max_radius()
            if r is not None:
                src = ("arrays.radius.max" if cert.arrays.get("radius") is not None
                       else "statements[S2.eigenpairs].evidence.max_radius")
                sent += (f"; the largest certified radius is {nums.add('certMaxRadius', r, src, rounding='up')} "
                         f"(rounded upward to three significant digits)")
                if ev.get("sector_lift_shift") is not None:
                    sent += ", including the certified shift from the sector block to the full operator"
                tgt = (ev.get("accuracy_target") or {}).get("value")
                if tgt is not None:
                    t_txt = nums.add("certAccuracyTarget", tgt, "statements[S2.eigenpairs].evidence.accuracy_target.value",
                                     rounding="nearest")
                    sent += (f", within the accuracy target {t_txt} "
                             f"({_esc(str((ev.get('accuracy_target') or {}).get('source', '')), style)})")
        if sid == "S3.completeness":
            if st == "CERTIFIED":
                if "next_eigenvalue_lower_bound" in ev:
                    nl = nums.add("certNextLower", ev["next_eigenvalue_lower_bound"],
                                  "statements[S3.completeness].evidence.next_eigenvalue_lower_bound", rounding="down", sig=10)
                    sent += f"; exactly the claimed number of eigenvalues lie below {nl}"
                if ev.get("gap_lower_bound") is not None:
                    g = nums.add("certGap", ev["gap_lower_bound"], "statements[S3.completeness].evidence.gap_lower_bound",
                                 rounding="down")
                    sent += f", and the gap to the next eigenvalue is at least {g} (rounded downward)"
                if ev.get("certified_window") and isinstance(ev["certified_window"][0], (int, float)):
                    lo = nums.add("certWindowCertLo", ev["certified_window"][0],
                                  "statements[S3.completeness].evidence.certified_window.0", rounding="up", sig=10)
                    hi = nums.add("certWindowCertHi", ev["certified_window"][1],
                                  "statements[S3.completeness].evidence.certified_window.1", rounding="down", sig=10)
                    sent += f"; the certified window is ({lo}, {hi}) (rounded inward)"
            elif st == "FAILED":
                key = "missing_at_least" if "missing_at_least" in ev else "missing_below_top_claim_at_least"
                miss = ev.get(key)
                if miss is not None:
                    sent += (f": at least {nums.add('certMissing', miss, f'statements[S3.completeness].evidence.{key}')} "
                             f"eigenvalue(s) are missing from the claim")
        if sid == "S4.eigenspaces" and st in ("CERTIFIED", "INCONCLUSIVE") and isinstance(ev.get("clusters"), list):
            n_c = sum(1 for c in ev["clusters"] if c.get("status") == "CERTIFIED")
            n_all = len(ev["clusters"])
            sent += (f" for {nums.add('certClustersCertified', n_c, 'count_clusters:statements[S4.eigenspaces].evidence.clusters:CERTIFIED')} "
                     f"of {nums.add('certClusters', n_all, 'len:statements[S4.eigenspaces].evidence.clusters')} cluster(s)")
        out.append(sent + ".")
    return out


def _hypotheses(cert, nums: _Numbers, style: str) -> str:
    ar = cert.body.get("arithmetic") or {}
    probe = ar.get("runtime_probe") or {}
    u = ar.get("unit_roundoff")
    utxt = (f" with unit roundoff at most {nums.add('certUnitRoundoff', u, 'arithmetic.unit_roundoff', rounding='up')}"
            if u else "")
    held = probe.get("standard_model_hypotheses_hold")
    if held is True:
        ptxt = "the run-time probe of the standard floating-point model passed"
    elif held is False:
        ptxt = ("the run-time probe of the standard floating-point model FAILED (" +
                _esc(", ".join(probe.get("failed", [])), style) + "), so no floating-point bound is certified")
    else:
        ptxt = "no run-time probe of the floating-point model was recorded"
    return (f"The floating-point bounds hold under the hypotheses that the arithmetic is IEEE 754 binary64 with "
            f"round-to-nearest{utxt}, that no underflow or overflow occurred in the bounded operations, and that "
            f"{ptxt}.")


def _run_stack(cert, nums: _Numbers, style: str) -> str:
    env = cert.body.get("environment") or {}
    bits = []
    for key, label in (("python", "Python"), ("numpy", "NumPy"), ("scipy", "SciPy"), ("python_flint", "python-flint")):
        if env.get(key):
            bits.append(f"{label} {_esc(str(env[key]), style)}")
    blas = env.get("blas")
    if isinstance(blas, list) and blas:
        b = blas[0]
        bits.append("BLAS " + _esc(" ".join(str(b.get(k)) for k in ("internal_api", "version") if b.get(k)), style)
                    + (f" ({b.get('num_threads')} threads)" if b.get("num_threads") else ""))
    plat = env.get("platform")
    if plat:
        bits.append(_esc(str(plat), style))
    secs = (cert.body.get("timing") or {}).get("total_seconds")
    t = (f" Certification took {nums.add('certSeconds', secs, 'timing.total_seconds', rounding='up', sig=2)} s."
         if secs else "")
    caveat = (" The run stack is recorded in the certificate; the statements do not depend on it beyond the "
              "hypotheses above, and a certificate produced on another stack carries its own hash.")
    return f"Run stack: {', '.join(bits)}.{t}{caveat}" if bits else t.strip() + caveat


def _integrity(cert, style: str) -> str:
    env = cert.body.get("environment") or {}
    ver = _esc(str(env.get("certify_ed") or "unknown"), style)
    integ = cert.body.get("integrity") or {}
    h = integ.get("body_sha256")
    schema = _esc(str(cert.body.get("schema") or ""), style)
    if h:
        return (f"The certificate (CERTIFY-ED {ver}" + (f", schema {schema}" if schema else "") +
                f") has body SHA-256 {h}.")
    return (f"The certificate was produced with CERTIFY-ED {ver}; its body hash is written when it is saved "
            f"(Certificate.save) and should be quoted from the saved file.")


def bibtex_entry(cert, style: str = "plain") -> str:
    env = cert.body.get("environment") or {}
    ver = str(env.get("certify_ed") or "unknown")
    created = str(env.get("created_utc") or "")
    year = created[:4] if re.match(r"^\d{4}", created) else "n.d."
    key = "certify_ed_" + re.sub(r"[^0-9A-Za-z]", "_", ver)
    return "\n".join([
        f"@software{{{key},",
        f"  author  = {{{' and '.join(AUTHORS)}}},",
        f"  title   = {{{TITLE}}},",
        f"  version = {{{ver}}},",
        f"  year    = {{{year}}},",
        f"  doi     = {{{CONCEPT_DOI}}},",
        f"  url     = {{https://doi.org/{CONCEPT_DOI}}},",
        f"  note    = {{Concept DOI for all versions; version DOI of this release: {VERSION_DOI_PLACEHOLDER}}}",
        "}",
    ])


def _build(cert, style: str):
    if style not in ("plain", "latex"):
        raise ValueError("style must be 'plain' or 'latex'")
    nums = _Numbers(style)
    claim = cert.body.get("claim") or {}
    sub = cert.body.get("subject") or {}
    dim = (sub.get("operator") or {}).get("dimension")
    model = _model_sentence(cert, nums, style)
    claim_txt = _claim_sentence(cert, nums, style)
    dim_txt = f" (matrix dimension {nums.add('certDim', dim, 'subject.operator.dimension')})" if dim else ""
    if sub.get("sector"):
        dim_txt += (f", a symmetry-adapted block of dimension "
                    f"{nums.add('certSectorDim', sub['sector']['dimension'], 'subject.sector.dimension')}")
    by = cert.counts_by_status()
    n_all = nums.add("certNstatements", len(cert.statements), "len:statements")
    n_cert = nums.add("certNcertified", len(by.get("CERTIFIED", [])), "count:CERTIFIED")
    intro = (f"We certified {claim_txt} for {model}{dim_txt} with CERTIFY-ED: {n_cert} of {n_all} statements "
             f"were certified.")
    para = [intro] + _statements_text(cert, nums, style) + [_hypotheses(cert, nums, style), _run_stack(cert, nums, style),
                                                            _integrity(cert, style)]
    nv = cert.body.get("not_verified") or []
    if nv:
        para.append("Not verified by the certificate: " + "; ".join(_esc(" ".join(str(x).split()), style) for x in nv) + ".")
    return " ".join(para), nums, claim


def methods_numbers(cert, style: str = "plain") -> List[Dict[str, Any]]:
    """The registry of every number the paragraph contains: name, value, displayed text, rounding, path."""
    return _build(cert, style)[1].entries


def methods_paragraph(cert, style: str = "plain") -> str:
    """Methods-section paragraph for `cert` (a Certificate), followed by a BibTeX entry; in style 'latex' also a
    `\\newcommand` block with every number (numbers inside `\\num{}` from siunitx)."""
    text, nums, _ = _build(cert, style)
    bib = bibtex_entry(cert, style)
    if style == "latex":
        macros = ("% numbers of this certificate (CERTIFY-ED); paste into the preamble, \\usepackage{siunitx}\n"
                  + nums.macros())
        return text + "\n\n" + macros + "\n\n" + bib + "\n"
    return text + "\n\n" + bib + "\n"
