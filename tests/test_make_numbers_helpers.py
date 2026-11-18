"""Helpers of experiments/make_numbers.py and make_numbers_apps.py that the v4 manuscript macros rely on.

The scripts run on import (they read a results directory), so the tests execute only their helper
prefix (everything above the first experiment section) in a namespace pointed at an empty directory.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
MARKERS = {"make_numbers.py": "# ============================================================== environment / provenance",
           "make_numbers_apps.py": "SKIPPED = []"}


def load_helpers(script, tmp_path):
    path = os.path.join(EXP, script)
    if not os.path.exists(path):
        pytest.skip(f"{script} not in this checkout")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    marker = MARKERS[script]
    assert marker in src, f"{script}: helper prefix marker missing"
    prefix = src[: src.index(marker)]
    ns = {"__file__": path, "__name__": "helpers_under_test"}
    argv, env = sys.argv, dict(os.environ)
    sys.argv = [path, str(tmp_path / "paper")]
    os.environ["CERTIFY_RESULTS"] = str(tmp_path / "results")
    try:
        exec(compile(prefix, path, "exec"), ns)
    finally:
        sys.argv = argv
        os.environ.clear()
        os.environ.update(env)
    return ns


@pytest.fixture
def mn(tmp_path):
    return load_helpers("make_numbers.py", tmp_path)


@pytest.fixture
def ma(tmp_path):
    return load_helpers("make_numbers_apps.py", tmp_path)


# ---------------------------------------------------------------- make_numbers.py
def test_code_lines_skips_blanks_comments_and_docstrings(mn, tmp_path):
    p = tmp_path / "m.py"
    p.write_text('"""module doc\nsecond line"""\n'
                 '# a comment\n'
                 'import os  # trailing comment counts as code\n'
                 '\n'
                 'class C:\n'
                 '    """class doc"""\n'
                 '    def f(self):\n'
                 '        """f doc\n'
                 '        more"""\n'
                 '        s = """not a docstring:\n'
                 '        assigned"""\n'
                 '        return (1 +\n'
                 '                2)\n'
                 '\n'
                 '# end\n', encoding="utf-8")
    # import, class, def, s = (2 lines), return (2 lines) = 7
    assert mn["code_lines"](str(p)) == 7


def test_code_lines_on_the_exact_layer(mn):
    for f in ("checker.py", "exact_sectors.py"):
        path = os.path.join(ROOT, "certify_ed", f)
        n = mn["code_lines"](path)
        with open(path, encoding="utf-8") as fh:
            raw = sum(1 for line in fh if line.strip() and not line.strip().startswith("#"))
        assert 0 < n < raw          # docstrings are excluded, so strictly fewer than the non-blank non-comment lines


def test_parse_collected(mn):
    assert mn["parse_collected"]("tests/test_a.py::test_x\n\n510 tests collected in 14.09s\n") == 510
    assert mn["parse_collected"]("1 test collected in 0.1s") == 1
    assert mn["parse_collected"]("3/5 tests collected (2 deselected)") == 3
    assert mn["parse_collected"]("no tests collected") is None
    assert mn["parse_collected"]("") is None


def test_duplicate_macros(mn, tmp_path):
    a, b, c = tmp_path / "numbers.tex", tmp_path / "numbers_apps.tex", tmp_path / "numbers_todo.tex"
    a.write_text("\\newcommand{\\Alpha}{1}\n\\newcommand{\\Beta}{2}  % note\n", encoding="utf-8")
    b.write_text("\\providecommand{\\Alpha}{1}\n\\newcommand{\\Gamma}{3}\n\\newcommand{\\Beta}{2}\n", encoding="utf-8")
    c.write_text("% nothing\n", encoding="utf-8")
    assert mn["duplicate_macros"]([str(a), str(b), str(c)]) == {"Beta": ["numbers.tex", "numbers_apps.tex"]}
    b.write_text("\\providecommand{\\Alpha}{1}\n\\providecommand{\\Beta}{2}\n", encoding="utf-8")
    assert mn["duplicate_macros"]([str(a), str(b), str(c)]) == {}


@pytest.mark.parametrize("p1,p2,expect", [
    (True, True, False),      # F0: neither flags
    (True, False, True),      # P2 flags
    (False, True, True),      # P1 flags
    (False, False, True),     # both flag
    (True, None, None),       # F12: P2 not applicable, P1 passes -> no verdict
    (False, None, True),      # P2 not applicable but P1 flags
    (None, None, None),
])
def test_flagged_by_either(mn, p1, p2, expect):
    assert mn["flagged_by_either"]({"P1_residual_only": p1, "P2_recomputation": p2}) is expect


def test_macro_notes_are_single_line(mn):
    mn["macro"]("Foo", 3, note="first line\n   second   line")
    assert mn["MACROS"]["Foo"] == "3"
    assert mn["NOTES"]["Foo"] == "first line second line"
    mn["macro"]("Bar", "x")
    assert "Bar" not in mn["NOTES"]


# ---------------------------------------------------------------- make_numbers_apps.py
LOG = ("[  0.6s]   star: floating-point eta 0.1110430220 (published 0.1110430220)\n"
       "[  0.6s]   largest symmetry-adapted bordered blocks [70, 70, 42, 42] (unreduced max 70)  [0s]\n"
       "[  0.6s] star: eta <= 0.1110430220 (published 0.1110430220); criterion holds: True\n"
       "[184.3s]   decorated_square: floating-point eta 0.2203543174 (published 0.2203543174)\n"
       "[187.4s]   largest symmetry-adapted bordered blocks [4034, 4034, 3959, 3959] (unreduced max 8842)  [3s]\n")


def test_unreduced_block_from_log(ma):
    assert ma["unreduced_block"](LOG, "decorated_square") == 8842
    assert ma["unreduced_block"](LOG, "star") == 70
    assert ma["unreduced_block"](LOG, "cross") is None
    # a lattice whose block line is missing must not pick up the next lattice's number
    log = LOG.replace("[  0.6s]   largest symmetry-adapted bordered blocks [70, 70, 42, 42] (unreduced max 70)  [0s]\n", "")
    assert ma["unreduced_block"](log, "star") is None
    assert ma["unreduced_block"]("", "star") is None


def test_symveig_outside_count(ma):
    assert ma["symveig_outside_count"]("field (0.05) on the L = 6 chain, 58 of 64 true eigenvalues lie outside every") == (58, 64)
    assert ma["symveig_outside_count"]("nothing here") is None
    assert ma["symveig_outside_count"]("") is None


def test_symveig_audit_note_states_the_count(ma):
    path = os.path.join(ROOT, "docs", "symveig_audit.md")
    if not os.path.exists(path):
        pytest.skip("docs/symveig_audit.md not in this checkout")
    with open(path, encoding="utf-8") as f:
        r = ma["symveig_outside_count"](f.read())
    assert r is not None and r[1] == 64 and 0 < r[0] <= 64


def test_write_pending(ma, tmp_path):
    p = tmp_path / "numbers_todo.tex"
    ma["write_pending"](str(p), {"CrownOneNamedSectorB": "E9 named-sector step pending (B)",
                                 "CrownOneNamedSectorA": "E9 named-sector step pending (A)"})
    txt = p.read_text(encoding="utf-8")
    assert txt.startswith("% generated by experiments/make_numbers_apps.py")
    lines = [line for line in txt.splitlines() if not line.startswith("%")]
    assert lines == ["\\newcommand{\\CrownOneNamedSectorA}{\\todo{E9 named-sector step pending (A)}}",
                     "\\newcommand{\\CrownOneNamedSectorB}{\\todo{E9 named-sector step pending (B)}}"]
    ma["write_pending"](str(p), {})
    txt = p.read_text(encoding="utf-8")
    assert "\\newcommand" not in txt and "no pending macros" in txt


def test_pending_and_provide_bookkeeping(ma):
    ma["macro"]("FlintCVersion", "3.3.1", note="env", provide=True)
    ma["macro"]("CrownOneRingEzero", -8.5, note=" a  note ")
    ma["pending"]("CrownOneNamedSectorA", "E9 named-sector step pending\n(3.1.0 final run)")
    assert ma["PROVIDE"] == {"FlintCVersion"}
    assert ma["M"]["CrownOneRingEzero"] == "-8.5" and ma["NOTES"]["CrownOneRingEzero"] == "a note"
    assert ma["PENDING"] == {"CrownOneNamedSectorA": "E9 named-sector step pending (3.1.0 final run)"}


def test_literature_constants_are_marked(ma):
    lit = ma["LITERATURE"]
    assert set(lit) == {"CrownHalfReducedPaper", "CrownHalfHeadlinePaper"}
    for val, src in lit.values():
        assert src.startswith("literature:") and "arXiv:2601.07800" in src
        assert val.startswith("-0.5") and val.endswith(")")


def test_thin(ma):
    assert ma["thin"](51315) == "51\\,315" and ma["thin"](999) == "999" and ma["thin"](1116229) == "1\\,116\\,229"
