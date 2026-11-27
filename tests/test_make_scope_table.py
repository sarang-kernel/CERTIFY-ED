"""experiments/make_scope_table.py writes every character of a cell or refuses the cell.

Up to round 7 of the review tex_escape dropped every character outside ASCII that it had no spelling for, so the
moments row read "tr(P O H O)/g" for "tr(P O Hᵏ O)/g".
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import make_scope_table as mst  # noqa: E402


def test_superscript_k_and_greek_letters_are_written():
    assert mst.tex_escape("tr(P O H\u1d4f O)/g") == "tr(P O H$^{k}$ O)/g"
    assert mst.tex_escape("A x = \u03bb B x") == "A x = $\\lambda$ B x"


def test_a_character_without_a_spelling_is_refused():
    with pytest.raises(ValueError, match="U\\+2264"):
        mst.tex_escape("eta \u2264 1")


def test_the_scope_matrix_is_written_whole(tmp_path):
    """The table of docs/SCOPE_MATRIX.md has no character without a spelling, and the moments row keeps its k."""
    assert mst.main(ROOT, str(tmp_path)) == 0
    tex = (tmp_path / "tables" / "scope_matrix.tex").read_text(encoding="utf-8")
    assert tex.isascii()
    assert "H$^{k}$ O" in tex
