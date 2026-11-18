"""E9 named-sector step (review B3-2): the two pair-spin sectors Kranitz & Penc name reproduce their S = 1
value at the printed couplings. The small sector (2495 states) is computed here; the 51 315-state sector is
left to the experiment (about half a minute)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
if not os.path.isdir(EXP):
    pytest.skip("experiments/ not in this checkout", allow_module_level=True)
sys.path.insert(0, EXP)

from fractions import Fraction  # noqa: E402

import e9_pyrochlore  # noqa: E402
from certify_ed.apps.pyrochlore.spin_one import PRINTED, canonical, landscape  # noqa: E402


def test_named_sectors_are_canonical_and_the_paper_pair():
    assert e9_pyrochlore.NAMED_SECTORS == ((0, 0, 0, 4, 0, 4), (0, 4, 4, 0, 4, 4))
    for cfg in e9_pyrochlore.NAMED_SECTORS:
        assert canonical(cfg) == cfg          # the record key equals the landscape's class key
        assert all(ts % 2 == 0 and 0 <= ts <= 4 for ts in cfg)


def test_small_named_sector_reproduces_the_published_value():
    c1, c3 = float(Fraction(PRINTED["c1"])), float(Fraction(PRINTED["c3"]))
    (e, cfg, d, m), = landscape(2, c1, c3, classes={(0, 0, 0, 4, 0, 4)})
    assert cfg == [0, 0, 0, 4, 0, 4] and d == 2495 and m == 6
    assert abs(e - float(Fraction(PRINTED["published_bound"]))) < 5e-6      # -1.632985 published; -1.6329867 here
    assert e > -1.6380246                                                    # above the all-singlet eigenvalue


def test_named_sectors_record_shape(monkeypatch):
    calls = {}

    def fake_landscape(S2, c1, c3, classes=None):
        calls["args"] = (S2, c1, c3, classes)
        return [(-1.63, [0, 0, 0, 4, 0, 4], 2495, 6), (-1.62, [0, 4, 4, 0, 4, 4], 51315, 3)]

    monkeypatch.setattr(e9_pyrochlore, "landscape", fake_landscape)
    out = e9_pyrochlore.named_sectors({"c1": "26733/500000", "c3": "41363/500000"})
    assert calls["args"][0] == 2 and calls["args"][3] == {(0, 0, 0, 4, 0, 4), (0, 4, 4, 0, 4, 4)}
    assert abs(calls["args"][1] - 0.053466) < 1e-15 and abs(calls["args"][2] - 0.082726) < 1e-15
    assert [s["pair_spins"] for s in out] == [[0, 0, 0, 2, 0, 2], [0, 2, 2, 0, 2, 2]]
    assert out[1] == {"pair_spins": [0, 2, 2, 0, 2, 2], "pair_spins2": [0, 4, 4, 0, 4, 4], "energy": -1.62, "dim": 51315,
                      "multiplicity": 3}
