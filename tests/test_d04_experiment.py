"""The comparison of experiments/d04_speedup.py: the fields that record time are matched by exact path, everything
else of the certificate body and arrays is compared, and a changed bit makes two certificates differ."""
import copy
import os
import sys

import numpy as np

import certify_ed as ce
from certify_ed import parallel as par

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import d04_speedup as d04  # noqa: E402


def test_dropped_fields_are_matched_by_exact_path():
    assert d04._dropped(("timing",))
    assert d04._dropped(("environment", "created_utc"))
    assert d04._dropped(("statements", "[]", "seconds"))
    assert d04._dropped(("statements", "[]", "evidence", "counts", "[]", "seconds"))
    # names that the substring rule of the first version would have dropped are kept
    assert not d04._dropped(("statements", "[]", "evidence", "timeout"))
    assert not d04._dropped(("scope", "options", "budget_seconds"))
    assert not d04._dropped(("claim", "timing"))
    assert not d04._dropped(("environment", "parallel"))
    dropped = set()
    kept = d04.strip({"timing": {"total_seconds": 1.0}, "statements": [{"id": "S2", "seconds": 0.5, "x": 0.25}]},
                     (), dropped)
    assert kept == {"statements": [{"id": "S2", "x": ("float", (0.25).hex())}]}
    assert dropped == {"timing", "statements/[]/seconds"}


def test_one_and_two_workers_compare_equal_and_a_changed_bit_differs():
    spec = ce.heisenberg_chain(8)
    jobs = [(d04.sector_claim(spec, nup, 3)[0], spec, ce.Options()) for nup in (2, 3, 4)]
    one, two = par.certify_many(jobs, workers=1), par.certify_many(jobs, workers=2)

    def key(certs, dropped):
        return [(d04.strip(c.body, (), dropped), d04.strip(dict(c.arrays), ("arrays",), dropped)) for c in certs]
    d1, d2 = set(), set()
    assert key(one, d1) == key(two, d2)                 # positive control
    assert "timing" in d1 and d1 == d2
    assert [d04._status(s)[:2] for s in one[0].statements][0] == ["S1.model_fidelity", one[0].statements[0]["status"]]
    arrays = dict(one[0].arrays)                        # negative controls: one ulp of a radius, one changed summary
    r = np.array(arrays["radius"], dtype=float)
    r[0] = np.nextafter(r[0], np.inf)
    assert d04.strip({**arrays, "radius": r}, ("arrays",)) != d04.strip(arrays, ("arrays",))
    body = copy.deepcopy(one[0].body)
    st = next(s for s in body["statements"] if s["id"] == "S2.eigenpairs")
    st["summary"] = st["summary"] + " "
    assert d04.strip(body, ()) != d04.strip(one[0].body, ())
