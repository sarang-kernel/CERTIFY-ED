"""experiments/compare_runs.py: the classes that absorb run-stack differences must not absorb a changed statement."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import compare_runs as cr  # noqa: E402

MUST_MISMATCH = [
    ("e7", "E7[0].status", "CERTIFIED", "FAILED"),
    ("e10", "lattices.star.unit_cosines_at_least", 32, 31),
    ("e9", "spin_half.lower_bound", -0.55015829882, -55.015829882),                       # a bound 100x off
    ("e2", "rows[0].certifier.statements.S2.eigenpairs.status", "CERTIFIED", "INCONCLUSIVE"),
    ("e1", "rows[3].P3_detail", "S1=CERTIFIED; S2=CERTIFIED; radius 1e-13", "S1=CERTIFIED; S2=FAILED; radius 1e-13"),
    ("e4", "W2.summaries.S2.eigenpairs", "4 claimed values matched", "3 claimed values matched"),   # a count in prose
    ("e5", "setup.dim", 12870, 12871),                                                      # outside the sampled rows
    ("e6", "baseline.status", "survived", "killed"),
]

ABSORBED = [
    ("e4", "W2.summaries.S2.eigenpairs", "4 matched; max radius 1.20e-13", "4 matched; max radius 8.37e-14", "text-numbers"),
    ("e2", "rows[0].producer.t_build", 0.45, 1.60, "timing"),
    ("e8", "E8[0].ratio", 0.75, 0.737, "fp-statistic"),
    ("e10", "lattices.cross.thresholds.x.beta", 6.12e9, 5.76e9, "stack-dependent"),
    ("e1", "rows[0].P3_detail", "S3=CERTIFIED", "S3=CERTIFIED; S8.degeneracy=CERTIFIED", "extended"),
    ("e5", "rows[3].correct", True, False, "sample-dependent"),
]


def test_changed_statements_are_mismatches():
    for exp, key, a, b in MUST_MISMATCH:
        assert cr.classify(key, a, b, 1e-9, exp)[0] == "MISMATCH", (exp, key)


def test_run_stack_differences_have_their_own_class():
    for exp, key, a, b, cls in ABSORBED:
        assert cr.classify(key, a, b, 1e-9, exp)[0] == cls, (exp, key)


def test_stack_dependent_bounds_beyond_tolerance_are_mismatches():
    assert cr.classify("lattices.cross.thresholds.x.beta", 1.0, 1.0 + 2 * cr.STACK_RTOL, 1e-9, "e10")[0] == "MISMATCH"
