"""E-04 (3.2): certified parameter intervals (certify_ed.sweep; docs/proofs/E-04_parameter_intervals.md).

The tubes must contain the dense-diagonalisation eigenvalues at every random intermediate coupling (100 % of the
points); a wrong B (negative control) breaks containment; a wrong declared H1 is caught by the affinity defect;
two sector scans of the J1-J2 chain bracket the k = 0 / k = pi ground-state crossing at J2 = 1/2."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed import sweep
from certify_ed.basis import cyclic_isometry, translation

FAST = ce.Options(eigenspaces="none", degeneracy=False)


@pytest.fixture(scope="module")
def tfim_scan():
    spec_fn = lambda h: ce.tfim_chain(6, J=1.0, h=h)  # noqa: E731
    rec = sweep.certified_sweep(spec_fn, [0.5, 1.0, 1.5], k=2, options=FAST, parameter_name="h")
    return spec_fn, rec


def test_tubes_contain_dense_eigenvalues_at_random_couplings(tfim_scan):
    spec_fn, rec = tfim_scan
    assert rec["exactly_affine"] and rec["B"] >= 6.0                 # ||sum_i sigma^x_i||_2 = N = 6 exactly
    assert all(p["rank_certified"] for p in rec["points"])
    rng = np.random.default_rng(1)
    n_ok = n = 0
    for c in rng.uniform(0.5, 1.5, 60):
        w = la.eigvalsh(ce.build_bitwise(spec_fn(c)).toarray())
        for j in range(2):
            t = sweep.tube_at(rec, j, float(c))
            lo, hi = t["interval"]
            n += 1
            n_ok += int(lo <= w[j] <= hi)
            assert t["rank_certified"]
        L = sweep.next_level_lower_bound_at(rec, float(c))
        assert L is not None and L <= w[2]
    assert n_ok == n
    # at a grid point the tube is the certificate's interval
    t = sweep.tube_at(rec, 0, 1.0)
    lo1, hi1 = rec["points"][1]["lower"][0], rec["points"][1]["upper"][0]
    assert np.nextafter(lo1, -np.inf) <= t["interval"][0] <= lo1 and hi1 <= t["interval"][1] <= np.nextafter(hi1, np.inf)
    # the width grows linearly with the distance: B |c - c_k| on each side
    t = sweep.tube_at(rec, 0, 0.6)
    assert t["interval"][0] <= rec["points"][0]["lower"][0] - rec["B"] * 0.1 * (1 - 1e-12)
    assert t["interval"][0] >= rec["points"][0]["lower"][0] - rec["B"] * 0.1 * (1 + 1e-9) - 1e-12
    assert rec["record_sha256"] and all(len(p["hash"]) == 64 for p in rec["points"])


def test_negative_control_wrong_B_breaks_containment(tfim_scan):
    spec_fn, rec = tfim_scan
    bad = dict(rec)
    bad["B"] = rec["B"] / 20.0
    rng = np.random.default_rng(2)
    violations = 0
    inconsistent = 0
    for c in rng.uniform(0.55, 0.95, 40):
        w = la.eigvalsh(ce.build_bitwise(spec_fn(c)).toarray())
        try:
            lo, hi = sweep.tube_at(bad, 0, float(c))["interval"]
        except ValueError:                      # the two grid points' tubes no longer intersect: detected
            inconsistent += 1
            continue
        violations += int(not (lo <= w[0] <= hi))
    assert violations + inconsistent > 0 and violations + inconsistent == 40


def test_wrong_declared_H1_shows_as_affinity_defect():
    spec_fn = lambda h: ce.tfim_chain(4, J=1.0, h=h)  # noqa: E731
    # declare H1 = 0 (every coefficient zero): the family F_k(c) = H(c_k) is constant on each segment
    base = spec_fn(0.5)
    from certify_ed.spec import ModelSpec, Term
    h1_zero = ModelSpec(base.n_sites, base.local, [Term(t.op, [(0.0, s) for _, s in t.couplings]) for t in base.terms],
                        state_order=base.state_order)
    rec = sweep.certified_sweep(spec_fn, [0.5, 1.0], k=1, options=FAST, h1_spec=h1_zero, parameter_name="h")
    assert not rec["exactly_affine"]
    assert rec["affinity_defects"][0] >= 0.5 * 4 * (1 - 1e-12)          # ||H(1.0) - H(0.5)||_2 = 0.5 ||sum sigma^x|| = 2
    assert rec["B"] <= 1e-300                                            # add_up(0, 0) is one ulp above 0
    # the tubes of the constant family still contain the eigenvalue of F_0(c) = H(0.5) for every c (left tube), and
    # the right tube, widened by the defect, contains it too (intersection is non-empty and contains lambda_0(H(0.5)))
    w0 = la.eigvalsh(ce.build_bitwise(spec_fn(0.5)).toarray())[0]
    for c in (0.6, 0.75, 0.9):
        lo, hi = sweep.tube_at(rec, 0, c)["interval"]
        assert lo <= w0 <= hi


def test_u1_block_scan_and_leak_refusal():
    spec_fn = lambda D: ce.heisenberg_chain(6, Delta=D)  # noqa: E731
    states_fn = lambda spec: ce.magnetization_state_set(spec, 0)  # noqa: E731
    rec = sweep.certified_sweep(spec_fn, [0.5, 1.0, 2.0], k=1, options=FAST, states_fn=states_fn, parameter_name="Delta")
    assert rec["exactly_affine"] and rec["block"]["dimension"] == 20
    rng = np.random.default_rng(3)
    for c in rng.uniform(0.5, 2.0, 30):
        st = states_fn(spec_fn(c))
        w = la.eigvalsh(ce.build_bitwise(spec_fn(c), st).toarray())
        lo, hi = sweep.tube_at(rec, 0, float(c))["interval"]
        assert lo <= w[0] <= hi
    # a derivative that does not conserve S^z is refused for a block scan
    bad_fn = lambda h: ce.tfim_chain(4, J=1.0, h=h)  # noqa: E731
    with pytest.raises(ValueError, match="conserve"):
        sweep.certified_sweep(bad_fn, [0.5, 1.0], k=1, options=FAST, states_fn=lambda s: ce.magnetization_state_set(s, 0))
    with pytest.raises(ValueError, match="increasing"):
        sweep.certified_sweep(spec_fn, [1.0, 0.5], k=1, options=FAST)
    with pytest.raises(ValueError, match="outside"):
        sweep.tube_at(rec, 0, 3.0)


@pytest.fixture(scope="module")
def j1j2_sectors():
    sf = lambda J2: ce.heisenberg_chain(8, J2=J2)  # noqa: E731
    states_fn = lambda spec: ce.magnetization_state_set(spec, 0)  # noqa: E731
    recs = {}
    for j in (0, 4):
        recs[j] = sweep.certified_sweep(sf, [0.3, 0.4, 0.5, 0.6, 0.7], k=1, options=FAST, states_fn=states_fn,
                                        sector_fn=lambda spec, st, j=j: cyclic_isometry(st, translation(spec), j),
                                        parameter_name="J2")
    return sf, states_fn, recs


def test_sector_scan_tubes_contain_block_eigenvalues(j1j2_sectors):
    sf, states_fn, recs = j1j2_sectors
    from certify_ed.basis import sector_matrix
    rng = np.random.default_rng(4)
    for j, rec in recs.items():
        assert all(p["S1"] == "CERTIFIED" and p["rank_scope"] == "sector block" for p in rec["points"])
        assert rec["sector"]["dimension"] in (8, 10)
        for c in rng.uniform(0.3, 0.7, 20):
            spec = sf(float(c))
            st = states_fn(spec)
            V0 = cyclic_isometry(st, translation(spec), j)
            w = la.eigvalsh(sector_matrix(ce.build_bitwise(spec, st), V0).toarray())
            lo, hi = sweep.tube_at(rec, 0, float(c))["interval"]
            assert lo <= w[0] <= hi


def test_crossing_interval_brackets_the_majumdar_ghosh_point(j1j2_sectors):
    sf, states_fn, recs = j1j2_sectors
    x = sweep.certified_crossing_interval(recs[0], 0, recs[4], 0, samples_per_segment=100)
    rels = [r["relation"] for r in x["runs"]]
    assert rels[0] == "<" and rels[-1] == ">"                       # k = 0 below at J2 = 0.3, k = pi below at 0.7
    assert len(x["crossing_intervals"]) == 1
    a, b = x["crossing_intervals"][0]["interval"]
    assert a < 0.5 < b and b - a < 0.25
    assert x["ranks_certified"] and "crossing interval" in x["statement"]
    # every certified run agrees with dense diagonalisation at its ends and midpoint
    from certify_ed.basis import sector_matrix
    for r in x["runs"]:
        if r["relation"] == "not separated":
            continue
        for c in (r["interval"][0], sum(r["interval"]) / 2, r["interval"][1]):
            spec = sf(float(c))
            st = states_fn(spec)
            e = []
            for j in (0, 4):
                V0 = cyclic_isometry(st, translation(spec), j)
                e.append(la.eigvalsh(sector_matrix(ce.build_bitwise(spec, st), V0).toarray())[0])
            assert (e[0] < e[1]) == (r["relation"] == "<")
    # levels of one scan with certified ranks never cross: a self comparison of level 0 with itself is not separated
    same = sweep.certified_crossing_interval(recs[0], 0, None, None, samples_per_segment=10)
    assert all(r["relation"] == "not separated" for r in same["runs"]) and not same["crossing_intervals"]


def test_cli_sweep(tmp_path):
    out = tmp_path / "scan.json"
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(ce.__file__))))
    env.pop("PYTHONSAFEPATH", None)
    r = subprocess.run([sys.executable, "-m", "certify_ed.cli", "sweep", "--model", "tfim_chain", "--sizes", "4", "--param", "h",
                        "--values", "0.5,1.0", "--k", "2", "--fast", "--crossing", "0,1", "--json", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "certified sweep over h" in r.stdout and "exactly affine" in r.stdout
    rec = json.loads(out.read_text())
    assert rec["schema"] == "certify-ed/sweep/1.0" and len(rec["points"]) == 2 and rec["crossing"]["level_a"] == 0
    assert "_certificates" not in rec
