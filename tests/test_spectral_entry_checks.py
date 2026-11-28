"""The certificate entry of M5 and the record check of M6 (3.3.1.dev0).

spectral_bounds_from_certificate treats spectrum_lower=None as not given, so the certificate's proved bound on the
lowest eigenvalue is used, as without the keyword; distribution_bounds refuses an M5 record that was changed after its
record_sha256 was written, unless check_record=False, and then marks its output; the driver starts its scripts
without PYTHONSAFEPATH.
"""
import copy
import json

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed import spectral_bounds as sb
from certify_ed.correlations import site_operator


@pytest.fixture(scope="module")
def ring6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).tocsr()
    O = sp.csr_matrix(sum(np.cos(np.pi * j) * site_operator(spec, "z", [j]) for j in range(6)) * np.sqrt(2.0 / 6))
    w, V = np.linalg.eigh(H.toarray())
    cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), None,
                      ce.Options(store_eigenvectors=True, eigenspaces="all"))
    return H, O, cert


def test_spectrum_lower_none_is_the_same_as_no_keyword(ring6):
    H, O, cert = ring6
    plain = sb.spectral_bounds_from_certificate(cert, H, O, K=4)
    none = sb.spectral_bounds_from_certificate(cert, H, O, K=4, spectrum_lower=None)
    assert none["record_sha256"] == plain["record_sha256"]
    assert none["chebyshev"]["proved_bounds"]["lower"] is not None
    assert none["chebyshev"]["spectrum_source"].startswith("lower: lowest eigenvalue from the certificate")


def test_a_number_replaces_the_certificates_bound(ring6):
    H, O, cert = ring6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=4, spectrum_lower=-10.0)
    ch = rec["chebyshev"]
    assert "proved_bounds" not in ch and ch["declared_bounds"]["lower"] == -10.0
    assert ch["spectrum_source"].startswith("lower: Gershgorin discs of the stored H (not below the declared bound)")


def test_an_unchanged_record_gives_m6(ring6):
    H, O, cert = ring6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=6)
    out = sb.distribution_bounds(rec, [-2.0, 0.0])
    st = {s["id"]: s["status"] for s in out["statements"]}
    assert st["M6.distribution_bounds"] == "CERTIFIED"


def test_a_changed_interval_is_refused(ring6):
    H, O, cert = ring6
    rec = copy.deepcopy(sb.spectral_bounds_from_certificate(cert, H, O, K=6))
    lo, hi = rec["chebyshev"]["moments"][2]["interval"]
    rec["chebyshev"]["moments"][2]["interval"] = [lo, hi + 0.5]
    with pytest.raises(ValueError, match="changed after M5"):
        sb.distribution_bounds(rec, [0.0])
    out = sb.distribution_bounds(rec, [0.0], check_record=False)          # taken as given, on purpose
    assert out["distribution"]["rows"]


@pytest.mark.parametrize("key,delta", [("center", 1e-3), ("scale", 1e-3)])
def test_a_changed_chebyshev_map_is_refused(ring6, key, delta):
    H, O, cert = ring6
    rec = copy.deepcopy(sb.spectral_bounds_from_certificate(cert, H, O, K=6))
    rec["chebyshev"][key] += delta
    with pytest.raises(ValueError, match="changed after M5"):
        sb.distribution_bounds(rec, [0.0])


def test_intervals_taken_as_given_are_marked_and_the_mark_is_kept(ring6):
    H, O, cert = ring6
    rec = sb.spectral_bounds_from_certificate(cert, H, O, K=6)
    plain = sb.distribution_bounds(rec, [0.0])
    assert plain["distribution"]["moment_intervals"] == sb.INTERVALS_COMPARED
    m6 = {s["id"]: s for s in plain["statements"]}["M6.distribution_bounds"]
    assert sb.GIVEN_ASSUMPTION not in m6["assumptions"]
    changed = copy.deepcopy(rec)
    lo, hi = changed["chebyshev"]["moments"][2]["interval"]
    changed["chebyshev"]["moments"][2]["interval"] = [lo - 0.3, hi + 0.3]
    given = sb.distribution_bounds(changed, [0.0], check_record=False)
    for out in (given, sb.distribution_bounds(json.loads(json.dumps(given)), [1.0])):   # a later call, checked
        assert out["distribution"]["moment_intervals"] == sb.INTERVALS_GIVEN
        m6 = {s["id"]: s for s in out["statements"]}["M6.distribution_bounds"]
        assert sb.GIVEN_ASSUMPTION in m6["assumptions"] and "taken as given" in m6["summary"]
    stripped = copy.deepcopy(given)
    stripped["distribution"]["moment_intervals"] = sb.INTERVALS_COMPARED            # the mark is inside the hash
    with pytest.raises(ValueError, match="changed after M5"):
        sb.distribution_bounds(stripped, [0.0])


def test_an_m6_record_without_the_mark_is_labelled_unrecorded(ring6):
    """An M6 record of 3.3.0 has no 'moment_intervals' key, and its hash was written by a call that compared nothing."""
    H, O, cert = ring6
    old = sb.distribution_bounds(sb.spectral_bounds_from_certificate(cert, H, O, K=6), [0.0])
    del old["distribution"]["moment_intervals"]
    old["statements"] = [s for s in old["statements"] if s["id"] != "M6.distribution_bounds"] + [
        {**s, "assumptions": [a for a in s["assumptions"] if a != sb.GIVEN_ASSUMPTION]}
        for s in old["statements"] if s["id"] == "M6.distribution_bounds"]
    old["record_sha256"] = sb._record_hash(old)                                      # the form 3.3.0 wrote
    out = sb.distribution_bounds(old, [1.0])
    assert out["distribution"]["moment_intervals"] == sb.INTERVALS_UNRECORDED
    m6 = {s["id"]: s for s in out["statements"]}["M6.distribution_bounds"]
    assert sb.GIVEN_ASSUMPTION in m6["assumptions"]
    again = sb.distribution_bounds(out, [1.0])
    assert again["distribution"]["moment_intervals"] == sb.INTERVALS_UNRECORDED


def test_a_record_without_its_hash_is_refused(ring6):
    H, O, cert = ring6
    rec = copy.deepcopy(sb.spectral_bounds_from_certificate(cert, H, O, K=6))
    del rec["record_sha256"]
    with pytest.raises(ValueError, match="no record_sha256"):
        sb.distribution_bounds(rec, [0.0])


def test_a_record_whose_m5_is_not_certified_needs_no_hash(ring6):
    """M6 is then NOT_CHECKED and no interval is used, so the check is not needed."""
    H, O, cert = ring6
    rec = copy.deepcopy(sb.spectral_bounds_from_certificate(cert, H, O, K=6))
    for s in rec["statements"]:
        if s["id"] == "M5.chebyshev_moments":
            s["status"] = "INCONCLUSIVE"
    del rec["record_sha256"]
    out = sb.distribution_bounds(rec, [0.0])
    st = {s["id"]: s for s in out["statements"]}
    assert st["M6.distribution_bounds"]["status"] == "NOT_CHECKED"


def test_the_driver_starts_scripts_without_the_safe_path_setting(monkeypatch):
    """The experiment scripts import their sibling modules (harness) from their own folder, which PYTHONSAFEPATH=1
    takes off the module path."""
    import subprocess
    import sys

    from certify_ed import cli
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    env = cli._script_env(PYTHONPATH="x")
    assert "PYTHONSAFEPATH" not in env and env["PYTHONPATH"] == "x"
    flag = subprocess.run([sys.executable, "-c", "import sys; print(getattr(sys.flags, 'safe_path', False))"],
                          env=env, capture_output=True, text=True).stdout.strip()
    assert flag in ("0", "False")                       # sys.flags.safe_path exists from Python 3.11 on
