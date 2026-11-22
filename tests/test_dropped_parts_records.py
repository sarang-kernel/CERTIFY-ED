"""Records of the imaginary parts that are dropped from complex claimed values (the statements concern the real
parts, as in 3.x), outside certify(): the generalized-pencil record (certify_generalized), the exact re-checks
(exact_recheck_lowest, exact_recheck_sector_lowest) and the ALPS reader (read_alps_spectrum, claim_from_alps).
Before, only certify() and the sector composition recorded them; the statements themselves are unchanged."""
import ast
import os
import warnings

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.basis import cyclic_isometry, translation
from certify_ed.checker import exact_eigenpair_radii, exact_recheck_lowest
from certify_ed.corroborate import corroborate
from certify_ed.exact_sectors import exact_recheck_sector_lowest
from certify_ed.generalized import certify_generalized
from certify_ed.io import FormatError, claim_from_alps, read_alps_spectrum
from certify_ed.schema import validate_record
from conftest import eigh_dense, needs_flint

import fixtures.ed_codes as fx


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # numpy's ComplexWarning when the real parts are taken
        yield


def _pencil(seed=4, n=12):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A + A.T
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    B = (Q * np.logspace(0, -1, n)) @ Q.T
    B = (B + B.T) / 2
    w, V = la.eigh(A, B)
    return A, B, w, V


def _statuses(rec):
    return [(s["id"], s["status"]) for s in rec["statements"]]


def _untimed(obj, drop=()):
    """The record without its timings ('seconds'), its hash and the keys in `drop` (the comparisons below)."""
    if isinstance(obj, dict):
        return {k: _untimed(v, drop) for k, v in obj.items() if k not in ("seconds", "record_sha256") + tuple(drop)}
    if isinstance(obj, list):
        return [_untimed(v, drop) for v in obj]
    return obj


def test_generalized_record_lists_the_dropped_imaginary_parts():
    A, B, w, V = _pencil()
    lam = w[:3].astype(complex)
    lam[0], lam[2] = complex(w[0], 0.125), complex(w[2], -0.375)     # the largest magnitude from a negative part
    rec = certify_generalized(A, B, lam, V[:, :3], kind="lowest")
    assert rec["claim"]["dropped_imaginary_parts"] == {"count": 2, "max_abs": 0.375}
    ref = certify_generalized(A, B, w[:3], V[:, :3], kind="lowest")
    assert "dropped_imaginary_parts" not in ref["claim"]
    assert _statuses(rec) == _statuses(ref)                          # the statements concern the real parts
    assert _untimed(rec, ("dropped_imaginary_parts",)) == _untimed(ref)
    assert validate_record(rec) in (True, None)


def test_generalized_record_without_nonzero_imaginary_parts_is_as_before():
    A, B, w, V = _pencil()
    ref = certify_generalized(A, B, w[:3], V[:, :3], kind="lowest")
    rec = certify_generalized(A, B, w[:3].astype(complex), V[:, :3], kind="lowest")    # zero imaginary parts
    assert "dropped_imaginary_parts" not in rec["claim"]
    assert _untimed(rec) == _untimed(ref)


def test_exact_recheck_records_dropped_parts_on_the_early_path():
    """A claim that is INCONCLUSIVE before any exact arithmetic (a nan value) still records the finite nonzero
    imaginary part of the other value (no flint needed on this path)."""
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = eigh_dense(H.toarray())
    lam = np.array([complex(w[0], 0.25), complex(np.nan, 0.0)])
    r = exact_recheck_lowest(spec, lam, V[:, :2], next_hint=w[2])
    assert r["status"] == "INCONCLUSIVE"
    assert r["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
    r = exact_recheck_lowest(spec, np.array([w[0], np.nan]), V[:, :2], next_hint=w[2])
    assert r["status"] == "INCONCLUSIVE" and "dropped_imaginary_parts" not in r


@needs_flint
def test_exact_recheck_lowest_records_dropped_parts_and_keeps_its_status():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = eigh_dense(H.toarray())
    k = next(i for i in range(1, 12) if w[i] - w[i - 1] > 1e-3)      # cut at a gap
    ref = exact_recheck_lowest(spec, w[:k], V[:, :k], next_hint=w[k])
    lam = w[:k].astype(complex)
    lam[0] = complex(w[0], -0.5)
    r = exact_recheck_lowest(spec, lam, V[:, :k], next_hint=w[k])
    assert r["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.5}
    assert "dropped_imaginary_parts" not in ref
    assert _untimed(r, ("dropped_imaginary_parts",)) == _untimed(ref)


@needs_flint
def test_exact_recheck_sector_lowest_records_dropped_parts_and_keeps_its_status():
    N = 8
    spec = ce.heisenberg_chain(N, J2=0.3)
    st = ce.u1_state_set(spec, N // 2)
    Hs = ce.build_bitwise(spec, st)
    T = translation(spec)
    V0 = cyclic_isometry(st, T, 0)
    w, Y = la.eigh((V0.conj().T @ Hs @ V0).toarray())
    X = (V0 @ Y[:, :1]).real
    ref = exact_recheck_sector_lowest(spec, st, [T], {0: 0}, w[:1], X, next_hint=w[1])
    r = exact_recheck_sector_lowest(spec, st, [T], {0: 0}, np.array([complex(w[0], 0.0625)]), X, next_hint=w[1])
    assert r["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.0625}
    assert "dropped_imaginary_parts" not in ref
    assert r["status"] == ref["status"]


def _alps_file(tmp_path, energies_as_stored):
    """An ALPS out.h5 with the open chain's full space (no basis table) whose energies dataset holds the given
    values (complex values stored as complex numbers)."""
    import h5py
    spec = ce.heisenberg_chain(6, boundary="open")
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    k = len(energies_as_stored)
    p = fx.write_alps_h5(str(tmp_path / "parm.task1.out.h5"),
                         [{"energies": w[:k], "quantumnumbers": {"Sz": 0.0}, "eigenvectors": V[:, :k]}])
    with h5py.File(p, "a") as f:
        del f["spectrum/sectors/0/energies"]
        f.create_dataset("spectrum/sectors/0/energies", data=np.asarray(energies_as_stored))
    return p, spec, H, w, V


def test_alps_reader_records_and_passes_the_dropped_parts(tmp_path):
    pytest.importorskip("h5py")
    spec = ce.heisenberg_chain(6, boundary="open")
    w = np.linalg.eigvalsh(ce.build_bitwise(spec).toarray())
    k = next(i for i in range(2, 12) if w[i] - w[i - 1] > 1e-3)
    lam = w[:k].astype(complex)
    lam[1] = complex(w[1], 0.25)
    p, spec, H, w, V = _alps_file(tmp_path, lam)
    s = read_alps_spectrum(p)[0]
    assert s["energies"].dtype == np.float64
    np.testing.assert_array_equal(s["energies"], lam.real)
    assert s["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
    np.testing.assert_array_equal(s["energies_as_given"], lam)
    claim = claim_from_alps(p, sector=0, operator=H, kind="lowest")
    cert = ce.certify(claim, spec, ce.Options(seed=1))
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
    assert any("imaginary parts" in line for line in cert.body["not_verified"])
    claim2 = claim_from_alps(p, sector=0, operator=H, kind="lowest", n_states=1)       # the part is not in the first
    assert "dropped_imaginary_parts" not in ce.certify(claim2, spec, ce.Options(seed=1)).body["claim"]


@pytest.mark.parametrize("stored", ["real", "complex_zero_imag"])
def test_alps_reader_without_nonzero_imaginary_parts_is_as_before(tmp_path, stored):
    pytest.importorskip("h5py")
    spec = ce.heisenberg_chain(6, boundary="open")
    w = np.linalg.eigvalsh(ce.build_bitwise(spec).toarray())
    vals = w[:3] if stored == "real" else w[:3].astype(complex)
    p, spec, H, w, V = _alps_file(tmp_path, vals)
    s = read_alps_spectrum(p)[0]
    assert s["dropped_imaginary_parts"] is None
    assert s["energies_as_given"] is s["energies"] and s["energies"].dtype == np.float64
    cert = ce.certify(claim_from_alps(p, sector=0, operator=H, kind="lowest"), spec, ce.Options(seed=1))
    assert "dropped_imaginary_parts" not in cert.body["claim"]


def test_alps_reader_refuses_a_non_finite_imaginary_part(tmp_path):
    pytest.importorskip("h5py")
    spec = ce.heisenberg_chain(6, boundary="open")
    w = np.linalg.eigvalsh(ce.build_bitwise(spec).toarray())
    lam = w[:2].astype(complex)
    lam[1] = complex(w[1], np.inf)
    p, *_ = _alps_file(tmp_path, lam)
    with pytest.raises(FormatError, match="non-finite energies"):
        read_alps_spectrum(p)


def test_exact_eigenpair_radii_records_dropped_parts():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = eigh_dense(H.toarray())
    ref = exact_eigenpair_radii(spec, w[:2], V[:, :2])
    r = exact_eigenpair_radii(spec, np.array([complex(w[0], 0.25), complex(w[1], 0.0)]), V[:, :2])
    assert r["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
    assert "dropped_imaginary_parts" not in ref
    assert {k: v for k, v in r.items() if k != "dropped_imaginary_parts"} == ref       # the radii of the real parts


def test_corroborate_records_dropped_parts_on_both_paths():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w = np.linalg.eigvalsh(H.toarray())
    lam = w[:3].astype(complex)
    lam[2] = complex(w[2], -0.125)
    r = corroborate(H, lam)
    assert r["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.125}
    assert r["status"] == corroborate(H, w[:3])["status"]
    assert "dropped_imaginary_parts" not in corroborate(H, w[:3])
    lam[0] = complex(np.nan, 0.0)                                  # the early INCONCLUSIVE path carries it too
    r = corroborate(H, lam)
    assert r["status"] == "INCONCLUSIVE" and r["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.125}


@needs_flint
def test_s6_evidence_is_unchanged_by_a_dropped_part():
    """certify passes the real parts to the exact re-check (S6), so its evidence has no record of its own and equals
    that of the real claim; the claim record holds the dropped part."""
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = eigh_dense(H.toarray())
    k = next(i for i in range(1, 12) if w[i] - w[i - 1] > 1e-3)
    lam = w[:k].astype(complex)
    lam[0] = complex(w[0], 0.5)
    opts = ce.Options(seed=1, exact_recheck=True)
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :k], operator=H, kind="lowest"), spec, opts)
    ref = ce.certify(ce.Claim(eigenvalues=w[:k], eigenvectors=V[:, :k], operator=H, kind="lowest"), spec, opts)
    s6, s6_ref = (next(s for s in c.statements if s["id"] == "S6.exact_recheck") for c in (cert, ref))
    assert s6["status"] == s6_ref["status"] == "CERTIFIED"
    assert "dropped_imaginary_parts" not in s6["evidence"]
    assert _untimed(s6) == _untimed(s6_ref)
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.5}


# every place that converts claimed values with precision.claimed_reals, and where the dropped parts are recorded
RECORDED = {
    ("certify.py", "_certify"): "the claim record of the certificate, and not_verified",
    ("checker.py", "_exact_recheck_lowest"): "exact_recheck_lowest adds it to its result",
    ("checker.py", "exact_eigenpair_radii"): "its result",
    ("compose.py", "_sector_record"): "the sector's record, and the composition's record and summary",
    ("corroborate.py", "_corroborate"): "corroborate adds it to its result",
    ("exact_sectors.py", "_exact_recheck_sector_lowest"): "exact_recheck_sector_lowest adds it to its result",
    ("generalized.py", "certify_generalized"): "the claim of the statement record",
    ("io/alps.py", "read_alps_spectrum"): "the sector dict; claim_from_alps passes the stored values to the claim",
    ("integrations/quspin.py", "certify_hamiltonian"): "passes values with a nonzero imaginary part to the claim as given",
    ("sweep.py", "certified_sweep"): "passes values with a nonzero imaginary part to the claim as given",
}
# numbers only, no record: certify passes real values; a direct call with complex values gets numpy's ComplexWarning,
# as in 3.x
NUMBERS = {
    ("enclosure.py", "enclose"): "an EnclosureSet",
    ("enclosure.py", "residuals"): "a ResidualReport",
    ("ozaki.py", "ozaki_residual"): "residuals and their error bounds",
    ("ozaki.py", "residual_report"): "residual radii",
    ("rigorous.py", "arb_residual_radii"): "residual radii",
    ("invariants.py", "spectrum_invariants"): "trace and sum of squares",
}
OPEN = {("io/bundle.py", "write_bundle"): "a bundle holds the real parts only; a manifest field is a schema decision"}


def _calls_in(node, scope, rel, found):
    for ch in ast.iter_child_nodes(node):
        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _calls_in(ch, scope + [ch.name], rel, found)
            continue
        if isinstance(ch, ast.Call):
            fn = ch.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "claimed_reals":
                found.add((rel, ".".join(scope) or "<module>"))
        _calls_in(ch, scope, rel, found)


def _claimed_reals_calls(root):
    found = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                _calls_in(tree, [], os.path.relpath(path, root).replace(os.sep, "/"), found)
    return found


def test_every_conversion_of_claimed_values_is_listed():
    """The lists above are complete: a new caller of claimed_reals must say where its dropped parts are recorded,
    or why it returns numbers only."""
    import certify_ed
    found = _claimed_reals_calls(os.path.dirname(certify_ed.__file__)) - {("precision.py", "<module>")}
    listed = set(RECORDED) | set(NUMBERS) | set(OPEN)
    assert found == listed, {"not listed": sorted(found - listed), "gone": sorted(listed - found)}
