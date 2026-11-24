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


def _alps_file_with_dataset(tmp_path, data, attrs=None):
    import h5py
    p, spec, H, w, V = _alps_file(tmp_path, w_placeholder(3))
    with h5py.File(p, "a") as f:
        del f["spectrum/sectors/0/energies"]
        ds = f.create_dataset("spectrum/sectors/0/energies", data=data)
        for key, val in (attrs or {}).items():
            ds.attrs[key] = val
    return p, spec, H, w, V


def w_placeholder(k):
    spec = ce.heisenberg_chain(6, boundary="open")
    return np.linalg.eigvalsh(ce.build_bitwise(spec).toarray())[:k]


def test_alps_reader_reads_the_complex_layout_of_the_alps_archive(tmp_path):
    """The ALPS HDF5 archive stores complex numbers with a trailing dimension of 2 and the attribute '__complex__':
    read as complex (before: flattened into twice as many energies)."""
    pytest.importorskip("h5py")
    w = w_placeholder(3)
    pairs = np.stack([w, np.array([0.0, 0.25, 0.0])], axis=-1)
    p, spec, H, *_ = _alps_file_with_dataset(tmp_path, pairs, {"__complex__": 1})
    s = read_alps_spectrum(p)[0]
    np.testing.assert_array_equal(s["energies"], w)
    assert s["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
    claim = claim_from_alps(p, sector=0, operator=H, kind="lowest")
    assert ce.certify(claim, spec, ce.Options(seed=1)).body["claim"]["dropped_imaginary_parts"]["count"] == 1


def test_alps_reader_refuses_energies_of_more_than_one_dimension(tmp_path):
    pytest.importorskip("h5py")
    pairs = np.stack([w_placeholder(3), np.zeros(3)], axis=-1)      # no '__complex__' attribute: not a known layout
    p, *_ = _alps_file_with_dataset(tmp_path, pairs)
    with pytest.raises(FormatError, match="expected one dimension"):
        read_alps_spectrum(p)


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
def test_s5_and_s6_evidence_are_unchanged_by_a_dropped_part():
    """certify passes the real parts to the exact re-check (S6) and to corroborate (S5), so their evidence has no
    record of its own and equals that of the real claim; the claim record holds the dropped part."""
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = eigh_dense(H.toarray())
    k = next(i for i in range(1, 12) if w[i] - w[i - 1] > 1e-3)
    lam = w[:k].astype(complex)
    lam[0] = complex(w[0], 0.5)
    opts = ce.Options(seed=1, exact_recheck=True, corroborate=("lapack_evr", "jacobi"))
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :k], operator=H, kind="lowest"), spec, opts)
    ref = ce.certify(ce.Claim(eigenvalues=w[:k], eigenvectors=V[:, :k], operator=H, kind="lowest"), spec, opts)
    for sid, status in (("S6.exact_recheck", "CERTIFIED"), ("S5.corroboration", "CORROBORATED")):
        got, want = (next(s for s in c.statements if s["id"] == sid) for c in (cert, ref))
        assert got["status"] == want["status"] == status, sid
        assert "dropped_imaginary_parts" not in got["evidence"], sid
        assert _untimed(got) == _untimed(want), sid
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.5}


# Scope of the audit below. It covers every use of precision.claimed_reals in the package: a call or any other
# reference (passing the function as an argument counts), under any name it is imported as (from-imports with an
# alias), and any attribute named claimed_reals (as reached through a module alias of precision). precision.py, which
# defines it, is not audited. Not followed: conversions that do not go through claimed_reals (np.real, .real,
# astype(float), float(), np.asarray(..., dtype=float)), and calls through a new name that claimed_reals is assigned
# to (the assignment itself is counted where it occurs).
#
# Each use, with where its dropped parts are recorded and the function (the user itself or its public wrapper, in the
# same module) whose code must name dropped_imaginary_parts or carries_imaginary_parts.
RECORDED = {
    ("certify.py", "_certify"): ("the claim record of the certificate, and not_verified", "_certify"),
    ("checker.py", "_exact_recheck_lowest"): ("exact_recheck_lowest adds it to its result", "exact_recheck_lowest"),
    ("checker.py", "exact_eigenpair_radii"): ("its result", "exact_eigenpair_radii"),
    ("compose.py", "_sector_record"): ("the sector's record, and the composition's record and summary",
                                       "_sector_record"),
    ("corroborate.py", "_corroborate"): ("corroborate adds it to its result", "corroborate"),
    ("exact_sectors.py", "_exact_recheck_sector_lowest"): ("exact_recheck_sector_lowest adds it to its result",
                                                           "exact_recheck_sector_lowest"),
    ("generalized.py", "certify_generalized"): ("the claim of the statement record", "certify_generalized"),
    ("io/alps.py", "read_alps_spectrum"): ("the sector dict; claim_from_alps passes the stored values to the claim",
                                          "read_alps_spectrum"),
    ("integrations/quspin.py", "certify_hamiltonian"): ("passes values with a nonzero imaginary part to the claim as "
                                                        "given", "certify_hamiltonian"),
    ("sweep.py", "certified_sweep"): ("passes values with a nonzero imaginary part to the claim as given",
                                      "certified_sweep"),
}
RECORD_NAMES = ("dropped_imaginary_parts", "carries_imaginary_parts")
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


def _local_names(tree):
    """The names claimed_reals is bound to by the imports of a module (from-imports, with or without an alias)."""
    return {a.asname or a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for a in node.names if a.name == "claimed_reals"}


def _uses_in(node, scope, rel, names, found):
    for ch in ast.iter_child_nodes(node):
        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _uses_in(ch, scope + [ch.name], rel, names, found)
            continue
        if (isinstance(ch, ast.Name) and ch.id in names) or (isinstance(ch, ast.Attribute)
                                                            and ch.attr == "claimed_reals"):
            found.add((rel, ".".join(scope) or "<module>"))
        _uses_in(ch, scope, rel, names, found)


def _claimed_reals_uses(root):
    """(module, enclosing function) of every call of or other reference to claimed_reals below root."""
    found = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                if rel == "precision.py":
                    continue
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                _uses_in(tree, [], rel, _local_names(tree), found)
    return found


def _names_a_record(path, func):
    """True if the code of the module-level function func (not its docstring) names one of RECORD_NAMES: as a name,
    an attribute, or a string constant such as a dictionary key."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func)
    for x in ast.walk(fn):
        if isinstance(x, ast.Name) and x.id in RECORD_NAMES:
            return True
        if isinstance(x, ast.Attribute) and x.attr in RECORD_NAMES:
            return True
        if isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value in RECORD_NAMES:
            return True
    return False


def test_every_use_of_claimed_reals_is_listed():
    """The lists above are complete for their scope (stated above them): a new use of claimed_reals must say where
    its dropped parts are recorded, or why it returns numbers only."""
    import certify_ed
    found = _claimed_reals_uses(os.path.dirname(certify_ed.__file__))
    listed = set(RECORDED) | set(NUMBERS) | set(OPEN)
    assert found == listed, {"not listed": sorted(found - listed), "gone": sorted(listed - found)}


def test_every_recorded_use_names_its_record():
    """Each RECORDED use, or its public wrapper, names dropped_imaginary_parts or carries_imaginary_parts in its
    code, so that the listing is checked against the property it states."""
    import certify_ed
    root = os.path.dirname(certify_ed.__file__)
    missing = [(rel, rec) for (rel, _), (_, rec) in RECORDED.items()
               if not _names_a_record(os.path.join(root, rel), rec)]
    assert missing == []


def test_the_use_audit_resolves_aliases_and_references(tmp_path):
    """Controls of the audit itself: aliased imports, a module alias and a reference that is not a call are found;
    a docstring that mentions a record name does not count as naming it."""
    (tmp_path / "a.py").write_text(
        "from certify_ed.precision import claimed_reals as cr\n"
        "def f(x):\n    return cr(x)\n"
        "def g(xs):\n    return list(map(cr, xs))\n"
        "def h(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "import certify_ed.precision as P\n"
        "def k(x):\n    return P.claimed_reals(x)\n", encoding="utf-8")
    (tmp_path / "c.py").write_text(
        "def only_docstring(x):\n    '''mentions dropped_imaginary_parts'''\n    return x\n"
        "def as_key(x):\n    return {'dropped_imaginary_parts': None}\n", encoding="utf-8")
    assert _claimed_reals_uses(str(tmp_path)) == {("a.py", "f"), ("a.py", "g"), ("b.py", "k")}
    assert not _names_a_record(str(tmp_path / "c.py"), "only_docstring")
    assert _names_a_record(str(tmp_path / "c.py"), "as_key")
