"""Interior windows everywhere (roadmap C-14): kind='window' with window=(a, b) through every importer, the
command line (`--window A B` and `--window A,B`), quick(), and the bundle's meta.json; the same validation of
the window in every path; a planted missing eigenvalue inside the window is FAILED on every path."""
import json
import os

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

import certify_ed as ce
from certify_ed.certify import parse_window
from certify_ed.cli import main
from certify_ed.importers import claim_from_arrays, claim_from_bundle, claim_from_file, claim_from_hdf5, claim_from_npz

try:
    import h5py
    HAVE_H5PY = True
except Exception:
    HAVE_H5PY = False


@pytest.fixture(scope="module")
def block():
    """Open J1-J2 chain, Sz = 0 block (dimension 70): non-degenerate interior spectrum."""
    spec = ce.heisenberg_chain(8, 1.0, J2=0.4, boundary="open")
    st = ce.u1_state_set(spec, 4, label="Sz=0")
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    return spec, st, H, w, V


def _window_claim(block, lo=20, hi=28):
    """Eigenvalues lo..hi-1 (0-based positions in the sorted spectrum) and a window in the gaps around them."""
    spec, st, H, w, V = block
    a = 0.5 * (w[lo - 1] + w[lo])
    b = 0.5 * (w[hi - 1] + w[hi])
    return w[lo:hi], V[:, lo:hi], (float(a), float(b))


def _statuses(c):
    return {s["id"]: s["status"] for s in c.statements}


def test_shift_invert_eigsh_window_is_certified(block):
    spec, st, H, w, V = block
    sigma = float(w[24])
    rng = np.random.default_rng(0)
    ws_, Vs = spla.eigsh(H, k=8, sigma=sigma, which="LM", tol=1e-12, v0=rng.standard_normal(H.shape[0]))
    order = np.argsort(ws_)
    ws_, Vs = ws_[order], Vs[:, order]
    a, b = 0.5 * (ws_[0] + ws_[1]), 0.5 * (ws_[-2] + ws_[-1])
    c = ce.quick(H, ws_[1:-1], Vs[:, 1:-1], spec, kind="window", window=(a, b), states=st)
    assert _statuses(c)["S3.completeness"] == "CERTIFIED"
    assert c.body["claim"]["kind"] == "window" and c.body["claim"]["window"] == [a, b]


def test_window_validation_is_the_same_everywhere():
    assert parse_window("1,2") == (1.0, 2.0) and parse_window("1 2") == (1.0, 2.0) and parse_window([-1, 3]) == (-1.0, 3.0)
    for bad in ("1", "2,1", "1,1", "a,b", [1, float("inf")], [float("nan"), 2], (1, 2, 3)):
        with pytest.raises(ValueError):
            parse_window(bad)
    with pytest.raises(ValueError, match="window"):
        ce.Claim([1.0], kind="window")                      # no window
    with pytest.raises(ValueError, match="a < b"):
        claim_from_arrays([1.0], kind="window", window=(2.0, 1.0))
    with pytest.raises(ValueError, match="one of"):
        ce.Claim([1.0], kind="inside")
    cl = ce.Claim([1.0], kind="window", window=[0, "2"])
    assert cl.window == (0.0, 2.0)


@pytest.mark.parametrize("path", ["arrays", "npz", "hdf5", "file", "bundle", "quick"])
def test_window_claim_through_every_importer(tmp_path, block, path):
    spec, st, H, w, V = block
    mu, X, (a, b) = _window_claim(block)
    states = np.sort(st.states_asc)
    if path == "arrays":
        claim = claim_from_arrays(mu, X, operator=H, kind="window", window=(a, b), states=st)
    elif path == "npz":
        p = ce.export_npz(str(tmp_path / "c.npz"), mu, X, H)
        claim = claim_from_npz(p, kind="window", window=(a, b), states=st)
    elif path == "hdf5":
        if not HAVE_H5PY:
            pytest.skip("h5py not installed")
        with h5py.File(str(tmp_path / "c.h5"), "w") as f:
            f["eigenvalues"] = mu
            f["eigenvectors"] = X
            g = f.create_group("H")
            g["data"], g["indices"], g["indptr"], g["shape"] = H.data, H.indices, H.indptr, np.array(H.shape)
        claim = claim_from_hdf5(str(tmp_path / "c.h5"), kind="window", window=(a, b), states=st)
    elif path == "file":
        np.save(str(tmp_path / "w.npy"), mu)
        np.save(str(tmp_path / "V.npy"), X)
        from certify_ed.io import write_operator, write_basis
        write_operator(str(tmp_path / "H.npz"), H)
        write_basis(str(tmp_path / "basis.npy"), states)
        claim = claim_from_file(str(tmp_path / "w.npy"), str(tmp_path / "V.npy"), str(tmp_path / "H.npz"),
                                str(tmp_path / "basis.npy"), kind="window", window=(a, b))
    elif path == "bundle":
        d = ce.write_bundle(str(tmp_path / "bundle"), mu, X, H, spec=spec, basis=states, kind="window", window=(a, b))
        claim, bspec, meta = claim_from_bundle(d)
        assert meta["window"] == [a, b] and claim.kind == "window" and claim.window == (a, b)
    else:
        c = ce.quick(H, mu, X, spec, kind="window", window=(a, b), states=st)
        assert _statuses(c)["S3.completeness"] == "CERTIFIED"
        return
    assert claim.kind == "window" and claim.window == (a, b)
    c = ce.certify(claim, spec)
    assert _statuses(c)["S3.completeness"] == "CERTIFIED", _statuses(c)
    assert _statuses(c)["S2.eigenpairs"] == "CERTIFIED"
    # negative control on the same path: drop one claimed pair inside the window -> FAILED, never CERTIFIED
    keep = [i for i in range(mu.size) if i != 3]
    claim2 = ce.Claim(mu[keep], X[:, keep], operator=H, kind="window", window=(a, b), states=st)
    c2 = ce.certify(claim2, spec)
    s3 = next(s for s in c2.statements if s["id"] == "S3.completeness")
    assert s3["status"] == "FAILED" and s3["evidence"]["missing_at_least"] >= 1


def test_cli_window_two_arguments_and_comma_form(tmp_path, block):
    spec, st, H, w, V = block
    mu, X, (a, b) = _window_claim(block)
    p = ce.export_npz(str(tmp_path / "c.npz"), mu, X, H)
    # the npz path needs the basis: give it through a bundle for the full S1; here no spec -> S3 on the stored matrix
    rc = main(["certify", p, "--kind", "window", "--window", repr(a), repr(b), "-o", str(tmp_path / "c1")])
    assert rc == 0
    c1 = ce.Certificate.load(str(tmp_path / "c1.json"))
    assert _statuses(c1)["S3.completeness"] == "CERTIFIED" and c1.body["claim"]["window"] == [a, b]
    # the comma form with a negative first endpoint needs the '=' spelling (argparse reads '-1.2,-0.6' as a flag)
    rc = main(["certify", p, "--kind", "window", f"--window={a!r},{b!r}", "-o", str(tmp_path / "c2")])
    assert rc == 0
    c2 = ce.Certificate.load(str(tmp_path / "c2.json"))
    assert c2.body["claim"]["window"] == [a, b]
    with pytest.raises(ValueError):
        main(["certify", p, "--kind", "window", "--window", repr(b), repr(a), "-o", str(tmp_path / "c3")])
    with pytest.raises(ValueError):
        main(["certify", p, "--kind", "window", "-o", str(tmp_path / "c4")])


def test_cli_import_bundle_window_from_meta_and_override(tmp_path, block):
    spec, st, H, w, V = block
    mu, X, (a, b) = _window_claim(block)
    d = ce.write_bundle(str(tmp_path / "bundle"), mu, X, H, spec=spec, basis=np.sort(st.states_asc), kind="window",
                        window=(a, b))
    rc = main(["import", d, "-o", str(tmp_path / "ci")])
    assert rc == 0
    ci = ce.Certificate.load(str(tmp_path / "ci.json"))
    assert ci.body["claim"]["kind"] == "window" and _statuses(ci)["S3.completeness"] == "CERTIFIED"
    # --window on the command line overrides meta.json; a window reaching past the list is FAILED (exit 1)
    rc = main(["import", d, "--window", repr(a), repr(b + 0.5), "-o", str(tmp_path / "cw")])
    assert rc == 1
    cw = ce.Certificate.load(str(tmp_path / "cw.json"))
    assert _statuses(cw)["S3.completeness"] == "FAILED" and cw.body["claim"]["window"] == [a, b + 0.5]
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    meta["window"] = [b, a]
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="a < b"):
        claim_from_bundle(d)
