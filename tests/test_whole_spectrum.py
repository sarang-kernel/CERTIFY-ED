"""Whole-spectrum path (roadmap C-17) and streaming vectors (A-06), 3.2.0.dev0.

The optimisation of enclosure.residuals / _cluster_bound / the Kato-Temple products changed how the bounds
are evaluated, not the bounds. These tests pin that down:

* bit-identity of the enclosure radii against a frozen copy of the pre-change module
  (tests/_enclosure_reference.py, dev/3.2 f080586) on Heisenberg Sz = 0 blocks, dense and sparse, with and
  without forced column blocking;
* the recorded pre-change radii of a fixed claim (tests/fixtures/whole_spectrum_252_*.npz: float32-exact
  vectors and dyadic eigenvalues, so the inputs are the same on every machine);
* the block width and the storage of the vectors (ndarray, numpy memmap, HDF5 dataset) do not change a
  single number of the certificate;
* negative controls: a planted error in a streamed claim is caught exactly as in memory.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.enclosure import (OperatorAdapter, StreamedColumns, column_blocks, enclose, is_streamed,
                                  kato_temple, read_columns, residuals)
from certify_ed.fp import exact_sum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _enclosure_reference as ref  # noqa: E402  (frozen pre-change enclosure module)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
try:
    import h5py
    HAVE_H5PY = True
except Exception:
    HAVE_H5PY = False


def _block(N):
    spec = ce.heisenberg_chain(N)
    st = ce.u1_state_set(spec, N // 2, label="Sz=0")
    H = ce.build_bitwise(spec, st)
    w, V = la.eigh(H.toarray())
    return spec, st, H, w, V


@pytest.fixture(scope="module")
def heis8():
    return _block(8)            # dimension 70


@pytest.fixture(scope="module")
def heis10():
    return _block(10)           # dimension 252


@pytest.fixture(scope="module")
def heis12():
    return _block(12)           # dimension 924


# Theorem 2's cluster radius changed on track/backlog-proofs (docs/proofs/thm2_polar-factor.md): the frozen module
# computes eps = (w + sqrt(w^2 + 4 rho^2))/2 <= (1 + sqrt 5)/2 rho, the package computes rho (Kahan's constant),
# and the Theorem 2' term uses norm2_upper (factor 1 + gamma_{g+2}) where the frozen module used (1 + 8u). The
# Theorem 1 pipeline (residuals, existence radii) is unchanged bit for bit; cluster radii lie in the bracket below.
_PHI = (1 + np.sqrt(5.0)) / 2


def _radius_bracket(e_new, e_old, cplx=False):
    sizes_new = np.array([e_new.groups[gi].size for gi in e_new.group_of])
    single = sizes_new == 1
    if cplx:
        # complex data: the division constant is gamma_{w+3} (3.2.0, thm1/D-02 corrections of VERDICTS.md), so the
        # Theorem 1 radii are >= the frozen module's and within the factor (1 - gamma_{w+1})/(1 - gamma_{w+3}) < 1 + 1e-12
        assert np.all(e_new.radius[single] >= e_old.radius[single])
        assert np.all(e_new.radius[single] <= e_old.radius[single] * (1 + 1e-12))
    else:
        assert np.array_equal(e_new.radius[single], e_old.radius[single])
    r_new, r_old = e_new.radius[~single], e_old.radius[~single]
    assert np.all(r_new <= r_old * (1 + 1e-12)) and np.all(r_new >= r_old / _PHI * (1 - 1e-12))


def _same_enclosure(e_new, e_old, cplx=False):
    """Bit identity with the frozen module (real data); for complex data the quantities that carry the gamma_{w+3}
    constant (residual upper bounds, existence radii, E norms) are >= the frozen values and within 1 + 1e-12."""
    _radius_bracket(e_new, e_old, cplx)
    E_norm_old = np.array([ref.norm2_upper(e_old.residual.E[:, j]) for j in range(e_old.mu.size)])
    if cplx:
        for a, b in ((e_new.existence_radius, e_old.existence_radius), (e_new.residual.upper, e_old.residual.upper),
                     (e_new.residual.E_norm, E_norm_old)):
            assert np.all(np.asarray(a) >= np.asarray(b)) and np.all(np.asarray(a) <= np.asarray(b) * (1 + 1e-12))
    else:
        assert np.array_equal(e_new.existence_radius, e_old.existence_radius)
        assert np.array_equal(e_new.residual.upper, e_old.residual.upper)
        assert np.array_equal(e_new.residual.E_norm, E_norm_old)
    assert np.array_equal(e_new.residual.computed, e_old.residual.computed)
    assert np.array_equal(e_new.residual.rel_estimate, e_old.residual.rel_estimate)
    assert np.array_equal(e_new.residual.xnorm_lower, e_old.residual.xnorm_lower)
    assert [g.members for g in e_new.groups] == [g.members for g in e_old.groups]
    assert e_new.all_distinct == e_old.all_distinct


def _identical_enclosure(e_a, e_b):
    """Bit identity between two evaluations of the package (the column-blocking invariance of C-17)."""
    assert np.array_equal(e_a.radius, e_b.radius)
    assert np.array_equal(e_a.existence_radius, e_b.existence_radius)
    assert np.array_equal(e_a.residual.upper, e_b.residual.upper)
    assert np.array_equal(e_a.residual.computed, e_b.residual.computed)
    assert np.array_equal(e_a.residual.E_norm, e_b.residual.E_norm)
    assert [g.members for g in e_a.groups] == [g.members for g in e_b.groups]
    assert [g.radius for g in e_a.groups] == [g.radius for g in e_b.groups]
    assert e_a.all_distinct == e_b.all_distinct


# ------------------------------------------------------------------------------ bit-identity (C-17)
@pytest.mark.parametrize("fixture", ["heis8", "heis10", "heis12"])
@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_enclosure_radii_bit_identical_to_pre_change_code(request, fixture, storage):
    spec, st, H, w, V = request.getfixturevalue(fixture)
    A = H.toarray() if storage == "dense" else H
    e_old = ref.enclose(ref.OperatorAdapter(A), w, V)
    e_new = enclose(OperatorAdapter(A), w, V)
    _same_enclosure(e_new, e_old)
    if storage == "sparse":
        # forced narrow column blocks (the whole-spectrum memory path): still the same bits
        e_blk = enclose(OperatorAdapter(A), w, V, block_mb=0.05)
        _identical_enclosure(e_blk, e_new)
        assert len(column_blocks(w.size, V.shape[0], 8, 0.05)) >= 3
    # the stored residual matrix is the pre-change one, and the recomputed columns equal it
    assert np.array_equal(e_new.residual.R, e_old.residual.R)
    idx = [5, 2, 9]
    assert np.array_equal(e_new.residual.residual_columns(idx), e_old.residual.R[:, idx])
    assert np.array_equal(e_new.residual.E, e_old.residual.E)


def test_enclosure_radii_bit_identical_complex_operator():
    rng = np.random.default_rng(3)
    n = 60
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = (M + M.conj().T) / 2
    w, V = la.eigh(A)
    V = V.copy()
    V[:, 3] += 1e-7 * V[:, 4]           # a slightly non-orthonormal pair near a cluster
    e_old = ref.enclose(ref.OperatorAdapter(A), w, V)
    e_new = enclose(OperatorAdapter(A), w, V)
    _same_enclosure(e_new, e_old, cplx=True)


def test_recorded_pre_change_radii_dim_252():
    """The radii dev/3.2 f080586 produced for a fixed claim (vectors stored as float32, hence exactly the same
    binary64 inputs everywhere; eigenvalues dyadic). Two pins: on every stack the package equals the frozen pre-change
    module bit for bit (both run here, on the same BLAS); and both stay within 1e-12 relative plus 1e-16 absolute of the
    recording, which another BLAS may round differently in the last bit. Exact equality with the recording, which holds
    on the machine that recorded it (OpenBLAS 0.3.31 Haswell kernels, numpy 2.4.4), is reported as a warning."""
    spec, st, H, w, V = _block(10)
    # recorded values: single radii, existence radii and residual bounds between 8.8e-15 and 8.9e-8. Another BLAS or
    # sparse kernel (numpy 1.22; numpy 2.5 with another OpenBLAS) rounds an entry of H x differently, and the computed
    # residual norms then differ from the recording by up to 2.9e-17 on the stacks tried; the absolute term 1e-16 is
    # about three times that, and only the comparison with the recording carries it
    atol = 1e-16
    z = np.load(os.path.join(FIXTURES, "whole_spectrum_252_claim.npz"))
    mu, X = z["eigenvalues"], z["eigenvectors32"].astype(np.float64)
    for storage, A in (("sparse", H), ("dense", H.toarray())):
        exp = np.load(os.path.join(FIXTURES, f"whole_spectrum_252_{storage}.npz"))
        e = enclose(OperatorAdapter(A), mu, X)
        e_ref = ref.enclose(ref.OperatorAdapter(A), mu, X)
        assert np.array_equal(e.group_of, exp["group_of"]) and bool(exp["all_distinct"]) == e.all_distinct
        single = np.array([e.groups[gi].size for gi in e.group_of]) == 1
        for key, got, frozen in (("radius", e.radius[single], e_ref.radius[single]),
                                 ("existence_radius", e.existence_radius, e_ref.existence_radius),
                                 ("residual_bound", e.residual.upper, e_ref.residual.upper),
                                 ("residual_computed", e.residual.computed, e_ref.residual.computed)):
            assert np.array_equal(got, frozen), (storage, key)          # same stack: bit for bit
            want = exp[key][single] if key == "radius" else exp[key]
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=atol)
            if not np.array_equal(got, want):
                warnings.warn(f"{storage} {key}: not bit-identical to the recorded radii (different BLAS kernels?)")
        # cluster radii: the recorded eps against the package's min(rho, eps*) (see _radius_bracket)
        assert np.all(e.radius[~single] <= exp["radius"][~single] * (1 + 1e-12) + atol)
        assert np.all(e.radius[~single] >= exp["radius"][~single] / _PHI * (1 - 1e-12) - atol)
        # the frozen module reproduces the recording on this machine's BLAS, or the recording is stale
        np.testing.assert_allclose(e_ref.radius, exp["radius"], rtol=1e-12, atol=atol)


def test_exact_sum_equals_fsum():
    """fp.exact_sum (the cascaded summation behind exact_dot for long vectors) returns math.fsum's value."""
    import math
    from certify_ed.fp import _two_product
    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.integers(2100, 6000))
        if trial % 3 == 0:
            t = rng.standard_normal(n) * 10.0 ** rng.integers(-30, 30, n)
        elif trial % 3 == 1:
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)
            y -= (x @ y) / (x @ x) * x                        # heavy cancellation, as in a Gram off-diagonal
            t = np.concatenate(_two_product(x, y))
        else:
            t = rng.integers(-3, 4, n).astype(float) * 2.0 ** rng.integers(-60, 60, n)   # ties and zeros
        assert exact_sum(t) == math.fsum(t.tolist())
    assert exact_sum(np.zeros(5000)) == 0.0
    assert exact_sum(np.array([1.0, np.inf] + [0.0] * 5000)) == math.inf


# ------------------------------------------------------------------------------ Kato-Temple (C-17)
def test_batched_kato_temple_equals_per_pair_for_sparse_operator():
    """The Kato-Temple products are formed for all isolated pairs at once; for a sparse operator a column of
    H X_iso is bit-identical to the single matvec of 3.1, hence so is every Theorem-3 interval (the record
    carries the one-ulp outward rounding of the declared-model shift, zero here)."""
    import scipy.sparse as sp
    from certify_ed.fp import add_up, sub_down
    rng = np.random.default_rng(5)
    n = 200
    M = sp.random(n, n, density=0.15, random_state=rng, data_rvs=rng.standard_normal).toarray()
    A = sp.csr_matrix(M + M.T)
    w, V = la.eigh(A.toarray())
    cert = ce.quick(A, w, V, kind="all")
    kt = cert.body["kato_temple"]
    assert len(kt) > 100
    ad = OperatorAdapter(A)
    for rec in kt[:40]:
        i = rec["index"]
        a, b = rec["isolation_interval"]
        lo, hi, th, r = kato_temple(ad, V[:, i], a, b)      # single matvec, as 3.1 did
        assert (float(sub_down(lo, 0.0)), float(add_up(hi, 0.0))) == (rec["lower"], rec["upper"])
    assert [r["index"] for r in kt] == sorted(r["index"] for r in kt)


# ------------------------------------------------------------------------------ streaming (A-06)
def _write_streams(tmp_path, V):
    mm = np.lib.format.open_memmap(str(tmp_path / "V.npy"), mode="w+", dtype=V.dtype, shape=V.shape)
    mm[:] = V
    mm.flush()
    del mm
    paths = {"memmap": str(tmp_path / "V.npy")}
    if HAVE_H5PY:
        with h5py.File(str(tmp_path / "V.h5"), "w") as f:
            f.create_dataset("eigenvectors", data=V, chunks=(V.shape[0], min(16, V.shape[1])))
        paths["h5"] = str(tmp_path / "V.h5")
    return paths


def _cert_signature(c):
    return {"arrays": {k: v.tolist() for k, v in c.arrays.items()},
            "statements": [(s["id"], s["status"], s["summary"]) for s in c.statements],
            "enclosures": c.body["enclosures"], "kato_temple": c.body["kato_temple"],
            "spectrum": c.body["spectrum_enclosure"], "sha": c.body["claim"]["eigenvectors_sha256"]}


@pytest.mark.parametrize("storage", ["sparse", "dense"])
@pytest.mark.parametrize("layout", ["F", "C"])
def test_streamed_vectors_give_the_same_certificate(tmp_path, heis10, storage, layout):
    spec, st, H, w, V = heis10
    V = np.asfortranarray(V) if layout == "F" else np.ascontiguousarray(V)
    A = H if storage == "sparse" else H.toarray()
    paths = _write_streams(tmp_path, V)
    kw = dict(kind="all", states=st, temperatures=[0.7], block_mb=0.05)
    ref_c = _cert_signature(ce.quick(A, w, V, spec, **kw))
    mm = np.load(paths["memmap"], mmap_mode="r")
    assert isinstance(mm, np.memmap) and is_streamed(mm)
    assert _cert_signature(ce.quick(A, w, mm, spec, **kw)) == ref_c
    if HAVE_H5PY:
        with h5py.File(paths["h5"], "r") as f:
            ds = f["eigenvectors"]
            assert is_streamed(ds)
            assert _cert_signature(ce.quick(A, w, ds, spec, **kw)) == ref_c
    # also the default (no preset) selection with S4/S8, which reads cluster columns from the stream
    ref_d = _cert_signature(ce.quick(A, w, V, spec, kind="lowest", states=st, preset="labels", block_mb=0.05))
    assert _cert_signature(ce.quick(A, w, mm, spec, kind="lowest", states=st, preset="labels", block_mb=0.05)) == ref_d


def test_streamed_residuals_do_not_store_R_and_recompute_columns(tmp_path, heis10):
    spec, st, H, w, V = heis10
    paths = _write_streams(tmp_path, V)
    mm = np.load(paths["memmap"], mmap_mode="r")
    r_mem = residuals(OperatorAdapter(H), w, V)
    r_mm = residuals(OperatorAdapter(H), w, mm, block_mb=0.05)
    assert r_mem.R is not None and r_mm.R is None
    idx = [7, 3, 3, 11]
    assert np.array_equal(r_mm.residual_columns(idx), r_mem.R[:, idx])
    assert np.array_equal(r_mm.upper, r_mem.upper) and np.array_equal(r_mm.computed, r_mem.computed)
    assert np.array_equal(read_columns(mm, idx), V[:, idx])
    if HAVE_H5PY:
        with h5py.File(paths["h5"], "r") as f:
            sc = StreamedColumns(f["eigenvectors"])
            assert np.array_equal(sc[:, idx], V[:, idx]) and np.array_equal(sc[:, 4], V[:, 4])
            assert np.array_equal(sc[:, 2:9], V[:, 2:9])


def test_column_blocks_never_width_one():
    assert column_blocks(1, 100, 8, 0.001) == [(0, 1)]
    for k in range(2, 40):
        bl = column_blocks(k, 1000, 8, 0.001)
        assert bl[0][0] == 0 and bl[-1][1] == k
        assert all(b - a >= 2 for a, b in bl)
        assert all(bl[i][1] == bl[i + 1][0] for i in range(len(bl) - 1))


def test_streamed_claim_with_planted_wrong_vector_is_not_certified(tmp_path, heis10):
    """Negative control: a wrong eigenvector in a memmap claim gives the same (non-CERTIFIED) S2 as in memory."""
    spec, st, H, w, V = heis10
    Vb = V.copy()
    Vb[:, 5] = V[:, 6]                       # a duplicated vector: the pair cannot be matched to a distinct eigenvalue
    paths = _write_streams(tmp_path, Vb)
    mm = np.load(paths["memmap"], mmap_mode="r")
    c_mem = ce.quick(H, w, Vb, spec, kind="all", states=st)
    c_mm = ce.quick(H, w, mm, spec, kind="all", states=st, block_mb=0.05)
    st_mem = {s["id"]: s["status"] for s in c_mem.statements}
    st_mm = {s["id"]: s["status"] for s in c_mm.statements}
    assert st_mem["S2.eigenpairs"] != "CERTIFIED" and st_mm == st_mem
    assert np.array_equal(c_mm.arrays["radius"], c_mem.arrays["radius"])
    Vn = V.copy()
    Vn[0, 3] = np.nan
    (tmp_path / "nan").mkdir()
    paths = _write_streams(tmp_path / "nan", Vn)
    mm = np.load(paths["memmap"], mmap_mode="r")
    c = ce.quick(H, w, mm, spec, kind="all", states=st, block_mb=0.05)
    assert {s["id"]: s["status"] for s in c.statements}["S2.eigenpairs"] == "INCONCLUSIVE"


def test_quick_kind_all_uses_whole_spectrum_preset(heis8):
    spec, st, H, w, V = heis8
    c = ce.quick(H, w, V, spec, kind="all", states=st)
    assert c.body["scope"]["preset"] == "whole-spectrum"
    stt = {s["id"]: s["status"] for s in c.statements}
    assert stt["S2.eigenpairs"] == "CERTIFIED" and stt["S3.completeness"] == "CERTIFIED"
    assert stt["S4.eigenspaces"] == "NOT_CHECKED" and stt["S7.thermodynamics"] == "NOT_CHECKED"
    c2 = ce.quick(H, w, V, spec, kind="all", states=st, only="S2")
    assert c2.body["scope"]["preset"] is None


@pytest.mark.skipif(not os.environ.get("CERTIFY_ED_LARGE_FIXTURE"), reason="set CERTIFY_ED_LARGE_FIXTURE to the "
                    "npz (w, V, N) of the 12870 Heisenberg block (experiments/whole_spectrum_bench.py builds it)")
def test_memmap_12870_fixture_identical_to_in_memory(tmp_path):
    z = np.load(os.environ["CERTIFY_ED_LARGE_FIXTURE"])
    w, V, N = z["w"], z["V"], int(z["N"])
    spec = ce.heisenberg_chain(N)
    st = ce.u1_state_set(spec, N // 2, label="Sz=0")
    H = ce.build_bitwise(spec, st)
    mm = np.lib.format.open_memmap(str(tmp_path / "V12870.npy"), mode="w+", dtype=V.dtype, shape=V.shape)
    mm[:] = V
    mm.flush()
    del mm
    c_mem = ce.quick(H, w, V, spec, kind="all", states=st, temperatures=[0.5, 1.0])
    del V
    mm = np.load(str(tmp_path / "V12870.npy"), mmap_mode="r")
    c_mm = ce.quick(H, w, mm, spec, kind="all", states=st, temperatures=[0.5, 1.0], block_mb=256)
    assert _cert_signature(c_mm) == _cert_signature(c_mem)
