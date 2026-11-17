"""D-05: GPU products for S2 (certify_ed/gpu.py). The tests that need a CUDA device skip without one; the fallback
tests run everywhere."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from certify_ed import gpu
from certify_ed.enclosure import OperatorAdapter, enclose, residuals


def _device_or_skip():
    try:
        rec = gpu.probe()
    except gpu.GPUUnavailable as e:
        pytest.skip(str(e))
    return rec


def _hermitian(n, density, cplx, seed):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csr", random_state=np.random.RandomState(seed))
    if cplx:
        B = A.copy()
        B.data = rng.standard_normal(B.nnz)
        A = (A + 1j * B).tocsr()
    H = ((A + A.conj().T) * 0.5).tocsr()
    H = H + sp.identity(n, format="csr") * 1e-3
    return H.tocsr()


def test_probe_holds():
    rec = _device_or_skip()
    assert rec["holds"], rec


@pytest.mark.parametrize("cplx_a", [False, True])
@pytest.mark.parametrize("sorted_idx", [True, False])
def test_products_equal_scipy_bit_for_bit(cplx_a, sorted_idx):
    _device_or_skip()
    rng = np.random.default_rng(7)
    A = gpu._with_empty_rows(gpu._random_csr(rng, 1500, 1400, 0.01, cplx_a, sorted_idx, extreme=True), (3, 4, 1499))
    D = gpu.DeviceCSR(A, spot_rows=1500)
    X = rng.standard_normal((1400, 7))
    Xc = X + 1j * rng.standard_normal((1400, 7))
    for Xv in (X, Xc, X[:, 2], Xc[:, 1], np.asfortranarray(X)):
        assert gpu.same_bits(D.matmul(Xv), np.asarray(A @ Xv))
        aX = np.abs(Xv)
        assert gpu.same_bits(D.matmul(aX, absolute=True), np.asarray(abs(A) @ aX))
    assert D.record["spot_checked_rows"] > 0


@pytest.mark.parametrize("cplx", [False, True])
def test_both_products_in_one_pass_equal_the_two_cpu_products(cplx):
    _device_or_skip()
    rng = np.random.default_rng(2)
    H = _hermitian(2000, 0.003, cplx, 21)
    for X in (rng.standard_normal((2000, 19)), rng.standard_normal((2000, 19)) + 1j * rng.standard_normal((2000, 19)),
              rng.standard_normal(2000)):
        ad = OperatorAdapter(H, device="gpu")
        hx, ahx = ad.both_products(X)
        assert gpu.same_bits(hx, H @ X) and gpu.same_bits(ahx, abs(H) @ np.abs(X))
        assert ad.device_record["gpu"]["products"] == 2 and "fallback" not in ad.device_record
    cpu = OperatorAdapter(H)
    X = rng.standard_normal((2000, 3))
    hx, ahx = cpu.both_products(X)
    assert gpu.same_bits(hx, cpu.matvec(X)) and gpu.same_bits(ahx, cpu.abs_matvec(X))


def test_column_blocks_change_no_bit():
    _device_or_skip()
    rng = np.random.default_rng(11)
    A = gpu._random_csr(rng, 800, 800, 0.02, True, True)
    D = gpu.DeviceCSR(A)
    D.budget = D.static_bytes + 16 * (800 + 800) * 3        # three columns per block
    X = rng.standard_normal((800, 10)) + 1j * rng.standard_normal((800, 10))
    assert gpu.same_bits(D.matmul(X), A @ X)


@pytest.mark.parametrize("cplx", [False, True])
def test_enclosure_on_gpu_equals_cpu_bit_for_bit(cplx):
    _device_or_skip()
    H = _hermitian(3000, 0.002, cplx, 3)
    w, V = spla.eigsh(H, k=6, which="SA", tol=1e-10)
    cpu = enclose(OperatorAdapter(H), w, V)
    ad = OperatorAdapter(H, device="gpu")
    assert ad.device_record["used"], ad.device_record
    g = enclose(ad, w, V)
    assert gpu.same_bits(g.radius, cpu.radius)
    assert gpu.same_bits(g.residual.upper, cpu.residual.upper)
    assert gpu.same_bits(g.existence_radius, cpu.existence_radius)
    assert ad.device_record["gpu"]["products"] >= 2
    assert "fallback" not in ad.device_record


def test_spot_check_rejects_a_wrong_row():
    _device_or_skip()
    rng = np.random.default_rng(13)
    A = gpu._random_csr(rng, 200, 200, 0.05, False, True)
    D = gpu.DeviceCSR(A, spot_rows=200)
    X = rng.standard_normal((200, 3))
    Y = A @ X
    Y[17, 1] = np.nextafter(Y[17, 1], np.inf)
    with pytest.raises(gpu.GPUUnavailable):
        D._spot_check(X, Y, False)


def test_fallback_is_recorded_and_changes_no_bit(monkeypatch):
    _device_or_skip()
    H = _hermitian(400, 0.02, False, 9)
    w, V = np.linalg.eigh(H.toarray())
    w, V = w[:3], V[:, :3]
    cpu = residuals(OperatorAdapter(H), w, V)
    ad = OperatorAdapter(H, device="gpu")

    def boom(self, X, *args, **kwargs):
        raise gpu.GPUUnavailable("injected failure")
    monkeypatch.setattr(gpu.DeviceCSR, "matmul", boom)
    monkeypatch.setattr(gpu.DeviceCSR, "products", boom)
    r = residuals(ad, w, V)
    assert ad.device_record["fallback"] == "injected failure"
    assert gpu.same_bits(r.upper, cpu.upper)


def test_no_device_falls_back_to_cpu(monkeypatch):
    def none():
        raise gpu.GPUUnavailable("no CUDA device")
    monkeypatch.setattr(gpu, "_cupy", none)
    H = _hermitian(200, 0.05, False, 1)
    ad = OperatorAdapter(H, device="gpu")
    assert ad.device_record == {"requested": "gpu", "used": False, "reason": "no CUDA device"}
    w, V = np.linalg.eigh(H.toarray())
    assert gpu.same_bits(residuals(ad, w[:2], V[:, :2]).upper, residuals(OperatorAdapter(H), w[:2], V[:, :2]).upper)


def _without_timings(obj):
    if isinstance(obj, dict):
        return {k: _without_timings(v) for k, v in obj.items()
                if not any(w in k for w in ("second", "time", "wall", "device"))}
    if isinstance(obj, (list, tuple)):
        return [_without_timings(v) for v in obj]
    return obj


def test_certify_with_device_gpu_gives_the_same_certificate():
    _device_or_skip()
    import json
    import certify_ed as ce
    spec = ce.heisenberg_chain(12)
    H = ce.build_bitwise(spec).tocsr()
    w, V = spla.eigsh(H, k=4, which="SA", tol=1e-12)
    claim = ce.Claim(eigenvalues=w, eigenvectors=V, operator=H, kind="lowest")
    a = ce.certify(claim, spec, ce.Options(seed=1))
    b = ce.certify(claim, spec, ce.Options(seed=1, device="gpu"))
    assert [(s["id"], s["status"]) for s in a.statements] == [(s["id"], s["status"]) for s in b.statements]
    dev = b.statement("S2.eigenpairs")["evidence"]["device"]
    assert dev["used"] and "fallback" not in dev and dev["gpu"]["spot_checked_rows"] > 0
    assert "device" not in a.statement("S2.eigenpairs")["evidence"]
    for s_a, s_b in zip(a.statements, b.statements):
        assert json.dumps(_without_timings(s_a), sort_keys=True, default=repr) == \
            json.dumps(_without_timings(s_b), sort_keys=True, default=repr), s_a["id"]
    assert sorted(a.arrays) == sorted(b.arrays)
    for key in a.arrays:
        x, y = np.asarray(a.arrays[key]), np.asarray(b.arrays[key])
        if x.dtype in (np.float64, np.complex128):
            assert gpu.same_bits(x, y), key
        else:
            assert x.dtype == y.dtype and np.array_equal(x, y), key


def test_options_device_without_gpu_records_the_fallback(monkeypatch):
    import certify_ed as ce

    def none():
        raise gpu.GPUUnavailable("no CUDA device")
    monkeypatch.setattr(gpu, "_cupy", none)
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    claim = ce.Claim(eigenvalues=w[:2], eigenvectors=V[:, :2], operator=H, kind="lowest")
    c = ce.certify(claim, spec, ce.Options(device="gpu"))
    dev = c.statement("S2.eigenpairs")["evidence"]["device"]
    assert dev == {"requested": "gpu", "used": False, "reason": "no CUDA device"}
    assert c.status_of("S2.eigenpairs") == ce.certify(claim, spec, ce.Options()).status_of("S2.eigenpairs")


def test_dense_and_unknown_devices_are_not_used():
    H = _hermitian(50, 0.2, False, 2)
    d = OperatorAdapter(H.toarray(), device="gpu").device_record
    assert d["used"] is False and "sparse" in d["reason"]
    u = OperatorAdapter(H, device="tpu").device_record
    assert u["used"] is False and "unknown device" in u["reason"]


class _StandIn:
    """A stand-in for gpu.DeviceCSR that forms the scipy products (no CUDA device needed) and counts them, so the
    wiring of residuals, Kato-Temple and the device record is tested everywhere. The call numbered fail_at and later
    ones raise, as a failed spot check or a CuPy error would."""
    fail_at = None

    def __init__(self, A, absA=None):
        self.A, self.absA = A, (abs(A) if absA is None else absA)
        self.record = {"products": 0}
        self.calls = 0

    def _count(self):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("injected device failure")
        self.record["products"] += 1

    def matmul(self, X, absolute=False):
        self._count()
        return (self.absA if absolute else self.A) @ X

    def products(self, X):
        self._count()
        return self.A @ X, self.absA @ np.abs(X)

    def close(self):
        pass


def _small_claim():
    import certify_ed as ce
    n = 300
    A = sp.random(n, n, density=0.03, format="csr", random_state=np.random.RandomState(3))
    H = ((A + A.T) * 0.5 + sp.diags(np.arange(n) * 0.01)).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    return ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest")


def _same_certificate(a, b):
    import json
    assert [(s["id"], s["status"]) for s in a.statements] == [(s["id"], s["status"]) for s in b.statements]
    for s_a, s_b in zip(a.statements, b.statements):
        assert json.dumps(_without_timings(s_a), sort_keys=True, default=repr) == \
            json.dumps(_without_timings(s_b), sort_keys=True, default=repr), s_a["id"]
    assert sorted(a.arrays) == sorted(b.arrays)
    for key in a.arrays:
        assert gpu.same_bits(np.asarray(a.arrays[key]), np.asarray(b.arrays[key])), key


def test_stand_in_device_runs_residuals_and_kato_temple_through_it(monkeypatch):
    import certify_ed as ce
    monkeypatch.setattr(gpu, "DeviceCSR", _StandIn)
    claim = _small_claim()
    a = ce.certify(claim, None, ce.Options(seed=1))
    b = ce.certify(claim, None, ce.Options(seed=1, device="gpu"))
    dev = b.statement("S2.eigenpairs")["evidence"]["device"]
    assert dev == {"requested": "gpu", "used": True, "gpu": {"products": 2}}      # residuals and Kato-Temple
    assert b.body["kato_temple"]
    _same_certificate(a, b)


def test_device_failure_after_products_recomputes_the_certificate_on_the_cpu(monkeypatch):
    import certify_ed as ce
    monkeypatch.setattr(gpu, "DeviceCSR", _StandIn)
    monkeypatch.setattr(_StandIn, "fail_at", 2)        # the Kato-Temple product fails after the residual product
    claim = _small_claim()
    a = ce.certify(claim, None, ce.Options(seed=1))
    b = ce.certify(claim, None, ce.Options(seed=1, device="gpu"))
    dev = b.statement("S2.eigenpairs")["evidence"]["device"]
    assert dev["used"] is False and dev["recomputed_on_cpu"] is True
    assert dev["gpu_products_at_fallback"] == 1 and dev["fallback"] == "RuntimeError: injected device failure"
    _same_certificate(a, b)


def test_device_record_says_unused_without_products_and_after_any_setup_error(monkeypatch):
    import certify_ed as ce
    monkeypatch.setattr(gpu, "DeviceCSR", _StandIn)
    claim = _small_claim()
    values_only = ce.Claim(eigenvalues=claim.eigenvalues, operator=claim.operator, kind="lowest")
    dev = ce.certify(values_only, None, ce.Options(device="gpu")).statement("S2.eigenpairs")["evidence"]["device"]
    assert dev == {"requested": "gpu", "used": False, "gpu": {"products": 0}}

    def broken(self, A, absA=None):
        raise MemoryError("out of device memory")
    monkeypatch.setattr(_StandIn, "__init__", broken)
    c = ce.certify(claim, None, ce.Options(seed=1, device="gpu"))
    dev = c.statement("S2.eigenpairs")["evidence"]["device"]
    assert dev == {"requested": "gpu", "used": False, "reason": "MemoryError: out of device memory"}
    _same_certificate(ce.certify(claim, None, ce.Options(seed=1)), c)
