"""GPU products for the residual bounds of S2 (roadmap D-05, 3.3.0.dev0; opt-in, never the default).

What runs on the GPU. For a sparse operator stored as CSR, the two products from which every S2 radius is built,
H X and |H| |X| (enclosure.residuals and enclosure._rounding_bound; also the Kato-Temple step of certify), are formed
on the GPU by kernels that perform, for each entry, the operations of scipy's CSR product in the same order: the sum
starts from +0.0 and adds the products a_ij x_j of the stored row in index order, each product and each sum rounded
once to nearest (__dmul_rn, __dadd_rn, compiled with --fmad=false, so that no product is fused with an addition).
For complex data each product is formed as scipy forms it, (ar xr - ai xi) + i (ar xi + ai xr): four products and
two sums, then the real and imaginary parts are added to the running sums. A real operator applied to complex
vectors, or the reverse, is promoted to complex with a zero imaginary part, as scipy promotes it. The moduli |a_ij|
are taken on the CPU (fp.abs_up, the function of the CPU path) and stored; the moduli |x_j| are taken on the CPU
(fp.modulus_up) for complex vectors and as fabs on the GPU for real ones (the same exact operation). One pass over the
stored matrix forms both products for up to eight vectors at a time; the vectors and the results keep numpy's C
layout, so no transpose is needed on either side.

Under IEEE-754 binary64 arithmetic with round to nearest and gradual underflow these are the same floating-point
operations as on the CPU, so H X and |H| |X| agree with the CPU products bit for bit, and so does every number of a
certificate computed with them: the bounds are those of docs/proofs/thm1_fp-accounting.md applied to the same
floating-point values, and no new rounding analysis enters.

Two checks guard that premise. probe() runs once per device and process: correctly rounded +, - and * on random
operands of three magnitude ranges, gradual underflow on fixed cases, and the kernels against scipy on random CSR
matrices (real and complex, empty rows, unsorted indices, subnormal and large entries), bit for bit. Every product
formed on the GPU is then spot-checked: a sample of rows is recomputed on the CPU and must agree bit for bit. A
failure of either check, a missing CuPy or device, an unsupported dtype or too little device memory raises
GPUUnavailable; enclosure.OperatorAdapter then computes the product on the CPU and records why.

The module uses only kernels of its own and host-device copies, no CuPy elementwise kernels, so it needs NVRTC but
not the CUDA headers. Device memory is bounded by `mem_gb` (default: CERTIFY_ED_GPU_MEM_GB or GPU_MEM_GB from the
environment, else 8) and by 90 % of the free device memory; vectors that do not fit at once are processed in column
blocks, which changes no bit (each column is an independent sum). CERTIFY_ED_GPU_DUTY or GPU_DUTY in (0, 1) makes
the module sleep after each block so that the device is busy about that fraction of the time (shared machines).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .fp import abs_up, modulus_up
from .precision import summed

SPOT_ROWS = 64
DEFAULT_MEM_GB = 8.0
_THREADS = 256
_KB = 8                       # vectors per pass over the matrix (registers per thread)


class GPUUnavailable(RuntimeError):
    """The GPU path cannot be used for this product; the caller computes it on the CPU."""


_SRC = r'''
#define KB 8
extern "C" {
__global__ void both_real(const long long* indptr, const IDX_T* indices, const double* data, const double* absdata,
                          const double* X, double* Y, double* Z, long long n, long long ldk, int c0, int kb,
                          int want_y, int want_z) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s[KB], t[KB];
    #pragma unroll
    for (int c = 0; c < KB; ++c) { s[c] = 0.0; t[c] = 0.0; }
    const long long e = indptr[i + 1];
    for (long long jj = indptr[i]; jj < e; ++jj) {
        const double a = data[jj];
        const double aa = absdata[jj];
        const double* xr = X + (long long)indices[jj] * ldk + c0;
        #pragma unroll
        for (int c = 0; c < KB; ++c) {
            if (c < kb) {
                const double xv = xr[c];
                if (want_y) s[c] = __dadd_rn(s[c], __dmul_rn(a, xv));
                if (want_z) t[c] = __dadd_rn(t[c], __dmul_rn(aa, fabs(xv)));
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < KB; ++c) {
        if (c < kb) {
            if (want_y) Y[i * ldk + c0 + c] = s[c];
            if (want_z) Z[i * ldk + c0 + c] = t[c];
        }
    }
}
__global__ void both_complex(const long long* indptr, const IDX_T* indices, const double2* data, const double* absdata,
                             const double2* X, const double* AX, double2* Y, double* Z, long long n, long long ldk,
                             int c0, int kb, int want_y, int want_z) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sr[KB], si[KB], t[KB];
    #pragma unroll
    for (int c = 0; c < KB; ++c) { sr[c] = 0.0; si[c] = 0.0; t[c] = 0.0; }
    const long long e = indptr[i + 1];
    for (long long jj = indptr[i]; jj < e; ++jj) {
        const double2 a = data[jj];
        const double aa = absdata[jj];
        const long long j = (long long)indices[jj];
        #pragma unroll
        for (int c = 0; c < KB; ++c) {
            if (c < kb) {
                if (want_y) {
                    const double2 b = X[j * ldk + c0 + c];
                    const double pr = __dadd_rn(__dmul_rn(a.x, b.x), -__dmul_rn(a.y, b.y));
                    const double pi = __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x));
                    sr[c] = __dadd_rn(sr[c], pr);
                    si[c] = __dadd_rn(si[c], pi);
                }
                if (want_z) t[c] = __dadd_rn(t[c], __dmul_rn(aa, AX[j * ldk + c0 + c]));
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < KB; ++c) {
        if (c < kb) {
            if (want_y) Y[i * ldk + c0 + c] = make_double2(sr[c], si[c]);
            if (want_z) Z[i * ldk + c0 + c] = t[c];
        }
    }
}
__global__ void ops(const double* a, const double* b, double* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[3 * i + 0] = __dadd_rn(a[i], b[i]);
    out[3 * i + 1] = __dadd_rn(a[i], -b[i]);
    out[3 * i + 2] = __dmul_rn(a[i], b[i]);
}
}
'''

_MODULES: Dict[Any, Any] = {}
_PROBES: Dict[int, dict] = {}


def _cupy():
    try:
        import cupy as cp
    except Exception as e:  # ImportError, or a CUDA library that does not load
        raise GPUUnavailable(f"CuPy is not available ({type(e).__name__}: {e})") from None
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise GPUUnavailable("no CUDA device")
    except GPUUnavailable:
        raise
    except Exception as e:
        raise GPUUnavailable(f"no usable CUDA device ({type(e).__name__}: {e})") from None
    return cp


def _module(cp, idx_t: str):
    key = (cp.cuda.Device().id, idx_t)
    if key not in _MODULES:
        _MODULES[key] = cp.RawModule(code=_SRC.replace("IDX_T", idx_t), options=("--fmad=false",))
    return _MODULES[key]


def device_info() -> Dict[str, Any]:
    """Name, compute capability, memory and library versions of the current device."""
    cp = _cupy()
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    free, total = cp.cuda.runtime.memGetInfo()
    return {"name": name.decode() if isinstance(name, bytes) else str(name), "id": int(dev.id),
            "compute_capability": f"{props['major']}.{props['minor']}", "memory_total_mb": round(total / 2 ** 20),
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": int(cp.cuda.runtime.driverGetVersion()), "cupy": cp.__version__}


def _duty() -> Optional[float]:
    for name in ("CERTIFY_ED_GPU_DUTY", "GPU_DUTY"):
        v = os.environ.get(name)
        if v:
            try:
                d = float(v)
            except ValueError:
                continue
            if 0.0 < d < 1.0:
                return d
    return None


def _mem_budget_bytes(mem_gb: Optional[float]) -> int:
    if mem_gb is None:
        for name in ("CERTIFY_ED_GPU_MEM_GB", "GPU_MEM_GB"):
            v = os.environ.get(name)
            if v:
                try:
                    mem_gb = float(v)
                    break
                except ValueError:
                    pass
    if mem_gb is None:
        mem_gb = DEFAULT_MEM_GB
    return int(float(mem_gb) * 2 ** 30)


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a.view(np.float64).view(np.uint64)


def same_bits(a, b) -> bool:
    """Whether two float64 or complex128 arrays are equal bit for bit (signed zeros and NaN payloads included)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(_bits(a), _bits(b)))


class _Ordered:
    """The ordered kernels on one CSR matrix held on the device (no probe, no spot check, no memory policy)."""

    def __init__(self, cp, A, absA):
        if not (absA.shape == A.shape and np.array_equal(absA.indptr, A.indptr)
                and np.array_equal(absA.indices, A.indices)):
            raise GPUUnavailable("|A| does not share the stored pattern of A")
        self.cp = cp
        self.shape = A.shape
        self.nnz = int(A.nnz)
        self.complex = A.dtype == np.complex128
        self.idx_t = "int" if A.indices.dtype == np.int32 else "long long"
        idx_np = np.int32 if self.idx_t == "int" else np.int64
        self.idx_bytes = np.dtype(idx_np).itemsize
        mod = _module(cp, self.idx_t)
        self.k_real = mod.get_function("both_real")
        self.k_cplx = mod.get_function("both_complex")
        self.A_data = A.data
        self.d_indptr = cp.asarray(A.indptr.astype(np.int64))
        self.d_indices = cp.asarray(A.indices.astype(idx_np))
        self.d_data = cp.asarray(A.data)
        self.d_abs = cp.asarray(np.ascontiguousarray(absA.data, dtype=np.float64))
        self.d_data_c = None
        self.d_dummy = cp.empty(1, dtype=cp.complex128)

    def static_bytes(self) -> int:
        es = 16 if self.complex else 8
        return 8 * (self.shape[0] + 1) + self.nnz * (self.idx_bytes + es + 8)

    def per_column_bytes(self, cplx: bool, want_y: bool, want_z: bool) -> int:
        nrows, ncols = self.shape
        if cplx:
            return (16 + (8 if want_z else 0)) * ncols + (16 * want_y + 8 * want_z) * nrows
        return 8 * ncols + 8 * (int(want_y) + int(want_z)) * nrows

    def extra_bytes(self, cplx: bool) -> int:
        return 16 * self.nnz if (cplx and not self.complex and self.d_data_c is None) else 0

    def run(self, X2: np.ndarray, want_y: bool, want_z: bool, kb: int, on_block=None):
        """(Y, Z) for the C-ordered n x k array X2 (float64 or complex128), column blocks of width kb."""
        cp = self.cp
        nrows, ncols = self.shape
        k = X2.shape[1]
        cplx = bool(want_y and (self.complex or np.iscomplexobj(X2)))
        if want_z and not want_y and np.iscomplexobj(X2):
            raise ValueError("|A| |X| alone takes the real moduli |X|")
        if cplx:
            if self.complex:
                data = self.d_data
            else:
                if self.d_data_c is None:
                    # promoted on the host (exact: zero imaginary parts); no CuPy elementwise kernel is needed
                    self.d_data_c = cp.asarray(self.A_data.astype(np.complex128))
                data = self.d_data_c
            ydt = np.complex128
        else:
            data = self.d_abs if (want_z and not want_y and self.complex) else self.d_data
            ydt = np.float64
        Y = np.empty((nrows, k), dtype=ydt) if want_y else None
        Z = np.empty((nrows, k), dtype=np.float64) if want_z else None
        grid_x = (nrows + _THREADS - 1) // _THREADS
        for a in range(0, k, kb):
            b = min(k, a + kb)
            w = b - a
            tb = time.perf_counter()
            Xb = X2 if (a, w) == (0, k) else X2[:, a:b]
            if cplx:
                Xd = cp.asarray(np.ascontiguousarray(Xb, dtype=np.complex128))
                AXd = cp.asarray(np.ascontiguousarray(modulus_up(Xb), dtype=np.float64)) if want_z else self.d_dummy
            else:
                Xd = cp.asarray(np.ascontiguousarray(Xb, dtype=np.float64))
                AXd = None
            Yd = cp.empty((nrows, w), dtype=ydt) if want_y else self.d_dummy
            Zd = cp.empty((nrows, w), dtype=np.float64) if want_z else self.d_dummy
            for c0 in range(0, w, _KB):
                kc = min(_KB, w - c0)
                if cplx:
                    self.k_cplx((grid_x,), (_THREADS,), (self.d_indptr, self.d_indices, data, self.d_abs, Xd, AXd, Yd,
                                                         Zd, np.int64(nrows), np.int64(w), np.int32(c0), np.int32(kc),
                                                         np.int32(want_y), np.int32(want_z)))
                else:
                    self.k_real((grid_x,), (_THREADS,), (self.d_indptr, self.d_indices, data, self.d_abs, Xd, Yd, Zd,
                                                         np.int64(nrows), np.int64(w), np.int32(c0), np.int32(kc),
                                                         np.int32(want_y), np.int32(want_z)))
            if want_y:
                if w == k:
                    Y = cp.asnumpy(Yd)
                else:
                    Y[:, a:b] = cp.asnumpy(Yd)
            if want_z:
                if w == k:
                    Z = cp.asnumpy(Zd)
                else:
                    Z[:, a:b] = cp.asnumpy(Zd)
            del Xd, AXd, Yd, Zd
            if on_block is not None:
                on_block(time.perf_counter() - tb)
        return Y, Z

    def close(self):
        self.d_indptr = self.d_indices = self.d_data = self.d_abs = self.d_data_c = self.d_dummy = None


class DeviceCSR:
    """A CSR matrix and its entrywise modulus on the current CUDA device, with the ordered products of scipy."""

    def __init__(self, A, absA=None, mem_gb: Optional[float] = None, spot_rows: int = SPOT_ROWS, seed: int = 0):
        cp = _cupy()
        self.cp = cp
        if not sp.issparse(A):
            raise GPUUnavailable("the GPU products are implemented for stored sparse matrices only")
        if A.dtype not in (np.float64, np.complex128):
            raise GPUUnavailable(f"dtype {A.dtype} is not float64 or complex128")
        A = sp.csr_matrix(summed(A, "gpu.DeviceCSR"))   # repeated entries summed (precision.summed), then CSR
        absA = abs_up(A) if absA is None else absA.tocsr()
        if not (np.array_equal(absA.indptr, A.indptr) and np.array_equal(absA.indices, A.indices)):
            raise GPUUnavailable("|A| does not share the pattern of A")
        rec = probe()
        if not rec.get("holds"):
            raise GPUUnavailable(f"the GPU floating-point probe does not hold: {rec.get('failed') or rec.get('error')}")
        self.A, self.absA = A, absA
        self.shape = A.shape
        self.complex = A.dtype == np.complex128
        self.budget = min(_mem_budget_bytes(mem_gb), int(0.9 * cp.cuda.runtime.memGetInfo()[0]))
        es = 16 if self.complex else 8
        idx_bytes = 4 if A.indices.dtype == np.int32 else 8
        static = 8 * (A.shape[0] + 1) + A.nnz * (idx_bytes + es + 8)
        if static + 48 * (A.shape[0] + A.shape[1]) > self.budget:
            raise GPUUnavailable(f"the matrix needs {static / 2 ** 20:.0f} MiB on the device, the budget is "
                                 f"{self.budget / 2 ** 20:.0f} MiB")
        self.core = _Ordered(cp, A, absA)
        self.static_bytes = self.core.static_bytes()
        self.spot_rows = int(spot_rows)
        self.rng = np.random.default_rng(seed)
        self.duty = _duty()
        self.record: Dict[str, Any] = {"device": device_info(), "probe": {"holds": True, "checks": rec["checks"]},
                                       "kernel": "ordered CSR (scipy order, --fmad=false), both products in one pass",
                                       "products": 0, "columns": 0, "spot_checked_rows": 0, "seconds": 0.0,
                                       "busy_seconds": 0.0, "spot_check_seconds": 0.0,
                                       "memory_budget_mb": round(self.budget / 2 ** 20)}

    def close(self):
        if getattr(self, "core", None) is not None:
            self.core.close()
            self.core = None
        self.cp.get_default_memory_pool().free_all_blocks()

    def _block_width(self, k: int, cplx: bool, want_y: bool, want_z: bool) -> int:
        free = self.budget - self.static_bytes - self.core.extra_bytes(cplx)
        per_col = self.core.per_column_bytes(cplx, want_y, want_z)
        kb = int(min(k, max(0, free) // max(per_col, 1)))
        if kb < 1:
            raise GPUUnavailable("not enough device memory for one column")
        return kb

    def _on_block(self, busy: float):
        self.record["busy_seconds"] = round(self.record["busy_seconds"] + busy, 6)
        if self.duty is not None:
            time.sleep(min(10.0, busy * (1.0 - self.duty) / self.duty))

    def products(self, X, hx: bool = True, ahx: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(A @ X, |A| @ |X|) as numpy and scipy compute them (same dtype, same shape, C order), either of them
        skipped with hx=False or ahx=False; bit for bit equal to the CPU products (spot-checked)."""
        if self.core is None:
            raise GPUUnavailable("the device copy was closed")
        t0 = time.perf_counter()
        X = np.asarray(X)
        one = X.ndim == 1
        X2 = X[:, None] if one else X
        nrows, ncols = self.shape
        if X2.ndim != 2 or X2.shape[0] != ncols:
            raise ValueError(f"dimension mismatch: {self.shape} @ {X.shape}")
        if X2.dtype not in (np.float64, np.complex128):
            raise GPUUnavailable(f"vector dtype {X2.dtype} is not float64 or complex128")
        if not hx and np.iscomplexobj(X2):
            X2 = modulus_up(X2)             # |A| |X| alone: the upper moduli, as the CPU path takes them
        k = X2.shape[1]
        if k == 0 or nrows == 0:
            Y = np.empty((nrows, k), dtype=np.result_type(self.A.dtype, X2.dtype)) if hx else None
            Z = np.empty((nrows, k)) if ahx else None
        else:
            cplx = bool(hx and (self.complex or np.iscomplexobj(X2)))
            kb = self._block_width(k, cplx, hx, ahx)
            X2 = np.ascontiguousarray(X2) if kb == k else X2
            Y, Z = self.core.run(X2, hx, ahx, kb, on_block=self._on_block)
            self.cp.get_default_memory_pool().free_all_blocks()
            ts = time.perf_counter()
            self._spot_check(X2, Y, Z)
            self.record["spot_check_seconds"] = round(self.record["spot_check_seconds"] + time.perf_counter() - ts, 6)
        self.record["products"] += int(hx) + int(ahx)
        self.record["columns"] += int(k)
        self.record["seconds"] = round(self.record["seconds"] + (time.perf_counter() - t0), 6)
        if one:
            Y = Y[:, 0] if Y is not None else None
            Z = Z[:, 0] if Z is not None else None
        return Y, Z

    def matmul(self, X, absolute: bool = False) -> np.ndarray:
        """A @ X (absolute=False) or |A| @ X with X = |X| real and nonnegative (absolute=True)."""
        if absolute:
            return self.products(X, hx=False, ahx=True)[1]
        return self.products(X, hx=True, ahx=False)[0]

    @staticmethod
    def _rows_compact(M, rows):
        """Rows `rows` of the CSR matrix M, entries in their stored order, with the columns renamed 0..c-1, and the
        original column indices: M_c @ X[cols] performs, row by row, the same operations as (M @ X)[rows]. Costs
        O(entries of the rows), not O(n)."""
        ip = M.indptr
        starts, ends = ip[rows].astype(np.int64), ip[rows + 1].astype(np.int64)
        lens = ends - starts
        pos = (np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
               if rows.size else np.empty(0, dtype=np.int64))
        ind = M.indices[pos]
        cols, inv = np.unique(ind, return_inverse=True)
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        Mc = sp.csr_matrix((M.data[pos], inv.astype(np.int64, copy=False), indptr), shape=(rows.size, cols.size))
        return Mc, cols

    def _spot_check(self, X2, Y, Z):
        nrows = self.shape[0]
        m = min(self.spot_rows, nrows)
        rows = np.sort(self.rng.choice(nrows, size=m, replace=False)) if m < nrows else np.arange(nrows)
        if Y is not None:
            Mc, cols = self._rows_compact(self.A, rows)
            ref = Mc @ X2[cols]
            if not same_bits(np.ascontiguousarray(Y[rows]), np.ascontiguousarray(ref.astype(Y.dtype, copy=False))):
                raise GPUUnavailable("spot check: a row of H X formed on the GPU differs from the CPU product")
        if Z is not None:
            Mc, cols = self._rows_compact(self.absA, rows)
            ref = Mc @ modulus_up(X2[cols])
            if not same_bits(np.ascontiguousarray(Z[rows]), np.ascontiguousarray(ref.astype(np.float64, copy=False))):
                raise GPUUnavailable("spot check: a row of |H| |X| formed on the GPU differs from the CPU product")
        self.record["spot_checked_rows"] += int(m)


def _random_csr(rng, n, m, density, cplx, sorted_indices=True, extreme=False):
    A = sp.random(n, m, density=density, format="csr", random_state=np.random.RandomState(int(rng.integers(2 ** 31))),
                  dtype=np.float64)
    if cplx:
        B = A.copy()
        B.data = rng.standard_normal(B.nnz)
        A = (A + 1j * B).tocsr()
    if extreme and A.nnz:
        idx = rng.choice(A.nnz, size=max(1, A.nnz // 10), replace=False)
        A.data[idx] *= rng.choice(np.array([2.0 ** -1060, 2.0 ** 500, 2.0 ** -500]), idx.size)
    if not sorted_indices:
        for i in range(A.shape[0]):
            s, t = A.indptr[i], A.indptr[i + 1]
            perm = rng.permutation(t - s)
            A.indices[s:t] = A.indices[s:t][perm]
            A.data[s:t] = A.data[s:t][perm]
        A.has_sorted_indices = False
    return A


def _with_empty_rows(A, rows):
    """A copy of A whose given rows store no entry (their data zeroed, then removed; the order of the rest is kept)."""
    A = A.copy()
    for r in rows:
        A.data[A.indptr[r]:A.indptr[r + 1]] = 0
    A.eliminate_zeros()
    return A


def _moduli_in_stored_order(A):
    """The entrywise moduli (modulus_up) of the CSR matrix A without repeated entries, on A's own stored pattern and
    order. abs_up returns the summed matrix, whose indices are sorted; the ordered kernels read |A| in A's stored
    order, so the probe's unsorted matrices need the moduli in that order."""
    return sp.csr_matrix((modulus_up(A.data), A.indices.copy(), A.indptr.copy()), shape=A.shape)


def probe(force: bool = False) -> Dict[str, Any]:
    """The floating-point checks of the current device that the ordered kernels rely on (cached per device).
    Returns {'holds': bool, 'checks': {...}, 'failed': [...], 'device': {...}} (or 'error' when a check could
    not run)."""
    cp = _cupy()
    dev = int(cp.cuda.Device().id)
    if dev in _PROBES and not force:
        return _PROBES[dev]
    rec: Dict[str, Any] = {"holds": False, "checks": {}, "failed": []}
    try:
        rec["device"] = device_info()
        rng = np.random.default_rng(20261117)
        ops = _module(cp, "int").get_function("ops")

        def bits_from(n, emin, emax):
            sign = rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
            ex = rng.integers(emin, emax + 1, n, dtype=np.uint64) << np.uint64(52)
            man = rng.integers(0, 2 ** 52, n, dtype=np.uint64)
            return (sign | ex | man).view(np.float64)

        fams = {"full_range": (1, 2046), "narrow": (1023 - 60, 1023 + 60), "low": (0, 60)}
        n = 20000
        for name, (lo, hi) in fams.items():
            a, b = bits_from(n, lo, hi), bits_from(n, lo, hi)
            o = cp.empty(3 * n, dtype=cp.float64)
            ops(((n + _THREADS - 1) // _THREADS,), (_THREADS,), (cp.asarray(a), cp.asarray(b), o, np.int64(n)))
            g = cp.asnumpy(o).reshape(n, 3)
            with np.errstate(all="ignore"):
                ok = same_bits(g[:, 0], a + b) and same_bits(g[:, 1], a - b) and same_bits(g[:, 2], a * b)
            rec["checks"][f"ops_{name}"] = bool(ok)
        tiny = 2.0 ** -1074
        a = np.array([tiny, 2.0 ** -1022, 3 * tiny, 2.0 ** -1000, 2.0 ** -1022, tiny])
        b = np.array([1.0, 0.5, tiny, 2.0 ** -60, -(2.0 ** -1022 - 2 * tiny), tiny])
        o = cp.empty(3 * a.size, dtype=cp.float64)
        ops((1,), (64,), (cp.asarray(a), cp.asarray(b), o, np.int64(a.size)))
        g = cp.asnumpy(o).reshape(a.size, 3)
        with np.errstate(all="ignore"):
            rec["checks"]["gradual_underflow"] = bool(same_bits(g[:, 0], a + b) and same_bits(g[:, 1], a - b)
                                                      and same_bits(g[:, 2], a * b) and g[0, 2] == tiny)
        kern_ok = True
        for cplx in (False, True):
            for sorted_idx in (True, False):
                A = _with_empty_rows(_random_csr(rng, 300, 280, 0.05, cplx, sorted_idx, extreme=True), (0, 10, 11, 299))
                aA = _moduli_in_stored_order(A)
                core = _Ordered(cp, A, aA)
                X = rng.standard_normal((280, 11)) * rng.choice(np.array([1.0, 2.0 ** -540, 2.0 ** 300]), (280, 11))
                Xc = X + 1j * rng.standard_normal((280, 11))
                for Xv in (X, Xc, np.ascontiguousarray(X[:, :1])):
                    for kb in (Xv.shape[1], 3):
                        Y, Z = core.run(np.ascontiguousarray(Xv), True, True, kb)
                        kern_ok = (kern_ok and same_bits(Y, np.asarray(A @ Xv))
                                   and same_bits(Z, np.asarray(aA @ modulus_up(Xv))))
                    _, Z = core.run(np.ascontiguousarray(modulus_up(Xv)), False, True, Xv.shape[1])
                    kern_ok = kern_ok and same_bits(Z, np.asarray(aA @ modulus_up(Xv)))
                core.close()
        rec["checks"]["kernels_match_scipy"] = bool(kern_ok)
        rec["failed"] = [k for k, v in rec["checks"].items() if not v]
        rec["holds"] = not rec["failed"]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["holds"] = False
    _PROBES[dev] = rec
    return rec
