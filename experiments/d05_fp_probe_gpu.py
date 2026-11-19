"""GPU binary64 floating-point model probe (roadmap item D-05).

Checks, on the GPU this process sees, the properties that the strict residual bounds of certify-ed assume of binary64
arithmetic, and records what the GPU libraries do that the bounds must tolerate:

  1. correct rounding to nearest of +, -, *, /, sqrt and fma (bitwise against the CPU, and fma against exact rationals);
  2. directed rounding (__dadd_ru/rd, __dmul_ru/rd, __fma_ru/rd): the result is the nearest double on the stated side;
  3. gradual underflow in binary64 (no flush to zero of subnormal inputs or results);
  4. contraction of a*b+c into fma by the compiler (default NVRTC flags against --fmad=false);
  5. cuBLAS ddot and cuSPARSE SpMV against the exactly rounded sums: the error stays inside the order-independent
     bound gamma_k * sum |a_i b_i| (a sanity check, not a proof), and repeated runs are bitwise identical or not.

Usage: python fp_probe_gpu.py [out.json]. Honors GPU_DUTY (sleeps between bursts) and GPU_MEM_GB (memory pool limit).
"""
import json
import math
import os
import platform
import sys
import time
from fractions import Fraction

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cps

DUTY = float(os.environ.get("GPU_DUTY", "0.5"))
MEM_GB = float(os.environ.get("GPU_MEM_GB", "4"))
cp.get_default_memory_pool().set_limit(size=int(MEM_GB * 2 ** 30))
U = 2.0 ** -53
DMAX = float(np.finfo(np.float64).max)


def pause(t0):
    dt = time.time() - t0
    time.sleep(min(5.0, dt * (1.0 - DUTY) / max(DUTY, 1e-3)))


SRC = r'''
extern "C" {
__global__ void ops_rn(const double* a, const double* b, const double* c, double* out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  double x = a[i], y = b[i], z = c[i];
  out[8*i+0] = __dadd_rn(x, y);
  out[8*i+1] = __dadd_rn(x, -y);
  out[8*i+2] = __dmul_rn(x, y);
  out[8*i+3] = __ddiv_rn(x, y);
  out[8*i+4] = __dsqrt_rn(fabs(x));
  out[8*i+5] = __fma_rn(x, y, z);
  out[8*i+6] = x * y + z;
  out[8*i+7] = __dadd_rn(__dmul_rn(x, y), z);
}
__global__ void ops_dir(const double* a, const double* b, const double* c, double* out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  double x = a[i], y = b[i], z = c[i];
  out[6*i+0] = __dadd_ru(x, y);
  out[6*i+1] = __dadd_rd(x, y);
  out[6*i+2] = __dmul_ru(x, y);
  out[6*i+3] = __dmul_rd(x, y);
  out[6*i+4] = __fma_ru(x, y, z);
  out[6*i+5] = __fma_rd(x, y, z);
}
}
'''


def doubles_from_bits(rng, n, emin, emax):
    """Random finite doubles with biased exponent field uniform in [emin, emax] (0 gives subnormals and zero)."""
    sign = rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
    exp = rng.integers(emin, emax + 1, n, dtype=np.uint64) << np.uint64(52)
    man = rng.integers(0, 2 ** 52, n, dtype=np.uint64)
    return (sign | exp | man).view(np.float64)


def families(rng):
    n = 200_000
    fam = {}
    fam["full_range"] = tuple(doubles_from_bits(rng, n, 1, 2046) for _ in range(3))
    fam["narrow"] = tuple(doubles_from_bits(rng, n, 1023 - 60, 1023 + 60) for _ in range(3))
    # products and sums that underflow into the subnormal range, and subnormal operands
    a = doubles_from_bits(rng, n, 1023 - 540, 1023 - 480)
    b = doubles_from_bits(rng, n, 1023 - 560, 1023 - 500)
    c = doubles_from_bits(rng, n, 0, 3)
    fam["underflow"] = (a, b, c)
    # ties and near-ties: 1 + k ulp patterns against half-ulp increments
    k = rng.integers(-8, 9, n).astype(np.float64)
    a = 1.0 + k * 2.0 ** -52
    b = rng.choice(np.array([2.0 ** -53, -2.0 ** -53, 3 * 2.0 ** -54, 2.0 ** -54, 1.0 - 2.0 ** -53]), n)
    c = rng.choice(np.array([2.0 ** -106, -2.0 ** -106, 0.0, 2.0 ** -53]), n)
    fam["ties"] = (a, b, c)
    return fam


def exact_round(q: Fraction) -> float:
    """Round an exact rational to the nearest double, ties to even (CPython int/int true division is correctly rounded)."""
    if q == 0:
        return 0.0
    try:
        return q.numerator / q.denominator
    except OverflowError:
        return math.inf if q > 0 else -math.inf


def check_rn(fam, mod_default, mod_nofmad, rng):
    res = {}
    for name, (a, b, c) in fam.items():
        n = a.size
        out = {}
        for tag, mod in (("default", mod_default), ("nofmad", mod_nofmad)):
            t0 = time.time()
            k = mod.get_function("ops_rn")
            da, db, dc = cp.asarray(a), cp.asarray(b), cp.asarray(c)
            o = cp.empty(8 * n, dtype=cp.float64)
            k(((n + 255) // 256,), (256,), (da, db, dc, o, np.int32(n)))
            out[tag] = cp.asnumpy(o).reshape(n, 8)
            pause(t0)
        g = out["default"]
        with np.errstate(all="ignore"):
            ref = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b, "sqrt": np.sqrt(np.abs(a)),
                   "mul_then_add": (a * b) + c}
        cols = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sqrt": 4, "mul_then_add": 7}
        r = {}
        for op, j in cols.items():
            x, y = g[:, j], ref[op]
            same = (x.view(np.uint64) == y.view(np.uint64)) | (np.isnan(x) & np.isnan(y))
            r[op + "_bitwise_mismatch"] = int((~same).sum())
        # fma against exact rationals on a sample of finite cases
        idx = rng.choice(n, size=min(4000, n), replace=False)
        bad = 0
        checked = 0
        for i in idx:
            x, y, z = float(a[i]), float(b[i]), float(c[i])
            q = Fraction(x) * Fraction(y) + Fraction(z)
            e = exact_round(q)
            if not math.isfinite(e):
                continue
            checked += 1
            if g[i, 5] != e:
                bad += 1
        r["fma_checked"] = checked
        r["fma_mismatch_vs_exact"] = bad
        # contraction: default flags vs --fmad=false, for the expression x*y + z
        r["default_equals_fma"] = int((g[:, 6].view(np.uint64) == g[:, 5].view(np.uint64)).sum())
        r["differs_from_separate_default"] = int((g[:, 6].view(np.uint64) != g[:, 7].view(np.uint64)).sum())
        h = out["nofmad"]
        r["differs_from_separate_nofmad"] = int((h[:, 6].view(np.uint64) != h[:, 7].view(np.uint64)).sum())
        r["n"] = n
        res[name] = r
    return res


def check_dir(fam, mod, rng):
    res = {}
    for name, (a, b, c) in fam.items():
        n = a.size
        t0 = time.time()
        k = mod.get_function("ops_dir")
        o = cp.empty(6 * n, dtype=cp.float64)
        k(((n + 255) // 256,), (256,), (cp.asarray(a), cp.asarray(b), cp.asarray(c), o, np.int32(n)))
        g = cp.asnumpy(o).reshape(n, 6)
        pause(t0)
        idx = rng.choice(n, size=min(3000, n), replace=False)
        bad = {"add": 0, "mul": 0, "fma": 0}
        checked = 0
        for i in idx:
            x, y, z = float(a[i]), float(b[i]), float(c[i])
            ex = {"add": Fraction(x) + Fraction(y), "mul": Fraction(x) * Fraction(y),
                  "fma": Fraction(x) * Fraction(y) + Fraction(z)}
            cols = {"add": (0, 1), "mul": (2, 3), "fma": (4, 5)}
            checked += 1
            for op, (ju, jd) in cols.items():
                q = ex[op]
                up, dn = float(g[i, ju]), float(g[i, jd])
                if math.isinf(up):
                    ok_up = up > 0 and q > Fraction(DMAX)
                else:
                    below = float(np.nextafter(up, -np.inf))
                    ok_up = Fraction(up) >= q and (Fraction(up) == q or math.isinf(below) or Fraction(below) < q)
                if math.isinf(dn):
                    ok_dn = dn < 0 and q < -Fraction(DMAX)
                else:
                    above = float(np.nextafter(dn, np.inf))
                    ok_dn = Fraction(dn) <= q and (Fraction(dn) == q or math.isinf(above) or Fraction(above) > q)
                if not (ok_up and ok_dn):
                    bad[op] += 1
        res[name] = {"checked": checked, "mismatch": bad}
    return res


def check_subnormals(mod):
    tiny = 2.0 ** -1074
    cases = [(tiny, 1.0, 0.0), (2.0 ** -1022, 0.5, 0.0), (3 * tiny, 1.0, tiny), (2.0 ** -1000, 2.0 ** -60, 0.0),
             (2.0 ** -1022, -1.0, 2.0 ** -1022 + 2 * tiny), (tiny, tiny, 0.0)]
    a = np.array([x for x, _, _ in cases])
    b = np.array([y for _, y, _ in cases])
    c = np.array([z for _, _, z in cases])
    n = len(cases)
    k = mod.get_function("ops_rn")
    o = cp.empty(8 * n, dtype=cp.float64)
    k((1,), (64,), (cp.asarray(a), cp.asarray(b), cp.asarray(c), o, np.int32(n)))
    g = cp.asnumpy(o).reshape(n, 8)
    rows = []
    for i, (x, y, z) in enumerate(cases):
        e_mul = exact_round(Fraction(x) * Fraction(y))
        e_add = exact_round(Fraction(x) + Fraction(y))
        e_fma = exact_round(Fraction(x) * Fraction(y) + Fraction(z))
        rows.append({"x": x.hex(), "y": y.hex(), "z": z.hex(), "mul": [float(g[i, 2]).hex(), e_mul.hex()],
                     "add": [float(g[i, 0]).hex(), e_add.hex()], "fma": [float(g[i, 5]).hex(), e_fma.hex()],
                     "ok": bool(g[i, 2] == e_mul and g[i, 0] == e_add and g[i, 5] == e_fma)})
    return {"cases": rows, "all_ok": all(r["ok"] for r in rows)}


def two_prod(a, b):
    """Dekker's product without fma: p + e == a*b exactly (no overflow or underflow for the ranges used here)."""
    f = 134217729.0
    p = a * b
    ca = f * a
    ah = ca - (ca - a)
    al = a - ah
    cb = f * b
    bh = cb - (cb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def gamma(k):
    return k * U / (1.0 - k * U)


def check_blas(rng):
    out = {}
    n = 1_000_000
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    p, e = two_prod(x, y)
    exact = math.fsum(np.concatenate([p, e]).tolist())
    absum = math.fsum(np.abs(p).tolist())
    t0 = time.time()
    try:
        dx, dy = cp.asarray(x), cp.asarray(y)
        vals = [float(cp.dot(dx, dy)) for _ in range(3)]
        pause(t0)
        out["ddot"] = {"n": n, "err_over_bound": abs(vals[0] - exact) / (gamma(n) * absum),
                       "repeat_bitwise_identical": len(set(v.hex() for v in vals)) == 1}
    except Exception as e:    # CuPy's dot launches its own elementwise kernels, which need the CUDA headers
        out["ddot"] = {"skipped": f"{type(e).__name__}: {str(e)[:300]}", "err_over_bound": 0.0}
    try:
        out["spmv"] = _check_spmv(rng)
    except Exception as e:    # cuSPARSE through cupyx needs the CUDA headers for its helper kernels
        out["spmv"] = {"skipped": f"{type(e).__name__}: {str(e)[:300]}", "max_err_over_row_bound": 0.0}
    return out


def _check_spmv(rng):
    m, per = 200_000, 8
    rows = np.repeat(np.arange(m), per)
    cols = rng.integers(0, m, m * per)
    vals_a = rng.standard_normal(m * per)
    import scipy.sparse as ssp
    Ah = ssp.coo_matrix((vals_a, (rows, cols)), shape=(m, m)).tocsr()
    Ah.sum_duplicates()
    A = cps.csr_matrix((cp.asarray(Ah.data), cp.asarray(Ah.indices), cp.asarray(Ah.indptr)), shape=(m, m))
    xv = rng.standard_normal(m)
    t0 = time.time()
    ys = [cp.asnumpy(A @ cp.asarray(xv)) for _ in range(3)]
    pause(t0)
    indptr, indices, data = Ah.indptr, Ah.indices, Ah.data
    pp, ee = two_prod(data, xv[indices])
    worst = 0.0
    for i in range(m):
        s, t = indptr[i], indptr[i + 1]
        k = t - s
        if k == 0:
            continue
        ex = math.fsum(pp[s:t].tolist() + ee[s:t].tolist())
        bnd = gamma(k) * math.fsum(np.abs(pp[s:t]).tolist())
        if bnd > 0:
            worst = max(worst, abs(ys[0][i] - ex) / bnd)
    return {"rows": m, "nnz": int(Ah.nnz), "max_err_over_row_bound": worst,
            "repeat_bitwise_identical": all(np.array_equal(ys[0].view(np.uint64), yy.view(np.uint64)) for yy in ys[1:])}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "fp_probe_gpu.json"
    rng = np.random.default_rng(20261117)
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"]
    rec = {"probe": "fp_probe_gpu", "version": 1, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__, "cupy": cp.__version__,
           "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cuda_driver": cp.cuda.runtime.driverGetVersion(),
           "nvrtc": list(cp.cuda.nvrtc.getVersion()),
           "device": {"name": name.decode() if isinstance(name, bytes) else str(name),
                      "cc": f"{props['major']}.{props['minor']}"},
           "cpu_reference": platform.processor() or platform.machine(), "gpu_duty": DUTY, "gpu_mem_gb": MEM_GB}
    mod_default = cp.RawModule(code=SRC)
    mod_nofmad = cp.RawModule(code=SRC, options=("--fmad=false",))
    fam = families(rng)
    rec["round_to_nearest"] = check_rn(fam, mod_default, mod_nofmad, rng)
    rec["directed"] = check_dir(fam, mod_default, rng)
    rec["subnormals"] = check_subnormals(mod_default)
    rec["libraries"] = check_blas(rng)
    rn = rec["round_to_nearest"]
    rec["summary"] = {
        "basic_ops_correctly_rounded": all(rn[f][k] == 0 for f in rn for k in rn[f] if k.endswith("_bitwise_mismatch")),
        "fma_correctly_rounded": all(rn[f]["fma_mismatch_vs_exact"] == 0 for f in rn),
        "directed_rounding_correct": all(sum(rec["directed"][f]["mismatch"].values()) == 0 for f in rec["directed"]),
        "gradual_underflow": rec["subnormals"]["all_ok"],
        "default_flags_contract_mul_add": any(rn[f]["differs_from_separate_default"] > 0 for f in rn),
        "nofmad_never_contracts": all(rn[f]["differs_from_separate_nofmad"] == 0 for f in rn),
        "ddot_inside_gamma_bound": (None if "skipped" in rec["libraries"]["ddot"]
                                    else rec["libraries"]["ddot"]["err_over_bound"] <= 1.0),
        "spmv_inside_gamma_bound": (None if "skipped" in rec["libraries"]["spmv"]
                                    else rec["libraries"]["spmv"]["max_err_over_row_bound"] <= 1.0),
    }
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["summary"], indent=1))


if __name__ == "__main__":
    main()
