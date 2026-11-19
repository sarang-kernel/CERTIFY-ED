"""D-05: the products of S2 (H X and |H| |X|) and the residual bounds on the GPU against the CPU, for Heisenberg chains
in the full space (n = 2^N), bit for bit and in time; then one certify() run with only S1 and S2 on both devices.

usage: python experiments/d05_gpu_bench.py [out.json] [N ...]

The CPU column is scipy's CSR product (one thread) as certify-ed uses it; the GPU column is gpu.DeviceCSR, wall time
including the transfers and the spot check, and busy time without the sleeps of a duty cycle (GPU_DUTY on shared
machines). The certificate comparison drops timing fields and S2's `device` record and compares everything else.
"""
import json
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

import certify_ed as ce
from certify_ed import gpu
from certify_ed.enclosure import OperatorAdapter, residuals


def _without_timings(obj):
    if isinstance(obj, dict):
        return {k: _without_timings(v) for k, v in obj.items()
                if not any(w in k for w in ("second", "time", "wall", "device"))}
    if isinstance(obj, (list, tuple)):
        return [_without_timings(v) for v in obj]
    return obj


def point(N, k, rng):
    t = time.perf_counter()
    H = ce.build_bitwise(ce.heisenberg_chain(N)).tocsr()
    t_build = time.perf_counter() - t
    n = H.shape[0]
    X = rng.standard_normal((n, k))
    mu = rng.standard_normal(k)
    absH = abs(H)
    t = time.perf_counter()
    HX = H @ X
    AX = absH @ np.abs(X)
    t_prod_cpu = time.perf_counter() - t
    t = time.perf_counter()
    rc = residuals(OperatorAdapter(H), mu, X)
    t_res_cpu = time.perf_counter() - t
    ad = OperatorAdapter(H, device="gpu")
    if not ad.device_record["used"]:
        return {"N": N, "n": n, "nnz": int(H.nnz), "gpu_unused": ad.device_record}
    D = ad._gpu
    b0, s0 = D.record.get("busy_seconds", 0.0), D.record.get("spot_check_seconds", 0.0)
    t = time.perf_counter()
    HXg, AXg = D.products(X)                 # one pass over H for both products
    t_prod_gpu = time.perf_counter() - t
    busy = D.record["busy_seconds"] - b0
    spot = D.record["spot_check_seconds"] - s0
    t = time.perf_counter()
    rg = residuals(ad, mu, X)
    t_res_gpu = time.perf_counter() - t
    return {"N": N, "n": n, "nnz": int(H.nnz), "k": k, "build_seconds": round(t_build, 3),
            "products_cpu_seconds": round(t_prod_cpu, 4), "products_gpu_wall_seconds": round(t_prod_gpu, 4),
            "products_gpu_busy_seconds": round(busy, 4), "products_gpu_spot_check_seconds": round(spot, 4),
            "residuals_cpu_seconds": round(t_res_cpu, 4), "residuals_gpu_seconds": round(t_res_gpu, 4),
            "products_same_bits": bool(gpu.same_bits(HX, HXg) and gpu.same_bits(AX, AXg)),
            "residual_upper_same_bits": bool(gpu.same_bits(rc.upper, rg.upper)),
            "fallback": ad.device_record.get("fallback"),
            "cpu_breakdown_seconds": breakdown(H, mu, X)}


def breakdown(H, mu, X):
    """Where the CPU time of enclosure.residuals goes (the same calls, timed one by one)."""
    from certify_ed.enclosure import _rounding_bound
    from certify_ed.fp import norm2_lower, norm2_upper
    ad = OperatorAdapter(H)
    out = {}
    t = time.perf_counter()
    HX = ad.matvec(X)
    out["H_X"] = time.perf_counter() - t
    t = time.perf_counter()
    R = HX - X * mu[None, :]
    out["R_formation"] = time.perf_counter() - t
    t = time.perf_counter()
    np.linalg.norm(R, axis=0)
    np.linalg.norm(X, axis=0)
    out["computed_norms"] = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(X.shape[1]):
        norm2_lower(X[:, i])
        norm2_upper(R[:, i])
    out["column_bounds_R_and_x"] = time.perf_counter() - t
    t = time.perf_counter()
    absHX = ad.abs_matvec(X)
    out["absH_absX"] = time.perf_counter() - t
    t = time.perf_counter()
    E = _rounding_bound(ad, mu, X, absHX)
    out["E_elementwise"] = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(X.shape[1]):
        norm2_upper(E[:, i])
    out["column_bounds_E"] = time.perf_counter() - t
    return {k: round(v, 4) for k, v in out.items()}


def certify_pair(N):
    spec = ce.heisenberg_chain(N)
    H = ce.build_bitwise(spec).tocsr()
    t = time.perf_counter()
    w, V = spla.eigsh(H, k=4, which="SA", tol=1e-12)
    t_eig = time.perf_counter() - t
    claim = ce.Claim(eigenvalues=w, eigenvectors=V, operator=H, kind="lowest")
    out = {"N": N, "n": int(H.shape[0]), "eigsh_seconds": round(t_eig, 2)}
    certs = {}
    for dev in ("cpu", "gpu"):
        t = time.perf_counter()
        try:
            opts = ce.Options(seed=1, device=dev, only=("S2",))
            certs[dev] = ce.certify(claim, spec, opts)
        except (ValueError, KeyError):
            t = time.perf_counter()
            certs[dev] = ce.certify(claim, spec, ce.Options(seed=1, device=dev, only=("S2.eigenpairs",)))
        out[f"certify_{dev}_seconds"] = round(time.perf_counter() - t, 2)
        out[f"S2_{dev}_seconds"] = certs[dev].statement("S2.eigenpairs")["seconds"]
    a, b = certs["cpu"], certs["gpu"]
    out["statuses"] = [(s["id"], s["status"]) for s in a.statements]
    out["same_statuses"] = out["statuses"] == [(s["id"], s["status"]) for s in b.statements]
    out["same_records_without_timings"] = all(
        json.dumps(_without_timings(x), sort_keys=True, default=repr) == json.dumps(_without_timings(y), sort_keys=True, default=repr)
        for x, y in zip(a.statements, b.statements))
    out["same_arrays"] = sorted(a.arrays) == sorted(b.arrays) and all(
        np.asarray(a.arrays[k_]).tobytes() == np.asarray(b.arrays[k_]).tobytes() for k_ in a.arrays)
    out["device_record"] = b.statement("S2.eigenpairs")["evidence"].get("device")
    try:
        from certify_ed import schema
        errs = schema.schema_errors(b.body) if hasattr(schema, "schema_errors") else None
        out["schema_errors_gpu_certificate"] = [str(e)[:200] for e in (errs or [])][:5] if errs is not None else "n/a"
    except Exception as e:
        out["schema_errors_gpu_certificate"] = f"check failed: {type(e).__name__}: {str(e)[:200]}"
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "d05_gpu_bench.json"
    Ns = [int(a) for a in sys.argv[2:]] or [16, 18, 20, 22, 24]
    rng = np.random.default_rng(1)
    rec = {"probe": gpu.probe(), "points": [], "certify": None}
    for N in Ns:
        p = point(N, 8, rng)
        rec["points"].append(p)
        print(json.dumps(p), flush=True)
        json.dump(rec, open(out, "w"), indent=1, default=str)
    rec["certify"] = certify_pair(18)
    print(json.dumps({k: v for k, v in rec["certify"].items() if k != "device_record"}), flush=True)
    json.dump(rec, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
