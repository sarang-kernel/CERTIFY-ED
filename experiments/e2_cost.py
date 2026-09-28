"""E2: cost of certification relative to the solve it certifies (time and peak memory).

Each data point runs two fresh processes:
  producer  : QuSpin builds the basis and H and computes the lowest eigenpairs (eigsh);
              the result is written to an .npz file
  certifier : loads the result and runs certify() (S1-S4); reports per-statement times
              and peak resident memory
Recomputing the result with a second code (the practice Referee 1 recommends) costs
about as much as the producer run.
"""

import json
import os
import subprocess
import sys
import time

from harness import Run, HERE, ROOT

PRODUCER = r'''
import sys, json, time, resource, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from quspin.basis import spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian
sys.path.insert(0, ROOT)
import certify_ed as ce
cfg = json.loads(sys.argv[1])
t0 = time.time()
if cfg["model"] == "chain":
    spec = ce.heisenberg_chain(cfg["N"], J2=cfg.get("J2", 0.0))
    kw = dict(Nup=cfg["N"] // 2, pauli=0)
    if cfg["block"] == "k0":
        kw["kblock"] = 0
    basis = spin_basis_1d(cfg["N"], **kw)
else:
    spec = ce.heisenberg_square(cfg["Lx"], cfg["Ly"], J2=cfg.get("J2", 0.0))
    basis = spin_basis_general(cfg["Lx"] * cfg["Ly"], Nup=cfg["Lx"] * cfg["Ly"] // 2, pauli=0)
static = spec.to_quspin_static()
dtype = np.float64
H = hamiltonian(static, [], basis=basis, dtype=dtype, check_herm=False, check_symm=False, check_pcon=False)
t_build = time.time() - t0
t1 = time.time()
w, V = H.eigsh(k=cfg["k"], which="SA", tol=0)
t_solve = time.time() - t1
order = np.argsort(w); w, V = w[order], V[:, order]
Hc = H.tocsr()
out = {"t_build": t_build, "t_solve": t_solve, "dim": int(basis.Ns), "nnz": int(Hc.nnz),
       "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
d = dict(H_data=Hc.data, H_indices=Hc.indices, H_indptr=Hc.indptr, H_shape=np.array(Hc.shape), w=w, V=V,
         static=json.dumps(static), states=np.asarray(basis.states, dtype=np.int64))
if cfg["block"] == "k0":
    P = sp.csr_matrix(basis.get_proj(np.float64))
    d.update(P_data=P.data, P_indices=P.indices, P_indptr=P.indptr, P_shape=np.array(P.shape))
np.savez(cfg["npz"], **d)
print("RESULT" + json.dumps(out))
'''.replace("ROOT", repr(ROOT))

CERTIFIER = r'''
import sys, json, time, resource, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
sys.path.insert(0, ROOT)
import certify_ed as ce
from certify_ed.construct import StateSet
cfg = json.loads(sys.argv[1])
z = np.load(cfg["npz"])
H = sp.csr_matrix((z["H_data"], z["H_indices"], z["H_indptr"]), shape=tuple(z["H_shape"]))
w, V = z["w"], z["V"]
if cfg["model"] == "chain":
    spec = ce.heisenberg_chain(cfg["N"], J2=cfg.get("J2", 0.0))
else:
    spec = ce.heisenberg_square(cfg["Lx"], cfg["Ly"], J2=cfg.get("J2", 0.0))
k = 4
while k < len(w) and w[k] - w[k - 1] < 1e-6:
    k += 1
k = min(k, len(w) - 1)
kw = {}
if cfg["block"] == "k0":
    kw["sector_isometry"] = sp.csr_matrix((z["P_data"], z["P_indices"], z["P_indptr"]), shape=tuple(z["P_shape"]))
else:
    kw["states"] = StateSet(np.sort(z["states"]), "descending", "U(1)")
claim = ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest", **kw)
opts = ce.Options(inertia_max_dim=cfg.get("inertia_max_dim", 10**7), inertia_method="sparse")
t0 = time.time()
cert = ce.certify(claim, spec, opts)
tot = time.time() - t0
st = {s["id"]: {"status": s["status"], "seconds": s["seconds"], "summary": s["summary"][:200]} for s in cert.statements}
s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]["evidence"]
fill = [c.get("fill_nnz") for c in s3.get("counts", [])]
out = {"total": tot, "statements": st, "k_claimed": int(k), "fill_nnz": fill,
       "gap_lower_bound": s3.get("gap_lower_bound"), "true_gap_estimate": float(w[k] - w[k - 1]) if k < len(w) else None,
       "max_radius": float(cert.arrays["radius"].max()),
       "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
print("RESULT" + json.dumps(out))
'''.replace("ROOT", repr(ROOT))


def run_child(code, cfg, timeout):
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", code, json.dumps(cfg)], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}, time.time() - t
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:]), time.time() - t
    return {"error": (p.stderr or p.stdout)[-600:]}, time.time() - t


def main():
    points = [dict(model="chain", block="u1", N=N) for N in (12, 14, 16, 18)] + \
             [dict(model="chain", block="k0", N=N) for N in (16, 18, 20, 22, 24)] + \
             [dict(model="square", block="u1", Lx=4, Ly=4), dict(model="square", block="u1", Lx=4, Ly=4, J2=0.5)]
    with Run("e2_cost") as run:
        rows = []
        for i, pt in enumerate(points):
            cfg = dict(pt, k=8, npz=os.path.join(run.dir, f"point{i}.npz"))
            label = (f"{pt['model']} N={pt.get('N', pt.get('Lx', 0) * pt.get('Ly', 0))} "
                     f"{pt['block']} J2={pt.get('J2', 0.0)}")
            prod, tp = run_child(PRODUCER, cfg, 1800)
            if "error" in prod:
                run.log(f"{label}: producer failed: {prod['error'][-200:]}")
                rows.append(dict(label=label, cfg=pt, producer=prod))
                continue
            cert, tc = run_child(CERTIFIER, cfg, 1500)
            rec = dict(label=label, cfg=pt, producer=prod, certifier=cert)
            if "error" in cert:
                # completeness (S3) unaffordable: measure S1+S2 alone
                cfg2 = dict(cfg, inertia_max_dim=0)
                cert2, _ = run_child(CERTIFIER, cfg2, 1500)
                rec["certifier_without_S3"] = cert2
                run.log(f"{label}: S3 failed ({cert['error'][-120:]}); without S3: "
                        f"{cert2.get('total', float('nan')):.1f}s, mem {cert2.get('maxrss_mb', float('nan')):.0f}MB")
            rows.append(rec)
            run.record("rows", rows)
            if "error" in cert:
                run.log(f"{label}: dim {prod['dim']} solve {prod['t_solve']:.1f}s; certifier failed: {cert['error'][-300:]}")
            else:
                s = cert["statements"]
                run.log(f"{label}: dim {prod['dim']:>7d} build {prod['t_build']:.1f}s solve {prod['t_solve']:.1f}s "
                        f"| certify {cert['total']:.1f}s (S1 {s['S1.model_fidelity']['seconds']:.1f}, "
                        f"S2 {s['S2.eigenpairs']['seconds']:.1f}, S3 {s['S3.completeness']['seconds']:.1f}) "
                        f"S3={s['S3.completeness']['status']} mem {cert['maxrss_mb']:.0f}MB vs {prod['maxrss_mb']:.0f}MB")
            try:
                os.remove(cfg["npz"])
            except OSError:
                pass
        run.record("rows", rows)


if __name__ == "__main__":
    main()
