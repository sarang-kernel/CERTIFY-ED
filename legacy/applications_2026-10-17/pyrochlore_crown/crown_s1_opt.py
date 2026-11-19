"""Spin-1 crown, reduced model with c5 = 0: maximise min over ALL pair-spin configurations.
Stage 1 optimises over the configurations that are lowest at the published couplings (cheap ones
first); stage 2 evaluates every configuration class at the optimum."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import json, time
import numpy as np, scipy.sparse.linalg as spla
from scipy.optimize import minimize
from crown_pairspin import config_hamiltonian, landscape
import certify_ed as ce, scipy.linalg as la

E6 = la.eigvalsh(ce.build_bitwise(ce.spin1_heisenberg_chain(6)).toarray())[0]
FOCUS = [(0, 0, 0, 4, 0, 4), (0, 4, 4, 0, 4, 4), (0, 4, 0, 4, 0, 4), (0, 2, 0, 4, 0, 4)]

def e_cfg(cfg, c1, c3):
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    if all(x == 0 for x in cfg):
        return -12 * c1 + c17 * E6
    H, _ = config_hamiltonian(2, cfg, c1, c3, c17)
    v0 = np.random.default_rng(1).standard_normal(H.shape[0])
    return float(spla.eigsh(H, k=1, which="SA", tol=1e-11, v0=v0, return_eigenvectors=False)[0])

def f(x):
    c1, c3 = x
    vals = [e_cfg((0,) * 6, c1, c3)] + [e_cfg(c, c1, c3) for c in FOCUS]
    return -min(vals)

if __name__ == "__main__":
    t = time.time()
    r = minimize(f, x0=[0.053466, 0.082726], method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-9, "initial_simplex": [[0.053466, 0.082726], [0.050, 0.0827], [0.0535, 0.085]]})
    c1, c3 = r.x
    print(f"stage 1: c1={c1:.7f} c3={c3:.7f} min over focus = {-r.fun:.8f}  ({time.time()-t:.0f}s)", flush=True)
    L = landscape(2, c1, c3)
    print("stage 2 (all 92 classes) lowest:", flush=True)
    for e, cfg, d, m in L[:6]:
        print(f"  {e:.8f} {tuple(x/2 for x in cfg)} dim {d}")
    json.dump({"c1": c1, "c3": c3, "focus_min": -r.fun, "landscape": [[e, list(cfg), d, m] for e, cfg, d, m in L]},
              open("crown_s1_opt.json", "w"), indent=1)
