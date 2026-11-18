"""Stage 3: include the all-S~=2 sector (d = 1,116,229) in the max-min over couplings."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import json, time
import numpy as np
from scipy.optimize import minimize
from crown_s1_opt import e_cfg, FOCUS
from crown_pairspin import landscape

ALL2 = (4, 4, 4, 4, 4, 4)
log = open("crown_s1_opt3.log", "a")
def f(x):
    c1, c3 = x
    vals = [e_cfg((0,) * 6, c1, c3)] + [e_cfg(c, c1, c3) for c in FOCUS + [ALL2]]
    log.write(f"{c1:.7f} {c3:.7f} {min(vals):.8f} {np.argmin(vals)}\n"); log.flush()
    return -min(vals)
x0 = json.load(open("crown_s1_opt.json"))
r = minimize(f, x0=[x0["c1"], x0["c3"]], method="Nelder-Mead",
             options={"xatol": 2e-7, "fatol": 2e-8, "maxfev": 45,
                      "initial_simplex": [[x0["c1"], x0["c3"]], [x0["c1"] - 0.0015, x0["c3"]], [x0["c1"], x0["c3"] + 0.0015]]})
c1, c3 = r.x
L = landscape(2, c1, c3)
json.dump({"c1": c1, "c3": c3, "maxmin": -r.fun, "landscape": [[e, list(cfg), d, m] for e, cfg, d, m in L]},
          open("crown_s1_opt3.json", "w"), indent=1)
log.write(f"DONE c1={c1:.7f} c3={c3:.7f} min over all classes = {L[0][0]:.8f}\n")
