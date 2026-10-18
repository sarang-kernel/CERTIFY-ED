import sys, numpy as np, scipy.sparse.linalg as spla
from scipy.optimize import minimize
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from crown_certify import spec_for
import certify_ed as ce
from certify_ed.basis import magnetization_state_set
st = None
def e0(x):
    global st
    spec, _ = spec_for(int(round(x[0])), int(round(x[1])))
    if st is None:
        st = magnetization_state_set(spec, 0)
    H = ce.build_bitwise(spec, st)
    v0 = np.random.default_rng(1).standard_normal(H.shape[0])
    return float(spla.eigsh(H, k=1, which="SA", tol=1e-12, v0=v0, return_eigenvectors=False)[0])
best = None
for d1 in range(-30, 31, 5):
    for d3 in range(-30, 31, 5):
        x = (381414 + d1, 817957 + d3); v = e0(x)
        if best is None or v > best[0]: best = (v, x)
print("grid best", best)
