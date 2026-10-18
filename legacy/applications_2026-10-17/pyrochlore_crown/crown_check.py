"""Reconstruct the Kranitz-Penc 18-site crown cluster (reduced model, c5 = 0) and compare E0 with
their Eq. (67): e_LB = -0.550158(2) at c1 = 0.038141(4), c3 = 0.081795(7)."""
import sys, time, numpy as np, scipy.sparse.linalg as spla
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import certify_ed as ce
from certify_ed.basis import magnetization_state_set

pairs = [(1, 2), (6, 12), (9, 16), (17, 18), (7, 13), (3, 10)]
hexagon = [4, 5, 11, 15, 14, 8]
# outer site of each pair -> its two hexagon neighbours (from the c3 bond list of Eq. D3)
tets = {(1, 2): (4, 5), (6, 12): (5, 11), (9, 16): (11, 15), (17, 18): (15, 14), (7, 13): (14, 8), (3, 10): (8, 4)}

def crown_bonds(c1, c3, c12, c17):
    b = []
    for p in pairs:
        b.append((c1, p[0], p[1]))
    for p, (h1, h2) in tets.items():
        for o in p:
            for h in (h1, h2):
                b.append((c3 if o in (1, 12, 9, 18, 7, 10) else c12, o, h))
    for i in range(6):
        b.append((c17, hexagon[i], hexagon[(i + 1) % 6]))
    return b

def spec_from_bonds(bonds):
    zz = [[c, i - 1, j - 1] for c, i, j in bonds]
    pm = [[c / 2, i - 1, j - 1] for c, i, j in bonds]
    return ce.ModelSpec.from_static(18, ce.LocalSpace.spin_space("1/2", "spin"), [["zz", zz], ["+-", pm], ["-+", pm]], name="crown")

c1, c3 = 0.038141, 0.081795
c17 = (1 - 2 * c1 - 8 * c3) / 2
bonds = crown_bonds(c1, c3, c3, c17)
assert len(bonds) == 36
spec = spec_from_bonds(bonds)
st = magnetization_state_set(spec, 0)
t = time.time()
H = ce.build_bitwise(spec, st)
print("d =", H.shape[0], "nnz", H.nnz, f"build {time.time()-t:.1f}s")
v0 = np.random.default_rng(1).standard_normal(H.shape[0])
w = spla.eigsh(H, k=4, which="SA", tol=0, v0=v0, return_eigenvectors=False)
print("c17 =", c17, " lowest:", np.sort(w))
