"""Floating-point estimate of eta for the decorated square (geometry check before certification)."""
import sys, os, time
import numpy as np, scipy.linalg as la
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import pw_reduced as P
from lattices import decorated_square
import flint
g, L, C, R = decorated_square()
t0 = time.time()
rL, rC, rR = (P.reduced_region(g, X) for X in (L, C, R))
Ti, di, si, _ = P.union_coords(g, [L, C], [rL, rC]); Tj, dj, sj, _ = P.union_coords(g, [C, R], [rC, rR])
f = lambda G: np.array([[float(x) for x in row] for row in G])
GL, GC, GR = f(rL["Gn"]), f(rC["Gn"]), f(rR["Gn"])
s = max(GL.max(), 1.0); 
nL, nC, nR, fi, fj = rL["n"], rC["n"], rR["n"], len(Ti), len(Tj)
def dense(T, shape):
    M = np.zeros(shape + (len(T),))
    for k, col in enumerate(T):
        for key, v in col.items():
            M[key + (k,)] += v
    return M
Ti_d = dense(Ti, (nL, nC)); Tj_d = dense(Tj, (nC, nR))
# normalise each factor metric (scale invariance of cosines)
GL /= np.max(np.abs(GL)); GC /= np.max(np.abs(GC)); GR /= np.max(np.abs(GR))
Gi = np.einsum("lca,lm,cd,mdb->ab", Ti_d, GL, GC, Ti_d, optimize=True); Gj = np.einsum("cra,cd,rs,dsb->ab", Tj_d, GC, GR, Tj_d, optimize=True)
Z = np.einsum("lm,mcb->lcb", GL, Ti_d); Y = np.einsum("cd,rs,dsb->crb", GC, GR, Tj_d)
labA, labB = {}, {}
for b in range(fi):
    for r in range(nR):
        labA.setdefault(int(si + di[b] + rR["downs"][r] + rR["inner"]), []).append((b, r))
for l in range(nL):
    for d in range(fj):
        labB.setdefault(int(sj + rL["downs"][l] + dj[d] + rL["inner"]), []).append((l, d))
allw = []
for k in sorted(labA):
    ra, cb = labA[k], labB.get(k, [])
    if not cb: continue
    bs = np.array([x[0] for x in ra]); rs = np.array([x[1] for x in ra]); ls = np.array([x[0] for x in cb]); ds = np.array([x[1] for x in cb])
    GA = Gi[np.ix_(bs, bs)] * GR[np.ix_(rs, rs)]; GB = GL[np.ix_(ls, ls)] * Gj[np.ix_(ds, ds)]
    Cb = np.einsum("lcb,crd->brld", Z, Y)[bs][:, :, ls][:, :, :, ds] if False else sum(Z[ls[None, :], c, bs[:, None]] * Y[c, rs[:, None], ds[None, :]] for c in range(nC))
    LB = la.cholesky(GB, lower=True); X = la.solve_triangular(LB, Cb.T, lower=True)
    w = la.eigh(X.T @ X, GA, eigvals_only=True)
    allw.extend(w.tolist())
    print(k, len(ra), len(cb), f"top {np.sort(w)[-3:]}", f"[{time.time()-t0:.0f}s]", flush=True)
w = np.sort(np.array(allw))[::-1]
n1 = int(np.sum(w > 1 - 1e-8))
print("unit:", n1, "eta =", np.sqrt(w[n1]), "next", np.sqrt(w[n1 + 1:n1 + 4]), "(paper 0.2203543174)")
