"""Exact eta certificate in fully reduced coordinates (scales to the decorated lattices).

Same quantity as pw_general (Pomata & Wei, arXiv:1911.01410): eta^2 is the largest squared cosine,
other than 1, between A = ker(H_Gi) (x) phys(R) and B = phys(L) (x) ker(H_Gj), computed in
V = range(U_L) (x) range(U_C) (x) range(U_R) (the rest of A is orthogonal to B). Differences from
pw_general:

* Boundary configurations are reduced throughout: a VBS map depends on the boundary virtual spins of a
  site only through their number of down spins, so only one representative column per count pattern
  is built, and every VBS state of a union of regions is written directly in V coordinates as a signed
  sum over the singlet configurations of the cut edges (no full-coordinate tensors).
* No inverse Gram matrix (and no determinant scaling). With t = p/q, the bordered integer matrix
      M = [[-p G_A, q C], [q C^T, -q G_B]]
  has, by the Haynsworth inertia additivity (G_B > 0), as many positive eigenvalues as
  C G_B^-1 C^T - t G_A. A floating-point LDL^T of M is re-checked exactly: ||L D L^T - M||_inf <= q beta.
  Weyl then gives #(eig(M/q) > beta) <= #positive pivots, and with mu <= lambda_min(G_B),
  nu <= lambda_min(G_A):  #(pencil eigenvalues > t') <= #positive pivots for
      t' = (1 + beta/mu) (t + beta/nu)
  (since (G_B + beta)^-1 >= G_B^-1 / (1 + beta/mu)).
* Lower bounds mu, nu come from certified lambda_min bounds of the four factor Gram matrices
  (G_A = G_i (x) G_R, G_B = G_L (x) G_j), each by an exactly re-checked LDL^T of G - lo I.
* The number of unit cosines is bounded below by the rank of the union VBS states in V coordinates
  (U_X injective, so V coordinates are faithful), computed as a rank modulo a prime of T^T T (a rank
  mod p never exceeds the rational rank, so this is a valid lower bound).
Everything is split into S^z blocks (total number of down virtual spins).
"""
import math
import time
from fractions import Fraction

import numpy as np
import scipy.linalg as la

import flint

from pw_general import Graph, ldl_inertia_exact, site_amp, site_weight, slot_reduction

PRIME = 2147483647


def _bits(x, n):
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


def reduced_region(g: Graph, X):
    """Reduced VBS map of region X: sparse columns (one per reduced boundary configuration), exact Gram
    matrix in the physical metric, injectivity check, and down counts."""
    sites = sorted(X)
    pos = {s: i for i, s in enumerate(sites)}
    inner = [e for e in g.edges if e[0] in pos and e[1] in pos]
    slots = g.boundary_slots(X)
    nb = len(slots)
    red, reps, downs = slot_reduction(slots)
    dims = [g.z[s] + 1 for s in sites]
    strides = [int(x) for x in np.cumprod([1] + dims[::-1][:-1])[::-1]]
    amps = [site_amp(g.z[s]) for s in sites]
    wts = [site_weight(g.z[s]) for s in sites]
    cols = []
    for bconf in reps:
        bdown = _bits(bconf, nb)
        col = {}
        for sconf in range(2 ** len(inner)):
            sb = _bits(sconf, len(inner))
            dn = [0] * len(sites)
            sign = 1
            for (u, v), sgl in zip(inner, sb):
                if sgl == 0:
                    dn[pos[v]] += 1
                else:
                    dn[pos[u]] += 1
                    sign = -sign
            for (kind, a, site), bd in zip(slots, bdown):
                dn[pos[site]] += bd
            amp = sign
            for i, k in enumerate(dn):
                amp *= int(amps[i][k])
            row = sum(k * st for k, st in zip(dn, strides))
            col[row] = col.get(row, 0) + amp
        cols.append({r: a for r, a in col.items() if a != 0})

    def weight(row):
        w = 1
        for i, st in enumerate(strides):
            w *= int(wts[i][(row // st) % dims[i]])
        return w

    n = len(cols)
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            a, b = cols[i], cols[j]
            if len(a) > len(b):
                a, b = b, a
            s = sum(weight(r) * v * b[r] for r, v in a.items() if r in b)
            G[i][j] = G[j][i] = s
    Gf = flint.fmpz_mat(G)
    assert Gf.rank() == n, "reduced VBS map not injective"
    return dict(sites=sites, slots=slots, red=red, reps=reps, downs=np.array(downs), n=n, G=Gf,
                Gn=np.array(G, dtype=object), inner=len(inner))


def union_coords(g: Graph, parts, regs):
    """VBS states of the union of 'parts' (list of site lists) in the product of the parts' reduced
    coordinates. Returns (dict col -> {row multi-index tuple: integer}, reduced union down counts,
    number of singlets inside the union)."""
    Y = sorted(set().union(*map(set, parts)))
    where = {}
    for pi, P in enumerate(parts):
        for s in P:
            where[s] = pi
    yslots = g.boundary_slots(Y)
    yred, yreps, ydowns = slot_reduction(yslots)
    cut = [e for e in g.edges if e[0] in where and e[1] in where and where[e[0]] != where[e[1]]]
    slot_pos = [{sl: k for k, sl in enumerate(r["slots"])} for r in regs]
    nsl = [len(r["slots"]) for r in regs]
    # where each union slot and each end of each cut edge goes
    ymap = [(where[sl[2]], slot_pos[where[sl[2]]][sl]) for sl in yslots]
    cmap = []
    for e in cut:
        u, v = e
        cmap.append(((where[u], slot_pos[where[u]][("edge", e, u)]),
                     (where[v], slot_pos[where[v]][("edge", e, v)])))
    out = []
    for yb in yreps:
        ybits = _bits(yb, len(yslots))
        col = {}
        for sconf in range(2 ** len(cut)):
            sb = _bits(sconf, len(cut))
            full = [0] * len(regs)
            for (pi, k), bit in zip(ymap, ybits):
                if bit:
                    full[pi] |= 1 << (nsl[pi] - 1 - k)
            sign = 1
            for ((pu, ku), (pv, kv)), sgl in zip(cmap, sb):
                if sgl == 0:            # u up, v down
                    full[pv] |= 1 << (nsl[pv] - 1 - kv)
                else:                   # - u down, v up
                    full[pu] |= 1 << (nsl[pu] - 1 - ku)
                    sign = -sign
            key = tuple(int(regs[pi]["red"][full[pi]]) for pi in range(len(regs)))
            col[key] = col.get(key, 0) + sign
        out.append({k: v for k, v in col.items() if v != 0})
    singlets = len([e for e in g.edges if e[0] in where and e[1] in where])
    return out, np.array(ydowns), singlets, yslots


def _gram_of_coords(cols, Gs):
    """Exact Gram matrix of vectors given as sparse dicts over multi-indices, metric (x) Gs."""
    n = len(cols)
    G = [[0] * n for _ in range(n)]
    items = [list(c.items()) for c in cols]
    for i in range(n):
        for j in range(i, n):
            s = 0
            for ka, va in items[i]:
                for kb, vb in items[j]:
                    p = va * vb
                    for f, (x, y) in enumerate(zip(ka, kb)):
                        p *= Gs[f][x, y]
                        if p == 0:
                            break
                    s += p
            G[i][j] = G[j][i] = s
    return G


def rank_mod_p_lower(cols, p=PRIME):
    """Lower bound on the rational rank of the integer vectors 'cols' (sparse dicts): rank mod p of T^T T."""
    keys = {}
    for c in cols:
        for k in c:
            keys.setdefault(k, len(keys))
    n = len(cols)
    M = flint.nmod_mat(n, n, p)
    byrow = {}
    for j, c in enumerate(cols):
        for k, v in c.items():
            byrow.setdefault(keys[k], []).append((j, v))
    acc = {}
    for entries in byrow.values():
        for (i, a) in entries:
            for (j, b) in entries:
                acc[(i, j)] = acc.get((i, j), 0) + a * b
    for (i, j), v in acc.items():
        M[i, j] = v % p
    return M.rank()


def lambda_min_lower(G):
    """Certified rational lower bound on lambda_min of a positive definite fmpz_mat."""
    n = G.nrows()
    Gn = np.array([[float(int(x)) for x in row] for row in G.tolist()])
    lo = Fraction(float(np.min(la.eigvalsh(Gn))) * 0.5).limit_denominator(10 ** 9)
    assert lo > 0
    A = G * lo.denominator - flint.fmpz_mat(n, n, [lo.numerator if i == j else 0 for i in range(n) for j in range(n)])
    neg, pos, bE = ldl_inertia_exact(A)
    assert neg == 0
    out = (lo.numerator - bE) / lo.denominator
    assert out > 0
    return out


def certify_eta_reduced(g: Graph, L, C, R, thresholds, log=print, int_threshold=2 ** 62):
    t0 = time.time()
    rL, rC, rR = (reduced_region(g, X) for X in (L, C, R))
    nL, nC, nR = rL["n"], rC["n"], rR["n"]
    log(f"  reduced dims L {nL}, C {nC}, R {nR} (V dim {nL * nC * nR})  [{time.time()-t0:.0f}s]")
    Ti, downs_i, sing_i, _ = union_coords(g, [L, C], [rL, rC])
    Tj, downs_j, sing_j, _ = union_coords(g, [C, R], [rC, rR])
    fi, fj = len(Ti), len(Tj)
    Gi = flint.fmpz_mat(_gram_of_coords(Ti, [rL["Gn"], rC["Gn"]]))
    Gj = flint.fmpz_mat(_gram_of_coords(Tj, [rC["Gn"], rR["Gn"]]))
    assert Gi.rank() == fi and Gj.rank() == fj
    Gin = np.array([[int(x) for x in row] for row in Gi.tolist()], dtype=object)
    Gjn = np.array([[int(x) for x in row] for row in Gj.tolist()], dtype=object)
    log(f"  Gamma boundaries {fi}, {fj}  [{time.time()-t0:.0f}s]")
    TU, _, _, _ = union_coords(g, [L, C, R], [rL, rC, rR])
    n_unit = rank_mod_p_lower(TU)
    log(f"  union VBS: rank >= {n_unit} (mod-p lower bound) -> at least {n_unit} unit cosines  "
        f"[{time.time()-t0:.0f}s]")
    lam = {k: lambda_min_lower(G) for k, G in (("L", rL["G"]), ("R", rR["G"]), ("i", Gi), ("j", Gj))}
    nu, mu = lam["i"] * lam["R"], lam["L"] * lam["j"]
    # Z[l, c, b] = sum_l' G_L[l, l'] Ti_b[l', c];  Y[c, r, d] = sum G_C[c, c'] G_R[r, r'] Tj_d[c', r']
    GL, GC, GR = rL["Gn"], rC["Gn"], rR["Gn"]
    Z = np.zeros((nL, nC, fi), dtype=object); Z[:] = 0
    for b, col in enumerate(Ti):
        for (l2, c), v in col.items():
            Z[:, c, b] += GL[:, l2] * v
    Y = np.zeros((nC, nR, fj), dtype=object); Y[:] = 0
    for d, col in enumerate(Tj):
        for (c2, r2), v in col.items():
            Y[:, :, d] += np.outer(GC[:, c2], GR[:, r2]) * v
    small = (max(abs(int(x)) for x in Z.ravel()) * max(abs(int(x)) for x in Y.ravel()) * nC < int_threshold)
    if small:
        Z, Y = Z.astype(np.int64), Y.astype(np.int64)
    labA, labB = {}, {}
    for b in range(fi):
        for r in range(nR):
            labA.setdefault(int(sing_i + downs_i[b] + rR["downs"][r] + rR["inner"]), []).append((b, r))
    for l in range(nL):
        for d in range(fj):
            labB.setdefault(int(sing_j + rL["downs"][l] + downs_j[d] + rL["inner"]), []).append((l, d))
    log(f"  pencil {fi * nR} x {nL * fj}; largest bordered block "
        f"{max(len(labA.get(k, [])) + len(labB.get(k, [])) for k in labA)}; lambda_min bounds "
        f"nu >= {float(nu):.4g}, mu >= {float(mu):.4g}  [{time.time()-t0:.0f}s]")
    out = {"V_dim": nL * nC * nR, "pencil_size": fi * nR, "unit_cosines_at_least": int(n_unit),
           "nu_lower": float(nu), "mu_lower": float(mu), "thresholds": {}}
    for t in thresholds:
        p, q = t.numerator, t.denominator
        pos_total, beta = 0, Fraction(0)
        for key in sorted(labA):
            ra, cb = labA[key], labB.get(key, [])
            bs = np.array([x[0] for x in ra]); rs = np.array([x[1] for x in ra])
            GA = Gin[np.ix_(bs, bs)] * GR[np.ix_(rs, rs)]
            if cb:
                ls = np.array([x[0] for x in cb]); ds = np.array([x[1] for x in cb])
                Cb = sum(Z[ls[None, :], c, bs[:, None]] * Y[c, rs[:, None], ds[None, :]] for c in range(nC))
                GB = GL[np.ix_(ls, ls)] * Gjn[np.ix_(ds, ds)]
                M = np.block([[-p * GA, q * Cb.astype(object)], [q * Cb.T.astype(object), -q * GB]])
            else:
                M = -p * GA
            Mf = flint.fmpz_mat([[int(x) for x in row] for row in M.tolist()])
            neg, pos, bE = ldl_inertia_exact(Mf)
            pos_total += pos
            beta = max(beta, bE / q)
        tprime = (1 + beta / mu) * (t + beta / nu)
        ok = pos_total <= n_unit
        out["thresholds"][str(t)] = {"positive": pos_total, "beta": float(beta), "t_prime": float(tprime),
                                     "t_prime_exact": f"{tprime.numerator}/{tprime.denominator}", "certified": ok,
                                     "eta_upper": float(tprime) ** 0.5}
        log(f"  t = {float(t):.6f}: {pos_total} positive pivots (need <= {n_unit}); beta = {float(beta):.2e}; "
            f"t' = {float(tprime):.10f}; eta <= {float(tprime) ** 0.5:.10f}: {ok}  [{time.time()-t0:.0f}s]")
    out["seconds"] = time.time() - t0
    return out
