"""VBS geometry for AKLT models on general graphs, in exact integer arithmetic.

Setting (Pomata & Wei, arXiv:1911.01410). A union graph is split into regions L, C, R with
Gamma_i = L u C and Gamma_j = C u R. Every site v has spin z_v/2, z_v its degree in the infinite
lattice; its z_v virtual spin-1/2 sit on its incident edges. For a region X, U_X is the VBS map from the
virtual spins on edges leaving X to the physical space of X (edges inside X carry singlets).

Scaled basis: per site of degree z, D = diag(sqrt(binom(z, k))) with k the number of down virtual spins.
VBS amplitudes become integers (site_amp: c_z / binom(z, k), c_z the lcm of the binomials) and the
physical metric is diag(binom(z, k)) (site_weight); all Gram matrices are integers. Boundary
configurations enter a VBS map only through the number of down spins per site, so they are reduced to
one representative per count pattern (slot_reduction).
"""
import math
from fractions import Fraction

import numpy as np

import flint
import itertools

from ..._compat import eigh
from ...checker import inertia_exact

PRIME = 2147483647      # 2^31 - 1; a rank modulo p never exceeds the rational rank


def _lcm(xs):
    out = 1
    for x in xs:
        out = out * x // math.gcd(out, x)
    return out


def site_amp(z):
    c = _lcm([math.comb(z, k) for k in range(z + 1)])
    return np.array([c // math.comb(z, k) for k in range(z + 1)], dtype=object)


def site_weight(z):
    return np.array([math.comb(z, k) for k in range(z + 1)], dtype=object)


class Graph:
    """Sites with lattice degrees z[v]; 'edges' are the union's edges; every site has z[v] slots, each
    either an edge of the union or a 'free' slot (an edge leaving the union)."""

    def __init__(self, z, edges):
        self.z = list(z)
        self.edges = [tuple(e) for e in edges]
        deg = [0] * len(z)
        for u, v in self.edges:
            deg[u] += 1; deg[v] += 1
        assert all(d <= zz for d, zz in zip(deg, z)), (deg, z)
        self.free = [zz - d for d, zz in zip(deg, z)]       # free slots per site

    def boundary_slots(self, region):
        """Ordered virtual slots on the boundary of a region: ('edge', (u,v), site) for union edges
        leaving the region, ('free', site, k) for free slots of region sites."""
        R = set(region)
        slots = []
        for e in self.edges:
            u, v = e
            if (u in R) != (v in R):
                slots.append(("edge", e, u if u in R else v))
        for s in sorted(R):
            for k in range(self.free[s]):
                slots.append(("free", (s, k), s))
        return slots


def region_map(g: Graph, region):
    """U_X as an object-dtype integer matrix (phys(X) x 2^{#boundary slots}), sites in sorted order,
    site 0 most significant, digit k = number of down virtual spins."""
    sites = sorted(region)
    pos = {s: i for i, s in enumerate(sites)}
    inner = [e for e in g.edges if e[0] in pos and e[1] in pos]
    bslots = g.boundary_slots(region)
    dims = [g.z[s] + 1 for s in sites]
    nphys = int(np.prod(dims))
    nb = len(bslots)
    U = np.zeros((nphys, 2 ** nb), dtype=object)
    U[:] = 0
    strides = np.cumprod([1] + dims[::-1][:-1])[::-1]
    amps = [site_amp(g.z[s]) for s in sites]
    for bconf in range(2 ** nb):
        bdown = [(bconf >> (nb - 1 - i)) & 1 for i in range(nb)]
        for sconf in itertools.product((0, 1), repeat=len(inner)):
            downs = [0] * len(sites)
            sign = 1
            for (u, v), sgl in zip(inner, sconf):
                if sgl == 0:
                    downs[pos[v]] += 1
                else:
                    downs[pos[u]] += 1
                    sign = -sign
            for (_kind, _a, site), bd in zip(bslots, bdown):
                downs[pos[site]] += bd
            amp = sign
            for i, k in enumerate(downs):
                amp *= amps[i][k]
            U[int(sum(k * st for k, st in zip(downs, strides))), bconf] += amp
    return U, bslots


def slot_reduction(slots):
    """Boundary configurations depend on the VBS only through the number of down spins among the
    slots of each site. Returns (red_of_full: array, reps: list of a full index per reduced index,
    downs_of_red: array)."""
    nb = len(slots)
    sites = []
    for sl in slots:
        if sl[2] not in sites:
            sites.append(sl[2])
    groups = [[i for i, sl in enumerate(slots) if sl[2] == s] for s in sites]
    sizes = [len(g_) for g_ in groups]
    radix = [m + 1 for m in sizes]
    red_of_full = np.zeros(2 ** nb, dtype=np.int64)
    downs_full = np.zeros(2 ** nb, dtype=np.int64)
    for f in range(2 ** nb):
        bits = [(f >> (nb - 1 - i)) & 1 for i in range(nb)]
        counts = [sum(bits[i] for i in g_) for g_ in groups]
        r = 0
        for c, rr in zip(counts, radix):
            r = r * rr + c
        red_of_full[f] = r
        downs_full[f] = sum(counts)
    nred = int(np.prod(radix)) if radix else 1
    reps = [int(np.nonzero(red_of_full == r)[0][0]) for r in range(nred)]
    downs_red = np.array([downs_full[reps[r]] for r in range(nred)])
    return red_of_full, reps, downs_red


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
            for (_kind, _a, site), bd in zip(slots, bdown):
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


def gram_of_coords(cols, Gs):
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
    lo = Fraction(float(np.min(eigh(Gn, eigvals_only=True))) * 0.5).limit_denominator(10 ** 9)   # a hint; 1 x 1 too
    assert lo > 0
    A = G * lo.denominator - flint.fmpz_mat(n, n, [lo.numerator if i == j else 0 for i in range(n) for j in range(n)])
    neg, pos, bE = inertia_exact(A)
    assert neg == 0
    out = (lo.numerator - bE) / lo.denominator
    assert out > 0
    return out
