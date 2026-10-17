"""Exact certification of the Pomata–Wei quantity eta for AKLT models on general graphs.

Setting (Pomata & Wei, arXiv:1911.01410). A union graph is split into regions L, C, R with
Gamma_i = L u C and Gamma_j = C u R. Every site v has spin z_v/2, z_v its degree in the infinite
lattice; its z_v virtual spin-1/2 sit on its incident edges. For a region X, U_X is the VBS map from the
virtual spins on edges leaving X to the physical space of X (edges inside X carry singlets). Pi_Gamma is
the projector onto the VBS range of Gamma. eta^2 = largest squared cosine, other than 1, of the
principal angles between A = range(Pi_i) (x) phys(R) and B = phys(L) (x) range(Pi_j); only the parts of
A and B inside V = range(U_L) (x) range(U_C) (x) range(U_R) have nonzero cosines (the rest of A is
orthogonal to B), so everything is computed in V coordinates.

Exact arithmetic: in the scaled basis (D = diag(sqrt binom(z, k))), VBS amplitudes are integers
z!/... (we use c_z / binom(z, k) with c_z = lcm of the binomials) and the physical metric is
diag(binom(z, k)). All Gram matrices are integers. The cosines are the eigenvalues of the pencil
(C G_B^-1 C^T, G_A); their count above a threshold t is an inertia count of an exact integer matrix,
established by a floating-point LDL^T whose residual is re-checked exactly (FLINT).
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import scipy.linalg as la

import flint


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
            for (kind, a, site), bd in zip(bslots, bdown):
                downs[pos[site]] += bd
            amp = sign
            for i, k in enumerate(downs):
                amp *= amps[i][k]
            U[int(sum(k * st for k, st in zip(downs, strides))), bconf] += amp
    return U, bslots


def region_gram(g: Graph, region):
    U, bslots = region_map(g, region)
    sites = sorted(region)
    w = np.array([1], dtype=object)
    for s in sites:
        w = np.kron(w, site_weight(g.z[s]))
    Uf = flint.fmpz_mat([[int(x) for x in row] for row in U.T])           # (2^nb) x phys
    Wd = flint.fmpz_mat([[int(w[i]) * int(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])])
    G = Uf * Wd
    return G, bslots


def contraction(slots_a, slots_b, region_a, region_b):
    """Tensor T for the VBS of region_a u region_b in terms of U_a (x) U_b: rows index the boundary
    configurations of a and b, columns the boundary configurations of the union (slots of a and b not
    on a-b edges, in order). Edges between a and b carry the singlet [[0, 1], [-1, 0]] (up-down = +1)."""
    A, B = set(region_a), set(region_b)
    inner_ab = {}
    for i, (kind, e, site) in enumerate(slots_a):
        if kind == "edge" and ((e[0] in A and e[1] in B) or (e[1] in A and e[0] in B)):
            inner_ab.setdefault(e, {})["a"] = i
    for i, (kind, e, site) in enumerate(slots_b):
        if kind == "edge" and ((e[0] in A and e[1] in B) or (e[1] in A and e[0] in B)):
            inner_ab.setdefault(e, {})["b"] = i
    keep_a = [i for i, s in enumerate(slots_a) if not (s[0] == "edge" and s[1] in inner_ab)]
    keep_b = [i for i, s in enumerate(slots_b) if not (s[0] == "edge" and s[1] in inner_ab)]
    na, nb = len(slots_a), len(slots_b)
    nk = len(keep_a) + len(keep_b)
    T = np.zeros((2 ** na * 2 ** nb, 2 ** nk), dtype=np.int64)
    edges = list(inner_ab.items())
    for out in range(2 ** nk):
        obits = [(out >> (nk - 1 - i)) & 1 for i in range(nk)]
        for sconf in itertools.product((0, 1), repeat=len(edges)):
            abits = [0] * na; bbits = [0] * nb
            for i, k in enumerate(keep_a):
                abits[k] = obits[i]
            for i, k in enumerate(keep_b):
                bbits[k] = obits[len(keep_a) + i]
            sign = 1
            for (e, idx), sgl in zip(edges, sconf):
                # orientation: the endpoint with smaller site index is 'u'
                u = min(e)
                ua = u in A
                if sgl == 0:          # u up, v down
                    up_a, dn = (0, 1)
                else:
                    up_a, dn = (1, 0)
                    sign = -sign
                bit_u, bit_v = (0, 1) if sgl == 0 else (1, 0)
                if ua:
                    abits[idx["a"]], bbits[idx["b"]] = bit_u, bit_v
                else:
                    bbits[idx["b"]], abits[idx["a"]] = bit_u, bit_v
            ia = int("".join(map(str, abits)), 2) if na else 0
            ib = int("".join(map(str, bbits)), 2) if nb else 0
            T[ia * 2 ** nb + ib, out] += sign
    return T, [slots_a[k] for k in keep_a], [slots_b[k] for k in keep_b]


# ------------------------------------------------------------------ exact inertia
def ldl_inertia_exact(A):
    """A: flint.fmpz_mat, symmetric. Float LDL^T of A (rounded to floats), residual against the exact A
    re-checked in integer arithmetic. Returns (n_negative, n_positive, upper bound on ||E||_inf) with
    E = Lt Dt Lt^T - A for dyadic factors Lt (unit lower triangular after the LDL permutation) and Dt."""
    from certify_ed.checker import negative_count_exact
    n = A.nrows()
    ent = [int(x) for x in A.entries()]
    maxbits = max(abs(x).bit_length() for x in ent)
    s_down = max(0, maxbits - 900)               # A 2^-s_down fits comfortably in binary64
    Af = np.array([float(x >> s_down) if x >= 0 else -float((-x) >> s_down) for x in ent]).reshape(n, n)
    lu, d, perm = la.ldl(Af, lower=True, hermitian=True)
    assert np.all(np.diag(lu[perm]) == 1.0)
    pL = 52
    pD = int(52 - np.ceil(np.log2(float(np.max(np.abs(d))))))
    Li = np.rint(np.ldexp(lu, pL)).astype(np.int64)
    Di = np.rint(np.ldexp(d, pD)).astype(np.int64)
    Dt = np.ldexp(Di.astype(np.float64), -pD)
    neg, amb = negative_count_exact(Dt)
    assert not amb
    # the factors approximate A 2^-s_down: Lt (2^s_down Dt) Lt^T approximates A; D is kept scaled
    sh = 2 * pL + pD - s_down
    L = flint.fmpz_mat(Li.tolist()); Dm = flint.fmpz_mat(Di.tolist())
    P = (L * Dm) * L.transpose()                 # = 2^sh Lt (2^s_down Dt) Lt^T
    if sh >= 0:
        E = P - A * (flint.fmpz(2) ** sh)       # = 2^sh E
        den = Fraction(2) ** sh
    else:                                        # P = 2^sh LDL with sh < 0: LDL = 2^-sh P exactly
        E = P * (flint.fmpz(2) ** (-sh)) - A
        den = Fraction(1)
    ents = E.entries()
    worst = 0
    for i in range(n):
        s = 0
        for x in ents[i * n:(i + 1) * n]:
            s += abs(int(x))
        if s > worst:
            worst = s
    return neg, n - neg, Fraction(worst) / den


def certify_eta(g: Graph, L, C, R, thresholds, log=print):
    """Exact certification of eta for the pair (Gamma_i = L u C, Gamma_j = C u R)."""
    import time
    t0 = time.time()
    GL, sL = region_gram(g, L)
    GC, sC = region_gram(g, C)
    GR, sR = region_gram(g, R)
    for name, G in (("L", GL), ("C", GC), ("R", GR)):
        assert G.rank() == G.nrows(), f"U_{name} not injective"
    Ti, kLi, kCi = contraction(sL, sC, L, C)          # Gamma_i
    Tj, kCj, kRj = contraction(sC, sR, C, R)          # Gamma_j
    nL, nC, nR = 2 ** len(sL), 2 ** len(sC), 2 ** len(sR)
    fi, fj = Ti.shape[1], Tj.shape[1]
    log(f"  regions: |dL|={len(sL)} |dC|={len(sC)} |dR|={len(sR)}; Gamma boundaries 2^{int(np.log2(fi))}, "
        f"2^{int(np.log2(fj))}  [{time.time()-t0:.0f}s]")
    GLn = np.array([[int(x) for x in row] for row in GL.tolist()], dtype=object)
    GCn = np.array([[int(x) for x in row] for row in GC.tolist()], dtype=object)
    GRn = np.array([[int(x) for x in row] for row in GR.tolist()], dtype=object)
    # Gram of Gamma_i in V coords: Ti^T (GL x GC) Ti ; of Gamma_j: Tj^T (GC x GR) Tj
    Tif = flint.fmpz_mat(Ti.tolist()); Tjf = flint.fmpz_mat(Tj.tolist())
    GLC = flint.fmpz_mat(np.kron(GLn, GCn).tolist()); GCR = flint.fmpz_mat(np.kron(GCn, GRn).tolist())
    Gi = Tif.transpose() * GLC * Tif
    Gj = Tjf.transpose() * GCR * Tjf
    assert Gi.rank() == fi and Gj.rank() == fj
    # Z[l, c, b] = sum_l1 GL[l1, l] Ti[(l1, c), b] ;  Y[(c, r), b'] = ((GC x GR) Tj)[(c, r), b']
    Ti3 = Ti.reshape(nL, nC, fi).astype(object)
    Z = np.einsum("pl,pcb->lcb", GLn, Ti3)
    Y = np.array([[int(x) for x in row] for row in (GCR * Tjf).tolist()], dtype=object).reshape(nC, nR, fj)
    # C[(b, r), (l, b')] = sum_c Z[l, c, b] Y[c, r, b']
    Cm = np.einsum("lcb,crd->brld", Z, Y).reshape(fi * nR, nL * fj)
    Cf = flint.fmpz_mat(Cm.tolist())
    GA = flint.fmpz_mat(np.kron(np.array(Gi.tolist(), dtype=object), GRn).tolist())
    # G_B = GL (x) Gj ; its inverse = GL^-1 (x) Gj^-1 ; scale to integers with det(GL) det(Gj)
    dGL, dGj = int(GL.det()), int(Gj.det())
    adjGL = flint.fmpq_mat(GL).inv() * dGL
    adjGj = flint.fmpq_mat(Gj).inv() * dGj
    adjGLn = np.array([[int(x) for x in row] for row in adjGL.tolist()], dtype=object)
    adjGjn = np.array([[int(x) for x in row] for row in adjGj.tolist()], dtype=object)
    Mid = flint.fmpz_mat(np.kron(adjGLn, adjGjn).tolist())
    scale = dGL * dGj
    Qm = Cf * Mid * Cf.transpose()                 # = scale * C G_B^-1 C^T
    log(f"  pencil of size {Qm.nrows()} assembled  [{time.time()-t0:.0f}s]")
    # lambda_min(G_A) lower bound: G_A = Gi (x) GR ; certify rational lower bounds for Gi and GR
    lam = Fraction(1)
    for G in (Gi, GR):
        Gn = np.array([[float(int(x)) for x in row] for row in G.tolist()])
        lo = Fraction(float(np.min(la.eigvalsh(Gn))) * 0.5).limit_denominator(10 ** 9)
        Gq = flint.fmpq_mat(G) - flint.fmpq_mat(G.nrows(), G.nrows(),
                                                [flint.fmpq(lo.numerator, lo.denominator) if i == j else 0
                                                 for i in range(G.nrows()) for j in range(G.nrows())])
        k = G.nrows()
        assert all(flint.fmpq_mat(m, m, [Gq[i, j] for i in range(m) for j in range(m)]).det() > 0
                   for m in range(1, k + 1)), "lambda_min lower bound not certified"
        lam *= lo
    # intersection: the union VBS states lie in A (built as (Ti x I_R) T_{LC,R}) and in B (built as
    # (I_L x Tj) T_{L,CR}); both constructions are computed and compared exactly, and their number of
    # independent states is the exact rank of their Gram matrix  T^T G_A T  (G_A = Gi x GR).
    T_LC_R, k1, k2 = contraction(kLi + kCi, sR, set(L) | set(C), R)
    T_L_CR, k3, k4 = contraction(sL, kCj + kRj, L, set(C) | set(R))
    slots_a, slots_b = k1 + k2, k3 + k4
    assert sorted(map(str, slots_a)) == sorted(map(str, slots_b))
    nu = len(slots_a)
    perm = [slots_b.index(x) for x in slots_a]          # bit i of order a = bit perm[i] of order b
    cols_b = []
    for ua in range(2 ** nu):
        bits_a = [(ua >> (nu - 1 - i)) & 1 for i in range(nu)]
        bits_b = [0] * nu
        for i, pi in enumerate(perm):
            bits_b[pi] = bits_a[i]
        cols_b.append(int("".join(map(str, bits_b)), 2) if nu else 0)
    TUa = np.einsum("xb,brU->xrU", Ti, T_LC_R.reshape(fi, nR, -1)).reshape(nL * nC * nR, -1)
    TUb = np.einsum("crd,ldU->lcrU", Tj.reshape(nC, nR, fj), T_L_CR.reshape(nL, fj, -1)).reshape(nL * nC * nR, -1)
    assert np.array_equal(TUa, TUb[:, cols_b]), "union VBS differs between the two constructions"
    TLCRf = flint.fmpz_mat(T_LC_R.tolist())
    GU = TLCRf.transpose() * GA * TLCRf
    n_unit = GU.rank()
    log(f"  union VBS states (in A and in B, checked exactly): {n_unit} independent  [{time.time()-t0:.0f}s]")
    out = {"pencil_size": Qm.nrows(), "unit_cosines_at_least": int(n_unit), "lambda_min_GA_lower": float(lam),
           "thresholds": {}}
    for t in thresholds:
        p, q = t.numerator, t.denominator
        K = Qm * q - GA * (p * scale)
        neg, pos, bE = ldl_inertia_exact(K)
        tprime = t + bE / (Fraction(q * scale) * lam)
        ok = pos <= n_unit
        lg = (bE.numerator.bit_length() - bE.denominator.bit_length()) if bE > 0 else None
        out["thresholds"][str(t)] = {"positive": pos, "bound_E_log2_approx": lg, "t_prime": float(tprime),
                                     "t_prime_exact": f"{tprime.numerator}/{tprime.denominator}",
                                     "certified": ok}
        log(f"  t = {float(t):.6f}: {pos} pencil eigenvalues above t' = {float(tprime):.10f}; "
            f"eta <= {float(tprime) ** 0.5:.10f}: {ok}  [{time.time()-t0:.0f}s]")
    return out


def eta_float(g: Graph, L, C, R, log=print):
    """Floating-point estimate of eta (validation of geometry; not certified)."""
    def fl(G):
        return np.array([[float(int(x)) for x in row] for row in G.tolist()])
    GL, sL = region_gram(g, L); GC, sC = region_gram(g, C); GR, sR = region_gram(g, R)
    Ti, kLi, kCi = contraction(sL, sC, L, C)
    Tj, kCj, kRj = contraction(sC, sR, C, R)
    gl, gc, gr = fl(GL), fl(GC), fl(GR)
    # orthonormalize in V coordinates: metric G_V = gl x gc x gr
    def inv_sqrt(G):
        w, V = la.eigh(G)
        return V @ np.diag(w ** -0.5) @ V.T, w
    sl, wl = inv_sqrt(gl); sc, wc = inv_sqrt(gc); sr, wr = inv_sqrt(gr)
    # A basis: (Ti x I_R) in V coords; B basis: (I_L x Tj)
    nL, nC, nR = gl.shape[0], gc.shape[0], gr.shape[0]
    Gi = Ti.T.astype(float) @ np.kron(gl, gc) @ Ti.astype(float)
    Gj = Tj.T.astype(float) @ np.kron(gc, gr) @ Tj.astype(float)
    fi, fj = Ti.shape[1], Tj.shape[1]
    Z = np.einsum("pl,pcb->lcb", gl, Ti.reshape(nL, nC, fi).astype(float))
    Y = (np.kron(gc, gr) @ Tj.astype(float)).reshape(nC, nR, fj)
    Cm = np.einsum("lcb,crd->brld", Z, Y).reshape(fi * nR, nL * fj)
    GA = np.kron(Gi, gr); GB = np.kron(gl, Gj)
    # cos^2 = eig(GA^-1 C GB^-1 C^T)
    LA = la.cholesky(GA, lower=True); LB = la.cholesky(GB, lower=True)
    M = la.solve_triangular(LA, Cm, lower=True)
    M = la.solve_triangular(LB, M.T, lower=True).T
    sv = la.svd(M, compute_uv=False)
    n1 = int(np.sum(sv > 1 - 1e-8))
    log(f"  V dims {nL}x{nC}x{nR}; unit cosines {n1}; eta = {sv[n1]:.10f}")
    return float(sv[n1]), n1


def _popcounts(nbits):
    return np.array([bin(x).count("1") for x in range(2 ** nbits)])


def certify_eta_blocks(g: Graph, L, C, R, thresholds, log=print):
    """As certify_eta, with the pencil split into blocks of fixed total S^z (number of down spins).
    VBS maps are SU(2) covariant, so a basis vector's S^z is fixed by its boundary bits; A and B vectors
    with different S^z are orthogonal and the Gram matrices are block diagonal."""
    import time
    t0 = time.time()
    GL, sL = region_gram(g, L); GC, sC = region_gram(g, C); GR, sR = region_gram(g, R)
    for name, G in (("L", GL), ("C", GC), ("R", GR)):
        assert G.rank() == G.nrows(), f"U_{name} not injective"
    Ti, kLi, kCi = contraction(sL, sC, L, C)
    Tj, kCj, kRj = contraction(sC, sR, C, R)
    nL, nC, nR = 2 ** len(sL), 2 ** len(sC), 2 ** len(sR)
    fi, fj = Ti.shape[1], Tj.shape[1]
    toint = lambda M: np.array([[int(x) for x in row] for row in M.tolist()], dtype=object)
    GLn, GCn, GRn = toint(GL), toint(GC), toint(GR)
    Tif = flint.fmpz_mat(Ti.tolist()); Tjf = flint.fmpz_mat(Tj.tolist())
    GLC = flint.fmpz_mat(np.kron(GLn, GCn).tolist()); GCR = flint.fmpz_mat(np.kron(GCn, GRn).tolist())
    Gi = Tif.transpose() * GLC * Tif
    Gj = Tjf.transpose() * GCR * Tjf
    assert Gi.rank() == fi and Gj.rank() == fj
    Gin, Gjn = toint(Gi), toint(Gj)
    # Z[l, c, b] = sum_p GL[p, l] Ti[(p, c), b]  -> matrix (l, (c, b))
    Zm = GL.transpose() * flint.fmpz_mat(Ti.reshape(nL, nC * fi).tolist())
    Z = toint(Zm).reshape(nL, nC, fi)
    Ym = GCR * Tjf                                          # ((c, r), d)
    Y = toint(Ym).reshape(nC, nR, fj)
    # T[(b, l), (r, d)] = sum_c Z[l, c, b] Y[c, r, d]
    Zbl = flint.fmpz_mat(Z.transpose(2, 0, 1).reshape(fi * nL, nC).tolist())
    Ycr = flint.fmpz_mat(Y.reshape(nC, nR * fj).tolist())
    T = Zbl * Ycr
    log(f"  cross Gram assembled ({fi * nR} x {nL * fj})  [{time.time()-t0:.0f}s]")
    # S^z labels (number of down virtual spins counts physical downs up to a constant per side)
    pb, pr = _popcounts(int(np.log2(fi))), _popcounts(len(sR))
    pl, pd = _popcounts(len(sL)), _popcounts(int(np.log2(fj)))
    # physical downs: each internal singlet contributes one down
    nsing = lambda regions_edges: len(regions_edges)
    LC = set(L) | set(C); CR = set(C) | set(R)
    sing_A = len([e for e in g.edges if (e[0] in LC and e[1] in LC) or (e[0] in R and e[1] in R)])
    sing_B = len([e for e in g.edges if (e[0] in L and e[1] in L) or (e[0] in CR and e[1] in CR)])
    labA = {}
    for b in range(fi):
        for r in range(nR):
            labA.setdefault(sing_A + pb[b] + pr[r], []).append((b, r))
    labB = {}
    for l in range(nL):
        for d in range(fj):
            labB.setdefault(sing_B + pl[l] + pd[d], []).append((l, d))
    # lambda_min(G_A) lower bound from Gi and GR
    lam = Fraction(1)
    for G in (Gi, GR):
        Gn = np.array([[float(int(x)) for x in row] for row in G.tolist()])
        lo = Fraction(float(np.min(la.eigvalsh(Gn))) * 0.5).limit_denominator(10 ** 9)
        Gq = flint.fmpq_mat(G) - flint.fmpq_mat(G.nrows(), G.nrows(),
                                                [flint.fmpq(lo.numerator, lo.denominator) if i == j else 0
                                                 for i in range(G.nrows()) for j in range(G.nrows())])
        assert all(flint.fmpq_mat(m, m, [Gq[i, j] for i in range(m) for j in range(m)]).det() > 0
                   for m in range(1, G.nrows() + 1))
        lam *= lo
    dGL, dGj = int(GL.det()), int(Gj.det())
    adjGL = toint(flint.fmpq_mat(GL).inv() * dGL); adjGj = toint(flint.fmpq_mat(Gj).inv() * dGj)
    scale = dGL * dGj
    # union VBS states: inside A and B (checked exactly), independent count
    T_LC_R, k1, k2 = contraction(kLi + kCi, sR, LC, R)
    T_L_CR, k3, k4 = contraction(sL, kCj + kRj, L, CR)
    slots_a, slots_b = k1 + k2, k3 + k4
    nu = len(slots_a)
    perm = [slots_b.index(x) for x in slots_a]
    cols_b = []
    for ua in range(2 ** nu):
        bits_a = [(ua >> (nu - 1 - i)) & 1 for i in range(nu)]
        bits_b = [0] * nu
        for i, pi in enumerate(perm):
            bits_b[pi] = bits_a[i]
        cols_b.append(int("".join(map(str, bits_b)), 2) if nu else 0)
    TUa = np.einsum("xb,brU->xrU", Ti, T_LC_R.reshape(fi, nR, -1)).reshape(nL * nC * nR, -1)
    TUb = np.einsum("crd,ldU->lcrU", Tj.reshape(nC, nR, fj), T_L_CR.reshape(nL, fj, -1)).reshape(nL * nC * nR, -1)
    assert np.array_equal(TUa, TUb[:, cols_b])
    GA_full = None
    TLCRf = flint.fmpz_mat(T_LC_R.tolist())
    # G_U = T^T (Gi x GR) T computed without forming the kron: (Gi x GR) T = vec(GR Tmat Gi^T)
    Tm = T_LC_R.reshape(fi, nR, -1)
    GiGR_T = np.einsum("bc,rs,csU->brU", Gin, GRn, Tm.astype(object)).reshape(fi * nR, -1)
    GU = flint.fmpz_mat(T_LC_R.T.tolist()) * flint.fmpz_mat(GiGR_T.tolist())
    n_unit = GU.rank()
    log(f"  union VBS: {n_unit} independent states in A and B  [{time.time()-t0:.0f}s]")
    out = {"pencil_size": fi * nR, "unit_cosines_at_least": int(n_unit), "lambda_min_GA_lower": float(lam),
           "thresholds": {}}
    for t in thresholds:
        p, q = t.numerator, t.denominator
        pos_total, worst = 0, Fraction(0)
        for key in sorted(set(labA) | set(labB)):
            ra, cb = labA.get(key, []), labB.get(key, [])
            if not ra:
                continue
            Cb = flint.fmpz_mat([[T[b * nL + l, r * fj + d] for (l, d) in cb] for (b, r) in ra]) \
                if cb else None
            GAb = flint.fmpz_mat([[Gin[b, b2] * GRn[r, r2] for (b2, r2) in ra] for (b, r) in ra])
            if Cb is not None:
                Mb = flint.fmpz_mat([[adjGL[l, l2] * adjGj[d, d2] for (l2, d2) in cb] for (l, d) in cb])
                Qb = Cb * Mb * Cb.transpose()
                K = Qb * q - GAb * (p * scale)
            else:
                K = GAb * (-(p * scale))
            neg, pos, bE = ldl_inertia_exact(K)
            pos_total += pos
            worst = max(worst, bE)
        tprime = t + worst / (Fraction(q * scale) * lam)
        ok = pos_total <= n_unit
        out["thresholds"][str(t)] = {"positive": pos_total, "t_prime": float(tprime),
                                     "t_prime_exact": f"{tprime.numerator}/{tprime.denominator}", "certified": ok}
        log(f"  t = {float(t):.6f}: {pos_total} pencil eigenvalues above t' = {float(tprime):.10f}; "
            f"eta <= {float(tprime) ** 0.5:.10f}: {ok}  [{time.time()-t0:.0f}s]")
    return out
