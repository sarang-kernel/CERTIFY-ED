"""Certified kernel equality ker H_Gamma = range(VBS map of Gamma) for AKLT models with mixed spins.

Both directions are exact:
  '>='  integer VBS states U (scaled basis) satisfy Ht U = 0 exactly, and rank U is computed exactly
        (rank of the integer Gram matrix; columns that differ only by a permutation of boundary slots at
        one site are identical, so the symmetrized columns suffice).
  '<='  Ht is an integer matrix similar to den * H_Gamma (Ht = den * D^-1 H D), so nullity_Q(H) =
        nullity_Q(Ht). For any prime p, rank over F_p <= rank over Q (a minor that is non-zero mod p is a
        non-zero integer), hence nullity_Q(Ht) <= nullity_p(Ht mod p). Ht is block diagonal in the number
        of down virtual spins (S^z), so the nullity is summed over blocks.
If the F_p nullity equals rank U the kernel is exactly the VBS range. A bad prime can only make the
check fail, never pass falsely. No floating point is involved.

Scaled basis per site of lattice degree z (spin z/2): digit k = number of down virtual spins,
D = diag(sqrt(binom(z, k))). Then D^-1 S+ D has (k-1, k) entry z-k+1, D^-1 S- D has (k, k-1) entry k,
and S^z = z/2 - k: all rational, so den * (projector onto maximal total spin of an edge) is an integer
matrix (polynomial in S_u . S_v).
"""
import math
import sys
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

import flint

from pw_general import Graph, region_map, slot_reduction

PRIME = 2147483647          # 2^31 - 1


def _scaled_ops(z):
    n = z + 1
    S = Fraction(z, 2)
    Z = [[Fraction(0)] * n for _ in range(n)]
    P = [[Fraction(0)] * n for _ in range(n)]
    M = [[Fraction(0)] * n for _ in range(n)]
    for k in range(n):
        Z[k][k] = S - k
    for k in range(1, n):
        P[k - 1][k] = Fraction(z - k + 1)
        M[k][k - 1] = Fraction(k)
    return np.array(Z, dtype=object), np.array(P, dtype=object), np.array(M, dtype=object)


def scaled_pmax(z1, z2):
    """(Pint, den): integer matrix with Pint/den = (D1 x D2)^-1 P_{S1+S2} (D1 x D2), S_i = z_i/2."""
    Z1, P1, M1 = _scaled_ops(z1)
    Z2, P2, M2 = _scaled_ops(z2)
    X = np.kron(Z1, Z2) + (np.kron(P1, M2) + np.kron(M1, P2)) * Fraction(1, 2)
    S1, S2 = Fraction(z1, 2), Fraction(z2, 2)
    xs = lambda J: (J * (J + 1) - S1 * (S1 + 1) - S2 * (S2 + 1)) / 2
    Jmax = S1 + S2
    I = np.array([[Fraction(int(i == j)) for j in range(X.shape[0])] for i in range(X.shape[0])], dtype=object)
    Pm = I.copy()
    J = abs(S1 - S2)
    while J < Jmax:
        Pm = Pm.dot((X - xs(J) * I) * (1 / (xs(Jmax) - xs(J))))
        J += 1
    den = 1
    for v in Pm.ravel():
        den = den * v.denominator // math.gcd(den, v.denominator)
    return np.array([[int(v * den) for v in row] for row in Pm], dtype=np.int64), den


def hamiltonian_region(g: Graph, region):
    """Sparse int64 matrix den * D^-1 H_region D (sites sorted, site 0 most significant), and den."""
    sites = sorted(region)
    pos = {s: i for i, s in enumerate(sites)}
    dims = [g.z[s] + 1 for s in sites]
    N = int(np.prod(dims))
    strides = np.cumprod([1] + dims[::-1][:-1])[::-1]
    idx = np.arange(N)
    digits = [(idx // strides[i]) % dims[i] for i in range(len(sites))]
    inner = [e for e in g.edges if e[0] in pos and e[1] in pos]
    mats = {}
    for u, v in inner:
        key = (g.z[u], g.z[v])
        if key not in mats:
            mats[key] = scaled_pmax(*key)
    den = 1
    for _, d in mats.values():
        den = den * d // math.gcd(den, d)
    rows, cols, vals = [], [], []
    for u, v in inner:
        P, d = mats[(g.z[u], g.z[v])]
        P = P * (den // d)
        iu, iv = pos[u], pos[v]
        a, b = digits[iu], digits[iv]
        dv = dims[iv]
        colpair = a * dv + b
        for rp in range(P.shape[0]):
            coef = P[rp, colpair]
            nz = coef != 0
            if not np.any(nz):
                continue
            a2, b2 = divmod(rp, dv)
            tgt = idx[nz] + (a2 - a[nz]) * strides[iu] + (b2 - b[nz]) * strides[iv]
            rows.append(tgt); cols.append(idx[nz]); vals.append(coef[nz])
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N),
                      dtype=np.int64)
    return H, den, digits


def nullity_mod_p(Hblock, p=PRIME):
    n = Hblock.shape[0]
    M = flint.nmod_mat(n, n, p)
    C = Hblock.tocoo()
    for i, j, v in zip(C.row.tolist(), C.col.tolist(), C.data.tolist()):
        M[i, j] = v % p
    return n - M.rank()


def kernel_check(g: Graph, region, log=print, p=PRIME):
    t0 = time.time()
    U, slots = region_map(g, region)
    red, reps, _ = slot_reduction(slots)
    Ur = U[:, reps]
    assert max(abs(int(x)) for x in Ur.ravel()) < 2 ** 40
    Ui = Ur.astype(np.int64)
    H, den, digits = hamiltonian_region(g, region)
    assert np.abs(H.data).max() * H.getnnz(axis=1).max() * np.abs(Ui).max() < 2 ** 62   # no int64 overflow
    HU_zero = bool(np.all((H @ Ui) == 0))
    assert HU_zero, "VBS states are not annihilated by H_Gamma"
    G = flint.fmpz_mat(Ui.T.tolist()) * flint.fmpz_mat(Ui.tolist())
    r = int(G.rank())
    downs = sum(digits)
    total = 0
    blocks = {}
    for k in np.unique(downs):
        sel = np.nonzero(downs == k)[0]
        Hb = H[sel][:, sel]
        assert (H[sel][:, np.nonzero(downs != k)[0]]).nnz == 0       # block diagonal in S^z
        nl = nullity_mod_p(Hb, p)
        blocks[int(k)] = (len(sel), nl)
        total += nl
    ok = total == r
    log(f"  {len(region)} sites, dim {H.shape[0]}: H U = 0, rank U = {r}; nullity of H mod {p} = {total}; "
        f"kernel = VBS range: {ok}  [{time.time()-t0:.0f}s]")
    return {"sites": len(region), "dim": int(H.shape[0]), "vbs_rank": r, "nullity_mod_p_upper": int(total),
            "prime": p, "kernel_equals_vbs_range": ok, "blocks": blocks, "seconds": time.time() - t0}


if __name__ == "__main__":
    import json, os
    from lattices import square_octagon, decorated_honeycomb, cross
    HERE = os.path.dirname(os.path.abspath(__file__))
    TRI = lambda a: [(a, a + 1), (a + 1, a + 2), (a, a + 2)]
    out = {}
    which = sys.argv[1:] or ["star", "square_octagon", "decorated_honeycomb"]
    if "star" in which:
        g = Graph([3] * 9, TRI(0) + TRI(3) + TRI(6) + [(2, 3), (4, 6)])
        print("star (expected 16, 16):", flush=True)
        out["star"] = [kernel_check(g, [0, 1, 2, 3, 4, 5]), kernel_check(g, [3, 4, 5, 6, 7, 8])]
    if "square_octagon" in which:
        g, L, C, R = square_octagon("par")
        print("square-octagon Gamma (two squares + bridge; same graph for both overlap types):", flush=True)
        out["square_octagon"] = [kernel_check(g, L + C)]
    if "decorated_honeycomb" in which:
        g, L, C, R = decorated_honeycomb()
        print("decorated honeycomb Gamma_i, Gamma_j:", flush=True)
        out["decorated_honeycomb"] = [kernel_check(g, L + C), kernel_check(g, C + R)]
    path = os.path.join(HERE, "kernel_certificate.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(out)
    json.dump(old, open(path, "w"), indent=1)
