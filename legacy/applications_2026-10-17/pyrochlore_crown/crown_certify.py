"""Certified lower bound on the smallest eigenvalue of the 18-site pyrochlore "crown" cluster
(Kranitz & Penc, arXiv:2601.07800, reduced model with c5 = 0), and hence on the ground-state energy
per site of the S=1/2 pyrochlore Heisenberg antiferromagnet (their Anderson-type argument, Eq. 11 with
M = N).

Rigour chain
  1. Exact model. Couplings are rationals c1 = n1/10^7, c3 = c12 = n3/10^7,
     c17 = (10^7 - 2 n1 - 8 n3) / (2 10^7), so that the embedding constraint
     J = 2 c1 + 4 c3 + 4 c12 + 2 c17 = 1 holds exactly. K = 8 10^7 makes K*H an integer matrix in
     the S^z basis (S.S has entries in {±1/4, 1/2}).
  2. SU(2): H commutes with total spin, 18 spins 1/2 have integer total spin, so every multiplet
     has an S^z = 0 member: lambda_min(H) = lambda_min(H restricted to S^z = 0).
  3. Abelian symmetry group G = Z2^7 (swap of the two outer sites of each of the six tetrahedra,
     which is a symmetry because c3 = c12, and global spin flip). For each character chi of G,
     W_chi has one column per orbit on which chi is trivial on the stabilizer, entries ±1. The
     W_chi are mutually orthogonal and their column spaces sum to the S^z = 0 space (checked:
     dimensions add up, and W^T W is diagonal with the orbit sizes).
  4. For each chi: A = W^T (K H - s I) W = W^T K H W - s G  is computed EXACTLY (integers).
     Lemma 1 of the CERTIFY-ED paper (inertia with backward error beta) on A: if A has no negative
     eigenvalue in the computed LDL^T then lambda_min(A) >= -beta, and for x = W y in the sector,
     x^T (K H - s) x >= -beta |y|^2 >= -(beta / g_min) |x|^2, g_min = smallest orbit size.
     Hence lambda_min(K H) >= s - max_chi beta_chi / g_min_chi.
  5. e_GS >= lambda_min(H) >= (s - max beta/g_min) / K.
"""
import json, sys, time
from fractions import Fraction
from itertools import product

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
import certify_ed as ce
from certify_ed.basis import magnetization_state_set
from certify_ed.inertia import inertia_dense

N = 18
PAIRS = [(1, 2), (6, 12), (9, 16), (17, 18), (7, 13), (3, 10)]
HEX = [4, 5, 11, 15, 14, 8]
TET_HEX = {(1, 2): (4, 5), (6, 12): (5, 11), (9, 16): (11, 15), (17, 18): (15, 14), (7, 13): (14, 8),
           (3, 10): (8, 4)}
C3_SITES = (1, 12, 9, 18, 7, 10)   # outer site whose two hexagon bonds carry c3 (the other carries c12)


def bonds(c1, c3, c12, c17):
    b = [(c1, p[0], p[1]) for p in PAIRS]
    for p, (h1, h2) in TET_HEX.items():
        for o in p:
            for h in (h1, h2):
                b.append((c3 if o in C3_SITES else c12, o, h))
    b += [(c17, HEX[i], HEX[(i + 1) % 6]) for i in range(6)]
    assert len(b) == 36 and len({tuple(sorted((i, j))) for _, i, j in b}) == 36
    return b


def spec_for(n1, n3, scale=1, strict=None):
    """ModelSpec of scale*H. With scale = K = 8e7 all couplings*S.S matrix elements are integers."""
    c1, c3 = Fraction(n1, 10 ** 7), Fraction(n3, 10 ** 7)
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    assert 2 * c1 + 4 * c3 + 4 * c3 + 2 * c17 == 1
    bl = bonds(c1, c3, c3, c17)
    zz = [[float(scale * c), i - 1, j - 1] for c, i, j in bl]
    pm = [[float(scale * c / 2), i - 1, j - 1] for c, i, j in bl]
    strict = scale != 1 if strict is None else strict
    for (c, _, _), a, b in zip(bl, zz, pm):     # couplings exactly representable in binary64
        assert not strict or Fraction(a[0]) == scale * c and Fraction(b[0]) == scale * c / 2
    return ce.ModelSpec.from_static(N, ce.LocalSpace.spin_space("1/2", "spin"),
                                    [["zz", zz], ["+-", pm], ["-+", pm]], name="pyrochlore_crown_reduced",
                                    parameters={"n1": n1, "n3": n3, "scale": scale}), (c1, c3, c17)


def bitperm(states, perm):
    """Apply a site permutation (0-based, site i -> perm[i]) to integer basis states (site0 = msb)."""
    out = np.zeros_like(states)
    for i in range(N):
        bit = (states >> (N - 1 - i)) & 1
        out |= bit << (N - 1 - perm[i])
    return out


def generators():
    gens = []
    for a, b in PAIRS:
        p = list(range(N))
        p[a - 1], p[b - 1] = b - 1, a - 1
        gens.append(("perm", p))
    gens.append(("flip", None))
    return gens


def apply(gen, states):
    kind, p = gen
    return bitperm(states, p) if kind == "perm" else states ^ ((1 << N) - 1)


def sector_isometries(states):
    """Unnormalised symmetrised bases W_chi (entries ±1), one per character of Z2^7."""
    d = len(states)
    index = {int(s): k for k, s in enumerate(states)}
    gens = generators()
    imgs = [np.array([index[int(t)] for t in apply(g, states)]) for g in gens]   # generator -> index map
    ng = len(gens)
    # all group elements as index maps and their bit-vectors
    elems = []
    for bits in product((0, 1), repeat=ng):
        m = np.arange(d)
        for k, bk in enumerate(bits):
            if bk:
                m = imgs[k][m]
        elems.append((np.array(bits), m))
    # orbit representative = smallest index in the orbit
    allimg = np.stack([m for _, m in elems])          # |G| x d
    rep = allimg.min(axis=0)
    reps = np.unique(rep)
    out = {}
    for chi in product((0, 1), repeat=ng):            # chi(g) = (-1)^{chi . bits(g)}
        chi = np.array(chi)
        rows, cols, vals = [], [], []
        col = 0
        gmin = None
        for r in reps:
            coef = {}
            for bits, m in elems:
                t = int(m[r])
                sgn = -1 if (chi @ bits) % 2 else 1
                coef[t] = coef.get(t, 0) + sgn
            if all(v == 0 for v in coef.values()):
                continue
            stab = len(elems) // len(coef)
            for t, v in coef.items():
                assert abs(v) == stab
                rows.append(t); cols.append(col); vals.append(v // stab)
            gmin = len(coef) if gmin is None else min(gmin, len(coef))
            col += 1
        if col:
            out[tuple(chi)] = (sp.csr_matrix((vals, (rows, cols)), shape=(d, col), dtype=np.float64), gmin)
    return out


def certify(n1, n3, margin_rel=2e-8, log=print):
    K = 8 * 10 ** 7
    spec_K, (c1, c3, c17) = spec_for(n1, n3, scale=K)
    st = magnetization_state_set(spec_K, 0)
    states = np.asarray(st.states_in_order(), dtype=np.int64)
    HK = ce.build_bitwise(spec_K, st).tocsr()
    assert np.all(HK.data == np.round(HK.data)), "K*H must be an integer matrix"
    assert abs(HK - HK.T).max() == 0
    # floating-point estimate of the ground energy (only places the shift)
    v0 = np.random.default_rng(12345).standard_normal(HK.shape[0])
    e0_est = float(spla.eigsh(HK / K, k=1, which="SA", tol=0, v0=v0, return_eigenvectors=False)[0])
    s = float(np.floor((e0_est - margin_rel * abs(e0_est)) * K))   # integer shift in K*H units
    t0 = time.time()
    secs = sector_isometries(states)
    total = sum(W.shape[1] for W, _ in secs.values())
    assert total == len(states), (total, len(states))
    rec = {"n1": n1, "n3": n3, "c1": str(c1), "c3": str(c3), "c17": str(c17), "K": K,
           "dim_Sz0": len(states), "n_sectors": len(secs), "e0_estimate": e0_est, "shift_K_units": s,
           "sectors": []}
    worst = 0.0
    for chi, (W, gmin) in secs.items():
        G = (W.T @ W).toarray()
        assert np.allclose(G, np.diag(np.diag(G))) and np.all(np.diag(G) == np.round(np.diag(G)))
        B = (W.T @ (HK @ W)).toarray()
        assert np.all(B == np.round(B)) and np.max(np.abs(B)) < 2 ** 50
        A = B - s * G                                   # exact: integers below 2^53
        assert np.max(np.abs(A)) < 2 ** 52
        r = inertia_dense(A, 0.0)
        if not r.conclusive or r.negative != 0:
            raise RuntimeError(f"sector {chi}: inertia {r}")
        worst = max(worst, r.beta / gmin)
        rec["sectors"].append({"chi": list(chi), "dim": W.shape[1], "g_min": gmin, "beta": r.beta,
                               "seconds": r.seconds})
    lb_K = s - worst
    rec["lower_bound"] = lb_K / K
    # the bound is reported rounded DOWN at 9 digits
    rec["lower_bound_rounded_down"] = float(np.floor(lb_K / K * 1e9) / 1e9)
    rec["max_beta_over_gmin_K_units"] = worst
    rec["largest_sector"] = max(x["dim"] for x in rec["sectors"])
    rec["seconds"] = time.time() - t0
    log(f"n1={n1} n3={n3}: e0 est {e0_est:.10f}; certified lambda_min >= {lb_K / K:.12f} "
        f"({len(secs)} sectors, largest {rec['largest_sector']}, {rec['seconds']:.1f}s)")
    return rec


if __name__ == "__main__":
    n1, n3 = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (381414, 817957)
    rec = certify(n1, n3)
    json.dump(rec, open(f"crown_{n1}_{n3}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def exact_check(n1, n3, shift_K_units, log=print):
    """Re-check every sector with the ball-arithmetic checker (no floating-point model assumed)."""
    from certify_ed.checker import check_semidefinite_exact as check_semidefinite_lower_bound
    K = 8 * 10 ** 7
    spec_K, _ = spec_for(n1, n3, scale=K)
    st = magnetization_state_set(spec_K, 0)
    states = np.asarray(st.states_in_order(), dtype=np.int64)
    HK = ce.build_bitwise(spec_K, st).tocsr()
    secs = sector_isometries(states)
    worst, rows = 0.0, []
    t0 = time.time()
    for chi, (W, gmin) in sorted(secs.items(), key=lambda kv: kv[1][0].shape[1]):
        G = (W.T @ W).toarray()
        A = (W.T @ (HK @ W)).toarray() - shift_K_units * G
        r = check_semidefinite_lower_bound(A)
        if r.get("nu") != 0:
            raise RuntimeError(f"sector {chi}: {r}")
        worst = max(worst, r["bound_E"] / gmin)
        rows.append({"chi": list(chi), "dim": W.shape[1], "g_min": gmin, "bound_E": r["bound_E"]})
        log(f"  sector dim {W.shape[1]:5d}: nu=0, ||E|| <= {r['bound_E']:.3e}  [{time.time()-t0:.0f}s]")
    return {"lower_bound_exact_check": (shift_K_units - worst) / K, "max_bound_E_over_gmin": worst,
            "sectors": rows, "seconds": time.time() - t0}
