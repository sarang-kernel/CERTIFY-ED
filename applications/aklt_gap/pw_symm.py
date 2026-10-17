"""pw_reduced with lattice symmetries: block-diagonalisation of the bordered matrix by an exact integer
congruence, so that the largest exactly checked factorisations shrink by up to a factor 4.

Symmetries used: site permutations sigma that map L, C and R to themselves and the union's edge set to
itself, and the global spin flip (on the self-conjugate S^z block). Each acts on every reduced
coordinate space (regions, Gamma boundaries) as a permutation; on the VBS states it acts up to a global
sign (singlet orientations). The action is verified exactly on the ingredients:
   G_L, G_C, G_R, G_i, G_j invariant;  Z[pi l, pi c, pi b] = s_Z Z[l, c, b];  Y[pi c, pi r, pi d] = s_Y Y
so that S_g = diag(Pi_A, s_Z s_Y Pi_B) satisfies S_g^T M S_g = M for the bordered matrix M.
For each character chi of the (abelian, involutive) group, integer vectors v = sum_g chi(g) S_g e_x
(one per orbit and character, dropped when zero) form W = [W_chi] with mutually orthogonal columns of
squared norm >= 1. W is nonsingular, W^T M W = blockdiag(W_chi^T M W_chi), so the inertia adds up
(Sylvester). For the perturbation argument, W = Q N with Q orthogonal and N >= I diagonal:
#(eig M > beta) = #pos(W^T (M - beta) W) <= #pos(W^T M W - beta I) <= sum of positive pivots, when beta
bounds every block's exact residual. The rest of the argument is as in pw_reduced.
"""
import itertools
import time
from fractions import Fraction

import numpy as np

import flint

from pw_general import Graph, ldl_inertia_exact, slot_reduction
from pw_reduced import reduced_region, union_coords, rank_mod_p_lower, lambda_min_lower, _gram_of_coords


def _slot_image(g, sl, sigma):
    kind, a, site = sl
    if kind == "free":
        return ("free", (sigma[a[0]], a[1]), sigma[site])
    u, v = a
    e = (sigma[u], sigma[v]) if (sigma[u], sigma[v]) in g.edges else (sigma[v], sigma[u])
    assert e in g.edges
    return ("edge", e, sigma[site])


def reduced_perm(g, slots, sigma=None, flip=False):
    """Permutation of reduced boundary configurations of a slot list induced by a site permutation (and/or
    the spin flip)."""
    red, reps, _ = slot_reduction(slots)
    nb = len(slots)
    if sigma is None:
        img_pos = list(range(nb))
    else:
        # slot k goes to the position of a slot of the image site with the same kind; free-slot indices
        # and edge identities are matched exactly, the counts per site are what matter
        pos = {sl: k for k, sl in enumerate(slots)}
        img_pos = []
        for sl in slots:
            im = _slot_image(g, sl, sigma)
            if im not in pos:            # free slot index may differ: any free slot of the same site
                cands = [k for k, s2 in enumerate(slots) if s2[0] == "free" and s2[2] == im[2]]
                im_k = cands[[s2 for s2 in slots if s2[0] == "free" and s2[2] == sl[2]].index(sl)]
            else:
                im_k = pos[im]
            img_pos.append(im_k)
        assert sorted(img_pos) == list(range(nb))
    perm = []
    for f in reps:
        bits = [(f >> (nb - 1 - i)) & 1 for i in range(nb)]
        out = 0
        for k, bt in enumerate(bits):
            if flip:
                bt ^= 1
            if bt:
                out |= 1 << (nb - 1 - img_pos[k])
        perm.append(int(red[out]))
    assert sorted(perm) == list(range(len(reps)))
    return np.array(perm)



def ldl_inertia_lean(M, rows_per_block=256):
    """Memory-lean version of pw_general.ldl_inertia_exact for an object ndarray M of Python ints
    (symmetric). Same guarantee: (#negative pivots, #positive pivots, bound on ||Lt Dt Lt^T - M||_inf),
    the residual computed exactly in integer arithmetic, block of rows by block of rows."""
    import scipy.linalg as la
    from certify_ed.checker import negative_count_exact
    n = M.shape[0]
    maxbits = max(abs(int(x)).bit_length() for x in M.ravel())
    s_down = max(0, maxbits - 900)
    if s_down:
        Af = np.array([[float(int(x) >> s_down) if x >= 0 else -float((-int(x)) >> s_down) for x in row]
                       for row in M])
    else:
        Af = M.astype(float)
    lu, d, perm = la.ldl(Af, lower=True, hermitian=True, overwrite_a=True)
    del Af
    assert np.all(np.diag(lu[perm]) == 1.0)
    pL = 52
    pD = int(52 - np.ceil(np.log2(float(np.max(np.abs(d))))))
    Li = np.rint(np.ldexp(lu, pL)).astype(np.int64)
    del lu
    Di = np.rint(np.ldexp(d, pD)).astype(np.int64)
    del d
    neg, amb = negative_count_exact(np.ldexp(Di.astype(np.float64), -pD))
    assert not amb
    sh = 2 * pL + pD - s_down
    LT = flint.fmpz_mat(Li.T.tolist())
    Dm = flint.fmpz_mat(Di.tolist())
    del Di
    two = flint.fmpz(2)
    worst = 0
    for r0 in range(0, n, rows_per_block):
        r1 = min(n, r0 + rows_per_block)
        Lb = flint.fmpz_mat(Li[r0:r1].tolist())
        Pb = (Lb * Dm) * LT                      # 2^sh (Lt 2^s_down Dt Lt^T)[rows]
        Ab = flint.fmpz_mat([[int(x) for x in row] for row in M[r0:r1]])
        if sh >= 0:
            Eb = Pb - Ab * (two ** sh)
        else:
            Eb = Pb * (two ** (-sh)) - Ab
        ents = Eb.entries()
        for i in range(r1 - r0):
            sm = 0
            for x in ents[i * n:(i + 1) * n]:
                sm += abs(int(x))
            worst = max(worst, sm)
        del Pb, Eb, Ab, Lb, ents
    den = Fraction(2) ** sh if sh >= 0 else Fraction(1)
    return neg, n - neg, Fraction(worst) / den


def certify_eta_sym(g: Graph, L, C, R, thresholds, sigmas=(), use_flip=True, log=print):
    t0 = time.time()
    rL, rC, rR = (reduced_region(g, X) for X in (L, C, R))
    nL, nC, nR = rL["n"], rC["n"], rR["n"]
    Ti, downs_i, sing_i, slots_i = union_coords(g, [L, C], [rL, rC])
    Tj, downs_j, sing_j, slots_j = union_coords(g, [C, R], [rC, rR])
    fi, fj = len(Ti), len(Tj)
    Gi = flint.fmpz_mat(_gram_of_coords(Ti, [rL["Gn"], rC["Gn"]]))
    Gj = flint.fmpz_mat(_gram_of_coords(Tj, [rC["Gn"], rR["Gn"]]))
    assert Gi.rank() == fi and Gj.rank() == fj
    Gin = np.array([[int(x) for x in row] for row in Gi.tolist()], dtype=object)
    Gjn = np.array([[int(x) for x in row] for row in Gj.tolist()], dtype=object)
    TU, _, _, _ = union_coords(g, [L, C, R], [rL, rC, rR])
    n_unit = rank_mod_p_lower(TU)
    lam = {k: lambda_min_lower(G) for k, G in (("L", rL["G"]), ("R", rR["G"]), ("i", Gi), ("j", Gj))}
    nu, mu = lam["i"] * lam["R"], lam["L"] * lam["j"]
    GL, GC, GR = rL["Gn"], rC["Gn"], rR["Gn"]
    Z = np.zeros((nL, nC, fi), dtype=object); Z[:] = 0
    for b, col in enumerate(Ti):
        for (l2, c), v in col.items():
            Z[:, c, b] += GL[:, l2] * v
    Y = np.zeros((nC, nR, fj), dtype=object); Y[:] = 0
    for d, col in enumerate(Tj):
        for (c2, r2), v in col.items():
            Y[:, :, d] += np.outer(GC[:, c2], GR[:, r2]) * v
    log(f"  reduced dims L {nL}, C {nC}, R {nR}; Gamma boundaries {fi}, {fj}; unit cosines >= {n_unit}  "
        f"[{time.time()-t0:.0f}s]")

    # ---- group elements: products of the given site permutations and the flip; exact equivariance
    def perms_for(sigma, flip):
        return dict(L=reduced_perm(g, rL["slots"], sigma, flip), C=reduced_perm(g, rC["slots"], sigma, flip),
                    R=reduced_perm(g, rR["slots"], sigma, flip), i=reduced_perm(g, slots_i, sigma, flip),
                    j=reduced_perm(g, slots_j, sigma, flip))
    n = len(g.z)
    ident = list(range(n))
    gens = [(list(s), False) for s in sigmas] + ([(ident, True)] if use_flip else [])
    elements = []
    for mask in itertools.product((0, 1), repeat=len(gens)):
        sig = ident
        fl = False
        for m, (s, f) in zip(mask, gens):
            if m:
                sig = [s[x] for x in sig]
                fl ^= f
        elements.append((mask, sig, fl))
    acts = []
    for mask, sig, fl in elements:
        P = perms_for(sig, fl)
        for G, p in ((GL, P["L"]), (GC, P["C"]), (GR, P["R"]), (Gin, P["i"]), (Gjn, P["j"])):
            assert np.array_equal(G[np.ix_(p, p)], G), "Gram matrix not invariant"
        Zp = Z[np.ix_(P["L"], P["C"], P["i"])]
        Yp = Y[np.ix_(P["C"], P["R"], P["j"])]
        sZ = 1 if np.array_equal(Zp, Z) else (-1 if np.array_equal(Zp, -Z) else 0)
        sY = 1 if np.array_equal(Yp, Y) else (-1 if np.array_equal(Yp, -Y) else 0)
        assert sZ and sY, f"not a symmetry: {mask}"
        acts.append(dict(mask=mask, flip=fl, P=P, sC=sZ * sY))
    log(f"  symmetry group of order {len(acts)} verified exactly; C signs {[a['sC'] for a in acts]}  "
        f"[{time.time()-t0:.0f}s]")

    labA, labB = {}, {}
    for b in range(fi):
        for r in range(nR):
            labA.setdefault(int(sing_i + downs_i[b] + rR["downs"][r] + rR["inner"]), []).append((b, r))
    for l in range(nL):
        for d in range(fj):
            labB.setdefault(int(sing_j + rL["downs"][l] + downs_j[d] + rL["inner"]), []).append((l, d))

    def block_bases(key):
        """Symmetry-adapted integer bases of block 'key': list over characters of (A list, B list), each
        entry a list of (index-in-block, coefficient)."""
        ra, cb = labA[key], labB.get(key, [])
        ia = {x: k for k, x in enumerate(ra)}; ib = {x: k for k, x in enumerate(cb)}
        # the flip maps this block to itself only if the labels are self-conjugate
        G_here = [a for a in acts if (not a["flip"]) or
                  all((a["P"]["i"][b], a["P"]["R"][r]) in ia for (b, r) in ra[:50])]
        if any(a["flip"] for a in G_here):
            assert all((a["P"]["i"][b], a["P"]["R"][r]) in ia for a in G_here for (b, r) in ra)
        partner = None
        flips = [a for a in acts if a["flip"] and not any(a["mask"][i] for i in range(len(a["mask"]) - 1))]
        if flips and not any(a["flip"] for a in G_here):
            fa = flips[0]
            b0, r0 = ra[0]
            img = (int(fa["P"]["i"][b0]), int(fa["P"]["R"][r0]))
            partner = next(k2 for k2, v in labA.items() if img in set(v))
            # the flip maps the whole block onto the partner block (A and B sides)
            assert {(int(fa["P"]["i"][b]), int(fa["P"]["R"][r])) for (b, r) in ra} == set(labA[partner])
            assert {(int(fa["P"]["L"][l]), int(fa["P"]["j"][d])) for (l, d) in cb} == set(labB.get(partner, []))
        # characters: sign vector on the generator masks
        gen_used = sorted({i for a in G_here for i, m in enumerate(a["mask"]) if m})
        chars = list(itertools.product((1, -1), repeat=len(gen_used)))
        def chi_val(ch, a):
            v = 1
            for c, gi in zip(ch, gen_used):
                if a["mask"][gi]:
                    v *= c
            return v
        out = []
        for ch in chars:
            Avecs, seenA = [], set()
            for k, (b, r) in enumerate(ra):
                if k in seenA:
                    continue
                vec = {}
                orbit = set()
                for a in G_here:
                    kk = ia[(a["P"]["i"][b], a["P"]["R"][r])]
                    orbit.add(kk)
                    vec[kk] = vec.get(kk, 0) + chi_val(ch, a)
                seenA |= orbit
                vec = [(kk, c) for kk, c in vec.items() if c != 0]
                if vec:
                    Avecs.append(vec)
            Bvecs, seenB = [], set()
            for k, (l, d) in enumerate(cb):
                if k in seenB:
                    continue
                vec = {}
                orbit = set()
                for a in G_here:
                    kk = ib[(a["P"]["L"][l], a["P"]["j"][d])]
                    orbit.add(kk)
                    vec[kk] = vec.get(kk, 0) + chi_val(ch, a) * a["sC"]
                seenB |= orbit
                vec = [(kk, c) for kk, c in vec.items() if c != 0]
                if vec:
                    Bvecs.append(vec)
            out.append((ch, Avecs, Bvecs))
        assert sum(len(x[1]) for x in out) == len(ra) and sum(len(x[2]) for x in out) == len(cb)
        return ra, cb, out, partner

    def padded(vecs, base_idx):
        m = max(len(v) for v in vecs)
        idx = np.zeros((len(vecs), m), dtype=np.int64); cf = np.zeros((len(vecs), m), dtype=object); cf[:] = 0
        for i, v in enumerate(vecs):
            for k, (kk, c) in enumerate(v):
                idx[i, k] = kk; cf[i, k] = c
        return idx, cf

    plans = {key: block_bases(key) for key in sorted(labA)}
    sizes = sorted((len(A) + len(B) for (_, _, o, pt) in plans.values() if pt is None or pt > -1 for (_, A, B) in o), reverse=True)
    log(f"  largest symmetry-adapted bordered blocks {sizes[:4]} (unreduced max "
        f"{max(len(labA[k]) + len(labB.get(k, [])) for k in labA)})  [{time.time()-t0:.0f}s]")
    out = {"pencil_size": fi * nR, "unit_cosines_at_least": int(n_unit), "group_order": len(acts),
           "largest_block": sizes[0], "nu_lower": float(nu), "mu_lower": float(mu), "thresholds": {}}
    for t in thresholds:
        p, q = t.numerator, t.denominator
        pos_total, beta = 0, Fraction(0)
        done = {}
        for key, (ra, cb, chars, partner) in plans.items():
            if partner is not None and partner in done:
                # the spin flip maps this block's bordered matrix onto the partner's by a signed
                # permutation (equivariance verified exactly above): same inertia, same residual bound
                pos_total += done[partner]
                done[key] = done[partner]
                continue
            pos_block = 0
            bsA = np.array([x[0] for x in ra]); rsA = np.array([x[1] for x in ra])
            if cb:
                lsB = np.array([x[0] for x in cb]); dsB = np.array([x[1] for x in cb])
            for ch, Avecs, Bvecs in chars:
                if not Avecs:
                    continue
                ia_, ca_ = padded(Avecs, None)
                # A-A part: sum over padded terms of coef coef' GA[...]
                nA = len(Avecs)
                GA = np.zeros((nA, nA), dtype=object); GA[:] = 0
                for s1 in range(ia_.shape[1]):
                    b1, r1 = bsA[ia_[:, s1]], rsA[ia_[:, s1]]
                    for s2 in range(ia_.shape[1]):
                        b2, r2 = bsA[ia_[:, s2]], rsA[ia_[:, s2]]
                        GA += np.outer(ca_[:, s1], ca_[:, s2]) * (Gin[np.ix_(b1, b2)] * GR[np.ix_(r1, r2)])
                if Bvecs:
                    ib_, cb_ = padded(Bvecs, None)
                    nB = len(Bvecs)
                    GB = np.zeros((nB, nB), dtype=object); GB[:] = 0
                    for s1 in range(ib_.shape[1]):
                        l1, d1 = lsB[ib_[:, s1]], dsB[ib_[:, s1]]
                        for s2 in range(ib_.shape[1]):
                            l2, d2 = lsB[ib_[:, s2]], dsB[ib_[:, s2]]
                            GB += np.outer(cb_[:, s1], cb_[:, s2]) * (GL[np.ix_(l1, l2)] * Gjn[np.ix_(d1, d2)])
                    Cm = np.zeros((nA, nB), dtype=object); Cm[:] = 0
                    for s1 in range(ia_.shape[1]):
                        b1, r1 = bsA[ia_[:, s1]], rsA[ia_[:, s1]]
                        for s2 in range(ib_.shape[1]):
                            l2, d2 = lsB[ib_[:, s2]], dsB[ib_[:, s2]]
                            Cs = sum(Z[l2[None, :], c, b1[:, None]] * Y[c, r1[:, None], d2[None, :]] for c in range(nC))
                            Cm += np.outer(ca_[:, s1], cb_[:, s2]) * Cs
                    M = np.empty((nA + nB, nA + nB), dtype=object)
                    M[:nA, :nA] = -p * GA; del GA
                    M[nA:, nA:] = -q * GB; del GB
                    M[:nA, nA:] = q * Cm
                    M[nA:, :nA] = M[:nA, nA:].T; del Cm
                else:
                    M = -p * GA
                neg, pos, bE = ldl_inertia_lean(M)
                del M
                pos_total += pos
                pos_block += pos
                beta = max(beta, bE / q)
                log(f"    block {key} chi {ch}: size {pos + neg}, {pos} positive pivots, "
                    f"||E|| <= {float(bE / q):.2e}  [{time.time()-t0:.0f}s]")
            done[key] = pos_block
        tprime = (1 + beta / mu) * (t + beta / nu)
        ok = pos_total <= n_unit
        out["thresholds"][str(t)] = {"positive": pos_total, "beta": float(beta), "t_prime": float(tprime),
                                     "t_prime_exact": f"{tprime.numerator}/{tprime.denominator}", "certified": ok,
                                     "eta_upper": float(tprime) ** 0.5}
        log(f"  t = {float(t):.6f}: {pos_total} positive pivots (need <= {n_unit}); beta = {float(beta):.2e}; "
            f"t' = {float(tprime):.10f}; eta <= {float(tprime) ** 0.5:.10f}: {ok}  [{time.time()-t0:.0f}s]")
    out["seconds"] = time.time() - t0
    return out
