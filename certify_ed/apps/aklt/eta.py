"""Exact certificate for the Pomata-Wei quantity eta (arXiv:1911.01410) on a union L - C - R.

eta^2 is the largest squared cosine, other than 1, of the principal angles between
A = ker(H_Gi) (x) phys(R) and B = phys(L) (x) ker(H_Gj), Gamma_i = L u C, Gamma_j = C u R. With
ker H_Gamma = range(U_Gamma) (kernel lemma; kernel.py checks it exactly where feasible), only the parts
of A and B inside V = range U_L (x) range U_C (x) range U_R have nonzero cosines, so everything is
computed in V coordinates, with reduced boundary configurations (graph.py).

* Unit cosines. dim(A n B) >= rank of the union VBS states in V coordinates (U_X injective), computed
  as a rank modulo a prime (a valid lower bound).
* Pencil. The cosines are the eigenvalues of (C G_B^-1 C^T, G_A). For t = p/q the bordered integer
  matrix M = [[-p G_A, q C], [q C^T, -q G_B]] has, by Haynsworth inertia additivity (G_B > 0), as many
  positive eigenvalues as C G_B^-1 C^T - t G_A. A floating-point LDL^T of M is re-checked exactly
  (checker.inertia_exact): ||L D L^T - M||_inf <= q beta. Weyl and (G_B + beta)^-1 >= G_B^-1/(1 + beta/mu)
  give #(pencil eigenvalues > t') <= #positive pivots for t' = (1 + beta/mu)(t + beta/nu), with mu, nu
  certified lower bounds on lambda_min(G_B), lambda_min(G_A).
* Symmetry. Site permutations sigma that preserve L, C, R and the edge set, and the global spin flip,
  act on every reduced coordinate space as permutations and on VBS states up to global signs. Their
  equivariance is verified exactly on the ingredients (G_L, G_C, G_R, G_i, G_j invariant;
  Z, Y equivariant up to signs s_Z, s_Y), so S_g = diag(Pi_A, s_Z s_Y Pi_B) satisfies S_g^T M S_g = M.
  For each 1D character, integer vectors sum_g chi(g) S_g e_x form W with orthogonal columns of squared
  norm >= 1; W^T M W is block diagonal (Sylvester), and with W = Q N, N >= I, the residual bound of every
  block covers M. Blocks exchanged by the spin flip have equal inertia and are factored once.
* Everything is split into S^z blocks (total number of down virtual spins).
If at most dim(A n B) pencil eigenvalues exceed t', then eta^2 <= t'. No step uses the floating-point
model: floating point only proposes factorizations that are checked exactly.
"""
import itertools
import time
from fractions import Fraction

import numpy as np

import flint

from ...checker import inertia_exact


def _sqrt_up(x: Fraction) -> float:
    """A float >= sqrt(x) for a non-negative rational x (float() and math.sqrt round to nearest, so one
    step up after each bounds the exact value)."""
    import math
    return float(np.nextafter(math.sqrt(float(np.nextafter(float(x), np.inf))), np.inf))
from .graph import (Graph, slot_reduction, reduced_region, union_coords, gram_of_coords, rank_mod_p_lower,
                    lambda_min_lower)


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


def certify_eta(g: Graph, L, C, R, thresholds, sigmas=(), use_flip=True, log=print):
    t0 = time.time()
    rL, rC, rR = (reduced_region(g, X) for X in (L, C, R))
    nL, nC, nR = rL["n"], rC["n"], rR["n"]
    Ti, downs_i, sing_i, slots_i = union_coords(g, [L, C], [rL, rC])
    Tj, downs_j, sing_j, slots_j = union_coords(g, [C, R], [rC, rR])
    fi, fj = len(Ti), len(Tj)
    Gi = flint.fmpz_mat(gram_of_coords(Ti, [rL["Gn"], rC["Gn"]]))
    Gj = flint.fmpz_mat(gram_of_coords(Tj, [rC["Gn"], rR["Gn"]]))
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
                neg, pos, bE = inertia_exact(M)
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
                                     "eta_upper": _sqrt_up(tprime)}
        log(f"  t = {float(t):.6f}: {pos_total} positive pivots (need <= {n_unit}); beta = {float(beta):.2e}; "
            f"t' = {float(tprime):.10f}; eta <= {float(tprime) ** 0.5:.10f}: {ok}  [{time.time()-t0:.0f}s]")
    out["seconds"] = time.time() - t0
    return out
