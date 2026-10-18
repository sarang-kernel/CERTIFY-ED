"""Certified Pomata–Wei criterion for the spin-3/2 AKLT model on the star lattice [3.12.12].

Claim: eta < 1/4 (z = 4), hence the model is gapped (Pomata & Wei, arXiv:1911.01410, Method I).
Here eta^2 is the largest squared cosine, other than 1, of the principal angles between
A = ker(H_Gi) (x) C^{4^3} and B = C^{4^3} (x) ker(H_Gj) on the union T_L - T - T_R.

Rigour chain (all integer arithmetic except two floating-point factorizations, whose residuals are
re-checked exactly):
  1. Psi_i, Psi_j, Psi_U (VBS maps, scaled integers) satisfy Ht Psi = 0 exactly and have exact ranks
     16, 16, 32 -> dim ker H_Gamma >= 16 and dim(A cap B) >= dim ker H_U >= 32.
  2. dim ker H_Gamma <= 16: inertia of the integer symmetric matrix W(Ht - s W^-1 ... ) congruent to
     H - s I at s = 1/8, factorization residual checked exactly.  Hence ker H_Gamma = range Psi.
  3. With exact Gram matrices G_A = G_i (x) W_R, G_B = W_L (x) G_j and cross Gram C, the squared cosines
     are the eigenvalues of the pencil (C G_B^-1 C^T, G_A). K = s (C G_B^-1 C^T - t G_A) is an exact
     integer matrix; an exactly re-checked LDL^T shows at most 32 eigenvalues of K exceed beta, so at
     most 32 pencil eigenvalues exceed t' = t + beta / (s lambda_min(G_A)). Together with 1., exactly
     32 cosines equal 1 and all others satisfy cos^2 <= t'.
"""
import json, os, sys, time
from fractions import Fraction

import numpy as np
import scipy.linalg as la

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from vbs_exact import hamiltonian_scaled, vbs_map, weight_vector, WEIGHT

import flint

TRI = lambda a: [(a, a + 1), (a + 1, a + 2), (a, a + 2)]
# union: T_L = 0,1,2   T = 3,4,5   T_R = 6,7,8 ; bridges (2,3) and (4,6)
GI = dict(n=6, edges=TRI(0) + TRI(3) + [(2, 3)], free=[0, 1, 4, 5])
GJ = dict(n=6, edges=TRI(0) + TRI(3) + [(1, 3)], free=[0, 2, 4, 5])           # local: T = 0,1,2 ; T_R = 3,4,5
GU = dict(n=9, edges=TRI(0) + TRI(3) + TRI(6) + [(2, 3), (4, 6)], free=[0, 1, 5, 7, 8])


def fmpz_mat(a):
    return flint.fmpz_mat([[int(x) for x in row] for row in a])


def exact_rank(Psi):
    # rank of an integer matrix: rank of the Gram matrix Psi^T Psi equals rank Psi (exact)
    G = fmpz_mat(Psi.T) * fmpz_mat(Psi)
    return G.rank()


def ldl_inertia_exact(A_int):
    """A_int: list of lists of Python ints (symmetric). Float LDL^T, exact residual with FLINT.
    Returns (negative count of D, positive count, bound on ||E||_inf)."""
    from certify_ed.checker import negative_count_exact
    n = len(A_int)
    Af = np.array([[float(x) for x in row] for row in A_int])
    lu, d, perm = la.ldl(Af, lower=True, hermitian=True)
    assert np.all(np.diag(lu[perm]) == 1.0)
    pL = 52
    dmax = float(np.max(np.abs(d)))
    pD = int(52 - np.ceil(np.log2(dmax)))
    Li = np.rint(np.ldexp(lu, pL)).astype(np.int64)
    Di = np.rint(np.ldexp(d, pD)).astype(np.int64)
    Dt = np.ldexp(Di.astype(np.float64), -pD)
    neg, amb = negative_count_exact(Dt)
    assert not amb
    # E * 2^(2pL+pD) = Li Di Li^T - A 2^(2pL+pD), exact
    sh = 2 * pL + pD
    L = flint.fmpz_mat(Li.tolist()); Dm = flint.fmpz_mat(Di.tolist())
    P = (L * Dm) * L.transpose()
    worst = 0
    for i in range(n):
        row = 0
        Ai = A_int[i]
        for j in range(n):
            a = Ai[j]
            ae = a << sh if sh >= 0 else None
            if sh >= 0:
                row += abs(int(P[i, j]) - ae)
            else:
                row_f = Fraction(int(P[i, j]), 1) - Fraction(a) * Fraction(2) ** sh
                row += abs(row_f)
        worst = max(worst, row)
    bound = Fraction(worst) / (Fraction(2) ** sh) if sh >= 0 else Fraction(worst) / Fraction(2) ** sh
    zero = 0
    pos = n - neg
    return neg, pos, bound


def main():
    rec = {"model": "spin-3/2 AKLT, star lattice", "criterion": "eta < 1/4 (z=4)"}
    t0 = time.time()
    # ---- 1. VBS maps, exact kernel membership and ranks
    out = {}
    for name, g in (("Gi", GI), ("Gj", GJ), ("U", GU)):
        Psi = vbs_map(g["n"], g["edges"], g["free"])
        Ht, den = hamiltonian_scaled(g["n"], g["edges"])
        HPsi = Ht @ Psi
        assert np.max(np.abs(Psi)) < 2 ** 40
        rec[f"{name}_HPsi_zero"] = bool(np.all(HPsi == 0))
        rec[f"{name}_rank"] = int(exact_rank(Psi))
        out[name] = (Psi, Ht, den)
    assert rec["Gi_HPsi_zero"] and rec["Gj_HPsi_zero"] and rec["U_HPsi_zero"]
    assert (rec["Gi_rank"], rec["Gj_rank"], rec["U_rank"]) == (16, 16, 32), rec
    print(f"1. VBS maps exact: ranks 16/16/32, H Psi = 0  [{time.time()-t0:.0f}s]", flush=True)

    # ---- 2. dim ker H_Gamma <= 16: inertia of W Ht - s W (congruent to den (H - s/den I))
    cache_path = os.path.join(HERE, "star_step2_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for name in ("Gi", "Gj"):
        Psi, Ht, den = out[name]
        if name in cache:          # deterministic; recomputed when the cache file is deleted
            rec[f"{name}_kernel_count"] = cache[name]
            print(f"2. {name}: from cache {cache[name]}", flush=True)
            continue
        W = weight_vector(6)
        M = (Ht.multiply(W[:, None])).toarray()            # W Ht, symmetric integer
        assert np.array_equal(M, M.T)
        s_num, s_den = 1, 8                                 # shift s = den/8 in units of den*H
        A = [[int(M[i, j]) * s_den - (s_num * den * int(W[i]) if i == j else 0) for j in range(M.shape[0])]
             for i in range(M.shape[0])]
        neg, pos, bE = ldl_inertia_exact(A)
        # #(eigenvalues of A < -||E||) <= neg; A = s_den * D (den H - (den/8) I) D, lambda_min(D^2) = 1
        rec[f"{name}_kernel_count"] = {"negative": neg, "bound_E": float(bE)}
        cache[name] = rec[f"{name}_kernel_count"]
        json.dump(cache, open(cache_path, "w"))
        assert neg == 16 and bE < den, (name, neg, float(bE), den)
        print(f"2. {name}: exactly 16 eigenvalues of H below 1/8 - eps (||E|| <= {float(bE):.2e})  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- 3. cosines via the pencil
    Psi_i, Psi_j = out["Gi"][0], out["Gj"][0]
    W3 = weight_vector(3)
    W6 = weight_vector(6)
    Gi = Psi_i.T @ (W6[:, None] * Psi_i)
    Gj = Psi_j.T @ (W6[:, None] * Psi_j)
    Pi = Psi_i.reshape(64, 64, 16)        # (l, c, a)
    Pj = Psi_j.reshape(64, 64, 16)        # (c, r, b)
    core = np.einsum("lca,c,crb->larb", Pi, W3, Pj)                       # int64
    C = np.einsum("l,r,larb->arlb", W3, W3, core).reshape(1024, 1024)      # rows (a,r), cols (l,b)
    assert np.max(np.abs(C)) < 2 ** 62
    # G_B^-1 = W_L^-1 (x) Gj^-1 ; adjugate scaling
    Gj_f = flint.fmpz_mat(Gj.tolist())
    detGj = int(Gj_f.det())
    adjGj = flint.fmpq_mat(Gj_f).inv() * detGj
    adjGj_int = [[int(adjGj[i, j]) for j in range(16)] for i in range(16)]
    lcmW = 27
    Wl_inv = [lcmW // int(w) for w in W3]
    # Mid = diag(27/W_L) (x) adj(Gj), indices (l, b)
    Mid = np.zeros((1024, 1024), dtype=object)
    for l in range(64):
        for b in range(16):
            for b2 in range(16):
                Mid[l * 16 + b, l * 16 + b2] = Wl_inv[l] * adjGj_int[b][b2]
    Cf = flint.fmpz_mat(C.tolist())
    Midf = flint.fmpz_mat(Mid.tolist())
    Q = Cf * Midf * Cf.transpose()                                       # = 27 det(Gj) C G_B^-1 C^T
    GA = np.kron(Gi, np.diag(W3))                                       # G_A = Gi (x) W_R
    scale = lcmW * detGj
    # lambda_min(G_A) >= lambda_min(Gi) * min W_R = lambda_min(Gi); certify a rational lower bound
    lam_gi = float(np.min(la.eigvalsh(Gi.astype(float))))
    lam_low = Fraction(lam_gi * 0.5).limit_denominator(10 ** 6)
    Mt = [[Fraction(int(Gi[i, j])) - (lam_low if i == j else 0) for j in range(16)] for i in range(16)]
    for k in range(1, 17):                       # Sylvester's criterion, exact
        assert flint.fmpq_mat(k, k, [flint.fmpq(Mt[i][j].numerator, Mt[i][j].denominator)
                                     for i in range(k) for j in range(k)]).det() > 0
    rec["lambda_min_GA_lower"] = float(lam_low)
    results = {}
    for t in (Fraction(1, 16), Fraction(12331, 1000000)):              # 1/16 (criterion) and just above 0.11104^2
        p, q = t.numerator, t.denominator
        K = [[int(Q[i, j]) * q - (p * scale * int(GA[i, j])) for j in range(1024)] for i in range(1024)]
        neg, pos, bE = ldl_inertia_exact(K)
        tprime = t + Fraction(bE) / (Fraction(q * scale) * lam_low)
        ok = pos <= 32
        results[str(t)] = {"negative": neg, "positive_in_D": pos, "bound_E": float(bE),
                           "t_prime": float(tprime), "at_most_32_above_tprime": ok}
        print(f"3. t = {t} ({float(t):.6f}): D has {pos} positive pivots; ||E|| <= {float(bE):.3e}; "
              f"cos^2 <= {float(tprime):.10f} for all non-unit angles -> eta <= {float(tprime)**0.5:.10f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    rec["pencil"] = results
    eta_up = min(v["t_prime"] for v in results.values() if v["at_most_32_above_tprime"]) ** 0.5
    rec["eta_upper"] = eta_up
    rec["criterion_holds"] = bool(eta_up < 0.25)
    rec["seconds"] = time.time() - t0
    json.dump(rec, open(os.path.join(HERE, "star_certificate.json"), "w"), indent=1)
    print(f"certified: eta <= {eta_up:.10f} < 1/4: {eta_up < 0.25}")


if __name__ == "__main__":
    main()
