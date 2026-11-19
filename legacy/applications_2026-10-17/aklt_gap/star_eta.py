"""Pomata & Wei (arXiv:1911.01410), spin-3/2 AKLT on the star lattice [3.12.12], Method I:
eta for neighbouring subgraphs, where the gap criterion needs eta < 1/4 (z = 4 neighbours).

Geometry (inference from the paper's description): Gamma = an inter-triangle edge with the two
triangles it joins (6 vertices, 7 edges). Two neighbouring Gammas share one triangle. Their union is
three triangles T_L - T - T_R joined by two inter-triangle edges attached to different vertices of T.

eta is the largest cosine of a principal angle between A = range(Pi_i (x) 1_R) and B = range(1_L (x)
Pi_j) other than 1, Pi_Gamma the projector onto ker H_Gamma, H_Gamma = sum over edges of the projector
onto total spin 3 of the two spin-3/2 sites. Sites are ordered T_L(3), T(3), T_R(3).
"""
import sys, time
import numpy as np
import scipy.linalg as la

S = 1.5
d = 4
m = np.arange(S, -S - 1, -1)                       # basis |3/2>, |1/2>, |-1/2>, |-3/2>
Sz = np.diag(m)
Sp = np.zeros((d, d))
for k in range(1, d):                              # S+ |m> = sqrt(S(S+1) - m(m+1)) |m+1>
    Sp[k - 1, k] = np.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
Sm = Sp.T


def two_site_SS():
    return (np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp)))


def P3():
    """Projector onto total spin 3 of two spin-3/2: polynomial in X = S1.S2 with eigenvalues
    x_J = (J(J+1) - 2 S(S+1)) / 2, J = 0..3."""
    X = two_site_SS()
    xs = {J: (J * (J + 1) - 2 * S * (S + 1)) / 2 for J in range(4)}
    P = np.eye(d * d)
    for J in (0, 1, 2):
        P = P @ (X - xs[J] * np.eye(d * d)) / (xs[3] - xs[J])
    return P


def embed2(P, n, u, v):
    """Two-site operator P on sites (u, v) of an n-site chain of spin-3/2."""
    P4 = P.reshape(d, d, d, d)
    full = np.zeros((d ** n, d ** n))
    # apply via einsum on basis: build as sparse would be better; n <= 6 here (4096)
    idx = np.indices([d] * n).reshape(n, -1).T
    out = np.zeros((d ** n, d ** n))
    for col in range(d ** n):
        s = idx[col]
        a, b = s[u], s[v]
        for a2 in range(d):
            for b2 in range(d):
                val = P4[a2, b2, a, b]
                if val != 0.0:
                    t = s.copy(); t[u] = a2; t[v] = b2
                    row = int(np.ravel_multi_index(t, [d] * n))
                    out[row, col] += val
    return out


def gamma_hamiltonian(tri1, tri2, bridge):
    """6 sites: tri1 = (0,1,2), tri2 = (3,4,5); bridge = (u, v) inter-triangle edge."""
    P = P3()
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), bridge]
    H = sum(embed2(P, 6, u, v) for u, v in edges)
    return H


def main():
    t0 = time.time()
    P = P3()
    w = np.linalg.eigvalsh(P)
    assert np.allclose(np.sort(w), np.r_[np.zeros(9), np.ones(7)], atol=1e-12)
    # Gamma_i on T_L (sites 0,1,2) and T (3,4,5), bridge from T_L vertex 2 to T vertex 3
    # Gamma_j on T (3,4,5) and T_R (6,7,8), bridge from T vertex 4 to T_R vertex 6
    Hi = gamma_hamiltonian(None, None, (2, 3))
    Hj = gamma_hamiltonian(None, None, (1, 3))     # in Gamma_j's local order T=(0,1,2), T_R=(3,4,5): bridge T vertex 1 -> T_R vertex 3
    wi, Vi = la.eigh(Hi)
    wj, Vj = la.eigh(Hj)
    ki, kj = int(np.sum(wi < 1e-9)), int(np.sum(wj < 1e-9))
    print(f"ker dims {ki}, {kj}; next eigenvalues {wi[ki]:.6f} {wj[kj]:.6f}  ({time.time()-t0:.1f}s)")
    Qi = Vi[:, :ki].reshape(d ** 3, d ** 3, ki)       # (L, C, a)
    Qj = Vj[:, :kj].reshape(d ** 3, d ** 3, kj)       # (C, R, b)
    # M[(a, r), (l, b)] = sum_c conj(Qi[l, c, a]) Qj[c, r, b]
    M = np.einsum("lca,crb->arlb", Qi.conj(), Qj).reshape(ki * d ** 3, d ** 3 * kj)
    sv = np.linalg.svd(M, compute_uv=False)
    n1 = int(np.sum(sv > 1 - 1e-9))
    eta = float(sv[n1])
    print(f"singular values = 1: {n1};  eta = {eta:.10f}  (paper 0.1110430220); next {sv[n1+1]:.6f}")


if __name__ == "__main__":
    main()
