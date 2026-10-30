"""Computational verification of docs/proofs/thm8_rho.md (Theorem 8, sector assembly with radius rho).

A  Heisenberg chains (spin-1/2, N = 4 and 6, exact dyadic matrices) in U(1) and momentum sectors: compose.certify_low_energy
   at several cuts; CERTIFIED results checked against the eigenvalues of the exact spec matrix (mpmath, 60 digits),
   plus the intermediate inequalities of the proof (Steps 2-3) in mp.
B  spin-1 chain (N = 3, irrational amplitudes, builder shift m > 0): eigenvalues of the declared model from
   checker.exact_matrix (D^-1 (W K) D^-1, Hermitian, exact rationals times sqrt of integer weights).
Adversarial: sector bases perturbed by 1e-14 .. 3e-11 (eta > 0) and blocks by 1e-13 (D > 0), within the audit tolerance: the statement must hold whenever CERTIFIED;
negative controls must not be CERTIFIED.
Usage: python docs/proofs/verify_thm8_rho.py [--cases 300] [--seed 4]
"""
from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from _common import Counter, TOL, bottleneck_sorted, eigh, func_herm, herm, norm2, to_mp  # noqa: E402

import certify_ed as ce  # noqa: E402
from certify_ed.basis import cyclic_isometry, translation  # noqa: E402
from certify_ed.checker import exact_matrix  # noqa: E402
from certify_ed.compose import SectorResult, certify_low_energy  # noqa: E402
from certify_ed.construct import full_state_set, hermitian_part  # noqa: E402


def exact_spectrum(spec):
    """Eigenvalues of the declared model (mp, 60 digits) from the exact rational matrix of checker.exact_matrix."""
    ex = exact_matrix(spec)
    n = len(ex.weights)
    S = mp.matrix(n, n)
    for r, c, a, b in zip(ex.rows, ex.cols, ex.re, ex.im):
        # W K entry: w_r (a + i b)/den
        S[int(r), int(c)] += mp.mpc(mp.mpf(int(ex.weights[r]) * int(a)), mp.mpf(int(ex.weights[r]) * int(b))) / int(ex.den)
    d = [1 / mp.sqrt(int(w)) for w in ex.weights]
    Hm = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            if S[i, j] != 0:
                Hm[i, j] = d[i] * S[i, j] * d[j]
    return eigh(Hm)[0], Hm


def u1_sectors(spec, H, rng, perturb_V=0.0, perturb_H=0.0):
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites * (spec.q - 1) + 1):          # all digit sums (q = 3 for spin 1)
        st = ce.u1_state_set(spec, nup)
        idx, found = full.index_of(st.states_in_order())
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        if perturb_V:
            Vd = Vs.toarray() + perturb_V * rng.standard_normal(Vs.shape)
            Vs = sp.csr_matrix(Vd)
        Hs = (Vs.conj().T @ H @ Vs).toarray()
        Hs = (Hs + Hs.conj().T) / 2
        if perturb_H:
            P = rng.standard_normal(Hs.shape)
            Hs = Hs + perturb_H * (P + P.T) / 2
        w, V = la.eigh(Hs)
        secs.append(SectorResult(f"nup={nup}", Vs, sp.csr_matrix(Hs), w, V))
    return secs


def momentum_sectors(spec, H, rng):
    st = ce.u1_state_set(spec, spec.n_sites // 2)
    full = full_state_set(spec)
    idx, _ = full.index_of(st.states_in_order())
    T = translation(spec)
    secs = []
    for k in range(spec.n_sites):
        Vk = cyclic_isometry(st, T, k)
        if Vk.shape[1] == 0:
            continue
        # embed the Sz=0 block isometry into the full space
        Vk = sp.csr_matrix(Vk)
        Vfull = sp.csr_matrix((Vk.data, (idx[Vk.tocoo().row], Vk.tocoo().col)), shape=(H.shape[0], Vk.shape[1]))
        Hs = (Vfull.conj().T @ H @ Vfull).toarray()
        Hs = (Hs + Hs.conj().T) / 2
        w, V = la.eigh(Hs)
        secs.append(SectorResult(f"k={k}", Vfull, sp.csr_matrix(Hs), w, V))
    # the other Sz blocks as U(1) sectors so that the decomposition is complete
    for nup in range(spec.n_sites + 1):
        if nup == spec.n_sites // 2:
            continue
        stn = ce.u1_state_set(spec, nup)
        idn, _ = full.index_of(stn.states_in_order())
        Vs = sp.csr_matrix((np.ones(idn.size), (idn, np.arange(idn.size))), shape=(H.shape[0], idn.size))
        Hs = (Vs.conj().T @ H @ Vs).toarray()
        w, V = la.eigh(Hs)
        secs.append(SectorResult(f"nup={nup}", Vs, sp.csr_matrix(Hs), w, V))
    return secs


def gap_cut(wf, rng, tol=1e-8):
    """A cut in a gap between two distinct levels of the sorted spectrum wf (never inside a multiplet)."""
    levels = [wf[0]]
    for v in wf[1:]:
        if v - levels[-1] > tol:
            levels.append(v)
    j = int(rng.integers(1, min(len(levels), 7)))
    return float((levels[j - 1] + levels[j]) / 2)


def check_statement(out, lam, counters, adversarial, tag, spec=None, secs=None, Hm=None):
    if out["status"] != "CERTIFIED":
        counters["T"].skipped += 1
        return
    K = out["K"]
    tol = TOL * (1 + float(abs(lam[-1])) + float(abs(lam[0])))
    ok = True
    for j in range(K):
        ok &= abs(lam[j] - mp.mpf(out["lowest"][j])) <= mp.mpf(out["radius"]) + tol
    if K < len(lam):
        ok &= lam[K] >= mp.mpf(out["next_lower_bound"]) - tol
    counters["T"].add(adversarial, ok, f"T {tag}: K={K} radius={out['radius']} eps={out['eps_decomposition']}")
    if secs is not None and Hm is not None and len(lam) <= 30:      # the mp steps cost O(n^3) at 60 digits
        # Steps 2-3 in mp: ||H W - W B|| <= D, ||H Qhat - Qhat B|| <= rho, sorted pairing within rho
        W = to_mp(sp.hstack([sp.csr_matrix(s.V) for s in secs]).toarray())
        B = to_mp(la.block_diag(*[s.H.toarray() for s in secs]))
        Hb = to_mp(hermitian_part(ce.build_bitwise(spec)).toarray())
        n = W.rows
        G = herm(W) * W
        eta = norm2(G - mp.eye(n))
        if not eta < 1:
            return
        Q = W * func_herm(G, lambda v: 1 / mp.sqrt(v))
        delta = 1 - mp.sqrt(1 - eta)
        Dex = norm2(Hb * W - W * B)
        rho_ex = Dex + (norm2(Hb) + norm2(B)) * delta
        Rhat = norm2(Hb * Q - Q * B)
        lamH, lamB = eigh(Hb)[0], eigh(B)[0]
        ok2 = (Dex <= mp.mpf(out["invariance_total"]) + tol and Rhat <= rho_ex + tol
               and bottleneck_sorted(lamH, lamB) <= Rhat + tol
               and mp.mpf(out["eta"]) >= eta and mp.mpf(out["eps_decomposition"]) >= rho_ex)
        # the K lowest of B are the pooled values within the sector radius
        ok2 &= all(abs(lamB[j] - mp.mpf(out["lowest"][j])) <= mp.mpf(out["max_sector_radius"]) + tol for j in range(K))
        counters["S"].add(adversarial, ok2, f"S {tag}: D={Dex} D_code={out['invariance_total']} rho={rho_ex} "
                                            f"eps_code={out['eps_decomposition']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    counters = {k: Counter() for k in ("T", "S", "N")}
    models = []
    for N in (4, 6):
        spec = ce.heisenberg_chain(N)
        H = hermitian_part(ce.build_bitwise(spec))
        lam, Hm = exact_spectrum(spec)
        models.append((f"heis{N}", spec, H, lam, Hm))
    spec1 = ce.spin1_heisenberg_chain(3)
    H1 = hermitian_part(ce.build_bitwise(spec1))
    lam1, Hm1 = exact_spectrum(spec1)
    models.append(("spin1_N3", spec1, H1, lam1, Hm1))
    # A/B: random cuts, exact decompositions
    for i in range(args.cases):
        name, spec, H, lam, Hm = models[i % len(models)]
        wf = [float(v) for v in lam]
        E_cut = gap_cut(wf, rng)
        if name.startswith("heis") and rng.random() < 0.5:
            secs = momentum_sectors(spec, H, rng)
        else:
            secs = u1_sectors(spec, H, rng)
        out = certify_low_energy(spec, secs, E_cut, H_full=H)
        check_statement(out, lam, counters, False, f"{name} cut={E_cut:.6g}", spec, secs, Hm)
    # adversarial: perturbed bases and blocks (statement must hold when CERTIFIED)
    for i in range(args.cases // 3):
        name, spec, H, lam, Hm = models[i % len(models)]
        wf = [float(v) for v in lam]
        E_cut = gap_cut(wf, rng)
        pv = float(rng.choice([1e-14, 1e-12, 3e-11]))        # the audit tolerance is 1e-10 on iso and inv
        ph = float(rng.choice([0.0, 1e-13]))
        secs = u1_sectors(spec, H, rng, perturb_V=pv, perturb_H=ph)
        out = certify_low_energy(spec, secs, E_cut, H_full=H)
        check_statement(out, lam, counters, True, f"{name} perturbed V {pv} H {ph}", spec, secs, Hm)
    # negative controls
    name, spec, H, lam, Hm = models[1]
    wf = [float(v) for v in lam]
    E_cut = gap_cut(wf, np.random.default_rng(0))
    secs = u1_sectors(spec, H, rng)
    out = certify_low_energy(spec, secs[:-1], E_cut, H_full=H)
    counters["N"].add(True, out["status"] != "CERTIFIED", "dropped sector")
    bad = [SectorResult(s.label, s.V, s.H, s.eigenvalues, s.eigenvectors) for s in secs]
    Hb = bad[3].H.tolil(copy=True)
    Hb[0, 0] += 1e-3
    bad[3] = SectorResult(bad[3].label, bad[3].V, Hb.tocsr(), bad[3].eigenvalues, bad[3].eigenvectors)
    out = certify_low_energy(spec, bad, E_cut, H_full=H)
    counters["N"].add(True, out["status"] != "CERTIFIED", "corrupted block")
    s3 = secs[3]
    below = np.nonzero(s3.eigenvalues < E_cut)[0]
    if below.size >= 1:
        keep = [t for t in range(s3.eigenvalues.size) if t != below[0]]
        miss = SectorResult(s3.label, s3.V, s3.H, s3.eigenvalues[keep], s3.eigenvectors[:, keep])
        out = certify_low_energy(spec, secs[:3] + [miss] + secs[4:], E_cut, H_full=H)
        counters["N"].add(True, out["status"] != "CERTIFIED", "omitted eigenvalue below the cut")
    if below.size >= 2:
        X = s3.eigenvectors.copy()
        X[:, below[1]] = X[:, below[0]]
        dup = SectorResult(s3.label, s3.V, s3.H, s3.eigenvalues, X)
        out = certify_low_energy(spec, secs[:3] + [dup] + secs[4:], E_cut, H_full=H)
        counters["N"].add(True, out["status"] != "CERTIFIED", "duplicated eigenvector")
    out = certify_low_energy(spec, secs, float(wf[3]), H_full=H)       # cut on an eigenvalue
    counters["N"].add(True, out["status"] != "CERTIFIED", "cut inside a claimed enclosure")
    ok = True
    for key, nm in (("T", "Theorem 8 statement against the exact spectrum"), ("S", "Steps 2-3 inequalities in mp"),
                    ("N", "negative controls not CERTIFIED")):
        ok &= counters[key].report(nm)
    print("RESULT:", "0 violations" if ok else "VIOLATIONS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
