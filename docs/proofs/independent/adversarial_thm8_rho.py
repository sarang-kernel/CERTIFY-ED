"""Independent adversarial search for Theorem 8 (docs/proofs/thm8_rho.md).

Part A (mpmath, 60 digits): the spectral core, |lambda_j(H) - lambda_j(B)| <= rho = D + (||H|| + ||B||) delta for
a square W with eta = ||W^H W - I|| < 1 and D = ||HW - WB||_F, at the hypothesis boundaries: eta -> 1, one-dimensional
sectors (W a perturbed unitary), repeated eigenvalues, complex entries, scales 1e+-150, B far from the compression.
Part B (exact rationals): the pooling argument (Steps 2-3): sector spectra, claims with radii <= r, cut (T3);
conclusion checked exactly, (T3) broken as a necessity control.
Part C (implementation): compose.certify_low_energy on Heisenberg chains with U(1) sectors; CERTIFIED outcomes are
checked against mpmath eigenvalues of the exact spec matrix; negative controls (non-orthonormal basis, missing
sector, missing claimed value, cut too close) must not be CERTIFIED.

Usage: python docs/proofs/independent/adversarial_thm8_rho.py [--cases 150] [--seed 13]
"""
import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, eig_herm, fro_norm, herm, mpf, report, spec_norm, to_mp  # noqa: E402
import certify_ed as ce  # noqa: E402
from certify_ed.checker import exact_matrix  # noqa: E402
from certify_ed.compose import SectorResult, certify_low_energy  # noqa: E402
from certify_ed.construct import build_bitwise, full_state_set, hermitian_part  # noqa: E402

SLACK = mp.mpf("1e-45")


def part_a(rng, cases, T):
    for i in range(cases):
        cplx = i % 3 == 0
        kind = i % 6
        n = int(rng.integers(2, 7))
        scale = {0: 1.0, 1: 1e150, 2: 1e-150}[i % 3] if kind == 4 else 1.0
        M = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)
        H = (M + M.conj().T) / 2 * scale
        if kind == 2:   # repeated eigenvalues
            Q = np.linalg.qr(M)[0]
            H = Q @ np.diag(rng.choice([-1.0, 0.0, 2.0], size=n)) @ Q.conj().T
        # sector dimensions
        if kind == 1:
            dims = [1] * n
        else:
            dims, left = [], n
            while left:
                d = int(rng.integers(1, left + 1)); dims.append(d); left -= d
        Q = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0))[0]
        pert = {0: 1e-3, 1: 0.3, 2: 1e-8, 3: 0.0, 4: 1e-6, 5: 0.49}[kind]
        W = Q + pert * (rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0.0)) / np.sqrt(n)
        if kind == 5:   # eta -> 1: stretch one column
            W = Q.copy(); W[:, 0] *= np.sqrt(2.0 - 2.0 ** -int(rng.integers(2, 30)))
        Wm, Hm = to_mp(W), to_mp(H)
        eta = spec_norm(herm(Wm) * Wm - mp.eye(n))
        if not eta < 1:
            T.skip(); continue
        # blocks: Hermitian part of the compression, perturbed for kind 3 (B far from the compression)
        _blocks, off = [], 0
        Bm = mp.zeros(n, n)
        for d in dims:
            Vs = Wm[:, off:off + d]
            Hs = herm(Vs) * Hm * Vs
            Hs = (Hs + herm(Hs)) / 2
            if kind == 3:
                Pm = to_mp(rng.standard_normal((d, d)) * float(rng.choice([0.1, 1.0, 10.0])))
                Hs = Hs + (Pm + herm(Pm)) / 2
            for a in range(d):
                for b in range(d):
                    Bm[off + a, off + b] = Hs[a, b]
            off += d
        D = fro_norm(Hm * Wm - Wm * Bm)
        delta = 1 - mp.sqrt(1 - eta)
        rho = D + (spec_norm(Hm) + spec_norm(Bm)) * delta
        lamH, _ = eig_herm(Hm); lamB, _ = eig_herm(Bm)
        worst = max(abs(x - y) for x, y in zip(lamH, lamB))
        T.add(worst <= rho + SLACK * (1 + spec_norm(Hm)), kind != 0,
              f"kind={kind} n={n} eta={mp.nstr(eta, 8)} worst={mp.nstr(worst, 12)} rho={mp.nstr(rho, 12)}")


def part_b(rng, cases, T, T_need):
    failed_without_T3, tried = 0, 0
    done = 0
    while done < cases:
        m = int(rng.integers(1, 5))
        spectra = [sorted(Fraction(int(v), 16) for v in rng.integers(-64, 64, size=int(rng.integers(1, 6)))) for _ in range(m)]
        r = Fraction(int(rng.integers(0, 4)), 64)
        E_cut = Fraction(int(rng.integers(-64, 64)), 16) + Fraction(1, 32)
        betas = [Fraction(int(rng.integers(0, 3)), 128) for _ in range(m)]
        claims, S, ok_T2 = [], [], True
        for s, lam in enumerate(spectra):
            ks = sum(1 for v in lam if v < E_cut - betas[s])        # Corollary 5: nu_s = k_s claimed below the cut
            # claimed values within r of the k_s lowest, all enclosures below E_cut - beta_s
            cs = [v + Fraction(int(rng.integers(-64, 65)), 64) * r for v in lam[:ks]]
            if any(c + r >= E_cut - betas[s] for c in cs):
                ok_T2 = False
            claims += cs; S += lam[:ks]
        if not ok_T2 or not claims:
            continue
        E_star = E_cut - max(betas)
        T3 = max(c + r for c in claims) < E_star
        allB = sorted(v for lam in spectra for v in lam)
        K = len(claims)
        concl = (all(abs(allB[j] - c) <= r for j, c in enumerate(sorted(claims)))
                 and (K == len(allB) or allB[K] >= E_star))
        if T3:
            done += 1
            T.add(concl, r == 0 or max(betas) > 0, f"spectra={[[str(v) for v in l] for l in spectra]} claims={[str(c) for c in claims]} r={r} E_cut={E_cut}")
        else:
            tried += 1
            if not concl:
                failed_without_T3 += 1
    # deterministic control: sector 1 (beta = 0) claims an eigenvalue at E_cut - 1/16; sector 2 (beta = 1/4) has an
    # unclaimed eigenvalue at E_cut - 1/8 >= E_cut - beta_2, so (T2) holds for both sectors, (T3) fails
    # (E_cut - 1/16 + r >= E_cut - 1/4), and the pooled "K lowest" statement is false
    E_cut = Fraction(0); spectra = [[Fraction(-1), E_cut - Fraction(1, 16)], [Fraction(-2), E_cut - Fraction(1, 8), Fraction(3)]]
    betas = [Fraction(0), Fraction(1, 4)]; r = Fraction(0)
    claims = [Fraction(-1), E_cut - Fraction(1, 16), Fraction(-2)]
    ks = [sum(1 for v in lam if v < E_cut - b) for lam, b in zip(spectra, betas)]
    assert ks == [2, 1] and all(c_ + r < E_cut - b for lam, b, k in zip(spectra, betas, ks) for c_ in lam[:k])
    T3 = max(c_ + r for c_ in claims) < E_cut - max(betas)
    allB = sorted(v for lam in spectra for v in lam)
    concl = all(abs(allB[j] - c_) <= r for j, c_ in enumerate(sorted(claims)))
    tried += 1
    if not T3 and not concl:
        failed_without_T3 += 1
    T_need.add(failed_without_T3 > 0, True, f"(T3) broken: conclusion never failed in {tried} cases")
    return failed_without_T3, tried


def u1_sectors(spec, H, rng, perturb_V=0.0):
    full = full_state_set(spec)
    secs = []
    for nup in range(spec.n_sites + 1):
        st = ce.u1_state_set(spec, nup)
        idx, _ = full.index_of(st.states_in_order())
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        if perturb_V:
            Vs = sp.csr_matrix(Vs.toarray() + perturb_V * rng.standard_normal(Vs.shape))
        Hs = (Vs.T @ H @ Vs).toarray(); Hs = (Hs + Hs.T) / 2
        w, V = np.linalg.eigh(Hs)
        secs.append(SectorResult(f"nup={nup}", Vs, sp.csr_matrix(Hs), w, V))
    return secs


def exact_mp(spec):
    """The declared model as an exact mp matrix (spin-1/2: all weights 1, K = H = (re + i im)/den)."""
    X = exact_matrix(spec)
    assert all(int(w) == 1 for w in X.weights)
    n = len(X.weights)
    M = mp.zeros(n, n)
    den = mp.mpf(int(X.den))
    for r_, c_, re_, im_ in zip(X.rows, X.cols, X.re, X.im):
        M[int(r_), int(c_)] += mp.mpc(mp.mpf(int(re_)) / den, mp.mpf(int(im_)) / den)
    return M


def part_c(rng, cases, T, T_neg):
    for _i in range(cases):
        N = int(rng.choice([4, 5, 6]))
        spec = ce.heisenberg_chain(N, J=1.0, Delta=float(rng.choice([1.0, 0.5, 1.7])))
        H = hermitian_part(build_bitwise(spec))
        secs = u1_sectors(spec, H, rng)
        wf = np.sort(np.concatenate([s.eigenvalues for s in secs]))
        gaps = np.diff(wf)
        j = int(np.argmax(gaps[: max(1, len(gaps) // 2)]))
        E_cut = float((wf[j] + wf[j + 1]) / 2)
        out = certify_low_energy(spec, secs, E_cut, H_full=H)
        def check(out, tag, spec=spec, N=N):
            if out["status"] != "CERTIFIED":
                T.skip(); return
            lam, _ = eig_herm(exact_mp(spec))
            K = out["K"]
            ok = all(abs(lam[k] - mpf(cv)) <= mpf(out["radius"]) for k, cv in enumerate(out["lowest"]))
            ok = ok and (K == len(lam) or lam[K] >= mpf(out["next_lower_bound"]))
            T.add(ok, tag != "gap cut", f"{tag} N={N} K={K} radius={out['radius']} next={out['next_lower_bound']}")
        check(out, "gap cut")
        # the cut exactly on an eigenvalue: CERTIFIED is legitimate (lambda_{K+1} = E_cut >= E_cut - beta - eps);
        # whichever status results, a CERTIFIED conclusion must hold
        check(certify_low_energy(spec, secs, float(wf[j + 1]), H_full=H), "cut on eigenvalue")
        # negative controls
        bad = u1_sectors(spec, H, rng, perturb_V=0.5)
        T_neg.add(certify_low_energy(spec, bad, E_cut, H_full=H)["status"] != "CERTIFIED", True, f"N={N} perturbed V certified")
        T_neg.add(certify_low_energy(spec, secs[:-1], E_cut, H_full=H)["status"] != "CERTIFIED", True, f"N={N} missing sector certified")
        s0 = max(range(len(secs)), key=lambda q: np.sum(secs[q].eigenvalues < E_cut))
        s = secs[s0]; keep = np.arange(s.eigenvalues.size) != 0
        miss = SectorResult(s.label, s.V, s.H, s.eigenvalues[keep], s.eigenvectors[:, keep])
        T_neg.add(certify_low_energy(spec, secs[:s0] + [miss] + secs[s0 + 1:], E_cut, H_full=H)["status"] != "CERTIFIED", True,
                  f"N={N} missing lowest value certified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_a = Tally("spectral core |lambda_j(H) - lambda_j(B)| <= rho (mp)")
    T_b = Tally("pooling argument, exact rationals")
    T_need = Tally("necessity of (T3)")
    T_c = Tally("certify_low_energy CERTIFIED outcomes vs mp eigenvalues of the exact spec matrix")
    T_neg = Tally("negative controls not CERTIFIED")
    part_a(rng, a.cases * 2, T_a)
    f, t = part_b(rng, a.cases * 3, T_b, T_need)
    part_c(rng, max(3, a.cases // 25), T_c, T_neg)
    print(f"necessity: without (T3) the conclusion failed in {f} of {t} cases")
    sys.exit(1 if report([T_a, T_b, T_need, T_c, T_neg]) else 0)


if __name__ == "__main__":
    main()
