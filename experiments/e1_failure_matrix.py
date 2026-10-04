"""E1: which verification practice detects which realistic ED failure?

Three practices are applied to the same producer output:
  P1 residual-only       the producer's own check: max_i ||H_p x_i - l_i x_i||/||x_i|| <= 1e-8,
                         computed with the producer's operator H_p
  P2 recomputation       an independent code (QuSpin, full space or U(1) block) is run on the
                         INTENDED model and its lowest eigenvalues are compared with the claim
                         (agreement to 1e-8 relative); this is the practice Referee 1 recommends
  P3 CERTIFY-ED          certify() / certify_low_energy() statements against the intended spec

Each failure is produced by a realistic mechanism (wrong input, wrong convention, solver
behaviour), not by editing numbers after the fact, except where noted.
"""

import time
import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

warnings.filterwarnings("ignore")
from harness import Run

import certify_ed as ce
from certify_ed.basis import magnetization_state_set, u1_state_set
from certify_ed.compose import SectorResult, certify_low_energy

from quspin.basis import spin_basis_1d, spinless_fermion_basis_1d
from quspin.operators import hamiltonian

N = 12
K = 4
RES_TOL = 1e-8
AGREE_TOL = 1e-8


def qs_H(static, basis, dtype=np.float64):
    return hamiltonian(static, [], basis=basis, dtype=dtype, check_herm=False,
                       check_symm=False, check_pcon=False)


def residual_only(Hp, lam, X):
    Hp = Hp.tocsr() if hasattr(Hp, "tocsr") else sp.csr_matrix(Hp)
    r = np.linalg.norm(Hp @ X - X * lam[None, :], axis=0) / np.linalg.norm(X, axis=0)
    return bool(np.max(r) <= RES_TOL * max(1.0, np.max(np.abs(lam)))), float(np.max(r))


_LAST_P2 = {"t": None}


def P2(static_intended, basis, lam):
    ok, diff, t = recompute(static_intended, basis, lam)
    _LAST_P2["t"] = t
    return ok


def recompute(static_intended, basis, lam, k=None):
    t = time.time()
    H = qs_H(static_intended, basis, np.complex128 if any(isinstance(c[0], complex) for _, cs in static_intended for c in cs) else np.float64)
    kk = k or len(lam)
    if basis.Ns <= 2000:
        w = np.sort(la.eigvalsh(H.toarray()))[:kk]
    else:
        v0 = np.random.default_rng(7).standard_normal(basis.Ns)   # reproducible start vector
        w = np.sort(H.eigsh(k=kk, which="SA", return_eigenvectors=False, tol=1e-13, v0=v0))
    lam = np.sort(np.asarray(lam))
    ok = w.size == lam.size and np.max(np.abs(w - lam)) <= AGREE_TOL * max(1.0, np.max(np.abs(w)))
    return bool(ok), float(np.max(np.abs(w[:lam.size] - lam[:w.size]))) if w.size and lam.size else None, time.time() - t


def lanczos_single_vector(A, m, v0, reorth=True):
    """Plain Lanczos from one start vector (returns Ritz pairs). In exact arithmetic it finds
    only ONE vector per degenerate eigenspace."""
    n = A.shape[0]
    Q = np.zeros((n, m))
    alpha, beta = np.zeros(m), np.zeros(m)
    q = v0 / np.linalg.norm(v0)
    qprev = np.zeros(n)
    b = 0.0
    for j in range(m):
        Q[:, j] = q
        w = A @ q - b * qprev
        alpha[j] = q @ w
        w -= alpha[j] * q
        if reorth:
            w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
        b = np.linalg.norm(w)
        beta[j] = b
        if b < 1e-14:
            m = j + 1
            break
        qprev, q = q, w / b
    T = np.diag(alpha[:m]) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
    th, S = la.eigh(T)
    return th, Q[:, :m] @ S


def k_at_gap(w, kmin):
    k = kmin
    while k < len(w) and w[k] - w[k - 1] < 1e-6:
        k += 1
    return k


def statuses(cert):
    return {s["id"]: s["status"] for s in cert.statements}


def main():
    with Run("e1_failure_matrix") as run:
        rows = []
        rng = np.random.default_rng(run.seed)
        spec = ce.heisenberg_chain(N)
        static = spec.to_quspin_static()
        b_u1 = spin_basis_1d(N, Nup=N // 2, pauli=0)
        H_true = qs_H(static, b_u1).tocsr()
        w_true, V_true = la.eigh(H_true.toarray())
        global K
        K = k_at_gap(w_true, 4)
        run.log(f"K = {K} (smallest k >= 4 not cutting a degenerate level of the Sz=0 block)")

        def add(case, mechanism, p1, p2, p3_status, p3_detail, localized, p2_time, p3_time, **extra):
            rec = dict(case=case, mechanism=mechanism, P1_residual_only=p1, P2_recomputation=p2,
                       P3_certify_ed=p3_status, P3_detail=p3_detail, P3_localizes=localized,
                       P2_seconds=p2_time, P3_seconds=p3_time)
            rec.update(extra)
            rows.append(rec)
            fmt = lambda v: "n/a   " if v is None else ("pass  " if v else "detect")
            run.log(f"{case:28s} P1={fmt(p1)}  P2={fmt(p2)}"
                    f"  P3={p3_status}  {p3_detail[:90]}")

        # ---------------------------------------------------------- F0 correct run (false alarms?)
        lam, X = w_true[:K], V_true[:, :K]
        t = time.time()
        claim, sp_ = ce.claim_from_quspin(qs_H(static, b_u1), b_u1, static, lam, X, kind="lowest")
        cert = ce.certify(claim, sp_)
        st = statuses(cert)
        p1 = residual_only(H_true, lam, X)[0]
        p2, _, t2 = recompute(static, b_u1, lam)
        add("F0 correct result", "reference run, no error", p1, p2,
            "all CERTIFIED" if all(v in ("CERTIFIED", "NOT_CHECKED") for v in st.values()) else str(st),
            "; ".join(f"{k}={v}" for k, v in st.items()), None, t2, time.time() - t)

        # ---------------------------------------------------------- F1 sign typo on one bond
        bad = [[op, [list(c) for c in cps]] for op, cps in static]
        bad[2][1][7][0] *= -1                       # zz on bond (7,8)
        Hb = qs_H(bad, b_u1)
        wb, Vb = la.eigh(Hb.toarray())
        t = time.time()
        claim, _ = ce.claim_from_quspin(Hb, b_u1, bad, wb[:K], Vb[:, :K], kind="lowest")
        cert = ce.certify(claim, spec)
        fit = cert.statements[0]["evidence"].get("diagnosis", {}).get("coupling_fit", {})
        top = fit.get("couplings", [{}])[0]
        add("F1 coupling sign typo", "static list: J -> -J on one zz bond",
            residual_only(Hb, wb[:K], Vb[:, :K])[0], P2(static, b_u1, wb[:K]),
            statuses(cert)["S1.model_fidelity"],
            f"fit: {top.get('op')} on {top.get('sites')} changed by {top.get('producer_minus_spec')}; "
            f"residual {fit.get('relative_residual', float('nan')):.1e}", True,
            _LAST_P2["t"], time.time() - t, fit_term=[top.get("op"), top.get("sites")],
            fit_delta=top.get("producer_minus_spec"), fit_relative_residual=fit.get("relative_residual"))

        # ---------------------------------------------------------- F2 convention factor
        # user writes spin-convention couplings but builds the basis with Pauli matrices
        b_p = spin_basis_1d(N, Nup=N // 2, pauli=1)
        Hc = qs_H(static, b_p)
        wc, Vc = la.eigh(Hc.toarray())
        t = time.time()
        claim, _ = ce.claim_from_quspin(Hc, b_p, static, wc[:K], Vc[:, :K], kind="lowest")
        cert = ce.certify(claim, spec)       # intended: spin convention
        diag = cert.statements[0]["evidence"].get("diagnosis", {})
        add("F2 Pauli/spin convention", "couplings for S=sigma/2 used with Pauli basis",
            residual_only(Hc, wc[:K], Vc[:, :K])[0], P2(static, b_u1, wc[:K]),
            statuses(cert)["S1.model_fidelity"],
            f"best global scale factor {diag.get('best_scale_factor', float('nan')):.3f}", True,
            _LAST_P2["t"], time.time() - t, scale_factor=diag.get("best_scale_factor"),
            relative_residual_after_scaling=diag.get("relative_residual_after_scaling"))

        # ---------------------------------------------------------- F3 site relabelling with a local field
        spec_f = ce.ModelSpec.from_static(N, spec.local, static + [["z", [[0.4, 0]]]])
        static_f = spec_f.to_quspin_static()
        rev = [[op, [[c[0]] + [N - 1 - s for s in c[1:]] for c in cps]] for op, cps in static_f]
        Hr = qs_H(rev, b_u1)
        wr, Vr = la.eigh(Hr.toarray())
        t = time.time()
        claim, _ = ce.claim_from_quspin(Hr, b_u1, rev, wr[:K], Vr[:, :K], kind="lowest")
        cert = ce.certify(claim, spec_f)
        fit = cert.statements[0]["evidence"].get("diagnosis", {}).get("coupling_fit", {})
        add("F3 site-order mismatch", "producer numbers sites right-to-left; field on site 0",
            residual_only(Hr, wr[:K], Vr[:, :K])[0], P2(static_f, b_u1, wr[:K]),
            statuses(cert)["S1.model_fidelity"],
            f"fit implicates {[ (c['op'], c['sites']) for c in fit.get('couplings', [])][:3]}; residual "
            f"{fit.get('relative_residual', float('nan')):.1e}", True,
            _LAST_P2["t"], time.time() - t, fit_relative_residual=fit.get("relative_residual"),
            fit_interpretation=fit.get("interpretation"),
            fit_terms=[(c["op"], c["sites"]) for c in fit.get("couplings", [])][:6])

        # ---------------------------------------------------------- F4 missing Jordan-Wigner sign at the PBC bond
        specF = ce.spinless_fermion_chain(N, t=1.0, V=1.0, boundary="periodic")
        bF = spinless_fermion_basis_1d(N, Nf=N // 2)
        HF = qs_H(specF.to_quspin_static(), bF).tocsr()
        # producer treats the wrap-around hopping bosonically (hard-core bosons): no string sign
        specB = ce.ModelSpec.from_static(N, ce.LocalSpace.spin_space("1/2", "pauli_pm1"),
                                         [["+-", [[-1.0, i, (i + 1) % N] for i in range(N)]],
                                          ["-+", [[-1.0, i, (i + 1) % N] for i in range(N)]],
                                          ["zz", [[0.25, i, (i + 1) % N] for i in range(N)]],
                                          ["z", [[0.5, i] for i in range(N)]]])
        # map spin language to fermion occupations: n = (1+sigma^z)/2 -> nn = (1 + z_i + z_j + z_i z_j)/4
        stF = u1_state_set(specF, N // 2)
        Hbos = ce.build_bitwise(specB, stF)
        Hbos = (Hbos + sp.identity(stF.size) * (N * 0.25)).tocsr()   # constant V*N/4 from n_i n_j
        wF, VF = la.eigh(Hbos.toarray())
        t = time.time()
        claim = ce.Claim(wF[:K], VF[:, :K], operator=Hbos, kind="lowest", states=stF)
        cert = ce.certify(claim, specF)
        fit = cert.statements[0]["evidence"].get("diagnosis", {}).get("coupling_fit", {})
        add("F4 missing JW sign (PBC)", "wrap-around hopping built without the fermionic string",
            residual_only(Hbos, wF[:K], VF[:, :K])[0], P2(specF.to_quspin_static(), bF, wF[:K]),
            statuses(cert)["S1.model_fidelity"],
            f"fit: {[ (c['op'], c['sites']) for c in fit.get('couplings', [])][:2]}, residual "
            f"{fit.get('relative_residual', float('nan')):.1e}", True, _LAST_P2["t"], time.time() - t,
            fit_relative_residual=fit.get("relative_residual"),
            fit_terms=[(c["op"], c["sites"]) for c in fit.get("couplings", [])][:6])

        # ---------------------------------------------------------- F5 sector result reported as global
        # the producer diagonalizes the Sz=0 block and reports its lowest states as "the K lowest
        # states of the model"; the eigenvectors are embedded into the full space
        specP = ce.heisenberg_chain(N)
        Hfull = ce.build_bitwise(specP)
        stS = magnetization_state_set(specP, 0)
        full_idx, _ = ce.full_state_set(specP).index_of(stS.states_in_order())
        E = sp.csr_matrix((np.ones(stS.size), (full_idx, np.arange(stS.size))), shape=(specP.dim, stS.size))
        lam5, X5 = w_true[:K], np.asarray(E @ V_true[:, :K])
        t = time.time()
        cert = ce.certify(ce.Claim(lam5, X5, operator=Hfull, kind="lowest"), specP)
        s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]
        b_full = spin_basis_1d(N, pauli=0)
        add("F5 sector result as global", "Sz=0 block's lowest states reported as the model's lowest states",
            residual_only(Hfull, lam5, X5)[0], P2(static, b_full, lam5),
            s3["status"], s3["summary"][:160], True, _LAST_P2["t"], time.time() - t)

        # ---------------------------------------------------------- F6 momentum-sector result reported for the U(1) block
        from certify_ed.basis import translation, cyclic_isometry
        Hs0 = ce.build_bitwise(specP, stS)
        V0 = cyclic_isometry(stS, translation(specP), 0)
        Hk0 = (V0.conj().T @ Hs0 @ V0).tocsr()
        wk, Yk = la.eigh(Hk0.toarray())
        Kk = k_at_gap(wk, 3)
        lam6, X6 = wk[:Kk], np.asarray(V0 @ Yk[:, :Kk])
        t = time.time()
        cert = ce.certify(ce.Claim(lam6, X6, operator=Hs0, kind="lowest", states=stS), specP)
        s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]
        add("F6 k=0 result as Sz=0 result", "lowest k=0 states reported as the lowest states of the Sz=0 block",
            residual_only(Hs0, lam6, X6)[0], P2(static, b_u1, lam6),
            s3["status"], s3["summary"][:160], True, _LAST_P2["t"], time.time() - t)

        # ---------------------------------------------------------- F7 ghost eigenvalues (no reorthogonalisation)
        th, Y = lanczos_single_vector(Hs0, 400, rng.standard_normal(Hs0.shape[0]), reorth=False)
        order = np.argsort(th)
        lam7, X7 = th[order[:K]], Y[:, order[:K]]
        dup = int(np.sum(np.diff(np.sort(lam7)) < 1e-9))
        t = time.time()
        cert = ce.certify(ce.Claim(lam7, X7, operator=Hs0, kind="lowest", states=stS), specP)
        st7 = statuses(cert)
        add("F7 Lanczos ghosts", "400 Lanczos steps without reorthogonalisation",
            residual_only(Hs0, lam7, X7)[0], P2(static, b_u1, lam7),
            f"S2={st7['S2.eigenpairs']}, S3={st7['S3.completeness']}",
            f"{dup} numerically duplicated Ritz value(s) among the {K} reported", True, _LAST_P2["t"],
            time.time() - t, duplicates=dup)

        # ---------------------------------------------------------- F8 under-converged result reported as accurate
        w8, X8 = spla.eigsh(H_true, k=K, which="SA", tol=1e-4, v0=np.ones(H_true.shape[0]))
        t = time.time()
        cert = ce.certify(ce.Claim(w8, X8, operator=H_true, kind="lowest", accuracy_target=1e-10,
                                   states=stS), specP)
        s2 = [s for s in cert.statements if s["id"] == "S2.eigenpairs"][0]
        add("F8 loose convergence", "eigsh tol=1e-4 reported with 10 digits (target 1e-10)",
            residual_only(H_true, w8, X8)[0], P2(static, b_u1, w8),
            s2["status"], s2["summary"][:140], False, _LAST_P2["t"], time.time() - t)

        # ---------------------------------------------------------- F9 eigenvector/eigenvalue mislabel
        lam9 = w_true[:K].copy()
        X9 = V_true[:, [1, 0] + list(range(2, K))]
        t = time.time()
        cert = ce.certify(ce.Claim(lam9, X9, operator=H_true, kind="lowest", states=stS), specP)
        s2 = [s for s in cert.statements if s["id"] == "S2.eigenpairs"][0]
        add("F9 swapped eigenvectors", "vectors of levels 0 and 1 interchanged (post-processing)",
            residual_only(H_true, lam9, X9)[0], P2(static, b_u1, lam9),
            s2["status"], f"max certified radius {s2['evidence']['max_radius']:.2e}", False, _LAST_P2["t"],
            time.time() - t)

        # ---------------------------------------------------------- F10/F11 sector workflows (pooled spectrum)
        Ns = 10
        specS = ce.heisenberg_chain(Ns)
        staticS = specS.to_quspin_static()
        wfull = np.sort(la.eigvalsh(ce.build_bitwise(specS).toarray()))
        gaps = np.where(np.diff(wfull[:30]) > 1e-3)[0]
        E_cut = float((wfull[gaps[3]] + wfull[gaps[3] + 1]) / 2)

        # locate the sector holding the ground state
        gs_sec, gs_e = None, np.inf
        for nup in range(Ns + 1):
            for k in range(Ns):
                b = spin_basis_1d(Ns, Nup=nup, kblock=k, pauli=0)
                if b.Ns == 0:
                    continue
                e0 = la.eigvalsh(qs_H(staticS, b, np.complex128).toarray())[0]
                if e0 < gs_e - 1e-9:
                    gs_sec, gs_e = (nup, k), e0
        target = {"phase_gs": gs_sec, "drop_gs": gs_sec, "phase": (2, 0), "drop": (0, 0)}
        run.log(f"ground-state sector (Nup,k) = {gs_sec}")

        def sector_run(bug):
            secs, pooled = [], []
            for nup in range(Ns + 1):
                for k in range(Ns):
                    if bug in ("drop", "drop_gs") and (nup, k) == target[bug]:
                        continue
                    b = spin_basis_1d(Ns, Nup=nup, kblock=k, pauli=0)
                    if b.Ns == 0:
                        continue
                    V = sp.csr_matrix(b.get_proj(np.complex128))
                    Hs = qs_H(staticS, b, np.complex128).tocsr()
                    if bug in ("phase", "phase_gs") and (nup, k) == target[bug]:
                        V = V.tolil()
                        r0 = V[:, 0].nonzero()[0][0]
                        V[r0, 0] *= -1
                        V = V.tocsr()
                        Hs = (V.conj().T @ ce.build_bitwise(specS) @ V).tocsr()
                    w, Yv = la.eigh(Hs.toarray())
                    sel = np.where(w < E_cut)[0]
                    pooled += w[sel].tolist()
                    secs.append(SectorResult(f"Nup={nup},k={k}", V, Hs, w[sel], Yv[:, sel]))
            return secs, np.sort(pooled)

        for bug, name, mech in [
                ("phase_gs", "F10a wrong phase, GS sector", "sign error in one momentum-basis vector of the ground-state sector"),
                ("phase", "F10b wrong phase, high sector", "same bug in the (Nup=2, k=0) block (no states below the cut)"),
                ("drop_gs", "F11a missing GS sector", "ground-state (Nup,k) block omitted from the pooled list"),
                ("drop", "F11b missing high sector", "fully polarised block omitted (no states below the cut)")]:
            secs, pooled = sector_run(bug)
            ok_res = all(residual_only(s.H, s.eigenvalues, s.eigenvectors)[0] for s in secs if len(s.eigenvalues))
            t = time.time()
            res = certify_low_energy(specS, secs, E_cut)
            b_all = spin_basis_1d(Ns, pauli=0)
            p2, _, t2 = recompute(staticS, b_all, pooled[:10])
            add(name, mech, ok_res, p2, res["status"], res["summary"][:140], True, t2, time.time() - t)

        # ---------------------------------------------------------- F12 basis-dependent multiplet observable
        j0 = int(np.argmax(np.diff(w_true) < 1e-9))
        lev = np.where(np.abs(w_true - w_true[j0]) < 1e-9)[0]
        kk = int(lev[-1]) + 1
        states = stS.states_in_order()
        Sz0 = np.array([((s >> (N - 1)) & 1) - 0.5 for s in states])
        O = sp.diags(Sz0 * np.array([((s >> (N - 2)) & 1) - 0.5 for s in states]))
        t = time.time()
        cert = ce.certify(ce.Claim(w_true[:kk], V_true[:, :kk], operator=H_true, kind="lowest", states=stS),
                          specP, ce.Options(observables={"Sz0Sz1": O}))
        cl = [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]["evidence"].get("clusters", [])
        info = cl[0]["observables"]["Sz0Sz1"] if cl else {}
        add("F12 per-vector multiplet data", "<psi|Sz0 Sz1|psi> quoted for one vector of a degenerate level",
            True, None, [s for s in cert.statements if s["id"] == "S4.eigenspaces"][0]["status"],
            (f"certified tr(P O)={info.get('value', float('nan')):.12f} +- {info.get('bound', float('nan')):.1e}; "
             f"per-vector values span {info.get('per_vector_spread_under_rotation', {}).get('spread', float('nan')):.3f} "
             f"across equally valid bases") if info else "n/a", True, None, time.time() - t,
            multiplicity=len(lev), value=info.get("value"), bound=info.get("bound"),
            spread=info.get("per_vector_spread_under_rotation", {}).get("spread"))

        run.record("rows", rows)
        run.record("settings", {"N": N, "K": K, "residual_tolerance": RES_TOL,
                                "agreement_tolerance": AGREE_TOL})


if __name__ == "__main__":
    main()
