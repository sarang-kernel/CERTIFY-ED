"""E3: how tight are the certified statements?

(a) Certified radius (Theorem 1), Kato-Temple interval (Theorem 3) and the ACTUAL error of
    the claimed eigenvalue, for eigsh results of decreasing tolerance.
(b) float64 residual bounds (standard floating-point model) vs interval-arithmetic (Arb)
    residuals computed without any model assumption.
(c) certified lower bound on the next eigenvalue vs the true next eigenvalue.
(d) sector-composition radius vs actual deviation of pooled sector values from the full spectrum.
"""

import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

warnings.filterwarnings("ignore")
from harness import Run

import certify_ed as ce
from certify_ed.basis import magnetization_state_set
from certify_ed.enclosure import OperatorAdapter, enclose, kato_temple
from certify_ed.rigorous import arb_residual_radii


def main():
    with Run("e3_tightness") as run:
        N = 12
        spec = ce.heisenberg_chain(N, J2=0.2)
        st = magnetization_state_set(spec, 0)
        H = ce.build_bitwise(spec, st)
        w_ref, V_ref = la.eigh(H.toarray())
        gap = float(w_ref[1] - w_ref[0])
        run.log(f"d={H.shape[0]}, E0={w_ref[0]:.12f}, gap={gap:.6f}")
        ad = OperatorAdapter(H)

        # (a) radius / Kato-Temple / actual error vs solver tolerance
        rows_a = []
        rng = np.random.default_rng(run.seed)
        v0 = rng.standard_normal(H.shape[0])
        for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0]:
            w, V = spla.eigsh(H, k=1, which="SA", tol=tol, v0=v0, maxiter=10 ** 6)
            x = V[:, 0] / np.linalg.norm(V[:, 0])
            theta = float(x @ (H @ x))
            E = enclose(ad, np.array([theta]), x[:, None])
            kt = kato_temple(ad, x, -1e3, float(w_ref[1]) - 1e-9 * abs(w_ref[1]))
            # the Kato-Temple separator beta must itself be certified: here w_ref[1] (dense LAPACK,
            # certified separately below by inertia); for the figure we use it as the reference
            rows_a.append({"tol": tol, "claimed": theta, "actual_error": abs(theta - w_ref[0]),
                           "radius": float(E.radius[0]),
                           "residual_computed": float(E.residual.computed[0]),
                           "kt_lower": kt[0] if kt else None, "kt_upper": kt[1] if kt else None,
                           "kt_halfwidth": (kt[1] - kt[0]) / 2 if kt else None,
                           "kt_contains": bool(kt[0] <= w_ref[0] <= kt[1]) if kt else None,
                           "radius_contains": bool(abs(theta - w_ref[0]) <= E.radius[0])})
            run.log(f"tol={tol:7.0e} err={abs(theta - w_ref[0]):.2e} radius={E.radius[0]:.2e} "
                    f"KT half-width={rows_a[-1]['kt_halfwidth'] if kt else float('nan'):.2e}")
        run.record("a_tolerance_sweep", rows_a)
        # certify the separator used above
        r = ce.inertia_count(H, float(w_ref[1]) - 1e-9 * abs(w_ref[1]), method="dense")
        run.record("a_separator_check", {"sigma": r.sigma, "negative": r.negative, "beta": r.beta,
                                         "note": "exactly one eigenvalue below the Kato-Temple separator"})

        # (b) float64 bound vs Arb residuals for converged eigenpairs
        k = 6
        X = V_ref[:, :k]
        E = enclose(ad, w_ref[:k], X)
        arb = arb_residual_radii(H, w_ref[:k], X)
        comp = E.residual.computed / np.linalg.norm(X, axis=0)
        run.record("b_float_vs_arb", {"float64_bound": E.residual.rel_bound.tolist(), "arb_exact": arb.tolist(),
                                      "computed_float": comp.tolist(),
                                      "ratio_bound_over_arb": (E.residual.rel_bound / arb).tolist()})
        run.log(f"float64 bound / Arb residual: min {np.min(E.residual.rel_bound / arb):.1f}, "
                f"max {np.max(E.residual.rel_bound / arb):.1f}")

        # (c) certified next-eigenvalue lower bound vs truth, several claim sizes and models
        rows_c = []
        for mdl, sp_ in [("heisenberg J2=0.2 N=12 Sz=0", spec),
                         ("tfim h=1 N=12", ce.tfim_chain(12, h=1.0)),
                         ("fermion t-V N=12 nf=6", ce.spinless_fermion_chain(12, V=1.5, boundary="periodic"))]:
            if sp_.local.kind == "fermion":
                sts = ce.u1_state_set(sp_, 6)
            elif sp_.name == "tfim_chain":
                sts = ce.full_state_set(sp_)
            else:
                sts = st
            Hm = ce.build_bitwise(sp_, sts)
            wm, Vm = la.eigh(Hm.toarray())
            for kk in (1, 2, 4, 8):
                while kk < len(wm) and wm[kk] - wm[kk - 1] < 1e-8:
                    kk += 1
                c = ce.certify(ce.Claim(wm[:kk], Vm[:, :kk], operator=Hm, kind="lowest", states=sts), sp_,
                               ce.Options(inertia_method="sparse"))
                s3 = [s for s in c.statements if s["id"] == "S3.completeness"][0]
                lb = s3["evidence"].get("next_eigenvalue_lower_bound")
                rows_c.append({"model": mdl, "k": kk, "status": s3["status"], "certified_lb": lb,
                               "true_next": float(wm[kk]), "true_gap": float(wm[kk] - wm[kk - 1]),
                               "certified_gap": s3["evidence"].get("gap_lower_bound"),
                               "fraction_of_gap_certified": (s3["evidence"].get("gap_lower_bound") or 0) / float(wm[kk] - wm[kk - 1]),
                               "n_factorizations": len(s3["evidence"].get("counts", [])),
                               "seconds": s3["seconds"]})
                run.log(f"{mdl:32s} k={kk:2d} {s3['status']} certified gap/true gap = "
                        f"{rows_c[-1]['fraction_of_gap_certified']:.3f} ({rows_c[-1]['n_factorizations']} factorizations)")
        run.record("c_gap", rows_c)

        # (d) global spectrum assembled from all (Nup, k) blocks vs the full-space spectrum
        from quspin.basis import spin_basis_1d
        from quspin.operators import hamiltonian
        from certify_ed.compose import SectorResult, certify_low_energy
        Nd = 12
        specd = ce.heisenberg_chain(Nd, J2=0.3)
        static = specd.to_quspin_static()
        Hfull = ce.build_bitwise(specd)
        wfull = la.eigvalsh(Hfull.toarray())
        jumps = np.where(np.diff(wfull[:40]) > 1e-4)[0]
        E_cut = float((wfull[jumps[5]] + wfull[jumps[5] + 1]) / 2)
        secs = []
        for nup in range(Nd + 1):
            for kb in range(Nd):
                b = spin_basis_1d(Nd, Nup=nup, kblock=kb, pauli=0)
                if b.Ns == 0:
                    continue
                Hs = hamiltonian(static, [], basis=b, dtype=np.complex128, check_herm=False,
                                 check_symm=False, check_pcon=False).tocsr()
                w, Y = la.eigh(Hs.toarray())
                sel = np.where(w < E_cut)[0]
                secs.append(SectorResult(f"{nup},{kb}", sp.csr_matrix(b.get_proj(np.complex128)), Hs, w[sel],
                                         Y[:, sel]))
        res = certify_low_energy(specd, secs, E_cut)
        K = res.get("K") or 0
        act = float(np.max(np.abs(np.array(res["lowest"]) - wfull[:K]))) if K else None
        run.record("d_compose", {"N": Nd, "J2": 0.3, "n_sectors": len(secs), "E_cut": E_cut, "status": res["status"],
                                 "K": K, "radius": res.get("radius"), "actual_max_error": act,
                                 "eta": res["eta"], "invariance_total": res["invariance_total"],
                                 "next_lower_bound": res.get("next_lower_bound"), "true_next": float(wfull[K]),
                                 "summary": res["summary"]})
        run.log(f"compose N={Nd}: {res['summary']}; actual max error {act:.2e}; true next {wfull[K]:.10f}")


if __name__ == "__main__":
    main()
