"""E5: an error that nobody planted.

While measuring cost (E2) we observed that ARPACK, asked through QuSpin for the eight lowest
states of the 4x4 Heisenberg S^z=0 block (default ncv, tol=0, random start vector), sometimes
returns only three of the four members of the fourfold level at E ~ -8.886, followed by two
members of the next level. The returned pairs are accurate eigenpairs; the list is not the
eight lowest eigenvalues.

For S seeded start vectors this script records
  truth : the eight lowest eigenvalues from a dense LAPACK solution of the same block
  P1    : residual check of the eight returned pairs (max relative residual <= 1e-8)
  P2    : a second ARPACK run from another random start, eight lowest values agree to 1e-8
  P3    : CERTIFY-ED on the claim "these are the eight lowest eigenvalues"
  P3k   : CERTIFY-ED on the claim a careful user would make: the lowest k states with k chosen at
          the first gap > 1e-6 in the returned values at or after the fifth (as in E2)
"""

import sys
import time
import warnings

import numpy as np
import scipy.linalg as la

warnings.filterwarnings("ignore")
from harness import Run

import certify_ed as ce
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
K = 8
TOL = 1e-8


def main():
    with Run("e5_arpack_multiplet") as run:
        spec = ce.heisenberg_square(4, 4)
        basis = spin_basis_general(16, Nup=8, pauli=0)
        Hq = hamiltonian(spec.to_quspin_static(), [], basis=basis, dtype=np.float64, check_herm=False,
                         check_symm=False, check_pcon=False)
        H = Hq.tocsr()
        d = H.shape[0]
        states = ce.StateSet(np.sort(np.asarray(basis.states, dtype=np.int64)), "descending", "U(1)")
        t0 = time.time()
        M = H.toarray()
        w_true = la.eigvalsh(M, overwrite_a=True, check_finite=False, subset_by_index=[0, 40])
        del M
        t_dense = time.time() - t0
        levels = []
        for x in w_true:
            if levels and abs(x - levels[-1][0]) < 1e-8:
                levels[-1][1] += 1
            else:
                levels.append([float(x), 1])
        run.record("setup", {"dim": d, "nnz": int(H.nnz), "dense_seconds": t_dense,
                             "true_lowest": w_true[:12].tolist(),
                             "levels": levels[:6]})
        run.log(f"d={d}; dense reference {t_dense:.0f}s; levels (value, multiplicity): "
                + ", ".join(f"({v:.8f}, {m})" for v, m in levels[:6]))
        scale = max(1.0, float(np.max(np.abs(w_true[:K]))))
        rows = []
        for s in range(N_SEEDS):
            v0 = np.random.default_rng(run.seed + s).standard_normal(d)
            t0 = time.time()
            w, V = Hq.eigsh(k=K, which="SA", tol=0, v0=v0)
            t_solve = time.time() - t0
            o = np.argsort(w)
            w, V = w[o], V[:, o]
            correct = bool(np.max(np.abs(w - w_true[:K])) <= TOL * scale)
            mult4 = int(np.sum(np.abs(w - levels[3][0]) < 1e-8))
            # P1: residual check with the producer's own matrix
            R = H @ V - V * w
            p1_res = float(np.max(np.linalg.norm(R, axis=0) / np.linalg.norm(V, axis=0)))
            p1_flag = p1_res > TOL * scale
            # P2: a second ARPACK run from another start vector
            v0b = np.random.default_rng(run.seed + 1000 + s).standard_normal(d)
            w2 = np.sort(Hq.eigsh(k=K, which="SA", tol=0, v0=v0b, return_eigenvectors=False))
            p2_diff = float(np.max(np.abs(w2 - w)))
            p2_flag = p2_diff > TOL * scale
            # P3: the claim "these are the K lowest"
            t0 = time.time()
            c = ce.certify(ce.Claim(w, V, operator=H, kind="lowest", states=states), spec,
                           ce.Options(inertia_method="sparse", inertia_max_dim=10 ** 6))
            t_cert = time.time() - t0
            st = {x["id"]: x["status"] for x in c.statements}
            s3 = [x for x in c.statements if x["id"] == "S3.completeness"][0]
            # P3k: claim up to the first gap in the returned values
            k = 4
            while k < len(w) and w[k] - w[k - 1] < 1e-6:
                k += 1
            k = min(k, len(w) - 1)
            t0 = time.time()
            ck = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest", states=states), spec,
                            ce.Options(inertia_method="sparse", inertia_max_dim=10 ** 6))
            t_certk = time.time() - t0
            s3k = [x for x in ck.statements if x["id"] == "S3.completeness"][0]
            rows.append({"seed": run.seed + s, "returned": w.tolist(), "correct": correct,
                         "members_of_fourfold_level": mult4, "solve_seconds": t_solve,
                         "P1_max_rel_residual": p1_res, "P1_flag": bool(p1_flag),
                         "P2_max_diff": p2_diff, "P2_flag": bool(p2_flag),
                         "P3_statuses": st, "P3_S3_summary": s3["summary"],
                         "P3_missing_at_least": s3["evidence"].get("missing_below_top_claim_at_least"),
                         "P3_factorizations": len(s3["evidence"].get("counts", [])),
                         "P3_seconds": t_cert,
                         "P3k_k": int(k), "P3k_S3": s3k["status"], "P3k_S3_summary": s3k["summary"],
                         "P3k_factorizations": len(s3k["evidence"].get("counts", [])),
                         "P3k_seconds": t_certk})
            run.log(f"seed {run.seed + s}: correct={correct} members={mult4} P1={'flag' if p1_flag else 'pass'} "
                    f"P2={'flag' if p2_flag else 'pass'} ({p2_diff:.1e}) P3 S3={st['S3.completeness']} "
                    f"[{len(s3['evidence'].get('counts', []))} fact., {t_cert:.0f}s] | k={k}: {s3k['status']} "
                    f"[{t_certk:.0f}s] {s3k['summary'][:90]}")
            run.record("rows", rows)


if __name__ == "__main__":
    main()
