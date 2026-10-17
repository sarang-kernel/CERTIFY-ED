"""E7: the W4 benchmark ground-state energies re-derived with no floating-point assumption.

For each system the producer is SciPy's eigsh (floating point). The claim 'E0 is the lowest eigenvalue
of the stated block' is then re-checked by certify_ed.exact_sectors.exact_recheck_sector_lowest: the
declared model as an exact rational matrix, the residual radius in integer arithmetic, and the count of
eigenvalues below a separation point by exact inertia in every symmetry sector of the block (1D
characters with Gaussian values, completeness of the decomposition checked by dimension).

usage: python3 e7_exact_benchmarks.py [system ...]   (default: all four W4 systems)
"""
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

from harness import Run
import certify_ed as ce
from certify_ed.basis import translation, reflection, spin_flip, cyclic_isometry
from certify_ed.exact_sectors import exact_recheck_sector_lowest, site_permutation


def square_translations(spec, Lx, Ly):
    site = lambda x, y: (y % Ly) * Lx + (x % Lx)
    tx = [site(x + 1, y) for y in range(Ly) for x in range(Lx)]
    ty = [site(x, y + 1) for y in range(Ly) for x in range(Lx)]
    # perm[i] = image of site i, sites enumerated i = y*Lx + x
    px = [0] * (Lx * Ly); py = [0] * (Lx * Ly)
    for y in range(Ly):
        for x in range(Lx):
            px[site(x, y)] = site(x + 1, y); py[site(x, y)] = site(x, y + 1)
    return site_permutation(spec, px, "Tx"), site_permutation(spec, py, "Ty")


SYSTEMS = {
    "chain_J2=0": ("Heisenberg chain, N=20, PBC (Sz=0,k=0 block)", "chain", dict(N=20, J2=0.0)),
    "chain_J2=0.5": ("J1-J2 chain, J2=0.5 (Majumdar-Ghosh), N=20, PBC (Sz=0,k=0)", "chain", dict(N=20, J2=0.5)),
    "square_J2=0": ("Heisenberg square 4x4, PBC (Sz=0)", "square", dict(Lx=4, Ly=4, J2=0.0)),
    "square_J2=0.5": ("J1-J2 square 4x4, J2=0.5, PBC (Sz=0)", "square", dict(Lx=4, Ly=4, J2=0.5)),
}


def main():
    which = sys.argv[1:] or list(SYSTEMS)
    with Run("e7_exact_benchmarks") as run:
        rows = []
        for key in which:
            name, kind, p = SYSTEMS[key]
            t0 = time.time()
            if kind == "chain":
                spec = ce.heisenberg_chain(p["N"], J2=p["J2"])
                st = ce.u1_state_set(spec, p["N"] // 2)
                H = ce.build_bitwise(spec, st)
                T = translation(spec)
                V0 = cyclic_isometry(st, T, 0)
                Hk = (V0.conj().T @ H @ V0).tocsr()
                w, Y = spla.eigsh(Hk, k=3, which="SA", tol=0, v0=np.random.default_rng(run.seed).standard_normal(Hk.shape[0]))
                o = np.argsort(w); w, Y = w[o], Y[:, o]
                X = np.real_if_close(V0 @ Y[:, :1])
                gens, fixed = [T, reflection(spec), spin_flip(spec)], {0: 0}
            else:
                spec = ce.heisenberg_square(p["Lx"], p["Ly"], J2=p["J2"])
                n = p["Lx"] * p["Ly"]
                st = ce.u1_state_set(spec, n // 2)
                H = ce.build_bitwise(spec, st)
                w, Y = spla.eigsh(H, k=3, which="SA", tol=0, v0=np.random.default_rng(run.seed).standard_normal(H.shape[0]))
                o = np.argsort(w); w, Y = w[o], Y[:, o]
                X = Y[:, :1]
                Tx, Ty = square_translations(spec, p["Lx"], p["Ly"])
                gens, fixed = [Tx, Ty, spin_flip(spec)], {}
            t_prod = time.time() - t0
            t1 = time.time()
            r = exact_recheck_sector_lowest(spec, st, gens, fixed, w[:1], X, next_hint=w[1], log=run.log)
            t_chk = time.time() - t1
            row = {"system": name, "u1_dim": int(st.size), "block_dim": r.get("dim_fixed"), "E0": float(w[0]),
                   "status": r["status"], "exact_radius": r.get("radii", [None])[0], "counts": r.get("counts"),
                   "sectors": r.get("sectors"), "group_order": r.get("group_order"),
                   "producer_seconds": t_prod, "exact_check_seconds": t_chk}
            rows.append(row)
            run.log(f"E7 {name}: E0 = {w[0]:.12f}, exact radius {row['exact_radius']}, status {r['status']}, "
                    f"{t_chk:.0f}s")
            run.record("E7", rows)


if __name__ == "__main__":
    main()
