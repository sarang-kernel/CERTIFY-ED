"""Experiment M6: certified bounds on the dynamical structure factor of the Heisenberg ring from a certified ground
state (statements M5 and M6; docs/proofs/C-07b_spectral_bounds.md, UNVERIFIED until Sarang's check, D-12).

For the antiferromagnetic Heisenberg ring of N sites (J = 1) and O = sqrt(2/N) sum_j cos(pi j) S^z_j, the spectral
measure of O on the ground state |0> is the structure factor at q = pi: F(E) = sum_{E_n <= E} |<n|O|0>|^2, with total
weight F(inf) = S(pi) = <0|O^2|0>. Everything is done in the S^z = 0 block of H (an exact submatrix: H and O conserve
S^z, which the script checks), where the ground state lies and O|0> stays.

Per N: the ground state is computed (dense eigh up to dimension 4000, eigsh above), certified ('lowest' claim,
store_eigenvectors, eigenspaces='all'), and M5/M6 are evaluated for each K on a grid of energies E_0 + omega. The bounds
are compared with a reference F: exact (dense diagonalization of the block) up to dimension 4000, otherwise a
floating-point Lanczos run with full reorthogonalization from O|0> (a reference, not a certified quantity). Read-out:
S(pi) = tau_0 (M5) and the weight of the lowest excitation with weight, mu((a, b]) in [L(b) - U(a), U(b) - L(a)] with a
and b halfway to the neighbouring levels of the reference (the window edges are query points only; the certified
bracket holds for any a < b).

Usage (from the repository root): python experiments/m6_spectral_bounds.py OUTDIR [--N 8 10 12 14 16]
                                   [--K 32 64 128] [--energies 81] [--omega-max 5.0]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from fractions import Fraction

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certify_ed as ce  # noqa: E402
from certify_ed import spectral_bounds as sb  # noqa: E402
from certify_ed.correlations import site_operator  # noqa: E402

DENSE_MAX = 4000


def sz0_block(N):
    spec = ce.heisenberg_chain(N)
    H = ce.build_bitwise(spec).tocsr()
    O = sp.csr_matrix(sum(np.cos(np.pi * j) * site_operator(spec, "z", [j]) for j in range(N)) * np.sqrt(2.0 / N))
    pop = np.array([bin(i).count("1") for i in range(2 ** N)])
    idx = np.nonzero(pop == N // 2)[0]
    rest = np.nonzero(pop != N // 2)[0]
    leak = H[idx][:, rest].nnz + O[idx][:, rest].nnz
    if leak:
        raise RuntimeError(f"the S^z = 0 block is not invariant ({leak} entries couple it to the rest)")
    return H[idx][:, idx].tocsr(), O[idx][:, idx].tocsr()


def lanczos_measure(A, v, m_max=800, tol=1e-12):
    """Ritz values and weights of the spectral measure of v under A (full reorthogonalization; floating point)."""
    b0 = float(np.linalg.norm(v))
    Q = np.zeros((A.shape[0], m_max + 1))
    Q[:, 0] = v / b0
    alphas, betas = [], []
    for k in range(m_max):
        w = A @ Q[:, k]
        a = float(Q[:, k] @ w)
        w = w - a * Q[:, k] - (betas[-1] * Q[:, k - 1] if k > 0 else 0.0)
        for _ in range(2):
            w = w - Q[:, :k + 1] @ (Q[:, :k + 1].T @ w)
        alphas.append(a)
        b = float(np.linalg.norm(w))
        if b < tol * max(1.0, abs(a)):
            break
        betas.append(b)
        Q[:, k + 1] = w / b
    m = len(alphas)
    theta, S = eigh_tridiagonal(np.array(alphas), np.array(betas[:m - 1]))
    return theta, b0 ** 2 * S[0, :] ** 2, m


def reference(Hb, Ob):
    n = Hb.shape[0]
    if n <= DENSE_MAX:
        w, V = np.linalg.eigh(Hb.toarray())
        psi = V[:, 0]
        amp = V.T @ (Ob @ psi)
        return {"kind": "exact (dense diagonalization of the S^z = 0 block)", "E0": float(w[0]), "psi": psi,
                "levels": w, "weights": amp ** 2}
    w0, v0 = spla.eigsh(Hb, k=1, which="SA", tol=1e-14, ncv=40)
    psi = v0[:, 0]
    theta, wts, m = lanczos_measure(Hb, Ob @ psi)
    return {"kind": f"floating point (Lanczos with full reorthogonalization, {m} steps)", "E0": float(w0[0]),
            "psi": psi, "levels": theta, "weights": wts}


def F_ref(ref, E):
    return float(np.sum(ref["weights"][ref["levels"] <= E]))


def round_down(x: Fraction) -> float:
    f = float(x)
    return float(np.nextafter(f, -np.inf)) if Fraction(f) > x else f


def round_up(x: Fraction) -> float:
    f = float(x)
    return float(np.nextafter(f, np.inf)) if Fraction(f) < x else f


def window(ref, rel=1e-6):
    """(a, b, E1, weight) for the lowest level with weight > rel * total, halfway to its neighbours with weight."""
    lv, wt = np.asarray(ref["levels"]), np.asarray(ref["weights"])
    keep = wt > rel * wt.sum()
    lv, wt = lv[keep], wt[keep]
    order = np.argsort(lv)
    lv, wt = lv[order], wt[order]
    e1 = lv[0]
    same = np.abs(lv - e1) < 1e-9
    w1 = float(wt[same].sum())
    nxt = lv[~same][0] if np.any(~same) else e1 + 1.0
    return 0.5 * (ref["E0"] + e1), 0.5 * (e1 + nxt), float(e1), w1


def run(N, Ks, n_energies, omega_max):
    t0 = time.perf_counter()
    Hb, Ob = sz0_block(N)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = reference(Hb, Ob)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    cert = ce.certify(ce.Claim(np.array([ref["E0"]]), ref["psi"][:, None], operator=Hb, kind="lowest"), None,
                      ce.Options(store_eigenvectors=True, eigenspaces="all"))
    t_cert = time.perf_counter() - t0
    status = {s["id"]: s["status"] for s in cert.statements}
    a, b, e1, w1 = window(ref)
    E0 = ref["E0"]
    energies = sorted(set([float(E0 + x) for x in np.linspace(-0.25, omega_max, n_energies)] + [float(a), float(b)]))
    out = {"N": N, "dimension": int(Hb.shape[0]), "reference": ref["kind"], "E0_reference": E0,
           "S_pi_reference": float(np.sum(ref["weights"])), "certificate_statuses": status,
           "seconds": {"build": round(t_build, 2), "reference": round(t_ref, 2), "certify": round(t_cert, 2)},
           "window": {"a": a, "b": b, "E1_reference": e1, "weight_reference": w1}, "runs": []}
    for K in Ks:
        t0 = time.perf_counter()
        rec = sb.spectral_bounds_from_certificate(cert, Hb, Ob, K=K)
        t_m5 = time.perf_counter() - t0
        t0 = time.perf_counter()
        rec = sb.distribution_bounds(rec, energies)
        t_m6 = time.perf_counter() - t0
        st = {s["id"]: s["status"] for s in rec["statements"]}
        rows = sorted(rec["distribution"]["rows"], key=lambda r: r["energy"])
        lows = np.array([r["lower_monotone"] for r in rows])
        ups = np.array([r["upper_monotone"] for r in rows])
        Fr = np.array([F_ref(ref, r["energy"]) for r in rows])
        tol = 1e-12 if ref["kind"].startswith("exact") else 1e-9
        viol = [(r["energy"], lo, f, hi) for r, lo, f, hi in zip(rows, lows, Fr, ups) if not (lo - tol <= f <= hi + tol)]
        by_e = {r["energy"]: r for r in rows}
        La, Ua = by_e[float(a)]["lower_monotone"], by_e[float(a)]["upper_monotone"]
        Lb, Ub = by_e[float(b)]["lower_monotone"], by_e[float(b)]["upper_monotone"]
        w_lo = max(0.0, round_down(Fraction(Lb) - Fraction(Ua)))
        w_hi = round_up(Fraction(Ub) - Fraction(La))
        t0_iv = rec["chebyshev"]["moments"][0]["interval"]
        methods = {}
        for r in rows:
            for k_ in ("upper_proof", "lower_proof"):
                m_ = (r.get(k_) or {}).get("method")
                if m_:
                    methods[m_] = methods.get(m_, 0) + 1
        out["runs"].append({
            "K": K, "statuses": st, "seconds": {"M5": round(t_m5, 2), "M6": round(t_m6, 2)},
            "spectrum_interval": rec["chebyshev"]["spectrum_interval"], "spectrum_source": rec["chebyshev"]["spectrum_source"],
            "S_pi_interval": t0_iv, "largest_moment_halfwidth": max(m["bound"] for m in rec["chebyshev"]["moments"]),
            "n_energies": len(rows), "violations": viol, "mean_width": float(np.mean(ups - lows)),
            "median_width": float(np.median(ups - lows)), "max_width": float(np.max(ups - lows)),
            "methods": methods,
            "lowest_window": {"certified": [w_lo, w_hi], "fraction_of_S_pi": [w_lo / t0_iv[1], w_hi / t0_iv[0]]},
            "curve": {"energy": [r["energy"] for r in rows], "lower": lows.tolist(), "upper": ups.tolist(),
                      "reference": Fr.tolist()}})
        print(f"N={N} K={K}: M5 {t_m5:.1f}s M6 {t_m6:.1f}s, violations {len(viol)}, mean width "
              f"{np.mean(ups - lows):.2e}, lowest window [{w_lo:.6f}, {w_hi:.6f}] vs ref {w1:.6f}", flush=True)
    return out


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [r for r in results if r["N"] in (12, max(x["N"] for x in results))]
    fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 3.9), squeeze=False)
    colors = {32: "#c6dbef", 64: "#6baed6", 128: "#2171b5", 256: "#08306b"}
    for ax, res in zip(axes[0], show):
        E0, S = res["E0_reference"], res["S_pi_reference"]
        for run_ in sorted(res["runs"], key=lambda r: r["K"]):          # widest band first, narrower ones on top
            c = run_["curve"]
            om = np.array(c["energy"]) - E0
            ax.fill_between(om, np.array(c["lower"]) / S, np.array(c["upper"]) / S, step="post",
                            color=colors.get(run_["K"], "#999999"), lw=0, label=f"certified bounds, K = {run_['K']}")
        c = res["runs"][0]["curve"]
        ax.step(np.array(c["energy"]) - E0, np.array(c["reference"]) / S, where="post", color="black", lw=0.8,
                label="exact" if res["reference"].startswith("exact") else "Lanczos reference")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        h, lab = ax.get_legend_handles_labels()
        order = list(range(len(h) - 2, -1, -1)) + [len(h) - 1]
        ax.legend([h[i] for i in order], [lab[i] for i in order], fontsize=8, loc="lower right", frameon=False)
        ax.set_xlabel(r"$\omega = E - E_0$  (units of $J$)")
        ax.set_ylabel(r"$\int_0^{\omega} S^{zz}(\pi, \omega')\, d\omega' \,/\, S(\pi)$")
        ax.set_title(f"Heisenberg ring, N = {res['N']} (S$^z$ = 0 block, dim {res['dimension']})", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--N", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    ap.add_argument("--K", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--energies", type=int, default=81)
    ap.add_argument("--omega-max", type=float, default=5.0)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from OUTDIR/m6_spectral_bounds.json only")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.replot:
        with open(os.path.join(args.outdir, "m6_spectral_bounds.json")) as fh:
            figure(json.load(fh)["results"], os.path.join(args.outdir, "m6_spectral_bounds.png"))
        print("done")
        return
    results = []
    try:
        import subprocess
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip())
    except OSError:
        head, dirty = None, None
    meta = {"git_head": head, "tracked_changes": dirty,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "machine": platform.machine(), "system": platform.system(), "args": vars(args),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    for N in args.N:
        results.append(run(N, args.K, args.energies, args.omega_max))
        with open(os.path.join(args.outdir, "m6_spectral_bounds.json"), "w") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=1)
    figure(results, os.path.join(args.outdir, "m6_spectral_bounds.png"))
    print("done")


if __name__ == "__main__":
    main()
