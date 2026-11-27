"""Experiment M6 on the square lattice: certified bounds on the dynamical structure factor of the J1-J2 Heisenberg
model on a periodic Lx x Ly lattice from a certified ground state (statements M5 and M6;
docs/proofs/C-07b_spectral_bounds.md, UNVERIFIED until Sarang's check, D-12).

H = sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j (ce.heisenberg_square, sites y*Lx + x) and, for a wavevector q whose
entries are 0 or pi, O_q = sqrt(2/N) sum_j cos(q.r_j) S^z_j; then cos(q.r_j) = exp(i q.r_j) = +-1, so O_q is the Fourier
component of S^z (Hermitian, normalized as in experiments/m6_spectral_bounds.py). The spectral measure of O_q on the
ground state |0> is the structure factor at q: F(E) = sum_{E_n <= E} |<n|O_q|0>|^2, total weight S(q) = <0|O_q^2|0>.
Everything is done in the S^z = 0 block (an exact submatrix: the script checks that no entry of H or O_q couples it to
the rest).

Per J2: the ground state of the block is computed (dense eigh up to dimension 4000, eigsh above) and certified once
('lowest' claim, store_eigenvectors, eigenspaces='all'); M5/M6 are then evaluated for every q and K on a grid of
energies E_0 + omega. Reference F: exact (dense diagonalization) up to dimension 4000, otherwise floating-point Lanczos
with full reorthogonalization from O_q|0> for the same ground-state vector (a reference, not a certified quantity).
Read-out as in the ring experiment: S(q) = tau_0 (M5) and the weight of the lowest excitation with weight,
mu((a, b]) in [L(b) - U(a), U(b) - L(a)] (a and b are query points only; the bracket holds for any a < b).

On the 4 x 4 torus at J2 = 0.5 the sparse factorization of S3 does not fit under 4.5 GiB (docs/project/experiments/
E2_cost.md), so S3 and the S4 cluster that M5 needs are not certified on a workstation; the default is J2 = 0.

Usage (from the repository root): python experiments/m6_square_lattice.py OUTDIR [--L 4 4] [--J2 0]
                                  [--q pi,pi pi,0] [--K 32 64 128 256] [--energies 81] [--omega-max 5.0]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify_ed as ce  # noqa: E402
from certify_ed import spectral_bounds as sb  # noqa: E402
from certify_ed.correlations import site_operator  # noqa: E402
from m6_spectral_bounds import (F_ref, fraction_interval, interval_text, lanczos_measure, round_down,  # noqa: E402
                                round_up, window)

DENSE_MAX = 4000
RESULT = "m6_square_lattice.json"


def parse_q(text):
    """'pi,0' -> (1, 0): the entries of q in units of pi (0 or 1 only, so that O_q is real)."""
    parts = [p.strip().lower() for p in text.split(",")]
    out = []
    for p in parts:
        if p in ("0", "0.0"):
            out.append(0)
        elif p in ("pi", "1pi"):
            out.append(1)
        else:
            raise ValueError(f"q entries must be 0 or pi (got {text!r})")
    if len(out) != 2:
        raise ValueError(f"q must have two entries (got {text!r})")
    return tuple(out)


def q_label(q, tex=False):
    pi = r"$\pi$" if tex else "pi"
    return "(" + ", ".join(pi if x else "0" for x in q) + ")"


def q_operator(spec, Lx, Ly, q):
    N = Lx * Ly
    signs = [(-1) ** (q[0] * (j % Lx) + q[1] * (j // Lx)) for j in range(N)]
    return sp.csr_matrix(sum(s * site_operator(spec, "z", [j]) for j, s in enumerate(signs)) * np.sqrt(2.0 / N))


def sz0_blocks(H, Os, N):
    pop = np.array([bin(i).count("1") for i in range(2 ** N)])
    idx = np.nonzero(pop == N // 2)[0]
    rest = np.nonzero(pop != N // 2)[0]
    leak = H[idx][:, rest].nnz + sum(O[idx][:, rest].nnz for O in Os)
    if leak:
        raise RuntimeError(f"the S^z = 0 block is not invariant ({leak} entries couple it to the rest)")
    return H[idx][:, idx].tocsr(), [O[idx][:, idx].tocsr() for O in Os]


def ground_state(Hb):
    n = Hb.shape[0]
    if n <= DENSE_MAX:
        w, V = np.linalg.eigh(Hb.toarray())
        return {"exact": True, "E0": float(w[0]), "psi": V[:, 0], "levels": w, "V": V, "E1_block": float(w[1])}
    w0, v0 = spla.eigsh(Hb, k=2, which="SA", tol=1e-14, ncv=60)
    o = np.argsort(w0)
    return {"exact": False, "E0": float(w0[o[0]]), "psi": v0[:, o[0]], "E1_block": float(w0[o[1]])}


def reference(gs, Hb, Ob):
    if gs["exact"]:
        amp = gs["V"].T @ (Ob @ gs["psi"])
        return {"kind": "exact (dense diagonalization of the S^z = 0 block)", "E0": gs["E0"], "levels": gs["levels"],
                "weights": amp ** 2}
    theta, wts, m = lanczos_measure(Hb, Ob @ gs["psi"])
    return {"kind": f"floating point (Lanczos with full reorthogonalization, {m} steps)", "E0": gs["E0"],
            "levels": theta, "weights": wts}


def run(Lx, Ly, J2, qs, Ks, n_energies, omega_max):
    N = Lx * Ly
    t0 = time.perf_counter()
    spec = ce.heisenberg_square(Lx, Ly, J2=J2)
    H = ce.build_bitwise(spec).tocsr()
    Hb, Obs = sz0_blocks(H, [q_operator(spec, Lx, Ly, q) for q in qs], N)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    gs = ground_state(Hb)
    t_gs = time.perf_counter() - t0
    t0 = time.perf_counter()
    cert = ce.certify(ce.Claim(np.array([gs["E0"]]), gs["psi"][:, None], operator=Hb, kind="lowest"), None,
                      ce.Options(store_eigenvectors=True, eigenspaces="all"))
    t_cert = time.perf_counter() - t0
    status = {s["id"]: s["status"] for s in cert.statements}
    out = {"Lx": Lx, "Ly": Ly, "N": N, "J2": J2, "dimension": int(Hb.shape[0]), "E0_reference": gs["E0"],
           "E1_block_reference": gs["E1_block"], "certificate_statuses": status,
           "seconds": {"build": round(t_build, 2), "ground_state": round(t_gs, 2), "certify": round(t_cert, 2)},
           "wavevectors": []}
    E0 = gs["E0"]
    for q, Ob in zip(qs, Obs):
        t0 = time.perf_counter()
        ref = reference(gs, Hb, Ob)
        t_ref = time.perf_counter() - t0
        a, b, e1, w1 = window(ref)
        energies = sorted(set([float(E0 + x) for x in np.linspace(-0.25, omega_max, n_energies)] +
                              [float(a), float(b)]))
        qo = {"q": list(q), "label": q_label(q), "reference": ref["kind"], "S_q_reference": float(np.sum(ref["weights"])),
              "seconds_reference": round(t_ref, 2),
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
            viol = [(r["energy"], lo, f, hi) for r, lo, f, hi in zip(rows, lows, Fr, ups)
                    if not (lo - tol <= f <= hi + tol)]
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
            frac = fraction_interval([w_lo, w_hi], t0_iv)
            qo["runs"].append({
                "K": K, "statuses": st, "seconds": {"M5": round(t_m5, 2), "M6": round(t_m6, 2)},
                "spectrum_interval": rec["chebyshev"]["spectrum_interval"],
                "spectrum_source": rec["chebyshev"]["spectrum_source"], "S_q_interval": t0_iv,
                "largest_moment_halfwidth": max(m["bound"] for m in rec["chebyshev"]["moments"]),
                "n_energies": len(rows), "violations": viol, "mean_width": float(np.mean(ups - lows)),
                "median_width": float(np.median(ups - lows)), "max_width": float(np.max(ups - lows)),
                "methods": methods, "lowest_window": {"certified": [w_lo, w_hi], "fraction_of_S_q": frac},
                "curve": {"energy": [r["energy"] for r in rows], "lower": lows.tolist(), "upper": ups.tolist(),
                          "reference": Fr.tolist()}})
            print(f"{Lx}x{Ly} J2={J2} q={q_label(q)} K={K}: M5 {t_m5:.1f}s M6 {t_m6:.1f}s, violations {len(viol)}, "
                  f"mean width {np.mean(ups - lows):.2e}, lowest window [{w_lo:.6f}, {w_hi:.6f}] vs ref {w1:.6f}",
                  flush=True)
        out["wavevectors"].append(qo)
    return out


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nq = max(len(r["wavevectors"]) for r in results)
    fig, axes = plt.subplots(len(results), nq, figsize=(4.6 * nq, 3.5 * len(results)), squeeze=False)
    colors = {32: "#c6dbef", 64: "#6baed6", 128: "#2171b5", 256: "#08306b"}
    for row, res in zip(axes, results):
        E0 = res["E0_reference"]
        for ax, qo in zip(row, res["wavevectors"]):
            S = qo["S_q_reference"]
            for run_ in sorted(qo["runs"], key=lambda r: r["K"]):      # widest band first, narrower ones on top
                c = run_["curve"]
                ax.fill_between(np.array(c["energy"]) - E0, np.array(c["lower"]) / S, np.array(c["upper"]) / S,
                                step="post", color=colors.get(run_["K"], "#999999"), lw=0,
                                label=f"certified bounds, K = {run_['K']}")
            c = qo["runs"][0]["curve"]
            ax.step(np.array(c["energy"]) - E0, np.array(c["reference"]) / S, where="post", color="black", lw=0.8,
                    label="exact" if qo["reference"].startswith("exact") else "Lanczos reference")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            h, lab = ax.get_legend_handles_labels()
            order = list(range(len(h) - 2, -1, -1)) + [len(h) - 1]
            ax.legend([h[i] for i in order], [lab[i] for i in order], fontsize=7, loc="lower right", frameon=False)
            ax.set_xlabel(r"$\omega = E - E_0$  (units of $J_1$)")
            ax.set_ylabel(r"$\int_0^{\omega} S^{zz}(q, \omega')\, d\omega' \,/\, S(q)$")
            ax.set_title(f"{res['Lx']}x{res['Ly']} torus, $J_2$ = {res['J2']:g}, q = {q_label(qo['q'], tex=True)} "
                         f"(dim {res['dimension']})", fontsize=10)
            ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tables(data):
    """Markdown tables of a result file: per lattice and J2, and per wavevector and K."""
    out = ["| lattice | J2 | dim | S2 / S3 / S4 | certify (s) | E0 (reference) | gap in the block (reference) |",
           "|---|---|---|---|---|---|---|"]
    for r in data["results"]:
        st = r["certificate_statuses"]
        out.append(f"| {r['Lx']}x{r['Ly']} | {r['J2']:g} | {r['dimension']} | {st.get('S2.eigenpairs')} / "
                   f"{st.get('S3.completeness')} / {st.get('S4.eigenspaces')} | {r['seconds']['certify']} | "
                   f"{r['E0_reference']:.10f} | {r['E1_block_reference'] - r['E0_reference']:.6f} |")
    out += ["", "| J2 | q | K | M6 (s) | violations | mean width | median width | S(q) certified | S(q) reference | "
            "lowest window, certified | reference | fraction of S(q) | polynomials (radau / chebyshev / LP) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in data["results"]:
        for qo in r["wavevectors"]:
            for x in sorted(qo["runs"], key=lambda y: y["K"]):
                lw, sq, m = x["lowest_window"]["certified"], x["S_q_interval"], x["methods"]
                fr = fraction_interval(lw, sq)          # from the certified ends (records before round 8 rounded it)
                out.append(f"| {r['J2']:g} | {qo['label']} | {x['K']} | {x['seconds']['M6']} | {len(x['violations'])} | "
                           f"{x['mean_width']:.2e} | {x['median_width']:.2e} | {interval_text(sq, 12)} | "
                           f"{qo['S_q_reference']:.12f} | {interval_text(lw, 6)} | "
                           f"{qo['window']['weight_reference']:.6f} | {interval_text(fr, 4)} | "
                           f"{m.get('radau-hermite', 0)} / {m.get('chebyshev-hermite', 0)} / "
                           f"{m.get('linear program', 0)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--L", type=int, nargs=2, default=[4, 4], metavar=("LX", "LY"))
    ap.add_argument("--J2", type=float, nargs="+", default=[0.0])
    ap.add_argument("--q", nargs="+", default=["pi,pi", "pi,0"])
    ap.add_argument("--K", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--energies", type=int, default=81)
    ap.add_argument("--omega-max", type=float, default=5.0)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from OUTDIR/" + RESULT + " only")
    ap.add_argument("--table", action="store_true", help="print the Markdown tables of OUTDIR/" + RESULT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, RESULT)
    if args.table:
        with open(path, encoding="utf-8") as fh:
            print(tables(json.load(fh)))
        return
    if args.replot:
        with open(path, encoding="utf-8") as fh:
            figure(json.load(fh)["results"], path.replace(".json", ".png"))
        print("done")
        return
    qs = [parse_q(t) for t in args.q]
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=here)
        r_dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                 text=True, cwd=here)
        ok = r_head.returncode == 0 and r_dirty.returncode == 0          # None, not False, when git gives no answer
        head = r_head.stdout.strip() if ok else None
        dirty = bool(r_dirty.stdout.strip()) if ok else None
    except OSError:
        head, dirty = None, None
    meta = {"git_head": head, "tracked_changes": dirty,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "machine": platform.machine(), "system": platform.system(), "args": vars(args),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    results = []
    for J2 in args.J2:
        results.append(run(args.L[0], args.L[1], J2, qs, args.K, args.energies, args.omega_max))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=1)
    figure(results, path.replace(".json", ".png"))
    print("done")


if __name__ == "__main__":
    main()
