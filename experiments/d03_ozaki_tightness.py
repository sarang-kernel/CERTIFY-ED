"""D-03 measurement: tightness and cost of the Ozaki error-free products against the default bounds on E2 points.

Not part of the evidence run (no `certify-ed reproduce` step calls it). For each E2 point of experiments/e2_cost.py
the producer of that experiment is repeated in-process (QuSpin basis and H, ARPACK eigsh with the start vector of
seed 12345 + index, as in the 3.2.1 run results/run_full_20261105), the claimed pairs are chosen by the
certifier's rule (k = 4 widened over degeneracies), and the following are compared:

* the Theorem-1 radius of every claimed pair from `enclosure.residuals` (default: ||fl(R)|| + ||E||, E the
  gamma_{w+2} rounding bound) and from `ozaki.ozaki_residual` (UNVERIFIED, docs/proofs/D-03_ozaki_draft.md);
* the S2 radius the certificate reports, i.e. the Theorem-1 radius plus the model shift of S1 (U(1) blocks with a
  spec; the k = 0 blocks carry none on the sector block), for both;
* the orthonormality defect eta of the claimed columns from the accurately rounded Gram (`fp.exact_gram`, the
  default up to EXACT_GRAM_BUDGET), from the BLAS Gram with its gamma_n bound (used above the budget), and from
  `ozaki.ozaki_gram`;
* wall seconds of each (minimum of `--repeat` runs, one BLAS thread).

A second table times exact_gram against ozaki_gram on random unit columns (n in 924, 12870, 112720; g in 2, 8, 32).

Usage: python experiments/d03_ozaki_tightness.py [--points 0,1,2,...] [--repeat 3] [--out FILE]
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
# one thread everywhere, as experiments/e2_cost.py sets for its children: QuSpin's OpenMP matvec inside eigsh sums in
# a thread-dependent order, so the regenerated vectors match the 3.2.1 run only with these settings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                      # noqa: E402
import scipy.sparse as sp               # noqa: E402

import certify_ed as ce                 # noqa: E402
from certify_ed import ozaki            # noqa: E402
from certify_ed.certify import _builder_shift          # noqa: E402
from certify_ed.construct import StateSet               # noqa: E402
from certify_ed.enclosure import OperatorAdapter, enclose, residuals    # noqa: E402
from certify_ed.fp import U, exact_gram, fro_upper, gamma_dot, gram_additive_term, up, ULP_MIN   # noqa: E402

# the points of experiments/e2_cost.py main(), in its order (the seed of point i is 12345 + i)
E2_POINTS = ([dict(model="chain", block="u1", N=N) for N in (12, 14, 16, 18)]
             + [dict(model="chain", block="k0", N=N) for N in (16, 18, 20, 22, 24)]
             + [dict(model="square", block="u1", Lx=4, Ly=4), dict(model="square", block="u1", Lx=4, Ly=4, J2=0.5)])
RUN_SEED = 12345


def label(pt):
    return (f"{pt['model']} N={pt.get('N', pt.get('Lx', 0) * pt.get('Ly', 0))} {pt['block']} J2={pt.get('J2', 0.0)}")


def produce(pt, seed, k_solve=8):
    """The producer of experiments/e2_cost.py, in-process."""
    from quspin.basis import spin_basis_1d, spin_basis_general
    from quspin.operators import hamiltonian
    if pt["model"] == "chain":
        spec = ce.heisenberg_chain(pt["N"], J2=pt.get("J2", 0.0))
        kw = dict(Nup=pt["N"] // 2, pauli=0)
        if pt["block"] == "k0":
            kw["kblock"] = 0
        basis = spin_basis_1d(pt["N"], **kw)
    else:
        spec = ce.heisenberg_square(pt["Lx"], pt["Ly"], J2=pt.get("J2", 0.0))
        basis = spin_basis_general(pt["Lx"] * pt["Ly"], Nup=pt["Lx"] * pt["Ly"] // 2, pauli=0)
    H = hamiltonian(spec.to_quspin_static(), [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False,
                    check_pcon=False)
    v0 = np.random.default_rng(seed).standard_normal(basis.Ns)
    w, V = H.eigsh(k=k_solve, which="SA", tol=0, v0=v0)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    k = 4
    while k < len(w) and w[k] - w[k - 1] < 1e-6:
        k += 1
    k = min(k, len(w) - 1)
    states = None if pt["block"] == "k0" else StateSet(np.sort(np.asarray(basis.states, dtype=np.int64)), "descending", "U(1)")
    Hc = sp.csr_matrix(H.tocsr())
    # as certify() does: a producer matrix that is not exactly Hermitian is replaced by its Hermitian part (the
    # k = 0 blocks of N = 22 and 24 differ from their transposes by one rounding in four entries)
    asym = Hc - Hc.conj().T
    dev = float(abs(asym).max()) if asym.nnz else 0.0
    if dev > 0:
        Hc = sp.csr_matrix((Hc + Hc.conj().T) / 2)
    return spec, Hc, w[:k], np.ascontiguousarray(V[:, :k]), states, dev


def timed(fn, repeat):
    best, out = math.inf, None
    for _ in range(max(1, repeat)):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def eta_exact(X):
    G, Eg = exact_gram(X)
    g = X.shape[1]
    return up(fro_upper(G - np.eye(g)) + fro_upper(Eg) + 4 * U * g)


def eta_blas(X):
    n, g = X.shape
    G = X.T @ X
    return up(fro_upper(G - np.eye(g)) + gamma_dot(n, False) * fro_upper(X) ** 2) + gram_additive_term(g, n, False) + 2 * ULP_MIN


def eta_ozaki(X):
    r = ozaki.ozaki_gram(X)
    g = X.shape[1]
    return up(fro_upper(r.G - np.eye(g)) + fro_upper(r.E) + 4 * U * g), r


def point_row(i, pt, repeat):
    t0 = time.time()
    spec, H, mu, X, states, asym = produce(pt, RUN_SEED + i)
    t_prod = time.time() - t0
    n, k = X.shape
    adapter = OperatorAdapter(H)
    shift = _builder_shift(spec, states) if states is not None else 0.0
    old, t_old = timed(lambda: residuals(adapter, mu, X), repeat)
    new, t_new = timed(lambda: ozaki.ozaki_residual(H, mu, X), repeat)
    e_ex, t_ex = timed(lambda: eta_exact(X), repeat)
    e_bl, t_bl = timed(lambda: eta_blas(X), repeat)
    (e_oz, gr), t_oz = timed(lambda: eta_ozaki(X), repeat)
    enc_d, t_ed = timed(lambda: enclose(adapter, mu, X), repeat)
    enc_o, t_eo = timed(lambda: enclose(adapter, mu, X, residual_report=ozaki.residual_report(H, mu, X)), repeat)

    def enc_rec(e, t):
        return {"radius": e.radius.tolist(), "max_radius": float(np.max(e.radius)), "all_distinct": bool(e.all_distinct),
                "groups": [{"members": [int(m) for m in g.members], "method": g.method, "radius": float(g.radius),
                            "bound": g.bound} for g in e.groups], "seconds": t}
    return {
        "index": i, "label": label(pt), "cfg": pt, "dim": n, "nnz": int(H.nnz), "w": int(adapter.width), "k": k,
        "producer_seconds": t_prod, "model_shift": shift, "max_abs_asymmetry": asym,
        "default": {"radius": old.rel_bound.tolist(), "R_up": old.upper.tolist(), "E_norm": old.E_norm.tolist(),
                    "seconds": t_old},
        "ozaki": {"radius": new.rel_bound.tolist(), "E_norm": new.E_norm.tolist(), "seconds": t_new, "info": new.info},
        "radius_ratio": (new.rel_bound / old.rel_bound).tolist(),
        "enclosure_default": enc_rec(enc_d, t_ed), "enclosure_ozaki": enc_rec(enc_o, t_eo),
        "s2_radius_default": float(np.max(enc_d.radius) + shift),
        "s2_radius_ozaki": float(np.max(enc_o.radius) + shift),
        "gram": {"eta_exact_gram": e_ex, "seconds_exact_gram": t_ex, "eta_blas": e_bl, "seconds_blas": t_bl,
                 "eta_ozaki": e_oz, "seconds_ozaki": t_oz, "slices": gr.slices_x, "exhausted": gr.exhausted,
                 "products": gr.products},
    }


def gram_cost_table(repeat, sizes=(924, 12870, 112720), widths=(2, 8, 32), seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        for g in widths:
            X = np.linalg.qr(rng.standard_normal((n, g)))[0]
            (G1, E1), t1 = timed(lambda X=X: exact_gram(X), repeat)
            r, t2 = timed(lambda X=X: ozaki.ozaki_gram(X), repeat)
            rows.append({"n": n, "g": g, "seconds_exact_gram": t1, "seconds_ozaki": t2, "speedup": t1 / t2,
                         "max_E_exact_gram": float(E1.max()), "max_E_ozaki": float(r.E.max()),
                         "max_abs_G_difference": float(np.max(np.abs(G1 - r.G))), "slices": r.slices_x})
            print(f"gram n={n} g={g}: exact_gram {t1:.4f} s, ozaki {t2:.4f} s ({t1 / t2:.1f}x), slices {r.slices_x}",
                  flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", default=",".join(str(i) for i in range(len(E2_POINTS))))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-gram-table", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "d03_ozaki_tightness.json"))
    a = ap.parse_args(argv)
    idx = [int(s) for s in a.points.split(",") if s.strip()]
    rows = []
    for i in idx:
        r = point_row(i, E2_POINTS[i], a.repeat)
        rows.append(r)
        print(f"{r['label']}: dim {r['dim']} k {r['k']} w {r['w']} | Thm-1 radius default {max(r['default']['radius']):.3e} "
              f"ozaki {max(r['ozaki']['radius']):.3e} (ratio {max(r['radius_ratio']):.3f}) | max enclosure radius "
              f"{r['enclosure_default']['max_radius']:.3e} -> {r['enclosure_ozaki']['max_radius']:.3e} | S2 {r['s2_radius_default']:.3e}"
              f" -> {r['s2_radius_ozaki']:.3e} | residual s {r['default']['seconds']:.4f} -> {r['ozaki']['seconds']:.4f} | "
              f"eta exact {r['gram']['eta_exact_gram']:.2e} ({r['gram']['seconds_exact_gram']:.4f} s) blas "
              f"{r['gram']['eta_blas']:.2e} ozaki {r['gram']['eta_ozaki']:.2e} ({r['gram']['seconds_ozaki']:.4f} s)", flush=True)
    out = {"experiment": "d03_ozaki_tightness", "status": "UNVERIFIED prototype (D-12); not the default path",
           "repeat": a.repeat, "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
           "rows": rows}
    if not a.no_gram_table:
        out["gram_cost"] = gram_cost_table(a.repeat)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
