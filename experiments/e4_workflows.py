"""E4: realistic research workflows.

W1  certify the lowest states of each (Nup, k) block computed with QuSpin and assemble a
    certified global low-energy spectrum (J1-J2 chain, all symmetry blocks)
W2  degenerate multiplet: certified quantum numbers and basis-independent observables
    (spin-1 Heisenberg chain, Haldane triplet)
W3  matrix-free producer at d = 2^20 (TFIM): what can and cannot be certified
W4  certified reference values for commonly used benchmark systems
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
from certify_ed.compose import SectorResult, certify_low_energy
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def qs(static, basis, dtype):
    return hamiltonian(static, [], basis=basis, dtype=dtype, check_herm=False, check_symm=False,
                       check_pcon=False)


def w1_sectors(run, N=16, J2=0.3, n_low=12):
    spec = ce.heisenberg_chain(N, J2=J2)
    static = spec.to_quspin_static()
    t0 = time.time()
    blocks = []
    for nup in range(N + 1):
        for k in range(N):
            b = spin_basis_1d(N, Nup=nup, kblock=k, pauli=0)
            if b.Ns == 0:
                continue
            H = qs(static, b, np.complex128)
            if b.Ns <= 400:
                w, Y = la.eigh(H.toarray())
            else:
                w, Y = H.eigsh(k=min(8, b.Ns - 2), which="SA", tol=0)
                o = np.argsort(w)
                w, Y = w[o], Y[:, o]
            blocks.append((nup, k, b, H, w, Y))
    t_prod = time.time() - t0
    pooled = np.sort(np.concatenate([bl[4] for bl in blocks]))
    E_cut = None
    for j in range(n_low, len(pooled)):
        if pooled[j] - pooled[j - 1] > 1e-4:
            E_cut = float(0.5 * (pooled[j] + pooled[j - 1]))
            break
    secs = []
    for nup, k, b, H, w, Y in blocks:
        sel = np.where(w < E_cut)[0]
        # a partial eigsh list may not reach E_cut: the certificate decides whether it is complete
        secs.append(SectorResult(f"Sz={nup - N / 2:+.0f},k={k}", sp.csr_matrix(b.get_proj(np.complex128)),
                                 H.tocsr(), w[sel], Y[:, sel]))
    t0 = time.time()
    res = certify_low_energy(spec, secs, E_cut)
    t_cert = time.time() - t0
    labels = []
    for nup, k, b, H, w, Y in blocks:
        for e in w[w < E_cut]:
            labels.append((float(e), nup - N / 2, k))
    labels.sort()
    run.record("W1", {"N": N, "J2": J2, "n_blocks": len(blocks), "E_cut": E_cut, "producer_seconds": t_prod,
                      "certify_seconds": t_cert, "status": res["status"], "summary": res["summary"],
                      "K": res.get("K"), "radius": res.get("radius"), "eta": res["eta"],
                      "invariance_total": res["invariance_total"], "next_lower_bound": res.get("next_lower_bound"),
                      "levels": labels[:40],
                      "sector_status_counts": {s: sum(1 for p in res["sectors"] if p.get("status") == s)
                                               for s in ("COMPLETE", "MISSING", "INCONCLUSIVE")}})
    run.log(f"W1 N={N}: {len(blocks)} blocks, producer {t_prod:.1f}s, certify {t_cert:.1f}s: {res['summary']}")


def w2_multiplet(run, N=8):
    spec = ce.spin1_heisenberg_chain(N)
    H = ce.build_bitwise(spec)
    t0 = time.time()
    w, V = spla.eigsh(H, k=8, which="SA", tol=0)
    o = np.argsort(w)
    w, V = w[o], V[:, o]
    t_prod = time.time() - t0
    k = 4
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    states = ce.full_state_set(spec).states_in_order()
    digits = np.array([[(s // 3 ** (N - 1 - i)) % 3 for i in range(N)] for s in states])
    Sz_site = digits - 1.0
    Sz_tot = Sz_site.sum(axis=1)
    O = sp.diags(Sz_site[:, 0] * Sz_site[:, 1])
    Sz0 = sp.diags(Sz_site[:, 0])
    opts = ce.Options(symmetries={"Sz_total": (Sz_tot, 1.0)},
                      observables={"Sz_0 Sz_1": O, "Sz_0": Sz0}, inertia_method="sparse")
    t0 = time.time()
    c = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, opts)
    t_cert = time.time() - t0
    s4 = [s for s in c.statements if s["id"] == "S4.eigenspaces"][0]
    run.record("W2", {"N": N, "dim": spec.dim, "k": k, "eigenvalues": w[:k].tolist(),
                      "statuses": {s["id"]: s["status"] for s in c.statements},
                      "summaries": {s["id"]: s["summary"] for s in c.statements},
                      "clusters": s4["evidence"].get("clusters", []),
                      "producer_seconds": t_prod, "certify_seconds": t_cert})
    run.log(f"W2: {c.summary()}")


def w3_matrix_free(run, N=18):
    """Producer: QuSpin's matrix-free quantum_LinearOperator (no stored matrix) + ARPACK.
    The certifier sees only a black-box matvec."""
    from quspin.operators import quantum_LinearOperator
    spec = ce.tfim_chain(N, h=1.0)
    pa = {"pauli": 1, "spin": 0, "pauli_pm1": -1}[spec.local.convention]
    b = spin_basis_1d(N, pauli=pa)
    qop = quantum_LinearOperator(spec.to_quspin_static(), basis=b, dtype=np.float64, check_herm=False,
                                 check_symm=False, check_pcon=False)
    op = spla.LinearOperator(qop.shape, matvec=qop.dot, dtype=np.float64)
    t0 = time.time()
    w, V = spla.eigsh(op, k=3, which="SA", tol=1e-12, ncv=24, v0=np.ones(spec.dim))
    o = np.argsort(w)
    w, V = w[o], V[:, o]
    t_prod = time.time() - t0
    t0 = time.time()
    c = ce.certify(ce.Claim(w, V, operator=op, kind="lowest"), spec, ce.Options())
    t_cert = time.time() - t0
    run.record("W3", {"N": N, "dim": spec.dim, "eigenvalues": w.tolist(),
                      "statuses": {s["id"]: s["status"] for s in c.statements},
                      "summaries": {s["id"]: s["summary"] for s in c.statements},
                      "seconds": {s["id"]: s["seconds"] for s in c.statements},
                      "operator_note": c.body.get("operator_note"),
                      "producer_seconds": t_prod, "certify_seconds": t_cert,
                      "max_radius": float(c.arrays["radius"].max())})
    run.log(f"W3: producer {t_prod:.1f}s certify {t_cert:.1f}s\n{c.summary()}")


def w4_benchmarks(run):
    rows = []
    cases = [
        ("Heisenberg chain, N=20, PBC (Sz=0,k=0 block)", "chain", dict(N=20, J2=0.0)),
        ("J1-J2 chain, J2=0.5 (Majumdar-Ghosh), N=20, PBC (Sz=0,k=0)", "chain", dict(N=20, J2=0.5)),
        ("Heisenberg square 4x4, PBC (Sz=0)", "square", dict(Lx=4, Ly=4, J2=0.0)),
        ("J1-J2 square 4x4, J2=0.5, PBC (Sz=0)", "square", dict(Lx=4, Ly=4, J2=0.5)),
    ]
    from quspin.basis import spin_basis_general
    for name, kind, p in cases:
        if kind == "chain":
            spec = ce.heisenberg_chain(p["N"], J2=p["J2"])
            b = spin_basis_1d(p["N"], Nup=p["N"] // 2, kblock=0, pauli=0)
            kw = {"sector_isometry": sp.csr_matrix(b.get_proj(np.float64))}
        else:
            spec = ce.heisenberg_square(p["Lx"], p["Ly"], J2=p["J2"])
            n = p["Lx"] * p["Ly"]
            b = spin_basis_general(n, Nup=n // 2, pauli=0)
            kw = {"states": ce.StateSet(np.sort(np.asarray(b.states, dtype=np.int64)), "descending", "U(1)")}
        H = qs(spec.to_quspin_static(), b, np.float64)
        t0 = time.time()
        w, V = H.eigsh(k=6, which="SA", tol=0)
        o = np.argsort(w)
        w, V = w[o], V[:, o]
        t_prod = time.time() - t0
        c = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H.tocsr(), kind="lowest", **kw), spec,
                       ce.Options(inertia_method="sparse", inertia_max_dim=10 ** 6))
        rad = float(c.arrays["radius"][0])
        s3 = [s for s in c.statements if s["id"] == "S3.completeness"][0]
        rows.append({"system": name, "dim": int(b.Ns), "E0": float(w[0]), "radius": rad,
                     "E0_per_site": float(w[0]) / spec.n_sites,
                     "statuses": {s["id"]: s["status"] for s in c.statements},
                     "gap_lower_bound": s3["evidence"].get("gap_lower_bound"),
                     "estimated_gap": float(w[1] - w[0]),
                     "producer_seconds": t_prod, "certify_seconds": c.body["timing"]["total_seconds"]})
        run.log(f"W4 {name}: E0={w[0]:.12f} +- {rad:.1e}; {rows[-1]['statuses']}; "
                f"gap >= {rows[-1]['gap_lower_bound']}")
        run.record("W4", rows)


def main():
    import sys
    parts = sys.argv[1:] or ["W1", "W2", "W3", "W4"]
    fns = {"W1": w1_sectors, "W2": w2_multiplet, "W3": w3_matrix_free, "W4": w4_benchmarks}
    with Run("e4_workflows_" + "".join(parts)) as run:
        for p in parts:
            fns[p](run)


if __name__ == "__main__":
    main()
