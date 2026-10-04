"""E0: validate the verifier against independent references.

(a) eigenvalue enclosures (S2) vs eigenvalues computed in ball arithmetic (Arb), and the
    float64 residual bound vs the residual computed in ball arithmetic
(b) inertia counts with backward-error bound (S3) vs the exact count from a dense
    eigensolver, dense and sparse factorizations
(c) the two reference builders (S1) against each other and against QuSpin on random
    operator-string Hamiltonians for every supported local space
"""

import sys
import time
import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

warnings.filterwarnings("ignore")
from harness import Run, ROOT

sys.path.insert(0, ROOT + "/tests/v2")
from conftest import LOCALS, random_spec  # noqa: E402

from certify_ed.construct import build_bitwise, build_kron, matrix_difference  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, enclose  # noqa: E402
from certify_ed.inertia import inertia  # noqa: E402
from certify_ed.rigorous import arb_eigenvalues, arb_residual_radii  # noqa: E402
from certify_ed.spec import ModelSpec  # noqa: E402


def random_spec_letters(local, N, rng, letters, n_terms=6, max_len=3):
    """As conftest.random_spec, restricted to the given letters (QuSpin's spin-S>1/2 bases accept
    only '+', '-', 'z')."""
    adj = {"+": "-", "-": "+"}
    static = []
    for _ in range(n_terms):
        L = int(rng.integers(1, max_len + 1))
        op = "".join(rng.choice(letters, L))
        sites = [int(s) for s in rng.choice(N, L, replace=L > N)]
        c = float(rng.normal())
        if rng.random() < 0.3:
            c = complex(c, float(rng.normal()))
        static.append([op, [[c] + sites]])
        static.append(["".join(adj.get(l, l) for l in op[::-1]), [[np.conj(c)] + sites[::-1]]])
    return ModelSpec.from_static(N, local, static)


def rand_herm(n, ev, rng, cplx):
    Z = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    Q, _ = np.linalg.qr(Z)
    A = (Q * ev) @ Q.conj().T
    return (A + A.conj().T) / 2


def spectrum(kind, n, rng):
    if kind == "generic":
        return rng.standard_normal(n)
    if kind == "degenerate":
        return np.round(rng.standard_normal(n), 1)
    if kind == "near-degenerate":
        return np.r_[0.5 + 1e-11 * rng.standard_normal(3), rng.standard_normal(n - 3)]
    return np.linspace(-1, 1, n)


KINDS = ["generic", "degenerate", "near-degenerate", "equispaced"]
NOISE = [0.0, 1e-10, 1e-7, 1e-4]


def part_a(run, rng):
    trials, viol_interval, viol_group, viol_resid, no_ref = 0, 0, 0, 0, 0
    ball_w = 0.0
    ratios, by_kind = [], {}
    t0 = time.time()
    cases = []
    for i in range(240):
        kind = KINDS[i % 4]
        n = int(rng.integers(4, 40))
        A = rand_herm(n, spectrum(kind, n, rng), rng, cplx=bool(i % 2))
        cases.append((f"random-{kind}", A))
    # many-body matrices from random operator-string models (small, all local spaces)
    for j in range(70):
        name, mk, N = LOCALS[j % len(LOCALS)]
        spec = random_spec(mk(), min(N, 5), rng)
        A = build_bitwise(spec).toarray()
        if A.shape[0] > 128:
            continue
        cases.append((f"spec-{name}", A))
    for label, A in cases:
        n = A.shape[0]
        w, V = la.eigh(A)
        noise = NOISE[trials % 4]
        if noise:
            V = V + noise * rng.standard_normal(V.shape)
        E = enclose(OperatorAdapter(A), w, V)
        try:
            ref = arb_eigenvalues(A)
        except ArithmeticError:
            no_ref += 1
            continue
        ball_w = max(ball_w, max(r[1] - r[0] for r in ref))
        bad_i = any(not any(r[0] <= E.interval(i)[1] and r[1] >= E.interval(i)[0] for r in ref) for i in range(n))
        bad_g = False
        for g in E.groups:
            if g.distinct:
                lo = min(E.mu[i] - E.radius[i] for i in g.members)
                hi = max(E.mu[i] + E.radius[i] for i in g.members)
                if sum(1 for r in ref if r[0] <= hi and r[1] >= lo) < g.size:
                    bad_g = True
        rr = arb_residual_radii(A, w, V)
        rr_lo = arb_residual_radii(A, w, V, lower=True)
        # a violation: the floating-point bound lies below the rigorous lower end of the true residual
        bad_r = bool(np.any(E.residual.rel_bound < rr_lo))
        pos = rr > 0
        ratios.extend((E.residual.rel_bound[pos] / rr[pos]).tolist())
        trials += 1
        viol_interval += bad_i
        viol_group += bad_g
        viol_resid += bad_r
        k = label.split("-")[0]
        by_kind[k] = by_kind.get(k, 0) + 1
    ratios = np.array(ratios)
    out = {"matrices": trials, "reference_unavailable": no_ref, "eigenpairs": int(ratios.size), "by_source": by_kind,
           "violations_interval": viol_interval, "violations_group_count": viol_group,
           "violations_residual_bound": viol_resid, "arb_ball_max_width": float(ball_w),
           "bound_over_arb_residual": {"min": float(ratios.min()), "median": float(np.median(ratios)),
                                       "p99": float(np.quantile(ratios, 0.99)), "max": float(ratios.max())},
           "seconds": time.time() - t0}
    run.record("a_enclosures", out)
    run.log(f"(a) {trials} matrices, {ratios.size} pairs: violations {viol_interval}/{viol_group}/{viol_resid}; "
            f"bound/Arb median {np.median(ratios):.2f} max {ratios.max():.2f}")


def part_b(run, rng):
    checks, conclusive, viol = 0, 0, 0
    betas = []
    t0 = time.time()
    for i in range(300):
        kind = KINDS[[0, 1, 3][i % 3]]
        n = int(rng.integers(5, 120))
        A = rand_herm(n, spectrum(kind, n, rng), rng, cplx=bool(i % 2))
        w = np.linalg.eigvalsh(A)
        shifts = list(rng.uniform(w.min() - 0.2, w.max() + 0.2, 3))
        shifts.append(float(w[n // 2]) + 1e-13)        # adversarial: shift on top of an eigenvalue
        for s in shifts:
            for meth, M in (("dense", A), ("sparse", sp.csr_matrix(A))):
                r = inertia(M, s, method=meth)
                checks += 1
                if r.conclusive:
                    conclusive += 1
                    betas.append(r.beta)
                    if not (np.sum(w < s - r.beta) <= r.negative <= np.sum(w <= s + r.beta)):
                        viol += 1
    run.record("b_inertia", {"matrices": 300, "checks": checks, "conclusive": conclusive, "violations": viol,
                             "beta_median": float(np.median(betas)), "beta_max": float(np.max(betas)),
                             "seconds": time.time() - t0})
    run.log(f"(b) {checks} inertia checks, {conclusive} conclusive, {viol} violations")


def part_c(run, rng):
    from quspin.basis import boson_basis_1d, spin_basis_1d, spinless_fermion_basis_1d
    from quspin.operators import hamiltonian
    rows = []
    t0 = time.time()
    for name, mk, N in LOCALS:
        local = mk()
        n_ok_kron, n_ok_qs, worst_kron, worst_qs, n = 0, 0, 0.0, 0.0, 0
        for rep in range(40):
            if local.kind == "spin" and local.spin != "1/2":
                spec = random_spec_letters(local, N, rng, ["+", "-", "z"])
            else:
                spec = random_spec(local, N, rng)
            Hb = build_bitwise(spec)
            Hk = build_kron(spec)
            dk = matrix_difference(Hk, Hb)
            if local.kind == "spin":
                pa = {"pauli": 1, "spin": 0, "pauli_pm1": -1}[local.convention]
                basis = spin_basis_1d(N, pauli=pa, S=local.spin)
            elif local.kind == "fermion":
                basis = spinless_fermion_basis_1d(N)
            else:
                basis = boson_basis_1d(N, sps=local.q)
            dtype = np.float64 if spec.is_real() else np.complex128
            Hq = hamiltonian(spec.to_quspin_static(), [], basis=basis, dtype=dtype, check_herm=False,
                             check_symm=False, check_pcon=False).tocsr()
            dq = matrix_difference(Hq, Hb)
            scale = max(1.0, dq.get("max_abs_entry", 1.0))
            worst_kron = max(worst_kron, dk["max_abs_diff"] / scale)
            worst_qs = max(worst_qs, dq["max_abs_diff"] / scale)
            n_ok_kron += dk["max_abs_diff"] <= 1e-13 * scale
            n_ok_qs += dq["max_abs_diff"] <= 1e-13 * scale
            n += 1
        rows.append({"local_space": name, "N": N, "dim": local.q ** N, "models": n,
                     "kron_agree": n_ok_kron, "quspin_agree": n_ok_qs,
                     "max_rel_diff_kron": worst_kron, "max_rel_diff_quspin": worst_qs})
        run.log(f"(c) {name:18s} N={N}: kron {n_ok_kron}/{n} (max {worst_kron:.1e}), "
                f"QuSpin {n_ok_qs}/{n} (max {worst_qs:.1e})")
    run.record("c_builders", {"rows": rows, "seconds": time.time() - t0})


def part_d(run, seed):
    """The Jordan-Wigner episode, reproduced: with the string on sites of SMALLER index (CERTIFY-ED's
    convention before commit 13d2149), random fermion models are compared with QuSpin (string on sites
    of larger index). Number-conserving strings agree in both conventions; number-changing ones need not."""
    from quspin.basis import spinless_fermion_basis_1d
    from quspin.operators import hamiltonian
    from certify_ed.localops import LocalSpace
    rng = np.random.default_rng(seed)
    N = 7
    local = LocalSpace.fermion_space("lower")
    basis = spinless_fermion_basis_1d(N)
    n_dis, n_dis_changing, n_conserving, n_conserving_agree = 0, 0, 0, 0
    for rep in range(80):
        spec = random_spec(local, N, rng)
        if rep >= 40:
            # second half: keep only the number-conserving strings (adjoint pairs stay together)
            terms = [t for t in spec.terms if t.op.count("+") == t.op.count("-")]
            if not terms:
                terms = ModelSpec.from_static(N, local, [["+-", [[0.5, 0, 1]]], ["-+", [[0.5, 0, 1]]]]).terms
            spec = ModelSpec(N, local, terms)
        changing = any(t.op.count("+") != t.op.count("-") for t in spec.terms)
        Hb = build_bitwise(spec)
        dtype = np.float64 if spec.is_real() else np.complex128
        Hq = hamiltonian(spec.to_quspin_static(), [], basis=basis, dtype=dtype, check_herm=False,
                         check_symm=False, check_pcon=False).tocsr()
        dq = matrix_difference(Hq, Hb)
        agree = dq["max_abs_diff"] <= 1e-13 * max(1.0, dq.get("max_abs_entry", 1.0))
        n_dis += not agree
        n_dis_changing += (not agree) and changing
        if not changing:
            n_conserving += 1
            n_conserving_agree += agree
    rec = {"models": 80, "disagree_with_quspin": n_dis, "disagreeing_with_number_changing_string": n_dis_changing,
           "number_conserving_models": n_conserving, "number_conserving_agree": n_conserving_agree}
    run.record("d_jw_convention_episode", rec)
    run.log(f"(d) JW string on lower sites vs QuSpin: {n_dis}/80 models disagree "
            f"({n_dis_changing} contain a number-changing string); number-conserving models agree "
            f"{n_conserving_agree}/{n_conserving}")


def main():
    with Run("e0_verifier_validation") as run:
        rng = np.random.default_rng(run.seed)
        part_c(run, rng)
        part_b(run, rng)
        part_a(run, rng)
        part_d(run, run.seed + 3)


if __name__ == "__main__":
    main()
