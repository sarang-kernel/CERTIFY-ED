"""Independent adversarial check of E-05 (variational certificates), written from the statement only.

Part A (exact rationals): Lemma 3 -- rayleigh_interval's [theta_lo, theta_hi] contains the exact Rayleigh quotient of
the stored data and r_stored^2 >= the exact r^2 = <psi, H^2 psi>/<psi, psi> - theta^2, for random sparse Hermitian H
(real and complex, n up to 40, scales 1e-3..1e3, rows of width 1..n), trial states random / near-eigenvectors (r down
to 1e-12) / sign patterns with theta ~ 0 / scales 1e-3..1e3.
Part B (mp, 60 digits): Temple -- certify_variational on stored matrices with rho = lambda_2 rounded down (declared):
V1 upper >= lambda_1, V2 interval contains an eigenvalue, V3 lower <= lambda_1 or INCONCLUSIVE exactly when
theta_hi >= rho; two-level trial states (equality up to the rounding slack); trial states orthogonal to the ground
state; negative control rho = lambda_3 (bound may exceed lambda_1; status must be CERTIFIED_UNDER_HYPOTHESIS and
name declared_gap).
Part C (mp): the spec route with a model shift sigma > 0 -- certify_variational(spec=...) on small spin / boson chains
with complex hopping (the raw build is not exactly Hermitian), the truth being the eigenvalues of the EXACT spec matrix
(checker.exact_matrix at 60 digits): V1, V2, V3 against lambda_1(H_spec), lambda_2(H_spec).
Part D (exact rationals): Lemma 4 -- ||psi_hat - psi_exact||_2 <= contraction_error for random MPS (L = 2..5, D = 1..5,
real / complex, scales 1e-2..1e2, tensors with +-1 entries (cancellation)); Lemma 5 (mp): |theta(psi) - theta(psi_hat)|
<= the recorded bound.
Part E: gap_from_certificate -- rho <= lambda_2 (mp) for certificates of k = 1 .. 4 lowest claims (Heisenberg chains).

Usage: python docs/proofs/independent/adversarial_E-05_variational.py [--cases 80] [--seed 91]
"""
from __future__ import annotations

import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import F, mpf, to_mp, herm, eig_herm, Tally, report  # noqa: E402
import certify_ed as ce  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, rayleigh_interval  # noqa: E402
from certify_ed.variational import certify_variational, state_from_mps, gap_from_certificate  # noqa: E402
from certify_ed.checker import exact_matrix  # noqa: E402
from certify_ed.spec import ModelSpec, Term  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402

mp.mp.dps = 60


def fc(z):
    z = complex(z)
    return (F(z.real), F(z.imag))


def rand_herm_sparse(rng, n, cplx, dens, scale):
    A = rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
    if cplx:
        A = A + 1j * rng.standard_normal((n, n)) * (rng.random((n, n)) < dens)
    H = (A + A.conj().T) * scale
    return sp.csr_matrix(H)


def exact_quadratics(H, x):
    """(N, D, Q) = (<x, H x>, <x, x>, <x, H^2 x>) exactly as Fractions (real parts; the imaginary parts vanish)."""
    Hc = sp.coo_matrix(H)
    n = x.size
    xf = [fc(v) for v in x]
    y = [(F(0), F(0)) for _ in range(n)]           # H x
    for r, c, v in zip(Hc.row, Hc.col, Hc.data):
        a = fc(v); b = xf[c]
        y[r] = (y[r][0] + a[0] * b[0] - a[1] * b[1], y[r][1] + a[0] * b[1] + a[1] * b[0])
    N = sum(xf[i][0] * y[i][0] + xf[i][1] * y[i][1] for i in range(n))
    D = sum(xf[i][0] ** 2 + xf[i][1] ** 2 for i in range(n))
    Q = sum(y[i][0] ** 2 + y[i][1] ** 2 for i in range(n))      # <Hx, Hx> = <x, H^2 x>
    return N, D, Q


def stmt(rec, sid):
    return [s for s in rec["statements"] if s["id"] == sid][0]


# ------------------------------------------------------------------------------------------------- Part A
def part_A(rng, cases, T_th, T_r):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 41))
        dens = float(rng.choice([0.05, 0.3, 1.0]))
        scale = float(rng.choice([1e-3, 1.0, 1e3]))
        H = rand_herm_sparse(rng, n, cplx, dens, scale)
        kind = i % 5
        if kind == 1:                                   # near-eigenvector
            w, V = np.linalg.eigh(H.toarray())
            x = V[:, int(rng.integers(0, n))] + rng.standard_normal(n) * float(rng.choice([1e-12, 1e-8, 1e-3]))
        elif kind == 2:                                 # sign pattern, theta near 0 for traceless-ish H
            x = rng.choice([-1.0, 1.0], size=n) * (1 + 0j if cplx else 1.0)
        elif kind == 3:                                 # wide dynamic range in the entries
            x = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 7, size=n)
        else:
            x = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0.0)
        x = np.asarray(x, dtype=complex if cplx else float) * float(rng.choice([1e-3, 1.0, 1e3]))
        ri = rayleigh_interval(OperatorAdapter(H), x)
        if ri is None:
            T_th.skip(); continue
        lo, hi, th, r_hat = ri
        N, D, Q = exact_quadratics(H, x)
        theta = N / D
        ok_th = F(lo) <= theta <= F(hi)
        T_th.add(ok_th, kind in (1, 2, 3), f"A n={n} cplx={cplx} kind={kind} theta={float(theta)!r} interval=({lo!r},{hi!r})")
        r_stored = float(np.nextafter(r_hat + max(hi - th, th - lo), np.inf)) * (1 + 8 * 2.0 ** -53)
        r2 = Q / D - theta * theta
        T_r.add(F(r_stored) ** 2 >= r2, kind in (1, 2, 3), f"A n={n} kind={kind} r_exact={float(r2) ** 0.5:.3e} r_stored={r_stored:.3e}")


# ------------------------------------------------------------------------------------------------- Part B
def dn(v):
    return float(np.nextafter(float(mp.mpf(v)), -np.inf))


def part_B(rng, cases, T, T_neg):
    for i in range(cases):
        cplx = i % 2 == 1
        n = int(rng.integers(2, 9))
        H = rand_herm_sparse(rng, n, cplx, 1.0, float(rng.choice([1e-2, 1.0, 1e2])))
        Hd = H.toarray()
        lam, V = eig_herm(to_mp(Hd))
        kind = i % 4
        wf, Vf = np.linalg.eigh(Hd)
        if kind == 1:                                   # two-level mixture (equality case)
            a = float(rng.uniform(0.05, 0.6))
            x = np.cos(a) * Vf[:, 0] + np.sin(a) * Vf[:, 1]
        elif kind == 2:                                 # orthogonal to the ground state
            x = Vf[:, 1:] @ rng.standard_normal(n - 1)
        else:
            x = Vf[:, 0] + rng.standard_normal(n) * float(rng.choice([1e-6, 1e-2, 0.3]))
        x = np.asarray(x, dtype=complex if cplx else float)
        rho = dn(lam[1])
        rec = certify_variational(H, x, gap_hypothesis=rho, gap_source="exact lambda_2 (test)")
        v1, v2, v3 = stmt(rec, "V1.rayleigh_upper"), stmt(rec, "V2.existence"), stmt(rec, "V3.temple_lower")
        ok = v1["status"] == "CERTIFIED" and mpf(v1["evidence"]["upper_bound"]) >= lam[0]
        lo_e, hi_e = v2["evidence"]["interval"]
        ok = ok and v2["status"] == "CERTIFIED" and any(mpf(lo_e) <= l <= mpf(hi_e) for l in lam)
        th_hi = rec["theta"]["upper"]
        if v3["status"] == "CERTIFIED_UNDER_HYPOTHESIS":
            ok = ok and mpf(v3["evidence"]["lower_bound"]) <= lam[0] and th_hi < rho
            ok = ok and any(h["name"] == "declared_gap" for h in v3.get("hypotheses", []))
            if kind == 1:                               # equality case: the slack is rounding only
                ok = ok and lam[0] - mpf(v3["evidence"]["lower_bound"]) <= mpf("1e-9") * (1 + abs(lam[0]))
        else:
            ok = ok and v3["status"] == "INCONCLUSIVE" and not th_hi < rho
        T.add(ok, kind in (1, 2), f"B n={n} kind={kind} V3={v3['status']} lower={v3['evidence'].get('lower_bound')} lam1={mp.nstr(lam[0], 12)}")
        if kind == 1 and n >= 3:                        # negative control: rho = lambda_3 > lambda_2 declared
            rec2 = certify_variational(H, x, gap_hypothesis=dn(lam[2]), gap_source="wrong (test)")
            v3b = stmt(rec2, "V3.temple_lower")
            okn = v3b["status"] in ("CERTIFIED_UNDER_HYPOTHESIS", "INCONCLUSIVE") and "CERTIFIED" != v3b["status"]
            if v3b["status"] == "CERTIFIED_UNDER_HYPOTHESIS":
                okn = okn and any(h["name"] == "declared_gap" for h in v3b.get("hypotheses", [])) and "UNDER THE DECLARED HYPOTHESIS" in v3b["summary"]
            T_neg.add(okn, True, f"B-neg n={n} status={v3b['status']}")


# ------------------------------------------------------------------------------------------------- Part C
def exact_spec_mp(spec):
    """Exact spec operator H = D K D^-1 (checker.exact_matrix) as an mp matrix (Hermitian to ~1e-58)."""
    ex = exact_matrix(spec)
    n = ex.weights.size
    d = [mp.sqrt(mp.mpf(int(w))) for w in ex.weights]
    H = mp.matrix(n, n)
    for r, c, a, b in zip(ex.rows, ex.cols, ex.re, ex.im):
        H[int(r), int(c)] = mp.mpc(mp.mpf(int(a)), mp.mpf(int(b))) / mp.mpf(int(ex.den)) * d[int(r)] / d[int(c)]
    return H


def specs_for_C(rng):
    t = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
    out = []
    L1 = LocalSpace.spin_space("1")
    out.append(ModelSpec(3, L1, [Term("+-", [(t, (i, i + 1)) for i in range(2)] + [(np.conj(t), (i + 1, i)) for i in range(2)]),
                                 Term("zz", [(float(rng.uniform(0.3, 1.5)), (i, i + 1)) for i in range(2)])]))
    Lb = LocalSpace.boson_space(2)
    out.append(ModelSpec(3, Lb, [Term("+-", [(t, (i, i + 1)) for i in range(2)] + [(np.conj(t), (i + 1, i)) for i in range(2)]),
                                 Term("nn", [(float(rng.uniform(0.1, 0.9)), (i, i + 1)) for i in range(2)])]))
    out.append(ce.heisenberg_chain(4, J=1.0, Delta=float(rng.uniform(0.5, 1.5)), J2=float(rng.uniform(0, 0.4))))
    for s in out:
        s.validate()
    return out


def part_C(rng, cases, T):
    for i in range(cases):
        for spec in specs_for_C(rng):
            Hm = exact_spec_mp(spec)
            lam, V = eig_herm(Hm)
            n = Hm.rows
            kind = i % 3
            Vf = np.array([[complex(V[r, k]) for k in range(n)] for r in range(n)])
            if kind == 0:
                x = Vf[:, 0] + rng.standard_normal(n) * 1e-7
            elif kind == 1:
                a = float(rng.uniform(0.05, 0.5)); x = np.cos(a) * Vf[:, 0] + np.sin(a) * Vf[:, 1]
            else:
                x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            rec = certify_variational(None, x, spec=spec, gap_hypothesis=dn(lam[1]), gap_source="exact lambda_2(H_spec) (test)")
            sigma = rec["operator"]["model_shift"]
            v1, v2, v3 = stmt(rec, "V1.rayleigh_upper"), stmt(rec, "V2.existence"), stmt(rec, "V3.temple_lower")
            if v1["status"] != "CERTIFIED":
                T.skip(); continue
            ok = mpf(v1["evidence"]["upper_bound"]) >= lam[0]
            lo_e, hi_e = v2["evidence"]["interval"]
            ok = ok and any(mpf(lo_e) <= l <= mpf(hi_e) for l in lam)
            if v3["status"] == "CERTIFIED_UNDER_HYPOTHESIS":
                ok = ok and mpf(v3["evidence"]["lower_bound"]) <= lam[0]
            else:
                ok = ok and v3["status"] == "INCONCLUSIVE"
            T.add(ok, kind == 1, f"C {spec.local.kind} n={n} kind={kind} sigma={sigma:.2e} V3={v3['status']} lower={v3['evidence'].get('lower_bound')} lam1={mp.nstr(lam[0], 12)}")


# ------------------------------------------------------------------------------------------------- Part D
def part_D(rng, cases, T4, T5):
    for i in range(cases):
        cplx = i % 2 == 1
        Lsites = int(rng.integers(2, 6)); d = int(rng.choice([2, 3]))
        Ds = [1] + [int(rng.integers(1, 6)) for _ in range(Lsites - 1)] + [1]
        scale = float(rng.choice([1e-2, 1.0, 1e2]))
        tensors = []
        for j in range(Lsites):
            A = rng.standard_normal((Ds[j], d, Ds[j + 1])) + (1j * rng.standard_normal((Ds[j], d, Ds[j + 1])) if cplx else 0.0)
            if i % 3 == 0:
                A = rng.choice([-1.0, 1.0], size=(Ds[j], d, Ds[j + 1])) * (1 + 0j if cplx else 1.0)
            tensors.append(A * scale)
        st = state_from_mps(tensors)
        # exact contraction in Fractions
        T = [[fc(v) for v in tensors[0][0, s, :]] for s in range(d)]      # list over s_1 of row vectors (length D_1)
        rows = T
        for A in tensors[1:]:
            new = []
            for row in rows:
                for s in range(A.shape[1]):
                    vec = []
                    for c in range(A.shape[2]):
                        acc = (F(0), F(0))
                        for k in range(A.shape[0]):
                            a = row[k]; b = fc(A[k, s, c])
                            acc = (acc[0] + a[0] * b[0] - a[1] * b[1], acc[1] + a[0] * b[1] + a[1] * b[0])
                        vec.append(acc)
                    new.append(vec)
            rows = new
        psi_exact = [r[0] for r in rows]
        err2 = sum((fc(st.vector[k])[0] - psi_exact[k][0]) ** 2 + (fc(st.vector[k])[1] - psi_exact[k][1]) ** 2 for k in range(len(psi_exact)))
        T4.add(err2 <= F(st.contraction_error) ** 2, i % 3 == 0, f"D L={Lsites} D={Ds} cplx={cplx} err={float(err2) ** 0.5:.3e} e={st.contraction_error:.3e}")
        # Lemma 5 through the record: theta(psi_exact) vs theta(psi_hat) within theta_source_vs_contracted
        n = len(psi_exact)
        H = rand_herm_sparse(rng, n, cplx, 0.5, 1.0)
        rec = certify_variational(H, {"mps": tensors})
        if "theta" not in rec or "theta_source_vs_contracted" not in rec["state"]:
            T5.skip(); continue
        Hm = to_mp(H.toarray())
        pe = mp.matrix([mp.mpc(mp.mpf(a.numerator) / a.denominator, mp.mpf(b.numerator) / b.denominator) for a, b in psi_exact])
        ph = to_mp(st.vector.reshape(-1, 1))
        th_e = mp.re((herm(pe) * Hm * pe)[0, 0] / (herm(pe) * pe)[0, 0])
        th_h = mp.re((herm(ph) * Hm * ph)[0, 0] / (herm(ph) * ph)[0, 0])
        T5.add(abs(th_e - th_h) <= mpf(rec["state"]["theta_source_vs_contracted"]) + mpf("1e-50"), i % 3 == 0,
               f"D5 L={Lsites} |dtheta|={mp.nstr(abs(th_e - th_h), 6)} bound={rec['state']['theta_source_vs_contracted']:.3e}")


# ------------------------------------------------------------------------------------------------- Part E
def part_E(rng, T):
    from certify_ed.construct import build_bitwise, hermitian_part
    for N, k in ((4, 1), (4, 2), (4, 3), (5, 4), (5, 8)):      # k respects the multiplets (S3 needs a gap)
        spec = ce.heisenberg_chain(N, Delta=float(rng.uniform(0.6, 1.4)))
        H = hermitian_part(build_bitwise(spec)); w, V = np.linalg.eigh(H.toarray())
        cert = ce.certify(ce.Claim(w[:k], V[:, :k], operator=H, kind="lowest"), spec, ce.Options(eigenspaces="none", degeneracy=False))
        g = gap_from_certificate(cert, spec, H.shape[0])
        lam, _ = eig_herm(exact_spec_mp(spec))
        if not g["ok"]:
            T.skip(); continue
        T.add(mpf(g["rho"]) <= lam[1], k == 1, f"E N={N} k={k} rho={g['rho']!r} lam2={mp.nstr(lam[1], 14)} ({g['source']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=80)
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_th = Tally("Lemma 3: [theta_lo, theta_hi] contains the exact Rayleigh quotient (exact rationals)")
    T_r = Tally("Lemma 3: r_stored >= exact residual r (exact rationals)")
    T_B = Tally("V1/V2/V3 on stored matrices with rho = lambda_2 (mp truth; Temple equality cases within 1e-9)")
    T_Bn = Tally("negative control: rho = lambda_3 declared -> never CERTIFIED, hypothesis named")
    T_C = Tally("spec route with model shift (non-exactly-Hermitian raw builds): V1/V2/V3 vs the exact spec matrix (mp)")
    T_4 = Tally("Lemma 4: ||psi_hat - psi_exact|| <= contraction_error (exact rationals)")
    T_5 = Tally("Lemma 5: |theta(psi) - theta(psi_hat)| <= theta_source_vs_contracted (mp)")
    T_E = Tally("gap_from_certificate: rho <= lambda_2 of the exact spec matrix (mp)")
    part_A(rng, a.cases, T_th, T_r)
    part_B(rng, a.cases, T_B, T_Bn)
    part_C(rng, max(4, a.cases // 8), T_C)
    part_D(rng, max(10, a.cases // 3), T_4, T_5)
    part_E(rng, T_E)
    report([T_th, T_r, T_B, T_Bn, T_C, T_4, T_5, T_E])


if __name__ == "__main__":
    main()
