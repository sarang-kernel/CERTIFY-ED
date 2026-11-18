"""Independent adversarial search for Appendix A (docs/proofs/appA_model-shift.md): builder shift and model shift.

Ground truth: H_spec from checker.exact_matrix (exact rationals, weights for spin 1) in mpmath at 60 digits; H_ref
= construct.build_bitwise(spec) and fl(H^) = hermitian_part(H_ref) taken as the stored binary64 numbers; spectral
norms and eigenvalues in mp. Checks: N1 ||H_ref - H_spec||_2 <= _builder_shift and ||fl(H^) - H_spec||_2 <=
_builder_shift (Lemma A3, both routes); W1 sorted eigenvalues within _builder_shift (Theorem A5); D1 a producer
matrix H_p = H_ref + perturbation (within product_tolerance, possibly non-Hermitian): check_model_fidelity's
model_shift_bound >= ||H_h - H_spec||_2 for the stored H_h = fl((H_p + H_p^H)/2) (Theorem A4); S3/S2 transfer on
the exact eigenvalues. Adversarial specs: genuinely complex couplings (hypot in |t^|), asymmetric |H|_abs (a
"+" term with cancelling couplings next to Hermitian terms), couplings 1e+-150, strings of length 4, spin 1
(irrational amplitudes), 12 couplings per term (m large), y letters, a single site.

Usage: python docs/proofs/independent/adversarial_appA_model-shift.py [--cases 120] [--seed 17]
"""
import argparse
import os
import sys

import mpmath as mp
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import Tally, eig_herm, herm, mpf, report, spec_norm, to_mp  # noqa: E402
from certify_ed.certify import Claim, Options, _builder_shift, check_model_fidelity  # noqa: E402
from certify_ed.checker import exact_matrix  # noqa: E402
from certify_ed.construct import build_bitwise, hermitian_part, product_tolerance  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402
from certify_ed.spec import ModelSpec  # noqa: E402


def exact_mp(spec):
    X = exact_matrix(spec)
    n = len(X.weights)
    den = mp.mpf(int(X.den))
    d = [1 / mp.sqrt(int(w)) for w in X.weights]
    M = mp.zeros(n, n)
    for r_, c_, re_, im_ in zip(X.rows, X.cols, X.re, X.im):
        r_, c_ = int(r_), int(c_)
        M[r_, c_] += d[r_] * mp.mpc(mp.mpf(int(X.weights[r_]) * int(re_)), mp.mpf(int(X.weights[r_]) * int(im_))) / den * d[c_]
    return M


def make_spec(rng, kind):
    half = LocalSpace.spin_space("1/2", "spin")
    one = LocalSpace.spin_space("1", "spin")
    if kind == "random":
        N = int(rng.integers(1, 4)); local = half if rng.random() < 0.5 else one; static = []
        for _ in range(int(rng.integers(1, 4))):
            L = int(rng.integers(1, min(N, 3) + 1))
            op = "".join(rng.choice(list("xyz+-")) for _ in range(L))
            sites = [int(s) for s in rng.choice(N, size=L, replace=False)]
            c = float(rng.standard_normal())
            static.append([op, [[c] + sites]])
            adj = op.translate(str.maketrans("+-", "-+"))[::-1]
            if adj != op:
                static.append([adj, [[c] + sites[::-1]]])
        return ModelSpec.from_static(N, local, static), False
    if kind == "complex_coupling":
        c = complex(rng.standard_normal(), rng.standard_normal())
        return ModelSpec.from_static(2, half, [["+-", [[[c.real, c.imag], 0, 1]]], ["-+", [[[c.real, -c.imag], 0, 1]]],
                                               ["zz", [[0.7, 0, 1]]]]), True
    if kind == "asymmetric_abs":
        c = float(rng.choice([0.1, 0.3, 1.0 / 3.0]))
        return ModelSpec.from_static(2, one, [["+", [[c, 0], [-c, 0]]], ["z", [[1.0, 0], [0.5, 1]]], ["zz", [[0.25, 0, 1]]]]), True
    if kind == "scale":
        s = 1e150 if rng.random() < 0.5 else 1e-150
        return ModelSpec.from_static(2, half, [["xx", [[s, 0, 1]]], ["yy", [[s, 0, 1]]], ["zz", [[0.3 * s, 0, 1]]]]), True
    if kind == "long_string":
        return ModelSpec.from_static(4, one, [["xyzx", [[0.5, 0, 1, 2, 3]]], ["xzyx", [[0.5, 3, 2, 1, 0]]],
                                              ["zz", [[1.0, 0, 1], [1.0, 1, 2]]]]), True
    if kind == "many_couplings":
        N = 3; cps = [[float(rng.standard_normal()), i, j] for i in range(N) for j in range(N) if i != j]
        return ModelSpec.from_static(N, one, [["xx", cps], ["yy", cps], ["zz", cps[:6]]]), True
    if kind == "single_site_y":
        return ModelSpec.from_static(1, one, [["y", [[0.3, 0]]], ["x", [[0.1, 0]]], ["z", [[-0.7, 0]]]]), True
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_n1 = Tally("N1: ||H_ref - H_spec||_2 and ||fl(H^) - H_spec||_2 <= _builder_shift")
    T_w1 = Tally("W1: sorted eigenvalues of fl(H^) and H_spec within _builder_shift (Weyl)")
    T_d1 = Tally("D1: check_model_fidelity model_shift_bound >= ||H_h - H_spec||_2 (planted producer matrix)")
    T_tr = Tally("A5 transfer: counts and sorted eigenvalues move by <= Delta on the exact eigenvalues")
    T_neg = Tally("negative control: producer matrix beyond the tolerance -> S1 not CERTIFIED")
    kinds = ["random", "random", "complex_coupling", "asymmetric_abs", "scale", "long_string", "many_couplings", "single_site_y"]
    for i in range(a.cases):
        kind = kinds[i % len(kinds)]
        spec, adv = make_spec(rng, kind)
        Href = build_bitwise(spec)
        Hh = hermitian_part(Href)
        Hs = exact_mp(spec)
        n = Hs.rows
        if not (spec_norm(Hs - herm(Hs)) == 0):
            T_n1.skip(); continue
        bs = _builder_shift(spec)
        n1, n2 = spec_norm(to_mp(Href.toarray()) - Hs), spec_norm(to_mp(Hh.toarray()) - Hs)
        T_n1.add(n1 <= mpf(bs) and n2 <= mpf(bs), adv, f"{kind}: ||E||={mp.nstr(n1, 8)} ||E_sym||={mp.nstr(n2, 8)} bs={bs}")
        lam_s, _ = eig_herm(Hs); lam_h, _ = eig_herm(to_mp(Hh.toarray()))
        T_w1.add(max(abs(x - y) for x, y in zip(lam_s, lam_h)) <= mpf(bs), adv, f"{kind}: Weyl gap > bs={bs}")
        # D1: planted producer matrix within the tolerance (non-Hermitian noise on the pattern of Href)
        tol = product_tolerance(spec, float(abs(Href).max()) if Href.nnz else 1.0)
        noise = sp.random(n, n, density=0.5, random_state=int(rng.integers(0, 2 ** 31)), format="csr")
        noise.data = rng.standard_normal(noise.nnz) + (1j * rng.standard_normal(noise.nnz) if np.iscomplexobj(Href.data) else 0.0)
        scale_n = float(abs(noise).max()) if noise.nnz else 1.0
        Hp = (Href + noise * (0.5 * tol / scale_n)).tocsr()
        st, _, _ = check_model_fidelity(Claim(eigenvalues=np.zeros(0), eigenvectors=None, operator=Hp), spec, Options())
        if st["status"] == "CERTIFIED":
            Hh_p = ((Hp + Hp.conj().T) / 2).tocsr() if float(abs(Hp - Hp.conj().T).max()) > 0 else Hp
            dist = spec_norm(to_mp(Hh_p.toarray()) - Hs)
            Delta = st["evidence"]["model_shift_bound"]
            T_d1.add(dist <= mpf(Delta), adv, f"{kind}: dist={mp.nstr(dist, 8)} Delta={Delta}")
            lam_p, _ = eig_herm(to_mp(Hh_p.toarray()))
            sig = mpf(float(lam_p[int(rng.integers(0, n))])) + mpf(1e-3) * (1 + abs(lam_p[0]))
            ok = (sum(1 for v in lam_s if v < sig - mpf(Delta)) <= sum(1 for v in lam_p if v < sig)
                  and sum(1 for v in lam_p if v <= sig) <= sum(1 for v in lam_s if v <= sig + mpf(Delta))
                  and all(abs(x - y) <= mpf(Delta) for x, y in zip(lam_s, lam_p)))
            T_tr.add(ok, adv, f"{kind}: transfer failed Delta={Delta}")
        else:
            T_d1.skip()
        Hbad = (Href + noise * (50 * tol / scale_n)).tocsr()
        stb, _, _ = check_model_fidelity(Claim(eigenvalues=np.zeros(0), eigenvectors=None, operator=Hbad), spec, Options())
        T_neg.add(stb["status"] != "CERTIFIED", True, f"{kind}: beyond tolerance certified")
    sys.exit(1 if report([T_n1, T_w1, T_d1, T_tr, T_neg]) else 0)


if __name__ == "__main__":
    main()
