"""Independent adversarial check of D-02 (matrix-free rounding width W, second pass, spec norm bound), from the
statement only. Ground truth: the exact spec operator H_spec is built INDEPENDENTLY of construct/matrixfree as a sparse
mp matrix (50 digits) from exact local matrices (mp square roots) composed as operator products with explicit
Jordan-Wigner strings; |H_spec|_abs is the same product with entrywise moduli; the per-row contribution count w_i is
the row sum of the corresponding 0/1 path-count matrices (exact integers).

Part A  Theorem D-02 (a): |fl(Hx)_i - (H_spec x)_i| <= gamma_W (|H_spec|_abs|x|)_i for random and adversarial x
        (dynamic range 1e-150..1e150, near-eigenvectors with cancellation, complex x with tiny imaginary parts, complex
        x for a real spec); Lemma M1/M2: w_i <= terms_per_row for every row, equality attained for q = 2.
Part B  Theorem D-02 (b): the Theorem-1 radius `residuals(...).rel_bound` >= ||H_spec x - mu x|| / ||x|| in mp for
        Ritz pairs of the floating-point build, perturbed values, and complex-typed vectors of a real spec.
Part C  Lemma M3 as stated: fl(|H|_abs|x|)_i >= (|H_spec|_abs|x|)_i (1 - gamma_{W-1}) -- counted separately for
        (real spec, real x), (complex spec, any x) and (real spec, complex x); a violation of the last family is the
        gap discussed in independent/D-02_matrix-free.md and is reported as INFORMATIONAL (not a violation of (a)/(b)).
Part D  Lemma N: norm_bound() >= ||H_spec||_2 (mp), local_norm_bound >= ||o||_2 for every letter; exact-letter decisions
        against an mp test; accuracy of NumPy's complex modulus on this platform (informational).

Usage: python docs/proofs/independent/adversarial_D-02_matrix-free.py [--cases 40] [--seed 107]
"""
from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indcommon import mpf, mpc_, Tally, report  # noqa: E402
from certify_ed.construct import build_bitwise  # noqa: E402
from certify_ed.enclosure import OperatorAdapter, residuals  # noqa: E402
from certify_ed.fp import U  # noqa: E402
from certify_ed.localops import LocalSpace  # noqa: E402
from certify_ed.matrixfree import MatrixFreeOperator, local_norm_bound  # noqa: E402
from certify_ed.spec import ModelSpec, Term  # noqa: E402

mp.mp.dps = 50


def gam(k):
    k = mp.mpf(int(k))
    return k * mpf(U) / (1 - k * mpf(U))


# ----------------------------------------------------------------------------------------------- exact local algebra
def exact_local(local, letter):
    """Exact q x q matrix <k'|o|k> as a dict {(k', k): mpc}, from the definitions (not from localops.matrix)."""
    q = local.q
    M = {}
    if letter == "I":
        return {(k, k): mp.mpc(1) for k in range(q)}
    if local.kind == "spin":
        S = mp.mpf(Fraction(local.spin).numerator) / Fraction(local.spin).denominator
        m = [mp.mpf(k) - S for k in range(q)]
        Sp = {(k + 1, k): mp.sqrt(S * (S + 1) - m[k] * (m[k] + 1)) for k in range(q - 1)}
        Sm = {(k, k + 1): v for (k1, k), v in Sp.items()}
        if letter == "z":
            M = {(k, k): mp.mpc(m[k]) for k in range(q)}
            scale = 2 if local.convention in ("pauli", "pauli_pm1") else 1
        elif letter == "+":
            M = {kk: mp.mpc(v) for kk, v in Sp.items()}; scale = 2 if local.convention == "pauli" else 1
        elif letter == "-":
            M = {kk: mp.mpc(v) for kk, v in Sm.items()}; scale = 2 if local.convention == "pauli" else 1
        elif letter == "x":
            M = {kk: mp.mpc(v) / 2 for kk, v in Sp.items()}; M.update({kk: mp.mpc(v) / 2 for kk, v in Sm.items()})
            scale = 2 if local.convention in ("pauli", "pauli_pm1") else 1
        elif letter == "y":
            M = {kk: mp.mpc(v) / (2j) for kk, v in Sp.items()}; M.update({kk: -mp.mpc(v) / (2j) for kk, v in Sm.items()})
            scale = 2 if local.convention in ("pauli", "pauli_pm1") else 1
        else:
            raise KeyError(letter)
        return {kk: v * scale for kk, v in M.items() if v != 0}
    if local.kind == "fermion":
        return {"+": {(1, 0): mp.mpc(1)}, "-": {(0, 1): mp.mpc(1)}, "n": {(1, 1): mp.mpc(1)},
                "z": {(0, 0): mp.mpc(-0.5), (1, 1): mp.mpc(0.5)}}[letter]
    # boson
    if letter == "+":
        return {(k + 1, k): mp.mpc(mp.sqrt(k + 1)) for k in range(q - 1)}
    if letter == "-":
        return {(k, k + 1): mp.mpc(mp.sqrt(k + 1)) for k in range(q - 1)}
    if letter == "n":
        return {(k, k): mp.mpc(k) for k in range(1, q)}
    raise KeyError(letter)


def site_operator(spec, letter, site):
    """Sparse dict of the full-space operator of `letter` on `site` (times its Jordan-Wigner string), the 0/1 path
    pattern, and the entrywise moduli; basis index = sum_s digit_s q^(N-1-s) (site 0 most significant)."""
    N, q, local = spec.n_sites, spec.q, spec.local
    loc = exact_local(local, letter)
    jw = local.kind == "fermion" and letter in "+-"
    place = q ** (N - 1 - site)
    n = q ** N
    O, A, C = {}, {}, {}
    for s in range(n):
        digit = (s // place) % q
        for (kp, k), v in loc.items():
            if k != digit:
                continue
            t = s + (kp - k) * place
            sign = 1
            if jw:
                # parity of the occupied sites on the string side of `site`
                if local.jw_string == "higher":
                    occ = sum((s // (q ** (N - 1 - l))) % q for l in range(site + 1, N))
                else:
                    occ = sum((s // (q ** (N - 1 - l))) % q for l in range(0, site))
                sign = -1 if occ % 2 else 1
            O[(t, s)] = v * sign
            A[(t, s)] = abs(v)
            C[(t, s)] = 1
    return O, A, C


def sp_mul(X, Y, add=lambda a, b: a + b):
    """(X Y) for sparse dicts {(i, j): value}."""
    by_row_Y = {}
    for (k, j), v in Y.items():
        by_row_Y.setdefault(k, []).append((j, v))
    Z = {}
    for (i, k), u in X.items():
        for j, v in by_row_Y.get(k, ()):
            key = (i, j)
            Z[key] = add(Z[key], u * v) if key in Z else u * v
    return Z


def sp_add_scaled(Z, X, c):
    for key, v in X.items():
        Z[key] = Z[key] + c * v if key in Z else c * v
    return Z


def exact_spec_operator(spec):
    """(H_spec, |H_spec|_abs, w) in ascending-integer basis order: sparse dicts and the exact per-row counts."""
    n = spec.q ** spec.n_sites
    H, Habs = {}, {}
    w = [0] * n
    for term in spec.terms:
        for c, sites in term.couplings:
            if c == 0:
                continue
            # rightmost letter acts first: O = O_1 O_2 ... O_L
            O = A = C = None
            for letter, site in zip(term.op, sites):
                o, a, cnt = site_operator(spec, letter, site)
                O = o if O is None else sp_mul(O, o)
                A = a if A is None else sp_mul(A, a)
                C = cnt if C is None else sp_mul(C, cnt)
            cm = mpc_(c)
            sp_add_scaled(H, O, cm)
            sp_add_scaled(Habs, A, abs(cm))
            for (i, _j), k in C.items():
                w[i] += k
    return H, Habs, w


def reorder(spec, D):
    """Map the ascending-integer dict to the operator's state order (descending reverses the index)."""
    if spec.state_order == "ascending":
        return D
    n = spec.q ** spec.n_sites
    return {(n - 1 - i, n - 1 - j): v for (i, j), v in D.items()}


def sp_matvec(D, x):
    y = [mp.mpc(0)] * len(x)
    for (i, j), v in D.items():
        y[i] += v * x[j]
    return y


def to_mp_vec(x):
    return [mpc_(v) for v in x]


# ----------------------------------------------------------------------------------------------- random specs
def random_spec(rng, i, hermitian=False):
    if hermitian:
        return hermitian_spec(rng, i)
    kind = i % 6
    if kind == 0:
        local = LocalSpace.spin_space("1/2", rng.choice(["spin", "pauli", "pauli_pm1"])); letters = "xyz+-"; N = int(rng.integers(2, 6))
    elif kind == 1:
        local = LocalSpace.spin_space("1"); letters = "xyz+-"; N = int(rng.integers(2, 4))
    elif kind == 2:
        local = LocalSpace.spin_space(rng.choice(["3/2", "2"])); letters = "xyz+-"; N = 2
    elif kind == 3:
        local = LocalSpace.fermion_space(rng.choice(["higher", "lower"])); letters = "+-nz"; N = int(rng.integers(2, 6))
    elif kind == 4:
        local = LocalSpace.boson_space(int(rng.integers(1, 4))); letters = "+-n"; N = 2 if local.q > 3 else int(rng.integers(2, 4))
    else:
        local = LocalSpace.spin_space("1/2"); letters = "xyz"; N = int(rng.integers(3, 6))     # Pauli strings, real or complex
    n_terms = int(rng.integers(1, 5))
    cplx_coeff = rng.random() < 0.4
    terms = []
    for _ in range(n_terms):
        L = int(rng.integers(1, min(N, 3) + 1))
        sites = tuple(int(s) for s in rng.choice(N, size=L, replace=False))
        op = "".join(rng.choice(list(letters)) for _ in range(L))
        couplings = []
        for _ in range(int(rng.integers(1, 3))):
            c = complex(rng.standard_normal() * 10.0 ** rng.integers(-3, 4), rng.standard_normal() if cplx_coeff else 0.0)
            sites = tuple(int(s) for s in rng.choice(N, size=L, replace=False))
            couplings.append((c, sites))
        terms.append(Term(op, couplings))
    spec = ModelSpec(N, local, terms, state_order=["descending", "ascending"][i % 2], name=f"adv-{i}")
    spec.validate()
    return spec


def hermitian_spec(rng, i):
    """Hermitian specs: Pauli/spin strings in x, y, z; fermion/boson hopping with both orientations, n and z."""
    kind = i % 4
    if kind == 0:
        local = LocalSpace.spin_space("1/2", rng.choice(["spin", "pauli", "pauli_pm1"])); N = int(rng.integers(2, 6)); letters = "xyz"
    elif kind == 1:
        local = LocalSpace.spin_space(rng.choice(["1", "3/2"])); N = 2 if local.q > 3 else int(rng.integers(2, 4)); letters = "xyz"
    elif kind == 2:
        local = LocalSpace.fermion_space(rng.choice(["higher", "lower"])); N = int(rng.integers(2, 6)); letters = "nz"
    else:
        local = LocalSpace.boson_space(int(rng.integers(1, 3))); N = int(rng.integers(2, 4)); letters = "n"
    terms = []
    for _ in range(int(rng.integers(1, 4))):
        L = int(rng.integers(1, min(N, 3) + 1))
        op = "".join(rng.choice(list(letters)) for _ in range(L))
        couplings = [(complex(rng.standard_normal() * 10.0 ** rng.integers(-2, 3)), tuple(int(s) for s in rng.choice(N, size=L, replace=False)))
                     for _ in range(int(rng.integers(1, 3)))]
        terms.append(Term(op, couplings))
    if kind >= 2 and N >= 2:
        a, b = (int(s) for s in rng.choice(N, size=2, replace=False))
        c = complex(rng.standard_normal(), rng.standard_normal() if rng.random() < 0.5 else 0.0)
        terms.append(Term("+-", [(c, (a, b)), (c.conjugate(), (b, a))]))
    spec = ModelSpec(N, local, terms, state_order=["descending", "ascending"][i % 2], name=f"adv-herm-{i}")
    spec.validate()
    return spec


def random_x(rng, n, cplx, style):
    if style == "wide":
        x = rng.standard_normal(n) * 10.0 ** rng.integers(-150, 151, size=n)
    else:
        x = rng.standard_normal(n)
    if cplx:
        if style == "tinyim":   # moduli that hypot rounds down: |x|(1 + ~0.95 u) -> |x|
            x = x.astype(complex) * (1 + 1j * np.sqrt(1.9 * U))
        else:
            x = x + 1j * (rng.standard_normal(n) * (10.0 ** rng.integers(-150, 151, size=n) if style == "wide" else 1))
    return x


# ----------------------------------------------------------------------------------------------- parts
def part_A(rng, cases, T_a, T_w, T_m3, info):
    for i in range(cases):
        spec = random_spec(rng, i)
        op = MatrixFreeOperator(spec)
        H, Habs, w = exact_spec_operator(spec)
        H, Habs = reorder(spec, H), reorder(spec, Habs)
        n = spec.dim
        if spec.state_order != "ascending":
            w = w[::-1]
        # Lemma M1 / M2
        T_w.add(max(w) <= op.terms_per_row, spec.q == 2,
                f"A-w {spec.name} q={spec.q}: max_i w_i={max(w)} > terms_per_row={op.terms_per_row}")
        if spec.q == 2 and spec.local.kind == "spin":
            info.append(("M2 sharp", max(w) == op.terms_per_row))
        real_op = op.dtype == np.float64
        gW = gam(op.W)
        for style in ["plain", "wide", "tinyim", "eig"]:
            for cplx_x in ([False, True] if real_op else [True]):
                if style == "eig":
                    Hd = build_bitwise(spec).toarray()
                    if not np.allclose(Hd, Hd.conj().T):
                        continue
                    _, V = np.linalg.eigh(Hd)
                    x = V[:, int(rng.integers(0, n))]
                    x = x.astype(complex) if cplx_x else (x.real if real_op else x)
                    if real_op and not cplx_x:
                        x = np.real(x)
                else:
                    x = random_x(rng, n, cplx_x, style)
                if real_op and not cplx_x:
                    x = np.real(x)
                y = op.matvec(x)
                xm = to_mp_vec(x)
                Hx = sp_matvec(H, xm)
                Ax = sp_matvec(Habs, [abs(v) for v in xm])
                ok = True
                for k in range(n):
                    lhs = abs(mpc_(y[k]) - Hx[k])
                    if not (lhs <= gW * abs(Ax[k])):
                        ok = False
                        break
                T_a.add(ok, style != "plain" or cplx_x != (not real_op),
                        f"A-(a) {spec.name} W={op.W} style={style} cplx_x={cplx_x}: entry {k} |fl-exact|={mp.nstr(lhs, 6)} > gamma_W |H|_abs|x| = {mp.nstr(gW * abs(Ax[k]), 6)}")
                # Lemma M3 as stated
                ya = op.abs_matvec(np.abs(x) if not cplx_x else x)
                gWm1 = gam(op.W - 1)
                ok3 = all(mpf(ya[k]) >= abs(Ax[k]) * (1 - gWm1) for k in range(n))
                fam = "real spec, real x" if (real_op and not cplx_x) else ("real spec, complex x" if real_op else "complex spec")
                if fam == "real spec, complex x":
                    info.append(("M3 real-spec complex-x", ok3))
                else:
                    T_m3.add(ok3, style != "plain", f"A-M3 {spec.name} [{fam}] style={style}: second pass below (1 - gamma_{{W-1}})")


def part_B(rng, cases, T_b):
    for i in range(cases):
        spec = random_spec(rng, i + 1000, hermitian=True)
        op = MatrixFreeOperator(spec)
        H, _, _ = exact_spec_operator(spec)
        H = reorder(spec, H)
        n = spec.dim
        Hd = build_bitwise(spec).toarray()
        if not np.allclose(Hd, Hd.conj().T):
            T_b.skip(); continue
        lam, V = np.linalg.eigh(Hd)
        real_op = op.dtype == np.float64
        k = min(n, 4)
        idx = rng.choice(n, size=k, replace=False)
        X = V[:, idx]
        mu = lam[idx].copy()
        variant = i % 3
        if variant == 1:        # perturbed values and vectors
            mu = mu + rng.standard_normal(k) * 10.0 ** rng.integers(-12, -3, size=k)
            X = X + rng.standard_normal(X.shape) * 1e-9
        elif variant == 2 and real_op:      # complex-typed vectors for a real spec (reachable: claim dtype is the producer's)
            X = X.astype(complex) * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(1, k)))
        if real_op and variant != 2:
            X = np.real(X)
        rep = residuals(OperatorAdapter(op), mu, X)
        ok = True
        for c in range(k):
            xm = to_mp_vec(X[:, c])
            Hx = sp_matvec(H, xm)
            r = mp.sqrt(sum(abs(Hx[j] - mpf(mu[c]) * xm[j]) ** 2 for j in range(n)))
            xn = mp.sqrt(sum(abs(v) ** 2 for v in xm))
            if not (mpf(rep.rel_bound[c]) >= r / xn):
                ok = False
                break
        T_b.add(ok, variant != 0, f"B {spec.name} variant={variant} rigorous={rep.rigorous}: rel_bound {rep.rel_bound[c]!r} < exact {mp.nstr(r / xn, 8)}")


def part_D(rng, cases, T_n, T_loc, T_ex, info):
    spaces = [LocalSpace.spin_space("1/2", cv) for cv in ("spin", "pauli", "pauli_pm1")] + \
             [LocalSpace.spin_space(s) for s in ("1", "3/2", "2", "5/2")] + \
             [LocalSpace.fermion_space("higher"), LocalSpace.fermion_space("lower")] + \
             [LocalSpace.boson_space(k) for k in (1, 2, 3, 4, 8)]
    for local in spaces:
        for letter in local.letters:
            if letter == "I":
                continue
            loc = exact_local(local, letter)
            M = mp.matrix(local.q, local.q)
            for (i, j), v in loc.items():
                M[i, j] = v
            w, _ = mp.eigh(M.transpose_conj() * M)
            nrm = mp.sqrt(max(mp.re(v) for v in w))
            T_loc.add(mpf(local_norm_bound(local, letter)) >= nrm, False, f"D-loc {local.kind} q={local.q} {letter}: bound below ||o||_2")
            # exact-letter decision vs mp: every stored nonzero amplitude equals the exact one?
            stored = local.shift_components(letter)
            exact_all = True
            for delta, amp in stored:
                for k in range(local.q):
                    a = complex(amp[k])
                    if a == 0:
                        continue
                    ex = loc.get((k + delta, k))
                    if ex is None or abs(mpc_(a) - ex) != 0:
                        exact_all = False
            from certify_ed.matrixfree import _exact_components
            T_ex.add((_exact_components(local, letter) is not None) == exact_all, False,
                     f"D-exact {local.kind} q={local.q} {letter}: code says exact={_exact_components(local, letter) is not None}, mp says {exact_all}")
    for i in range(cases):
        spec = random_spec(rng, i + 2000)
        if spec.dim > 32:
            T_n.skip(); continue
        op = MatrixFreeOperator(spec)
        H, _, _ = exact_spec_operator(spec)
        n = spec.dim
        M = mp.matrix(n, n)
        for (a, b), v in H.items():
            M[a, b] = v
        w, _ = mp.eigh(M.transpose_conj() * M)
        nrm = mp.sqrt(max(mp.re(v) for v in w)) if n else mp.mpf(0)
        T_n.add(mpf(op.norm_bound()) >= nrm, spec.n_terms > 2, f"D-N {spec.name}: norm_bound {op.norm_bound()!r} < ||H_spec||_2 {mp.nstr(nrm, 10)}")
        ad = OperatorAdapter(op)
        T_n.add(mpf(ad._norm) >= nrm, True, f"D-N adapter {spec.name}: adapter norm {ad._norm!r} < ||H_spec||_2 {mp.nstr(nrm, 10)}")
    # hypot accuracy on this platform (informational)
    z = rng.standard_normal(100000) * 10.0 ** rng.integers(-5, 6, size=100000) + 1j * rng.standard_normal(100000) * 10.0 ** rng.integers(-5, 6, size=100000)
    z = np.concatenate([z, rng.standard_normal(20000) * (1 + 1j * np.sqrt(1.9 * U)), rng.standard_normal(20000) * (1 + 1j * np.sqrt(0.5 * U))])
    worst = mp.mpf(0)
    for v in z:
        ex = abs(mpc_(v))
        worst = max(worst, abs(mpf(abs(v)) - ex) / ex)
    info.append(("hypot max relative error / u", mp.nstr(worst / mpf(U), 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=107)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T_a = Tally("Theorem D-02 (a): |fl(Hx) - H_spec x| <= gamma_W |H_spec|_abs |x| entrywise (mp, independent H_spec)")
    T_w = Tally("Lemma M1: exact per-row contribution count w_i <= terms_per_row")
    T_m3 = Tally("Lemma M3 as stated: second pass >= (1 - gamma_{W-1}) exact  [real spec/real x and complex spec]")
    T_b = Tally("Theorem D-02 (b): Theorem-1 radius rel_bound >= exact ||H_spec x - mu x||/||x|| (mp)")
    T_n = Tally("Lemma N: norm_bound() and the adapter norm >= ||H_spec||_2 (mp)")
    T_loc = Tally("Lemma N: local_norm_bound >= ||o||_2 for every letter of every local space")
    T_ex = Tally("exact-letter decision of _exact_components agrees with an mp test")
    info = []
    part_A(rng, a.cases, T_a, T_w, T_m3, info)
    part_B(rng, a.cases, T_b)
    part_D(rng, a.cases, T_n, T_loc, T_ex, info)
    report([T_a, T_w, T_m3, T_b, T_n, T_loc, T_ex])
    m3 = [ok for tag, ok in info if tag == "M3 real-spec complex-x"]
    m2 = [ok for tag, ok in info if tag == "M2 sharp"]
    print(f"INFORMATIONAL: Lemma M3 as stated for (real spec, complex x): {sum(1 for ok in m3 if not ok)} of {len(m3)} cases below (1 - gamma_(W-1)) "
          "(the gap discussed in independent/D-02_matrix-free.md; (a) and (b) above are unaffected)")
    print(f"INFORMATIONAL: Lemma M2 sharpness (q = 2 spin): w_row attained in {sum(1 for ok in m2 if ok)} of {len(m2)} specs")
    for tag, v in info:
        if tag.startswith("hypot"):
            print(f"INFORMATIONAL: {tag} = {v}")


if __name__ == "__main__":
    main()
