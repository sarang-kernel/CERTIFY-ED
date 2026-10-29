"""Generate the planted-error corpus (roadmap B-07): small exact-diagonalization results with one known error
each, stored as producer bundles with an `expected.json` that names the statements a correct checker must
not certify.

    python corpus/make_corpus.py [--out corpus/cases] [--models heis8,j1j2_8,tfim8,heis10]

Each case directory is a producer bundle (docs/producer_bundle.md: eigenvalues.npz, eigenvectors.npz,
operator.npz, spec.json, meta.json) plus `expected.json` (corpus/README.md describes the fields). The
generator is deterministic: fixed seed, dense `scipy.linalg.eigh` on the reference operator, and every
planted change is a stated arithmetic operation on the arrays; `index.json` records a SHA-256 of every
array so that a regenerated corpus can be compared with the committed one. The error classes extend the
mechanisms of experiments/e1_failure_matrix.py (F1 coupling sign, F5 sector result as global, F7 ghosts,
F8/F9 convergence and swaps) to a form any checker can be run on.

Expectations come from the definitions of the statements (docs/theory.md, docs/SCOPE_MATRIX.md), not from
running a checker: a statement is in `must_not_certify` when the planted error makes it false or makes
its hypotheses unverifiable for the claimed data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import certify_ed as ce  # noqa: E402
from certify_ed.io.bundle import write_bundle  # noqa: E402

CORPUS_SCHEMA = "certify-ed/corpus-case/1.0"
SEED = 20261029

S1, S2, S3, S4, S5, S6, S7, S8 = ("S1.model_fidelity", "S2.eigenpairs", "S3.completeness", "S4.eigenspaces",
                                  "S5.corroboration", "S6.exact_recheck", "S7.thermodynamics", "S8.degeneracy")
ALL = (S1, S2, S3, S4, S5, S6, S7, S8)

MODELS = {
    "heis8": lambda: ce.heisenberg_chain(8),
    "j1j2_8": lambda: ce.heisenberg_chain(8, J2=0.5),
    "tfim8": lambda: ce.tfim_chain(8, J=1.0, h=1.0),
    "heis10": lambda: ce.heisenberg_chain(10),
}
DEGENERACY_TOL = 1e-9
KMIN = 5                    # the claim is 'the k lowest' with the smallest k >= KMIN at a spectral gap
ACCURACY_TARGET = 1e-8      # every claim declares this accuracy (meta.json); a wrong value by more is a false claim

# error class -> description (also the README table)
CLASSES = {
    "correct": "Control: the k lowest eigenpairs of the declared model with k at a spectral gap. No error.",
    "wrong_eigenvalue": "One claimed eigenvalue is shifted by delta (1e-5, above the declared accuracy 1e-8; or, on "
                        "the top pair, past the next level, so that the claimed set is no longer the k lowest); the "
                        "vector is the true eigenvector.",
    "unnormalised_isolated": "One isolated eigenvector is scaled by 1.7 (norm not 1). The eigenvalue statement is "
                             "still true for the direction; a checker that normalises may certify S2.",
    "unnormalised_cluster_member": "One member of a degenerate multiplet is scaled by 1.7: the Gram matrix of the "
                                   "claimed cluster vectors is far from the identity.",
    "swapped_pairs": "The eigenvectors of two non-degenerate levels are interchanged (eigenvalue i with vector j).",
    "missing_multiplet_member": "The claim 'the k lowest' stops inside a degenerate multiplet: members are missing.",
    "duplicated_vector": "The ground-state pair appears twice (a Lanczos ghost) and the top pair is dropped.",
    "wrong_coupling": "The producer's operator has the sign of one bond coupling flipped (J -> -J on bond (0,1); for "
                      "the TFIM the field on site 0 is doubled); the eigenpairs are correct for the producer's "
                      "operator but the declared model is the intended one.",
    "wrong_matrix_entry": "One symmetric pair of entries of the producer's matrix is changed by +0.3 (a matrix element "
                          "the model does not have); the eigenpairs are those of the wrong matrix.",
    "non_hermitian": "One entry of the producer's matrix is changed without its transpose: the stored matrix is not "
                     "Hermitian. The eigenpairs are those of the lower triangle (what LAPACK's eigh reads).",
    "nan_eigenvector_entry": "One eigenvector entry is NaN.",
    "nan_eigenvalue": "One claimed eigenvalue is NaN.",
    "mislabeled_sector": "The lowest states of one symmetry sector (Sz = 0 for the spin chains; even spin-flip parity "
                         "for the TFIM) are reported as the lowest states of the whole model.",
    "subnormal_entries": "A numerically zero eigenvector entry is replaced by the subnormal 1e-310: the no-underflow "
                         "hypothesis of the standard floating-point model is not established for this input.",
    "wrong_window_count": "A window claim lists the eigenvalues in (a, b) but one interior non-degenerate level is "
                          "left out.",
}


def array_sha256(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def levels(w, tol=DEGENERACY_TOL):
    """[(start, stop)] index ranges of (numerically) degenerate levels in the sorted spectrum."""
    out, s = [], 0
    for i in range(1, len(w) + 1):
        if i == len(w) or w[i] - w[i - 1] > tol:
            out.append((s, i))
            s = i
    return out


def k_at_gap(w, kmin=4):
    k = kmin
    while k < len(w) and w[k] - w[k - 1] < DEGENERACY_TOL:
        k += 1
    return k


def expected(model, cls, params, must_not, must, note="", refusal_ok=False, conditional=False, kind="lowest"):
    may = [s for s in ALL if s not in must_not and s not in must]
    return {"schema": CORPUS_SCHEMA, "case": "", "model": model, "error_class": cls, "claim_kind": kind,
            "parameters": params, "description": CLASSES[cls], "note": note,
            "must_not_certify": list(must_not), "must_certify": list(must), "may_certify": may,
            "refusal_ok": bool(refusal_ok), "conditional": bool(conditional)}


class Case:
    def __init__(self, cid, lam, X, H, spec, exp, kind="lowest", window=None):
        self.cid, self.lam, self.X, self.H, self.spec, self.exp = cid, lam, X, H, spec, exp
        self.kind, self.window = kind, window


def _restrict(spec, keep):
    """A spec holding only the couplings for which keep(sites) is true (same local space and encoding)."""
    terms = []
    for t in spec.terms:
        cps = [(c, s) for c, s in t.couplings if keep(s)]
        if cps:
            terms.append(ce.Term(t.op, cps))
    return ce.ModelSpec(spec.n_sites, spec.local, terms, site_order=spec.site_order, state_order=spec.state_order,
                        name="part")


def _flip_bond(spec, H):
    """Producer operator with J -> -J on bond (0,1): H minus twice the bond-(0,1) part of the spec."""
    H01 = ce.build_bitwise(_restrict(spec, lambda s: len(s) == 2 and sorted(s) == [0, 1])).tocsr()
    return (H - 2 * H01).tocsr()


def _double_field(spec, H):
    """Producer operator with the one-site terms on site 0 doubled."""
    H0 = ce.build_bitwise(_restrict(spec, lambda s: len(s) == 1 and s[0] == 0)).tocsr()
    return (H + H0).tocsr()


def _sector_claim(spec, H, k):
    """Lowest states of one symmetry sector embedded in the full space, claimed as the model's lowest."""
    n = H.shape[0]
    if spec.name.startswith("heisenberg"):
        stS = ce.magnetization_state_set(spec, 0)
        full_idx, _ = ce.full_state_set(spec).index_of(stS.states_in_order())
        E = sp.csr_matrix((np.ones(stS.size), (full_idx, np.arange(stS.size))), shape=(n, stS.size))
        label = "Sz=0"
    else:
        E = ce.cyclic_isometry(ce.full_state_set(spec), ce.spin_flip(spec), 0)
        label = "even spin-flip parity"
    Hs = (E.conj().T @ H @ E).tocsr()
    ws, Ys = la.eigh(Hs.toarray())
    ks = k_at_gap(ws, 3)
    lam = ws[:ks]
    X = np.asarray(E @ Ys[:, :ks])
    if np.iscomplexobj(X) and np.max(np.abs(X.imag)) < 1e-14:
        X = np.ascontiguousarray(X.real)
    w_all = la.eigvalsh(H.toarray())
    missing = int(np.sum(w_all <= lam[-1] + DEGENERACY_TOL)) - ks
    if missing <= 0:
        return None
    return lam, X, label, missing


def make_cases_for(model_name: str):
    spec = MODELS[model_name]()
    H = ce.build_bitwise(spec).tocsr()
    H.sort_indices()
    w, V = la.eigh(H.toarray())
    k = k_at_gap(w, KMIN)
    lv = levels(w)
    desc = f"{spec.name} N={spec.n_sites}"
    cases = []

    def C(cls, suffix, lam, X, Hp, exp, kind="lowest", window=None):
        cid = f"{model_name}_{cls}" + (f"_{suffix}" if suffix else "")
        exp["case"] = cid
        cases.append(Case(cid, np.asarray(lam, float), None if X is None else np.asarray(X), Hp, spec, exp,
                          kind=kind, window=window))

    # ---------------------------------------------------------------- control
    C("correct", "", w[:k], V[:, :k], H,
      expected(desc, "correct", {"k": k}, [], [S1, S2, S3],
               note="k is the smallest k >= 4 at a gap; S4 and S8 are decidable when the lowest k contain a "
                    "multiplet and have no clusters otherwise"))
    # ---------------------------------------------------------------- wrong eigenvalue
    lam = w[:k].copy()
    lam[1] += 1e-5
    C("wrong_eigenvalue", "1e-05", lam, V[:, :k], H,
      expected(desc, "wrong_eigenvalue", {"k": k, "pair": 1, "delta": 1e-5}, [S2], [S1],
               note="the residual of pair 1 is 1e-5 in norm: no radius below the declared accuracy 1e-8 exists for "
                    "the pair; a radius of 1e-5 would be true but does not meet the claim"))
    lam = w[:k].copy()
    delta = float(w[k] - w[k - 1]) + 0.1               # past the next level by 0.1
    lam[k - 1] += delta
    assert lam[k - 1] > w[k] + DEGENERACY_TOL, "the shifted top value must cross the next level"
    C("wrong_eigenvalue", "large", lam, V[:, :k], H,
      expected(desc, "wrong_eigenvalue", {"k": k, "pair": k - 1, "delta": delta}, [S2, S3], [S1],
               note=f"the top claimed value is {delta:.4g} above the true eigenvalue and above the next level: no "
                    f"radius below {delta:.4g} exists for the pair, and more than k = {k} eigenvalues lie below the "
                    "top of the claim"))
    # ---------------------------------------------------------------- unnormalised vectors
    iso = [i for (a, b) in lv if b - a == 1 and a < k for i in range(a, b)]
    if iso:
        X = V[:, :k].copy()
        X[:, iso[0]] *= 1.7
        C("unnormalised_isolated", "", w[:k], X, H,
          expected(desc, "unnormalised_isolated", {"k": k, "pair": iso[0], "scale": 1.7}, [], [S1],
                   note="Theorem 1 uses ||Hx - mu x|| / ||x||: S2 may be certified; a checker that requires unit "
                        "vectors must say so (INCONCLUSIVE), not refute S2"))
    clus = [(a, b) for (a, b) in lv if b - a > 1 and b <= k]
    if clus:
        a, b = clus[0]
        X = V[:, :k].copy()
        X[:, a] *= 1.7
        C("unnormalised_cluster_member", "", w[:k], X, H,
          expected(desc, "unnormalised_cluster_member", {"k": k, "pair": a, "cluster": [a, b], "scale": 1.7},
                   [S2, S4], [S1], conditional=True,
                   note="the Gram matrix of the claimed cluster vectors differs from the identity by 1.89 in norm: "
                        "neither the distinctness of the cluster's eigenvalues (Theorem 2) nor the eigenspace "
                        "statement (Theorem 3) can be certified for the vectors as claimed; a checker that "
                        "re-orthonormalises before checking and records it may certify (conditional case)"))
    # ---------------------------------------------------------------- swapped pairs
    if len(iso) >= 2:
        i, j = iso[0], iso[1]
        X = V[:, :k].copy()
        X[:, [i, j]] = X[:, [j, i]]
        C("swapped_pairs", "", w[:k], X, H,
          expected(desc, "swapped_pairs", {"k": k, "pairs": [i, j]}, [S2], [S1],
                   note=f"the residual of (mu_{i}, x_{j}) is |mu_{i} - mu_{j}| = {abs(w[i] - w[j]):.3g}: no small "
                        "radius exists for either pair"))
    # ---------------------------------------------------------------- missing multiplet member
    cut = [(a, b) for (a, b) in lv if b - a > 1 and a >= 1]
    if cut:
        a, b = cut[0]
        kk = a + 1                                  # the claim stops after the first member of the multiplet
        C("missing_multiplet_member", "", w[:kk], V[:, :kk], H,
          expected(desc, "missing_multiplet_member", {"k": kk, "multiplet": [a, b], "claimed_members": 1},
                   [S3], [S1, S2],
                   note="'the k lowest eigenvalues' with k inside a degenerate level: the gap above the claim is "
                        f"zero and {b - a - 1} member(s) of the top level are missing. S8 speaks only about clusters "
                        "of claimed pairs; the single claimed member is not a cluster, so S8 may be certified for "
                        "the other clusters or say that there are none"))
    # ---------------------------------------------------------------- duplicated vector (ghost)
    lam = np.concatenate([[w[0], w[0]], w[1:k - 1]])
    X = np.column_stack([V[:, 0], V[:, 0], V[:, 1:k - 1]])
    C("duplicated_vector", "", lam, X, H,
      expected(desc, "duplicated_vector", {"k": k, "duplicated_pair": 0, "dropped_pair": k - 1}, [S2, S3], [S1],
               note="two identical vectors cannot be matched to two distinct eigenvalues, and the k values are "
                    "not the k lowest: the top pair is missing (one eigenvalue is claimed once too often)"))
    # ---------------------------------------------------------------- wrong coupling (F1 mechanism)
    if spec.name.startswith("heisenberg"):
        Hb, what = _flip_bond(spec, H), "sign of the couplings on bond (0,1) flipped"
    else:
        Hb, what = _double_field(spec, H), "transverse field on site 0 doubled"
    wb, Vb = la.eigh(Hb.toarray())
    kb = k_at_gap(wb, KMIN)
    C("wrong_coupling", "", wb[:kb], Vb[:, :kb], Hb,
      expected(desc, "wrong_coupling", {"k": kb, "change": what}, [S1], [],
               note="the producer's operator is not the declared model; S2-S4 may be certified for the producer's "
                    "operator (the certificate must say which operator the radii refer to)"))
    # ---------------------------------------------------------------- wrong matrix entry
    Hm = H.tolil(copy=True)
    i0, j0 = 0, H.shape[0] - 1
    Hm[i0, j0] = Hm[i0, j0] + 0.3
    Hm[j0, i0] = Hm[j0, i0] + 0.3
    Hm = Hm.tocsr()
    wm, Vm = la.eigh(Hm.toarray())
    km = k_at_gap(wm, KMIN)
    C("wrong_matrix_entry", "", wm[:km], Vm[:, :km], Hm,
      expected(desc, "wrong_matrix_entry", {"k": km, "entry": [i0, j0], "added": 0.3}, [S1], [],
               note="one matrix element of the producer's operator is not in the model (it couples the two fully "
                    "polarised states)"))
    # ---------------------------------------------------------------- non-Hermitian producer matrix
    Hn = H.tolil(copy=True)
    Hn[j0, i0] = Hn[j0, i0] + 0.2             # lower triangle only: eigh reads it, the transpose is untouched
    Hn = Hn.tocsr()
    wn, Vn = la.eigh(Hn.toarray(), lower=True)
    kn = k_at_gap(wn, KMIN)
    C("non_hermitian", "", wn[:kn], Vn[:, :kn], Hn,
      expected(desc, "non_hermitian", {"k": kn, "entry": [j0, i0], "added": 0.2}, [S1], [],
               note="the stored matrix is not Hermitian, so it is not the declared model; a checker may certify "
                    "S2-S4 for its Hermitian part if the certificate says so"))
    # ---------------------------------------------------------------- NaN
    X = V[:, :k].copy()
    X[3, 1] = np.nan
    C("nan_eigenvector_entry", "", w[:k], X, H,
      expected(desc, "nan_eigenvector_entry", {"k": k, "entry": [3, 1]}, [S2, S3, S4, S8], [], refusal_ok=True,
               note="a NaN entry cannot be enclosed; refusing the input is a correct answer"))
    lam = w[:k].copy()
    lam[2] = np.nan
    C("nan_eigenvalue", "", lam, V[:, :k], H,
      expected(desc, "nan_eigenvalue", {"k": k, "pair": 2}, [S2, S3, S4, S8], [], refusal_ok=True,
               note="a NaN eigenvalue cannot be enclosed; refusing the input is a correct answer"))
    # ---------------------------------------------------------------- mislabeled sector (F5 mechanism)
    sec = _sector_claim(spec, H, k)
    if sec is not None:
        lam_s, X_s, label, missing = sec
        C("mislabeled_sector", "", lam_s, X_s, H,
          expected(desc, "mislabeled_sector", {"k": len(lam_s), "sector": label, "missing_in_lowest_k": missing},
                   [S3], [S1, S2],
                   note=f"the {label} block's lowest states are claimed as the model's lowest {len(lam_s)}; "
                        f"{missing} eigenvalue(s) of other sectors lie at or below the top of the claim"))
    # ---------------------------------------------------------------- subnormal entries
    zero_rows = np.where(V[:, 0] == 0.0)[0]
    if zero_rows.size:
        X = V[:, :k].copy()
        X[zero_rows[0], 0] = 1e-310
        C("subnormal_entries", "", w[:k], X, H,
          expected(desc, "subnormal_entries", {"k": k, "entry": [int(zero_rows[0]), 0], "value": 1e-310},
                   [S2], [S1], conditional=True,
                   note="the entry is below 2^-900, so the no-underflow hypothesis of the standard model is not "
                        "established for this input; a checker whose floating-point model covers gradual "
                        "underflow (Higham 2002, eq. 2.6) may certify (conditional case)"))
    # ---------------------------------------------------------------- wrong claimed window count
    inner = [a for (a, b) in lv if b - a == 1 and 1 <= a <= k - 2] or [a for (a, b) in lv if b - a == 1 and a < k]
    if inner:
        drop = inner[0]
        keep = [i for i in range(k) if i != drop]
        lo, hi = float(w[0]) - 1e-3, float(w[k - 1]) + 1e-3
        C("wrong_window_count", "", w[keep], V[:, keep], H,
          expected(desc, "wrong_window_count", {"window": [lo, hi], "eigenvalues_in_window": k, "claimed": k - 1,
                                                "dropped_pair": drop}, [S3], [S1, S2], kind="window",
                   note=f"the window holds {k} eigenvalues and the claim lists {k - 1}: completeness is false"),
          kind="window", window=(lo, hi))
    return cases, {"k": k, "dimension": int(H.shape[0]), "levels_in_lowest_k": [list(x) for x in lv if x[0] < k]}


def write_case(case: Case, out_dir: str) -> dict:
    d = os.path.join(out_dir, case.cid)
    meta = {"producer": {"name": "certify-ed planted-error corpus", "version": "1.0"},
            "description": f"{case.exp['model']}: {case.exp['error_class']}", "corpus_case": case.cid,
            "accuracy_target": ACCURACY_TARGET}
    write_bundle(d, case.lam, case.X, case.H, case.spec, fmt="npz", meta=meta, kind=case.kind, window=case.window)
    with open(os.path.join(d, "expected.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(case.exp, f, indent=1, sort_keys=True)
        f.write("\n")
    return {"case": case.cid, "model": case.exp["model"], "error_class": case.exp["error_class"],
            "k": int(case.lam.size), "dimension": int(case.H.shape[0]),
            "sha256": {"eigenvalues": array_sha256(case.lam),
                       "eigenvectors": array_sha256(case.X) if case.X is not None else None,
                       "operator_data": array_sha256(case.H.data)},
            "must_not_certify": case.exp["must_not_certify"], "must_certify": case.exp["must_certify"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases"))
    p.add_argument("--models", default=",".join(MODELS))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    index = {"schema": "certify-ed/corpus-index/1.0", "seed": SEED, "generator": "corpus/make_corpus.py",
             "certify_ed": ce.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
             "models": {}, "cases": []}
    for m in a.models.split(","):
        cases, info = make_cases_for(m)
        index["models"][m] = info
        for c in cases:
            index["cases"].append(write_case(c, a.out))
            print(f"{c.cid:45s} k={c.lam.size:3d} must_not_certify={[s.split('.')[0] for s in c.exp['must_not_certify']]}")
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1, sort_keys=True)
        f.write("\n")
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(a.out) for fn in fns)
    print(f"{len(index['cases'])} cases, {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
