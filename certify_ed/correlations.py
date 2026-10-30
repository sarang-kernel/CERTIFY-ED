"""S4 extensions: batches of certified observables, two-point functions and static structure factors.

Everything here is Theorem 6 (one certified tr(P O) per Hermitian operator O) plus Proposition 15 (a
linear combination of certified traces carries the sum of |coefficient| x bound). Site operators are built
from the declared model's local space with the same bitwise builder that S1 uses, so the operator the
statement refers to is defined by the spec, not by the producer. Derivation and machine check:
docs/proofs/P15_summed_bounds.md (docs/theory.md 9.2); independent re-derivation pending (D-12).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .construct import StateSet, build_bitwise, full_state_set
from .eigenspace import EigenspaceCertificate, certify_observable
from .fp import U, add_up, fro_upper, gamma, gamma_dot, inf_norm_upper, up
from .spec import ModelSpec, Term

__all__ = ["site_operator", "certify_observables_batch", "certify_correlations", "structure_factor",
           "combine_certified", "hermitian_parts", "certify_observable_general", "compress_operator",
           "certify_observable_lifted", "certify_correlations_lifted"]


def site_operator(spec: ModelSpec, letters: str, sites: Sequence[int], coeff: complex = 1.0,
                  states: Optional[StateSet] = None) -> sp.csr_matrix:
    """The operator string `letters` on `sites` (e.g. 'z' on (3,), '+-' on (1, 2)) in the claim space.

    Built from the spec's local space by the bitwise builder, restricted to `states` (the producer's U(1)
    block) when given. Operators that leave the state set are an error (the statement would refer to a
    truncated operator)."""
    sub = ModelSpec(spec.n_sites, spec.local, [Term(letters, [(complex(coeff), tuple(int(s) for s in sites))])],
                    state_order=spec.state_order)
    st = states if states is not None else full_state_set(spec)
    O, leak = build_bitwise(sub, st, return_leak=True)
    if leak > 0:
        raise ValueError(f"operator {letters!r} on {tuple(sites)} leaves the producer's state set (leak {leak:.2e})")
    return sp.csr_matrix(O)


def _hermitian_part_or_fail(O, name: str):
    O = sp.csr_matrix(O)
    dev = abs(O - O.conj().T).max() if O.nnz else 0.0
    if dev > 0:
        raise ValueError(f"{name}: operator is not Hermitian (max asymmetry {dev:.2e}); Theorem 6 needs Hermitian O")
    return O


def _two_sum_exact(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether every entry of fl(a + b) equals a + b exactly (Knuth's TwoSum: the error term is zero)."""
    s_ = a + b
    bb = s_ - a
    err = (a - (s_ - bb)) + (b - bb)
    return bool(np.all(err == 0)) and bool(np.all(np.isfinite(s_)))


def hermitian_parts(O, name: str = "O"):
    """Hermitian and anti-Hermitian parts of O: H = (O + O^H)/2 and K = (O - O^H)/(2i), both Hermitian, with
    O = H + iK. Classical decomposition; both parts are formed in floating point and the formation is checked
    to be exact entrywise (Knuth's TwoSum error term zero; halving and the multiplication by -i are exact).
    Returns (H, K, exact). When `exact` is False the parts are rounded and the record that uses them must not
    be CERTIFIED (no bound on the rounding is claimed)."""
    O = sp.csr_matrix(O)
    Oh = sp.csr_matrix(O.conj().T)
    S = (O + Oh).tocsr()
    D = (O - Oh).tocsr()
    # the entrywise sums on the union of both patterns, checked for exactness with TwoSum
    Un = (abs(O) + abs(Oh)).tocoo()
    rows, cols = Un.row, Un.col
    a = np.asarray(O[rows, cols]).ravel() if rows.size else np.zeros(0)
    b = np.asarray(Oh[rows, cols]).ravel() if rows.size else np.zeros(0)
    exact = _two_sum_exact(a, b) and _two_sum_exact(a, -b)
    Hp = (S * 0.5).tocsr()                     # halving is exact (no underflow by hypothesis)
    Dd = D.data
    if np.iscomplexobj(Dd):
        K_data = (Dd.imag * 0.5) + 1j * (-(Dd.real) * 0.5)        # (x + iy)/(2i) = y/2 - i x/2, exact
    else:
        K_data = 1j * (-(Dd) * 0.5)
    Kp = sp.csr_matrix((K_data, D.indices, D.indptr), shape=D.shape)
    Kp.eliminate_zeros()
    Hp.eliminate_zeros()
    return Hp, Kp, bool(exact)


def certify_observable_general(ec: EigenspaceCertificate, X: np.ndarray, O, name: str) -> dict:
    """Theorem 6 for a Hermitian O; for a non-Hermitian O (C-05) the Hermitian decomposition O = H + iK:
    tr(P H) and tr(P K) are certified separately (Theorem 6 each) and recombined as tr(P O) = tr(P H) + i tr(P K)
    with bound b_H + b_K (Proposition 15 with coefficients 1 and i). The record names both parts and how to
    recombine them; `value` is then [re, im]. Records land in ec.observables[name]."""
    Oc = sp.csr_matrix(O)
    dev = abs(Oc - Oc.conj().T).max() if Oc.nnz else 0.0
    if dev == 0:
        rec = certify_observable(ec, X, Oc, name)
        rec["hermitian"] = True
        return rec
    Hp, Kp, exact = hermitian_parts(Oc, name)
    rh = certify_observable(ec, X, Hp, f"{name}.hermitian_part")
    rk = certify_observable(ec, X, Kp, f"{name}.anti_hermitian_part")
    for k in (f"{name}.hermitian_part", f"{name}.anti_hermitian_part"):
        ec.observables.pop(k, None)
    bound = float(add_up(rh["bound"], rk["bound"]))
    rec = {"hermitian": False, "hermitian_decomposition": True, "max_asymmetry": float(dev),
           "definition": "O = H + iK with H = (O + O^H)/2 and K = (O - O^H)/(2i), both Hermitian",
           "parts": {"hermitian": rh, "anti_hermitian": rk},
           "parts_formed_exactly": bool(exact),
           "recombine": "tr(P O) = tr(P H) + i tr(P K); |tr(P O) - (v_H + i v_K)| <= b_H + b_K (Proposition 15, a = 1, i)",
           "value": [float(rh["value"]), float(rk["value"])], "bound": bound,
           "statement": (f"tr(P {name}) = tr(P H) + i tr(P K) with tr(P H) in [{rh['value'] - rh['bound']:.15g}, "
                         f"{rh['value'] + rh['bound']:.15g}] and tr(P K) in [{rk['value'] - rk['bound']:.15g}, "
                         f"{rk['value'] + rk['bound']:.15g}]")}
    if not exact:
        rec["status"] = "INCONCLUSIVE"
        rec["reason"] = ("the Hermitian and anti-Hermitian parts could not be formed exactly in floating point; the "
                         "certified values refer to the rounded parts, not to O")
    ec.observables[name] = rec
    return rec


def certify_observables_batch(ec: EigenspaceCertificate, X: np.ndarray, operators: Dict[str, object]) -> Dict[str, dict]:
    """Theorem 6 for each named operator; non-Hermitian operators go through the Hermitian decomposition
    (certify_observable_general). Records land in ec.observables."""
    out = {}
    for name, O in operators.items():
        out[name] = certify_observable_general(ec, X, O, name)
    return out


def modulus_upper(z: complex) -> float:
    """Upper bound on |z| from IEEE +, x and sqrt only (Proposition 15, Lemma A; no libm call): with s =
    fl(sqrt(fl(fl(x^2) + fl(y^2)))) >= |z| (1 - u)^2 when neither square underflows, up(s) >= |z| (1 - u)^3 (1 + 8u)
    >= |z|. When a component is below 2^-500 the bound up(|x| + |y|) >= |z| is used instead (looser by at most
    sqrt(2), never affected by underflow)."""
    x, y = float(np.real(z)), float(np.imag(z))
    if y == 0.0:
        return abs(x)
    if x == 0.0:
        return abs(y)
    if min(abs(x), abs(y)) < 2.0 ** -500:
        return float(up(abs(x) + abs(y)))
    return float(up(math.sqrt(x * x + y * y)))


def combine_certified(coefficients: Sequence[complex], values: Sequence[float], bounds: Sequence[float]) -> Tuple[complex, float]:
    """Proposition 15: (fl(sum a_k v_k), sum |a_k| b_k + rounding of the sum), the bound rounded outward.

    The moduli |a_k|, |v_k| enter only through `modulus_upper` (an upper bound from IEEE +, x, sqrt; no libm
    call), the products and sums of the bound are rounded up, and the rounding term is 2 gamma_{m+2} sum
    |a_k||v_k| (docs/proofs/P15_summed_bounds.md, Section 2, Lemma B: the actual rounding of fl(sum a_k v_k) is
    at most (sqrt(2) gamma_2 + gamma_{m-1}(1 + sqrt(2) gamma_2)) sum |a_k||v_k| <= gamma_{m+2} sum |a_k||v_k|)."""
    a = np.asarray(coefficients, dtype=complex)
    v = np.asarray(values, dtype=complex)      # certified values may be complex (Hermitian decomposition, C-05)
    b = np.asarray(bounds, dtype=float)
    m = a.size
    val = complex(np.sum(a * v))
    bound = 0.0
    sum_av = 0.0
    for ak, vk, bk in zip(a.tolist(), v.tolist(), b.tolist()):
        ma = modulus_upper(ak)
        bound = float(add_up(bound, up(ma * bk)))
        sum_av = float(add_up(sum_av, up(ma * modulus_upper(vk))))
    rnd = float(up(2 * gamma(m + 2) * sum_av)) if m else 0.0
    return val, float(add_up(bound, rnd))


def certify_correlations(ec: EigenspaceCertificate, X: np.ndarray, spec: ModelSpec, letter: str,
                         sites: Sequence[int], states: Optional[StateSet] = None, name: str = "C",
                         pairs: Optional[Sequence[Tuple[int, int]]] = None) -> dict:
    """Two-point functions C_ij = tr(P O_i O_j) for the given sites.

    `letter` is one Hermitian single-site letter ('z', 'x', 'y', 'n', ...): O_i O_j is Hermitian for commuting
    Hermitian site operators (different sites), O_i^2 for i = j; only i <= j is computed and C_ji = C_ij.
    Since 3.2 (C-05) `letter` may be two letters, e.g. '+-' for S+_i S-_j: the product is built as one
    two-site term from the spec; where it is not Hermitian (i != j) it is certified through the Hermitian
    decomposition O = H + iK (certify_observable_general), the pair record carries `value` = Re tr(P O),
    `imag_value` = Im tr(P O), the summed bound and the `hermitian_decomposition` record, and
    C_ji = conj(C_ij) (tr(P O^H) = conj tr(P O)). Per-state values are tr(P O)/g with bound b/g (g = cluster
    size)."""
    sites = [int(s) for s in sites]
    g = len(ec.members)
    two = len(letter) == 2
    if not two:
        ops = {i: site_operator(spec, letter, (i,), 1.0, states) for i in sites}
        for i, O in ops.items():
            _hermitian_part_or_fail(O, f"{letter}_{i}")
    if pairs is None:
        pairs = [(i, j) for a, i in enumerate(sites) for j in sites[a:]]
    rec = {"letter": letter, "sites": sites, "cluster_size": g, "pairs": [],
           "conjugate_symmetry": "C_ji = conj(C_ij)" if two else "C_ji = C_ij",
           "operator_hypothesis": ("the statement refers to the stored floating-point operator matrix of each pair "
                                   "(exact for dyadic local entries such as spin-1/2 letters; rounded for entries like "
                                   "1/sqrt(2)); docs/proofs/P15_summed_bounds.md (D2)")}
    for i, j in pairs:
        if two:
            O = site_operator(spec, letter, (i, j), 1.0, states)
            r = certify_observable_general(ec, X, O, f"{name}_{i}{j}")
            if r.get("hermitian_decomposition"):
                val, imag = r["value"]
                pr = {"i": i, "j": j, "value": float(val), "imag_value": float(imag), "bound": r["bound"],
                      "hermitian_decomposition": {k: r[k] for k in ("definition", "parts", "parts_formed_exactly", "recombine")}}
                if r.get("status") == "INCONCLUSIVE":
                    pr["status"] = "INCONCLUSIVE"
                    pr["reason"] = r["reason"]
            else:
                pr = {"i": i, "j": j, "value": r["value"], "imag_value": 0.0, "bound": r["bound"]}
        else:
            O = (ops[i] @ ops[j]).tocsr()
            O = ((O + O.conj().T) / 2).tocsr() if i != j else O            # exact for commuting Hermitian factors
            r = certify_observable(ec, X, O, f"{name}_{i}{j}")
            pr = {"i": i, "j": j, "value": r["value"], "bound": r["bound"]}
        pr["per_state_value"] = pr["value"] / g
        pr["per_state_bound"] = float(up(pr["bound"] / g))
        rec["pairs"].append(pr)
    if two:
        rec["statement"] = (f"tr(P {letter[0]}_i {letter[1]}_j) certified for {len(rec['pairs'])} pairs; Theorem 6 per "
                            f"Hermitian part, Hermitian decomposition O = H + iK for i != j (Proposition 15 for the "
                            f"recombination) [derived: docs/proofs/P15_summed_bounds.md; UNVERIFIED by "
                            f"independent re-derivation, D-12]")
    else:
        rec["statement"] = (f"tr(P {letter}_i {letter}_j) certified for {len(rec['pairs'])} pairs; "
                            f"Theorem 6 per pair, Proposition 15 for combinations [derived: "
                            f"docs/proofs/P15_summed_bounds.md; UNVERIFIED by independent re-derivation, D-12]")
    return rec


def structure_factor(corr: dict, positions: Sequence[Sequence[float]], q_list: Sequence[Sequence[float]],
                     n_sites: Optional[int] = None) -> List[dict]:
    """S(q) = (1/N) sum_{i,j} e^{i q.(r_i - r_j)} C_ij with the summed bound of Proposition 15.

    `corr` is the record of certify_correlations (pairs i <= j; C_ji = C_ij). positions[i] is r_i for
    site i (any dimension); N = len(sites) unless n_sites is given. Reports the real part of S(q) per
    state (tr(P .)/g) with the bound (1/N) sum |a| b / g."""
    sites = corr["sites"]
    N = int(n_sites or len(sites))
    g = corr["cluster_size"]
    pos = {int(i): np.asarray(positions[k], dtype=float) for k, i in enumerate(sites)} if len(positions) == len(sites) \
        else {int(i): np.asarray(positions[int(i)], dtype=float) for i in sites}
    table = {}
    for p in corr["pairs"]:
        v = complex(p["value"], p.get("imag_value", 0.0))
        table[(p["i"], p["j"])] = (v, p["bound"])
        table[(p["j"], p["i"])] = (v.conjugate() if corr.get("conjugate_symmetry", "").startswith("C_ji = conj") else v,
                                   p["bound"])
    out = []
    for q in q_list:
        q = np.asarray(q, dtype=float)
        coeffs, vals, bnds = [], [], []
        for i in sites:
            for j in sites:
                if (i, j) not in table:
                    continue
                phase = np.exp(1j * float(np.dot(q, pos[i] - pos[j]))) / N
                v, b = table[(i, j)]
                coeffs.append(phase)
                vals.append(v)
                bnds.append(b)
        val, bound = combine_certified(coeffs, vals, bnds)
        out.append({"q": q.tolist(), "value": val.real, "imag_part": val.imag, "bound": float(bound),
                    "per_state_value": val.real / g, "per_state_bound": float(up(bound / g)),
                    "n_terms": len(coeffs),
                    "phase_hypothesis": ("the coefficients are the floating-point phases exp(i q.(r_i - r_j))/N as "
                                         "computed; their distance to the exact phases rests on the accuracy of the "
                                         "platform's cos/sin (docs/proofs/P15_summed_bounds.md, D1)")})
    return out


def positions_chain(N: int) -> List[List[float]]:
    return [[float(i)] for i in range(N)]


def q_chain(N: int) -> List[List[float]]:
    return [[2 * math.pi * k / N] for k in range(N)]


# -------------------------------------------------------------------------------------------- C-04: sector lift
# A symmetry-adapted claim certifies eigenpairs of the producer's block H_s = V^H H V (S1 audits V: isometry
# defect iota >= ||V^H V - I||, invariance defect >= ||H V - V H_s||). Theorem 6 in the block gives P_s, the spectral
# projector of H_s on the cluster, with ||X X^H - P_s|| <= s + eta. For a full-space operator O the lifted quantity is
#   tr(P_s V^H O V) = tr(Pi_V O),  Pi_V := V P_s V^H          (cyclicity of the trace; exact),
# and with Q the polar factor of V (an exact isometry with the same range) and Pi_Q := Q P_s Q^H (an orthogonal
# projector of rank g onto a subspace of range V):
#   |tr(Pi_V O) - tr(Pi_Q O)| <= 2 g ||O||_2 delta (1 + sqrt(1 + iota)),  delta = 1 - sqrt(1 - iota).
# The compression V^H O V is formed in floating point; its rounding error enters as g ||E||_2 (||P_s||_1 = g).
# Derivation and proof: docs/proofs/C-04_correlation_lift.md.

def compress_operator(V, O):
    """Hermitian part of fl(V^H (O V)) with a bound rnd on ||Otilde_h - V^H O V||_2 for Hermitian O.

    Standard-model analysis (docs/proofs/C-04_correlation_lift.md, Lemma 1): entrywise
    |fl(V^H fl(O V)) - V^H O V| <= gamma_{w_O + w_V} (|V|^T |O| |V|), w_O the row width of O and w_V the column width
    of V; the Hermitian part of an approximation to the Hermitian V^H O V is at least as close; forming it in
    floating point adds at most u |Otilde_fl| entrywise. ||.||_2 <= ||.||_F closes the bound.
    Returns (Otilde_h (csr, exactly Hermitian), rnd, norm_upper) with norm_upper >= ||Otilde_h||_2."""
    V = sp.csr_matrix(V)
    O = sp.csr_matrix(O)
    cplx = np.iscomplexobj(V.data) or np.iscomplexobj(O.data)
    OV = (O @ V).tocsr()
    Ot = (V.conj().T @ OV).tocsr()
    w_O = int(np.diff(O.indptr).max()) if O.shape[0] else 1
    w_V = int(np.diff(sp.csc_matrix(V).indptr).max()) if V.shape[1] else 1
    absprod = (abs(V).T @ (abs(O) @ abs(V))).tocsr()
    # the computed |V|^T |O| |V| underestimates the exact one by at most a relative gamma_{w_O + w_V} entrywise
    g_abs = gamma_dot(w_O + w_V, cplx)
    rnd = g_abs * fro_upper(absprod) * up(1.0 / (1.0 - g_abs))
    Oh = ((Ot + Ot.conj().T) * 0.5).tocsr()
    Oh.eliminate_zeros()
    rnd = up(rnd + U * fro_upper(Ot) * (1 + 4 * U))
    return Oh, float(rnd), float(inf_norm_upper(Oh))


def _isometry_term(g: int, O_norm: float, iota: float) -> float:
    """2 g ||O|| delta (1 + sqrt(1 + iota)), delta = 1 - sqrt(1 - iota) (rounded up); inf when iota >= 1."""
    if not iota < 1:
        return float("inf")
    delta = up(1.0 - math.sqrt(1.0 - iota)) + 4 * U
    return float(up(2 * g * O_norm * delta * (1.0 + math.sqrt(1.0 + iota)) * (1 + 4 * U)))


def certify_observable_lifted(ec: EigenspaceCertificate, X: np.ndarray, V, O, name: str, isometry_defect: float,
                              invariance_defect: Optional[float] = None) -> dict:
    """Certified tr(Pi_Q O) for a full-space operator O through the sector isometry V (C-04).

    Hermitian O: Theorem 6 for the compressed Hermitian Otilde_h = (V^H O V)_h in the block (certify_observable),
    plus g * rnd for the compression rounding (bound for tr(Pi_V O), `bound_V`), plus the isometry term for
    tr(Pi_Q O) (`bound`). Non-Hermitian O: the exact Hermitian decomposition O = H + iK (C-05) in the full space,
    each part lifted, recombined with bound b_H + b_K (Proposition 15); `value` is then [re, im]."""
    Oc = sp.csr_matrix(O)
    g = len(ec.members)
    iota = float(isometry_defect)
    dev = abs(Oc - Oc.conj().T).max() if Oc.nnz else 0.0
    if dev == 0:
        Oh, rnd, Oh_norm = compress_operator(V, Oc)
        r6 = certify_observable(ec, X, Oh, f"{name}.compressed", O_norm=Oh_norm)
        ec.observables.pop(f"{name}.compressed", None)
        O_norm = float(inf_norm_upper(Oc))
        bound_V = float(add_up(r6["bound"], up(g * rnd)))
        iso_term = _isometry_term(g, O_norm, iota)
        bound = float(add_up(bound_V, iso_term))
        rec = {"hermitian": True, "lifted": True, "value": float(r6["value"]), "bound": bound, "bound_V": bound_V,
               "isometry_term": iso_term, "compression_rounding": float(rnd), "O_norm_bound": O_norm,
               "compressed_norm_bound": float(Oh_norm), "theorem6_bound_in_block": float(r6["bound"]),
               "isometry_defect": iota, "invariance_defect": None if invariance_defect is None else float(invariance_defect),
               "statement": (f"tr(Pi_Q {name}) in [{r6['value'] - bound:.15g}, {r6['value'] + bound:.15g}]; "
                             f"tr(Pi_V {name}) in [{r6['value'] - bound_V:.15g}, {r6['value'] + bound_V:.15g}]")}
    else:
        Hp, Kp, exact = hermitian_parts(Oc, name)
        rh = certify_observable_lifted(ec, X, V, Hp, f"{name}.hermitian_part", iota, invariance_defect)
        rk = certify_observable_lifted(ec, X, V, Kp, f"{name}.anti_hermitian_part", iota, invariance_defect)
        for k in (f"{name}.hermitian_part", f"{name}.anti_hermitian_part"):
            ec.observables.pop(k, None)
        rec = {"hermitian": False, "lifted": True, "hermitian_decomposition": True, "max_asymmetry": float(dev),
               "definition": "O = H + iK with H = (O + O^H)/2 and K = (O - O^H)/(2i), both Hermitian",
               "parts": {"hermitian": rh, "anti_hermitian": rk}, "parts_formed_exactly": bool(exact),
               "recombine": "tr(Pi O) = tr(Pi H) + i tr(Pi K); bound b_H + b_K (Proposition 15, a = 1, i)",
               "value": [float(rh["value"]), float(rk["value"])],
               "bound": float(add_up(rh["bound"], rk["bound"])), "bound_V": float(add_up(rh["bound_V"], rk["bound_V"])),
               "isometry_defect": iota, "invariance_defect": None if invariance_defect is None else float(invariance_defect),
               "statement": (f"tr(Pi_Q {name}) = tr(Pi_Q H) + i tr(Pi_Q K) with tr(Pi_Q H) in [{rh['value'] - rh['bound']:.15g}, "
                             f"{rh['value'] + rh['bound']:.15g}] and tr(Pi_Q K) in [{rk['value'] - rk['bound']:.15g}, "
                             f"{rk['value'] + rk['bound']:.15g}]")}
        if not exact:
            rec["status"] = "INCONCLUSIVE"
            rec["reason"] = ("the Hermitian and anti-Hermitian parts could not be formed exactly in floating point; the "
                             "certified values refer to the rounded parts, not to O")
    rec["refers_to"] = ("Pi_Q = Q P_s Q^H, the orthogonal projector onto the lift by the polar factor Q of V of the "
                        "certified H_s-eigenspace (rank g); Pi_V = V P_s V^H for the stored V (exact identity "
                        "tr(P_s V^H O V) = tr(Pi_V O)). Whether range(Pi_Q) is an H-eigenspace rests on S1's invariance "
                        "audit; the lift does not certify that another sector has no eigenvalue in the same level")
    ec.observables[name] = rec
    return rec


def certify_correlations_lifted(ec: EigenspaceCertificate, X: np.ndarray, spec: ModelSpec, V, letter: str,
                                sites: Sequence[int], isometry_defect: float, invariance_defect: Optional[float] = None,
                                name: str = "C", pairs: Optional[Sequence[Tuple[int, int]]] = None) -> dict:
    """Two-point functions C_ij = tr(Pi_Q O_i O_j) for a symmetry-adapted claim (C-04): the site operators are built
    in the full space from the spec and lifted through V (certify_observable_lifted). Record shape as in
    certify_correlations (pairs with value, imag_value, bound, per-state value and bound), plus `bound_V` per pair and
    the isometry and invariance defects."""
    sites = [int(s) for s in sites]
    g = len(ec.members)
    two = len(letter) == 2
    if not two:
        ops = {i: site_operator(spec, letter, (i,), 1.0, None) for i in sites}
        for i, O in ops.items():
            _hermitian_part_or_fail(O, f"{letter}_{i}")
    if pairs is None:
        pairs = [(i, j) for a, i in enumerate(sites) for j in sites[a:]]
    rec = {"letter": letter, "sites": sites, "cluster_size": g, "pairs": [], "lifted": True,
           "isometry_defect": float(isometry_defect),
           "invariance_defect": None if invariance_defect is None else float(invariance_defect),
           "conjugate_symmetry": "C_ji = conj(C_ij)" if two else "C_ji = C_ij"}
    for i, j in pairs:
        if two:
            O = site_operator(spec, letter, (i, j), 1.0, None)
        else:
            O = (ops[i] @ ops[j]).tocsr()
            O = ((O + O.conj().T) / 2).tocsr() if i != j else O            # exact for commuting Hermitian factors
        r = certify_observable_lifted(ec, X, V, O, f"{name}_{i}{j}", isometry_defect, invariance_defect)
        if r.get("hermitian_decomposition"):
            val, imag = r["value"]
            pr = {"i": i, "j": j, "value": float(val), "imag_value": float(imag), "bound": r["bound"], "bound_V": r["bound_V"],
                  "hermitian_decomposition": {k: r[k] for k in ("definition", "parts", "parts_formed_exactly", "recombine")}}
            if r.get("status") == "INCONCLUSIVE":
                pr["status"] = "INCONCLUSIVE"
                pr["reason"] = r["reason"]
        else:
            pr = {"i": i, "j": j, "value": r["value"], "imag_value": 0.0, "bound": r["bound"], "bound_V": r["bound_V"]}
        pr["per_state_value"] = pr["value"] / g
        pr["per_state_bound"] = float(up(pr["bound"] / g))
        rec["pairs"].append(pr)
    rec["statement"] = (f"tr(Pi_Q {letter[0]}_i {letter[-1]}_j) certified for {len(rec['pairs'])} pairs through the sector "
                        f"isometry (Theorem 6 in the block on V^H O V, compression rounding, isometry term "
                        f"2 g ||O|| delta (1 + sqrt(1 + iota)); docs/proofs/C-04_correlation_lift.md)")
    rec["refers_to"] = ("Pi_Q = Q P_s Q^H with Q the polar factor of V: the lift of the certified H_s-eigenspace; its "
                        "identification with an H-eigenspace rests on S1's invariance audit, and other sectors may hold "
                        "eigenvalues of the same level")
    return rec
