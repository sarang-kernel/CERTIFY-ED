"""S4 extensions: batches of certified observables, two-point functions and static structure factors.

Everything here is Theorem 6 (one certified tr(P O) per Hermitian operator O) plus Proposition 15 (a
linear combination of certified traces carries the sum of |coefficient| x bound). Site operators are built
from the declared model's local space with the same bitwise builder that S1 uses, so the operator the
statement refers to is defined by the spec, not by the producer. **UNVERIFIED** (D-12; docs/theory.md 9.2).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .construct import StateSet, build_bitwise, full_state_set
from .eigenspace import EigenspaceCertificate, certify_observable
from .fp import U, add_up, gamma, up
from .spec import ModelSpec, Term

__all__ = ["site_operator", "certify_observables_batch", "certify_correlations", "structure_factor",
           "combine_certified"]


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


def certify_observables_batch(ec: EigenspaceCertificate, X: np.ndarray, operators: Dict[str, object]) -> Dict[str, dict]:
    """Theorem 6 for each named Hermitian operator; records land in ec.observables."""
    out = {}
    for name, O in operators.items():
        Oh = _hermitian_part_or_fail(O, name)
        out[name] = certify_observable(ec, X, Oh, name)
    return out


def combine_certified(coefficients: Sequence[complex], values: Sequence[float], bounds: Sequence[float]) -> Tuple[complex, float]:
    """Proposition 15: (sum a_k v_k, sum |a_k| b_k + rounding of the sum), rounded outward."""
    a = np.asarray(coefficients, dtype=complex)
    v = np.asarray(values, dtype=float)
    b = np.asarray(bounds, dtype=float)
    m = a.size
    val = complex(np.sum(a * v))
    bound = 0.0
    for ak, bk in zip(np.abs(a), b):
        bound = float(add_up(bound, up(ak * bk)))
    # rounding of the computed sum: |fl(sum a_k v_k) - sum a_k v_k| <= gamma_{m+1} sum |a_k||v_k| (complex: x2)
    rnd = 2 * gamma(m + 2) * float(np.sum(np.abs(a) * np.abs(v))) * (1 + 8 * U) if m else 0.0
    return val, float(add_up(bound, up(rnd)))


def certify_correlations(ec: EigenspaceCertificate, X: np.ndarray, spec: ModelSpec, letter: str,
                         sites: Sequence[int], states: Optional[StateSet] = None, name: str = "C",
                         pairs: Optional[Sequence[Tuple[int, int]]] = None) -> dict:
    """Two-point functions C_ij = tr(P O_i O_j) for the single-site letter on the given sites.

    O_i O_j is Hermitian for commuting Hermitian site operators (different sites); for i = j the operator
    is O_i^2. Only i <= j is computed; C_ji = C_ij. Letters must be Hermitian single-site letters ('z',
    'x', 'y', 'n', ...); for '+'/'-' use the symmetrised combinations through `operators`. Per-state values
    are tr(P O)/g with bound b/g (g = cluster size)."""
    sites = [int(s) for s in sites]
    g = len(ec.members)
    ops = {i: site_operator(spec, letter, (i,), 1.0, states) for i in sites}
    for i, O in ops.items():
        _hermitian_part_or_fail(O, f"{letter}_{i}")
    if pairs is None:
        pairs = [(i, j) for a, i in enumerate(sites) for j in sites[a:]]
    rec = {"letter": letter, "sites": sites, "cluster_size": g, "pairs": []}
    for i, j in pairs:
        O = (ops[i] @ ops[j]).tocsr()
        O = ((O + O.conj().T) / 2).tocsr() if i != j else O            # exact for commuting Hermitian factors
        r = certify_observable(ec, X, O, f"{name}_{i}{j}")
        rec["pairs"].append({"i": i, "j": j, "value": r["value"], "bound": r["bound"],
                             "per_state_value": r["value"] / g, "per_state_bound": float(up(r["bound"] / g))})
    rec["statement"] = (f"tr(P {letter}_i {letter}_j) certified for {len(rec['pairs'])} pairs; "
                        f"Theorem 6 per pair, Proposition 15 for combinations [UNVERIFIED]")
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
        table[(p["i"], p["j"])] = (p["value"], p["bound"])
        table[(p["j"], p["i"])] = (p["value"], p["bound"])
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
                    "n_terms": len(coeffs)})
    return out


def positions_chain(N: int) -> List[List[float]]:
    return [[float(i)] for i in range(N)]


def q_chain(N: int) -> List[List[float]]:
    return [[2 * math.pi * k / N] for k in range(N)]
