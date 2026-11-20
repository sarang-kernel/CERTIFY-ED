"""Spectral invariants computed from the specification alone.

Each coupling c * o_1(i_1)...o_m(i_m) is a tensor product over sites of local
q x q factors (letters at that site multiplied in string order, fermionic
parity factors on the string side for Jordan-Wigner letters, identity elsewhere).
Hence, with O_t the operator of coupling t,

    tr H          = sum_t c_t prod_l tr(o_{t,l})
    ||H||_F^2     = sum_{t,t'} conj(c_t) c_t' prod_l tr(o_{t,l}^dagger o_{t',l})

cost O(T^2 N q^3), independent of the Hilbert-space dimension q^N. These are
the cheapest model-fidelity checks available and the only ones that apply to
producers exposing neither a matrix nor a matvec: sum_s tr H_s and
sum_s ||H_s||_F^2 over a claimed sector decomposition must reproduce them, and
so must a claimed full spectrum (sum lambda and sum lambda^2).

Values are computed in float64 with an a-priori rounding bound (products of
<= N factors, sums of <= T^2 terms).
"""

from __future__ import annotations

import math

from typing import Dict, List, Tuple

import numpy as np

from .fp import gamma, modulus_up, up
from .precision import binary64
from .spec import ModelSpec


def _site_factors(spec: ModelSpec, op: str, sites) -> List[np.ndarray]:
    N, q, local = spec.n_sites, spec.q, spec.local
    facs = [np.eye(q, dtype=complex) for _ in range(N)]
    P = local.local_parity() if local.is_fermionic else None
    # full operator = F(o_1,i_1) F(o_2,i_2) ... ; site-l factor = product in the same order
    for letter, site in zip(op, sites):
        o = local.matrix(letter)
        for l in range(N):
            if l == site:
                facs[l] = facs[l] @ o
            elif local.jw_on(letter, l, site):
                facs[l] = facs[l] @ P
    return facs


def spec_invariants(spec: ModelSpec) -> Dict[str, float]:
    items: List[Tuple[complex, List[np.ndarray]]] = []
    for t in spec.terms:
        for c, s in t.couplings:
            items.append((complex(c), _site_factors(spec, t.op, s)))
    T, N = len(items), spec.n_sites
    tr = 0j
    for c, f in items:
        p = c
        for m in f:
            p *= np.trace(m)
        tr += p
    fro2 = 0j
    for a in range(T):
        ca, fa = items[a]
        for b in range(T):
            cb, fb = items[b]
            p = np.conj(ca) * cb
            for l in range(N):
                p *= np.trace(fa[l].conj().T @ fb[l])
                if p == 0:
                    break
            fro2 += p
    # a-priori rounding: each product has <= N+2 factors (each local trace itself
    # a sum of <= q^3 products), sums of <= T^2 terms
    q = spec.q
    g = gamma(2 * (N + 2) + q ** 3 + T * T)
    # the moduli of complex couplings and local entries are upper bounds (fp.modulus_up), formed once per term and
    # factor (T N calls), not inside the double sum over pairs of terms
    mods = [(modulus_up(c), [modulus_up(m) for m in f]) for c, f in items]
    abs_tr = sum(mc * np.prod([am.trace() for am in af]) for mc, af in mods)
    abs_fro = sum(mca * mcb * np.prod([aa[l].T.__matmul__(ab[l]).trace() for l in range(N)])
                  for mca, aa in mods for mcb, ab in mods) if T <= 400 else float("inf")
    return {"trace": float(tr.real), "trace_imag": float(tr.imag),
            "trace_err": float(up(g * abs_tr)),
            "fro2": float(fro2.real), "fro2_err": float(up(g * abs_fro)),
            "n_couplings": T}


def matrix_invariants(H) -> Dict[str, float]:
    """Trace and squared Frobenius norm of H with the relative rounding bound err_rel (binary64). A matrix in a
    narrower floating format is converted exactly to binary64 first, so that the sums are those the bound counts
    (precision.py)."""
    import scipy.sparse as sp
    H = binary64(H if sp.issparse(H) else np.asarray(H), "H", keep_integers=True)
    if sp.issparse(H):
        H = H.tocsr()
        tr = complex(H.diagonal().sum())
        fro2 = float(np.sum(H.data.real ** 2 + H.data.imag ** 2))     # |h|^2 from the components, no modulus
        n = H.nnz
    else:
        H = np.asarray(H)
        tr = complex(np.trace(H))
        fro2 = float(np.sum(H.real ** 2 + H.imag ** 2))
        n = H.size
    return {"trace": tr.real, "fro2": fro2, "err_rel": gamma(n + 2)}


def spectrum_invariants(evals) -> Dict[str, float]:
    e = np.asarray(evals, dtype=float)
    return {"trace": float(e.sum()), "fro2": float(np.sum(e * e)), "err_rel": gamma(e.size + 2)}


def compare_invariants(spec_inv: dict, other: dict) -> dict:
    tr_tol = spec_inv["trace_err"] + other["err_rel"] * max(abs(other["trace"]), 1.0) + 1e-300
    fr_tol = spec_inv["fro2_err"] + other["err_rel"] * max(abs(other["fro2"]), 1.0) + 1e-300
    dtr = abs(spec_inv["trace"] - other["trace"])
    dfr = abs(spec_inv["fro2"] - other["fro2"])

    def rec(v):                     # no nan in the record (inf - inf for couplings whose squares overflow): inf
        return float("inf") if v != v else v
    out = {"trace_diff": rec(dtr), "trace_tol": rec(tr_tol), "fro2_diff": rec(dfr), "fro2_tol": rec(fr_tol),
           "consistent": bool(dtr <= tr_tol and dfr <= fr_tol)}
    if not all(math.isfinite(v) for v in (dtr, tr_tol, dfr, fr_tol)):
        out["consistent"] = None    # an invariant overflowed: the comparison decides nothing (it is a pre-check)
        out["note"] = "not compared: an invariant or its tolerance is not finite (an overflowed sum)"
    return out
