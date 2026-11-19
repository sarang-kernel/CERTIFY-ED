"""Two independent constructions of the matrix defined by a ModelSpec.

`build_kron`   : sum over couplings of Kronecker products of dense local
                 matrices (LocalSpace.matrix), Jordan-Wigner strings inserted as
                 explicit parity factors. Full Hilbert space only.
`build_bitwise`: acts on integer-encoded basis states digit by digit using the
                 shift decomposition (LocalSpace.shift_components) and explicit
                 sign counting; supports restriction to a subset of basis
                 states (e.g. a U(1) sector) and reports amplitude leaking out
                 of that subset.

The two paths share only the ModelSpec data structure. `cross_construct`
compares them; agreement is the evidence for the statement "the matrix is the
one the spec defines". A producer's matrix (QuSpin, custom code, ...) is then
compared with either reference.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import reduce
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .fp import gamma
from .precision import binary64
from .spec import ModelSpec


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _order_permutation(spec: ModelSpec, n: int) -> Optional[np.ndarray]:
    """Permutation mapping ascending-integer order to the spec's state order."""
    if spec.state_order == "ascending":
        return None
    return np.arange(n - 1, -1, -1)


def _dtype(spec: ModelSpec):
    return np.float64 if spec.is_real() else np.complex128


def _finalize(H: sp.spmatrix, spec: ModelSpec) -> sp.csr_matrix:
    H = H.tocsr()
    H.sum_duplicates()
    H.eliminate_zeros()
    if spec.is_real():
        if np.iscomplexobj(H.data):
            if H.nnz and np.abs(H.data.imag).max() > 0:
                raise AssertionError("spec declared real but matrix has imaginary entries")
            H = H.real.tocsr()
    H.sort_indices()
    return H


# ---------------------------------------------------------------------------
# Path A: Kronecker products
# ---------------------------------------------------------------------------

def build_kron(spec: ModelSpec) -> sp.csr_matrix:
    """Reference matrix from Kronecker products of the dense local matrices (independent of build_bitwise)."""
    N, q, local = spec.n_sites, spec.q, spec.local
    I = sp.identity(q, dtype=complex, format="csr")
    P = sp.csr_matrix(local.local_parity()) if local.is_fermionic else None

    def site_operator(letter: str, site: int) -> sp.csr_matrix:
        o = sp.csr_matrix(local.matrix(letter))
        factors = []
        for l in range(N):
            if l == site:
                factors.append(o)
            elif local.jw_on(letter, l, site):
                factors.append(P)
            else:
                factors.append(I)
        return reduce(lambda a, b: sp.kron(a, b, format="csr"), factors)

    dim = q ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    cache: Dict[Tuple[str, int], sp.csr_matrix] = {}
    for term in spec.terms:
        for c, sites in term.couplings:
            ops = []
            for letter, s in zip(term.op, sites):
                key = (letter, s)
                if key not in cache:
                    cache[key] = site_operator(letter, s)
                ops.append(cache[key])
            prod = reduce(lambda a, b: a @ b, ops)
            H = H + complex(c) * prod
    perm = _order_permutation(spec, dim)
    if perm is not None:
        H = H[perm][:, perm]
    return _finalize(H, spec)


# ---------------------------------------------------------------------------
# Path B: bitwise action on basis states
# ---------------------------------------------------------------------------

@dataclass
class StateSet:
    """A set of basis states (integers, site0_msb encoding) with an index map."""
    states_asc: np.ndarray       # sorted ascending, int64
    order: str                   # 'ascending' | 'descending'
    label: str = "full"

    @property
    def size(self) -> int:
        return int(self.states_asc.size)

    def index_of(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (index, found_mask) for integer states s."""
        pos = np.searchsorted(self.states_asc, s)
        pos_c = np.minimum(pos, self.size - 1)
        found = self.states_asc[pos_c] == s
        idx = pos_c if self.order == "ascending" else (self.size - 1 - pos_c)
        return idx, found

    def states_in_order(self) -> np.ndarray:
        return self.states_asc if self.order == "ascending" else self.states_asc[::-1]


def full_state_set(spec: ModelSpec) -> StateSet:
    """All basis states of the spec (integer encoding, site 0 most significant)."""
    return StateSet(np.arange(spec.dim, dtype=np.int64), spec.state_order, "full")


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.int64)
    while np.any(x):
        c += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    return c


def _apply_string(spec: ModelSpec, op: str, sites: Sequence[int], coeff: complex,
                  s0: np.ndarray):
    """Apply coeff * op_string to each state in s0; yield (s_final, amp) per monomial."""
    N, q, local = spec.n_sites, spec.q, spec.local
    comps = [local.shift_components(letter) for letter in op]
    for choice in itertools.product(*[range(len(c)) for c in comps]):
        s = s0.copy()
        amp = np.full(s0.shape, complex(coeff))
        ok = np.ones(s0.shape, dtype=bool)
        for pos in range(len(op) - 1, -1, -1):          # rightmost acts first
            letter, site = op[pos], sites[pos]
            delta, ampvec = comps[pos][choice[pos]]
            place = q ** (N - 1 - site)
            digit = (s // place) % q
            new_digit = digit + delta
            valid = (new_digit >= 0) & (new_digit < q)
            factor = ampvec[digit]
            if local.jw_letter(letter):
                # (-1)^(occupied sites on the string side); q == 2, site 0 is the most significant bit
                if local.jw_string == "higher":
                    side = s & ((1 << (N - 1 - site)) - 1)     # sites site+1 .. N-1
                else:
                    side = s >> (N - site)                      # sites 0 .. site-1
                factor = factor * np.where(_popcount(side) % 2 == 1, -1.0, 1.0)
            amp = amp * factor
            ok &= valid & (factor != 0)
            s = np.where(valid, s + delta * place, s)
        yield s, amp, ok


def build_bitwise(spec: ModelSpec, states: Optional[StateSet] = None,
                  return_leak: bool = False):
    """Build H restricted to `states` (default: full space) by digit arithmetic.

    Returns the CSR matrix (and, if return_leak, the Frobenius norm of the
    amplitude that maps outside the state set: zero for a conserved quantity).
    """
    st = states if states is not None else full_state_set(spec)
    s_order = st.states_in_order()
    n = st.size
    col_idx = np.arange(n)
    rows, cols, vals = [], [], []
    leak2 = 0.0
    for term in spec.terms:
        for c, sites in term.couplings:
            for s_fin, amp, ok in _apply_string(spec, term.op, sites, c, s_order):
                m = ok & (amp != 0)
                if not np.any(m):
                    continue
                r, found = st.index_of(s_fin[m])
                if not np.all(found):
                    leak2 += float(np.sum(np.abs(amp[m][~found]) ** 2))
                rows.append(r[found])
                cols.append(col_idx[m][found])
                vals.append(amp[m][found])
    if rows:
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
    else:
        H = sp.csr_matrix((n, n), dtype=complex)
    H = _finalize(H, spec)
    if return_leak:
        return H, float(np.sqrt(leak2))
    return H


def hermitian_part(H: sp.spmatrix) -> sp.csr_matrix:
    """H itself when it is exactly Hermitian, else (H + H^H)/2 (one rounding per entry).

    The bitwise builder forms the (i, j) and (j, i) entries from different monomials (a different order of
    the same amplitude factors), so a built matrix with irrational local amplitudes (S >= 1, bosons) can
    differ from its adjoint in the last bit. The statements of Theorems 1, 2 and 8 and Weyl's inequality
    need a Hermitian matrix; the builder shift certify._builder_shift covers the symmetrisation
    (docs/proofs/appA_model-shift.md, Lemma A2). A matrix in a narrower floating format is converted exactly to
    binary64 first, so that the sum is formed in binary64 (precision.py)."""
    H = sp.csr_matrix(binary64(H if sp.issparse(H) else np.asarray(H), "H", keep_integers=True))
    asym = H - H.conj().T
    if asym.nnz == 0 or float(abs(asym).max()) == 0.0:
        return H
    return ((H + H.conj().T) / 2).tocsr()


# ---------------------------------------------------------------------------
# Matrix-free application (for producers that never store H)
# ---------------------------------------------------------------------------

class SpecOperator:
    """Matrix-free H (and |H| entrywise-dominating operator) defined by a spec.

    `matvec` applies H on the full space by the bitwise path without storing a
    matrix. `abs_matvec` applies sum_t |c_t| |O_t| which dominates |H|
    entrywise and is used for rounding-error bounds of matrix-free residuals.
    """

    def __init__(self, spec: ModelSpec, states: Optional[StateSet] = None):
        self.spec = spec
        self.states = states if states is not None else full_state_set(spec)
        self.shape = (self.states.size, self.states.size)
        self.dtype = _dtype(spec)
        self._s = self.states.states_in_order()

    def _apply(self, x: np.ndarray, absolute: bool, transpose: bool = False) -> np.ndarray:
        if transpose and not absolute:
            raise NotImplementedError("transposed application is provided for |H|_abs only")
        x = np.asarray(x)
        out_dtype = np.float64 if (absolute or (self.dtype == np.float64 and not np.iscomplexobj(x))) else np.complex128
        y = np.zeros(x.shape, dtype=out_dtype)
        col = np.arange(self.states.size)
        for term in self.spec.terms:
            for c, sites in term.couplings:
                for s_fin, amp, ok in _apply_string(self.spec, term.op, sites, c, self._s):
                    m = ok & (amp != 0)
                    if not np.any(m):
                        continue
                    r, found = self.states.index_of(s_fin[m])
                    a = amp[m][found]
                    src = col[m][found]
                    rr = r[found]
                    if transpose:          # (|H|_abs)^T x: gather at the target, scatter to the source
                        src, rr = rr, src
                    if absolute:
                        contrib = np.abs(a)[:, None] * np.abs(x[src]).reshape(len(src), -1) if x.ndim > 1 \
                            else np.abs(a) * np.abs(x[src])
                    else:
                        contrib = (a[:, None] * x[src].reshape(len(src), -1)) if x.ndim > 1 else a * x[src]
                        if out_dtype == np.float64:
                            contrib = contrib.real
                    np.add.at(y, rr, contrib)
        return y

    def matvec(self, x):
        return self._apply(x, absolute=False)

    def abs_matvec(self, x):
        return self._apply(x, absolute=True)

    def abs_rmatvec(self, x):
        """(|H|_abs)^T x, with |H|_abs the entrywise sum of absolute values of all contributions."""
        return self._apply(x, absolute=True, transpose=True)

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(self.shape, matvec=self.matvec, dtype=self.dtype)


# ---------------------------------------------------------------------------
# Cross-construction
# ---------------------------------------------------------------------------

def matrix_difference(A: sp.spmatrix, B: sp.spmatrix) -> Dict[str, float]:
    """Entrywise comparison with a rounding allowance for non-dyadic products. Matrices in a narrower floating format
    are converted exactly to binary64 first, so that repeated entries are summed in binary64 (precision.py)."""
    A = sp.csr_matrix(binary64(A if sp.issparse(A) else np.asarray(A), "A", keep_integers=True))
    B = sp.csr_matrix(binary64(B if sp.issparse(B) else np.asarray(B), "B", keep_integers=True))
    if A.shape != B.shape:
        return {"shape_match": False, "max_abs_diff": float("inf"), "pattern_match": False}
    D = (A - B).tocsr()
    D.eliminate_zeros()
    maxdiff = float(np.abs(D.data).max()) if D.nnz else 0.0
    scale = float(max(np.abs(A.data).max() if A.nnz else 0.0, np.abs(B.data).max() if B.nnz else 0.0))
    pattern = (A != 0).astype(np.int8) - (B != 0).astype(np.int8)
    return {"shape_match": True, "max_abs_diff": maxdiff, "max_abs_entry": scale,
            "n_differing_entries": int(D.nnz), "pattern_match": bool(pattern.nnz == 0)}


def product_tolerance(spec: ModelSpec, scale: float) -> float:
    """Allowance for re-association of products of <=len(op)+1 floats per entry
    plus summation of at most n_terms contributions per matrix entry."""
    max_len = max((len(t.op) for t in spec.terms), default=1)
    return gamma(2 * (max_len + 2) + spec.n_terms) * max(scale, 1.0) * spec.n_terms


def cross_construct(spec: ModelSpec, kron_max_dim: int = 2 ** 16) -> Dict[str, object]:
    """Build H by both paths and compare. Returns a report dict."""
    B = build_bitwise(spec)
    report = {"dimension": spec.dim, "nnz": int(B.nnz), "paths": ["bitwise"]}
    if spec.dim <= kron_max_dim:
        A = build_kron(spec)
        report["paths"].append("kron")
        diff = matrix_difference(A, B)
        tol = product_tolerance(spec, diff.get("max_abs_entry", 1.0))
        report.update(diff)
        report["tolerance"] = tol
        report["agree"] = bool(diff["shape_match"] and diff["max_abs_diff"] <= tol)
    else:
        report["agree"] = None
        report["note"] = f"Kronecker path skipped (dim {spec.dim} > {kron_max_dim})"
    return report
