"""Matrix-free spec operator with an a-priori rounding count and a spec norm bound (D-02, docs/proofs/D-02_matrix-free.md).

`MatrixFreeOperator(spec, states)` applies H term by term from the spec, exactly as `construct.SpecOperator`
does (same monomial passes, same arithmetic), and adds what the Theorem 1 accounting of `enclosure.residuals`
needs when no matrix is ever stored:

* `rounding_width`  -- W such that |fl(H x)_i - (H_spec x)_i| <= gamma_W (|H|_abs |x|)_i for every stored x, where
  H_spec is the EXACT spec operator (exact couplings as stored, exact local amplitudes), so that no separate model
  shift is needed: the roundings of the local amplitudes (spin >= 1, bosons: square roots) are counted here.
  W = w_row + L + rho + 1 (real data), where w_row bounds the number of monomial contributions that can land in
  one output entry (combinatorial count from the spec, section 2 of the document), L the longest string, rho the
  number of letters per contribution whose stored amplitudes are not exact (one rounding each), and the +1 the
  product with x_j. Complex data: W = w_row + 3 L + rho + 3 (complex products cost gamma_3 each in modulus).
  The adapter uses width = W - 2 so that `gamma_dot(width + 2)` of `_rounding_bound` equals gamma_W (real) or
  2 gamma_{W+4} >= gamma_W (complex).
* `norm_bound()`    -- ||H_spec||_2 <= sum_t |c_t| prod_l ||o_l||_2, with each local norm bounded by
  sqrt(||o||_1 ||o||_inf) of the exact local matrix (Jordan-Wigner strings are unitary and do not change norms).
* `amplitude_hypothesis` -- the named hypothesis recorded in the certificate: which letters are exact.

The per-row count w_row is exact combinatorics on the spec (no floating point): for one coupling with string
o_1 ... o_L on distinct sites, a target state receives at most prod_l c(o_l) contributions, c(o) = max over the
target digit k of the number of shift components (delta, amp) of o with a valid source digit k - delta and
amp[k - delta] != 0; summed over couplings. For spin-1/2 every letter has c = 1, so w_row = n_terms exactly
(a Pauli string maps each basis state to one basis state); the 3.1 width n_terms 2^L + 2L + 2 was 2^L times
larger. Repeated sites in one string (allowed by the spec) are counted with c(o) per letter, an upper bound.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, Optional

import numpy as np

from .construct import SpecOperator, StateSet, _apply_string
from .fp import up, _GUARD
from .spec import ModelSpec


def _exact_components(local, letter: str):
    """Exact (delta, [(re, im)]) components of the UNscaled letter, or None when a component is irrational.
    Spin: amplitudes sqrt(S(S+1) - m(m+-1)) times the convention scale; fermion/boson: see localops."""
    q = local.q
    comps = local.shift_components(letter)
    out = []
    for delta, amp in comps:
        ex = []
        for k in range(q):
            a = complex(amp[k])
            if a == 0:
                ex.append((Fraction(0), Fraction(0)))
                continue
            if local.kind == "spin":
                S = Fraction(local.spin)
                m = k - S
                a2 = S * (S + 1) - m * (m + delta) if delta != 0 else None
                if delta == 0:
                    v = m * (2 if local.convention in ("pauli", "pauli_pm1") else 1)
                    ex.append((v, Fraction(0)))
                    continue
                scale2 = Fraction(1)
                if letter in "+-":
                    scale2 = Fraction(4) if local.convention == "pauli" else Fraction(1)
                else:   # x, y: half of the ladder amplitude times the xyz scale
                    xyz = Fraction(2) if local.convention in ("pauli", "pauli_pm1") else Fraction(1)
                    scale2 = (xyz / 2) ** 2
                mag2 = a2 * scale2
                r = Fraction(a.real), Fraction(a.imag)
                if r[0] ** 2 + r[1] ** 2 != mag2:
                    return None            # irrational amplitude, stored rounded
                ex.append(r)
            else:
                # fermion letters are 0/1/+-1/2 (exact); bosons: sqrt(k) exact only for perfect squares
                r = Fraction(a.real), Fraction(a.imag)
                if local.kind == "boson" and letter in "+-":
                    kk = k + 1 if delta > 0 else k
                    if r[0] ** 2 + r[1] ** 2 != Fraction(kk):
                        return None
                ex.append(r)
        out.append((delta, ex))
    return out


def letter_counts(local, letter: str) -> int:
    """c(o): the largest number of shift components of `letter` that can contribute to one target digit."""
    q = local.q
    comps = local.shift_components(letter)
    best = 0
    for k in range(q):
        c = 0
        for delta, amp in comps:
            src = k - delta
            if 0 <= src < q and amp[src] != 0:
                c += 1
        best = max(best, c)
    return best


def local_norm_bound(local, letter: str) -> float:
    """Upper bound on ||o||_2 from sqrt(||o||_1 ||o||_inf) of the exact local matrix (moduli rounded up)."""
    M = np.abs(local.matrix(letter))
    rows = M.sum(axis=1).max()
    cols = M.sum(axis=0).max()
    return up(math.sqrt(float(rows) * float(cols)) * (1 + 8 * 2.0 ** -53))


class MatrixFreeOperator(SpecOperator):
    """SpecOperator with the D-02 a-priori rounding width, the spec norm bound and the amplitude hypothesis."""

    def __init__(self, spec: ModelSpec, states: Optional[StateSet] = None):
        super().__init__(spec, states)
        local = spec.local
        letters = sorted({ch for t in spec.terms for ch in t.op})
        self.exact_letters: Dict[str, bool] = {}
        for ch in letters:
            self.exact_letters[ch] = _exact_components(local, ch) is not None
        self.max_len = max((len(t.op) for t in spec.terms), default=0)
        # rho: the largest number of rounded amplitudes in one contribution (letters of one string that are inexact)
        self.rho = max((sum(0 if self.exact_letters[ch] else 1 for ch in t.op) for t in spec.terms), default=0)
        counts = {ch: letter_counts(local, ch) for ch in letters}
        self.terms_per_row = int(sum(math.prod(counts[ch] for ch in t.op) * len(t.couplings) for t in spec.terms))
        cplx = self.dtype == np.complex128
        L = self.max_len
        self.W = self.terms_per_row + (3 * L + self.rho + 3 if cplx else L + self.rho + 1)
        self.rounding_width = max(1, self.W - 2)
        norms = {ch: local_norm_bound(local, ch) for ch in letters}
        s = 0.0
        for t in spec.terms:
            ht = math.prod(norms[ch] for ch in t.op) if t.op else 1.0
            for c, _sites in t.couplings:
                s += abs(complex(c)) * ht
        # the sum of n_terms non-negative products: (1 + gamma_{n_terms + L + 2}) covers its roundings
        from .fp import gamma
        self._norm = up(s * (1.0 + gamma(max(1, spec.n_terms + L + 2)))) if spec.n_terms else 0.0
        self._full = self.states.label == "full" and bool(np.array_equal(self.states.states_asc, np.arange(self.states.size)))
        self._desc = self.states.order == "descending"

    def norm_bound(self) -> float:
        """||H_spec||_2 <= sum_t |c_t| prod_l ||o_l||_2 (D-02, Lemma N)."""
        return self._norm

    @property
    def amplitude_hypothesis(self) -> dict:
        return {"exact_letters": {k: bool(v) for k, v in self.exact_letters.items()},
                "rounded_amplitudes_per_contribution": int(self.rho),
                "terms_per_row": int(self.terms_per_row), "max_string_length": int(self.max_len),
                "rounding_width_W": int(self.W),
                "statement": ("every local amplitude of the letters " + ",".join(k for k, v in self.exact_letters.items() if v)
                              + " is exactly representable in binary64"
                              + (("; the letters " + ",".join(k for k, v in self.exact_letters.items() if not v)
                                  + " have amplitudes rounded once each, counted in W") if self.rho else ""))}

    # the application itself: as SpecOperator._apply, with two changes that do not alter any rounding: the full
    # state set needs no searchsorted (the state is its own index), and within one monomial pass the targets are
    # distinct, so y[rr] += contrib (one rounded addition per target) replaces np.add.at
    def _apply(self, x: np.ndarray, absolute: bool, transpose: bool = False) -> np.ndarray:
        if transpose and not absolute:
            raise NotImplementedError("transposed application is provided for |H|_abs only")
        x = np.asarray(x)
        out_dtype = np.float64 if (absolute or (self.dtype == np.float64 and not np.iscomplexobj(x))) else np.complex128
        y = np.zeros(x.shape, dtype=out_dtype)
        n = self.states.size
        col = np.arange(n)
        for term in self.spec.terms:
            for c, sites in term.couplings:
                for s_fin, amp, ok in _apply_string(self.spec, term.op, sites, c, self._s):
                    m = ok & (amp != 0)
                    if not np.any(m):
                        continue
                    if self._full:        # the state is its own index (reversed for descending order)
                        rr = (n - 1 - s_fin[m]) if self._desc else s_fin[m]
                        src = col[m]
                        a = amp[m]
                    else:
                        r, found = self.states.index_of(s_fin[m])
                        a = amp[m][found]
                        src = col[m][found]
                        rr = r[found]
                    if transpose:
                        src, rr = rr, src
                    if absolute:
                        contrib = np.abs(a)[:, None] * np.abs(x[src]).reshape(len(src), -1) if x.ndim > 1 \
                            else np.abs(a) * np.abs(x[src])
                    else:
                        contrib = (a[:, None] * x[src].reshape(len(src), -1)) if x.ndim > 1 else a * x[src]
                        if out_dtype == np.float64:
                            contrib = contrib.real
                    y[rr] += contrib
        return y


def matrix_free_operator(spec: ModelSpec, states: Optional[StateSet] = None) -> MatrixFreeOperator:
    return MatrixFreeOperator(spec, states)


__all__ = ["MatrixFreeOperator", "matrix_free_operator", "letter_counts", "local_norm_bound", "_GUARD"]
