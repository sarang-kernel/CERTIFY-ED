"""Operator ergonomics (roadmap C-15): observables and correlations written as text or operator lists.

Every form below is converted to the sparse operator in the claim space that `Options.observables` and the
S4 machinery already accept, built from the declared model's local space by the same bitwise builder S1
uses (`correlations.site_operator`), so the statement refers to an operator defined by the spec. Accepted:

  Pauli strings       "XIZY"                one letter per site (I X Y Z; site 0 first), sigma matrices
                      "X0 Z2"               letter followed by the site index; omitted sites are identity
  site-index letters  "Sz_0 Sz_3", "Sx_1", "S+_0 S-_3", "n_0 n_1", "+_0 -_3"
                      (S<letter>_<site> for spins, <letter>_<site> for any local space; a product of factors)
  linear combinations "0.5 S+_0 S-_3 + 0.5 S-_0 S+_3", "Sz_0 Sz_1 - 0.25 I"
  dictionaries        {"letter": "z", "sites": [0, 3]}             the product z_0 z_3
                      {"op": "zz", "sites": [0, 3], "coeff": 1.0}
  QuSpin operator lists  [["zz", [[J, 0, 1], [J, 1, 2]]], ["x", [[h, 0]]]]
  matrices            dense / scipy.sparse, passed through unchanged (as before)

Pauli letters are the Pauli matrices regardless of the spec's spin convention (sigma = 2 S in the "spin"
convention); they need a spin-1/2 local space. Non-Hermitian results are refused with the message of
`correlations._hermitian_part_or_fail` (Theorem 6 needs a Hermitian operator): "S+_0 S-_3" alone is refused,
its symmetrised form "0.5 S+_0 S-_3 + 0.5 S-_0 S+_3" is accepted. The module decides which operator a
statement refers to (its text is recorded in the certificate's scope options); the numbers come from
build_bitwise and certify_observable.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .construct import StateSet, build_bitwise, full_state_set
from .spec import ModelSpec, Term

__all__ = ["parse_operator", "operator_from_static", "resolve_observables", "resolve_correlation_request",
           "OperatorSyntaxError"]

_PAULI = {"I": "I", "X": "x", "Y": "y", "Z": "z"}
_TOKEN = re.compile(r"^(?:S)?([xyzXYZ+\-nI])_?(\d+)$")           # Sz_0, S+_3, z_1, n_0, X0 (Pauli compact)
_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[jJ]?$")


class OperatorSyntaxError(ValueError):
    """The operator text could not be read; the message says which token and what is accepted."""


def _pauli_scale(spec: ModelSpec, letter: str) -> float:
    """Factor turning the spec's letter into the Pauli matrix: 2 for S^{x,y,z} in the 'spin' convention,
    2 for +/- there as well (sigma^+ = 2 S^+), 1 where the letter already is the Pauli matrix."""
    conv = spec.local.convention
    if conv == "pauli":
        return 1.0
    if conv == "pauli_pm1":
        return 1.0 if letter in "xyz" else 2.0
    return 2.0


def _require_spin_half(spec: ModelSpec, what: str):
    if not (spec.local.kind == "spin" and spec.local.q == 2):
        raise OperatorSyntaxError(f"{what}: Pauli letters need a spin-1/2 local space (this model: "
                                  f"{spec.local.kind}, q = {spec.local.q}); use the letter_site form instead")


def _split_terms(text: str) -> List[Tuple[float, str]]:
    """'a - b + c' -> [(+1, 'a'), (-1, 'b'), (+1, 'c')]; a leading sign is allowed; '+'/'-' glued to a site
    letter (S+_0) are not separators."""
    out = []
    sign, buf = 1.0, ""
    i = 0
    s = text.strip()
    while i < len(s):
        ch = s[i]
        if ch in "+-" and (i + 1 < len(s) and s[i + 1] in " \t") and (i == 0 or s[i - 1] in " \t"):
            if buf.strip():
                out.append((sign, buf.strip()))
            sign, buf = (1.0 if ch == "+" else -1.0), ""
            i += 1
            continue
        buf += ch
        i += 1
    if buf.strip():
        out.append((sign, buf.strip()))
    return out


def _parse_factor_tokens(tokens: Sequence[str], spec: ModelSpec, text: str) -> Tuple[complex, str, Tuple[int, ...]]:
    """Tokens of one product term -> (coefficient, letters, sites)."""
    coeff = 1.0 + 0j
    letters, sites = "", []
    for tok in tokens:
        t = tok.strip().rstrip("*")
        if not t:
            continue
        if _NUMBER.match(t):
            coeff *= complex(t)
            continue
        if t == "I":
            continue
        m = _TOKEN.match(t)
        if m is None:
            raise OperatorSyntaxError(f"{text!r}: cannot read {tok!r}; accepted factors are a number, I, "
                                      "S<x|y|z|+|->_<site>, <letter>_<site> or a Pauli letter with a site index (X0)")
        letter, site = m.group(1), int(m.group(2))
        if letter in "XYZ":                      # upper case: the Pauli matrix on that site
            _require_spin_half(spec, text)
            coeff *= _pauli_scale(spec, letter.lower())
            letter = letter.lower()
        if letter not in spec.local.letters:
            raise OperatorSyntaxError(f"{text!r}: letter {letter!r} is not defined for the {spec.local.kind} local "
                                      f"space (letters: {', '.join(spec.local.letters)})")
        if not 0 <= site < spec.n_sites:
            raise OperatorSyntaxError(f"{text!r}: site {site} out of range for {spec.n_sites} sites")
        letters += letter
        sites.append(site)
    return coeff, letters, tuple(sites)


def _parse_pauli_string(word: str, spec: ModelSpec, text: str) -> Tuple[complex, str, Tuple[int, ...]]:
    if len(word) != spec.n_sites:
        raise OperatorSyntaxError(f"{text!r}: a Pauli string needs one letter per site ({spec.n_sites}), got "
                                  f"{len(word)}")
    _require_spin_half(spec, text)
    coeff = 1.0 + 0j
    letters, sites = "", []
    for site, ch in enumerate(word):
        if ch == "I":
            continue
        coeff *= _pauli_scale(spec, _PAULI[ch])
        letters += _PAULI[ch]
        sites.append(site)
    return coeff, letters, tuple(sites)


def parse_terms(text: str, spec: ModelSpec) -> List[Tuple[complex, str, Tuple[int, ...]]]:
    """Operator text -> list of (coefficient, letters, sites) in the spec's letters."""
    if not isinstance(text, str) or not text.strip():
        raise OperatorSyntaxError(f"empty operator text {text!r}")
    out = []
    for sign, term in _split_terms(text):
        tokens = [t for t in re.split(r"[\s*]+", term) if t]
        words = [t for t in tokens if not _NUMBER.match(t)]
        if len(words) == 1 and re.fullmatch(r"[IXYZ]+", words[0]) and (len(words[0]) > 1 or spec.n_sites == 1) \
                and len(words[0]) == spec.n_sites:
            c, letters, sites = _parse_pauli_string(words[0], spec, text)
            for t in tokens:
                if _NUMBER.match(t):
                    c *= complex(t)
        else:
            c, letters, sites = _parse_factor_tokens(tokens, spec, text)
        out.append((sign * c, letters, sites))
    return out


def _build(spec: ModelSpec, terms: Sequence[Tuple[complex, str, Tuple[int, ...]]], states: Optional[StateSet],
           name: str) -> sp.csr_matrix:
    st = states if states is not None else full_state_set(spec)
    n = st.size
    O = sp.csr_matrix((n, n), dtype=complex)
    for c, letters, sites in terms:
        if letters == "":
            O = O + complex(c) * sp.identity(n, dtype=complex, format="csr")
            continue
        sub = ModelSpec(spec.n_sites, spec.local, [Term(letters, [(complex(c), tuple(sites))])],
                        state_order=spec.state_order)
        M, leak = build_bitwise(sub, st, return_leak=True)
        if leak > 0:
            raise ValueError(f"{name}: operator {letters!r} on {tuple(sites)} leaves the producer's state set "
                             f"(leak {leak:.2e})")
        O = O + sp.csr_matrix(M, dtype=complex)
    O = O.tocsr()
    O.sum_duplicates()
    O.eliminate_zeros()
    if O.nnz == 0 or np.abs(O.data.imag).max() == 0:
        O = sp.csr_matrix(O.real)
    O.sort_indices()
    return O


def operator_from_static(spec: ModelSpec, static: Sequence, states: Optional[StateSet] = None,
                         name: str = "operator") -> sp.csr_matrix:
    """QuSpin-style operator list [["zz", [[J, i, j], ...]], ...] -> sparse operator in the claim space."""
    sub = ModelSpec.from_static(spec.n_sites, spec.local, static, state_order=spec.state_order)
    terms = [(c, t.op, s) for t in sub.terms for c, s in t.couplings]
    return _build(spec, terms, states, name)


def parse_operator(expr: Any, spec: ModelSpec, states: Optional[StateSet] = None, name: str = "operator",
                   hermitian: bool = True):
    """Any accepted form (module docstring) -> sparse operator in the claim space; matrices pass through.
    With hermitian=True a non-Hermitian result is refused (Theorem 6 needs Hermitian O)."""
    if sp.issparse(expr) or isinstance(expr, np.ndarray):
        return expr
    if spec is None:
        raise ValueError(f"{name}: operator text or lists need a model specification (spec) to define the letters")
    if isinstance(expr, str):
        O = _build(spec, parse_terms(expr, spec), states, name)
    elif isinstance(expr, dict):
        if "op" in expr:
            letters, sites = str(expr["op"]), tuple(int(s) for s in expr["sites"])
        elif "letter" in expr:
            sites = tuple(int(s) for s in expr["sites"])
            letters = str(expr["letter"]) * len(sites)
        else:
            raise OperatorSyntaxError(f"{name}: a dictionary operator needs 'op' + 'sites' or 'letter' + 'sites'")
        if len(letters) != len(sites):
            raise OperatorSyntaxError(f"{name}: {letters!r} needs {len(letters)} sites, got {len(sites)}")
        O = _build(spec, [(complex(expr.get("coeff", 1.0)), letters, sites)], states, name)
    elif isinstance(expr, (list, tuple)):
        O = operator_from_static(spec, expr, states, name)
    else:
        raise OperatorSyntaxError(f"{name}: unsupported operator form {type(expr).__name__}")
    if hermitian:
        from .correlations import _hermitian_part_or_fail
        O = _hermitian_part_or_fail(O, name)
    return O


def resolve_observables(spec: Optional[ModelSpec], observables: Dict[str, Any],
                        states: Optional[StateSet] = None) -> Dict[str, Any]:
    """Options.observables with every text / dict / list entry converted to a sparse operator."""
    return {name: parse_operator(expr, spec, states, name) for name, expr in (observables or {}).items()}


def resolve_correlation_request(spec: Optional[ModelSpec], name: str, req: Any) -> dict:
    """Options.correlations entry -> the {'letter', 'sites', ...} dictionary correlations.py works with.

    A string "Sz_0 Sz_3" / "z_0 z_3" names one Hermitian single-site letter on a list of sites; mixed letters
    or non-Hermitian letters ('+', '-') are refused: the two-point function tr(P O_i O_j) needs commuting
    Hermitian factors (Proposition 15), and "S+_0 S-_3" is not a Hermitian product."""
    if isinstance(req, dict):
        return req
    if not isinstance(req, str):
        raise OperatorSyntaxError(f"{name}: a correlation request is a dictionary or a string of letter_site factors")
    if spec is None:
        raise ValueError(f"{name}: operator text needs a model specification (spec) to define the letters")
    terms = parse_terms(req, spec)
    if len(terms) != 1:
        raise OperatorSyntaxError(f"{name}: a correlation request names one letter on several sites, not a sum")
    c, letters, sites = terms[0]
    if any(ch in "+-" for ch in letters):
        raise ValueError(f"{name}: operator is not Hermitian ({letters!r} has +/- factors); Theorem 6 needs Hermitian "
                         f"O. Request the symmetrised product as an observable: '0.5 S+_i S-_j + 0.5 S-_i S+_j'")
    if not letters or len(set(letters)) != 1:
        raise OperatorSyntaxError(f"{name}: a correlation request uses one letter on every site (got {letters!r})")
    letter = letters[0]
    if c != 1:
        raise OperatorSyntaxError(f"{name}: a correlation request carries no coefficient (got {c})")
    return {"letter": letter, "sites": list(sites)}
