"""Pauli-string Hamiltonians as ModelSpecs (roadmap F-05): plain strings, OpenFermion QubitOperator, Qiskit
SparsePauliOp.

    spec = ModelSpec.from_pauli_strings({"XXI": 0.5, "ZZI": 1.0, "IIZ": -0.3, "III": 0.1})
    spec = ModelSpec.from_openfermion(qubit_operator)      # openfermion.QubitOperator
    spec = ModelSpec.from_qiskit(sparse_pauli_op)          # qiskit.quantum_info.SparsePauliOp

Conventions (fixed here so that S1 compares a producer's matrix against the same operator the library builds):

  * Local space: spin-1/2 in the `pauli` convention (letters x, y, z are sigma^x, sigma^y, sigma^z; `I` is the
    identity). Digit 1 of the certify-ed encoding is spin up (sigma^z = +1), i.e. the qubit state |0>; digit 0 is
    |1>. With `state_order="descending"` and `site_order="site0_msb"`, basis index i is the computational basis
    state whose big-endian bit string is i, with site 0 the most significant bit.
  * A Pauli string is read left to right: character j acts on site j. For OpenFermion strings and
    `QubitOperator` terms (qubit q acts on tensor factor q counted from the left) site = qubit, and the spec
    matrix equals `openfermion.get_sparse_operator(op, n_qubits)` entry by entry. For Qiskit labels (little
    endian: the rightmost character is qubit 0) character j is qubit n-1-j; the spec matrix equals
    `SparsePauliOp.to_matrix()` entry by entry, and the certify-ed site index is n-1-qubit. The spec's
    `parameters["qubit_order"]` records which.
  * Coefficients must be real (a nonzero imaginary part makes the sum non-Hermitian): ValueError otherwise.
    Terms with coefficient exactly 0 are dropped and counted in `parameters["dropped_zero_terms"]`.
  * The all-identity string is kept as the term `("I", [[c, 0]])`: c times the identity.

The third-party packages are never imported here; the readers use the objects' interfaces (`.terms` for a
QubitOperator; `.to_list()` and `.num_qubits` for a SparsePauliOp), which hand-built objects can mimic. This
module builds specs and computes no bound (provenance.NON_COMPUTATIONAL); the matrices come from
certify_ed.construct as for every other spec.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ..localops import LocalSpace
from ..spec import ModelSpec

__all__ = ["pauli_terms_to_static", "spec_from_pauli_terms", "spec_from_openfermion", "spec_from_qiskit",
           "QUBIT_ORDERS"]

QUBIT_ORDERS = ("big_endian", "little_endian")
_LETTERS = {"I": "I", "X": "x", "Y": "y", "Z": "z"}

TermList = Union[Mapping[str, Any], Iterable[Tuple[str, Any]]]


def _real_coefficient(c, label: str) -> float:
    z = complex(c)
    if z.imag != 0.0:
        raise ValueError(f"Pauli term {label!r} has a non-real coefficient {z}: a sum of Pauli strings is Hermitian "
                         f"only with real coefficients")
    if z.real != z.real:
        raise ValueError(f"Pauli term {label!r} has a NaN coefficient")
    return float(z.real)


def pauli_terms_to_static(terms: TermList, n_qubits: Optional[int] = None) -> Tuple[list, int, int]:
    """(static list, n_qubits, dropped_zero_terms) from {string: coefficient} or [(string, coefficient), ...].

    Strings use the letters I X Y Z (lower case accepted); all must have the same length (= n_qubits when given).
    Character j acts on site j. Each string becomes one coupling of the QuSpin-style static list, grouped by the
    operator-letter pattern: 'XXI' -> ['xx', [[c, 0, 1]]], 'III' -> ['I', [[c, 0]]]."""
    items = list(terms.items()) if isinstance(terms, Mapping) else list(terms)
    if not items:
        raise ValueError("no Pauli terms")
    groups: Dict[str, List[list]] = {}
    order: List[str] = []
    n = n_qubits
    dropped = 0
    for label, coeff in items:
        s = str(label).strip().upper()
        if n is None:
            n = len(s)
        if len(s) != n:
            raise ValueError(f"Pauli string {label!r} has length {len(s)}, expected {n}")
        bad = sorted(set(s) - set(_LETTERS))
        if bad:
            raise ValueError(f"Pauli string {label!r}: letters {bad} are not I, X, Y or Z")
        c = _real_coefficient(coeff, s)
        if c == 0.0:
            dropped += 1
            continue
        sites = [j for j, ch in enumerate(s) if ch != "I"]
        op = "".join(_LETTERS[s[j]] for j in sites) if sites else "I"
        if not sites:
            sites = [0]
        if op not in groups:
            groups[op] = []
            order.append(op)
        groups[op].append([c] + sites)
    if n is None or n < 1:
        raise ValueError("n_qubits must be at least 1")
    static = [[op, groups[op]] for op in order]
    if not static:
        raise ValueError("all Pauli coefficients are zero: the operator is 0 and has no spec")
    return static, n, dropped


def spec_from_pauli_terms(terms: TermList, n_qubits: Optional[int] = None, qubit_order: str = "big_endian",
                          name: str = "pauli_strings", parameters: Optional[Dict[str, Any]] = None,
                          notes: str = "") -> ModelSpec:
    """ModelSpec of sum_k c_k P_k for Pauli strings P_k (see the module docstring for the conventions)."""
    if qubit_order not in QUBIT_ORDERS:
        raise ValueError(f"qubit_order must be one of {QUBIT_ORDERS}")
    static, n, dropped = pauli_terms_to_static(terms, n_qubits)
    params = {"source": "pauli_strings", "qubit_order": qubit_order,
              "site_of_qubit": "site = qubit" if qubit_order == "big_endian" else "site = n_qubits - 1 - qubit",
              "n_terms": sum(len(cps) for _op, cps in static), "dropped_zero_terms": dropped,
              "basis": "index i = computational basis state with big-endian bit string i (site 0 = MSB); digit 1 = |0>"}
    params.update(parameters or {})
    return ModelSpec.from_static(n, LocalSpace.spin_space("1/2", "pauli"), static, state_order="descending",
                                 name=name, parameters=params, notes=notes)


def spec_from_openfermion(op, n_qubits: Optional[int] = None, name: str = "openfermion_qubit_operator",
                          **kw) -> ModelSpec:
    """ModelSpec from an OpenFermion QubitOperator (or any object with a `.terms` mapping
    {((qubit, 'X'), (qubit, 'Z'), ...): coefficient}; the empty tuple is the identity). Site = qubit; the matrix
    equals openfermion.get_sparse_operator(op, n_qubits). `n_qubits` defaults to max qubit + 1."""
    terms = getattr(op, "terms", None)
    if not isinstance(terms, Mapping):
        raise TypeError(f"expected a QubitOperator-like object with a .terms mapping (got {type(op).__name__})")
    if n_qubits is None:
        n_qubits = 1 + max((int(q) for t in terms for q, _l in t), default=0)
    strings: List[Tuple[str, Any]] = []
    for t, c in terms.items():
        chars = ["I"] * n_qubits
        for q, letter in t:
            q = int(q)
            if not (0 <= q < n_qubits):
                raise ValueError(f"term {t} acts on qubit {q} outside 0..{n_qubits - 1}")
            if chars[q] != "I":
                raise ValueError(f"term {t} names qubit {q} twice")
            if str(letter).upper() not in ("X", "Y", "Z"):
                raise ValueError(f"term {t}: operator {letter!r} is not X, Y or Z")
            chars[q] = str(letter).upper()
        strings.append(("".join(chars), c))
    return spec_from_pauli_terms(strings, n_qubits, qubit_order="big_endian", name=name,
                                 parameters={"source": "openfermion.QubitOperator", **kw.pop("parameters", {})}, **kw)


def spec_from_qiskit(op, name: str = "qiskit_sparse_pauli_op", **kw) -> ModelSpec:
    """ModelSpec from a Qiskit SparsePauliOp (or any object with `.to_list()` -> [(label, coeff), ...] and
    `.num_qubits`). Labels are little endian (rightmost character = qubit 0): character j is site j and qubit
    n-1-j; the matrix equals op.to_matrix()."""
    to_list = getattr(op, "to_list", None)
    if to_list is None:
        paulis, coeffs = getattr(op, "paulis", None), getattr(op, "coeffs", None)
        if paulis is None or coeffs is None:
            raise TypeError(f"expected a SparsePauliOp-like object with .to_list() (got {type(op).__name__})")
        items = [(str(p), c) for p, c in zip(paulis, coeffs)]
    else:
        items = [(str(label), c) for label, c in to_list()]
    n = getattr(op, "num_qubits", None)
    labels = []
    for label, c in items:
        s = label.strip()
        if s[:1] in "+-":                              # a phase prefix would not be a Pauli string with a real coefficient
            raise ValueError(f"label {label!r} carries a phase prefix; use SparsePauliOp.simplify() / to_list()")
        if s.startswith("i"):
            raise ValueError(f"label {label!r} carries a phase prefix")
        labels.append((s, c))
    return spec_from_pauli_terms(labels, int(n) if n is not None else None, qubit_order="little_endian", name=name,
                                 parameters={"source": "qiskit.quantum_info.SparsePauliOp", **kw.pop("parameters", {})},
                                 **kw)

