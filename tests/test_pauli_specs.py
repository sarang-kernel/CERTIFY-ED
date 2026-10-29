"""F-05: ModelSpec from Pauli strings, OpenFermion QubitOperator and Qiskit SparsePauliOp.

The spec matrix must equal (entry by entry) the matrix the library convention defines: numpy Kronecker products
of the Pauli matrices with site 0 the leftmost factor (OpenFermion: qubit 0 leftmost; Qiskit: qubit n-1 leftmost,
which is what its little-endian labels read left to right give). The transverse-field Ising spec from strings
equals `tfim_chain`; a 2-qubit H2-like Hamiltonian certifies with kind='all' against numpy. OpenFermion and
Qiskit are not installed in the test environment: hand-built objects mimic their interfaces; when a package is
present the same tests run against the real objects."""
import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.integrations import pauli as cp
from certify_ed.spec import ModelSpec

I2 = np.eye(2)
PAULI = {"I": I2, "X": np.array([[0, 1], [1, 0]], dtype=complex), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0, -1.0]).astype(complex)}

# H2 at R = 0.735 A in the Bravyi-Kitaev-reduced 2-qubit form (O'Malley et al. 2016, Table I; rounded):
# the test compares against numpy, so the values only need to be a Hermitian 2-qubit operator with Y terms.
H2_TERMS = {"II": -0.4804, "ZI": 0.3435, "IZ": -0.4347, "ZZ": 0.5716, "XX": 0.0910, "YY": 0.0910}


def _matrix(terms, n=None):
    """sum_k c_k P_k by Kronecker products, character j the j-th factor from the left."""
    n = n or len(next(iter(terms)))
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for s, c in (terms.items() if isinstance(terms, dict) else terms):
        M = np.array([[1.0 + 0j]])
        for ch in s.upper():
            M = np.kron(M, PAULI[ch])
        H += complex(c) * M
    return H


def _status(cert, sid):
    return {s["id"].split(".")[0]: s["status"] for s in cert.statements}[sid]


def _tfim_terms(N, J, h):
    terms = {}
    for i in range(N):
        s = ["I"] * N
        s[i] = "Z"
        s[(i + 1) % N] = "Z"
        terms["".join(s)] = -J
    for i in range(N):
        s = ["I"] * N
        s[i] = "X"
        terms["".join(s)] = -h
    return terms


# ---------------------------------------------------------------------------------------- strings
@pytest.mark.parametrize("N,J,h", [(4, 1.0, 0.7), (5, 0.8, 1.3)])
def test_tfim_from_pauli_strings_equals_tfim_spec(N, J, h):
    spec = ModelSpec.from_pauli_strings(_tfim_terms(N, J, h))
    ref = ce.tfim_chain(N, J=J, h=h)
    assert spec.local == ref.local and spec.n_sites == N and spec.state_order == ref.state_order
    A = ce.build_bitwise(spec)
    B = ce.build_bitwise(ref)
    assert abs(A - B).max() == 0.0
    assert abs(ce.build_kron(spec) - B).max() == 0.0
    assert abs(A.toarray() - _matrix(_tfim_terms(N, J, h))).max() == 0.0
    assert spec.parameters["qubit_order"] == "big_endian" and spec.parameters["n_terms"] == 2 * N


def test_h2_like_two_qubit_hamiltonian_certifies_all(tmp_path):
    spec = ModelSpec.from_pauli_strings(H2_TERMS, name="H2 BK 2-qubit")
    Href = _matrix(H2_TERMS)
    assert np.abs(Href - Href.conj().T).max() == 0.0
    assert spec.is_real()                                       # YY: an even number of y letters, real matrix
    A = ce.build_bitwise(spec).toarray()
    assert np.abs(A - Href).max() == 0.0
    w, V = la.eigh(Href)
    assert np.allclose(w, np.linalg.eigvalsh(Href))
    cert = ce.certify(ce.Claim(w, V, operator=Href, kind="all"), spec)
    assert _status(cert, "S1") == "CERTIFIED", cert.summary()
    assert _status(cert, "S2") == "CERTIFIED"
    assert _status(cert, "S3") == "CERTIFIED"
    assert cert.body["subject"]["spec"]["name"] == "H2 BK 2-qubit"
    # and through the spec's own matrix (no producer operator)
    cert2 = ce.certify(ce.Claim(w, V, kind="all"), spec)
    assert _status(cert2, "S2") == "CERTIFIED"


def test_complex_pauli_hamiltonian_certifies():
    terms = {"XYI": 0.5, "YXI": -0.5, "ZZI": 1.0, "IIX": 0.3, "IZZ": 0.4, "YIY": 0.2}
    spec = ModelSpec.from_pauli_strings(terms)
    Href = _matrix(terms)
    H = ce.build_bitwise(spec)
    assert np.abs(H.toarray() - Href).max() == 0.0 and np.iscomplexobj(H.toarray())
    w, V = la.eigh(Href)
    cert = ce.certify(ce.Claim(w[:3], V[:, :3], operator=H, kind="lowest"), spec)
    assert _status(cert, "S1") == "CERTIFIED", cert.summary()
    assert _status(cert, "S2") == "CERTIFIED"


def test_identity_and_zero_terms_and_list_input():
    spec = ModelSpec.from_pauli_strings([("II", 0.25), ("ZI", 0.0), ("IZ", -1.0), ("iz", 2.0)])
    assert spec.parameters["dropped_zero_terms"] == 1
    assert spec.to_quspin_static() == [["I", [[0.25, 0]]], ["z", [[-1.0, 1], [2.0, 1]]]]
    assert np.abs(ce.build_bitwise(spec).toarray() - (0.25 * np.eye(4) + np.kron(I2, PAULI["Z"]))).max() == 0.0


def test_static_grouping_and_sites():
    static, n, dropped = cp.pauli_terms_to_static({"XXI": 0.5, "ZZI": 1.0, "IXX": 0.25, "ZIZ": -1.0})
    assert n == 3 and dropped == 0
    assert static == [["xx", [[0.5, 0, 1], [0.25, 1, 2]]], ["zz", [[1.0, 0, 1], [-1.0, 0, 2]]]]


def test_errors():
    with pytest.raises(ValueError, match="length"):
        ModelSpec.from_pauli_strings({"XX": 1.0, "XXX": 1.0})
    with pytest.raises(ValueError, match="not I, X, Y or Z"):
        ModelSpec.from_pauli_strings({"XA": 1.0})
    with pytest.raises(ValueError, match="non-real"):
        ModelSpec.from_pauli_strings({"XX": 1.0 + 1e-9j})
    with pytest.raises(ValueError, match="zero"):
        ModelSpec.from_pauli_strings({"XX": 0.0})
    with pytest.raises(ValueError, match="no Pauli terms"):
        ModelSpec.from_pauli_strings({})
    with pytest.raises(ValueError, match="qubit_order"):
        ModelSpec.from_pauli_strings({"XX": 1.0}, qubit_order="middle")
    with pytest.raises(ValueError, match="expected 3"):
        ModelSpec.from_pauli_strings({"XX": 1.0}, n_qubits=3)


def test_spec_round_trips_through_json():
    spec = ModelSpec.from_pauli_strings(H2_TERMS)
    again = ModelSpec.from_dict(spec.to_dict())
    assert again.sha256() == spec.sha256()
    assert abs(ce.build_bitwise(again) - ce.build_bitwise(spec)).max() == 0.0


# ----------------------------------------------------------------------- negative controls
def test_negative_control_wrong_producer_coefficient_fails_s1():
    spec = ModelSpec.from_pauli_strings(H2_TERMS)
    bad = dict(H2_TERMS)
    bad["XX"] += 1e-3                                              # the producer built a different operator
    Hbad = _matrix(bad)
    w, V = la.eigh(Hbad)
    cert = ce.certify(ce.Claim(w, V, operator=Hbad, kind="all"), spec)
    assert _status(cert, "S1") == "FAILED", cert.summary()


def test_negative_control_planted_eigenvalue_not_certified():
    spec = ModelSpec.from_pauli_strings(H2_TERMS)
    H = _matrix(H2_TERMS)
    w, V = la.eigh(H)
    w = w.copy()
    w[0] -= 1e-4
    cert = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec)
    assert _status(cert, "S2") != "CERTIFIED"


def test_negative_control_qubit_order_matters():
    """A producer that used the other endianness builds a different matrix for an asymmetric operator: S1 FAILED."""
    terms = {"XZ": 1.0, "ZI": 0.5}
    spec = ModelSpec.from_pauli_strings(terms)                     # site 0 = leftmost character
    H_swapped = _matrix({"ZX": 1.0, "IZ": 0.5})                    # the same strings read right to left
    w, V = la.eigh(H_swapped)
    cert = ce.certify(ce.Claim(w, V, operator=H_swapped, kind="all"), spec)
    assert _status(cert, "S1") == "FAILED"


# ----------------------------------------------------------------------------- OpenFermion
class _FakeQubitOperator:
    """Mimics openfermion.QubitOperator: .terms = {((q, 'X'), ...): coeff}, () the identity."""

    def __init__(self, terms):
        self.terms = dict(terms)


def _openfermion_h2():
    try:
        from openfermion import QubitOperator
    except ImportError:
        return _FakeQubitOperator({(): -0.4804, ((0, "Z"),): 0.3435, ((1, "Z"),): -0.4347,
                                   ((0, "Z"), (1, "Z")): 0.5716, ((0, "X"), (1, "X")): 0.0910,
                                   ((0, "Y"), (1, "Y")): 0.0910})
    op = QubitOperator((), -0.4804) + QubitOperator("Z0", 0.3435) + QubitOperator("Z1", -0.4347)
    op += QubitOperator("Z0 Z1", 0.5716) + QubitOperator("X0 X1", 0.0910) + QubitOperator("Y0 Y1", 0.0910)
    return op


def test_openfermion_qubit_operator():
    op = _openfermion_h2()
    spec = ModelSpec.from_openfermion(op)
    assert spec.n_sites == 2 and spec.parameters["source"] == "openfermion.QubitOperator"
    assert np.abs(ce.build_bitwise(spec).toarray() - _matrix(H2_TERMS)).max() == 0.0
    try:
        from openfermion import get_sparse_operator
        assert np.abs(ce.build_bitwise(spec).toarray() - get_sparse_operator(op, 2).toarray()).max() == 0.0
    except ImportError:
        pass
    # n_qubits padding and a term on a higher qubit
    op3 = _FakeQubitOperator({((2, "X"), (0, "Z")): 0.7, (): 1.0})
    spec3 = ModelSpec.from_openfermion(op3)
    assert spec3.n_sites == 3
    assert np.abs(ce.build_bitwise(spec3).toarray() - _matrix({"ZIX": 0.7, "III": 1.0})).max() == 0.0
    spec4 = ModelSpec.from_openfermion(op3, n_qubits=4)
    assert spec4.n_sites == 4 and spec4.dim == 16
    with pytest.raises(ValueError, match="twice"):
        ModelSpec.from_openfermion(_FakeQubitOperator({((0, "X"), (0, "Z")): 1.0}))
    with pytest.raises(ValueError, match="outside"):
        ModelSpec.from_openfermion(_FakeQubitOperator({((3, "X"),): 1.0}), n_qubits=2)
    with pytest.raises(ValueError, match="non-real"):
        ModelSpec.from_openfermion(_FakeQubitOperator({((0, "X"),): 1j}))
    with pytest.raises(TypeError, match="QubitOperator-like"):
        ModelSpec.from_openfermion({"XX": 1.0})


# ---------------------------------------------------------------------------------- Qiskit
class _FakeSparsePauliOp:
    """Mimics qiskit.quantum_info.SparsePauliOp: .to_list() -> [(label, coeff)], .num_qubits; little-endian labels."""

    def __init__(self, items, num_qubits=None):
        self._items = [(lab, complex(c)) for lab, c in items]
        self.num_qubits = num_qubits if num_qubits is not None else len(items[0][0])

    def to_list(self):
        return list(self._items)


def _qiskit_op(items):
    try:
        from qiskit.quantum_info import SparsePauliOp
        return SparsePauliOp.from_list(items)
    except ImportError:
        return _FakeSparsePauliOp(items)


def test_qiskit_sparse_pauli_op_little_endian():
    items = [("XZ", 1.0), ("ZI", 0.5), ("IY", 0.25), ("YY", -0.75)]
    op = _qiskit_op(items)
    spec = ModelSpec.from_qiskit(op)
    assert spec.parameters["qubit_order"] == "little_endian" and spec.n_sites == 2
    # Qiskit's to_matrix: qubit 0 is the rightmost tensor factor, i.e. the label read left to right
    Href = _matrix(dict(items))
    assert np.abs(ce.build_bitwise(spec).toarray() - Href).max() == 0.0
    try:
        assert np.abs(ce.build_bitwise(spec).toarray() - np.asarray(op.to_matrix())).max() == 0.0
    except AttributeError:
        pass
    # the site of qubit q is n-1-q: the single-qubit term 'IY' acts on qubit 0 = site 1
    assert spec.to_quspin_static()[2] == ["y", [[0.25, 1]]]
    w, V = la.eigh(Href)
    cert = ce.certify(ce.Claim(w, V, operator=Href, kind="all"), spec)
    assert _status(cert, "S1") == "CERTIFIED" and _status(cert, "S2") == "CERTIFIED", cert.summary()
    with pytest.raises(ValueError, match="phase prefix"):
        ModelSpec.from_qiskit(_FakeSparsePauliOp([("-XZ", 1.0)]))
    with pytest.raises(TypeError, match="SparsePauliOp-like"):
        ModelSpec.from_qiskit(object())


def test_qiskit_paulis_coeffs_fallback():
    class _Legacy:
        num_qubits = 2
        paulis = ["ZZ", "XI"]
        coeffs = np.array([1.0, 0.5])
    spec = ModelSpec.from_qiskit(_Legacy())
    assert np.abs(ce.build_bitwise(spec).toarray() - _matrix({"ZZ": 1.0, "XI": 0.5})).max() == 0.0


def test_openfermion_and_qiskit_agree_on_the_same_physical_operator():
    """OpenFermion 'X0 Z1' (qubit 0 leftmost) and Qiskit label 'ZX' (qubit 0 rightmost) are the same operator in
    two basis orderings: each spec reproduces its library's matrix, the two matrices are related by the
    bit-reversal permutation of the basis and have the same spectrum."""
    of = ModelSpec.from_openfermion(_FakeQubitOperator({((0, "X"), (1, "Z")): 1.0, ((1, "Y"),): 0.3}))
    qk = ModelSpec.from_qiskit(_FakeSparsePauliOp([("ZX", 1.0), ("YI", 0.3)]))
    A = ce.build_bitwise(of).toarray()
    B = ce.build_bitwise(qk).toarray()
    assert np.abs(A - _matrix({"XZ": 1.0, "IY": 0.3})).max() == 0.0
    assert np.abs(B - _matrix({"ZX": 1.0, "YI": 0.3})).max() == 0.0
    rev = [int(f"{i:02b}"[::-1], 2) for i in range(4)]
    assert np.abs(A - B[np.ix_(rev, rev)]).max() == 0.0
    assert np.allclose(np.linalg.eigvalsh(A), np.linalg.eigvalsh(B))
