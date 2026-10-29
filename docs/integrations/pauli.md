# Pauli strings, OpenFermion and Qiskit operators as model specs

`ModelSpec.from_pauli_strings`, `ModelSpec.from_openfermion` and `ModelSpec.from_qiskit`
(`certify_ed/integrations/pauli.py`) declare a qubit Hamiltonian H = sum_k c_k P_k as a certify-ed model, so
that S1 can compare the matrix a quantum-computing or ED producer built against the matrix certify-ed builds
from the same strings, and S2-S4 run as for any other spec.

```python
import certify_ed as ce
from certify_ed import ModelSpec

spec = ModelSpec.from_pauli_strings({"II": -0.4804, "ZI": 0.3435, "IZ": -0.4347,
                                     "ZZ": 0.5716, "XX": 0.0910, "YY": 0.0910}, name="H2, 2 qubits")
H = producer_matrix                     # numpy array or scipy.sparse, in the convention below
w, V = numpy.linalg.eigh(H)
cert = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec)
```

Neither OpenFermion nor Qiskit is imported by certify-ed: the converters read `op.terms` (QubitOperator) or
`op.to_list()` and `op.num_qubits` (SparsePauliOp), so any object with that interface works and the packages are
not dependencies.

## Conventions

* Local space: spin-1/2, `pauli` convention (`x`, `y`, `z` are sigma^x, sigma^y, sigma^z; `I` the identity).
  Digit 1 of the certify-ed encoding is sigma^z = +1, the qubit state |0>; digit 0 is |1>.
* `state_order = "descending"`, `site_order = "site0_msb"`: basis index i is the computational basis state whose
  big-endian bit string is i, with site 0 the most significant bit. This is the ordering of
  `openfermion.get_sparse_operator` and of `SparsePauliOp.to_matrix()`.
* A Pauli string is read left to right: character j acts on site j.
  * OpenFermion (`qubit_order="big_endian"`, the default): site = qubit. `from_openfermion` writes each term
    `((0, 'X'), (1, 'Z'))` as the string `XZ...`; the spec matrix equals `get_sparse_operator(op, n_qubits)`.
  * Qiskit (`qubit_order="little_endian"`): labels put qubit 0 at the right, so character j is qubit n-1-j and
    site j = qubit n-1-j. The spec matrix equals `op.to_matrix()`. `spec.parameters["site_of_qubit"]` records
    the rule.
  * The same physical operator written in the two conventions gives two specs whose matrices are related by the
    bit-reversal permutation of the basis (test `test_openfermion_and_qiskit_agree_on_the_same_physical_operator`).
    A producer that used the other endianness is caught by S1 (test `test_negative_control_qubit_order_matters`).
* Coefficients must be real (`ValueError` otherwise: a sum of Pauli strings with a non-real coefficient is not
  Hermitian). Terms with coefficient 0 are dropped and counted (`parameters["dropped_zero_terms"]`). The
  all-identity string becomes the term `("I", [[c, 0]])`, c times the identity.
* The strings are grouped by letter pattern into the QuSpin-style static list (`"XXI" -> ["xx", [[c, 0, 1]]]`),
  so `spec.to_quspin_static()` and the spec JSON look like every other certify-ed model. `is_real()` is True
  when all coefficients are real and every string has an even number of Y letters; otherwise the reference matrix
  is complex and the claim follows the complex path.

## What is checked

`tests/test_pauli_specs.py`: the transverse-field Ising chain from strings equals `tfim_chain` entry by entry
(both builders); a 2-qubit H2-like Hamiltonian (Bravyi-Kitaev reduced form, coefficients rounded) certifies
with `kind="all"` against a numpy Kronecker-product matrix, S1-S3 CERTIFIED; a complex (odd-Y) three-qubit
operator certifies; OpenFermion- and Qiskit-like objects (hand-built when the packages are absent; the real
objects when present) give the matrices of `get_sparse_operator` / `to_matrix()`; negative controls: a producer
matrix with one coefficient changed by 1e-3 gives S1 FAILED, a planted eigenvalue is not certified, the wrong
endianness gives S1 FAILED.

## Limits

* Qubit operators only (spin-1/2 Pauli algebra). Fermionic operators (OpenFermion `FermionOperator`,
  Qiskit Nature `FermionicOp`) must be mapped to qubits by the producer first (Jordan-Wigner, Bravyi-Kitaev);
  the spec then describes the qubit operator, and the certificate's statements concern that operator.
* Labels with a phase prefix (`-iXZ`) are refused; call `SparsePauliOp.simplify()` first.
* Nothing here changes a bound: the module builds a `ModelSpec`, the matrices come from `certify_ed.construct`
  (`provenance.NON_COMPUTATIONAL`).
