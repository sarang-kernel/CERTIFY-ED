"""Integrations with producer codes and quantum-computing libraries (3.2, roadmap F-01, F-05).

Each submodule imports its third-party package lazily, so `certify_ed.integrations` can be imported without
QuSpin, OpenFermion or Qiskit installed:

  * `certify_ed.integrations.quspin`: `certify_hamiltonian(H, w, V, ...)`, one call from a QuSpin
    `hamiltonian` object and the output of `H.eigsh` / `H.eigh` to a Certificate.
  * `certify_ed.integrations.pauli`: Pauli-string Hamiltonians (`ModelSpec.from_pauli_strings`), OpenFermion
    `QubitOperator` and Qiskit `SparsePauliOp` to a ModelSpec.

Nothing here computes a bound: the modules build Claims and ModelSpecs and call `certify()`
(provenance.NON_COMPUTATIONAL).
"""
