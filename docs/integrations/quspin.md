# QuSpin integration: `certify_hamiltonian`

`certify_ed.integrations.quspin.certify_hamiltonian(H, w, V=None, basis=None, static=None, spec=None,
kind="lowest", window=None, **options)` turns a QuSpin `hamiltonian` object and the output of `H.eigsh` or
`H.eigh` into a certificate in one call. It is the QuSpin-specific counterpart of `certify_ed.quick`.

```python
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

import certify_ed as ce
from certify_ed.integrations.quspin import certify_hamiltonian

L = 10
static = ce.heisenberg_chain(L, J2=0.3).to_quspin_static()
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pblock=1, pauli=0)
H = hamiltonian(static, [], basis=basis, dtype=np.float64,
                check_symm=False, check_herm=False, check_pcon=False)
E, V = H.eigsh(k=4, which="SA")

cert = certify_hamiltonian(H, E, V)         # kind="lowest": "E are the 4 lowest eigenvalues of H"
print(cert.summary())
cert.save("heisenberg_k0_p1.certificate.json")
```

## What the call does

1. Takes the basis from `H.basis` and the operator-string list from the object (`H._static_opstr_list`, the
   list QuSpin received). A `hamiltonian` built from matrices keeps no string list: pass `static=[...]`. A
   non-empty dynamic list (time dependence) is refused with `ValueError`.
2. Sorts the claimed eigenvalues in ascending order, with the matching columns of `V`, when `kind` is `lowest`
   or `window` (`eigsh` does not promise an order; `eigh` returns ascending). The certificate records
   `claim.producer.eigenvalues_sorted_by_wrapper`.
3. Builds the Claim and the ModelSpec with `certify_ed.claim_from_quspin`. The basis decides the audit path:
   a full basis is compared entry by entry with the spec matrix; a U(1) basis (`Nup`, `Nf`, `Nb`) through its
   state list; a symmetry-adapted basis (`kblock`, `pblock`, `zblock`, ...) through its isometry
   `basis.get_proj`, so S1 reports the isometry and invariance defects of the block and S2 carries the
   certified shift from the block to the full operator. This is the same path as a direct `claim_from_quspin`
   call; the test `test_wrapper_matches_direct_route` pins that the statuses agree for all four basis types.
4. Calls `certify()` with `Options` built from `**options` (fields of `Options`, including `preset`, `only`,
   `skip`, `budget_seconds`, `exact_recheck`, `progress`) and the Claim fields `producer`, `accuracy_target`,
   `description`.

## What the spec is

By default the ModelSpec is rebuilt from the string list QuSpin received. S1 then states that QuSpin's matrix
for that string list equals the matrix certify-ed builds for the same strings (two independent constructions
of one declaration). It does not state that the string list is the model you meant. To certify against an
independently written model, pass `spec=`:

```python
cert = certify_hamiltonian(H, E, V, spec=ce.heisenberg_chain(L, J2=0.3))
```

The certificate's `claim.producer.spec_source` says which (`"user"` or `"static list of H"`). A wrong `spec=`
(different coupling) gives S1 FAILED, as the test `test_spec_override_is_used` shows.

## What the call does not do

* It adds no check and changes no bound: `certify_hamiltonian(H, w, V, **o)` returns what
  `certify(*claim_from_quspin(H, H.basis, static, w_sorted, V_sorted, kind=kind), Options(**o))` returns.
  The module is listed in `provenance.NON_COMPUTATIONAL`.
* It does not read QuSpin's own checks (`check_symm`, `check_herm`, `check_pcon`); the certificate's S1 is the
  check of the matrix against the spec.
* It does not certify time-dependent Hamiltonians, `quantum_operator` or `quantum_LinearOperator` objects
  (pass their matrix and a spec to `certify_ed.quick`).

## Upstream

`integrations/quspin_PR.md` in the repository is a draft pull-request text proposing a thin
`quspin.tools.certify` wrapper around this function. It is text only; nothing in certify-ed depends on it.
