# CERTIFY-ED

**Producer-independent certificates for exact-diagonalization (ED) results.**

CERTIFY-ED checks a result produced by any ED code (QuSpin, XDiag, HΦ, your own) against a
declared model and returns a *certificate*: a machine-readable list of statements, each marked
`CERTIFIED`, `FAILED` (with a diagnosis), `INCONCLUSIVE` (with the reason) or `NOT_CHECKED`
(with the reason). It does not diagonalize anything for you and it never calls your code.

| Statement | What it says when CERTIFIED |
|---|---|
| **S1** model fidelity | your matrix (or symmetry block) is the matrix of the declared model, built independently; on failure, a least-squares fit names the terms whose coefficients differ |
| **S2** enclosures | each claimed eigenvalue lies within a stated radius of a *distinct* eigenvalue of the declared model |
| **S3** completeness | no eigenvalue below the top claimed one is missing; certified lower bound on the next one |
| **S4** eigenspaces | for degenerate levels: projector error, basis-independent observables tr(PO), certified quantum numbers |
| **S5** corroboration | (optional) independent solvers reproduce the values; their shared components are recorded |

`certify_low_energy` assembles certified symmetry-block results into a certified statement
about the low-energy spectrum of the full Hilbert space.

The bounds come from residual, Kato–Temple, Davis–Kahan and Sylvester-inertia arguments with
explicit floating-point error terms (standard model, IEEE binary64). They are tested against
ball arithmetic (Arb). The manuscript in preparation describes the mathematics and the
experiments; `experiments/` regenerates every number in it.

## Install

```bash
pip install -e .                 # numpy, scipy
pip install -e .[validation]     # QuSpin importer and examples
pip install -e .[rigorous,hdf5]  # python-flint (Arb) reference checks, HDF5 import/export
pip install -e .[dev]            # pytest
```

Tested with Python 3.11, NumPy 2.4, SciPy 1.17, QuSpin 1.0.1.

## Certify a QuSpin result

```python
import numpy as np, certify_ed as ce
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

L = 16
spec = ce.heisenberg_chain(L, J2=0.3)                 # the model you intend
static = spec.to_quspin_static()
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pauli=0)
H = hamiltonian(static, [], basis=basis, dtype=np.float64)
E, V = H.eigsh(k=6, which="SA")

claim, _ = ce.claim_from_quspin(H, basis, static, E, V, kind="lowest")
cert = ce.certify(claim, spec)
print(cert.summary())
cert.save("result.json")                              # + result.arrays.npz, hashed
```

Write the specification independently of the producer's input (as above) so that S1 can catch
input errors. A reader re-checks the file with `ce.CertificateV2.load("result.json", verify=True)`.

## Other producers

Export eigenvalues, eigenvectors and the sparse matrix (CSR triplet) to NPZ or HDF5 and use
`claim_from_npz` / `claim_from_hdf5`; pass `states=` (a fixed-particle-number basis) or
`sector_isometry=` (an n × d sparse isometry) if you worked in a symmetry block. For a
matrix-free producer pass a `scipy.sparse.linalg.LinearOperator`; S1 then compares it with the
model on random vectors (probabilistic), and S2–S4 are evaluated with CERTIFY-ED's reference
operator.

## Model specifications

`ModelSpec` uses QuSpin's `static` operator-string format. Local spaces: spin S (three
conventions for S = 1/2), spinless fermions (Jordan–Wigner string on higher-index sites as in
QuSpin, or on lower-index sites), bosons with a cutoff. Factories: `heisenberg_chain`,
`tfim_chain`, `heisenberg_square`, `spinless_fermion_chain`, `bose_hubbard_chain`,
`spin1_heisenberg_chain`, or `ModelSpec.from_static(N, local_space, static)`.

## Limits (read these)

* **Completeness (S3)** needs a sparse LDLᵀ factorization. Its fill-in limits S3 to blocks of
  dimension ~10⁴ on a workstation (4×4 Heisenberg S^z=0: 31 s, 1.9 GB). Larger blocks get
  `NOT_CHECKED`, never an uncertified count.
* **Model fidelity (S1)** for momentum blocks is audited in the parent U(1) block (N times
  larger). Black-box operators are checked probabilistically.
* Bounds assume the standard floating-point model (no Strassen-type BLAS, no reduced-precision
  modes). They are not interval arithmetic.
* Hermitian, time-independent operator-string Hamiltonians only; single process (no MPI);
  native importer for QuSpin only.

## Layout

```
certify_ed/
  spec.py, localops.py      model specification, local spaces and conventions
  construct.py, basis.py    two independent reference builders, matrix-free operator, sectors
  enclosure.py, fp.py       S2 bounds, floating-point error model, error-free inner products
  inertia.py                S3 inertia counts with backward-error bounds
  eigenspace.py, sectors.py S4 bounds, sector audits, U(1) restriction
  compose.py                certified global spectrum from symmetry blocks
  certify.py                orchestration: certify(claim, spec, options)
  certificate_v2.py         certificate format (canonical JSON + hashed NPZ, HDF5 export)
  importers.py              QuSpin, NPZ, HDF5
  corroborate.py, rigorous.py  S5 solvers (incl. Jacobi); Arb reference used in tests
tests/v2/                   tests of the verification core
experiments/                E0–E4 of the paper, run_all.sh, make_numbers.py
```

The v1 modules (`hamiltonian.py`, `oracles.py`, `certificates.py`, `validators/`) remain for
backward compatibility; they are code-verification utilities, not part of the certification
described above.

## Tests

```bash
pytest tests/ -q
```

## License and citation

MIT (see `LICENSE`). Citation metadata: `CITATION.cff`.
