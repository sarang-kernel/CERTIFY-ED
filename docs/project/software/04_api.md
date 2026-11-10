# The Python interface

This document describes the main entry points of the Python interface of certify-ed at `a814d28`, with the patterns of
use that the code repository documents. The complete list of public names with their signatures is generated from the
docstrings by `docs/make_api.py` into `code/certify-ed/docs/api.md`; the runnable versions of the snippets below are
`examples/01` to `examples/04` (`docs/CODE_INDEX.md`). Facts are from `code/certify-ed/README.md` and
`code/certify-ed/docs/api.md`.

## Specifying the model

A model is declared as a `ModelSpec` with `n_sites`, a `LocalSpace` and a list of terms, or built by a factory
function. The factories of `certify_ed.spec` cover the models of the experiments: `heisenberg_chain(N, J, Delta, J2,
boundary)`, `tfim_chain`, `heisenberg_square(Lx, Ly, J, J2, boundary)`, `spinless_fermion_chain`, `bose_hubbard_chain`
and `spin1_heisenberg_chain`. `spec_from_quspin(static, basis)` reads a QuSpin static list.
`ModelSpec.from_pauli_strings`, `from_openfermion` and `from_qiskit` (`integrations/pauli.py`) read Pauli-string
Hamiltonians, with the conventions fixed in that module so that S1 compares the producer's matrix with the same
operator the library builds. `spec.to_quspin_static()` writes the spec back as a QuSpin static list, which is how a
test or an example builds a producer matrix from the same declaration. A local space of fermions takes the direction
of its Jordan-Wigner string as an option, with QuSpin's convention as default (decision D-21).

## Certifying a claim

The general call is `certify(claim, spec, options)`, which returns a `Certificate`. A `Claim` holds the eigenvalues,
the optional eigenvectors, operator and basis, the kind (`subset`, `lowest`, `window`, `all`) and an optional accuracy
target. `Options` holds every switch of the run; the defaults include `inertia_max_dim=60000`, `exact_max_dim=4000`,
`gap_fractions=(0.9, 0.5, 0.2, 0.05)` (the probe ladder of S3, decision D-20), `seed=12345`, `eigenspaces='clusters'`,
`degeneracy=True` and `underflow_check=True`, and `Options.from_preset(name, ...)` starts from a preset.

```python
import numpy as np, certify_ed as ce
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

L = 16
spec = ce.heisenberg_chain(L, J2=0.3)                 # the intended model, written independently
basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pauli=0)
H = hamiltonian(spec.to_quspin_static(), [], basis=basis, dtype=np.float64)
E, V = H.eigsh(k=6, which="SA")
claim, _ = ce.claim_from_quspin(H, basis, spec.to_quspin_static(), E, V, kind="lowest")
cert = ce.certify(claim, spec)
print(cert.summary())                                 # one sentence; cert.report() is the full text
cert.save("result.json")                              # and result.arrays.npz, hashed
```

The one-call form `quick(H, w, V, spec, kind="lowest", **options)` (`api.py`) builds the claim and the options from
arrays and accepts the same keywords, for example `preset=`, `budget_seconds="auto"` and `progress=True`. A
matrix-free producer passes a `scipy.sparse.linalg.LinearOperator` as H; S1 then uses random probes and is marked
probabilistic.

## Reading a certificate

`Certificate.load(path, verify=True)` re-checks the body hash, every array hash and, since `fix/3.1.1`, the
consistency of the scope block (`verify_scope`), and validates the body against the JSON Schema when `jsonschema` is
installed. `cert.summary()` gives one sentence (statement counts by status, the largest S2 radius, completeness and
its gap), `cert.report()` the full text, `cert.statement("S3.completeness")` one record, `cert.counts_by_status()` and
`cert.max_radius()` the numbers behind the summary, and `cert.methods_paragraph("latex")` a methods paragraph in which
every number is a field of the certificate, with `\newcommand` macros and a BibTeX entry (`report.py`). In a notebook
a certificate renders as a table of statement, status and one line of evidence with the next action. `export_npz`,
`claim_from_npz` and `claim_from_hdf5` move claims between processes and codes.

## The exact layer

```python
cert = ce.certify(claim, spec, ce.Options(exact_recheck=True))          # adds S6
r = ce.exact_recheck_lowest(spec, E[:k], V[:, :k], states)               # dense blocks up to about 4000
r = ce.exact_recheck_sector_lowest(spec, states, [ce.translation(spec), ce.reflection(spec),
                                   ce.spin_flip(spec)], {0: 0}, E[:k], X)  # momentum 0, any size that blocks
```

The exact functions need python-flint (`pip install -e .[exact]`). Without it, S6 is NOT_CHECKED with the reason code
`MISSING_DEPENDENCY` and the evidence names the extra to install. Exact sectors need one-dimensional characters with
values in {±1, ±i}: momenta 0 and π on any lattice, all momenta on lattices of side 1, 2 or 4, reflections and the
spin flip (`code/certify-ed/README.md`, Limits).

## Whole spectra and sectors

`enclose_spectrum(H, S_diag=Sz)` encloses the whole spectrum sector by sector without diagonalising H as a whole; the
result has `complete`, `max_halfwidth` and `intervals`. `certify_ed.compat.symveig.enclose_global` and
`enclose_sectors` keep the symveig 1.0.0 interface on the new engine. `compose.certify_low_energy` assembles
per-sector certificates into the low spectrum of the whole Hamiltonian. The reference sectors are built with
`u1_state_set`, `magnetization_state_set`, `translation`, `reflection`, `spin_flip`, `cyclic_isometry` and
`sector_matrix` (`basis.py`), and the reference operator with `build_bitwise`, `build_kron`, `cross_construct` and
`SpecOperator` (`construct.py`).

## Producers that write files or run in Python

`claim_from_bundle(directory)` returns the claim, the spec and the metadata of a producer bundle, and
`write_bundle(...)` writes one. A producer that runs in Python can use the record hook, `with ce.record(spec,
"run42.bundle") as rec: rec.capture(w, V, H)`, or the decorator `@ce.recorded(spec, "run42.bundle", operator=H)`; the
arrays are written exactly as given, so a wrong value in the producer's output is in the bundle and is found when the
bundle is certified (`07_importers_and_producers.md`). `integrations.quspin` turns a QuSpin `hamiltonian` and its
eigensolver output into a certificate in one call.

## Planning and budgets

`estimate_seconds(dimension, model)`, `auto_budget(dimension, ...)` and `plan_for_claim(claim, spec, options)`
(`cost.py`) predict the time and memory of each statement; `Options(plan=True)` prints the plan before running and
`certify-ed plan` prints it without running. The estimates are advisory: they are fitted to the E2 records of the
3.1.0 evidence run and never change a status.
