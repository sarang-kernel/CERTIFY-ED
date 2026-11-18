# Importers and producers

A certificate is only as useful as the route by which a producer's output reaches the certifier. The decision behind
the routes is D-49: one universal layout (the producer bundle) for every code that can write files, plus native
readers where a code has its own output format, instead of an importer for each producer. This document lists every
route at `a814d28`, what each one assumes, and how the certificate records the assumption. Facts are from the
docstrings of `certify_ed/importers.py`, `certify_ed/io/*.py`, `certify_ed/integrations/*.py` and
`certify_ed/recorder.py`, and from `code/certify-ed/docs/producer_bundle.md`.

## Routes into a claim

| Route | Module | Input | Notes |
|---|---|---|---|
| QuSpin objects | `importers.claim_from_quspin` | a `hamiltonian`, its basis, the static list, the eigenpairs | full bases, U(1) bases and symmetry-adapted bases (momentum, parity, spin inversion) through the basis projector |
| QuSpin in one call | `integrations.quspin.certify_hamiltonian` | a `hamiltonian` and its eigensolver output | takes the basis and the operator strings from the object, refuses time-dependent Hamiltonians, sorts the eigenvalues with their vectors |
| arrays and NPZ | `importers.claim_from_npz`, `export_npz` | a matrix (or none) and eigenpairs | the route for XDiag and for in-house C++, Fortran or Julia codes |
| HDF5 | `importers.claim_from_hdf5` | datasets `eigenvalues`, optional `eigenvectors`, optional CSR triplet under `H/` | the layout a referee of v1 proposed for publishing residual data |
| matrix-free | `Claim` with a `LinearOperator` | a matrix-vector product | S1 by random probes, marked probabilistic; S3 needs a stored matrix |
| any file, or a bundle | `io.claim_from_file`, `io.claim_from_bundle` | the formats of `06_formats_and_schema.md` | the universal route |
| HΦ | `io/hphi.py` | `##_energy.dat` (Lanczos and LOBCG, and the older one-line form), `Eigenvalue.dat` (full diagonalization), the binary eigenvector files | layouts taken from the HΦ manual, version 3.5 |
| DiagHam | `io/diagham.py` | binary vectors `*.vec` and spectrum text files | the real vector layout is documented by the DiagHam wiki; the complex layout is assumed by analogy, decided from the file size, and refused when the size matches neither |
| SPINPACK | `io/spinpack.py` | text tables of eigenvalues with sector labels | no file specification was available, so the layout is declared as assumed and named in the claim's producer record |
| ALPS | `io/alps.py` | the HDF5 output of `sparsediag` and `fulldiag` (`/spectrum/sectors/<n>/...`, with eigenvectors and basis when saved) | ALPS 2.x layout |
| Pauli strings | `integrations/pauli.py` | plain strings, an OpenFermion `QubitOperator`, a Qiskit `SparsePauliOp` | builds a `ModelSpec` with fixed conventions (spin 1/2 in the Pauli convention; digit 1 is spin up, the qubit state with σ^z = +1) |

Where a reader rests on an assumed layout (the complex DiagHam vectors, every SPINPACK layout), the assumption is
written into `claim.producer` of the certificate, so that a reader of the certificate sees which assumption a result
depends on. The test files for these readers are generated to the documented layouts by `tests/fixtures/ed_codes.py`
and exercised by `tests/test_io_ed_codes.py` (22 tests of layouts, truncation and header checks, and round trips
through `certify()` on a 6-site chain with S1 to S3 CERTIFIED). The CHANGELOG records one open point: the basis order
of HΦ, DiagHam and SPINPACK relative to a `ModelSpec` is not established, so the caller passes `operator=` and
`states=` explicitly (`code/certify-ed/CHANGELOG.md`, section Producers).

## Writing a bundle from a Python producer

The record hook (`recorder.py`, roadmap item A-07) captures a solver's output in the producing process and writes a
bundle:

```python
import certify_ed as ce
with ce.record(spec, "run42.bundle") as rec:          # spec may be None, to declare the model later
    w, V = scipy.sparse.linalg.eigsh(H, k=4, which="SA")
    rec.capture(w, V, H)
```

On exit the directory holds `eigenvalues.npz`, `eigenvectors.npz`, `operator.npz`, `spec.json` and `meta.json` with
the kind (default `lowest`), the producer, the environment and the timing of the producing process. The arrays are
written as given: nothing is sorted, rounded or filtered, so a wrong value in the producer's output is in the bundle
and is found when the bundle is certified. If the block raises, nothing is written; if it ends without eigenvalues,
the hook raises `ValueError`. The decorator form `@ce.recorded(spec, "run42.bundle", operator=H)` wraps a function
returning `(w, V)` or `(w, V, H)`. The hook is non-computational in the sense of `05_provenance.md`: it never changes
a number.

## Checking a producer route

`certify-ed conform BUNDLE...` (`conformance.py`) runs four checks on a bundle: the schema of `meta.json`, the import,
the certification, and the schema of the resulting certificate. Three reference bundles in
`code/certify-ed/conformance/reference/` (`dense-arrays`, `matrix-free`, `npz`) are the fixtures, built by
`conformance/make_reference_bundles.py` and checked by `conformance/check_bundle.py`. The planted-error corpus in
`code/certify-ed/corpus/` (60 cases in 14 error classes) and its runner score any checker against known errors;
`08_tests_and_mutation.md` describes it. Decision D-11 planned to ask the authors of the producer codes concrete
questions about their exports; `project_docs/PROJECT_STATE.md` lists that outreach as planned, so the routes above
rest on the published documentation of each code.
