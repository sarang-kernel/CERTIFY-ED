# File formats and the JSON schemas

certify-ed reads and writes three kinds of files: the certificate (a JSON body and an NPZ side file of arrays), the
producer bundle (a directory of plain files that any ED code can write), and the run directory of the reproduction
driver. This document describes the first two and their schemas; the run directory is described in
`docs/reproduction/README.md`. Facts are from `code/certify-ed/docs/schema.md`,
`code/certify-ed/docs/producer_bundle.md`, `code/certify-ed/verifier/README.md` and the schema files in
`code/certify-ed/schema/`.

## The certificate

A certificate is written by `Certificate.save(path)` as a JSON file and, when arrays are saved, a side file
`<stem>.arrays.npz`. The JSON body identifies itself with the field `"schema": "certify-ed/certificate/2.1"`. Its
top-level fields are `subject` (the declared model and the operator: the spec with its SHA-256, the hash of the
operator matrix, a norm bound, and the sector when the claim concerns a symmetry block), `claim` (the eigenvalues, the
kind, the window, the accuracy target, the hash of the eigenvectors and the producer's provenance), `statements` (one
record per statement reached, each with `id`, `status` and `evidence`), `enclosures` (the S2 record of every claimed
eigenvalue: claimed value, radius, existence radius, residual bound, lower and upper ends, group, method,
distinctness, index), `kato_temple`, `arithmetic` (the floating-point model the bounds assume, the run-time probe and
the underflow check), `environment` (versions, platform, time and the source fingerprint), `hermitian_part` (present
when the producer's matrix was not exactly Hermitian), `operator_note`, `spectrum_enclosure` (the interval list of a
whole-spectrum claim), `scope` (schema `certify-ed/scope/1.0`: what was in scope, narrowed and out of scope, with
reason codes), `not_verified` (what the certificate does not say), `timing`, `arrays` (the manifest of the side file)
and `integrity`.

Integrity rests on two hashes. The body hash `integrity.body_sha256` is the SHA-256 of the canonical JSON of the body
without the field `integrity`; the canonical form, named `certify-ed-canonical-json-1`, has sorted keys, the
separators `,` and `:`, non-ASCII characters written as they are, floats in Python's shortest round-trip form and no
NaN (non-finite numbers are written as the strings `"nan"`, `"inf"` and `"-inf"`). Each array of the side file is
hashed as the SHA-256 of its dtype string, the JSON list of its shape and its bytes in little-endian C order, and the
hashes are listed in the body's `arrays` block. The side file holds `eigenvalues`, `radius`, `existence_radius` and
`residual_bound`, and also `eigenvectors` when the certificate was made with `Options(store_eigenvectors=True)`. The
exported `radius` and `existence_radius` arrays carry the recorded radii plus the model shift, rounded upward since
`fix/3.1.1` (INT-03, `docs/failures/02_integration_findings.md`). A certificate can also be exported to HDF5.

`Certificate.load(path, verify=True)` recomputes both hashes and, since `fix/3.1.1`, checks the scope block
(`verify_scope`). The hash and the schema answer different questions. The hash says that the file was not modified
after it was written; the schema says that the file has the shape a reader may rely on. Neither says that the
computation behind it was right: only running `certify` again, or the independent re-check of `09_verifier.md`,
addresses that.

## The JSON schemas

Two JSON Schema documents (draft 2020-12) are published in `code/certify-ed/schema/`: `certificate-2.1.schema.json`
for the certificate body and `producer-bundle-1.0.schema.json` for `meta.json` of a producer bundle, with `spec.json`
under `$defs/model_spec`. Both are rendered from `certify_ed/schema.py` by `python -m certify_ed.schema --write
schema`, and their enumerations come from the code: statement identifiers from `certificate_v2.STATEMENT_IDS`,
statuses from `certificate_v2.STATUSES` plus `CORROBORATED`, reason codes from `scope.REASON_CODES`. The test
`tests/test_schema_conformance.py` fails when the files and the definitions drift apart, so after a change that adds a
reason code the schema has to be rendered again. Without the package `jsonschema` the loader runs only the structural
checks that JSON Schema cannot express: every statement identifier appears exactly once in the scope block, an
in-scope status equals the statement's status and is not NOT_CHECKED, every reason code used is declared and known,
and `claim.n_eigenvalues` equals the number of eigenvalues.

The schema carries a stability promise. Within major version 2 a field is never removed or re-typed, an enumeration
value is never removed and the meaning of a field does not change; new optional fields may be added, and readers must
tolerate unknown fields. Adding an enumeration value is a minor version of the schema. Removing or re-typing a field,
or changing the canonical JSON behind the body hash, is a major version and goes through
`code/certify-ed/docs/SCHEMA_RFC_PROCESS.md`. The array hashing and the canonical JSON are frozen for every 2.x
version.

## The producer bundle

A producer bundle (decision D-49) is a directory that any code able to write files can produce:

```
my_run/
  eigenvalues.<ext>      required   1-D real array of the claimed eigenvalues, in any order
  eigenvectors.<ext>     optional   n x k array; column j belongs to eigenvalue j
  operator.<ext>         optional   the producer's matrix (dense or sparse), or operator.py (matrix-free)
  basis.<ext>            optional   a 1-D list of U(1) basis states, or a 2-D sector isometry
  spec.json              optional   the declared model, ModelSpec.to_dict()
  meta.json              optional   the kind of claim and how to read the files
```

The reader chooses by extension: `.npy`, `.npz`, `.h5`/`.hdf5`/`.hdf` (with `h5py`), `.csv`/`.tsv`/`.txt`/`.dat`,
`.json`, `.mtx` (Matrix Market) and, for a matrix-free operator only, `.py`, which runs the producer's code and is
therefore accepted only with `--allow-code`. When several files share a stem, the first in the order `.npz .npy .h5
.hdf5 .hdf .mtx .json .csv .tsv .txt .dat .py` wins, and the chosen file names are recorded in `claim.producer.files`.
Sparse operators are read as SciPy `save_npz` files, CSR triplets, COO triplets or groups of the same names in HDF5,
and complex entries are accepted in each format. Non-finite entries anywhere are rejected when the file is read, so a
NaN eigenvalue never reaches the enclosure code (finding IMP-21,
`docs/failures/03_review_and_derivation_findings.md`).

A one-dimensional integer basis file is the list of basis states of a U(1) block in the spec's integer encoding (site
0 most significant), with `meta.state_order` giving the producer's order. A two-dimensional basis file is the isometry
of a symmetry-adapted block, with rows in the full-space encoding and orthonormal columns; S1 then audits invariance
and orthonormality, and the certificate says that its statements concern the sector block, with the certified shift to
the full operator. Without a basis file the operator acts on the full space of the spec. `meta.json` may give `kind`,
`window`, `accuracy_target`, `description`, `producer`, the keys of keyed formats, layout hints, `temperatures`, the
scope the producer suggests (`preset`, `only`, `skip`, budgets, which the command line overrides) and the
`environment` and `timing` written by the record hook. If a producer uses a different site order or state numbering
from the spec, the advice of `producer_bundle.md` is to write `spec.json` in the producer's convention rather than to
permute the arrays, because the spec is the declaration and S1 says whether the matrix matches it.

The scope block records what each missing file costs: no `eigenvectors.*` narrows S2 to S4 with `NEEDS_EIGENVECTORS`,
no `spec.json` narrows S1 with `NEEDS_SPEC`, and so on. `certify-ed conform` checks a bundle end to end (the schema of
`meta.json`, the import, the certification and the schema of the resulting certificate), and three reference bundles
in `code/certify-ed/conformance/reference/` serve as fixtures (`code/certify-ed/conformance/README.md`).
