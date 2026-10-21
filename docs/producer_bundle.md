# The producer bundle (`dd if=` for ED results)

Any code that can write files can be certified. A *producer bundle* is a directory:

```
my_run/
  eigenvalues.<ext>      required   1-D real array, the claimed eigenvalues (any order)
  eigenvectors.<ext>     optional   n x k array; column j belongs to eigenvalue j
  operator.<ext>         optional   the producer's matrix (dense or sparse), or operator.py (matrix-free)
  basis.<ext>            optional   the producer's sector basis (see below)
  spec.json              optional   the declared model, ModelSpec.to_dict()
  meta.json              optional   what kind of claim this is, and how to read the files
```

```
certify-ed certify --from my_run                      # default preset: everything that fits
certify-ed certify --from my_run --preset lab-minimal # S1 + S2 only
certify-ed certify --from my_run --only S1,S2,S3 --budget-seconds 600 -o my_run.certificate
```

In Python: `claim, spec, meta = certify_ed.claim_from_bundle("my_run")`, then `certify(claim, spec, Options(...))`;
`certify_ed.write_bundle(...)` writes one (also what the round-trip tests use).

## Formats (`<ext>`)

The reader decides by extension. The same reader serves eigenvalues, eigenvectors and bases
(`certify_ed.io.read_array`), and the operator (`certify_ed.io.read_operator`).

| ext | arrays | operator | notes |
|---|---|---|---|
| `.npy` | yes | dense | |
| `.npz` | yes (`keys`) | `scipy.sparse.save_npz`, CSR triplet `H_data/H_indices/H_indptr/H_shape`, COO `row/col/data/shape`, or dense | default key names: see below |
| `.h5` `.hdf5` `.hdf` | yes (`keys`, nested paths allowed) | group `data/indices/indptr/shape` (CSR) or `row/col/data/shape` (COO), or a dense dataset | needs `h5py` |
| `.csv` `.tsv` `.txt` `.dat` | numeric columns; `#` and `%` comment lines and one header line are skipped; complex entries as `1+2j`; `(re, im)` column pairs with `complex_pairs` | dense table, or COO triplets `i j value [imag]` (`layout: coo` in meta, or the header CERTIFY-ED writes) | |
| `.json` | list (nested for 2-D) or object with the array under a key; complex as `{"re":..,"im":..}` or `"1+2j"` | dense nested list, or `{"format": "coo"\|"csr", "shape": [n,n], ...}` | |
| `.mtx` | Matrix Market | Matrix Market (coordinate or array) | |
| `.py` | – | matrix-free: `matvec(x)` and `dim` (or `shape`), or `operator()` returning a `LinearOperator` | runs the producer's code: only with `--allow-code` / `allow_code=True` |

When several files share a stem, the first in this order wins: `.npz .npy .h5 .hdf5 .hdf .mtx .json .csv
.tsv .txt .dat .py`; the chosen file names are recorded in the certificate (`claim.producer.files`).

Default key / dataset names tried when `meta.keys` does not name one: eigenvalues `eigenvalues, evals, E,
energies, w, values, lambda`; eigenvectors `eigenvectors, evecs, V, vectors, psi, X`; operator `H, operator,
matrix, A, hamiltonian`; basis `basis, states, isometry, V, projector, P`. A file holding a single array
needs no key.

Non-finite entries (NaN, inf) anywhere are rejected at the door; a NaN eigenvalue never reaches the
enclosure code (review IMP-21).

## The basis file

Two forms, told apart by shape:

* **1-D integer array** — the list of basis states of a U(1) block in the spec's integer encoding
  (site 0 most significant, local states numbered as in the spec's local space; e.g. QuSpin's
  `basis.states`). `meta.state_order` says how the producer ordered them (`descending`, the QuSpin
  convention, or `ascending`); the default is the spec's `state_order`.
* **2-D (dense or sparse) n_full x n_sector matrix** — the isometry of a symmetry-adapted block (momentum,
  parity, ...), rows in the spec's full-space encoding, columns orthonormal. S1 then audits invariance
  and orthonormality (`sectors.audit_sector`), and the certificate says the statements concern the
  sector block, with the certified shift to the full operator.

Without a basis file the operator is taken to act on the full space of the spec.

## meta.json

All fields optional.

| field | meaning | default |
|---|---|---|
| `schema` | `certify-ed/producer-bundle/1.0` | |
| `kind` | `subset` \| `lowest` \| `window` \| `all` (what completeness is claimed; see `Claim`) | `subset` |
| `window` | `[a, b]` for `kind: window` | |
| `accuracy_target` | absolute accuracy the producer claims for its eigenvalues; S2 is FAILED when a certified radius exceeds it | `1e-8 * max(1, ‖H‖)` |
| `description` | free text | |
| `producer` | `{"name": ..., "version": ..., ...}` copied into the certificate | `{"name": "producer bundle"}` |
| `keys` | `{"eigenvalues": "...", "eigenvectors": "...", "operator": "...", "basis": "..."}` for keyed formats | conventional names |
| `complex_pairs` | `{"eigenvectors": true}` when a text file holds `(re, im)` column pairs | false |
| `eigenvectors_layout` | `columns` (default) or `rows` | inferred when unambiguous |
| `layout` | `{"operator": "coo"}` for text triplets | `auto` |
| `operator_shape`, `one_based` | for COO text without a header | |
| `state_order`, `basis_label` | for a 1-D basis file | spec's order / `producer basis` |
| `temperatures` | list of T for S7 (whole-spectrum claims) | none |
| `preset`, `only`, `skip`, `budget_seconds`, `budget_mb` | the scope the producer suggests; the command line overrides | |

## What the certificate says about the import

`claim.producer` records the bundle path, the chosen files and `meta.producer`; `subject.operator` records
the matrix hash; the `scope` block records which statements ran and why the others did not (for instance
`NEEDS_EIGENVECTORS` when there is no `eigenvectors.*` file, `NEEDS_SPEC` when there is no `spec.json`).
See `docs/presets.md` and `docs/SCOPE_MATRIX.md`.

## Writing a bundle from another code

The files are plain: `numpy.save`, `h5py`, a CSV writer, `mmwrite` or a text dump all work. For a
Fortran/C++/Julia code without Python, text or Matrix Market is enough; a 4096 x 4096 dense operator
in `.txt` is large but reads correctly. If the producer uses a different local-state numbering or site
order than the spec, write `spec.json` in the producer's convention (the spec is the declaration; S1 says
whether the matrix matches it) rather than permuting the arrays.
