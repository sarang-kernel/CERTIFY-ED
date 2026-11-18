# Conformance: is my producer bundle one CERTIFY-ED can certify?

A *producer bundle* is a directory any exact-diagonalization code can write (`docs/producer_bundle.md`).
This directory holds three reference bundles and the tool that checks a bundle of yours against them.

```
certify-ed conform my_run/                 # schema, import, certify, schema of the certificate
certify-ed conform my_run/ --json report.json -o my_run/certificate
python conformance/check_bundle.py my_run/ # the same without installing the console script
```

Exit code 0 and the last line `CONFORMANT` mean: the directory has the required files, `meta.json` and
`spec.json` conform to `schema/producer-bundle-1.0.schema.json`, the arrays were read and agree in shape,
`certify()` ran, and the certificate it wrote conforms to `schema/certificate-2.1.schema.json`. Whether
the statements come out CERTIFIED is a property of your data, not of conformance; the report prints them.
When `expected.json` is present in the bundle (`{"S2.eigenpairs": "CERTIFIED", ...}`), the statuses are
compared as a seventh step, which is how the reference bundles are regression-tested.

The steps and what a failure at each one means:

| step | checks | a failure means |
|---|---|---|
| `layout` | `eigenvalues.<ext>` exists, one file per stem is chosen | wrong file names or extensions |
| `meta_schema` | `meta.json`, `spec.json` against the bundle schema | a misspelled field, a wrong enumeration value, `kind: window` without `window` |
| `import` | arrays read; eigenvectors `n x k` with `k` eigenvalues; no NaN/inf; real eigenvalues | layout (`eigenvectors_layout: rows`), a key name (`keys`), a complex eigenvalue, a non-finite entry |
| `certify` | `certify(claim, spec, options)` completes | no operator and no spec (nothing to certify against); an operator whose dimension differs from the spec's |
| `certificate_schema` | the certificate body validates | a bug in CERTIFY-ED: report it |
| `expected_statuses` | statuses equal `expected.json` | the data changed, or the checker did |

Without the optional `jsonschema` package (`pip install certify-ed[schema]`) the two schema steps run the
structural checks only and say so in the report; `--require-jsonschema` turns that into a failure.

## The reference bundles

All three are the S=1/2 Heisenberg chain on 6 sites (dimension 64), eigenpairs from `scipy.linalg.eigh`,
written by `make_reference_bundles.py`; diff yours against the one closest to your format.

| bundle | arrays | operator | what it exercises |
|---|---|---|---|
| `reference/dense-arrays/` | `eigenvalues.csv`, `eigenvectors.csv` (columns) | `operator.txt`, dense 64 x 64 | text formats written by any language |
| `reference/npz/` | `eigenvalues.npz`, `eigenvectors.npz` | `operator.npz` (`scipy.sparse.save_npz`) | the NumPy/SciPy route |
| `reference/matrix-free/` | `eigenvalues.csv`, `eigenvectors.csv` | none: `spec.json` only, preset `lab-minimal` | a producer that does not store its matrix; the statements concern the reference matrix built from the declared model |

Each has `spec.json` (`ModelSpec.to_dict()`, schema `certify-ed/model-spec/1.0`), `meta.json`
(`kind: lowest`, a `producer` record, a `description`) and `expected.json`.

## Writing a bundle from a code that is not Python

1. Dump the eigenvalues as one column of text (`eigenvalues.csv` or `.txt`; `#` comment lines are skipped).
2. Dump the eigenvectors as a table with one column per vector (`eigenvectors.csv`), or one row per vector
   with `"eigenvectors_layout": "rows"` in `meta.json`. Complex entries as `1+2j`, or `(re, im)` column
   pairs with `"complex_pairs": {"eigenvectors": true}`.
3. Dump the matrix as a dense table (`operator.txt`), as COO triplets `i j value` with
   `"layout": {"operator": "coo"}` and `"operator_shape": [n, n]` (`"one_based": true` for Fortran
   indices), or as Matrix Market (`operator.mtx`). Without a matrix, the declared model is used.
4. Write `spec.json`. The simplest way is one line of Python once:
   `python -c "import json, certify_ed as ce; json.dump(ce.heisenberg_chain(6).to_dict(), open('spec.json','w'), indent=1)"`
   or build `ModelSpec` from your own term list (`docs/api.md`). The spec is the declaration; S1 says
   whether your matrix matches it, so write the spec in *your* site and state conventions.
5. Write `meta.json`: at least `{"schema": "certify-ed/producer-bundle/1.0", "kind": "lowest"}`
   (`subset` when you make no completeness claim; `window` with `"window": [a, b]`; `all` for a whole spectrum).
6. `certify-ed conform my_run/`.

`write_bundle(...)` in `certify_ed.io` writes a bundle from Python and is what the generator here uses.

## Tests

`tests/test_schema_conformance.py` checks that the reference bundles are conformant, that they are what the
generator writes, and the negative controls: a bundle with a planted wrong eigenvalue is not conformant
against an `expected.json` that says CERTIFIED (S2 is FAILED), a bad `meta.json` fails the schema step, and
an empty directory fails at layout.
