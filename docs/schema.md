# The certificate schema (2.1) and the producer-bundle schema (1.0)

Two JSON Schema (draft 2020-12) documents are published in `schema/`:

| file | identifier in the data | describes |
|---|---|---|
| `schema/certificate-2.1.schema.json` | `"schema": "certify-ed/certificate/2.1"` | the JSON body a certificate consists of (`Certificate.save`) |
| `schema/producer-bundle-1.0.schema.json` | `"schema": "certify-ed/producer-bundle/1.0"` | `meta.json` of a producer bundle, with `spec.json` under `$defs/model_spec` |

Both are rendered from `certify_ed/schema.py` (`python -m certify_ed.schema --write schema`), where the
enumerations come from the code: statement ids from `certificate_v2.STATEMENT_IDS`, statuses from
`certificate_v2.STATUSES` plus `CORROBORATED`, reason codes from `scope.REASON_CODES`. A test fails when
the files and the definitions drift apart (`tests/test_schema_conformance.py`).

## Validation

```python
cert = Certificate.load("run.certificate.json")                    # hashes + schema when jsonschema is installed
cert = Certificate.load("run.certificate.json", validate_schema=True)   # ImportError without jsonschema
cert = Certificate.load("run.certificate.json", validate_schema=False)  # hashes only
cert.validate_schema()                                             # an in-memory certificate, before or after save
```

```
pip install certify-ed[schema]                 # jsonschema
python -m certify_ed.schema run.certificate.json
certify-ed conform my_bundle/                  # a producer bundle: schema, import, certify, schema of the result
```

Without `jsonschema` the loader runs the *structural checks* only (below) and `validate_certificate_body`
returns `None`; with it, a non-conforming body raises `certify_ed.schema.SchemaValidationError`, a
`ValueError`. The integrity hash and the schema answer different questions: the hash says the file was not
modified after it was written; the schema says the file has the shape a reader may rely on. Neither says
the computation behind it was right; only re-running `certify` does (see `certificate_v2.py`).

### Structural checks (not expressible in JSON Schema)

* every statement id of `STATEMENT_IDS` appears exactly once in `scope.in_scope`, `scope.narrowed` or
  `scope.out_of_scope` (as `quantity`);
* `scope.in_scope[].status` equals the statement's status and is not `NOT_CHECKED`;
* every reason code used in `scope.narrowed` / `scope.out_of_scope` is declared in `scope.reason_codes`,
  and every declared code is in `REASON_CODES`;
* a `NOT_CHECKED` statement's `evidence.reason_code`, when present, is in `REASON_CODES`;
* `claim.n_eigenvalues == len(claim.eigenvalues)`.

## Stability promise

* Within schema major version 2, a field that exists is never removed or re-typed, an enumeration value
  is never removed, and the meaning of a field does not change. New optional fields may be added (2.1 added
  `scope` and `spectrum_enclosure`; a reader must tolerate unknown fields and the schema allows them).
* Adding an enumeration value (a new statement id, status or reason code) is a minor version bump of the
  schema (2.2), announced in the CHANGELOG and documented in `docs/SCOPE_MATRIX.md`; a 2.1 validator rejects
  such files, which is the intended signal to upgrade.
* Removing or re-typing a field, or changing the canonicalization behind `integrity.body_sha256`, is a
  major version (3.0) and goes through `docs/SCHEMA_RFC_PROCESS.md`.
* The `arrays` sidecar hashing (SHA-256 over `dtype.str`, the JSON shape list and the little-endian C-order
  bytes) and the body canonicalization (`certify-ed-canonical-json-1`: sorted keys, separators `,` and `:`,
  shortest round-trip floats, no NaN) are part of the format and frozen for 2.x.

## Field-by-field

Types are JSON types. "number*" is a number or one of the strings `"nan"`, `"inf"`, `"-inf"` (JSON has
no non-finite numbers; `_jsonable` writes them as strings). All SHA-256 values are 64 lowercase hex digits.

### Top level

| field | type | required | meaning |
|---|---|---|---|
| `schema` | const `certify-ed/certificate/2.1` | yes | format identifier |
| `subject` | object | yes | what was certified: the declared model and the operator |
| `claim` | object | yes | the producer's claim: eigenvalues, kind of completeness, provenance |
| `statements` | array of statement | yes | S1..S8 records, in order, for every statement that was reached |
| `enclosures` | array of enclosure | no | the S2 record per claimed eigenvalue (empty when S2 did not run) |
| `kato_temple` | array | no | Kato-Temple intervals where an isolation interval was available |
| `arithmetic` | object | no | the floating-point model assumed by the bounds and the run-time probe |
| `environment` | object | yes | versions, platform, time, source fingerprint |
| `hermitian_part` | object or null | no | present when the producer matrix was not exactly Hermitian |
| `operator_note` | string or null | no | how the operator was applied when it was not the producer's matrix |
| `spectrum_enclosure` | object or null | no | Prop 9 interval list for whole-spectrum claims |
| `scope` | object | yes | what was in scope, narrowed, out of scope, and why (schema `certify-ed/scope/1.0`) |
| `not_verified` | array of string | no | what the certificate does not say |
| `timing` | object | no | `total_seconds` |
| `arrays` | object | no (present when arrays were saved) | sidecar manifest |
| `integrity` | object | yes | hash of the canonical body without this field |

### `subject`

| field | type | meaning |
|---|---|---|
| `description` | string | spec name, or `operator supplied without spec` |
| `spec` | model_spec or null | `ModelSpec.to_dict()` of the declared model (schema `certify-ed/model-spec/1.0`) |
| `spec_sha256` | sha256 or null | hash of the canonical spec JSON |
| `operator.kind` | `matrix` / `spec` / `blackbox` / `stable` | how the operator was applied (`OperatorAdapter.kind`) |
| `operator.dimension` | integer | dimension of the space the statements concern (the sector block, if any) |
| `operator.matrix_sha256` | sha256 or null | hash of the canonical CSR form of a stored matrix |
| `operator.norm_upper_bound` | number* | upper bound on the spectral norm used by the bounds |
| `sector` | object or null | for symmetry-adapted blocks: `isometry_sha256`, `dimension`, `lift_shift` (certified shift to the full operator) |

`model_spec`: `schema`, `n_sites`, `local_space` (`kind` spin/boson/fermion, `q`, and kind-specific fields),
`encoding` (`site_order`, `state_order` descending/ascending, `base`), `terms` (list of
`[letters, [[coefficient, site, site, ...], ...]]` with a real coefficient or `[re, im]`), `name`,
`parameters`, `notes`.

### `claim`

| field | type | meaning |
|---|---|---|
| `description` | string | free text (`<kind> claim, <n> eigenvalues` by default) |
| `kind` | `subset` / `lowest` / `window` / `all` | what completeness is claimed |
| `window` | [number*, number*] or null | required for `kind: window` |
| `n_eigenvalues` | integer | equals `len(eigenvalues)` |
| `has_eigenvectors` | boolean | whether eigenvectors were supplied (S2-S4 need them) |
| `producer` | object | free form: `name`, `version`, `bundle`, `files` and whatever the producer recorded |
| `accuracy_target` | number or null | the producer's claimed absolute accuracy (S2 FAILED when a radius exceeds it) |
| `eigenvalues` | array of number | the claimed values, in the claim's order |
| `eigenvectors_sha256` | sha256 or null | hash of the eigenvector array (whether or not it was stored) |
| `dropped_imaginary_parts` | object, optional | `count` and `max_abs` of the finite nonzero imaginary parts of complex claimed values; the statements concern the real parts, as in 3.x (a non-finite imaginary part makes the value non-finite) |

### `statements[]`

| field | type | meaning |
|---|---|---|
| `id` | one of `S1.model_fidelity`, `S2.eigenpairs`, `S3.completeness`, `S4.eigenspaces`, `S5.corroboration`, `S6.exact_recheck`, `S7.thermodynamics`, `S8.degeneracy` | |
| `status` | `CERTIFIED` / `FAILED` / `INCONCLUSIVE` / `NOT_CHECKED`; `CORROBORATED` for S5 only | see `certificate_v2.py` for the meaning of each |
| `summary` | string | one line |
| `evidence` | object | statement-specific; `reason_code` (a `REASON_CODES` key) when `NOT_CHECKED` |
| `assumptions` | array of string | what the statement rests on (theorem references, encoding assumptions) |
| `seconds` | number | wall time of the statement |

The schema names the evidence fields each statement writes (as of 3.1.0) with their types, all optional,
so that a reader knows what to expect; it does not forbid additional ones. The important ones:

* S1: `comparison` (`max_abs_diff`, ...), `reference`, `internal_cross_construction`, `model_shift_bound`,
  `hermiticity_max_abs`, `spec_sha256`, `audit_space` / `sector_audit` for sector blocks.
* S2: `n_pairs`, `max_radius`, `median_radius`, `max_existence_radius`, `model_shift`, `sector_lift_shift`,
  `radius_refers_to`, `rigorous`, `accuracy_target`, `groups` (cluster records with `bound.eps`, `bound.eps_star`).
* S3: `dimension`, `count_from_inertia`, `claimed_certified_inside`, `missing_at_least`, `gap_lower_bound`,
  `next_eigenvalue_lower_bound`, `certified_window`, `window`, `beta_*` (backward-error bounds), `counts`;
  since 3.2 `gap_interval` (`[gap_lower_bound, "inf"]`: the gap above the highest claimed level of a 'lowest'
  claim has no upper bound because the next eigenvalue is not claimed; written only when S2 and S3 are CERTIFIED);
  for an eigenvalue-only window claim (C-02) `vectorless: true`, `n_claimed`, `count_in_window`, `count_statement`,
  `pairing`, `edges` (per edge: `nu`, `beta`, `delta`, `shifts`, `nu_below`, `nu_above`, `beta_below`, `beta_above`,
  `exact`, `eigenvalue_free`), and `missing_at_least` or `spurious_at_least` when FAILED.
* S4: `clusters` (projector error, observables, correlations, quantum numbers per cluster).
* S5: `oracles` (per solver: `max_abs_diff_vs_claim`, `n_compared`, `seconds`), `independence`, `tolerance`, `note`.
* S7: `temperatures`, `rows` (interval bounds for Z, F, U, C), `n_intervals`, `energy_offset`, `hypothesis`.
* S8: `clusters` (bracketing inertia counts per cluster).
* any statement: `budget` (the events that stopped it) and `narrowed` (`by`, `reason`) when the scope did.

### `enclosures[]`

`index`, `claimed`, `lower`, `upper`, `radius` (required); `existence_radius`, `residual_bound`,
`residual_computed`, `method` (`single` / `cluster` / `residual interval`), `group`, `group_size`, `distinct`.
`radius` includes the model shift and the sector lift shift when they apply (`S2.evidence.radius_refers_to`).

### `gaps` (since 3.2.0.dev0, roadmap C-01)

`null` unless S2 and S3 are both CERTIFIED and the claim kind is `lowest`, `window` or `all`. Otherwise an
object (`certify_ed.gaps.gap_records`): `refers_to` (what the intervals refer to, as `S2.evidence.radius_refers_to`),
`ranked` (true for `lowest` / `all`: the levels carry eigenvalue ranks, 1 = lowest), `rank_origin`, `n_levels`,
`levels[]` (`lower`, `upper`, `members` = indices into `claim.eigenvalues`, `count`, `ranks` = `[first, last]`
when ranked): maximal chains of overlapping S2 enclosures (radii include the model shift), each holding exactly
`count` eigenvalues (Corollary 5 for `lowest`, the inertia counts for `window`, count = dimension for `all`, then
the pigeonhole step of theory §6); `gaps[]` (`interval` = `[lower, upper]` rounded outward, `lower_from` / `upper_from`
= `S2` or `S3` / `null`, `within_level`, `levels` = the two level indices, `between_ranks` when ranked, `note`):
one record per pair of consecutive levels (both ends from S2), one `[0, width]` record per level with `count > 1`,
and for `lowest` claims the gap above the highest claimed level (`top_gap`, repeated in the list): lower end
`S3.evidence.gap_lower_bound`, upper end `"inf"`. No new statement: the record is a presentation of S2 and S3.

### `arithmetic`

`arithmetic` (`IEEE-754 binary64, round-to-nearest`), `unit_roundoff`, `model`, `assumptions`,
`complex_inner_product_constant`, `runtime_probe` (`standard_model_hypotheses_hold`, `failed`, `checks`).
When the probe fails, every CERTIFIED floating-point statement is downgraded to INCONCLUSIVE and
`not_verified` says why.

### `environment`

`python`, `platform`, `machine`, `numpy`, `scipy`, `created_utc` (required); `blas` (threadpoolctl
records or numpy's build dependency record), `certify_ed`, `source_hash` / `source_dirty` / `git_commit`
(the computational source fingerprint, `provenance.py`; null outside a checkout), `python_flint`,
`flint_c_library`. `distribution_sha256` is reserved (not written by 3.1; see `packaging/SIGNING.md`).

### `scope` (`certify-ed/scope/1.0`)

| field | type | meaning |
|---|---|---|
| `schema` | const | `certify-ed/scope/1.0` |
| `preset` | string or null | the preset, if one was used (`docs/presets.md`) |
| `selection` | object | `wanted`, `implied`, `narrowed`, `only`, `skip`, `narrowing_reason`: the resolved selection |
| `budgets` | object | `seconds`, `mb`, `events[]` (`statement`, `code` BUDGET_TIME/BUDGET_MEMORY, `note`, `estimate_mb`) |
| `in_scope[]` | `id`, `status` (not NOT_CHECKED), `implied` | statements that ran |
| `narrowed[]` | `id`, `by` (`default`/`dependency`/`only`/`skip`/`preset`), `reason`, `reason_code`, `status` (NOT_CHECKED) | statements the user or the defaults excluded |
| `out_of_scope[]` | `quantity`, `reason_code`, `detail`, `implied` | NOT_CHECKED statements (quantity = statement id) and the standing quantities no statement certifies |
| `requested` | `kind`, `has_eigenvectors`, `has_spec`, `has_matrix` | what the input offered |
| `options` | object | the options record |
| `operator_scope` | string | `operator`, or the sector-block note |
| `reason_codes` | object | the codes used in this certificate, each with its one-line meaning |

Reason codes (closed list, `scope.REASON_CODES`; `docs/SCOPE_MATRIX.md` says which apply to which quantity):
`NARROWED_BY_USER`, `DEFAULT_SELECTION`, `DEPENDENCY_NARROWED`, `BUDGET_TIME`, `BUDGET_MEMORY`,
`NEEDS_EIGENVECTORS`, `NEEDS_SPEC`, `NEEDS_MATRIX`, `NEEDS_WHOLE_SPECTRUM`, `NEEDS_TEMPERATURES`,
`NEEDS_DEPENDENCY`, `NEEDS_OBSERVABLES`, `NO_COMPLETENESS_CLAIM`, `NO_CLUSTERS`, `SIZE_LIMIT`,
`UNSUPPORTED_CLAIM`, `UNSUPPORTED_MODEL`, `NOT_IMPLEMENTED`, `OUT_OF_SCOPE_DYNAMICS`,
`OUT_OF_SCOPE_EXTRAPOLATION`, `NOT_APPLICABLE`.

### `spectrum_enclosure`

`complete`, `n_intervals`, `max_halfwidth`, `intervals[]` (`lo`, `hi`, `count`, `counted`, `method`).

### `arrays` (sidecar manifest)

| field | meaning |
|---|---|
| `sidecar` | file name of the NPZ next to the JSON (`<stem>.arrays.npz`); null in the HDF5 export |
| `sha256` | per array name, SHA-256 over `dtype.str` + JSON shape + little-endian C-order bytes (`array_sha256`) |
| `shapes` | per array name, the shape |
| `dtypes` | per array name, numpy `dtype.str` (`<f8`, `<c16`, ...) |

Arrays written by 3.1: `eigenvalues`, `radius`, `existence_radius`, `residual_bound`, and `eigenvectors`
when `Options(store_eigenvectors=True)`.

### `integrity`

`algorithm` (`sha256`), `canonicalization` (`certify-ed-canonical-json-1`), `body_sha256`: the SHA-256 of
the canonical JSON of the body with `integrity` removed (`arrays` included).

## Producing a conformant certificate from another tool

A third-party checker that writes certificates (rather than producer bundles for CERTIFY-ED to read) must:

1. write the body with `schema: certify-ed/certificate/2.1` and the required sections above;
2. use only the closed enumerations for `statements[].id`, `statements[].status`, `claim.kind`, and reason
   codes; a quantity it cannot certify goes into `scope.out_of_scope` with a reason code, never into a
   statement with a status of its own invention;
3. account for every one of the eight statement ids exactly once in `scope` (a statement the tool does not
   implement is `out_of_scope` with `UNSUPPORTED_CLAIM`, `NOT_IMPLEMENTED` or `NOT_APPLICABLE`);
4. write `CERTIFIED` only for a statement that rests on a stated theorem with its assumptions listed in
   `statements[].assumptions`; evidence from recomputation is `S5.corroboration: CORROBORATED`;
5. serialize the body canonically (sorted keys, `,`/`:` separators, shortest round-trip floats, no NaN;
   non-finite values as the strings above), hash it, and add `integrity`; hash each sidecar array as
   `array_sha256` does;
6. validate: `python -m certify_ed.schema the_certificate.json` or any draft 2020-12 validator against
   `schema/certificate-2.1.schema.json`, plus `certify_ed.schema.structural_check` for the scope invariants.

Producing a *bundle* for CERTIFY-ED to certify is the easier route and is what `docs/producer_bundle.md`
and `conformance/README.md` describe.
