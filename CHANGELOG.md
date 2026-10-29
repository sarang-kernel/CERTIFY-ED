# Changelog

## [3.2.0.dev0] (unreleased; branch `dev/3.2`)

`__version__` = 3.2.0.dev0. The computational source changes (certify.py, fp.py, scope.py), so records made
with it are not comparable by `source_hash` with the 3.1.0 evidence run; no statement that 3.1.0 certified
becomes *more* certified here (soundness rule), and the only status changes are downgrades to INCONCLUSIVE
under the new reason code below.

### Usability (roadmap A-01, A-02, A-09, A-04, A-05; track/core-api)
- `certify_ed.quick(H, w, V=None, spec=None, kind="lowest", **options)`: builds the Claim and Options and calls
  `certify()` (no check of its own). `Certificate.summary()` is now one sentence; the former multi-line text is
  `Certificate.report()`; `_repr_html_` / `_repr_markdown_` render a statement table in Jupyter.
- `Certificate.methods_paragraph(style="plain"|"latex")` and `certify-ed statement CERT.json` (`certify_ed/report.py`,
  outside the fingerprint): a methods-section paragraph whose every number is a certificate field, rounded in the safe
  direction (radii up, gap bounds down), with hypotheses, run-stack caveat, body hash and a BibTeX entry (concept DOI;
  version DOI placeholder).
- `evidence["next_action"]` on every INCONCLUSIVE and NOT_CHECKED statement (`certify_ed/diagnostics.py`).
- New reason code `MISSING_DEPENDENCY`: without python-flint, S6 is NOT_CHECKED instead of an import error; the QuSpin
  importer names the extra to install. `certify_ed.apps` and the exact checker itself still need python-flint.
- `Options(budget_seconds="auto")` / `--budget auto`: a default budget from a cost model fitted to seven E2 numbers of the
  3.1.0 evidence run (`certify_ed/cost.py`; advisory, never changes a status); `Options(progress=True)` / `--progress`.

### Producers (roadmap F-01, A-07, F-02, F-05; track/producers)
- F-01 `certify_ed.integrations.quspin.certify_hamiltonian(H, w, V, basis=None, static=None, spec=None, kind="lowest",
  **options)`: one call from a QuSpin `hamiltonian` and the output of `H.eigsh` / `H.eigh` to a Certificate. The
  basis and the operator-string list are taken from the object; eigenvalues are sorted for `lowest` / `window`
  claims; the Claim and ModelSpec come from `claim_from_quspin`, so a symmetry block (kblock, pblock, ...) keeps
  the S1 sector audit through `basis.get_proj`. Time-dependent Hamiltonians are refused. `spec=` certifies
  against an independently written model. Docs: `docs/integrations/quspin.md`; a draft upstream pull-request
  text proposing `quspin.tools.certify` is `integrations/quspin_PR.md` (text only). Tests:
  `tests/test_integrations_quspin.py` (full, U(1), kblock and kblock+pblock bases give the statuses of the direct
  route; unsorted eigsh output; spec override with a wrong coupling -> S1 FAILED; negative controls: a planted
  eigenvalue and a swapped eigenvector are not certified).
- A-07 producer record hook `certify_ed.record(spec, path, kind=..., window=..., fmt=...)`: `with record(...) as
  rec: ...; rec.capture(w, V, H)` (or the attributes `rec.eigenvalues`, `rec.eigenvectors`, `rec.operator`,
  `rec.basis`) writes a producer bundle (schema 1.0) on exit with `meta.environment` (Python, numpy, scipy,
  certify-ed versions, platform) and `meta.timing` (UTC start/end, wall and process seconds); the decorator
  `certify_ed.recorded(spec, path)` wraps a function returning `(w, V)` or `(w, V, H)`. Nothing is written when
  the block raises; exiting without eigenvalues raises. The bundle schema (`certify_ed/schema.py`,
  `schema/producer-bundle-1.0.schema.json`) documents the two new optional fields. Module `certify_ed/recorder.py`
  (NON_COMPUTATIONAL). Tests: `tests/test_recorder.py` (bundle passes `check_bundle` and the CLI `conform` and
  `certify --from`; statuses equal the direct Claim; decorator; U(1) basis; window kind and csv format; meta
  validates with jsonschema; negative controls: a planted eigenvalue and a planted operator entry written through
  the hook are not certified).
- F-02 file importers for other ED codes (`certify_ed/io/hphi.py`, `diagham.py`, `spinpack.py`, `alps.py`;
  `claim_from_hphi`, `claim_from_diagham`, `claim_from_spinpack`, `claim_from_alps`; docs/integrations/ed_codes.md).
  Layout sources: HPhi `energy.dat` / `Eigenvalue.dat` / binary `eigenvec` (manual 3.5: int iterations, unsigned long
  local_size, complex double x (local_size + 1), first component unused; the 4-byte-long variant is detected from the
  file size); DiagHam binary real vectors (wiki: int dimension + doubles) with complex vectors ASSUMED Re/Im
  interleaved, spectrum text files (quantum-number columns, energy last: observed convention); SPINPACK: ASSUMED
  text layouts (`table`, `keyvalue`), recorded as `layout_status` in the claim's producer record, no vectors; ALPS
  `/spectrum/sectors/<n>/energies`, `quantumnumbers`, and the `SAVE_EIGENVECTORS` eigenvector and basis tables
  (reordered into the spec encoding when a spec is given). Every reader checks header against file size and column
  counts and raises `FormatError`. The basis order of HPhi/DiagHam/SPINPACK relative to a ModelSpec is not
  established: the caller passes `operator=` and `states=`. Fixture generators: `tests/fixtures/ed_codes.py`.
  Tests: `tests/test_io_ed_codes.py` (22: layouts, truncation and header checks, round trips through certify()
  on a 6-site chain with S1-S3 CERTIFIED, MPI rank concatenation, negative controls: planted energy and swapped
  vector in the producer's files are not certified).
- `schema/certificate-2.1.schema.json` re-rendered: the file on dev/3.2 lacked the reason codes
  `MISSING_DEPENDENCY` and `UNDERFLOW_RISK` that `scope.REASON_CODES` already carried (the Wave 0 merge did not
  re-render).

### INT-01: the no-underflow hypothesis is enforced (E-06; design decision D-57, conservative option)
- The standard model behind every floating-point bound assumes no underflow: nonzero operands, radii and
  residual bounds of magnitude at least 2^-900 (`fp.UNDERFLOW_THRESHOLD`). 3.1.0 stated this and probed
  flush-to-zero, but never looked at the input. `certify()` now scans, before S2, the claimed eigenvalues,
  the eigenvector entries, the operator's stored entries (sparse data array; for a spec or matrix-free
  operator the norm bound and one matvec on a seeded random probe) and the model shift, and after the
  enclosure the radii, existence radii, residual bounds and residual norms. Exact zeros are fine.
- Any finding makes S2 INCONCLUSIVE with the new reason code `UNDERFLOW_RISK` (evidence: where, first
  indices, minimal magnitude, threshold; the summary keeps what the status would otherwise have been), and
  S3, S4, S7, S8 INCONCLUSIVE with the same code, since each rests on the S2 enclosure. S1 and S6 (exact
  arithmetic) are not affected. Statuses of inputs without a finding are identical to a run without the
  check (test `test_standard_fixtures_unchanged_by_underflow_check`); the existing fixtures have none.
- This is the conservative choice and it is reversible: `Options(underflow_check=False)` (CLI
  `--assume-no-underflow`) skips the scan and records `underflow_check: false` in the certificate's
  `scope.options` and `arithmetic.underflow_check.performed`; the certificate then carries the hypothesis
  as the user's. Rescaling the input instead of rejecting it was not implemented (it would need a
  statement about the rescaled operator). Alternatives remain open to Sarang (D-57).
- `docs/theory.md` section 1 names the hypothesis and the enforcement; `docs/SCOPE_MATRIX.md` and
  `scope.REASON_CODES` carry the code. Tests: `tests/test_underflow_check.py` (19: planted subnormal
  eigenvector entry, planted 2^-950 dense and sparse matrix entries, a radius of 1e-274 from an exact
  eigenpair of a 1e-258-scaled operator with eigenvectors of norm 1e125, tiny claimed eigenvalue,
  matrix-free probe, spec operator probe, complex components, opt-out recorded, CLI flag, fixtures
  unchanged, LAPACK numerical zeros not flagged, negative control: a finding never leaves S2 or a
  dependent statement CERTIFIED).
- **INT-05 (new known issue, found while implementing INT-01; needs a design decision).** The scan covers
  the stored inputs and the reported outputs. The *intermediate* quantities of the Theorem 1 bound are not
  covered by the hypothesis as stated, and routine inputs violate it there: `scipy.linalg.eigh` returns
  eigenvector entries of order 1e-265 (numerical zeros of a symmetry sector; above 2^-900), for which the
  residual vector R and the entrywise rounding bound E = gamma (|H||x| + |x||mu|) have entries of order
  1e-278 (below 2^-900), and their squares in `np.linalg.norm` underflow to zero. The lost mass is tiny
  compared with the guards (1 + 8u) of `fp.norm2_upper`, but saying so is a bound the theory does not
  contain (D-12): either the hypothesis is strengthened to cover the intermediates (which would make
  almost every LAPACK input INCONCLUSIVE), or the standard model is replaced by the underflow-tolerant
  model fl(a op b) = (a op b)(1 + delta) + eta, |eta| <= u lambda (Higham 2002, eq. 2.6; Demmel 1984)
  with an absolute term carried through `norm2_upper`, `residuals` and the Gram bounds. An early version
  of this branch scanned R and E and flagged `test_wrong_bond_sign_is_localized`; it was withdrawn. Test
  `test_lapack_eigenvectors_with_numerically_zero_entries_are_not_flagged` pins the observation.

### INT-04: code-facing accounting of the Theorem 1 radius (documentation)
- `docs/theory.md` section 2 now lists, line by line, how `enclosure.residuals` evaluates the Theorem 1
  radius in floating point: three roundings each covered by a factor (1 + 4u), the norm bounds of
  `fp.norm2_upper` / `fp.norm2_lower` with their (1 + gamma_{n+2}) and (1 + 8u) guards, and the model shift
  added afterwards with `add_up`. The code's bound is unchanged. The manuscript's Appendix A (k <= 6
  roundings, 1 + 8u, shift multiplied in) is to be rewritten from this list by Sarang (D-16).
  Test `tests/test_theorem1_accounting.py` recomputes the radius of a fixed 3 x 3 case with `Fraction`
  arithmetic and pins the code's value between the exact radius and the documented-factor evaluation.

### Merged from `fix/3.1.1` (f441227)

Two of the known issues of 3.1.0, fixed without touching what any statement asserts (merged with a merge commit;
under D-44 a version the paper cites needs its own full run).

- **INT-02.** `Certificate.verify_scope()` checks the scope block: every statement id exactly once (in scope, narrowed,
  or out of scope), known reason codes, every statement in scope present, a narrowed statement that is present is
  `NOT_CHECKED`, nothing reported outside the scope lists; `None` for certificates written before the scope block.
  `verify_integrity()` reports `scope_ok`/`scope_problems` and fails on `False`, so `Certificate.load()` refuses such a
  certificate even when its body hash was recomputed.
- **INT-03.** The exported `radius` and `existence_radius` arrays add the model shift with upward rounding (`add_up`), as
  the statements do; without a shift the radii are exported unchanged. The builder's rounding bound makes the shift
  nonzero for almost every certificate, so this changes almost every exported radius by at most one ulp.
- Tests: `tests/test_fix_3_1_1.py` (12; including a writer that drops a statement and recomputes the hash).
  Full suite 556 passed, 1 skipped; ruff clean.
### Format, packaging and governance (roadmap B-02, F-09, A-03, H-01, H-05, B-09, G-05, G-12; track/format-packaging)

No statement, bound or number changes in this track; the version is bumped in the known-issues track.
`certify_ed/schema.py` and `certify_ed/conformance.py` are listed in `provenance.NON_COMPUTATIONAL`.

- **B-02.** Published JSON Schema (draft 2020-12) of the certificate body, `schema/certificate-2.1.schema.json`,
  and of the producer bundle, `schema/producer-bundle-1.0.schema.json`, rendered from `certify_ed/schema.py`
  (the enumerations are the code's; a test keeps the files in sync). `Certificate.load(validate_schema=None|True|False)`
  validates when `jsonschema` is installed (extra `schema`) and otherwise runs the structural checks (every
  statement id exactly once in the scope block, reason codes declared and known, in_scope statuses equal to the
  statements); a non-conforming body raises `SchemaValidationError`. This also closes INT-02 independently of
  `fix/3.1.1`. `docs/schema.md`: field by field, stability promise, how a third-party writer conforms.
- **F-09.** `certify-ed conform DIR...` (`certify_ed/conformance.py`): bundle schema, import, certify, certificate
  schema, expected statuses. Three reference bundles under `conformance/reference/` (dense text, npz, matrix-free)
  with `expected.json`; `conformance/README.md` for third-party writers. Negative controls: unknown status,
  unknown reason code, CORROBORATED outside S5, duplicated/missing scope entries, tampered files with a recomputed
  hash, a bundle with a planted wrong eigenvalue.
- **A-03.** Extras `exact`, `quspin`, `hdf5`, `plots`, `schema`, `all` (the union), `dev`; `MANIFEST.in` ships the
  schemas, the reference bundles, docs and tests in the sdist. `python -m build` gives an sdist and a wheel; the
  wheel installs into a fresh venv and `certify-ed env` runs. `packaging/conda-forge/meta.yaml` (noarch draft),
  `packaging/RELEASE_CHECKLIST.md` (version bump, annotated tag, full run per D-44, checksums, sigstore, Zenodo
  new version under 10.5281/zenodo.20066565, GitHub release; never a force-push).
- **H-01.** `.github/workflows/ci.yml` replaces `tests.yml`: lint, build, schema sync; ubuntu/macos/windows x
  Python 3.11/3.12 with `.[all]` and QuSpin 1.0.1 (`--no-deps` after its real dependencies; the wheel declares the
  obsolete `importlib` backport; documented in `docs/reproducibility.md`), test log uploaded; `nightly-mutation.yml`
  runs a 32-mutant E6 sample without a failure gate.
- **H-05.** `containers/Dockerfile` and `containers/certify-ed.def` pin the 3.1.0 evidence-run stack (Python 3.11,
  numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0, QuSpin 1.0.1, h5py, matplotlib, mpmath).
- **B-09.** `packaging/SIGNING.md`: keyless sigstore signing and verification of release assets and `SHA256SUMS`.
  The certificate schema reserves `environment.distribution_sha256`; 3.1 does not write it (the hash of an installed
  distribution is not well defined; `source_hash` identifies the code from a checkout).
- **G-05 / G-12.** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `GOVERNANCE.md` (maintainers,
  decisions, release cadence, deprecation policy: schema fields never removed within a major version),
  `docs/SCHEMA_RFC_PROCESS.md`, issue templates (bug, feature, *I could not certify X* feeding `docs/SCOPE_MATRIX.md`),
  `ROADMAP.md` summarising the scored expansion roadmap.
- Tests: `tests/test_schema_conformance.py` (17), `tests/test_packaging.py` (7).

## [3.1.0] - 2026-10-22

`__version__` = 3.1.0 (tag `v3.1.0`; evidence run `results/run_full_20261022`, made from the tagged commit on the
Windows workstation with the reference library versions, D-54). Two soundness fixes found
while writing the proofs for the manuscript scaffold (`ms_v3`), sharper application bounds, and the
fixes of the independent review of 2026-10-21 (ids IMP-xx: code audit; SS-xx: second stack; MS-Nxx:
manuscript check). The computational source hash changes with this release, so the records of the run
of 2026-10-21 (source 7fb63586baf7ea6f) are superseded by the next full run.

### Known issues of 3.1.0 (found while filling the manuscript's evidence pointers, 2026-10-22)
- **INT-01.** The no-underflow hypothesis of the floating-point statements is assumed, not enforced: `fp.py` probes
  gradual underflow and denormals-are-zero, but no input check rejects operands or radii below 2^-900 (the
  manuscript said it did). Needs a design decision (reject, rescale, or state the hypothesis as the user's).
- **INT-02.** No validator rejects a certificate whose scope block omits a statement: `verify_integrity` checks the
  hashes only, and the completeness check of the scope block exists only as a test helper. Fixed on `fix/3.1.1`.
- **INT-03.** The exported `radius` and `existence_radius` arrays are `enc.radius + eps_shift` in round-to-nearest
  (`certify.py` 1161-1162) and can understate the sum by half an ulp; the statements themselves use `add_up`.
  Fixed on `fix/3.1.1`.
- **INT-04.** Appendix A's accounting of the Theorem 1 radius (k <= 6 roundings, factor 1 + 8u, model shift
  multiplied in) does not describe the code (three roundings, factor 1 + 4u, model shift added with upward
  rounding). The code's bound is sound; the text is to be rewritten (D-16).

### Evidence run of 3.1.0 (2026-10-22)
- `results/run_full_20261022`: tag `v3.1.0` (`e860a83`), computational source `acbef11af991c860` for every step, no step
  dirty; Windows workstation, Python 3.11.17, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0
  (FLINT 3.6.0), QuSpin 1.0.1 (the library versions of the Linux 3.0.0 run). Tests: 541 passed, 1 skipped, 1 warning.
  E0–E10 ok. Numbers: 61/61 and 45/45 checks, after one change to `make_numbers.py` (`55c2ef9`): the W4 check
  asserted the Linux outcome for the 4×4 J2 = 0.5 completeness (NOT_CHECKED); here S3 is CERTIFIED (gap ≥ 0.2870)
  because a job object counts committed memory where RLIMIT_AS counts address space. E6: 38 of 64 mutants killed;
  15 survivors await the human verdict of MS-N03 (`results/run_full_20261022/E6_inspection_worksheet.md`).
- Two stacks: against `run_full_20261021` (Linux, 3.0.0) `compare_runs.py` finds no MISMATCH leaf
  (`RESULT_evidence_run_3.1.0.md`). With the library versions pinned, the E9 exact lower bound and every E10
  factorization residual are bit-identical across the two operating systems (the up-to-6 % differences of the
  candidate run came from library versions); E7 exact radii still differ (ratios 0.25–1.47).
- `compare_runs.py`: named classes for prose with run-stack digits (floating-point numbers only; counts stay strict),
  records extended by a newer version, records drawn from the stack (E5 ARPACK rows, E6 mutant samples; compared by
  their invariants in the RESULT document), explained differences (`EXPLAINED`, with reasons), and timing/memory
  leaves; negative controls in `tests/test_compare_runs.py`.

### Windows portability (2026-10-22, found by the first attempt of the evidence run)
The first full run on the Windows workstation (`results/run_full_20261022_attempt1`, kept with an ATTEMPT_NOTE)
failed in four places that the Linux runs never exercised. None touched a certified statement.
- **Matrix-free importer.** scipy 1.17's `LinearOperator.matmat` calls the producer's `matvec` with (n, 1)
  columns, which a function written for vectors broadcasts to (n, n). `certify_ed/io/formats.py` now always calls
  the producer's `matvec`/`rmatvec` with a vector of shape (n,), checks that n numbers come back (FormatError
  naming the file otherwise) and builds `matmat` column by column. Test: `test_matrix_free_matvec_is_called_with_vectors_and_checked`.
- **E2, E4: memory cap and peak memory.** `resource` and `preexec_fn` do not exist on Windows. New
  `experiments/portable.py`: RLIMIT_AS on POSIX (unchanged), a job object with a per-process memory limit on
  Windows; peak working set on Windows; CPU model and RAM from the registry and GlobalMemoryStatusEx (the harness
  recorded neither on Windows, and `make_numbers.py` read /proc unguarded). E2 rows and E4 settings record the cap
  mechanism (`memory_cap`). Tests: `tests/test_portable.py` (a capped child must raise MemoryError).
- **E6: the mutants face the shipped suite.** The mutation harness copied only `certify_ed/`, `tests/`,
  `examples/`; six 3.1.0 tests read `pyproject.toml` or `experiments/` (so the baseline failed on every platform) and
  25 tests skipped in the copy against 1 in the repository. The copy is now every tracked file (`git ls-files`).
  `tests/test_driver.py`'s `needs_repo` also requires a git checkout, as `test_imp06` already did.

### Candidate run and integration (2026-10-22)
- `release/3.1.0` at `v3.1.0-candidate`: fix track + scope framework + survey/scope generators + v4 macro emission +
  `compare_runs.py` + `make_result_candidate.py` + E9 named-sector step. Full suite on the Windows integration stack
  (Python 3.12.15, numpy 2.5.4, scipy 1.18.1, python-flint 0.8.0): 502 passed, 31 skipped (30 need QuSpin, 1 needs gcc);
  ruff clean. Computational source `1d4e33529dfce716`.
- `results/run_candidate_20261021` (E7, E9, E10, E8 with the pre-merge source `e707838f6852d474`) reproduces every statement
  of `run_full_20261021`; see `RESULT_candidate_run.md`. The 3.1.0 evidence run (all experiments, tests enabled, one
  commit, no edits in flight) is still to be made on the reference stack.

### Fixed (review of 2026-10-21)
- **IMP-18 (critical).** `checker.radius_exact` returned radius 0 for a zero test vector, so a wrong
  eigenvalue claimed with a zero vector (or, in `exact_recheck_sector_lowest`, with a vector from another
  sector whose exact projection vanishes) was CERTIFIED "with no floating-point assumption". The function
  now returns None for ‖x′‖ = 0 and `exact_recheck_lowest`, `exact_recheck_sector_lowest` and S6 report
  INCONCLUSIVE; the hypothesis x ≠ 0 is stated in the docstrings.
- **IMP-21, IMP-22 (critical).** A NaN claimed eigenvalue passed every comparison: `certify()` reported S2
  and S3 CERTIFIED for a full-spectrum claim and `enclose_spectrum` returned `complete=True` with an
  eigenvalue in no interval. `enclose()` and `enclose_spectrum()` refuse non-finite claims (ValueError);
  `certify()` and the exact re-checks report S2/S3/S6 INCONCLUSIVE; the accuracy comparison is written
  fail-safe (`not (radius <= target)`); a group with a non-finite radius is never 'distinct'. A zero
  claimed vector no longer crashes S3 (IMP-23): INCONCLUSIVE with the pair named.
- **IMP-13 (major).** A correct claim with valid but non-orthogonal eigenvectors of a degenerate level was
  S2 FAILED (cluster radius 4.9 against a target 4.5e-8). Theorem 2′ (`docs/theory.md`, UNVERIFIED, D-12)
  gives the cluster radius ε* = ‖R‖/√(1−η) + ‖XᴴR‖/(1−η) + δ·spread(Λ)/√(1−η); `_cluster_bound` uses
  min(ε, ε*) and records both. Every cluster member keeps its Theorem-1 radius as `existence_radius`
  (records, arrays); the cluster radius remains the radius of the distinct matching and of every
  indexed statement. When only the cluster radius misses the accuracy target, S2 is INCONCLUSIVE with
  the diagnosis, not FAILED. The review's case is now CERTIFIED (radius 8e-14).
- **IMP-12 / IMP-20.** `compose.certify_low_energy` uses ε = ρ (Weyl for the square W) instead of
  (1+√5)/2·ρ, and records symmetrised producer blocks (`hermitian_part` per sector, `symmetrised_blocks`).
- **IMP-08.** The certified window of S3 'window' claims is rounded inward (`add_up(a, β_a)`,
  `sub_down(b, β_b)`); the same window feeds Kato–Temple and the S4 separations.
- **IMP-14.** `inertia_dense` checks that the permuted LDLᵀ factor is lower triangular, as
  `inertia_exact` does.
- **IMP-24.** `fp.probe_float_environment` detects round-toward-zero (1 + 3·2⁻⁵⁴ must give 1 + 2⁻⁵²,
  and the mirror). The rounding-mode tests run on Windows through `ucrtbase` as well as on glibc.
- **IMP-05 / SS-02.** Provenance calls git with `core.filemode=false`: a lost executable bit (Windows
  checkouts) no longer marks the computational source as dirty.
- **IMP-06.** The numbers step records its provenance (`provenance.numbers_fingerprint`: commit, blob
  hashes of `make_numbers*.py`, source and tests hashes) in `manifest.steps.numbers.provenance`.
- **IMP-07 / MS-N02.** `\NtestsTotal`, `\NtestsVtwo` (and new `\NtestsPassed`, `\NtestsSkipped`) come
  from the run's tests step (`manifest.steps.tests.summary`), not from a live collection.
- **IMP-26.** `__version__` = 3.1.0.dev0 (pyproject, `.zenodo.json`); every certificate's `environment`
  carries `source_hash`, `source_dirty`, `git_commit`, `python_flint` and `flint_c_library`.
- **SS-04.** `cli.NEEDS["E3"]` includes QuSpin (part d imports it); `experiments/harness.Run` logs
  "ABORTED" and records `_aborted` when the experiment raises, instead of "finished".
- **MS-N12.** The FLINT C-library version (`flint.__FLINT_VERSION__`) is recorded in
  `environment.json`, `certify-ed env` and certificates; macros `\FlintCVersion`, `\PythonFlintVersion`,
  `\CertifyEdVersion`.
- **MS-N03.** `e6_recheck.py` (`_provenance`) and `e6_classify.py` (`provenance`) stamp their outputs with
  the commit, `source_hash` and `tests_hash`; `make_numbers_apps.py` requires the stamps on
  `classification.json`, `recheck.json` and `inspection.json` (CHECK FAILED otherwise), checks that they
  refer to the E6 run's source, and that every "test added, mutant now killed" verdict has a recheck
  entry (`\MutRecheckKilled`, `\MutRecheckTestsHash`).
- **IMP-28 / MS-N01.** `\AKLTCritMin` is rounded down (0.44, was 0.45); a check verifies that the
  printed range contains every criterion value.
- **IMP-27 / SS-15.** The AKLT agreement check tests exactly the printed `\AKLTMaxExcess` (the fixed
  5e-9 threshold is gone).
- **MS-N04 / MS-N05.** E2 text macros (`\CSoneUoneMax`, `\CStwoMax`, `\CkTwentyTwoSone`,
  `\CkTwentyTwoMem`) come from the same run as the table cells; `\CSoneUoneMax` is the maximum over
  all S^z rows (new `\CSoneUoneAtMaxDim` for the largest block).
- **MS-N06.** S3 feasibility is stated through nnz/fill: `\CnotFitAtOrBelowMaxDim`, `\CnotFitDim`,
  `\CnotFitNnzMin` and a check that every block of dimension ≤ `\CmaxFeasibleDim` whose S3 did not fit
  has more nonzeros than the largest certified block.
- **MS-N07.** `\TcShiftFraction` names the 0.9 of E3(c) as the shift-placement fraction; the check says so;
  `\TcCasesRecorded` keeps the count of recorded rows (one duplicate) next to `\TcCases`.
- **MS-N13.** `\VaMatricesGenerated` (matrices with and without an Arb reference) next to `\VaMatrices`
  and `\VaNoRef`; `\TbN` gives n for the E3(b) ratios, whose range is now rounded outward.
- Statement check "same computational source" falls back to the records' `source_hash` when the
  recorded commits are not in the repository (rewritten history).

### Added (review of 2026-10-21)
- `tests/test_adversarial.py`: the 14 attack cases of the review (7 failed on the reviewed HEAD).
- `tests/test_review_fixes.py`: one or more regression tests per finding above.

### Added (E9 named sectors, branch feat/e9-named-sectors; changes the computational source hash)
- `e9_pyrochlore.py` step S1N: the lowest eigenvalue of the two pair-spin sectors Kranitz & Penc name,
  S~ = (0,0,0,2,0,2) and (0,2,2,0,2,2), at the printed couplings (S^z = 0 block, ARPACK, floating point),
  recorded under `spin1_printed.named_sectors`; `make_numbers_apps.py` emits `\CrownOneNamedSectorA/B`
  from it (review B3-2) and checks that both reproduce the published value to 1e-5 and lie above the
  certified all-singlet eigenvalue. `tests/test_e9_named_sectors.py`. To be merged after the candidate
  run at source e707838f6852d474 has finished (the step adds about half a minute to E9).

### Added (manuscript v4 macros, branch feat/v4-macros; post-processing only, source hash unchanged)
- `make_numbers.py` / `make_numbers_apps.py` emit every macro the v4 scaffold used from its hand-written
  `generated/numbers_todo.tex`, each with a comment naming the record and field it comes from:
  `\AKLTCritMinDown`, `\AKLTMaxExcessObserved`, `\AKLTExcessCheck` (the check's own threshold, RA-23),
  `\TbRatioMinDown`, `\TbRatioMaxUp`, `\TbPairs`, `\TcCasesUnique` (MS-N13, MS-N07), `\CSoneSzMaxTable`,
  `\CkTwentyTwoSoneTable`, `\CnnzSquareJtwoNotChecked` (MS-N04/05/06), `\NtestsRun`,
  `\NtestsAddedAfterRun` (MS-N02), `\FnPoneOrPtwo`, `\FnBothMissed`, `\FnBothMissedList` and the
  "P1 or P2" column of Table e1 (MS-H02), `\CrownOneRingEzero`, `\CrownOneRingRadiusExact`,
  `\CrownOneRingRadiusFP` (RA-09), `\CrownOneMinClassDim`, `\CrownOneBigClassDim`,
  `\CrownOneBigClassAbove` (MS-N09), `\SourceHashNew` (IMP-26), `\FlintLibVersion` (MS-N12),
  `\ExactCheckerLines` (+ per-file, MS-N11), `\DecSquareNoSymDim` (from the E10 run.log),
  `\SymveigPtwoOutside` (from docs/symveig_audit.md, RA-32), the literature values
  `\CrownHalfReducedPaper`, `\CrownHalfHeadlinePaper` (marked as such, B2-4/RA-21) and, when the record
  carries them, `\MutGapsClosedStamped` (MS-N03) and `\CrownOneNamedSectorA/B` (B3-2).
- `generated/numbers_todo.tex` is now written by `make_numbers_apps.py`: the macros whose value no record of
  the run contains, as `\todo{reason}` (nothing is substituted); empty when every macro has a record.
- `make_numbers.py` checks that no macro is defined twice across the generated files; the environment
  macros that both scripts know are written by `make_numbers_apps.py` with `\providecommand` (before this,
  `\FlintCVersion`, `\PythonFlintVersion` and `\CertifyEdVersion` were defined twice, which would have stopped LaTeX with "command already defined"; found by the duplicate-macro check, no LaTeX build was run).
- `tests/test_make_numbers_helpers.py`: the helpers behind these macros (code-line counter, pytest collection
  parser, duplicate detection, P1-or-P2 verdicts, E10 log and symveig-audit parsers,
  pending-file writer).

### Added (scope framework, branch feat/scope-framework)
- **Universal producer import** (`certify_ed.io`, `docs/producer_bundle.md`): eigenvalues, eigenvectors,
  operators and sector bases from .npy, .npz (dense, `save_npz`, CSR/COO triplets), HDF5, CSV/TSV/text,
  JSON and Matrix Market; matrix-free operators from a Python file (opt-in `--allow-code`); the
  *producer bundle* directory convention (`eigenvalues.*`, `eigenvectors.*`, `operator.*`, `basis.*`,
  `spec.json`, `meta.json`) so that any code is certified by writing files; `claim_from_file`,
  `claim_from_bundle`, `write_bundle`; non-finite input is rejected at the door (review IMP-21).
  `certify-ed certify --from DIR` / `certify-ed import DIR`.
- **Statement selection and presets** (`presets.py`, `docs/presets.md`): `Options.preset / only / skip /
  narrowing_reason / budget_seconds / budget_mb`; presets `lab-minimal`, `benchmark`, `proof-lemma`,
  `whole-spectrum`, `labels`, `full` as data; `certify-ed presets`. Budgets turn statements into
  NOT_CHECKED (reason `BUDGET_TIME` / `BUDGET_MEMORY`), never into exceptions.
- **Scope block** (`scope.py`, `docs/SCOPE_MATRIX.md`): every certificate (schema 2.1) carries `scope`
  with the preset, the statements in scope, the narrowed statements with the user's reason, and the
  out-of-scope quantities with reason codes; `Certificate.summary()` prints it.
- **New statements, all UNVERIFIED (D-12) with proofs written before the code (D-46, `docs/theory.md` §9):**
  `S7.thermodynamics` (Prop 16: interval bounds on Z, F, U, C from a complete whole-spectrum enclosure,
  dyadic interval arithmetic with enclosed exp/ln, `ivl.py`, `thermo.py`); S4 extensions (Prop 15:
  batches of observables, two-point functions, static structure factors, `correlations.py`,
  `Options.correlations`); `S8.degeneracy` (Prop 14: number of eigenvalues in each cluster by bracketing
  inertia counts, `degeneracy.py`). `spectrum.intervals_from_enclosure` / `merge_intervals` expose the
  Prop 9 interval list, recorded as `spectrum_enclosure` for whole-spectrum claims.
- Tests: 58 (import round trips), 18 (presets/scope), 8 (S7), 8 (S4 extensions), 8 (S8).

### Fixed
- **Exact 'k lowest' re-check (S6, `exact_recheck_lowest`, `exact_recheck_sector_lowest`).** A component
  of overlapping residual intervals with m ≥ 2 claimed values was taken to hold m eigenvalues
  whenever one count above it agreed. Residual intervals give *an* eigenvalue each, not distinct
  ones: two copies of the first-excited vector, claimed as "the two lowest", were certified although
  the ground state is missing. Such components are now bracketed by counts on both sides, and their
  eigenvalues are certified inside the bracket ("level enclosure"). Single-value components keep
  their residual interval: the counts on the two sides pin the index. Results carry `enclosures`
  and `next_lower_bound`. FAILED now means an unclaimed eigenvalue lies under a claimed one; a cut
  multiplet is INCONCLUSIVE (it was FAILED). `checker.certify_by_counts` is shared by both routines.
- **Outward rounding of reported endpoints.** μ − r, μ + r, σ ∓ β, separations and cut comparisons
  are formed with `fp.sub_down` / `fp.add_up` (one ulp outward). Round-to-nearest could place an
  endpoint half an ulp inside the exact interval. This affects enclosures and their records,
  cluster merging, S3 margins and gaps, Kato–Temple, S4 separations, sector assembly and
  whole-spectrum intervals.
- `inertia_exact` checks that the permuted LDLᵀ factor is exactly unit lower triangular.

### Changed
- AKLT: thresholds are placed just above the floating-point η² (margins 1e-9, 1e-7, 1e-5). The
  certified η now exceeds the published values by 2e-11 to 2e-9 (it was about 2e-6). On the cross
  and the decorated square, the remaining excess comes from the factorization term t′ − t (β/μ, β/ν).
  Reported η and the square-octagon sum are floats rounded upward from the exact rationals.
- Pyrochlore S = 1/2 crown: a ladder of shifts (relative margins 1e-10, 1e-9, 2e-8). The certified bound
  is λ_min ≥ −0.550158299 (it was −0.550158314), within 1e-9 of the Lanczos value.
- Provenance: the post-processing scripts left the computational fingerprint; the test suite has its
  own `tests_hash`, and `--resume` re-runs E6 when the tests changed. E6 is classified
  automatically after it runs.
- `make_numbers_apps.py`: E6 table, E8 width-scaling figure (`fig_widths.pdf`), criterion values for
  AKLT, run-manifest table, tighter statement checks.
- Tests: 370 before the review fixes; 7 of them close the 8 test gaps found by the mutation analysis of
  the run of 2026-10-21: 2×2 pivots with det > 0, the Dekker splitter, exact residuals with huge entries,
  the round-up probe, the SU(2) check in mixed Pauli letters, the sector compression diagnostic and the
  TFIM bonds. The review fixes add `test_adversarial.py` (14) and `test_review_fixes.py` (see the
  README for the current count).

## [3.0.0] - 2026-10-18

One package for the combined CERTIFY-ED / symveig paper.

### Added
- **Exact re-check layer** (no floating-point assumption), for every local space:
  - `checker.exact_matrix`: the declared model in the scaled basis D⁻¹HD with integer weights;
    spin-S binom(2S, k), bosons k!.
  - `exact_eigenpair_radii`, `exact_recheck_lowest`.
  - `exact_sectors.exact_recheck_sector_lowest` and `exact_count_below`: symmetry sectors with
    Gaussian characters, non-abelian groups allowed when the irreps met are one-dimensional.
  - `checker.inertia_exact`: lean exact LDLᵀ check.
- **S6.exact_recheck** statement (`Options(exact_recheck=True)`).
- **`certify_ed.spectrum.enclose_spectrum`**: whole-spectrum enclosure by sectors (formerly
  symveig). Counts are certified by the cluster theorem, conservation and Hermiticity are checked
  exactly, and rounding bounds use |A||x|.
- **`certify_ed.compat.symveig`**: the symveig 1.0.0 API on the new engine, with symveig's tests ported.
- **`certify_ed.apps.aklt`**: certified Pomata–Wei criteria for five lattices, one pipeline;
  kernel-lemma checks.
- **`certify_ed.apps.pyrochlore`**: exact S = 1/2 crown bound (now through `exact_count_below`);
  S = 1 printed-coupling check; pair-spin landscape.
- **CLI** `certify-ed` (`env`, `certify`, `spectrum`, `list`, `test`, `reproduce`, `numbers`) and
  `run.py`.
- Experiments E7 (exact benchmarks), E8 (symveig comparison), E9 (pyrochlore), E10 (AKLT).
  `make_numbers_apps.py` reads their records.
- `fp.probe_float_environment` is recorded in every experiment's environment.
- Run-time float probe, total-spin and momentum label certification, manufactured and metamorphic
  tests, mutation analysis E6 with recheck and inspection tooling (2026-10-17).
- **Provenance** (`certify_ed.provenance`): every experiment record and every reproduction step
  carries a fingerprint of the computational source (`certify_ed/` and `experiments/` without the
  command line, test helpers and this module). Both numbers scripts check that all records they read
  share one fingerprint, with no uncommitted changes.
- `certify-ed reproduce` / `run.py`:
  - `--jobs N` runs experiments in parallel, longest first; E2, E6 and E8 always run alone;
  - `--resume` keeps the steps of an interrupted run that finished with the same source;
  - unknown experiment names are rejected.
- `apps.pyrochlore.spin_one.printed_coupling_check` also re-derives the six-ring energy in exact
  arithmetic (scaled spin-1 basis). The S = 1 contradiction now needs no floating-point assumption.
- Tests (356) for every AKLT lattice geometry, the spin-1 pair-spin tools, the sector-decomposition
  audits (coverage, orthogonality, trace moments, certified spectral shift) and the driver
  (parallel jobs, resume, numbers on partial runs), and for six gaps found by mutation analysis of the
  final tree (internal builder-disagreement path, 1-D eigenvectors, dense non-Hermitian producers,
  fermionic translation sign, float-model caveat, square-lattice default).
- Tooling: `ruff check .` is clean (configuration in `pyproject.toml`). CI runs lint, builds the
  wheel, runs the suite on Python 3.10–3.12 and does a quick reproduction. SPDX license metadata.

### Changed
- Package metadata in `pyproject.toml` (`setup.py` removed). Version 3.0.0. Authors: Sarang Vehale,
  Ritu Goel.
- Tests moved from `tests/v2/` to `tests/`. Helpers moved to `certify_ed.testing`.
- `Certificate` is the v2 certificate class (`CertificateV2` is kept as an alias).
- Experiment results can be directed with `CERTIFY_RESULTS`.

### Removed (moved to `legacy/`)
- v1 modules (`hamiltonian`, `oracles`, `certificates`, `observables`, `symmetries`), `validators/`,
  v1 tests and examples. Tag `v1.0.0-baseline`.
- symveig 1.0.0 (kept unchanged in `legacy/symveig_1_0_0` for E8).
- The script versions of the applications and their certificate files (2026-10-17).

### Fixed
- The numbers step compared whole commits, so a run that crossed a commit touching only the command
  line failed its provenance check. It now compares computational sources.
- The E2 check "the N=22 momentum block has the largest S1 time" compared runs instead of blocks. It
  failed when the same block's run with S3 took longer than its run without.
- Mutation workspaces now include `examples/`, which the test suite runs.

### Fixed (found while merging symveig; see docs/symveig_audit.md)
- Merged-interval multiplicities are reported as certified counts only when proved.
- Sector decompositions are rejected when the quantity is not conserved.

## [2.0.0.dev0] - 2026-10-04
v2 verification core S1–S5, sector assembly, importers, certificate v2, experiments E0–E5.

## [1.0.0] - 2026-05
CERTIFY-ED v1 (arXiv:2605.11787), now in `legacy/v1`.
