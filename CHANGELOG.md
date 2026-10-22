# Changelog

## [3.1.0] - 2026-10-22

`__version__` = 3.1.0 (tag `v3.1.0`; evidence run `results/run_full_20261022`, made from the tagged commit on the
Windows workstation with the reference library versions, D-54). Two soundness fixes found
while writing the proofs for the manuscript scaffold (`ms_v3`), sharper application bounds, and the
fixes of the independent review of 2026-10-21 (ids IMP-xx: code audit; SS-xx: second stack; MS-Nxx:
manuscript check). The computational source hash changes with this release, so the records of the run
of 2026-10-21 (source 7fb63586baf7ea6f) are superseded by the next full run.

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
