# Changelog

## [Unreleased] - 2026-10-21 (3.1.0 candidate; `__version__` stays 3.0.0 until the next full run)

Two soundness fixes, both found while writing the proofs for the manuscript scaffold (`ms_v3`), and
sharper application bounds.

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
  certified η now agrees with the published values to about 1e-10 (it was about 2e-6). Reported η and
  the square-octagon sum are floats rounded upward from the exact rationals.
- Pyrochlore S = 1/2 crown: a ladder of shifts (relative margins 1e-10, 1e-9, 2e-8). The certified bound
  is λ_min ≥ −0.550158299 (it was −0.550158314), within 1e-9 of the Lanczos value.
- Provenance: the post-processing scripts left the computational fingerprint; the test suite has its
  own `tests_hash`, and `--resume` re-runs E6 when the tests changed. E6 is classified
  automatically after it runs.
- `make_numbers_apps.py`: E6 table, E8 width-scaling figure (`fig_widths.pdf`), criterion values for
  AKLT, run-manifest table, tighter statement checks.
- Tests: 363.

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
