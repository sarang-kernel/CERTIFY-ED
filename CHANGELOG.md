# Changelog

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

### Fixed (found while merging symveig; see docs/symveig_audit.md)
- Merged-interval multiplicities are reported as certified counts only when proved.
- Sector decompositions are rejected when the quantity is not conserved.

## [2.0.0.dev0] - 2026-10-04
v2 verification core S1–S5, sector assembly, importers, certificate v2, experiments E0–E5.

## [1.0.0] - 2026-05
CERTIFY-ED v1 (arXiv:2605.11787), now in `legacy/v1`.
