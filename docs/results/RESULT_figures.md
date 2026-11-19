# RESULT — Figures track (branch `track/figures`, 2026-10-30)

Base: `dev/3.2` = 38b0212 (certify-ed 3.2.0.dev0). Head: see the last line of this file. Data: the project
archive v4 (`code/certify-ed-3.1/results/run_full_20261022`, the Windows 3.1.0 evidence run; `run_full_20261021`,
the Linux 3.0.0 run; `survey/`, `project_docs/roadmap/`, `review/wave01/RESULT_integration_3.2.md`) and the
repository (`corpus/results/certify-ed/report.json`, `docs/SCOPE_MATRIX.md`, `experiments/survey/`).

## Items

| id | figure | status | notes |
|---|---|---|---|
| C1 | `fig_certificate_anatomy` | done | blocks of the certificate body (schema 2.1 keys, statement ids of `certificate_v2.STATEMENT_IDS`), statuses as an example; dependency panel S1 → S2 → S3 → S4/S8, S6 exact, S7 from the whole-spectrum enclosure, S5 corroboration |
| C2 | `fig_enclosure_geometry` | done | four rows on one number line: Thm 1 radii, Thm 2/2′ cluster radius ε, Kato–Temple with (α, β), S3 inertia with collar σ ± β and λ_{k+1} ≥ σ − β |
| C3 | `fig_inertia_completeness` | done | counting-function staircase: the sandwich (a), the vectorless window count of C-02 (b), the refusals ν > k and collar hit (c) |
| C4 | `fig_rounding_accounting` | done | the six steps of theory.md §2 with their factors (1 + 4u), the norm guards and the added shift; footer names the slack and the pinning test |
| E1 | `fig_failure_matrix` | done, with a deviation | 15 cases × {P1, P2, P3 outcome, statement named, localizes}; the record stores per-statement statuses only for F0, so the "statement named" column is derived from `P3_detail` by keyword rules (stated in the caption and the script) |
| E2 | `fig_cost_memory` | done | `certifier.total` and `certifier.maxrss_mb` against dimension for the 11 E2 blocks on both stacks, S3 outcome by marker fill, the 4.5 GiB cap |
| E3 | `fig_two_stack` | done | stacked bars per experiment of the compare-CSV classes; the CSV's class names are used (`float-differs` = relative ≤ 1e-9, plus `extended` and `text-numbers`, which the plan did not name); 0 MISMATCH in 3 885 leaves |
| E4 | `fig_mutation` | done | per-module killed/survived with the five survivor categories of `classification.json`; the dev/3.2 16-mutant sample (13 killed, three named survivors) parsed from RESULT_integration_3.2.md because its record directory is not in the repository |
| E5 | `fig_corpus_detection` | done | statuses per planted-error class and statement, refusals hatched, `must_not_certify` framed, verdict column |
| E6 | `fig_scope_vs_survey` | done | survey percentages with the JSON's Wilson intervals, coloured by scope-matrix row status; the mapping table `STATUS_OF` is written from SCOPE_MATRIX.md on dev/3.2; per-paper coverage count from `survey_coded.csv` |
| E7 | `fig_roadmap_progress` | done | points by wave and items by theme, status of 2026-10-29; the wave column was added to the CSV snapshot from the wave lists of ROADMAP_expansion.md (point totals 66/99/76/120/109 reproduced) |
| A1 | `fig_crown_sectors` | done, with a deviation | the record's landscape is at the re-optimised couplings (`spin1_landscape.couplings`), not at the printed ones; drawn as recorded and said so. The named sectors have floating-point energies and no certified interval in the record; drawn as such (open markers, "(FP)") |
| A2 | `fig_aklt_eta` | done | certified η against the published values, thresholds 1/z, the square-octagon joint criterion, criterion values against 1 |
| index | `docs/figures/INDEX.md` | done | rendered by `experiments/figures/make_index.py` from the `.numbers.json` files; filename, caption, data source (record path and run), script |
| placement | `docs/figures/MANUSCRIPT_PLACEMENT.md` | done | section and report chapter per figure, `\caption` scaffolds with the existing macros; 30 macros still needed are named with their record field |
| tests | `tests/test_figures.py` | done | 17 tests with the records, 11 passed + 6 skipped without them |
| provenance | `certify_ed/provenance.py` | done | `experiments/figures` added to `NON_COMPUTATIONAL` |

E3 "tightness" of the plan step is the manuscript's existing `fig_tightness` (make_numbers.py); it was not
duplicated.

## Proof documents written

None. This track derives no mathematics; the concept figures restate theory.md as drawn.

## Verification counts

- Figure scripts: 13, each run on the archive records; layout self-check (`_common.check_layout`: no two text
  boxes overlap) clean on every committed figure except where a label intentionally extends past the axes
  frame (reported as "outside figure" by the bbox check, visible in the saved PNG). Every figure was viewed
  after rendering and corrected until legible.
- Deterministic output: a PDF regenerated from the same inputs is byte-identical (no CreationDate).
- INDEX.md regenerated from the numbers files: `make_index.py --check` passes (tested).

## Files touched

- `certify_ed/provenance.py` (one tuple entry)
- `experiments/figures/_common.py`, `fig_certificate_anatomy.py`, `fig_enclosure_geometry.py`,
  `fig_inertia_completeness.py`, `fig_rounding_accounting.py`, `fig_failure_matrix.py`, `fig_cost_memory.py`,
  `fig_two_stack.py`, `fig_mutation.py`, `fig_corpus_detection.py`, `fig_scope_vs_survey.py`,
  `fig_roadmap_progress.py`, `fig_crown_sectors.py`, `fig_aklt_eta.py`, `make_index.py`
- `experiments/figures/data/ROADMAP_expansion_items_2026-10-29.csv`, `data/README.md`
- `docs/figures/*.pdf`, `*.png`, `*.numbers.json` (13 each), `INDEX.md`, `MANUSCRIPT_PLACEMENT.md`
- `tests/test_figures.py`
- `mkdocs.yml` (Figures nav section), `CHANGELOG.md`, `RESULT_figures.md`

## Tests added and negative controls

- `tests/test_figures.py`: one parametrised smoke test per script (exit 0 and PNG + numbers file written; exit
  code 3 "data absent" is a skip), an index-present check, `make_index.py --check`, a concept figure run with
  a non-existent results directory (must pass), and the negative control
  `test_data_absent_exit_code_is_a_skip`: a record-driven script pointed at an empty results directory exits
  with code 3 and writes no PNG (so a missing record can never register as a pass).

## What was not done, and why

- No per-statement status matrix for E1 in the strict sense: the E1 record does not store statuses per
  statement for the failing cases (only the outcome string and the diagnosis). A future `e1_failure_matrix.py`
  could write `P3_statuses` per case; the figure would then switch to reading it.
- The E9 landscape at the printed couplings: not in the record (the record holds it at the re-optimised
  couplings); nothing substituted.
- No certified intervals for the two named crown sectors: the record has floating-point energies only.
- The 3.2 mutation sample is read from RESULT_integration_3.2.md, not from a record (the record directory of
  that run is not in the repository); the figure says so.
- The manuscript (`manuscript_v4`) was not edited (task); `MANUSCRIPT_PLACEMENT.md` proposes the placement.
- Docs build (`mkdocs build --strict`) was not run here (mkdocs is not in the environment); the nav entries point
  at existing files and the Markdown uses relative image paths within `docs/figures/`.

## Final checks

- `ruff check certify_ed tests experiments conformance corpus`: All checks passed.
- `pytest -q tests` (full suite, PYTHONSAFEPATH removed, CERTIFY_RESULTS pointing at the archive run):
  `970 passed, 2 skipped, 5 warnings in 2025.26s (0:33:45)` (953 of dev/3.2 plus the 17 of test_figures.py).
- Branch head: `track/figures` = 00e391e (+ this amend).
