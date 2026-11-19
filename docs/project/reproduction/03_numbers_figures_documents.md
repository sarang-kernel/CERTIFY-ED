# Numbers, figures and documents

The numbers, tables and figures of the manuscript, of the consolidated report and of this documentation are made from
run records by scripts that are listed in `provenance.NON_COMPUTATIONAL`: they read records and documents and never
compute a bound. They can therefore be remade from the archived run directories without running an experiment again.
All commands are run from the root of a checkout (`01_environment.md`); paths in angle brackets are to be replaced.

## Macros and tables from a run

`numbers` runs `experiments/make_numbers.py` and `experiments/make_numbers_apps.py` on the records of one run
directory and writes `numbers.tex`, `numbers_apps.tex`, `numbers_todo.tex`, `tables/` and `figures/` into the given
output directory. It also records the regeneration in the run directory: it rewrites `manifest.json` (a new `numbers`
step and the time in `numbers_regenerated`) and `SUMMARY.md`. To keep the archived record unchanged, copy the run
directory first and run the command on the copy:

```
Copy-Item -Recurse <archive>\code\certify-ed\results\run_full_20261105 ..\runs\run_full_20261105
python -m certify_ed.cli numbers --run ..\runs\run_full_20261105 --manuscript ..\generated
```

The POSIX form is `cp -r <archive>/code/certify-ed/results/run_full_20261105 ../runs/` followed by the same `python`
line. The command prints the provenance of the two scripts (git blob, SHA-256, whether modified, source and test
fingerprints) and the checks each script makes of the statements the text makes about its numbers. For the 3.2.1 run
they are "44/44 checks pass" for `make_numbers.py` and "61/61 checks pass" for `make_numbers_apps.py`, with the E6
inspection skipped because 18 survivors await a hand verdict. A failed check prints a `CHECK FAILED` line and the
command exits with 1.

This was done on 10 November 2026 in a clone of the bundle at `a814d28` on a copy of `run_full_20261105`. The
provenance printed the blobs `a29559d6` and `e67808a2` of the two scripts, unmodified, and the fingerprints
`cb73dec3e4433a3b` and `9da2ba19ca572962`. The new `numbers.tex` (154 lines), `numbers_apps.tex` (57 lines),
`numbers_todo.tex` and the 12 files of `tables/` are byte-identical to those in `paper/` of the archived run; the
three figure PDFs (`fig_cost.pdf`, `fig_tightness.pdf`, `fig_widths.pdf`) differ only in their `/CreationDate` entry.

## Figures of the documentation and the report

Each figure is one script in `experiments/figures/`, which writes `<name>.pdf`, `<name>.png` and `<name>.numbers.json`
(the numbers the caption prints) into `--out`. Record-driven figures read the run named by `--results` or
`CERTIFY_RESULTS` and, where they compare two runs, the second run named by `--results-b` or `CERTIFY_RESULTS_B`;
`--png-only` skips the PDF.

```
$env:CERTIFY_RESULTS = "<archive>\code\certify-ed\results\run_full_20261105"
$env:CERTIFY_RESULTS_B = "<archive>\code\certify-ed\results\run_full_20261022"
python experiments\figures\fig_failure_matrix.py --out ..\figures
python experiments\figures\make_index.py --dir ..\figures
python experiments\figures\make_index.py --check --dir docs\figures_run_3.2.1
```

The POSIX form sets the variables in front of the command (`CERTIFY_RESULTS=<run> CERTIFY_RESULTS_B=<run b> python
experiments/figures/<script>.py --out <dir>`). The figure scripts only read the run directories, so they may point at
the archive. `make_index.py` renders `INDEX.md` of a figure directory from the numbers files, and `--check` exits with
1 when `INDEX.md` is out of date. The record-driven figures are `fig_failure_matrix` (E1), `fig_cost_memory` (E2 of
both runs), `fig_two_stack` (the compare CSV in the directory of the first run), `fig_mutation` (E6),
`fig_crown_sectors` (E9) and `fig_aklt_eta` (E10); the concept and document-driven figures (`fig_certificate_anatomy`,
`fig_enclosure_geometry`, `fig_inertia_completeness`, `fig_rounding_accounting`, `fig_corpus_detection`,
`fig_scope_vs_survey`, `fig_roadmap_progress`) read no run. On 10 November 2026 the six record-driven scripts were run
in the clone with the two variables above and `--png-only`; every `numbers.json` they wrote is equal, key for key, to
the one in `docs/figures_run_3.2.1/` of the repository. The copies in `docs/project/experiments/figures/` of this
documentation are those files. `make_index.py --check` reported `INDEX.md is in sync` for `docs/figures/` and for
`docs/figures_run_3.2.1/`.

## The manuscript

The manuscript scaffold `manuscript_v4/` is built in a copy of the archive, from the archive root, because the first
three steps write into `manuscript_v4/generated/`. That directory holds the macros of the 3.2.1 run and
`generated_3.1.0/` the 3.1.0 set it replaced; the two are swapped to change the cited run (`README_ARCHIVE.md`). Which
run the paper cites is Sarang's call (decision D-61); the cited version stays 3.1.0 until then, so a build that cites
it swaps the two directories first.

```
python -m certify_ed.cli numbers --run <copy of the run> --manuscript manuscript_v4\generated
python <repo>\experiments\make_survey_table.py <repo>\experiments\survey manuscript_v4\generated
python <repo>\experiments\make_scope_table.py <repo> manuscript_v4\generated <repo>\experiments\survey\survey_for_scope.json
python manuscript_v4\tools\make_figure_macros.py <repo>\docs\figures_run_3.2.1 <repo>\docs\figures manuscript_v4\generated\numbers_figures.tex
cd manuscript_v4
python tools\check_macros.py
python tools\check_duplicates.py
latexmk -pdf main.tex
```

`make_survey_table.py` writes `numbers_survey.tex` and `tables/survey_table.tex` from the coded survey, and labels the
automatically coded fractions as provisional, because the pre-registered check reading left 9 of 13 fields below 80 %
agreement (its docstring and `experiments/survey/RESULT_practice_survey.md`). `make_scope_table.py` writes
`tables/scope_matrix.tex` from `docs/SCOPE_MATRIX.md`. `make_figure_macros.py` writes the caption macros from the
numbers files of the two figure directories. The two checks report undefined and duplicate macros; the scaffold README
records no undefined macro (232 names) and no duplicate. The last built PDF in the archive is
`manuscript_v4/manuscript_v4_3.1.0.pdf` (39 pages, pdflatex and bibtex, from the 3.1.0 run; `README_ARCHIVE.md`).
`tectonic main.tex` is the alternative engine; it failed in the restricted Windows environment of 21 October
(`manuscript_v4/build_tectonic.log`). No TeX engine was available for the checks of 10 November 2026, so the PDF was
not rebuilt. The prose of the manuscript is Sarang's (decision D-16); the scaffold carries only claims with evidence
pointers.

## The consolidated report

The report sources are in `report/` of the archive, and the builder finds the repository and `project_docs/` relative
to the archive root (the parent of `report/`), unless `--repo` and `--docs` are given. It writes
`CONSOLIDATED_REPORT.md` next to itself, so it is run in a copy of the archive, not in the archive.

```
python report\check_render.py <run dir> <generated dir> [--repo DIR] [--docs DIR]
python report\build_report.py <run dir> <generated dir> <out.pdf> [--repo DIR] [--docs DIR]
```

`check_render.py` assembles the Markdown with the builder's own functions and stops before pandoc. It reports
unresolved macros, leftover placeholders, the headings and the image paths, and exits with 1 if a macro or placeholder
remains. `build_report.py` then converts to PDF with pandoc and xelatex. On 10 November 2026 `check_render.py` was run
on a copy of `report/` with the copy of the 3.2.1 run, the regenerated macros of the first section and `--repo` and
`--docs` pointing into the archive. It reported 0 unresolved macros, 0 leftover placeholders and 27 top-level
headings, and exit code 0. Its 13 image paths are relative to the archive root and were not found from the copy
outside the archive layout; this is the reason to copy the whole archive rather than `report/` alone. The PDF was not
built, because pandoc was not installed on the machine of the check. The last built PDF is
`report/report_v1.1_3.1.0.pdf`.

## The note on a candidate run

```
python experiments\make_result_candidate.py <reference run> <candidate run> <compare.csv> <out.md> [title]
```

writes a short Markdown note with the statements and headline numbers of E7, E8, E9 and E10 of both runs, the
stack-dependent bounds with their relative difference and the summary of the comparison CSV (`04_checks.md`). The
statements of the `RESULT_evidence_run_*.md` files of the repository that the records do not support are listed in
`docs/project/failures/06_found_while_documenting.md`.
