# Data snapshots used by the figure scripts

`ROADMAP_expansion_items_2026-10-29.csv`: the 78 roadmap items of `project_docs/roadmap/ROADMAP_expansion_items.csv`
(project archive v4, 2026-10-29) with one added column, `wave` (0-4), read from the wave lists of
`ROADMAP_expansion.md` (section "Waves", items written as `ID (score, effort)`); the per-wave point totals of
that document (66, 99, 76, 120, 109) are reproduced by summing `score` over the column. `status_2026-10-29` is the
status after roadmap Waves 0-1 (D-58). The column is a snapshot: the roadmap document is the source of record.

- `ROADMAP_expansion_items_2026-11-03.csv`: the same 78 items with the status column `status_2026-11-03` (after the completion plan, Phases 1-2: 43 done, 6 partial, 1 blocked, 28 not started; 270 of 470 points). The figure reads the latest dated status column.
