# Defects, failures and open gaps

This folder records everything that went wrong in the project and how each problem was found, explained and closed, or
why it is still open. It is the most direct answer to the question how far the evidence can be trusted: a checker is
believable only if the record of its own errors is complete. The main source is the defect chapter of the consolidated
report, `report/08_defects.md`, with the decision log, the review files in `review/`, the result files of the code
repository and the run records.

| Document | Contents |
|---|---|
| `01_defect_ledger.md` | every defect from the first draft to 21 October 2026: the audits of v1, the C1 pilot, the version 2 core, the delivery check, mutation analysis, the symveig merge, and the two soundness gaps found by writing proofs against the code |
| `02_integration_findings.md` | INT-01 to INT-05, the places where the manuscript and the code of 3.1.0 disagreed, and the underflow issue of routine LAPACK output |
| `03_review_and_derivation_findings.md` | the 179 findings of the independent review of 21 October 2026, with its critical and major findings, and the verdicts of the independent re-derivation of the 22 derivation documents, with the two ERROR rows and the five GAP rows |
| `04_failed_runs_and_attempts.md` | every failed run, build and attempt: seeding, the first E5 run, an empty figure, memory failures, the provenance check, the candidate run, attempt 1 of the 3.1.0 run, the 3.2.0 release sample, the E4 crash of the 3.2.0 run, the number-script check of 3.2.1, the figure index, the toolchains |
| `05_open_items.md` | what is still open: Sarang's check of the derivations, the E6 verdicts and the seven soundness-relevant test gaps, the memory gate, the cited version, the known issues of 3.1.0, and the acts outside the code |
| `06_found_while_documenting.md` | inaccuracies of the result files and limits of the comparison script found while these documents were written |

## How the defects were found

The defect chapter of the report names five routes, and the record bears it out. Writing a proof against the code
found the ordering hypothesis of the sector assembly (decision D-23) and both soundness gaps of 21 October 2026
(decision D-46). Independent references found the Jordan-Wigner error (E0, decision D-21). Running at scale or on real
data found the ARPACK truncation (E5), the memory failures, the seeding problem and the AKLT excess. Mutation analysis
found 34 test gaps over five runs (6, 12, 2, 6 and 8, per `report/08_defects.md`). The fifth route is adversarial
reading from outside the project: the independent review found three input paths that gave a false CERTIFIED, one
false FAILED, the wrong reading of the S = 1 crown and the privacy problem of the commit history, and the independent
re-derivation found two errors and five gaps in proofs that their own verification scripts had passed. The same
reading of text against code found INT-01 to INT-05. Each route found something that the others had not: mutation
analysis cannot resolve rounding at the level of half a unit in the last place, tests do not use inputs nobody thought
of, and a proof check does not run the code at scale.

## What the record does and does not show

According to the record, no defect found after an evidence run changed a certified statement of that run in substance.
Where the code paths behind a run were found unsound, as with the two soundness gaps of 21 October 2026 and the runs
of 18 October, the run was made again at the corrected source (decision D-46). Otherwise the defects were closed by a
new version with its own run, or recorded as known issues of the cited version with the reason no rerun was needed
(decisions D-57 and D-61). The record cannot show the absence of defects not yet found; what it shows is the rate at
which each route found them, and that the found ones were closed in the open.
