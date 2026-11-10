# Open items: known gaps that are not yet closed

This document lists what is known to be open at the state of archive v5 (6 November 2026): defects and gaps found but
not yet closed, and checks that only a person can make. Each entry says why it is open and what would close it.
Sources: `project_docs/SESSION_HANDOVER_2026-11-06.md` (section 2, "What remains"), `report/08_defects.md`,
`code/certify-ed/E6_inspection_worksheet_3.2.0.md`, `code/certify-ed/docs/proofs/VERDICTS.md`, and decisions D-57 to
D-61.

## The check of the derivation documents by Sarang Vehale

All 22 documents in `code/certify-ed/docs/proofs/` are derived, machine-checked and independently re-derived (a second
derivation written from the statement alone, not a human check), with Sarang's check pending (decisions D-12 and
D-59). They are open in this sense until he has re-derived them. The handover suggests an order: first the documents
that changed bounds (`thm2_polar-factor`, `thm1_fp-accounting`, `int05_underflow`, `appA_model-shift`), then the ERROR
and GAP rows of `VERDICTS.md` (`int05`, `E-04`, `thm1`, `C-04`, `E-07`, `E-05`, `D-02`), then the rest. Each document
ends with a status line, and a test accepts "done" with a date once the check is made. Two consequences hang on this
check. If the check of `docs/proofs/D-01_sparse-residual.md` finds a gap, the two E3 lower bounds that 3.2 tightened
revert to the a-priori bound and a 3.2.2 needs its own run (decision D-61). The five GAP rows need proof text that
only Sarang writes (decision D-59).

## The E6 survivors

No person has yet given verdicts on the mutation survivors that need inspection: 15 of the 26 survivors of the 3.1.0
run and 18 of the 28 survivors of the 3.2.1 run (finding RA-25). The worksheets in the code repository carry suggested
readings, not verdicts. The suggested readings for the 3.2 sample name seven soundness-relevant test gaps, places
where a mutant changes a bound or count that no test pins:

| Site | What the mutant changes |
|---|---|
| `sectors.py:57` | the real or complex flag of `audit_sector`: real rounding constants would be used for complex arithmetic |
| `eigenspace.py:151` | the rounding term of the label bound scaled by 1/g instead of g |
| `eigenspace.py:147` | κ computed with √(1 − η) instead of 1/√(1 − η) |
| `eigenspace.py:129` | the main term of the Theorem 6 bound on observables |
| `exact_sectors.py:202` | ν times the factor instead of ν divided by it in the real embedding of a complex block |
| `localops.py:189` | the amplitude vector of the fermion lowering letter |
| `spectrum.py:103` | a merged interval marked counted when only one side is |

None of them changes a statement of an evidence run, because the records were produced by the unmutated code. The
remedy proposed in the worksheet is a test that pins each formula, as `tests/test_backlog_proofs.py` does for the
cluster bound; the tests belong to a 3.2.2 together with the verdicts. One further survivor (`checker.py:535`) is a
test gap through which a false FAILED, never a false CERTIFIED, would be possible.

## The memory gate

The E4 crash of the 3.2.0 run showed that a multi-threaded allocation failure under a cap on committed memory can end
the process. 3.2.1 avoids it by computing the residual on one thread by default. What remains open is a memory gate
that uses the process's memory cap when no memory budget (`budget_mb`) is given, so that a statement whose estimated
memory exceeds the cap is NOT_CHECKED before it starts (handover of 6 November 2026, item 3).

## The cited version

The manuscript cites 3.1.0 and its run `run_full_20261022`. Whether 3.2.1 becomes the cited version is Sarang's
decision after reading `code/certify-ed/RESULT_evidence_run_3.2.1.md` (decisions D-60 and D-61). The manuscript macros
of the archive are already generated from the 3.2.1 run, with the 3.1.0 set kept beside them, so the choice is made by
swapping two directories.

## The known issues of 3.1.0

INT-01 and INT-04 were left open for 3.1.0 by decision D-57. INT-01 is enforced conservatively in 3.2
(`UNDERFLOW_RISK`), but the choice among rejecting, rescaling and stating the hypothesis as the user's is Sarang's.
INT-04 needs Appendix A of the manuscript rewritten to describe the accounting the code implements, which is prose
under decision D-16. INT-02 and INT-03 are fixed on `fix/3.1.1` and in 3.2, but not in the cited 3.1.0
(`02_integration_findings.md`).

## Items outside the code

Several open items are decisions or acts of the authors. The note to Kránitz and Penc (version 0.3) waits for Sarang's
check of three statements against the arXiv text before it is sent, and nothing about the S = 1 discrepancy is stated
publicly before their reply (decisions D-34 and D-51). The corrected replacement of the symveig preprint and the
deprecation of its public code (decision D-53) have not been carried out. The push of the rewritten history to GitHub
waits for a credential with write access and for Sarang's word (decisions D-56 and D-60), and the Zenodo version after
it. The question of whether the Zenodo DOI 20066565 or 20066566 is the concept DOI of v1, and a threat model for the
authenticity of certificates, are open from the audit of v1 (`01_defect_ledger.md`). The PDFs of manuscript v4 and
report v1.2 have not been built from the archive v5 sources (`04_failed_runs_and_attempts.md`). The manuscript prose
is Sarang's (decision D-16): v4 carries scaffold captions only, and every section except Appendix D still names 3.1.0.

## Items found while writing these documents

Writing these documents found a few inaccuracies in the record and limits of one instrument. They are listed in
`06_found_while_documenting.md`; none changes a statement or status of any run.
