# Open items: known gaps that are not yet closed

This document lists what is known to be open at the state of archive v5 (6 November 2026): defects and gaps found but
not yet closed, and checks that only a person can make. Each entry says why it is open and what would close it.
Sources: `project_docs/SESSION_HANDOVER_2026-11-06.md` (section 2, "What remains"), `report/08_defects.md`,
`code/certify-ed/E6_inspection_worksheet_3.2.0.md`, `code/certify-ed/docs/proofs/VERDICTS.md`, and decisions D-57 to
D-61.

## The check of the derivation documents by Sarang Vehale

All 22 documents in `code/certify-ed/docs/proofs/` are derived, machine-checked and independently re-derived (a second
derivation written before the original proof was read, not a human check), with Sarang's check pending (decisions D-12 and
D-59). They are open in this sense until he has re-derived them. The handover suggests an order: first the documents
that changed bounds (`thm2_polar-factor`, `thm1_fp-accounting`, `int05_underflow`, `appA_model-shift`), then the ERROR
and GAP rows of `VERDICTS.md` (`int05`, `E-04`, `thm1`, `C-04`, `E-07`, `E-05`, `D-02`), then the rest. Each document
ends with a status line, and a test accepts "done" with a date once the check is made. Two consequences hang on this
check. If the check of `docs/proofs/D-01_sparse-residual.md` finds a gap, the two E3 lower bounds that 3.2 tightened
revert to the a-priori bound and a 3.2.2 needs its own run (decision D-61). The five GAP rows need proof text that
only Sarang writes (decision D-59).

Added in 3.3.0.dev0 and open in the same sense: `docs/proofs/C-07b_spectral_bounds.md` (statements M5 and M6, the
Chebyshev moments and the bounds on the spectral distribution; machine check `docs/proofs/verify_C-07b.py`). It has no
independent re-derivation yet. The points to check first are Lemma 7 (the error of the shifted and scaled product
fl(s(fl(Hz) − fl(cz))), the constant γ̇_{w+2}), Lemma 5 (b) (the lowest eigenvalue read from a certificate whose S2 and
S3 are CERTIFIED) and Lemma 8 (c) (de Casteljau with rounding down).

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
test gap through which a false FAILED, never a false CERTIFIED, would be possible. The 3.2.3 run drew a different sample
of 64, because the number of mutation sites grew: 37 of its mutants were killed, and `make_numbers_apps.py` counts 19 of
the 27 survivors as awaiting a verdict.

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
it; since 18 November 2026 the push of the joined repository also waits for the reply of Kránitz and Penc, because
its history states the S = 1 discrepancy (D-34). The question of whether the Zenodo DOI 20066565 or 20066566 is the concept DOI of v1, and a threat model for the
authenticity of certificates, are open from the audit of v1 (`01_defect_ledger.md`). The PDFs of manuscript v4 and
report v1.2 have not been built from the archive v5 sources (`04_failed_runs_and_attempts.md`). The manuscript prose
is Sarang's (decision D-16): v4 carries scaffold captions only, and every section except Appendix D still names 3.1.0.

## Items found while writing these documents

Writing these documents found a few inaccuracies in the record and limits of one instrument. They are listed in
`06_found_while_documenting.md`; none changes a statement or status of any run.

## Items found by the Lean formalization

The Lean re-verification found that the code does not check the size range under which the floating-point guards of
Theorem 1 hold: `_rounding_bound` in `certify_ed/enclosure.py` and the norm guards of `certify_ed/fp.py` accept any row
width w and dimension n, while Lambda6 and Lambda4 are established for w and n up to 2^26 only (finding L-3 of
[07_found_while_formalizing.md](07_found_while_formalizing.md)). No stored operator comes near that size, so no
certificate is affected; a check that refuses to certify above the range, with a test, is a candidate for 3.2.2 together
with the E6 pinning tests and the memory gate. The other findings of that document are corrected by documentation
commits or need no change. The statements that the formalization left partial or did not take up are listed in
`docs/lean/VERIFICATION_REPORT.md`; whether a Lean theorem can stand in for Sarang's re-derivation is the
subject of the proposal `docs/project/decisions/05_proposed_d62_lean.md`, which is not adopted.

## After the first development phase (11 to 13 November 2026)

The first phase of [the development roadmap](../roadmap/DEVELOPMENT_ROADMAP.md) closed three items of this document in
code and opened others. Sources: the sections [3.2.2] and [3.3.0.dev0] of `CHANGELOG.md` on the code branches
`fix/3.2.2` and `dev/3.3` (`docs/project/sources/SOURCES.md`), and `docs/lean/NONHERMITIAN.md`, `LEAN_PRODUCT.md`,
`APPS_NEXT.md` and `VERIFICATION_REPORT.md`.

Version 3.2.2 (tag `v3.2.2`) adds the range check of finding L-3, the pinning tests of the E6 survivors and the memory
gate. Outside w and n at most 2^26 the floating-point guards form no bound, and S2 with the statements that rest on it
is INCONCLUSIVE with the reason code `FP_GUARD_RANGE`. `tests/test_e6_pins.py` pins the seven survivors of the table
above and the `checker.py:535` gap, and each of the eight mutants fails it. Without `budget_mb`, the memory left under
the process memory cap now gates statements and factorizations (NOT_CHECKED with `BUDGET_MEMORY`). The pin tests do not
replace the verdicts on the survivors, which remain Sarang's, and the worksheet's reading of `localops.py:189` does not
hold: that mutant changes no built matrix, so its test pins the contract of `shift_components` instead. The evidence run
of 3.2.2 (`run_full_20261112`) did not complete: the memory gate refused S1 in two cases that the 3.2.1 run had
certified (for the chain N=22 at k = 0 it estimated 16896 MB, where the 3.2.1 certifier peaked at 2989 MB), and the E6
baseline failed because git could not create a repository under the temporary directory of the restricted environment
(`docs/results/RESULT_evidence_run_3.2.2.md`). Version 3.2.3 (tag `v3.2.3`) exempts S1 from the gate derived from the
cap (an explicit `budget_mb` still gates it) and makes E6 copy next to the repository. Its evidence run
`run_full_20261112b` completed on 13 November 2026 at source `bfc115e824386443`, after a change of the number script
alone (`5dedf9d`). Against the 3.2.1 run no record reached the guard range and no statement changed status, except one
E2 point whose 3.2.1 certifier had reached the time limit and which now ends with S1 and S2 CERTIFIED and S3, S4 and S8
NOT_CHECKED by the gate; the other differences are the summary texts of statements NOT_CHECKED in both runs and the size
of the E6 copy (`docs/results/RESULT_evidence_run_3.2.3.md`). Which version the manuscript cites (3.1.0, 3.2.1 or 3.2.3)
stays Sarang's decision.

Two new derivation documents are on `dev/3.3`. `docs/proofs/E-02_non_hermitian.md` (non-Hermitian eigenpairs:
backward errors, a bound on the inverse of the eigenvector matrix, Bauer-Fike discs and a count per group of discs) is
derived, machine-checked (`docs/proofs/runs/verify_E-02.txt`: 2696 random and 547 adversarial cases, 0 violations) and
independently re-derived (verdict AGREES-WITH-REMARKS in `docs/proofs/VERDICTS.md`), with Sarang's check pending; the
claim kind stays UNVERIFIED until then. `docs/proofs/D-03_ozaki_draft.md` (error-free residual and Gram products) is a
draft without an independent re-derivation, and no default path uses it.

The applications left open questions for Sarang. At the nine couplings that Kránitz and Penc print, the floating-point
layer gives -0.5498329145683885 <= lambda_min <= -0.5498329144473251 for the crown. The exact re-check, stopped after 3
of 8 blocks in its first run, was completed on 22 November 2026 with one process per block
(`lean/results/apps_next/n11_crown_nine_exact_8_blocks.json`): every count is 0, and lambda_min >= -0.5498329145062226
without a floating-point assumption, resting on the lemmas of the exact layer, which are marked UNVERIFIED. The printed
-0.549832(8), with its digit in parentheses appended, lies above that interval; whether this concerns the note to
Kránitz and Penc (decisions D-34 and D-51) depends on how the digit is meant. The honeycomb criterion of Pomata and Wei
needs a block of 1293292 rows and was not run, and the manuscript sentence that names only the companion paper for the
honeycomb criterion is incomplete (prose under decision D-16).

The Lean work left four mechanical gaps (Theorem U5 and D-01 part (c) for complex data, C-12 part (a), E-07 Lemma 5),
the count theorems of the three kernel certificates and the refinement of `CertifyCore.K.bareiss`
(`LEAN_PRODUCT.md`). It also found that the kernel checks `checkLowest` and `pairOK` of
`CertifyCore/Kernel/Defs.lean` accept an asymmetric matrix and a zero vector, which `cecheck` rejects; no stored
certificate is affected, and adding the two tests to the kernel checks is open. nanoda could not finish
`npz_S2` and `npz_theorem1` under the limit of 4096 MiB.

On the side of scale, the worker pool of D-04 was measured on 22 November 2026 on 16 pinned cores of a shared Linux
host (`docs/results/RESULT_d04_speedup.md`, single runs): on eleven sector claims of the chain of 16 sites it gives a
speed-up of 1.77 with 2 workers and 2.75 with 8, bounded by the longest claim, and the statement records and arrays
equal the one-worker ones. The
thresholds of the performance tests (`pytest --perf`) were not tried on an idle machine; the baseline is to be recorded
again on an idle machine, since the 3.2.2 and 3.2.3 runs shared the machine with other work. One comparison at a
relative tolerance of 1e-12 in `tests/test_whole_spectrum.py` failed once on the loaded machine and passed when run
again, which marks it as a test that can fail by chance.

## Non-finite claimed values in certificates

Since `fix/small-followups` (November 2026) a claimed value with a non-finite real or imaginary part is a non-finite
value at every entry point: S2 and the exact re-checks are INCONCLUSIVE ("claim contains non-finite values"). Three
things remain open. For a claim with eigenvectors, S3, S4 and S8 give `NEEDS_EIGENVECTORS` with the reason "the claim
has eigenvalues only", which is not true of such a claim; closing it needs a reason code (`scope.REASON_CODES`,
`docs/SCOPE_MATRIX.md`). A certificate whose claim holds a non-finite value does not load under the default schema
validation, because the items of `claim.eigenvalues` are plain numbers in schema 2.1; and a certificate with an
enclosure of method `existence-only` (claimed vectors scaled by about 1e-170 or 1e155) does not either, because the
method enumeration lacks that value. Both were so before the branch, and both are schema decisions
(`docs/SCHEMA_RFC_PROCESS.md`). Until they are made, such certificates are read with
`Certificate.load(path, validate_schema=False)`.

The record of dropped finite imaginary parts (`dropped_imaginary_parts`) is made by `certify`, its two wrappers and the
sector composition and, since the changelog section "Dropped imaginary parts: records at the remaining entry points", by
`certify_generalized` (in the claim of its record), `exact_recheck_lowest`, `exact_recheck_sector_lowest`,
`exact_eigenpair_radii` and `corroborate` (in their result) and the ALPS reader (`read_alps_spectrum`,
`claim_from_alps`). The functions that return numbers only (`enclose`, `residuals`, the Ozaki residuals,
`arb_residual_radii`, `spectrum_invariants`) make no record; a direct call with complex values gets numpy's
ComplexWarning, as in 3.x. `write_bundle` (a bundle holds the real parts only) still drops such parts without a record;
a bundle manifest field would close that, which is a schema decision (`docs/SCHEMA_RFC_PROCESS.md`).
`tests/test_dropped_parts_records.py` keeps this list complete.

## Packaging

The `test` and `dev` extras of `pyproject.toml` do not list mpmath, although four test files use it for
high-precision references (`tests/test_entanglement.py`, `test_nonhermitian.py`, `test_proofs_props.py`,
`test_thermo.py`) and skip those tests without it, so an install with `pip install .[test]` runs fewer tests. The
container recipes install it. Adding `"mpmath>=1.2"` to both extras is open.
