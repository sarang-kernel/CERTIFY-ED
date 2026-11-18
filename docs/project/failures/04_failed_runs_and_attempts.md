# Failed runs and failed attempts

This document collects every run, build or attempt that failed, in time order, with what failed, why it failed, what
was done, and what was kept as a record. A failed attempt is part of the evidence: it shows which failure modes the
project met and how each was closed. Defects of the code found by other means are in `01_defect_ledger.md`; the
integration findings are in `02_integration_findings.md`. Sources are named in each section.

## October 2026: reproducibility of iterative solvers

Two E2 runs at consecutive commits gave different numbers of eigenvalues for the 4x4 block. The cause was that SciPy
from version 1.15 seeds ARPACK from operating-system entropy when no start vector is given, so the iterative solver
calls of the certifier (its gap hint), of the experiments' reference and of the experiment producers were not
reproducible. No certified quantity was affected, because the gap hint never enters one, but shifts, gaps and run
times were. Every iterative call now receives a seeded start vector (decision D-26, commit `6143b5d`).

## 4 October 2026: the first E5 run and its interrupted rerun

The first E5 run reported the truncated ARPACK lists as INCONCLUSIVE instead of FAILED. The cause was that near the
fourfold level of the 4x4 block the backward error of the factorization (about 5e-7) exceeded the margin of the probe
below the top claimed value (1.1e-7). The probe was given the same remedy as the probe above the claim, a retry at 4
times (β plus the model shift) below the top enclosure (decision D-28, commit `e432742`), and every experiment was
rerun at that commit; the first run was superseded and the episode is reported in the paper. The E5 rerun at the final
tree of 4 October was then interrupted at the end of the working day. Its partial record made `make_numbers.py` fail,
and the PDF of that date was built with the E5 table of the superseded first run, which contradicted the text.
Decision D-29 moved records of interrupted runs to `experiments/results/superseded/incomplete/`, which the number
scripts never read, made `make_numbers.py` report the run date of every experiment when records come from different
days, and reran E5 on 11 October at `536b814`, which reproduced the outcomes of `e432742` seed by seed. Reusing the
`e432742` record was rejected, although it would have given the same numbers, because the record set would no longer
come from one tree.

## 11 October 2026: an empty figure panel

The fill-in panel of Figure 3 was empty in the PDF of 4 October, because the logarithmic axis was set before its
limits and collapsed. The figure had passed the independent review and the 36 sentence checks, which read data, not
pixels. The fix is `5c3f8b8`, and decision D-31 added a visual check of every rendered figure to the checklist.

## 17 October 2026: the exact AKLT check out of memory

The exact check of the decorated-square lattice was killed for lack of memory at 5.9 GB. The cause was that the
residual check formed the full product matrix. A residual check by row blocks replaced it, and the bordered pencil
with exact symmetry blocks cut the largest exactly checked block from 8842 to 4034 (decision D-38).

## 18 October 2026: the first full run of 3.0.0

The number script of `run_full_20261018` failed two checks. "All experiments ran with the same certify_ed source tree"
failed because the run crossed a commit (`764a9e8` to `bcc6f12`) that changed only the command line: the check
compared commits, not the code that can change a number. "E2: the N=22 momentum block has the largest S1 time" failed
because the check compared runs instead of blocks, and failed when the same block's run with S3 took longer than its
run without. The first led to the computational source fingerprint (decision D-44, `docs/project/software/05_provenance.md`),
the second was corrected in the check. At the same time the mutation workspaces were found to lack `examples/`, which
the suite runs, so the E6 baseline would have failed; they were made to include it.

## 22 October 2026: the candidate run's three source hashes

The candidate run `run_candidate_20261021` (E7 to E10 on Windows) recorded three source hashes and one dirty step,
although the computational files did not change between the steps. `code/certify-ed/RESULT_candidate_run.md` gives the
cause, reproduced after the run: the list `NON_COMPUTATIONAL` was extended while the run was in flight, the running
process computed each step's hash with the list it had imported at its start, and a post-processing script created in
`experiments/` during the last step made that step's tree dirty. The hashes were computed again for this documentation
on 10 November 2026 from the three step commits (`04d95e9` for E7, `0882699` for E9 and E10, `edb92b2` for E8) with
the functions of `tools/codeindex/source_hash.py`. With the `NON_COMPUTATIONAL` list of `04d95e9` (7 entries) the
commits give the recorded hashes `e707838f6852d474`, `819671ccacd816f4` and `209234087abaea2a`; with the list of
`edb92b2` (11 entries) or of `v3.1.0` (12 entries) all three give `e707838f6852d474`, so the computational files are
the same at the three commits. The applications reproduced without a MISMATCH leaf, but the run could not serve as an
evidence run. The rule drawn from it is that an evidence run is made from one commit with no edits in flight.

## 22 October 2026: attempt 1 of the 3.1.0 evidence run

The first attempt (`run_full_20261022_attempt1`, commit `ada1362`, 10:28 to 13:56) failed four steps, all through
portability defects of the 3.1.0 code on Windows, none through a certified statement
(`code/certify-ed/results/run_full_20261022_attempt1/ATTEMPT_NOTE.md`). The tests step had 1 failure: in
`test_io_formats.py::test_matrix_free_operator_requires_allow_code`, SciPy 1.17.1's `LinearOperator.matmat` called the
producer's matrix-vector function with an (n, 1) column, which a function written for vectors (`_d * x`) broadcast to
(n, n). The importer now always calls the producer's function with a vector of shape (n,), checks the length of its
output (`FormatError` otherwise) and assembles `matmat` column by column. E4 failed at `import resource`, a POSIX-only
module used for the 5 GiB memory cap. E2 failed at `subprocess.run(..., preexec_fn=...)`, also POSIX-only, used for
the 4.5 GiB cap of each child, and at `resource.getrusage` for the peak memory; E2 and E4 now use
`experiments/portable.py`, which keeps the address-space limit on POSIX and uses a Windows job object with a
per-process memory limit and the peak working set on Windows, and records the mechanism. E6 failed because the
mutation copy held only `certify_ed/`, `tests/` and `examples/`, while six 3.1.0 tests read `pyproject.toml` or
`experiments/`, so the baseline failed on any platform, and 25 tests were skipped in the copy against 1 in the
repository; the copy is now every tracked file. The number scripts then failed or were skipped for lack of records.
The fixes went into `e860a83`, the tag `v3.1.0` was moved there before it was published, and the attempt is kept as a
record; its successful steps are valid records of `ada1362` but not of the cited commit (decision D-44).

## 3 to 4 November 2026: the 3.2.0 release sample

The first attempt at the 16-mutant release sample of 3.2.0, at `b832d8e`, stopped because the unmutated baseline
failed: `tests/test_whole_spectrum.py::test_enclosure_radii_bit_identical_complex_operator` compared complex-data
radii bit for bit with a frozen reference, and the looser constant γ_{w+3} for complex data, introduced after the
independent re-derivation (`03_review_and_derivation_findings.md`), had changed those radii. The comparison was
changed to "at least, and within a factor 1 + 1e-12" for complex data, with real data still bit-identical, in
`e1ebaed`. The suite at `e1ebaed` then gave 1168 passed and 8 skipped, and the release sample killed 10 of 16 mutants
(`code/certify-ed/RESULT_integration_3.2.md`).

## 4 to 5 November 2026: the E4 crash of the 3.2.0 evidence run

The 3.2.0 evidence run `run_full_20261104` completed every step except E4, whose W4 case "J1-J2 square 4x4, J2 = 0.5"
(dimension 12870) ended the process with return code 3221225477, a native access violation, under the 5 GiB job-object
cap. The diagnosis in a separate clone found the cause in the new a-posteriori S3 residual
(`ldl_residual.sparse_ldl_residual_bound`), which formed its row blocks on eight threads: under a cap on committed
memory, an allocation failure inside SciPy's sparse product in a worker thread ends the process instead of raising
`MemoryError` in the calling thread, so the certifier's handling of memory failures never runs. With one thread the
case certifies S3 under the cap (gap at least 0.287); without the cap both settings certify. Version 3.2.1 makes one
thread the default, with `CERTIFY_ED_RESIDUAL_THREADS` to opt in, and pins with a test that the bound does not depend
on the thread count. The 3.2.1 run repeated every step (decision D-61), and its comparison with the 3.2.0 run has no
MISMATCH and no float difference. The same 3.2.0 run's E2 record shows, for the same matrix, a certifier child that
ended with an empty error text; this is consistent with the same crash, but the record does not state the cause
(`docs/project/experiments/02_evidence_runs.md`). Decision D-61 rejected keeping eight threads by default, because the crash
depends on the kind of memory cap and gives no message. A memory gate that uses the process cap when no memory budget
is given remains open (`05_open_items.md`).

## 5 November 2026: the E2 check of the number script

After the 3.2.1 run, `make_numbers.py` treated the two E2 points whose certifier reached the experiment's 900 s limit,
and left a certificate without S3, as crashed certifier runs, and failed. The check was wrong, not the run: a
certifier that reaches the limit is an expected outcome of E2, recorded as such, and in 3.1.0 both points had S3
NOT_CHECKED. Commit `44d318c` changed the check (a non-computational file, so the run's source hash is unchanged) and
added the macro `CnumSthreeTimedOut`; the numbers were regenerated from the same records, 44 of 44 and 61 of 61
checks.

## 6 November 2026: the labels of the 3.2.1 figure index

`docs/figures_run_3.2.1/INDEX.md` of the code repository named the 3.2.1 record directories in its data-source lines
but labelled the run as the 3.1.0 evidence run. The cause was a label written into `make_index.py` rather than read
from the records. Commit `004a40f` makes the data-source lines take the run labels from the records
(`report/08_defects.md`).

## Toolchains

Two toolchain attempts failed. For the independent verifier, a Rust toolchain (conda-forge `rust` 1.99.1, target
`x86_64-pc-windows-msvc`) was installed but could not link: the workstation has no MSVC linker, self-contained linking
is not supported on that target, and the GNU-target standard library was not available. The verifier was written in C
instead and built with the MinGW gcc 14.2 shipped with GNU Octave (`code/certify-ed/verifier/README.md`). For the
documents, the PDFs of manuscript v4 and report v1.2 were not built: TeX Live's installer could not be run in the
build environment and tectonic 0.17.0 failed on the Windows known-folder interface (`report/09_manuscripts.md`,
`project_docs/SESSION_HANDOVER_2026-11-06.md`). Both source trees pass their own checks (`check_macros.py` reports no
undefined macro, `check_duplicates.py` no duplicate), and the last built manuscript PDF is the 3.1.0 version of
archive v3 and v4.

## The test environment on the Windows workstation

One test fails under a particular temporary directory, not because of the code.
`tests/test_driver.py::test_reproduce_parallel_resume_and_numbers` spawns two experiments, which failed within 3 s
when pytest used the default temporary directory under `AppData`; with `--basetemp` in a writable working folder the
test passes, and `certify-ed reproduce --only E1,E3` succeeds on its own
(`review/completion/RESULT_independent_check.md`). The reproduction commands in `docs/project/reproduction/` therefore pass
`--basetemp`.
