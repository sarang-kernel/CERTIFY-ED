# Findings made while writing these documents

Writing the experiment documents meant reading every comparison file and record that the result files summarise. That
reading found three statements of the result files that the records do not support as written, two limits of the
comparison script, and one trace of the 3.2.0 crash in a second experiment. None changes a certified statement or a
status of any run. They are listed here so that a reader of the result files can correct them, and each is shown with
the file and the field that settles it. The code repository was not edited.

## The E6 leaf `copy.files` of the comparison of 3.2.1 with 3.1.0

`code/certify-ed/RESULT_evidence_run_3.2.1.md` explains the MISMATCH leaf `copy.files` (215 in 3.1.0, 915 in 3.2.1) as
the mutation driver copying every tracked file instead of five folders. The records do not support that reading: the
E6 records of both runs give `copy.source` as `git ls-files`
(`results/run_full_20261022/e6_mutation_20261022_205005/data.json` and
`results/run_full_20261105/e6_mutation_20261105_154612/data.json`), and the copy was changed to every tracked file for
3.1.0 already, after its first attempt. The number of files that git tracks is 215 at `v3.1.0` and 915 at `v3.2.0` and
`v3.2.1` (counted with `git ls-tree -r --name-only` on each tag). The leaf differs because the repository grew between
the versions, which is still a documented and harmless difference. `docs/experiments/03_comparisons.md` gives the
corrected explanation.

## The missing leaves of the comparison of 3.2.1 with 3.2.0

The same result file attributes the 64 missing leaves of the comparison with the 3.2.0 run to "the 64 leaves E4 lacks
in the 3.2.0 record". The comparison file `results/run_full_20261105/compare_vs_run_full_20261104.csv` shows 48 of
them in E2 and 16 in E4. The 48 E2 leaves are the certifier fields of the momentum blocks N = 22 and N = 24 (rows 7
and 8), where the 3.2.0 certifier reached the 900 s limit and the 3.2.1 certifier finished with S3 NOT_CHECKED; the 16
E4 leaves are the fourth W4 case and the elapsed time, which the crashed 3.2.0 record lacks.

## A status that differs between the two stacks of 3.1.0

`code/certify-ed/RESULT_evidence_run_3.1.0.md` concludes that every status agrees between the Linux 3.0.0 run and the
Windows 3.1.0 run. One status leaf differs, and the comparison classes it as explained rather than as a MISMATCH:
`W4[3].statuses.S3.completeness` of E4 is NOT_CHECKED on Linux and CERTIFIED on Windows, because the factorization of
the J1-J2 square at J2 = 0.5 fits under a job-object cap and not under an address-space cap of the same size
(`results/run_full_20261022/compare_vs_run_full_20261021.csv`). The result file lists this explanation among the
explained leaves, so the information is in it; the summary sentence needs the exception.

## Two limits of `compare_runs.py`

First, the classes `stack-dependent` and `float-differs` divide the difference of two values by max(1, |A|). For a
bound smaller than 1 the comparison is therefore in effect absolute: the tolerance 0.10 of `stack-dependent` accepts
any two radii below 0.1, so the W4 radii 4.905e-10 (3.1.0) and 3.0335e-10 (3.2.1) of E4 are classed as
stack-dependent. The status a bound supports is compared strictly, so no status can change unnoticed, but the class
does not show that small bounds reproduce to some relative accuracy. Second, a rule of the table `EXPLAINED` matches
by path, and the attached sentence describes the case it was written for. The E2 rule for S3 summaries was written for
the wording of NOT_CHECKED reasons on two platforms; in the comparison of 3.2.1 with 3.1.0 it also matched six
CERTIFIED summaries. Those six were read one by one: the certified counts are equal, the printed gaps are equal in
five and move in the fourth digit in one (8.285e-02 against 8.286e-02), and the printed shifts differ from the sixth
significant digit on in two rows and from the seventh or later in four (`docs/experiments/03_comparisons.md`).

## The 3.2.0 crash in E2

In the 3.2.0 run, the E2 point "square N=16 u1 J2=0.5" (dimension 12870), the same matrix as the crashed W4 case of
E4, has a certifier child that ended with an empty error text (`certifier failed: ` in
`results/run_full_20261104/E2.log`; `rows[10].certifier.error` is the empty string in the record). E2 runs its
certifier in a child process, so the experiment continued and its step is ok. The record does not say why the child
ended; that it was the same multi-threaded allocation failure is an inference from the identical matrix, cap and code
path. `RESULT_evidence_run_3.2.1.md` and decision D-61 describe the crash in E4 only. In the 3.2.1 run the same point
reached the 900 s limit instead.
