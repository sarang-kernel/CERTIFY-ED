# E5: an error that nobody planted

## Why this experiment

The planted errors of E1 invite an objection: errors chosen by the authors of a method may be the ones the method
catches. While measuring cost in E2, ARPACK, asked through QuSpin for the eight lowest states of the 4x4 Heisenberg
S^z = 0 block, sometimes returned only three of the four members of a fourfold level, followed by two members of the
next level. Each returned pair is an accurate eigenpair; the list is not the eight lowest eigenvalues. This is an
error that nobody designed, and decision D-27 made it an experiment and a subsection of the paper, reported at
whatever frequency the seeded runs give. Two alternatives were rejected under that decision: re-seeding E2 until the
run came out clean, which would have been a selection of favourable runs, and dropping the 4x4 point. The experiment
also tests whether disagreement between two solver runs can say which of them is wrong.

## Design

The script is `experiments/e5_arpack_multiplet.py`. For 12 seeded start vectors, ARPACK is asked for the eight lowest
states of the 4x4 Heisenberg S^z = 0 block (dimension 12,870, default `ncv`, `tol=0`). The truth is the eight lowest
eigenvalues of a dense LAPACK solution of the same block. Four checks are applied to each run. P1 is the residual
check of the eight returned pairs (largest relative residual at most 1e-8). P2 is a second ARPACK run from another
random start, accepted when its eight lowest values agree with the first to 1e-8. P3 is certify-ed on the claim "these
are the eight lowest eigenvalues". P3k is certify-ed on the claim a careful user would make: the lowest k states, with
k chosen at the first gap larger than 1e-6 in the returned values at or after the fifth, as E2 does.

## How the instrument is checked

The dense solution gives the truth for every run, so each verdict can be scored. The correct runs are the controls: a
check that flags them gives a false alarm. The seeds are fixed (decision D-26), so the frequency of truncation is
reproducible on a given stack; on another stack the seeds that truncate can differ, and the two-stack comparison
checks invariants instead of rows (no incorrect result certified).

## Results

| Quantity | 3.0.0 `run_full_20261021` | 3.1.0 `run_full_20261022` | 3.2.1 `run_full_20261105` |
|---|---|---|---|
| ARPACK runs | 12 | 12 | 12 |
| runs that returned an incomplete level | 3 | 3 | 3 |
| multiplicity of the level | fourfold | fourfold | fourfold |
| energy of the level | -8.88644241 | -8.88644241 | -8.88644241 |
| members returned in an incomplete run | 3 | 3 | 3 |
| energy of the next level | -8.79437891 | -8.79437891 | -8.79437891 |
| incomplete runs flagged by P1 | none | none | none |
| incomplete runs flagged by P2 | 3 | 2 | 2 |
| correct runs flagged by P2 (false alarms) | 2 | 3 | 3 |
| correct runs | 9 | 9 | 9 |
| incomplete runs with S3 FAILED under P3 | 3 | 3 | 3 |
| S3 of the correct runs under P3 | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| factorizations until S3 FAILED | 4 | 4 | 4 |
| k chosen by P3k in a correct run | 7 | 7 | 7 |
| k chosen by P3k in an incomplete run | 6 | 6 | 6 |

The values are the macros of `code/certify-ed/results/<run>/paper/numbers.tex` of each run. In every run, 3 of the 12
ARPACK runs returned the fourfold level at -8.88644241 with only 3 of its 4 members. P1 flags none of them, because
every returned pair is accurate. P2 flags some of the incomplete runs (3 in 3.0.0, 2 in 3.1.0 and 3.2.1) but also some
correct ones (2 in 3.0.0, 3 in 3.1.0 and 3.2.1): when two runs disagree, the disagreement does not say which is wrong.
P3 gives S3 FAILED for all 3 incomplete runs, after 4 factorizations, and INCONCLUSIVE for the 9 correct runs, because
the eighth value belongs to a degenerate level that continues beyond it, and a claim that cuts a multiplet is
INCONCLUSIVE, not FAILED (decision D-46). For the claim up to the first gap (P3k), S3 certifies the correct runs (k
= 7) and diagnoses the cut multiplet in the incomplete ones. The invariants of the two-stack comparison hold on both
stacks: 12 runs, 3 incorrect results, 0 incorrect results certified, S3 of the full claim FAILED 3 times and
INCONCLUSIVE 9 times, S3 of the k lowest CERTIFIED 9 times and INCONCLUSIVE 3 times
(`code/certify-ed/RESULT_evidence_run_3.1.0.md`).

## The history of this experiment

The first E5 run, on 4 October 2026, reported the truncated lists as INCONCLUSIVE instead of FAILED. On the 4x4 block
the backward error of the factorization (about 5e-7) exceeded the margin (1.1e-7) near the fourfold level, so the
probe below the top claimed value could not decide. Decision D-28 made the probe retry at 4 times (β plus the model
shift) below the top enclosure, commit `e432742`, and the first run was superseded; all experiments were rerun at that
commit. Decision D-27 had already changed S3 to check the count below the top claimed value at the first probe that
over-counts, so that it fails after two factorizations instead of up to six. A later rerun of E5 was interrupted at
the end of a working period on 4 October 2026, and its partial record made the number script fail; decision D-29 moved
such records to a folder the number scripts never read and reran E5 at `536b814`, which reproduced the outcomes seed
by seed (`docs/failures/04_failed_runs_and_attempts.md`).

## What the result shows and does not show

E5 shows, on one block and twelve seeds, an error of a common solver that the producer's own residual check cannot
see, that recomputation cannot attribute, and that the completeness statement catches. It does not estimate how often
ARPACK truncates multiplets in general; the frequency depends on the matrix, the solver settings and the stack, and
the experiment reports the frequency its seeds give, as decision D-27 requires.
