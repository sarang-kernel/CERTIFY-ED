# RESULT_d04_speedup.md: the worker pool of D-04 on 16 pinned cores of a shared host

Roadmap item D-04 (`certify_ed/parallel.py`) runs independent certifications in worker processes, each with a share of
the process memory cap. It had shown no speed-up: the measurements before 22 November 2026 were made on a machine
without free cores (`docs/project/failures/05_open_items.md`). This document records the first measurement on cores
to which the run was pinned, on a host that other work shared, and a second run of 27 November.

## What was run

`experiments/d04_speedup.py --N 16 --nups 3,4,5,6,7,8,9,10,11,12,13 --workers 1,2,4,8 --cap-gib 32`, at commit 57607a70
(source hash 337392e110cebd3d). The jobs are the eleven U(1) sectors nup = 3 to 13 of the periodic Heisenberg chain of
16 sites (dimensions 560 to 12870). Each sector claims its lowest eigenpairs from scipy's eigsh, at most 4 and cut at a
gap (3 in every sector), and `certify()` runs with the default Options. For each worker count `parallel.certify_many`
certifies the eleven claims, and the record compares, for every claim, the statement records and the arrays of the
certificate with those of the one-worker run, after dropping every key whose name contains "second", "time", "elapsed"
or "parallel" (the comparison of `tests/test_parallel.py`; floats are compared by their bits). The host was Linux
(Python 3.12.3, numpy 2.5.4, scipy 1.18.1), with one pin of 16 cores, nice 10, BLAS and OpenMP threads 1, and a memory
cap of 32 GiB (RLIMIT_AS). The host is shared: its load average was about 113 on 208 cores during the run, so other
work may also have run on the 16 pinned cores.

The experiment file of this record is the one of commit 57607a70. Since then (rounds 7 and 8 of the review) the
defaults of the script are this configuration, and the script records more: the CPU seconds of each run (this process
and its finished worker processes), the id and reason code next to each status, and the comparison covers the whole
certificate body and the arrays, with only the fields that record time left out (`timing`, `environment/created_utc`
and the keys `seconds` below `statements`, matched by exact path and listed in the record as `fields_not_compared`).
None of these changes alters what the command above measures. The record of 22 November has none of the additions; its
equality column covers the statement records and arrays only.

## What happened

| workers | wall (s) | speed-up | sum of task times (s) | longest task (s) | statements and arrays equal to one worker |
|---|---|---|---|---|---|
| 1 | 324.8 | 1.00 | 324.8 | 111.3 | (reference) |
| 2 | 183.0 | 1.77 | 334.0 | 112.5 | yes |
| 4 | 125.9 | 2.58 | 354.4 | 118.6 | yes |
| 8 | 118.0 | 2.75 | 349.5 | 117.4 | yes |

All eleven tasks ran in workers at every worker count above one. None was deferred because its predicted peak exceeded
the per-worker budget (rule 3), and none was redone in this process (rule 4). Each configuration ran once. The
one-worker times and statuses per sector (statements S1, S2, S3, S4, S8 in this order):

| nup | dimension | claimed pairs | seconds | statuses |
|---|---|---|---|---|
| 3 | 560 | 3 | 0.2 | all CERTIFIED |
| 4 | 1820 | 3 | 3.1 | all CERTIFIED |
| 5 | 4368 | 3 | 6.7 | all CERTIFIED |
| 6 | 8008 | 3 | 26.2 | all CERTIFIED |
| 7 | 11440 | 3 | 104.1 | all CERTIFIED |
| 8 | 12870 | 3 | 36.9 | S1, S2, S3 CERTIFIED; S4, S8 NOT_CHECKED |
| 9 | 11440 | 3 | 111.3 | all CERTIFIED |
| 10 | 8008 | 3 | 27.4 | all CERTIFIED |
| 11 | 4368 | 3 | 5.4 | all CERTIFIED |
| 12 | 1820 | 3 | 3.3 | all CERTIFIED |
| 13 | 560 | 3 | 0.2 | all CERTIFIED |

In the sector nup = 8 the three claimed pairs contain no degenerate level, so S4 and S8 have nothing to certify and are
NOT_CHECKED with the reason `NO_CLUSTERS`. In the sectors nup = 7 and 9, whose dimension is close to that of nup = 8, the
claim contains a degenerate level, and S8 counts its degeneracy with two further sparse factorizations. That is why
these two sectors take about three times as long as the largest one. A profile of the same certifications on the
Windows workstation, which was not kept as a record and is an observation only, gave 250 s for nup = 7, of which 82 s
were S3 and 166 s S8, and 92 s for nup = 8.

## What it shows and does not show

The pool shortens the run by a factor of 1.77 with 2 workers, 2.58 with 4 and 2.75 with 8. From 4 workers on, the wall
time is set by the longest single certification: the sectors nup = 7 and 9 took 104 s and 111 s of the 325 s serial
total, so no schedule of whole claims can finish in less than about 111 s. That bounds the speed-up of this set at 2.92,
and 8 workers reach 94 % of the bound.

The tasks are handed to the workers in input order (nup = 3 to 13), so the two long tasks start late. A list schedule of
the one-worker task times in that order ends after 177.5 s with 2 workers and 118.0 s with 4, against 164.9 s and 111.3 s
when the longest task is handed out first; the measured 183.0 s and 125.9 s are close to the first pair. Part of the
difference to the bound at 2 and 4 workers is therefore the order of the list, not the pool.

The sum of the task times exceeds the serial run by 3 % with 2 workers and by 8 to 9 % with 4 and 8. Starting processes,
writing and reading the pickles, and other work on the shared cores probably account for it; the overhead was not broken
down, and with single runs on a shared host the differences between 4 and 8 workers are within what the load could
cause.

The statement records (timing keys removed) and the arrays of every certificate of the parallel runs equal those of the
one-worker run. This is what rules 1 to 4 of `parallel.py` promise, and `tests/test_parallel.py` checks it on small cases.

The run does not show the speed-up on an idle machine, for other models, or for claims much larger than 12870 states.
D-04 parallelizes across claims and sectors, not inside one certification, so a set dominated by one large claim gains
little.

A first attempt with the sectors of the chain of 18 sites (dimensions up to 48620) was cancelled after 81 minutes. Its
one-worker run had not finished, and the job would not have finished within its clock; no number of it is recorded.

## Second run (27 November 2026)

The same command ran again at commit f5864761 (source hash d689f4159bae8dd2, no tracked changes), with the script of
rounds 7 and 8, on the same host, pin, thread settings and cap. The load average was between 117 and 124 before and
after every worker count (the record keeps both readings).

| workers | wall (s) | speed-up | sum of task times (s) | longest task (s) | CPU s, this process | CPU s, workers | certificates equal to one worker |
|---|---|---|---|---|---|---|---|
| 1 | 310.2 | 1.00 | 310.2 | 104.2 | 309.7 | 0.19 | (reference) |
| 2 | 175.6 | 1.77 | 320.5 | 108.0 | 2.2 | 321.3 | yes |
| 4 | 116.6 | 2.66 | 329.5 | 109.3 | 1.6 | 332.1 | yes |
| 8 | 110.7 | 2.80 | 318.5 | 110.2 | 1.4 | 325.2 | yes |

Here the equality covers the whole certificate body and the arrays, except the seven time fields that the record lists
as `fields_not_compared`. The statuses and reason codes are the same at every worker count and are those of the first
run: every statement CERTIFIED except S4 and S8 at nup = 8, NOT_CHECKED with `NO_CLUSTERS`. The CPU seconds of the
workers come from `os.times`, which counts only child processes that have been waited for; with 2, 4 and 8 workers they
exceed the sum of the task times by 0.3 %, 0.8 % and 2.1 %, so every worker had finished and been waited for when the
run was recorded.

The speed-ups agree with the first run to within 0.09 (1.77 and 1.77, 2.58 and 2.66, 2.75 and 2.80). The
longest one-worker task (nup = 9, 104.2 s of 310.2 s) bounds the speed-up at 2.98, and 8 workers reach 94 % of it. A list
schedule of the one-worker times in input order ends after 173.7 s with 2 workers and 111.4 s with 4 (157.1 s and
104.2 s with the longest task first), close to the measured 175.6 s and 116.6 s. The sum of the task times exceeds
the serial run by 3.3 %, 6.2 % and 2.7 % with 2, 4 and 8 workers. Each configuration ran once.

## Files

- `docs/results/d04_speedup_n16_20261122.json`: the record of the run (`d04_speedup`) with the versions and the
  commit of its environment.
- `docs/results/d04_speedup_n16_20261127.json`: the record of the second run, in the same layout, with its
  environment.
- `experiments/d04_speedup.py`: the experiment.
