# RESULT_d04_speedup.md: the worker pool of D-04 on free cores

Roadmap item D-04 (`certify_ed/parallel.py`) runs independent certifications in worker processes, each with a share of
the process memory cap. It had shown no speed-up: the measurements before 22 November 2026 were made on a machine
without free cores (`docs/project/failures/05_open_items.md`). This document records the first measurement on free
cores.

## What was run

`experiments/d04_speedup.py --N 16 --nups 3,4,5,6,7,8,9,10,11,12,13 --workers 1,2,4,8 --cap-gib 32`, at commit 57607a70
(source hash 337392e110cebd3d). The jobs are the eleven U(1) sectors nup = 3 to 13 of the periodic Heisenberg chain of
16 sites (dimensions 560 to 12870). Each sector claims its 3 lowest eigenpairs from scipy's eigsh, cut at a gap, and
`certify()` runs with the default Options. For each worker count `parallel.certify_many` certifies the eleven claims,
and the record compares every status and every recorded number, timings excluded, with the one-worker run. The host was
Linux (Python 3.12.3, numpy 2.5.4, scipy 1.18.1), with one pin of 16 cores, nice 10, BLAS and OpenMP threads 1, and a
memory cap of 32 GiB (RLIMIT_AS). The host is shared: its load average was about 113 on 208 cores during the run, so
other work may also have run on the 16 pinned cores.

## What happened

| workers | wall (s) | speed-up | sum of task times (s) | longest task (s) | certificates equal to one worker |
|---|---|---|---|---|---|
| 1 | 324.8 | 1.00 | 324.8 | 111.3 | yes |
| 2 | 183.0 | 1.77 | 334.0 | 112.5 | yes |
| 4 | 125.9 | 2.58 | 354.4 | 118.6 | yes |
| 8 | 118.0 | 2.75 | 349.5 | 117.4 | yes |

All eleven tasks ran in workers at every worker count above one. None was deferred because its predicted peak exceeded
the per-worker budget (rule 3), and none was redone in this process (rule 4). The one-worker times per sector:

| nup | dimension | claimed pairs | seconds |
|---|---|---|---|
| 3 | 560 | 3 | 0.2 |
| 4 | 1820 | 3 | 3.1 |
| 5 | 4368 | 3 | 6.7 |
| 6 | 8008 | 3 | 26.2 |
| 7 | 11440 | 3 | 104.1 |
| 8 | 12870 | 3 | 36.9 |
| 9 | 11440 | 3 | 111.3 |
| 10 | 8008 | 3 | 27.4 |
| 11 | 4368 | 3 | 5.4 |
| 12 | 1820 | 3 | 3.3 |
| 13 | 560 | 3 | 0.2 |

## What it shows and does not show

The pool shortens the run by a factor of 1.77 with 2 workers, 2.58 with 4 and 2.75 with 8. From 4 workers on, the wall
time is set by the longest single certification. The sectors nup = 7 and 9 took 104 s and 111 s of the 325 s serial
total, so no schedule of whole claims can finish in less than about 111 s. That bounds the speed-up of this set at 2.92,
and 8 workers reach 94 % of the bound. The sum of the task times exceeds the serial run by 3 % with 2 workers and by 8
to 9 % with 4 and 8, the cost of starting processes, of the pickles and of the shared cores. Every certificate of the
parallel runs equals the one-worker certificate in every status and recorded number. This is what rules 1 to 4 of
`parallel.py` promise, and `tests/test_parallel.py` checks it on small cases.

The run does not show the speed-up on an idle machine, for other models, or for claims much larger than 12870 states.
D-04 parallelizes across claims and sectors, not inside one certification, so a set dominated by one large claim gains
little. Why the largest sector (nup = 8) took 37 s and its neighbours more than 100 s was not examined.

A first attempt with the sectors of the chain of 18 sites (dimensions up to 48620) was cancelled after 81 minutes. Its
one-worker run had not finished, and the job would not have finished within its clock; no number of it is recorded.

## Files

- `docs/results/d04_speedup_n16_20261122.json`: the record of the run (`d04_speedup`) with the versions and the
  commit of its environment.
- `experiments/d04_speedup.py`: the experiment.
