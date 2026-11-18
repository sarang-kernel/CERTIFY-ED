#!/bin/sh
# Regenerate every number in the paper. Two single-threaded queues:
#   queue A: E2 (cost; timing-sensitive, up to 4.5 GiB per child), then E4 (workflows, up to 5 GiB),
#            then E5 (ARPACK multiplets; dense reference of a 12870-dimensional block, about 3 GiB)
#   queue B: E0, E1, E3 (small memory)
# E2, E4 and E5 are never run concurrently: together they can exceed the memory available to the job.
# Afterwards: python3 make_numbers.py  (writes the manuscript's numbers.tex, tables, figures)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
( python3 e2_cost.py && python3 e4_workflows.py W1 W2 W3 W4 N=20 && python3 e5_arpack_multiplet.py 12 ) \
  > results/queueA.log 2>&1 &
( python3 e0_verifier_validation.py && python3 e1_failure_matrix.py && python3 e3_tightness.py ) \
  > results/queueB.log 2>&1 &
wait
echo "all experiments finished"
