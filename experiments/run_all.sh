#!/bin/sh
# Regenerate every number in the paper. Two independent single-threaded queues:
#   queue A: E2 (cost; timing-sensitive, runs alone on one core)
#   queue B: E0, E1, E3, E4 (sequential on the other core)
# Afterwards: python3 make_numbers.py  (writes the manuscript's numbers.tex, tables, figures)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
( python3 e2_cost.py > results/queueA.log 2>&1 ) &
( python3 e0_verifier_validation.py && python3 e1_failure_matrix.py && python3 e3_tightness.py \
  && python3 e4_workflows.py W1 W2 W3 W4 N=20 ) > results/queueB.log 2>&1 &
wait
echo "all experiments finished"
