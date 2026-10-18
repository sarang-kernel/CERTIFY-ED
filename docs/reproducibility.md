# Reproducibility

## One command

```bash
pip install -e .[all]          # numpy, scipy, python-flint, quspin, h5py, matplotlib, pytest
python run.py                  # quick mode
python run.py --full           # every experiment at paper size
```

`run.py` is `certify-ed reproduce`. Options:

| option | effect |
|---|---|
| `--full` | paper-size runs of all experiments except E6 |
| `--only E0,E9` / `--skip E2` | select experiments |
| `--with-mutation` | add E6 (mutation analysis; about 15 min for 64 mutants on two workers) |
| `--no-tests`, `--no-numbers` | skip the test suite or the numbers step |
| `--out DIR` | run directory (default `results/run_<mode>_<timestamp>/`); an existing run directory of the same mode is extended, with the commit recorded per step |
| `--manuscript DIR` | where numbers and tables go (default `<run dir>/paper/`) |

Experiments run one after another, each single-threaded (`OMP_NUM_THREADS=1` and the like). That is
needed for the timing experiment E2 and keeps memory below about 5 GB. An experiment whose optional
dependency is missing is skipped and recorded as such, never silently dropped.

## What a run directory contains

```
manifest.json      commit, dirty flag, environment (versions, CPU, float probe), per-step status/time/records
SUMMARY.md         the same as a table
run.log            driver log;   tests.log   pytest output
E<k>.log           stdout of each experiment
e<k>_<name>_<ts>/  one record per experiment: data.json, environment.json, run.log, figures/
paper/             numbers.tex, numbers_apps.tex, tables/, figures/ and the statement checks
make_numbers*.log  every check of a statement in the text, with pass/fail
```

## Experiments

| | content | quick | full (one core) |
|---|---|---|---|
| E0 | verifier vs Arb balls, exact eigenvalue counts, QuSpin construction | same | ~5 min |
| E1 | planted errors: which statement catches which failure | same | ~1 min |
| E2 | cost and memory of S2/S3 against size | skipped | ~21 min |
| E3 | tightness of radii against Arb | same | ~1 min |
| E4 | workflows: global spectrum from sectors, multiplets, matrix-free, benchmark values | skipped | ~5 min |
| E5 | unplanted error: ARPACK misses a multiplet member | skipped | ~27 min |
| E6 | mutation analysis (opt-in) | opt-in | ~15 min (64 mutants) |
| E7 | benchmark E0 values with no floating-point assumption | 4×4 Heisenberg | ~12 min (four systems) |
| E8 | symveig 1.0.0 vs `certify_ed.spectrum` | same | ~2 min |
| E9 | pyrochlore crown | S = 1 parts | + exact S = 1/2 certificate and the full S = 1 landscape (~25 min) |
| E10 | AKLT criteria | five lattices, star kernel | + decorated square (~45 min) and kernel checks (~25 min) |

## Comparing two machines (two-stack replication)

1. On each machine, check out the same commit and run `python run.py --full`.
2. Compare `manifest.json → environment` (library versions, BLAS, float probe) between the two.
3. Compare the records. Certified values must agree exactly when a step is deterministic (exact
   arithmetic: E7, E9, E10, S6) and within their certified radii otherwise. Timings and memory
   (E2) are machine-specific. `paper/make_numbers*.log` must show every check passing on both.

The exact steps do not depend on BLAS or on the floating-point environment for their *validity*.
A different BLAS can only change the proposed factorization, and therefore β and the time taken.
