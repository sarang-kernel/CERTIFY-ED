# The experiments E0 to E10

The evidence for what certify-ed does comes from eleven experiments, E0 to E10, each a script in
`code/certify-ed/experiments/` that writes a record (`data.json`, `environment.json`, `run.log`) into a run directory.
The reproduction driver (`certify-ed reproduce`, the same as `python run.py`) runs the test suite, the experiments and
the two number scripts into one directory; the number scripts turn the records into the macros, tables and figures of
the manuscript and check every qualitative sentence the text makes about them. This folder describes why each
experiment exists, how it is designed, how its instrument is checked, and what every evidence run recorded.

## Why there are eleven experiments

Each experiment answers one question that an earlier result, a referee report or a cost argument made necessary. E0
asks whether the bounds hold, using references that share no code with certify-ed (ball arithmetic, exact counts, an
independent builder and QuSpin). E1 answers the second SciPost report, which said that the gap between ordinary
practice and certification was not demonstrated: it applies three practices to the same erroneous results. E2 measures
the cost of S1 to S3 up to the point where they fail, because a certificate that costs more than a user will pay is
not used, and the failure points are part of the claim. E3 asks whether the bounds are tight enough to be useful,
since a true bound that is orders of magnitude too loose says little. E4 runs four workflows of the kind researchers
use. E5 records an error that nobody planted, which answers the objection that the planted errors of E1 were chosen to
suit the method (decision D-27). E6 measures how many deliberate faults in the certifier the test suite detects
(decision D-35). E7 re-derives the benchmark values of E4 with no floating-point assumption (decision D-41). E8
compares the whole-spectrum enclosure with the symveig package it replaced (decision D-42). E9 and E10 are the
applications to published proofs: the pyrochlore crown cluster of Kránitz and Penc (decisions D-34, D-40, D-51) and
the AKLT gap criteria of Pomata and Wei (decisions D-36 to D-41).

| Experiment | Script | Question | Document |
|---|---|---|---|
| E0 | `e0_verifier_validation.py` | do the bounds hold against independent references? | `E0_verifier_validation.md` |
| E1 | `e1_failure_matrix.py` | which practice catches which realistic error? | `E1_failure_matrix.md` |
| E2 | `e2_cost.py` | what do S1 to S3 cost, and where do they stop? | `E2_cost.md` |
| E3 | `e3_tightness.py` | how tight are the certified statements? | `E3_tightness.md` |
| E4 | `e4_workflows.py` | do realistic workflows certify? | `E4_workflows.md` |
| E5 | `e5_arpack_multiplet.py` | what happens with an error nobody planted? | `E5_arpack_multiplet.md` |
| E6 | `e6_mutation.py` | how many faults in the certifier do the tests detect? | `E6_mutation.md` |
| E7 | `e7_exact_benchmarks.py` | do the benchmark values hold with no floating-point assumption? | `E7_exact_benchmarks.md` |
| E8 | `e8_symveig_comparison.py` | how does the new whole-spectrum enclosure compare with symveig? | `E8_symveig_comparison.md` |
| E9 | `e9_pyrochlore.py` | are the cluster eigenvalues of the pyrochlore bound certified? | `E9_pyrochlore.md` |
| E10 | `e10_aklt.py` | do the AKLT gap criteria hold in exact arithmetic? | `E10_aklt.md` |

Three further documents cover the runs as a whole: `01_run_history.md` lists every run directory in the archive and
why each was made, `02_evidence_runs.md` describes the three evidence runs of 3.1.0, 3.2.0 and 3.2.1 step by step,
including the failed first attempt of 22 October 2026 and the E4 crash of the 3.2.0 run, and `03_comparisons.md` gives
the record-by-record comparisons between runs with every MISMATCH leaf explained. Two scripts outside E0 to E10
reproduce single results of the 3.2 scale work: `experiments/e2_s3_reach.py` (the reach of the sparse S3 path with the
a-posteriori residual) and `experiments/d02_matrixfree_scale.py` (S2 radii for claims whose matrix is never stored, up
to dimension 2^24); they are listed in `docs/CODE_INDEX.md` and are not part of the evidence runs.

## Rules that every run follows

The evidence policy is decision D-24: every number, table and figure of the paper is generated from experiment records
by the number scripts, which also check the qualitative sentences, and all experiments of a cited run come from one
computational source (decision D-44, `docs/software/05_provenance.md`). Every iterative solver call gets a seeded
start vector (decision D-26). A record of an interrupted run is never read (decision D-29). Two runs on different
stacks must certify the same statements, while digits that depend on the floating-point stack may differ (decision
D-54). E2, E6 and E8 run alone because they measure time or memory. Each experiment runs single-threaded. Memory caps
are 4.5 GiB per child in E2 and 5 GiB in E4; on Linux they are address-space limits and on Windows job objects, and
the mechanism is recorded per data point (`experiments/portable.py`).

## Which run is cited

The manuscript still cites 3.1.0 and its evidence run `run_full_20261022` (decision D-61). The 3.2 evidence run is
`run_full_20261105` of version 3.2.1. The archive's manuscript macros are generated from the 3.2.1 run, and those of
3.1.0 are kept beside them, so that the cited run is switched by swapping two directories; the switch is Sarang
Vehale's decision. Each experiment document therefore gives the numbers of the 3.0.0 run `run_full_20261021` (the
first run with a clean single source, cited by the report up to version 1.1), of the 3.1.0 run and of the 3.2.1 run,
and notes where the 3.2.0 run `run_full_20261104` differs. Numbers come from the macro files `paper/numbers.tex` and
`paper/numbers_apps.tex` of each run directory, or from the records themselves, and each table names its source.
