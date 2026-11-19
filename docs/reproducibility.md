# Reproducibility

## One command

```bash
pip install -e .[all]          # numpy, scipy, python-flint, quspin, h5py, matplotlib, pytest
python run.py                  # quick mode
python run.py --full           # every experiment at paper size
python run.py --full --jobs 4  # the same, four experiments at a time (E2, E6, E8 still run alone)
```

### Installing QuSpin 1.0.1

The QuSpin 1.0.1 wheel declares the obsolete PyPI backport `importlib` as a requirement, which cannot be
installed on Python 3.12 (`docs/results/RESULT_candidate_run.md`). Install its real runtime dependencies first and then
the wheel without dependency resolution:

```bash
pip install quspin-extensions parallel-sparse-tools dill joblib numba numexpr six gmpy2
pip install quspin==1.0.1 --no-deps
```

This is the sequence `.github/workflows/ci.yml` and the container recipes in `containers/` use; the
versions of the 3.1.0 evidence run are Python 3.11, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0 (FLINT
3.6.0), QuSpin 1.0.1, quspin-extensions 0.1.8, parallel-sparse-tools 0.2.5.

`run.py` is `certify-ed reproduce`. Options:

| option | effect |
|---|---|
| `--full` | paper-size runs of all experiments except E6 |
| `--only E0,E9` / `--skip E2` | select experiments |
| `--with-mutation` | add E6 (mutation analysis; about 15 min for 64 mutants on two workers) |
| `--no-tests`, `--no-numbers` | skip the test suite or the numbers step |
| `--out DIR` | run directory (default `results/run_<mode>_<timestamp>/`); an existing run directory of the same mode is extended, with the commit and source fingerprint recorded per step |
| `--resume` | with `--out`: keep every step that already succeeded with the same computational source (see Provenance) |
| `--jobs N` | run up to N experiments at once, longest first; E2, E6 and E8 (timing or memory is what they measure) always run alone afterwards |
| `--manuscript DIR` | where numbers and tables go (default `<run dir>/paper/`) |

Each experiment runs single-threaded (`OMP_NUM_THREADS=1` and the like). By default they run one
after another; `--jobs N` runs N at a time, which needs about N × 2.5 GB of memory. An experiment
whose optional dependency is missing is skipped and recorded as such, never silently dropped.

## Provenance

Every record (`environment.json`) and every step of `manifest.json` carries a *source fingerprint*
(`certify_ed.provenance.source_fingerprint`):

* `git_commit`;
* `source_hash`: a hash of the git blobs of `certify_ed/` and `experiments/`, leaving out the files
  that cannot change a number (the command line, the test helpers, the provenance module itself,
  and the post-processing scripts `make_numbers*.py`, `e6_classify.py`, `e6_recheck.py`, which only
  read records);
* `source_dirty`: whether those paths had uncommitted changes;
* `tests_hash`, `tests_dirty`: the same for `tests/`. The test suite never changes a computed number,
  but it decides the mutation score, so E6 is tied to it.

The numbers scripts check that all records they read share one `source_hash` with
`source_dirty = false`, and that E6 ran against committed tests. Commits that differ only in
documentation, tests, post-processing or the command line therefore still count as one computation.
`--resume` relies on the same rule to keep earlier steps, and it also re-runs E6 when `tests_hash`
changed. After E6 the driver runs `e6_classify.py` on the new record. The hand inspection of the
survivors that need it (`inspection.json`) remains a separate step.

Since 3.1.0 (review of 2026-10-21):

* git is called with `core.filemode=false`, so a lost executable bit (Windows checkouts and some
  copies) does not count as an uncommitted change (IMP-05 / SS-02).
* The numbers step records its own provenance in `manifest.steps.numbers.provenance`: the commit, the
  git blob hash and content hash of `make_numbers.py` and `make_numbers_apps.py`, whether either is
  modified, and the source and tests hashes at that time (IMP-06). The commit of the number scripts
  that produced a set of macros can therefore be read from the run directory.
* `\NtestsTotal`, `\NtestsPassed`, `\NtestsSkipped` come from the run's own tests step
  (`manifest.steps.tests.summary`), never from a collection on the machine that regenerates the
  numbers (IMP-07 / MS-N02).
* The post-run files of an E6 record (`classification.json`, `recheck.json`, `inspection.json`) must
  carry a provenance stamp (`source_hash`, `tests_hash`); `e6_classify.py` and `e6_recheck.py` write it,
  the hand-written `inspection.json` must include one, and `make_numbers_apps.py` reports CHECK FAILED
  without it (MS-N03). Records of 3.0.0 runs therefore fail these three checks when regenerated with
  3.1.0 scripts; that is intended.
* `environment.json` and every certificate record the FLINT C-library version (`flint_c_library`),
  the `certify_ed` version and the `source_hash` (IMP-26, MS-N12).
* An experiment that raises writes `_aborted` into `data.json` and "ABORTED" into its log (SS-04).

Platform notes: 37 of the 3.0.0 tests needed QuSpin (35) or gcc/glibc (2). The two floating-point
probe tests now run on Windows as well (through `ucrtbase` `fesetround`); only the `-ffast-math`
shared-library test still needs gcc on Linux/macOS.

### The manuscript's macros (v4)

`certify-ed numbers --run DIR --manuscript MS` writes three macro files into `MS/`, and the
manuscript inputs all three:

* `numbers.tex` (`make_numbers.py`, E0–E5 records, the run manifest and the repository);
* `numbers_apps.tex` (`make_numbers_apps.py`, E6–E10 records, the manifest and the repository);
* `numbers_todo.tex` (`make_numbers_apps.py`): the macros the text uses whose value **no record of this
  run contains**, each defined as `\todo{reason}`. The file is generated like the other two and is never
  edited by hand; when every macro has a record it holds a comment only. `make_numbers_apps.py` prints
  one `pending` line per entry.

Rules that the two scripts follow (so that the manuscript builds from `certify-ed numbers` alone):

* **One definition per macro.** `make_numbers.py` runs last and reports `CHECK FAILED` if a macro is
  defined by `\newcommand` in two files of `MS/`. The environment macros (`\FlintCVersion`,
  `\FlintLibVersion`, `\PythonFlintVersion`, `\CertifyEdVersion`) are owned by `make_numbers.py` (E0
  record); `make_numbers_apps.py` writes them with `\providecommand`, which only takes effect in a quick
  run without `numbers.tex`.
* **Every macro names its source.** A macro added since 3.1.0 carries a LaTeX comment after its
  definition naming the record and field (or the repository file, or the paper) its value comes from.
  Values typed from the literature (`\CrownHalfReducedPaper`, `\CrownHalfHeadlinePaper`) are marked
  `literature:` in that comment.
* **Nothing is substituted.** A number the text needs that the run did not record goes to
  `numbers_todo.tex`; the scripts never fill it from another coupling, another run or an earlier
  document. Examples: `\CrownOneNamedSectorA/B` (the two pair-spin sectors named by Kranitz & Penc at
  the printed couplings) need `spin1_printed.named_sectors` in the E9 record; `\MutGapsClosedStamped`
  needs stamped `inspection.json` and `recheck.json` (MS-N03); `\CrownHalfHeadlineFloat` needs a
  nine-coupling floating-point step that E9 does not have yet.
* **Code facts come from the checkout.** `\ExactCheckerLines` (code lines of `certify_ed/checker.py` +
  `exact_sectors.py` without blanks, comments and docstrings, from the token stream) and
  `\NtestsAddedAfterRun` are computed from the repository that regenerates the numbers.
  `\NtestsAddedAfterRun` is `0` by identity when the run's `tests_hash` equals the repository's;
  otherwise it is `pytest --collect-only` on the checkout minus the run's count, a live collection on the
  regenerating host, on which a module skipped at import (`tests/test_label_audit.py` without QuSpin)
  is not counted. The comment in `numbers.tex` says which case applied.
* **Numbers that live only in a log.** `\DecSquareNoSymDim` (the unreduced decorated-square block) is
  read from the E10 record's `run.log`, the only place the 3.1.0 harness writes it; `\SymveigPtwoOutside`
  is read from `docs/symveig_audit.md` (the project's audit of symveig 1.0.0; the E8 record compares
  conserving models only). Both comments say so.

## What a run directory contains

```
manifest.json      commit, source fingerprint, environment (versions, CPU, float probe), per-step status/time/records/fingerprint
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
| E0 | verifier vs Arb balls, exact eigenvalue counts, QuSpin construction | same | ~4 min |
| E1 | planted errors: which statement catches which failure | same | ~1 min |
| E2 | cost and memory of S2/S3 against size | skipped | ~22 min |
| E3 | tightness of radii against Arb | same | ~1 min |
| E4 | workflows: global spectrum from sectors, multiplets, matrix-free, benchmark values | skipped | ~6 min |
| E5 | unplanted error: ARPACK misses a multiplet member | skipped | ~38 min |
| E6 | mutation analysis (opt-in) | opt-in | ~19 min (64 mutants, two workers) |
| E7 | benchmark E0 values with no floating-point assumption | 4×4 Heisenberg | ~11 min (four systems) |
| E8 | symveig 1.0.0 vs `certify_ed.spectrum` | same | ~3 min |
| E9 | pyrochlore crown | S = 1 parts | + exact S = 1/2 certificate and the full S = 1 landscape (~17 min) |
| E10 | AKLT criteria | five lattices, star kernel | + decorated square (~17 min) and kernel checks (~20 min); ~40 min in all |

Measured on the 2-core reference machine (run `results/run_full_final`, 2026-10-18): 2.7 h of
experiment time in all (E6 included). With `--jobs 2` the wall time was 1 h 46 min.

## Comparing two machines (two-stack replication)

1. On each machine, check out the same commit and run `python run.py --full` (add `--jobs N` on a
   machine with N free cores and N × 2.5 GB of memory).
2. Compare `manifest.json → environment` (library versions, BLAS, float probe) between the two.
3. Compare the records. Certified values must agree exactly when a step is deterministic (exact
   arithmetic: E7, E9, E10, S6) and within their certified radii otherwise. Timings and memory
   (E2) are machine-specific. `paper/make_numbers*.log` must show every check passing on both.

The exact steps do not depend on BLAS or on the floating-point environment for their *validity*.
A different BLAS can only change the proposed factorization, and therefore β and the time taken.
