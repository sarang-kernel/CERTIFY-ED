# The command line, the presets and the budgets

The package installs one command, `certify-ed` (`certify_ed/cli.py`); `python run.py` at the root of the code
repository is the same as `certify-ed reproduce`. This document lists every subcommand of `a814d28` with its options,
then the presets, the selection rules, the budgets and the reason codes. The option lists are read from the argument
parser in `code/certify-ed/certify_ed/cli.py`; the presets and reason codes from `code/certify-ed/docs/presets.md` and
`certify_ed/presets.py`.

## Subcommands for certification

| Subcommand | What it does | Main options |
|---|---|---|
| `env` | runs the floating-point probe and prints the versions of every library | none |
| `certify` | certifies a claim given as a file (`.npz`, `.h5`, `.csv` and the other formats of `06_formats_and_schema.md`) or as a producer bundle (`--from DIR`) | `--spec`, `--kind auto,subset,lowest,window,all`, `--window A B`, `--exact` (adds S6), `--preset`, `--only`, `--skip`, `--reason`, `--budget-seconds`, `--budget` (a number or `auto`), `--budget-mb`, `--progress`, `--temperatures` (S7), `--assume-no-underflow`, `--allow-code`, `--plan`, `--block-mb`, `--eigenvectors`, `--operator`, `--basis`, `-o` |
| `import` | the same as `certify --from BUNDLE` | the bundle directory |
| `plan` | prints the predicted seconds and peak memory per statement without running anything (`certify_ed.cost`) | the claim or `--from` |
| `statement` | writes the methods-section paragraph and a BibTeX entry from a saved certificate | `--style plain,latex`, `--no-verify`, `-o` |
| `view` | writes an offline HTML page of a saved certificate (self-contained, no scripts) | `-o`, `--max-pairs` |
| `conform` | checks producer bundles: the schema of `meta.json`, the import, the certification, and the schema of the resulting certificate | `--allow-code`, `--require-jsonschema`, `-o`, `--json` |
| `spectrum` | encloses a whole spectrum from an `.npz` holding H | `--sectors KEY` (a conserved diagonal), `-o`, `--show` |
| `presets` | prints the presets and statements table | `--markdown` |

## Subcommands for records built on certificates

| Subcommand | What it does | Main options |
|---|---|---|
| `crossing` | orders the intervals of the eigenvalue of a given rank in two certificates (roadmap item C-11) | `--level`, `--no-verify`, `--json` |
| `sweep` | certified eigenvalue tubes over a coupling interval for a built-in model family (roadmap item E-04) | `--model`, `--sizes`, `--param`, `--values`, `--fixed`, `--k`, `--two-sz`, `--momentum`, `--crossing`, `--fast`, `--save-dir`, `--json` |
| `variational` | Rayleigh upper bound, existence radius and Temple lower bound for a trial state (roadmap item E-05) | `--operator` or `--spec`, `--state`, `--mps`, `--ci-indices`, `--dim`, `--gap` with `--gap-source`, or `--gap-certificate`, `--declared`, `--json` |
| `levelstats` | interval r-ratios and a spacing histogram of a complete counted spectrum (roadmap item C-09) | `--unfold none` or `poly:D`, `--bins`, `--smax`, `--json` |
| `extrapolate` | finite-size extrapolation report from per-size certificates (roadmap item C-10); not certified, and the report says so | `--model 1/N` or `1/N^2`, `--sizes`, `--per-site`, `--level`, `-o`, `--json` |

In `variational`, a gap given with `--gap` is recorded as a declared hypothesis together with `--gap-source`; only a
gap taken from a certificate with `--gap-certificate` is a certified number. Producer estimates passed with
`--declared` (for example an energy and a variance from a Monte Carlo run) are recorded as declared and never enter a
certified bound.

## Subcommands for reproduction

| Subcommand | What it does | Main options |
|---|---|---|
| `list` | lists the experiments E0 to E10 | none |
| `test` | runs the test suite through pytest | arguments passed to pytest |
| `reproduce` | runs the tests, the experiments and the number scripts into one run directory | `--full`, `--only`, `--skip`, `--with-mutation` (adds E6), `--no-tests`, `--no-numbers`, `--out DIR`, `--resume`, `--jobs N`, `--manuscript DIR` |
| `numbers` | regenerates the numbers, tables and figures of the manuscript from a run directory | `--run DIR`, `--manuscript DIR` |

Three rules of the driver matter for the records. E2, E6 and E8 always run alone, after the others, because they
measure time or memory (decision D-44). `--out` with an existing run directory of the same mode extends it, recording
the commit and the source fingerprint per step, and `--resume` keeps a finished step only if its computational source
is unchanged; for E6 it also requires an unchanged test suite. After E6 the driver runs the rule-based classification
of the surviving mutants (`experiments/e6_classify.py`). The commands that recreate every record are in
`docs/reproduction/`.

## Presets

A preset is a named set of statements and option values, in the manner of the `-preset` option of ffmpeg (decision
D-49). The default run accepts whatever the producer handed over and runs every statement whose inputs exist, which
means S1 to S4 and S8; S5, S6 and S7 switch on through the options `corroborate`, `exact_recheck` and `temperatures`.
A whole-spectrum claim without a preset, `--only` or `--skip` uses the `whole-spectrum` preset.

| Preset | Statements | Options set | Intended use |
|---|---|---|---|
| `lab-minimal` | S1 S2 | `inertia=False`, `eigenspaces='none'` | is the matrix the declared model, and are the claimed pairs eigenpairs (no factorization) |
| `benchmark` | S1 S2 S3 S5 | `corroborate=('jacobi', 'arpack')`, `eigenspaces='none'` | a reference value for other methods: enclosures, completeness, a gap, and corroboration |
| `proof-lemma` | S1 S2 S3 S6 | `exact_recheck=True`, `eigenspaces='none'` | a number that enters a proof: S1 to S3, then the exact integer re-check |
| `whole-spectrum` | S1 S2 S3 S7 | `eigenspaces='none'`, `whole_spectrum=True` | every eigenvalue enclosed and counted; thermodynamic intervals when temperatures are given |
| `labels` | S1 S2 S3 S4 S8 | `eigenspaces='all'` | quantum numbers, degeneracies and observables of every claimed level |
| `full` | S1 to S8 | `corroborate=('jacobi', 'arpack')`, `exact_recheck=True`, `eigenspaces='all'` | everything; S5 to S7 run only when their inputs exist |

## Selection rules and budgets

`--only A,B` runs A and B and whatever they need; implied statements are listed in `scope.selection.implied`. `--skip
X` removes X, and a statement that needs X is narrowed with the reason `DEPENDENCY_NARROWED`. A preset's option values
apply first and explicit options override them. Narrowing does not change what is said about a statement still in
scope: the same claim with `--skip S3` gives the same S1 and S2 records, which `tests/test_presets_scope.py` tests.

Budgets never raise an exception. With `--budget-seconds S`, a statement that has not started S seconds after
`certify()` began is NOT_CHECKED with the reason `BUDGET_TIME`; a running statement is not interrupted, because the
check happens between statements. With `--budget-mb M`, a statement whose memory estimate exceeds M is NOT_CHECKED
with the reason `BUDGET_MEMORY`; the estimates are coarse and are written into the statement. A factorization that
fails for lack of memory at run time is also NOT_CHECKED, as `resource_failure`, not an error. `--budget auto` sets
the time budget to max(60 s, 5 times the estimate) from the advisory cost model in `cost.py`, which is fitted to the
E2 records of the 3.1.0 evidence run (for example 0.7, 2.9 and 65.5 s for the chain blocks of dimension 924, 3432 and
12870, and 175.4 s for the 4x4 square, per `code/certify-ed/docs/presets.md`). The estimate never changes a status,
and `scope.options.budget_auto` records how it was derived. `--progress` prints one line per finished statement to
standard error, and `timing.per_statement` in the certificate holds the seconds of every statement.

## Reason codes

A statement that did not run, and every quantity declared out of scope, carries a reason code from
`certify_ed.scope.REASON_CODES`. The codes are listed with their meaning in `code/certify-ed/docs/presets.md`; their
groups are as follows. User choices: `NARROWED_BY_USER`, `DEFAULT_SELECTION`, `DEPENDENCY_NARROWED`. Budgets:
`BUDGET_TIME`, `BUDGET_MEMORY`. Missing inputs: `NEEDS_EIGENVECTORS`, `NEEDS_SPEC`, `NEEDS_MATRIX`,
`NEEDS_WHOLE_SPECTRUM`, `NEEDS_TEMPERATURES`, `NEEDS_DEPENDENCY`, `NEEDS_OBSERVABLES`, `MISSING_DEPENDENCY` (an
optional package such as python-flint is not installed; the evidence names the extra to install). Properties of the
claim: `NO_COMPLETENESS_CLAIM` (a `subset` claim says nothing about other eigenvalues), `NO_CLUSTERS`, `SIZE_LIMIT`,
`UNSUPPORTED_CLAIM`, `UNSUPPORTED_MODEL`, `NOT_APPLICABLE`. Design slots and standing exclusions: `NOT_IMPLEMENTED`,
`OUT_OF_SCOPE_DYNAMICS` (dynamical response and time evolution are not statements about the eigenpairs of one matrix),
`OUT_OF_SCOPE_EXTRAPOLATION` (finite-size extrapolation and phase boundaries are not statements about any finite
matrix). Version 3.2 adds `UNDERFLOW_RISK` (a nonzero operand, radius or residual bound below 2^-900, so the
no-underflow hypothesis is not established; `--assume-no-underflow` skips the scan and the certificate records that
choice) and `DECLARED_ESTIMATE` (a stochastic estimate declared by the producer, recorded and never certified).

Every statement identifier appears exactly once across the lists `in_scope`, `narrowed` and `out_of_scope` of the
scope block, and the standing out-of-scope quantities are listed in every certificate with their reason codes, so that
a reader never has to guess what a certificate is silent about. Since `fix/3.1.1` (merged into 3.2),
`Certificate.verify_scope()` checks this when a certificate is loaded (INT-02 in
`docs/failures/02_integration_findings.md`).
