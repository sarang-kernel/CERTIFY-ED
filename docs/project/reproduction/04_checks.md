# Checks of runs, certificates and derivations

This document gives the commands that check a result rather than produce one: the comparison of two runs, the
independent verifier, the scripts that test the derivations of `docs/proofs/`, the planted-error corpus, the
conformance bundles, the JSON Schemas and the examples. Each section says what was run again on 10 November 2026, in a
clone of the bundle at `a814d28` on the Windows workstation, and what came out. Commands are run from the root of the
checkout with the settings of `01_environment.md`.

## Comparing two runs

```
python experiments\compare_runs.py <run A> <run B> <out.csv>
```

The script flattens the `data.json` of every experiment present in both runs into leaves and classifies each leaf
(`same`, `float-differs`, `stack-dependent`, `fp-statistic`, `text-numbers`, `extended`, `sample-dependent`,
`explained`, `timing`, `missing`, `MISMATCH`). It prints a summary per experiment and per class, writes one CSV row
per leaf, and exits with 1 when a leaf is a `MISMATCH`. The classes, their order and the two limits of the instrument
(a tolerance divided by max(1, |A|), and `EXPLAINED` rules that match by path) are described in
`docs/project/experiments/03_comparisons.md`. Run A is the older run: in the archived files `compare_vs_<run A>.csv` lie in
the directory of run B.

The comparison of the 3.1.0 run with the 3.2.1 run was made again from the archived records:

```
python experiments\compare_runs.py <results>\run_full_20261022 <results>\run_full_20261105 ..\compare_vs_run_full_20261022.csv
```

Its CSV is equal, row for row, to `run_full_20261105/compare_vs_run_full_20261022.csv`, and the exit code is 1 because
the comparison has 10 `MISMATCH` leaves. The run of E1 alone (`02_tests_and_experiments.md`) compared with the 3.2.1
run gave no `MISMATCH`.

## The independent verifier

```
python verifier\build.py --cc "<path to gcc.exe>"
verifier\build\cev.exe <certificate.json> --operator <bundle>\operator.npz [--report out.txt] [--json out.json] [--quiet]
python verifier\validation\run_validation.py --cc "<path to gcc.exe>"
```

`build.py` compiles `verifier/src/cev.c` with `-frounding-math -ffp-contract=off` into `verifier/build/cev` (`cev.exe`
on Windows); the compiler is taken from `--cc`, then `CEV_CC`, then `gcc`, `clang` or `cc` on `PATH`. The verifier
needs a certificate written with `Options(store_eigenvectors=True)`, so that the sidecar carries the claimed vectors;
it prints one line per check and a verdict, and exits with 0 when every check passes and with 1 when one fails.
`run_validation.py` builds the verifier, produces the validation certificates with `verifier/make_validation_set.py`
from `corpus/cases` and `conformance/reference`, runs the verifier on each, makes four kinds of forged certificates
and writes `RESULTS.md` and `results.json` next to itself.

On 10 November 2026 the validation took about four minutes and printed a summary line with 55 certificates checked
(pass 55, fail 0, error 0), 18 FAILED statuses of which 18 reproduced, and 20 of 20 forgeries caught in each of the
four kinds (`radius`, `array_hash`, `body_hash`, `status`). Its `RESULTS.md` (63 certificates produced, 8 refused by
certify-ed) has the same content as the committed one, and `results.json` differs from the committed one only by a
final newline; both are tracked files, so the run leaves them marked as modified in the checkout. The verifier run
alone on `verifier/validation/certs/heis10_correct.json` with the operator of `corpus/cases/heis10_correct` printed
`verdict: PASS` with no failed check. The validation and its limits are described in `docs/project/software/09_verifier.md`.

## The derivations of docs/proofs

```
python docs\proofs\verify_<id>.py [--cases N] [--seed S]
python docs\proofs\independent\adversarial_<id>.py
```

Each `verify_<id>.py` tests the statement of `docs/proofs/<id>.md` against ground truth (mpmath at high precision or
exact rationals), prints the number of random and adversarial cases, the number skipped because the hypothesis was not
met and the number of violations, and exits with 1 when a violation is found. The logs of the long runs are in
`docs/proofs/runs/` (ten files); the first line of `appA_model-shift.txt`, for example, records `--cases 400 --seed 5`
and 235 minutes of wall time. The scripts of `docs/proofs/independent/` belong to the second derivations, each written
before the original proof was read, and search for counterexamples at the boundaries of the hypotheses; their verdicts are
collected in `docs/proofs/VERDICTS.md`. These are machine checks. The status of every derivation is: derived,
machine-checked, independently re-derived (a second derivation written before the original proof was read, not a human check),
Sarang's check pending (decision D-12).

As a check that the scripts run from a fresh clone, `python docs\proofs\verify_thm8_rho.py --cases 10 --seed 1` was
run; it took about one minute and reported 0 violations in each of its three parts.

## The planted-error corpus

```
python corpus\make_corpus.py [--out corpus\cases] [--models heis8,j1j2_8,tfim8,heis10]
python corpus\run_corpus.py [--results <dir>]
python corpus\run_corpus.py --checker "mychecker {case} --out {out}" --name mychecker
```

`make_corpus.py` regenerates the 60 cases deterministically, and `cases/index.json` holds the SHA-256 of every array,
so that a regenerated corpus can be compared with the committed one. `run_corpus.py` runs a checker on every case and
scores it; the default checker is certify-ed of the checkout (`{python} -m certify_ed.cli certify --from {case} -o
{out}`), and `--checker` takes the command of any other checker as a template. The committed report
`corpus/results/certify-ed/report.md`, of 29 October 2026, reads "detected 53 / missed 0 (unsound 0, silent 0,
error 0) / false alarms 0 (weak 0) of 53 planted cases, 7 controls (clean 7)". On 10 November 2026 the default checker was
run on the committed cases with `--results` pointing outside the checkout; it took about 13 minutes and printed the
same summary line, and the verdict and the five statuses of every one of the 60 cases equal those of the committed
report.

## Conformance bundles and schemas

```
python conformance\make_reference_bundles.py
python conformance\check_bundle.py conformance\reference\dense-arrays conformance\reference\npz conformance\reference\matrix-free
python -m certify_ed.schema --check schema
python -m certify_ed.schema --write schema
```

`make_reference_bundles.py` writes the three reference producer bundles (the S=1/2 Heisenberg chain on 6 sites in
three formats). `check_bundle.py` is the same as `certify-ed conform` and checks the layout, the metadata schema, the
import, the certification, the certificate schema and the expected statuses of each bundle. On 10 November 2026 each
of the three reference bundles was reported `CONFORMANT`. `--check schema` compares the files of `schema/` with the
definitions in `certify_ed/schema.py` and reported `in sync` for `certificate-2.1.schema.json`,
`producer-bundle-1.0.schema.json` and `statement-record-1.0.schema.json`; `--write` renders them again.

## Examples

```
python examples\01_certify_quspin.py
python examples\02_exact_recheck.py
python examples\03_whole_spectrum.py
python examples\04_applications.py
```

The first needs QuSpin (`01_environment.md`); the fourth runs the two applications on their quick instances, and its
docstring notes that the S = 1/2 crown certificate takes about 13 minutes through
`certify_ed.apps.pyrochlore.spin_half.certify_crown`. The examples are described in `docs/project/software/04_api.md`. They
were not run again for this documentation.
