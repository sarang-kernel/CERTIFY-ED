# Integration of roadmap Waves 0 and 1 into `dev/3.2` (2026-10-29)

Lead's record of the merge of seven tracks (plan of 2026-10-23, D-58). Branch `dev/3.2`, certify-ed 3.2.0.dev0.

## Merge order and resolutions

| step | branch | head | result |
|---|---|---|---|
| base | `dev/3.2-known-issues` (fix/3.1.1 merged, INT-01 check, INT-04 note, docs site) | c5d2560 | start of `dev/3.2` |
| 1 | `track/core-api` | 3771f59 | fb85f29; two additive conflicts in certify.py (Options fields; underflow record before next-action annotation); `UNDERFLOW_RISK` given a next_action |
| 2 | `track/format-packaging` | 8b718a4 | f080586; additive conflicts (provenance, docs, CHANGELOG consolidated) |
| 3 | integration fix | — | 68e26d0: schema re-rendered with the merged reason codes; `Certificate.load` validates the schema before the hashes; two tests adjusted to the two validators guarding the same tampering |
| 4 | `track/producers` | 306d27b | clean merge |
| 5 | `track/statements` | 1f0240c | 986f8ae; additive conflicts (provenance, make_api, api.md regenerated) |
| 6 | `track/whole-spectrum` | 80819df (re-written to LF as `track/whole-spectrum-lf`, content identical under `--ignore-cr-at-eol`) | cd874f8; one real interaction: the observable loop applies C-05's Hermitian decomposition to C-15's parsed operators; `gaps` and `plan` records both kept; an inverted window is now refused at Claim construction (C-14), one C-02 test adjusted |
| 7 | `track/viewer-corpus` | 40d93a3 | 73e5c2e; additive conflicts in cli.py (new subcommands) and api.md |
| 8 | CHANGELOG integration note | — | d932bfd |

## Checks on the merged branch

- `pytest -q tests` at 73e5c2e: **953 passed, 2 skipped** (1412 s); PYTHONSAFEPATH removed from the environment of the run, as for every run on this host (the sandbox's default breaks the experiment driver test independently of the code).
- Before the viewer merge (cd874f8): 862 passed, 2 skipped. Base `release/3.1.0`: 541 passed, 1 skipped.
- `ruff check certify_ed tests experiments conformance corpus`: clean. `python -m certify_ed.schema --check schema`: in sync.
- CLI: env, certify, import, plan, statement, crossing, levelstats, extrapolate, view, presets, conform, spectrum, list, test, reproduce, numbers.
- Mutation sample (16 mutants, 2 workers): 13 of 16 killed; survivors at the end of this file. A full 64-mutant sample belongs to the next evidence run (each mutant runs the 23-minute suite).
- Private-address scan of the tree at `dev/3.2`: none. Authors and committers of all 34 new commits: the paper address; no trailers.

## Item status

Roadmap section 7 (`ROADMAP_expansion.md`) and the CSV: 30 done, 3 partial (B-03: run pending; B-09: no package hash in the certificate, no SBOM; F-02: SPINPACK and DiagHam complex-vector layouts assumed), G-01 blocked (GitHub token).

## Caveats carried from the tracks

CI workflows and container recipes syntax-checked only; wheel installed offline; cost model and planner calibrated on three Heisenberg-chain blocks of this machine; OpenFermion/Qiskit tested through interface-mimicking objects; corpus expectation for `unnormalised_cluster_member` revised from [S4] to [S2, S4] after a first run (49/53 → 53/53; both runs committed; certifier unchanged); HTML viewer checked as text and by tests; new known issue INT-05 (subnormal intermediate residual entries) awaits a design decision.

## Mutation sample on the merged branch (2026-10-29)

`experiments/e6_mutation.py 16 2` at c59131e (baseline timeout raised to 3600 s; the baseline suite took 1195 s): **13 of 16 killed** (3.1.0's 64-mutant sample: 38 of 64). Record: `experiments/results/e6_mutation_20261029_164910/`. Survivors, with the lead's suggestions (not verdicts, RA-25):

| mutant | suggestion |
|---|---|
| `certify.py:1236` `pos > 0` → `pos > 1` | the second-lowest claimed pair loses its lower neighbour in the Kato-Temple isolation window; the lowest pair has none anyway. Possible test gap: a test where the second pair's isolation interval decides a Kato-Temple sharpening. |
| `sectors.py:240` `defects[k] + rnd` → `defects[k] - rnd` | **tightens a bound** (the rounding allowance of the sector defect is subtracted instead of added). No test resolves a sector-audit radius short by `rnd`; a soundness-relevant gap by the project's own rule (rounding allowances are sound when looser, undetectable when tighter unless a test resolves them). Add a test with a planted defect within `rnd` of the threshold. |
| `checker.py:261` `2**53` → `2**54` in the mantissa scaling of the exact-rational conversion | with the exponent left at `e − 53`, every converted value doubles; the tests pass, so either the downstream common-exponent normalisation absorbs a common factor (then the mutant is equivalent for ratios but not for absolute values) or no test checks an absolute exact value through this path. Needs a careful read before a verdict. |
