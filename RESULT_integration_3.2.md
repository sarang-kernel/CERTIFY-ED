# Integration of roadmap Waves 0 and 1 into `dev/3.2` (2026-10-29)

Record of the merge of seven tracks (plan of 2026-10-23, D-58). Branch `dev/3.2`, certify-ed 3.2.0.dev0.

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

- `pytest -q tests` at 73e5c2e: **953 passed, 2 skipped** (1412 s); PYTHONSAFEPATH removed from the environment of the run, as for every run on this host (the default of the development environment breaks the experiment driver test independently of the code).
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

`experiments/e6_mutation.py 16 2` at c59131e (baseline timeout raised to 3600 s; the baseline suite took 1195 s): **13 of 16 killed** (3.1.0's 64-mutant sample: 38 of 64). Record: `experiments/results/e6_mutation_20261029_164910/`. Survivors, with suggested readings (automatic, not verdicts, RA-25):

| mutant | suggestion |
|---|---|
| `certify.py:1236` `pos > 0` → `pos > 1` | the second-lowest claimed pair loses its lower neighbour in the Kato-Temple isolation window; the lowest pair has none anyway. Possible test gap: a test where the second pair's isolation interval decides a Kato-Temple sharpening. |
| `sectors.py:240` `defects[k] + rnd` → `defects[k] - rnd` | **tightens a bound** (the rounding allowance of the sector defect is subtracted instead of added). No test resolves a sector-audit radius short by `rnd`; a soundness-relevant gap by the project's own rule (rounding allowances are sound when looser, undetectable when tighter unless a test resolves them). Add a test with a planted defect within `rnd` of the threshold. |
| `checker.py:261` `2**53` → `2**54` in the mantissa scaling of the exact-rational conversion | with the exponent left at `e − 53`, every converted value doubles; the tests pass, so either the downstream common-exponent normalisation absorbs a common factor (then the mutant is equivalent for ratios but not for absolute values) or no test checks an absolute exact value through this path. Needs a careful read before a verdict. |


## Completion plan Phase 1 integration (2026-10-30)

Merged into `dev/3.2` on 38b0212: track/new-statements (ec8163a), track/figures (9c8b043), track/backlog-proofs (dffa41b), track/props-proofs (f7a5983). Conflicts resolved in the merge commits: CHANGELOG.md (additive, four times), certify_ed/enclosure.py `_cluster_bound` (Backlog's min(rho, eps*) with INT-05 guard, plus the Davis-Kahan ingredient keys of New statements), certify_ed/correlations.py (fp import union), docs/SCOPE_MATRIX.md (degeneracies/thermodynamics rows from Props proofs, labels/entanglement rows from New statements, correlation row with the P15 derivation reference). docs/api.md regenerated; schema in sync.

Full suite at d5ddf17: 1047 passed, 8 skipped, 5 warnings in 1992.79s (0:33:12). ruff clean.

Open from the tracks (carried into Phase 2/3): Appendix A verification script `docs/proofs/verify_appA_model-shift.py` written but not yet run; C-16 sector path at the API only; `exact_bracket` capped at dimension 600; E6 survivors of the 3.1.0 run still need Sarang's verdicts; fig_scope_vs_survey and fig_rounding_accounting to be regenerated from the final tree (Phase 3).


## Completion plan Phase 2 integration (2026-11-03)

Merged into `dev/3.2` on ac52920: track/variational (fda4d74; E-05, E-03, C-07 first part), track/verifier (be4456a; C verifier `verifier/src/cev.c`, builds with `CEV_CC` = MinGW gcc 14.2 from GNU Octave 10.3 on this host; 7 tests pass, skipped without a C compiler), track/scale (4150a79; D-01 a-posteriori sparse residual, D-02 matrix-free S2), track/independent-check (8b598d7; `docs/proofs/VERDICTS.md`, 17 documents: 5 AGREES, 7 AGREES-WITH-REMARKS, 3 GAP (thm1, C-04, E-07: proof text), 2 ERROR corrected by looser code with negative controls (int05 `norm2_lower` overflow; E-04 crossing slack at interior grid points)). Conflicts resolved in the merge commits: CHANGELOG.md (additive), docs/SCOPE_MATRIX.md (producer-formats row; the D-02 row now says S1 and S2 at 2^20-2^24, the only statements the scale run timed). Lead commit 2e1dc59: thm1 correction implemented as the looser option (`gamma(width + 3)` for complex data in `enclosure._rounding_bound`).

Full suite at e69bca5 (+2e1dc59 for subprocess tests): 1161 passed, 8 skipped, 8 warnings in 6404 s (1:46:44, run concurrently with the Appendix A verification script). ruff clean (certify_ed tests experiments conformance corpus docs/proofs verifier). Schema in sync (certificate-2.1, producer-bundle-1.0, statement-record-1.0).

Open: independent re-derivation of the five Phase 2 documents (E-05, E-03, C-07, D-01, D-02) in progress on track/independent-check-2; `verify_appA_model-shift.py` running (user approved 2026-11-03); E6 survivors of the 3.1.0 run (Sarang); figures fig_scope_vs_survey / fig_rounding_accounting to regenerate from the final tree.


## 3.2.0 release sample (2026-11-04)

Full suite at e1ebaed (release tree, idle machine, `--basetemp` in the workspace): **1168 passed, 8 skipped, 8 warnings in 2582.80 s (43:02)**.

`experiments/e6_mutation.py 16 2` at e1ebaed (record `experiments/results/e6_mutation_20261104_043326/`; baseline 2355 s): **10 of 16 killed** (3.2.0.dev0 sample: 13 of 16; 3.1.0's 64-mutant sample: 38 of 64). An earlier attempt at b832d8e aborted because the baseline failed: `tests/test_whole_spectrum.py::test_enclosure_radii_bit_identical_complex_operator` compared complex-data radii bit for bit against the frozen oracle after the gamma_{w+3} change (fixed in e1ebaed: the complex comparison is >= and within 1 + 1e-12; real data stay bit-identical). The full suite of 2026-11-03 (1161 passed) had been started before that change was imported and did not see it; the release suite above does.

Survivors, with suggested readings (automatic, not verdicts, RA-25):

| mutant | reading |
|---|---|
| `construct.py:220` constant `0` → `1` in `asym.nnz == 0` of `hermitian_part`'s early return | equivalent: for an exactly Hermitian H the second clause (`abs(asym).max() == 0.0`) still returns H, for a stored explicit zero the mutant's `nnz == 1` returns H as the original did, and otherwise both paths symmetrise; no entry changes |
| `certify.py:1039` `bad_x[:8]` → `[:9]` | message text only (how many offending column indices are listed) |
| `exact_sectors.py:213` string `+` → `-` in the weights-violation reason | reachable error path never exercised: a generator that does not preserve the scaling weights would raise instead of returning (False, reason). Test added: `test_check_commutes_reports_a_generator_that_does_not_preserve_the_weights` (fails on the mutant) |
| `sectors.py:179` dropped `not` in `restrict_to_u1` | outcome-equivalent: for a single-sector support the mutant returns None (the caller falls back to the full-space operator, slower but the same statements); for a multi-sector support it proceeds and `st.index_of` reports missing states, returning None at line 194 |
| `checker.py:521` `>` → `>=` in `top_excess` | heuristic: triggers the extra witness point also when the count equals the claim; FAILED still needs `w["count"] > w["claimed_below"]` at line 536 (unmutated), so no status can change wrongly |
| `eigenspace.py:100` `and` → `or` in the `sin_theta_alt` guard | reachable robustness path: `certify_eigenspace(..., sin_theta_alt=None)` would raise. Test added: `test_certify_eigenspace_without_an_alternative_bound_uses_theorem_6_only` (fails on the mutant) |

Killed: localops.py:193, certify.py:1142, certify.py:1155, certify.py:1213, fp.py:333, localops.py:41, fp.py:239, spec.py:247, localops.py:186, enclosure.py:624 (first failing tests in the run log). The 64-mutant sample belongs to the 3.2.0 evidence run (`run.py --full --with-mutation`).

## Evidence runs (2026-11-04 to 2026-11-06)

The 3.2.0 run `results/run_full_20261104` failed its E4 step (native crash of the eight-thread residual under the memory
cap); 3.2.1 (tag f3c478b, one residual thread by default) has the complete run `results/run_full_20261105`: tests 1172
passed, E0-E10 ok, E6 36 of 64 killed, numbers 44/44 and 61/61; against 3.1.0 10 MISMATCH leaves, all documented changes;
E9 and E10 identical. Details: `RESULT_evidence_run_3.2.1.md`; decision D-61.
