# Presets, statement selection and budgets (the ffmpeg `-preset` model)

The default run accepts whatever the producer handed over and runs every statement whose inputs exist
(S1-S4 and S8; S5, S6 and S7 switch on through `corroborate`, `exact_recheck` and `temperatures`). The
user narrows with a preset, `--only`, `--skip` and budgets. Narrowing is recorded, never silent: a
statement that does not run is NOT_CHECKED with a reason code, and the certificate's `scope` block lists
the preset, the statements in scope, the narrowed statements with the user's reason, and the quantities
out of scope (`docs/SCOPE_MATRIX.md`).

```
certify-ed certify --from RUN --preset benchmark
certify-ed certify --from RUN --only S1,S2 --reason "vectors not stored for the excited states"
certify-ed certify --from RUN --skip S3 --budget-seconds 600 --budget-mb 4000
certify-ed presets                       # this table
```

In Python: `Options.from_preset("benchmark", budget_seconds=600)`, or `Options(only=["S1", "S2"])`,
`Options(skip=["S3"], narrowing_reason="...")`.

## Presets (data: `certify_ed.presets.PRESETS`)

| preset | statements | options set | use |
|---|---|---|---|
| `lab-minimal` | S1 S2 | inertia=False, eigenspaces='none' | is the matrix the declared model, and are the claimed pairs eigenpairs? (no factorization) |
| `benchmark` | S1 S2 S3 S5 | corroborate=('jacobi', 'arpack'), eigenspaces='none' | a reference value for other methods: enclosures, completeness and a gap, plus corroboration |
| `proof-lemma` | S1 S2 S3 S6 | exact_recheck=True, eigenspaces='none' | a number that enters a proof: S1-S3, then the exact integer re-check with no floating-point assumption |
| `whole-spectrum` | S1 S2 S3 S7 | eigenspaces='none', whole_spectrum=True | every eigenvalue enclosed and counted (Prop 9); thermodynamic intervals when temperatures are given |
| `labels` | S1 S2 S3 S4 S8 | eigenspaces='all' | quantum numbers, degeneracies and observables of every claimed level |
| `full` | S1 S2 S3 S4 S5 S6 S7 S8 | corroborate=('jacobi', 'arpack'), exact_recheck=True, eigenspaces='all' | everything; S5-S7 run only when their inputs exist (corroboration solvers, python-flint, a whole spectrum) |

| statement | needs | inputs | what it says |
|---|---|---|---|
| S1 `S1.model_fidelity` | - | spec | producer operator equals the declared model (entrywise, sector audit, or probes) |
| S2 `S2.eigenpairs` | - | eigenvectors | each claimed value within a certified radius of a distinct eigenvalue (Thm 1-3) |
| S3 `S3.completeness` | S2 | eigenvectors, matrix, completeness claim | no eigenvalue missing in the claimed window; gap to the next (Lemma 4, Cor 5, Prop 9) |
| S4 `S4.eigenspaces` | S2 S3 | eigenvectors | projector error, observables, correlations, structure factors, quantum numbers (Thm 6, Prop 7, Prop 15) |
| S5 `S5.corroboration` | - | matrix | recomputation by algorithmically distinct solvers (evidence, not a certified statement) |
| S6 `S6.exact_recheck` | S2 | spec, eigenvectors, python-flint | S2 radii and the S3 count re-derived in integer arithmetic (Lemmas 10-12, Prop 13) |
| S7 `S7.thermodynamics` | S2 S3 | whole spectrum, temperatures | interval bounds on Z, F, U, C from a complete enclosure (Prop 16; UNVERIFIED) |
| S8 `S8.degeneracy` | S2 | eigenvectors, matrix | number of eigenvalues in each cluster by bracketing inertia counts (Prop 14; UNVERIFIED) |

## Rules

* `--only A,B` runs A and B and whatever they need (S4 needs S2 and S3; S3, S6 and S8 need S2; S7 needs
  S2 and S3). Implied statements are listed as such in `scope.selection.implied`.
* `--skip X` removes X; statements that need X are narrowed with `by: dependency`.
* A preset's option values apply first; keyword arguments / command-line flags override them.
* **Budgets never raise.** `--budget-seconds S`: a statement that has not started S seconds after
  `certify()` began is NOT_CHECKED with reason `BUDGET_TIME` (a running statement is not interrupted; the
  check happens between statements). `--budget-mb M`: a statement whose memory estimate exceeds M is
  NOT_CHECKED with reason `BUDGET_MEMORY`. The estimates are coarse and declared in the statement:
  S1 reference build `dim x terms x 2 entries x 32 B`, S3/S8 dense factorization `2 n^2` entries (only when
  the dense path would be used), S6 `48 n^2` bytes. A factorization that fails for memory at run time is
  also NOT_CHECKED (`resource_failure`), not an exception.
* **Narrowing does not change what is said about a statement still in scope.** The same claim with
  `--skip S3` gives the same S1 and S2 records (tested in `tests/test_presets_scope.py`).
* The bundle's `meta.json` may suggest `preset`, `only`, `skip`, `budget_seconds`, `budget_mb`,
  `temperatures`; the command line wins.

## Reason codes

Used in `statements[].evidence.reason_code` and in `scope.out_of_scope[].reason_code`
(`certify_ed.scope.REASON_CODES`):

| code | meaning |
|---|---|
| `NARROWED_BY_USER` | the user excluded the statement (--only / --skip / preset) |
| `DEFAULT_SELECTION` | not part of the default selection; switched on by its option, a preset or --only |
| `DEPENDENCY_NARROWED` | the statement needs another statement that the user excluded |
| `BUDGET_TIME` | the time budget was spent before the statement could start |
| `BUDGET_MEMORY` | the statement's estimated memory exceeds the memory budget |
| `NEEDS_EIGENVECTORS` | the claim has eigenvalues only |
| `NEEDS_SPEC` | no model specification was declared |
| `NEEDS_MATRIX` | the operator is matrix-free and no stored reference matrix exists |
| `NEEDS_WHOLE_SPECTRUM` | the claim is not a complete whole-spectrum claim (kind 'all', S3 certified) |
| `NEEDS_TEMPERATURES` | no temperatures were given |
| `NEEDS_DEPENDENCY` | an optional dependency (python-flint, h5py, a corroborating solver) is missing |
| `NEEDS_OBSERVABLES` | no observables, correlations or structure factors were requested |
| `NO_COMPLETENESS_CLAIM` | the claim is a 'subset' claim: nothing is said about other eigenvalues |
| `NO_CLUSTERS` | no (quasi-)degenerate clusters among the claimed pairs |
| `SIZE_LIMIT` | dimension above the configured limit for this statement (inertia_max_dim, exact_max_dim, ...) |
| `UNSUPPORTED_CLAIM` | the statement is not implemented for this claim kind or sector form |
| `UNSUPPORTED_MODEL` | the model class is outside what the statement handles |
| `NOT_IMPLEMENTED` | a design slot: the mathematics is identified but not implemented |
| `OUT_OF_SCOPE_DYNAMICS` | dynamical response and time evolution are not statements about eigenpairs of one matrix |
| `OUT_OF_SCOPE_EXTRAPOLATION` | finite-size extrapolation and phase boundaries are not statements about any finite matrix |
| `NOT_APPLICABLE` | the statement does not apply to this input |

## The scope block

```
"scope": {
  "schema": "certify-ed/scope/1.0",
  "preset": "benchmark" | null,
  "selection": {"wanted": [...], "implied": {...}, "narrowed": {...}, "only": ..., "skip": ..., "narrowing_reason": ...},
  "budgets": {"seconds": ..., "mb": ..., "events": [...]},
  "in_scope":     [{"id": "S2.eigenpairs", "status": "CERTIFIED", "implied": null}, ...],
  "narrowed":     [{"id": "S3.completeness", "by": "skip", "reason": "...", "reason_code": "NARROWED_BY_USER", "status": "NOT_CHECKED"}, ...],
  "out_of_scope": [{"quantity": "S7.thermodynamics", "reason_code": "NEEDS_WHOLE_SPECTRUM", "detail": "..."},
                   {"quantity": "entanglement entropy and entanglement spectrum", "reason_code": "NOT_IMPLEMENTED", "detail": "..."}, ...],
  "requested": {"kind": ..., "has_eigenvectors": ..., "has_spec": ..., "has_matrix": ...},
  "options": {...}, "operator_scope": "operator" | "sector block: ...", "reason_codes": {...}
}
```

Every statement id appears exactly once across `in_scope`, `narrowed` and `out_of_scope`; the standing
out-of-scope quantities (entanglement, dynamics, time evolution, level statistics, extrapolation, phase
boundaries) are listed in every certificate with their reason codes so that a reader never has to guess
what the certificate is silent about.
