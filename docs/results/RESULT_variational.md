# RESULT: track/variational (E-05, E-03, C-07 first part)

Branch `track/variational` on `dev/3.2` (ac52920). Head: the commit that adds this file (five commits: the four below and this record). Four content commits: c0aa14b (E-05), 8319055 (E-03),
f3c5f1a (C-07 code checkpoint), d383d07 (C-07 derivation and docs).

## Item status

| item | status | documents | verification (actually obtained) | tests | negative controls |
|---|---|---|---|---|---|
| E-05 variational certificates | done | `docs/proofs/E-05_variational.md`, `verify_E-05.py`, run log `docs/proofs/runs/verify_E-05.txt` | 666 random + 294 adversarial cases, 0 violations (`--cases 200`) | `tests/test_variational.py` (12) | false declared gap ρ = λ₃ > λ₂: bound broken 66/66 while the record stays CERTIFIED_UNDER_HYPOTHESIS and names `declared_gap`; trial state orthogonal to the ground state (INCONCLUSIVE); mismatched certificate; tiny norms (2⁻⁴⁵⁰ valid, 2⁻⁶⁰⁰ NOT_CHECKED, 2⁻⁹⁵⁰ UNDERFLOW_RISK); non-Hermitian operator refused; duplicate CI indices; MPS size limit |
| E-03 generalized eigenproblems | done | `docs/proofs/E-03_generalized.md`, `verify_E-03.py`, run log `docs/proofs/runs/verify_E-03.txt` | 303 random + 100 adversarial cases, 0 violations (`--cases 80`; mpmath Cholesky / L⁻¹AL⁻ᴴ ground truth) | `tests/test_generalized.py` (8) | B indefinite refused (UNSUPPORTED_MODEL, nothing certified); wrong eigenvalue (radius valid, G3 not certified); swapped pairs (G3 INCONCLUSIVE); missing eigenvalue (`missing_at_least`); near-singular B; false declared λ_min(B) with the inertia disabled: radii undercut the exact B⁻¹-norm residual 40/40 while named `declared_lambda_min_B` |
| C-07 moments and sum rules (first part) | done (first part); the continued-fraction part is documented as open, not implemented | `docs/proofs/C-07_moments.md`, `verify_C-07.py`, run log `docs/proofs/runs/verify_C-07.txt` | 418 random + 192 adversarial cases, 0 violations (`--cases 40`) | `tests/test_moments.py` (6) | s understated 10⁴× on tilted vectors: enclosure broken 132/138 moments; random orthonormal X with s = η = 0 claimed: 138/138; wrong level interval (shifted beyond the certified widths): M3 FAILED 40/40 |

## What was built

**E-05** (`certify_ed/variational.py`, CLI `certify-ed variational`): V1 Rayleigh upper bound λ₁ ≤ θ (unconditional;
`enclosure.rayleigh_interval` for the enclosure of the rounded quotient; model shift σ of a spec added), V2 Theorem 1
existence radius at θ (+σ), V3 Temple's lower bound λ₁ ≥ θ − r²/(ρ − θ) for θ < ρ ≤ λ₂ with ρ from a certificate of a
'lowest' claim on the same operator (CERTIFIED; `gap_from_certificate`: second smallest lower endpoint of the S2
enclosures for k ≥ 2, S3's `next_eigenvalue_lower_bound` for k = 1) or from a declared value (CERTIFIED_UNDER_HYPOTHESIS,
hypothesis `declared_gap` with the producer's source text, repeated in the statement text), V4 producer-declared
stochastic estimates (DECLARED, reason code DECLARED_ESTIMATE, nothing rests on them). Adapters: dense vector; MPS
(`state_from_mps`, left-to-right contraction ≤ 2²² with the entrywise bound γ_M|A⁽¹⁾|⋯|A⁽ᴸ⁾| recorded as
`contraction_error`, and Lemma 5 for the Rayleigh quotient of the source state; the certified statements are about the
contracted vector); sparse CI (`state_from_sparse_ci`). Matrix-free and non-Hermitian operators are refused.

**Representation of "certified under hypothesis"** (asked for in the plan): the three modules write standalone JSON
records (not certificates) validated by the new `schema/statement-record-1.0.schema.json` (`schema.STATEMENT_RECORD_SCHEMA`,
`schema.validate_record`). Their status vocabulary is CERTIFIED, CERTIFIED_UNDER_HYPOTHESIS, DECLARED, INCONCLUSIVE,
NOT_CHECKED, FAILED; every statement carries a `hypotheses` list, the schema requires it to be non-empty for
CERTIFIED_UNDER_HYPOTHESIS, `validate_record` re-checks that and the record hash, and the statement text repeats
"UNDER THE DECLARED HYPOTHESIS … source: …". The certificate schema 2.1 and the viewer are untouched (no certificate
status was added, so no existing reader can mistake a hypothesis for a certification). `diagnostics.record_next_action`
/ `annotate_record` give every non-certified record statement a `next_action`.

**E-03** (`certify_ed/generalized.py`, dense path n ≤ 4000): G1 certified lower bound β_c on λ_min(B) by the inertia of
B − βI (Bunch–Kaufman LDLᴴ with its backward error; B with a negative inertia at 0 is refused), G2 per pair
|λ − μ| ≤ ‖Ax − μBx‖₂/(√β_c ‖x‖_B) (Theorem 1 for L⁻¹AL⁻ᴴ, ‖L⁻¹‖₂² = 1/λ_min(B); residual and xᴴBx with the standard-model
accounting), G3 completeness of a 'lowest' claim by the inertia of A − σB (Sylvester's law for the congruence, collar
b/β_c including the rounding of forming A − σB) with λ_{k+1} ≥ σ − b/β_c. A declared λ_min(B) (used only when the
inertia is inconclusive) is the named hypothesis `declared_lambda_min_B`. Clusters (overlapping enclosures) are
INCONCLUSIVE with NOT_IMPLEMENTED; 'window'/'all' pencils are not implemented.

**C-07 first part** (`certify_ed/moments.py`): M1 certified m_k = tr(P O Hᵏ O)/g, k = 0..K, from the stored vectors of a
CERTIFIED S4 cluster (O once and H k times applied to the vectors, Hᵏ never formed; Theorem 1 style rounding per
application; projector term (2gs + gη)‖O‖²‖H‖ᵏ from the trace-norm bound), M2 the zeroth-moment sum rule and M3 the
f-sum rule, each certified by a second route (Theorem 6 on the formed O² and [O,[H,O]] with their product rounding
bounds) and reported CERTIFIED / FAILED by overlap of the two intervals; M4 continued-fraction bounds NOT_IMPLEMENTED
(`docs/theory.md` §9.4 names the mathematics; scope-matrix row).

## Files touched

- `CHANGELOG.md`
- `certify_ed/cli.py`
- `certify_ed/diagnostics.py`
- `certify_ed/generalized.py`
- `certify_ed/moments.py`
- `certify_ed/schema.py`
- `certify_ed/scope.py`
- `certify_ed/variational.py`
- `docs/SCOPE_MATRIX.md`
- `docs/proofs/C-07_moments.md`
- `docs/proofs/E-03_generalized.md`
- `docs/proofs/E-05_variational.md`
- `docs/proofs/runs/verify_C-07.txt`
- `docs/proofs/runs/verify_E-03.txt`
- `docs/proofs/runs/verify_E-05.txt`
- `docs/proofs/verify_C-07.py`
- `docs/proofs/verify_E-03.py`
- `docs/proofs/verify_E-05.py`
- `docs/theory.md`
- `schema/certificate-2.1.schema.json`
- `schema/statement-record-1.0.schema.json`
- `tests/test_generalized.py`
- `tests/test_moments.py`
- `tests/test_variational.py`

## Tests and lint

Targeted suites: `tests/test_variational.py` 12 passed, `tests/test_generalized.py` 8 passed, `tests/test_moments.py` 6
passed; `tests/test_schema_conformance.py`, `tests/test_diagnostics.py`, `tests/test_presets_scope.py` 128 passed after
the schema and diagnostics changes. Full suite (`pytest -q` on the branch head): 1073 passed, 8 skipped, 5 warnings in 7735.87s (2:08:55) (host CPU saturated by external processes during the run).
`ruff check certify_ed tests experiments conformance corpus docs/proofs`: All checks passed!.

## Not done, and why

- C-07 second part (continued-fraction / Lanczos bounds on the spectral function): out of the track's scope by design;
  documented as open (M4 NOT_IMPLEMENTED, `docs/theory.md` §9.4, scope matrix).
- E-03: no cluster bound (Theorem 2 in the B-inner product), no 'window'/'all' pencils, no sparse LDLᴴ path; each refused
  with a reason code, listed in the scope-matrix row.
- E-05: an MPS beyond 2²² is SIZE_LIMIT (a residual bound by MPO contraction is not implemented); stochastic estimates are
  recorded, never bounded (no certified sampling-error statement).
- No claim kind "variational" was added to `certify.Claim`: the variational statements live in a standalone record
  (dedicated function and CLI command), which keeps `certify.py`, `schema.py`'s certificate schema and the viewer
  unchanged and avoids the conflict surface named in the plan.
- The verification scripts were run at `--cases 200` (E-05), `80` (E-03) and `40` (C-07); the documents and this file
  record exactly those counts.

## Expected merge conflicts with other tracks

- `CHANGELOG.md`: one new subsection "Variational and dynamics track" inserted before "New statements track" (additive;
  textual conflict likely with any track that inserts at the same place; keep both).
- `docs/SCOPE_MATRIX.md`: three new rows inserted before the "producer formats" row and one reason-code row after
  `UNDERFLOW_RISK` (additive).
- `docs/theory.md`: new §9.4 (before §10) and §10.5–10.7 appended (additive; a track that also appends §10.x will
  conflict on the numbering — renumber).
- `certify_ed/schema.py`: `RECORD_SCHEMAS`, `RECORD_STATUSES`, `_record_schema`, `STATEMENT_RECORD_SCHEMA`, `SCHEMA_FILES`
  (one new entry), `validate_record` (additive; the certificate schema is unchanged); `schema/statement-record-1.0.schema.json`
  is new; regenerate with `python -m certify_ed.schema --write schema` after merging.
- `certify_ed/scope.py`: one reason code `DECLARED_ESTIMATE` (additive). `certify_ed/diagnostics.py`: one entry in
  `NEXT_ACTION_BY_CODE` and two new functions at the end (additive).
- `certify_ed/cli.py`: `cmd_variational` and its parser block (inserted before `levelstats`; additive).
- `certify_ed/certify.py`: untouched. `certify_ed/variational.py`, `generalized.py`, `moments.py`: new files.
