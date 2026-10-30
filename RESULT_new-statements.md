# Track: new statements (C-03, C-04, E-04, C-12, C-13, E-07) — result (2026-10-30)

Branch `track/new-statements` from `dev/3.2` (38b0212). Decision D-59: every statement derived here, written to
`docs/proofs/<id>_<name>.md` (sections Statement / Proof / Where the code implements it / Computational
verification / Analytic cross-check / Status) with its verification script `docs/proofs/verify_<id>.py`.
Independent re-derivation (Phase 2) and Sarang's check (D-12) are pending for all four documents. No existing
bound became tighter except where a new bound is proven and combined as min(old, new) (E-07; both recorded).

## Item status

| item | status | proof document | verification (seed 20261030) | module / hook |
|---|---|---|---|---|
| C-03 entanglement entropy and spectrum | done | `docs/proofs/C-03_entanglement.md` | `verify_C-03.py`: 1 150 random + 20 026 adversarial cases, 0 violations | `certify_ed/entanglement.py`; `Options.entanglement`; one block in the S4 loop |
| C-04 correlations for symmetry-adapted claims | done | `docs/proofs/C-04_correlation_lift.md` | `verify_C-04.py`: 240 random + 14 adversarial, 0 violations | `correlations.compress_operator / certify_observable_lifted / certify_correlations_lifted`; `_correlations(..., audit=)` and the observable loop in `certify.py` |
| E-04 certified parameter intervals | done | `docs/proofs/E-04_parameter_intervals.md` | `verify_E-04.py`: 1 049 random + 107 adversarial, 0 violations; tubes contain dense eigenvalues at 100 % of 300 random couplings; wrong B detected 50/50; wrong H1 flagged | `certify_ed/sweep.py` (`certified_sweep`, `tube_at`, `certified_crossing_interval`); CLI `certify-ed sweep` |
| C-12 / C-13 / E-07 Davis–Kahan | done | `docs/proofs/E-07_davis_kahan.md` (one document, three propositions) | `verify_E-07.py`: 529 random + 24 adversarial, 0 violations; negative controls 20/20 and 20/20 | `certify_ed/davis_kahan.py`; `eigenspace.certify_eigenspace(sin_theta_alt=)`; `enclosure._cluster_bound` exports its ingredients; S4 hook (dist, s_dk, labels_hook); `Options.approximate_symmetries` |

## What each statement says (short)

* **C-03.** For a certified cluster (Theorem 6: ‖Q̂Q̂ᴴ − P‖ ≤ s, η) and a declared cut A|B, ρ_A = tr_B(P)/g and
  ρ̂_A = tr_B(XXᴴ)/tr(XᴴX) satisfy T = ‖ρ_A − ρ̂_A‖₁/2 ≤ s + η/(1−η) (= s for g = 1), |S(ρ_A) − S(ρ̂_A)| ≤ T* ln(d_A−1) + h(T*)
  (Fannes–Audenaert, Audenaert 2007; T* ≤ 1 − 1/d_A required, else NOT_CHECKED with the reason), spectrum by Weyl
  (2T*). The reduced matrix of the stored vectors is exact (rationals), its spectrum enclosed by Ostrowski + Weyl
  with exactly computed defects, x ln x / h / ln on dyadic intervals (`ivl.py`). Which error enters: S4's sin θ
  (Davis–Kahan), not S2's existence radius — the document says why.
* **C-04.** tr(P_s VᴴOV) = tr(Π_V O) exactly (Π_V = VP_sVᴴ); with Q the polar factor of V and Π_Q = QP_sQᴴ,
  |tr(Π_V O) − tr(Π_Q O)| ≤ 2g‖O‖δ(1 + √(1+ι)), δ = 1 − √(1−ι); Theorem 6 in the block on the compressed Hermitian
  operator plus the compression rounding (standard model, exact-rational check) gives the certified interval for
  tr(Π_Q O). The record says what Π_Q is and that other sectors may share the level.
* **E-04.** F_k(c) = H(c_k) + (c − c_k)H1; Weyl per sorted index widens every S2 enclosure by B|c − c_k| with the S3
  rank preserved; the neighbouring certificate's tube (shifted by the affinity defect, exactly zero for binary64-affine
  coefficients) is intersected; the sampling criterion max(m_i, m_{i+1}) + BD < 0 certifies separation on a whole
  sample interval; crossings are bracketed. Sector scans concern the stored block family (ranks in the block; lift
  to the full operator certified at grid points only, recorded).
* **E-07.** ‖Q̂Q̂ᴴ − P‖ ≤ ‖R‖_F/(√(1−η)(dist − w′)) with w′ = ‖XᴴR‖/(1−η) + δ·spread/√(1−η) (Frobenius sin Θ, proved);
  S4 uses min(Theorem 6, this), both recorded. **C-12.** With ‖[H,S]‖ ≤ c recorded, each label of XᴴSX has an
  eigenvalue of S within (2s + 2s² + η)‖S‖ + (c + ‖S‖w/2)/γ₀ + rounding; the tight case A = diag(0, γ), S = c′σ_x
  attains the bound. **C-13.** |tr(P₁P₂) − ‖X₁ᴴX₂‖_F²| ≤ g(s₁+η₁) + g(1+η₁)(s₂+η₂) + rounding.

## Files touched

New: `certify_ed/entanglement.py`, `certify_ed/sweep.py`, `certify_ed/davis_kahan.py`, `docs/proofs/` (4 documents,
4 scripts), `tests/test_entanglement.py` (11), `tests/test_correlation_lift.py` (7), `tests/test_sweep.py` (7),
`tests/test_davis_kahan.py` (7), `RESULT_new-statements.md`.
Modified: `certify_ed/certify.py` (Options fields `entanglement`, `approximate_symmetries`; S4 hooks: entanglement
records, lifted observables, `_correlations(..., audit=)`, `dist`/`s_dk`/`sin_theta_alt`, `labels_hook`; options
record), `certify_ed/correlations.py` (lift functions appended), `certify_ed/eigenspace.py`
(`sin_theta_bounds`, `certify_eigenspace(sin_theta_alt=)`), `certify_ed/enclosure.py` (`_cluster_bound` result
dict: `R_fro`, `XR`, `delta`, `spread`), `certify_ed/scope.py` (entanglement row DEFAULT_SELECTION, omitted when
requested), `certify_ed/diagnostics.py` (`entanglement_next_action`), `certify_ed/presets.py` (S4 "what"),
`certify_ed/cli.py` (`sweep`), `CHANGELOG.md`, `docs/theory.md` (§10), `docs/SCOPE_MATRIX.md`.
No schema change: the new evidence lives in the open cluster records of S4 (`entanglement`, `sin_theta_bounds`,
`quantum_numbers[...]`, `correlations[...].lifted`) and in the new standalone records (sweep, crossing interval,
fidelity); no new reason code or statement id. `python -m certify_ed.schema --check schema`: in sync.

## Negative controls (every status-affecting path)

* C-03: T* above 1 − 1/d_A → NOT_CHECKED with reason and `next_action`, no interval (two planted rotations through
  `certify()`); a planted rotation inside the range keeps the exact entropy inside the interval; size limit, bad
  subsets, no spec, sector claim; the beyond-range Fannes–Audenaert counter-example reproduced in the verification.
* C-04: a planted 0.1 % stretch of one sector basis vector fails S1 and leaves the correlation records INCONCLUSIVE
  while S2–S4 in the block stay CERTIFIED; S1 skipped → NOT_CHECKED (DEPENDENCY_NARROWED); a planted ι = 0.05 shows
  |v − tr(Π_Q O)| > b_V but ≤ b (the isometry term is needed and sufficient).
* E-04: B/20 → violation or empty intersection at every sampled coupling (tests: 40/40; verification: 50/50); a
  declared H1 = 0 → `exactly_affine` False with defect = ‖ΔH‖; non-conserving H1 refused for a block scan.
* E-07: dist overstated (an eigenvalue inside the claimed gap) undercuts the exact sin θ in 20/20 planted cases — the
  hypothesis must come from S3's window, as the hook does. C-12: c halved in the tight case breaks the bound 20/20;
  a 0.6 S^x admixture gives INCONCLUSIVE (D ≥ 1/2); no reference matrix → NOT-CHECKED. C-13: shape mismatch and
  missing stored vectors raise.

## What was not done, and why

* No new statement id or schema change for any item: the task scoped the results as S4 evidence and standalone
  records; a statement id for the sweep (e.g. S9) would need the schema RFC process (`docs/SCHEMA_RFC_PROCESS.md`).
* C-03 for symmetry-adapted claims (lifted reduced states) is NOT_CHECKED (UNSUPPORTED_CLAIM): propagating the
  isometry and invariance defects into the reduced state was not derived.
* C-04 does not assemble per-sector lifts into a full-level statement (Theorem 8-type); the record says that other
  sectors may share the level.
* E-04 for sector scans certifies the lift to the full operator at grid points only; a continuous lift would need the
  invariance defect of V for H1 (`audit_sector(H1, V, ·)`), noted in the document as the next step.
* C-12 uses a first-order geometric term (2s + 2s² + η)‖S‖; Proposition 7's second-order term is not available
  without exact commutation. C-13 has no CLI command (API only).
* The 3.1.0 evidence run, the figures and the manuscript are other tracks.

## Checks

* Branch head: see the git log (the commit after 94ab4d2 adds this summary).

* `ruff check certify_ed tests experiments conformance corpus docs/proofs`: clean.
* `python -m certify_ed.schema --check schema`: in sync.
* Full suite (`pytest -q tests`, PYTHONSAFEPATH removed, TEMP inside the workspace) at 94ab4d2: **985 passed, 2 skipped** in 2288.8 s (38 min on this host; the 953 tests of dev/3.2 plus the 32 new ones).
* Verification scripts (each prints case counts and violations): `verify_C-03.py` 21 176 cases, `verify_C-04.py`
  254, `verify_E-04.py` 1 156, `verify_E-07.py` 553 — 0 violations each.

## Conflicts expected at integration

`certify_ed/certify.py` S4 loop (Backlog track edits the S2 block for INT-05; Props track may touch
`correlations.py` for the Prop 15 proof — my lift functions are appended at the end of that file);
`certify_ed/enclosure.py` `_cluster_bound` return dict (Backlog track owns enclosure.py: the change is four extra
keys in the dict); `docs/theory.md` (§10 appended at the end; the other tracks edit §2–§9); `CHANGELOG.md` (a new
section inserted before the Statements-track section); `docs/SCOPE_MATRIX.md` (rows edited in place).
