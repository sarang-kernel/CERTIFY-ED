# RESULT: track/backlog-proofs (D-59 derivations of the UNVERIFIED backlog)

Branch `track/backlog-proofs` on `dev/3.2` (38b0212). Head: see the last line. Date: 2026-10-30.

## Per item

| item | status | proof document | verification (random + adversarial, violations) |
|---|---|---|---|
| 1 Theorem 2 with ρ and the polar-factor lemma | done | `docs/proofs/thm2_polar-factor.md` | 17 224 cases (34 448 lemma checks, 5 000 Kahan-block checks) + 600 adversarial (900 + 600), 0 violations (`docs/proofs/runs/thm2_polar-factor.txt`) |
| 1 Theorem 2′ (ε*) | done | `docs/proofs/thm2prime_eps-star.md` | 17 094 (51 282 inequality checks) + 621 (1 863), 0 violations |
| 2 Corollary 5 pairing | done | `docs/proofs/cor5_pairing.md` | 20 000 lemma + 347 code cases + 400 negative controls + 500 ties, 0 violations |
| 2 Theorem 8 ρ | done | `docs/proofs/thm8_rho.md` | 240 + 49 statement checks against the exact spectrum, 160 + 32 inequality checks, 5 negative controls, 0 violations |
| 3 Appendix A model shift Δ | partial | `docs/proofs/appA_model-shift.md` | derivation complete; `verify_appA_model-shift.py` written but **not executed**: it had not been run when this record was written, and the document's status line said NOT RUN (the run of 2026-11-03 is `docs/proofs/runs/appA_model-shift.txt`). Pilot evidence of the builder bound exists only through `tests/test_backlog_proofs.py::test_builder_shift_dominates_the_build_error_for_a_spin1_chain` |
| 4 Theorem 1 floating-point accounting | done | `docs/proofs/thm1_fp-accounting.md`, scaffold `docs/proofs/appendixA_scaffold.tex` | 20 000 + 246 radius checks, 40 000 + 480 stage checks, 20 012 scalar-lemma grid points, 0 violations |
| 5 INT-05 underflow term | done | `docs/proofs/int05_underflow.md` | 18 992 model operations, 5 000 dot, 5 200 exact_dot, 205 023 norm/T_N, 2 650 residual, 2 500 monotonicity, 0 violations |
| 6a sectors.py:240 mutant | done | — | `test_label_audit_defect_within_rounding_allowance_of_the_threshold` (planted defect two ulps below 1/2; the mutant `defects[k] − rnd` would certify, found by bisection over binary64 ε) |
| 6b certify.py Kato–Temple `pos > 0` | done (by test) | — | `test_kato_temple_second_lowest_pair_uses_its_lower_neighbour_as_isolation_end`: with the mutant the isolation interval of the second-lowest pair starts at −∞ and Kato–Temple's upper end collapses to the Rayleigh quotient θ < λ₂ of a ground-state-contaminated vector; the test asserts λ₂ ∈ [lower, upper] and a finite lower isolation end |
| 6c checker.py:261 mantissa 2⁵³ | done (test + equivalence argument) | — | `test_float_vector_to_int_reconstructs_the_vector_exactly`. Equivalence note: the mutant 2⁵⁴ doubles every entry of y; `radius_exact` is homogeneous of degree 0 in (yr, yi) and `_float_vector_to_int` is used only there (and the zero test is scale invariant), so no certificate output changes — the mutant is equivalent at the output level and is killed only by the reconstruction test |
| 6d 15 E6 survivors of the 3.1.0 run | skipped | — | the worksheet `results/run_full_20261022/E6_inspection_worksheet.md` is not in the clone, as expected |

## Mathematics: what was found

- **Theorem 2.** `docs/theory.md` stated the cluster radius ε = (w + √(w² + 4ρ²))/2 from the Weyl-perturbation
  proof of the manuscript appendix (perturbation matrix [[S, Tᴴ], [T, 0]], whose norm exceeds ‖[S; T]‖ unless
  S = 0). Kahan's inclusion theorem holds for an arbitrary Hermitian block (proved from the Courant–Fischer
  interval count and Hall's theorem), so ρ itself is a valid radius and ε ≥ ρ always. The code now uses min(ρ, ε*);
  `_subspace_shift` and `decomposition_shift` return ρ instead of (1 + √5)/2 ρ. Cluster radii shrink by up to the
  factor 1.618 where ρ is the smaller bound — a bound change justified by the proof document (soundness rule).
- **Theorem 2′** holds as stated; the proof sketch of theory.md is complete once ‖[B, C]‖ ≤ 2‖B‖‖C‖ with
  ‖Λ − cI‖ = spread/2 is written out.
- **Corollary 5**: the sorted pairing needs a *uniform* radius within each group (counterexample with
  non-uniform radii in the document, 238 of 2 000 random cases); the code satisfies this because cluster members
  share the cluster radius and groups have disjoint union intervals.
- **Theorem 8** needs a Hermitian built matrix; `build_bitwise` can differ from its adjoint in the last bit for
  irrational local amplitudes (different multiplication order of the same factors). `compose.py` and the sector
  path of `certify()` now symmetrise it (`construct.hermitian_part`); the builder shift covers the symmetrisation
  (Lemma A2).
- **Appendix A**: `_builder_shift` multiplied by (1 + γ_W) where 1/(1 − γ_W) is what the derivation needs (the
  former suffices only for Wu ≲ 10⁻⁸); changed. `OperatorAdapter.width` for a spec operator did not count the
  amplitude roundings: m + L_max + 2 is short by L_max − 4 roundings for strings of five or more letters;
  changed to m + 2L_max + 2.
- **Theorem 1 accounting**: the INT-04 list is correct; the slack of the norm guards is 3u (needed now for the
  fourth rounding of step 5). Hypotheses n, w ≤ 2²⁶ are stated (second-order terms (nu)² ≤ u/2).
- **INT-05**: the pure standard model indeed fails inside the cluster bound as well — the pilot run of the
  Theorem 2 script on an operator scaled by 10⁻¹⁵⁰ returned a cluster radius of 0 (the squares in the norms
  underflowed) before the fix. The underflow-tolerant model is derived from the binary64 format (no Higham
  quotation: the text was not available offline; §2.5/eq. (2.6) is cited by name only). `exact_dot`'s Dekker
  TwoProduct was also exposed to underflow (products of two entries of 10⁻²⁶⁵ flush); products below 2⁻⁹⁶⁷ are
  now left out and bounded.

## Files touched

`certify_ed/enclosure.py` (Theorem 2 docstring, `_cluster_bound`, spec width, residual additive terms, kato_temple),
`certify_ed/fp.py` (ULP_MIN, norm_underflow_term, residual_entry_count, residual_additive_term, gram_additive_term,
norm2_upper/lower, TwoProduct split, exact_dot), `certify_ed/certify.py` (_subspace_shift, _builder_shift,
hermitian_part of the sector reference, post-enclosure scan removed, docstring of _underflow_operands),
`certify_ed/compose.py` (hermitian_part, norm2_upper for D, + ULP_MIN), `certify_ed/sectors.py`
(decomposition_shift), `certify_ed/construct.py` (hermitian_part), `tests/test_whole_spectrum.py` (cluster radii
within the Theorem 2 bracket; blocking invariance compared new-vs-new), `tests/test_underflow_check.py` (two tests
adjusted to the removed post scan), `tests/test_backlog_proofs.py` (new, 25 tests), `docs/theory.md`,
`docs/SCOPE_MATRIX.md`, `CHANGELOG.md`, `docs/proofs/` (7 documents, 7 scripts, `_common.py`, `runs/`,
`appendixA_scaffold.tex`).

## Tests added and negative controls

`tests/test_backlog_proofs.py`: scalar lemma constants; cluster bound = min(ρ, ε*) and ρ pinned to its formula;
ε* terms are upper bounds; duplicate columns → no cluster bound (negative control); Theorem 2 radius contains
distinct eigenvalues (random); `_subspace_shift`/`decomposition_shift` return ρ; `_certify_lowest` outputs pinned
to Corollary 5 with a Weyl-transfer check on H + E; missing eigenvalue and duplicated vector not CERTIFIED
(negative controls); compose ε = up(ρ + m_build) and the statement holds; `hermitian_part`; spec adapter width;
builder shift dominates the build error (spin-1 chain); norm bounds with underflowed squares; `exact_dot` with tiny
and subnormal products; scaled-operator radii enclose the eigenvalues; radii never shrink against the pre-branch
formula (bit identity for normal radii); tiny-vector radius in exact arithmetic (and +∞ below resolution); the
three mutation tests. The verification scripts carry their own negative controls (Cor 5: 400; Thm 8: 5; App. A:
written, not run).

## What was not done and why

- Item 3's computational verification was not executed at the time (it ran on 2026-11-03). The document
  and the script are in place; running `python docs/proofs/verify_appA_model-shift.py` and filling the counts into
  §4 and §6 of `appA_model-shift.md` is the remaining step.
- Item 6d: worksheet unavailable.
- The Theorem 8 verification uses N = 4, 6 spin-½ chains and a 3-site spin-1 chain (the 60-digit ground truth
  costs O(n³) per case); larger chains are covered by the statement check only through the fixtures of the test
  suite.
- No manuscript prose was written (D-16); the scaffold is statement-level.

## Suite and lint

`ruff check certify_ed tests experiments conformance corpus docs/proofs`: clean.
Full suite (`pytest -q tests`, Windows, QuSpin installed): **978 passed, 2 skipped, 5 warnings in 1670.21s (0:27:50)**
(953 + 25 new tests).

Branch head: bea0cd5 (plus this amendment commit recording the head).
