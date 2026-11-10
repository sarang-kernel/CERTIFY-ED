# COUNTING.md: counting, assembly and eigenspace statements of CERTIFY-ED on Mathlib

This document describes the Lean 4 formalization of the statements of CERTIFY-ED that count eigenvalues and put counts together: Sylvester's law of inertia, Lemma 4 (inertia with backward error), Corollary 5 (completeness of a 'lowest' claim and the sorted pairing), the exact part of D-01 (the a-posteriori backward error of the sparse factorization), the spectrum of a block-diagonal matrix and Theorem 8 (sector assembly), the separation inequality behind the Davis-Kahan statement E-07, Proposition 14 (degeneracy bracketing), and the exact parts of Propositions 15 and 16, C-06 and E-04. The files are in `reverification_lean/CertifyMathlib/Counting/`, the root module is `CertifyMathlib.Counting` (file `CertifyMathlib/Counting.lean`), and the machine-readable record of every statement, every theorem name and its axioms is `reverification_lean/results/mathlib_counting.json`. The toolchain is Lean 4.35.0 with Mathlib `v4.35.0`, as in `reverification_lean/README.md`.

Every path of the form `code/...`, `manuscript_v4/...` or `project_docs/...` is relative to `sources/archive_v5/`. The statements are taken from `code/certify-ed/docs/theory.md` (sections 3, 4, 5, 9 and 10), `manuscript_v4/sections/s3_math.tex` and the derivation documents in `code/certify-ed/docs/proofs/` named in the table below.

## What a status means here

Each statement has one of three statuses. A statement is "proved" when every Lean theorem named for it is kernel-checked in Lean: it elaborates with no `sorry`, no `native_decide` and no new `axiom`, and `#print axioms` lists only `propext`, `Classical.choice` and `Quot.sound`, the three axioms of Mathlib's classical logic. In the trust scale of `README.md` such a theorem is at level K. A statement is "partial" when the exact core is kernel-checked in Lean and a named part is missing; the missing lemma is named below and in the results file. A statement is "not attempted" when no Lean file states it, and the reason is given.

A kernel-checked Lean proof is a formal proof of the statement as written in Lean. The difference between that statement and the document's statement is given for each result below, under its own heading. Whether such a proof meets the project's rule D-12 (new mathematics counts as established only after Sarang Vehale has re-derived it, `project_docs/DECISIONS_LOG.md`) is his decision. Until he takes it, the derivation documents keep their status "derived, machine-checked, independently re-derived (a second derivation written from the statement alone, not a human check), Sarang's check pending". No result of this document has been checked by a person, and neither have the correspondence notes.

## Status of each statement

| statement | source | Lean theorems (main) | file | status |
|---|---|---|---|---|
| Sylvester's law of inertia | `theory.md` §3; `D-01_sparse-residual.md` Step 5 | `sylvester_card`, `sylvester_inertia`, `cntLt_eq_of_factor` | `QuadForm.lean`, `Inertia.lean` | proved |
| min-max count (subspace form) | used by every count of `theory.md` §3 | `finrank_le_cntLt`, `exists_negOn_cntLt`, `finrank_le_cntLe`, `exists_nonposOn_cntLe` | `Inertia.lean` | proved |
| Weyl's inequality, count form | `cor5_pairing.md` Step 4; `thm8_rho.md` Steps 3 and 4 | `weyl_cntLt`, `weyl_cntLe` | `Inertia.lean` | proved |
| spectral-norm hypotheses | `theory.md` §3 and §5 (norm bounds) | `abs_qf_le_of_toEuclideanLin`, `lemma4_opNorm`, `weyl_opNorm`, `thm8_opNorm` | `Norms.lean` | proved |
| Lemma 4, inertia with backward error | `theory.md` §3; `lem:inertia` in `s3_math.tex` | `lemma4`, `lemma4_doc`, `lemma4_hermitian`, `lemma4_perm` | `Lemma4.lean` | proved |
| count of a 2 × 2 pivot block | `theory.md` §3 ("2 × 2 blocks by the sign of a rational determinant") | `cntLt_fin_two` | `Lemma4.lean` | proved |
| Corollary 5, completeness and sorted pairing | `cor5_pairing.md`; `cor:complete` in `s3_math.tex` | `image_eq_of_match`, `injective_of_groups`, `cor5_le`, `cor5_lt`, `cor5_sorted`, `cor5_model` | `Completeness.lean` | proved |
| D-01, a-posteriori backward error | `D-01_sparse-residual.md` | `abs_qf_le_rowSum`, `lemma4_rowSum` | `Lemma4.lean` | partial |
| spectrum of a block-diagonal matrix | `thm8_rho.md` Step 2 | `cnt_blockDiagonal'`, `isHermitian_blockDiagonal'` | `Assembly.lean` | proved |
| Theorem 8, sector assembly | `thm8_rho.md`; `thm:compose` in `s3_math.tex` | `thm8_blocks`, `thm8`, `cnt_conj` | `Assembly.lean` | partial |
| Theorem 6, projector bound | `theory.md` §4; `thm:dk` in `s3_math.tex` | none | none | not attempted |
| E-07, Lemma 1 (Frobenius sin Θ) | `E-07_davis_kahan.md` §1 | `frob_sylvester`, `frob_eq_re_trace` | `Eigenspace.lean` | partial |
| E-07, Lemma 2, Proposition 1, Lemma 3; C-12; C-13 | `E-07_davis_kahan.md` §1 | none | none | not attempted |
| Proposition 14, degeneracy bracketing | `P14_degeneracy_bracketing.md` | `bracket_count`, `bracket_exact`, `bracket_ge`, `card_Ioo_add_cntLe` | `Lemma4.lean` | proved |
| C-16, exact bracket | `C16_exact_bracket.md` | none | none | not attempted |
| Proposition 15, summed bounds | `P15_summed_bounds.md` | `summed_bound` | `Thermo.lean` | partial |
| Proposition 16, thermodynamic intervals | `P16_thermo_intervals.md` | `thermo_Z`, `thermo_energy`, `thermo_second`, `quotient_bounds`, `free_energy_bounds`, `variance_nonneg`, `variance_bounds`, `heat_capacity_bounds` | `Thermo.lean`, `Variance.lean` | partial |
| C-06, truncated thermodynamics | `C06_truncated_thermodynamics.md` | `tail_weight` | `Thermo.lean` | partial |
| C-04, correlation lift | `C-04_correlation_lift.md` | none | none | not attempted |
| C-07, moments | `C-07_moments.md` | none | none | not attempted |
| E-04, parameter intervals | `E-04_parameter_intervals.md` | `weyl_affine`, `weyl_affine_right`, `qf_triangle`, `sampling_criterion` | `Inertia.lean`, `Parameter.lean` | partial |

All Lean names are in the namespace `CertifyMathlib.Counting`. The library has 98 theorems; the list with their axioms is in `reverification_lean/results/mathlib_counting.json`.

## Why the library builds its own counting argument

Every counting statement of CERTIFY-ED rests on one fact: the number of eigenvalues of a Hermitian matrix `A` below `t` is the largest dimension of a subspace on which the form `xᴴ (A - t I) x` is negative. Sylvester's law, Weyl's inequality, Lemma 4 and the union of block spectra all follow from it in a few lines. Mathlib `v4.35.0` has the spectral theorem for Hermitian matrices (`Matrix.IsHermitian.spectral_theorem`, with eigenvalues `hA.eigenvalues` indexed by the index type of the matrix) and the existence half of Sylvester's law for real quadratic forms (`QuadraticForm.equivalent_signType_weighted_sum_squared` in `Mathlib/LinearAlgebra/QuadraticForm/Real.lean`). A search of the Mathlib sources for `Courant`, `min_max` and `inertia` found no min-max theorem for matrices and no statement that the number of negative squares does not depend on the diagonalization. Comparing characteristic polynomials (Mathlib's `Matrix.IsHermitian.roots_charpoly_eq_eigenvalues`) would have covered exact similarity, but not the inequalities between counts that Lemma 4 and Weyl's inequality need. The library therefore proves the subspace fact once, in `QuadForm.lean` and `Inertia.lean`, and derives the rest from it. A min-max theorem in Mathlib would have made this part shorter; nothing else in the plan depended on it.

The argument in `QuadForm.lean` uses no eigenvalues. If `M = L diag(d) Lᴴ` with `Lᴴ` invertible, the map `x ↦ Lᴴ x` carries the form of `M` to the form of `diag(d)`; on the coordinate subspace where `d < 0` that form is negative, on the one where `d ≥ 0` it is non-negative, and two subspaces that meet only in `0` have dimensions adding to at most `n` (`finrank_add_finrank_le`). `Inertia.lean` applies this to the spectral decomposition `A - t I = U diag(λ - t) Uᴴ`.

## Two conventions

Counting form. Mathlib lists the eigenvalues of `A` as a function `hA.eigenvalues : n → ℝ` with no order on `n` (its sorted list `eigenvalues₀` is decreasing). The library states everything through the counts `cntLt hA t = #{i | λ i < t}` and `cntLe hA t = #{i | λ i ≤ t}`. For the eigenvalues sorted increasingly as `λ_1 ≤ … ≤ λ_n`, `λ_j ≤ x` holds exactly when `j ≤ cntLe hA x`, and `λ_j ≥ y` exactly when `cntLt hA y ≤ j - 1`. Every statement of the documents about "the j-th eigenvalue" is formalized in this form. The translation itself is the definition of the j-th smallest element of a finite multiset and is not formalized.

Norm hypotheses. The core theorems take a perturbation `E` through the bound `|xᴴ E x| ≤ β ‖x‖²` for all `x`. The documents state `‖E‖₂ ≤ β`. The first follows from the second by the Cauchy-Schwarz inequality; this step is kernel-checked in `Norms.lean` (`abs_qf_le_of_toEuclideanLin`), for the operator form `‖E z‖ ≤ β ‖z‖` on `EuclideanSpace 𝕜 n` that Mathlib and `ENCLOSURE.md` use, and Lemma 4, Weyl's inequality and Theorem 8 are restated there with the norm hypotheses of the documents. For a Hermitian `E` the bound also follows from the largest absolute row sum (`abs_qf_le_rowSum`), which is the route of D-01.

## Correspondence, statement by statement

### Sylvester's law of inertia

Document: if `A = L D Lᴴ` with `L` invertible and `D` real diagonal, the number of negative eigenvalues of `A` is the number of negative entries of `D` (`theory.md` §3; used in Step 5 of `code/certify-ed/docs/proofs/D-01_sparse-residual.md`). Lean: `sylvester_inertia` gives `cntLt hA 0 = #{i | d i < 0}` and `cntLe hA 0 = #{i | d i ≤ 0}`, so the numbers of zero and of positive eigenvalues agree as well. `cntLt_eq_of_factor` is the same at a shift `t` with a rectangular `L` that has a two-sided inverse, and `sylvester_card` compares two such factorizations of one matrix without any eigenvalue. The statement is the document's.

### Lemma 4, inertia with backward error

Document: if `‖L D Lᴴ - P(A - σ I)Pᵀ‖ ≤ β` for unit lower-triangular `L` and block-diagonal `D` with `ν` negative eigenvalues, then `#{λ < σ - β} ≤ ν ≤ #{λ ≤ σ + β}` (`theory.md` §3). The manuscript states the strict upper count `ν ≤ #{λ < σ + β}` (`lem:inertia` in `manuscript_v4/sections/s3_math.tex`). Lean: `lemma4` proves the strict form for any `L` whose conjugate transpose has a two-sided inverse and diagonal `D`, with `E` given by `L D Lᴴ = A - σ I + E` and the quadratic-form bound on `E`; `lemma4_doc` gives the `theory.md` form for square invertible `L`; `lemma4_opNorm` takes `‖E‖₂ ≤ β`; `lemma4_hermitian` allows any Hermitian `D` (so a block-diagonal `D` with 1 × 1 and 2 × 2 blocks) with `ν` the number of its negative eigenvalues; `lemma4_perm` allows the permutation, as any unitary `P`. Unit triangularity is not needed, only invertibility. `cntLt_fin_two` is the rule by which the code counts a 2 × 2 block: negative determinant, one negative eigenvalue; positive determinant, two or none according to the sign of the trace. Together with `cnt_blockDiagonal'` it gives `ν` of a block-diagonal `D` as the sum over its blocks.

### Corollary 5, completeness and sorted pairing

Document: `code/certify-ed/docs/proofs/cor5_pairing.md`, section 1 (hypotheses H1 to H3, conclusions for `A_u` and, with the model shift `Δ`, for the declared model `A_m`). Lean: the claimed values `μ` and radii `r` are indexed by a finite type `ι` with `k` elements. (H1) is formalized as a group map `grp : ι → G`, radii constant on each group, group hulls `[lo g, hi g]` containing each member's interval and pairwise disjoint, and an injection `φ` of the claims into the eigen-indices with `|λ (φ j) - μ j| ≤ r j`; `injective_of_groups` builds this injection from injections on the groups (Step 1). (H2) is `cntLt hA c ≤ k` at the cut `c = σ - β`, the lower half of Lemma 4 with `ν = k`. (H3) is `μ j + r j < c`. `image_eq_of_match` gives Step 2 (exactly `k` eigenvalues below `c`, and they are the matched ones, so `λ_{k+1} ≥ c`); `cor5_le` and `cor5_lt` give the pairing in counting form; `cor5_sorted` restates it for sorted claimed values `μ : Fin k → ℝ` as `j + 1 ≤ cntLe hA (μ j + r j)` and `cntLt hA (μ j - r j) ≤ j`, which is `λ_{j+1} ∈ [μ_j - r_j, μ_j + r_j]`; `cor5_model` is Step 4 with the document's (H3) `μ_j + r_j + Δ < σ - β - Δ`, and gives exactly `k` eigenvalues of `A_m` below `σ - β - Δ` and the pairing for `A_m` with radii `r_j + Δ`. The manuscript's two sufficient conditions (enclosures pairwise disjoint, or a common radius) are the cases "every claim its own group" and "one group". The gap bound of the document is the difference of the two bounds and is not stated separately.

### D-01, a-posteriori backward error

Document: `code/certify-ed/docs/proofs/D-01_sparse-residual.md`. Its parts (a) to (c) bound the exact residual of the stored factors through the floating-point products the code forms (Steps 1 to 4); Lemma R1 bounds `‖E‖₂` by `‖E‖_∞` for Hermitian `E`; Step 5 adds the rounding of the shifted diagonal and applies Lemma 4. Lean: `abs_qf_le_rowSum` is Lemma R1 in quadratic-form form, and `lemma4_rowSum` is Step 5: with a diagonal rounding `|Δ_i| ≤ s` and a Hermitian residual of row sums at most `R`, Lemma 4 holds with `β = R + s`. Missing for parts (a) to (c): a model of binary64 arithmetic with the standard error bound for inner products (Higham 2002, Lemma 3.1), which this library does not have. This was not attempted, because a proof of (a) would be a statement about a model of SciPy's arithmetic, and that model, not the inequality, would carry the trust.

### Spectrum of a block-diagonal matrix and Theorem 8

Document: `code/certify-ed/docs/proofs/thm8_rho.md` and `thm:compose` in `manuscript_v4/sections/s3_math.tex`. Lean: `cnt_blockDiagonal'` proves, for `B = blockDiagonal' H` over sectors `s` of sizes `d_s`, that `cntLt hB t = Σ_s cntLt (hH s) t` and the same for `cntLe`, for every `t`; equal counts at every threshold are the same as equal multisets of eigenvalues, so this is "the spectrum of `B` is the multiset union of the sector spectra". `thm8_blocks` is Steps 1 and 2 with a uniform radius `r`: from sector matchings and sector counts `cntLt (hH s) (E_cut - β_s) ≤ k_s` it gives exactly `K = Σ k_s` eigenvalues of `B` below `E_cut - max β` and the sorted pairing of the pooled claimed values with them. `thm8` is Steps 3 to 5 for a square `Q` with `Qᴴ Q = 1` and `Q Qᴴ = 1` (`cnt_conj` moves the counts between the index type of `H` and that of `B`) and the bounds `|yᴴ (Qᴴ H Q - B) y| ≤ ρ ‖y‖²` and `|xᴴ (H - H_spec) x| ≤ m_b ‖x‖²`: the pairing holds for `H_spec` within `r + ρ + m_b`, and `λ_{K+1}(H_spec) ≥ E_cut - max β - (ρ + m_b)`. With `m_b = 0` this is the manuscript's statement. `thm8_opNorm` takes the two bounds as spectral norms. Missing: the step from the audit quantities to the bound on `Qᴴ H Q - B`, that is, the polar-factor lemma of `code/certify-ed/docs/proofs/thm2_polar-factor.md` for the square `W = [V_1 ⋯ V_m]`, with `Q` the polar factor and `ρ = D + (‖H‖ + ‖B‖) δ`. That lemma is kernel-checked in operator form in the enclosure part of this repository (`CertifyMathlib.Enclosure.polarQ_residual_le`, see `ENCLOSURE.md`). Joining the two needs a matrix `Q` for the operator `polarQ` with `Qᴴ Q = Q Qᴴ = 1` and the passage from `‖H Q z - Q B z‖ ≤ ρ ‖z‖` to the bound on `Qᴴ H Q - B`; this was not done here, so that this library does not depend on files of another part that were still changing.

### Theorem 6 and E-07

Document: Theorem 6 (`theory.md` §4, `thm:dk` in the manuscript) bounds `‖Q̂ Q̂ᴴ - P‖` by `s = ρ/sep`; `code/certify-ed/docs/proofs/E-07_davis_kahan.md` gives the alternative bound `s_DK` through Lemma 1 (the Frobenius form of the sin Θ theorem), Lemma 2 (two polar identities) and Proposition 1. Lean: `frob_sylvester` is the separation inequality inside the proof of Lemma 1, for any Hermitian `A` and `H` and any `Y`: if every nonzero coordinate of `Y` in the two eigenbases pairs eigenvalues at distance at least `δ`, then `δ² ‖Y‖_F² ≤ ‖A Y - Y H‖_F²`. For `Y = (I - P) Q̂` the coordinates on the cluster eigenvectors vanish, which is the hypothesis of Lemma 1. Missing for Lemma 1: the identity `(I - P) R' = A_⊥ Y - Y H'`, which needs the spectral projector `P` as a matrix and its commutation with `A`. Not attempted: Lemma 2, Proposition 1, Lemma 3, Propositions 2 and 3 (C-12, C-13) and Theorem 6. They need the polar factor and, for Theorem 6, Kahan's theorem and a spectral-norm sin Θ theorem; the effort went first to the counting statements that S3 and S5 use directly.

### Proposition 14, degeneracy bracketing

Document: `code/certify-ed/docs/proofs/P14_degeneracy_bracketing.md`, parts (a) to (e). Lean: `bracket_count` gives (a) and (b) without subtraction, as `#I_in + ν_lo ≤ ν_hi` and `ν_hi ≤ ν_lo + #I_out`, under the hypothesis `σ_lo + β_lo < σ_hi - β_hi` (`I_in` not empty); `bracket_exact` gives (c) and `bracket_ge` gives (e), with (H1) formalized as `g` distinct eigen-indices with eigenvalues in `I_in`. Part (d) is (a) read for `ν_hi - ν_lo > g`. The hypothesis that `I_in` is not empty is in part (b) of the derivation document but not in the form of `theory.md` §9.1, which states the two inequalities without it (see the findings below).

### Propositions 15 and 16 and C-06

Proposition 15 (`code/certify-ed/docs/proofs/P15_summed_bounds.md`): `summed_bound` is inequality (1), the exact statement. The floating-point corollary (2) and Lemmas A and B are statements about binary64 arithmetic and were not attempted, for the reason given under D-01.

Proposition 16 (`code/certify-ed/docs/proofs/P16_thermo_intervals.md`): hypothesis (H) is formalized as an assignment `κ` of the eigen-indices to the intervals with `l (κ i) ≤ λ_i ≤ h (κ i)`, the count `c_k` of interval `k` being the size of its fibre. `thermo_Z`, `thermo_energy` and `thermo_second` enclose `Z e^{β E_0}`, `Σ_i e_i e^{-β e_i}` and `Σ_i e_i² e^{-β e_i}` (`e_i = λ_i - E_0 ≥ 0`) between the sums of the interval ends; `quotient_bounds` encloses `⟨e⟩` and `⟨e²⟩` as quotients and `free_energy_bounds` encloses `F = E_0 - T ln Z̃`. The partition function is the sum over Mathlib's eigenvalues, `Σ_i e^{-β λ_i}`. In `Variance.lean`, `variance_nonneg` gives `⟨e²⟩ - ⟨e⟩² ≥ 0` for non-negative weights with positive sum (a weighted Cauchy-Schwarz inequality, `weighted_sq_le`); `variance_bounds` gives the interval `[max(0, a₂ - b₁²), b₂ - a₁²]` for the variance from enclosures `a₁ ≤ ⟨e⟩ ≤ b₁` with `a₁ ≥ 0` and `a₂ ≤ ⟨e²⟩ ≤ b₂`, which is the interval of the document after its intersection with `[0, ∞)`; `heat_capacity_bounds` multiplies it by `β²`. The shift identity `Var(E) = Var(e)` of the proof is not stated as a separate theorem. Missing: the dyadic interval arithmetic of Lemma 17, which belongs to `ENCLOSURE.md`.

C-06 (`code/certify-ed/docs/proofs/C06_truncated_thermodynamics.md`): `tail_weight` is part (i), the omitted weight: the eigenvalues at or above `L` number `n - cntLt hA L`, and their shifted weight lies in `(n - k)·[e^{-β(M - E_0)}, e^{-β(L - E_0)}]`. Missing: Lemma C (the envelopes of `e e^{-βe}` and `e² e^{-βe}`, a calculus statement) and the assembled intervals of part (iii).

### E-04, parameter intervals

Document: `code/certify-ed/docs/proofs/E-04_parameter_intervals.md`. Lean: `weyl_affine` is Proposition 1 (first tube) and Proposition 2 in counting form for `F = H(c_k) + (c - c_k) H1` with `|xᴴ H1 x| ≤ B ‖x‖²`; `weyl_affine_right` is the second tube with the affinity defect `d_k`; `sampling_criterion` is Lemma 3 for a margin `m` that is `2B`-Lipschitz on the whole sample interval. Missing: Lemma 4 of the document (the term-wise bound on `‖H1‖₂`, which needs norms of Kronecker products) and the continuity argument for crossings.

### C-16, C-04 and C-07

Not attempted. C-16 (`code/certify-ed/docs/proofs/C16_exact_bracket.md`) counts eigenvalues in a rational bracket with Sturm sequences of the squarefree factors of the characteristic polynomial; Mathlib has no Sturm theorem, and the statement's hypothesis (H2) makes the exactness of python-flint an assumption that a Lean proof could not discharge. C-04 (`code/certify-ed/docs/proofs/C-04_correlation_lift.md`) and C-07 (`code/certify-ed/docs/proofs/C-07_moments.md`) need trace norms, Hölder's inequality for traces and Theorem 6, none of which is in this library.

## Earlier evidence for the same statements

Before this formalization the statements had been machine-checked by the scripts of the code repository and derived a second time from the statement alone. The table copies the case counts from section 6 of each derivation document and the verdict of the second derivation from the same place (`code/certify-ed/docs/proofs/VERDICTS.md` has the details). These are machine checks, not human checks.

| statement | derivation document | machine-checked cases (copied from section 6 of the document) | verdict of the second derivation |
|---|---|---|---|
| Corollary 5 | `code/certify-ed/docs/proofs/cor5_pairing.md` | 20 347 random + 900 adversarial cases, 0 violations | AGREES |
| Theorem 8 | `code/certify-ed/docs/proofs/thm8_rho.md` | 240 (160 inequality checks) random + 49 (32 inequality checks, 5 negative controls) adversarial cases, 0 violations | AGREES |
| D-01 | `code/certify-ed/docs/proofs/D-01_sparse-residual.md` | 400 random + 17 adversarial cases, 0 violations | AGREES-WITH-REMARKS |
| E-07 | `code/certify-ed/docs/proofs/E-07_davis_kahan.md` | 529 random + 24 adversarial cases, 0 violations; both negative controls detected 20/20 | GAP |
| Proposition 14 | `code/certify-ed/docs/proofs/P14_degeneracy_bracketing.md` | 3060 random + 1006 adversarial cases, 0 violations | AGREES |
| Proposition 15 | `code/certify-ed/docs/proofs/P15_summed_bounds.md` | 1111 random + 32 adversarial cases, 0 violations | AGREES |
| Proposition 16 | `code/certify-ed/docs/proofs/P16_thermo_intervals.md` | 4000 random + 95 adversarial cases, 0 violations | AGREES-WITH-REMARKS |
| C-06 | `code/certify-ed/docs/proofs/C06_truncated_thermodynamics.md` | 300 random + 150 adversarial cases, 0 violations | AGREES-WITH-REMARKS |
| C-16 | `code/certify-ed/docs/proofs/C16_exact_bracket.md` | 1501 random + 42 adversarial cases, 0 violations | AGREES-WITH-REMARKS |
| E-04 | `code/certify-ed/docs/proofs/E-04_parameter_intervals.md` | 1 049 random + 107 adversarial cases, 0 violations; negative controls detected | ERROR |
| C-04 | `code/certify-ed/docs/proofs/C-04_correlation_lift.md` | 240 random + 14 adversarial cases, 0 violations | GAP |
| C-07 | `code/certify-ed/docs/proofs/C-07_moments.md` | 418 random + 192 adversarial cases, 0 violations; negative controls 132/138, 138/138 and 40/40 | AGREES |

## Findings made while formalizing

The upper count of Lemma 4. The Lean proof gives the strict count `ν ≤ #{λ < σ + β}` without extra hypotheses, as the manuscript states it (`manuscript_v4/sections/s3_math.tex`, `lem:inertia`). The `theory.md` form with `≤` follows from it. Nothing in the code depends on the difference.

The non-empty inner interval of Proposition 14. The inequality `#I_in ≤ ν_hi - ν_lo` is false without `σ_lo + β_lo < σ_hi - β_hi`. For a 1 × 1 matrix with eigenvalue 0, shifts `σ_lo = -s`, `σ_hi = s` and backward errors `β_lo = β_hi = 2s` for any `s > 0`, the values `ν_lo = 1` and `ν_hi = 0` satisfy both Lemma 4 sandwiches, `I_in` is empty and `ν_hi - ν_lo = -1`. Part (b) of `code/certify-ed/docs/proofs/P14_degeneracy_bracketing.md` carries the hypothesis; the summary in `theory.md` §9.1 does not. The code uses the inequality only with `J ⊂ I_in`, which implies the hypothesis, so no certified statement is affected.

The sorted pairing without the exchange argument. In counting form, Corollary 5 needs no exchange of an inversion: each of `cor5_le` and `cor5_lt` compares two counts at one threshold and uses the equality of radii within a group exactly once. This is where the remark of `cor5_pairing.md` (the pairing fails for non-uniform radii) enters the proof. The second derivation recorded in `VERDICTS.md` also used a counting argument.

The sampling criterion of E-04. `sampling_criterion` needs the margin to be `2B`-Lipschitz on the whole sample interval. At an interior grid point the tubes can jump by the affinity defects, and there the hypothesis fails; this is the case in which the second derivation found the error recorded in `VERDICTS.md` for `E-04_parameter_intervals.md` (fixed in the code by adding the defects to the slack). The Lean statement makes the hypothesis explicit.

## How the files were checked

Each file was elaborated with `lake env lean` until it gave no error, then built as a module. The eleven files (the root module and ten modules) contain no `sorry`, no `admit`, no `native_decide` and no `axiom` declaration; this was checked by searching the files. `#print axioms` was run for every theorem by `reverification_lean/Drafts/Counting/PrintAxioms.lean`; every one of the 98 theorems depends exactly on `propext`, `Classical.choice` and `Quot.sound` (the output is kept in `reverification_lean/Drafts/Counting/print_axioms_output.txt`). The second checker of the toolchain, `leanchecker`, replayed the nine modules other than `Variance` and the root module as it stood before `Variance.lean` was added, and accepted them (exit code 0, no message); a second run replayed `Variance` and the root module with its new import and accepted them as well. Every theorem of the eleven files is therefore at level K2. The commands, exit codes and times of the two runs are in `reverification_lean/results/mathlib_counting.json`.

In a Windows command window, with the environment of `reverification_lean/README.md` (the Lean binaries and Git on `PATH`, `MATHLIB_CACHE_DIR` set), from the folder `reverification_lean`:

```
lake build CertifyMathlib.Counting
lake env lean Drafts/Counting/PrintAxioms.lean
lake env leanchecker CertifyMathlib.Counting.QuadForm CertifyMathlib.Counting.Inertia CertifyMathlib.Counting.Lemma4 CertifyMathlib.Counting.Completeness CertifyMathlib.Counting.Assembly CertifyMathlib.Counting.Thermo CertifyMathlib.Counting.Eigenspace CertifyMathlib.Counting.Parameter CertifyMathlib.Counting.Norms CertifyMathlib.Counting.Variance CertifyMathlib.Counting
```

## Open items

First, Theorem 8 becomes complete once the polar-factor lemma of the enclosure part is joined to `thm8` as described above. Second, the remaining part of Lemma 1 of E-07 (the projector identity) and then Lemma 2 and Proposition 1 would give the bound `s_DK` itself; Theorem 6 also needs Kahan's theorem, which is in the enclosure part. Third, Lemma C of C-06 is elementary and was left for lack of time. Fourth, the floating-point parts of D-01 and Proposition 15 would need a model of binary64 arithmetic in Lean, which is a separate project. Fifth, Sarang Vehale's check under D-12 is pending for every derivation document named here.
