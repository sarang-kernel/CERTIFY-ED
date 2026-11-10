# ENCLOSURE.md: the enclosure theorems of CERTIFY-ED on Mathlib

This document describes the Lean 4 formalization of the enclosure theorems of CERTIFY-ED: Theorem 1 (the residual enclosure), its multi-pair form and its version with the S1 model shift, Theorem 3 (Kato-Temple), Gershgorin's theorem for Hermitian matrices, Weyl's inequality and the model shift of Appendix A, Kahan's inclusion theorem, the polar-factor lemma, Theorem 2 and Theorem 2', the exact part of the variational statements E-05, and Lemma 2 of the generalized eigenproblems E-03. The files are in `reverification_lean/CertifyMathlib/Enclosure/`, the root module is `CertifyMathlib.Enclosure` (file `CertifyMathlib/Enclosure.lean`), and the machine-readable record of every statement, every theorem name and its axioms is `reverification_lean/results/mathlib_enclosure.json`. The toolchain is Lean 4.35.0 with Mathlib `v4.35.0`, as in `reverification_lean/README.md`.

Every path of the form `code/...`, `manuscript_v4/...` or `project_docs/...` is relative to `sources/archive_v5/`. The statements are taken from `code/certify-ed/docs/theory.md` (sections 2 and 10.5), `manuscript_v4/sections/s3_math.tex` and the derivation documents `code/certify-ed/docs/proofs/thm2_polar-factor.md`, `thm2prime_eps-star.md`, `appA_model-shift.md` and `E-05_variational.md` and `E-03_generalized.md`.

## What a status means here

Each statement has one of three statuses. A statement is "proved" when every Lean theorem named for it is kernel-checked in Lean: it elaborates with no `sorry`, no `native_decide` and no new `axiom`, and `#print axioms` lists only `propext`, `Classical.choice` and `Quot.sound`, the three axioms of Mathlib's classical logic. In the trust scale of `README.md` such a theorem is at level K. A statement is "partial" when the exact core is kernel-checked in Lean and a named part is missing; the missing lemma is named in the table and in the results file. A statement is "not attempted" when no Lean file states it.

A kernel-checked Lean proof is a formal proof of the statement as written in Lean. The difference between that statement and the document's statement is given for each result below, under "Correspondence". Whether such a proof meets the project's rule D-12 (new mathematics counts as established only after Sarang Vehale has re-derived it, `project_docs/DECISIONS_LOG.md`) is his decision. Until he takes it, the derivation documents keep their status "derived, machine-checked, independently re-derived (a second derivation written from the statement alone, not a human check), Sarang's check pending". No result of this document has been checked by a person.

## Status of each statement

| statement | source | Lean theorem (main) | file | status |
|---|---|---|---|---|
| Theorem 1, residual enclosure | `theory.md` §2, `thm:residual` | `residual_enclosure` | `Residual.lean` | proved |
| Theorem 1, multi-pair form of S2 | `theory.md` §2 | `residual_enclosure_family` | `Residual.lean` | proved |
| Theorem 1 with the model shift sigma | `theory.md` §2 step 6 | `residual_enclosure_of_shift`, `residual_enclosure_of_l2_opNorm` | `Residual.lean` | proved |
| Theorem 3, Kato-Temple | `theory.md` §2, `thm:kato` | `kato_temple` | `KatoTemple.lean` | proved |
| Gershgorin and `‖A‖₂ ≤ ‖A‖_∞` | Mathlib `eigenvalue_mem_ball`, `thm2_polar-factor.md` §3 | `eigenvalue_mem_gershgorin`, `l2_opNorm_le_of_row_sum` | `Residual.lean` | proved |
| Weyl's inequality | `appA_model-shift.md` Theorem A5 | `weyl_eigenvalues₀`, `card_le_le_card_le_add` | `Weyl.lean` | proved |
| Lemmas A1 and A2 (builder rounding) | `appA_model-shift.md` §1 | none | none | not attempted |
| Lemma A3 | `appA_model-shift.md` §1 | `schur_bound`, `norm_le_max_of_row_col` | `Weyl.lean` | proved |
| Theorem A4, the model shift | `appA_model-shift.md` §1 | `model_shift_bound` | `Weyl.lean` | partial |
| Theorem A5, Weyl transfer for S2 and S3 | `appA_model-shift.md` §1 | `weyl_transfer_enclosure`, `weyl_transfer_count` | `Weyl.lean` | proved |
| polar-factor lemma | `thm2_polar-factor.md` §1, `lem:polar` | `polarQ_norm`, `sub_polarQ_le`, `polarQ_residual_le` | `Cluster.lean` | proved |
| Kahan's inclusion theorem | `thm2_polar-factor.md` §2 | `kahan`, `kahan_antitone` | `Kahan.lean` | proved |
| Theorem 2, radius rho | `thm2_polar-factor.md`, `thm:cluster` | `cluster_enclosure` | `Cluster.lean` | proved |
| Theorem 2', radius eps* and min(rho, eps*) | `thm2prime_eps-star.md`, `lem:epsstar` | `cluster_enclosure_eps_star`, `cluster_min` | `ClusterPrime.lean` | proved |
| E-05 variational certificates | `E-05_variational.md` §1 | `rayleigh_ritz_matrix`, `temple_matrix` | `Variational.lean` | partial |
| D-02 matrix-free residual bounds | `D-02_matrix-free.md` | none | none | not attempted |
| E-03 generalized eigenproblems | `E-03_generalized.md` | `generalized_residual_enclosure_beta` | `Generalized.lean` | partial |
| Lemma 17, dyadic exp and ln | `L17_dyadic_exp_ln.md` | none | none | not attempted |

All Lean names are in the namespace `CertifyMathlib.Enclosure`; the operator forms are in `CertifyMathlib.Enclosure.EigenData`. The results file lists every theorem of the nine files with its axioms.

## The common framework

All theorems are first proved for a symmetric linear map `T` on a finite-dimensional inner product space `E` over a field `𝕜` that is the real or the complex numbers (Mathlib's class `RCLike`), together with an orthonormal basis of eigenvectors with real eigenvalues. This pair is the structure `EigenData` of `Spectral.lean`. For a Hermitian matrix `A` the map is `Matrix.toEuclideanLin A` on `EuclideanSpace 𝕜 n` (column vectors with the Euclidean norm), and the eigenbasis is Mathlib's `Matrix.IsHermitian.eigenvectorBasis` with the eigenvalues `Matrix.IsHermitian.eigenvalues` (`hermitianEigenData`), or the sorted basis with the eigenvalues `eigenvalues₀` in decreasing order (`sortedEigenData` in `Weyl.lean`). The reason for this choice is that every proof of the documents is a computation in the eigenvector basis or a dimension count, and both are short for an abstract eigenbasis; the matrix statements are then one line each.

`Spectral.lean` holds the coordinate formulas (Parseval, the residual norm and the Rayleigh quotient in the eigenvector basis) and one counting lemma, `EigenData.finrank_le_card_of_support`: if no nonzero vector of a subspace `Y` has all its coefficients with index in a set `s` equal to zero, then the dimension of `Y` is at most the size of `s`. Its proof intersects `Y` with the span of the remaining eigenvectors (`exists_ne_zero_mem_inf`, from the dimension formula for the sum and the intersection of two subspaces). The two uses are the Courant-Fischer counts: a subspace on which `‖T y - c y‖ ≤ r ‖y‖` forces at least `dim Y` eigenvalues, counted with multiplicity, within `r` of `c` (`finrank_le_card_near`), and a subspace on which the Rayleigh quotient is at most `t` forces at least `dim Y` eigenvalues at most `t` (`finrank_le_card_le`, strict form `finrank_le_card_lt`). Weyl's inequality and Step 1 of Kahan's theorem are instances.

Norms of operators are written by their defining bound: "`‖R‖ ≤ nR`" is the hypothesis `∀ z, ‖R z‖ ≤ nR * ‖z‖`. This is equivalent to a bound on the spectral norm, and it allows any upper bound on the spectral norm to be used, as `manuscript_v4/sections/s3_math.tex` allows ("any upper bound on the spectral norm may be substituted"). For the model shift there is also a form with Mathlib's spectral norm of matrices (`Matrix.Norms.L2Operator`).

## Correspondence

### Theorem 1 and its two forms

Document (`theory.md` §2): for `x ≠ 0` and real `μ`, some eigenvalue satisfies `|λ - μ| ≤ ‖Ax - μx‖/‖x‖`. Lean, `residual_enclosure`: for a Hermitian `A : Matrix n n 𝕜`, real `μ` and `x : EuclideanSpace 𝕜 n` with `x ≠ 0`, there is an index `i` with `|hA.eigenvalues i - μ| ≤ ‖toEuclideanLin A x - μ • x‖ / ‖x‖`. The only difference is that the field may be the real or the complex numbers. The proof writes the squared residual as `∑ (λ_i - μ)² |c_i|²` and compares it with `r² ∑ |c_i|²`.

The multi-pair form, `residual_enclosure_family`, gives one index per claimed pair. The index map is not claimed to be injective, which is the document's remark that Theorem 1 gives existence only: two copies of one vector are enclosed by the same eigenvalue.

The version with the S1 model shift, `residual_enclosure_of_shift`, encloses an eigenvalue of a Hermitian `H` with `‖(H - A) v‖ ≤ σ ‖v‖` for all `v` (and `residual_enclosure_of_l2_opNorm` with `‖H - A‖ ≤ σ` in Mathlib's spectral norm) by the radius `‖Ax - μx‖/‖x‖ + σ`, computed with the stored matrix `A`. The document's radius is `add_up(r, σ)`, the same sum stepped one unit outward; the outward rounding is a floating-point step and is not part of the Lean statement. The Lean proof uses the triangle inequality and Theorem 1 for `H`, so it needs neither Weyl's inequality nor symmetry of `A`; the version that keeps the sorted index is Theorem A5 below.

### Theorem 3 (Kato-Temple)

Document: with `‖x‖ = 1`, `θ = xᴴAx`, `r = ‖Ax - θx‖` and an interval `(α, β)` containing `θ` and exactly one eigenvalue `λ`, `θ - r²/(β - θ) ≤ λ ≤ θ + r²/(θ - α)`. Lean, `kato_temple`: the same, with `θ = re ⟪x, A x⟫` (the lemma `inner_apply_self_eq_re` shows that the quadratic form of a Hermitian matrix is real, so this is `xᴴAx`) and "exactly one eigenvalue" written as "every eigenvalue in `(α, β)` equals `λ`". The eigenvalue `λ` may therefore have any multiplicity. The operator forms are more general: `EigenData.kato_temple_upper` needs only `α < θ` and `EigenData.kato_temple_lower` only `θ < β`, `λ` may be any real number in `[α, β]` that every eigenvalue equals or avoids `(α, β)`, and the division-free forms `(λ - θ)(θ - α) ≤ r²` and `(θ - λ)(β - θ) ≤ r²` need no hypothesis on `θ`. The proof is the classical identity `∑ w_i (λ_i - λ)(λ_i - a) = r² + (θ - λ)(θ - a)` with weights `w_i = |⟨b_i, x⟩|²` (`kato_temple_identity`), whose left side is nonnegative for `a = α` and `a = β`.

### Gershgorin

Mathlib's `eigenvalue_mem_ball` states Gershgorin's disc theorem for an eigenvalue of the linear map `Matrix.toLin' A` over any normed field. `eigenvalue_mem_gershgorin` applies it to each `hA.eigenvalues i` of a Hermitian matrix, with the eigenvector `eigenvectorBasis i`: some `k` has `‖λ_i - A k k‖ ≤ ∑_{j ≠ k} ‖A k j‖`. The statement that the code uses is its consequence: `inf_norm_upper(A) ≥ ‖A‖_∞ ≥ ‖A‖₂` for Hermitian `A` (`code/certify-ed/docs/proofs/thm2_polar-factor.md` §3, the row for `‖A‖`). Lean: `abs_eigenvalue_le_row_sum` gives `|λ_i| ≤ ∑_j ‖A k j‖` for some `k`, `norm_toEuclideanLin_le_of_row_sum` gives `‖A v‖ ≤ R ‖v‖` for every `R ≥ 0` above every row sum, and `l2_opNorm_le_of_row_sum` gives `‖A‖ ≤ R` in Mathlib's spectral norm. The hypothesis `R ≥ 0` is added; a maximum row sum always satisfies it.

### Weyl's inequality and Appendix A

Document (`appA_model-shift.md`, Theorem A5, citing Horn and Johnson, Corollary 4.3.15): Hermitian `H_h`, `H_spec` with `‖H_h - H_spec‖₂ ≤ Δ` have `|λ_j(H_spec) - λ_j(H_h)| ≤ Δ` for every `j`. Lean, `weyl_eigenvalues₀`: the same for Mathlib's sorted eigenvalues `eigenvalues₀`, which are in decreasing order. The document sorts increasingly; reversing both sequences leaves the statement unchanged. The proof goes through the counting forms `#{λ(T) ≤ t} ≤ #{λ(S) ≤ t + ε}` (`card_le_le_card_le_add`), `#{λ(T) < t} ≤ #{λ(S) < t + ε}` (`card_lt_le_card_lt_add`) and `#{λ(T) ≥ t} ≤ #{λ(S) ≥ t - ε}` (`card_ge_le_card_ge_sub`), each a Courant-Fischer count.

Theorem A5 itself has two parts. For S2, `weyl_transfer_enclosure`: if claimed values `μ_c` are matched to sorted indices `f c` of `H_h` with radii `r_c`, the same indices of `H_spec` lie within `r_c + Δ`, so a matching to distinct indices remains one. For S3, `weyl_transfer_count`: `#{λ(H_h) < σ - β} ≤ ν ≤ #{λ(H_h) ≤ σ + β}` gives `#{λ(H_spec) < σ - β - Δ} ≤ ν ≤ #{λ(H_spec) ≤ σ + β + Δ}`. The statement for S4 (separations shrink by `2Δ`) follows from the S2 part and is not stated separately.

Lemma A3 (`‖E‖₂ ≤ √(‖E‖₁ ‖E‖_∞) ≤ max(‖E‖₁, ‖E‖_∞)`) is `schur_bound` (`‖E x‖² ≤ R C ‖x‖²` for any bounds `R ≥ 0` on the row sums and `C` on the column sums, for a rectangular `E`) and `norm_le_max_of_row_col` (`‖E x‖ ≤ max(R, C) ‖x‖`, with `C ≥ 0`). The document's proof bounds `λ_max(EᴴE)` by an induced norm; the Lean proof is the Schur test, Cauchy-Schwarz row by row, and gives the same inequality.

Theorem A4 is `model_shift_bound`: with `D = H_h - H_ref` bounded by `R` and `C` and `‖(H_ref - H_spec) x‖ ≤ Δ_b ‖x‖`, `‖(H_h - H_spec) x‖ ≤ (max(R, C) + Δ_b) ‖x‖`. This is the composition step of the document. Its other ingredient, the inequality `Δ_builder = γ_W √(r c) ≥ ‖H_ref - H_spec‖₂`, rests on Lemmas A1 and A2, which bound the binary64 rounding of the builder; it enters the Lean theorem as the hypothesis on `Δ_b`. The status is therefore partial, and the missing part is named below.

### The polar-factor lemma

Document (`thm2_polar-factor.md` §1): `G = XᴴX` with `η = ‖G - I‖ < 1`, `M = G^{1/2}`, `Q̂ = XM⁻¹`; then `Q̂ᴴQ̂ = I` (a), `‖M - I‖ ≤ δ` and `‖M⁻¹‖ ≤ (1 - η)^{-1/2}` (b), `‖X - Q̂‖ ≤ δ` (c), and `‖AQ̂ - Q̂Λ‖ ≤ ρ` (d), with `δ = 1 - √(1 - η)` and `ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ`. Lean (`Cluster.lean`): `X` is a linear map `F → E`, `G` is any map with `⟪X u, X v⟫ = ⟪G u, v⟫` (that is `G = XᴴX`; for matrices `inner_toEuclideanLin_gram`), and `M` and `M⁻¹` are built in an orthonormal eigenbasis of `G` by multiplying the coefficients by `√g_k` and `1/√g_k` (`diagMap`, `polarM`, `polarMinv`). Part (a) is stated as `‖Q̂ z‖ = ‖z‖` (`polarQ_norm`), which is what Kahan's theorem uses; preservation of inner products, and with it `Q̂ᴴQ̂ = I`, is `polarQ_inner` and `adjoint_polarQ_polarQ` in `ClusterPrime.lean`. Part (b) is `sqrt_dev_le` with `polarM_sub_le` and `polarMinv_le`, part (c) is `sub_polarQ_le`, and part (d) is `polarQ_residual_le` with upper bounds `nR`, `nA ≥ 0` and `nΛ` on the three norms. The hypothesis `0 ≤ η` is added; in the document `η` is a norm.

### Kahan's inclusion theorem

Document (`thm2_polar-factor.md` §1 and §2): for `Q` with orthonormal columns, `H` Hermitian with eigenvalues `θ_1 ≤ … ≤ θ_g` and `r = ‖AQ - QH‖`, there are distinct indices `i_1, …, i_g` with `|λ_{i_j} - θ_j| ≤ r`; `H` need not be `QᴴAQ`. Lean, `kahan` (eigenvalues in increasing order) and `kahan_antitone` (Mathlib's decreasing order): for an isometry `Q` the conclusion is a strictly increasing index map, which is injective and therefore gives distinct indices; it is stronger than the document's statement. Step 1, the interval lemma (`kahan_interval_count`: `#{θ_l ∈ [a, b]} ≤ #{λ_i ∈ [a - r, b + r]}`), follows the document: it is the Courant-Fischer count for the image under `Q` of the eigenvectors of `H` with eigenvalues in `[a, b]`. Step 2 differs. The document builds a bipartite graph and applies Hall's marriage theorem after splitting a union of intervals into connected components. The Lean proof uses an explicit matching on the line instead (`exists_strictMono_matching`): the index of `θ_j` is `p_j = max_{l ≤ j} (#{i : λ_i < θ_l - r} + j - l)`, and the interval lemma for the window `[θ_l, θ_j]` shows `λ_{p_j} ≤ θ_j + r`. The reason for the change is that the explicit matching needs only counts of sorted sequences, where Hall's theorem needs the component decomposition, which is long to formalize; the explicit matching also yields an order-preserving pairing.

### Theorem 2

Document (`theory.md` §2, `thm2_polar-factor.md`, `thm:cluster`): under the hypotheses of the polar-factor lemma there are `g` distinct indices with `|λ_{i_j} - μ_j| ≤ ρ`, `μ_j` the eigenvalues of `Λ`. Lean, `cluster_enclosure`: `A` Hermitian `n × n`, `X : Matrix n m 𝕜`, `Λ` Hermitian `m × m`, `‖(XᴴX - I) z‖ ≤ η ‖z‖` with `0 ≤ η < 1`, and upper bounds `nR` on `‖AX - XΛ‖`, `nA ≥ 0` on `‖A‖` and `nΛ` on `‖Λ‖`; the conclusion is a strictly increasing map `σ` with `|λ_{σ(j)}(A) - μ_j| ≤ nR + (nA + nΛ)(1 - √(1 - η))`, both sequences in Mathlib's order `eigenvalues₀`. With the exact norms the radius is the document's `ρ`. The abstract form is `cluster_rho`, which is Kahan's theorem for `Q̂` and the block `Λ`.

### Theorem 2' and the radius min(ρ, ε*)

Document (`thm2prime_eps-star.md`, `lem:epsstar`): there are `g` distinct indices with `|λ_{i_j} - μ_j| ≤ ε* = ‖R‖/√(1 - η) + ‖XᴴR‖/(1 - η) + δ·spread(Λ)/√(1 - η)`, and `min(ρ, ε*)` is a valid common radius. Lean, `cluster_enclosure_eps_star`: the same with upper bounds `nR ≥ 0` on `‖AX - XΛ‖` and `nXR ≥ 0` on `‖Xᴴ(AX - XΛ)‖`, and with `spread(Λ)` replaced by `hi - lo` for any `lo ≤ hi` with `lo ≤ μ_j ≤ hi`; for `lo = μ_min` and `hi = μ_max` this is the document's statement. The proof has the document's three steps. Step (i), `rayleigh_residual_le`: `H′ = Q̂ᴴAQ̂` (`rayleighBlock`) is symmetric and `‖AQ̂ z - Q̂H′ z‖ ≤ nR (1 - η)^{-1/2} ‖z‖`, because `AQ̂ z - Q̂H′ z = (I - Q̂Q̂ᴴ)(R M⁻¹ z)` and `I - Q̂Q̂ᴴ` does not increase norms (`norm_sub_proj_le`). Step (ii), `rayleigh_sub_le`: `‖H′ z - Λ z‖ ≤ (nXR/(1 - η) + δ(hi - lo)/√(1 - η)) ‖z‖`, from `H′ - Λ = M⁻¹XᴴRM⁻¹ + (MΛM⁻¹ - Λ)` and `MΛy - ΛMy = (M - I)(Λ - c)y - (Λ - c)(M - I)y` with `c = (lo + hi)/2`. Step (iii): Kahan's theorem for `H′` and Weyl's inequality for `H′` and `Λ`, composed (`cluster_eps_star`). `cluster_min` takes the matchings of Theorem 2 and Theorem 2' and returns one with the radius `min(ρ, ε*)`.

### E-05: Rayleigh-Ritz and Temple

Document (`E-05_variational.md` §1): Lemma 1, `λ_1 ≤ θ ≤ λ_n` for `ψ ≠ 0`; Lemma 2, an eigenvalue within `r` of `θ` and `r² = ⟨ψ, H²ψ⟩/⟨ψ, ψ⟩ - θ²`; Proposition 1 (Temple), `θ < ρ ≤ λ_2` gives `λ_1 ≥ θ - r²/(ρ - θ)`. Lean (`Variational.lean`): `rayleigh_ritz_matrix` (some eigenvalue `≤ θ` and some `≥ θ`); Theorem 1 at `μ = θ` is `residual_enclosure`; the identity is `EigenData.residual_sq_eq` (`‖Hψ - θψ‖² = ‖Hψ‖² - θ²‖ψ‖²`) with `EigenData.norm_apply_sq_eq` (`‖Hψ‖² = re ⟨ψ, H²ψ⟩`); Temple is `temple_matrix`, where `ρ ≤ λ_2` is written as "at most one eigenvalue, counted with multiplicity, lies below `ρ`" and `λ_1` is an eigenvalue below all others. The Rayleigh quotient is given by `θ ‖ψ‖² = re ⟨ψ, Hψ⟩`. Lemmas 3 to 5 of the document are not formalized, so the status is partial.

### E-03: the Cholesky reduction of a pencil

Document (`E-03_generalized.md` §1): with `B = LLᴴ` and `C = L⁻¹AL⁻ᴴ`, `C` is Hermitian, the pencil eigenvalues are those of `C`, `‖L⁻¹v‖ = ‖v‖_{B⁻¹}` and `‖Lᴴx‖ = ‖x‖_B` (Lemma 1); for `x ≠ 0` and real `μ` some pencil eigenvalue satisfies `|λ - μ| ≤ ‖Ax - μBx‖_{B⁻¹}/‖x‖_B ≤ ‖Ax - μBx‖₂/(√β_c ‖x‖_B)` for `0 < β_c ≤ λ_min(B)` (Lemma 2). Lean (`Generalized.lean`): `L` is any invertible linear map with `B v = L(Lᴴ v)`, `C` is `reducedOp L A`, symmetric when `A` is (`reducedOp_isSymmetric`), and an eigenpair `(y, λ)` of `C` gives the pencil eigenpair `(L⁻ᴴ y, λ)` (`pencil_eigenpair_of_reduced`). `generalized_residual_enclosure` is the first inequality of Lemma 2 for the eigenvalues of `C`, written with `‖L⁻¹(Ax - μBx)‖` (the `B⁻¹`-norm) and `‖Lᴴx‖` (the `B`-norm, `norm_sq_adjoint_eq`); `generalized_residual_enclosure_beta` is the second, with `λ_min(B) ≥ β_c` written as `β_c ‖w‖² ≤ wᴴBw` for all `w`, through `norm_inv_le_of_lower_bound` (`‖L⁻¹s‖ ≤ ‖s‖/√β_c`). The converse half of Lemma 1, that every pencil eigenvalue (a root of `det(A - λB)`) is an eigenvalue of `C` with the same multiplicity, is not formalized, so the theorems are about the eigenvalues of `C` and the status is partial.

## What is not formalized, and why

Lemmas A1 and A2 of Appendix A, the rounding parts of Theorem A4, the rounding count of D-02 and Lemmas 3 and 4 of E-05 are statements about binary64 arithmetic: they apply the standard model `fl(a ∘ b) = (a ∘ b)(1 + δ)`, `|δ| ≤ u`, and Higham's constants `γ_m` to the operations of the code. A formal proof needs a formal floating-point model in Lean with those lemmas (Higham 2002, Lemmas 3.1 and 3.3), which this library does not have. They were not attempted, because the result would be a proof about a model of the code's arithmetic, and that model, not the mathematics of the enclosures, would carry the trust.

Lemma 5 of E-05 (the change of the Rayleigh quotient between the source state and the contracted vector), the exact part of D-02 (`‖H‖₂ ≤ ∑_t |c_t| ∏_l ‖o_{t,l}‖₂`, which needs the norm of Kronecker products and of Jordan-Wigner strings), the converse half of Lemma 1 of E-03 (the pencil eigenvalues, as roots of `det(A - λB)`, are exactly the eigenvalues of `C` with multiplicity) and Lemma 17 (Taylor and atanh remainders of real `exp` and `ln` and their dyadic rounding) are exact mathematics. They were not attempted: the time available was spent on the statements that S2 uses directly (Theorems 1 to 3, Kahan's theorem and Appendix A), which come first in the order of the certificate, and on the exact parts of E-05 and E-03 that rest on them.

## How the files were checked

Each file was elaborated with `lake env lean` until it gave no error, then built as a module. The nine files contain no `sorry`, no `admit`, no `native_decide` and no `axiom` declaration; this was checked by searching the files. `#print axioms` was run for every theorem of the nine files by `reverification_lean/Drafts/Enclosure/PrintAxioms.lean`; every one depends exactly on `propext`, `Classical.choice` and `Quot.sound`. The second checker of the toolchain, `leanchecker`, replayed the eight modules from `Spectral` to `Variational` and the root module as they stood before `Generalized.lean` was added, and accepted them (exit code 0, no message); a second run replayed `Generalized` and accepted it as well. Every theorem of the nine files is therefore at level K2. The counts, the per-theorem axioms, the build command and its time, and the second-checker runs with their times are in `reverification_lean/results/mathlib_enclosure.json`.

## Recreating the check

In a Windows command window (`cmd.exe`), with the environment of `reverification_lean/README.md` (the Lean binaries and Git on `PATH`, `MATHLIB_CACHE_DIR` set), from the folder `reverification_lean`:

```
lake build CertifyMathlib.Enclosure
lake env lean Drafts/Enclosure/PrintAxioms.lean
```

The first command builds the nine modules and the root module; the second prints the axioms of every theorem. The second checker is run with `lake env leanchecker CertifyMathlib.Enclosure.Spectral CertifyMathlib.Enclosure.Residual` and so on for every module name; the times of the two runs on the shared build machine are recorded in the results file.
