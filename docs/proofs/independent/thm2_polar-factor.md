# Independent re-derivation: Theorem 2 (cluster radius ρ) and the polar-factor lemma

Fresh-context session, 2026-10-30. Protocol (D-12): sections 1 and 3 of `../thm2_polar-factor.md` were read
first; this proof was written before section 2 was opened; the comparison follows at the end.
Notation as in `docs/theory.md`: A Hermitian n × n, λ₁ ≤ … ≤ λₙ, ‖·‖ the spectral norm.

## Statement checked

X ∈ ℂⁿˣᵍ, G = XᴴX, η = ‖G − I‖ < 1, Λ ∈ ℂᵍˣᵍ Hermitian with eigenvalues μ₁ ≤ … ≤ μ_g, R = AX − XΛ,
δ = 1 − √(1 − η), ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ. Claims: polar-factor lemma (a)–(d), Kahan's inclusion theorem for an
arbitrary Hermitian block H, Theorem 2 (g distinct indices with |λ_{i_j} − μ_j| ≤ ρ).

## My proof

### Polar-factor lemma

(a) G is Hermitian positive semidefinite. Its eigenvalues σ satisfy |σ − 1| ≤ ‖G − I‖ = η < 1, so
σ ∈ [1 − η, 1 + η] ⊂ (0, 2): G is positive definite, M := G^{1/2} (unique Hermitian positive definite square
root, Horn–Johnson, *Matrix Analysis*, Thm 7.2.6) exists and is invertible, and M commutes with G.
Q̂ := XM⁻¹ gives Q̂ᴴQ̂ = M⁻¹ XᴴX M⁻¹ = M⁻¹ M² M⁻¹ = I.

(b) By the spectral mapping theorem the eigenvalues of M are √σ with σ ∈ [1 − η, 1 + η], hence lie in
[√(1 − η), √(1 + η)]. M − I is Hermitian with eigenvalues √σ − 1, so
‖M − I‖ = max_σ |√σ − 1| = max(1 − √(1 − η), √(1 + η) − 1) = 1 − √(1 − η) = δ,
because √(1 + η) + √(1 − η) ≤ 2 (concavity of √, Jensen with equal weights). ‖M⁻¹‖ = 1/λ_min(M) ≤ (1 − η)^{−1/2}.

(c) X − Q̂ = Q̂M − Q̂ = Q̂(M − I); since Q̂ has orthonormal columns ‖Q̂B‖ = ‖B‖ for every B, so
‖X − Q̂‖ = ‖M − I‖ ≤ δ.

(d) AQ̂ − Q̂Λ = (AX − XΛ) − A(X − Q̂) + (X − Q̂)Λ. Triangle inequality and submultiplicativity:
‖AQ̂ − Q̂Λ‖ ≤ ‖R‖ + ‖A‖‖X − Q̂‖ + ‖X − Q̂‖‖Λ‖ ≤ ‖R‖ + (‖A‖ + ‖Λ‖)δ = ρ.

### Kahan's inclusion theorem (arbitrary Hermitian block)

Let Q ∈ ℂⁿˣᵍ have orthonormal columns, H Hermitian g × g with eigenvalues θ₁ ≤ … ≤ θ_g, r = ‖AQ − QH‖.

*Counting lemma.* Let Θ be the set of eigenvalues of H in an interval [a, b] (with multiplicity), m = |Θ| ≥ 1,
Y ∈ ℂᵍˣᵐ orthonormal eigenvectors of H for them, D = diag(Θ). Then A has at least m eigenvalues (with
multiplicity) in [a − r, b + r].
Proof. Z := QY has orthonormal columns and AZ − ZD = (AQ − QH)Y, so ‖AZ − ZD‖ ≤ r. Let W be the span of
the eigenvectors of A whose eigenvalues lie outside [a − r, b + r], and suppose fewer than m eigenvalues lie
inside, so dim W > n − m. Then dim range(Z) + dim W > n and there is a unit z = Zy ∈ W (‖y‖ = 1). With
c = (a + b)/2: ‖(A − c)z‖ ≤ ‖(AZ − ZD)y‖ + ‖Z(D − cI)y‖ ≤ r + (b − a)/2. But z ∈ W expands in eigenvectors
with |λ − c| > (b − a)/2 + r, so ‖(A − c)z‖² = Σ|λ − c|²|⟨v_λ, z⟩|² > ((b − a)/2 + r)². Contradiction. ∎

*Matching.* Bipartite graph: left vertices j = 1..g (eigenvalues of H), right vertices i = 1..n (indices of
eigenvalues of A), edge (j, i) iff |λ_i − θ_j| ≤ r. Hall's condition: for S ⊂ {1..g}, the closed intervals
[θ_j − r, θ_j + r], j ∈ S, split into connected components (maximal chains of pairwise-overlapping intervals);
component k covers [a_k − r, b_k + r] with a_k = min, b_k = max of its θ's, has m_k members from S, and the
components' intervals are pairwise disjoint. All eigenvalues of H in [a_k, b_k] number m'_k ≥ m_k; the counting
lemma gives ≥ m'_k ≥ m_k eigenvalue indices in [a_k − r, b_k + r] ⊂ N(S). Disjointness gives |N(S)| ≥ Σ_k m_k =
|S|. Hall's marriage theorem (P. Hall 1935; e.g. Diestel, *Graph Theory*, Thm 2.1.2) yields a matching saturating
the left side: distinct i_j with |λ_{i_j} − θ_j| ≤ r. ∎ (Kahan 1967, *Inclusion theorems for clusters of
eigenvalues of Hermitian matrices*; Parlett, *The Symmetric Eigenvalue Problem*, Thm 11.5.1 — stated there for any
Hermitian H, not only the Rayleigh quotient.)

### Theorem 2

By the lemma Q̂ has orthonormal columns and ‖AQ̂ − Q̂Λ‖ ≤ ρ. Kahan's theorem with H = Λ (Hermitian, eigenvalues
μ_j) gives g distinct indices with |λ_{i_j} − μ_j| ≤ ρ. ∎ The hypothesis η < 1 is used only in (a) (invertibility
of G); it is necessary: X = [x, x] with ‖x‖ = 1 gives η = 1, R = 0 when x is an eigenvector, and no two distinct
eigenvalue indices need lie within 0 of μ.

### Remark ε ≥ ρ

ε = (w + √(w² + 4ρ²))/2 ≥ (0 + 2ρ)/2 = ρ for every w ≥ 0.

### Corollaries used by section 3 (stated there without proof text in sections 1/3)

(i) With `iso` ≥ ‖VᴴV − I‖ and `inv` ≥ ‖HV − VH_s‖, the same algebra with (X, Λ) = (V, H_s) gives
‖HV̂ − V̂H_s‖ ≤ inv + (‖H‖ + ‖H_s‖)δ(iso). (ii) For W = [V₁ ⋯ V_m] and B = ⊕H_s, ‖WᴴW − I‖_F² =
Σ_s‖V_sᴴV_s − I‖_F² + Σ_{s≠t}‖V_sᴴV_t‖_F², bounded by ‖(iso_1..iso_m, orth × m(m − 1))‖₂ when each
off-diagonal block has Frobenius norm ≤ orth; ‖HW − WB‖_F² = Σ_s ‖HV_s − V_sH_s‖_F². Both are the Frobenius
bound ‖·‖₂ ≤ ‖·‖_F followed by the lemma.

## Comparison with section 2 of the original (read after the proof above was written)

Step by step: (a)–(d) identical in content (same decomposition of AQ̂ − Q̂Λ, same concavity argument for δ).
Kahan: the original's Step 1 proves the interval lemma with Courant–Fischer on (A − cI)²; mine uses the
dimension-intersection argument; both are valid and give the same count. Step 2 (connected components of the
union of intervals, Hall's theorem) is the same argument as mine. Theorem 2 and the remark ε ≥ ρ agree. Constant: ρ
in both. Corollaries (i), (ii) as I reconstructed them.

**Code mapping (current `dev/3.2` lines).** `enclosure._cluster_bound` 458–513: η exact path 476, BLAS path
481, guard `eta < 1` 482, δ 486–487, ‖R‖ 489, ‖A‖ 490, ‖Λ‖ 493, ρ 494, min(ρ, ε*) 511. `certify._subspace_shift`
296–303, `sectors.decomposition_shift` 125–148. Rounding directions checked by hand against `fp.up` (93–95,
one rounding, (1 + 8u)), `fp.norm2_upper` (149–155), `fp.fro_upper` (165–170): every factor is rounded towards
+∞; δ: 1 − fl(√fl(1 − e)) ≥ δ(e) − 2.5u − O(u²) (I count the subtraction's rounding at u, not 1.1u), `up` adds
≥ 6u relative, `+ 4u` absolute closes it since δ ≤ 1 — agrees. The additive `+ ULP_MIN`/`gram_additive_term`
terms outside `up` are lost to rounding when the main term is normal-sized, but then the ≥ 4u relative slack of
`up` (after its ≤ 3 roundings) exceeds 2⁻¹⁰⁷⁵ whenever the main term is ≥ 2⁻¹⁰²², and below that the addition is
exact — sound. Section 3 cites the line numbers of `track/backlog-proofs`; they moved by the merges (listed above).

**Remarks.** (1) The code's η is a Frobenius bound ‖G − I‖_F; cases with ‖G − I‖₂ < 1 ≤ ‖G − I‖_F are declined
(my adversarial run: 140 of 391 cases) — conservative, not a violation. (2) `sectors.audit_orthogonality`
(sectors.py 74–82) returns `up(fro_upper(V_aᴴV_b))` for the *computed* sparse product, with no term for the
rounding error of that product (unlike `iso` in `audit_sector` line 60, which adds γ_dot(w)‖V‖_F²). Section 3
asserts η = norm2_upper([iso_s, orth × m(m − 1)]) ≥ ‖WᴴW − I‖_F; this needs `orth` ≥ ‖V_aᴴV_b‖_F for the *stored*
V's, which the missing rounding term does not guarantee when the sectors overlap in support. This concerns the
Theorem 8 constant (thm8_rho.md), not Theorem 2 itself; recorded in VERDICTS.md under Required corrections.

**Verdict: AGREES-WITH-REMARKS** (proof valid, same constant; remark (2) is a code-side gap in an input of the
corollary, filed under thm8_rho).

## Runs

- `verify_thm2_polar-factor.py --cases 2000 --seed 7`: lemma 3452 random + 860 adversarial (444 skipped), Kahan
  1726 + 430, Theorem 2 1726 + 430, Kahan random block 500 + 600, implemented bounds 3291 + 809 (212 skipped);
  0 violations (`docs/proofs/independent/runs/` is not kept; counts recorded in VERDICTS.md).
- `adversarial_thm2_polar-factor.py --cases 500 --seed 3`: lemma 36 random + 355 adversarial (109 skipped,
  η ≥ 1), Kahan 36 + 355, Theorem 2 36 + 355, implemented bounds 26 + 225 (140 skipped, Frobenius η ≥ 1),
  `_subspace_shift` 36 + 355, negative controls 10 + 119 (duplicate columns and η ≥ 1: code returns None, exact
  matching distance > 0); 0 violations.
