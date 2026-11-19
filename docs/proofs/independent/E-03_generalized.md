# Independent re-derivation: E-03 (generalized eigenproblems A x = λ B x, B ≻ 0)

Not a human check. Written from sections 1 and 3 of `docs/proofs/E-03_generalized.md` only (section 2 read afterwards;
comparison at the end). Notation as there. Classical facts: Cholesky factorisation; the residual theorem (Parlett Thm
4.5.1); Sylvester's law of inertia; Ostrowski's theorem (Horn & Johnson Thm 4.5.9); Weyl; Higham (2002) eq. (3.5), Lemma
3.3; the contracts of `fp.exact_dot` (thm1 note) and of `inertia.inertia_dense` (β ≥ ‖LDLᴴ − P(A − σI)Pᵀ‖₂, with the
rounding of forming the shifted matrix included — theory.md Lemma 4, taken as a hypothesis here).

## Proofs

**Lemma 1.** B = LLᴴ with L nonsingular (B ≻ 0). C = L⁻¹AL⁻ᴴ is Hermitian. Ax = λBx ⇔ L⁻¹A L⁻ᴴ(Lᴴx) = λ Lᴴx, and
y = Lᴴx ≠ 0 ⇔ x ≠ 0; so the pencil eigenpairs (x, λ) correspond bijectively to the eigenpairs (Lᴴx, λ) of C, and
det(A − λB) = |det L|² det(C − λI) gives the multiplicities. ‖L⁻¹‖₂² = ‖L⁻¹L⁻ᴴ‖₂ = ‖B⁻¹‖₂ = 1/λ_min(B);
‖L⁻¹v‖² = vᴴL⁻ᴴL⁻¹v = vᴴB⁻¹v; ‖Lᴴx‖² = xᴴLLᴴx = xᴴBx. ∎

**Lemma 2.** With y = Lᴴx: Cy − μy = L⁻¹AL⁻ᴴLᴴx − μLᴴx = L⁻¹(Ax − μBx) = L⁻¹s. The residual theorem for the Hermitian
C gives an eigenvalue λ of C (= of the pencil) with |λ − μ| ≤ ‖L⁻¹s‖/‖y‖ = ‖s‖_{B⁻¹}/‖x‖_B, and ‖L⁻¹s‖ ≤ ‖L⁻¹‖₂‖s‖₂ =
‖s‖₂/√λ_min(B) ≤ ‖s‖₂/√β_c for 0 < β_c ≤ λ_min(B). ∎

**Lemma 3.** A − σB = L(C − σI)Lᴴ is a congruence, so ν(A − σB) = ν(C − σI) = #{λ_j < σ} = N(σ) (Sylvester). For the
quantitative part use Ostrowski: the sorted eigenvalues satisfy λ_j(A − σB) = θ_j(λ_j − σ) with θ_j ∈ [λ_min(B),
λ_max(B)] (λ_j the pencil eigenvalues, sorted alike). Weyl for M′: ν(M′) ≥ #{λ_j(A − σB) < −b} and
ν(M′) ≤ #{λ_j(A − σB) ≤ b}. If λ_j < σ − b/β_c then λ_j − σ < −b/β_c ≤ −b/θ_j, so θ_j(λ_j − σ) < −b: N(σ − b/β_c) ≤ ν.
If θ_j(λ_j − σ) ≤ b then either λ_j ≤ σ or λ_j − σ ≤ b/θ_j ≤ b/β_c: ν ≤ #{λ_j ≤ σ + b/β_c}. ∎

**Proposition 1.** Let M′ := PᵀLDLᴴP be the factorised matrix, ‖M′ − (B − βI)‖₂ ≤ b. ν(M′) = ν(D) = 0 (Sylvester; D
block diagonal, its inertia counted exactly by `_count_block_diagonal`), so M′ ⪰ 0 and by Weyl every eigenvalue of
B − βI is ≥ −b: λ_min(B) ≥ β − b = β_c. If at shift 0 ν(M′) = ν ≥ 1 then M′ has ν negative eigenvalues and by Weyl B
has ν eigenvalues ≤ b. ∎

**Proposition 2.** Standard model, n terms per inner product (dense A, B). |a − Ax| ≤ γ̇_n|A||x| and |b′ − Bx| ≤ γ̇_n|B||x|
entrywise. c = fl(μb′) = μb′(1 + δ), |δ| ≤ u: |c − μBx| ≤ |μ|(|b′ − Bx| + u|b′|) ≤ |μ|(γ̇_n + u(1 + γ̇_n))|B||x| ≤
|μ|γ̇_{n+1}|B||x| (γ_a + u + uγ_a ≤ γ_{a+1}, likewise for 2γ_{a+4}). ŝ = fl(a − c) = (a − c)(1 + δ′): |ŝ − (a − c)| ≤
u|a − c| ≤ u|ŝ|/(1 − u) ≤ u(1 + 2u)|ŝ|. Summing: |ŝ − s| ≤ γ̇_n|A||x| + |μ|γ̇_{n+1}|B||x| + u(1 + 2u)|ŝ|. The
non-negative products â, b̂ underestimate |A||x|, |B||x| by at most a relative γ_n (real data; for complex data the
moduli add ≤ 2u per product, absorbed by the factor-2 slack of γ̇_n = 2γ_{n+4} against the true complex constant
≤ γ_{n+2}), whence E. ‖s‖₂ ≤ ‖ŝ‖₂ + ‖ŝ − s‖₂ ≤ ‖ŝ‖₂ + ‖E‖₂ ≤ S (norm2_upper, `up`). Denominator: xᴴBx = Re⟨x, b′⟩ −
Re⟨x, b′ − Bx⟩ ≥ Re D̂ − e_D − |x|ᵀ|b′ − Bx| ≥ Re D̂ − e_D − γ̇_n|x|ᵀ|B||x|, and |x|ᵀ|B||x| ≤ fl(|x|ᵀb̂)(1 + γ_{n+2})/(1 − γ_n)
(non-negative inner products of n terms) — so xᴴBx ≥ D_lo. If D_lo > 0: ‖s‖_{B⁻¹}/‖x‖_B ≤ ‖s‖₂/(√β_c‖x‖_B) ≤
S/√(β_c D_lo). The computed denominator fl(√(fl(β_c D_lo))) is within (1 + u)^{3/2} of √(β_c D_lo) and the factor
(1 − 4u) (one more rounding) brings it below; the division and `up` give r ≥ S/√(β_c D_lo). Lemma 2 closes. ∎

**Proposition 3.** M̂ = fl(A − fl(σB)) is exactly Hermitian (entrywise rounding commutes with conjugation; A, B exactly
Hermitian by `_dense_hermitian`). |M̂ − (A − σB)| ≤ u|σ||B| + u|A − fl(σB)| ≤ u|σ||B| + u(1 + 2u)|M̂| entrywise; the
matrix on the right is symmetric, so its 2-norm is ≤ its ∞-norm; the u² shortfall of the code's u|M̂| is covered by
the `up` guard on b (b ≥ u‖|M̂|‖_∞). Hence ‖M′ − (A − σB)‖₂ ≤ b_f + ‖M̂ − (A − σB)‖₂ ≤ b and Lemma 3 gives
N(σ − c) ≤ ν ≤ #{λ_j ≤ σ + c} with c = b/β_c (rounded up). Now let the k enclosures be pairwise disjoint with
T ≤ σ − c. By Lemma 2 each contains a pencil eigenvalue, and disjointness makes these k eigenvalues distinct, all
< T ≤ σ − c, so N(σ − c) ≥ k. If ν = k: N(σ − c) ≤ k, hence exactly k eigenvalues lie below σ − c, each enclosure holds
exactly one (k eigenvalues distributed over k disjoint enclosures that each hold ≥ 1), they are λ₁, …, λ_k, and
λ_{k+1} ≥ σ − c. If ν > k: at least ν eigenvalues are ≤ σ + c, of which k are matched to the enclosures, so ≥ ν − k are
not (they may lie in the collar or in an enclosure as a second member): no completeness statement. If ν < k with
disjoint enclosures below σ − c, N(σ − c) ≥ k > ν ≥ N(σ − c) is a contradiction: this case cannot occur under correct
inputs and signals an inconsistency (a false β_c or a wrong inertia); the code reports INCONCLUSIVE, which is safe
(remark: the message "some claimed enclosures share an eigenvalue" explains the overlapping case, not the disjoint
one). Overlapping enclosures: Lemma 2 gives no distinctness; INCONCLUSIVE. ∎

**Hypothesis path.** A declared β_c enters Lemma 2 and the collar only through 0 < β_c ≤ λ_min(B); if the
declaration is false, r and c are too small — hence CERTIFIED_UNDER_HYPOTHESIS and the named hypothesis. Refusal of
an indefinite B (conclusive inertia with ν ≥ 1 at shift 0) is independent of declarations; by Proposition 1 B then
has ν eigenvalues ≤ b, which does not exclude λ_min(B) ∈ (0, b] — the refusal is conservative (no false statement is
made, a positive definite but nearly singular B may be refused). ∎

## Code (current `certify_ed/generalized.py`)

* `_dense_hermitian` 49–61: exact Hermiticity of A and B enforced (56–58), finite entries.
* Proposition 1: `certify_B_positive_definite` 71–110: inertia at 0 (81), refusal when conclusive with ν ≥ 1 (83–87),
  β = f·λ̂_min with f = 1/2 then ×1/4 per retry (97–108), β_c = sub_down(β, b) then −4u relative (102–103),
  ok only if β_c > 0.
* Proposition 2: `pencil_residual_radii` 115–152: γ̇_n, γ̇_{n+1}, γ_n (125–127), AX, BX dense products (128–129), â/(1 − γ_n),
  b̂/(1 − γ_n) (130–131), c, ŝ, E (137–139), S = up(norm2_upper(ŝ) + norm2_upper(E)) (141), D̂ via exact_dot (142–143),
  err = up(e_D + γ̇_n fl(|x|ᵀb̂)(1 + γ_{n+2})) (144; b̂ already divided by 1 − γ_n), D_lo (146), den = √(β_c·max(D_lo, 0))(1 − 4u)
  (149), r = up(S/den) or ∞ (150–151).
* Lemma 3 / Proposition 3: `pencil_count_below` 157–169: M̂ (161), `inertia_dense(M̂, 0)` (162), form = inf_norm_upper(u|σ||B|
  + u|M̂|) (163), b = up(β_f + form) (164), collar up (165), σ ∓ c by sub_down/add_up (168–169). Decision in
  `certify_generalized` 292–359: sorted enclosures, `disjoint` = strict separation of consecutive outward ends (303),
  T = max upper (304), shifts between T and the computed λ̂_{k+1} (306–317; a hint only), skip when the collar
  reaches T (323–324), ν = k ∧ disjoint → status `under` (326–334; λ_{k+1} ≥ σ − c recorded), ν < k (335–340),
  ν > k with `missing_at_least` (341–348), overlap (349–354), none conclusive (355–358). G2 only when all radii
  finite (277–290). Hypothesis path 244–252 (`declared_lambda_min_B`, `under` = CERTIFIED_UNDER_HYPOTHESIS at 267);
  refusal of an indefinite B at 226–233; `max_dim` at 214–221.
* Section 3's line numbers agree with the current file (shifts ≤ 2).

### Comparison with section 2 (read after the above)

Lemmas 1–2, Propositions 1–3 agree with my derivations. Section 2 proves the quantitative part of Lemma 3 by the
Loewner sandwich M − (b/β_c)B ⪯ M′ ⪯ M + (b/β_c)B and the antitonicity of ν; I used Ostrowski's theorem — both are
correct and give the same inequalities. Rounding directions in the code as stated.

**Remarks.** (1) Proposition 3, forming M̂: "the subtraction adds ≤ u|M̂_ij|" is u|a − c| ≤ u(1 + 2u)|M̂_ij|; the 2u²|M̂|
shortfall is covered by the `up` guard on b (b ≥ u‖|M̂|‖_∞), as in my proof — a wording fix. (2) The ν < k paragraph:
with disjoint enclosures below σ − c the case cannot occur under correct inputs (N(σ − c) ≥ k > ν ≥ N(σ − c)); the text
"the claim's vectors must be duplicates up to the radii" describes overlapping enclosures, which the branch does not
see (`disjoint` is not re-checked there). The status INCONCLUSIVE is safe either way; the message could say
"inconsistent inputs (a false β_c or inertia)". (3) Refusing a B whose inertia at 0 is conclusive with ν ≥ 1 is
conservative: Proposition 1 shows λ(B) ≤ b for ν eigenvalues, not λ_min(B) ≤ 0; the record never states a false
claim. (4) The contract of `inertia.inertia_dense` (β ≥ ‖LDLᴴ − P(A − σI)Pᵀ‖₂ incl. forming the shift) is a
hypothesis from theory.md Lemma 4 here, not re-derived.

**Verdict: AGREES-WITH-REMARKS.**

### Runs

* `python docs/proofs/verify_E-03.py --cases 60 --seed 7`: A 170, B 58 certified records vs the exact spectrum, C 75
  adversarial (+ negative control 30/30); 228 random + 75 adversarial cases, 0 violations (211 s).
* `python docs/proofs/independent/adversarial_E-03_generalized.py --cases 80 --seed 97`: Proposition 2 exact-rational
  checks 80 (S ≥ ‖s‖, D_lo ≤ xᴴBx; κ(B) up to 1e8, near-eigenvectors with residual 1e-13, scales 10^±3) and mp radius
  checks 80 (r ≥ ‖s‖_{B⁻¹}/‖x‖_B, eigenvalue within r), Proposition 1 62 (κ(B) up to 1e15; 18 skipped: inertia
  inconclusive) + 62 indefinite-B negative controls, Lemma 3 160 shifts (80 next to an eigenvalue), end to end 80 (G3
  CERTIFIED in all 80: exactly the k lowest, λ_{k+1} ≥ the recorded bound), 172 negative controls (skipped eigenvalue,
  duplicated vector, false declared λ_min(B) on a near-singular B: never CERTIFIED); 0 violations (207 s).
