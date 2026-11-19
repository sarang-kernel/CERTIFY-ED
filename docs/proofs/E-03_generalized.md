# E-03 Generalized eigenproblems A x = λ B x with B positive definite: residual enclosures in the B-norm and completeness by inertia

Notation as in `docs/theory.md`: A, B Hermitian n × n, B positive definite with eigenvalues β₁ = λ_min(B) ≤ … ≤ λ_max(B);
the pencil's eigenvalues λ₁ ≤ … ≤ λ_n are the roots of det(A − λB); (x, μ) a claimed pair, x ≠ 0, μ real; s = Ax − μBx;
‖v‖_B = (vᴴBv)^{1/2}, ‖v‖_{B⁻¹} = (vᴴB⁻¹v)^{1/2}; u = 2⁻⁵³, γ_m, γ̇_m as in `fp.py`; N(τ) := #{j : λ_j < τ} (pencil
eigenvalues strictly below τ), ν(M) := number of negative eigenvalues of a Hermitian M.

## 1 Statement

**Lemma 1 (Cholesky reduction; Parlett 1998 §15.2, Stewart & Sun 1990 Thm VI.1.15).** Let B = LLᴴ (Cholesky, L
nonsingular lower triangular) and C := L⁻¹AL⁻ᴴ. Then C is Hermitian, the eigenvalues of the pencil (A, B) are exactly
the eigenvalues of C (with multiplicity), and (x, λ) is an eigenpair of the pencil iff (Lᴴx, λ) is an eigenpair of C.
Moreover ‖L⁻¹‖₂² = ‖B⁻¹‖₂ = 1/λ_min(B), ‖L⁻¹v‖ = ‖v‖_{B⁻¹} and ‖Lᴴx‖ = ‖x‖_B for every v, x.

**Lemma 2 (residual enclosure in the B-norm).** For every x ≠ 0 and real μ some eigenvalue λ of the pencil satisfies

    |λ − μ| ≤ ‖Ax − μBx‖_{B⁻¹} / ‖x‖_B ≤ ‖Ax − μBx‖₂ / ( √β_c · ‖x‖_B )

for every β_c with 0 < β_c ≤ λ_min(B).

**Lemma 3 (counting by inertia of A − σB).** For real σ, ν(A − σB) = N(σ) (Sylvester's law of inertia for the congruence
A − σB = L(C − σI)Lᴴ). If a Hermitian M′ with ‖M′ − (A − σB)‖₂ ≤ b has ν(M′) = ν, then

    N(σ − b/β_c) ≤ ν ≤ #{j : λ_j ≤ σ + b/β_c}.

**Proposition 1 (G1, certified lower bound on λ_min(B); `generalized.certify_B_positive_definite`).** Let β > 0 and let the
Bunch–Kaufman factorization of B − βI (`inertia.inertia_dense`) return the count ν = 0 of negative pivots with backward
error b (‖LDLᴴ − P(B − βI)Pᵀ‖₂ ≤ b). Then λ_min(B) ≥ β − b =: β_c. If instead the factorization of B itself (shift 0)
returns ν ≥ 1 negative pivots, B has at least ν eigenvalues ≤ b (B is not certified positive definite; the record
refuses the pencil).

**Proposition 2 (G2; `generalized.pencil_residual_radii`).** With a = fl(Ax), b′ = fl(Bx), c = fl(μb′), ŝ = fl(a − c)
and the computed absolute products â = fl(|A||x|), b̂ = fl(|B||x|):

    |ŝ − s| ≤ E := γ̇_n â/(1 − γ_n) + |μ| γ̇_{n+1} b̂/(1 − γ_n) + u(1 + 2u)|ŝ|   entrywise,

so ‖s‖₂ ≤ S := up(norm2_upper(ŝ) + norm2_upper(E)); and with D̂ = `exact_dot`(x, b′) (error e_D),
xᴴBx ≥ D_lo := Re D̂ − up(e_D + γ̇_n fl(|x|ᵀb̂)(1 + γ_{n+2})/(1 − γ_n)). If D_lo > 0 then

    r := up( S / ( fl(√(fl(β_c D_lo))) (1 − 4u) ) ) ≥ ‖s‖₂/(√β_c ‖x‖_B),

and by Lemma 2 some pencil eigenvalue lies in [μ − r, μ + r].

**Proposition 3 (G3, completeness of a 'lowest' claim; `generalized.pencil_count_below`).** Let the k claimed
enclosures [μ_i − r_i, μ_i + r_i] be pairwise disjoint with top T = max(μ_i + r_i), let σ be a shift, M̂ = fl(A − fl(σB)),
and let `inertia_dense`(M̂, 0) return ν negative pivots with backward error b_f. Put b := up(b_f + ‖u|σ||B| + u|M̂|‖_∞) and
the collar c := up(b/β_c). If σ − c ≥ T and ν = k, then exactly k pencil eigenvalues lie below σ − c, each enclosure
contains exactly one of them, they are λ₁, …, λ_k, and λ_{k+1} ≥ σ − c. If ν > k, at least ν − k eigenvalues ≤ σ + c are
not accounted for by the enclosures (INCONCLUSIVE, `missing_at_least`); if ν < k the enclosures do not hold k distinct
eigenvalues (INCONCLUSIVE); overlapping enclosures give no distinctness statement (INCONCLUSIVE, NOT_IMPLEMENTED: the
cluster bound of Theorem 2 is not derived for pencils here).

**Hypothesis path.** When Proposition 1 cannot certify a β_c (inconclusive factorization), a user-declared β_c > 0
(`lambda_min_B`, with its source text) is recorded as the hypothesis `declared_lambda_min_B`; G1, G2 and G3 are then
CERTIFIED_UNDER_HYPOTHESIS, never CERTIFIED. A B whose inertia at 0 is conclusive with ν ≥ 1 is refused regardless of
any declaration.

## 2 Proof

**Lemma 1.** L exists and is nonsingular because B ≻ 0 (Horn & Johnson 2013, Cor. 7.2.9). C = L⁻¹AL⁻ᴴ is Hermitian
since A is. det(A − λB) = det(L(C − λI)Lᴴ) = |det L|² det(C − λI), so the pencil and C have the same characteristic
polynomial, hence the same eigenvalues with multiplicity. Ax = λBx ⟺ L⁻¹Ax = λLᴴx ⟺ C(Lᴴx) = λ(Lᴴx) (write x = L⁻ᴴy).
‖L⁻¹v‖² = vᴴL⁻ᴴL⁻¹v = vᴴ(LLᴴ)⁻¹v = vᴴB⁻¹v; ‖Lᴴx‖² = xᴴLLᴴx = xᴴBx; ‖L⁻¹‖₂² = ‖L⁻¹L⁻ᴴ‖₂ = ‖B⁻¹‖₂ = 1/λ_min(B). ∎

**Lemma 2.** Put y = Lᴴx ≠ 0. Then Cy − μy = L⁻¹AL⁻ᴴLᴴx − μLᴴx = L⁻¹(Ax − μBx) = L⁻¹s (using L⁻¹Bx = L⁻¹LLᴴx = Lᴴx).
Theorem 1 (`docs/theory.md` §2) for the Hermitian C and the vector y gives an eigenvalue λ of C — by Lemma 1 of the
pencil — with |λ − μ| ≤ ‖Cy − μy‖/‖y‖ = ‖L⁻¹s‖/‖Lᴴx‖ = ‖s‖_{B⁻¹}/‖x‖_B. Finally ‖L⁻¹s‖ ≤ ‖L⁻¹‖₂‖s‖₂ = ‖s‖₂/√λ_min(B) ≤
‖s‖₂/√β_c. ∎

**Lemma 3.** A − σB = L(C − σI)Lᴴ is a congruence (L nonsingular), so by Sylvester's law of inertia (Horn & Johnson
Thm 4.5.8) ν(A − σB) = ν(C − σI) = #{eigenvalues of C below σ} = N(σ). Now let M = A − σB and ‖M′ − M‖₂ ≤ b. Then
M − bI ⪯ M′ ⪯ M + bI, and since B ⪰ β_c I, bI ⪯ (b/β_c)B, so M − (b/β_c)B ⪯ M′ ⪯ M + (b/β_c)B. The map X ↦ ν(X) is
antitone (X ⪯ Y ⟹ ν(X) ≥ ν(Y), Weyl's monotonicity theorem). Hence ν(M + (b/β_c)B) ≤ ν(M′) ≤ ν(M − (b/β_c)B), and
M ± (b/β_c)B = A − (σ ∓ b/β_c)B, whose ν is N(σ ∓ b/β_c) by the first part. The upper member: N(σ + b/β_c) counts
eigenvalues *strictly* below σ + b/β_c, which is ≤ #{λ_j ≤ σ + b/β_c}. ∎

**Proposition 1.** `inertia_dense` computes ν = ν(LDLᴴ) with ‖LDLᴴ − P(B − βI)Pᵀ‖₂ ≤ b (its backward error, including the
rounding of forming B − βI; `docs/theory.md` §3). By Weyl, the eigenvalues of B − βI and of the congruent-to-D matrix
differ by at most b, so #{eigenvalues of B − βI below −b} ≤ ν = 0: every eigenvalue of B − βI is ≥ −b, i.e. λ_min(B) ≥
β − b. The code forms β − b with `sub_down` and then steps a relative 4u down (one more rounding than needed). For the
shift 0: ν ≥ 1 negative pivots means #{λ(B) ≤ b} ≥ ν ≥ 1 (the other half of Weyl's sandwich). ∎

**Proposition 2.** *Entrywise residual.* Each entry of a = fl(Ax) is an inner product of n terms: |a − Ax| ≤ γ̇_n|A||x|
(Higham 2002 eq. (3.5), `gamma_dot` for complex data). Likewise |b′ − Bx| ≤ γ̇_n|B||x|. c = fl(μb′) = μb′(1 + δ), so
|c − μBx| ≤ |μ|(|b′ − Bx| + u|b′|) ≤ |μ|(γ̇_n + u(1 + γ̇_n))|B||x| ≤ |μ|γ̇_{n+1}|B||x| (γ_a + u(1 + γ_a) ≤ γ_{a+1}; for the
complex constant 2γ_{n+4} + u(1 + 2γ_{n+4}) ≤ 2γ_{n+5}). ŝ = fl(a − c) = (a − c)(1 + δ′): |ŝ − (a − c)| ≤ u|a − c| ≤
u|ŝ|/(1 − u) ≤ u(1 + 2u)|ŝ|. Adding the three parts gives E with the exact |A||x|, |B||x|; the computed â, b̂ are sums of
non-negative products with relative error ≤ γ_n, so |A||x| ≤ â/(1 − γ_n), |B||x| ≤ b̂/(1 − γ_n). Then ‖s‖₂ ≤ ‖ŝ‖₂ + ‖ŝ − s‖₂ ≤
‖ŝ‖₂ + ‖E‖₂, each bounded by `norm2_upper`, the sum guarded by `up`. *Denominator.* xᴴBx = xᴴb′ + xᴴ(Bx − b′), with
|xᴴ(Bx − b′)| ≤ |x|ᵀγ̇_n|B||x| ≤ γ̇_n fl(|x|ᵀb̂)(1 + γ_{n+2})/(1 − γ_n) — the same accounting as Lemma 3 (i) of
`docs/proofs/E-05_variational.md` (in the code the division by 1 − γ_n is applied to b̂ once, `absBX`, and the
non-negative dot |x|ᵀ(b̂/(1 − γ_n)) is inflated by 1 + γ_{n+2}); Re D̂ is within e_D of xᴴb′ (`exact_dot`); `up` guards
the sum. *Quotient.*
β_c D_lo ≤ β_c xᴴBx; the computed p = fl(β_c D_lo) ≥ β_c D_lo(1 − u) and fl(√p) ≥ √p(1 − u) ≥ √(β_c D_lo)(1 − u)^{3/2}, so
fl(√p)(1 − 4u) ≤ √(β_c D_lo) ≤ √β_c ‖x‖_B (the product by (1 − 4u) rounds once more; (1 − u)^{3/2}(1 − 4u)(1 + u) ≤ 1 −
4u + 1.5u + u + O(u²) < 1). Hence S/(fl(√p)(1 − 4u)) ≥ ‖s‖₂/(√β_c‖x‖_B), and `up` covers the division. ∎

**Proposition 3.** *Forming M̂.* fl(σB_ij) = σB_ij(1 + δ), error ≤ u|σ||B_ij|; the subtraction adds ≤ u|M̂_ij|. So
|M̂ − M| ≤ u|σ||B| + u|M̂| entrywise; this bound matrix is symmetric with non-negative entries, hence ‖M̂ − M‖₂ ≤
‖|M̂ − M|‖₂ ≤ ‖u|σ||B| + u|M̂|‖_∞ (for a symmetric non-negative matrix ‖X‖₂ ≤ √(‖X‖₁‖X‖_∞) = ‖X‖_∞), bounded by
`inf_norm_upper`. M̂ is exactly Hermitian (σ real, B Hermitian: rounding commutes with conjugation). *Count.* The
factorization gives M′ (congruent to D) with ‖M′ − M̂‖₂ ≤ b_f, so ‖M′ − M‖₂ ≤ b_f + ‖M̂ − M‖₂ ≤ b and Lemma 3 applies with
collar c = b/β_c (rounded up): N(σ − c) ≤ ν ≤ #{λ_j ≤ σ + c}. *Conclusion for ν = k.* Each of the k pairwise disjoint
enclosures contains at least one pencil eigenvalue (Lemma 2), and these k eigenvalues are distinct (disjoint intervals)
and all < T ≤ σ − c, so N(σ − c) ≥ k; with N(σ − c) ≤ ν = k, N(σ − c) = k: the k eigenvalues below σ − c are exactly the
ones in the enclosures, hence λ₁, …, λ_k, one per enclosure (an enclosure with two of them would leave another enclosure
empty, contradicting Lemma 2), and λ_{k+1} ≥ σ − c. *ν > k:* N(σ + c) ≥ ν > k, so at least ν − k eigenvalues ≤ σ + c lie
outside the k enclosures' certified eigenvalues — the claim is incomplete or the collar hides them; no statement.
*ν < k:* N(σ − c) ≤ ν < k contradicts k distinct eigenvalues below T unless the enclosures overlap; the code reaches this
branch only when they do not (then the claim's vectors must be duplicates up to the radii, and nothing is certified). ∎

## 3 Where the code implements it

`certify_ed/generalized.py` (branch `track/variational`):

* Proposition 1: `certify_B_positive_definite` (71–112): inertia at shift 0 at 81 (refusal when ν ≥ 1, 83–87), shifts
  β = ½λ̂_min, then ¼ of that per retry (96–110), β_c = β − b rounded down at 102 (`sub_down`) with the extra 4u at 103.
* Proposition 2: `pencil_residual_radii` (115–153): γ̇_n, γ̇_{n+1} at 122–123, â, b̂ divided by (1 − γ_n) at 127–128, E at
  139, S at 141 (`up(norm2_upper + norm2_upper)`), D̂ by `exact_dot` at 142, D_lo at 144–146, the denominator at 149
  (`sqrt`, `(1 − 4u)`), the radius at 150–151 (`up`).
* Lemma 3 and Proposition 3: `pencil_count_below` (157–169): M̂ at 161, `inertia_dense`(M̂, 0) at 162, the forming bound
  at 163 (`inf_norm_upper`), b at 164, the collar at 165 (`up`), σ ∓ c at 167–168 (`sub_down`/`add_up`). The decision in
  `certify_generalized` (299–358): disjointness at 303, shifts between T and the computed λ̂_{k+1} (a hint only, 306–317),
  σ − c ≥ T required at 323, ν = k and disjoint → CERTIFIED at 326–334, ν < k at 335, ν > k at 341 (`missing_at_least`),
  overlap at 349 (NOT_IMPLEMENTED), no conclusive shift at 355.
* Hypothesis path: 244–253 (`declared_lambda_min_B`, status CERTIFIED_UNDER_HYPOTHESIS for G1 and, through `under` at 267,
  for G2/G3); refusal of an indefinite B at 226–233 regardless of a declaration; the dense limit `max_dim` at 214–221.

Rounding directions: residual and radius up (`norm2_upper`, `up`), xᴴBx and β_c down (`sub_down`, (1 − 4u)), the collar
up, σ − c down.

**Correction (2026-11-18), complex moduli; UNVERIFIED (D-12).** Proposition 2 and G3 use â = fl(|A||x|),
b̂ = fl(|B||x|), u(1 + 2u)|ŝ|, fl(|x|ᵀb̂) and the forming error u|σ||B| + u|M| with the computed moduli taken as exact
("sums of non-negative products with relative error ≤ γ_n"). For complex data the code took them from `numpy.abs`,
whose error is not bounded (`docs/proofs/fp_modulus-up.md`). Since 3.3.0.dev0 `generalized` takes them from `fp.abs_up` and
`fp.modulus_up` (≥ the exact moduli, Lemma M there), so the steps hold as written. No constant changes.

## 4 Computational verification

`docs/proofs/verify_E-03.py` (seed 20261030, `--cases 80`; mpmath at 60 digits: Cholesky of B, C = L⁻¹AL⁻ᴴ and its
eigenvalues, the exact ‖s‖_{B⁻¹}/‖x‖_B and λ_min(B)). Run of 2026-10-30 (`docs/proofs/runs/verify_E-03.txt`):

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | random pencils (n = 3…8, real and complex, cond(B) = 1…10⁸; claimed pairs from `scipy.linalg.eigh(A, B)`, half of them perturbed by 10⁻¹²…10⁻² in the vectors and values): β_c ≤ λ_min(B); every radius ≥ the exact ‖Ax − μBx‖_{B⁻¹}/‖x‖_B and ≥ the distance to the nearest pencil eigenvalue | 225 | 0 |
| B | G3 CERTIFIED records: exactly k pencil eigenvalues below `complete_below` and λ_{k+1} ≥ `next_eigenvalue_lower_bound` | 78 | 0 |
| C | adversarial (n = 6): B made indefinite along the first eigenvector → every statement NOT_CHECKED (UNSUPPORTED_MODEL); a wrong eigenvalue (μ₂ shifted by 0.3 of the spread) keeps a valid radius and G3 is not CERTIFIED; swapped pairs: radii valid, G3 not CERTIFIED; cond(B) = 10¹⁰: valid when certified, otherwise nothing certified; **negative control**: a declared λ_min(B) ten times the true one with the inertia path disabled — the record is CERTIFIED_UNDER_HYPOTHESIS, names `declared_lambda_min_B` with source "WRONG", and its radii fall below the exact B⁻¹-norm residual in 40/40 pairs | 100 adversarial | 0 |

Total: 303 random + 100 adversarial cases, 0 violations. Tests `tests/test_generalized.py` (8): random pencils
(real/complex, n = 40, cond 100) against `scipy.linalg.eigh`, perturbed vectors, indefinite B refused, non-Hermitian A
refused, wrong eigenvalue, swapped pairs, a missing eigenvalue detected (`missing_at_least`), 'subset' claims, the
declared-β path (stubbed inertia) with a false declaration (100·λ_max(B)) undercutting every exact radius while named,
the dense size limit, JSON round trip and the record schema.

## 5 Analytic cross-check

1. *B = I.* L = I, C = A, ‖·‖_B = ‖·‖₂, β_c ≤ 1 from the inertia of I − βI (exact pivots 1 − β), the radius is
   ‖Ax − μx‖₂/(√β_c‖x‖) ≥ the Theorem 1 radius, with equality in the limit β_c → 1; Lemma 3 with B = I is Lemma 4 of
   `docs/theory.md` (collar b/1 = b). The pencil statements reduce to S2/S3 of the standard problem.
2. *Diagonal pencil.* A = diag(a_j), B = diag(b_j) with b_j > 0: the eigenvalues are a_j/b_j, exact eigenvectors e_j,
   s = 0, r = 0 up to rounding, and ν(A − σB) = #{a_j − σb_j < 0} = #{a_j/b_j < σ} exactly — Lemma 3 is Sylvester's law
   read off the diagonal.
3. *Tightness of the B-norm bound.* For x an exact eigenvector of the pencil for λ and μ = λ + d: s = dBx, ‖s‖_{B⁻¹} =
   |d|‖x‖_B, so the first inequality of Lemma 2 is an equality, |λ − μ| = |d| = ‖s‖_{B⁻¹}/‖x‖_B. The second inequality
   loses at most the factor √(λ_max(B)/β_c): ‖s‖₂² = d²xᴴB²x ≤ d²λ_max(B)·xᴴBx, so the computable radius
   ‖s‖₂/(√β_c‖x‖_B) ≤ |d|√(λ_max(B)/β_c), i.e. at most √(λ_max(B)/β_c) times the exact one (the √cond(B) factor seen in
   part A and in the tests).
4. *Scaling.* Replacing (A, B) by (tA, tB), t > 0, leaves the eigenvalues, ν(A − σB) and the ratio ‖s‖₂/(√λ_min(B)‖x‖_B)
   unchanged (both scale by √t·√t/√t·√t); β_c scales by t and the collar b/β_c is scale-free as b scales by t.

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (303 random + 100 adversarial cases, 0 violations; negative control
undercut 40/40); independent re-derivation: done 2026-10-30 (fresh-context session, verdict AGREES-WITH-REMARKS, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
Correction for complex moduli on 2026-11-18 (end of section 3; UNVERIFIED, D-12).
