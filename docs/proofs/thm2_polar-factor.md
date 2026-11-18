# Theorem 2 (cluster radius ρ) and the polar-factor lemma

Derivation document in the D-59 format. Notation as in `docs/theory.md`: A is Hermitian n × n with
eigenvalues λ₁ ≤ … ≤ λₙ, ‖·‖ is the spectral norm unless marked F (Frobenius), u = 2⁻⁵³, γ_m = mu/(1 − mu).

## 1 Statement

**Hypotheses.** X ∈ ℂⁿˣᵍ with Gram matrix G = XᴴX and defect η := ‖G − I‖ < 1. Λ ∈ ℂᵍˣᵍ Hermitian (in the
implementation Λ = diag(μ₁, …, μ_g) real). R := AX − XΛ. δ := 1 − √(1 − η). ρ := ‖R‖ + (‖A‖ + ‖Λ‖) δ.
A finite; all quantities exact (the floating-point evaluation is the subject of section 3).

**Polar-factor lemma.** Let M := G^{1/2} (the Hermitian positive definite square root) and Q̂ := X M⁻¹. Then

- (a) Q̂ is an isometry: Q̂ᴴQ̂ = I;
- (b) the eigenvalues of M lie in [√(1 − η), √(1 + η)], so ‖M − I‖ ≤ δ and ‖M⁻¹‖ ≤ (1 − η)^{−1/2};
- (c) ‖X − Q̂‖ ≤ δ;
- (d) the residual of the pair (Q̂, Λ) satisfies ‖AQ̂ − Q̂Λ‖ ≤ ‖R‖ + (‖A‖ + ‖Λ‖) δ = ρ.

**Kahan's inclusion theorem (general Hermitian block).** Let Q ∈ ℂⁿˣᵍ have orthonormal columns, H ∈ ℂᵍˣᵍ be
Hermitian with eigenvalues θ₁ ≤ … ≤ θ_g, and r := ‖AQ − QH‖. Then there are g distinct indices i₁, …, i_g
with |λ_{i_j} − θ_j| ≤ r for j = 1, …, g. (H need not be the Rayleigh quotient QᴴAQ.)

**Theorem 2.** Under the hypotheses above there are g distinct indices i₁, …, i_g with |λ_{i_j} − μ_j| ≤ ρ,
where μ₁ ≤ … ≤ μ_g are the eigenvalues of Λ.

**Remark (what changed).** `docs/theory.md` §2 stated Theorem 2 with the radius ε = (w + √(w² + 4ρ²))/2,
w ≥ ‖Q̂ᴴ(AQ̂ − Q̂Λ)‖, from the Weyl-perturbation proof of the manuscript appendix (the perturbation matrix
[[S, Tᴴ], [T, 0]] in the basis [Q̂, Q̂⊥] has norm (‖S‖ + √(‖S‖² + 4‖T‖²))/2, which exceeds ‖[S; T]‖ unless S = 0).
Since ε ≥ ρ always (set w = 0), Kahan's constant ρ is never worse, and it is what the code now uses together
with Theorem 2′ (`docs/proofs/thm2prime_eps-star.md`): the cluster radius is min(ρ, ε*), both proven here.

## 2 Proof

**Polar-factor lemma.** G is Hermitian positive semidefinite; by hypothesis every eigenvalue g_k of G satisfies
|g_k − 1| ≤ η < 1, so g_k ∈ [1 − η, 1 + η] ⊂ (0, 2) and G is positive definite. M = G^{1/2} is defined by the
spectral theorem with eigenvalues √g_k ∈ [√(1 − η), √(1 + η)] and the same eigenvectors; M is Hermitian and
invertible, M⁻¹ has eigenvalues 1/√g_k ≤ (1 − η)^{−1/2}, which gives the second half of (b).

(a) Q̂ᴴQ̂ = M⁻¹XᴴXM⁻¹ = M⁻¹M²M⁻¹ = I (M⁻¹ is Hermitian).

(b) ‖M − I‖ = max_k |√g_k − 1| ≤ max(1 − √(1 − η), √(1 + η) − 1). By concavity of the square root,
(√(1 − η) + √(1 + η))/2 ≤ √1 = 1, i.e. √(1 + η) − 1 ≤ 1 − √(1 − η) = δ. Hence ‖M − I‖ ≤ δ.

(c) X − Q̂ = Q̂M − Q̂ = Q̂(M − I), and for an isometry Q̂ and any B, ‖Q̂B‖ = ‖B‖ (‖Q̂Bz‖² = zᴴBᴴQ̂ᴴQ̂Bz = ‖Bz‖²).
So ‖X − Q̂‖ = ‖M − I‖ ≤ δ.

(d) AQ̂ − Q̂Λ = (AX − XΛ) + A(Q̂ − X) − (Q̂ − X)Λ (expand and cancel). By the triangle inequality and
submultiplicativity, ‖AQ̂ − Q̂Λ‖ ≤ ‖R‖ + ‖A‖ ‖Q̂ − X‖ + ‖Q̂ − X‖ ‖Λ‖ ≤ ‖R‖ + (‖A‖ + ‖Λ‖) δ = ρ. ∎

**Kahan's inclusion theorem.** Classical: W. Kahan, *Inclusion theorems for clusters of eigenvalues of Hermitian
matrices*, Computer Science Report, University of Toronto, 1967; B. N. Parlett, *The Symmetric Eigenvalue
Problem*, SIAM Classics 1998, Theorem 11.5.1 (stated there for the Rayleigh quotient block; the proof below
covers any Hermitian H and is self-contained). Two steps.

*Step 1 (interval lemma).* For any real a ≤ b let m′ := #{j : θ_j ∈ [a, b]}. Then #{i : λ_i ∈ [a − r, b + r]} ≥ m′
(eigenvalues counted with multiplicity on both sides).

Proof. If m′ = 0 there is nothing to show. Otherwise let U ∈ ℂᵍˣᵐ′ hold orthonormal eigenvectors of H for the
eigenvalues θ_j ∈ [a, b], HU = UΘ′ with Θ′ = diag(those θ_j), and put Y := QU ∈ ℂⁿˣᵐ′. Y has orthonormal
columns (YᴴY = UᴴQᴴQU = UᴴU = I), and AY − YΘ′ = (AQ − QH)U, so ‖AY − YΘ′‖ ≤ r‖U‖ = r. Let c := (a + b)/2 and
h := (b − a)/2; then ‖Θ′ − cI‖ ≤ h. For every unit vector y = Yz in the range of Y (‖z‖ = 1),

    ‖(A − cI)y‖ = ‖(AY − YΘ′)z + Y(Θ′ − cI)z‖ ≤ r + h.

The Hermitian matrix (A − cI)² has eigenvalues (λ_i − c)². By the Courant–Fischer min-max theorem (any text,
e.g. Parlett 1998 Thm 10.2.1, Horn–Johnson *Matrix Analysis* Thm 4.2.6) its m′-th smallest eigenvalue is at most
max over unit y in any m′-dimensional subspace of yᴴ(A − cI)²y; with the range of Y this gives: at least m′
eigenvalues of (A − cI)² are ≤ (r + h)², i.e. at least m′ indices i with |λ_i − c| ≤ r + h, i.e.
λ_i ∈ [a − r, b + r]. ∎

*Step 2 (Hall's condition).* Build the bipartite graph with left vertices j = 1, …, g (the θ_j) and right vertices
i = 1, …, n (the λ_i), j ~ i iff |λ_i − θ_j| ≤ r. We show Hall's condition: for every J ⊆ {1, …, g} the
neighbourhood N(J) = {i : ∃ j ∈ J, |λ_i − θ_j| ≤ r} has |N(J)| ≥ |J|. The set ⋃_{j∈J} [θ_j − r, θ_j + r] is a
finite union of closed intervals; take its connected components C₁, …, C_p. Each C_k is [a_k − r, b_k + r] with
a_k = min and b_k = max of the θ_j (j ∈ J) whose intervals lie in C_k; let J_k be the set of those j. The J_k
partition J, and every j ∈ J_k has θ_j ∈ [a_k, b_k], so |J_k| ≤ m′_k := #{j ∈ {1, …, g} : θ_j ∈ [a_k, b_k]}. The
components are pairwise disjoint, hence

    |N(J)| = Σ_k #{i : λ_i ∈ [a_k − r, b_k + r]} ≥ Σ_k m′_k ≥ Σ_k |J_k| = |J|,

the middle inequality by Step 1 applied to [a_k, b_k]. Hall's marriage theorem (P. Hall 1935; e.g. Diestel,
*Graph Theory*, Thm 2.1.2) gives a matching that saturates every left vertex: an injection j ↦ i_j with
|λ_{i_j} − θ_j| ≤ r. ∎

**Theorem 2.** By the polar-factor lemma (a) and (d), Q̂ has orthonormal columns and ‖AQ̂ − Q̂Λ‖ ≤ ρ. Kahan's
theorem with Q = Q̂, H = Λ (Hermitian by hypothesis; its eigenvalues are the μ_j) gives distinct indices with
|λ_{i_j} − μ_j| ≤ ‖AQ̂ − Q̂Λ‖ ≤ ρ. ∎

**Corollary (uses in the code).** (i) X = V a sector isometry (n × d_s), Λ = H_s Hermitian: every eigenvalue of
H_s is within ρ = ‖HV − VH_s‖ + (‖H‖ + ‖H_s‖)δ of a distinct eigenvalue of H (`certify._subspace_shift`).
(ii) X = W = [V₁ ⋯ V_m] square, Λ = B = ⊕H_s: Theorem 8's constant (`sectors.decomposition_shift`); for the
square case Weyl's inequality gives the same constant by a shorter route (`docs/proofs/thm8_rho.md`).

## 3 Where the code implements it

`certify_ed/enclosure.py`, function `_cluster_bound` (the line numbers are those of branch
`track/backlog-proofs`; `up(x) = x (1 + 8u)`, one rounding; `(1 − u)^{k+1}(1 + 8u) ≥ 1` for k ≤ 6, so `up` of
an expression with at most six round-to-nearest operations on exact upper-bound inputs is ≥ the exact value
of the expression). Every input of the chain is itself an upper bound on the exact quantity it stands for.

| quantity | code | bound it implements | roundings inside `up` |
|---|---|---|---|
| η | `eta = up(fro_upper(G - I) + fro_upper(Eg) + 4 U g)` (exact Gram path) | ‖XᴴX − I‖₂ ≤ ‖XᴴX − I‖_F ≤ ‖fl(G − I)‖_F/(1 − u) + ‖Eg‖_F; `fro_upper(M) ≥ ‖M‖_F (1 + 5u)` covers the diagonal subtraction's rounding; `exact_dot` returns G with entrywise error ≤ Eg | 2 additions |
| η (BLAS path) | `eta = up(fro_upper(G - I) + gd * xf2) + gram_additive_term(g, n, cplx) + 2 ULP_MIN` | entrywise error of fl(XᴴX) ≤ γ_dot(n) (|X|ᴴ|X|)_ab + additive underflow term; ‖·‖_F ≤ γ_dot(n) ‖X‖_F² | 1 product, 1 addition; `xf2` = `fro_upper(Xg) ** 2` carries the slack (1 + 5u)² ≥ (1 + 10u) for its square and `gd`'s two roundings |
| δ | `delta = up(1 - sqrt(1 - eta)) + 4 U` | 1 − √(1 − η): fl(1 − η) = (1 − η)(1 + d₁), fl(√·) = √(1 − η)(1 + d₁)^{1/2}(1 + d₂), so 1 − fl(√·) ≥ δ − 1.5u√(1 − η) − O(u²) ≥ δ − 1.6u; the subtraction's rounding costs ≤ u; `up` adds ≥ 6u relative; `+ 4u` absolute covers the ≤ 2.6u absolute shortfall (δ ≤ 1) | — (absolute accounting) |
| ‖R‖ | `Rg_fro = norm2_upper(res.upper[idx])` | ‖R‖₂ ≤ ‖R‖_F = ‖(‖r₁‖, …, ‖r_g‖)‖₂ ≤ ‖(res.upper)‖₂ (each `res.upper[i]` ≥ ‖r_i‖, `docs/proofs/thm1_fp-accounting.md`); `norm2_upper` is an upper bound on the 2-norm of the stored vector (same document, step 4; underflow term `docs/proofs/int05_underflow.md`) | — |
| ‖A‖ | `Hn = adapter.norm_upper()` | `inf_norm_upper(A) ≥ ‖A‖_∞ ≥ ‖A‖₂` for Hermitian A (‖A‖₂ ≤ √(‖A‖₁‖A‖_∞) = ‖A‖_∞) | — |
| ‖Λ‖ | `lam_n = max(abs(mu[idx]))` | exact for diagonal Λ | 0 |
| ρ | `rho = up(Rg_fro + (Hn + lam_n) * delta) + ULP_MIN` | ρ = ‖R‖ + (‖A‖ + ‖Λ‖)δ, each factor an upper bound, all terms non-negative | 3 (addition, product, addition); `+ ULP_MIN` covers the absolute error of the product when its result is subnormal |
| radius | `min(rho, eps_star)` | both proven (Theorem 2 here, Theorem 2′ in its document) | 0 |

`certify._subspace_shift(iso, inv, H_norm, M_norm)` evaluates `up(inv + (H_norm + M_norm) * delta) + ULP_MIN`
with `delta` as above (Corollary (i); its inputs `iso ≥ ‖VᴴV − I‖_F`, `inv ≥ ‖HV − VH_s‖_F` come from
`sectors.audit_sector`). `sectors.decomposition_shift` evaluates the same expression with
η = `norm2_upper([iso_1, …, iso_m, orth × m(m − 1)])` ≥ ‖WᴴW − I‖_F (block Frobenius bound, m diagonal blocks and
m(m − 1) off-diagonal blocks) and D = `norm2_upper([inv_1, …, inv_m])` ≥ ‖HW − WB‖_F (Corollary (ii)). Before
this branch both returned (ρ + √5 ρ)/2.

Rounding direction: every quantity is rounded towards +∞ relative to the exact value it bounds (the bound can
only be larger than the exact ρ, never smaller).

## 4 Computational verification

`docs/proofs/verify_thm2_polar-factor.py` (mpmath at 30 decimal digits on the binary64 inputs taken exactly;
the exact quantities are algebraic, so the ground truth is the 30-digit value with a slack of 10⁻²⁴, far below
the 10⁻¹⁶ relative margin every floating-point bound carries). Per case it checks

- L1: ‖M − I‖ ≤ δ, ‖M⁻¹‖ ≤ (1 − η)^{−1/2}, ‖X − Q̂‖ ≤ δ, Q̂ᴴQ̂ = I (lemma (a)–(c));
- L2: ‖AQ̂ − Q̂Λ‖ ≤ ρ (lemma (d));
- K: the bottleneck matching distance d* = min over injections of max_j |λ_{i_j} − μ_j| (brute force) is
  ≤ ‖AQ̂ − Q̂Λ‖ (Kahan with the non-Rayleigh block Λ);
- T2: d* ≤ ρ (Theorem 2);
- KG: Kahan's theorem for an orthonormal Q and a random Hermitian H that is not QᴴAQ (adversarial: H far from
  the Rayleigh quotient, so that QᴴR dominates the orthogonal part of the residual);
- C: the code's `rho` ≥ exact ρ, its `eta` ≥ exact η, its radius min(ρ, ε*) ≥ d*, and `_subspace_shift` ≥ the
  exact ρ on its inputs.

Random cases: n ∈ [2, 6], g ∈ [1, 4], real or complex A with a planted cluster of spread 0 to 10, X = V_cluster C
+ noise with ‖C − I‖ from 0 to 0.9 and noise from 0 to 0.3, μ exact or perturbed. Adversarial families (60 each):
η → 1 (two nearly parallel columns); exactly orthonormal X (signed unit vectors, η = 0); Λ far from the spectrum;
A = 0; A = cI; square X (g = n); large spread of Λ; A and μ scaled by 10^{±40}, 10^{±150}; column scaling;
n = g = 1. Cases with η ≥ 1 are counted as skipped (hypothesis not met).

Counts (run of 2026-10-30, `--cases 20000 --seed 1`, output in `docs/proofs/runs/thm2_polar-factor.txt`):
20 000 generated random cases (17 224 met η < 1 − 10⁻²⁰; 2 776 skipped) and 600 adversarial cases, plus 5 000 + 600
random-block Kahan cases; per counter: L1/L2 34 448 random + 900 adversarial checks, K and T2 17 224 + 450 each, KG
5 000 + 600, C 32 738 + 859 (1 751 skipped where the code's guarded η reached 1); 0 violations. A 60-case pilot
run *before* the INT-05 fix had found 2 violations in the C counter, both in the 10^{−150}-scaled family: the
code returned radius 0 for a residual of 7 × 10⁻¹⁶⁷ because the squares in its norms underflowed — the INT-05
defect (`docs/proofs/int05_underflow.md`); the final run passes that family.

## 5 Analytic cross-check

(i) *Reduction to Theorem 1.* g = 1: X = x, η = |‖x‖² − 1|, M = ‖x‖, Q̂ = x/‖x‖, and (d) reads
‖Ax/‖x‖ − μx/‖x‖‖ = ‖Ax − μx‖/‖x‖ ≤ ρ, which is Theorem 1's radius (with slack (‖A‖ + |μ|)δ); Kahan's theorem
with g = 1 is Theorem 1 itself (the m′ = 1 case of the interval lemma is the statement that some eigenvalue
lies within r of θ).

(ii) *Orthonormal X.* η = 0 gives δ = 0, M = I, Q̂ = X, ρ = ‖R‖: Theorem 2 is Kahan's theorem, and for Λ = XᴴAX
(Ritz values) it is the classical Rayleigh–Ritz residual bound.

(iii) *Square X.* g = n, Q̂ unitary: Q̂ᴴAQ̂ has the spectrum of A and ‖Q̂ᴴAQ̂ − Λ‖ = ‖Q̂ᴴ(AQ̂ − Q̂Λ)‖ ≤ ρ, so Weyl's
inequality pairs the sorted eigenvalues within ρ — the same constant as Theorem 2 by a different theorem
(`docs/proofs/thm8_rho.md`).

(iv) *The old constant.* With w ≤ ρ, ε = (w + √(w² + 4ρ²))/2 ∈ [ρ, (1 + √5)ρ/2]; at η = 0 and Λ = XᴴAX it gives
ε = ‖R‖ = ρ, so the two statements agree exactly where the appendix's proof is sharp (S = 0).

(v) *Tightness of (b).* X = [√(1 − η)] (n = g = 1) gives G = 1 − η, M = √(1 − η), ‖M − I‖ = δ exactly.

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (17 224 (34 448 lemma checks, 5 000 Kahan-block checks) random + 600 (900 lemma checks, 600 Kahan-block checks) adversarial cases, 0 violations); independent re-derivation: done 2026-10-30 (fresh-context session, verdict AGREES-WITH-REMARKS, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
