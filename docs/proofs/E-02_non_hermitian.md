# E-02 Non-Hermitian eigenproblems: backward errors of residuals, Bauer–Fike discs from a computed eigenvector matrix with a certified bound on its inverse, and the count of eigenvalues per group of discs

Notation: A ∈ ℂⁿˣⁿ is the stored matrix (binary64 entries, real or complex), **not** assumed Hermitian or normal; its
eigenvalues λ₁, …, λₙ are the roots of det(zI − A) listed with algebraic multiplicity, Λ(A) their set. A claim lists values
μ̂₁, …, μ̂ₖ ∈ ℂ with right vectors xᵢ ≠ 0 (the columns of V̂) and optionally left vectors yᵢ ≠ 0; for a complete claim
(k = n) D̂ = diag(μ̂₁, …, μ̂ₙ) and R = AV̂ − V̂D̂ is the exact residual matrix of the stored numbers, with columns rᵢ = Axᵢ − μ̂ᵢxᵢ.
‖·‖ is the Euclidean norm of vectors and the spectral norm of matrices, ‖·‖_F the Frobenius norm, σ_min the smallest
singular value, κ₂(M) = ‖M‖‖M⁻¹‖, D(c, ρ) = {z ∈ ℂ : |z − c| ≤ ρ} the closed disc, Λ_ε(A) = {z : σ_min(A − zI) ≤ ε} the
closed ε-pseudospectrum. u = 2⁻⁵³, u_min = 2⁻¹⁰⁷⁵, γ_m = mu/(1 − mu), γ̇_m = `fp.gamma_dot(m, complex)` (γ_m for real data,
2γ_{m+4} for complex data), `up`, `norm2_upper`, `norm2_lower`, `inf_norm_upper`, `safe_sqrt_upper`, `sub_down` as in `fp.py`.

## 1 Statement

**Lemma 1 (backward error of a right or a left residual; Kahan, Parlett and Jiang 1982; Trefethen and Embree 2005,
Thm 2.1).** For x ≠ 0, μ ∈ ℂ and r = Ax − μx, the matrix E = −r xᴴ/(xᴴx) satisfies (A + E)x = μx and ‖E‖ = ‖r‖/‖x‖; hence
σ_min(A − μI) ≤ ‖r‖/‖x‖, i.e. μ ∈ Λ_ε(A) with ε = ‖r‖/‖x‖. For y ≠ 0 and the left residual s = Aᴴy − μ̄y (sᴴ = yᴴA − μyᴴ),
F = −y sᴴ/(yᴴy) satisfies yᴴ(A + F) = μyᴴ and ‖F‖ = ‖s‖/‖y‖. When A is not normal, no eigenvalue of A need lie within ε
of μ.

**Theorem 2 (Bauer and Fike 1960).** Let A = VDV⁻¹ with D = diag(d₁, …, dₙ) and V invertible, and let E ∈ ℂⁿˣⁿ. Every
eigenvalue λ of A + E satisfies min_i |λ − dᵢ| ≤ ‖V⁻¹EV‖ ≤ κ₂(V)‖E‖. *Residual form:* for every x ≠ 0 and μ ∈ ℂ,
min_i |dᵢ − μ| ≤ κ₂(V)‖Ax − μx‖/‖x‖.

**Lemma 3 (the inverse from a computed inverse).** Let V̂, W ∈ ℂⁿˣⁿ, G = I − WV̂ and ‖G‖ ≤ θ < 1. Then W and V̂ are
invertible, V̂⁻¹ = (I − G)⁻¹W, and ‖V̂⁻¹‖ ≤ ‖W‖/(1 − θ).

**Theorem 4 (discs from a computed eigenvector matrix, and their count).** Let V̂ be invertible, F = V̂⁻¹R, and ρ ≥ ‖F‖
(for instance ρ ≥ β‖R‖_F with β ≥ ‖V̂⁻¹‖). A need not be diagonalizable.

* (a) A = V̂(D̂ + F)V̂⁻¹: the eigenvalues of A are those of D̂ + F with the same algebraic multiplicities.
* (b) Every eigenvalue of A lies in U = ⋃ᵢ D(μ̂ᵢ, ρ).
* (c) Let {1, …, n} be divided into groups G₁, …, G_p such that D(μ̂ᵢ, ρ) ∩ D(μ̂ⱼ, ρ) = ∅ whenever i and j lie in different
  groups, and put U_q = ⋃_{i∈G_q} D(μ̂ᵢ, ρ). Then U_q contains exactly |G_q| eigenvalues of A, counted with algebraic
  multiplicity.
* (d) A disc that meets no other disc contains exactly one eigenvalue of A, and it is simple. If all n discs are pairwise
  disjoint, A has n distinct eigenvalues and is diagonalizable.
* (e) With the relative residuals r̃ᵢ = rᵢ/‖xᵢ‖: β‖R‖_F ≤ β‖V̂‖ (Σᵢ‖r̃ᵢ‖²)^{1/2} and β‖V̂‖ ≥ κ₂(V̂). The radius is therefore at
  most a certified bound on κ₂(V̂) times the Frobenius norm of the relative residuals: the bound of Theorem 2, with the
  computed V̂ in place of the unknown exact eigenvector matrix.

**Floating-point hypotheses (FP).** binary64 round-to-nearest; every rounded product, quotient or square root obeys
fl(z) = z(1 + δ) + η with |δ| ≤ u, |η| ≤ u_min, and every rounded addition or subtraction obeys fl(z) = z(1 + δ) (it is
exact when the result is subnormal): the underflow-tolerant model (M-U) of `docs/proofs/int05_underflow.md`. BLAS inner
products of m terms are within γ̇_m times the sum of the moduli of the terms, in any order (`fp.py`, Higham 2002 §3.1
and §3.6). The modulus of a complex number (`np.abs`, C `hypot`) has relative error at most 3u (H-mod, the hypothesis
of `docs/proofs/D-01_sparse-residual.md`; moduli of real numbers are exact). `fp.gamma(m)` is within a factor (1 − u)² of
γ_m (two roundings; `gamma_dot` doubles exactly). Elementwise complex products of NumPy may be contracted to fused
multiply-adds by the compiler; their error is then at most 2u|xy| (Jeannerod, Kornerup, Louvet and Muller 2017), below
the γ₃|xy| used here. With `underflow_check=False`, (FP) and H-mod are assumed also for stored numbers with subnormal
components, whose computed moduli are rounded to the subnormal grid. An overflow produces ±∞, which the code treats as
"no bound".

**Proposition 5 (residual bounds of N1 and N3; `nonhermitian.residual_bounds`).** For a column x of V̂ (n entries), a
value μ and w = n, let a = fl(Ax), c = fl(μx) entrywise, ŝ = fl(a − c), â = fl(|A||x|) (computed moduli, then a BLAS
product), p̂ = fl(fl(|x|)·fl(|μ|)), G = γ̇_{w+2} and g = γ_{w+6} for complex data (γ_{w+1} for real data), and let E be the
computed value of G·(â/(1 − g) + p̂(1 + 8u))·(1 + 12u) in the order written. Under (FP), for every entry j:

* (i) |ŝⱼ − rⱼ| ≤ G((|A||x|)ⱼ + |μ||xⱼ|) + (8w + 8)u_min with r = Ax − μx;
* (ii) in the standard model (η = 0), (|A||x|)ⱼ ≤ âⱼ/(1 − g) and |μ||xⱼ| ≤ p̂ⱼ(1 + 8u); under (M-U) the shortfall of â,
  of p̂ and of the rounding of E is at most (2w + 10)u_min per entry of E;
* (iii) |ŝⱼ − rⱼ| ≤ Eⱼ + (10w + 20)u_min;
* (iv) with T = ldexp(√n (10w + 20)(1 + 8u)², −1075) + 2⁻¹⁰⁷⁴ ≥ √n (10w + 20)u_min:
  ‖r‖ ≤ S := up(norm2_upper(ŝ) + norm2_upper(E) + T), and ε := up(S/norm2_lower(x) + 2⁻¹⁰⁷³) ≥ ‖r‖/‖x‖ whenever
  norm2_lower(x) > 0.

**Proposition 6 (θ, β and the norm bounds; `nonhermitian.inverse_bound`, `nonhermitian.matrix_norm2_upper`).** For any W
(the code takes W = `np.linalg.inv(V̂)`), P̂ = fl(WV̂), Ĝ = fl(I − P̂), g = γ_{n+6} (complex) or γ_{n+1} (real), B the
computed value of (γ̇ₙ·fl(|W||V̂|)/(1 − g) + 4u·|Ĝᵢᵢ|·[i = j])·(1 + 12u), and T_G = ldexp(n(10n + 20)(1 + 8u)², −1075) + 2⁻¹⁰⁷⁴:

* (i) |(I − WV̂)ᵢⱼ − Ĝᵢⱼ| ≤ Bᵢⱼ + (10n + 20)u_min;
* (ii) ‖I − WV̂‖ ≤ ‖I − WV̂‖_F ≤ θ := up(norm2_upper(Ĝ) + norm2_upper(B) + T_G);
* (iii) c(M) := min(norm2_upper(vec M), safe_sqrt_upper(up(inf_norm_upper(Mᵀ)·inf_norm_upper(M)))) ≥ ‖M‖ for every stored
  M with at most 2²⁶ columns;
* (iv) if θ < 1: V̂ is invertible, β := up(c(W)/sub_down(1, θ) + 2⁻¹⁰⁷³) ≥ ‖V̂⁻¹‖ and up(β·c(V̂)) ≥ κ₂(V̂).

**Proposition 7 (radius and groups; `certify_nonhermitian`, `nonhermitian.disc_groups`).**

* (i) With Sᵢ of Proposition 5 for each column, ρ_R := norm2_upper((S₁, …, Sₙ)) ≥ ‖R‖_F and ρ := up(β ρ_R + 2⁻¹⁰⁷³) ≥
  β‖R‖_F ≥ ‖F‖.
* (ii) For i ≠ j, δᵢⱼ := fl(max(|fl(Re μ̂ᵢ − Re μ̂ⱼ)|, |fl(Im μ̂ᵢ − Im μ̂ⱼ)|)·(1 − 2u)) satisfies δᵢⱼ ≤ |μ̂ᵢ − μ̂ⱼ|. If
  δᵢⱼ > 2ρ (2ρ is exact), the discs D(μ̂ᵢ, ρ) and D(μ̂ⱼ, ρ) are disjoint.
* (iii) The connected components of the graph on {1, …, n} that joins i and j unless δᵢⱼ > 2ρ satisfy the hypothesis of
  Theorem 4 (c).

**Statements of the record** (`certify_ed.nonhermitian`, schema `certify-ed/nonhermitian/1.0`; every statement carries
the marker UNVERIFIED):

* **N1.backward_errors** (any claim): μ̂ᵢ ∈ Λ_{εᵢ}(A) for every pair (Lemma 1, Proposition 5); with left vectors also the
  left εᵢ′, and the record gives min(εᵢ, εᵢ′). This is the pseudospectral statement, not an enclosure of an eigenvalue.
* **N2.eigenvector_inverse** (complete claim, kind `all`): V̂ is invertible, ‖V̂⁻¹‖ ≤ β, κ₂(V̂) ≤ up(β·c(V̂)) (Lemma 3,
  Proposition 6).
* **N3.inclusion**: Λ(A) ⊂ ⋃ᵢ D(μ̂ᵢ, ρ) (Theorem 4 (b), Proposition 7 (i)).
* **N4.counting**: the groups of Proposition 7 (iii) and the count of Theorem 4 (c) for each; (d) for isolated discs and
  for the case that every disc is isolated.

Status rules: θ ≥ 1, a non-finite W or a V̂ singular in floating point gives N2 INCONCLUSIVE with reason
`ILL_CONDITIONED_EIGENVECTORS` and N3, N4 NOT_CHECKED (`NEEDS_DEPENDENCY`); a partial claim (k < n) gives N2 to N4
NOT_CHECKED (`NEEDS_WHOLE_SPECTRUM`); a stored nonzero magnitude below 2⁻⁹⁰⁰ in A, μ̂, V̂ or the left vectors gives every
statement INCONCLUSIVE (`UNDERFLOW_RISK`) unless `underflow_check=False`; n > `max_dim` (2000) gives NOT_CHECKED
(`SIZE_LIMIT`); a vector norm that cannot be bounded away from zero, or an overflowing bound, gives N1 INCONCLUSIVE
(`NOT_APPLICABLE`). For a complete claim N3 and N4 account for all n eigenvalues, so no separate completeness statement
is needed; for a partial claim no inclusion statement is made, because for a non-normal A the residuals of some pairs say
nothing about where the eigenvalues are (Lemma 1, last sentence). Not implemented: per-disc radii, pseudospectral
portraits, Lindbladian structure, sparse or matrix-free operators.

## 2 Proof

**Lemma 1.** (A + E)x = Ax − r(xᴴx)/(xᴴx) = μx. E = uvᴴ with u = −r/‖x‖, v = x/‖x‖ has rank one, so ‖E‖ = ‖u‖‖v‖ =
‖r‖/‖x‖. A + E − μI is singular, hence σ_min(A − μI) ≤ ‖E‖ (Weyl's inequality for singular values,
σ_min(M + E) ≥ σ_min(M) − ‖E‖, with M + E singular); directly, σ_min(A − μI) = min_{z≠0}‖(A − μI)z‖/‖z‖ ≤ ‖r‖/‖x‖. The left
statement is the right statement for Aᴴ, y and μ̄: (Aᴴ + F̃)y = μ̄y with F̃ = −s yᴴ/(yᴴy), and F = F̃ᴴ has the same norm and
gives yᴴ(A + F) = μyᴴ. The last sentence: A = [[0, 1], [0, 0]], x = (1, t), μ = t, 0 < t < 1 give r = (0, −t²) and
‖r‖/‖x‖ = t²/√(1 + t²) < t = |μ − 0|, while 0 is the only eigenvalue. ∎

**Theorem 2.** Let λ be an eigenvalue of A + E with λ ≠ dᵢ for all i (otherwise there is nothing to prove). Then
A + E − λI = V(D − λI)(I + (D − λI)⁻¹V⁻¹EV)V⁻¹ is singular, so I + K with K = (D − λI)⁻¹V⁻¹EV is singular, which forces
‖K‖ ≥ 1 (if ‖K‖ < 1, I + K is invertible by the Neumann series). Since ‖(D − λI)⁻¹‖ = 1/min_i|dᵢ − λ| for a diagonal
matrix, 1 ≤ ‖V⁻¹EV‖/min_i|dᵢ − λ|, i.e. min_i|dᵢ − λ| ≤ ‖V⁻¹EV‖ ≤ ‖V⁻¹‖‖E‖‖V‖. Residual form: by Lemma 1, μ is an
eigenvalue of A + E with ‖E‖ = ‖Ax − μx‖/‖x‖; apply the first part with λ = μ. ∎

**Lemma 3.** ‖G‖ < 1 makes I − G = WV̂ invertible, with (I − G)⁻¹ = Σ_{m≥0} Gᵐ and ‖(I − G)⁻¹‖ ≤ Σ θᵐ = 1/(1 − θ).
det W · det V̂ = det(WV̂) ≠ 0, so both factors are invertible, and V̂⁻¹ = (WV̂)⁻¹W = (I − G)⁻¹W, ‖V̂⁻¹‖ ≤ ‖W‖/(1 − θ). ∎

**Theorem 4.** (a) AV̂ = V̂D̂ + R gives V̂⁻¹AV̂ = D̂ + F; similar matrices have the same characteristic polynomial.
(b) Theorem 2 for the diagonal matrix D̂ (V = I, κ₂ = 1) and the perturbation F: every eigenvalue λ of D̂ + F, hence of A,
has min_i|λ − μ̂ᵢ| ≤ ‖F‖ ≤ ρ. For the stated choice, ‖F‖ ≤ ‖V̂⁻¹‖‖R‖ ≤ β‖R‖_F.
(c) For t ∈ [0, 1] let M(t) = D̂ + tF. By (b) applied to tF (‖tF‖ ≤ ρ) every eigenvalue of M(t) lies in U. The
characteristic polynomial of M(t) is monic of degree n with coefficients that are polynomials in t. By the continuity of
the roots of a polynomial on a real parameter interval (Kato 1995, Chapter II, §5.2; Horn and Johnson 2013, Appendix D)
there are continuous functions λ₁(t), …, λₙ(t) on [0, 1] that list the eigenvalues of M(t) with algebraic multiplicity
for every t. Let U_q′ = ⋃_{i∉G_q} D(μ̂ᵢ, ρ). U_q and U_q′ are closed (finite unions of closed discs), disjoint (by the
hypothesis on the groups) and U_q ∪ U_q′ = U. Each λⱼ([0, 1]) is a connected subset of U, so it lies entirely in U_q or
entirely in U_q′. At t = 0 the λⱼ(0) are μ̂₁, …, μ̂ₙ; μ̂ᵢ ∈ U_q for i ∈ G_q and μ̂ᵢ ∈ U_q′, so μ̂ᵢ ∉ U_q, for i ∉ G_q.
Exactly |G_q| of the functions therefore start in U_q, and the same functions end there: M(1) = D̂ + F has exactly |G_q|
eigenvalues in U_q, counted with multiplicity, and by (a) so has A. This is the continuity argument of the counting part
of Gershgorin's theorem (Horn and Johnson 2013, Theorem 6.1.1), applied to D̂ + tF.
(d) An isolated disc forms its own group (|G_q| = 1): one eigenvalue of algebraic multiplicity one. If all discs are
isolated, each of the n discs holds one simple eigenvalue and these are distinct (the discs are disjoint); a matrix with n
distinct eigenvalues has n linearly independent eigenvectors and is diagonalizable.
(e) R = R̃N with N = diag(‖xᵢ‖) gives ‖R‖_F ≤ ‖R̃‖_F maxᵢ‖xᵢ‖, and maxᵢ‖xᵢ‖ = maxᵢ‖V̂eᵢ‖ ≤ ‖V̂‖; β‖V̂‖ ≥ ‖V̂⁻¹‖‖V̂‖ =
κ₂(V̂). ∎

**Proposition 5.** *Additive part first.* Under (FP) the computation of one entry ŝⱼ involves at most 4w + 4 rounded real
products (4w in the complex inner product (Ax)ⱼ, four in μxⱼ; fewer for real data) and no other operation with an η;
each η is at most u_min and reaches ŝⱼ multiplied by at most (1 + u)^{w+1} ≤ 2 (the later additions of the sum and the
subtraction). The computation of Eⱼ involves at most w + 5 rounded multiplications or divisions (w products in âⱼ, the
quotient by fl(1 − g), the product p̂ⱼ, the product with 1 + 8u, the product with G, the product with 1 + 12u), each η
multiplied by at most 2 on its way (the factors 1/fl(1 − g) ≤ 2, G ≤ 1/2, 1 + 12u and the additions). Hence every
inequality below that holds in the standard model (η = 0) holds under (M-U) up to (8w + 8)u_min in ŝⱼ and
(2w + 10)u_min in Eⱼ, together at most (10w + 20)u_min; this gives the additive terms of (i) to (iii). The rest of the
proof is in the standard model.

(i) |aⱼ − (Ax)ⱼ| ≤ γ̇_w(|A||x|)ⱼ (inner product of w terms, `fp.gamma_dot`). |cⱼ − μxⱼ| ≤ γ₃|μ||xⱼ| for complex data
(Higham 2002, Lemma 3.5: √2γ₂ ≤ γ₃; with a fused multiply-add 2u ≤ γ₃) and ≤ u|μ||xⱼ| for real data. ŝⱼ = fl(aⱼ − cⱼ) rounds each component once:
|ŝⱼ − (aⱼ − cⱼ)| ≤ u|aⱼ − cⱼ|, and |aⱼ − cⱼ| ≤ |rⱼ| + |aⱼ − (Ax)ⱼ| + |cⱼ − μxⱼ| with |rⱼ| ≤ (|A||x|)ⱼ + |μ||xⱼ|. So

  |ŝⱼ − rⱼ| ≤ (γ̇_w + u(1 + γ̇_w))(|A||x|)ⱼ + (γ₃ + u(1 + γ₃))|μ||xⱼ|.

With γ_k + u(1 + γ_k) ≤ γ_{k+1} (Higham 2002, Lemma 3.3): for complex data 2γ_{w+4} + u(1 + 2γ_{w+4}) ≤ 2γ_{w+5} ≤ γ̇_{w+2}
and γ₃ + u(1 + γ₃) ≤ γ₄ ≤ γ̇_{w+2}; for real data γ_w + u(1 + γ_w) ≤ γ_{w+1} ≤ γ_{w+2} and 2u + u² ≤ γ₂ ≤ γ_{w+2}.

(ii) Under H-mod a computed modulus m̂ of z satisfies m̂ ≥ (1 − 3u)|z|. âⱼ is a sum of w computed products of computed
moduli, all non-negative, so âⱼ ≥ (1 − γ_w)(1 − 3u)²(|A||x|)ⱼ ≥ (1 − γ_{w+6})(|A||x|)ⱼ, because (1 − 3u)² ≥ 1 − 6u and
γ_{w+6} ≥ γ_w + 6u/(1 − wu) ≥ γ_w + 6u. For real data the moduli are exact and âⱼ ≥ (1 − γ_w)(|A||x|)ⱼ ≥ (1 − γ_{w+1})(|A||x|)ⱼ.
p̂ⱼ ≥ (1 − u)(1 − 3u)²|μ||xⱼ| ≥ (1 − 7u)|μ||xⱼ|, and 1/(1 − 7u) ≤ 1 + 8u because (1 + 8u)(1 − 7u) = 1 + u − 56u² ≥ 1.

(iii) Let Φⱼ = G_e(âⱼ/(1 − g_e) + p̂ⱼ(1 + 8u)) with the exact constants G_e = γ̇_{w+2} and g_e. By (i) and (ii),
|ŝⱼ − rⱼ| ≤ Φⱼ. The computed Eⱼ evaluates Φⱼ(1 + 12u) with six roundings of non-negative quantities (fl(1 − ĝ), the
quotient, the product p̂ⱼ(1 + 8u), the sum, the product with Ĝ, the product with 1 + 12u), each lowering the result by at
most a factor 1 − u, and with the computed constants: Ĝ ≥ G_e(1 − u)², and 1/(1 − ĝ) ≥ (1 − u)/(1 − g_e) because ĝ ≥
g_e(1 − u)² gives 1 − ĝ ≤ (1 − g_e)(1 + 2ug_e/(1 − g_e)) ≤ (1 − g_e)(1 + u) for g_e ≤ 1/3. In total
Eⱼ ≥ Φⱼ(1 + 12u)(1 − u)⁹ ≥ Φⱼ(1 + 12u)(1 − 9u) = Φⱼ(1 + 3u − 108u²) ≥ Φⱼ. (1 + 8u and 1 + 12u are binary64 numbers.)

(iv) ‖r‖ ≤ ‖ŝ‖ + ‖ŝ − r‖ ≤ ‖ŝ‖ + ‖E‖ + √n (10w + 20)u_min, since an entrywise bound gives a norm bound and the vector of
additive terms has norm at most √n times its largest entry. `norm2_upper(v) ≥ ‖v‖` for every stored vector of any
magnitude (`docs/proofs/int05_underflow.md`, Lemma U3, with the overflow clause of `docs/proofs/VERDICTS.md`). T is formed
as `fp.residual_additive_term` is: the factor √n(10w + 20)(1 + 8u)² is a normal number whose four roundings the two guards
cover, `ldexp(·, −1075)` rounds it to the subnormal grid by at most u_min, and 2⁻¹⁰⁷⁴ = 2u_min covers that. The sum of
three non-negative terms has two roundings and `up` multiplies by 1 + 8u with one more: (1 − u)³(1 + 8u) ≥ 1. For ε, with
q = S/norm2_lower(x) (and norm2_lower(x) ≤ ‖x‖): if q ≥ 2⁻¹⁰⁰⁰ the quotient, the sum with 2⁻¹⁰⁷³ and the product in `up`
are normal and ε ≥ q(1 − u)³(1 + 8u) ≥ q; if q is smaller, the quotient is within u_min + uq of q, the sum with
2⁻¹⁰⁷³ = 4u_min is exact when it is subnormal and otherwise rounded by a relative u, and the product by 1 + 8u does not
decrease a binary64 number (rounding is monotone), so ε ≥ q + 3u_min − 2uq ≥ q. ∎

**Proposition 6.** (i) Entry (i, j) of P̂ is a BLAS inner product of n terms: |P̂ᵢⱼ − (WV̂)ᵢⱼ| ≤ γ̇ₙ(|W||V̂|)ᵢⱼ. Ĝ = fl(I − P̂)
is exact off the diagonal (Ĝᵢⱼ = −P̂ᵢⱼ) and on the diagonal rounds the real part once and negates the imaginary part, so
|Ĝᵢᵢ − (1 − P̂ᵢᵢ)| ≤ u|1 − P̂ᵢᵢ| ≤ u|Ĝᵢᵢ|/(1 − u) ≤ 2u·m̂ᵢᵢ (m̂ᵢᵢ the computed modulus, H-mod). By (ii) of Proposition 5
for the moduli, (|W||V̂|)ᵢⱼ ≤ fl(|W||V̂|)ᵢⱼ/(1 − g); the computed B evaluates its formula with at most nine roundings or
constant errors (as in Proposition 5 (iii); the product 4u·m̂ᵢᵢ is exact in the normal range), and (1 + 12u)(1 − u)⁹ ≥ 1.
Additive part: an entry of P̂ involves 4n rounded real products (complex) and an entry of B at most n + 5 rounded
products or quotients, each η multiplied by at most 2: at most (10n + 10)u_min ≤ (10n + 20)u_min.
(ii) ‖G‖ ≤ ‖G‖_F; the entrywise bound gives ‖G − Ĝ‖_F ≤ ‖B‖_F + n(10n + 20)u_min (n² entries); T_G is formed as T; the
two sums and `up` as in Proposition 5 (iv).
(iii) ‖M‖ ≤ ‖M‖_F = ‖vec M‖ ≤ norm2_upper(vec M), and ‖M‖² ≤ ‖M‖₁‖M‖_∞ (Golub and Van Loan 2013, Corollary 2.3.2) with
‖M‖₁ = ‖Mᵀ‖_∞. `inf_norm_upper(M) ≥ ‖M‖_∞`: it returns m(1 + γ_{w+1})(1 + 8u) for the computed largest row sum m of the
computed moduli of a row of w entries. The exact row sum is at most m/((1 − 3u)(1 − γ_{w−1})) (moduli under H-mod; the
sum of w non-negative terms), and (1 + γ_{w+1})(1 − γ_{w−1}) ≥ (1 − 2wu)/(1 − (w + 1)u)² ≥ 1 + 2u − 4w(w + 1)u², so with the two
roundings of the product (1 + γ_{w+1})(1 + 8u)(1 − u)²(1 − 3u)(1 − γ_{w−1}) ≥ (1 + 2u − 4w(w + 1)u²)(1 + 3u − 40u²) ≥ 1 for
w ≤ 2²⁶. `up` and `safe_sqrt_upper` add one guard each over one rounding.
(iv) Lemma 3 with ‖W‖ ≤ c(W) and θ < 1; `sub_down(1, θ)` ≤ 1 − θ; the quotient and its rounding as ε in Proposition 5 (iv).
κ₂(V̂) = ‖V̂‖‖V̂⁻¹‖ ≤ c(V̂)β, rounded up by `up`. ∎

**Proposition 7.** (i) Sᵢ ≥ ‖rᵢ‖ gives ‖R‖_F = ‖(‖r₁‖, …, ‖rₙ‖)‖ ≤ ‖(S₁, …, Sₙ)‖ ≤ ρ_R; ρ is rounded as ε of Proposition 5 (iv);
‖F‖ ≤ ‖V̂⁻¹‖‖R‖ ≤ β‖R‖_F. (ii) Let x = Re μ̂ᵢ − Re μ̂ⱼ and x̂ = fl(x). If x̂ is subnormal the subtraction is exact; otherwise
|x̂| ≤ (1 + u)|x|; if x̂ = ±∞ then |x| exceeds the largest binary64 number and hence 2ρ whenever 2ρ is finite. The same holds
for the imaginary parts. With m = max(|x̂|, |ŷ|): if m is normal, |μ̂ᵢ − μ̂ⱼ| ≥ m/(1 + u) ≥ m(1 − 2u)(1 + u) ≥ fl(m(1 − 2u)) = δᵢⱼ;
if m is subnormal, m ≤ |μ̂ᵢ − μ̂ⱼ| and δᵢⱼ = fl(m(1 − 2u)) ≤ fl(m) = m (rounding is monotone). 2ρ is exact (a doubling)
unless it overflows, and then no pair is separated. δᵢⱼ > 2ρ gives |μ̂ᵢ − μ̂ⱼ| > 2ρ, so the closed discs of radius ρ are
disjoint. (iii) Indices in different components are not joined, so δᵢⱼ > 2ρ for them. ∎

## 3 Where the code implements it

`certify_ed/nonhermitian.py` (branch `track/non-hermitian`):

* Proposition 5: `residual_bounds` (114–145): ŝ at 126, â at 128, the quotient by 1 − g at 129
  (g, G from `_bound_constants`, 107), p̂(1 + 8u) at 131, the products with G and 1 + 12u at 132–133, T from
  `additive_term` (96), S at 143, ε at 145. The left residuals call the same function on Aᴴ, μ̄ and the left vectors (344).
* Lemma 3 and Proposition 6: `inverse_bound` (163–218): W at 180, P̂ at 191, Ĝ at 192, B at 194–198, T_G at 199,
  θ at 200, the refusal θ ≥ 1 at 208 (reason code `ILL_CONDITIONED_EIGENVECTORS` in the driver, 383), β at 214,
  κ at 218; `matrix_norm2_upper` (154) for c(W) and c(V̂).
* Proposition 7: ρ_R and ρ at 399–400; `disc_groups` (224): δ at 238–240, the test δ > 2ρ at 241, the components at 243.
* The driver `certify_nonhermitian` (284): size limit 323, the no-underflow scan 330, N1 358, the partial claim 374,
  N2 392, N3 408, N4 427 (groups from `disc_groups` at 413).

Rounding directions: residual, bounds, norms of W and V̂, θ, β, ρ and κ upward (`up`, `norm2_upper`, guards); ‖x‖ and 1 − θ
downward (`norm2_lower`, `sub_down`); the separations δᵢⱼ downward. Every statement summary ends with the UNVERIFIED
marker (`nonhermitian.UNVERIFIED`); the reason code `ILL_CONDITIONED_EIGENVECTORS` is in `scope.REASON_CODES` and the
schema `certify-ed/nonhermitian/1.0` in `schema.RECORD_SCHEMAS`; the CLI is `certify-ed nonhermitian`.

**Correction (2026-11-18), H-mod replaced; UNVERIFIED (D-12).** The hypothesis H-mod of (FP), the modulus of a complex
number within 3u, is not established for NumPy's complex absolute value (2.47u measured on the reference host; no
bound is known; `docs/proofs/fp_modulus-up.md`). Since 3.3.0.dev0 `residual_bounds` and `inverse_bound` take every modulus from
`fp.modulus_up` (≥ the exact modulus, Lemma M there, subnormal components included), and `matrix_norm2_upper` gets
upper moduli through `fp.inf_norm_upper`. Propositions 5 (ii) and 6 then hold without H-mod: (|A||x|)_j ≤ â_j/(1 − g)
needs only the w products and sums, and |μ||x_j| ≤ p̂_j(1 + 8u) only the product. The constants g = γ_{w+6} (complex)
and the factor (1 + 8u) are kept, with spare counts; the additive budget of (M-U) is kept as stated. The record's
arithmetic text names `fp.modulus_up` instead of H-mod.

## 4 Computational verification

`docs/proofs/verify_E-02.py` (`--cases 200 --seed 20261111`; exact rational arithmetic with `fractions.Fraction`, complex
numbers as pairs, for every floating-point inequality; mpmath at 60 digits for ‖V̂⁻¹‖, ‖W‖, ‖V̂‖, ‖V̂⁻¹R‖, σ_min(A − μI) and
the eigenvalues of matrices whose spectrum is not planted). Run of 2026-11-11 (`docs/proofs/runs/verify_E-02.txt`; one
case counts one checked statement or stage):

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | random matrices (n = 2 to 8, real and complex, Gaussian and non-normal Schur forms with strict upper part scaled by 1 to 100, claims from `scipy.linalg.eig`, a quarter with values and a quarter with vectors perturbed by 10⁻¹² to 10⁻³, left vectors given): Propositions 5 (iii) and 6 (i) entry by entry exactly, Sᵢ ≥ ‖rᵢ‖ and εᵢ ≥ ‖rᵢ‖/‖xᵢ‖ exactly, σ_min(A − μ̂ᵢI) ≤ εᵢ, the left εᵢ′ exactly, θ ≥ ‖I − WV̂‖_F exactly, β ≥ ‖V̂⁻¹‖, c(W) ≥ ‖W‖, c(V̂) ≥ ‖V̂‖, ρ ≥ ‖V̂⁻¹R‖, every eigenvalue in the union of the discs, the cross-group discs disjoint (exactly) and the count of every group, δᵢⱼ ≤ |μ̂ᵢ − μ̂ⱼ| exactly | 1808 | 0 |
| B | planted spectra A = SJS⁻¹ (S unimodular integer, J a dyadic Jordan form: distinct, complex, repeated semisimple, defective with a block of size 2 or 3, a pair 2⁻³⁰ apart; n = 3 to 7): N3 and N4 against the exact eigenvalues with multiplicity | 888 | 0 |
| C | adversarial: the Bauer–Fike equality family (36 cases), exact Jordan blocks of size 2 to 6, underflow inside the products (A ≈ 2⁻⁸⁹⁵, vectors ≈ 2⁻¹⁵⁰), a stored subnormal entry (every statement INCONCLUSIVE with UNDERFLOW_RISK; with the scan off every certified bound still holds), a duplicated eigenvector (N2 not certified), entries near 2⁵⁰⁰, companion matrices of ∏(z − j), column scalings 2^±30, a wrong eigenvalue, real ill-conditioned matrices with conjugate pairs | 547 adversarial | 0 |

Total: 2696 random + 547 adversarial cases, 0 violations. Certified statements checked: 375 N1, 371 N2, 371 N3 and 371 N4. On the Bauer–Fike
equality family the certified radius exceeds the exact distance ‖V̂⁻¹R‖ = bτ by a relative 9.318e-10 to 3.500e+00. Negative controls,
which must be detected: the backward error εᵢ read as an enclosure radius fails on the equality family in 36/36; Bauer–Fike
without the factor ‖V̂⁻¹‖ (radius ‖R‖_F) in 36/36; the radius ρ/2 in 34/36; the exact stage check of Proposition 5 (iii)
with E replaced by 0 in 200/200 random cases. They show that the conditioning factor is needed and that the harness can fail.
Tests `tests/test_nonhermitian.py` (14): random real matrices with conjugate pairs, the equality family with its two
negative controls, an exact Jordan block that is never over-counted, a duplicated vector (ILL_CONDITIONED_EIGENVECTORS),
left vectors checked exactly, a partial claim, the underflow scan and its opt-out, the size limit and input checks,
`inverse_bound` and `disc_groups` directly, the CLI round trip, a forged status caught by the record hash, this document's
sections and status line, the verification script at reduced size, and the planted-error cases of `corpus/nonhermitian/`.

## 5 Analytic cross-check

1. *Normal A.* For A normal with an orthonormal V̂ (up to rounding) W ≈ V̂ᴴ, θ is of the order of nu, β ≈ 1, and Theorem 4 (b)
   is the Bauer–Fike theorem for normal matrices (κ₂ = 1): every eigenvalue lies within ‖R‖_F of a claimed value. Compared
   with Theorem 1 of `docs/theory.md` for Hermitian A (radius ‖rᵢ‖/‖xᵢ‖ per pair, distinctness from Theorem 2) the E-02
   radius is uniform and at most √n times the largest relative residual larger, the price of not using Hermitian structure.
2. *Diagonal A, V̂ = I.* R = 0, so ρ is made of the additive terms and the rounding guards only; equal diagonal entries
   form one group whose size is their multiplicity, and Theorem 4 (c) returns the multiplicities exactly.
3. *Equality.* For A = [[a, b], [0, a]], x = (1, ±τ), μ = a ± bτ: V̂⁻¹R = (bτ/2)[[−1, −1], [1, 1]], so ‖V̂⁻¹R‖ = bτ = |μ − a|,
   and Theorem 4 (b) holds with equality; no smaller constant is possible. The backward error bτ²/√(1 + τ²) and ‖R‖_F = √2bτ²
   are smaller by a factor of order τ: for a non-normal matrix neither the residual alone nor Bauer–Fike without κ encloses
   the eigenvalue (negative controls D1 and D2 of section 4).
4. *Jordan blocks.* For J_m(λ) + εe_me₁ᵀ the eigenvalues are λ + ε^{1/m}ω^k (ω an m-th root of unity), so an O(u) backward
   error (N1) is compatible with a forward error of order u^{1/m}. Theorem 4 reflects this through κ₂(V̂), which for the
   LAPACK eigenvectors of a near-Jordan block grows like the inverse of the eigenvector separation; for the exact block the
   computed V̂ is still invertible in exact arithmetic but its κ₂ makes the discs overlap, and N4 then reports one group of
   m discs holding m eigenvalues, which is true.
5. *Scaling.* (A, μ̂) → (tA, tμ̂) with t a power of two scales R and ρ by t exactly. A column scaling V̂ → V̂N changes W to N⁻¹W
   and R to RN; the radius depends on N, and by van der Sluis' theorem (1969) columns of equal norm are within a factor √n
   of the best diagonal scaling for κ₂. The code uses V̂ as claimed and records the condition estimate.

## 6 Status

Derived on 2026-11-11; machine-checked (2696 random + 547 adversarial cases, 0 violations; negative controls detected in
36/36, 36/36, 34/36 and 200/200 cases); independent re-derivation: done 2026-11-11 (a second derivation written from section 1 and
section 3 alone, not a human check; verdict AGREES-WITH-REMARKS, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending. The code is
marked UNVERIFIED until that check.
Correction for complex moduli on 2026-11-18 (end of section 3; UNVERIFIED, D-12).
