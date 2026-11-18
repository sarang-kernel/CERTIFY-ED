# Independent re-derivation: E-02 (non-Hermitian eigenproblems: backward errors, Bauer–Fike discs, counting)

Written from sections 1 and 3 of `docs/proofs/E-02_non_hermitian.md` only (section 2 read afterwards; comparison at the
end). This is a second derivation produced automatically, without the first proof in view; it is not a human check.
Notation and hypotheses (FP), (M-U), H-mod as there. Classical facts used: the variational characterisation of the
smallest singular value, σ_min(M) = min_{z≠0}‖Mz‖/‖z‖ = σ_min(Mᴴ); the argument principle and Hurwitz's theorem for
polynomials; Higham (2002) Lemmas 3.1, 3.3 and 3.5 and eq. (3.5); the contracts of `fp.norm2_upper`, `fp.norm2_lower` and
`fp.inf_norm_upper` as proved in `int05_underflow.md` (taken as given here).

## Proofs

**Lemma 1.** σ_min(A − μI) ≤ ‖(A − μI)x‖/‖x‖ = ‖r‖/‖x‖ by the variational characterisation. The matrix E = −rxᴴ/(xᴴx)
satisfies Ex = −r, so (A + E)x = Ax − r = μx, and for any z, ‖Ez‖ = ‖r‖|xᴴz|/‖x‖² ≤ ‖r‖‖z‖/‖x‖ with equality at z = x:
‖E‖ = ‖r‖/‖x‖. For the left vector: σ_min(A − μI) = σ_min(Aᴴ − μ̄I) ≤ ‖s‖/‖y‖, and F = −ysᴴ/(yᴴy) gives
yᴴF = −sᴴ, yᴴ(A + F) = yᴴA − sᴴ = μyᴴ, ‖F‖ = ‖s‖/‖y‖ by the same computation. The example: for A = [[0, 1], [0, 0]],
x = (1, t), μ = t, r = (t − t, −t²) = (0, −t²), so ‖r‖/‖x‖ < t while Λ(A) = {0}. ∎

**Theorem 2 (vector route).** Let (A + E)v = λv with v ≠ 0 and put w = V⁻¹v ≠ 0. Then VDw + EVw = λVw, so
(D − λI)w = −V⁻¹EVw and min_i|dᵢ − λ|·‖w‖ ≤ ‖(D − λI)w‖ = ‖V⁻¹EVw‖ ≤ ‖V⁻¹EV‖‖w‖ (for a diagonal matrix
‖(D − λI)w‖² = Σᵢ|dᵢ − λ|²|wᵢ|² ≥ min_i|dᵢ − λ|²‖w‖²). Divide by ‖w‖ > 0. The residual form follows from Lemma 1 (μ is an
eigenvalue of A + E with ‖E‖ = ‖r‖/‖x‖). This route needs no case distinction λ = dᵢ. ∎

**Lemma 3 (vector route).** For every z: ‖z‖ ≤ ‖WV̂z‖ + ‖Gz‖ ≤ ‖W‖‖V̂z‖ + θ‖z‖, so (1 − θ)‖z‖ ≤ ‖W‖‖V̂z‖. Hence V̂z = 0
forces z = 0: V̂ is injective, and a square matrix that is injective is invertible. For y = V̂z this reads
‖V̂⁻¹y‖ ≤ ‖W‖‖y‖/(1 − θ). W is invertible because WV̂ = I − G is (‖G‖ < 1) and V̂ is; then V̂⁻¹ = (I − G)⁻¹W. ∎

**Theorem 4.** (a) V̂⁻¹AV̂ = V̂⁻¹(V̂D̂ + R) = D̂ + F. (b) Theorem 2 (vector route) for the matrix D̂ + F = I·D̂·I⁻¹ + F gives
min_i|λ − μ̂ᵢ| ≤ ‖F‖ ≤ ρ for every eigenvalue λ. ‖F‖ ≤ ‖V̂⁻¹‖‖R‖ ≤ β‖R‖_F.
(c) (argument principle instead of continuous root functions) Let δ = dist(U_q, U_q′) > 0 (two disjoint compact sets) and
Ω = {z : dist(z, U_q) < δ/2}, a bounded open set whose boundary is a finite union of circular arcs (Ω is the union of the
open discs of radius ρ + δ/2 around the centres of G_q). For t ∈ [0, 1] the zeros of p_t(z) = det(zI − D̂ − tF) lie in U
(part (b) for tF), and no point of ∂Ω is in U: such a point has distance δ/2 from U_q and at least δ/2 from U_q′. The number
of zeros of p_t in Ω, N(t) = (2πi)⁻¹∮_{∂Ω} p_t′(z)/p_t(z) dz, is an integer that depends continuously on t (the integrand is
continuous in (t, z) on [0, 1] × ∂Ω, where p_t does not vanish), hence constant. At t = 0, p_0(z) = ∏(z − μ̂ᵢ) has in Ω
exactly the zeros μ̂ᵢ with i ∈ G_q (a centre outside G_q lies in U_q′, at distance ≥ δ from U_q). The zeros of p_1 in Ω are
those in U_q (zeros lie in U, and Ω ∩ U = U_q). So U_q holds |G_q| eigenvalues of D̂ + F, hence of A, with multiplicity.
(d) |G_q| = 1 gives one eigenvalue of multiplicity one; n disjoint discs with one simple eigenvalue each give n distinct
eigenvalues, so A is diagonalizable. (e) ‖R‖_F² = Σᵢ‖xᵢ‖²‖r̃ᵢ‖² ≤ maxᵢ‖xᵢ‖² Σᵢ‖r̃ᵢ‖² and ‖xᵢ‖ = ‖V̂eᵢ‖ ≤ ‖V̂‖. ∎

**Proposition 5 (own accounting).** Standard model first. For complex data each entry of a = fl(Ax) satisfies
|aⱼ − (Ax)ⱼ| ≤ √2γ_{w+2}(|A||x|)ⱼ ≤ 2γ_{w+4}(|A||x|)ⱼ (Higham §3.6); cⱼ = fl(μxⱼ) has |cⱼ − μxⱼ| ≤ √2γ₂|μ||xⱼ| ≤ γ₃|μ||xⱼ|
(Lemma 3.5; with a fused multiply-add the bound is 2u, Jeannerod, Kornerup, Louvet and Muller 2017, also ≤ γ₃); the
subtraction is componentwise, |ŝⱼ − (aⱼ − cⱼ)| ≤ u|aⱼ − cⱼ|. Adding, with |aⱼ − cⱼ| ≤ (1 + 2γ_{w+4})(|A||x|)ⱼ + (1 + γ₃)|μ||xⱼ|:
|ŝⱼ − rⱼ| ≤ 2γ_{w+5}(|A||x|)ⱼ + γ₄|μ||xⱼ| ≤ G·((|A||x|)ⱼ + |μ||xⱼ|), G = 2γ_{w+6}. For real data the same with γ_w, u and
G = γ_{w+2}. The computed moduli m̂ ≥ |z|(1 − 3u) (H-mod) and the w-term non-negative BLAS sum give âⱼ ≥ (|A||x|)ⱼ(1 − 3u)²(1 − γ_w)
≥ (|A||x|)ⱼ(1 − γ_{w+6}); p̂ⱼ ≥ |μ||xⱼ|(1 − 3u)²(1 − u). The code's E = G(â/(1 − g) + p̂(1 + 8u))(1 + 12u) with g = γ_{w+6}:
the exact value of the formula dominates G((|A||x|)ⱼ + |μ||xⱼ|) because 1/(1 − γ_{w+6}) undoes the shortfall of â and
(1 + 8u)(1 − 3u)²(1 − u) = (1 + 8u)(1 − 7u + 15u² − 9u³) = 1 + u − 41u² + 111u³ − 72u⁴ ≥ 1 undoes that of p̂.
The computed E has six roundings and three constant errors (Ĝ two, ĝ one): with (1 + 12u)(1 − u)⁹ ≥ 1 it dominates the
exact formula. Under (M-U): the products of the computation of ŝⱼ (at most 4w + 4) and the multiplicative operations of
Eⱼ (at most w + 5) add at most u_min each, multiplied on the way by factors ≤ 2: per entry ≤ (8w + 8 + 2w + 10)u_min ≤
(10w + 20)u_min, and T ≥ √n(10w + 20)u_min covers the norm of these. The norm, the sum and the quotient: as in the
statement (norm2_upper, norm2_lower contracts; `up` covers three roundings of non-negative quantities, (1 − u)³(1 + 8u) ≥ 1;
the subnormal case of the quotient is covered by the added 2⁻¹⁰⁷³). ∎

**Proposition 6 (own accounting).** P̂ = fl(WV̂): |P̂ᵢⱼ − (WV̂)ᵢⱼ| ≤ γ̇ₙ(|W||V̂|)ᵢⱼ. Ĝ = fl(I − P̂): off-diagonal entries are
negations (exact); diagonal entries round one real subtraction, error ≤ u|1 − P̂ᵢᵢ| ≤ u|Ĝᵢᵢ|/(1 − u) ≤ u m̂ᵢᵢ/((1 − u)(1 − 3u))
≤ 2u m̂ᵢᵢ. The code adds 4u m̂ᵢᵢ (exact product, power of two) before the final guard: a factor 2 to spare. (|W||V̂|)ᵢⱼ ≤
fl(|W||V̂|)ᵢⱼ/(1 − g) as in Proposition 5. Rounding of B: five roundings and three constant errors, (1 + 12u)(1 − u)⁸ ≥ 1.
(M-U): ≤ 4n products in P̂ᵢⱼ and n + 4 in Bᵢⱼ, so ≤ (10n + 8)u_min per entry, and the n² entries give the Frobenius term
n(10n + 20)u_min of T_G. ‖I − WV̂‖ ≤ ‖I − WV̂‖_F ≤ ‖Ĝ‖_F + ‖B‖_F + T_G-term, all rounded up. Norm bounds: ‖M‖ ≤ ‖M‖_F; ‖M‖ ≤
√(‖M‖₁‖M‖_∞) (for x of unit norm, ‖Mx‖² = xᴴMᴴMx ≤ ‖MᴴM‖ ≤ ‖Mᴴ‖_∞‖M‖_∞ = ‖M‖₁‖M‖_∞, the ∞-norm being submultiplicative and
an upper bound of the spectral radius of the Hermitian MᴴM). inf_norm_upper for complex M: the guard (1 + γ_{w+1})(1 + 8u)
against w − 1 additions, the 3u of each modulus and the two roundings of the guard product: (1 + γ_{w+1})(1 − γ_{w−1}) ≥
1 + 2u − 4w(w + 1)u² and (1 + 8u)(1 − u)²(1 − 3u) ≥ 1 + 3u − 33u², so the product is at least 1 + 5u − (4w(w + 1) + 40)u², which
exceeds 1 for w ≤ 10⁸. β, κ: Lemma 3 (vector route) and the
rounding of a quotient as for ε. ∎

**Proposition 7.** (i) as for ε. (ii) Own argument: |μ̂ᵢ − μ̂ⱼ| ≥ max(|Re(μ̂ᵢ − μ̂ⱼ)|, |Im(μ̂ᵢ − μ̂ⱼ)|); a computed difference
x̂ satisfies |x̂| ≤ |x|(1 + u) (exact when subnormal), so max(|x̂|, |ŷ|)/(1 + u) ≤ |μ̂ᵢ − μ̂ⱼ|, and
fl(m(1 − 2u)) ≤ m(1 − 2u)(1 + u) ≤ m/(1 + u) when m(1 − 2u) is normal, fl(m(1 − 2u)) ≤ m when it is subnormal (then x̂, ŷ are
exact). (iii) immediate. ∎

## Code mapping (section 3 checked against `certify_ed/nonhermitian.py`)

The line numbers of section 3 agree with the file: `residual_bounds` 114, ŝ at 126, â at 128, the
quotient at 129, p̂(1 + 8u) at 131, G and 1 + 12u at 132 and 133, S at 143, ε at 145; `inverse_bound` 163
with P̂ 191, Ĝ 192, B 194 to 198, θ 200, β 214; `disc_groups` 224 with δ 238 to 240 and the strict test
241. The directions are as stated: every bound upward, ‖x‖ and 1 − θ downward, separations downward, and the test δ > 2ρ
strict (touching discs are joined).

### Comparison with section 2 (read after the above)

Lemma 1, Theorem 2, Lemma 3, Theorem 4 and Propositions 5 to 7 agree with my derivations. Section 2 proves Theorem 2 by
the factorisation A + E − λI = V(D − λI)(I + K)V⁻¹ and the Neumann series and Lemma 3 by the Neumann series; I used the
vector forms ((D − λI)w = −V⁻¹EVw and (1 − θ)‖z‖ ≤ ‖W‖‖V̂z‖), which are the forms a formal proof most easily takes.
Section 2 proves the count of Theorem 4 (c) with continuous root functions on [0, 1] (Kato); I used the argument
principle on the boundary of a δ/2-neighbourhood of U_q. Both are valid; mine also shows directly that the groups need
not be connected sets. The constants of Propositions 5 and 6 are the same; my own count of the roundings of E and B gives
the same budgets (nine and eight against the guard 1 + 12u).

**Remarks.** (1) Proposition 5 (ii) as first written stated the two inequalities "each up to (2w + 10)u_min together with
the rounding of E", which mixes the standard-model inequality with the additive budget; the precise statement is: in the
standard model (ii) holds as written, and under (M-U) the shortfall of â, p̂ and the rounding of E together is at most
(2w + 10)u_min per entry (this is how section 2 uses it). (2) The hypothesis (FP) speaks of BLAS inner products and of
moduli; the elementwise complex products fl(μxⱼ) of NumPy may be contracted to fused multiply-adds by the compiler, for
which Higham's Lemma 3.5 does not apply literally; the bound 2u of Jeannerod, Kornerup, Louvet and Muller (2017) is also
below γ₃, so the proof stands, and (FP) should mention it. (3) With `underflow_check=False`, H-mod is assumed also for
subnormal components of complex stored numbers, where the modulus is rounded to the subnormal grid and its relative
error is not bounded by 3u; the record lists the assumptions, and the project's convention (the opt-out assumes the
no-underflow hypothesis instead) covers it, but the derivation should say that the opt-out path assumes H-mod for such
moduli. With the default scan no stored component is below 2⁻⁹⁰⁰ and H-mod is the hypothesis of D-01. (4) The radius is
uniform (one ρ for all discs), so a single ill-conditioned eigenvalue widens every disc; per-disc radii (Gershgorin on
D̂ + F with entrywise bounds on F) would be sharper and are listed as not implemented. (5) N4 is certified also when it is
uninformative (one group holding all n discs, for instance an exact Jordan block with LAPACK vectors, where κ₂(V̂) is
enormous); the statement is then true and the summary says that no disc is separated.

Remarks (1) to (3) were applied to section 1 of the document on 2026-11-11 (wording of Proposition 5 (ii) and two
sentences in (FP)); no constant, bound or status changed. Remarks (4) and (5) are limitations, recorded in section 1
(not implemented) and in the N4 summary text.

**Verdict: AGREES-WITH-REMARKS.**

### Runs

* `python docs/proofs/verify_E-02.py --cases 40 --seed 7` (the original script at reduced size): 539 random +
  558 adversarial cases, 0 violations; negative controls D1_eps_as_radius: detected in 36/36 cases; D2_no_conditioning: detected in 36/36 cases; D3_half_radius: detected in 34/36 cases; D4_zero_E: detected in 40/40 cases.
* `python docs/proofs/independent/adversarial_E-02_non_hermitian.py --cases 120 --seed 113` (own ground truth, mpmath at
  60 digits, 500 digits for Part D): A: 120 matrices (N3 certified in 96, N2 INCONCLUSIVE in 24); B: 360 centre pairs at the separation boundary; D: 59 opt-out matrices with subnormal complex components and vectors up to 2^500 (1 skipped: LAPACK geev did not converge); C negative controls: Theorem 1 radius misses the nearest eigenvalue in 57/76 pairs; Bauer-Fike without beta leaves an eigenvalue outside in 10/24 matrices; violations: 0.
* While Part D was written, a fault of the instrument of the original script was found and corrected: `verify_E-02.py`
  computed σ_min(A − μI) from the smallest eigenvalue of (A − μI)ᴴ(A − μI) at 60 digits, which is not resolved when
  σ_min is below about 10⁻³⁰‖A − μI‖; on a matrix with subnormal complex entries it reported σ_min = 7.9e-31 against
  ε = 1.2e-32 although the exact checks ε ≥ ‖r‖/‖x‖ held. The script now uses mpmath's SVD, and the recorded runs are
  from the corrected script. No bound of the code was involved.
