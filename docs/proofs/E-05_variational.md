# E-05 Variational certificates for non-ED states: Rayleigh upper bound, existence radius, Temple lower bound

Notation as in `docs/theory.md`: H Hermitian n × n with eigenvalues λ₁ ≤ λ₂ ≤ … ≤ λ_n; ψ ≠ 0 a stored vector (the
trial state); θ = ⟨ψ, Hψ⟩/⟨ψ, ψ⟩ its Rayleigh quotient; r = ‖Hψ − θψ‖/‖ψ‖ its residual (so r² = ⟨ψ, H²ψ⟩/⟨ψ, ψ⟩ − θ²,
the energy variance); u = 2⁻⁵³, γ_m = mu/(1 − mu), γ̇_m = `fp.gamma_dot(m)` (γ_m for real data, 2γ_{m+4} for complex
data); w the largest number of nonzeros in a row of H (`adapter.width`). When the operator is built from a spec, H_ref
is the floating-point build, H_spec the operator the spec defines, and σ ≥ ‖H_ref − H_spec‖₂ the model shift
(`certify._builder_shift`, `docs/proofs/appA_model-shift.md`). ‖·‖ is the Euclidean norm of a vector and the
spectral norm of a matrix.

## 1 Statement

**Lemma 1 (Rayleigh–Ritz; e.g. Parlett 1998, Thm 1.3.2 / Horn & Johnson 2013, Thm 4.2.2).** λ₁ ≤ θ ≤ λ_n for every
ψ ≠ 0. No hypothesis on ψ.

**Lemma 2 (Theorem 1 at the Rayleigh quotient).** Some eigenvalue λ_j of H satisfies |λ_j − θ| ≤ r. Moreover
r² = ⟨ψ, H²ψ⟩/⟨ψ, ψ⟩ − θ².

**Proposition 1 (Temple 1928; Kato 1949, Lemma 1 / Parlett 1998, Thm 4.6.3).** Let ρ be a real number with
θ < ρ ≤ λ₂. Then

    λ₁ ≥ θ − r² / (ρ − θ).

If λ₁ = λ₂ no such ρ exists (θ ≥ λ₁ = λ₂ ≥ ρ contradicts θ < ρ); the statement is empty, not false. For the
two-level trial state ψ = cos a · v₁ + sin a · v₂ (v_i exact eigenvectors) and ρ = λ₂ the bound is an equality.

**Lemma 3 (floating-point enclosure of θ and r; `enclosure.rayleigh_interval`, `variational.certify_variational`).**
Let ŷ = fl(Hψ), N̂ and D̂ the values returned by `fp.exact_dot(ψ, ŷ)` and `fp.exact_dot(ψ, ψ)` with their error
bounds e_N, e_D (|N̂ − ⟨ψ, ŷ⟩| ≤ e_N, |D̂ − ⟨ψ, ψ⟩| ≤ e_D), and â = fl(|H||ψ|). Put

    E_N = e_N + γ̇_{w+2} · fl(|ψ|ᵀ â) · (1 + γ_{n+2}) / (1 − γ_{w+1}),   E_D = e_D,

both rounded up by the guard of `fp.up`. If D̂ − E_D > 0 then θ ∈ [θ_lo, θ_hi], where θ_lo, θ_hi are the minimum
and maximum of the four quotients (N̂ ∓ E_N)/(D̂ ± E_D), each then moved outward by a relative 4u. With
θ̂ = fl(N̂/D̂) and r̂ the Theorem 1 radius at θ̂ (`enclosure.residuals`, `docs/proofs/thm1_fp-accounting.md`:
r̂ ≥ ‖Hψ − θ̂ψ‖/‖ψ‖),

    r ≤ r_stored := up( r̂ + max(θ_hi − θ̂, θ̂ − θ_lo) ).

All of this is under the standard model with the no-underflow hypothesis for the entries of ψ and H (recorded;
`fp.underflow_scan` makes the statements INCONCLUSIVE with UNDERFLOW_RISK when an entry below 2⁻⁹⁰⁰ is found) and is
about the stored matrix. For a spec (model shift σ): λ₁(H_spec) ≤ θ_hi + σ; some eigenvalue of H_spec lies in
[θ_lo − r_stored − σ, θ_hi + r_stored + σ]; and if θ_hi < ρ − σ, with ρ ≤ λ₂(H_spec),

    λ₁(H_spec) ≥ θ_lo − r² / (ρ − σ − θ_hi) − σ   for any r ≥ r_stored.

**Lemma 4 (MPS contraction; `variational.state_from_mps`).** Let A⁽¹⁾, …, A⁽ᴸ⁾ be the tensors of an open-boundary
MPS (A⁽ʲ⁾ of shape D_{j−1} × d_j × D_j, D₀ = D_L = 1), ψ the exact vector ψ[s₁…s_L] = A⁽¹⁾[s₁]⋯A⁽ᴸ⁾[s_L] and ψ̂ the
left-to-right floating-point contraction T̂₁ = A⁽¹⁾[0], T̂_j = fl(T̂_{j−1} A⁽ʲ⁾) (each entry an inner product of
length D_{j−1}). Then, entrywise,

    |ψ̂ − ψ| ≤ c_L · |A⁽¹⁾|⋯|A⁽ᴸ⁾|,   c_L = ∏_{j=2}^{L} (1 + g_j) − 1 ≤ γ_M,   g_j = γ̇_{D_{j−1}},

with M = Σ_{j≥2} D_{j−1} for real data and Σ_{j≥2} 2(D_{j−1} + 4) for complex data, so
‖ψ̂ − ψ‖ ≤ e := `norm2_upper`(up(γ_M) · P̂/(1 − γ_M)), where P̂ = fl(|A⁽¹⁾|⋯|A⁽ᴸ⁾|).

**Lemma 5 (Rayleigh quotient of the source state).** If ‖ψ − ψ̂‖ ≤ e < ‖ψ̂‖ then

    |θ(ψ) − θ(ψ̂)| ≤ ( 2 r(ψ̂) ‖ψ̂‖ e + (‖H‖ + |θ(ψ̂)|) e² ) / (‖ψ̂‖ − e)².

The certified statements V1–V3 are about the contracted vector ψ̂ as stored (Lemma 1 needs no hypothesis on the
vector, so ψ̂ is a legitimate trial state in its own right); Lemma 5 is recorded as information about the source MPS.

**Statements of the record** (`certify_ed.variational`, schema `certify-ed/variational/1.0`): V1 λ₁ ≤ θ_hi (+σ),
CERTIFIED unconditionally; V2 an eigenvalue in [θ_lo − r, θ_hi + r], CERTIFIED; V3 the Temple bound, with status
CERTIFIED when ρ is a certified lower bound on λ₂ read from a certificate of a 'lowest' claim on the same operator
(second smallest lower endpoint of its S2 enclosures when k ≥ 2 — two distinct eigenvalues lie in two distinct
enclosures, Corollary 5 — or S3's `next_eigenvalue_lower_bound` when k = 1), and status CERTIFIED_UNDER_HYPOTHESIS
with the named hypothesis `declared_gap` (ρ ≤ λ₂, the producer's source text) otherwise; V4 producer-declared
stochastic estimates, status DECLARED (reason code DECLARED_ESTIMATE), nothing rests on them.

## 2 Proof

**Lemma 1.** Expand ψ = Σ c_j v_j in an orthonormal eigenbasis: θ = Σ λ_j |c_j|² / Σ |c_j|², a convex combination of
the eigenvalues, hence in [λ₁, λ_n]. ∎

**Lemma 2.** Theorem 1 (`docs/theory.md` §2) with μ = θ. For the variance identity: ‖Hψ − θψ‖² = ⟨ψ, H²ψ⟩ −
2θ⟨ψ, Hψ⟩ + θ²⟨ψ, ψ⟩ = ⟨ψ, H²ψ⟩ − θ²⟨ψ, ψ⟩ because ⟨ψ, Hψ⟩ = θ⟨ψ, ψ⟩; divide by ⟨ψ, ψ⟩. ∎

**Proposition 1.** Every eigenvalue λ_j satisfies (λ_j − λ₁)(λ_j − ρ) ≥ 0: for j = 1 the first factor is 0; for j ≥ 2,
λ_j ≥ λ₂ ≥ ρ and λ_j ≥ λ₁, so both factors are ≥ 0. Hence the Hermitian operator (H − λ₁)(H − ρ) is positive
semidefinite and ⟨ψ, (H − λ₁)(H − ρ)ψ⟩ ≥ 0, i.e. ⟨ψ, H²ψ⟩ − (λ₁ + ρ)⟨ψ, Hψ⟩ + λ₁ρ⟨ψ, ψ⟩ ≥ 0. Dividing by ⟨ψ, ψ⟩ and
using Lemma 2: r² + θ² − (λ₁ + ρ)θ + λ₁ρ ≥ 0, i.e. λ₁(ρ − θ) ≥ ρθ − θ² − r² = θ(ρ − θ) − r². Since ρ − θ > 0,
λ₁ ≥ θ − r²/(ρ − θ). Equality case: for ψ = cos a · v₁ + sin a · v₂ only j = 1, 2 carry weight and (λ_j − λ₁)(λ_j − λ₂) = 0
for both, so the inequality is an equality when ρ = λ₂. ∎

**Lemma 3.** (i) *Numerator.* ⟨ψ, Hψ⟩ = ⟨ψ, ŷ⟩ + ⟨ψ, Hψ − ŷ⟩. Each entry of ŷ is an inner product of at most w
terms, so |ŷ − Hψ| ≤ γ̇_w |H||ψ| entrywise (Higham 2002, eq. (3.5); the complex constant of `gamma_dot` dominates
§3.6), hence |⟨ψ, Hψ − ŷ⟩| ≤ |ψ|ᵀ γ̇_w |H||ψ|. The computed â = fl(|H||ψ|) has non-negative entries, each an inner
product of at most w non-negative terms, so |H||ψ| ≤ â/(1 − γ_w) ≤ â/(1 − γ_{w+1}); the computed fl(|ψ|ᵀ â) of
non-negative numbers satisfies |ψ|ᵀâ ≤ fl(|ψ|ᵀâ)/(1 − γ_n) ≤ fl(|ψ|ᵀâ)(1 + γ_{n+2}) (since 1/(1 − γ_n) = 1 +
nu/(1 − 2nu) ≤ 1 + γ_{n+2}). γ̇_w ≤ γ̇_{w+2}. Together |⟨ψ, Hψ⟩ − N̂| ≤ e_N + γ̇_{w+2} fl(|ψ|ᵀâ)(1 + γ_{n+2})/(1 − γ_{w+1}) =
E_N before the guard; the guard (1 + 8u) of `fp.up` covers the four roundings of forming this expression (product,
product, division, sum; each ≤ u relative, (1 + u)⁴ ≤ 1 + 8u). (ii) *Denominator.* |⟨ψ, ψ⟩ − D̂| ≤ e_D by the
contract of `exact_dot`. (iii) *Quotient.* θ = ⟨ψ, Hψ⟩/⟨ψ, ψ⟩ with numerator in [N̂ − E_N, N̂ + E_N] and denominator
in [D̂ − E_D, D̂ + E_D] ⊂ (0, ∞): the quotient is monotone in each argument on these intervals (the numerator interval
may contain 0, which is why both signs are tried), so its range is between the smallest and the largest of the
four corner quotients. Each corner is formed with two roundings (the sum or difference, the division) and the exact
corner lies within a relative (1 + u)² − 1 < 4u of the computed one; the outward step by a relative 4u covers it
(for θ_lo < 0 the step is away from zero, as required by the sign of the correction). (iv) *Residual.* For the exact θ,
‖Hψ − θψ‖ ≤ ‖Hψ − θ̂ψ‖ + |θ − θ̂|‖ψ‖, so r ≤ r̂ + |θ − θ̂| ≤ r̂ + max(θ_hi − θ̂, θ̂ − θ_lo) because θ ∈ [θ_lo, θ_hi] and
θ̂ ∈ [θ_lo, θ_hi] as well (θ̂ = fl(N̂/D̂) lies within a relative u of N̂/D̂, which lies between the corners); one
addition and the guard of `up` cover the roundings. (v) *Model shift.* By Weyl's inequality |λ_j(H_spec) −
λ_j(H_ref)| ≤ ‖H_spec − H_ref‖ ≤ σ for every j. V1: λ₁(H_spec) ≤ λ₁(H_ref) + σ ≤ θ_hi + σ. V2: the eigenvalue of
H_ref within r_stored of θ ∈ [θ_lo, θ_hi] (Lemma 2 for H_ref) has an eigenvalue of H_spec within σ of it. V3: the
hypothesis ρ ≤ λ₂(H_spec) gives ρ − σ ≤ λ₂(H_ref); if θ_hi < ρ − σ then Proposition 1 for H_ref with ρ′ = ρ − σ and
the exact θ, r of H_ref (θ ∈ [θ_lo, θ_hi], r ≤ r_stored ≤ r) gives λ₁(H_ref) ≥ θ − r²/(ρ′ − θ) ≥ θ_lo − r²/(ρ′ − θ_hi) (the
function θ ↦ θ − r²/(ρ′ − θ) is concave on (−∞, ρ′), so its minimum over [θ_lo, θ_hi] is at an endpoint, and
θ_lo − r²/(ρ′ − θ_hi) is below both endpoint values); then λ₁(H_spec) ≥ λ₁(H_ref) − σ. ∎

**Lemma 4.** Induction on j with P_j := |A⁽¹⁾|⋯|A⁽ʲ⁾| (entrywise absolute values) and the claim |T̂_j − T_j| ≤ c_j P_j,
c₁ = 0 (the first tensor is stored, not computed). Step: each entry of fl(T̂_{j−1}A⁽ʲ⁾) is an inner product of D_{j−1}
stored numbers, so fl(T̂_{j−1}A⁽ʲ⁾) = T̂_{j−1}A⁽ʲ⁾ + F with |F| ≤ g_j |T̂_{j−1}||A⁽ʲ⁾| (Higham eq. (3.5) and `gamma_dot` for
complex data). Then |T̂_j − T_j| ≤ |T̂_{j−1} − T_{j−1}||A⁽ʲ⁾| + g_j(|T_{j−1}| + |T̂_{j−1} − T_{j−1}|)|A⁽ʲ⁾| ≤ (c_{j−1} +
g_j(1 + c_{j−1})) P_j = ((1 + c_{j−1})(1 + g_j) − 1) P_j, using |T_{j−1}| ≤ P_{j−1} and |T_{j−1}||A| ≤ P_j. The computed P̂
(non-negative inner products, D_{j−1} terms per step) satisfies P_L ≤ P̂/∏(1 − γ_{D_{j−1}}) ≤ P̂/(1 − γ_M). Both
products of factors use Higham 2002, Lemma 3.3: ∏(1 + γ_{a_j}) ≤ 1 + γ_{Σa_j} (so c_L ≤ γ_M with a_j = D_{j−1} for real
data), and for complex data 1 + 2γ_{a+4} ≤ (1 + γ_{a+4})² ≤ 1 + γ_{2(a+4)} (so a_j = 2(D_{j−1} + 4); the real P̂ needs
only the real constants, so M overstates there). γ_M = Mu/(1 − Mu) is computed with two roundings (Mu is exact),
1 − γ_M with one, the division and the product with one each: five roundings, covered by the guard (1 + 8u) of
`fp.up` applied to γ_M together with the (1 + 8u) guard inside `norm2_upper` ((1 + u)⁵ ≤ (1 + 8u)). Hence
‖ψ̂ − ψ‖₂ ≤ ‖γ_M P_L[:, 0]‖₂ ≤ `norm2_upper`(computed entrywise bound). ∎

**Lemma 5.** Write ψ = ψ̂ + d with ‖d‖ ≤ e, θ̂ = θ(ψ̂). Then θ(ψ) − θ̂ = ⟨ψ, (H − θ̂)ψ⟩/⟨ψ, ψ⟩ and
⟨ψ, (H − θ̂)ψ⟩ = ⟨ψ̂, (H − θ̂)ψ̂⟩ + 2 Re⟨d, (H − θ̂)ψ̂⟩ + ⟨d, (H − θ̂)d⟩, whose first term vanishes by the definition of θ̂.
Cauchy–Schwarz: |2 Re⟨d, (H − θ̂)ψ̂⟩| ≤ 2e‖(H − θ̂)ψ̂‖ = 2e r(ψ̂)‖ψ̂‖ and |⟨d, (H − θ̂)d⟩| ≤ (‖H‖ + |θ̂|)e². The denominator
satisfies ‖ψ‖ ≥ ‖ψ̂‖ − e > 0. ∎

**Statement V3 from a certificate (ρ certified).** For a 'lowest' claim with S2 and S3 CERTIFIED, the k claimed
enclosures contain k distinct eigenvalues (Corollary 5) and no other eigenvalue lies below the certified count
threshold, so {λ₁, …, λ_k} are the eigenvalues in the enclosures. If k ≥ 2, λ₁ and λ₂ lie in two distinct enclosures
I_a, I_b, so λ₂ ≥ max(lo_a, lo_b) ≥ the second smallest lower endpoint. If k = 1, λ₂ ≥ `next_eigenvalue_lower_bound`
is S3's gap statement. The certificate's radii already contain its model shift (`radius_refers_to`), so ρ bounds
λ₂ of the operator the certificate refers to; the record requires the same spec hash and dimension. ∎

## 3 Where the code implements it

`certify_ed/variational.py` (branch `track/variational`):

* Lemma 3 (i)–(iv): `enclosure.rayleigh_interval` (enclosure.py lines 660–695; E_N at 682, E_D at 683, the four
  corners 687–688, the outward 4u at 690–691, r̂ from `residuals` at 693), called at variational.py line 334 (only
  when the no-underflow scan at 325–332 found nothing); r_stored at 348 (`up`); r = r_stored + σ at 349 (`add_up`);
  the V1 upper bound at 350–351 (`add_up` with σ, then one `nextafter` outward); the V2 interval at 376
  (`sub_down`/`add_up` with r).
* Proposition 1 and Lemma 3 (v): `_temple` (405–460): ρ_eff = ρ − σ rounded down at 435, the hypothesis check θ_hi < ρ_eff
  at 438 (INCONCLUSIVE otherwise, never a bound), r² rounded up (+ ULP_MIN for a subnormal product) at 444, the
  correction r²/(ρ_eff − θ_hi) rounded up at 445 (its denominator rounded down by `sub_down`), lower = θ_lo − correction
  rounded down at 446, − σ at 448, and a relative 8u step away from zero at 449. Status CERTIFIED only when
  `gap_from_certificate` (168–217: second smallest lower endpoint at 202, k = 1 path at 211) returned a certified ρ;
  CERTIFIED_UNDER_HYPOTHESIS with `hypotheses = [{'name': 'declared_gap', ...}]` and the words "UNDER THE DECLARED
  HYPOTHESIS" in the summary otherwise; `_vstmt` asserts that a statement under hypothesis names one, and
  `schema.validate_record` rejects a record that does not.
* Lemma 4: `state_from_mps` (101–149): M accumulated at 140, γ_M at 145, the entrywise bound at 146 (`up(γ_M)`, division
  by 1 − γ_M), e at 147 (`norm2_upper`); SIZE_LIMIT above 2²² at 130–132. Lemma 5: lines 355–369 (`dth` at 363, with
  `norm2_upper`/`norm2_lower` for ‖ψ̂‖ and the adapter's norm bound for ‖H‖).
* Model shift σ: `certify._builder_shift` at line 267; the no-underflow scan at 325–332 (state entries, operator entries
  or norm bound and σ); its findings make V1–V3 INCONCLUSIVE (UNDERFLOW_RISK) before any bound is formed (334–341).
* V4: lines 385–396, status DECLARED with reason code DECLARED_ESTIMATE (`scope.REASON_CODES`).

Rounding directions: every upper bound is rounded up (`up`, `add_up`, `nextafter(·, inf)`), the lower bound of V3
down (`sub_down`, the relative 8u step away from zero), the Temple denominator ρ_eff − θ_hi down (`sub_down`,
which makes the correction larger).

## 4 Computational verification

`docs/proofs/verify_E-05.py` (seed 20261030, `--cases 200`; exact rationals for θ and r², mpmath at 60 digits for
the eigenvalues). Run of 2026-10-30:

| part | what is compared | cases | violations |
|---|---|---:|---:|
| A | θ_lo ≤ θ_exact ≤ θ_hi (exact Fraction quotient of the stored data) and λ₁ ≤ the V1 upper bound; random Hermitian H (n = 2…6, real and complex, scales 10⁻³…10³), trial states random or ground state + perturbation 10⁻⁸…10⁻¹, vector scales 10⁻²…10² | 200 | 0 |
| B | r_impl² ≥ r_exact² = ⟨ψ, H²ψ⟩/⟨ψ, ψ⟩ − θ² (exact rational) and an eigenvalue inside the V2 interval | 200 | 0 |
| C | Temple with ρ = fl↓(λ₂) (hypothesis true): lower ≤ λ₁, or INCONCLUSIVE exactly when θ_hi ≥ ρ; adversarial: two-level mixtures (equality case: λ₁ − lower < 10⁻⁶(1 + |λ₁|)), trial states orthogonal to the ground state (INCONCLUSIVE with the θ < ρ message), **negative control** ρ = λ₃ > λ₂ declared on the two-level mixtures: the record stays CERTIFIED_UNDER_HYPOTHESIS, names `declared_gap`, and its bound exceeds λ₁ in 66/66 cases | 200 random + 198 adversarial | 0 |
| D | MPS contraction: ‖ψ̂ − ψ_exact‖² (exact rational contraction of the binary64 tensors, L = 2…5, D = 1…4, real and complex, scales 10⁻²…10²) ≤ e²; Lemma 5: \|θ(ψ) − θ(ψ̂)\| ≤ the recorded bound | 66 + 66 | 0 |
| E | tiny norms: entries ≈ 2⁻⁴⁵⁰ keep A–C valid; entries ≈ 2⁻⁶⁰⁰ (squares underflow) give NOT_CHECKED, never CERTIFIED; entries ≈ 2⁻⁹⁵⁰ are INCONCLUSIVE with UNDERFLOW_RISK | 30 adversarial | 0 |

Total: 666 random + 294 adversarial cases, 0 violations; negative control broken 66/66. Tests
`tests/test_variational.py` cover the 8-site Heisenberg ring (V1 above λ₁, V3 under a declared gap and from a
certificate, the negative controls: false gap hypothesis, orthogonal state, mismatched certificate, tiny norms,
non-Hermitian operator, duplicate CI indices, MPS size limit), the adapters, the declared estimates, the record
schema and the CLI.

## 5 Analytic cross-check

1. *Exact eigenvector.* ψ = v₁ exactly: θ = λ₁, r = 0, V1 gives λ₁ ≤ λ₁ + rounding, V2 the interval [λ₁ − rounding,
   λ₁ + rounding], and Temple λ₁ ≥ λ₁ − 0 for any ρ ∈ (λ₁, λ₂]: the three statements collapse to the S2 enclosure of
   an exact eigenpair, as they must (the 8-site singlet in the tests: width ≈ 10⁻¹³).
2. *Two-level mixture.* ψ = cos a · v₁ + sin a · v₂, ρ = λ₂: θ = c²λ₁ + s²λ₂, r² = c²s²(λ₂ − λ₁)², ρ − θ = c²(λ₂ − λ₁), so
   θ − r²/(ρ − θ) = c²λ₁ + s²λ₂ − s²(λ₂ − λ₁) = λ₁: Temple is attained (part C, 66 cases to 10⁻⁶ of the floating-point
   slack) and cannot be improved without further hypotheses.
3. *Kato–Temple of the code (Theorem 3).* For an isolating interval (α, β) around λ_j, `enclosure.kato_temple` gives
   λ_j ≥ θ − r²/(β − θ): with j = 1 and β = λ₂ (α = −∞) this is Proposition 1 — the S2 sharpening and the variational
   lower bound are the same inequality, the former with β certified by S3, the latter with ρ declared or read from a
   certificate.
4. *First-order perturbation.* For ψ = v₁ + εd (d ⊥ v₁, ‖d‖ = 1): θ − λ₁ = ε²(⟨d, Hd⟩ − λ₁)/(1 + ε²) = O(ε²) while
   r = O(ε), so the Rayleigh excess is second order and the Temple correction r²/(ρ − θ) is second order too: the
   upper and lower bounds close at the same rate, which is what part A/C show (perturbations 10⁻⁸ … 10⁻¹).

## 6 Status

Derived by Claude on 2026-10-30; machine-checked (666 random + 294 adversarial cases, 0 violations; negative
control broken 66/66); independent re-derivation: done 2026-10-30 (fresh-context session, verdict GAP, docs/proofs/VERDICTS.md); Sarang's check (D-12): pending.
