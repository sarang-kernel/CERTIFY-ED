# Independent re-derivation: E-05 (variational certificates: Rayleigh upper bound, existence radius, Temple)

Written from sections 1 and 3 of `docs/proofs/E-05_variational.md` only (section 2 read afterwards; comparison at
the end). Notation as there. Classical facts: Rayleigh–Ritz; the residual theorem (Parlett Thm 4.5.1); Temple's
inequality via the operator inequality (H − λ₁)(H − ρ) ⪰ 0; Weyl; Higham (2002) Lemma 3.1/3.3 and the complex
inner-product constant; the error-free `fp.exact_dot` (checked in my thm1 note).

## Proofs

**Lemma 1.** Expand ψ = Σ c_j v_j in an orthonormal eigenbasis: θ = Σ λ_j|c_j|²/Σ|c_j|² is a convex combination of
the eigenvalues, hence in [λ₁, λ_n]. ∎

**Lemma 2.** Residual theorem with μ = θ: if no eigenvalue were within r of θ, ‖(H − θ)ψ‖ ≥ min_j|λ_j − θ|‖ψ‖ > r‖ψ‖,
a contradiction. Identity: ‖(H − θ)ψ‖² = ⟨Hψ, Hψ⟩ − 2θ⟨ψ, Hψ⟩ + θ²⟨ψ, ψ⟩ = ⟨ψ, H²ψ⟩ − θ²⟨ψ, ψ⟩ (θ real, θ⟨ψ,ψ⟩ = ⟨ψ,Hψ⟩). ∎

**Proposition 1 (Temple).** For ρ ≤ λ₂ every eigenvalue satisfies (λ_j − λ₁)(λ_j − ρ) ≥ 0 (j = 1: the first
factor is 0; j ≥ 2: both factors ≥ 0), so (H − λ₁)(H − ρ) ⪰ 0 and ⟨ψ, (H − λ₁)(H − ρ)ψ⟩ ≥ 0. Dividing by ‖ψ‖² and
using Lemma 2's identity ⟨ψ, H²ψ⟩/‖ψ‖² = r² + θ²: r² + θ² − (λ₁ + ρ)θ + λ₁ρ ≥ 0, i.e. λ₁(ρ − θ) ≥ ρθ − θ² − r² =
θ(ρ − θ) − r²; with ρ − θ > 0, λ₁ ≥ θ − r²/(ρ − θ). Equality: for ψ ∈ span{v₁, v₂} and ρ = λ₂, (H − λ₁)(H − λ₂)ψ = 0.
If λ₁ = λ₂, θ ≥ λ₁ = λ₂ ≥ ρ contradicts θ < ρ: no admissible ρ. ∎

**Lemma 3.** Let N := ⟨ψ, Hψ⟩ (real), D := ⟨ψ, ψ⟩ > 0, ŷ = fl(Hψ). Entry i of ŷ is an inner product with ≤ w nonzero
terms, so |ŷ − Hψ| ≤ γ̇_w |H||ψ| entrywise (Higham eq. (3.5); complex via Lemma 3.5 with the project's γ̇). Hence
|⟨ψ, ŷ⟩ − N| ≤ |ψ|ᵀ|ŷ − Hψ| ≤ γ̇_w a, a := |ψ|ᵀ|H||ψ|, and |Re N̂ − N| ≤ |N̂ − ⟨ψ, ŷ⟩| + |⟨ψ, ŷ⟩ − N| ≤ e_N + γ̇_w a. The
code bounds a from the computed â = fl(|H||ψ|) and fl(|ψ|ᵀâ): for real data â ≥ (|H||ψ|)(1 − γ_w) entrywise (non-negative
sums), fl(|ψ|ᵀâ) ≥ |ψ|ᵀâ (1 − γ_n), so a ≤ fl(|ψ|ᵀâ)/((1 − γ_w)(1 − γ_n)) ≤ fl(|ψ|ᵀâ)(1 + γ_{n+2})/(1 − γ_{w+1}); for
complex data the moduli |H_ij|, |ψ_j| carry one rounding each (hypot), which costs two more u per product —
(1 − γ_{w+1}) does not cover γ_{w+2} by itself (the thm1 (S2) remark), but the code uses γ̇_{w+2} = 2γ_{w+6} in place
of the γ̇_w = 2γ_{w+4} that the error needs, a relative slack ≥ 2/(w + 4) ≫ u, so E_N ≥ e_N + γ̇_w a holds with room.
E_D = e_D. If D̂ − E_D > 0, N ∈ [N̂ − E_N, N̂ + E_N] and D ∈ [D̂ − E_D, D̂ + E_D] ⊂ (0, ∞), and N/D is monotone in N and in
D on that box, so θ lies between the smallest and the largest of the four corner quotients. Each corner is computed
with ≤ 3 roundings (one sum/difference, one division; the sums are exact or carry a relative u of the result), so a
relative 4u step outward encloses the exact corners. Radius: ‖(H − θ)ψ‖ ≤ ‖(H − θ̂)ψ‖ + |θ − θ̂|‖ψ‖, so r ≤ r̂ +
|θ − θ̂| ≤ r̂ + max(θ_hi − θ̂, θ̂ − θ_lo) (also when θ̂ falls outside [θ_lo, θ_hi]: then one of the two terms exceeds
|θ − θ̂|); `up` covers the addition. ∎
*Model shift (H_spec = H_ref + Δ, ‖Δ‖₂ ≤ σ).* θ_spec := ⟨ψ, H_spec ψ⟩/D ∈ [θ − σ, θ + σ] (|⟨ψ, Δψ⟩| ≤ σD), so
λ₁(H_spec) ≤ θ_spec ≤ θ_hi + σ (V1). ‖(H_spec − θ)ψ‖ ≤ ‖(H_ref − θ)ψ‖ + σ‖ψ‖: an eigenvalue of H_spec within r + σ of θ
(V2). Temple for H_spec directly: r_spec := ‖(H_spec − θ_spec)ψ‖/‖ψ‖ ≤ ‖(H_spec − θ)ψ‖/‖ψ‖ ≤ r + σ (the Rayleigh
quotient minimises the residual over shifts), θ_spec ≤ θ_hi + σ < ρ ≤ λ₂(H_spec), hence
λ₁(H_spec) ≥ θ_spec − r_spec²/(ρ − θ_spec) ≥ (θ_lo − σ) − (r + σ)²/(ρ − σ − θ_hi) — exactly what the code forms (its
r is r_stored + σ). This route needs only H_spec Hermitian. The document's section 1 formula with "any r ≥
r_stored" (no σ in r) instead needs the route "Temple for H_ref with ρ − σ ≤ λ₂(H_ref), then λ₁(H_spec) ≥
λ₁(H_ref) − σ", which presupposes that the floating-point build H_ref is Hermitian: it is not in general (a boson
chain with complex hopping gives max|H_ref − H_refᴴ| = 1.1·10⁻¹⁶ in my check; spin chains with real couplings give
0). With a non-Hermitian H_ref, "λ₂(H_ref)" and Temple are not available, and the σ-free formula is not justified
(the deficit is of order σ·r/(ρ − θ) — rounding-level, but the statement as written is not proven). See the
comparison.

**Lemma 4.** Induction on j with E_j := |T̂_j − T_j| (entrywise). T̂_j = fl(T̂_{j−1}A⁽ʲ⁾): every entry is an inner product
of length D_{j−1}, so |T̂_j − T̂_{j−1}A⁽ʲ⁾| ≤ g_j |T̂_{j−1}||A⁽ʲ⁾|, and T̂_{j−1}A⁽ʲ⁾ − T_j = (T̂_{j−1} − T_{j−1})A⁽ʲ⁾. With
|T̂_{j−1}| ≤ |T_{j−1}| + E_{j−1} ≤ (1 + c_{j−1})Π_{<j} (Π_{<j} := |A⁽¹⁾|⋯|A⁽ʲ⁻¹⁾|, and |T_{j−1}| ≤ Π_{<j}):
E_j ≤ (g_j(1 + c_{j−1}) + c_{j−1})Π_{≤j} = ((1 + g_j)(1 + c_{j−1}) − 1)Π_{≤j}, so c_L = Π(1 + g_j) − 1. Real data: Higham
Lemma 3.3 gives Π(1 + γ_{D_{j−1}}) ≤ 1 + γ_{ΣD_{j−1}}. Complex data: 2γ_m ≤ γ_{2m} (2mu/(1 − mu) ≤ 2mu/(1 − 2mu)), so
1 + g_j ≤ 1 + γ_{2(D_{j−1}+4)} and Lemma 3.3 again gives c_L ≤ γ_M with M = Σ 2(D_{j−1} + 4) — the composition is
done through γ of a doubled index, which is legitimate (unlike adding γ̇'s, cf. my C-04 note). The computed P̂ =
fl(|A⁽¹⁾|⋯|A⁽ᴸ⁾|) is a product of non-negative inner products: P̂ ≥ Π(1 − γ_{ΣD}) ≥ Π(1 − γ_M) (complex: the moduli
add one rounding per entry, absorbed by M ≥ ΣD + 8(L − 1)), so ‖ψ̂ − ψ‖₂ ≤ ‖c_L Π‖₂ ≤ ‖γ_M P̂/(1 − γ_M)‖₂ ≤ e. The
first tensor is copied exactly (T̂₁ = A⁽¹⁾[0]); BLAS summation order and FMA do not weaken the γ bounds. ∎

**Lemma 5.** Write ψ = ψ̂ + δ, ‖δ‖ ≤ e, θ̂ := θ(ψ̂), r̂ := ‖(H − θ̂)ψ̂‖/‖ψ̂‖. Since ⟨ψ̂, (H − θ̂)ψ̂⟩ = 0,
⟨ψ, (H − θ̂)ψ⟩ = 2 Re⟨δ, (H − θ̂)ψ̂⟩ + ⟨δ, (H − θ̂)δ⟩, and θ(ψ) − θ̂ = ⟨ψ, (H − θ̂)ψ⟩/‖ψ‖². Hence
|θ(ψ) − θ̂| ≤ (2 e r̂‖ψ̂‖ + (‖H‖ + |θ̂|)e²)/(‖ψ̂‖ − e)². ∎

**Statements of the record.** V1: Lemma 1 + model shift. V2: Lemma 2 + Lemma 3 + shift. V3: Proposition 1 with ρ
from a certificate: for k ≥ 2 with S2/S3 CERTIFIED (Corollary 5) λ₁ and λ₂ lie in two distinct enclosures, so
λ₂ ≥ max of their two lower ends ≥ the second smallest lower end overall (at most one lower end is below both);
for k = 1, λ₂ ≥ S3's `next_eigenvalue_lower_bound`. Declared ρ: the hypothesis is named and the status says so. ∎

## Code (current files)

* `enclosure.rayleigh_interval` 669–704: ŷ = adapter.matvec (680), `exact_dot` for N̂, D̂ (681–684; real parts), γ̇_{w+2}
  (685), absAx = fl(|A||x|)/(1 − γ_{w+1}) (687–688), num_err = up(e_N + γ̇_{w+2}·fl(|x|ᵀabsAx)·(1 + γ_{n+2})) (691),
  den_lo ≤ 0 → None (693–695), the four corners (696–697), ±4u outward (699–700), θ̂ = N̂/D̂ and r̂ from `residuals`
  (701–704). `variational.certify_variational` 238–403: Hermiticity of a stored matrix enforced exactly (280–287);
  spec route with σ = `_builder_shift` (266–268); underflow scan (325–333) → INCONCLUSIVE UNDERFLOW_RISK (336–341);
  r_stored = up(r̂ + max(…)) (349), r = add_up(r_stored, σ) (350), V1 upper = nextafter(add_up(θ_hi, σ)) (351–352),
  V2 interval sub_down/add_up with r (377). Lemma 5 at 356–370 (xn ≥ ‖ψ̂‖, xl ≤ ‖ψ̂‖, Hn ≥ ‖H‖, r_stored ≥ r̂; one
  `up` for ≤ 7 roundings).
* `_temple` 406–460: ρ_eff = sub_down(ρ, σ) (436), hypothesis θ_hi < ρ_eff (439; INCONCLUSIVE otherwise), r² up +
  ULP_MIN (445), corr = up(r²/sub_down(ρ_eff, θ_hi)) (446), lower = sub_down(θ_lo, corr) (447), − σ (449), relative
  8u away from zero (450); CERTIFIED only with `gap_from_certificate` (169–216: spec hash and dimension checks, 'lowest'
  kind, S2 and S3 CERTIFIED, second smallest lower end `sort(sub_down(mu, rad))[1]` for k ≥ 2 (202–204), S3's bound
  for k = 1 (207–212)); otherwise CERTIFIED_UNDER_HYPOTHESIS with `declared_gap` named (428–429).
* `state_from_mps` 102–150: left-to-right `tensordot` (139), absolute product (140), M (141: 2(D + 4) complex, D
  real), γ_M (146), entrywise bound up(γ_M)·P̂/(1 − γ_M) (147), e = norm2_upper (148); SIZE_LIMIT (131–133).
* Section 3's line numbers (branch `track/variational`) agree with the current file up to small shifts (`_temple`
  445–450 for 444–449; `rayleigh_interval` 669–704 for 660–695).

### Comparison with section 2 (read after the above)

Lemmas 1, 2, 4, 5, Proposition 1 and Lemma 3 (i)–(iv) agree with my derivations step by step (same operator
inequality for Temple, same induction and the γ_{2(D+4)} route for the complex MPS constant, same expansion for
Lemma 5, same Corollary 5 argument for ρ from a certificate).

**GAP (Lemma 3 (v), model shift).** Section 2 derives V1–V3 for the spec route through the eigenvalues of the
floating-point build H_ref: "by Weyl |λ_j(H_spec) − λ_j(H_ref)| ≤ σ", Lemma 1 and Proposition 1 applied to H_ref.
All of this presupposes that H_ref is Hermitian, which the raw spec build is not in general (a 4-site boson chain
with complex hopping: max|H_ref − H_refᴴ| = 1.1·10⁻¹⁶; the real spin chains give exactly 0): Weyl, Rayleigh–Ritz
and Temple are not available for a non-Hermitian H_ref, and "λ₂(H_ref)" is undefined. The statements are
nevertheless true by the direct route (my proof): θ_spec ∈ [θ − σ, θ + σ], r_spec ≤ r + σ, Temple for the Hermitian
H_spec — and that route needs r ≥ r_stored + σ in the Temple formula, which is what the code uses (`r =
add_up(r_stored, sigma)`). Section 1's "for any r ≥ r_stored" (without σ) is therefore not established; the
implemented bound is. Correction: proof text (replace (v) by the direct route) and the statement's r ≥ r_stored + σ,
or symmetrise the spec build (then the H_ref route is valid). No certified bound affected.

**Remarks.** (1) Lemma 3 (i) for complex data: the moduli |H_ij|, |ψ_j| are rounded (hypot), so â can fall short of
|H||ψ| by γ_{w+2}, not the γ_{w+1} used; the γ̇_{w+2} in place of γ̇_w (relative slack ≥ 2/(w + 4)) covers it — the
thm1 (S2) remark again, harmless here. (2) `gap_from_certificate` with a stored matrix and no spec checks the
dimension only; the caveat is in the record's `refers_to`. (3) Section 3's line numbers are shifted by ≤ 9 lines.

**Verdict: GAP** (model-shift proof presupposes a Hermitian floating-point build; statements true by the direct
route with r_stored + σ, which the code uses; no certified bound affected).

### Runs

* `python docs/proofs/verify_E-05.py --cases 60 --seed 5`: A 60, B 60, C 60 + 60 (negative control 20/20), D 20 + 20,
  E 30; 200 random + 110 adversarial cases, 0 violations (16 s).
* `python docs/proofs/independent/adversarial_E-05_variational.py --cases 80 --seed 91`: Lemma 3 θ-enclosure 80 and
  r_stored 80 (exact rationals; n ≤ 40, near-eigenvectors with r ~ 1e-12, θ ≈ 0 sign patterns, entries 10^±6,
  scales 10^±3), V1–V3 on stored matrices 80 (Temple equality within 1e-9) + 18 negative controls (ρ = λ₃: never
  CERTIFIED, hypothesis named), spec route with σ > 0 on spin-1/boson chains with complex hopping and Heisenberg
  chains 30 (truth: eigenvalues of `checker.exact_matrix` at 60 digits), Lemma 4 26 (exact rational contraction,
  L ≤ 5, D ≤ 5, ±1 tensors) and Lemma 5 26, `gap_from_certificate` 3 (2 skipped: S3 INCONCLUSIVE when k splits a
  multiplet); 0 violations (~8 min, dominated by mp eigensolves).
