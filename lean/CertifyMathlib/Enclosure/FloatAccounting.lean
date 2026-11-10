import CertifyMathlib.Enclosure.FloatModel
import CertifyMathlib.Enclosure.Residual
import CertifyMathlib.Enclosure.Weyl

/-!
# Theorem 1 in floating point: the computed radius bounds the exact residual

This file follows the computation of `residuals` in `certify_ed/enclosure.py` for real data, one
rounding at a time, in the standard model of `FloatModel.lean` (`eta = 0`, `u ≤ 2⁻⁵³`), and proves
the statement of `docs/proofs/thm1_fp-accounting.md`: the computed relative radius `rel` is an upper
bound of `‖A x - μ x‖ / ‖x‖` (`residual_radius_fl`), hence of the distance from `μ` to the spectrum
(`residual_enclosure_fl`).

The modelled computation, for a vector `x`, a claimed value `μ` and rows with at most `w` stored
entries (`n` is the dimension):

* `ŷ_i`: the products `A_ij x_j` rounded and summed along a summation tree `T i` (`matvecFl`);
* `R̂_i = fl(ŷ_i - fl(x_i μ))` (`residFl`);
* `Ê_i = fl(fl(fl(fl(fl(â_i / fl(1 - γ̂_{w+1})) (1 + 4u)) + fl(|x_i| |μ|)) γ̂_{w+2}) (1 + 4u))`,
  where `â_i` is `|A| |x|` computed along a second tree and `γ̂_k = fl(k u / fl(1 - k u))` is
  `fp.gamma` (`roundBoundFl`);
* `fp.norm2_upper` and `fp.norm2_lower`: `fl(‖v‖) = fl(√(∑ fl(v_j²)))` along a tree over all
  entries, then `(1 ± γ̂_{n+2})`, the guard `1 + 8u` and an additive tail `≥ 0` (`norm2Up`,
  `norm2Low`);
* `rel = fl(fl(fl(fl(fl(R_up + E_up) + c₁) / x_low) (1 + 4u)) + c₀)` with `c₀, c₁ ≥ 0` (the INT-05
  additive terms of the code, which the standard model does not need).

The summation trees are arbitrary (any order and association of the sums), products are rounded
separately (no fused multiply-add), and `k u` is exact (an integer times a power of two). The
numeric hypotheses are `w ≤ 2²⁶` and `n ≤ 2²⁶`.

For complex data (`residual_radius_fl_c`, `residual_enclosure_fl_c`) complex sums and products
with a real number are rounded componentwise (`crnd`), a complex product is formed from four real
products and two real sums (`cmul`, error `γ₃`, Higham's Lemma 3.5), a modulus is rounded once, the
computed `|A| |x|` is divided by `1 - γ̂_{w+3}`, the entrywise bound uses `2 γ̂_{w+6}` (`gamma_dot`),
and the norms sum the `2n` squares of the real components with the factor `1 ± γ̂_{n+2}` of
`fp.norm2_upper` (`w + 4 ≤ 2²⁶`, `n ≤ 2²⁶`).

Appendix A (`docs/proofs/appA_model-shift.md`): the amplitude of a monomial computed as a chain of
rounded products (`monoFl`, `mono_error`: `γ_{2L}`), an entry of `H_ref` as a rounded sum of at most
`m` monomials in any order (Lemma A1, `builder_entry_error`, `builder_error`: `γ_{m+2L-1}`), the
symmetrised build `fl(H_ij + conj H_ji) / 2` (Lemma A2, `symmetrised_error`), the bound
`Δ_builder = γ_W √(r c)` of `‖H_ref - H_spec‖₂` and of `‖fl(Ĥ) - H_spec‖₂` (Lemma A3 with the Schur
test, `builder_shift`, `symmetrised_shift`, `schur_split`), and Theorem A4 with this `Δ_builder`
(`model_shift_fl`) and its Weyl transfer (`weyl_transfer_fl`). The second sentence of Lemma A1:
the computed row and column sums of `|H|_abs` (`abs_matvec(ones)`, `abs_rmatvec(ones)`, sums of the
computed moduli of at most `m` monomials in any order, the products with the entries `1.0` of
`ones` exact) are within the factor `1 ± γ_{m+2L-1}` of the exact sums for real amplitudes
(`abs_sum_bounds_real`) and `1 ± γ_{m+2L}` for complex ones, where `np.abs` rounds the modulus once
(`abs_sum_bounds_complex`); `abs_sum_error` is the general form.
-/

open Finset Matrix

namespace CertifyMathlib.Enclosure.FP

/-! ### Rounding envelopes in the standard model -/

namespace Rounding

theorem abs_rnd_sub_le (R : Rounding ℝ) (h0 : R.eta = 0) (z : ℝ) : |R.rnd z - z| ≤ R.u * |z| := by
  have h := R.err z
  rwa [h0, add_zero, Real.norm_eq_abs, Real.norm_eq_abs] at h

/-- An upper bound `z ≥ 0` of the exact result gives the upper bound `z (1 + u)`. -/
theorem rnd_le_of_le (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {w z : ℝ} (hw : w ≤ z)
    (hz : 0 ≤ z) : R.rnd w ≤ z * (1 + R.u) := by
  have hu := R.u_nonneg
  have h := R.rnd_le_abs h0 w
  rcases le_total 0 w with hw0 | hw0
  · rw [abs_of_nonneg hw0] at h
    nlinarith
  · rw [abs_of_nonpos hw0] at h
    nlinarith

/-- A lower bound `0 ≤ z` of the exact result gives the lower bound `z (1 - u)`. -/
theorem le_rnd_of_le (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {w z : ℝ} (hw : z ≤ w)
    (hz : 0 ≤ z) : z * (1 - R.u) ≤ R.rnd w := by
  have h := R.rnd_ge_of_nonneg h0 (hz.trans hw)
  nlinarith

theorem rnd_nonneg (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {w : ℝ} (hw : 0 ≤ w) :
    0 ≤ R.rnd w := by
  have := R.le_rnd_of_le h0 hu1 hw le_rfl
  simpa using this

theorem le_rnd_mul (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {a b a' b' : ℝ}
    (ha' : 0 ≤ a') (hb' : 0 ≤ b') (ha : a' ≤ a) (hb : b' ≤ b) :
    a' * b' * (1 - R.u) ≤ R.rnd (a * b) :=
  R.le_rnd_of_le h0 hu1 (mul_le_mul ha hb hb' (ha'.trans ha)) (mul_nonneg ha' hb')

theorem rnd_le_mul (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {a b a' b' : ℝ}
    (ha0 : 0 ≤ a) (ha : a ≤ a') (hb : b ≤ b') (hb'0 : 0 ≤ b') (ha'0 : 0 ≤ a') :
    R.rnd (a * b) ≤ a' * b' * (1 + R.u) :=
  R.rnd_le_of_le h0 hu1 ((mul_le_mul_of_nonneg_left hb ha0).trans
    (mul_le_mul_of_nonneg_right ha hb'0)) (mul_nonneg ha'0 hb'0)

end Rounding

/-! ### Numeric facts -/

theorem one_sub_mul_le_pow {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1) :
    ∀ n : ℕ, 1 - (n : ℝ) * u ≤ (1 - u) ^ n
  | 0 => by simp
  | n + 1 => by
    have ih := one_sub_mul_le_pow hu hu1 n
    have h1 : 0 ≤ 1 - u := by linarith
    rw [pow_succ]
    push_cast
    nlinarith [mul_le_mul_of_nonneg_right ih h1, mul_nonneg (Nat.cast_nonneg (α := ℝ) n)
      (mul_nonneg hu hu)]

theorem one_add_pow_five_le {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) :
    (1 + u) ^ 5 ≤ 1 + 6 * u := by
  have h2 : u ^ 2 ≤ u / 1000 := by nlinarith
  have h3 : u ^ 3 ≤ u / 1000 := by nlinarith [mul_le_mul_of_nonneg_left hu1 (pow_nonneg hu 2)]
  have h4 : u ^ 4 ≤ u / 1000 := by nlinarith [mul_le_mul_of_nonneg_left hu1 (pow_nonneg hu 3)]
  have h5 : u ^ 5 ≤ u / 1000 := by nlinarith [mul_le_mul_of_nonneg_left hu1 (pow_nonneg hu 4)]
  have e : (1 + u) ^ 5 = 1 + 5 * u + 10 * u ^ 2 + 10 * u ^ 3 + 5 * u ^ 4 + u ^ 5 := by ring
  rw [e]
  linarith

/-- For `k ≤ 2²⁶ + 2` and `u ≤ 2⁻⁵³`: `k u ≤ 10⁻⁷`, `γ_k ≤ 10⁻³` and `γ_k² ≤ u`. -/
theorem gam_regime {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {k : ℕ}
    (hk : k ≤ 2 ^ 26 + 2) :
    (k : ℝ) * u ≤ 1 / 10 ^ 7 ∧ gam u k ≤ 1 / 1000 ∧ gam u k ^ 2 ≤ u := by
  have hkr : (k : ℝ) ≤ 2 ^ 26 + 2 := by exact_mod_cast hk
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have h1 : (k : ℝ) * u ≤ (2 ^ 26 + 2) * (2 : ℝ)⁻¹ ^ 53 := mul_le_mul hkr hu53 hu (by norm_num)
  have hc : ((2 : ℝ) ^ 26 + 2) * (2 : ℝ)⁻¹ ^ 53 ≤ 1 / 10 ^ 7 := by norm_num
  have hku : (k : ℝ) * u ≤ 1 / 10 ^ 7 := h1.trans hc
  have hku0 : 0 ≤ (k : ℝ) * u := mul_nonneg hk0 hu
  have hd : 0 < 1 - (k : ℝ) * u := by linarith
  refine ⟨hku, ?_, ?_⟩
  · unfold gam
    rw [div_le_iff₀ hd]
    linarith
  · unfold gam
    rw [div_pow, div_le_iff₀ (by positivity)]
    have h2 : (k : ℝ) * ((k : ℝ) * u) ≤ (2 ^ 26 + 2) * ((2 ^ 26 + 2) * (2 : ℝ)⁻¹ ^ 53) :=
      mul_le_mul hkr h1 hku0 (by norm_num)
    have h3 : ((2 : ℝ) ^ 26 + 2) * ((2 ^ 26 + 2) * (2 : ℝ)⁻¹ ^ 53) ≤ 3 / 4 := by norm_num
    have h4 : (3 : ℝ) / 4 ≤ (1 - (k : ℝ) * u) ^ 2 := by nlinarith
    calc ((k : ℝ) * u) ^ 2 = ((k : ℝ) * ((k : ℝ) * u)) * u := by ring
      _ ≤ 3 / 4 * u := mul_le_mul_of_nonneg_right (h2.trans h3) hu
      _ ≤ u * (1 - (k : ℝ) * u) ^ 2 := by nlinarith

/-- The constants used for a dimension or row width `k ≤ 2²⁶`. -/
theorem small_consts {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {k : ℕ}
    (hk : k ≤ 2 ^ 26) :
    u ≤ 1 / 1000 ∧ ((k + 2 : ℕ) : ℝ) * u ≤ 1 / 10 ^ 7 ∧ 0 ≤ gam u k ∧
      gam u k + u ≤ gam u (k + 1) ∧ gam u (k + 1) + u ≤ gam u (k + 2) ∧
      gam u (k + 2) ≤ 1 / 1000 ∧ gam u (k + 2) ^ 2 ≤ u := by
  obtain ⟨h1, h2, h3⟩ := gam_regime hu hu53 (k := k + 2) (by omega)
  have hk2 : ((k + 2 : ℕ) : ℝ) * u < 1 := by linarith
  have hk1 : ((k + 1 : ℕ) : ℝ) * u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k + 1 ≤ k + 2)) hu) hk2
  have hk0 : ((k : ℕ) : ℝ) * u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k ≤ k + 2)) hu) hk2
  have s1 : gam u k + u * (1 + gam u k) ≤ gam u (k + 1) := gam_succ hu hk1
  have s2 : gam u (k + 1) + u * (1 + gam u (k + 1)) ≤ gam u (k + 2) := gam_succ hu hk2
  have g0 := gam_nonneg hu hk0
  have g1 := gam_nonneg hu hk1
  have hu1 : u ≤ 1 / 1000 := hu53.trans (by norm_num)
  exact ⟨hu1, h1, g0, by nlinarith, by nlinarith, h2, h3⟩

/-- The factor of `fp.norm2_upper`: `(1 - γ_n)(1 + γ̂_{n+2})(1 - u)⁵ (1 + 8u) ≥ 1 + u`. -/
theorem num_norm_up {u g G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (hg : 0 ≤ g)
    (hgG : g + 2 * u ≤ G) (hG : G ≤ 1 / 1000) (hG2 : G ^ 2 ≤ u) :
    1 + u ≤ (1 - g) * (1 + G * (1 - u) ^ 2) * (1 - u) ^ 5 * (1 + 8 * u) := by
  have hG0 : 0 ≤ G := by linarith
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith
  have hP : 1 + 2 * u - 2 * u * G - G ^ 2 ≤ (1 - g) * (1 + G * (1 - u) ^ 2) := by
    have h1 : (1 - g) * (1 + G * (1 - 2 * u)) ≤ (1 - g) * (1 + G * (1 - u) ^ 2) :=
      mul_le_mul_of_nonneg_left (by linarith [mul_le_mul_of_nonneg_left hsq hG0]) (by linarith)
    nlinarith [mul_nonneg hG0 (by linarith : (0 : ℝ) ≤ G - g), mul_nonneg (mul_nonneg hu hg) hG0]
  have hP' : 1 + u / 2 ≤ 1 + 2 * u - 2 * u * G - G ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hG hu]
  have hQ : 1 + 2 * u ≤ (1 - u) ^ 5 * (1 + 8 * u) := by
    have h5 := one_sub_mul_le_pow hu (by linarith) 5
    push_cast at h5
    have : (1 - 5 * u) * (1 + 8 * u) ≤ (1 - u) ^ 5 * (1 + 8 * u) :=
      mul_le_mul_of_nonneg_right h5 (by linarith)
    nlinarith [mul_le_mul_of_nonneg_left hu1 hu]
  have hPQ := mul_le_mul (hP'.trans hP) hQ (by linarith) (by linarith)
  calc 1 + u ≤ (1 + u / 2) * (1 + 2 * u) := by nlinarith
    _ ≤ (1 - g) * (1 + G * (1 - u) ^ 2) * ((1 - u) ^ 5 * (1 + 8 * u)) := hPQ
    _ = (1 - g) * (1 + G * (1 - u) ^ 2) * (1 - u) ^ 5 * (1 + 8 * u) := by ring

/-- The factor of `fp.norm2_lower`: `(1 + γ_n)(1 - γ̂_{n+2})(1 + u)⁵ ≤ (1 - u)(1 + 8u)`. -/
theorem num_norm_low {u g G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (hg : 0 ≤ g)
    (hgG : g + 2 * u ≤ G) (hG : G ≤ 1 / 1000) :
    (1 + g) * (1 - G * (1 - u) ^ 2) * (1 + u) ^ 5 ≤ (1 - u) * (1 + 8 * u) := by
  have hG0 : 0 ≤ G := by linarith
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith
  have hsq1 : (1 - u) ^ 2 ≤ 1 := by nlinarith
  have hX : (1 + g) * (1 - G * (1 - u) ^ 2) ≤ 1 - u := by
    have h1 : (1 + g) * (1 - G * (1 - u) ^ 2) ≤ (1 + g) * (1 - G * (1 - 2 * u)) :=
      mul_le_mul_of_nonneg_left (by linarith [mul_le_mul_of_nonneg_left hsq hG0]) (by linarith)
    nlinarith [mul_nonneg hg (mul_nonneg hG0 (by linarith : (0 : ℝ) ≤ 1 - 2 * u)),
      mul_le_mul_of_nonneg_left hG hu]
  have hC : 0 ≤ 1 - G * (1 - u) ^ 2 := by nlinarith [mul_le_mul_of_nonneg_left hsq1 hG0]
  have hX0 : 0 ≤ (1 + g) * (1 - G * (1 - u) ^ 2) := mul_nonneg (by linarith) hC
  have h5 := one_add_pow_five_le hu hu1
  calc (1 + g) * (1 - G * (1 - u) ^ 2) * (1 + u) ^ 5 ≤ (1 - u) * (1 + 6 * u) :=
        mul_le_mul hX h5 (by positivity) (by linarith)
    _ ≤ (1 - u) * (1 + 8 * u) := mul_le_mul_of_nonneg_left (by linarith) (by linarith)

/-- The factor of `|A| |x| / (1 - γ̂_{w+1}) (1 + 4u)`. -/
theorem num_abs {u g G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (hg : 0 ≤ g) (hgG : g + u ≤ G)
    (hG : G ≤ 1 / 1000) :
    (1 - G * (1 - u) ^ 2) * (1 + u) ≤ (1 - g) * (1 - u) ^ 2 * (1 + 4 * u) := by
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith
  have hG0 : 0 ≤ G := by linarith
  have hL : (1 - g) * (1 - 2 * u) * (1 + 4 * u) ≤ (1 - g) * (1 - u) ^ 2 * (1 + 4 * u) := by
    apply mul_le_mul_of_nonneg_right _ (by linarith)
    exact mul_le_mul_of_nonneg_left hsq (by linarith)
  have hR : (1 - G * (1 - u) ^ 2) * (1 + u) ≤ (1 - G * (1 - 2 * u)) * (1 + u) :=
    mul_le_mul_of_nonneg_right (by linarith [mul_le_mul_of_nonneg_left hsq hG0]) (by linarith)
  have key : (1 - G * (1 - 2 * u)) * (1 + u) ≤ (1 - g) * (1 - 2 * u) * (1 + 4 * u) := by
    nlinarith [mul_le_mul_of_nonneg_left hu1 hu, mul_le_mul_of_nonneg_left hG hu,
      mul_le_mul_of_nonneg_left (by linarith : g ≤ 1 / 1000) hu,
      mul_nonneg (mul_nonneg hu hu) hg, mul_le_mul_of_nonneg_left hG (mul_nonneg hu hu)]
  linarith

/-- The factor of `γ̂_{w+2} (…)(1 + 4u)`: `γ_k ≤ γ_{k+1} (1 - u)⁶ (1 + 4u)`. -/
theorem num_E {u g G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (hg : 0 ≤ g) (hgG : g + u ≤ G)
    (hG : G ≤ 1 / 1000) : g ≤ G * ((1 - u) ^ 6 * (1 + 4 * u)) := by
  have h6 := one_sub_mul_le_pow hu (by linarith) 6
  push_cast at h6
  have hG0 : 0 ≤ G := by linarith
  have hQ : 1 - 3 * u ≤ (1 - u) ^ 6 * (1 + 4 * u) := by
    have := mul_le_mul_of_nonneg_right h6 (by linarith : (0 : ℝ) ≤ 1 + 4 * u)
    nlinarith [mul_le_mul_of_nonneg_left hu1 hu]
  have := mul_le_mul_of_nonneg_left hQ hG0
  nlinarith [mul_le_mul_of_nonneg_left hG hu]

/-- The factor of the final quotient: `(1 + u)(1 - u)⁴ (1 + 4u) ≥ 1`. -/
theorem num_final {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) :
    1 ≤ (1 + u) * (1 - u) ^ 4 * (1 + 4 * u) := by
  have h4 := one_sub_mul_le_pow hu (by linarith) 4
  push_cast at h4
  have : (1 + u) * (1 - 4 * u) * (1 + 4 * u) ≤ (1 + u) * (1 - u) ^ 4 * (1 + 4 * u) := by
    apply mul_le_mul_of_nonneg_right _ (by linarith)
    exact mul_le_mul_of_nonneg_left h4 (by linarith)
  nlinarith [mul_le_mul_of_nonneg_left hu1 hu, mul_le_mul_of_nonneg_left hu1 (mul_nonneg hu hu)]

/-! ### The computed `γ_k` -/

/-- `fp.gamma(k)`: `nu / (1 - nu)` with `nu = k u` exact. -/
noncomputable def gamHat (R : Rounding ℝ) (k : ℕ) : ℝ :=
  R.rnd ((k : ℝ) * R.u / R.rnd (1 - (k : ℝ) * R.u))

theorem gamHat_bounds (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 2) {k : ℕ}
    (hk : (k : ℝ) * R.u ≤ 1 / 2) :
    gam R.u k * (1 - R.u) ^ 2 ≤ gamHat R k ∧ gamHat R k ≤ 3 * gam R.u k := by
  have hu := R.u_nonneg
  have hu1' : R.u ≤ 1 := by linarith
  have hku : 0 ≤ (k : ℝ) * R.u := mul_nonneg (Nat.cast_nonneg k) hu
  have hd0 : (0 : ℝ) < 1 - (k : ℝ) * R.u := by linarith
  have hd1 : R.rnd (1 - (k : ℝ) * R.u) ≤ (1 - (k : ℝ) * R.u) * (1 + R.u) :=
    R.rnd_le_of_le h0 hu1' le_rfl hd0.le
  have hd2 : (1 - (k : ℝ) * R.u) * (1 - R.u) ≤ R.rnd (1 - (k : ℝ) * R.u) :=
    R.le_rnd_of_le h0 hu1' le_rfl hd0.le
  have hdp : 0 < (1 - (k : ℝ) * R.u) * (1 - R.u) := mul_pos hd0 (by linarith)
  have hdpos : 0 < R.rnd (1 - (k : ℝ) * R.u) := lt_of_lt_of_le hdp hd2
  have hg : gam R.u k * (1 - (k : ℝ) * R.u) = (k : ℝ) * R.u := by
    unfold gam
    exact div_mul_cancel₀ _ hd0.ne'
  have hg0 : 0 ≤ gam R.u k := gam_nonneg hu (by linarith)
  have hq1 : gam R.u k * (1 - R.u) ≤ (k : ℝ) * R.u / R.rnd (1 - (k : ℝ) * R.u) := by
    rw [le_div_iff₀ hdpos]
    calc gam R.u k * (1 - R.u) * R.rnd (1 - (k : ℝ) * R.u)
        ≤ gam R.u k * (1 - R.u) * ((1 - (k : ℝ) * R.u) * (1 + R.u)) :=
          mul_le_mul_of_nonneg_left hd1 (mul_nonneg hg0 (by linarith))
      _ = (gam R.u k * (1 - (k : ℝ) * R.u)) * (1 - R.u ^ 2) := by ring
      _ = (k : ℝ) * R.u * (1 - R.u ^ 2) := by rw [hg]
      _ ≤ (k : ℝ) * R.u := by nlinarith [mul_nonneg hku (sq_nonneg R.u)]
  have hq2 : (k : ℝ) * R.u / R.rnd (1 - (k : ℝ) * R.u) * (1 + R.u) ≤ 3 * gam R.u k := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hdpos]
    calc (k : ℝ) * R.u * (1 + R.u) ≤ 3 * ((k : ℝ) * R.u) * (1 - R.u) := by
          nlinarith [mul_nonneg hku (by linarith : (0 : ℝ) ≤ 2 - 4 * R.u)]
      _ = 3 * (gam R.u k * (1 - (k : ℝ) * R.u)) * (1 - R.u) := by rw [hg]
      _ = 3 * gam R.u k * ((1 - (k : ℝ) * R.u) * (1 - R.u)) := by ring
      _ ≤ 3 * gam R.u k * R.rnd (1 - (k : ℝ) * R.u) :=
          mul_le_mul_of_nonneg_left hd2 (mul_nonneg (by norm_num) hg0)
  have hq0 : 0 ≤ (k : ℝ) * R.u / R.rnd (1 - (k : ℝ) * R.u) := div_nonneg hku hdpos.le
  refine ⟨?_, ?_⟩
  · have := R.le_rnd_of_le h0 hu1' hq1 (mul_nonneg hg0 (by linarith))
    unfold gamHat
    calc gam R.u k * (1 - R.u) ^ 2 = gam R.u k * (1 - R.u) * (1 - R.u) := by ring
      _ ≤ _ := this
  · unfold gamHat
    exact (R.rnd_le_of_le h0 hu1' le_rfl hq0).trans hq2

/-! ### Rounding chains -/

namespace Rounding

variable (R : Rounding ℝ)

/-- The chain of `fp.norm2_upper` from a lower bound `s` of `fl(‖v‖)`. -/
theorem chain_up (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {N s c g G tail : ℝ} (hN : 0 ≤ N)
    (hs : N * (1 - g) * (1 - R.u) ≤ s) (hc : (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ≤ c)
    (hg1 : g ≤ 1) (hG0 : 0 ≤ G) (ht : 0 ≤ tail)
    (hnum : 1 + R.u ≤ (1 - g) * (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ^ 5 * (1 + 8 * R.u)) :
    N * (1 + R.u) ≤ R.rnd (R.rnd (R.rnd (s * c) * (1 + 8 * R.u)) + tail) := by
  have hu := R.u_nonneg
  have h1u : 0 ≤ 1 - R.u := by linarith
  have h8 : 0 ≤ 1 + 8 * R.u := by linarith
  have hL : 0 ≤ N * (1 - g) * (1 - R.u) := mul_nonneg (mul_nonneg hN (by linarith)) h1u
  have hC : 0 ≤ (1 + G * (1 - R.u) ^ 2) * (1 - R.u) :=
    mul_nonneg (by linarith [mul_nonneg hG0 (sq_nonneg (1 - R.u))]) h1u
  have hp := R.le_rnd_mul h0 hu1 hL hC hs hc
  have hp2 := R.le_rnd_mul h0 hu1 (mul_nonneg (mul_nonneg hL hC) h1u) h8 hp le_rfl
  have hp3 := R.le_rnd_of_le h0 hu1 (hp2.trans (le_add_of_nonneg_right ht))
    (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg hL hC) h1u) h8) h1u)
  calc N * (1 + R.u)
      ≤ N * ((1 - g) * (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ^ 5 * (1 + 8 * R.u)) :=
        mul_le_mul_of_nonneg_left hnum hN
    _ = N * (1 - g) * (1 - R.u) * ((1 + G * (1 - R.u) ^ 2) * (1 - R.u)) * (1 - R.u) *
          (1 + 8 * R.u) * (1 - R.u) * (1 - R.u) := by ring
    _ ≤ _ := hp3

/-- The chain of `fp.norm2_lower` from an upper bound of `fl(‖v‖)`. -/
theorem chain_low (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {N s c g G tail : ℝ} (hN : 0 ≤ N)
    (hs0 : 0 ≤ s) (hs : s ≤ N * (1 + g) * (1 + R.u))
    (hc : c ≤ (1 - G * (1 - R.u) ^ 2) * (1 + R.u)) (hg0 : 0 ≤ g)
    (hC0 : 0 ≤ 1 - G * (1 - R.u) ^ 2) (ht : 0 ≤ tail)
    (hnum : (1 + g) * (1 - G * (1 - R.u) ^ 2) * (1 + R.u) ^ 5 ≤ (1 - R.u) * (1 + 8 * R.u)) :
    max 0 (R.rnd (R.rnd (R.rnd (s * c) / (1 + 8 * R.u)) - tail)) ≤ N * (1 - R.u) := by
  have hu := R.u_nonneg
  have h8 : 0 < 1 + 8 * R.u := by linarith
  have h1 : 0 ≤ 1 + R.u := by linarith
  have hS : 0 ≤ N * (1 + g) * (1 + R.u) := mul_nonneg (mul_nonneg hN (by linarith)) h1
  have hC : 0 ≤ (1 - G * (1 - R.u) ^ 2) * (1 + R.u) := mul_nonneg hC0 h1
  have hp := R.rnd_le_mul h0 hu1 hs0 hs hc hC hS
  have hpS : 0 ≤ N * (1 + g) * (1 + R.u) * ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) * (1 + R.u) :=
    mul_nonneg (mul_nonneg hS hC) h1
  have hp2 := R.rnd_le_of_le h0 hu1 (div_le_div_of_nonneg_right hp h8.le) (div_nonneg hpS h8.le)
  have hp3 := R.rnd_le_of_le h0 hu1 ((sub_le_self _ ht).trans hp2)
    (mul_nonneg (div_nonneg hpS h8.le) h1)
  have hfin : N * (1 + g) * (1 + R.u) * ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) * (1 + R.u) /
      (1 + 8 * R.u) * (1 + R.u) * (1 + R.u) ≤ N * (1 - R.u) := by
    rw [show N * (1 + g) * (1 + R.u) * ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) * (1 + R.u) /
        (1 + 8 * R.u) * (1 + R.u) * (1 + R.u) =
        N * ((1 + g) * (1 - G * (1 - R.u) ^ 2) * (1 + R.u) ^ 5) / (1 + 8 * R.u) by ring]
    rw [div_le_iff₀ h8]
    calc N * ((1 + g) * (1 - G * (1 - R.u) ^ 2) * (1 + R.u) ^ 5)
        ≤ N * ((1 - R.u) * (1 + 8 * R.u)) := mul_le_mul_of_nonneg_left hnum hN
      _ = N * (1 - R.u) * (1 + 8 * R.u) := by ring
  exact max_le (mul_nonneg hN (by linarith)) (hp3.trans hfin)

/-- `|A| |x| / fl(1 - γ̂_{w+1}) (1 + 4u)`, two roundings, bounds `(|A| |x|)_i` from above. -/
theorem chain_abs (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {a ah d g G : ℝ} (ha : 0 ≤ a)
    (hah : a * (1 - g) ≤ ah) (hg1 : g ≤ 1) (hd0 : 0 < d)
    (hd : d ≤ (1 - G * (1 - R.u) ^ 2) * (1 + R.u))
    (hnum : (1 - G * (1 - R.u) ^ 2) * (1 + R.u) ≤ (1 - g) * (1 - R.u) ^ 2 * (1 + 4 * R.u)) :
    a ≤ R.rnd (R.rnd (ah / d) * (1 + 4 * R.u)) := by
  have hu := R.u_nonneg
  have h1u : 0 ≤ 1 - R.u := by linarith
  have hD0 : 0 < (1 - G * (1 - R.u) ^ 2) * (1 + R.u) := lt_of_lt_of_le hd0 hd
  have hah0 : 0 ≤ a * (1 - g) := mul_nonneg ha (by linarith)
  have hq : a * (1 - g) / ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) ≤ ah / d :=
    (div_le_div_of_nonneg_left hah0 hd0 hd).trans (div_le_div_of_nonneg_right hah hd0.le)
  have hq0 : 0 ≤ a * (1 - g) / ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) := div_nonneg hah0 hD0.le
  have hb1 := R.le_rnd_of_le h0 hu1 hq hq0
  have hb2 := R.le_rnd_mul h0 hu1 (mul_nonneg hq0 h1u) (by linarith : (0 : ℝ) ≤ 1 + 4 * R.u)
    hb1 le_rfl
  refine le_trans ?_ hb2
  rw [show a * (1 - g) / ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) * (1 - R.u) * (1 + 4 * R.u) *
      (1 - R.u) = a * ((1 - g) * (1 - R.u) ^ 2 * (1 + 4 * R.u)) /
        ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) by ring, le_div_iff₀ hD0]
  exact mul_le_mul_of_nonneg_left hnum ha

/-- The last three roundings of `_rounding_bound`. -/
theorem chain_E (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {a q b2 qh gh G : ℝ} (ha : 0 ≤ a) (hq : 0 ≤ q)
    (hb2 : a ≤ b2) (hqh : q * (1 - R.u) ≤ qh) (hG0 : 0 ≤ G) (hgh : G * (1 - R.u) ^ 2 ≤ gh) :
    G * ((1 - R.u) ^ 6 * (1 + 4 * R.u)) * (a + q) ≤
      R.rnd (R.rnd (R.rnd (b2 + qh) * gh) * (1 + 4 * R.u)) := by
  have hu := R.u_nonneg
  have h1u : 0 ≤ 1 - R.u := by linarith
  have haq : 0 ≤ a + q := add_nonneg ha hq
  have hz : (a + q) * (1 - R.u) ≤ b2 + qh := by nlinarith [mul_nonneg ha hu]
  have hb3 := R.le_rnd_of_le h0 hu1 hz (mul_nonneg haq h1u)
  have hG2 : 0 ≤ G * (1 - R.u) ^ 2 := mul_nonneg hG0 (sq_nonneg _)
  have hb4 := R.le_rnd_mul h0 hu1 (mul_nonneg (mul_nonneg haq h1u) h1u) hG2 hb3 hgh
  have hb5 := R.le_rnd_mul h0 hu1
    (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg haq h1u) h1u) hG2) h1u)
    (by linarith : (0 : ℝ) ≤ 1 + 4 * R.u) hb4 le_rfl
  calc G * ((1 - R.u) ^ 6 * (1 + 4 * R.u)) * (a + q)
      = (a + q) * (1 - R.u) * (1 - R.u) * (G * (1 - R.u) ^ 2) * (1 - R.u) * (1 + 4 * R.u) *
          (1 - R.u) := by ring
    _ ≤ _ := hb5

/-- One residual entry: `fl(ŷ - fl(x μ))` against `y - μ x`. -/
theorem resid_entry (h0 : R.eta = 0) {y yh x μ a : ℝ} (g : ℝ) (hy : |yh - y| ≤ g * a)
    (hya : |y| ≤ a) :
    |R.rnd (yh - R.rnd (x * μ)) - (y - μ * x)| ≤
      (g + R.u * (1 + g)) * a + (2 * R.u + R.u ^ 2) * (|x| * |μ|) := by
  have hu := R.u_nonneg
  have hm := R.abs_rnd_sub_le h0 (x * μ)
  rw [abs_mul] at hm
  have hr := R.abs_rnd_sub_le h0 (yh - R.rnd (x * μ))
  obtain ⟨m1, m2⟩ := abs_le.mp hm
  obtain ⟨x1, x2⟩ := abs_le.mp (le_of_eq (abs_mul x μ))
  obtain ⟨y1, y2⟩ := abs_le.mp hy
  obtain ⟨a1, a2⟩ := abs_le.mp hya
  have hD : |yh - R.rnd (x * μ)| ≤ a * (1 + g) + |x| * |μ| * (1 + R.u) :=
    abs_le.mpr ⟨by nlinarith, by nlinarith⟩
  have hrD := (hr.trans (mul_le_mul_of_nonneg_left hD hu))
  obtain ⟨r1, r2⟩ := abs_le.mp hrD
  exact abs_le.mpr ⟨by nlinarith, by nlinarith⟩

/-- The final quotient and its three roundings. -/
theorem chain_final (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {nr nRc nE nx xl Rup Eup c1 c0 : ℝ}
    (hr : nr ≤ nRc + nE) (hnRc : 0 ≤ nRc) (hnE : 0 ≤ nE) (hRup : nRc * (1 + R.u) ≤ Rup)
    (hEup : nE * (1 + R.u) ≤ Eup) (hc1 : 0 ≤ c1) (hc0 : 0 ≤ c0) (hxl0 : 0 < xl)
    (hxl : xl ≤ nx * (1 - R.u)) (hu1' : R.u < 1)
    (hnum : 1 ≤ (1 + R.u) * (1 - R.u) ^ 4 * (1 + 4 * R.u)) :
    nr / nx ≤ R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (Rup + Eup) + c1) / xl) * (1 + 4 * R.u)) + c0) := by
  have hu := R.u_nonneg
  have h1u : 0 < 1 - R.u := by linarith
  have hnx : 0 < nx := by
    by_contra hneg
    push Not at hneg
    nlinarith
  have hS0 : 0 ≤ (nRc + nE) * (1 + R.u) := mul_nonneg (add_nonneg hnRc hnE) (by linarith)
  have hS : (nRc + nE) * (1 + R.u) ≤ Rup + Eup := by nlinarith
  have hU1 := R.le_rnd_of_le h0 hu1 hS hS0
  have hU2 := R.le_rnd_of_le h0 hu1 (hU1.trans (le_add_of_nonneg_right hc1))
    (mul_nonneg hS0 h1u.le)
  have hS2 : 0 ≤ (nRc + nE) * (1 + R.u) * (1 - R.u) * (1 - R.u) :=
    mul_nonneg (mul_nonneg hS0 h1u.le) h1u.le
  have hQ : (nRc + nE) * (1 + R.u) * (1 - R.u) * (1 - R.u) / (nx * (1 - R.u)) ≤
      R.rnd (R.rnd (Rup + Eup) + c1) / xl :=
    (div_le_div_of_nonneg_left hS2 hxl0 hxl).trans (div_le_div_of_nonneg_right hU2 hxl0.le)
  have hQ0 : 0 ≤ (nRc + nE) * (1 + R.u) * (1 - R.u) * (1 - R.u) / (nx * (1 - R.u)) :=
    div_nonneg hS2 (mul_pos hnx h1u).le
  have hq := R.le_rnd_of_le h0 hu1 hQ hQ0
  have hq2 := R.le_rnd_mul h0 hu1 (mul_nonneg hQ0 h1u.le) (by linarith : (0 : ℝ) ≤ 1 + 4 * R.u)
    hq le_rfl
  have hq3 := R.le_rnd_of_le h0 hu1 (hq2.trans (le_add_of_nonneg_right hc0))
    (mul_nonneg (mul_nonneg (mul_nonneg hQ0 h1u.le) (by linarith)) h1u.le)
  have hne1 : (1 - R.u) ≠ 0 := h1u.ne'
  have hnx' : nx ≠ 0 := hnx.ne'
  calc nr / nx ≤ (nRc + nE) * ((1 + R.u) * (1 - R.u) ^ 4 * (1 + 4 * R.u)) / nx :=
        div_le_div_of_nonneg_right (hr.trans (le_mul_of_one_le_right (add_nonneg hnRc hnE) hnum))
          hnx.le
    _ = (nRc + nE) * (1 + R.u) * (1 - R.u) * (1 - R.u) / (nx * (1 - R.u)) * (1 - R.u) *
          (1 + 4 * R.u) * (1 - R.u) * (1 - R.u) := by
        field_simp
    _ ≤ _ := hq3

end Rounding

/-! ### Euclidean norms -/

section Norms

variable {ι : Type*} [Fintype ι]

/-- The Euclidean norm of a real vector given by its entries. -/
noncomputable def l2 (v : ι → ℝ) : ℝ := ‖(WithLp.toLp 2 v : EuclideanSpace ℝ ι)‖

theorem l2_eq (v : ι → ℝ) : l2 v = Real.sqrt (∑ i, v i ^ 2) := by
  rw [l2, EuclideanSpace.norm_eq]
  simp [Real.norm_eq_abs, sq_abs]

theorem l2_nonneg (v : ι → ℝ) : 0 ≤ l2 v := norm_nonneg _

theorem l2_le_add (a b : ι → ℝ) : l2 a ≤ l2 b + l2 (fun i => a i - b i) := by
  have e : (WithLp.toLp 2 a : EuclideanSpace ℝ ι) =
      WithLp.toLp 2 b + WithLp.toLp 2 (fun i => a i - b i) := by
    refine PiLp.ext fun k => ?_
    simp
  unfold l2
  rw [e]
  exact norm_add_le _ _

theorem l2_le_of_abs_le {a b : ι → ℝ} (h : ∀ i, |a i| ≤ b i) : l2 a ≤ l2 b := by
  rw [l2_eq, l2_eq]
  apply Real.sqrt_le_sqrt
  refine Finset.sum_le_sum fun i _ => ?_
  rw [← sq_abs (a i)]
  exact pow_le_pow_left₀ (abs_nonneg _) (h i) 2

theorem l2_ofLp (x : EuclideanSpace ℝ ι) : l2 (fun i => x i) = ‖x‖ := by
  rw [l2_eq, EuclideanSpace.norm_eq]
  simp [Real.norm_eq_abs, sq_abs]

variable [DecidableEq ι]

theorem l2_residual (A : Matrix ι ι ℝ) (x : EuclideanSpace ℝ ι) (μ : ℝ) :
    l2 (fun i => ∑ j, A i j * x j - μ * x i) = ‖toEuclideanLin A x - μ • x‖ := by
  rw [l2_eq, EuclideanSpace.norm_eq]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  have hc : (toEuclideanLin A x) i = ∑ j, A i j * x j := by
    rw [Matrix.toLpLin_apply]
    rfl
  rw [PiLp.sub_apply, PiLp.smul_apply, hc, smul_eq_mul, Real.norm_eq_abs, sq_abs]

/-- A summation tree over all indices, each once (the sums of squares of the norms). -/
def FullTree (T : STree ι) : Prop := T.leaves.Nodup ∧ ∀ j, j ∈ T.leaves

/-- Row summation trees: `T i` sums row `i` over its stored entries, at most `w` of them. -/
def RowTrees (T : ι → STree ι) (A : ι → ι → ℝ) (w : ℕ) : Prop :=
  ∀ i, (T i).leaves.Nodup ∧ (T i).adds + 1 ≤ w ∧ ∀ j, j ∉ (T i).leaves → A i j = 0

/-- `np.linalg.norm` of a real vector: squares rounded, summed along `T`, rounded square root. -/
noncomputable def normFl (R : Rounding ℝ) (T : STree ι) (v : ι → ℝ) : ℝ :=
  R.rnd (Real.sqrt (T.comp R.rnd (fun j => R.rnd (v j * v j))))

/-- `fp.norm2_upper`: `fl(‖v‖) (1 + γ̂_{n+2}) (1 + 8u) + tail`, every operation rounded. -/
noncomputable def norm2Up (R : Rounding ℝ) (T : STree ι) (tail : ℝ) (v : ι → ℝ) : ℝ :=
  R.rnd (R.rnd (R.rnd (normFl R T v * R.rnd (1 + gamHat R (Fintype.card ι + 2))) *
    (1 + 8 * R.u)) + tail)

/-- `fp.norm2_lower`: `max 0 (fl(‖v‖) (1 - γ̂_{n+2}) / (1 + 8u) - tail)`, every operation
rounded. -/
noncomputable def norm2Low (R : Rounding ℝ) (T : STree ι) (tail : ℝ) (v : ι → ℝ) : ℝ :=
  max 0 (R.rnd (R.rnd (R.rnd (normFl R T v * R.rnd (1 - gamHat R (Fintype.card ι + 2))) /
    (1 + 8 * R.u)) - tail))

theorem sum_sq_error (R : Rounding ℝ) (h0 : R.eta = 0) {T : STree ι} (hT : FullTree T)
    (hn : (Fintype.card ι : ℝ) * R.u < 1) (v : ι → ℝ) :
    |T.comp R.rnd (fun j => R.rnd (v j * v j)) - ∑ j, v j ^ 2| ≤
      gam R.u (Fintype.card ι) * ∑ j, v j ^ 2 := by
  have hu := R.u_nonneg
  have hlen : T.adds + 1 ≤ Fintype.card ι := by
    rw [← STree.length_leaves]
    exact hT.1.length_le_card
  have hk : ((T.adds + 1 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hu) hn
  have h := dot_error R v v T hk
  rw [h0, mul_zero, add_zero] at h
  have he : T.eval (fun j => v j * v j) = ∑ j, v j ^ 2 := by
    rw [STree.eval_eq_finset_sum _ T hT.1 (fun j hj => absurd (hT.2 j) hj)]
    exact Finset.sum_congr rfl fun j _ => (sq (v j)).symm
  have ha : T.absEval (fun j => v j * v j) = ∑ j, v j ^ 2 := by
    rw [STree.absEval_eq_finset_sum _ T hT.1 (fun j hj => absurd (hT.2 j) hj)]
    exact Finset.sum_congr rfl fun j _ => by rw [Real.norm_eq_abs, abs_mul_self, sq]
  rw [he, ha, Real.norm_eq_abs] at h
  have hs : 0 ≤ ∑ j, v j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg (v j)
  exact h.trans (mul_le_mul_of_nonneg_right (gam_mono hu hlen hn) hs)

theorem normFl_bounds (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {T : STree ι}
    (hT : FullTree T) (hn : (Fintype.card ι : ℝ) * R.u < 1) (hg1 : gam R.u (Fintype.card ι) ≤ 1)
    (v : ι → ℝ) :
    l2 v * (1 - gam R.u (Fintype.card ι)) * (1 - R.u) ≤ normFl R T v ∧
      normFl R T v ≤ l2 v * (1 + gam R.u (Fintype.card ι)) * (1 + R.u) := by
  have hu := R.u_nonneg
  have hg0 : 0 ≤ gam R.u (Fintype.card ι) := gam_nonneg hu hn
  have herr := sum_sq_error R h0 hT hn v
  have hs : 0 ≤ ∑ j, v j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg (v j)
  obtain ⟨hlo, hhi⟩ := abs_le.mp herr
  have hl2 := l2_eq v
  have hsq_lo : l2 v * (1 - gam R.u (Fintype.card ι)) ≤
      Real.sqrt (T.comp R.rnd (fun j => R.rnd (v j * v j))) := by
    have h1 : (∑ j, v j ^ 2) * (1 - gam R.u (Fintype.card ι)) ≤
        T.comp R.rnd (fun j => R.rnd (v j * v j)) := by nlinarith
    have h2 := Real.sqrt_le_sqrt h1
    rw [Real.sqrt_mul hs] at h2
    have h3 : 1 - gam R.u (Fintype.card ι) ≤ Real.sqrt (1 - gam R.u (Fintype.card ι)) := by
      have := Real.sqrt_le_sqrt (show (1 - gam R.u (Fintype.card ι)) ^ 2 ≤
        1 - gam R.u (Fintype.card ι) by nlinarith)
      rwa [Real.sqrt_sq (by linarith)] at this
    rw [hl2]
    exact (mul_le_mul_of_nonneg_left h3 (Real.sqrt_nonneg _)).trans h2
  have hsq_hi : Real.sqrt (T.comp R.rnd (fun j => R.rnd (v j * v j))) ≤
      l2 v * (1 + gam R.u (Fintype.card ι)) := by
    have h1 : T.comp R.rnd (fun j => R.rnd (v j * v j)) ≤
        (∑ j, v j ^ 2) * (1 + gam R.u (Fintype.card ι)) := by nlinarith
    have h2 := Real.sqrt_le_sqrt h1
    rw [Real.sqrt_mul hs] at h2
    have h3 : Real.sqrt (1 + gam R.u (Fintype.card ι)) ≤ 1 + gam R.u (Fintype.card ι) := by
      have := Real.sqrt_le_sqrt (show 1 + gam R.u (Fintype.card ι) ≤
        (1 + gam R.u (Fintype.card ι)) ^ 2 by nlinarith)
      rwa [Real.sqrt_sq (by linarith)] at this
    rw [hl2]
    exact h2.trans (mul_le_mul_of_nonneg_left h3 (Real.sqrt_nonneg _))
  exact ⟨R.le_rnd_of_le h0 hu1 hsq_lo (mul_nonneg (l2_nonneg v) (by linarith)),
    R.rnd_le_of_le h0 hu1 hsq_hi (mul_nonneg (l2_nonneg v) (by linarith))⟩

/-- **`fp.norm2_upper` is an upper bound**, with the slack `1 + u` used by the final quotient. -/
theorem norm2Up_ge (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    (hn : Fintype.card ι ≤ 2 ^ 26) {T : STree ι} (hT : FullTree T) {tail : ℝ} (ht : 0 ≤ tail)
    (v : ι → ℝ) : l2 v * (1 + R.u) ≤ norm2Up R T tail v := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hk2, hg0, hs1, hs2, hG, hG2⟩ := small_consts hu hu53 hn
  have hu1' : R.u ≤ 1 := by linarith
  have hnu : (Fintype.card ι : ℝ) * R.u < 1 := by
    have : (Fintype.card ι : ℝ) * R.u ≤ ((Fintype.card ι + 2 : ℕ) : ℝ) * R.u :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : Fintype.card ι ≤ Fintype.card ι + 2))
        hu
    linarith
  have hg1 : gam R.u (Fintype.card ι) ≤ 1 := by linarith
  obtain ⟨hs_lo, -⟩ := normFl_bounds R h0 hu1' hT hnu hg1 v
  obtain ⟨hgh, -⟩ := gamHat_bounds R h0 (by linarith) (k := Fintype.card ι + 2) (by linarith)
  have hG0 : 0 ≤ gam R.u (Fintype.card ι + 2) := by linarith
  have hc := R.le_rnd_of_le h0 hu1' (w := 1 + gamHat R (Fintype.card ι + 2))
    (z := 1 + gam R.u (Fintype.card ι + 2) * (1 - R.u) ^ 2) (by linarith)
    (by linarith [mul_nonneg hG0 (sq_nonneg (1 - R.u))])
  exact R.chain_up h0 hu1' (l2_nonneg v) hs_lo hc hg1 hG0 ht
    (num_norm_up hu hu1 hg0 (by linarith) hG hG2)

/-- **`fp.norm2_lower` is a lower bound**, with the slack `1 - u`. -/
theorem norm2Low_le (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    (hn : Fintype.card ι ≤ 2 ^ 26) {T : STree ι} (hT : FullTree T) {tail : ℝ} (ht : 0 ≤ tail)
    (v : ι → ℝ) : norm2Low R T tail v ≤ l2 v * (1 - R.u) := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hk2, hg0, hs1, hs2, hG, hG2⟩ := small_consts hu hu53 hn
  have hu1' : R.u ≤ 1 := by linarith
  have hnu : (Fintype.card ι : ℝ) * R.u < 1 := by
    have : (Fintype.card ι : ℝ) * R.u ≤ ((Fintype.card ι + 2 : ℕ) : ℝ) * R.u :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : Fintype.card ι ≤ Fintype.card ι + 2))
        hu
    linarith
  have hg1 : gam R.u (Fintype.card ι) ≤ 1 := by linarith
  obtain ⟨hs_lo, hs_hi⟩ := normFl_bounds R h0 hu1' hT hnu hg1 v
  obtain ⟨hgh, -⟩ := gamHat_bounds R h0 (by linarith) (k := Fintype.card ι + 2) (by linarith)
  have hG0 : 0 ≤ gam R.u (Fintype.card ι + 2) := by linarith
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hC0 : 0 ≤ 1 - gam R.u (Fintype.card ι + 2) * (1 - R.u) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsq1 hG0]
  have hc := R.rnd_le_of_le h0 hu1' (w := 1 - gamHat R (Fintype.card ι + 2))
    (z := 1 - gam R.u (Fintype.card ι + 2) * (1 - R.u) ^ 2) (by linarith) hC0
  have hs0 : 0 ≤ normFl R T v :=
    (mul_nonneg (mul_nonneg (l2_nonneg v) (by linarith)) (by linarith)).trans hs_lo
  exact R.chain_low h0 hu1' (l2_nonneg v) hs0 hs_hi hc hg0 hC0 ht
    (num_norm_low hu hu1 hg0 (by linarith) hG)

end Norms

/-! ### The residual and its rounding bound -/

section Residual

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- `adapter.matvec`: row `i` of `A x`, products rounded and summed along `T i`. -/
noncomputable def matvecFl (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℝ) (x : ι → ℝ)
    (i : ι) : ℝ :=
  (T i).comp R.rnd (fun j => R.rnd (A i j * x j))

/-- The computed residual `HX - X mu`. -/
noncomputable def residFl (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℝ) (x : ι → ℝ) (μ : ℝ)
    (i : ι) : ℝ :=
  R.rnd (matvecFl R T A x i - R.rnd (x i * μ))

/-- `adapter.abs_matvec`: row `i` of `|A| |x|`. -/
noncomputable def absMatvecFl (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℝ) (x : ι → ℝ)
    (i : ι) : ℝ :=
  (T i).comp R.rnd (fun j => R.rnd (|A i j| * |x j|))

/-- `_rounding_bound`, real data:
`γ̂_{w+2} (|A| |x| / (1 - γ̂_{w+1}) (1 + 4u) + |x| |μ|) (1 + 4u)`, operation by operation. -/
noncomputable def roundBoundFl (R : Rounding ℝ) (T : ι → STree ι) (w : ℕ) (A : ι → ι → ℝ)
    (x : ι → ℝ) (μ : ℝ) (i : ι) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (absMatvecFl R T A x i / R.rnd (1 - gamHat R (w + 1))) *
    (1 + 4 * R.u)) + R.rnd (|x i| * |μ|)) * gamHat R (w + 2)) * (1 + 4 * R.u))

/-- The computed relative radius `rel` of `residuals`. -/
noncomputable def residualRadiusFl (R : Rounding ℝ) (T T' : ι → STree ι) (TR TE TX : STree ι)
    (w : ℕ) (A : ι → ι → ℝ) (x : ι → ℝ) (μ tR tE tX c1 c0 : ℝ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (norm2Up R TR tR (residFl R T A x μ) +
    norm2Up R TE tE (roundBoundFl R T' w A x μ)) + c1) / norm2Low R TX tX x) *
      (1 + 4 * R.u)) + c0)

/-- **Matrix-vector error** (Higham eq. (3.11) row by row): `|ŷ_i - (A x)_i| ≤ γ_w (|A| |x|)_i`. -/
theorem matvec_error (R : Rounding ℝ) (h0 : R.eta = 0) {T : ι → STree ι} {A : ι → ι → ℝ} {w : ℕ}
    (hT : RowTrees T A w) (hw : (w : ℝ) * R.u < 1) (x : ι → ℝ) (i : ι) :
    |matvecFl R T A x i - ∑ j, A i j * x j| ≤ gam R.u w * ∑ j, |A i j| * |x j| := by
  obtain ⟨hnd, hlen, hsupp⟩ := hT i
  have hu := R.u_nonneg
  have hk : (((T i).adds + 1 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hu) hw
  have h := dot_error R (A i) x (T i) hk
  rw [h0, mul_zero, add_zero] at h
  have he : (T i).eval (fun j => A i j * x j) = ∑ j, A i j * x j :=
    STree.eval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, zero_mul])
  have ha : (T i).absEval (fun j => A i j * x j) = ∑ j, |A i j| * |x j| := by
    rw [STree.absEval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, zero_mul])]
    exact Finset.sum_congr rfl fun j _ => by rw [Real.norm_eq_abs, abs_mul]
  rw [he, ha, Real.norm_eq_abs] at h
  unfold matvecFl
  exact h.trans (mul_le_mul_of_nonneg_right (gam_mono hu hlen hw)
    (Finset.sum_nonneg fun j _ => by positivity))

theorem absMatvec_ge (R : Rounding ℝ) (h0 : R.eta = 0) {T : ι → STree ι} {A : ι → ι → ℝ} {w : ℕ}
    (hT : RowTrees T A w) (hw : (w : ℝ) * R.u < 1) (x : ι → ℝ) (i : ι) :
    (∑ j, |A i j| * |x j|) * (1 - gam R.u w) ≤ absMatvecFl R T A x i := by
  obtain ⟨hnd, hlen, hsupp⟩ := hT i
  have hu := R.u_nonneg
  have hk : (((T i).adds + 1 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hu) hw
  have h := dot_error R (fun j => |A i j|) (fun j => |x j|) (T i) hk
  rw [h0, mul_zero, add_zero] at h
  have he : (T i).eval (fun j => |A i j| * |x j|) = ∑ j, |A i j| * |x j| :=
    STree.eval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, abs_zero, zero_mul])
  have ha : (T i).absEval (fun j => |A i j| * |x j|) = ∑ j, |A i j| * |x j| := by
    rw [STree.absEval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, abs_zero, zero_mul])]
    exact Finset.sum_congr rfl fun j _ => by
      rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  rw [he, ha, Real.norm_eq_abs] at h
  have hs : 0 ≤ ∑ j, |A i j| * |x j| := Finset.sum_nonneg fun j _ => by positivity
  have hg := mul_le_mul_of_nonneg_right (gam_mono hu hlen hw) hs
  unfold absMatvecFl
  nlinarith [(abs_le.mp h).1]

/-- **Theorem 1 accounting, entrywise**: `|R̂_i - (A x - μ x)_i| ≤ Ê_i`. -/
theorem resid_le_roundBound (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    {w : ℕ} (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26) {A : ι → ι → ℝ} {T T' : ι → STree ι}
    (hT : RowTrees T A w) (hT' : RowTrees T' A w) (x : ι → ℝ) (μ : ℝ) (i : ι) :
    |residFl R T A x μ i - (∑ j, A i j * x j - μ * x i)| ≤ roundBoundFl R T' w A x μ i := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, -⟩ := small_consts hu hu53 hw
  have hu1' : R.u ≤ 1 := by linarith
  have hw2 : ((w + 2 : ℕ) : ℝ) * R.u < 1 := by linarith
  have hw12 : ((w + 1 : ℕ) : ℝ) * R.u ≤ ((w + 2 : ℕ) : ℝ) * R.u :=
    mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 1 ≤ w + 2)) hu
  have hw1' : ((w + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt hw12 hw2
  have hw0 : (w : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w ≤ w + 2)) hu) hw2
  -- `2u + u² ≤ γ_{w+1}`
  have hg2u : 2 * R.u + R.u ^ 2 ≤ gam R.u (w + 1) := by
    have h1 := u_le_gam_one hu (by linarith)
    have h2 : ((1 + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 1 + 1 ≤ w + 2)) hu) hw2
    have s : gam R.u 1 + R.u * (1 + gam R.u 1) ≤ gam R.u (1 + 1) := gam_succ hu h2
    have m : gam R.u (1 + 1) ≤ gam R.u (w + 1) := gam_mono hu (by omega) hw1'
    nlinarith
  set a := ∑ j, |A i j| * |x j| with ha_def
  set q := |x i| * |μ| with hq_def
  have ha0 : 0 ≤ a := Finset.sum_nonneg fun j _ => by positivity
  have hq0 : 0 ≤ q := by positivity
  -- the residual entry
  have hy := matvec_error R h0 hT hw0 x i
  have hya : |∑ j, A i j * x j| ≤ a := (Finset.abs_sum_le_sum_abs _ _).trans
    (le_of_eq (Finset.sum_congr rfl fun j _ => abs_mul _ _))
  have h1 : |residFl R T A x μ i - (∑ j, A i j * x j - μ * x i)| ≤
      (gam R.u w + R.u * (1 + gam R.u w)) * a + (2 * R.u + R.u ^ 2) * q :=
    R.resid_entry h0 (gam R.u w) hy hya
  have hsucc : gam R.u w + R.u * (1 + gam R.u w) ≤ gam R.u (w + 1) := gam_succ hu hw1'
  have h2 : |residFl R T A x μ i - (∑ j, A i j * x j - μ * x i)| ≤ gam R.u (w + 1) * (a + q) := by
    nlinarith [mul_le_mul_of_nonneg_right hsucc ha0, mul_le_mul_of_nonneg_right hg2u hq0]
  -- the rounding bound
  obtain ⟨hgh1, hgh1'⟩ := gamHat_bounds R h0 (by linarith) (k := w + 1) (by linarith)
  obtain ⟨hgh2, -⟩ := gamHat_bounds R h0 (by linarith) (k := w + 2) (by linarith)
  have hG1 : gam R.u (w + 1) ≤ 1 / 1000 := by linarith
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hG10 : 0 ≤ gam R.u (w + 1) := by linarith
  have hC0 : 0 ≤ 1 - gam R.u (w + 1) * (1 - R.u) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsq1 hG10]
  have hd : R.rnd (1 - gamHat R (w + 1)) ≤ (1 - gam R.u (w + 1) * (1 - R.u) ^ 2) * (1 + R.u) :=
    R.rnd_le_of_le h0 hu1' (by linarith) hC0
  have hd0 : 0 < R.rnd (1 - gamHat R (w + 1)) := by
    have := R.le_rnd_of_le h0 hu1' (w := 1 - gamHat R (w + 1)) (z := 1 - 3 * gam R.u (w + 1))
      (by linarith) (by linarith)
    nlinarith
  have hah := absMatvec_ge R h0 hT' hw0 x i
  have hb2 := R.chain_abs h0 hu1' ha0 hah (by linarith) hd0 hd
    (num_abs hu hu1 hgw0 hgw1 hG1)
  have hqh := R.le_rnd_of_le h0 hu1' (w := |x i| * |μ|) (z := q) le_rfl hq0
  have hE := R.chain_E h0 hu1' ha0 hq0 hb2 hqh (by linarith) hgh2
  have hnE := num_E hu hu1 hG10 hgw2 hGw
  have h3 := mul_le_mul_of_nonneg_right hnE (add_nonneg ha0 hq0)
  exact h2.trans (h3.trans hE)

/-- **Theorem 1 in floating point** (`docs/proofs/thm1_fp-accounting.md`, real data, standard
model): when the computed lower bound of `‖x‖` is positive, the computed radius `rel` of `residuals`
bounds `‖A x - μ x‖ / ‖x‖`. -/
theorem residual_radius_fl (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    {w : ℕ} (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26) (hn : Fintype.card ι ≤ 2 ^ 26)
    {A : ι → ι → ℝ} {T T' : ι → STree ι} (hT : RowTrees T A w) (hT' : RowTrees T' A w)
    {TR TE TX : STree ι} (hTR : FullTree TR) (hTE : FullTree TE) (hTX : FullTree TX)
    (x : ι → ℝ) (μ : ℝ) {tR tE tX c1 c0 : ℝ} (htR : 0 ≤ tR) (htE : 0 ≤ tE) (htX : 0 ≤ tX)
    (hc1 : 0 ≤ c1) (hc0 : 0 ≤ c0) (hxl : 0 < norm2Low R TX tX x) :
    l2 (fun i => ∑ j, A i j * x j - μ * x i) / l2 x ≤
      residualRadiusFl R T T' TR TE TX w A x μ tR tE tX c1 c0 := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 1000 := hu53.trans (by norm_num)
  have hentry := resid_le_roundBound R h0 hu53 hw1 hw hT hT' x μ
  have hr : l2 (fun i => ∑ j, A i j * x j - μ * x i) ≤
      l2 (residFl R T A x μ) + l2 (roundBoundFl R T' w A x μ) := by
    refine (l2_le_add _ (residFl R T A x μ)).trans (add_le_add le_rfl (l2_le_of_abs_le ?_))
    intro i
    rw [abs_sub_comm]
    exact hentry i
  exact R.chain_final h0 (by linarith) hr (l2_nonneg _) (l2_nonneg _)
    (norm2Up_ge R h0 hu53 hn hTR htR _) (norm2Up_ge R h0 hu53 hn hTE htE _) hc1 hc0 hxl
    (norm2Low_le R h0 hu53 hn hTX htX x) (by linarith) (num_final hu hu1)

/-- **Theorem 1 with the computed radius**: for a real symmetric matrix, some eigenvalue lies within
the computed `rel` of `μ`. -/
theorem residual_enclosure_fl {A : Matrix ι ι ℝ} (hA : A.IsHermitian) (R : Rounding ℝ)
    (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) {w : ℕ} (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26)
    (hn : Fintype.card ι ≤ 2 ^ 26) {T T' : ι → STree ι} (hT : RowTrees T A w)
    (hT' : RowTrees T' A w) {TR TE TX : STree ι} (hTR : FullTree TR) (hTE : FullTree TE)
    (hTX : FullTree TX) (x : EuclideanSpace ℝ ι) (μ : ℝ) {tR tE tX c1 c0 : ℝ} (htR : 0 ≤ tR)
    (htE : 0 ≤ tE) (htX : 0 ≤ tX) (hc1 : 0 ≤ c1) (hc0 : 0 ≤ c0)
    (hxl : 0 < norm2Low R TX tX (fun i => x i)) :
    ∃ i, |hA.eigenvalues i - μ| ≤
      residualRadiusFl R T T' TR TE TX w A (fun i => x i) μ tR tE tX c1 c0 := by
  have hu := R.u_nonneg
  have hlow := norm2Low_le R h0 hu53 hn hTX htX (fun i => x i)
  have hx : x ≠ 0 := by
    intro h
    have : l2 (fun i => x i) = 0 := by rw [l2_ofLp, h, norm_zero]
    rw [this, zero_mul] at hlow
    linarith
  obtain ⟨i, hi⟩ := residual_enclosure hA μ hx
  refine ⟨i, hi.trans ?_⟩
  have := residual_radius_fl R h0 hu53 hw1 hw hn hT hT' hTR hTE hTX (fun i => x i) μ htR htE htX
    hc1 hc0 hxl
  rwa [l2_residual, l2_ofLp] at this

end Residual

/-! ### Appendix A: the builder error (Lemmas A1, A2, A3 and Theorem A4) -/

namespace Rounding

/-- The standard model (`eta = 0`) with every value representable: the hypothesis `IsFloat` of
`add_err` is then not needed. -/
def std {V : Type*} [NormedAddCommGroup V] (R : Rounding V) (h0 : R.eta = 0) : Rounding V where
  rnd := R.rnd
  IsFloat := fun _ => True
  isFloat_rnd := fun _ => trivial
  u := R.u
  eta := 0
  u_nonneg := R.u_nonneg
  eta_nonneg := le_rfl
  err := fun v => by simpa [h0] using R.err v
  add_err := fun a b _ _ => by simpa [h0] using R.err (a + b)

end Rounding

section Builder

variable {𝕂 : Type*} [RCLike 𝕂]

theorem norm_le_add_norm_sub (a b : 𝕂) : ‖a‖ ≤ ‖b‖ + ‖a - b‖ := by
  calc ‖a‖ = ‖b + (a - b)‖ := by congr 1; abel
    _ ≤ ‖b‖ + ‖a - b‖ := norm_add_le _ _

/-- Relative errors compose under products: `γ_j` and `γ_k` give `γ_{j+k}` (Higham, Lemma 3.3). -/
theorem rel_mul {u : ℝ} (hu : 0 ≤ u) {j k : ℕ} (h : ((j + k : ℕ) : ℝ) * u < 1) {a ah b bh : 𝕂}
    (ha : ‖ah - a‖ ≤ gam u j * ‖a‖) (hb : ‖bh - b‖ ≤ gam u k * ‖b‖) :
    ‖ah * bh - a * b‖ ≤ gam u (j + k) * ‖a * b‖ := by
  have hj : (j : ℝ) * u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : j ≤ j + k)) hu) h
  have hk : (k : ℝ) * u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k ≤ j + k)) hu) h
  have gj := gam_nonneg hu hj
  have gk := gam_nonneg hu hk
  have hadd := gam_add hu h
  have hbh : ‖bh‖ ≤ (1 + gam u k) * ‖b‖ := by
    have := norm_le_add_norm_sub bh b
    nlinarith
  have e : ah * bh - a * b = (ah - a) * bh + a * (bh - b) := by ring
  rw [e, norm_mul a b]
  have hab := mul_le_mul_of_nonneg_right hadd (mul_nonneg (norm_nonneg a) (norm_nonneg b))
  calc ‖(ah - a) * bh + a * (bh - b)‖ ≤ ‖ah - a‖ * ‖bh‖ + ‖a‖ * ‖bh - b‖ := by
        refine (norm_add_le _ _).trans ?_
        rw [norm_mul, norm_mul]
    _ ≤ gam u j * ‖a‖ * ((1 + gam u k) * ‖b‖) + ‖a‖ * (gam u k * ‖b‖) :=
        add_le_add (mul_le_mul ha hbh (norm_nonneg _) (mul_nonneg gj (norm_nonneg _)))
          (mul_le_mul_of_nonneg_left hb (norm_nonneg _))
    _ ≤ gam u (j + k) * (‖a‖ * ‖b‖) := by nlinarith

/-- One more rounding adds one to the index of `γ`. -/
theorem rel_rnd (R : Rounding 𝕂) (h0 : R.eta = 0) {k : ℕ} (hk : ((k + 1 : ℕ) : ℝ) * R.u < 1)
    {p ph : 𝕂} (h : ‖ph - p‖ ≤ gam R.u k * ‖p‖) : ‖R.rnd ph - p‖ ≤ gam R.u (k + 1) * ‖p‖ := by
  have hu := R.u_nonneg
  have hs := gam_succ hu hk
  have hk0 : (k : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k ≤ k + 1)) hu) hk
  have g0 := gam_nonneg hu hk0
  have hr := R.err ph
  rw [h0, add_zero] at hr
  have hph : ‖ph‖ ≤ (1 + gam R.u k) * ‖p‖ := by
    have := norm_le_add_norm_sub ph p
    nlinarith
  have hsp := mul_le_mul_of_nonneg_right hs (norm_nonneg p)
  calc ‖R.rnd ph - p‖ = ‖(R.rnd ph - ph) + (ph - p)‖ := by congr 1; abel
    _ ≤ ‖R.rnd ph - ph‖ + ‖ph - p‖ := norm_add_le _ _
    _ ≤ R.u * ((1 + gam R.u k) * ‖p‖) + gam R.u k * ‖p‖ :=
        add_le_add (hr.trans (mul_le_mul_of_nonneg_left hph hu)) h
    _ ≤ gam R.u (k + 1) * ‖p‖ := by nlinarith

/-- The builder's amplitude of a monomial `c f₁ ⋯ f_L`: `fl(⋯fl(fl(c f̂₁) f̂₂)⋯ f̂_L)` with
`f̂ = fl(f)` (`construct._apply_string`). -/
noncomputable def monoFl (R : Rounding 𝕂) : 𝕂 → List 𝕂 → 𝕂
  | c, [] => c
  | c, f :: fs => monoFl R (R.rnd (c * R.rnd f)) fs

/-- **The amplitude of a monomial** with `L` factors: `|t̂ - t| ≤ γ_{2L} |t|` (from an exact
coupling, `k = 0`). -/
theorem mono_error (R : Rounding 𝕂) (h0 : R.eta = 0) :
    ∀ (fs : List 𝕂) (ch c : 𝕂) (k : ℕ), ‖ch - c‖ ≤ gam R.u k * ‖c‖ →
      ((k + 2 * fs.length : ℕ) : ℝ) * R.u < 1 →
      ‖monoFl R ch fs - c * fs.prod‖ ≤ gam R.u (k + 2 * fs.length) * ‖c * fs.prod‖
  | [], ch, c, k, h, _ => by simpa [monoFl] using h
  | f :: fs, ch, c, k, h, hk => by
    have hu := R.u_nonneg
    have e : k + 2 * (f :: fs).length = k + 1 + 1 + 2 * fs.length := by
      simp only [List.length_cons]
      omega
    have hk' : ((k + 1 + 1 + 2 * fs.length : ℕ) : ℝ) * R.u < 1 := by
      rw [← e]
      exact hk
    have hk2 : ((k + 1 + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k + 1 + 1 ≤ k + 1 + 1 + 2 * fs.length))
        hu) hk'
    have hk1 : ((k + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k + 1 ≤ k + 1 + 1)) hu) hk2
    have hu1 : R.u < 1 := lt_of_le_of_lt
      (le_mul_of_one_le_left hu (by exact_mod_cast (by omega : 1 ≤ k + 1))) hk1
    have hf : ‖R.rnd f - f‖ ≤ gam R.u 1 * ‖f‖ := by
      have := R.err f
      rw [h0, add_zero] at this
      exact this.trans (mul_le_mul_of_nonneg_right (u_le_gam_one hu hu1) (norm_nonneg f))
    have hm := rel_mul hu (j := k) (k := 1) hk1 h hf
    have hr := rel_rnd R h0 (k := k + 1) hk2 hm
    have ih := mono_error R h0 fs _ (c * f) (k + 1 + 1) hr hk'
    simp only [monoFl, List.prod_cons]
    rw [← mul_assoc, e]
    exact ih

/-- An entry of `H_ref`: the computed amplitudes `th k` of the monomials `k ∈ S`, summed in any
order along a tree with at most `m` leaves (`0` when `S` is empty). -/
def EntryBuilt (R : Rounding 𝕂) (m : ℕ) {κ : Type*} (th : κ → 𝕂) (S : Finset κ) (h : 𝕂) :
    Prop :=
  (S = ∅ ∧ h = 0) ∨
    ∃ T : STree κ, T.leaves.Nodup ∧ (∀ k, k ∈ T.leaves ↔ k ∈ S) ∧ T.adds + 1 ≤ m ∧
      h = T.comp R.rnd th

/-- **Lemma A1** (entrywise builder error) for one entry:
`|H_ref - H_spec|_{ij} ≤ γ_{m+2L-1} (|H|_abs)_{ij}`. -/
theorem builder_entry_error (R : Rounding 𝕂) (h0 : R.eta = 0) {m L : ℕ}
    (hW : ((m + 2 * L : ℕ) : ℝ) * R.u < 1) {κ : Type*} [DecidableEq κ] (t th : κ → 𝕂)
    (S : Finset κ) (hth : ∀ k ∈ S, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) {h : 𝕂}
    (hb : EntryBuilt R m th S h) :
    ‖h - ∑ k ∈ S, t k‖ ≤ gam R.u (m + 2 * L - 1) * ∑ k ∈ S, ‖t k‖ := by
  have hu := R.u_nonneg
  rcases hb with ⟨hS, hh⟩ | ⟨T, hnd, hmem, hlen, hh⟩
  · subst hS
    subst hh
    simp
  · have hW1 : ((m + 2 * L - 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : m + 2 * L - 1 ≤ m + 2 * L)) hu) hW
    have hT : ((2 * L + T.adds : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 2 * L + T.adds ≤ m + 2 * L)) hu) hW
    have h2L : ((2 * L : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 2 * L ≤ m + 2 * L)) hu) hW
    let t0 : κ → 𝕂 := fun k => if k ∈ S then t k else th k
    have hleaf : ∀ k, ‖th k - t0 k‖ ≤ gam R.u (2 * L) * ‖t0 k‖ + 0 := by
      intro k
      by_cases hk : k ∈ S
      · simp only [t0, hk, ↓reduceIte, add_zero]
        exact hth k hk
      · simp only [t0, hk, ↓reduceIte, sub_self, norm_zero, add_zero]
        exact mul_nonneg (gam_nonneg hu h2L) (norm_nonneg _)
    have herr := STree.comp_error (R.std h0) le_rfl t0 th hleaf (fun _ => trivial) T hT
    have hfs : T.leaves.toFinset = S := Finset.ext fun k => by simp [hmem]
    have he : T.eval t0 = ∑ k ∈ S, t k := by
      rw [STree.eval_eq_sum, ← List.sum_toFinset _ hnd, hfs]
      exact Finset.sum_congr rfl fun k hk => by simp [t0, hk]
    have ha : T.absEval t0 = ∑ k ∈ S, ‖t k‖ := by
      rw [STree.absEval_eq_sum, ← List.sum_toFinset _ hnd, hfs]
      exact Finset.sum_congr rfl fun k hk => by simp [t0, hk]
    have hg : gam R.u (2 * L + T.adds) ≤ gam R.u (m + 2 * L - 1) := gam_mono hu (by omega) hW1
    have hs0 : 0 ≤ ∑ k ∈ S, ‖t k‖ := Finset.sum_nonneg fun k _ => norm_nonneg _
    rw [hh]
    calc ‖T.comp R.rnd th - ∑ k ∈ S, t k‖ = ‖T.comp (R.std h0).rnd th - T.eval t0‖ := by
          rw [he]
          rfl
      _ ≤ gam R.u (2 * L + T.adds) * T.absEval t0 +
            ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * 0 := herr
      _ ≤ gam R.u (m + 2 * L - 1) * ∑ k ∈ S, ‖t k‖ := by
          rw [ha, mul_zero, add_zero]
          exact mul_le_mul_of_nonneg_right hg hs0

variable {ι κ : Type*} [Fintype ι] [DecidableEq ι]

/-- `H_spec`: entry `(i, j)` is the sum of the exact amplitudes of the monomials `S i j`. -/
def specMat (S : ι → ι → Finset κ) (t : κ → 𝕂) : Matrix ι ι 𝕂 :=
  Matrix.of fun i j => ∑ k ∈ S i j, t k

/-- `|H|_abs`: entry `(i, j)` is the sum of the moduli of those amplitudes. -/
def absMat (S : ι → ι → Finset κ) (t : κ → 𝕂) : ι → ι → ℝ := fun i j => ∑ k ∈ S i j, ‖t k‖

/-- `construct.hermitian_part`: `fl(H_ij + conj H_ji) / 2`, the halving exact. -/
noncomputable def symFl (R : Rounding 𝕂) (H : Matrix ι ι 𝕂) : Matrix ι ι 𝕂 :=
  Matrix.of fun i j => R.rnd (H i j + star (H j i)) / 2

omit [Fintype ι] [DecidableEq ι] in
theorem absMat_nonneg (S : ι → ι → Finset κ) (t : κ → 𝕂) (i j : ι) : 0 ≤ absMat S t i j :=
  Finset.sum_nonneg fun _ _ => norm_nonneg _

omit [Fintype ι] [DecidableEq ι] in
theorem norm_specMat_le (S : ι → ι → Finset κ) (t : κ → 𝕂) (i j : ι) :
    ‖specMat S t i j‖ ≤ absMat S t i j := norm_sum_le _ _

theorem norm_star_eq (z : 𝕂) : ‖star z‖ = ‖z‖ := by
  rw [← starRingEnd_apply]
  exact RCLike.norm_conj z

omit [Fintype ι] [DecidableEq ι] in
/-- **Lemma A1** (matrix form). -/
theorem builder_error [DecidableEq κ] (R : Rounding 𝕂) (h0 : R.eta = 0) {m L : ℕ}
    (hW : ((m + 2 * L : ℕ) : ℝ) * R.u < 1) (t th : κ → 𝕂) (S : ι → ι → Finset κ)
    (hth : ∀ i j, ∀ k ∈ S i j, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) (Href : Matrix ι ι 𝕂)
    (hb : ∀ i j, EntryBuilt R m th (S i j) (Href i j)) (i j : ι) :
    ‖(Href - specMat S t) i j‖ ≤ gam R.u (m + 2 * L - 1) * absMat S t i j := by
  rw [Matrix.sub_apply]
  exact builder_entry_error R h0 hW t th (S i j) (hth i j) (hb i j)

/-- The rounding of one symmetrised entry. -/
theorem sym_entry_error (R : Rounding 𝕂) (h0 : R.eta = 0) {a : ℕ}
    (ha : ((a + 1 : ℕ) : ℝ) * R.u < 1) {x y hy : 𝕂} {Px Py : ℝ}
    (hxe : ‖x - star hy‖ ≤ gam R.u a * Px) (hye : ‖y - hy‖ ≤ gam R.u a * Py)
    (hhx : ‖star hy‖ ≤ Px) (hhy : ‖hy‖ ≤ Py) :
    ‖R.rnd (x + star y) / 2 - star hy‖ ≤ gam R.u (a + 1) / 2 * (Px + Py) := by
  have hu := R.u_nonneg
  have ha0 : (a : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : a ≤ a + 1)) hu) ha
  have g0 := gam_nonneg hu ha0
  have hs := gam_succ hu ha
  have hr := R.err (x + star y)
  rw [h0, add_zero] at hr
  have hxn : ‖x‖ ≤ (1 + gam R.u a) * Px := by
    have := norm_le_add_norm_sub x (star hy)
    nlinarith
  have hyn : ‖y‖ ≤ (1 + gam R.u a) * Py := by
    have := norm_le_add_norm_sub y hy
    nlinarith
  have hsn : ‖x + star y‖ ≤ (1 + gam R.u a) * (Px + Py) := by
    have := norm_add_le x (star y)
    rw [norm_star_eq] at this
    nlinarith
  have e : R.rnd (x + star y) / 2 - star hy =
      ((R.rnd (x + star y) - (x + star y)) + (x - star hy) + star (y - hy)) / 2 := by
    rw [star_sub]
    ring
  rw [e, norm_div, RCLike.norm_two]
  have h3 : ‖(R.rnd (x + star y) - (x + star y)) + (x - star hy) + star (y - hy)‖ ≤
      R.u * ((1 + gam R.u a) * (Px + Py)) + gam R.u a * Px + gam R.u a * Py := by
    refine (norm_add_le _ _).trans ?_
    rw [norm_star_eq]
    have := norm_add_le (R.rnd (x + star y) - (x + star y)) (x - star hy)
    have := hr.trans (mul_le_mul_of_nonneg_left hsn hu)
    linarith
  have hPP : 0 ≤ Px + Py := by linarith [norm_nonneg hy, norm_nonneg (star hy)]
  have h4 := mul_le_mul_of_nonneg_right hs hPP
  rw [div_le_iff₀ (by norm_num : (0 : ℝ) < 2)]
  nlinarith

omit [Fintype ι] [DecidableEq ι] in
/-- **Lemma A2** (symmetrisation): `|fl(Ĥ) - H_spec|_{ij} ≤ γ_{m+2L} ((|H|_abs)_{ij} + (|H|_abs)_{ji}) / 2
≤ γ_{m+2L} max((|H|_abs)_{ij}, (|H|_abs)_{ji})`. -/
theorem symmetrised_error [DecidableEq κ] (R : Rounding 𝕂) (h0 : R.eta = 0) {m L : ℕ}
    (hm : 1 ≤ m) (hW : ((m + 2 * L : ℕ) : ℝ) * R.u < 1) (t th : κ → 𝕂) (S : ι → ι → Finset κ)
    (hth : ∀ i j, ∀ k ∈ S i j, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) (Href : Matrix ι ι 𝕂)
    (hb : ∀ i j, EntryBuilt R m th (S i j) (Href i j)) (hH : (specMat S t).IsHermitian)
    (i j : ι) :
    ‖(symFl R Href - specMat S t) i j‖ ≤
      gam R.u (m + 2 * L) / 2 * (absMat S t i j + absMat S t j i) := by
  have e1 : m + 2 * L - 1 + 1 = m + 2 * L := by omega
  have hW' : ((m + 2 * L - 1 + 1 : ℕ) : ℝ) * R.u < 1 := by rw [e1]; exact hW
  have hsym : specMat S t i j = star (specMat S t j i) := (hH.apply i j).symm
  have hxe := builder_error R h0 hW t th S hth Href hb i j
  have hye := builder_error R h0 hW t th S hth Href hb j i
  rw [Matrix.sub_apply, hsym] at hxe ⊢
  rw [Matrix.sub_apply] at hye
  have hhx : ‖star (specMat S t j i)‖ ≤ absMat S t i j := by
    rw [← hsym]
    exact norm_specMat_le S t i j
  have := sym_entry_error R h0 hW' hxe hye hhx (norm_specMat_le S t j i)
  rw [e1] at this
  simpa [symFl] using this

theorem add_le_two_max (a b : ℝ) : (a + b) / 2 ≤ max a b := by
  have := le_max_left a b
  have := le_max_right a b
  linarith

/-- From the squared Schur bound to the operator bound `κ √(r c)`. -/
theorem le_of_sq_le_schur {A X κ r c : ℝ} (hA : 0 ≤ A) (hX : 0 ≤ X) (hκ : 0 ≤ κ) (hr : 0 ≤ r)
    (hc : 0 ≤ c) (h : A ^ 2 ≤ κ * r * (κ * c) * X ^ 2) : A ≤ κ * Real.sqrt (r * c) * X := by
  have hrc : 0 ≤ r * c := mul_nonneg hr hc
  have hB : 0 ≤ κ * Real.sqrt (r * c) * X := mul_nonneg (mul_nonneg hκ (Real.sqrt_nonneg _)) hX
  refine (sq_le_sq₀ hA hB).mp ?_
  rw [mul_pow, mul_pow, Real.sq_sqrt hrc]
  linarith [h]

/-- **Schur test for a symmetric entrywise bound**: `|M_ij| ≤ κ (P_ij + P_ji)` with row sums of `P`
at most `r` and column sums at most `c` gives `‖M‖₂ ≤ 2 κ √(r c)`. -/
theorem schur_split (M : Matrix ι ι 𝕂) (P : ι → ι → ℝ) (hP : ∀ i j, 0 ≤ P i j) {κ r c : ℝ}
    (hκ : 0 ≤ κ) (hr0 : 0 ≤ r) (hc0 : 0 ≤ c) (hM : ∀ i j, ‖M i j‖ ≤ κ * (P i j + P j i))
    (hr : ∀ i, ∑ j, P i j ≤ r) (hc : ∀ j, ∑ i, P i j ≤ c) (x : EuclideanSpace 𝕂 ι) :
    ‖toEuclideanLin M x‖ ≤ 2 * κ * Real.sqrt (r * c) * ‖x‖ := by
  classical
  let wt : ι → ι → ℝ := fun i j => if P i j + P j i = 0 then 0 else P i j / (P i j + P j i)
  let M1 : Matrix ι ι 𝕂 := Matrix.of fun i j => (wt i j : 𝕂) * M i j
  have hzero : ∀ i j, P i j + P j i = 0 → ‖M i j‖ = 0 := fun i j hs =>
    le_antisymm (by simpa [hs] using hM i j) (norm_nonneg _)
  have hw0 : ∀ i j, 0 ≤ wt i j := by
    intro i j
    simp only [wt]
    split_ifs
    · exact le_rfl
    · exact div_nonneg (hP i j) (add_nonneg (hP i j) (hP j i))
  have hw1 : ∀ i j, wt i j ≤ 1 := by
    intro i j
    simp only [wt]
    split_ifs with hs
    · exact zero_le_one
    · exact div_le_one_of_le₀ (le_add_of_nonneg_right (hP j i)) (add_nonneg (hP i j) (hP j i))
  have h1 : ∀ i j, ‖M1 i j‖ ≤ κ * P i j := by
    intro i j
    simp only [M1, Matrix.of_apply, norm_mul, RCLike.norm_ofReal, abs_of_nonneg (hw0 i j)]
    by_cases hs : P i j + P j i = 0
    · rw [hzero i j hs, mul_zero]
      exact mul_nonneg hκ (hP i j)
    · have hpos : 0 < P i j + P j i :=
        lt_of_le_of_ne (add_nonneg (hP i j) (hP j i)) (Ne.symm hs)
      simp only [wt, hs, ↓reduceIte]
      calc P i j / (P i j + P j i) * ‖M i j‖
          ≤ P i j / (P i j + P j i) * (κ * (P i j + P j i)) :=
            mul_le_mul_of_nonneg_left (hM i j) (div_nonneg (hP i j) hpos.le)
        _ = κ * P i j := by field_simp
  have h2 : ∀ i j, ‖(M - M1) i j‖ ≤ κ * P j i := by
    intro i j
    have e : (M - M1) i j = ((1 - wt i j : ℝ) : 𝕂) * M i j := by
      simp only [Matrix.sub_apply, M1, Matrix.of_apply]
      push_cast
      ring
    rw [e, norm_mul, RCLike.norm_ofReal, abs_of_nonneg (by linarith [hw1 i j])]
    by_cases hs : P i j + P j i = 0
    · rw [hzero i j hs, mul_zero]
      exact mul_nonneg hκ (hP j i)
    · have hpos : 0 < P i j + P j i :=
        lt_of_le_of_ne (add_nonneg (hP i j) (hP j i)) (Ne.symm hs)
      simp only [wt, hs, ↓reduceIte]
      have h1w : 0 ≤ 1 - P i j / (P i j + P j i) := by
        have := div_le_one_of_le₀ (le_add_of_nonneg_right (hP j i)) hpos.le
        linarith
      calc (1 - P i j / (P i j + P j i)) * ‖M i j‖
          ≤ (1 - P i j / (P i j + P j i)) * (κ * (P i j + P j i)) :=
            mul_le_mul_of_nonneg_left (hM i j) h1w
        _ = κ * P j i := by field_simp; ring
  have s1 := schur_bound M1 (Rb := κ * r) (Cb := κ * c) (mul_nonneg hκ hr0)
    (fun i => (Finset.sum_le_sum fun j _ => h1 i j).trans
      (by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left (hr i) hκ))
    (fun j => (Finset.sum_le_sum fun i _ => h1 i j).trans
      (by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left (hc j) hκ)) x
  have s2 := schur_bound (M - M1) (Rb := κ * c) (Cb := κ * r) (mul_nonneg hκ hc0)
    (fun i => (Finset.sum_le_sum fun j _ => h2 i j).trans
      (by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left (hc i) hκ))
    (fun j => (Finset.sum_le_sum fun i _ => h2 i j).trans
      (by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left (hr j) hκ)) x
  have n1 := le_of_sq_le_schur (norm_nonneg _) (norm_nonneg x) hκ hr0 hc0 s1
  have n2 := le_of_sq_le_schur (norm_nonneg _) (norm_nonneg x) hκ hc0 hr0 s2
  rw [mul_comm c r] at n2
  have e : toEuclideanLin M x = toEuclideanLin M1 x + toEuclideanLin (M - M1) x := by
    rw [← LinearMap.add_apply, ← map_add, add_sub_cancel]
  rw [e]
  calc ‖toEuclideanLin M1 x + toEuclideanLin (M - M1) x‖
      ≤ ‖toEuclideanLin M1 x‖ + ‖toEuclideanLin (M - M1) x‖ := norm_add_le _ _
    _ ≤ κ * Real.sqrt (r * c) * ‖x‖ + κ * Real.sqrt (r * c) * ‖x‖ := add_le_add n1 n2
    _ = 2 * κ * Real.sqrt (r * c) * ‖x‖ := by ring

/-- **Lemma A3 with Lemma A1: `Δ_builder = γ_W √(r c)` bounds `‖H_ref - H_spec‖₂`**, where `r` and `c`
bound the row and column sums of `|H|_abs` and `W ≥ m + 2 L`. -/
theorem builder_shift [DecidableEq κ] (R : Rounding 𝕂) (h0 : R.eta = 0) {m L W : ℕ}
    (hmW : m + 2 * L ≤ W) (hW : (W : ℝ) * R.u < 1) (t th : κ → 𝕂) (S : ι → ι → Finset κ)
    (hth : ∀ i j, ∀ k ∈ S i j, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) (Href : Matrix ι ι 𝕂)
    (hb : ∀ i j, EntryBuilt R m th (S i j) (Href i j)) {r c : ℝ} (hr0 : 0 ≤ r) (hc0 : 0 ≤ c)
    (hr : ∀ i, ∑ j, absMat S t i j ≤ r) (hc : ∀ j, ∑ i, absMat S t i j ≤ c)
    (x : EuclideanSpace 𝕂 ι) :
    ‖toEuclideanLin (Href - specMat S t) x‖ ≤ gam R.u W * Real.sqrt (r * c) * ‖x‖ := by
  have hu := R.u_nonneg
  have hW2 : ((m + 2 * L : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast hmW) hu) hW
  have hW1 : ((m + 2 * L - 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : m + 2 * L - 1 ≤ m + 2 * L)) hu) hW2
  have hg0 := gam_nonneg hu hW1
  have hg : gam R.u (m + 2 * L - 1) ≤ gam R.u W := gam_mono hu (by omega) hW
  have hE := builder_error R h0 hW2 t th S hth Href hb
  have s := schur_bound (Href - specMat S t) (Rb := gam R.u (m + 2 * L - 1) * r)
    (Cb := gam R.u (m + 2 * L - 1) * c) (mul_nonneg hg0 hr0)
    (fun i => (Finset.sum_le_sum fun j _ => hE i j).trans
      (by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left (hr i) hg0))
    (fun j => (Finset.sum_le_sum fun i _ => hE i j).trans
      (by rw [← Finset.mul_sum]; exact mul_le_mul_of_nonneg_left (hc j) hg0)) x
  have n := le_of_sq_le_schur (norm_nonneg _) (norm_nonneg x) hg0 hr0 hc0 s
  exact n.trans (mul_le_mul_of_nonneg_right
    (mul_le_mul_of_nonneg_right hg (Real.sqrt_nonneg _)) (norm_nonneg x))

/-- **Lemma A3 for the symmetrised build**: `Δ_builder` also bounds `‖fl(Ĥ) - H_spec‖₂`. -/
theorem symmetrised_shift [DecidableEq κ] (R : Rounding 𝕂) (h0 : R.eta = 0) {m L W : ℕ}
    (hm : 1 ≤ m) (hmW : m + 2 * L ≤ W) (hW : (W : ℝ) * R.u < 1) (t th : κ → 𝕂)
    (S : ι → ι → Finset κ) (hth : ∀ i j, ∀ k ∈ S i j, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖)
    (Href : Matrix ι ι 𝕂) (hb : ∀ i j, EntryBuilt R m th (S i j) (Href i j))
    (hH : (specMat S t).IsHermitian) {r c : ℝ} (hr0 : 0 ≤ r) (hc0 : 0 ≤ c)
    (hr : ∀ i, ∑ j, absMat S t i j ≤ r) (hc : ∀ j, ∑ i, absMat S t i j ≤ c)
    (x : EuclideanSpace 𝕂 ι) :
    ‖toEuclideanLin (symFl R Href - specMat S t) x‖ ≤ gam R.u W * Real.sqrt (r * c) * ‖x‖ := by
  have hu := R.u_nonneg
  have hW2 : ((m + 2 * L : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast hmW) hu) hW
  have hg0 := gam_nonneg hu hW2
  have hg : gam R.u (m + 2 * L) ≤ gam R.u W := gam_mono hu hmW hW
  have hE := symmetrised_error R h0 hm hW2 t th S hth Href hb hH
  have n := schur_split (symFl R Href - specMat S t) (absMat S t) (absMat_nonneg S t)
    (div_nonneg hg0 (by norm_num : (0 : ℝ) ≤ 2)) hr0 hc0 hE hr hc x
  calc ‖toEuclideanLin (symFl R Href - specMat S t) x‖
      ≤ 2 * (gam R.u (m + 2 * L) / 2) * Real.sqrt (r * c) * ‖x‖ := n
    _ = gam R.u (m + 2 * L) * Real.sqrt (r * c) * ‖x‖ := by ring
    _ ≤ gam R.u W * Real.sqrt (r * c) * ‖x‖ := mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right hg (Real.sqrt_nonneg _)) (norm_nonneg x)

/-- **Theorem A4** (the model shift), with `Δ_builder` derived from the builder model:
`‖H_h - H_spec‖₂ ≤ max(‖D‖₁, ‖D‖_∞) + γ_W √(r c)` for `D = H_h - H_ref`. -/
theorem model_shift_fl [DecidableEq κ] (R : Rounding 𝕂) (h0 : R.eta = 0) {m L W : ℕ}
    (hmW : m + 2 * L ≤ W) (hW : (W : ℝ) * R.u < 1) (t th : κ → 𝕂) (S : ι → ι → Finset κ)
    (hth : ∀ i j, ∀ k ∈ S i j, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) (Href Hh : Matrix ι ι 𝕂)
    (hb : ∀ i j, EntryBuilt R m th (S i j) (Href i j)) {r c Rb Cb : ℝ} (hr0 : 0 ≤ r)
    (hc0 : 0 ≤ c) (hr : ∀ i, ∑ j, absMat S t i j ≤ r) (hc : ∀ j, ∑ i, absMat S t i j ≤ c)
    (hR0 : 0 ≤ Rb) (hC0 : 0 ≤ Cb) (hR : ∀ i, ∑ j, ‖(Hh - Href) i j‖ ≤ Rb)
    (hC : ∀ j, ∑ i, ‖(Hh - Href) i j‖ ≤ Cb) (x : EuclideanSpace 𝕂 ι) :
    ‖toEuclideanLin (Hh - specMat S t) x‖ ≤
      (max Rb Cb + gam R.u W * Real.sqrt (r * c)) * ‖x‖ :=
  model_shift_bound Hh Href (specMat S t) hR0 hC0 hR hC
    (fun v => builder_shift R h0 hmW hW t th S hth Href hb hr0 hc0 hr hc v) x

/-- **Theorems A4 and A5 together**: an enclosure for the certified operator `H_h` with radius
`r_c` is an enclosure for `H_spec` with radius `r_c + max(‖D‖₁, ‖D‖_∞) + γ_W √(r c)`, for the same
sorted indices. -/
theorem weyl_transfer_fl [DecidableEq κ] (R : Rounding 𝕂) (h0 : R.eta = 0) {m L W : ℕ}
    (hmW : m + 2 * L ≤ W) (hW : (W : ℝ) * R.u < 1) (t th : κ → 𝕂) (S : ι → ι → Finset κ)
    (hth : ∀ i j, ∀ k ∈ S i j, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) (Href Hh : Matrix ι ι 𝕂)
    (hb : ∀ i j, EntryBuilt R m th (S i j) (Href i j)) (hh : Hh.IsHermitian)
    (hs : (specMat S t).IsHermitian) {r c Rb Cb : ℝ} (hr0 : 0 ≤ r)
    (hc0 : 0 ≤ c) (hr : ∀ i, ∑ j, absMat S t i j ≤ r) (hc : ∀ j, ∑ i, absMat S t i j ≤ c)
    (hR0 : 0 ≤ Rb) (hC0 : 0 ≤ Cb) (hR : ∀ i, ∑ j, ‖(Hh - Href) i j‖ ≤ Rb)
    (hC : ∀ j, ∑ i, ‖(Hh - Href) i j‖ ≤ Cb) {ν : Type*} (f : ν → Fin (Fintype.card ι))
    (μ rc : ν → ℝ) (hf : ∀ k, |hh.eigenvalues₀ (f k) - μ k| ≤ rc k) (k : ν) :
    |hs.eigenvalues₀ (f k) - μ k| ≤ rc k + (max Rb Cb + gam R.u W * Real.sqrt (r * c)) := by
  refine weyl_transfer_enclosure hh hs (fun v => ?_) f μ rc hf k
  have e : specMat S t - Hh = -(Hh - specMat S t) := by abel
  rw [e, map_neg, LinearMap.neg_apply, norm_neg]
  exact model_shift_fl R h0 hmW hW t th S hth Href Hh hb hr0 hc0 hr hc hR0 hC0 hR hC v

end Builder

/-! ### Theorem 1 in floating point for complex data -/

section ComplexData

theorem abs_add_le' (a b : ℝ) : |a + b| ≤ |a| + |b| :=
  abs_le.mpr ⟨by linarith [neg_abs_le a, neg_abs_le b], by linarith [le_abs_self a, le_abs_self b]⟩

/-- Componentwise rounding of a complex result: a complex sum or difference, or a complex number
times a real one. -/
noncomputable def crnd (R : Rounding ℝ) (z : ℂ) : ℂ := ⟨R.rnd z.re, R.rnd z.im⟩

theorem crnd_err (R : Rounding ℝ) (h0 : R.eta = 0) (z : ℂ) : ‖crnd R z - z‖ ≤ R.u * ‖z‖ := by
  have hu := R.u_nonneg
  have h1 := pow_le_pow_left₀ (abs_nonneg _) (R.abs_rnd_sub_le h0 z.re) 2
  have h2 := pow_le_pow_left₀ (abs_nonneg _) (R.abs_rnd_sub_le h0 z.im) 2
  rw [sq_abs, mul_pow, sq_abs] at h1 h2
  refine (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hu (norm_nonneg _))).mp ?_
  rw [mul_pow, Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply, Complex.normSq_apply]
  simp only [crnd, Complex.sub_re, Complex.sub_im]
  nlinarith

/-- The componentwise rounding as a rounding on `ℂ` (standard model). -/
noncomputable def cRounding (R : Rounding ℝ) (h0 : R.eta = 0) : Rounding ℂ where
  rnd := crnd R
  IsFloat := fun _ => True
  isFloat_rnd := fun _ => trivial
  u := R.u
  eta := 0
  u_nonneg := R.u_nonneg
  eta_nonneg := le_rfl
  err := fun v => by rw [add_zero]; exact crnd_err R h0 v
  add_err := fun a b _ _ => crnd_err R h0 (a + b)

/-- `fl(fl(p) ± fl(q))` against `p ± q` (`ε = ±1`). -/
theorem pair_err (R : Rounding ℝ) (h0 : R.eta = 0) (p q ε : ℝ) (hε : |ε| = 1) :
    |R.rnd (R.rnd p + ε * R.rnd q) - (p + ε * q)| ≤ (2 * R.u + R.u ^ 2) * (|p| + |q|) := by
  have hu := R.u_nonneg
  have hp := R.abs_rnd_sub_le h0 p
  have hq := R.abs_rnd_sub_le h0 q
  have hd := R.abs_rnd_sub_le h0 (R.rnd p + ε * R.rnd q)
  have h1 : |(R.rnd p + ε * R.rnd q) - (p + ε * q)| ≤ R.u * (|p| + |q|) := by
    have e : (R.rnd p + ε * R.rnd q) - (p + ε * q) = (R.rnd p - p) + ε * (R.rnd q - q) := by ring
    rw [e]
    refine (abs_add_le' _ _).trans ?_
    rw [abs_mul, hε, one_mul]
    linarith
  have h2 : |R.rnd p + ε * R.rnd q| ≤ (1 + R.u) * (|p| + |q|) := by
    have e : R.rnd p + ε * R.rnd q = (p + ε * q) + ((R.rnd p + ε * R.rnd q) - (p + ε * q)) := by
      ring
    rw [e]
    refine (abs_add_le' _ _).trans ?_
    have : |p + ε * q| ≤ |p| + |q| := by
      refine (abs_add_le' _ _).trans ?_
      rw [abs_mul, hε, one_mul]
    linarith
  have e : R.rnd (R.rnd p + ε * R.rnd q) - (p + ε * q) =
      (R.rnd (R.rnd p + ε * R.rnd q) - (R.rnd p + ε * R.rnd q)) +
        ((R.rnd p + ε * R.rnd q) - (p + ε * q)) := by ring
  rw [e]
  refine (abs_add_le' _ _).trans ?_
  have := mul_le_mul_of_nonneg_left h2 hu
  nlinarith

/-- A complex product from real operations, `(ac - bd) + i(ad + bc)`: four products and two sums
rounded (no fused multiply-add). -/
noncomputable def cmul (R : Rounding ℝ) (a b : ℂ) : ℂ :=
  ⟨R.rnd (R.rnd (a.re * b.re) - R.rnd (a.im * b.im)),
    R.rnd (R.rnd (a.re * b.im) + R.rnd (a.im * b.re))⟩

theorem sq_pair_bound {α β γ δ : ℝ} (hγ : 0 ≤ γ) (hδ : 0 ≤ δ) :
    (α * γ + β * δ) ^ 2 + (α * δ + β * γ) ^ 2 ≤ 2 * ((α ^ 2 + β ^ 2) * (γ ^ 2 + δ ^ 2)) := by
  have k1 : 2 * (α * β) ≤ α ^ 2 + β ^ 2 := by nlinarith [sq_nonneg (α - β)]
  have k2 : 2 * (γ * δ) ≤ γ ^ 2 + δ ^ 2 := by nlinarith [sq_nonneg (γ - δ)]
  have k3 := mul_le_mul k1 k2 (by positivity) (by positivity)
  linarith

theorem pair_err_sq (R : Rounding ℝ) (h0 : R.eta = 0) (p q ε : ℝ) (hε : |ε| = 1) :
    (R.rnd (R.rnd p + ε * R.rnd q) - (p + ε * q)) ^ 2 ≤
      (2 * R.u + R.u ^ 2) ^ 2 * (|p| + |q|) ^ 2 := by
  rw [← mul_pow, ← sq_abs (R.rnd (R.rnd p + ε * R.rnd q) - (p + ε * q))]
  exact pow_le_pow_left₀ (abs_nonneg _) (pair_err R h0 p q ε hε) 2

/-- **Complex product error** (Higham 2002, Lemma 3.5, in the form `√2 γ₂ ≤ γ₃`). -/
theorem cmul_err (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) (a b : ℂ) :
    ‖cmul R a b - a * b‖ ≤ gam R.u 3 * ‖a * b‖ := by
  have hu := R.u_nonneg
  have hre := pair_err_sq R h0 (a.re * b.re) (a.im * b.im) (-1) (by norm_num)
  have him := pair_err_sq R h0 (a.re * b.im) (a.im * b.re) 1 (by norm_num)
  have ere : (cmul R a b - a * b).re = R.rnd (R.rnd (a.re * b.re) + -1 * R.rnd (a.im * b.im)) -
      (a.re * b.re + -1 * (a.im * b.im)) := by
    simp only [cmul, Complex.sub_re, Complex.mul_re]
    ring_nf
  have eim : (cmul R a b - a * b).im = R.rnd (R.rnd (a.re * b.im) + 1 * R.rnd (a.im * b.re)) -
      (a.re * b.im + 1 * (a.im * b.re)) := by
    simp only [cmul, Complex.sub_im, Complex.mul_im]
    ring_nf
  rw [← ere, abs_mul, abs_mul] at hre
  rw [← eim, abs_mul, abs_mul] at him
  have hsum := sq_pair_bound (α := |a.re|) (β := |a.im|) (abs_nonneg b.re) (abs_nonneg b.im)
  rw [sq_abs, sq_abs, sq_abs, sq_abs] at hsum
  have hc2 : 0 ≤ (2 * R.u + R.u ^ 2) ^ 2 := sq_nonneg _
  have hmul := mul_le_mul_of_nonneg_left hsum hc2
  have hnorm : ‖cmul R a b - a * b‖ ^ 2 ≤
      (2 * R.u + R.u ^ 2) ^ 2 * (2 * ((a.re ^ 2 + a.im ^ 2) * (b.re ^ 2 + b.im ^ 2))) := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    linarith
  have hab : ‖a * b‖ ^ 2 = (a.re ^ 2 + a.im ^ 2) * (b.re ^ 2 + b.im ^ 2) := by
    rw [norm_mul, mul_pow, Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply,
      Complex.normSq_apply]
    ring
  have h3 : R.u ^ 3 ≤ R.u ^ 2 / 1000 := by
    have := mul_le_mul_of_nonneg_left hu1 (sq_nonneg R.u)
    nlinarith
  have h4 : R.u ^ 4 ≤ R.u ^ 2 / 1000 := by
    have := mul_le_mul_of_nonneg_left h3 hu
    nlinarith [mul_le_mul_of_nonneg_left hu1 (sq_nonneg R.u)]
  have hc3 : (2 * R.u + R.u ^ 2) ^ 2 * 2 ≤ (3 * R.u) ^ 2 := by
    have e : (2 * R.u + R.u ^ 2) ^ 2 * 2 = 8 * R.u ^ 2 + 8 * R.u ^ 3 + 2 * R.u ^ 4 := by ring
    rw [e]
    nlinarith
  have hP : 0 ≤ (a.re ^ 2 + a.im ^ 2) * (b.re ^ 2 + b.im ^ 2) := by positivity
  have hfin : ‖cmul R a b - a * b‖ ≤ 3 * R.u * ‖a * b‖ := by
    refine (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (by linarith) (norm_nonneg _))).mp ?_
    rw [mul_pow, hab]
    calc ‖cmul R a b - a * b‖ ^ 2
        ≤ (2 * R.u + R.u ^ 2) ^ 2 * (2 * ((a.re ^ 2 + a.im ^ 2) * (b.re ^ 2 + b.im ^ 2))) := hnorm
      _ = ((2 * R.u + R.u ^ 2) ^ 2 * 2) * ((a.re ^ 2 + a.im ^ 2) * (b.re ^ 2 + b.im ^ 2)) := by
          ring
      _ ≤ (3 * R.u) ^ 2 * ((a.re ^ 2 + a.im ^ 2) * (b.re ^ 2 + b.im ^ 2)) :=
          mul_le_mul_of_nonneg_right hc3 hP
  have h3u : 3 * R.u ≤ gam R.u 3 := by
    unfold gam
    push_cast
    rw [le_div_iff₀ (by linarith)]
    nlinarith
  exact hfin.trans (mul_le_mul_of_nonneg_right h3u (norm_nonneg _))

/-- The real components of a complex vector, `2n` of them. -/
def comps {ι : Type*} (v : ι → ℂ) : ι ⊕ ι → ℝ := Sum.elim (fun i => (v i).re) (fun i => (v i).im)

variable {ι : Type*} [Fintype ι]

/-- The Euclidean norm of a complex vector given by its entries. -/
noncomputable def l2c (v : ι → ℂ) : ℝ := ‖(WithLp.toLp 2 v : EuclideanSpace ℂ ι)‖

theorem l2c_eq_sqrt (v : ι → ℂ) : l2c v = Real.sqrt (∑ i, ‖v i‖ ^ 2) := by
  rw [l2c, EuclideanSpace.norm_eq]

theorem l2c_eq (v : ι → ℂ) : l2c v = l2 (comps v) := by
  rw [l2c_eq_sqrt, l2_eq, Fintype.sum_sum_type, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Complex.sq_norm, Complex.normSq_apply]
  simp only [comps, Sum.elim_inl, Sum.elim_inr]
  ring

theorem l2c_le_add (a b : ι → ℂ) : l2c a ≤ l2c b + l2c (fun i => a i - b i) := by
  have e : (WithLp.toLp 2 a : EuclideanSpace ℂ ι) =
      WithLp.toLp 2 b + WithLp.toLp 2 (fun i => a i - b i) := by
    refine PiLp.ext fun k => ?_
    simp
  unfold l2c
  rw [e]
  exact norm_add_le _ _

theorem l2c_le_of_norm_le {a : ι → ℂ} {b : ι → ℝ} (h : ∀ i, ‖a i‖ ≤ b i) : l2c a ≤ l2 b := by
  rw [l2c_eq_sqrt, l2_eq]
  apply Real.sqrt_le_sqrt
  exact Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (norm_nonneg _) (h i) 2

theorem l2c_ofLp (x : EuclideanSpace ℂ ι) : l2c (fun i => x i) = ‖x‖ := by
  rw [l2c_eq_sqrt, EuclideanSpace.norm_eq]

variable [DecidableEq ι]

theorem l2c_residual (A : Matrix ι ι ℂ) (x : EuclideanSpace ℂ ι) (μ : ℝ) :
    l2c (fun i => ∑ j, A i j * x j - (μ : ℂ) * x i) = ‖toEuclideanLin A x - (μ : ℂ) • x‖ := by
  rw [l2c_eq_sqrt, EuclideanSpace.norm_eq]
  refine congrArg Real.sqrt (Finset.sum_congr rfl fun i _ => ?_)
  have hc : (toEuclideanLin A x) i = ∑ j, A i j * x j := by
    rw [Matrix.toLpLin_apply]
    rfl
  rw [PiLp.sub_apply, PiLp.smul_apply, hc, smul_eq_mul]

/-- Row summation trees for a complex matrix. -/
def RowTreesC (T : ι → STree ι) (A : ι → ι → ℂ) (w : ℕ) : Prop :=
  ∀ i, (T i).leaves.Nodup ∧ (T i).adds + 1 ≤ w ∧ ∀ j, j ∉ (T i).leaves → A i j = 0

/-- `adapter.matvec`, complex data: complex products `cmul`, complex sums rounded
componentwise. -/
noncomputable def matvecC (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℂ) (x : ι → ℂ)
    (i : ι) : ℂ :=
  (T i).comp (crnd R) (fun j => cmul R (A i j) (x j))

/-- The computed residual, complex data: `fl(ŷ_i - fl(x_i μ))`. -/
noncomputable def residC (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℂ) (x : ι → ℂ) (μ : ℝ)
    (i : ι) : ℂ :=
  crnd R (matvecC R T A x i - crnd R (x i * (μ : ℂ)))

/-- `adapter.abs_matvec`, complex data: the moduli rounded once (`hypot`), then the product. -/
noncomputable def absMatvecC (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℂ) (x : ι → ℂ)
    (i : ι) : ℝ :=
  (T i).comp R.rnd (fun j => R.rnd (R.rnd ‖A i j‖ * R.rnd ‖x j‖))

/-- `_rounding_bound`, complex data:
`2γ̂_{w+6} (|A| |x| / (1 - γ̂_{w+3}) (1 + 4u) + |x| |μ|) (1 + 4u)`, operation by operation. -/
noncomputable def roundBoundC (R : Rounding ℝ) (T : ι → STree ι) (w : ℕ) (A : ι → ι → ℂ)
    (x : ι → ℂ) (μ : ℝ) (i : ι) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (absMatvecC R T A x i / R.rnd (1 - gamHat R (w + 3))) *
    (1 + 4 * R.u)) + R.rnd (R.rnd ‖x i‖ * |μ|)) * R.rnd (2 * gamHat R (w + 6))) * (1 + 4 * R.u))

/-- `fp.norm2_upper` for a complex vector: `np.linalg.norm` sums the `2n` squares of the
components, the factor is still `1 + γ̂_{n+2}`. -/
noncomputable def norm2UpC (R : Rounding ℝ) (T : STree (ι ⊕ ι)) (tail : ℝ) (v : ι → ℂ) : ℝ :=
  R.rnd (R.rnd (R.rnd (normFl R T (comps v) * R.rnd (1 + gamHat R (Fintype.card ι + 2))) *
    (1 + 8 * R.u)) + tail)

/-- `fp.norm2_lower` for a complex vector. -/
noncomputable def norm2LowC (R : Rounding ℝ) (T : STree (ι ⊕ ι)) (tail : ℝ) (v : ι → ℂ) : ℝ :=
  max 0 (R.rnd (R.rnd (R.rnd (normFl R T (comps v) * R.rnd (1 - gamHat R (Fintype.card ι + 2))) /
    (1 + 8 * R.u)) - tail))

/-- The computed relative radius of `residuals`, complex data. -/
noncomputable def residualRadiusFlC (R : Rounding ℝ) (T T' : ι → STree ι) (TR TX : STree (ι ⊕ ι))
    (TE : STree ι) (w : ℕ) (A : ι → ι → ℂ) (x : ι → ℂ) (μ tR tE tX c1 c0 : ℝ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (norm2UpC R TR tR (residC R T A x μ) +
    norm2Up R TE tE (roundBoundC R T' w A x μ)) + c1) / norm2LowC R TX tX x) *
      (1 + 4 * R.u)) + c0)

theorem sqrt_one_sub_ge {x : ℝ} (hx0 : 0 ≤ x) (hx : x ≤ 1 / 2) :
    1 - x / 2 - x ^ 2 / 4 ≤ Real.sqrt (1 - x) := by
  have hy : 0 ≤ 1 - x / 2 - x ^ 2 / 4 := by nlinarith
  have h : (1 - x / 2 - x ^ 2 / 4) ^ 2 ≤ 1 - x := by
    nlinarith [mul_nonneg (mul_nonneg hx0 hx0) hx0, mul_nonneg (mul_nonneg hx0 hx0) (by linarith : (0:ℝ) ≤ 1 / 2 - x)]
  have := Real.sqrt_le_sqrt h
  rwa [Real.sqrt_sq hy] at this

theorem sqrt_one_add_le {x : ℝ} (hx0 : 0 ≤ x) : Real.sqrt (1 + x) ≤ 1 + x / 2 := by
  have h : 1 + x ≤ (1 + x / 2) ^ 2 := by nlinarith
  have := Real.sqrt_le_sqrt h
  rwa [Real.sqrt_sq (by linarith)] at this

/-- `fl(‖v‖)` against `‖v‖ √(1 ∓ γ_N)` (`N` the number of real components). -/
theorem normFl_sqrt_bounds {α : Type*} [Fintype α] [DecidableEq α] (R : Rounding ℝ)
    (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {T : STree α} (hT : FullTree T)
    (hn : (Fintype.card α : ℝ) * R.u < 1) (v : α → ℝ) :
    l2 v * Real.sqrt (1 - gam R.u (Fintype.card α)) * (1 - R.u) ≤ normFl R T v ∧
      normFl R T v ≤ l2 v * Real.sqrt (1 + gam R.u (Fintype.card α)) * (1 + R.u) := by
  have hu := R.u_nonneg
  have hg0 : 0 ≤ gam R.u (Fintype.card α) := gam_nonneg hu hn
  have herr := sum_sq_error R h0 hT hn v
  have hs : 0 ≤ ∑ j, v j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg (v j)
  obtain ⟨hlo, hhi⟩ := abs_le.mp herr
  have hl2 := l2_eq v
  have hsq_lo : l2 v * Real.sqrt (1 - gam R.u (Fintype.card α)) ≤
      Real.sqrt (T.comp R.rnd (fun j => R.rnd (v j * v j))) := by
    have h1 : (∑ j, v j ^ 2) * (1 - gam R.u (Fintype.card α)) ≤
        T.comp R.rnd (fun j => R.rnd (v j * v j)) := by nlinarith
    have h2 := Real.sqrt_le_sqrt h1
    rw [Real.sqrt_mul hs] at h2
    rw [hl2]
    exact h2
  have hsq_hi : Real.sqrt (T.comp R.rnd (fun j => R.rnd (v j * v j))) ≤
      l2 v * Real.sqrt (1 + gam R.u (Fintype.card α)) := by
    have h1 : T.comp R.rnd (fun j => R.rnd (v j * v j)) ≤
        (∑ j, v j ^ 2) * (1 + gam R.u (Fintype.card α)) := by nlinarith
    have h2 := Real.sqrt_le_sqrt h1
    rw [Real.sqrt_mul hs] at h2
    rw [hl2]
    exact h2
  exact ⟨R.le_rnd_of_le h0 hu1 hsq_lo (mul_nonneg (l2_nonneg v) (Real.sqrt_nonneg _)),
    R.rnd_le_of_le h0 hu1 hsq_hi (mul_nonneg (l2_nonneg v) (Real.sqrt_nonneg _))⟩

/-- Constants of a complex vector of length `n ≤ 2²⁶`: with `t = n u`, `t² ≤ u / 2`,
`γ_{2n} ≤ 2t + 6t²` and `γ_{n+2} ≥ t + 2u`. -/
theorem complex_norm_consts {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {n : ℕ}
    (hn : n ≤ 2 ^ 26) :
    (n : ℝ) * u ≤ 1 / 10 ^ 7 ∧ ((n : ℝ) * u) ^ 2 ≤ u / 2 ∧
      gam u (n + n) ≤ 2 * ((n : ℝ) * u) + 6 * ((n : ℝ) * u) ^ 2 ∧ 0 ≤ gam u (n + n) ∧
      (n : ℝ) * u + 2 * u ≤ gam u (n + 2) := by
  have hnr : (n : ℝ) ≤ 2 ^ 26 := by exact_mod_cast hn
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have h1 : (n : ℝ) * u ≤ 2 ^ 26 * (2 : ℝ)⁻¹ ^ 53 := mul_le_mul hnr hu53 hu (by norm_num)
  have ht : (n : ℝ) * u ≤ 1 / 10 ^ 7 := h1.trans (by norm_num)
  have ht0 : 0 ≤ (n : ℝ) * u := mul_nonneg hn0 hu
  have h2 : (n : ℝ) * ((n : ℝ) * u) ≤ 2 ^ 26 * (2 ^ 26 * (2 : ℝ)⁻¹ ^ 53) :=
    mul_le_mul hnr h1 ht0 (by norm_num)
  have h3 : (2 : ℝ) ^ 26 * (2 ^ 26 * (2 : ℝ)⁻¹ ^ 53) = 1 / 2 := by norm_num
  have ht2 : ((n : ℝ) * u) ^ 2 ≤ u / 2 := by
    calc ((n : ℝ) * u) ^ 2 = (n : ℝ) * ((n : ℝ) * u) * u := by ring
      _ ≤ 1 / 2 * u := mul_le_mul_of_nonneg_right (h2.trans h3.le) hu
      _ = u / 2 := by ring
  have hnn : ((n + n : ℕ) : ℝ) * u = 2 * ((n : ℝ) * u) := by push_cast; ring
  have hd : 0 < 1 - 2 * ((n : ℝ) * u) := by linarith
  refine ⟨ht, ht2, ?_, ?_, ?_⟩
  · unfold gam
    rw [hnn, div_le_iff₀ hd]
    nlinarith [mul_nonneg ht0 (mul_nonneg ht0 ht0)]
  · unfold gam
    rw [hnn]
    exact div_nonneg (by linarith) hd.le
  · have hk : ((n + 2 : ℕ) : ℝ) * u = (n : ℝ) * u + 2 * u := by push_cast; ring
    have hd2 : 0 < 1 - ((n + 2 : ℕ) : ℝ) * u := by rw [hk]; nlinarith
    unfold gam
    rw [le_div_iff₀ hd2, hk]
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ (n : ℝ) * u + 2 * u)
      (by linarith : (0 : ℝ) ≤ (n : ℝ) * u + 2 * u)]

/-- The factor of `fp.norm2_upper` for complex data. -/
theorem num_norm_up_c {u t x G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (ht0 : 0 ≤ t)
    (ht : t ≤ 1 / 10 ^ 7) (ht2 : t ^ 2 ≤ u / 2) (hx0 : 0 ≤ x) (hx : x ≤ 2 * t + 6 * t ^ 2)
    (hG : t + 2 * u ≤ G) (hG1 : G ≤ 1 / 1000) :
    1 + u ≤ (1 - (x / 2 + x ^ 2 / 4)) * (1 + G * (1 - u) ^ 2) * (1 - u) ^ 5 * (1 + 8 * u) := by
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith
  have hx2 : x / 2 ≤ t + 3 * t ^ 2 := by linarith
  have hxx : x ^ 2 / 4 ≤ (11 / 10) * t ^ 2 := by
    have h1 : x / 2 ≤ t + 3 * t ^ 2 := hx2
    have h2 : 0 ≤ x / 2 := by linarith
    have h3 : (x / 2) ^ 2 ≤ (t + 3 * t ^ 2) ^ 2 := pow_le_pow_left₀ h2 h1 2
    have h4 : (t + 3 * t ^ 2) ^ 2 ≤ (11 / 10) * t ^ 2 := by
      have : t * t ≤ t * (1 / 10 ^ 7) := mul_le_mul_of_nonneg_left ht ht0
      nlinarith [mul_nonneg ht0 (sq_nonneg t), mul_nonneg (sq_nonneg t) (sq_nonneg t)]
    nlinarith
  have hA : 1 - t - (41 / 10) * t ^ 2 ≤ 1 - (x / 2 + x ^ 2 / 4) := by linarith
  have hA0 : 0 ≤ 1 - t - (41 / 10) * t ^ 2 := by nlinarith
  have hG0 : 0 ≤ G := by linarith
  have hB : 1 + t + 2 * u - 2 * u * t - 4 * u ^ 2 ≤ 1 + G * (1 - u) ^ 2 := by
    have h1 : (t + 2 * u) * (1 - 2 * u) ≤ G * (1 - u) ^ 2 :=
      mul_le_mul hG hsq (by linarith) hG0
    nlinarith
  have hB0 : 0 ≤ 1 + t + 2 * u - 2 * u * t - 4 * u ^ 2 := by nlinarith
  have hAB := mul_le_mul hA hB hB0 (by linarith)
  have hP : 1 - (6 / 10) * u ≤
      (1 - t - (41 / 10) * t ^ 2) * (1 + t + 2 * u - 2 * u * t - 4 * u ^ 2) := by
    have f2 : u * t ≤ u * (1 / 10 ^ 7) := mul_le_mul_of_nonneg_left ht hu
    have f3 : u * u ≤ u * (1 / 1000) := mul_le_mul_of_nonneg_left hu1 hu
    have f4 : t * t ^ 2 ≤ (1 / 10 ^ 7) * (u / 2) := mul_le_mul ht ht2 (sq_nonneg t) (by norm_num)
    have f5 : u * t ^ 2 ≤ u * (u / 2) := mul_le_mul_of_nonneg_left ht2 hu
    nlinarith [mul_nonneg hu (sq_nonneg t), mul_nonneg (sq_nonneg u) ht0,
      mul_nonneg (mul_nonneg hu ht0) (sq_nonneg t), mul_nonneg (sq_nonneg u) (sq_nonneg t)]
  have hQ : 1 + 2 * u ≤ (1 - u) ^ 5 * (1 + 8 * u) := by
    have h5 := one_sub_mul_le_pow hu (by linarith) 5
    push_cast at h5
    have : (1 - 5 * u) * (1 + 8 * u) ≤ (1 - u) ^ 5 * (1 + 8 * u) :=
      mul_le_mul_of_nonneg_right h5 (by linarith)
    nlinarith [mul_le_mul_of_nonneg_left hu1 hu]
  have hPQ := mul_le_mul (hP.trans hAB) hQ (by linarith) (by nlinarith)
  calc 1 + u ≤ (1 - (6 / 10) * u) * (1 + 2 * u) := by nlinarith [mul_le_mul_of_nonneg_left hu1 hu]
    _ ≤ (1 - (x / 2 + x ^ 2 / 4)) * (1 + G * (1 - u) ^ 2) * ((1 - u) ^ 5 * (1 + 8 * u)) := hPQ
    _ = (1 - (x / 2 + x ^ 2 / 4)) * (1 + G * (1 - u) ^ 2) * (1 - u) ^ 5 * (1 + 8 * u) := by ring

/-- The factor of `fp.norm2_lower` for complex data. -/
theorem num_norm_low_c {u t x G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (ht0 : 0 ≤ t)
    (ht : t ≤ 1 / 10 ^ 7) (ht2 : t ^ 2 ≤ u / 2) (hx0 : 0 ≤ x) (hx : x ≤ 2 * t + 6 * t ^ 2)
    (hG : t + 2 * u ≤ G) (hG1 : G ≤ 1 / 1000) :
    (1 + x / 2) * (1 - G * (1 - u) ^ 2) * (1 + u) ^ 5 ≤ (1 - u) * (1 + 8 * u) := by
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith
  have hsq1 : (1 - u) ^ 2 ≤ 1 := by nlinarith
  have hG0 : 0 ≤ G := by linarith
  have hC : 1 - G * (1 - u) ^ 2 ≤ 1 - (t + 2 * u) * (1 - 2 * u) := by
    have := mul_le_mul hG hsq (by linarith) hG0
    linarith
  have hC0 : 0 ≤ 1 - G * (1 - u) ^ 2 := by nlinarith [mul_le_mul_of_nonneg_left hsq1 hG0]
  have hX : (1 + x / 2) * (1 - G * (1 - u) ^ 2) ≤
      (1 + t + 3 * t ^ 2) * (1 - (t + 2 * u) * (1 - 2 * u)) :=
    mul_le_mul (by linarith) hC hC0 (by nlinarith)
  have hX2 : (1 + t + 3 * t ^ 2) * (1 - (t + 2 * u) * (1 - 2 * u)) ≤ 1 - (9 / 10) * u := by
    have f2 : u * t ≤ u * (1 / 10 ^ 7) := mul_le_mul_of_nonneg_left ht hu
    have f3 : u * u ≤ u * (1 / 1000) := mul_le_mul_of_nonneg_left hu1 hu
    have f4 : t * t ^ 2 ≤ (1 / 10 ^ 7) * (u / 2) := mul_le_mul ht ht2 (sq_nonneg t) (by norm_num)
    have f5 : u * t ^ 2 ≤ u * (u / 2) := mul_le_mul_of_nonneg_left ht2 hu
    have f6 : u * u * t ≤ u * u * (1 / 10 ^ 7) := mul_le_mul_of_nonneg_left ht (mul_nonneg hu hu)
    have f7 : u * u * t ^ 2 ≤ u * u * (u / 2) := mul_le_mul_of_nonneg_left ht2 (mul_nonneg hu hu)
    nlinarith [mul_nonneg hu (sq_nonneg t), mul_nonneg (sq_nonneg u) ht0,
      mul_nonneg (mul_nonneg hu ht0) (sq_nonneg t), mul_nonneg (sq_nonneg u) (sq_nonneg t)]
  have hX0 : 0 ≤ (1 + x / 2) * (1 - G * (1 - u) ^ 2) := mul_nonneg (by linarith) hC0
  have h5 := one_add_pow_five_le hu hu1
  calc (1 + x / 2) * (1 - G * (1 - u) ^ 2) * (1 + u) ^ 5 ≤ (1 - (9 / 10) * u) * (1 + 6 * u) :=
        mul_le_mul (hX.trans hX2) h5 (by positivity) (by linarith)
    _ ≤ (1 - u) * (1 + 8 * u) := by nlinarith [mul_le_mul_of_nonneg_left hu1 hu]

theorem norm2UpC_ge (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    (hn : Fintype.card ι ≤ 2 ^ 26) {T : STree (ι ⊕ ι)} (hT : FullTree T) {tail : ℝ}
    (ht : 0 ≤ tail) (v : ι → ℂ) : l2c v * (1 + R.u) ≤ norm2UpC R T tail v := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hk2, -, -, -, hG, -⟩ := small_consts hu hu53 hn
  obtain ⟨ht1, ht2, hx, hx0, hGt⟩ := complex_norm_consts hu hu53 hn
  have hu1' : R.u ≤ 1 := by linarith
  have hcard : Fintype.card (ι ⊕ ι) = Fintype.card ι + Fintype.card ι := Fintype.card_sum
  have hnu : (Fintype.card (ι ⊕ ι) : ℝ) * R.u < 1 := by
    rw [hcard]
    push_cast
    linarith
  have hx1 : gam R.u (Fintype.card (ι ⊕ ι)) ≤ 1 / 2 := by
    rw [hcard]
    nlinarith
  obtain ⟨hs_lo, -⟩ := normFl_sqrt_bounds R h0 hu1' hT hnu (comps v)
  rw [hcard] at hs_lo
  have hsq := sqrt_one_sub_ge hx0 (by rw [← hcard]; exact hx1)
  have hs' : l2c v * (1 - (gam R.u (Fintype.card ι + Fintype.card ι) / 2 +
      gam R.u (Fintype.card ι + Fintype.card ι) ^ 2 / 4)) * (1 - R.u) ≤ normFl R T (comps v) := by
    refine le_trans ?_ hs_lo
    rw [l2c_eq]
    apply mul_le_mul_of_nonneg_right _ (by linarith)
    apply mul_le_mul_of_nonneg_left _ (l2_nonneg _)
    linarith
  obtain ⟨hgh, -⟩ := gamHat_bounds R h0 (by linarith) (k := Fintype.card ι + 2) (by linarith)
  have hG0 : 0 ≤ gam R.u (Fintype.card ι + 2) := by nlinarith
  have hc := R.le_rnd_of_le h0 hu1' (w := 1 + gamHat R (Fintype.card ι + 2))
    (z := 1 + gam R.u (Fintype.card ι + 2) * (1 - R.u) ^ 2) (by linarith)
    (by linarith [mul_nonneg hG0 (sq_nonneg (1 - R.u))])
  have hl0 : 0 ≤ l2c v := norm_nonneg _
  exact R.chain_up h0 hu1' hl0 hs' hc (by nlinarith) hG0 ht
    (num_norm_up_c hu hu1 (by positivity) ht1 ht2 hx0 hx hGt hG)

theorem norm2LowC_le (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    (hn : Fintype.card ι ≤ 2 ^ 26) {T : STree (ι ⊕ ι)} (hT : FullTree T) {tail : ℝ}
    (ht : 0 ≤ tail) (v : ι → ℂ) : norm2LowC R T tail v ≤ l2c v * (1 - R.u) := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hk2, -, -, -, hG, -⟩ := small_consts hu hu53 hn
  obtain ⟨ht1, ht2, hx, hx0, hGt⟩ := complex_norm_consts hu hu53 hn
  have hu1' : R.u ≤ 1 := by linarith
  have hcard : Fintype.card (ι ⊕ ι) = Fintype.card ι + Fintype.card ι := Fintype.card_sum
  have hnu : (Fintype.card (ι ⊕ ι) : ℝ) * R.u < 1 := by
    rw [hcard]
    push_cast
    linarith
  have hx1 : gam R.u (Fintype.card (ι ⊕ ι)) ≤ 1 / 2 := by
    rw [hcard]
    nlinarith
  obtain ⟨hs_lo, hs_hi⟩ := normFl_sqrt_bounds R h0 hu1' hT hnu (comps v)
  rw [hcard] at hs_lo hs_hi
  have hsq := sqrt_one_add_le hx0
  have hs' : normFl R T (comps v) ≤ l2c v *
      (1 + gam R.u (Fintype.card ι + Fintype.card ι) / 2) * (1 + R.u) := by
    refine hs_hi.trans ?_
    rw [l2c_eq]
    apply mul_le_mul_of_nonneg_right _ (by linarith)
    exact mul_le_mul_of_nonneg_left hsq (l2_nonneg _)
  have hs0 : 0 ≤ normFl R T (comps v) :=
    (mul_nonneg (mul_nonneg (l2_nonneg _) (Real.sqrt_nonneg _)) (by linarith)).trans hs_lo
  obtain ⟨hgh, -⟩ := gamHat_bounds R h0 (by linarith) (k := Fintype.card ι + 2) (by linarith)
  have hG0 : 0 ≤ gam R.u (Fintype.card ι + 2) := by nlinarith
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hC0 : 0 ≤ 1 - gam R.u (Fintype.card ι + 2) * (1 - R.u) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsq1 hG0]
  have hc := R.rnd_le_of_le h0 hu1' (w := 1 - gamHat R (Fintype.card ι + 2))
    (z := 1 - gam R.u (Fintype.card ι + 2) * (1 - R.u) ^ 2) (by linarith) hC0
  exact R.chain_low h0 hu1' (norm_nonneg _) hs0 hs' hc (by linarith) hC0 ht
    (num_norm_low_c hu hu1 (by positivity) ht1 ht2 hx0 hx hGt hG)

namespace Rounding

variable (R : Rounding ℝ)

/-- The last three roundings of `_rounding_bound`, complex data (`|x_i|` and `2 γ̂` rounded
once more). -/
theorem chain_E_c (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {a q b2 qh gh G : ℝ} (ha : 0 ≤ a)
    (hq : 0 ≤ q) (hb2 : a ≤ b2) (hqh : q * (1 - R.u) ^ 2 ≤ qh) (hG0 : 0 ≤ G)
    (hgh : G * (1 - R.u) ^ 3 ≤ gh) :
    G * ((1 - R.u) ^ 8 * (1 + 4 * R.u)) * (a + q) ≤
      R.rnd (R.rnd (R.rnd (b2 + qh) * gh) * (1 + 4 * R.u)) := by
  have hu := R.u_nonneg
  have h1u : 0 ≤ 1 - R.u := by linarith
  have haq : 0 ≤ a + q := add_nonneg ha hq
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hz : (a + q) * (1 - R.u) ^ 2 ≤ b2 + qh := by
    nlinarith [mul_le_mul_of_nonneg_left hsq1 ha]
  have hb3 := R.le_rnd_of_le h0 hu1 hz (mul_nonneg haq (sq_nonneg _))
  have hG3 : 0 ≤ G * (1 - R.u) ^ 3 := mul_nonneg hG0 (pow_nonneg h1u 3)
  have hb4 := R.le_rnd_mul h0 hu1 (mul_nonneg (mul_nonneg haq (sq_nonneg _)) h1u) hG3 hb3 hgh
  have hb5 := R.le_rnd_mul h0 hu1
    (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg haq (sq_nonneg _)) h1u) hG3) h1u)
    (by linarith : (0 : ℝ) ≤ 1 + 4 * R.u) hb4 le_rfl
  calc G * ((1 - R.u) ^ 8 * (1 + 4 * R.u)) * (a + q)
      = (a + q) * (1 - R.u) ^ 2 * (1 - R.u) * (G * (1 - R.u) ^ 3) * (1 - R.u) *
          (1 + 4 * R.u) * (1 - R.u) := by ring
    _ ≤ _ := hb5

end Rounding

theorem resid_entry_c (R : Rounding ℝ) (h0 : R.eta = 0) {y yh xm : ℂ} {a q g : ℝ}
    (hy : ‖yh - y‖ ≤ g * a) (hya : ‖y‖ ≤ a) (hxm : ‖xm‖ ≤ q) :
    ‖crnd R (yh - crnd R xm) - (y - xm)‖ ≤ (g + R.u * (1 + g)) * a + (2 * R.u + R.u ^ 2) * q := by
  have hu := R.u_nonneg
  have hm := (crnd_err R h0 xm).trans (mul_le_mul_of_nonneg_left hxm hu)
  have hmn : ‖crnd R xm‖ ≤ q * (1 + R.u) := by
    have := norm_le_add_norm_sub (crnd R xm) xm
    nlinarith
  have hyn : ‖yh‖ ≤ a + g * a := by
    have := norm_le_add_norm_sub yh y
    linarith
  have hd : ‖yh - crnd R xm‖ ≤ a + g * a + q * (1 + R.u) :=
    (norm_sub_le _ _).trans (by linarith)
  have hr := (crnd_err R h0 (yh - crnd R xm)).trans (mul_le_mul_of_nonneg_left hd hu)
  have e : crnd R (yh - crnd R xm) - (y - xm) =
      (crnd R (yh - crnd R xm) - (yh - crnd R xm)) + (yh - y) - (crnd R xm - xm) := by ring
  rw [e]
  calc ‖(crnd R (yh - crnd R xm) - (yh - crnd R xm)) + (yh - y) - (crnd R xm - xm)‖
      ≤ ‖(crnd R (yh - crnd R xm) - (yh - crnd R xm)) + (yh - y)‖ + ‖crnd R xm - xm‖ :=
        norm_sub_le _ _
    _ ≤ ‖crnd R (yh - crnd R xm) - (yh - crnd R xm)‖ + ‖yh - y‖ + ‖crnd R xm - xm‖ := by
        linarith [norm_add_le (crnd R (yh - crnd R xm) - (yh - crnd R xm)) (yh - y)]
    _ ≤ (g + R.u * (1 + g)) * a + (2 * R.u + R.u ^ 2) * q := by nlinarith

theorem matvecC_error (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000)
    {T : ι → STree ι} {A : ι → ι → ℂ} {w : ℕ} (hT : RowTreesC T A w)
    (hw : ((w + 2 : ℕ) : ℝ) * R.u < 1) (x : ι → ℂ) (i : ι) :
    ‖matvecC R T A x i - ∑ j, A i j * x j‖ ≤ gam R.u (w + 2) * ∑ j, ‖A i j‖ * ‖x j‖ := by
  obtain ⟨hnd, hlen, hsupp⟩ := hT i
  have hu := R.u_nonneg
  have hk : ((3 + (T i).adds : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 3 + (T i).adds ≤ w + 2)) hu) hw
  have hleaf : ∀ j, ‖cmul R (A i j) (x j) - A i j * x j‖ ≤
      gam (cRounding R h0).u 3 * ‖A i j * x j‖ + 0 := fun j => by
    rw [add_zero]
    exact cmul_err R h0 hu1 _ _
  have h := STree.comp_error (cRounding R h0) le_rfl (fun j => A i j * x j)
    (fun j => cmul R (A i j) (x j)) hleaf (fun _ => trivial) (T i) hk
  rw [mul_zero, add_zero] at h
  have he : (T i).eval (fun j => A i j * x j) = ∑ j, A i j * x j :=
    STree.eval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, zero_mul])
  have ha : (T i).absEval (fun j => A i j * x j) = ∑ j, ‖A i j‖ * ‖x j‖ := by
    rw [STree.absEval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, zero_mul])]
    exact Finset.sum_congr rfl fun j _ => norm_mul _ _
  rw [he, ha] at h
  have hg : gam R.u (3 + (T i).adds) ≤ gam R.u (w + 2) := gam_mono hu (by omega) hw
  exact h.trans (mul_le_mul_of_nonneg_right hg
    (Finset.sum_nonneg fun j _ => mul_nonneg (norm_nonneg _) (norm_nonneg _)))

theorem absMatvecC_ge (R : Rounding ℝ) (h0 : R.eta = 0) {T : ι → STree ι} {A : ι → ι → ℂ}
    {w : ℕ} (hT : RowTreesC T A w) (hw : ((w + 2 : ℕ) : ℝ) * R.u < 1) (x : ι → ℂ) (i : ι) :
    (∑ j, ‖A i j‖ * ‖x j‖) * (1 - gam R.u (w + 2)) ≤ absMatvecC R T A x i := by
  obtain ⟨hnd, hlen, hsupp⟩ := hT i
  have hu := R.u_nonneg
  have hk : ((3 + (T i).adds : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 3 + (T i).adds ≤ w + 2)) hu) hw
  have h3 : ((2 + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 2 + 1 ≤ 3 + (T i).adds)) hu) hk
  have h2 : ((1 + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 1 + 1 ≤ 2 + 1)) hu) h3
  have hu1 : R.u < 1 := lt_of_le_of_lt
    (le_mul_of_one_le_left hu (by norm_num)) h2
  have hone : ∀ z : ℝ, ‖R.rnd z - z‖ ≤ gam R.u 1 * ‖z‖ := fun z => by
    have := R.err z
    rw [h0, add_zero] at this
    exact this.trans (mul_le_mul_of_nonneg_right (u_le_gam_one hu hu1) (norm_nonneg z))
  have hleaf : ∀ j, ‖R.rnd (R.rnd ‖A i j‖ * R.rnd ‖x j‖) - ‖A i j‖ * ‖x j‖‖ ≤
      gam (R.std h0).u 3 * ‖‖A i j‖ * ‖x j‖‖ + 0 := fun j => by
    rw [add_zero]
    have hm := rel_mul hu (j := 1) (k := 1) h2 (hone ‖A i j‖) (hone ‖x j‖)
    exact rel_rnd R h0 (k := 1 + 1) h3 hm
  have h := STree.comp_error (R.std h0) le_rfl (fun j => ‖A i j‖ * ‖x j‖)
    (fun j => R.rnd (R.rnd ‖A i j‖ * R.rnd ‖x j‖)) hleaf (fun _ => trivial) (T i) hk
  rw [mul_zero, add_zero] at h
  have he : (T i).eval (fun j => ‖A i j‖ * ‖x j‖) = ∑ j, ‖A i j‖ * ‖x j‖ :=
    STree.eval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, norm_zero, zero_mul])
  have ha : (T i).absEval (fun j => ‖A i j‖ * ‖x j‖) = ∑ j, ‖A i j‖ * ‖x j‖ := by
    rw [STree.absEval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, norm_zero, zero_mul])]
    exact Finset.sum_congr rfl fun j _ => by
      rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (norm_nonneg _) (norm_nonneg _))]
  rw [he, ha, Real.norm_eq_abs] at h
  have hs : 0 ≤ ∑ j, ‖A i j‖ * ‖x j‖ :=
    Finset.sum_nonneg fun j _ => mul_nonneg (norm_nonneg _) (norm_nonneg _)
  have hg : gam R.u (3 + (T i).adds) ≤ gam R.u (w + 2) := gam_mono hu (by omega) hw
  have := mul_le_mul_of_nonneg_right hg hs
  have h' : |(T i).comp R.rnd (fun j => R.rnd (R.rnd ‖A i j‖ * R.rnd ‖x j‖)) -
      ∑ j, ‖A i j‖ * ‖x j‖| ≤ gam R.u (3 + (T i).adds) * ∑ j, ‖A i j‖ * ‖x j‖ := h
  have hl := (abs_le.mp h').1
  unfold absMatvecC
  nlinarith

/-- The constants of the complex accounting for a row width `w` with `w + 4 ≤ 2²⁶`. -/
theorem complex_width_consts {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {w : ℕ}
    (hw1 : 1 ≤ w) (hw : w + 4 ≤ 2 ^ 26) :
    u ≤ 1 / 1000 ∧ ((w + 3 : ℕ) : ℝ) * u ≤ 1 / 2 ∧ ((w + 6 : ℕ) : ℝ) * u ≤ 1 / 2 ∧
      0 ≤ gam u (w + 2) ∧ gam u (w + 2) + u * (1 + gam u (w + 2)) ≤ gam u (w + 3) ∧
      2 * u + u ^ 2 ≤ gam u (w + 3) ∧ gam u (w + 3) ≤ 1 / 1000 ∧
      gam u (w + 3) ≤ gam u (w + 6) ∧ gam u (w + 6) ≤ 1 / 1000 := by
  obtain ⟨hu1, hkw, -, -, -, hG3, -⟩ := small_consts hu hu53 (k := w + 1) (by omega)
  obtain ⟨-, hkw6, -, -, -, hG6, -⟩ := small_consts hu hu53 (k := w + 4) (by omega)
  have e3 : w + 1 + 2 = w + 3 := by omega
  have e6 : w + 4 + 2 = w + 6 := by omega
  rw [e3] at hkw hG3
  rw [e6] at hkw6 hG6
  have hw3 : ((w + 3 : ℕ) : ℝ) * u < 1 := by linarith
  have hw6 : ((w + 6 : ℕ) : ℝ) * u < 1 := by linarith
  have hw2 : ((w + 2 : ℕ) : ℝ) * u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 2 ≤ w + 3)) hu) hw3
  have hg2 : 0 ≤ gam u (w + 2) := gam_nonneg hu hw2
  have hsucc : gam u (w + 2) + u * (1 + gam u (w + 2)) ≤ gam u (w + 3) :=
    gam_succ hu (k := w + 2) hw3
  have hg2u : 2 * u + u ^ 2 ≤ gam u (w + 3) := by
    have h1 := u_le_gam_one hu (by linarith)
    have h2 : ((1 + 1 : ℕ) : ℝ) * u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 1 + 1 ≤ w + 3)) hu) hw3
    have s : gam u 1 + u * (1 + gam u 1) ≤ gam u (1 + 1) := gam_succ hu h2
    have m : gam u (1 + 1) ≤ gam u (w + 3) := gam_mono hu (by omega) hw3
    nlinarith
  have hG36 : gam u (w + 3) ≤ gam u (w + 6) := gam_mono hu (by omega) hw6
  exact ⟨hu1, by linarith, by linarith, hg2, hsucc, hg2u, hG3, hG36, hG6⟩

/-- The residual entry of complex data against the exact one: `γ_{w+3}((|A| |x|)_i + |x_i| |μ|)`. -/
theorem residC_entry_err (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    {w : ℕ} (hw1 : 1 ≤ w) (hw : w + 4 ≤ 2 ^ 26) {A : ι → ι → ℂ} {T : ι → STree ι}
    (hT : RowTreesC T A w) (x : ι → ℂ) (μ : ℝ) (i : ι) :
    ‖residC R T A x μ i - (∑ j, A i j * x j - (μ : ℂ) * x i)‖ ≤
      gam R.u (w + 3) * (∑ j, ‖A i j‖ * ‖x j‖ + ‖x i‖ * |μ|) := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hk3, -, hg2, hsucc, hg2u, -, -, -⟩ := complex_width_consts hu hu53 hw1 hw
  have hw3 : ((w + 3 : ℕ) : ℝ) * R.u < 1 := by linarith
  have hw2 : ((w + 2 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 2 ≤ w + 3)) hu) hw3
  have hy := matvecC_error R h0 hu1 hT hw2 x i
  obtain ⟨a, ha⟩ : ∃ a, a = ∑ j, ‖A i j‖ * ‖x j‖ := ⟨_, rfl⟩
  obtain ⟨q, hq⟩ : ∃ q, q = ‖x i‖ * |μ| := ⟨_, rfl⟩
  rw [← ha] at hy
  rw [← ha, ← hq]
  have ha0 : 0 ≤ a := by
    rw [ha]
    exact Finset.sum_nonneg fun j _ => mul_nonneg (norm_nonneg _) (norm_nonneg _)
  have hq0 : 0 ≤ q := by
    rw [hq]
    exact mul_nonneg (norm_nonneg _) (abs_nonneg _)
  have hya : ‖∑ j, A i j * x j‖ ≤ a := by
    rw [ha]
    exact (norm_sum_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun j _ => norm_mul _ _))
  have hxm : ‖x i * (μ : ℂ)‖ ≤ q := by
    rw [hq, norm_mul, Complex.norm_real, Real.norm_eq_abs]
  have h1 := resid_entry_c R h0 hy hya hxm
  have e1 : residC R T A x μ i - (∑ j, A i j * x j - (μ : ℂ) * x i) =
      crnd R (matvecC R T A x i - crnd R (x i * (μ : ℂ))) -
        (∑ j, A i j * x j - x i * (μ : ℂ)) := by
    unfold residC
    rw [mul_comm (μ : ℂ) (x i)]
  rw [e1]
  have k1 := mul_le_mul_of_nonneg_right hsucc ha0
  have k2 := mul_le_mul_of_nonneg_right hg2u hq0
  linarith

/-- The computed rounding bound of complex data dominates `γ_{w+3}((|A| |x|)_i + |x_i| |μ|)`. -/
theorem roundBoundC_ge (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    {w : ℕ} (hw1 : 1 ≤ w) (hw : w + 4 ≤ 2 ^ 26) {A : ι → ι → ℂ} {T' : ι → STree ι}
    (hT' : RowTreesC T' A w) (x : ι → ℂ) (μ : ℝ) (i : ι) :
    gam R.u (w + 3) * (∑ j, ‖A i j‖ * ‖x j‖ + ‖x i‖ * |μ|) ≤ roundBoundC R T' w A x μ i := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hk3, hk6, hg2, hsucc, -, hG3, hG36, hG6⟩ := complex_width_consts hu hu53 hw1 hw
  have hu1' : R.u ≤ 1 := by linarith
  have hw3 : ((w + 3 : ℕ) : ℝ) * R.u < 1 := by linarith
  have hw2 : ((w + 2 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 2 ≤ w + 3)) hu) hw3
  have hs23 : gam R.u (w + 2) + R.u ≤ gam R.u (w + 3) := by nlinarith
  obtain ⟨hgh3, hgh3'⟩ := gamHat_bounds R h0 (by linarith) (k := w + 3) hk3
  obtain ⟨hgh6, -⟩ := gamHat_bounds R h0 (by linarith) (k := w + 6) hk6
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hG30 : 0 ≤ gam R.u (w + 3) := by linarith
  have hC0 : 0 ≤ 1 - gam R.u (w + 3) * (1 - R.u) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsq1 hG30]
  have hd : R.rnd (1 - gamHat R (w + 3)) ≤ (1 - gam R.u (w + 3) * (1 - R.u) ^ 2) * (1 + R.u) :=
    R.rnd_le_of_le h0 hu1' (by linarith) hC0
  have hd0 : 0 < R.rnd (1 - gamHat R (w + 3)) := by
    have := R.le_rnd_of_le h0 hu1' (w := 1 - gamHat R (w + 3)) (z := 1 - 3 * gam R.u (w + 3))
      (by linarith) (by linarith)
    nlinarith
  have ha0 : 0 ≤ ∑ j, ‖A i j‖ * ‖x j‖ :=
    Finset.sum_nonneg fun j _ => mul_nonneg (norm_nonneg _) (norm_nonneg _)
  have hq0 : 0 ≤ ‖x i‖ * |μ| := mul_nonneg (norm_nonneg _) (abs_nonneg _)
  have hah := absMatvecC_ge R h0 hT' hw2 x i
  have hb2 := R.chain_abs h0 hu1' ha0 hah (by linarith) hd0 hd (num_abs hu hu1 hg2 hs23 hG3)
  have hxr := R.le_rnd_of_le h0 hu1' (w := ‖x i‖) (z := ‖x i‖) le_rfl (norm_nonneg _)
  have hqh := R.le_rnd_mul h0 hu1' (mul_nonneg (norm_nonneg _) (by linarith)) (abs_nonneg μ)
    hxr (le_refl |μ|)
  have hqh' : ‖x i‖ * |μ| * (1 - R.u) ^ 2 ≤ R.rnd (R.rnd ‖x i‖ * |μ|) := by
    calc ‖x i‖ * |μ| * (1 - R.u) ^ 2 = ‖x i‖ * (1 - R.u) * |μ| * (1 - R.u) := by ring
      _ ≤ _ := hqh
  have hG60 : 0 ≤ gam R.u (w + 6) := by linarith
  have h2g := R.le_rnd_of_le h0 hu1' (w := 2 * gamHat R (w + 6))
    (z := 2 * (gam R.u (w + 6) * (1 - R.u) ^ 2)) (by linarith)
    (mul_nonneg (by norm_num) (mul_nonneg hG60 (sq_nonneg _)))
  have hgh : 2 * gam R.u (w + 6) * (1 - R.u) ^ 3 ≤ R.rnd (2 * gamHat R (w + 6)) := by
    calc 2 * gam R.u (w + 6) * (1 - R.u) ^ 3 =
          2 * (gam R.u (w + 6) * (1 - R.u) ^ 2) * (1 - R.u) := by ring
      _ ≤ _ := h2g
  have hE := R.chain_E_c h0 hu1' ha0 hq0 hb2 hqh' (by linarith) hgh
  have hnE : gam R.u (w + 3) ≤ 2 * gam R.u (w + 6) * ((1 - R.u) ^ 8 * (1 + 4 * R.u)) := by
    have h8 := one_sub_mul_le_pow hu hu1' 8
    push_cast at h8
    have h9 : (1 : ℝ) / 2 ≤ (1 - R.u) ^ 8 * (1 + 4 * R.u) := by
      have := mul_le_mul_of_nonneg_right h8 (by linarith : (0 : ℝ) ≤ 1 + 4 * R.u)
      nlinarith
    have := mul_le_mul_of_nonneg_left h9 hG60
    linarith
  have h3 := mul_le_mul_of_nonneg_right hnE (add_nonneg ha0 hq0)
  exact h3.trans hE

/-- **Theorem 1 accounting, complex data, entrywise**: `|R̂_i - (A x - μ x)_i| ≤ Ê_i`. -/
theorem residC_le_roundBound (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    {w : ℕ} (hw1 : 1 ≤ w) (hw : w + 4 ≤ 2 ^ 26) {A : ι → ι → ℂ} {T T' : ι → STree ι}
    (hT : RowTreesC T A w) (hT' : RowTreesC T' A w) (x : ι → ℂ) (μ : ℝ) (i : ι) :
    ‖residC R T A x μ i - (∑ j, A i j * x j - (μ : ℂ) * x i)‖ ≤ roundBoundC R T' w A x μ i :=
  (residC_entry_err R h0 hu53 hw1 hw hT x μ i).trans (roundBoundC_ge R h0 hu53 hw1 hw hT' x μ i)

/-- **Theorem 1 in floating point, complex data** (`γ_{w+3}` for the computed `|A| |x|`,
`2 γ_{w+6}` for the entrywise bound, `2n` squares in the norms). -/
theorem residual_radius_fl_c (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    {w : ℕ} (hw1 : 1 ≤ w) (hw : w + 4 ≤ 2 ^ 26) (hn : Fintype.card ι ≤ 2 ^ 26)
    {A : ι → ι → ℂ} {T T' : ι → STree ι} (hT : RowTreesC T A w) (hT' : RowTreesC T' A w)
    {TR TX : STree (ι ⊕ ι)} {TE : STree ι} (hTR : FullTree TR) (hTE : FullTree TE)
    (hTX : FullTree TX) (x : ι → ℂ) (μ : ℝ) {tR tE tX c1 c0 : ℝ} (htR : 0 ≤ tR) (htE : 0 ≤ tE)
    (htX : 0 ≤ tX) (hc1 : 0 ≤ c1) (hc0 : 0 ≤ c0) (hxl : 0 < norm2LowC R TX tX x) :
    l2c (fun i => ∑ j, A i j * x j - (μ : ℂ) * x i) / l2c x ≤
      residualRadiusFlC R T T' TR TX TE w A x μ tR tE tX c1 c0 := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 1000 := hu53.trans (by norm_num)
  have hentry := residC_le_roundBound R h0 hu53 hw1 hw hT hT' x μ
  have hr : l2c (fun i => ∑ j, A i j * x j - (μ : ℂ) * x i) ≤
      l2c (residC R T A x μ) + l2 (roundBoundC R T' w A x μ) := by
    refine (l2c_le_add _ (residC R T A x μ)).trans (add_le_add le_rfl (l2c_le_of_norm_le ?_))
    intro i
    rw [norm_sub_rev]
    exact hentry i
  exact R.chain_final h0 (by linarith) hr (norm_nonneg _) (l2_nonneg _)
    (norm2UpC_ge R h0 hu53 hn hTR htR _) (norm2Up_ge R h0 hu53 hn hTE htE _) hc1 hc0 hxl
    (norm2LowC_le R h0 hu53 hn hTX htX x) (by linarith) (num_final hu hu1)

/-- **Theorem 1 with the computed radius, complex Hermitian matrices.** -/
theorem residual_enclosure_fl_c {A : Matrix ι ι ℂ} (hA : A.IsHermitian) (R : Rounding ℝ)
    (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) {w : ℕ} (hw1 : 1 ≤ w) (hw : w + 4 ≤ 2 ^ 26)
    (hn : Fintype.card ι ≤ 2 ^ 26) {T T' : ι → STree ι} (hT : RowTreesC T A w)
    (hT' : RowTreesC T' A w) {TR TX : STree (ι ⊕ ι)} {TE : STree ι} (hTR : FullTree TR)
    (hTE : FullTree TE) (hTX : FullTree TX) (x : EuclideanSpace ℂ ι) (μ : ℝ)
    {tR tE tX c1 c0 : ℝ} (htR : 0 ≤ tR) (htE : 0 ≤ tE) (htX : 0 ≤ tX) (hc1 : 0 ≤ c1)
    (hc0 : 0 ≤ c0) (hxl : 0 < norm2LowC R TX tX (fun i => x i)) :
    ∃ i, |hA.eigenvalues i - μ| ≤
      residualRadiusFlC R T T' TR TX TE w A (fun i => x i) μ tR tE tX c1 c0 := by
  have hlow := norm2LowC_le R h0 hu53 hn hTX htX (fun i => x i)
  have hx : x ≠ 0 := by
    intro h
    have : l2c (fun i => x i) = 0 := by rw [l2c_ofLp, h, norm_zero]
    rw [this, zero_mul] at hlow
    linarith
  obtain ⟨i, hi⟩ := residual_enclosure hA μ hx
  refine ⟨i, hi.trans ?_⟩
  have := residual_radius_fl_c R h0 hu53 hw1 hw hn hT hT' hTR hTE hTX (fun i => x i) μ htR htE
    htX hc1 hc0 hxl
  rwa [l2c_residual, l2c_ofLp] at this

end ComplexData

/-! ### E-05, Lemma 3 (i): the numerator of the Rayleigh quotient -/

section RayleighNumerator

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **E-05, Lemma 3 (i)**, real data: with `ŷ = fl(H ψ)` (rows of at most `w` stored entries),
`|⟨ψ, ŷ⟩ - ⟨ψ, H ψ⟩| ≤ γ_w |ψ|ᵀ |H| |ψ|`. -/
theorem rayleigh_numerator_error (R : Rounding ℝ) (h0 : R.eta = 0) {T : ι → STree ι}
    {A : ι → ι → ℝ} {w : ℕ} (hT : RowTrees T A w) (hw : (w : ℝ) * R.u < 1) (ψ : ι → ℝ) :
    |∑ i, ψ i * matvecFl R T A ψ i - ∑ i, ψ i * ∑ j, A i j * ψ j| ≤
      gam R.u w * ∑ i, |ψ i| * ∑ j, |A i j| * |ψ j| := by
  rw [← Finset.sum_sub_distrib, Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
  rw [← mul_sub, abs_mul]
  calc |ψ i| * |matvecFl R T A ψ i - ∑ j, A i j * ψ j|
      ≤ |ψ i| * (gam R.u w * ∑ j, |A i j| * |ψ j|) :=
        mul_le_mul_of_nonneg_left (matvec_error R h0 hT hw ψ i) (abs_nonneg _)
    _ = gam R.u w * (|ψ i| * ∑ j, |A i j| * |ψ j|) := by ring

/-- **E-05, Lemma 3 (i)** with the contract of `fp.exact_dot` (`|N̂ - ⟨ψ, ŷ⟩| ≤ e_N`):
`|N̂ - ⟨ψ, H ψ⟩| ≤ e_N + γ_w |ψ|ᵀ |H| |ψ|`. -/
theorem rayleigh_numerator_bound (R : Rounding ℝ) (h0 : R.eta = 0) {T : ι → STree ι}
    {A : ι → ι → ℝ} {w : ℕ} (hT : RowTrees T A w) (hw : (w : ℝ) * R.u < 1) (ψ : ι → ℝ)
    {Nh eN : ℝ} (hN : |Nh - ∑ i, ψ i * matvecFl R T A ψ i| ≤ eN) :
    |Nh - ∑ i, ψ i * ∑ j, A i j * ψ j| ≤ eN + gam R.u w * ∑ i, |ψ i| * ∑ j, |A i j| * |ψ j| := by
  have h := rayleigh_numerator_error R h0 hT hw ψ
  calc |Nh - ∑ i, ψ i * ∑ j, A i j * ψ j|
      = |(Nh - ∑ i, ψ i * matvecFl R T A ψ i) +
          (∑ i, ψ i * matvecFl R T A ψ i - ∑ i, ψ i * ∑ j, A i j * ψ j)| := by ring_nf
    _ ≤ |Nh - ∑ i, ψ i * matvecFl R T A ψ i| +
          |∑ i, ψ i * matvecFl R T A ψ i - ∑ i, ψ i * ∑ j, A i j * ψ j| := abs_add_le' _ _
    _ ≤ eN + gam R.u w * ∑ i, |ψ i| * ∑ j, |A i j| * |ψ j| := add_le_add hN h

end RayleighNumerator

section AbsSums

variable {𝕂 : Type*} [RCLike 𝕂]

/-- **Lemma A1, second sentence** (the computed `|H|_abs` sums). A row sum of
`abs_matvec(ones)` (or a column sum of `abs_rmatvec(ones)`) is the sum, in any order along a
tree with at most `m` leaves, of the computed moduli `m̂_k` of the monomial amplitudes; the
multiplication by the entry `1.0` of `ones` is exact. If each modulus is within `γ_a |t_k|` of the
exact `|t_k|`, the computed sum is within `γ_{m+a-1} Σ |t_k|` of the exact sum. -/
theorem abs_sum_error (R : Rounding ℝ) (h0 : R.eta = 0) {m a : ℕ}
    (hW : ((m + a : ℕ) : ℝ) * R.u < 1) {κ : Type*} [DecidableEq κ] (t : κ → 𝕂) (mh : κ → ℝ)
    (S : Finset κ) (hmh : ∀ k ∈ S, |mh k - ‖t k‖| ≤ gam R.u a * ‖t k‖) {s : ℝ}
    (hb : EntryBuilt R m mh S s) :
    |s - ∑ k ∈ S, ‖t k‖| ≤ gam R.u (m + a - 1) * ∑ k ∈ S, ‖t k‖ := by
  have hu := R.u_nonneg
  rcases hb with ⟨hS, hh⟩ | ⟨T, hnd, hmem, hlen, hh⟩
  · subst hS
    subst hh
    simp
  · have hW1 : ((m + a - 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : m + a - 1 ≤ m + a)) hu) hW
    have hT : ((a + T.adds : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : a + T.adds ≤ m + a)) hu) hW
    have ha : ((a : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : a ≤ m + a)) hu) hW
    let t0 : κ → ℝ := fun k => if k ∈ S then ‖t k‖ else mh k
    have hleaf : ∀ k, ‖mh k - t0 k‖ ≤ gam R.u a * ‖t0 k‖ + 0 := by
      intro k
      by_cases hk : k ∈ S
      · simp only [t0, hk, ↓reduceIte, add_zero, Real.norm_eq_abs, abs_norm]
        exact hmh k hk
      · simp only [t0, hk, ↓reduceIte, sub_self, norm_zero, add_zero]
        exact mul_nonneg (gam_nonneg hu ha) (norm_nonneg _)
    have herr := STree.comp_error (R.std h0) le_rfl t0 mh hleaf (fun _ => trivial) T hT
    have hfs : T.leaves.toFinset = S := Finset.ext fun k => by simp [hmem]
    have he : T.eval t0 = ∑ k ∈ S, ‖t k‖ := by
      rw [STree.eval_eq_sum, ← List.sum_toFinset _ hnd, hfs]
      exact Finset.sum_congr rfl fun k hk => by simp [t0, hk]
    have hae : T.absEval t0 = ∑ k ∈ S, ‖t k‖ := by
      rw [STree.absEval_eq_sum, ← List.sum_toFinset _ hnd, hfs]
      exact Finset.sum_congr rfl fun k hk => by simp [t0, hk]
    have hg : gam R.u (a + T.adds) ≤ gam R.u (m + a - 1) := gam_mono hu (by omega) hW1
    have hs0 : 0 ≤ ∑ k ∈ S, ‖t k‖ := Finset.sum_nonneg fun k _ => norm_nonneg _
    rw [hh]
    calc |T.comp R.rnd mh - ∑ k ∈ S, ‖t k‖| = ‖T.comp (R.std h0).rnd mh - T.eval t0‖ := by
          rw [he, Real.norm_eq_abs]
          rfl
      _ ≤ gam R.u (a + T.adds) * T.absEval t0 +
            ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * 0 := herr
      _ ≤ gam R.u (m + a - 1) * ∑ k ∈ S, ‖t k‖ := by
          rw [hae, mul_zero, add_zero]
          exact mul_le_mul_of_nonneg_right hg hs0

/-- The factor form: `(1 - γ_{m+a-1}) Σ |t_k| ≤ ŝ ≤ (1 + γ_{m+a-1}) Σ |t_k|`. -/
theorem abs_sum_bounds (R : Rounding ℝ) (h0 : R.eta = 0) {m a : ℕ}
    (hW : ((m + a : ℕ) : ℝ) * R.u < 1) {κ : Type*} [DecidableEq κ] (t : κ → 𝕂) (mh : κ → ℝ)
    (S : Finset κ) (hmh : ∀ k ∈ S, |mh k - ‖t k‖| ≤ gam R.u a * ‖t k‖) {s : ℝ}
    (hb : EntryBuilt R m mh S s) :
    (1 - gam R.u (m + a - 1)) * ∑ k ∈ S, ‖t k‖ ≤ s ∧
      s ≤ (1 + gam R.u (m + a - 1)) * ∑ k ∈ S, ‖t k‖ := by
  have h := abs_le.mp (abs_sum_error R h0 hW t mh S hmh hb)
  constructor <;> nlinarith [h.1, h.2]

/-- Real amplitudes (`np.abs` exact): with `|t̂_k - t_k| ≤ γ_{2L} |t_k|` (`mono_error`), the
computed sums of `|t̂_k|` are within the factor `1 ± γ_{m+2L-1}` of Lemma A1. -/
theorem abs_sum_bounds_real (R : Rounding ℝ) (h0 : R.eta = 0) {m L : ℕ}
    (hW : ((m + 2 * L : ℕ) : ℝ) * R.u < 1) {κ : Type*} [DecidableEq κ] (t th : κ → ℝ)
    (S : Finset κ) (hth : ∀ k ∈ S, |th k - t k| ≤ gam R.u (2 * L) * |t k|) {s : ℝ}
    (hb : EntryBuilt R m (fun k => |th k|) S s) :
    (1 - gam R.u (m + 2 * L - 1)) * ∑ k ∈ S, |t k| ≤ s ∧
      s ≤ (1 + gam R.u (m + 2 * L - 1)) * ∑ k ∈ S, |t k| := by
  have h := abs_sum_bounds R h0 hW t (fun k => |th k|) S (fun k hk => by
    rw [Real.norm_eq_abs]
    exact (abs_abs_sub_abs_le_abs_sub _ _).trans (hth k hk)) hb
  simpa [Real.norm_eq_abs] using h

/-- Complex amplitudes (`np.abs` is one rounding of the modulus): the factor is
`1 ± γ_{m+2L}`, one index more than the document states for real amplitudes. -/
theorem abs_sum_bounds_complex (R : Rounding ℝ) (h0 : R.eta = 0) {m L : ℕ}
    (hW : ((m + 2 * L + 1 : ℕ) : ℝ) * R.u < 1) {κ : Type*} [DecidableEq κ] (t th : κ → ℂ)
    (S : Finset κ) (hth : ∀ k ∈ S, ‖th k - t k‖ ≤ gam R.u (2 * L) * ‖t k‖) {s : ℝ}
    (hb : EntryBuilt R m (fun k => R.rnd ‖th k‖) S s) :
    (1 - gam R.u (m + 2 * L)) * ∑ k ∈ S, ‖t k‖ ≤ s ∧
      s ≤ (1 + gam R.u (m + 2 * L)) * ∑ k ∈ S, ‖t k‖ := by
  have hu := R.u_nonneg
  have hk : ((2 * L + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 2 * L + 1 ≤ m + 2 * L + 1)) hu) hW
  have hmh : ∀ k ∈ S, |R.rnd ‖th k‖ - ‖t k‖| ≤ gam R.u (2 * L + 1) * ‖t k‖ := by
    intro k hk'
    have h1 := hth k hk'
    have hs := gam_succ hu hk
    have hk0 : ((2 * L : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 2 * L ≤ 2 * L + 1)) hu) hk
    have g0 := gam_nonneg hu hk0
    have hr := R.abs_rnd_sub_le h0 ‖th k‖
    rw [abs_norm] at hr
    have hn : |‖th k‖ - ‖t k‖| ≤ gam R.u (2 * L) * ‖t k‖ :=
      (abs_norm_sub_norm_le _ _).trans h1
    have hth' : ‖th k‖ ≤ (1 + gam R.u (2 * L)) * ‖t k‖ := by
      have := (abs_le.mp hn).2
      linarith
    have hsp := mul_le_mul_of_nonneg_right hs (norm_nonneg (t k))
    calc |R.rnd ‖th k‖ - ‖t k‖| = |(R.rnd ‖th k‖ - ‖th k‖) + (‖th k‖ - ‖t k‖)| := by ring_nf
      _ ≤ |R.rnd ‖th k‖ - ‖th k‖| + |‖th k‖ - ‖t k‖| := abs_add_le _ _
      _ ≤ R.u * ((1 + gam R.u (2 * L)) * ‖t k‖) + gam R.u (2 * L) * ‖t k‖ :=
          add_le_add (hr.trans (mul_le_mul_of_nonneg_left hth' hu)) hn
      _ ≤ gam R.u (2 * L + 1) * ‖t k‖ := by nlinarith
  have h := abs_sum_bounds R h0 (a := 2 * L + 1) (m := m) (by simpa [add_assoc] using hW) t
    (fun k => R.rnd ‖th k‖) S hmh hb
  have e : m + (2 * L + 1) - 1 = m + 2 * L := by omega
  rw [e] at h
  exact h

end AbsSums

end CertifyMathlib.Enclosure.FP
