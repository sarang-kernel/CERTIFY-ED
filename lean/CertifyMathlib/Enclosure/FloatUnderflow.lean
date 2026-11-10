import CertifyMathlib.Enclosure.FloatAccounting

/-!
# Theorem 1 in floating point under the underflow-tolerant model (INT-05, Theorem U5)

The rounding `R` satisfies the underflow-tolerant model of `docs/proofs/int05_underflow.md`:
`|rnd z - z| ≤ u |z| + eta` for every exact result, and a sum of two representable numbers has no
absolute term (`Rounding`). `UModel R` adds the binary64 facts used here: `u = 2⁻⁵³`,
`eta ≤ 2⁻¹⁰⁷⁵`, no absolute term for an exact result of magnitude at least `2⁻¹⁰²²` (the normal
range), and representable numbers are closed under negation.

The code (`certify_ed/fp.py`, `certify_ed/enclosure.py`) adds four terms to the standard-model
computation of `residuals`:

* `T_N = norm_underflow_term(N)`, added by `norm2_upper` and subtracted by `norm2_lower`
  (`normUnder`);
* `A_{n,w} = residual_additive_term(n, w)`, added to `R_up + E_up` (`addTerm`, with the computed
  `c_w = residual_entry_count(w)`, `cwHat`);
* `2 ULP_MIN = 2⁻¹⁰⁷³`, added to the quotient `upper / xl (1 + 4u)`.

`residual_radius_int05` proves that the radius computed with these terms bounds
`‖A x - μ x‖ / ‖x‖` for every input (real data), with no magnitude hypothesis.
-/

open Finset

namespace CertifyMathlib.Enclosure.FP

/-! ### The model -/

/-- The smallest normal binary64 number `2⁻¹⁰²²`. -/
noncomputable def normalMin : ℝ := (2 : ℝ)⁻¹ ^ 1022

theorem normalMin_pos : 0 < normalMin := by unfold normalMin; positivity

theorem normalMin_le_pow {k : ℕ} (hk : k ≤ 1022) : normalMin ≤ (2 : ℝ)⁻¹ ^ k :=
  pow_le_pow_of_le_one (by norm_num) (by norm_num) hk

theorem normalMin_le_half : normalMin ≤ 1 / 2 := by
  calc normalMin ≤ (2 : ℝ)⁻¹ ^ 1 := normalMin_le_pow (by norm_num)
    _ = 1 / 2 := by norm_num

/-- The binary64 facts used with the underflow-tolerant model. -/
structure UModel (R : Rounding ℝ) : Prop where
  u_eq : R.u = (2 : ℝ)⁻¹ ^ 53
  eta_le : R.eta ≤ (2 : ℝ)⁻¹ ^ 1075
  normal : ∀ z : ℝ, normalMin ≤ |z| → |R.rnd z - z| ≤ R.u * |z|
  neg : ∀ a, R.IsFloat a → R.IsFloat (-a)

namespace Rounding

variable (R : Rounding ℝ)

theorem abs_err (z : ℝ) : |R.rnd z - z| ≤ R.u * |z| + R.eta := by
  have h := R.err z
  rwa [Real.norm_eq_abs, Real.norm_eq_abs] at h

theorem abs_add_err {a b : ℝ} (ha : R.IsFloat a) (hb : R.IsFloat b) :
    |R.rnd (a + b) - (a + b)| ≤ R.u * |a + b| + 0 := by
  have h := R.add_err a b ha hb
  rw [Real.norm_eq_abs, Real.norm_eq_abs] at h
  linarith

end Rounding

/-- Lower envelope of a rounded value: from `L ≤ z` and `-m ≤ z`. -/
theorem lo_of_err {y z u e L m : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1) (he : |y - z| ≤ u * |z| + e)
    (hL : L ≤ z) (hm0 : 0 ≤ m) (hm : -m ≤ z) : L * (1 - u) - 2 * u * m - e ≤ y := by
  have h1 := (abs_le.mp he).1
  have h2 := mul_le_mul_of_nonneg_left hL (by linarith : (0 : ℝ) ≤ 1 - u)
  rcases le_total 0 z with hz | hz
  · rw [abs_of_nonneg hz] at h1
    nlinarith [mul_nonneg hu hm0]
  · rw [abs_of_nonpos hz] at h1
    nlinarith [mul_le_mul_of_nonneg_left hm hu]

/-- Upper envelope of a rounded value: from `z ≤ H` with `0 ≤ H`. -/
theorem hi_of_err {y z u e H : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1) (he : |y - z| ≤ u * |z| + e)
    (hH : z ≤ H) (hH0 : 0 ≤ H) : y ≤ H * (1 + u) + e := by
  have h1 := (abs_le.mp he).2
  rcases le_total 0 z with hz | hz
  · rw [abs_of_nonneg hz] at h1
    nlinarith [mul_le_mul_of_nonneg_left hH hu]
  · rw [abs_of_nonpos hz] at h1
    nlinarith [mul_nonneg hu hH0]

namespace UModel

variable {R : Rounding ℝ} (M : UModel R)

include M

theorem u_pos : 0 < R.u := by rw [M.u_eq]; positivity

theorem u_le53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53 := M.u_eq.le

theorem u_small : R.u ≤ 1 / 10 ^ 15 := by rw [M.u_eq]; norm_num

/-- In the normal range a lower bound `z` of the exact result gives `z (1 - u)`. -/
theorem le_rnd_normal {w z : ℝ} (hz : normalMin ≤ z) (hw : z ≤ w) :
    z * (1 - R.u) ≤ R.rnd w := by
  have hw0 : 0 ≤ w := le_trans (le_trans normalMin_pos.le hz) hw
  have h := M.normal w (by rw [abs_of_nonneg hw0]; linarith)
  rw [abs_of_nonneg hw0] at h
  have hu := R.u_nonneg
  have h1 := (abs_le.mp h).1
  have hu1 : R.u ≤ 1 := by linarith [M.u_small]
  nlinarith [mul_le_mul_of_nonneg_left hw (by linarith : (0 : ℝ) ≤ 1 - R.u)]

/-- In the normal range an upper bound `z` of the exact result gives `z (1 + u)`. -/
theorem rnd_le_normal {w z : ℝ} (hw0 : normalMin ≤ w) (hw : w ≤ z) :
    R.rnd w ≤ z * (1 + R.u) := by
  have hw0' : 0 ≤ w := le_trans normalMin_pos.le hw0
  have h := M.normal w (by rw [abs_of_nonneg hw0']; exact hw0)
  rw [abs_of_nonneg hw0'] at h
  have hu := R.u_nonneg
  have h1 := (abs_le.mp h).2
  nlinarith [mul_le_mul_of_nonneg_left hw hu]

/-- The computed `γ̂_k` under the model: `γ_k (1 - u)² ≤ γ̂_k ≤ γ_k (1 + 3u)`. -/
theorem gamHat_bounds {k : ℕ} (hk1 : 1 ≤ k) (hk : k ≤ 2 ^ 26 + 2) :
    gam R.u k * (1 - R.u) ^ 2 ≤ gamHat R k ∧ gamHat R k ≤ gam R.u k * (1 + 3 * R.u) := by
  have hu := R.u_nonneg
  have hus := M.u_small
  obtain ⟨hku, -, -⟩ := gam_regime hu M.u_le53 hk
  have hk1r : (1 : ℝ) ≤ k := by exact_mod_cast hk1
  have hku1 : R.u ≤ (k : ℝ) * R.u := by nlinarith
  set d := 1 - (k : ℝ) * R.u with hd_def
  have hd : 1 / 2 ≤ d := by rw [hd_def]; nlinarith
  have hd0 : 0 < d := by linarith
  have htiny := normalMin_le_half
  have hd1 : R.rnd d ≤ d * (1 + R.u) := M.rnd_le_normal (htiny.trans hd) le_rfl
  have hd2 : d * (1 - R.u) ≤ R.rnd d := M.le_rnd_normal (htiny.trans hd) le_rfl
  have hdp : 0 < d * (1 - R.u) := mul_pos hd0 (by linarith)
  have hdpos : 0 < R.rnd d := lt_of_lt_of_le hdp hd2
  have hg : gam R.u k * d = (k : ℝ) * R.u := by
    unfold gam
    exact div_mul_cancel₀ _ hd0.ne'
  have hg0 : 0 ≤ gam R.u k := gam_nonneg hu (by linarith)
  have hgk : (k : ℝ) * R.u ≤ gam R.u k := by nlinarith
  -- the quotient
  have hq1 : gam R.u k * (1 - R.u) ≤ (k : ℝ) * R.u / R.rnd d := by
    rw [le_div_iff₀ hdpos]
    calc gam R.u k * (1 - R.u) * R.rnd d ≤ gam R.u k * (1 - R.u) * (d * (1 + R.u)) :=
          mul_le_mul_of_nonneg_left hd1 (mul_nonneg hg0 (by linarith))
      _ = (gam R.u k * d) * (1 - R.u ^ 2) := by ring
      _ = (k : ℝ) * R.u * (1 - R.u ^ 2) := by rw [hg]
      _ ≤ (k : ℝ) * R.u := by
        nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ k) hu) (sq_nonneg R.u)]
  have hq2 : (k : ℝ) * R.u / R.rnd d ≤ gam R.u k / (1 - R.u) := by
    rw [div_le_div_iff₀ hdpos (by linarith)]
    calc (k : ℝ) * R.u * (1 - R.u) = gam R.u k * (d * (1 - R.u)) := by rw [← hg]; ring
      _ ≤ gam R.u k * R.rnd d := mul_le_mul_of_nonneg_left hd2 hg0
  have hgt : normalMin ≤ gam R.u k * (1 - R.u) := by
    have h54 : normalMin ≤ (2 : ℝ)⁻¹ ^ 54 := normalMin_le_pow (by norm_num)
    have : (2 : ℝ)⁻¹ ^ 54 ≤ R.u * (1 - R.u) := by
      rw [M.u_eq]; norm_num
    nlinarith [mul_le_mul_of_nonneg_right hku1 (by linarith : (0 : ℝ) ≤ 1 - R.u),
      mul_le_mul_of_nonneg_right hgk (by linarith : (0 : ℝ) ≤ 1 - R.u)]
  refine ⟨?_, ?_⟩
  · have := M.le_rnd_normal hgt hq1
    unfold gamHat
    calc gam R.u k * (1 - R.u) ^ 2 = gam R.u k * (1 - R.u) * (1 - R.u) := by ring
      _ ≤ _ := this
  · have h3 := M.rnd_le_normal (hgt.trans hq1) hq2
    unfold gamHat
    refine h3.trans ?_
    rw [div_mul_eq_mul_div, div_le_iff₀ (by linarith)]
    nlinarith [mul_nonneg hg0 (mul_nonneg hu hu)]

end UModel

/-! ### Square roots -/

theorem sqrt_sub_le' {a b S : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a - b ≤ S) :
    Real.sqrt a - Real.sqrt b ≤ Real.sqrt S := by
  rcases le_total (Real.sqrt a) (Real.sqrt b) with hab | hab
  · linarith [Real.sqrt_nonneg S]
  · apply Real.le_sqrt_of_sq_le
    have h1 := Real.sq_sqrt ha
    have h2 := Real.sq_sqrt hb
    have h3 : Real.sqrt b * Real.sqrt b ≤ Real.sqrt a * Real.sqrt b :=
      mul_le_mul_of_nonneg_right hab (Real.sqrt_nonneg b)
    nlinarith

theorem sqrt_add_le' {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Real.sqrt (a + b) ≤ Real.sqrt a + Real.sqrt b := by
  rw [Real.sqrt_le_iff]
  refine ⟨by positivity, ?_⟩
  have h1 := Real.sq_sqrt ha
  have h2 := Real.sq_sqrt hb
  nlinarith [mul_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b)]

/-! ### `norm_underflow_term` and the two norm bounds -/

/-- `fp.norm_underflow_term(N)`: `sqrt(0.5 N) 2⁻⁵³⁷ (1 + 2 γ̂_{N+2}) (1 + 8u) (1 + 8u)`,
evaluated left to right with every operation rounded. -/
noncomputable def normUnder (R : Rounding ℝ) (N : ℕ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (Real.sqrt (R.rnd (2⁻¹ * (N : ℝ)))) * (2 : ℝ)⁻¹ ^ 537) *
    R.rnd (1 + R.rnd (2 * gamHat R (N + 2)))) * (1 + 8 * R.u)) * (1 + 8 * R.u))

/-- `√(N / 2) 2⁻⁵³⁷`, which equals `√(N u_min)` for `u_min = 2⁻¹⁰⁷⁵`. -/
noncomputable def rootN (N : ℕ) : ℝ := Real.sqrt (2⁻¹ * (N : ℝ)) * (2 : ℝ)⁻¹ ^ 537

theorem rootN_nonneg (N : ℕ) : 0 ≤ rootN N := by unfold rootN; positivity

theorem half_le_sqrt {N : ℕ} (hN : 1 ≤ N) : 1 / 2 ≤ Real.sqrt (2⁻¹ * (N : ℝ)) := by
  have hN' : (1 : ℝ) ≤ N := by exact_mod_cast hN
  apply Real.le_sqrt_of_sq_le
  nlinarith

theorem rootN_ge {N : ℕ} (hN : 1 ≤ N) : (2 : ℝ)⁻¹ ^ 538 ≤ rootN N := by
  unfold rootN
  have h := half_le_sqrt hN
  calc (2 : ℝ)⁻¹ ^ 538 = 1 / 2 * (2 : ℝ)⁻¹ ^ 537 := by rw [pow_succ]; ring
    _ ≤ _ := mul_le_mul_of_nonneg_right h (by positivity)

/-- The additive part `σ = N (1 + g) eta` of a sum of `N` squares, as a norm. -/
theorem sqrt_sigma_le {N : ℕ} {g eta : ℝ} (hg : 0 ≤ g)
    (he : eta ≤ (2 : ℝ)⁻¹ ^ 1075) :
    Real.sqrt ((N : ℝ) * (1 + g) * eta) ≤ rootN N * (1 + g) := by
  have hN : (0 : ℝ) ≤ N := Nat.cast_nonneg N
  set p := (2 : ℝ)⁻¹ ^ 537 with hp
  have hp0 : 0 ≤ p := by positivity
  have e1 : (2 : ℝ)⁻¹ ^ 1075 = 2⁻¹ * p ^ 2 := by
    rw [hp, ← pow_mul, ← pow_succ']
  have h1 : (N : ℝ) * (1 + g) * eta ≤ (2⁻¹ * (N : ℝ) * (1 + g)) * p ^ 2 := by
    calc (N : ℝ) * (1 + g) * eta ≤ (N : ℝ) * (1 + g) * (2 : ℝ)⁻¹ ^ 1075 :=
          mul_le_mul_of_nonneg_left he (by positivity)
      _ = _ := by rw [e1]; ring
  have h3 : Real.sqrt (1 + g) ≤ 1 + g := by
    rw [Real.sqrt_le_left (by linarith)]
    nlinarith
  have e2 : Real.sqrt ((2⁻¹ * (N : ℝ) * (1 + g)) * p ^ 2) =
      Real.sqrt (2⁻¹ * (N : ℝ)) * Real.sqrt (1 + g) * p := by
    rw [Real.sqrt_mul (by positivity : (0 : ℝ) ≤ 2⁻¹ * (N : ℝ) * (1 + g)) (p ^ 2),
      Real.sqrt_sq hp0, Real.sqrt_mul (by positivity : (0 : ℝ) ≤ 2⁻¹ * (N : ℝ)) (1 + g)]
  unfold rootN
  calc Real.sqrt ((N : ℝ) * (1 + g) * eta) ≤ Real.sqrt ((2⁻¹ * (N : ℝ) * (1 + g)) * p ^ 2) :=
        Real.sqrt_le_sqrt h1
    _ = Real.sqrt (2⁻¹ * (N : ℝ)) * Real.sqrt (1 + g) * p := e2
    _ ≤ Real.sqrt (2⁻¹ * (N : ℝ)) * (1 + g) * p := by
        gcongr
    _ = Real.sqrt (2⁻¹ * (N : ℝ)) * p * (1 + g) := by ring

/-- The smallest unit is negligible against `rootN N u`: `35 eta ≤ rootN N u / 1000`. -/
theorem eta_le_rootN {R : Rounding ℝ} (M : UModel R) {N : ℕ} (hN : 1 ≤ N) :
    35 * R.eta ≤ rootN N * R.u / 1000 := by
  have h1 := rootN_ge hN
  have e : (2 : ℝ)⁻¹ ^ 1075 = (2 : ℝ)⁻¹ ^ 538 * (2 : ℝ)⁻¹ ^ 53 * (2 : ℝ)⁻¹ ^ 484 := by
    rw [← pow_add, ← pow_add]
  have h2 : (2 : ℝ)⁻¹ ^ 484 ≤ (2 : ℝ)⁻¹ ^ 20 := pow_le_pow_of_le_one (by norm_num) (by norm_num)
    (by norm_num)
  have h3 : (2 : ℝ)⁻¹ ^ 20 ≤ 1 / 35000 := by norm_num
  have h4 : 0 ≤ (2 : ℝ)⁻¹ ^ 538 * (2 : ℝ)⁻¹ ^ 53 := by positivity
  calc 35 * R.eta ≤ 35 * (2 : ℝ)⁻¹ ^ 1075 := by linarith [M.eta_le]
    _ = 35 * ((2 : ℝ)⁻¹ ^ 538 * (2 : ℝ)⁻¹ ^ 53 * (2 : ℝ)⁻¹ ^ 484) := by rw [e]
    _ ≤ 35 * ((2 : ℝ)⁻¹ ^ 538 * (2 : ℝ)⁻¹ ^ 53 * (1 / 35000)) := by
        gcongr
        exact h2.trans h3
    _ = (2 : ℝ)⁻¹ ^ 538 * (2 : ℝ)⁻¹ ^ 53 / 1000 := by ring
    _ ≤ rootN N * R.u / 1000 := by
        rw [M.u_eq]
        gcongr

/-- **`norm_underflow_term` from below.** -/
theorem normUnder_ge {R : Rounding ℝ} (M : UModel R) {N : ℕ} (hN1 : 1 ≤ N) (hN : N ≤ 2 ^ 26) :
    rootN N * ((1 + 2 * gam R.u (N + 2) * (1 - R.u) ^ 3) * (1 - R.u) ^ 7 *
      (1 + 8 * R.u) ^ 2) ≤ normUnder R N := by
  have hu := R.u_nonneg
  have hus := M.u_small
  have h1u : 0 ≤ 1 - R.u := by linarith
  have h1u2 : 1 / 2 ≤ 1 - R.u := by linarith
  have h8 : 0 ≤ 1 + 8 * R.u := by linarith
  have hfac : 1 ≤ (1 - R.u) * (1 + 8 * R.u) := by nlinarith
  obtain ⟨-, hk2, hg0, -, -, -, -⟩ := small_consts hu M.u_le53 hN
  have hG1 : R.u ≤ gam R.u (N + 2) := by
    have h := u_le_gam_one hu (by linarith)
    have hm : gam R.u 1 ≤ gam R.u (N + 2) := gam_mono hu (by omega) (by linarith)
    linarith
  have hG0 : 0 ≤ gam R.u (N + 2) := le_trans hu hG1
  obtain ⟨hgh, -⟩ := M.gamHat_bounds (k := N + 2) (by omega) (by omega)
  have hN' : (1 : ℝ) ≤ N := by exact_mod_cast hN1
  set h := 2⁻¹ * (N : ℝ) with hh
  set p := (2 : ℝ)⁻¹ ^ 537 with hp
  have hp0 : 0 ≤ p := by positivity
  have hh0 : 1 / 2 ≤ h := by rw [hh]; linarith
  have hsq : 1 / 2 ≤ Real.sqrt h := half_le_sqrt hN1
  have hmin2 : normalMin ≤ 1 / 4 := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 2 := normalMin_le_pow (by norm_num)
      _ = 1 / 4 := by norm_num
  have hmin541 : normalMin ≤ p * (1 / 16) := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 541 := normalMin_le_pow (by norm_num)
      _ = (2 : ℝ)⁻¹ ^ 537 * (2 : ℝ)⁻¹ ^ 4 := by rw [← pow_add]
      _ = p * (1 / 16) := by rw [hp]; norm_num
  -- the square root
  have a1 : h * (1 - R.u) ≤ R.rnd h := M.le_rnd_normal (normalMin_le_half.trans hh0) le_rfl
  have a2 : Real.sqrt h * (1 - R.u) ≤ Real.sqrt (R.rnd h) := by
    have h2 := Real.sqrt_le_sqrt a1
    rw [Real.sqrt_mul (by linarith)] at h2
    have h3 : 1 - R.u ≤ Real.sqrt (1 - R.u) := by
      apply Real.le_sqrt_of_sq_le
      nlinarith
    exact (mul_le_mul_of_nonneg_left h3 (Real.sqrt_nonneg h)).trans h2
  have q2 : 1 / 4 ≤ Real.sqrt h * (1 - R.u) := by nlinarith
  have a3 := M.le_rnd_normal (hmin2.trans q2) a2
  have q3 : 1 / 8 ≤ Real.sqrt h * (1 - R.u) * (1 - R.u) := by nlinarith
  have q3' : p * (1 / 16) ≤ Real.sqrt h * (1 - R.u) * (1 - R.u) * p := by nlinarith
  have a4 := M.le_rnd_normal (hmin541.trans q3') (mul_le_mul_of_nonneg_right a3 hp0)
  -- the factor `1 + 2 γ̂`
  have hsq1 : 1 / 2 ≤ (1 - R.u) ^ 2 := by nlinarith
  have q5 : R.u ≤ 2 * gam R.u (N + 2) * (1 - R.u) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left hsq1 (by linarith : (0 : ℝ) ≤ 2 * gam R.u (N + 2))]
  have h53 : normalMin ≤ R.u := by rw [M.u_eq]; exact normalMin_le_pow (by norm_num)
  have b2 := M.le_rnd_normal (h53.trans q5) (by linarith : 2 * gam R.u (N + 2) * (1 - R.u) ^ 2 ≤
    2 * gamHat R (N + 2))
  have q6 : 1 ≤ 1 + 2 * gam R.u (N + 2) * (1 - R.u) ^ 3 := by
    have : 0 ≤ 2 * gam R.u (N + 2) * (1 - R.u) ^ 3 := by positivity
    linarith
  have b3 := M.le_rnd_normal (normalMin_le_half.trans (by linarith)) (by
    have : 2 * gam R.u (N + 2) * (1 - R.u) ^ 3 = 2 * gam R.u (N + 2) * (1 - R.u) ^ 2 * (1 - R.u) :=
      by ring
    linarith : 1 + 2 * gam R.u (N + 2) * (1 - R.u) ^ 3 ≤ 1 + R.rnd (2 * gamHat R (N + 2)))
  -- the products
  set X1 := Real.sqrt h * (1 - R.u) * (1 - R.u) * p * (1 - R.u) with hX1
  set X2 := (1 + 2 * gam R.u (N + 2) * (1 - R.u) ^ 3) * (1 - R.u) with hX2
  have hX1lo : p * (1 / 16) ≤ X1 := by
    rw [hX1]
    nlinarith [mul_le_mul_of_nonneg_right q3' h1u]
  have hX2lo : 1 / 2 ≤ X2 := by rw [hX2]; nlinarith
  have hX1p : 0 ≤ X1 := le_trans (by positivity) hX1lo
  have hmin542 : normalMin ≤ p * (1 / 16) * (1 / 2) := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 542 := normalMin_le_pow (by norm_num)
      _ = (2 : ℝ)⁻¹ ^ 537 * (2 : ℝ)⁻¹ ^ 5 := by rw [← pow_add]
      _ = p * (1 / 16) * (1 / 2) := by rw [hp]; ring
  have c1n : normalMin ≤ X1 * X2 :=
    hmin542.trans (mul_le_mul hX1lo hX2lo (by norm_num) hX1p)
  have c1 := M.le_rnd_normal c1n (mul_le_mul a4 b3 (by rw [hX2]; positivity) (hX1p.trans a4))
  have c2n : normalMin ≤ X1 * X2 * (1 - R.u) * (1 + 8 * R.u) := by
    have h0 : 0 ≤ X1 * X2 := le_trans normalMin_pos.le c1n
    calc normalMin ≤ X1 * X2 := c1n
      _ ≤ X1 * X2 * ((1 - R.u) * (1 + 8 * R.u)) := le_mul_of_one_le_right h0 hfac
      _ = _ := by ring
  have c2 := M.le_rnd_normal c2n (mul_le_mul_of_nonneg_right c1 h8)
  have c3n : normalMin ≤ X1 * X2 * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) * (1 + 8 * R.u) := by
    have h0 : 0 ≤ X1 * X2 * (1 - R.u) * (1 + 8 * R.u) := le_trans normalMin_pos.le c2n
    calc normalMin ≤ X1 * X2 * (1 - R.u) * (1 + 8 * R.u) := c2n
      _ ≤ X1 * X2 * (1 - R.u) * (1 + 8 * R.u) * ((1 - R.u) * (1 + 8 * R.u)) :=
          le_mul_of_one_le_right h0 hfac
      _ = _ := by ring
  have c3 := M.le_rnd_normal c3n (mul_le_mul_of_nonneg_right c2 h8)
  unfold normUnder
  refine le_trans (le_of_eq ?_) c3
  rw [hX1, hX2]
  unfold rootN
  ring


/-! ### Numeric facts for the norm bounds -/

theorem num_under_up {u g G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 10 ^ 15) (hg : 0 ≤ g)
    (hgG : g + 2 * u ≤ G) (hG : G ≤ 1 / 1000) (hG2 : G ^ 2 ≤ u) :
    (1 + g) * (1 - u) ^ 3 * ((1 + G * (1 + 3 * u)) * (1 + u)) * (1 + 8 * u) + u / 1000 ≤
      (1 + 2 * G * (1 - u) ^ 3) * (1 - u) ^ 7 * (1 + 8 * u) ^ 2 := by
  have hG0 : 0 ≤ G := by linarith
  have hgG' : g ≤ G := by linarith
  have hgGu : g * G ≤ u := le_trans (mul_le_mul_of_nonneg_right hgG' hG0) (by nlinarith)
  have hA : (1 - u) ^ 3 * (1 + u) * (1 + 8 * u) ≤ 1 + 6 * u := by
    have e : (1 - u) ^ 3 * (1 + u) * (1 + 8 * u) =
        1 + 6 * u - 16 * u ^ 2 + 2 * u ^ 3 + 15 * u ^ 4 - 8 * u ^ 5 := by ring
    rw [e]
    have a1 : u ^ 3 ≤ u ^ 2 / 10 ^ 15 := by
      have := mul_le_mul_of_nonneg_left hu1 (pow_nonneg hu 2)
      calc u ^ 3 = u ^ 2 * u := by ring
        _ ≤ _ := this
        _ = _ := by ring
    have a2 : u ^ 4 ≤ u ^ 3 := by
      have := mul_le_mul_of_nonneg_left (hu1.trans (by norm_num) : u ≤ 1) (pow_nonneg hu 3)
      calc u ^ 4 = u ^ 3 * u := by ring
        _ ≤ u ^ 3 * 1 := this
        _ = u ^ 3 := by ring
    nlinarith [pow_nonneg hu 2, pow_nonneg hu 5]
  have h3 : 1 - 3 * u ≤ (1 - u) ^ 3 := by
    have := one_sub_mul_le_pow hu (by linarith) 3
    push_cast at this
    linarith
  have hB : 1 + 8 * u ≤ (1 - u) ^ 7 * (1 + 8 * u) ^ 2 := by
    have h7 := one_sub_mul_le_pow hu (by linarith) 7
    push_cast at h7
    have h16 : 1 + 16 * u ≤ (1 + 8 * u) ^ 2 := by nlinarith
    have h0 : 0 ≤ 1 - 7 * u := by linarith
    calc 1 + 8 * u ≤ (1 - 7 * u) * (1 + 16 * u) := by nlinarith
      _ ≤ (1 - u) ^ 7 * (1 + 8 * u) ^ 2 :=
          mul_le_mul h7 h16 (by linarith) (pow_nonneg (by linarith) 7)
  have hL : (1 + g) * (1 - u) ^ 3 * ((1 + G * (1 + 3 * u)) * (1 + u)) * (1 + 8 * u) ≤
      (1 + g) * (1 + G * (1 + 3 * u)) * (1 + 6 * u) := by
    have e : (1 + g) * (1 - u) ^ 3 * ((1 + G * (1 + 3 * u)) * (1 + u)) * (1 + 8 * u) =
        (1 + g) * (1 + G * (1 + 3 * u)) * ((1 - u) ^ 3 * (1 + u) * (1 + 8 * u)) := by ring
    rw [e]
    exact mul_le_mul_of_nonneg_left hA (by positivity)
  have hR : (1 + 2 * G * (1 - 3 * u)) * (1 + 8 * u) ≤
      (1 + 2 * G * (1 - u) ^ 3) * (1 - u) ^ 7 * (1 + 8 * u) ^ 2 := by
    have h1 : 1 + 2 * G * (1 - 3 * u) ≤ 1 + 2 * G * (1 - u) ^ 3 := by nlinarith
    have h2 : 0 ≤ 1 + 2 * G * (1 - u) ^ 3 := by
      have : 0 ≤ (1 - u) ^ 3 := pow_nonneg (by linarith) 3
      nlinarith
    calc (1 + 2 * G * (1 - 3 * u)) * (1 + 8 * u)
        ≤ (1 + 2 * G * (1 - u) ^ 3) * ((1 - u) ^ 7 * (1 + 8 * u) ^ 2) :=
          mul_le_mul h1 hB (by linarith) h2
      _ = _ := by ring
  have k1 : u * g ≤ u / 1000 := by nlinarith
  have k2 : u * (g * G) ≤ u / 10 ^ 6 := by
    have : g * G ≤ 1 / 10 ^ 6 := by nlinarith
    nlinarith
  have k3 : u ^ 2 * G ≤ u / 10 ^ 6 := by nlinarith [mul_le_mul_of_nonneg_left hu1 hu]
  have k4 : u ^ 2 * (g * G) ≤ u / 10 ^ 6 := by
    have : g * G ≤ 1 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left hu1 hu]
  have hmid : (1 + g) * (1 + G * (1 + 3 * u)) * (1 + 6 * u) + u / 1000 ≤
      (1 + 2 * G * (1 - 3 * u)) * (1 + 8 * u) := by
    nlinarith [mul_nonneg hu hG0]
  linarith

theorem num_under_low {u g G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 10 ^ 15) (hg : 0 ≤ g)
    (hgG : g + 2 * u ≤ G) (hG : G ≤ 1 / 1000) :
    (1 + g) * (1 + u) ^ 4 + u / 1000 ≤ (1 + 2 * G * (1 - u) ^ 3) * (1 - u) ^ 7 * (1 + 8 * u) ^ 2 := by
  have hG0 : 0 ≤ G := by linarith
  have h4 : (1 + u) ^ 4 ≤ 1 + 5 * u := by
    have e : (1 + u) ^ 4 = 1 + 4 * u + 6 * u ^ 2 + 4 * u ^ 3 + u ^ 4 := by ring
    rw [e]
    nlinarith [pow_nonneg hu 3, pow_nonneg hu 4, mul_le_mul_of_nonneg_left hu1 hu,
      mul_le_mul_of_nonneg_left hu1 (pow_nonneg hu 2), mul_le_mul_of_nonneg_left hu1 (pow_nonneg hu 3)]
  have hB : 1 + 8 * u ≤ (1 - u) ^ 7 * (1 + 8 * u) ^ 2 := by
    have h7 := one_sub_mul_le_pow hu (by linarith) 7
    push_cast at h7
    have h16 : 1 + 16 * u ≤ (1 + 8 * u) ^ 2 := by nlinarith
    calc 1 + 8 * u ≤ (1 - 7 * u) * (1 + 16 * u) := by nlinarith
      _ ≤ (1 - u) ^ 7 * (1 + 8 * u) ^ 2 :=
          mul_le_mul h7 h16 (by linarith) (pow_nonneg (by linarith) 7)
  have h3 : 1 - 3 * u ≤ (1 - u) ^ 3 := by
    have := one_sub_mul_le_pow hu (by linarith) 3
    push_cast at this
    linarith
  have hR : (1 + 2 * G * (1 - 3 * u)) * (1 + 8 * u) ≤
      (1 + 2 * G * (1 - u) ^ 3) * (1 - u) ^ 7 * (1 + 8 * u) ^ 2 := by
    have h1 : 1 + 2 * G * (1 - 3 * u) ≤ 1 + 2 * G * (1 - u) ^ 3 := by nlinarith
    have h2 : 0 ≤ 1 + 2 * G * (1 - u) ^ 3 := by
      have : 0 ≤ (1 - u) ^ 3 := pow_nonneg (by linarith) 3
      nlinarith
    calc (1 + 2 * G * (1 - 3 * u)) * (1 + 8 * u)
        ≤ (1 + 2 * G * (1 - u) ^ 3) * ((1 - u) ^ 7 * (1 + 8 * u) ^ 2) :=
          mul_le_mul h1 hB (by linarith) h2
      _ = _ := by ring
  have hL : (1 + g) * (1 + u) ^ 4 ≤ (1 + g) * (1 + 5 * u) := mul_le_mul_of_nonneg_left h4 (by linarith)
  nlinarith [mul_nonneg hu hG0, mul_nonneg hu hg]

/-! ### The norm chains with the absolute term -/

/-- `np.linalg.norm` under the model: with `σ = N (1 + γ_N) eta`,
`(‖v‖ (1 - γ_N) - √σ)(1 - u) - eta ≤ fl(‖v‖) ≤ (‖v‖ (1 + γ_N) + √σ)(1 + u) + eta`. -/
theorem normFl_U {ι : Type*} [Fintype ι] [DecidableEq ι] (R : Rounding ℝ) (hu1 : R.u ≤ 1)
    {T : STree ι} (hT : FullTree T) (hn : (Fintype.card ι : ℝ) * R.u < 1)
    (hg1 : gam R.u (Fintype.card ι) ≤ 1) (v : ι → ℝ) :
    (l2 v * (1 - gam R.u (Fintype.card ι)) -
        Real.sqrt ((Fintype.card ι : ℝ) * (1 + gam R.u (Fintype.card ι)) * R.eta)) * (1 - R.u) -
        R.eta ≤ normFl R T v ∧ -R.eta ≤ normFl R T v ∧
      normFl R T v ≤ (l2 v * (1 + gam R.u (Fintype.card ι)) +
        Real.sqrt ((Fintype.card ι : ℝ) * (1 + gam R.u (Fintype.card ι)) * R.eta)) * (1 + R.u) +
        R.eta := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  set n := Fintype.card ι with hn_def
  set g := gam R.u n with hg_def
  have hg0 : 0 ≤ g := gam_nonneg hu hn
  have hlen : T.adds + 1 ≤ n := by
    rw [← STree.length_leaves]
    exact hT.1.length_le_card
  have hk : ((T.adds + 1 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hu) hn
  have h := dot_error R v v T hk
  have hev : T.eval (fun j => v j * v j) = ∑ j, v j ^ 2 := by
    rw [STree.eval_eq_finset_sum _ T hT.1 (fun j hj => absurd (hT.2 j) hj)]
    exact Finset.sum_congr rfl fun j _ => (sq (v j)).symm
  have ha : T.absEval (fun j => v j * v j) = ∑ j, v j ^ 2 := by
    rw [STree.absEval_eq_finset_sum _ T hT.1 (fun j hj => absurd (hT.2 j) hj)]
    exact Finset.sum_congr rfl fun j _ => by rw [Real.norm_eq_abs, abs_mul_self, sq]
  rw [hev, ha, Real.norm_eq_abs] at h
  set S := ∑ j, v j ^ 2 with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun j _ => sq_nonneg (v j)
  have hlenr : (T.adds : ℝ) + 1 ≤ n := by exact_mod_cast hlen
  have hgm : gam R.u T.adds ≤ g := gam_mono hu (by omega) hn
  have hgk : gam R.u (T.adds + 1) ≤ g := gam_mono hu hlen hn
  set σ := (n : ℝ) * (1 + g) * R.eta with hσ
  have hσ0 : 0 ≤ σ := by positivity
  have hadd : ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * R.eta ≤ σ := by
    have h1 : 0 ≤ 1 + gam R.u T.adds := by
      have := gam_nonneg hu (k := T.adds) (lt_of_le_of_lt (mul_le_mul_of_nonneg_right
        (by exact_mod_cast (by omega : T.adds ≤ n)) hu) hn)
      linarith
    calc ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * R.eta
        ≤ (n : ℝ) * (1 + g) * R.eta := by
          apply mul_le_mul_of_nonneg_right _ he
          exact mul_le_mul hlenr (by linarith) h1 (Nat.cast_nonneg n)
      _ = σ := rfl
  have hmul : gam R.u (T.adds + 1) * S ≤ g * S := mul_le_mul_of_nonneg_right hgk hS0
  obtain ⟨hlo, hhi⟩ := abs_le.mp (h.trans (add_le_add hmul hadd))
  set Sh := T.comp R.rnd (fun j => R.rnd (v j * v j)) with hSh
  have hl2 : l2 v = Real.sqrt S := l2_eq v
  -- lower bound of the square root
  have hsq_lo : l2 v * (1 - g) - Real.sqrt σ ≤ Real.sqrt Sh := by
    have h1 : S * (1 - g) - σ ≤ Sh := by linarith
    have h2 := sqrt_sub_le' (mul_nonneg hS0 (by linarith)) hσ0 h1
    rw [Real.sqrt_mul hS0] at h2
    have h3 : 1 - g ≤ Real.sqrt (1 - g) := by
      apply Real.le_sqrt_of_sq_le
      nlinarith
    rw [hl2]
    nlinarith [mul_le_mul_of_nonneg_left h3 (Real.sqrt_nonneg S)]
  have hsq_hi : Real.sqrt Sh ≤ l2 v * (1 + g) + Real.sqrt σ := by
    have h1 : Sh ≤ S * (1 + g) + σ := by linarith
    have h2 := (Real.sqrt_le_sqrt h1).trans (sqrt_add_le' (mul_nonneg hS0 (by linarith)) hσ0)
    rw [Real.sqrt_mul hS0] at h2
    have h3 : Real.sqrt (1 + g) ≤ 1 + g := by
      rw [Real.sqrt_le_left (by linarith)]
      nlinarith
    rw [hl2]
    nlinarith [mul_le_mul_of_nonneg_left h3 (Real.sqrt_nonneg S)]
  have herr := R.abs_err (Real.sqrt Sh)
  have hs0 := Real.sqrt_nonneg Sh
  refine ⟨?_, ?_, ?_⟩
  · have := lo_of_err hu hu1 herr hsq_lo le_rfl (by linarith : -(0 : ℝ) ≤ Real.sqrt Sh)
    unfold normFl
    linarith
  · have := lo_of_err hu hu1 herr hs0 le_rfl (by linarith : -(0 : ℝ) ≤ Real.sqrt Sh)
    unfold normFl
    linarith
  · have := hi_of_err hu hu1 herr hsq_hi (by nlinarith [l2_nonneg v, Real.sqrt_nonneg σ])
    unfold normFl
    linarith

/-- The chain of `fp.norm2_upper` with absolute terms. -/
theorem chain_up_U (R : Rounding ℝ) (hu1 : R.u ≤ 1) {N g s c G ch Q tail : ℝ} (hN : 0 ≤ N)
    (hg1 : g ≤ 1) (hQ : 0 ≤ Q) (hs : N * (1 - g) * (1 - R.u) - Q ≤ s) (hs' : -R.eta ≤ s)
    (hc : (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ≤ c) (hch : c ≤ ch) (hG0 : 0 ≤ G)
    (hfl : R.IsFloat tail)
    (htail : Q * (ch * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u)) + R.eta * (2 * R.u * ch *
      (1 + 8 * R.u) * (1 - R.u) + (1 + 8 * R.u) * (1 - R.u) + 2 * R.u * ((ch * (1 + R.u) + 1) *
        (1 + 8 * R.u)) + 1) ≤ tail)
    (hnum : 1 + R.u ≤ (1 - g) * (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ^ 5 * (1 + 8 * R.u)) :
    N * (1 + R.u) ≤ R.rnd (R.rnd (R.rnd (s * c) * (1 + 8 * R.u)) + tail) := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have h1u : 0 ≤ 1 - R.u := by linarith
  have h8 : 0 ≤ 1 + 8 * R.u := by linarith
  set P := N * (1 - g) * (1 - R.u) with hP
  have hP0 : 0 ≤ P := mul_nonneg (mul_nonneg hN (by linarith)) h1u
  set cl := (1 + G * (1 - R.u) ^ 2) * (1 - R.u) with hcl
  have hcl0 : 0 ≤ cl := mul_nonneg (by positivity) h1u
  have hc0 : 0 ≤ c := hcl0.trans hc
  have hch0 : 0 ≤ ch := hc0.trans hch
  -- first product
  have hz1 : P * cl - Q * ch ≤ s * c := by
    have a1 := mul_le_mul_of_nonneg_right hs hc0
    have a2 := mul_le_mul_of_nonneg_left hc hP0
    have a3 := mul_le_mul_of_nonneg_left hch hQ
    nlinarith
  have hz1' : -(R.eta * ch) ≤ s * c := by
    have a1 := mul_le_mul_of_nonneg_right hs' hc0
    have a2 := mul_le_mul_of_nonneg_left hch he
    nlinarith
  have hm1 : 0 ≤ R.eta * ch := mul_nonneg he hch0
  have hx1 := lo_of_err hu hu1 (R.abs_err (s * c)) hz1 hm1 hz1'
  have hx1' := lo_of_err hu hu1 (R.abs_err (s * c)) hz1' hm1 hz1'
  set x1 := R.rnd (s * c) with hx1d
  set m2 := R.eta * ch * (1 + R.u) + R.eta with hm2
  have hm20 : 0 ≤ m2 := by positivity
  have hx1n : -m2 ≤ x1 := by rw [hm2]; nlinarith
  -- second product
  have hz2 : ((P * cl - Q * ch) * (1 - R.u) - 2 * R.u * (R.eta * ch) - R.eta) * (1 + 8 * R.u) ≤
      x1 * (1 + 8 * R.u) := mul_le_mul_of_nonneg_right hx1 h8
  have hz2' : -(m2 * (1 + 8 * R.u)) ≤ x1 * (1 + 8 * R.u) := by
    have := mul_le_mul_of_nonneg_right hx1n h8
    linarith
  have hx2 := lo_of_err hu hu1 (R.abs_err (x1 * (1 + 8 * R.u))) hz2 (mul_nonneg hm20 h8) hz2'
  set x2 := R.rnd (x1 * (1 + 8 * R.u)) with hx2d
  -- the sum with the tail
  set L3 := P * cl * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) with hL3
  have hL30 : 0 ≤ L3 := by positivity
  have hz3 : L3 ≤ x2 + tail := by
    rw [hL3]
    nlinarith
  have hx3 := lo_of_err hu hu1 (R.abs_add_err (R.isFloat_rnd _) hfl) hz3 le_rfl
    (by linarith : -(0 : ℝ) ≤ x2 + tail)
  calc N * (1 + R.u) ≤ N * ((1 - g) * (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ^ 5 * (1 + 8 * R.u)) :=
        mul_le_mul_of_nonneg_left hnum hN
    _ = L3 * (1 - R.u) := by rw [hL3, hP, hcl]; ring
    _ ≤ R.rnd (x2 + tail) := by linarith

/-- The chain of `fp.norm2_lower` with absolute terms. -/
theorem chain_low_U (R : Rounding ℝ) (hu1 : R.u ≤ 1) {N g s c G Q tail : ℝ} (hN : 0 ≤ N)
    (hg0 : 0 ≤ g) (hQ : 0 ≤ Q) (hs : s ≤ N * (1 + g) * (1 + R.u) + Q) (hc0 : 0 ≤ c)
    (hc : c ≤ (1 - G * (1 - R.u) ^ 2) * (1 + R.u)) (hC0 : 0 ≤ 1 - G * (1 - R.u) ^ 2)
    (hfl : R.IsFloat (-tail))
    (htail : (Q * ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) * (1 + R.u) + R.eta) * (1 + R.u) /
      (1 + 8 * R.u) + R.eta ≤ tail)
    (hnum : (1 + g) * (1 - G * (1 - R.u) ^ 2) * (1 + R.u) ^ 5 ≤ (1 - R.u) * (1 + 8 * R.u)) :
    max 0 (R.rnd (R.rnd (R.rnd (s * c) / (1 + 8 * R.u)) - tail)) ≤ N * (1 - R.u) := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have h8 : 0 < 1 + 8 * R.u := by linarith
  have h1 : 0 ≤ 1 + R.u := by linarith
  set P := N * (1 + g) * (1 + R.u) with hP
  have hP0 : 0 ≤ P := by positivity
  set ch := (1 - G * (1 - R.u) ^ 2) * (1 + R.u) with hch
  have hch0 : 0 ≤ ch := mul_nonneg hC0 h1
  have hz1 : s * c ≤ (P + Q) * ch := by
    rcases le_total 0 s with hs0 | hs0
    · calc s * c ≤ s * ch := mul_le_mul_of_nonneg_left hc hs0
        _ ≤ (P + Q) * ch := mul_le_mul_of_nonneg_right hs hch0
    · have : s * c ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hs0 hc0
      have : 0 ≤ (P + Q) * ch := mul_nonneg (by linarith) hch0
      linarith
  have hH1 : 0 ≤ (P + Q) * ch := mul_nonneg (by linarith) hch0
  have hx1 := hi_of_err hu hu1 (R.abs_err (s * c)) hz1 hH1
  set x1 := R.rnd (s * c) with hx1d
  set H2 := ((P + Q) * ch * (1 + R.u) + R.eta) / (1 + 8 * R.u) with hH2
  have hH20 : 0 ≤ H2 := div_nonneg (by positivity) h8.le
  have hz2 : x1 / (1 + 8 * R.u) ≤ H2 := div_le_div_of_nonneg_right hx1 h8.le
  have hx2 := hi_of_err hu hu1 (R.abs_err (x1 / (1 + 8 * R.u))) hz2 hH20
  set x2 := R.rnd (x1 / (1 + 8 * R.u)) with hx2d
  set H3 := P * ch * (1 + R.u) * (1 + R.u) / (1 + 8 * R.u) with hH3
  have hH30 : 0 ≤ H3 := div_nonneg (by positivity) h8.le
  have hsplit : H2 * (1 + R.u) + R.eta =
      H3 + ((Q * ch * (1 + R.u) + R.eta) * (1 + R.u) / (1 + 8 * R.u) + R.eta) := by
    rw [hH2, hH3]
    field_simp
    ring
  have hz3 : x2 + -tail ≤ H3 := by linarith
  have hx3 := hi_of_err hu hu1 (R.abs_add_err (R.isFloat_rnd _) hfl) hz3 hH30
  have hfin : H3 * (1 + R.u) ≤ N * (1 - R.u) := by
    rw [hH3, hP, hch, div_mul_eq_mul_div, div_le_iff₀ h8]
    calc N * (1 + g) * (1 + R.u) * ((1 - G * (1 - R.u) ^ 2) * (1 + R.u)) * (1 + R.u) * (1 + R.u) *
          (1 + R.u) = N * ((1 + g) * (1 - G * (1 - R.u) ^ 2) * (1 + R.u) ^ 5) := by ring
      _ ≤ N * ((1 - R.u) * (1 + 8 * R.u)) := mul_le_mul_of_nonneg_left hnum hN
      _ = N * (1 - R.u) * (1 + 8 * R.u) := by ring
  have key : R.rnd (x2 + -tail) ≤ N * (1 - R.u) := by linarith
  have key' : R.rnd (R.rnd (R.rnd (s * c) / (1 + 8 * R.u)) - tail) ≤ N * (1 - R.u) := by
    rw [sub_eq_add_neg]
    exact key
  exact max_le (mul_nonneg hN (by linarith)) key'

theorem kb_bound {u ch : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) (hch0 : 0 ≤ ch) (hch2 : ch ≤ 2) :
    ch * (1 - u) * (1 + 8 * u) * (1 - u) + (2 * u * ch * (1 + 8 * u) * (1 - u) +
      (1 + 8 * u) * (1 - u) + 2 * u * ((ch * (1 + u) + 1) * (1 + 8 * u)) + 1) ≤ 35 := by
  have ha0 : 0 ≤ 1 - u := by linarith
  have ha1 : 1 - u ≤ 1 := by linarith
  have hb1 : 1 + 8 * u ≤ 2 := by linarith
  have hb0 : 0 ≤ 1 + 8 * u := by linarith
  have p0 : (1 - u) * (1 + 8 * u) ≤ 2 := by nlinarith only [ha0, ha1, hb0, hb1]
  have p0' : 0 ≤ (1 - u) * (1 + 8 * u) := mul_nonneg ha0 hb0
  have hX0 : 0 ≤ ch * ((1 - u) * (1 + 8 * u)) := mul_nonneg hch0 p0'
  have p1 : ch * (1 - u) * (1 + 8 * u) * (1 - u) ≤ 4 := by
    have : ch * ((1 - u) * (1 + 8 * u)) ≤ 2 * 2 := mul_le_mul hch2 p0 p0' (by norm_num)
    nlinarith only [this, ha0, ha1, hX0]
  have p2 : 2 * u * ch * (1 + 8 * u) * (1 - u) ≤ 1 := by
    have : ch * ((1 - u) * (1 + 8 * u)) ≤ 2 * 2 := mul_le_mul hch2 p0 p0' (by norm_num)
    nlinarith only [this, hu, hu1, hX0]
  have p4 : 2 * u * ((ch * (1 + u) + 1) * (1 + 8 * u)) ≤ 1 := by
    have h1 : ch * (1 + u) + 1 ≤ 4 := by nlinarith only [hch2, hch0, hu, hu1]
    have h2 : (ch * (1 + u) + 1) * (1 + 8 * u) ≤ 4 * 2 :=
      mul_le_mul h1 hb1 hb0 (by norm_num)
    nlinarith only [h2, hu, hu1, mul_nonneg (by positivity : (0 : ℝ) ≤ ch * (1 + u) + 1) hb0]
  linarith

theorem tail_low_aux {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) :
    (1 + u) ^ 3 + (1 + u) + 1 ≤ 35 := by
  have h : (1 + u) ^ 3 ≤ 8 := by
    have h1 : 1 + u ≤ 2 := by linarith
    calc (1 + u) ^ 3 ≤ 2 ^ 3 := pow_le_pow_left₀ (by linarith) h1 3
      _ = 8 := by norm_num
  linarith

theorem fullTree_card_pos {ι : Type*} [Fintype ι] {T : STree ι} (hT : FullTree T) :
    1 ≤ Fintype.card ι := by
  have h1 := hT.1.length_le_card
  rw [STree.length_leaves] at h1
  omega

theorem normUnder_isFloat (R : Rounding ℝ) (N : ℕ) : R.IsFloat (normUnder R N) :=
  R.isFloat_rnd _

/-- **`fp.norm2_upper` is an upper bound under the model**, for a vector of any magnitude. -/
theorem norm2Up_ge_U {R : Rounding ℝ} (M : UModel R) {ι : Type*} [Fintype ι] [DecidableEq ι]
    (hn : Fintype.card ι ≤ 2 ^ 26) {T : STree ι} (hT : FullTree T) (v : ι → ℝ) :
    l2 v * (1 + R.u) ≤ norm2Up R T (normUnder R (Fintype.card ι)) v := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hus := M.u_small
  have hu1' : R.u ≤ 1 := by linarith
  have hn1 := fullTree_card_pos hT
  obtain ⟨hu1, hk2, hg0, hs1, hs2, hG, hG2⟩ := small_consts hu M.u_le53 hn
  set n := Fintype.card ι with hn_def
  have hnu : (n : ℝ) * R.u < 1 := by
    have : (n : ℝ) * R.u ≤ ((n + 2 : ℕ) : ℝ) * R.u :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : n ≤ n + 2)) hu
    linarith
  set g := gam R.u n with hg_def
  set G := gam R.u (n + 2) with hG_def
  have hgG : g + 2 * R.u ≤ G := by linarith
  have hG0 : 0 ≤ G := by linarith
  have hg1 : g ≤ 1 := by linarith
  obtain ⟨hs_lo, hs_neg, -⟩ := normFl_U R hu1' hT hnu hg1 v
  obtain ⟨hgh_lo, hgh_hi⟩ := M.gamHat_bounds (k := n + 2) (by omega) (by omega)
  have hgh0 : 0 ≤ gamHat R (n + 2) := le_trans (by positivity) hgh_lo
  have hc_lo : (1 + G * (1 - R.u) ^ 2) * (1 - R.u) ≤ R.rnd (1 + gamHat R (n + 2)) :=
    M.le_rnd_normal (normalMin_le_half.trans (by nlinarith [mul_nonneg hG0 (sq_nonneg (1 - R.u))]))
      (by linarith)
  have hc_hi : R.rnd (1 + gamHat R (n + 2)) ≤ (1 + G * (1 + 3 * R.u)) * (1 + R.u) :=
    M.rnd_le_normal (normalMin_le_half.trans (by linarith)) (by linarith)
  set σ := (n : ℝ) * (1 + g) * R.eta with hσ
  set Q := Real.sqrt σ * (1 - R.u) + R.eta with hQ
  have hQ0 : 0 ≤ Q := by positivity
  have hs : l2 v * (1 - g) * (1 - R.u) - Q ≤ normFl R T v := by rw [hQ]; linarith
  -- the tail
  set r := rootN n with hr
  have hr0 : 0 ≤ r := rootN_nonneg n
  have hsig : Real.sqrt σ ≤ r * (1 + g) := sqrt_sigma_le hg0 M.eta_le
  have hQle : Q ≤ r * (1 + g) * (1 - R.u) + R.eta := by
    rw [hQ]
    nlinarith [mul_le_mul_of_nonneg_right hsig (by linarith : (0 : ℝ) ≤ 1 - R.u)]
  have hX := normUnder_ge M hn1 hn
  have hnumU := num_under_up hu hus hg0 hgG hG hG2
  have h35 := eta_le_rootN M hn1
  set ch := (1 + G * (1 + 3 * R.u)) * (1 + R.u) with hch
  have hch0 : 0 ≤ ch := by positivity
  have hch2 : ch ≤ 2 := by
    rw [hch]
    nlinarith [mul_le_mul_of_nonneg_left hus hG0]
  set K := ch * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) with hK
  have hK0 : 0 ≤ K := by
    rw [hK]
    have : 0 ≤ 1 - R.u := by linarith
    positivity
  set B := 2 * R.u * ch * (1 + 8 * R.u) * (1 - R.u) + (1 + 8 * R.u) * (1 - R.u) +
    2 * R.u * ((ch * (1 + R.u) + 1) * (1 + 8 * R.u)) + 1 with hB
  have hKB : K + B ≤ 35 := by
    rw [hK, hB]
    exact kb_bound hu hu1 hch0 hch2
  have htail : Q * K + R.eta * B ≤ normUnder R n := by
    have a1 : Q * K ≤ (r * (1 + g) * (1 - R.u) + R.eta) * K := mul_le_mul_of_nonneg_right hQle hK0
    have a2 : r * (1 + g) * (1 - R.u) * K =
        r * ((1 + g) * (1 - R.u) ^ 3 * ((1 + G * (1 + 3 * R.u)) * (1 + R.u)) * (1 + 8 * R.u)) := by
      rw [hK, hch]; ring
    have a3 : R.eta * (K + B) ≤ R.eta * 35 := mul_le_mul_of_nonneg_left hKB he
    have a4 := mul_le_mul_of_nonneg_left hnumU hr0
    calc Q * K + R.eta * B ≤ (r * (1 + g) * (1 - R.u) + R.eta) * K + R.eta * B := by
          linarith only [a1]
      _ = r * ((1 + g) * (1 - R.u) ^ 3 * ((1 + G * (1 + 3 * R.u)) * (1 + R.u)) * (1 + 8 * R.u)) +
          R.eta * (K + B) := by rw [add_mul, a2]; ring
      _ ≤ r * ((1 + g) * (1 - R.u) ^ 3 * ((1 + G * (1 + 3 * R.u)) * (1 + R.u)) * (1 + 8 * R.u)) +
          r * R.u / 1000 := by linarith only [a3, h35]
      _ = r * ((1 + g) * (1 - R.u) ^ 3 * ((1 + G * (1 + 3 * R.u)) * (1 + R.u)) * (1 + 8 * R.u) +
          R.u / 1000) := by ring
      _ ≤ _ := a4
      _ ≤ normUnder R n := hX
  have hnum := num_norm_up hu hu1 hg0 hgG hG hG2
  unfold norm2Up
  exact chain_up_U R hu1' (l2_nonneg v) hg1 hQ0 hs hs_neg hc_lo hc_hi hG0
    (normUnder_isFloat R n) htail hnum

/-- **`fp.norm2_lower` is a lower bound under the model**, for a vector of any magnitude. -/
theorem norm2Low_le_U {R : Rounding ℝ} (M : UModel R) {ι : Type*} [Fintype ι] [DecidableEq ι]
    (hn : Fintype.card ι ≤ 2 ^ 26) {T : STree ι} (hT : FullTree T) (v : ι → ℝ) :
    norm2Low R T (normUnder R (Fintype.card ι)) v ≤ l2 v * (1 - R.u) := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hus := M.u_small
  have hu1' : R.u ≤ 1 := by linarith
  have hn1 := fullTree_card_pos hT
  obtain ⟨hu1, hk2, hg0, hs1, hs2, hG, hG2⟩ := small_consts hu M.u_le53 hn
  set n := Fintype.card ι with hn_def
  have hnu : (n : ℝ) * R.u < 1 := by
    have : (n : ℝ) * R.u ≤ ((n + 2 : ℕ) : ℝ) * R.u :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : n ≤ n + 2)) hu
    linarith
  set g := gam R.u n with hg_def
  set G := gam R.u (n + 2) with hG_def
  have hgG : g + 2 * R.u ≤ G := by linarith
  have hG0 : 0 ≤ G := by linarith
  have hg1 : g ≤ 1 := by linarith
  obtain ⟨-, -, hs_hi⟩ := normFl_U R hu1' hT hnu hg1 v
  obtain ⟨hgh_lo, hgh_hi⟩ := M.gamHat_bounds (k := n + 2) (by omega) (by omega)
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hC0 : 0 ≤ 1 - G * (1 - R.u) ^ 2 := by nlinarith [mul_le_mul_of_nonneg_left hsq1 hG0]
  have hw1 : 1 / 2 ≤ 1 - gamHat R (n + 2) := by nlinarith [mul_le_mul_of_nonneg_left hus hG0]
  have hc0 : 0 ≤ R.rnd (1 - gamHat R (n + 2)) := by
    have := M.le_rnd_normal (normalMin_le_half) hw1
    nlinarith
  have hc_hi : R.rnd (1 - gamHat R (n + 2)) ≤ (1 - G * (1 - R.u) ^ 2) * (1 + R.u) :=
    M.rnd_le_normal (normalMin_le_half.trans hw1) (by linarith)
  set σ := (n : ℝ) * (1 + g) * R.eta with hσ
  set Q := Real.sqrt σ * (1 + R.u) + R.eta with hQ
  have hQ0 : 0 ≤ Q := by positivity
  have hs : normFl R T v ≤ l2 v * (1 + g) * (1 + R.u) + Q := by rw [hQ]; linarith
  set r := rootN n with hr
  have hr0 : 0 ≤ r := rootN_nonneg n
  have hsig : Real.sqrt σ ≤ r * (1 + g) := sqrt_sigma_le hg0 M.eta_le
  have hQle : Q ≤ r * (1 + g) * (1 + R.u) + R.eta := by
    rw [hQ]
    nlinarith [mul_le_mul_of_nonneg_right hsig (by linarith : (0 : ℝ) ≤ 1 + R.u)]
  have hX := normUnder_ge M hn1 hn
  have hnumL := num_under_low hu hus hg0 hgG hG
  have h35 := eta_le_rootN M hn1
  set ch := (1 - G * (1 - R.u) ^ 2) * (1 + R.u) with hch
  have hch0 : 0 ≤ ch := mul_nonneg hC0 (by linarith)
  have hch1 : ch ≤ 1 + R.u := by
    rw [hch]
    have : 0 ≤ G * (1 - R.u) ^ 2 := by positivity
    nlinarith
  set Y := (Q * ch * (1 + R.u) + R.eta) * (1 + R.u) with hY
  have hY0 : 0 ≤ Y := by positivity
  have htail : (Q * ch * (1 + R.u) + R.eta) * (1 + R.u) / (1 + 8 * R.u) + R.eta ≤
      normUnder R n := by
    have a1 : Y / (1 + 8 * R.u) ≤ Y := div_le_self hY0 (by linarith)
    have a2 : Q * ch ≤ (r * (1 + g) * (1 + R.u) + R.eta) * (1 + R.u) :=
      mul_le_mul hQle hch1 hch0 (by positivity)
    have a3 : Y ≤ r * ((1 + g) * (1 + R.u) ^ 4) + R.eta * ((1 + R.u) ^ 3 + (1 + R.u)) := by
      have h := mul_le_mul_of_nonneg_right a2 (by positivity : (0 : ℝ) ≤ (1 + R.u) * (1 + R.u))
      calc Y = Q * ch * ((1 + R.u) * (1 + R.u)) + R.eta * (1 + R.u) := by rw [hY]; ring
        _ ≤ (r * (1 + g) * (1 + R.u) + R.eta) * (1 + R.u) * ((1 + R.u) * (1 + R.u)) +
            R.eta * (1 + R.u) := by linarith only [h]
        _ = _ := by ring
    have a4 := tail_low_aux hu hu1
    have a5 := mul_le_mul_of_nonneg_left a4 he
    have a6 := mul_le_mul_of_nonneg_left hnumL hr0
    calc Y / (1 + 8 * R.u) + R.eta ≤ Y + R.eta := by linarith only [a1]
      _ ≤ r * ((1 + g) * (1 + R.u) ^ 4) + R.eta * ((1 + R.u) ^ 3 + (1 + R.u) + 1) := by
          linarith only [a3]
      _ ≤ r * ((1 + g) * (1 + R.u) ^ 4) + r * R.u / 1000 := by linarith only [a5, h35]
      _ = r * ((1 + g) * (1 + R.u) ^ 4 + R.u / 1000) := by ring
      _ ≤ _ := a6
      _ ≤ normUnder R n := hX
  have hfl : R.IsFloat (-normUnder R n) := M.neg _ (normUnder_isFloat R n)
  have hnum := num_norm_low hu hu1 hg0 hgG hG
  unfold norm2Low
  exact chain_low_U R hu1' (l2_nonneg v) hg0 hQ0 hs hc0 hc_hi hC0 hfl htail hnum

/-! ### The residual entry and the rounding bound under the model (Lemma U4) -/

section ResidualU

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

theorem matvec_error_U (R : Rounding ℝ) {T : ι → STree ι} {A : ι → ι → ℝ} {w : ℕ}
    (hT : RowTrees T A w) (hw : (w : ℝ) * R.u < 1) (x : ι → ℝ) (i : ι) :
    |matvecFl R T A x i - ∑ j, A i j * x j| ≤
      gam R.u w * ∑ j, |A i j| * |x j| + (w : ℝ) * (1 + gam R.u w) * R.eta := by
  obtain ⟨hnd, hlen, hsupp⟩ := hT i
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hk : (((T i).adds + 1 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hu) hw
  have h := dot_error R (A i) x (T i) hk
  have hev : (T i).eval (fun j => A i j * x j) = ∑ j, A i j * x j :=
    STree.eval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, zero_mul])
  have ha : (T i).absEval (fun j => A i j * x j) = ∑ j, |A i j| * |x j| := by
    rw [STree.absEval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, zero_mul])]
    exact Finset.sum_congr rfl fun j _ => by rw [Real.norm_eq_abs, abs_mul]
  rw [hev, ha, Real.norm_eq_abs] at h
  have hs : 0 ≤ ∑ j, |A i j| * |x j| := Finset.sum_nonneg fun j _ => by positivity
  have h1 : gam R.u ((T i).adds + 1) * ∑ j, |A i j| * |x j| ≤
      gam R.u w * ∑ j, |A i j| * |x j| := mul_le_mul_of_nonneg_right (gam_mono hu hlen hw) hs
  have hlenr : ((T i).adds : ℝ) + 1 ≤ w := by exact_mod_cast hlen
  have hgm : gam R.u (T i).adds ≤ gam R.u w := gam_mono hu (by omega) hw
  have hg0 : 0 ≤ gam R.u (T i).adds := gam_nonneg hu (lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : (T i).adds ≤ w)) hu) hw)
  have h2 : (((T i).adds : ℝ) + 1) * (1 + gam R.u (T i).adds) * R.eta ≤
      (w : ℝ) * (1 + gam R.u w) * R.eta :=
    mul_le_mul_of_nonneg_right (mul_le_mul hlenr (by linarith) (by linarith) (Nat.cast_nonneg w)) he
  unfold matvecFl
  linarith

theorem absMatvec_ge_U (R : Rounding ℝ) {T : ι → STree ι} {A : ι → ι → ℝ} {w : ℕ}
    (hT : RowTrees T A w) (hw : (w : ℝ) * R.u < 1) (x : ι → ℝ) (i : ι) :
    (∑ j, |A i j| * |x j|) * (1 - gam R.u w) - (w : ℝ) * (1 + gam R.u w) * R.eta ≤
      absMatvecFl R T A x i := by
  obtain ⟨hnd, hlen, hsupp⟩ := hT i
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hk : (((T i).adds + 1 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hu) hw
  have h := dot_error R (fun j => |A i j|) (fun j => |x j|) (T i) hk
  have hev : (T i).eval (fun j => |A i j| * |x j|) = ∑ j, |A i j| * |x j| :=
    STree.eval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, abs_zero, zero_mul])
  have ha : (T i).absEval (fun j => |A i j| * |x j|) = ∑ j, |A i j| * |x j| := by
    rw [STree.absEval_eq_finset_sum _ _ hnd (fun j hj => by rw [hsupp j hj, abs_zero, zero_mul])]
    exact Finset.sum_congr rfl fun j _ => by rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  rw [hev, ha, Real.norm_eq_abs] at h
  have hs : 0 ≤ ∑ j, |A i j| * |x j| := Finset.sum_nonneg fun j _ => by positivity
  have h1 : gam R.u ((T i).adds + 1) * ∑ j, |A i j| * |x j| ≤
      gam R.u w * ∑ j, |A i j| * |x j| := mul_le_mul_of_nonneg_right (gam_mono hu hlen hw) hs
  have hlenr : ((T i).adds : ℝ) + 1 ≤ w := by exact_mod_cast hlen
  have hgm : gam R.u (T i).adds ≤ gam R.u w := gam_mono hu (by omega) hw
  have hg0 : 0 ≤ gam R.u (T i).adds := gam_nonneg hu (lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : (T i).adds ≤ w)) hu) hw)
  have h2 : (((T i).adds : ℝ) + 1) * (1 + gam R.u (T i).adds) * R.eta ≤
      (w : ℝ) * (1 + gam R.u w) * R.eta :=
    mul_le_mul_of_nonneg_right (mul_le_mul hlenr (by linarith) (by linarith) (Nat.cast_nonneg w)) he
  have h3 := (abs_le.mp h).1
  unfold absMatvecFl
  nlinarith

omit [Fintype ι] [DecidableEq ι] in
theorem matvecFl_isFloat (R : Rounding ℝ) (T : ι → STree ι) (A : ι → ι → ℝ) (x : ι → ℝ)
    (i : ι) : R.IsFloat (matvecFl R T A x i) :=
  STree.comp_isFloat R _ (fun _ => R.isFloat_rnd _) (T i)

/-- One residual entry under the model: the additive part is `(w (1 + γ_w) + 1)(1 + u) eta`. -/
theorem resid_entry_U {R : Rounding ℝ} (M : UModel R) {w : ℕ} (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26)
    {A : ι → ι → ℝ} {T : ι → STree ι} (hT : RowTrees T A w) (x : ι → ℝ) (μ : ℝ) (i : ι) :
    |residFl R T A x μ i - (∑ j, A i j * x j - μ * x i)| ≤
      gam R.u (w + 1) * (∑ j, |A i j| * |x j| + |x i| * |μ|) +
        ((w : ℝ) * (1 + gam R.u w) + 1) * (1 + R.u) * R.eta := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, -⟩ := small_consts hu M.u_le53 hw
  have hw2 : ((w + 2 : ℕ) : ℝ) * R.u < 1 := by linarith
  have hw1' : ((w + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 1 ≤ w + 2)) hu) hw2
  have hw0 : (w : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w ≤ w + 2)) hu) hw2
  have hg2u : 2 * R.u + R.u ^ 2 ≤ gam R.u (w + 1) := by
    have h1 := u_le_gam_one hu (by linarith)
    have h2 : ((1 + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 1 + 1 ≤ w + 2)) hu) hw2
    have s : gam R.u 1 + R.u * (1 + gam R.u 1) ≤ gam R.u (1 + 1) := gam_succ hu h2
    have m : gam R.u (1 + 1) ≤ gam R.u (w + 1) := gam_mono hu (by omega) hw1'
    nlinarith
  have hsucc : gam R.u w + R.u * (1 + gam R.u w) ≤ gam R.u (w + 1) := gam_succ hu hw1'
  set a := ∑ j, |A i j| * |x j| with ha_def
  set q := |x i| * |μ| with hq_def
  have ha0 : 0 ≤ a := Finset.sum_nonneg fun j _ => by positivity
  have hq0 : 0 ≤ q := by positivity
  set β := (w : ℝ) * (1 + gam R.u w) * R.eta with hβ
  have hβ0 : 0 ≤ β := by positivity
  have hy := matvec_error_U R hT hw0 x i
  have hya : |∑ j, A i j * x j| ≤ a := (Finset.abs_sum_le_sum_abs _ _).trans
    (le_of_eq (Finset.sum_congr rfl fun j _ => abs_mul _ _))
  set yh := matvecFl R T A x i with hyh
  set mh := R.rnd (x i * μ) with hmh
  have hm : |mh - x i * μ| ≤ R.u * q + R.eta := by
    have := R.abs_err (x i * μ)
    rwa [abs_mul] at this
  have hr := R.abs_add_err (matvecFl_isFloat R T A x i) (M.neg _ (R.isFloat_rnd (x i * μ)))
  have hres : residFl R T A x μ i = R.rnd (yh + -mh) := by
    unfold residFl
    rw [sub_eq_add_neg]
  rw [hres]
  obtain ⟨y1, y2⟩ := abs_le.mp hy
  obtain ⟨m1, m2⟩ := abs_le.mp hm
  obtain ⟨a1, a2⟩ := abs_le.mp hya
  obtain ⟨x1, x2⟩ := abs_le.mp (le_of_eq (abs_mul (x i) μ))
  have hD : |yh + -mh| ≤ a * (1 + gam R.u w) + β + q * (1 + R.u) + R.eta := by
    exact abs_le.mpr ⟨by linarith only [y1, y2, a1, a2, m1, m2, x1, x2],
      by linarith only [y1, y2, a1, a2, m1, m2, x1, x2]⟩
  have hrD := hr.trans (by linarith only [mul_le_mul_of_nonneg_left hD hu] :
    R.u * |yh + -mh| + 0 ≤ R.u * (a * (1 + gam R.u w) + β + q * (1 + R.u) + R.eta))
  obtain ⟨r1, r2⟩ := abs_le.mp hrD
  have hxm : x i * μ = μ * x i := mul_comm _ _
  have key : |R.rnd (yh + -mh) - (∑ j, A i j * x j - μ * x i)| ≤
      (gam R.u w + R.u * (1 + gam R.u w)) * a + (2 * R.u + R.u ^ 2) * q + (β + R.eta) * (1 + R.u) := by
    rw [← hxm]
    refine abs_le.mpr ⟨?_, ?_⟩
    · linarith only [r1, y1, m2]
    · linarith only [r2, y2, m1]
  have e : (β + R.eta) * (1 + R.u) = ((w : ℝ) * (1 + gam R.u w) + 1) * (1 + R.u) * R.eta := by
    rw [hβ]; ring
  calc _ ≤ _ := key
    _ ≤ gam R.u (w + 1) * (a + q) + ((w : ℝ) * (1 + gam R.u w) + 1) * (1 + R.u) * R.eta := by
        rw [e]
        linarith only [mul_le_mul_of_nonneg_right hsucc ha0, mul_le_mul_of_nonneg_right hg2u hq0]

end ResidualU


/-- The absolute part of the computed `E_i` against `c_w`. -/
theorem num_D {u W G : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 10 ^ 15) (hW : 1 ≤ W) (hG0 : 0 ≤ G) :
    ((((101 / 100 * W * (1 + u) + 1) * (1 + 4 * u) * (1 + u) + 2) * (1 + u) * G * (1 + 3 * u) *
      (1 + u) + 1) * (1 + 4 * u) * (1 + u) + 1) ≤
      2 * G * W * (1 + 4 * u) + 3 * G * (1 + 4 * u) ^ 2 + 2 * (1 + 4 * u) := by
  have hW0 : 0 ≤ W := by linarith
  have r1 : (1 + 4 * u) * (1 + u) ≤ 10001 / 10000 := by nlinarith only [hu, hu1]
  have r1' : 1 ≤ (1 + 4 * u) * (1 + u) := by nlinarith only [hu]
  have hu2 : u * u ≤ u := by nlinarith only [hu, hu1]
  have hu3 : u * u * u ≤ u := by nlinarith only [hu, hu2, mul_nonneg hu hu]
  have r2 : (1 + u) * (1 + 3 * u) * (1 + u) ≤ 10001 / 10000 := by
    nlinarith only [hu, hu1, hu2, hu3]
  have s1 : 101 / 100 * W * (1 + u) ≤ 102 / 100 * W := by nlinarith only [hu, hu1, hW0]
  set Bv := (101 / 100 * W * (1 + u) + 1) * (1 + 4 * u) * (1 + u) + 2 with hBv
  have hBv0 : 0 ≤ Bv := by rw [hBv]; positivity
  have s3 : Bv ≤ (102 / 100 * W + 1) * (10001 / 10000) + 2 := by
    have t : (101 / 100 * W * (1 + u) + 1) * ((1 + 4 * u) * (1 + u)) ≤
        (102 / 100 * W + 1) * (10001 / 10000) :=
      mul_le_mul (by linarith only [s1]) r1 (by linarith only [r1']) (by positivity)
    rw [hBv]
    nlinarith only [t]
  have s4 : Bv * (1 + u) * G * (1 + 3 * u) * (1 + u) ≤
      ((102 / 100 * W + 1) * (10001 / 10000) + 2) * G * (10001 / 10000) := by
    have t : Bv * G * ((1 + u) * (1 + 3 * u) * (1 + u)) ≤
        ((102 / 100 * W + 1) * (10001 / 10000) + 2) * G * (10001 / 10000) :=
      mul_le_mul (mul_le_mul_of_nonneg_right s3 hG0) r2 (by positivity) (by positivity)
    nlinarith only [t]
  set Z := Bv * (1 + u) * G * (1 + 3 * u) * (1 + u) with hZ
  have hZ0 : 0 ≤ Z := by rw [hZ]; positivity
  have s5 : (Z + 1) * (1 + 4 * u) * (1 + u) + 1 ≤
      ((102 / 100 * W + 1) * (10001 / 10000) + 2) * G * (10001 / 10000) * (10001 / 10000) +
        ((1 + 4 * u) * (1 + u) + 1) := by
    have t : Z * ((1 + 4 * u) * (1 + u)) ≤
        ((102 / 100 * W + 1) * (10001 / 10000) + 2) * G * (10001 / 10000) * (10001 / 10000) :=
      mul_le_mul s4 r1 (by linarith only [r1']) (by positivity)
    nlinarith only [t]
  have s6 : ((102 / 100 * W + 1) * (10001 / 10000) + 2) * G * (10001 / 10000) * (10001 / 10000) ≤
      2 * G * W * (1 + 4 * u) + 3 * G * (1 + 4 * u) ^ 2 := by
    have t : ((102 / 100 * W + 1) * (10001 / 10000) + 2) * (10001 / 10000) * (10001 / 10000) ≤
        2 * W + 3 := by nlinarith only [hW]
    have t2 : 2 * W + 3 ≤ 2 * W * (1 + 4 * u) + 3 * (1 + 4 * u) ^ 2 := by
      nlinarith only [hu, hW0, mul_nonneg hW0 hu]
    nlinarith only [mul_le_mul_of_nonneg_left (t.trans t2) hG0]
  have s7 : (1 + 4 * u) * (1 + u) + 1 ≤ 2 * (1 + 4 * u) := by nlinarith only [hu, hu1]
  linarith only [s5, s6, s7]

section RoundBoundU

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **The computed `E_i` under the model**: it falls short of the standard-model bound
`γ_{w+1} ((|A| |x|)_i + |x_i| |μ|)` by at most
`(2 γ_{w+2} w (1 + γ_w)(1 + 4u) + 3 γ_{w+2} (1 + 4u)² + 2 (1 + 4u)) eta`. -/
theorem roundBound_ge_U {R : Rounding ℝ} (M : UModel R) {w : ℕ} (hw1 : 1 ≤ w)
    (hw : w ≤ 2 ^ 26) {A : ι → ι → ℝ} {T' : ι → STree ι} (hT' : RowTrees T' A w) (x : ι → ℝ)
    (μ : ℝ) (i : ι) :
    gam R.u (w + 1) * (∑ j, |A i j| * |x j| + |x i| * |μ|) -
      (2 * gam R.u (w + 2) * w * (1 + gam R.u w) * (1 + 4 * R.u) +
        3 * gam R.u (w + 2) * (1 + 4 * R.u) ^ 2 + 2 * (1 + 4 * R.u)) * R.eta ≤
      roundBoundFl R T' w A x μ i := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hus := M.u_small
  have hu1' : R.u ≤ 1 := by linarith
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, -⟩ := small_consts hu M.u_le53 hw
  have hw2 : ((w + 2 : ℕ) : ℝ) * R.u < 1 := by linarith
  have hw0 : (w : ℝ) * R.u < 1 := lt_of_le_of_lt
    (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w ≤ w + 2)) hu) hw2
  set gw := gam R.u w with hgw
  set g1 := gam R.u (w + 1) with hg1
  set G := gam R.u (w + 2) with hG
  have hg10 : 0 ≤ g1 := by linarith
  have hG0 : 0 ≤ G := by linarith
  have hg1s : g1 ≤ 1 / 1000 := by linarith
  have hgw1' : gw ≤ 1 := by linarith
  obtain ⟨hγ1lo, hγ1hi⟩ := M.gamHat_bounds (k := w + 1) (by omega) (by omega)
  obtain ⟨hγ2lo, hγ2hi⟩ := M.gamHat_bounds (k := w + 2) (by omega) (by omega)
  set a := ∑ j, |A i j| * |x j| with ha_def
  set q := |x i| * |μ| with hq_def
  have ha0 : 0 ≤ a := Finset.sum_nonneg fun j _ => by positivity
  have hq0 : 0 ≤ q := by positivity
  set W := (w : ℝ) * (1 + gw) with hW
  have hW1 : 1 ≤ W := by
    have : (1 : ℝ) ≤ w := by exact_mod_cast hw1
    rw [hW]; nlinarith
  set β := W * R.eta with hβ
  have hβ0 : 0 ≤ β := by positivity
  have hah := absMatvec_ge_U R hT' hw0 x i
  set ah := absMatvecFl R T' A x i with hahd
  have hah' : a * (1 - gw) - β ≤ ah := by rw [hβ, hW]; linarith only [hah]
  -- the divisor `fl(1 - γ̂_{w+1})`
  have hz0 : 1 / 2 ≤ 1 - g1 * (1 + 3 * R.u) := by nlinarith only [hg1s, hg10, hus, hu]
  have hdw : 1 - g1 * (1 + 3 * R.u) ≤ 1 - gamHat R (w + 1) := by linarith only [hγ1hi]
  have hd_lo : (1 - g1 * (1 + 3 * R.u)) * (1 - R.u) ≤ R.rnd (1 - gamHat R (w + 1)) :=
    M.le_rnd_normal (normalMin_le_half.trans hz0) hdw
  have hd_hi : R.rnd (1 - gamHat R (w + 1)) ≤ (1 - g1 * (1 - R.u) ^ 2) * (1 + R.u) :=
    M.rnd_le_normal (normalMin_le_half.trans (hz0.trans hdw)) (by linarith only [hγ1lo])
  set d := R.rnd (1 - gamHat R (w + 1)) with hd
  set dl := (1 - g1 * (1 + 3 * R.u)) * (1 - R.u) with hdl
  set dh := (1 - g1 * (1 - R.u) ^ 2) * (1 + R.u) with hdh
  have hdl1 : 100 / 101 ≤ dl := by
    rw [hdl]; nlinarith only [hg1s, hg10, hus, hu, mul_nonneg hg10 hu]
  have hd0 : 0 < d := by linarith only [hd_lo, hdl1]
  have hdh0 : 0 < dh := lt_of_lt_of_le hd0 hd_hi
  have hdl0 : 0 < dl := by linarith only [hdl1]
  set X := a * (1 - gw) / dh with hX
  set Y := β / dl with hY
  have hY0 : 0 ≤ Y := div_nonneg hβ0 hdl0.le
  have hag : 0 ≤ a * (1 - gw) := mul_nonneg ha0 (by linarith only [hgw1'])
  have hz1 : X - Y ≤ ah / d := by
    have t1 : (a * (1 - gw) - β) / d ≤ ah / d := div_le_div_of_nonneg_right hah' hd0.le
    have t2 : X ≤ a * (1 - gw) / d := div_le_div_of_nonneg_left hag hd0 hd_hi
    have t3 : β / d ≤ Y := div_le_div_of_nonneg_left hβ0 hdl0 hd_lo
    have t4 : (a * (1 - gw) - β) / d = a * (1 - gw) / d - β / d := sub_div _ _ _
    linarith only [t1, t2, t3, t4]
  have hz1' : -Y ≤ ah / d := by
    have t1 : -β ≤ ah := by linarith only [hah', hag]
    have t2 : -β / d ≤ ah / d := div_le_div_of_nonneg_right t1 hd0.le
    have t3 : β / d ≤ Y := div_le_div_of_nonneg_left hβ0 hdl0 hd_lo
    have t4 : -β / d = -(β / d) := neg_div _ _
    linarith only [t2, t3, t4]
  have hb1 := lo_of_err hu hu1' (R.abs_err (ah / d)) hz1 hY0 hz1'
  have hb1' := lo_of_err hu hu1' (R.abs_err (ah / d)) hz1' hY0 hz1'
  set b1 := R.rnd (ah / d) with hb1d
  have h4 : 0 ≤ 1 + 4 * R.u := by linarith
  set m2 := (Y * (1 + R.u) + R.eta) * (1 + 4 * R.u) with hm2
  have hm20 : 0 ≤ m2 := by positivity
  have hz2 : ((X - Y) * (1 - R.u) - 2 * R.u * Y - R.eta) * (1 + 4 * R.u) ≤ b1 * (1 + 4 * R.u) :=
    mul_le_mul_of_nonneg_right hb1 h4
  have hz2' : -m2 ≤ b1 * (1 + 4 * R.u) := by
    have t := mul_le_mul_of_nonneg_right hb1' h4
    have e : (-Y * (1 - R.u) - 2 * R.u * Y - R.eta) * (1 + 4 * R.u) = -m2 := by rw [hm2]; ring
    linarith only [t, e]
  have hb2 := lo_of_err hu hu1' (R.abs_err (b1 * (1 + 4 * R.u))) hz2 hm20 hz2'
  have hb2' := lo_of_err hu hu1' (R.abs_err (b1 * (1 + 4 * R.u))) hz2' hm20 hz2'
  set b2 := R.rnd (b1 * (1 + 4 * R.u)) with hb2d
  set S2 := m2 * (1 + R.u) + R.eta with hS2
  have hXa : a ≤ X * (1 - R.u) ^ 2 * (1 + 4 * R.u) := by
    have hnum := num_abs hu hu1 hgw0 hgw1 hg1s
    have e : X * (1 - R.u) ^ 2 * (1 + 4 * R.u) =
        a * ((1 - gw) * (1 - R.u) ^ 2 * (1 + 4 * R.u)) / dh := by rw [hX]; ring
    rw [e, le_div_iff₀ hdh0]
    exact mul_le_mul_of_nonneg_left hnum ha0
  have hb2a : a - S2 ≤ b2 := by
    have e : ((X - Y) * (1 - R.u) - 2 * R.u * Y - R.eta) * (1 + 4 * R.u) * (1 - R.u) -
        2 * R.u * m2 - R.eta = X * (1 - R.u) ^ 2 * (1 + 4 * R.u) - S2 := by rw [hS2, hm2]; ring
    linarith only [hb2, e, hXa]
  have hb2n : -S2 ≤ b2 := by
    have e : -m2 * (1 - R.u) - 2 * R.u * m2 - R.eta = -S2 := by rw [hS2]; ring
    linarith only [hb2', e]
  -- `fl(|x_i| |μ|)`
  have hq := lo_of_err hu hu1' (R.abs_err (|x i| * |μ|)) (le_refl q) le_rfl
    (by linarith only [hq0] : -(0 : ℝ) ≤ |x i| * |μ|)
  set qh := R.rnd (|x i| * |μ|) with hqh
  have hqu : 0 ≤ q * (1 - R.u) := mul_nonneg hq0 (by linarith)
  have hau : 0 ≤ a * R.u := mul_nonneg ha0 hu
  have hS20 : 0 ≤ S2 + R.eta := by rw [hS2]; positivity
  have hz3 : (a + q) * (1 - R.u) - (S2 + R.eta) ≤ b2 + qh := by linarith only [hb2a, hq, hau]
  have hz3' : -(S2 + R.eta) ≤ b2 + qh := by linarith only [hb2n, hq, hqu]
  have hb3 := lo_of_err hu hu1' (R.abs_add_err (R.isFloat_rnd _) (R.isFloat_rnd _)) hz3 hS20 hz3'
  have hb3' := lo_of_err hu hu1' (R.abs_add_err (R.isFloat_rnd _) (R.isFloat_rnd _)) hz3' hS20 hz3'
  set b3 := R.rnd (b2 + qh) with hb3d
  set P3 := (a + q) * (1 - R.u) ^ 2 with hP3
  set Q3 := (S2 + R.eta) * (1 + R.u) with hQ3
  have hP30 : 0 ≤ P3 := by rw [hP3]; positivity
  have hQ30 : 0 ≤ Q3 := by rw [hQ3]; positivity
  have hb3a : P3 - Q3 ≤ b3 := by
    have e : ((a + q) * (1 - R.u) - (S2 + R.eta)) * (1 - R.u) - 2 * R.u * (S2 + R.eta) - 0 =
        P3 - Q3 := by rw [hP3, hQ3]; ring
    linarith only [hb3, e]
  have hb3n : -Q3 ≤ b3 := by
    have e : -(S2 + R.eta) * (1 - R.u) - 2 * R.u * (S2 + R.eta) - 0 = -Q3 := by rw [hQ3]; ring
    linarith only [hb3', e]
  set gh := gamHat R (w + 2) with hgh
  have hgh0 : 0 ≤ gh := le_trans (by positivity) hγ2lo
  set m4 := Q3 * (G * (1 + 3 * R.u)) with hm4
  have hm40 : 0 ≤ m4 := by rw [hm4]; positivity
  have hz4 : P3 * (G * (1 - R.u) ^ 2) - m4 ≤ b3 * gh := by
    have t1 := mul_le_mul_of_nonneg_right hb3a hgh0
    have t2 := mul_le_mul_of_nonneg_left hγ2lo hP30
    have t3 := mul_le_mul_of_nonneg_left hγ2hi hQ30
    rw [hm4]
    linarith only [t1, t2, t3]
  have hz4' : -m4 ≤ b3 * gh := by
    have t1 := mul_le_mul_of_nonneg_right hb3n hgh0
    have t3 := mul_le_mul_of_nonneg_left hγ2hi hQ30
    rw [hm4]
    linarith only [t1, t3]
  have hb4 := lo_of_err hu hu1' (R.abs_err (b3 * gh)) hz4 hm40 hz4'
  have hb4' := lo_of_err hu hu1' (R.abs_err (b3 * gh)) hz4' hm40 hz4'
  set b4 := R.rnd (b3 * gh) with hb4d
  have hb4a : (a + q) * G * (1 - R.u) ^ 5 - (m4 * (1 + R.u) + R.eta) ≤ b4 := by
    have e : (P3 * (G * (1 - R.u) ^ 2) - m4) * (1 - R.u) - 2 * R.u * m4 - R.eta =
        (a + q) * G * (1 - R.u) ^ 5 - (m4 * (1 + R.u) + R.eta) := by rw [hP3]; ring
    linarith only [hb4, e]
  have hb4n : -(m4 * (1 + R.u) + R.eta) ≤ b4 := by
    have e : -m4 * (1 - R.u) - 2 * R.u * m4 - R.eta = -(m4 * (1 + R.u) + R.eta) := by ring
    linarith only [hb4', e]
  set m5 := (m4 * (1 + R.u) + R.eta) * (1 + 4 * R.u) with hm5
  have hm50 : 0 ≤ m5 := by rw [hm5]; positivity
  have hz5 : ((a + q) * G * (1 - R.u) ^ 5 - (m4 * (1 + R.u) + R.eta)) * (1 + 4 * R.u) ≤
      b4 * (1 + 4 * R.u) := mul_le_mul_of_nonneg_right hb4a h4
  have hz5' : -m5 ≤ b4 * (1 + 4 * R.u) := by
    have t := mul_le_mul_of_nonneg_right hb4n h4
    rw [hm5]
    linarith only [t]
  have hE := lo_of_err hu hu1' (R.abs_err (b4 * (1 + 4 * R.u))) hz5 hm50 hz5'
  have hEa : (a + q) * G * (1 - R.u) ^ 6 * (1 + 4 * R.u) - (m5 * (1 + R.u) + R.eta) ≤
      R.rnd (b4 * (1 + 4 * R.u)) := by
    have e : ((a + q) * G * (1 - R.u) ^ 5 - (m4 * (1 + R.u) + R.eta)) * (1 + 4 * R.u) *
        (1 - R.u) - 2 * R.u * m5 - R.eta =
        (a + q) * G * (1 - R.u) ^ 6 * (1 + 4 * R.u) - (m5 * (1 + R.u) + R.eta) := by
      rw [hm5]; ring
    linarith only [hE, e]
  -- the relative part
  have hnE := num_E hu hu1 hg10 hgw2 hGw
  have hmain : g1 * (a + q) ≤ (a + q) * G * (1 - R.u) ^ 6 * (1 + 4 * R.u) := by
    have := mul_le_mul_of_nonneg_right hnE (add_nonneg ha0 hq0)
    linarith only [this]
  -- the absolute part
  have hYb : Y ≤ 101 / 100 * W * R.eta := by
    rw [hY, hβ, div_le_iff₀ hdl0]
    have hWe : 0 ≤ W * R.eta := by positivity
    nlinarith only [hWe, hdl1]
  have hm2b : m2 ≤ (101 / 100 * W * R.eta * (1 + R.u) + R.eta) * (1 + 4 * R.u) := by
    rw [hm2]
    exact mul_le_mul_of_nonneg_right (by nlinarith only [hYb, hu]) h4
  set Bv := (101 / 100 * W * (1 + R.u) + 1) * (1 + 4 * R.u) * (1 + R.u) + 2 with hBv
  have hS2b : S2 + R.eta ≤ R.eta * Bv := by
    have t := mul_le_mul_of_nonneg_right hm2b (by linarith : (0 : ℝ) ≤ 1 + R.u)
    rw [hS2, hBv]
    nlinarith only [t]
  have hm4b : m4 ≤ R.eta * Bv * (1 + R.u) * G * (1 + 3 * R.u) := by
    rw [hm4, hQ3]
    have t := mul_le_mul_of_nonneg_right hS2b (by positivity :
      (0 : ℝ) ≤ (1 + R.u) * (G * (1 + 3 * R.u)))
    nlinarith only [t]
  have hm5b : m5 * (1 + R.u) + R.eta ≤
      R.eta * ((((101 / 100 * W * (1 + R.u) + 1) * (1 + 4 * R.u) * (1 + R.u) + 2) * (1 + R.u) * G *
        (1 + 3 * R.u) * (1 + R.u) + 1) * (1 + 4 * R.u) * (1 + R.u) + 1) := by
    rw [hm5]
    have t := mul_le_mul_of_nonneg_right hm4b (by positivity :
      (0 : ℝ) ≤ (1 + R.u) * ((1 + 4 * R.u) * (1 + R.u)))
    rw [hBv] at t
    nlinarith only [t]
  have hnD := num_D hu hus hW1 hG0
  have hD : m5 * (1 + R.u) + R.eta ≤ (2 * G * w * (1 + gw) * (1 + 4 * R.u) +
      3 * G * (1 + 4 * R.u) ^ 2 + 2 * (1 + 4 * R.u)) * R.eta := by
    have t := mul_le_mul_of_nonneg_left hnD he
    have e : 2 * G * W * (1 + 4 * R.u) = 2 * G * w * (1 + gw) * (1 + 4 * R.u) := by rw [hW]; ring
    rw [e] at t
    linarith only [hm5b, t]
  unfold roundBoundFl
  linarith only [hEa, hmain, hD]

/-- The constant `c_w` of `fp.residual_entry_count` (real data), in exact arithmetic. -/
noncomputable def cwExact (u : ℝ) (w : ℕ) : ℝ :=
  ((w : ℝ) + 1) * (1 + gam u (w + 1)) + 2 * gam u (w + 2) * w * (1 + gam u w) * (1 + 4 * u) +
    3 * gam u (w + 2) * (1 + 4 * u) ^ 2 + 2 * (1 + 4 * u)

/-- **Lemma U4** (real data): `|R̂_i - r_i| ≤ Ê_i + c_w eta` for inputs of any magnitude. -/
theorem resid_le_roundBound_U {R : Rounding ℝ} (M : UModel R) {w : ℕ} (hw1 : 1 ≤ w)
    (hw : w ≤ 2 ^ 26) {A : ι → ι → ℝ} {T T' : ι → STree ι} (hT : RowTrees T A w)
    (hT' : RowTrees T' A w) (x : ι → ℝ) (μ : ℝ) (i : ι) :
    |residFl R T A x μ i - (∑ j, A i j * x j - μ * x i)| ≤
      roundBoundFl R T' w A x μ i + cwExact R.u w * R.eta := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, -⟩ := small_consts hu M.u_le53 hw
  have h1 := resid_entry_U M hw1 hw hT x μ i
  have h2 := roundBound_ge_U M hw1 hw hT' x μ i
  have hw1r : (0 : ℝ) ≤ w := Nat.cast_nonneg w
  have hu_g1 : R.u ≤ gam R.u (w + 1) := by linarith
  have hD1 : ((w : ℝ) * (1 + gam R.u w) + 1) * (1 + R.u) ≤
      ((w : ℝ) + 1) * (1 + gam R.u (w + 1)) := by
    have hw2 : ((w + 2 : ℕ) : ℝ) * R.u < 1 := by linarith
    have hw1' : ((w + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 1 ≤ w + 2)) hu) hw2
    have t1 : (1 + gam R.u w) * (1 + R.u) ≤ 1 + gam R.u (w + 1) := by
      have := gam_succ hu (k := w) hw1'
      nlinarith only [this]
    have t2 := mul_le_mul_of_nonneg_left t1 hw1r
    nlinarith only [t2, hu_g1]
  have hD1' := mul_le_mul_of_nonneg_right hD1 he
  unfold cwExact
  nlinarith only [h1, h2, hD1']

end RoundBoundU

/-! ### `residual_entry_count` and `residual_additive_term` as computed -/

/-- `fp.residual_entry_count(w)` for real data, left to right, every operation rounded:
`(w + 1)(1 + γ̂_{w+1}) + 2 γ̂_{w+2} w (1 + γ̂_w)(1 + 4u) + 3 γ̂_{w+2} (1 + 4u)² + 2 (1 + 4u)`. -/
noncomputable def cwHat (R : Rounding ℝ) (w : ℕ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (((w + 1 : ℕ) : ℝ) * R.rnd (1 + gamHat R (w + 1))) +
      R.rnd (R.rnd (R.rnd (R.rnd (2 * gamHat R (w + 2)) * w) * R.rnd (1 + gamHat R w)) *
        (1 + 4 * R.u))) +
    R.rnd (R.rnd (3 * gamHat R (w + 2)) * R.rnd ((1 + 4 * R.u) ^ 2))) + R.rnd (2 * (1 + 4 * R.u)))

/-- `fp.residual_additive_term(n, w)`:
`ldexp(sqrt(n) c_w (1 + 8u)(1 + 8u), -1075) + ULP_MIN`, every operation rounded. -/
noncomputable def addTerm (R : Rounding ℝ) (n w : ℕ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (Real.sqrt n) * cwHat R w) * (1 + 8 * R.u)) *
    (1 + 8 * R.u)) * (2 : ℝ)⁻¹ ^ 1075) + (2 : ℝ)⁻¹ ^ 1074)

/-- The lower bound of the computed `c_w`. -/
noncomputable def cwLow (u : ℝ) (w : ℕ) : ℝ :=
  ((((w : ℝ) + 1) * (1 + gam u (w + 1) * (1 - u) ^ 2) * (1 - u) ^ 2 +
    2 * gam u (w + 2) * w * (1 + gam u w * (1 - u) ^ 2) * (1 + 4 * u) * (1 - u) ^ 7) * (1 - u) +
    3 * gam u (w + 2) * (1 + 4 * u) ^ 2 * (1 - u) ^ 5) * (1 - u) * (1 - u) +
    2 * (1 + 4 * u) * (1 - u) * (1 - u)

theorem pow_lower {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 10 ^ 15) (k : ℕ) (hk : k ≤ 16) :
    1 - (k : ℝ) * u ≤ (1 - u) ^ k ∧ 0 ≤ 1 - (k : ℝ) * u := by
  have h := one_sub_mul_le_pow hu (by linarith) k
  have hk' : (k : ℝ) ≤ 16 := by exact_mod_cast hk
  exact ⟨h, by nlinarith⟩

/-- `(1 - u)^k (1 + 8u)² ≥ 1 + (16 - k) u - 16 k u²`. -/
theorem pow_guard {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 10 ^ 15) (k : ℕ) (hk : k ≤ 16) :
    1 + (16 - (k : ℝ)) * u - 16 * k * u ^ 2 ≤ (1 - u) ^ k * (1 + 8 * u) ^ 2 := by
  obtain ⟨h1, h0⟩ := pow_lower hu hu1 k hk
  have h16 : 1 + 16 * u ≤ (1 + 8 * u) ^ 2 := by nlinarith
  calc 1 + (16 - (k : ℝ)) * u - 16 * k * u ^ 2 = (1 - (k : ℝ) * u) * (1 + 16 * u) := by ring
    _ ≤ (1 - u) ^ k * (1 + 8 * u) ^ 2 :=
        mul_le_mul h1 h16 (by linarith) (pow_nonneg (by linarith) k)

/-- **The computed `c_w` has the slack the additive term needs**:
`cwLow (1 - u)⁶ (1 + 8u)² ≥ c_w (1 + u) + u`, term by term. -/
theorem num_cw {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 10 ^ 15) {w : ℝ} (hw1 : 1 ≤ w)
    {gw g1 G : ℝ} (hgw0 : 0 ≤ gw) (hgw1 : gw + u ≤ g1) (hg1 : g1 + u ≤ G) (hG : G ≤ 1 / 1000)
    (hGw : G * w ≤ 1) :
    (w + 1) * (1 + g1) * (1 + u) + 2 * G * w * (1 + gw) * (1 + 4 * u) * (1 + u) +
        3 * G * (1 + 4 * u) ^ 2 * (1 + u) + 2 * (1 + 4 * u) * (1 + u) + u ≤
      ((w + 1) * (1 + g1 * (1 - u) ^ 2) * (1 - u) ^ 2 * ((1 - u) ^ 9 * (1 + 8 * u) ^ 2) +
        2 * G * w * (1 + gw * (1 - u) ^ 2) * (1 + 4 * u) * ((1 - u) ^ 16 * (1 + 8 * u) ^ 2) +
        3 * G * (1 + 4 * u) ^ 2 * ((1 - u) ^ 13 * (1 + 8 * u) ^ 2) +
        2 * (1 + 4 * u) * ((1 - u) ^ 8 * (1 + 8 * u) ^ 2)) := by
  have hg10 : 0 ≤ g1 := by linarith
  have hG0 : 0 ≤ G := by linarith
  have hw0 : 0 ≤ w := by linarith
  have hu2 : u ^ 2 ≤ u / 10 ^ 15 := by nlinarith
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith
  -- (a) the first term
  have a1 := pow_guard hu hu1 11 (by norm_num)
  push_cast at a1
  have a2 : 1 + 4 * u ≤ (1 - u) ^ 2 * ((1 - u) ^ 9 * (1 + 8 * u) ^ 2) := by
    have e : (1 - u) ^ 2 * ((1 - u) ^ 9 * (1 + 8 * u) ^ 2) = (1 - u) ^ 11 * (1 + 8 * u) ^ 2 := by
      ring
    rw [e]
    nlinarith only [a1, hu2, hu]
  have a3 : (1 + g1) * (1 + u) ≤ (1 + g1 * (1 - u) ^ 2) * (1 + 4 * u) := by
    have t1 := mul_le_mul_of_nonneg_left hsq hg10
    have t2 : 0 ≤ g1 * u := mul_nonneg hg10 hu
    have t3 : g1 * u * u ≤ g1 * u / 1000 := by nlinarith only [t2, hu, hu1]
    nlinarith only [t1, t2, t3, hu, mul_nonneg t2 hu, hg10, hG]
  have ha : (w + 1) * (1 + g1) * (1 + u) ≤
      (w + 1) * (1 + g1 * (1 - u) ^ 2) * (1 - u) ^ 2 * ((1 - u) ^ 9 * (1 + 8 * u) ^ 2) := by
    have t3 : (1 + g1 * (1 - u) ^ 2) * (1 + 4 * u) ≤
        (1 + g1 * (1 - u) ^ 2) * ((1 - u) ^ 2 * ((1 - u) ^ 9 * (1 + 8 * u) ^ 2)) :=
      mul_le_mul_of_nonneg_left a2 (by positivity)
    have t4 := mul_le_mul_of_nonneg_left (a3.trans t3) (by linarith : (0 : ℝ) ≤ w + 1)
    calc (w + 1) * (1 + g1) * (1 + u) = (w + 1) * ((1 + g1) * (1 + u)) := by ring
      _ ≤ _ := t4
      _ = _ := by ring
  -- (b) the third term
  have b1 := pow_guard hu hu1 13 (by norm_num)
  push_cast at b1
  have hb : 3 * G * (1 + 4 * u) ^ 2 * (1 + u) ≤
      3 * G * (1 + 4 * u) ^ 2 * ((1 - u) ^ 13 * (1 + 8 * u) ^ 2) := by
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    nlinarith only [b1, hu2, hu]
  -- (c) the last term
  have c1 := pow_guard hu hu1 8 (by norm_num)
  push_cast at c1
  have hc : 2 * (1 + 4 * u) * (1 + u) + 12 * u ≤
      2 * (1 + 4 * u) * ((1 - u) ^ 8 * (1 + 8 * u) ^ 2) := by
    have t : 1 + 7 * u ≤ (1 - u) ^ 8 * (1 + 8 * u) ^ 2 := by nlinarith only [c1, hu2, hu]
    have t2 := mul_le_mul_of_nonneg_left t (by linarith : (0 : ℝ) ≤ 2 * (1 + 4 * u))
    nlinarith only [t2, hu]
  -- (d) the second term
  have d1 := pow_guard hu hu1 16 (by norm_num)
  push_cast at d1
  have d2 : (1 + gw) * (1 - 2 * u) ≤ 1 + gw * (1 - u) ^ 2 := by
    nlinarith only [mul_le_mul_of_nonneg_left hsq hgw0, hu]
  have h1 : 1 - u ≤ (1 - u) ^ 16 * (1 + 8 * u) ^ 2 := by nlinarith only [d1, hu2, hu]
  have d3 : (1 + gw) * (1 + u) - 4 * u * (1 + gw) ≤
      (1 + gw * (1 - u) ^ 2) * ((1 - u) ^ 16 * (1 + 8 * u) ^ 2) := by
    have h0 : 0 ≤ (1 + gw) * (1 - 2 * u) := mul_nonneg (by linarith) (by linarith)
    have t := mul_le_mul d2 h1 (by linarith) (le_trans h0 d2)
    have t2 : (1 + gw) * (1 + u) - 4 * u * (1 + gw) ≤ (1 + gw) * (1 - 2 * u) * (1 - u) := by
      nlinarith only [mul_nonneg (by linarith : (0 : ℝ) ≤ 1 + gw) (mul_nonneg hu hu)]
    linarith only [t, t2]
  have hgw1' : gw ≤ 1 := by linarith
  have hd : 2 * G * w * (1 + gw) * (1 + 4 * u) * (1 + u) - 9 * u ≤
      2 * G * w * (1 + gw * (1 - u) ^ 2) * (1 + 4 * u) * ((1 - u) ^ 16 * (1 + 8 * u) ^ 2) := by
    have hGw0 : 0 ≤ G * w := mul_nonneg hG0 hw0
    have hK0 : (0 : ℝ) ≤ 2 * (G * w) * (1 + 4 * u) := by positivity
    have t := mul_le_mul_of_nonneg_left d3 hK0
    have t2 : 2 * (G * w) * (1 + 4 * u) * (4 * u * (1 + gw)) ≤ 9 * u := by
      have hgws : gw ≤ 1 / 1000 := by linarith
      have h3 : (1 + 4 * u) * (1 + gw) ≤ 1002 / 1000 := by nlinarith only [hu, hu1, hgw0, hgws]
      have h4 : (G * w) * ((1 + 4 * u) * (1 + gw)) ≤ 1 * (1002 / 1000) :=
        mul_le_mul hGw h3 (by positivity) (by norm_num)
      nlinarith only [h4, hu]
    have e1 : 2 * G * w * (1 + gw) * (1 + 4 * u) * (1 + u) =
        2 * (G * w) * (1 + 4 * u) * ((1 + gw) * (1 + u)) := by ring
    have e2 : 2 * G * w * (1 + gw * (1 - u) ^ 2) * (1 + 4 * u) * ((1 - u) ^ 16 * (1 + 8 * u) ^ 2) =
        2 * (G * w) * (1 + 4 * u) * ((1 + gw * (1 - u) ^ 2) * ((1 - u) ^ 16 * (1 + 8 * u) ^ 2)) := by
      ring
    rw [e1, e2]
    nlinarith only [t, t2]
  linarith only [ha, hb, hc, hd, hu]

theorem gw_le_one {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {w : ℕ} (hw : w ≤ 2 ^ 26) :
    gam u (w + 2) * w ≤ 1 := by
  obtain ⟨hk, -, -⟩ := gam_regime hu hu53 (k := w + 2) (by omega)
  have hwr : (w : ℝ) ≤ 2 ^ 26 := by exact_mod_cast hw
  have hw0 : (0 : ℝ) ≤ w := Nat.cast_nonneg w
  push_cast at hk
  have hd : 0 < 1 - ((w : ℝ) + 2) * u := by linarith
  have hG : gam u (w + 2) * (1 - ((w : ℝ) + 2) * u) = ((w : ℝ) + 2) * u := by
    unfold gam; push_cast; exact div_mul_cancel₀ _ hd.ne'
  have h1 : (w : ℝ) * (((w : ℝ) + 2) * u) ≤ 51 / 100 := by
    have : (w : ℝ) * (((w : ℝ) + 2) * u) ≤ 2 ^ 26 * ((2 ^ 26 + 2) * (2 : ℝ)⁻¹ ^ 53) :=
      mul_le_mul hwr (mul_le_mul (by linarith) hu53 hu (by norm_num)) (by positivity) (by norm_num)
    have h2 : (2 : ℝ) ^ 26 * ((2 ^ 26 + 2) * (2 : ℝ)⁻¹ ^ 53) ≤ 51 / 100 := by norm_num
    linarith
  have hG0 : 0 ≤ gam u (w + 2) := gam_nonneg hu (by push_cast; linarith)
  have hGw0 : 0 ≤ gam u (w + 2) * w := mul_nonneg hG0 hw0
  have e : gam u (w + 2) * w = gam u (w + 2) * w * (1 - ((w : ℝ) + 2) * u) +
      gam u (w + 2) * w * (((w : ℝ) + 2) * u) := by ring
  have e2 : gam u (w + 2) * w * (1 - ((w : ℝ) + 2) * u) = (w : ℝ) * (((w : ℝ) + 2) * u) := by
    rw [mul_right_comm, hG]; ring
  have t := mul_le_mul_of_nonneg_left hk hGw0
  nlinarith only [e, e2, t, h1, hGw0]

theorem mul_ge' {a b x y : ℝ} (hx : a ≤ x) (hy : b ≤ y) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    a * b ≤ x * y := mul_le_mul hx hy hb (ha.trans hx)

section AddTerm

variable {R : Rounding ℝ}

theorem cwHat_ge (M : UModel R) {w : ℕ} (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26) :
    cwLow R.u w ≤ cwHat R w := by
  have hu := R.u_nonneg
  have hus := M.u_small
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, -⟩ := small_consts hu M.u_le53 hw
  obtain ⟨hγ1lo, -⟩ := M.gamHat_bounds (k := w + 1) (by omega) (by omega)
  obtain ⟨hγ2lo, -⟩ := M.gamHat_bounds (k := w + 2) (by omega) (by omega)
  obtain ⟨hγ0lo, -⟩ := M.gamHat_bounds (k := w) hw1 (by omega)
  set gw := gam R.u w with hgwd
  set g1 := gam R.u (w + 1) with hg1d
  set G := gam R.u (w + 2) with hGd
  set c := 1 - R.u with hc
  have h12 : 1 / 2 ≤ c := by rw [hc]; linarith
  have hc0 : 0 ≤ c := by linarith
  have hg10 : 0 ≤ g1 := by linarith
  have hG0 : 0 ≤ G := by linarith
  have hGu : R.u ≤ G := by linarith
  have hwr : (1 : ℝ) ≤ w := by exact_mod_cast hw1
  have hmin := normalMin_le_half
  have hnu : normalMin ≤ R.u / 64 := by
    rw [M.u_eq]
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 59 := normalMin_le_pow (by norm_num)
      _ = (2 : ℝ)⁻¹ ^ 53 / 64 := by norm_num
  have hc2 : 1 / 4 ≤ c ^ 2 := by nlinarith only [h12]
  have hu0 : 0 ≤ R.u / 64 := by positivity
  have h4 : 1 ≤ 1 + 4 * R.u := by linarith
  -- the first term
  have q1 : 1 ≤ 1 + g1 * c ^ 2 := by have : 0 ≤ g1 * c ^ 2 := by positivity
                                     linarith
  have t1a := M.le_rnd_normal (hmin.trans (by linarith)) (by linarith :
    1 + g1 * c ^ 2 ≤ 1 + gamHat R (w + 1))
  have hw1c : (2 : ℝ) ≤ ((w + 1 : ℕ) : ℝ) := by push_cast; linarith
  have v1 : 1 / 2 ≤ (1 + g1 * c ^ 2) * c := by nlinarith only [q1, h12]
  have t1n : 1 / 2 ≤ ((w + 1 : ℕ) : ℝ) * ((1 + g1 * c ^ 2) * c) := by nlinarith only [hw1c, v1]
  have t1 := M.le_rnd_normal (hmin.trans t1n) (mul_le_mul_of_nonneg_left t1a (by positivity))
  -- the second term
  have v2 : R.u / 2 ≤ 2 * G * c ^ 2 := by nlinarith only [mul_ge' hGu hc2 hu (by norm_num)]
  have t2a := M.le_rnd_normal (hnu.trans (by linarith : R.u / 64 ≤ 2 * G * c ^ 2))
    (by linarith : 2 * G * c ^ 2 ≤ 2 * gamHat R (w + 2))
  have v3 : R.u / 4 ≤ 2 * G * c ^ 2 * c * w := by
    have := mul_ge' v2 h12 (by positivity) (by norm_num)
    have := mul_ge' this hwr (by positivity) (by norm_num)
    linarith only [this]
  have t2b := M.le_rnd_normal (hnu.trans (by linarith : R.u / 64 ≤ 2 * G * c ^ 2 * c * w))
    (mul_le_mul_of_nonneg_right t2a (by linarith))
  have q2 : 1 ≤ 1 + gw * c ^ 2 := by have : 0 ≤ gw * c ^ 2 := by positivity
                                     linarith
  have t2c := M.le_rnd_normal (hmin.trans (by linarith)) (by linarith :
    1 + gw * c ^ 2 ≤ 1 + gamHat R w)
  have v4 : 1 / 2 ≤ (1 + gw * c ^ 2) * c := by nlinarith only [q2, h12]
  have v5 : R.u / 16 ≤ 2 * G * c ^ 2 * c * w * c * ((1 + gw * c ^ 2) * c) := by
    have := mul_ge' v3 h12 (by positivity) (by norm_num)
    have := mul_ge' this v4 (by positivity) (by norm_num)
    linarith only [this]
  have t2d := M.le_rnd_normal (hnu.trans (by linarith : R.u / 64 ≤
      2 * G * c ^ 2 * c * w * c * ((1 + gw * c ^ 2) * c)))
    (mul_le_mul t2b t2c (by positivity) (le_trans (by positivity) t2b))
  have v6 : R.u / 32 ≤ 2 * G * c ^ 2 * c * w * c * ((1 + gw * c ^ 2) * c) * c * (1 + 4 * R.u) := by
    have := mul_ge' v5 h12 (by positivity) (by norm_num)
    have := mul_ge' this h4 (by positivity) (by norm_num)
    linarith only [this]
  have t2 := M.le_rnd_normal (hnu.trans (by linarith : R.u / 64 ≤
      2 * G * c ^ 2 * c * w * c * ((1 + gw * c ^ 2) * c) * c * (1 + 4 * R.u)))
    (mul_le_mul_of_nonneg_right t2d (by linarith))
  -- the third term
  have v7 : R.u / 2 ≤ 3 * G * c ^ 2 := by nlinarith only [mul_ge' hGu hc2 hu (by norm_num), hu]
  have t3a := M.le_rnd_normal (hnu.trans (by linarith : R.u / 64 ≤ 3 * G * c ^ 2))
    (by linarith : 3 * G * c ^ 2 ≤ 3 * gamHat R (w + 2))
  have q3 : 1 ≤ (1 + 4 * R.u) ^ 2 := by nlinarith only [hu, mul_nonneg hu hu]
  have t3b := M.le_rnd_normal (hmin.trans (by linarith)) (le_refl ((1 + 4 * R.u) ^ 2))
  have v8 : R.u / 8 ≤ 3 * G * c ^ 2 * c * ((1 + 4 * R.u) ^ 2 * c) := by
    have := mul_ge' v7 h12 (by positivity) (by norm_num)
    have h5 : 1 / 2 ≤ (1 + 4 * R.u) ^ 2 * c := by nlinarith only [q3, h12]
    have := mul_ge' this h5 (by positivity) (by norm_num)
    linarith only [this]
  have t3 := M.le_rnd_normal (hnu.trans (by linarith : R.u / 64 ≤
      3 * G * c ^ 2 * c * ((1 + 4 * R.u) ^ 2 * c)))
    (mul_le_mul t3a t3b (by positivity) (le_trans (by positivity) t3a))
  -- the fourth term
  have t4 := M.le_rnd_normal (hmin.trans (by linarith)) (le_refl (2 * (1 + 4 * R.u)))
  -- the three sums
  set T1 := ((w + 1 : ℕ) : ℝ) * ((1 + g1 * c ^ 2) * c) * c with hT1
  set T2 := 2 * G * c ^ 2 * c * w * c * ((1 + gw * c ^ 2) * c) * c * (1 + 4 * R.u) * c with hT2
  set T3 := 3 * G * c ^ 2 * c * ((1 + 4 * R.u) ^ 2 * c) * c with hT3
  set T4 := 2 * (1 + 4 * R.u) * c with hT4
  have hT1lo : 1 / 4 ≤ T1 := by
    have := mul_ge' t1n h12 (by norm_num) (by norm_num)
    rw [hT1]; linarith only [this]
  have hT2lo : 0 ≤ T2 := by rw [hT2]; positivity
  have hT3lo : 0 ≤ T3 := by rw [hT3]; positivity
  have hT4lo : 0 ≤ T4 := by rw [hT4]; positivity
  have hm4 : normalMin ≤ 1 / 4 := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 2 := normalMin_le_pow (by norm_num)
      _ = 1 / 4 := by norm_num
  have s1 := M.le_rnd_normal (hm4.trans (by linarith only [hT1lo, hT2lo] : (1 : ℝ) / 4 ≤ T1 + T2))
    (add_le_add t1 t2)
  have s2n : 1 / 8 ≤ (T1 + T2) * c + T3 := by
    have := mul_ge' (by linarith only [hT1lo, hT2lo] : (1 : ℝ) / 4 ≤ T1 + T2) h12 (by norm_num)
      (by norm_num)
    linarith only [this, hT3lo]
  have hm8 : normalMin ≤ 1 / 8 := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 3 := normalMin_le_pow (by norm_num)
      _ = 1 / 8 := by norm_num
  have s2 := M.le_rnd_normal (hm8.trans s2n) (add_le_add s1 t3)
  have s3n : 1 / 16 ≤ ((T1 + T2) * c + T3) * c + T4 := by
    have := mul_ge' s2n h12 (by norm_num) (by norm_num)
    linarith only [this, hT4lo]
  have hm16 : normalMin ≤ 1 / 16 := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 4 := normalMin_le_pow (by norm_num)
      _ = 1 / 16 := by norm_num
  have s3 := M.le_rnd_normal (hm16.trans s3n) (add_le_add s2 t4)
  unfold cwHat cwLow
  rw [← hGd, ← hg1d, ← hgwd, ← hc]
  refine le_trans (le_of_eq ?_) s3
  rw [hT1, hT2, hT3, hT4]
  push_cast
  ring

theorem cw_bridge {u : ℝ} {w : ℕ} (h : ((w : ℝ) + 1) * (1 + gam u (w + 1)) * (1 + u) +
      2 * gam u (w + 2) * w * (1 + gam u w) * (1 + 4 * u) * (1 + u) +
      3 * gam u (w + 2) * (1 + 4 * u) ^ 2 * (1 + u) + 2 * (1 + 4 * u) * (1 + u) + u ≤
    (((w : ℝ) + 1) * (1 + gam u (w + 1) * (1 - u) ^ 2) * (1 - u) ^ 2 * ((1 - u) ^ 9 * (1 + 8 * u) ^ 2) +
      2 * gam u (w + 2) * w * (1 + gam u w * (1 - u) ^ 2) * (1 + 4 * u) *
        ((1 - u) ^ 16 * (1 + 8 * u) ^ 2) +
      3 * gam u (w + 2) * (1 + 4 * u) ^ 2 * ((1 - u) ^ 13 * (1 + 8 * u) ^ 2) +
      2 * (1 + 4 * u) * ((1 - u) ^ 8 * (1 + 8 * u) ^ 2))) :
    cwExact u w * (1 + u) + u ≤ cwLow u w * ((1 - u) ^ 6 * (1 + 8 * u) ^ 2) := by
  unfold cwExact cwLow
  have e1 : (((w : ℝ) + 1) * (1 + gam u (w + 1)) + 2 * gam u (w + 2) * w * (1 + gam u w) * (1 + 4 * u) +
      3 * gam u (w + 2) * (1 + 4 * u) ^ 2 + 2 * (1 + 4 * u)) * (1 + u) + u =
      ((w : ℝ) + 1) * (1 + gam u (w + 1)) * (1 + u) +
      2 * gam u (w + 2) * w * (1 + gam u w) * (1 + 4 * u) * (1 + u) +
      3 * gam u (w + 2) * (1 + 4 * u) ^ 2 * (1 + u) + 2 * (1 + 4 * u) * (1 + u) + u := by ring
  rw [e1]
  refine h.trans (le_of_eq ?_)
  ring

/-- **`residual_additive_term` covers `√n c_w (1 + u) u_min`.** -/
theorem addTerm_ge (M : UModel R) {n w : ℕ} (hn1 : 1 ≤ n) (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26) :
    Real.sqrt n * cwExact R.u w * (1 + R.u) * (2 : ℝ)⁻¹ ^ 1075 ≤ addTerm R n w := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hus := M.u_small
  have hu1' : R.u ≤ 1 := by linarith
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, -⟩ := small_consts hu M.u_le53 hw
  have hwr : (1 : ℝ) ≤ w := by exact_mod_cast hw1
  have hGw1 := gw_le_one hu M.u_le53 hw
  have hnum := cw_bridge (num_cw hu hus hwr hgw0 hgw1 hgw2 hGw hGw1)
  have hcw := cwHat_ge M hw1 hw
  have hg10 : 0 ≤ gam R.u (w + 1) := by linarith
  have hG0 : 0 ≤ gam R.u (w + 2) := by linarith
  have hcl1 : 1 ≤ cwLow R.u w := by
    have h0 : 0 ≤ 1 - R.u := by linarith
    have h1 : 0 ≤ ((((w : ℝ) + 1) * (1 + gam R.u (w + 1) * (1 - R.u) ^ 2) * (1 - R.u) ^ 2 +
        2 * gam R.u (w + 2) * w * (1 + gam R.u w * (1 - R.u) ^ 2) * (1 + 4 * R.u) * (1 - R.u) ^ 7) *
          (1 - R.u) + 3 * gam R.u (w + 2) * (1 + 4 * R.u) ^ 2 * (1 - R.u) ^ 5) *
            (1 - R.u) * (1 - R.u) := by positivity
    have h2 : 1 ≤ 2 * (1 + 4 * R.u) * (1 - R.u) * (1 - R.u) := by
      nlinarith only [hu, hus, mul_nonneg hu hu, mul_nonneg (mul_nonneg hu hu) hu]
    unfold cwLow; linarith only [h1, h2]
  set cl := cwLow R.u w with hcl
  set ε := (2 : ℝ)⁻¹ ^ 1075 with hε
  have hε0 : 0 ≤ ε := by positivity
  have hsn : 1 ≤ Real.sqrt n := by
    rw [Real.one_le_sqrt]; exact_mod_cast hn1
  set s := Real.sqrt n with hs
  have hmin := normalMin_le_half
  have hm4 : normalMin ≤ 1 / 4 := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 2 := normalMin_le_pow (by norm_num)
      _ = 1 / 4 := by norm_num
  have hm8 : normalMin ≤ 1 / 8 := by
    calc normalMin ≤ (2 : ℝ)⁻¹ ^ 3 := normalMin_le_pow (by norm_num)
      _ = 1 / 8 := by norm_num
  have h12 : 1 / 2 ≤ 1 - R.u := by linarith
  have h8 : 1 ≤ 1 + 8 * R.u := by linarith
  -- `t = sqrt(n) c_w (1 + 8u)²` from below
  have p0 := M.le_rnd_normal (hmin.trans (by linarith only [hsn])) (le_refl s)
  have z1 : 1 / 2 ≤ s * (1 - R.u) * cl := by
    have := mul_ge' (mul_ge' hsn h12 (by norm_num) (by norm_num)) hcl1 (by norm_num) (by norm_num)
    linarith only [this]
  have p1 := M.le_rnd_normal (hmin.trans z1) (mul_le_mul p0 hcw (by linarith only [hcl1])
    (le_trans (by nlinarith only [hsn, h12]) p0))
  have z2 : 1 / 4 ≤ s * (1 - R.u) * cl * (1 - R.u) * (1 + 8 * R.u) := by
    have := mul_ge' (mul_ge' z1 h12 (by norm_num) (by norm_num)) h8 (by norm_num) (by norm_num)
    linarith only [this]
  have p2 := M.le_rnd_normal (hm4.trans z2) (mul_le_mul_of_nonneg_right p1 (by linarith only [h8]))
  have z3 : 1 / 8 ≤ s * (1 - R.u) * cl * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) * (1 + 8 * R.u) := by
    have := mul_ge' (mul_ge' z2 h12 (by norm_num) (by norm_num)) h8 (by norm_num) (by norm_num)
    linarith only [this]
  have p3 := M.le_rnd_normal (hm8.trans z3) (mul_le_mul_of_nonneg_right p2 (by linarith only [h8]))
  set tlo := s * (1 - R.u) * cl * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u)
    with htlo
  set tHat := R.rnd (R.rnd (R.rnd (R.rnd s * cwHat R w) * (1 + 8 * R.u)) * (1 + 8 * R.u)) with htHat
  have htlo0 : 0 ≤ tlo := by rw [htlo]; nlinarith only [z3, h12]
  -- the ldexp and the `+ ULP_MIN`
  have hz : tlo * ε ≤ tHat * ε := mul_le_mul_of_nonneg_right p3 hε0
  have hL := lo_of_err hu hu1' (R.abs_err (tHat * ε)) hz le_rfl
    (by nlinarith only [hz, htlo0, hε0] : -(0 : ℝ) ≤ tHat * ε)
  set L := R.rnd (tHat * ε) with hLd
  have e2 : (2 : ℝ)⁻¹ ^ 1074 = 2 * ε := by
    have h : (2 : ℝ)⁻¹ ^ 1075 = (2 : ℝ)⁻¹ ^ 1074 * 2⁻¹ := pow_succ _ _
    rw [hε, h]; ring
  have heta := M.eta_le
  rw [← hε] at heta
  have hzA : tlo * ε * (1 - R.u) - R.eta + 2 * ε ≤ L + (2 : ℝ)⁻¹ ^ 1074 := by
    rw [e2]; linarith only [hL]
  have hzA0 : 0 ≤ tlo * ε * (1 - R.u) - R.eta + 2 * ε := by
    have : 0 ≤ tlo * ε * (1 - R.u) := mul_nonneg (mul_nonneg htlo0 hε0) (by linarith)
    linarith only [this, heta, hε0]
  have hA := lo_of_err hu hu1' (R.abs_err (L + (2 : ℝ)⁻¹ ^ 1074)) hzA le_rfl
    (by linarith only [hzA, hzA0] : -(0 : ℝ) ≤ L + (2 : ℝ)⁻¹ ^ 1074)
  -- the slack
  have e3 : tlo * (1 - R.u) * (1 - R.u) = s * (cl * ((1 - R.u) ^ 6 * (1 + 8 * R.u) ^ 2)) := by
    rw [htlo]; ring
  have hk : s * (cwExact R.u w * (1 + R.u) + R.u) ≤ s * (cl * ((1 - R.u) ^ 6 * (1 + 8 * R.u) ^ 2)) :=
    mul_le_mul_of_nonneg_left hnum (by linarith only [hsn])
  have hsu : R.u ≤ s * R.u := by nlinarith only [hsn, hu]
  have hfin : s * cwExact R.u w * (1 + R.u) * ε ≤
      (tlo * ε * (1 - R.u) - R.eta + 2 * ε) * (1 - R.u) - 2 * R.u * 0 - R.eta := by
    have t1 : (s * cwExact R.u w * (1 + R.u) + R.u) * ε ≤ tlo * (1 - R.u) * (1 - R.u) * ε := by
      apply mul_le_mul_of_nonneg_right _ hε0
      rw [e3]; nlinarith only [hk, hsu]
    have t2 : 0 ≤ (ε - R.eta) * (1 - R.u) := mul_nonneg (by linarith only [heta]) (by linarith)
    nlinarith only [t1, t2, heta, hε0, hu, mul_nonneg hu hε0]
  unfold addTerm
  exact hfin.trans hA

end AddTerm

/-! ### Theorem U5 -/

/-- The final quotient of `residuals` with the additive terms. -/
theorem chain_final_U (R : Rounding ℝ) (hu1 : R.u ≤ 1 / 1000) (heta : R.eta ≤ (2 : ℝ)⁻¹ ^ 1075)
    {nr nRc nE D nx xl Rup Eup c1 : ℝ} (hr : nr ≤ nRc + nE + D) (hnRc : 0 ≤ nRc) (hnE : 0 ≤ nE)
    (hD : 0 ≤ D) (hRup : nRc * (1 + R.u) ≤ Rup) (hEup : nE * (1 + R.u) ≤ Eup)
    (hc1 : D * (1 + R.u) ≤ c1) (hf1 : R.IsFloat Rup) (hf2 : R.IsFloat Eup) (hf3 : R.IsFloat c1)
    (hxl0 : 0 < xl) (hxl : xl ≤ nx * (1 - R.u)) :
    nr / nx ≤ R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (Rup + Eup) + c1) / xl) * (1 + 4 * R.u)) +
      (2 : ℝ)⁻¹ ^ 1073) := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hu1' : R.u ≤ 1 := by linarith
  have h1u : 0 < 1 - R.u := by linarith
  have hnx : 0 < nx := by
    by_contra hneg
    push Not at hneg
    nlinarith
  set Mv := nRc + nE + D with hMv
  have hMv0 : 0 ≤ Mv := by positivity
  -- the two sums
  have hz1 : (nRc + nE) * (1 + R.u) ≤ Rup + Eup := by linarith only [hRup, hEup]
  have hz10 : 0 ≤ (nRc + nE) * (1 + R.u) := by positivity
  have hU1 := lo_of_err hu hu1' (R.abs_add_err hf1 hf2) hz1 le_rfl (by linarith only [hz1, hz10])
  have hz2 : Mv * (1 + R.u) * (1 - R.u) ≤ R.rnd (Rup + Eup) + c1 := by
    rw [hMv]
    nlinarith only [hU1, hc1, hD, hu, mul_nonneg hD hu]
  have hz20 : 0 ≤ Mv * (1 + R.u) * (1 - R.u) := by
    have : 0 ≤ 1 + R.u := by linarith
    positivity
  have hU2 := lo_of_err hu hu1' (R.abs_add_err (R.isFloat_rnd _) hf3) hz2 le_rfl
    (by linarith only [hz2, hz20])
  set U2 := R.rnd (R.rnd (Rup + Eup) + c1) with hU2d
  set Qv := Mv / nx with hQv
  have hQv0 : 0 ≤ Qv := div_nonneg hMv0 hnx.le
  have hU20 : 0 ≤ Mv * (1 + R.u) * (1 - R.u) * (1 - R.u) := by
    have : 0 ≤ 1 + R.u := by linarith
    positivity
  have hZ : Qv * (1 + R.u) * (1 - R.u) ≤ U2 / xl := by
    have t1 : Mv * (1 + R.u) * (1 - R.u) * (1 - R.u) / (nx * (1 - R.u)) ≤
        Mv * (1 + R.u) * (1 - R.u) * (1 - R.u) / xl := div_le_div_of_nonneg_left hU20 hxl0 hxl
    have t2 : Mv * (1 + R.u) * (1 - R.u) * (1 - R.u) / xl ≤ U2 / xl :=
      div_le_div_of_nonneg_right (by linarith only [hU2]) hxl0.le
    have e : Mv * (1 + R.u) * (1 - R.u) * (1 - R.u) / (nx * (1 - R.u)) = Qv * (1 + R.u) * (1 - R.u) := by
      rw [hQv]; field_simp
    linarith only [t1, t2, e]
  have hZ0 : 0 ≤ Qv * (1 + R.u) * (1 - R.u) := by
    have : 0 ≤ 1 + R.u := by linarith
    positivity
  have hq := lo_of_err hu hu1' (R.abs_err (U2 / xl)) hZ le_rfl (by linarith only [hZ, hZ0])
  have hq' := lo_of_err hu hu1' (R.abs_err (U2 / xl)) (by linarith only [hZ, hZ0] : (0 : ℝ) ≤ U2 / xl)
    le_rfl (by linarith only [hZ, hZ0])
  set q := R.rnd (U2 / xl) with hqd
  have h4 : 0 ≤ 1 + 4 * R.u := by linarith
  have hzp : (Qv * (1 + R.u) * (1 - R.u) * (1 - R.u) - R.eta) * (1 + 4 * R.u) ≤ q * (1 + 4 * R.u) :=
    mul_le_mul_of_nonneg_right (by linarith only [hq]) h4
  have hzp' : -(R.eta * (1 + 4 * R.u)) ≤ q * (1 + 4 * R.u) := by
    have := mul_le_mul_of_nonneg_right (by linarith only [hq'] : -R.eta ≤ q) h4
    linarith only [this]
  have hp := lo_of_err hu hu1' (R.abs_err (q * (1 + 4 * R.u))) hzp (by positivity) hzp'
  set p := R.rnd (q * (1 + 4 * R.u)) with hpd
  have e4 : (2 : ℝ)⁻¹ ^ 1073 = 4 * (2 : ℝ)⁻¹ ^ 1075 := by
    have h : (2 : ℝ)⁻¹ ^ 1075 = (2 : ℝ)⁻¹ ^ 1073 * (2 : ℝ)⁻¹ ^ 2 := pow_add _ 1073 2
    rw [h]; ring
  set Lr := Qv * (1 + R.u) * (1 - R.u) ^ 3 * (1 + 4 * R.u) +
    ((2 : ℝ)⁻¹ ^ 1073 - R.eta * (2 + 5 * R.u + 4 * R.u ^ 2)) with hLr
  have hzr : Lr ≤ p + (2 : ℝ)⁻¹ ^ 1073 := by
    rw [hLr]
    nlinarith only [hp]
  have hbr : R.eta * (2 - 5 * R.u - 4 * R.u ^ 2) ≤ (2 : ℝ)⁻¹ ^ 1073 - R.eta * (2 + 5 * R.u + 4 * R.u ^ 2) := by
    rw [e4]; nlinarith only [heta, he]
  have hbr0 : 0 ≤ R.eta * (2 - 5 * R.u - 4 * R.u ^ 2) := by
    apply mul_nonneg he; nlinarith only [hu, hu1]
  have hLr0 : 0 ≤ Lr := by
    have : 0 ≤ Qv * (1 + R.u) * (1 - R.u) ^ 3 * (1 + 4 * R.u) := by
      have : 0 ≤ 1 + R.u := by linarith
      positivity
    rw [hLr]; linarith only [this, hbr, hbr0]
  have hrel := lo_of_err hu hu1' (R.abs_err (p + (2 : ℝ)⁻¹ ^ 1073)) hzr le_rfl
    (by linarith only [hzr, hLr0])
  have hnum := num_final hu hu1
  have hfin : Qv ≤ Lr * (1 - R.u) - 2 * R.u * 0 - R.eta := by
    have t1 : Qv ≤ Qv * ((1 + R.u) * (1 - R.u) ^ 4 * (1 + 4 * R.u)) := le_mul_of_one_le_right hQv0 hnum
    have t2 : R.eta ≤ R.eta * (2 - 5 * R.u - 4 * R.u ^ 2) * (1 - R.u) := by
      have : 1 ≤ (2 - 5 * R.u - 4 * R.u ^ 2) * (1 - R.u) := by nlinarith only [hu, hu1]
      nlinarith only [this, he]
    have t3 := mul_le_mul_of_nonneg_right hbr h1u.le
    rw [hLr]
    nlinarith only [t1, t2, t3]
  calc nr / nx ≤ Qv := by rw [hQv]; exact div_le_div_of_nonneg_right hr hnx.le
    _ ≤ _ := hfin
    _ ≤ _ := hrel

section TheoremU5

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [DecidableEq ι] in
theorem l2_add_le' (a b : ι → ℝ) : l2 (fun i => a i + b i) ≤ l2 a + l2 b := by
  have h := l2_le_add (fun i => a i + b i) a
  have e : (fun i => a i + b i - a i) = b := by funext i; ring
  simpa only [e] using h

omit [DecidableEq ι] in
theorem l2_const (c : ℝ) (hc : 0 ≤ c) :
    l2 (fun _ : ι => c) = Real.sqrt (Fintype.card ι) * c := by
  have h : ∑ _i : ι, c ^ 2 = (Fintype.card ι : ℝ) * c ^ 2 := by
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [l2_eq, h, Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq hc]

theorem cwExact_nonneg {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {w : ℕ}
    (hw : w ≤ 2 ^ 26) : 0 ≤ cwExact u w := by
  obtain ⟨_, _, hgw0, hgw1, hgw2, _, _⟩ := small_consts hu hu53 hw
  have h1 : 0 ≤ gam u (w + 1) := by linarith
  have h2 : 0 ≤ gam u (w + 2) := by linarith
  unfold cwExact
  positivity

/-- **Theorem U5 (Theorem 1 in floating point under the underflow-tolerant model, real data).**
For every input (no magnitude hypothesis), the radius computed by `enclosure.residuals` with the
additive terms of the code (`norm_underflow_term` in both norm bounds, `residual_additive_term`
added to `R_up + E_up`, and `2 ULP_MIN` added to the quotient) bounds `‖A x - μ x‖ / ‖x‖` when
the computed lower bound of `‖x‖` is positive. -/
theorem residual_radius_int05 {R : Rounding ℝ} (M : UModel R) {w : ℕ} (hw1 : 1 ≤ w)
    (hw : w ≤ 2 ^ 26) (hn : Fintype.card ι ≤ 2 ^ 26) {A : ι → ι → ℝ} {T T' : ι → STree ι}
    (hT : RowTrees T A w) (hT' : RowTrees T' A w) {TR TE TX : STree ι} (hTR : FullTree TR)
    (hTE : FullTree TE) (hTX : FullTree TX) (x : ι → ℝ) (μ : ℝ)
    (hxl : 0 < norm2Low R TX (normUnder R (Fintype.card ι)) x) :
    l2 (fun i => ∑ j, A i j * x j - μ * x i) / l2 x ≤
      residualRadiusFl R T T' TR TE TX w A x μ (normUnder R (Fintype.card ι))
        (normUnder R (Fintype.card ι)) (normUnder R (Fintype.card ι))
        (addTerm R (Fintype.card ι) w) ((2 : ℝ)⁻¹ ^ 1073) := by
  have hu := R.u_nonneg
  have he := R.eta_nonneg
  have hu1 : R.u ≤ 1 / 1000 := M.u_le53.trans (by norm_num)
  have hn1 := fullTree_card_pos hTX
  set n := Fintype.card ι with hn_def
  have hcw0 := cwExact_nonneg hu M.u_le53 hw
  set c := cwExact R.u w * R.eta with hc
  have hc0 : 0 ≤ c := mul_nonneg hcw0 he
  have hentry := resid_le_roundBound_U M hw1 hw hT hT' x μ
  have hr : l2 (fun i => ∑ j, A i j * x j - μ * x i) ≤ l2 (residFl R T A x μ) +
      l2 (roundBoundFl R T' w A x μ) + Real.sqrt n * c := by
    have h1 := l2_le_add (fun i => ∑ j, A i j * x j - μ * x i) (residFl R T A x μ)
    have h2 : l2 (fun i => (∑ j, A i j * x j - μ * x i) - residFl R T A x μ i) ≤
        l2 (fun i => roundBoundFl R T' w A x μ i + c) := by
      apply l2_le_of_abs_le
      intro i
      rw [abs_sub_comm]
      exact hentry i
    have h3 := l2_add_le' (roundBoundFl R T' w A x μ) (fun _ => c)
    have h4 : l2 (fun _ : ι => c) = Real.sqrt n * c := l2_const c hc0
    linarith only [h1, h2, h3, h4]
  have hRup := norm2Up_ge_U M hn hTR (residFl R T A x μ)
  have hEup := norm2Up_ge_U M hn hTE (roundBoundFl R T' w A x μ)
  have hxlb := norm2Low_le_U M hn hTX x
  have hA := addTerm_ge M hn1 hw1 hw
  have hc1 : Real.sqrt n * c * (1 + R.u) ≤ addTerm R n w := by
    refine le_trans ?_ hA
    rw [hc]
    have hs : 0 ≤ Real.sqrt n * cwExact R.u w * (1 + R.u) := by
      have : 0 ≤ 1 + R.u := by linarith
      positivity
    calc Real.sqrt n * (cwExact R.u w * R.eta) * (1 + R.u) =
        Real.sqrt n * cwExact R.u w * (1 + R.u) * R.eta := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left M.eta_le hs
  unfold residualRadiusFl
  exact chain_final_U R hu1 M.eta_le hr (l2_nonneg _) (l2_nonneg _)
    (mul_nonneg (Real.sqrt_nonneg _) hc0) hRup hEup hc1 (R.isFloat_rnd _) (R.isFloat_rnd _)
    (R.isFloat_rnd _) hxl hxlb

/-- **Theorem U5 with the eigenvalue**: for a real symmetric matrix some eigenvalue lies within
the radius computed under the underflow-tolerant model. -/
theorem residual_enclosure_int05 {A : Matrix ι ι ℝ} (hA : A.IsHermitian) {R : Rounding ℝ}
    (M : UModel R) {w : ℕ} (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26) (hn : Fintype.card ι ≤ 2 ^ 26)
    {T T' : ι → STree ι} (hT : RowTrees T A w) (hT' : RowTrees T' A w) {TR TE TX : STree ι}
    (hTR : FullTree TR) (hTE : FullTree TE) (hTX : FullTree TX) (x : EuclideanSpace ℝ ι) (μ : ℝ)
    (hxl : 0 < norm2Low R TX (normUnder R (Fintype.card ι)) (fun i => x i)) :
    ∃ i, |hA.eigenvalues i - μ| ≤
      residualRadiusFl R T T' TR TE TX w A (fun i => x i) μ (normUnder R (Fintype.card ι))
        (normUnder R (Fintype.card ι)) (normUnder R (Fintype.card ι))
        (addTerm R (Fintype.card ι) w) ((2 : ℝ)⁻¹ ^ 1073) := by
  have hlow := norm2Low_le_U M hn hTX (fun i => x i)
  have hx : x ≠ 0 := by
    intro h
    have : l2 (fun i => x i) = 0 := by rw [l2_ofLp, h, norm_zero]
    rw [this, zero_mul] at hlow
    linarith
  obtain ⟨i, hi⟩ := residual_enclosure hA μ hx
  refine ⟨i, hi.trans ?_⟩
  have := residual_radius_int05 M hw1 hw hn hT hT' hTR hTE hTX (fun i => x i) μ hxl
  rwa [l2_residual, l2_ofLp] at this

end TheoremU5


end CertifyMathlib.Enclosure.FP
