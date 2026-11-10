import CertifyMathlib.Apps.Bordered

/-!
# From the certified count to the Pomata-Wei criterion

Pomata and Wei (arXiv:1911.01410) prove the AKLT spectral gap from a finite-size criterion on
`eta`, the largest cosine other than 1 of the principal angles between two ground spaces `A` and `B`:
`eta < 1/z` (`z = 4` or `3` for the lattices of E10) or, for the square-octagon lattice,
`4 eta_perp + 2 eta_par < 1`. With bases of `A` and `B` of Gram matrices `G_A`, `G_B` and cross Gram
matrix `C`, and any invertible `F` with `G_A = F Fᴴ`, the squared cosines are the eigenvalues of the
symmetric matrix `K = F⁻¹ C G_B⁻¹ Cᴴ F⁻ᴴ` (the standard description of principal angles, taken here as
the definition); the unit eigenvalues belong to `A ∩ B`.

* `cntLt_smul_one_sub`, `pencil_count_eq`: the number of eigenvalues of `K` above `t` is the number of
  negative eigenvalues of `t G_A - C G_B⁻¹ Cᴴ`, the quantity bounded in `Bordered.lean`;
* `eig_le_of_count`: if at most `u` eigenvalues exceed `t' < 1` and at least `u` equal 1, every eigenvalue
  below 1 is at most `t'`;
* `e10_eta_bound`: the composed statement of `eta.py`: the certified count (`eta_count`) and the lower
  bound on the unit cosines give `eta² ≤ t'`;
* `criterion_of_eta_sq`, `sqrt_le_isqrt_div`, `square_octagon_criterion`: the comparisons with which
  `certify_ed/apps/aklt/__init__.py` decides the criterion in rationals (`t' < 1/z²`, and
  `4 s_perp + 2 s_par < 1` for rational upper bounds `s` of the square roots) imply the criterion on
  `eta` itself.
-/

namespace CertifyMathlib.Apps

open Matrix CertifyMathlib.Counting

section Pencil

variable {a b : Type*} [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b]

/-- The number of eigenvalues of `K` above `t` is the number of negative eigenvalues of `t I - K`. -/
theorem cntLt_smul_one_sub {K : Matrix a a ℝ} (hK : K.IsHermitian) (t : ℝ)
    (h : (t • (1 : Matrix a a ℝ) - K).IsHermitian) :
    cntLt h 0 = (Finset.univ.filter fun i => t < hK.eigenvalues i).card := by
  set U := (hK.eigenvectorUnitary : Matrix a a ℝ) with hU
  have h1 : Uᴴ * U = 1 := conjTranspose_mul_unitary _
  have h2 : U * Uᴴ = 1 := unitary_mul_conjTranspose _
  have hs := sub_smul_one_eq hK t
  have hfac : t • (1 : Matrix a a ℝ) - K - ((0 : ℝ) : ℝ) • (1 : Matrix a a ℝ) =
      U * dg (fun i => t - hK.eigenvalues i) * Uᴴ := by
    have hneg : dg (𝕜 := ℝ) (fun i => t - hK.eigenvalues i) = -dg (fun i => hK.eigenvalues i - t) := by
      ext i j
      by_cases hij : i = j
      · subst hij; simp [dg]
      · simp [dg, hij]
    rw [hneg, Matrix.mul_neg, Matrix.neg_mul, ← hs]
    simp
  have := (cntLt_eq_of_factor h 0 U U h1 h2 _ hfac).1
  rw [this]
  congr 1
  ext i
  simp

/-- **The pencil count.** If `G_A = F Fᴴ` with `F` invertible and `K = F⁻¹ (C G_B⁻¹ Cᴴ) F⁻ᴴ`, the number
of negative eigenvalues of `t G_A - C G_B⁻¹ Cᴴ` is the number of eigenvalues of `K` above `t`. -/
theorem pencil_count_eq {GA : Matrix a a ℝ} {GB : Matrix b b ℝ} {C : Matrix a b ℝ}
    (F : Matrix a a ℝ) (hF : IsUnit F.det) (hGA : GA = F * Fᴴ) {K : Matrix a a ℝ}
    (hKdef : K = F⁻¹ * (C * GB⁻¹ * Cᴴ) * (F⁻¹)ᴴ) (hK : K.IsHermitian) (t : ℝ)
    (hP : (t • GA - C * GB⁻¹ * Cᴴ).IsHermitian) :
    cntLt hP 0 = (Finset.univ.filter fun i => t < hK.eigenvalues i).card := by
  have hFt : IsUnit Fᴴ.det := by rw [det_conjTranspose]; exact hF.star
  have hM : C * GB⁻¹ * Cᴴ = F * K * Fᴴ := by
    rw [hKdef]
    have h1 : F * F⁻¹ = 1 := mul_nonsing_inv F hF
    have h2 : (F⁻¹)ᴴ * Fᴴ = 1 := by rw [← conjTranspose_mul, h1, conjTranspose_one]
    calc C * GB⁻¹ * Cᴴ = (F * F⁻¹) * (C * GB⁻¹ * Cᴴ) * ((F⁻¹)ᴴ * Fᴴ) := by
          rw [h1, h2, Matrix.one_mul, Matrix.mul_one]
      _ = F * (F⁻¹ * (C * GB⁻¹ * Cᴴ) * (F⁻¹)ᴴ) * Fᴴ := by simp only [Matrix.mul_assoc]
  have hT : (t • (1 : Matrix a a ℝ) - K).IsHermitian :=
    (isHermitian_smul_real isHermitian_one t).sub hK
  have heq : t • GA - C * GB⁻¹ * Cᴴ = F * (t • (1 : Matrix a a ℝ) - K) * Fᴴ := by
    rw [hGA, hM, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul]
  rw [cntLt_congr hP hT F (Fᴴ)⁻¹ (mul_nonsing_inv _ hFt) (nonsing_inv_mul _ hFt) heq,
    cntLt_smul_one_sub hK t hT]

end Pencil

/-- If at most `u` of the values `λ i` exceed `t' < 1` and at least `u` of them equal 1, every value
below 1 is at most `t'`. -/
theorem eig_le_of_count {ι : Type*} [Fintype ι] (lam : ι → ℝ) {t' : ℝ} (ht' : t' < 1) (u : ℕ)
    (hcount : (Finset.univ.filter fun i => t' < lam i).card ≤ u)
    (hunit : u ≤ (Finset.univ.filter fun i => lam i = 1).card) (i : ι) (hi : lam i < 1) :
    lam i ≤ t' := by
  by_contra hlt
  rw [not_le] at hlt
  have hsub : (Finset.univ.filter fun i => lam i = 1) ⊆ (Finset.univ.filter fun i => t' < lam i) := by
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    rw [hj]
    exact ht'
  have heq := Finset.eq_of_subset_of_card_le hsub (hcount.trans hunit)
  have hmem : i ∈ (Finset.univ.filter fun i => t' < lam i) := by simp [hlt]
  rw [← heq] at hmem
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hmem
  linarith

section Eta

variable {a b : Type*} [Fintype a] [Fintype b] [DecidableEq a] [DecidableEq b]

/-- **The E10 eta bound, composed.** Under the hypotheses of `eta_count` (Gram bounds `μ`, `ν`, threshold
`p / q`, a symmetric `N'` within `q β` of the bordered matrix), if `N'` has at most `u` negative
eigenvalues (the positive pivots of the exact-checked factorization) and at least `u` eigenvalues of the
cosine matrix `K` equal 1 (the unit cosines, `dim (A ∩ B) ≥ u`), then every squared cosine other than 1
is at most `t' = (1 + β/μ)(p/q + β/ν)`, provided `t' < 1`. -/
theorem e10_eta_bound {GA : Matrix a a ℝ} {GB : Matrix b b ℝ} {C : Matrix a b ℝ}
    (hGA : GA.IsHermitian) (hGB : GB.IsHermitian) {μ ν β p q : ℝ}
    (hμ : 0 < μ) (hν : 0 < ν) (hβ : 0 ≤ β) (hq : 0 < q)
    (hGBμ : ∀ y, μ * nsq y ≤ qf GB y) (hGAν : ∀ x, ν * nsq x ≤ qf GA x)
    {N' : Matrix (a ⊕ b) (a ⊕ b) ℝ} (hN' : N'.IsHermitian)
    (hres : ∀ x, |qf (N' - bordered (p • GA) C GB q) x| ≤ q * β * nsq x)
    (F : Matrix a a ℝ) (hF : IsUnit F.det) (hGAF : GA = F * Fᴴ) {K : Matrix a a ℝ}
    (hKdef : K = F⁻¹ * (C * GB⁻¹ * Cᴴ) * (F⁻¹)ᴴ) (hK : K.IsHermitian) (u : ℕ)
    (hpiv : cntLt hN' 0 ≤ u) (hunit : u ≤ (Finset.univ.filter fun i => hK.eigenvalues i = 1).card)
    (ht' : (1 + β / μ) * (p / q + β / ν) < 1) (i : a) (hi : hK.eigenvalues i < 1) :
    hK.eigenvalues i ≤ (1 + β / μ) * (p / q + β / ν) := by
  set t' := (1 + β / μ) * (p / q + β / ν)
  have hP : (t' • GA - C * GB⁻¹ * Cᴴ).IsHermitian :=
    (isHermitian_smul_real hGA t').sub (isHermitian_mul_mul_conjTranspose C hGB.inv)
  have h1 := eta_count hGA hGB hμ hν hβ hq hGBμ hGAν hN' hres hP
  rw [pencil_count_eq F hF hGAF hKdef hK t' hP] at h1
  exact eig_le_of_count hK.eigenvalues ht' u (h1.trans hpiv) hunit i hi

/-- **The E10 criterion, composed.** Under the hypotheses of `e10_eta_bound`, if the rational test of
the code `t' < 1/z²` holds with `z ≥ 1`, then every cosine other than 1 is below `1/z`: `η < 1/z`. -/
theorem e10_criterion {GA : Matrix a a ℝ} {GB : Matrix b b ℝ} {C : Matrix a b ℝ}
    (hGA : GA.IsHermitian) (hGB : GB.IsHermitian) {μ ν β p q : ℝ}
    (hμ : 0 < μ) (hν : 0 < ν) (hβ : 0 ≤ β) (hq : 0 < q)
    (hGBμ : ∀ y, μ * nsq y ≤ qf GB y) (hGAν : ∀ x, ν * nsq x ≤ qf GA x)
    {N' : Matrix (a ⊕ b) (a ⊕ b) ℝ} (hN' : N'.IsHermitian)
    (hres : ∀ x, |qf (N' - bordered (p • GA) C GB q) x| ≤ q * β * nsq x)
    (F : Matrix a a ℝ) (hF : IsUnit F.det) (hGAF : GA = F * Fᴴ) {K : Matrix a a ℝ}
    (hKdef : K = F⁻¹ * (C * GB⁻¹ * Cᴴ) * (F⁻¹)ᴴ) (hK : K.IsHermitian) (u : ℕ)
    (hpiv : cntLt hN' 0 ≤ u) (hunit : u ≤ (Finset.univ.filter fun i => hK.eigenvalues i = 1).card)
    {z : ℝ} (hz : 1 ≤ z) (hcrit : (1 + β / μ) * (p / q + β / ν) < 1 / z ^ 2) (i : a)
    (hi : hK.eigenvalues i < 1) : Real.sqrt (hK.eigenvalues i) < 1 / z := by
  set t' := (1 + β / μ) * (p / q + β / ν)
  have hz0 : 0 < z := by linarith
  have hz2 : 1 / z ^ 2 ≤ 1 := by
    rw [div_le_one (by positivity)]
    nlinarith
  have ht' : t' < 1 := lt_of_lt_of_le hcrit hz2
  have hle := e10_eta_bound hGA hGB hμ hν hβ hq hGBμ hGAν hN' hres F hF hGAF hKdef hK u hpiv hunit
    ht' i hi
  have h1 : Real.sqrt (hK.eigenvalues i) ≤ Real.sqrt t' := Real.sqrt_le_sqrt hle
  have h2 : Real.sqrt t' < 1 / z := by
    rw [Real.sqrt_lt' (by positivity)]
    rw [div_pow, one_pow]
    exact hcrit
  exact lt_of_le_of_lt h1 h2

end Eta

/-- The single-lattice criterion as decided in rationals: `η² ≤ t'` and `t' < 1/z²` give
`η < 1/z`. -/
theorem criterion_of_eta_sq {η t' z : ℝ} (hz : 0 < z) (hle : η ^ 2 ≤ t')
    (hlt : t' < 1 / z ^ 2) : η < 1 / z := by
  have h1 : η ^ 2 < (1 / z) ^ 2 := by rw [div_pow, one_pow]; linarith
  exact lt_of_pow_lt_pow_left₀ 2 (by positivity) h1

/-- The rational square-root bound of `square_octagon_criterion` in `__init__.py`:
`sqrt (num / den) ≤ (isqrt (num den) + 1) / den`. -/
theorem sqrt_le_isqrt_div (num den : ℕ) (hden : 0 < den) :
    Real.sqrt ((num : ℝ) / den) ≤ ((Nat.sqrt (num * den) + 1 : ℕ) : ℝ) / den := by
  have hd : (0 : ℝ) < den := by exact_mod_cast hden
  have hs : num * den < (Nat.sqrt (num * den) + 1) * (Nat.sqrt (num * den) + 1) :=
    Nat.lt_succ_sqrt (num * den)
  have hsR : (num : ℝ) * den < ((Nat.sqrt (num * den) + 1 : ℕ) : ℝ) ^ 2 := by
    rw [sq]; exact_mod_cast hs
  have hrhs : 0 ≤ ((Nat.sqrt (num * den) + 1 : ℕ) : ℝ) / den := by positivity
  rw [Real.sqrt_le_left hrhs]
  rw [div_pow, le_div_iff₀ (by positivity)]
  have : (num : ℝ) / den * (den : ℝ) ^ 2 = num * den := by field_simp
  rw [this]
  exact hsR.le

/-- **The square-octagon criterion.** If `η⊥² ≤ a`, `η∥² ≤ b` and `s⊥, s∥ ≥ 0` are upper bounds of
the square roots (`a ≤ s⊥²`, `b ≤ s∥²`) with `4 s⊥ + 2 s∥ < 1`, then `4 η⊥ + 2 η∥ < 1`. -/
theorem square_octagon_criterion {ηp ηq a b sp sq : ℝ}
    (hp : ηp ^ 2 ≤ a) (hq : ηq ^ 2 ≤ b) (hsp : 0 ≤ sp) (hsq : 0 ≤ sq) (hap : a ≤ sp ^ 2)
    (hbq : b ≤ sq ^ 2) (hlt : 4 * sp + 2 * sq < 1) : 4 * ηp + 2 * ηq < 1 := by
  have h1 : ηp ≤ sp := by nlinarith [hp.trans hap]
  have h2 : ηq ≤ sq := by nlinarith [hq.trans hbq]
  linarith

end CertifyMathlib.Apps
