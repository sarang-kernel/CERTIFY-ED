import CertifyMathlib.Counting.Thermo
import CertifyMathlib.Counting.Variance

/-!
# C-06 (Proposition 16'): Lemma C, the omitted moments and the assembled sums;
# the shift identity of Proposition 16

Source: `docs/proofs/C06_truncated_thermodynamics.md` and `docs/proofs/P16_thermo_intervals.md` in
the code repository. `φ(e) = e e^{-β e}`, `ψ(e) = e² e^{-β e}`, `β > 0`.

* Lemma C: `phiT_le_max` (`φ ≤ e^{-1}/β`, the value at `1/β`), `phiT_mono` (increasing on
  `[0, 1/β]`), `phiT_anti` (decreasing on `[1/β, ∞)`), `phiT_ge_min` (on `[ℓ, μ]`,
  `φ ≥ min(φ(ℓ), φ(μ))`), `phiT_le_left` / `phiT_le_right` (the two regimes the code uses for the
  supremum); the same for `ψ` with `2/β` and `4 e^{-2}/β²` (`psiT_eq`: `ψ_β = φ_{β/2}²`).
* `sum_fiber_bounds_on`: Proposition 16's counting of interval contributions over a subset.
* `c06_tail_phi`, `c06_tail_psi`: (ii), `Σ_{j>k} e_j e^{-β e_j} ∈ (n - k) [inf φ, sup φ]` and the
  same for `ψ`, for every pair of bounds valid on `[ℓ, μ]`.
* `c06_sums`: (iii) without the dyadic rounding: the claimed eigenvalues (those below `L`) are
  enclosed by their intervals, the omitted ones by `[ℓ, μ]`, and the sums for `Z e^{β E_0}`,
  `Σ e_i e^{-β e_i}` and `Σ e_i² e^{-β e_i}` lie between the sums of the two parts' bounds. The
  quotients, the variance and `C` then follow by `quotient_bounds`, `variance_bounds` and
  `heat_capacity_bounds` exactly as in Proposition 16.
* `c06_sums_noM`: the same without the bound `M` (the case `μ = ∞`), with lower ends `0` for the
  omitted terms.
* `mean_shift`, `variance_shift`: `⟨e + c⟩ = ⟨e⟩ + c` and `Var(e + c) = Var(e)` (the step
  `Var(E) = Var(e)` and `U = E_0 + ⟨e⟩` of the proof of Proposition 16).
-/

namespace CertifyMathlib.Counting

/-- `φ(e) = e e^{-β e}`. -/
noncomputable def phiT (β e : ℝ) : ℝ := e * Real.exp (-(β * e))

/-- `ψ(e) = e² e^{-β e}`. -/
noncomputable def psiT (β e : ℝ) : ℝ := e ^ 2 * Real.exp (-(β * e))

theorem psiT_eq (β e : ℝ) : psiT β e = phiT (β / 2) e ^ 2 := by
  unfold psiT phiT
  have h : -(β * e) = -(β / 2 * e) + -(β / 2 * e) := by ring
  rw [h, Real.exp_add]
  ring

theorem phiT_nonneg (β : ℝ) {e : ℝ} (he : 0 ≤ e) : 0 ≤ phiT β e :=
  mul_nonneg he (Real.exp_pos _).le

theorem psiT_nonneg (β e : ℝ) : 0 ≤ psiT β e :=
  mul_nonneg (sq_nonneg e) (Real.exp_pos _).le

/-- **Lemma C**, the maximum: `φ(e) ≤ e^{-1}/β = φ(1/β)`. -/
theorem phiT_le_max {β : ℝ} (hβ : 0 < β) (e : ℝ) : phiT β e ≤ Real.exp (-1) / β := by
  rw [le_div_iff₀ hβ]
  have h2 : β * e ≤ Real.exp (β * e - 1) := by linarith [Real.add_one_le_exp (β * e - 1)]
  have h3 : Real.exp (β * e - 1) * Real.exp (-(β * e)) = Real.exp (-1) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc phiT β e * β = (β * e) * Real.exp (-(β * e)) := by unfold phiT; ring
    _ ≤ Real.exp (β * e - 1) * Real.exp (-(β * e)) :=
        mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
    _ = Real.exp (-1) := h3

theorem phiT_at_peak {β : ℝ} (hβ : 0 < β) : phiT β (1 / β) = Real.exp (-1) / β := by
  unfold phiT
  rw [mul_one_div_cancel hβ.ne']
  ring

/-- **Lemma C**: `φ` is increasing on `[0, 1/β]`. -/
theorem phiT_mono {β x y : ℝ} (hβ : 0 < β) (hx : 0 ≤ x) (hxy : x ≤ y) (hy : y ≤ 1 / β) :
    phiT β x ≤ phiT β y := by
  have hy' : β * y ≤ 1 := by
    rw [le_div_iff₀ hβ] at hy
    linarith
  have hy0 : 0 ≤ y := hx.trans hxy
  have h1 : 1 - β * (y - x) ≤ Real.exp (-(β * (y - x))) := by
    linarith [Real.add_one_le_exp (-(β * (y - x)))]
  have h2 : x ≤ y * Real.exp (-(β * (y - x))) := by
    have h4 := mul_le_mul_of_nonneg_left h1 hy0
    nlinarith [mul_nonneg (sub_nonneg.2 hy') (sub_nonneg.2 hxy)]
  have h3 : Real.exp (-(β * (y - x))) * Real.exp (-(β * x)) = Real.exp (-(β * y)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  unfold phiT
  calc x * Real.exp (-(β * x)) ≤ y * Real.exp (-(β * (y - x))) * Real.exp (-(β * x)) :=
        mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
    _ = y * Real.exp (-(β * y)) := by rw [mul_assoc, h3]

/-- **Lemma C**: `φ` is decreasing on `[1/β, ∞)`. -/
theorem phiT_anti {β x y : ℝ} (hβ : 0 < β) (hx : 1 / β ≤ x) (hxy : x ≤ y) :
    phiT β y ≤ phiT β x := by
  have hx' : 1 ≤ β * x := by
    rw [div_le_iff₀ hβ] at hx
    linarith
  have hx0 : 0 ≤ x := by
    have : 0 < 1 / β := by positivity
    linarith
  have h1 : 1 + β * (y - x) ≤ Real.exp (β * (y - x)) := by
    linarith [Real.add_one_le_exp (β * (y - x))]
  have h2 : y ≤ x * Real.exp (β * (y - x)) := by
    have h4 := mul_le_mul_of_nonneg_left h1 hx0
    nlinarith [mul_nonneg (sub_nonneg.2 hx') (sub_nonneg.2 hxy)]
  have h3 : Real.exp (β * (y - x)) * Real.exp (-(β * y)) = Real.exp (-(β * x)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  unfold phiT
  calc y * Real.exp (-(β * y)) ≤ x * Real.exp (β * (y - x)) * Real.exp (-(β * y)) :=
        mul_le_mul_of_nonneg_right h2 (Real.exp_pos _).le
    _ = x * Real.exp (-(β * x)) := by rw [mul_assoc, h3]

/-- **Lemma C**, infimum on `[ℓ, μ]`: `min(φ(ℓ), φ(μ)) ≤ φ(e)`. -/
theorem phiT_ge_min {β ℓ μ e : ℝ} (hβ : 0 < β) (hℓ : 0 ≤ ℓ) (h1 : ℓ ≤ e) (h2 : e ≤ μ) :
    min (phiT β ℓ) (phiT β μ) ≤ phiT β e := by
  by_cases h : e ≤ 1 / β
  · exact (min_le_left _ _).trans (phiT_mono hβ hℓ h1 h)
  · exact (min_le_right _ _).trans (phiT_anti hβ (le_of_lt (not_le.mp h)) h2)

/-- **Lemma C**, supremum when `ℓ ≥ 1/β`: `φ(e) ≤ φ(ℓ)`. -/
theorem phiT_le_left {β ℓ e : ℝ} (hβ : 0 < β) (hℓ : 1 / β ≤ ℓ) (h1 : ℓ ≤ e) :
    phiT β e ≤ phiT β ℓ :=
  phiT_anti hβ hℓ h1

/-- **Lemma C**, supremum when `μ ≤ 1/β`: `φ(e) ≤ φ(μ)`. -/
theorem phiT_le_right {β μ e : ℝ} (hβ : 0 < β) (he : 0 ≤ e) (hμ : μ ≤ 1 / β) (h2 : e ≤ μ) :
    phiT β e ≤ phiT β μ :=
  phiT_mono hβ he h2 hμ

/-- **Lemma C** for `ψ`, the maximum: `ψ(e) ≤ 4 e^{-2}/β² = ψ(2/β)` for `e ≥ 0`. -/
theorem psiT_le_max {β : ℝ} (hβ : 0 < β) {e : ℝ} (he : 0 ≤ e) :
    psiT β e ≤ 4 * Real.exp (-2) / β ^ 2 := by
  have hβ2 : 0 < β / 2 := by positivity
  rw [psiT_eq]
  have h := phiT_le_max hβ2 e
  have h0 := phiT_nonneg (β / 2) he
  have hsq : phiT (β / 2) e ^ 2 ≤ (Real.exp (-1) / (β / 2)) ^ 2 := pow_le_pow_left₀ h0 h 2
  have e2 : (Real.exp (-1) / (β / 2)) ^ 2 = 4 * Real.exp (-2) / β ^ 2 := by
    have : Real.exp (-1) ^ 2 = Real.exp (-2) := by
      rw [sq, ← Real.exp_add]
      norm_num
    rw [div_pow, this]
    field_simp
    ring
  linarith

/-- **Lemma C** for `ψ`: increasing on `[0, 2/β]`. -/
theorem psiT_mono {β x y : ℝ} (hβ : 0 < β) (hx : 0 ≤ x) (hxy : x ≤ y) (hy : y ≤ 2 / β) :
    psiT β x ≤ psiT β y := by
  have hβ2 : 0 < β / 2 := by positivity
  have hy' : y ≤ 1 / (β / 2) := by rw [one_div_div]; exact hy
  rw [psiT_eq, psiT_eq]
  exact pow_le_pow_left₀ (phiT_nonneg _ hx) (phiT_mono hβ2 hx hxy hy') 2

/-- **Lemma C** for `ψ`: decreasing on `[2/β, ∞)`. -/
theorem psiT_anti {β x y : ℝ} (hβ : 0 < β) (hx : 2 / β ≤ x) (hxy : x ≤ y) :
    psiT β y ≤ psiT β x := by
  have hβ2 : 0 < β / 2 := by positivity
  have hx' : 1 / (β / 2) ≤ x := by rw [one_div_div]; exact hx
  have hx0 : 0 ≤ x := by
    have : 0 < 2 / β := by positivity
    linarith
  rw [psiT_eq, psiT_eq]
  exact pow_le_pow_left₀ (phiT_nonneg _ (hx0.trans hxy)) (phiT_anti hβ2 hx' hxy) 2

theorem psiT_ge_min {β ℓ μ e : ℝ} (hβ : 0 < β) (hℓ : 0 ≤ ℓ) (h1 : ℓ ≤ e) (h2 : e ≤ μ) :
    min (psiT β ℓ) (psiT β μ) ≤ psiT β e := by
  by_cases h : e ≤ 2 / β
  · exact (min_le_left _ _).trans (psiT_mono hβ hℓ h1 h)
  · exact (min_le_right _ _).trans (psiT_anti hβ (le_of_lt (not_le.mp h)) h2)

theorem psiT_le_left {β ℓ e : ℝ} (hβ : 0 < β) (hℓ : 2 / β ≤ ℓ) (h1 : ℓ ≤ e) :
    psiT β e ≤ psiT β ℓ :=
  psiT_anti hβ hℓ h1

theorem psiT_le_right {β μ e : ℝ} (hβ : 0 < β) (he : 0 ≤ e) (hμ : μ ≤ 2 / β) (h2 : e ≤ μ) :
    psiT β e ≤ psiT β μ :=
  psiT_mono hβ he h2 hμ

/-! ### Sums over a subset with interval counts -/

theorem sum_fiber_bounds_on {n K : Type*} [Fintype K] [DecidableEq K] (s : Finset n)
    (κ : n → K) (f : n → ℝ) (lo hi : K → ℝ) (hf : ∀ i ∈ s, lo (κ i) ≤ f i ∧ f i ≤ hi (κ i)) :
    ∑ k, ((s.filter fun i => κ i = k).card : ℝ) * lo k ≤ ∑ i ∈ s, f i ∧
    ∑ i ∈ s, f i ≤ ∑ k, ((s.filter fun i => κ i = k).card : ℝ) * hi k := by
  rw [← Finset.sum_fiberwise s κ f]
  constructor
  · refine Finset.sum_le_sum fun k _ => ?_
    rw [← nsmul_eq_mul]
    refine Finset.card_nsmul_le_sum _ _ _ fun i hi => ?_
    have hi' := Finset.mem_filter.mp hi
    rw [← hi'.2]
    exact (hf i hi'.1).1
  · refine Finset.sum_le_sum fun k _ => ?_
    rw [← nsmul_eq_mul]
    refine Finset.sum_le_card_nsmul _ _ _ fun i hi => ?_
    have hi' := Finset.mem_filter.mp hi
    rw [← hi'.2]
    exact (hf i hi'.1).2

theorem sum_const_bounds {n : Type*} (s : Finset n) (f : n → ℝ) {lo hi : ℝ}
    (hf : ∀ i ∈ s, lo ≤ f i ∧ f i ≤ hi) :
    (s.card : ℝ) * lo ≤ ∑ i ∈ s, f i ∧ ∑ i ∈ s, f i ≤ (s.card : ℝ) * hi := by
  constructor
  · rw [← nsmul_eq_mul]
    exact Finset.card_nsmul_le_sum _ _ _ fun i hi => (hf i hi).1
  · rw [← nsmul_eq_mul]
    exact Finset.sum_le_card_nsmul _ _ _ fun i hi => (hf i hi).2

/-! ### Proposition 16' (ii) and (iii) -/

section Spectrum

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]
  {A : Matrix n n 𝕜} (hA : A.IsHermitian)

/-- **C-06 (ii)** for `φ`: the omitted first moments lie in `(n - k) [lo, hi]` for any `lo`,
`hi` bounding `φ` on `[ℓ, μ]` (Lemma C supplies them). -/
theorem c06_tail_phi (L M E0 β : ℝ) (hM : ∀ i, hA.eigenvalues i ≤ M) {lo hi : ℝ}
    (hlohi : ∀ e, L - E0 ≤ e → e ≤ M - E0 → lo ≤ phiT β e ∧ phiT β e ≤ hi) :
    ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ) * lo ≤
      ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L),
        phiT β (hA.eigenvalues i - E0) ∧
    ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L), phiT β (hA.eigenvalues i - E0) ≤
      ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ) * hi :=
  sum_const_bounds _ _ fun i hi => by
    have h := (Finset.mem_filter.mp hi).2
    exact hlohi _ (by linarith [not_lt.mp h]) (by linarith [hM i])

/-- **C-06 (ii)** for `ψ`. -/
theorem c06_tail_psi (L M E0 β : ℝ) (hM : ∀ i, hA.eigenvalues i ≤ M) {lo hi : ℝ}
    (hlohi : ∀ e, L - E0 ≤ e → e ≤ M - E0 → lo ≤ psiT β e ∧ psiT β e ≤ hi) :
    ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ) * lo ≤
      ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L),
        psiT β (hA.eigenvalues i - E0) ∧
    ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L), psiT β (hA.eigenvalues i - E0) ≤
      ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ) * hi :=
  sum_const_bounds _ _ fun i hi => by
    have h := (Finset.mem_filter.mp hi).2
    exact hlohi _ (by linarith [not_lt.mp h]) (by linarith [hM i])

/-- Splitting a sum over all eigen-indices into the claimed ones (`λ_i < L`) and the omitted
ones, each part enclosed. -/
theorem sum_split_bounds (L : ℝ) (f : n → ℝ) {lo1 hi1 lo2 hi2 : ℝ}
    (h1 : lo1 ≤ ∑ i ∈ Finset.univ.filter (fun i => hA.eigenvalues i < L), f i ∧
      ∑ i ∈ Finset.univ.filter (fun i => hA.eigenvalues i < L), f i ≤ hi1)
    (h2 : lo2 ≤ ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L), f i ∧
      ∑ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L), f i ≤ hi2) :
    lo1 + lo2 ≤ ∑ i, f i ∧ ∑ i, f i ≤ hi1 + hi2 := by
  rw [← Finset.sum_filter_add_sum_filter_not Finset.univ (fun i => hA.eigenvalues i < L) f]
  exact ⟨add_le_add h1.1 h2.1, add_le_add h1.2 h2.2⟩

variable {K : Type*} [Fintype K] [DecidableEq K]

/-- **C-06 (iii)**, the three sums. The claimed eigenvalues `λ_i < L` lie in their intervals
`[l (κ i), h (κ i)]` (counts `c_k`), all eigenvalues are at most `M`, `E_0 ≤ l_k`, and the bounds
`φlo, φhi, ψlo, ψhi` hold on `[ℓ, μ] = [L - E_0, M - E_0]` (Lemma C). -/
theorem c06_sums (κ : n → K) (l h : K → ℝ) (L M E0 β : ℝ) (hβ : 0 ≤ β)
    (hin : ∀ i, hA.eigenvalues i < L → l (κ i) ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ h (κ i))
    (hE0 : ∀ k, E0 ≤ l k) (hM : ∀ i, hA.eigenvalues i ≤ M) {φlo φhi ψlo ψhi : ℝ}
    (hφ : ∀ e, L - E0 ≤ e → e ≤ M - E0 → φlo ≤ phiT β e ∧ phiT β e ≤ φhi)
    (hψ : ∀ e, L - E0 ≤ e → e ≤ M - E0 → ψlo ≤ psiT β e ∧ psiT β e ≤ ψhi) :
    let S := Finset.univ.filter fun i => hA.eigenvalues i < L
    let c : K → ℝ := fun k => ((S.filter fun i => κ i = k).card : ℝ)
    let m : ℝ := ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ)
    (∑ k, c k * Real.exp (-(β * (h k - E0))) + m * Real.exp (-(β * (M - E0))) ≤
        ∑ i, Real.exp (-(β * (hA.eigenvalues i - E0))) ∧
      ∑ i, Real.exp (-(β * (hA.eigenvalues i - E0))) ≤
        ∑ k, c k * Real.exp (-(β * (l k - E0))) + m * Real.exp (-(β * (L - E0)))) ∧
    (∑ k, c k * ((l k - E0) * Real.exp (-(β * (h k - E0)))) + m * φlo ≤
        ∑ i, phiT β (hA.eigenvalues i - E0) ∧
      ∑ i, phiT β (hA.eigenvalues i - E0) ≤
        ∑ k, c k * ((h k - E0) * Real.exp (-(β * (l k - E0)))) + m * φhi) ∧
    (∑ k, c k * ((l k - E0) ^ 2 * Real.exp (-(β * (h k - E0)))) + m * ψlo ≤
        ∑ i, psiT β (hA.eigenvalues i - E0) ∧
      ∑ i, psiT β (hA.eigenvalues i - E0) ≤
        ∑ k, c k * ((h k - E0) ^ 2 * Real.exp (-(β * (l k - E0)))) + m * ψhi) := by
  intro S c m
  have hexp : ∀ {x y : ℝ}, x ≤ y → Real.exp (-(β * y)) ≤ Real.exp (-(β * x)) := fun hxy =>
    Real.exp_le_exp.mpr (by nlinarith)
  have htail : ∀ i ∈ Finset.univ.filter (fun i => ¬ hA.eigenvalues i < L),
      L - E0 ≤ hA.eigenvalues i - E0 ∧ hA.eigenvalues i - E0 ≤ M - E0 := fun i hi => by
    have h' := (Finset.mem_filter.mp hi).2
    exact ⟨by linarith [not_lt.mp h'], by linarith [hM i]⟩
  refine ⟨?_, ?_, ?_⟩
  · refine sum_split_bounds hA L _ (sum_fiber_bounds_on S κ _ _ _ fun i hi => ?_)
      (sum_const_bounds _ _ fun i hi => ⟨hexp (htail i hi).2, hexp (htail i hi).1⟩)
    have hb := hin i (Finset.mem_filter.mp hi).2
    exact ⟨hexp (by linarith [hb.2]), hexp (by linarith [hb.1])⟩
  · refine sum_split_bounds hA L _ (sum_fiber_bounds_on S κ _ _ _ fun i hi => ?_)
      (sum_const_bounds _ _ fun i hi => hφ _ (htail i hi).1 (htail i hi).2)
    have hb := hin i (Finset.mem_filter.mp hi).2
    have h0 : 0 ≤ l (κ i) - E0 := by linarith [hE0 (κ i)]
    unfold phiT
    constructor
    · exact mul_le_mul (by linarith [hb.1]) (hexp (by linarith [hb.2]))
        (Real.exp_pos _).le (by linarith [hb.1])
    · exact mul_le_mul (by linarith [hb.2]) (hexp (by linarith [hb.1]))
        (Real.exp_pos _).le (by linarith [hb.1, hb.2])
  · refine sum_split_bounds hA L _ (sum_fiber_bounds_on S κ _ _ _ fun i hi => ?_)
      (sum_const_bounds _ _ fun i hi => hψ _ (htail i hi).1 (htail i hi).2)
    have hb := hin i (Finset.mem_filter.mp hi).2
    have h0 : 0 ≤ l (κ i) - E0 := by linarith [hE0 (κ i)]
    unfold psiT
    constructor
    · exact mul_le_mul (pow_le_pow_left₀ h0 (by linarith [hb.1]) 2) (hexp (by linarith [hb.2]))
        (Real.exp_pos _).le (sq_nonneg _)
    · exact mul_le_mul (pow_le_pow_left₀ (by linarith [hb.1]) (by linarith [hb.2]) 2)
        (hexp (by linarith [hb.1])) (Real.exp_pos _).le (sq_nonneg _)

/-- **C-06 (iii)** without the bound `M` (the case `μ = ∞` of the document): the omitted terms are
enclosed by `[0, e^{-β ℓ}]`, `[0, φhi]` and `[0, ψhi]`, for bounds `φhi`, `ψhi` of `φ` and `ψ` on
`[ℓ, ∞)` (by Lemma C, `φ(max(1/β, ℓ))` and `ψ(max(2/β, ℓ))`). -/
theorem c06_sums_noM (κ : n → K) (l h : K → ℝ) (L E0 β : ℝ) (hβ : 0 ≤ β)
    (hin : ∀ i, hA.eigenvalues i < L → l (κ i) ≤ hA.eigenvalues i ∧ hA.eigenvalues i ≤ h (κ i))
    (hE0 : ∀ k, E0 ≤ l k) (hL : E0 ≤ L) {φhi ψhi : ℝ}
    (hφ : ∀ e, L - E0 ≤ e → phiT β e ≤ φhi) (hψ : ∀ e, L - E0 ≤ e → psiT β e ≤ ψhi) :
    let S := Finset.univ.filter fun i => hA.eigenvalues i < L
    let c : K → ℝ := fun k => ((S.filter fun i => κ i = k).card : ℝ)
    let m : ℝ := ((Finset.univ.filter fun i => ¬ hA.eigenvalues i < L).card : ℝ)
    (∑ k, c k * Real.exp (-(β * (h k - E0))) ≤
        ∑ i, Real.exp (-(β * (hA.eigenvalues i - E0))) ∧
      ∑ i, Real.exp (-(β * (hA.eigenvalues i - E0))) ≤
        ∑ k, c k * Real.exp (-(β * (l k - E0))) + m * Real.exp (-(β * (L - E0)))) ∧
    (∑ k, c k * ((l k - E0) * Real.exp (-(β * (h k - E0)))) ≤
        ∑ i, phiT β (hA.eigenvalues i - E0) ∧
      ∑ i, phiT β (hA.eigenvalues i - E0) ≤
        ∑ k, c k * ((h k - E0) * Real.exp (-(β * (l k - E0)))) + m * φhi) ∧
    (∑ k, c k * ((l k - E0) ^ 2 * Real.exp (-(β * (h k - E0)))) ≤
        ∑ i, psiT β (hA.eigenvalues i - E0) ∧
      ∑ i, psiT β (hA.eigenvalues i - E0) ≤
        ∑ k, c k * ((h k - E0) ^ 2 * Real.exp (-(β * (l k - E0)))) + m * ψhi) := by
  intro S c m
  have hM : ∀ i, hA.eigenvalues i ≤ ∑ j, |hA.eigenvalues j| := fun i =>
    (le_abs_self _).trans (Finset.single_le_sum (f := fun j => |hA.eigenvalues j|)
      (fun j _ => abs_nonneg _) (Finset.mem_univ i))
  have h0 := c06_sums hA κ l h L (∑ j, |hA.eigenvalues j|) E0 β hβ hin hE0 hM
    (φlo := 0) (ψlo := 0)
    (fun e he _ => ⟨phiT_nonneg β (by linarith), hφ e he⟩)
    (fun e he _ => ⟨psiT_nonneg β e, hψ e he⟩)
  obtain ⟨⟨z1, z2⟩, ⟨p1, p2⟩, ⟨q1, q2⟩⟩ := h0
  have hm : 0 ≤ m := Nat.cast_nonneg _
  have hz : 0 ≤ m * Real.exp (-(β * (∑ j, |hA.eigenvalues j| - E0))) :=
    mul_nonneg hm (Real.exp_pos _).le
  exact ⟨⟨(le_add_of_nonneg_right hz).trans z1, z2⟩,
    ⟨(le_add_of_nonneg_right (mul_nonneg hm le_rfl)).trans p1, p2⟩,
    ⟨(le_add_of_nonneg_right (mul_nonneg hm le_rfl)).trans q1, q2⟩⟩

end Spectrum

/-! ### The shift identity of Proposition 16 -/

theorem mean_shift {ι : Type*} (s : Finset ι) (w e : ι → ℝ) (c : ℝ) (hZ : ∑ i ∈ s, w i ≠ 0) :
    (∑ i ∈ s, w i * (e i + c)) / ∑ i ∈ s, w i = (∑ i ∈ s, w i * e i) / ∑ i ∈ s, w i + c := by
  have h : ∑ i ∈ s, w i * (e i + c) = ∑ i ∈ s, w i * e i + c * ∑ i ∈ s, w i := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [h]
  field_simp

/-- `Var(e + c) = Var(e)`: `⟨(e + c)²⟩ - ⟨e + c⟩² = ⟨e²⟩ - ⟨e⟩²`. -/
theorem variance_shift {ι : Type*} (s : Finset ι) (w e : ι → ℝ) (c : ℝ)
    (hZ : ∑ i ∈ s, w i ≠ 0) :
    (∑ i ∈ s, w i * (e i + c) ^ 2) / ∑ i ∈ s, w i -
        ((∑ i ∈ s, w i * (e i + c)) / ∑ i ∈ s, w i) ^ 2 =
      (∑ i ∈ s, w i * e i ^ 2) / ∑ i ∈ s, w i - ((∑ i ∈ s, w i * e i) / ∑ i ∈ s, w i) ^ 2 := by
  have h1 : ∑ i ∈ s, w i * (e i + c) ^ 2 =
      ∑ i ∈ s, w i * e i ^ 2 + 2 * c * ∑ i ∈ s, w i * e i + c ^ 2 * ∑ i ∈ s, w i := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  have h2 : ∑ i ∈ s, w i * (e i + c) = ∑ i ∈ s, w i * e i + c * ∑ i ∈ s, w i := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [h1, h2]
  field_simp
  ring

end CertifyMathlib.Counting
