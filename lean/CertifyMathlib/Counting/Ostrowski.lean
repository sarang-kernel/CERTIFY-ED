import CertifyMathlib.Counting.Theorem6

/-!
# Ostrowski's theorem in counting form (E-07, Proposition 2 (b))

Source: `docs/proofs/E-07_davis_kahan.md`, proof of Proposition 2 (b), in the code repository:
`Xᴴ S X = M (Q̂ᴴ S Q̂) M` with `M` Hermitian positive definite, so by Ostrowski's theorem
(Horn and Johnson, Theorem 4.5.9) `λ_i(Xᴴ S X) = θ_i λ_i(Q̂ᴴ S Q̂)` with `θ_i ∈ [1 - η, 1 + η]`.

* `ostrowski_pos`, `ostrowski_neg`: for Hermitian `K`, a Hermitian invertible `M` with
  `a ‖x‖² ≤ ‖M x‖² ≤ b ‖x‖²` (`0 < a`) and `N = M K M`: for `t ≥ 0`,
  `#{λ(K) < t / b} ≤ #{λ(N) < t} ≤ #{λ(K) < t / a}`, and for `t ≤ 0` the same with `a` and `b`
  exchanged. This is the counting form of `λ_i(N) = θ_i λ_i(K)`, `θ_i ∈ [a, b]`.
* `ostrowski_polar`: the case of the document, `K = Q̂ᴴ S Q̂`, `N = Xᴴ S X`, `M = (Xᴴ X)^{1/2}`,
  `a = 1 - η`, `b = 1 + η`.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {g : Type*} [Fintype g] [DecidableEq g]

theorem qf_one (x : g → 𝕜) : qf (1 : Matrix g g 𝕜) x = nsq x := by
  simp only [qf, one_mulVec]
  exact re_star_dotProduct_self x

/-- `‖M x‖² = xᴴ Mᴴ M x`. -/
theorem nsq_mulVec (M : Matrix g g 𝕜) (x : g → 𝕜) : nsq (M *ᵥ x) = qf (Mᴴ * M) x := by
  have h := qf_congr Mᴴ (1 : Matrix g g 𝕜) x
  rw [conjTranspose_conjTranspose, Matrix.mul_one] at h
  rw [h, qf_one]

/-- Transport of a negative subspace by an invertible matrix `P` under a comparison of the two
quadratic forms. -/
theorem cntLt_le_of_transport {K N : Matrix g g 𝕜} (hK : K.IsHermitian) (hN : N.IsHermitian)
    (P Pinv : Matrix g g 𝕜) (h1 : P * Pinv = 1) (h2 : Pinv * P = 1) {s t : ℝ}
    (hrel : ∀ y, y ≠ 0 → qf K y < s * nsq y → qf N (P *ᵥ y) < t * nsq (P *ᵥ y)) :
    cntLt hK s ≤ cntLt hN t := by
  obtain ⟨V, hdim, hV⟩ := exists_negOn_cntLt hK s
  have hW : NegOn (N - (t : 𝕜) • (1 : Matrix g g 𝕜))
      (V.map (mulVecEquiv P Pinv h1 h2 : (g → 𝕜) →ₗ[𝕜] (g → 𝕜))) := by
    intro x hx hx0
    obtain ⟨y, hy, rfl⟩ := Submodule.mem_map.mp hx
    have hy0 : y ≠ 0 := by
      rintro rfl
      exact hx0 (map_zero _)
    have hneg := hV y hy hy0
    rw [qf_sub_smul_one] at hneg
    have hlt := hrel y hy0 (by linarith)
    show qf (N - (t : 𝕜) • (1 : Matrix g g 𝕜)) (mulVecEquiv P Pinv h1 h2 y) < 0
    rw [qf_sub_smul_one, mulVecEquiv_apply]
    linarith
  calc cntLt hK s ≤ Module.finrank 𝕜 V := hdim
    _ = Module.finrank 𝕜 (V.map (mulVecEquiv P Pinv h1 h2 : (g → 𝕜) →ₗ[𝕜] (g → 𝕜))) :=
        (finrank_map_equiv (mulVecEquiv P Pinv h1 h2) V).symm
    _ ≤ cntLt hN t := finrank_le_cntLt hN t _ hW

theorem cntLt_congr {A B : Matrix g g 𝕜} (hA : A.IsHermitian) (hB : B.IsHermitian) (h : A = B)
    (t : ℝ) : cntLt hA t = cntLt hB t := by
  subst h
  rfl

omit [DecidableEq g] in
theorem qf_MKM {K M : Matrix g g 𝕜} (hM : Mᴴ = M) (x : g → 𝕜) :
    qf (M * K * M) x = qf K (M *ᵥ x) := by
  have h := qf_congr M K x
  rw [hM] at h
  exact h

section Ostrowski

variable {K M Mi : Matrix g g 𝕜} (hK : K.IsHermitian) (hM : Mᴴ = M) (h1 : M * Mi = 1)
  (h2 : Mi * M = 1) {a b : ℝ} (ha : 0 < a) (hab : a ≤ b)
  (hMa : ∀ x, a * nsq x ≤ nsq (M *ᵥ x)) (hMb : ∀ x, nsq (M *ᵥ x) ≤ b * nsq x)
include hK hM h1 h2 ha hab hMa hMb

/-- **Ostrowski, `t ≥ 0`**: `#{λ(K) < t/b} ≤ #{λ(M K M) < t} ≤ #{λ(K) < t/a}`. -/
theorem ostrowski_pos (hN : (M * K * M).IsHermitian) {t : ℝ} (ht : 0 ≤ t) :
    cntLt hK (t / b) ≤ cntLt hN t ∧ cntLt hN t ≤ cntLt hK (t / a) := by
  have hb : 0 < b := lt_of_lt_of_le ha hab
  constructor
  · refine cntLt_le_of_transport hK hN Mi M h2 h1 (fun y _ hy => ?_)
    rw [qf_MKM hM, mulVec_mulVec, h1, one_mulVec]
    have e1 := hMb (Mi *ᵥ y)
    rw [mulVec_mulVec, h1, one_mulVec] at e1
    have e2 : t / b * nsq y ≤ t * nsq (Mi *ᵥ y) := by
      rw [div_mul_eq_mul_div, div_le_iff₀ hb]
      nlinarith
    linarith
  · refine cntLt_le_of_transport hN hK M Mi h1 h2 (fun x _ hx => ?_)
    rw [qf_MKM hM] at hx
    have e1 := hMa x
    have e2 : t * nsq x ≤ t / a * nsq (M *ᵥ x) := by
      rw [div_mul_eq_mul_div, le_div_iff₀ ha]
      nlinarith
    linarith

/-- **Ostrowski, `t ≤ 0`**: `#{λ(K) < t/a} ≤ #{λ(M K M) < t} ≤ #{λ(K) < t/b}`. -/
theorem ostrowski_neg (hN : (M * K * M).IsHermitian) {t : ℝ} (ht : t ≤ 0) :
    cntLt hK (t / a) ≤ cntLt hN t ∧ cntLt hN t ≤ cntLt hK (t / b) := by
  have hb : 0 < b := lt_of_lt_of_le ha hab
  constructor
  · refine cntLt_le_of_transport hK hN Mi M h2 h1 (fun y _ hy => ?_)
    rw [qf_MKM hM, mulVec_mulVec, h1, one_mulVec]
    have e1 := hMa (Mi *ᵥ y)
    rw [mulVec_mulVec, h1, one_mulVec] at e1
    have e2 : t / a * nsq y ≤ t * nsq (Mi *ᵥ y) := by
      rw [div_mul_eq_mul_div, div_le_iff₀ ha]
      nlinarith
    linarith
  · refine cntLt_le_of_transport hN hK M Mi h1 h2 (fun x _ hx => ?_)
    rw [qf_MKM hM] at hx
    have e1 := hMb x
    have e2 : t * nsq x ≤ t / b * nsq (M *ᵥ x) := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hb]
      nlinarith
    linarith

end Ostrowski

/-- **E-07, Proposition 2 (b), Ostrowski step**: the eigenvalues of `Xᴴ S X` and of `Q̂ᴴ S Q̂`,
in counting form with `a = 1 - η`, `b = 1 + η`. -/
theorem ostrowski_polar {n : Type*} [Fintype n] [DecidableEq n] {X : Matrix n g 𝕜} {η : ℝ}
    (hη1 : η < 1) (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) {S : Matrix n n 𝕜}
    (hK : ((polarMat X)ᴴ * S * polarMat X).IsHermitian) (hN : (Xᴴ * S * X).IsHermitian)
    (hη0 : 0 ≤ η) (t : ℝ) :
    (0 ≤ t → cntLt hK (t / (1 + η)) ≤ cntLt hN t ∧ cntLt hN t ≤ cntLt hK (t / (1 - η))) ∧
    (t ≤ 0 → cntLt hK (t / (1 - η)) ≤ cntLt hN t ∧ cntLt hN t ≤ cntLt hK (t / (1 + η))) := by
  have hNeq : Xᴴ * S * X =
      polarSqrt X * ((polarMat X)ᴴ * S * polarMat X) * polarSqrt X := by
    conv_lhs => rw [← polarMat_mul_polarSqrt hη1 hη]
    rw [conjTranspose_mul, polarSqrt_conjTranspose]
    simp only [Matrix.mul_assoc]
  have hN' : (polarSqrt X * ((polarMat X)ᴴ * S * polarMat X) * polarSqrt X).IsHermitian :=
    hNeq ▸ hN
  have hcnt : ∀ s, cntLt hN s = cntLt hN' s := fun s => cntLt_congr hN hN' hNeq s
  have hq : ∀ x, qf (Xᴴ * X) x - nsq x ≤ η * nsq x ∧ -(η * nsq x) ≤ qf (Xᴴ * X) x - nsq x :=
    fun x => by
    have h := abs_qf_le_of_toEuclideanLin (Xᴴ * X - 1) η (fun z => by
      rw [map_sub, LinearMap.sub_apply, toEuclideanLin_one_apply]
      exact hη z) x
    rw [qf_sub, qf_one] at h
    exact ⟨(abs_le.mp h).2, (abs_le.mp h).1⟩
  have hsq : ∀ x, nsq (polarSqrt X *ᵥ x) = qf (Xᴴ * X) x := fun x => by
    rw [nsq_mulVec, polarSqrt_conjTranspose, polarSqrt_mul_self hη1 hη]
  have hMa : ∀ x, (1 - η) * nsq x ≤ nsq (polarSqrt X *ᵥ x) := fun x => by
    rw [hsq]
    linarith [(hq x).2]
  have hMb : ∀ x, nsq (polarSqrt X *ᵥ x) ≤ (1 + η) * nsq x := fun x => by
    rw [hsq]
    linarith [(hq x).1]
  have ha : (0 : ℝ) < 1 - η := by linarith
  have hab : 1 - η ≤ 1 + η := by linarith
  refine ⟨fun ht => ?_, fun ht => ?_⟩
  · rw [hcnt]
    exact ostrowski_pos hK (polarSqrt_conjTranspose X) (polarSqrt_mul_inv hη1 hη)
      (polarSqrtInv_mul hη1 hη) ha hab hMa hMb hN' ht
  · rw [hcnt]
    exact ostrowski_neg hK (polarSqrt_conjTranspose X) (polarSqrt_mul_inv hη1 hη)
      (polarSqrtInv_mul hη1 hη) ha hab hMa hMb hN' ht

end CertifyMathlib.Counting
