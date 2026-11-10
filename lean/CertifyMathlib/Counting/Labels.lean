import CertifyMathlib.Counting.Sorted
import CertifyMathlib.Counting.Ostrowski
import CertifyMathlib.Counting.Observables

/-!
# E-07, Proposition 2 (b) (C-12, labels) and Proposition 3 (C-13, overlaps)

Source: `docs/proofs/E-07_davis_kahan.md` in the code repository, with its notation: `X` (`n × g`)
the claimed vectors, `‖Xᴴ X - I‖ ≤ η < 1`, `Q̂` the polar factor of `X`, `Q` an orthonormal basis
of the range of the spectral projector `P` and `s ≥ ‖(I - Q Qᴴ) Q̂‖`.

* `eigAsc_conj`: the sorted eigenvalues do not change under a unitary change of basis.
* `abs_eigAsc_le`: `|λ_j(K)| ≤ ‖K‖`.
* `ostrowski_sorted`: `|λ_j(Xᴴ S X) - λ_j(Q̂ᴴ S Q̂)| ≤ η |λ_j(Q̂ᴴ S Q̂)|`, the sorted form of
  `ostrowski_polar`.
* `c12_prop2_b`: Proposition 2 (b), `|m̂_j - m_j| ≤ (2 s + 2 s² + η) ‖S‖` for the sorted
  eigenvalues `m̂_j` of `Xᴴ S X` and `m_j` of `Qᴴ S Q`.
* `norm_apply_le_sqrt`, `norm_trace_le_card`, `c13_prop3`: Proposition 3,
  `|tr(P₁ P₂) - tr(X₁ X₁ᴴ X₂ X₂ᴴ)| ≤ k₂ e₁ + g (1 + η₁) e₂` for `P_i = Q_i Q_iᴴ` with
  `‖X_i X_iᴴ - P_i‖ ≤ e_i` (Theorem 6 gives `e_i = s_i + η_i`; the document's bound is the case
  `k₂ = g`), and `Re tr(X₁ X₁ᴴ X₂ X₂ᴴ) = ‖X₁ᴴ X₂‖_F²`.
-/

namespace CertifyMathlib.Counting

open Matrix
open scoped InnerProductSpace

variable {𝕜 : Type*} [RCLike 𝕜] {n g : Type*} [Fintype n] [DecidableEq n] [Fintype g]
  [DecidableEq g]

/-- Sorted eigenvalues do not change under a unitary change of basis `K = Uᴴ C U`. -/
theorem eigAsc_conj {C K : Matrix g g 𝕜} (hC : C.IsHermitian) (hK : K.IsHermitian)
    (U : Matrix g g 𝕜) (hU1 : Uᴴ * U = 1) (hU2 : U * Uᴴ = 1) (hKeq : K = Uᴴ * C * U)
    (j : Fin (Fintype.card g)) : eigAsc hK j = eigAsc hC j := by
  have hc : ∀ t, cntLe hK t = cntLe hC t := fun t => (cnt_conj hC U hU1 hU2 hK hKeq t).2
  apply le_antisymm
  · rw [eigAsc_le_iff, hc]
    exact (eigAsc_le_iff hC j _).mp le_rfl
  · rw [eigAsc_le_iff, ← hc]
    exact (eigAsc_le_iff hK j _).mp le_rfl

/-- `|λ_j(K)| ≤ ‖K‖`. -/
theorem abs_eigAsc_le {K : Matrix g g 𝕜} (hK : K.IsHermitian) {nK : ℝ}
    (h : ∀ z, ‖toEuclideanLin K z‖ ≤ nK * ‖z‖) (j : Fin (Fintype.card g)) :
    |eigAsc hK j| ≤ nK := by
  have h0 : (0 : Matrix g g 𝕜).IsHermitian := isHermitian_zero
  have hz : eigAsc h0 j = 0 := by
    simp [eigAsc_eq, h0.eigenvalues_eq_zero_iff.mpr rfl]
  have := weyl_sorted_opNorm hK h0 nK (fun z => by rw [sub_zero]; exact h z) j
  rwa [hz, sub_zero] at this

/-- **Ostrowski, sorted form**: `|λ_j(Xᴴ S X) - λ_j(Q̂ᴴ S Q̂)| ≤ η |λ_j(Q̂ᴴ S Q̂)|`. -/
theorem ostrowski_sorted {X : Matrix n g 𝕜} {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) {S : Matrix n n 𝕜}
    (hK : ((polarMat X)ᴴ * S * polarMat X).IsHermitian) (hN : (Xᴴ * S * X).IsHermitian)
    (j : Fin (Fintype.card g)) :
    |eigAsc hN j - eigAsc hK j| ≤ η * |eigAsc hK j| := by
  have ho := ostrowski_polar hη1 hη hK hN hη0
  have hkle : ∀ t, t ≤ eigAsc hK j → cntLt hK t ≤ j := fun t ht => (le_eigAsc_iff hK j t).mp ht
  have hklt : ∀ t, eigAsc hK j < t → (j : ℕ) + 1 ≤ cntLt hK t := fun t ht => by
    have := mt (le_eigAsc_iff hK j t).mpr (not_le.mpr ht)
    omega
  have hnle : ∀ t, cntLt hN t ≤ j → t ≤ eigAsc hN j := fun t => (le_eigAsc_iff hN j t).mpr
  have hnlt : ∀ t, (j : ℕ) + 1 ≤ cntLt hN t → eigAsc hN j < t := fun t h => by
    by_contra hc
    have := (le_eigAsc_iff hN j t).mp (not_lt.mp hc)
    omega
  generalize eigAsc hK j = k at hkle hklt ⊢
  generalize eigAsc hN j = m at hnle hnlt ⊢
  have hpos : (0 : ℝ) < 1 - η := by linarith
  have hpos' : (0 : ℝ) < 1 + η := by linarith
  rw [abs_sub_le_iff]
  constructor
  · -- `m ≤ k + η |k|`
    by_contra hc
    rw [not_le] at hc
    have key : ∀ t, k + η * |k| < t → (j : ℕ) + 1 ≤ cntLt hN t := by
      intro t ht
      rcases le_or_gt 0 k with hk | hk
      · rw [abs_of_nonneg hk] at ht
        have ht0 : 0 ≤ t := by nlinarith
        have hlt : k < t / (1 + η) := (lt_div_iff₀ hpos').mpr (by linarith)
        exact (hklt _ hlt).trans ((ho t).1 ht0).1
      · rw [abs_of_neg hk] at ht
        rcases le_or_gt t 0 with ht0 | ht0
        · have hlt : k < t / (1 - η) := (lt_div_iff₀ hpos).mpr (by linarith)
          exact (hklt _ hlt).trans ((ho t).2 ht0).1
        · have h0 := ((ho 0).2 le_rfl).1
          rw [zero_div] at h0
          exact ((hklt 0 hk).trans h0).trans (cntLt_mono hN ht0.le)
    have := hnlt _ (key ((k + η * |k| + m) / 2) (by linarith))
    linarith
  · -- `k - η |k| ≤ m`
    have : k - η * |k| ≤ m := by
      apply hnle
      rcases le_or_gt 0 k with hk | hk
      · have ht : 0 ≤ k - η * |k| := by rw [abs_of_nonneg hk]; nlinarith
        have e : (k - η * |k|) / (1 - η) = k := by
          rw [abs_of_nonneg hk, div_eq_iff hpos.ne']
          ring
        calc cntLt hN (k - η * |k|) ≤ cntLt hK ((k - η * |k|) / (1 - η)) := ((ho _).1 ht).2
          _ = cntLt hK k := by rw [e]
          _ ≤ j := hkle k le_rfl
      · have ht : k - η * |k| ≤ 0 := by rw [abs_of_neg hk]; nlinarith
        have e : (k - η * |k|) / (1 + η) = k := by
          rw [abs_of_neg hk, div_eq_iff hpos'.ne']
          ring
        calc cntLt hN (k - η * |k|) ≤ cntLt hK ((k - η * |k|) / (1 + η)) := ((ho _).2 ht).2
          _ = cntLt hK k := by rw [e]
          _ ≤ j := hkle k le_rfl
    linarith

/-- **E-07, Proposition 2 (b)** (C-12): for the sorted eigenvalues `m̂_j` of `Xᴴ S X` and `m_j` of
the exact compression `Qᴴ S Q`, `|m̂_j - m_j| ≤ (2 s + 2 s² + η) ‖S‖` when
`‖(I - Q Qᴴ) Q̂‖ ≤ s < 1`. -/
theorem c12_prop2_b {X : Matrix n g 𝕜} {η : ℝ} (hη0 : 0 ≤ η) (hη1 : η < 1)
    (hη : ∀ z, ‖toEuclideanLin (Xᴴ * X) z - z‖ ≤ η * ‖z‖) (Q : Matrix n g 𝕜) (hQ : Qᴴ * Q = 1)
    {s : ℝ} (hs0 : 0 ≤ s) (hs1 : s < 1)
    (hY : ∀ z, ‖toEuclideanLin ((1 - Q * Qᴴ) * polarMat X) z‖ ≤ s * ‖z‖)
    {S : Matrix n n 𝕜} (hS : S.IsHermitian) {nS : ℝ} (hnS0 : 0 ≤ nS)
    (hnS : ∀ v, ‖toEuclideanLin S v‖ ≤ nS * ‖v‖)
    (hC : (Qᴴ * S * Q).IsHermitian) (hN : (Xᴴ * S * X).IsHermitian)
    (j : Fin (Fintype.card g)) :
    |eigAsc hN j - eigAsc hC j| ≤ (2 * s + 2 * s ^ 2 + η) * nS := by
  have hQh := polarMat_conjTranspose_mul hη1 hη
  obtain ⟨hU1, hU2, hU⟩ := e07_lemma3 Q (polarMat X) hQ hQh rfl hs0 hs1 hY
  have hK : ((polarMat X)ᴴ * S * polarMat X).IsHermitian :=
    isHermitian_conjTranspose_mul_mul _ hS
  have hK' : ((Q * polarMat (Qᴴ * polarMat X))ᴴ * S *
      (Q * polarMat (Qᴴ * polarMat X))).IsHermitian :=
    isHermitian_conjTranspose_mul_mul _ hS
  have hss : 0 ≤ s + s ^ 2 := by positivity
  -- Ostrowski between `Xᴴ S X` and `Q̂ᴴ S Q̂`
  have h1 := ostrowski_sorted hη0 hη1 hη hK hN j
  have hQhc : ∀ w, ‖toEuclideanLin (polarMat X)ᴴ w‖ ≤ 1 * ‖w‖ :=
    norm_conjTranspose_le _ zero_le_one (fun z => by
      rw [norm_apply_of_isometry _ hQh, one_mul])
  have h1' : |eigAsc hK j| ≤ nS := abs_eigAsc_le hK (fun z => by
    simp only [toEuclideanLin_mul_apply]
    calc ‖toEuclideanLin (polarMat X)ᴴ (toEuclideanLin S (toEuclideanLin (polarMat X) z))‖
        ≤ 1 * ‖toEuclideanLin S (toEuclideanLin (polarMat X) z)‖ := hQhc _
      _ ≤ 1 * (nS * ‖toEuclideanLin (polarMat X) z‖) :=
          mul_le_mul_of_nonneg_left (hnS _) zero_le_one
      _ = nS * ‖z‖ := by rw [norm_apply_of_isometry _ hQh, one_mul]) j
  -- Weyl between `Q̂ᴴ S Q̂` and `(Q U)ᴴ S (Q U)`
  have hQU : (Q * polarMat (Qᴴ * polarMat X))ᴴ * (Q * polarMat (Qᴴ * polarMat X)) = 1 := by
    rw [conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Qᴴ, hQ, Matrix.one_mul, hU1]
  have hQUh : ∀ w, ‖toEuclideanLin (Q * polarMat (Qᴴ * polarMat X))ᴴ w‖ ≤ 1 * ‖w‖ :=
    norm_conjTranspose_le _ zero_le_one (fun z => by
      rw [norm_apply_of_isometry _ hQU, one_mul])
  have hDh := norm_conjTranspose_le _ hss hU
  have e : (polarMat X)ᴴ * S * polarMat X -
      (Q * polarMat (Qᴴ * polarMat X))ᴴ * S * (Q * polarMat (Qᴴ * polarMat X)) =
      (polarMat X - Q * polarMat (Qᴴ * polarMat X))ᴴ * S * polarMat X +
        (Q * polarMat (Qᴴ * polarMat X))ᴴ * S * (polarMat X - Q * polarMat (Qᴴ * polarMat X)) := by
    rw [conjTranspose_sub]
    simp only [Matrix.sub_mul, Matrix.mul_sub]
    abel
  have h2 := weyl_sorted_opNorm hK hK' (2 * (s + s ^ 2) * nS) (fun z => by
    rw [e, map_add, LinearMap.add_apply]
    simp only [toEuclideanLin_mul_apply]
    have a1 := hDh (toEuclideanLin S (toEuclideanLin (polarMat X) z))
    have a2 := hnS (toEuclideanLin (polarMat X) z)
    rw [norm_apply_of_isometry _ hQh] at a2
    have a3 := hQUh (toEuclideanLin S
      (toEuclideanLin (polarMat X - Q * polarMat (Qᴴ * polarMat X)) z))
    have a4 := hnS (toEuclideanLin (polarMat X - Q * polarMat (Qᴴ * polarMat X)) z)
    have a5 := hU z
    refine (norm_add_le _ _).trans ?_
    calc _ ≤ (s + s ^ 2) * (nS * ‖z‖) + 1 * (nS * ((s + s ^ 2) * ‖z‖)) :=
          add_le_add (a1.trans (mul_le_mul_of_nonneg_left a2 hss))
            (a3.trans (mul_le_mul_of_nonneg_left
              (a4.trans (mul_le_mul_of_nonneg_left a5 hnS0)) zero_le_one))
      _ = 2 * (s + s ^ 2) * nS * ‖z‖ := by ring) j
  -- `(Q U)ᴴ S (Q U) = Uᴴ (Qᴴ S Q) U`
  have h3 := eigAsc_conj hC hK' _ hU1 hU2 (by
    rw [conjTranspose_mul]
    simp only [Matrix.mul_assoc]) j
  rw [h3] at h2
  calc |eigAsc hN j - eigAsc hC j| ≤
        |eigAsc hN j - eigAsc hK j| + |eigAsc hK j - eigAsc hC j| := abs_sub_le _ _ _
    _ ≤ η * nS + 2 * (s + s ^ 2) * nS :=
        add_le_add (h1.trans (mul_le_mul_of_nonneg_left h1' hη0)) h2
    _ = (2 * s + 2 * s ^ 2 + η) * nS := by ring

/-- `‖V u‖ ≤ √(1 + ι) ‖u‖` when `‖Vᴴ V - I‖ ≤ ι`. -/
theorem norm_apply_le_sqrt {d : Type*} [Fintype d] [DecidableEq d] {V : Matrix n d 𝕜} {ι : ℝ}
    (hι0 : 0 ≤ ι) (hι : ∀ z, ‖toEuclideanLin (Vᴴ * V) z - z‖ ≤ ι * ‖z‖)
    (u : EuclideanSpace 𝕜 d) : ‖toEuclideanLin V u‖ ≤ √(1 + ι) * ‖u‖ := by
  have e2 : ‖toEuclideanLin V u‖ ^ 2 ≤ (1 + ι) * ‖u‖ ^ 2 := by
    have h1 := Enclosure.inner_toEuclideanLin_gram V u u
    rw [← inner_self_eq_norm_sq (𝕜 := 𝕜), h1]
    have h2 : RCLike.re ⟪toEuclideanLin (Vᴴ * V) u, u⟫_𝕜 =
        RCLike.re ⟪u, u⟫_𝕜 + RCLike.re ⟪toEuclideanLin (Vᴴ * V) u - u, u⟫_𝕜 := by
      rw [inner_sub_left, map_sub]
      ring
    rw [h2, inner_self_eq_norm_sq (𝕜 := 𝕜)]
    have h3 : RCLike.re ⟪toEuclideanLin (Vᴴ * V) u - u, u⟫_𝕜 ≤ ι * ‖u‖ * ‖u‖ :=
      (RCLike.re_le_norm _).trans ((norm_inner_le_norm _ _).trans
        (mul_le_mul_of_nonneg_right (hι u) (norm_nonneg _)))
    nlinarith
  calc ‖toEuclideanLin V u‖ = √(‖toEuclideanLin V u‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ √((1 + ι) * ‖u‖ ^ 2) := Real.sqrt_le_sqrt e2
    _ = √(1 + ι) * ‖u‖ := by rw [Real.sqrt_mul (by linarith), Real.sqrt_sq (norm_nonneg _)]

/-- `|tr M| ≤ card g · ‖M‖`. -/
theorem norm_trace_le_card (M : Matrix g g 𝕜) {nM : ℝ}
    (hM : ∀ v, ‖toEuclideanLin M v‖ ≤ nM * ‖v‖) : ‖trace M‖ ≤ Fintype.card g * nM := by
  rw [trace_eq_sum_inner_matrix M (EuclideanSpace.basisFun g 𝕜)]
  calc ‖∑ i, ⟪EuclideanSpace.basisFun g 𝕜 i,
          toEuclideanLin M (EuclideanSpace.basisFun g 𝕜 i)⟫_𝕜‖
      ≤ ∑ i, ‖⟪EuclideanSpace.basisFun g 𝕜 i,
          toEuclideanLin M (EuclideanSpace.basisFun g 𝕜 i)⟫_𝕜‖ := norm_sum_le _ _
    _ ≤ ∑ _i : g, nM := Finset.sum_le_sum fun i _ => by
        have h1 : ‖EuclideanSpace.basisFun g 𝕜 i‖ = 1 :=
          (EuclideanSpace.basisFun g 𝕜).orthonormal.1 i
        refine (norm_inner_le_norm _ _).trans ?_
        rw [h1, one_mul]
        exact (hM _).trans (by rw [h1, mul_one])
    _ = Fintype.card g * nM := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- **E-07, Proposition 3** (C-13): for `P_i = Q_i Q_iᴴ` and `‖X_i X_iᴴ - P_i‖ ≤ e_i`,
`|tr(P₁ P₂) - tr(X₁ X₁ᴴ X₂ X₂ᴴ)| ≤ k₂ e₁ + g (1 + η₁) e₂` (`k₂` the rank of `P₂`), and
`Re tr(X₁ X₁ᴴ X₂ X₂ᴴ) = ‖X₁ᴴ X₂‖_F²`. -/
theorem c13_prop3 {k₁ k₂ : Type*} [Fintype k₁] [Fintype k₂] [DecidableEq k₂]
    (Q1 : Matrix n k₁ 𝕜) (Q2 : Matrix n k₂ 𝕜) (hQ2 : Q2ᴴ * Q2 = 1) (X1 X2 : Matrix n g 𝕜)
    {η1 e1 e2 : ℝ} (hη10 : 0 ≤ η1)
    (hη1 : ∀ z, ‖toEuclideanLin (X1ᴴ * X1) z - z‖ ≤ η1 * ‖z‖) (he2' : 0 ≤ e2)
    (he1 : ∀ v, ‖toEuclideanLin (X1 * X1ᴴ - Q1 * Q1ᴴ) v‖ ≤ e1 * ‖v‖)
    (he2 : ∀ v, ‖toEuclideanLin (X2 * X2ᴴ - Q2 * Q2ᴴ) v‖ ≤ e2 * ‖v‖) :
    ‖trace (Q1 * Q1ᴴ * (Q2 * Q2ᴴ)) - trace (X1 * X1ᴴ * (X2 * X2ᴴ))‖ ≤
      Fintype.card k₂ * e1 + Fintype.card g * ((1 + η1) * e2) ∧
    RCLike.re (trace (X1 * X1ᴴ * (X2 * X2ᴴ))) = frob (X1ᴴ * X2) := by
  have hsplit : trace (Q1 * Q1ᴴ * (Q2 * Q2ᴴ)) - trace (X1 * X1ᴴ * (X2 * X2ᴴ)) =
      trace ((Q1 * Q1ᴴ - X1 * X1ᴴ) * (Q2 * Q2ᴴ)) +
        trace (X1 * X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ)) := by
    rw [Matrix.sub_mul, Matrix.mul_sub, trace_sub, trace_sub]
    ring
  have t1 : trace ((Q1 * Q1ᴴ - X1 * X1ᴴ) * (Q2 * Q2ᴴ)) =
      trace (Q2ᴴ * (Q1 * Q1ᴴ - X1 * X1ᴴ) * Q2) := by
    calc trace ((Q1 * Q1ᴴ - X1 * X1ᴴ) * (Q2 * Q2ᴴ)) =
          trace ((Q1 * Q1ᴴ - X1 * X1ᴴ) * Q2 * Q2ᴴ) := by rw [Matrix.mul_assoc]
      _ = trace (Q2ᴴ * ((Q1 * Q1ᴴ - X1 * X1ᴴ) * Q2)) := Matrix.trace_mul_comm _ _
      _ = trace (Q2ᴴ * (Q1 * Q1ᴴ - X1 * X1ᴴ) * Q2) := by rw [Matrix.mul_assoc]
  have t2 : trace (X1 * X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ)) =
      trace (X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ) * X1) := by
    calc trace (X1 * X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ)) =
          trace (X1 * (X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ))) := by rw [Matrix.mul_assoc]
      _ = trace (X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ) * X1) := Matrix.trace_mul_comm _ _
  have hE1 : ∀ v, ‖toEuclideanLin (Q1 * Q1ᴴ - X1 * X1ᴴ) v‖ ≤ e1 * ‖v‖ := fun v => by
    have hneg : Q1 * Q1ᴴ - X1 * X1ᴴ = -(X1 * X1ᴴ - Q1 * Q1ᴴ) := (neg_sub _ _).symm
    rw [hneg, map_neg, LinearMap.neg_apply, norm_neg]
    exact he1 v
  have hE2 : ∀ v, ‖toEuclideanLin (Q2 * Q2ᴴ - X2 * X2ᴴ) v‖ ≤ e2 * ‖v‖ := fun v => by
    have hneg : Q2 * Q2ᴴ - X2 * X2ᴴ = -(X2 * X2ᴴ - Q2 * Q2ᴴ) := (neg_sub _ _).symm
    rw [hneg, map_neg, LinearMap.neg_apply, norm_neg]
    exact he2 v
  have hQ2h : ∀ w, ‖toEuclideanLin Q2ᴴ w‖ ≤ 1 * ‖w‖ :=
    norm_conjTranspose_le Q2 zero_le_one (fun z => by
      rw [norm_apply_of_isometry Q2 hQ2, one_mul])
  have b1 : ∀ z, ‖toEuclideanLin (Q2ᴴ * (Q1 * Q1ᴴ - X1 * X1ᴴ) * Q2) z‖ ≤ e1 * ‖z‖ := fun z => by
    simp only [toEuclideanLin_mul_apply]
    calc _ ≤ 1 * ‖toEuclideanLin (Q1 * Q1ᴴ - X1 * X1ᴴ) (toEuclideanLin Q2 z)‖ := hQ2h _
      _ ≤ 1 * (e1 * ‖toEuclideanLin Q2 z‖) := mul_le_mul_of_nonneg_left (hE1 _) zero_le_one
      _ = e1 * ‖z‖ := by rw [norm_apply_of_isometry Q2 hQ2, one_mul]
  have hX1 := norm_apply_le_sqrt hη10 hη1
  have hX1h := norm_conjTranspose_le X1 (Real.sqrt_nonneg _) hX1
  have b2 : ∀ z, ‖toEuclideanLin (X1ᴴ * (Q2 * Q2ᴴ - X2 * X2ᴴ) * X1) z‖ ≤
      ((1 + η1) * e2) * ‖z‖ := fun z => by
    simp only [toEuclideanLin_mul_apply]
    calc _ ≤ √(1 + η1) * ‖toEuclideanLin (Q2 * Q2ᴴ - X2 * X2ᴴ) (toEuclideanLin X1 z)‖ :=
          hX1h _
      _ ≤ √(1 + η1) * (e2 * (√(1 + η1) * ‖z‖)) :=
          mul_le_mul_of_nonneg_left ((hE2 _).trans (mul_le_mul_of_nonneg_left (hX1 z) he2'))
            (Real.sqrt_nonneg _)
      _ = (√(1 + η1) * √(1 + η1)) * e2 * ‖z‖ := by ring
      _ = ((1 + η1) * e2) * ‖z‖ := by rw [Real.mul_self_sqrt (by linarith)]
  refine ⟨?_, ?_⟩
  · rw [hsplit, t1, t2]
    exact (norm_add_le _ _).trans (add_le_add (norm_trace_le_card _ b1) (norm_trace_le_card _ b2))
  · rw [frob_eq_re_trace, conjTranspose_mul, conjTranspose_conjTranspose]
    congr 1
    calc trace (X1 * X1ᴴ * (X2 * X2ᴴ)) = trace (X1 * X1ᴴ * X2 * X2ᴴ) := by
          simp only [Matrix.mul_assoc]
      _ = trace (X2ᴴ * (X1 * X1ᴴ * X2)) := Matrix.trace_mul_comm _ _
      _ = trace (X2ᴴ * X1 * (X1ᴴ * X2)) := by simp only [Matrix.mul_assoc]

end CertifyMathlib.Counting
