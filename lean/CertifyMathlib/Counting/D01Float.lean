import CertifyMathlib.Enclosure.FloatAccounting
import CertifyMathlib.Counting.Ostrowski

/-!
# D-01 in floating point: the entrywise bound (a), the row-sum bound (b) and the count

Source: `docs/proofs/D-01_sparse-residual.md` in the code repository (Steps 1 to 3 and 5) and
`certify_ed/ldl_residual.py`. The arithmetic is the standard model of
`CertifyMathlib.Enclosure.FloatModel` (a rounding `R` with `eta = 0`, the error model of the
document); every sum of the sparse products is evaluated along an arbitrary summation tree (any
order and association, as the CSR kernels of SciPy may use). `T i j` sums the products of entry
`(i, j)`: its leaves are distinct indices `k`, and every `k` it omits has `L_ik = 0` or `L_jk = 0`;
`t_ij = (T i j).adds + 1` is the number of products summed.

Real data (`prodFl`, `residFl`): `F_c,ij` is the sum along `T i j` of `fl(L_ik fl(D_k L_jk))` and
`E_c,ij = fl(F_c,ij - S_P,ij)`.

* `d01_a`: **(a)**, `|E_ij| ≤ |E_c,ij| / (1 - u) + γ_{w+1} G_ij` for every `w ≥ t_ij` (the
  document's `g_{t_ij+1} G_ij` with `w = t_ij`), `G = |L| |D| |L|ᵀ`.
* `d01_b`: **(b)**, the absolute row sums of `E` are at most
  `Σ_j (|E_c,ij| / (1 - u) + γ_{c_i+1} G_ij)` when `t_ij ≤ c_i`.

Complex data (`prodFlC`, `residFlC`): `fl(D_k conj L_jk)` rounded componentwise (`crnd`), the
complex products formed from four real products and two real sums (`cmul`), the sums and the
subtraction rounded componentwise.

* `d01_a_c`: **(a)**, `|E_ij| ≤ |E_c,ij| / (1 - u) + γ_{w+3} G_ij` for `w ≥ t_ij`; `γ_{t+3}` is below
  the document's `g_{t+1} = 2 γ_{t+5}`.
* `d01_b_c`: **(b)** for complex data with `γ_{c_i+3}`.

The count (Step 5): `lemma4_perm_rowSum`, for `L D Lᴴ = P (A - σ I + diag Δ) Pᴴ + E` with `P`
unitary (the permutation), `|Δ_i| ≤ s` (the rounding of the shifted diagonal) and `E` Hermitian with
absolute row sums at most `R`: `#{λ < σ - (R + s)} ≤ ν ≤ #{λ ≤ σ + (R + s)}`. `d01_count` and
`d01_count_c` join it to (b).

Part (c), real data (section `PartC`; `sparse_ldl_residual_bound`, lines 108 to 161 of
`certify_ed/ldl_residual.py`): `ecLow` takes `|E_c,ij|` from the lower triangle and `|E_c,ji|` above
the diagonal (the mirror of Lemma R1); `rowsAbs` sums row `i` of these terms and `nb + 1` zeros (the
initial value and the partial sums of the `nb` row blocks) along an arbitrary tree `Trow i`;
`colsumFl` and `G1Fl` are `fl(|L|ᵀ 1)` and `fl(|L| fl(|D| colsum))` along arbitrary trees;
`residInfFl`, `roundInfFl`, `shiftFl` and `betaFl` are lines 157, 159, 160 and 161, with
`gamma(k)` as `gamHat` of `FloatAccounting` and every product by a constant rounded.

* `lambda_r1`, `lambda_r2`: the scalar inequalities (Λ-R1) and (Λ-R2) in the form the proof uses,
  for `m = n + nb + 1 ≤ 2²⁷` and `c_i ≤ n ≤ 2²⁶`.
* `d01_c`: **(c)**, `Σ_j |E_ij| + 2u |S_i'i'| ≤ β` for all `i` and `i'`, for `u ≤ 2⁻⁵³`,
  `t_ij ≤ c_i`, `t_ij ≤ c_j` and `c_i ≤ n`.
* `d01_count_fl`: the sandwich `#{λ(A) < σ - β} ≤ ν ≤ #{λ(A) ≤ σ + β}` with this `β`, for the stored
  `S = A - σ I + diag Δ` with `|Δ_i| ≤ 2u |S_ii|`.
-/

namespace CertifyMathlib.Counting

open CertifyMathlib.Enclosure.FP Matrix

section Count

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

/-- `‖Pᴴ x‖ = ‖x‖` for a unitary `P`. -/
theorem nsq_conjTranspose_mulVec (P : Matrix n n 𝕜) (hP2 : P * Pᴴ = 1) (x : n → 𝕜) :
    nsq (Pᴴ *ᵥ x) = nsq x := by
  rw [nsq_mulVec, conjTranspose_conjTranspose, hP2, qf_one]

/-- `(L diag(d) Lᴴ)_ij = Σ_k L_ik d_k conj(L_jk)`. -/
theorem ldl_apply (L : Matrix n n 𝕜) (d : n → ℝ) (i j : n) :
    (L * (dg d : Matrix n n 𝕜) * Lᴴ) i j = ∑ k, L i k * ((d k : 𝕜) * star (L j k)) := by
  rw [Matrix.mul_apply]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [dg, Matrix.mul_diagonal, conjTranspose_apply, mul_assoc]

/-- **D-01, Step 5** with the permutation: Lemma 4 with `β = R + s`. -/
theorem lemma4_perm_rowSum {A : Matrix n n 𝕜} (hA : A.IsHermitian) (σ R s : ℝ)
    (P : Matrix n n 𝕜) (hP1 : Pᴴ * P = 1) (hP2 : P * Pᴴ = 1) (L : Matrix n n 𝕜) (hL : IsUnit L)
    (d : n → ℝ) (Δ : n → ℝ) (hΔ : ∀ i, |Δ i| ≤ s) (E : Matrix n n 𝕜) (hEh : E.IsHermitian)
    (hR : ∀ i, ∑ j, ‖E i j‖ ≤ R)
    (hfac : L * (dg d : Matrix n n 𝕜) * Lᴴ = P * (A - (σ : 𝕜) • (1 : Matrix n n 𝕜) + (dg Δ : Matrix n n 𝕜)) * Pᴴ + E) :
    cntLt hA (σ - (R + s)) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + (R + s)) := by
  refine lemma4_perm hA σ (R + s) P hP1 hP2 L hL d (P * (dg Δ : Matrix n n 𝕜) * Pᴴ + E) ?_ ?_
  · rw [hfac, Matrix.mul_add, Matrix.add_mul, add_assoc]
  · intro x
    rw [qf_add, qf_congr, qf_dg]
    have hE := abs_qf_le_rowSum E hEh R hR x
    have hD : |∑ i, Δ i * ‖(Pᴴ *ᵥ x) i‖ ^ 2| ≤ s * nsq x := by
      rw [← nsq_conjTranspose_mulVec P hP2 x]
      unfold nsq
      rw [Finset.mul_sum]
      refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
      rw [abs_mul, abs_of_nonneg (sq_nonneg ‖(Pᴴ *ᵥ x) i‖)]
      exact mul_le_mul_of_nonneg_right (hΔ i) (sq_nonneg _)
    calc |∑ i, Δ i * ‖(Pᴴ *ᵥ x) i‖ ^ 2 + qf E x| ≤
          |∑ i, Δ i * ‖(Pᴴ *ᵥ x) i‖ ^ 2| + |qf E x| := abs_add_le _ _
      _ ≤ (R + s) * nsq x := by linarith

end Count

/-- `|z| ≤ |fl(z)| / (1 - u)` (the subtraction of Step 3). -/
theorem abs_le_rnd_div (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u < 1) (z : ℝ) :
    |z| ≤ |R.rnd z| / (1 - R.u) := by
  rw [le_div_iff₀ (by linarith)]
  have h1 := R.abs_rnd_sub_le h0 z
  have h2 := abs_sub_abs_le_abs_sub z (R.rnd z)
  rw [abs_sub_comm] at h2
  nlinarith

theorem norm_le_crnd_div (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u < 1) (z : ℂ) :
    ‖z‖ ≤ ‖crnd R z‖ / (1 - R.u) := by
  rw [le_div_iff₀ (by linarith)]
  have h1 := crnd_err R h0 z
  have h2 := norm_sub_norm_le z (crnd R z)
  rw [norm_sub_rev] at h2
  nlinarith

theorem two_u_le_gam_two {u : ℝ} (hu : 0 ≤ u) (hu1 : 2 * u < 1) : 2 * u + u ^ 2 ≤ gam u 2 := by
  unfold gam
  rw [le_div_iff₀ (by push_cast; linarith)]
  push_cast
  nlinarith [mul_nonneg hu (sq_nonneg u)]

section Real

variable {n : Type*} [Fintype n] [DecidableEq n] (R : Rounding ℝ)

/-- `F_c,ij`: the products `fl(L_ik fl(D_k L_jk))` summed along `T i j`. -/
noncomputable def prodFl (T : n → n → STree n) (L : Matrix n n ℝ) (d : n → ℝ) (i j : n) : ℝ :=
  (T i j).comp R.rnd (fun k => R.rnd (L i k * R.rnd (d k * L j k)))

/-- `E_c,ij = fl(F_c,ij - S_P,ij)`. -/
noncomputable def residFl (T : n → n → STree n) (L : Matrix n n ℝ) (d : n → ℝ)
    (Sp : Matrix n n ℝ) (i j : n) : ℝ :=
  R.rnd (prodFl R T L d i j - Sp i j)

/-- `G = |L| |D| |L|ᵀ`. -/
noncomputable def Gabs {𝕜 : Type*} [RCLike 𝕜] (L : Matrix n n 𝕜) (d : n → ℝ) (i j : n) : ℝ :=
  ∑ k, ‖L i k‖ * |d k| * ‖L j k‖

/-- The pattern of the summation trees. -/
def PatternTrees {𝕜 : Type*} [RCLike 𝕜] (T : n → n → STree n) (L : Matrix n n 𝕜) : Prop :=
  ∀ i j, (T i j).leaves.Nodup ∧ ∀ k, k ∉ (T i j).leaves → L i k = 0 ∨ L j k = 0

variable {R}

/-- **D-01 (a)**, real data. -/
theorem d01_a (h0 : R.eta = 0) {T : n → n → STree n} {L : Matrix n n ℝ} (hT : PatternTrees T L)
    (d : n → ℝ) (Sp : Matrix n n ℝ) (i j : n) {w : ℕ} (hw : (T i j).adds + 1 ≤ w)
    (hwu : ((w + 1 : ℕ) : ℝ) * R.u < 1) :
    |(L * (dg d : Matrix n n ℝ) * Lᴴ - Sp) i j| ≤ |residFl R T L d Sp i j| / (1 - R.u) +
      gam R.u (w + 1) * Gabs L d i j := by
  have hu := R.u_nonneg
  have hw1 : 1 ≤ w := le_trans (by omega) hw
  have h2u : 2 * R.u < 1 := by
    have h2 : (2 : ℝ) ≤ ((w + 1 : ℕ) : ℝ) := by exact_mod_cast (by omega : 2 ≤ w + 1)
    nlinarith
  have hleaf : ∀ k, ‖R.rnd (L i k * R.rnd (d k * L j k)) - L i k * (d k * L j k)‖ ≤
      gam R.u 2 * ‖L i k * (d k * L j k)‖ + 0 := fun k => by
    rw [Real.norm_eq_abs, Real.norm_eq_abs, add_zero]
    have e1 := R.abs_rnd_sub_le h0 (d k * L j k)
    have e2 := R.abs_rnd_sub_le h0 (L i k * R.rnd (d k * L j k))
    have e3 : |R.rnd (d k * L j k)| ≤ (1 + R.u) * |d k * L j k| := by
      have := abs_sub_abs_le_abs_sub (R.rnd (d k * L j k)) (d k * L j k)
      linarith
    have hg2 := two_u_le_gam_two hu h2u
    calc |R.rnd (L i k * R.rnd (d k * L j k)) - L i k * (d k * L j k)| ≤
          |R.rnd (L i k * R.rnd (d k * L j k)) - L i k * R.rnd (d k * L j k)| +
            |L i k * R.rnd (d k * L j k) - L i k * (d k * L j k)| := abs_sub_le _ _ _
      _ = |R.rnd (L i k * R.rnd (d k * L j k)) - L i k * R.rnd (d k * L j k)| +
            |L i k| * |R.rnd (d k * L j k) - d k * L j k| := by rw [← mul_sub, abs_mul]
      _ ≤ R.u * (|L i k| * ((1 + R.u) * |d k * L j k|)) + |L i k| * (R.u * |d k * L j k|) := by
          refine add_le_add ?_ (mul_le_mul_of_nonneg_left e1 (abs_nonneg _))
          refine e2.trans (mul_le_mul_of_nonneg_left ?_ hu)
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_left e3 (abs_nonneg _)
      _ = (2 * R.u + R.u ^ 2) * |L i k * (d k * L j k)| := by
          rw [abs_mul (L i k) (d k * L j k)]; ring
      _ ≤ gam R.u 2 * |L i k * (d k * L j k)| := mul_le_mul_of_nonneg_right hg2 (abs_nonneg _)
  have hle : 2 + (T i j).adds ≤ w + 1 := by omega
  have hk : ((2 + (T i j).adds : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hle) hu) hwu
  have h := STree.comp_error R (a := 2) (β := 0) le_rfl (fun k => L i k * (d k * L j k))
    (fun k => R.rnd (L i k * R.rnd (d k * L j k))) hleaf (fun k => R.isFloat_rnd _) (T i j) hk
  have hsupp : ∀ k, k ∉ (T i j).leaves → L i k * (d k * L j k) = 0 := fun k hk' => by
    rcases (hT i j).2 k hk' with h1 | h1 <;> simp [h1]
  rw [STree.eval_eq_finset_sum _ _ (hT i j).1 hsupp,
    STree.absEval_eq_finset_sum _ _ (hT i j).1 hsupp] at h
  simp only [mul_zero, add_zero] at h
  have hG : ∑ k, ‖L i k * (d k * L j k)‖ = Gabs L d i j := by
    unfold Gabs
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [norm_mul, norm_mul, Real.norm_eq_abs (d k), mul_assoc]
  have hentry : (L * (dg d : Matrix n n ℝ) * Lᴴ) i j = ∑ k, L i k * (d k * L j k) := by
    rw [ldl_apply]
    simp
  have hF : |prodFl R T L d i j - (L * (dg d : Matrix n n ℝ) * Lᴴ) i j| ≤ gam R.u (w + 1) * Gabs L d i j := by
    rw [hentry, ← Real.norm_eq_abs, ← hG]
    refine h.trans (mul_le_mul_of_nonneg_right (gam_mono hu hle hwu)
      (Finset.sum_nonneg fun k _ => norm_nonneg _))
  have hS := abs_le_rnd_div R h0 (by linarith) (prodFl R T L d i j - Sp i j)
  rw [Matrix.sub_apply]
  calc |(L * (dg d : Matrix n n ℝ) * Lᴴ) i j - Sp i j| ≤
        |(L * (dg d : Matrix n n ℝ) * Lᴴ) i j - prodFl R T L d i j| + |prodFl R T L d i j - Sp i j| :=
        abs_sub_le _ _ _
    _ ≤ gam R.u (w + 1) * Gabs L d i j + |residFl R T L d Sp i j| / (1 - R.u) := by
        rw [abs_sub_comm]
        exact add_le_add hF hS
    _ = _ := by ring

/-- **D-01 (b)**, real data: the absolute row sums of `E`. -/
theorem d01_b (h0 : R.eta = 0) {T : n → n → STree n} {L : Matrix n n ℝ} (hT : PatternTrees T L)
    (d : n → ℝ) (Sp : Matrix n n ℝ) (c : n → ℕ) (hc : ∀ i j, (T i j).adds + 1 ≤ c i)
    (hcu : ∀ i, ((c i + 1 : ℕ) : ℝ) * R.u < 1) (i : n) :
    ∑ j, ‖(L * (dg d : Matrix n n ℝ) * Lᴴ - Sp) i j‖ ≤
      ∑ j, (|residFl R T L d Sp i j| / (1 - R.u) + gam R.u (c i + 1) * Gabs L d i j) :=
  Finset.sum_le_sum fun j _ => by
    rw [Real.norm_eq_abs]
    exact d01_a h0 hT d Sp i j (hc i j) (hcu i)

end Real

section Complex

variable {n : Type*} [Fintype n] [DecidableEq n] (R : Rounding ℝ)

/-- `F_c,ij` for complex data. -/
noncomputable def prodFlC (T : n → n → STree n) (L : Matrix n n ℂ) (d : n → ℝ) (i j : n) : ℂ :=
  (T i j).comp (crnd R) (fun k => cmul R (L i k) (crnd R ((d k : ℂ) * star (L j k))))

/-- `E_c,ij` for complex data. -/
noncomputable def residFlC (T : n → n → STree n) (L : Matrix n n ℂ) (d : n → ℝ)
    (Sp : Matrix n n ℂ) (i j : n) : ℂ :=
  crnd R (prodFlC R T L d i j - Sp i j)

variable {R}

/-- **D-01 (a)**, complex data. -/
theorem d01_a_c (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) {T : n → n → STree n}
    {L : Matrix n n ℂ} (hT : PatternTrees T L) (d : n → ℝ) (Sp : Matrix n n ℂ) (i j : n) {w : ℕ}
    (hw : (T i j).adds + 1 ≤ w) (hwu : ((w + 3 : ℕ) : ℝ) * R.u < 1) :
    ‖(L * (dg d : Matrix n n ℂ) * Lᴴ - Sp) i j‖ ≤ ‖residFlC R T L d Sp i j‖ / (1 - R.u) +
      gam R.u (w + 3) * Gabs L d i j := by
  have hu := R.u_nonneg
  have h4 : ((3 + 1 : ℕ) : ℝ) * R.u < 1 := by
    have : ((3 + 1 : ℕ) : ℝ) ≤ ((w + 3 : ℕ) : ℝ) := by exact_mod_cast (by omega : 3 + 1 ≤ w + 3)
    nlinarith
  have hg := gam_succ hu h4
  have hg3 : 0 ≤ gam R.u 3 := gam_nonneg hu (by
    have : ((3 : ℕ) : ℝ) ≤ ((3 + 1 : ℕ) : ℝ) := by exact_mod_cast (by omega : 3 ≤ 3 + 1)
    nlinarith)
  have hleaf : ∀ k, ‖cmul R (L i k) (crnd R ((d k : ℂ) * star (L j k))) -
      L i k * ((d k : ℂ) * star (L j k))‖ ≤ gam R.u (3 + 1) * ‖L i k * ((d k : ℂ) * star (L j k))‖ +
        0 := fun k => by
    rw [add_zero]
    set b : ℂ := (d k : ℂ) * star (L j k) with hb
    have e1 := crnd_err R h0 b
    have e2 := cmul_err R h0 hu1 (L i k) (crnd R b)
    have e3 : ‖crnd R b‖ ≤ (1 + R.u) * ‖b‖ := by
      have := norm_sub_norm_le (crnd R b) b
      linarith
    calc ‖cmul R (L i k) (crnd R b) - L i k * b‖ ≤
          ‖cmul R (L i k) (crnd R b) - L i k * crnd R b‖ + ‖L i k * crnd R b - L i k * b‖ :=
          norm_sub_le_norm_sub_add_norm_sub _ _ _
      _ = ‖cmul R (L i k) (crnd R b) - L i k * crnd R b‖ + ‖L i k‖ * ‖crnd R b - b‖ := by
          rw [← mul_sub, norm_mul]
      _ ≤ gam R.u 3 * (‖L i k‖ * ((1 + R.u) * ‖b‖)) + ‖L i k‖ * (R.u * ‖b‖) := by
          refine add_le_add ?_ (mul_le_mul_of_nonneg_left e1 (norm_nonneg _))
          refine e2.trans (mul_le_mul_of_nonneg_left ?_ hg3)
          rw [norm_mul]
          exact mul_le_mul_of_nonneg_left e3 (norm_nonneg _)
      _ = (gam R.u 3 + R.u * (1 + gam R.u 3)) * ‖L i k * b‖ := by
          rw [norm_mul (L i k) b]; ring
      _ ≤ gam R.u (3 + 1) * ‖L i k * b‖ := mul_le_mul_of_nonneg_right hg (norm_nonneg _)
  have hle : (3 + 1) + (T i j).adds ≤ w + 3 := by omega
  have hk : (((3 + 1) + (T i j).adds : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hle) hu) hwu
  have h := STree.comp_error (cRounding R h0) (a := 3 + 1) (β := 0) le_rfl
    (fun k => L i k * ((d k : ℂ) * star (L j k)))
    (fun k => cmul R (L i k) (crnd R ((d k : ℂ) * star (L j k)))) hleaf (fun _ => trivial)
    (T i j) hk
  have hsupp : ∀ k, k ∉ (T i j).leaves → L i k * ((d k : ℂ) * star (L j k)) = 0 := fun k hk' => by
    rcases (hT i j).2 k hk' with h1 | h1 <;> simp [h1]
  rw [STree.eval_eq_finset_sum _ _ (hT i j).1 hsupp,
    STree.absEval_eq_finset_sum _ _ (hT i j).1 hsupp] at h
  simp only [mul_zero, add_zero] at h
  have hG : ∑ k, ‖L i k * ((d k : ℂ) * star (L j k))‖ = Gabs L d i j := by
    unfold Gabs
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [norm_mul, norm_mul, norm_star, Complex.norm_real, Real.norm_eq_abs, mul_assoc]
  have hF : ‖prodFlC R T L d i j - (L * (dg d : Matrix n n ℂ) * Lᴴ) i j‖ ≤ gam R.u (w + 3) * Gabs L d i j := by
    rw [ldl_apply, ← hG]
    exact h.trans (mul_le_mul_of_nonneg_right (gam_mono hu hle hwu)
      (Finset.sum_nonneg fun k _ => norm_nonneg _))
  have hS := norm_le_crnd_div R h0 (by nlinarith) (prodFlC R T L d i j - Sp i j)
  rw [Matrix.sub_apply]
  calc ‖(L * (dg d : Matrix n n ℂ) * Lᴴ) i j - Sp i j‖ ≤
        ‖(L * (dg d : Matrix n n ℂ) * Lᴴ) i j - prodFlC R T L d i j‖ + ‖prodFlC R T L d i j - Sp i j‖ :=
        norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ gam R.u (w + 3) * Gabs L d i j + ‖residFlC R T L d Sp i j‖ / (1 - R.u) := by
        rw [norm_sub_rev]
        exact add_le_add hF hS
    _ = _ := by ring

/-- **D-01 (b)**, complex data. -/
theorem d01_b_c (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) {T : n → n → STree n}
    {L : Matrix n n ℂ} (hT : PatternTrees T L) (d : n → ℝ) (Sp : Matrix n n ℂ) (c : n → ℕ)
    (hc : ∀ i j, (T i j).adds + 1 ≤ c i) (hcu : ∀ i, ((c i + 3 : ℕ) : ℝ) * R.u < 1) (i : n) :
    ∑ j, ‖(L * (dg d : Matrix n n ℂ) * Lᴴ - Sp) i j‖ ≤
      ∑ j, (‖residFlC R T L d Sp i j‖ / (1 - R.u) + gam R.u (c i + 3) * Gabs L d i j) :=
  Finset.sum_le_sum fun j _ => d01_a_c h0 hu1 hT d Sp i j (hc i j) (hcu i)

end Complex

section Count2

variable {n : Type*} [Fintype n] [DecidableEq n] {R : Rounding ℝ}

/-- **D-01**, real data, with Step 5: if `S_P = P (A - σ I + diag Δ) Pᴴ` (the stored shifted
matrix, permuted exactly), `|Δ_i| ≤ s`, `L` is invertible and `β ≥ R + s` with `R` at least every
row bound of (b), then `#{λ < σ - β} ≤ ν ≤ #{λ ≤ σ + β}`. -/
theorem d01_count (h0 : R.eta = 0) {A : Matrix n n ℝ} (hA : A.IsHermitian) (σ s Rb : ℝ)
    (P : Matrix n n ℝ) (hP1 : Pᴴ * P = 1) (hP2 : P * Pᴴ = 1) {T : n → n → STree n}
    {L : Matrix n n ℝ} (hL : IsUnit L) (hT : PatternTrees T L) (d : n → ℝ) (Δ : n → ℝ)
    (hΔ : ∀ i, |Δ i| ≤ s) (c : n → ℕ) (hc : ∀ i j, (T i j).adds + 1 ≤ c i)
    (hcu : ∀ i, ((c i + 1 : ℕ) : ℝ) * R.u < 1)
    (hRb : ∀ i, ∑ j, (|residFl R T L d (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) i j| /
      (1 - R.u) + gam R.u (c i + 1) * Gabs L d i j) ≤ Rb) :
    cntLt hA (σ - (Rb + s)) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + (Rb + s)) := by
  have hσ : (σ • (1 : Matrix n n ℝ)).IsHermitian := by
    show (σ • (1 : Matrix n n ℝ))ᴴ = σ • 1
    rw [conjTranspose_smul, conjTranspose_one, star_trivial]
  have hdg : ∀ e : n → ℝ, (dg e : Matrix n n ℝ).IsHermitian := fun e => by
    unfold dg
    exact isHermitian_diagonal _
  have hM : (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)).IsHermitian := (hA.sub hσ).add (hdg Δ)
  have hEh : (L * (dg d : Matrix n n ℝ) * Lᴴ - P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ).IsHermitian :=
    (isHermitian_mul_mul_conjTranspose L (hdg d)).sub (isHermitian_mul_mul_conjTranspose P hM)
  refine lemma4_perm_rowSum hA σ Rb s P hP1 hP2 L hL d Δ hΔ
    (L * (dg d : Matrix n n ℝ) * Lᴴ - P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) hEh
    (fun i => (d01_b h0 hT d _ c hc hcu i).trans (hRb i)) ?_
  simp only [RCLike.ofReal_real_eq_id, id_eq]
  abel

/-- **D-01**, complex data, with Step 5 (as `d01_count`). -/
theorem d01_count_c (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) {A : Matrix n n ℂ}
    (hA : A.IsHermitian) (σ s Rb : ℝ) (P : Matrix n n ℂ) (hP1 : Pᴴ * P = 1) (hP2 : P * Pᴴ = 1)
    {T : n → n → STree n} {L : Matrix n n ℂ} (hL : IsUnit L) (hT : PatternTrees T L) (d : n → ℝ)
    (Δ : n → ℝ) (hΔ : ∀ i, |Δ i| ≤ s) (c : n → ℕ) (hc : ∀ i j, (T i j).adds + 1 ≤ c i)
    (hcu : ∀ i, ((c i + 3 : ℕ) : ℝ) * R.u < 1)
    (hRb : ∀ i, ∑ j, (‖residFlC R T L d (P * (A - (σ : ℂ) • (1 : Matrix n n ℂ) + (dg Δ : Matrix n n ℂ)) * Pᴴ) i j‖ /
      (1 - R.u) + gam R.u (c i + 3) * Gabs L d i j) ≤ Rb) :
    cntLt hA (σ - (Rb + s)) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + (Rb + s)) := by
  have hσ : ((σ : ℂ) • (1 : Matrix n n ℂ)).IsHermitian := by
    show ((σ : ℂ) • (1 : Matrix n n ℂ))ᴴ = (σ : ℂ) • 1
    rw [conjTranspose_smul, conjTranspose_one, Complex.star_def, Complex.conj_ofReal]
  have hdg : ∀ e : n → ℝ, (dg e : Matrix n n ℂ).IsHermitian := fun e => by
    unfold dg
    refine isHermitian_diagonal_of_self_adjoint _ ?_
    funext i
    simp
  have hM : (A - (σ : ℂ) • (1 : Matrix n n ℂ) + (dg Δ : Matrix n n ℂ)).IsHermitian := (hA.sub hσ).add (hdg Δ)
  have hEh : (L * (dg d : Matrix n n ℂ) * Lᴴ - P * (A - (σ : ℂ) • (1 : Matrix n n ℂ) + (dg Δ : Matrix n n ℂ)) * Pᴴ).IsHermitian :=
    (isHermitian_mul_mul_conjTranspose L (hdg d)).sub (isHermitian_mul_mul_conjTranspose P hM)
  refine lemma4_perm_rowSum hA σ Rb s P hP1 hP2 L hL d Δ hΔ
    (L * (dg d : Matrix n n ℂ) * Lᴴ - P * (A - (σ : ℂ) • (1 : Matrix n n ℂ) + (dg Δ : Matrix n n ℂ)) * Pᴴ) hEh
    (fun i => (d01_b_c h0 hu1 hT d _ c hc hcu i).trans (hRb i)) ?_
  simp only [RCLike.ofReal_eq_complex_ofReal]
  abel

end Count2

/-! ### Part (c): the computed `β` (real data) -/

theorem rnd_zero (R : Rounding ℝ) (h0 : R.eta = 0) : R.rnd 0 = 0 := by
  have h := R.abs_rnd_sub_le h0 0
  rw [sub_zero, abs_zero, mul_zero] at h
  exact abs_nonpos_iff.mp h

/-- A sum of nonnegative terms evaluated along any tree loses at most the factor `(1 - u)^adds`. -/
theorem comp_ge_pow (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {α : Type*} (t : α → ℝ)
    (ht : ∀ i, 0 ≤ t i) : ∀ T : STree α, (1 - R.u) ^ T.adds * T.eval t ≤ T.comp R.rnd t
  | .leaf i => by simp [STree.adds, STree.eval, STree.comp]
  | .node l r => by
    have hu := R.u_nonneg
    have h1u : 0 ≤ 1 - R.u := by linarith
    have h1u' : 1 - R.u ≤ 1 := by linarith
    have hl := comp_ge_pow R h0 hu1 t ht l
    have hr := comp_ge_pow R h0 hu1 t ht r
    have hev : ∀ T : STree α, 0 ≤ T.eval t := fun T => by
      rw [STree.eval_eq_sum]
      exact List.sum_nonneg (by simpa using fun i _ => ht i)
    have hpl : (1 - R.u) ^ (l.adds + r.adds) ≤ (1 - R.u) ^ l.adds :=
      pow_le_pow_of_le_one h1u h1u' (by omega)
    have hpr : (1 - R.u) ^ (l.adds + r.adds) ≤ (1 - R.u) ^ r.adds :=
      pow_le_pow_of_le_one h1u h1u' (by omega)
    have hsum : (1 - R.u) ^ (l.adds + r.adds) * (l.eval t + r.eval t) ≤
        l.comp R.rnd t + r.comp R.rnd t := by
      have a1 := mul_le_mul_of_nonneg_right hpl (hev l)
      have a2 := mul_le_mul_of_nonneg_right hpr (hev r)
      nlinarith
    have h := R.le_rnd_of_le h0 hu1 hsum (mul_nonneg (pow_nonneg h1u _) (add_nonneg (hev l) (hev r)))
    simp only [STree.adds, STree.eval, STree.comp]
    calc (1 - R.u) ^ (l.adds + r.adds + 1) * (l.eval t + r.eval t) =
          (1 - R.u) ^ (l.adds + r.adds) * (l.eval t + r.eval t) * (1 - R.u) := by ring
      _ ≤ _ := h

/-- The sum of a tree whose omitted leaves are zero, as a finite sum, with the loss `(1 - u)^K`
for `K ≥ adds`. -/
theorem comp_ge_sum (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {α : Type*} [Fintype α]
    [DecidableEq α] (t : α → ℝ) (ht : ∀ i, 0 ≤ t i) (T : STree α) (hnd : T.leaves.Nodup)
    (hsupp : ∀ i, i ∉ T.leaves → t i = 0) {K : ℕ} (hK : T.adds ≤ K) :
    (1 - R.u) ^ K * ∑ i, t i ≤ T.comp R.rnd t := by
  have hu := R.u_nonneg
  have h := comp_ge_pow R h0 hu1 t ht T
  rw [STree.eval_eq_finset_sum t T hnd hsupp] at h
  exact le_trans (mul_le_mul_of_nonneg_right (pow_le_pow_of_le_one (by linarith) (by linarith) hK)
    (Finset.sum_nonneg fun i _ => ht i)) h

/-- `γ_k ≥ k u`. -/
theorem mul_le_gam {u : ℝ} (hu : 0 ≤ u) {k : ℕ} (hk : (k : ℝ) * u < 1) : (k : ℝ) * u ≤ gam u k := by
  unfold gam
  rw [le_div_iff₀ (by linarith)]
  nlinarith [mul_nonneg (Nat.cast_nonneg k) hu]

/-- **Λ-R1** (real data): `(1 - u)^{K+5} (1 + γ_m (1 - u)²)(1 + 4u)(1 + 8u) ≥ 1` for `m = K + 1 ≤ 2²⁷`. -/
theorem lambda_r1 {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {K : ℕ} (hK : K + 1 ≤ 2 ^ 27) :
    1 ≤ (1 - u) ^ (K + 5) * (1 + gam u (K + 1) * (1 - u) ^ 2) * (1 + 4 * u) * (1 + 8 * u) := by
  have hu1 : u ≤ 1 / 10 ^ 15 := hu53.trans (by norm_num)
  have hKr : ((K + 1 : ℕ) : ℝ) ≤ 2 ^ 27 := by exact_mod_cast hK
  have hX0 : (0 : ℝ) ≤ ((K + 1 : ℕ) : ℝ) := Nat.cast_nonneg _
  have hx0 : 0 ≤ ((K + 1 : ℕ) : ℝ) * u := mul_nonneg hX0 hu
  have hxs : ((K + 1 : ℕ) : ℝ) * u ≤ 1 / 10 ^ 7 := by
    have := mul_le_mul hKr hu53 hu (by positivity)
    have h2 : (2 : ℝ) ^ 27 * (2 : ℝ)⁻¹ ^ 53 ≤ 1 / 10 ^ 7 := by norm_num
    linarith
  have hx2 : (((K + 1 : ℕ) : ℝ) * u) ^ 2 ≤ 2 * u := by
    have h1 : ((K + 1 : ℕ) : ℝ) ^ 2 * u ≤ (2 ^ 27) ^ 2 * (2 : ℝ)⁻¹ ^ 53 :=
      mul_le_mul (pow_le_pow_left₀ hX0 hKr 2) hu53 hu (by positivity)
    have h2 : ((2 : ℝ) ^ 27) ^ 2 * (2 : ℝ)⁻¹ ^ 53 = 2 := by norm_num
    have h3 : (((K + 1 : ℕ) : ℝ) * u) ^ 2 = ((K + 1 : ℕ) : ℝ) ^ 2 * u * u := by ring
    rw [h3]
    nlinarith
  have hg : ((K + 1 : ℕ) : ℝ) * u ≤ gam u (K + 1) := mul_le_gam hu (by linarith)
  have hpow : 1 - ((K + 5 : ℕ) : ℝ) * u ≤ (1 - u) ^ (K + 5) := one_sub_mul_le_pow hu (by linarith) _
  have e5 : ((K + 5 : ℕ) : ℝ) * u = ((K + 1 : ℕ) : ℝ) * u + 4 * u := by push_cast; ring
  rw [e5] at hpow
  have hf : 1 + ((K + 1 : ℕ) : ℝ) * u * (1 - 2 * u) ≤ 1 + gam u (K + 1) * (1 - u) ^ 2 := by
    have : ((K + 1 : ℕ) : ℝ) * u * (1 - 2 * u) ≤ gam u (K + 1) * (1 - u) ^ 2 :=
      mul_le_mul hg (by nlinarith) (by linarith) (hx0.trans hg)
    linarith
  have hA : (1 - (((K + 1 : ℕ) : ℝ) * u + 4 * u)) * (1 + ((K + 1 : ℕ) : ℝ) * u * (1 - 2 * u)) ≤
      (1 - u) ^ (K + 5) * (1 + gam u (K + 1) * (1 - u) ^ 2) :=
    mul_le_mul hpow hf (by nlinarith) (pow_nonneg (by linarith) _)
  have hB : 1 - 7 * u ≤
      (1 - (((K + 1 : ℕ) : ℝ) * u + 4 * u)) * (1 + ((K + 1 : ℕ) : ℝ) * u * (1 - 2 * u)) := by
    have t1 := mul_le_mul_of_nonneg_right hxs hu
    have t2 := mul_nonneg (mul_nonneg hx0 hx0) (mul_nonneg hu hu)
    have t3 := mul_nonneg hx0 (mul_nonneg hu hu)
    nlinarith
  have hC : 1 ≤ (1 - 7 * u) * (1 + 4 * u) * (1 + 8 * u) := by
    nlinarith [mul_nonneg hu hu, mul_nonneg (mul_nonneg hu hu) hu]
  calc (1 : ℝ) ≤ (1 - 7 * u) * (1 + 4 * u) * (1 + 8 * u) := hC
    _ ≤ (1 - u) ^ (K + 5) * (1 + gam u (K + 1) * (1 - u) ^ 2) * (1 + 4 * u) * (1 + 8 * u) := by
        have := mul_le_mul_of_nonneg_right (hB.trans hA) (by linarith : (0 : ℝ) ≤ 1 + 4 * u)
        exact mul_le_mul_of_nonneg_right this (by linarith)

/-- **Λ-R2** (real data): `(1 - u)^{c+N+9} (1 + γ_{2N+8} (1 - u)²)(1 + 8u)² ≥ 1` for `c ≤ N ≤ 2²⁶`. -/
theorem lambda_r2 {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {c N : ℕ} (hc : c ≤ N)
    (hN : N ≤ 2 ^ 26) :
    1 ≤ (1 - u) ^ (c + N + 9) * (1 + gam u (2 * N + 8) * (1 - u) ^ 2) * (1 + 8 * u) ^ 2 := by
  have hu1 : u ≤ 1 / 10 ^ 15 := hu53.trans (by norm_num)
  have hN' : N ≤ 67108864 := by have := hN; norm_num at this; exact this
  have hNr : ((2 * N + 8 : ℕ) : ℝ) ≤ 2 ^ 27 + 8 := by
    have : 2 * N + 8 ≤ 134217736 := by omega
    have h' : ((2 * N + 8 : ℕ) : ℝ) ≤ (134217736 : ℕ) := by exact_mod_cast this
    norm_num at h' ⊢
    linarith
  have hZ0 : (0 : ℝ) ≤ ((2 * N + 8 : ℕ) : ℝ) := Nat.cast_nonneg _
  have hz0 : 0 ≤ ((2 * N + 8 : ℕ) : ℝ) * u := mul_nonneg hZ0 hu
  have hzs : ((2 * N + 8 : ℕ) : ℝ) * u ≤ 1 / 10 ^ 7 := by
    have := mul_le_mul hNr hu53 hu (by positivity)
    have h2 : ((2 : ℝ) ^ 27 + 8) * (2 : ℝ)⁻¹ ^ 53 ≤ 1 / 10 ^ 7 := by norm_num
    linarith
  have hz2 : (((2 * N + 8 : ℕ) : ℝ) * u) ^ 2 ≤ 3 * u := by
    have h1 : ((2 * N + 8 : ℕ) : ℝ) ^ 2 * u ≤ (2 ^ 27 + 8) ^ 2 * (2 : ℝ)⁻¹ ^ 53 :=
      mul_le_mul (pow_le_pow_left₀ hZ0 hNr 2) hu53 hu (by positivity)
    have h2 : ((2 : ℝ) ^ 27 + 8) ^ 2 * (2 : ℝ)⁻¹ ^ 53 ≤ 3 := by norm_num
    have h3 : (((2 * N + 8 : ℕ) : ℝ) * u) ^ 2 = ((2 * N + 8 : ℕ) : ℝ) ^ 2 * u * u := by ring
    rw [h3]
    nlinarith
  have hg : ((2 * N + 8 : ℕ) : ℝ) * u ≤ gam u (2 * N + 8) := mul_le_gam hu (by linarith)
  have hpow1 : (1 - u) ^ (2 * N + 9) ≤ (1 - u) ^ (c + N + 9) :=
    pow_le_pow_of_le_one (by linarith) (by linarith) (by omega)
  have hpow : 1 - ((2 * N + 9 : ℕ) : ℝ) * u ≤ (1 - u) ^ (c + N + 9) :=
    (one_sub_mul_le_pow hu (by linarith) _).trans hpow1
  have e9 : ((2 * N + 9 : ℕ) : ℝ) * u = ((2 * N + 8 : ℕ) : ℝ) * u + u := by push_cast; ring
  rw [e9] at hpow
  have hf : 1 + ((2 * N + 8 : ℕ) : ℝ) * u * (1 - 2 * u) ≤ 1 + gam u (2 * N + 8) * (1 - u) ^ 2 := by
    have : ((2 * N + 8 : ℕ) : ℝ) * u * (1 - 2 * u) ≤ gam u (2 * N + 8) * (1 - u) ^ 2 :=
      mul_le_mul hg (by nlinarith) (by linarith) (hz0.trans hg)
    linarith
  have hA : (1 - (((2 * N + 8 : ℕ) : ℝ) * u + u)) * (1 + ((2 * N + 8 : ℕ) : ℝ) * u * (1 - 2 * u)) ≤
      (1 - u) ^ (c + N + 9) * (1 + gam u (2 * N + 8) * (1 - u) ^ 2) :=
    mul_le_mul hpow hf (by nlinarith) (pow_nonneg (by linarith) _)
  have hB : 1 - 5 * u ≤
      (1 - (((2 * N + 8 : ℕ) : ℝ) * u + u)) * (1 + ((2 * N + 8 : ℕ) : ℝ) * u * (1 - 2 * u)) := by
    have t1 := mul_le_mul_of_nonneg_right hzs hu
    have t2 := mul_nonneg (mul_nonneg hz0 hz0) (mul_nonneg hu hu)
    have t3 := mul_nonneg hz0 (mul_nonneg hu hu)
    nlinarith
  have hC : 1 ≤ (1 - 5 * u) * (1 + 8 * u) ^ 2 := by
    nlinarith [mul_nonneg hu hu, mul_nonneg (mul_nonneg hu hu) hu]
  calc (1 : ℝ) ≤ (1 - 5 * u) * (1 + 8 * u) ^ 2 := hC
    _ ≤ _ := mul_le_mul_of_nonneg_right (hB.trans hA) (sq_nonneg _)

section PartC

variable {n : Type*} [Fintype n] [DecidableEq n] [LinearOrder n] (R : Rounding ℝ)

/-- `|E_c|` assembled from the lower triangle: `|E_c,ij|` for `j ≤ i`, `|E_c,ji|` for `j > i`. -/
noncomputable def ecLow (T : n → n → STree n) (L : Matrix n n ℝ) (d : n → ℝ) (Sp : Matrix n n ℝ)
    (i j : n) : ℝ :=
  if j ≤ i then |residFl R T L d Sp i j| else |residFl R T L d Sp j i|

/-- `rows_abs_i`: the terms `ecLow i j` and `nb + 1` zeros (the initial value and the empty parts
of the blocks) summed along `Trow i`. -/
noncomputable def rowsAbs (nb : ℕ) (Trow : n → STree (n ⊕ Fin (nb + 1))) (T : n → n → STree n)
    (L : Matrix n n ℝ) (d : n → ℝ) (Sp : Matrix n n ℝ) (i : n) : ℝ :=
  (Trow i).comp R.rnd (Sum.elim (fun j => ecLow R T L d Sp i j) fun _ => 0)

/-- `colsum_k = fl(Σ_j |L_jk|)` along `Tcol k`. -/
noncomputable def colsumFl (Tcol : n → STree n) (L : Matrix n n ℝ) (k : n) : ℝ :=
  (Tcol k).comp R.rnd (fun j => |L j k|)

/-- `G1_i = fl(|L| fl(|D| colsum))_i` along `Tmv i`. -/
noncomputable def G1Fl (Tcol Tmv : n → STree n) (L : Matrix n n ℝ) (d : n → ℝ) (i : n) : ℝ :=
  (Tmv i).comp R.rnd (fun k => R.rnd (|L i k| * R.rnd (|d k| * colsumFl R Tcol L k)))

variable [Nonempty n]

/-- `resid_inf` (line 157). -/
noncomputable def residInfFl (nb : ℕ) (Trow : n → STree (n ⊕ Fin (nb + 1))) (T : n → n → STree n)
    (L : Matrix n n ℝ) (d : n → ℝ) (Sp : Matrix n n ℝ) : ℝ :=
  R.rnd (R.rnd (R.rnd (Finset.univ.sup' Finset.univ_nonempty (rowsAbs R nb Trow T L d Sp) *
    R.rnd (1 + gamHat R (Fintype.card n + nb + 1))) * (1 + 4 * R.u)) * (1 + 8 * R.u))

/-- `round_inf` (line 159). -/
noncomputable def roundInfFl (Tcol Tmv : n → STree n) (L : Matrix n n ℝ) (d : n → ℝ) (c : n → ℕ) :
    ℝ :=
  R.rnd (R.rnd (R.rnd (Finset.univ.sup' Finset.univ_nonempty
    (fun i => R.rnd (gamHat R (c i + 1) * G1Fl R Tcol Tmv L d i)) *
      R.rnd (1 + gamHat R (2 * Fintype.card n + 8))) * (1 + 8 * R.u)) * (1 + 8 * R.u))

/-- `shift_term` (line 160). -/
noncomputable def shiftFl (S : Matrix n n ℝ) : ℝ :=
  R.rnd (2 * R.u * Finset.univ.sup' Finset.univ_nonempty (fun i => |S i i|))

/-- `beta = up(resid_inf + round_inf + shift_term)` (line 161). -/
noncomputable def betaFl (nb : ℕ) (Trow : n → STree (n ⊕ Fin (nb + 1))) (Tcol Tmv : n → STree n)
    (T : n → n → STree n) (L : Matrix n n ℝ) (d : n → ℝ) (Sp S : Matrix n n ℝ) (c : n → ℕ) : ℝ :=
  R.rnd (R.rnd (R.rnd (residInfFl R nb Trow T L d Sp + roundInfFl R Tcol Tmv L d c) +
    shiftFl R S) * (1 + 8 * R.u))

variable {R}

theorem small_mul_u {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {k : ℕ}
    (hk : k ≤ 2 ^ 27 + 8) : (k : ℝ) * u ≤ 1 / 2 := by
  have hk' : (k : ℝ) ≤ 2 ^ 27 + 8 := by exact_mod_cast hk
  have := mul_le_mul hk' hu53 hu (by positivity)
  have h2 : ((2 : ℝ) ^ 27 + 8) * (2 : ℝ)⁻¹ ^ 53 ≤ 1 / 2 := by norm_num
  linarith

theorem pow3_mul_ge {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 1000) :
    1 ≤ (1 - u) ^ 3 * (1 + 8 * u) := by
  have h3 := one_sub_mul_le_pow hu (by linarith) 3
  push_cast at h3
  have h6 := mul_le_mul_of_nonneg_right h3 (by linarith : (0 : ℝ) ≤ 1 + 8 * u)
  nlinarith

/-- **D-01 (c)**, Step B: `resid_inf ≥ Σ_j |E_c,ij| / (1 - u)` (the low triangle and its mirror). -/
theorem d01_c_resid (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) (nb : ℕ)
    (hm : Fintype.card n + nb + 1 ≤ 2 ^ 27) (T : n → n → STree n) (L : Matrix n n ℝ) (d : n → ℝ)
    (Sp : Matrix n n ℝ) (Trow : n → STree (n ⊕ Fin (nb + 1)))
    (hrow : ∀ i, (Trow i).leaves.Nodup ∧
      ∀ j, Sum.inl j ∉ (Trow i).leaves → ecLow R T L d Sp i j = 0) (i : n) :
    (∑ j, ecLow R T L d Sp i j) / (1 - R.u) ≤ residInfFl R nb Trow T L d Sp := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 10 ^ 15 := hu53.trans (by norm_num)
  have hu1' : R.u ≤ 1 := by linarith
  have h1u : 0 ≤ 1 - R.u := by linarith
  have hecl0 : ∀ j, 0 ≤ ecLow R T L d Sp i j := fun j => by
    unfold ecLow
    split_ifs <;> exact abs_nonneg _
  have hS10 : 0 ≤ ∑ j, ecLow R T L d Sp i j := Finset.sum_nonneg fun j _ => hecl0 j
  have hrowge : (1 - R.u) ^ (Fintype.card n + nb) * ∑ j, ecLow R T L d Sp i j ≤
      rowsAbs R nb Trow T L d Sp i := by
    have hlen : (Trow i).adds ≤ Fintype.card n + nb := by
      have h1 := (hrow i).1.length_le_card
      rw [STree.length_leaves, Fintype.card_sum, Fintype.card_fin] at h1
      omega
    have h := comp_ge_sum R h0 hu1' (Sum.elim (fun j => ecLow R T L d Sp i j) fun _ => 0)
      (fun q => by cases q <;> simp [hecl0]) (Trow i) (hrow i).1 (fun q hq => by
        cases q with
        | inl j => exact (hrow i).2 j hq
        | inr _ => rfl) hlen
    calc (1 - R.u) ^ (Fintype.card n + nb) * ∑ j, ecLow R T L d Sp i j =
          (1 - R.u) ^ (Fintype.card n + nb) *
            ∑ q, (Sum.elim (fun j => ecLow R T L d Sp i j) fun _ => (0 : ℝ)) q := by
          rw [Fintype.sum_sum_type]
          simp
      _ ≤ _ := h
  have hm' : ((Fintype.card n + nb + 1 : ℕ) : ℝ) * R.u ≤ 1 / 2 :=
    small_mul_u hu hu53 (by have := hm; norm_num at this ⊢; omega)
  have hgm := (gamHat_bounds R h0 (by linarith) hm').1
  have hgm0 : 0 ≤ gam R.u (Fintype.card n + nb + 1) := gam_nonneg hu (by linarith)
  have hq1 : 0 ≤ 1 + gam R.u (Fintype.card n + nb + 1) * (1 - R.u) ^ 2 :=
    add_nonneg zero_le_one (mul_nonneg hgm0 (sq_nonneg _))
  have hc1 : (1 + gam R.u (Fintype.card n + nb + 1) * (1 - R.u) ^ 2) * (1 - R.u) ≤
      R.rnd (1 + gamHat R (Fintype.card n + nb + 1)) :=
    R.le_rnd_of_le h0 hu1' (by linarith) hq1
  have hM1 := Finset.le_sup' (rowsAbs R nb Trow T L d Sp) (Finset.mem_univ i)
  have hp1 : 0 ≤ (1 - R.u) ^ (Fintype.card n + nb) * ∑ j, ecLow R T L d Sp i j :=
    mul_nonneg (pow_nonneg h1u _) hS10
  have hx1 := R.le_rnd_mul h0 hu1' hp1 (mul_nonneg hq1 h1u) (hrowge.trans hM1) hc1
  have hp2 := mul_nonneg (mul_nonneg hp1 (mul_nonneg hq1 h1u)) h1u
  have hx2 := R.le_rnd_of_le h0 hu1'
    (mul_le_mul_of_nonneg_right hx1 (by linarith : (0 : ℝ) ≤ 1 + 4 * R.u))
    (mul_nonneg hp2 (by linarith))
  have hx3 := R.le_rnd_of_le h0 hu1'
    (mul_le_mul_of_nonneg_right hx2 (by linarith : (0 : ℝ) ≤ 1 + 8 * R.u))
    (mul_nonneg (mul_nonneg (mul_nonneg hp2 (by linarith)) h1u) (by linarith))
  have hr1 := lambda_r1 hu hu53 (K := Fintype.card n + nb) hm
  have hB : (∑ j, ecLow R T L d Sp i j) / (1 - R.u) ≤ residInfFl R nb Trow T L d Sp := by
    refine le_trans ?_ hx3
    rw [div_le_iff₀ (by linarith)]
    calc ∑ j, ecLow R T L d Sp i j ≤ (∑ j, ecLow R T L d Sp i j) *
          ((1 - R.u) ^ (Fintype.card n + nb + 5) *
            (1 + gam R.u (Fintype.card n + nb + 1) * (1 - R.u) ^ 2) * (1 + 4 * R.u) *
              (1 + 8 * R.u)) := le_mul_of_one_le_right hS10 hr1
      _ = _ := by ring
  exact hB

omit [LinearOrder n] in
/-- **D-01 (c)**, Step C: `round_inf ≥ γ_{c_i+1} Σ_j G_ij`. -/
theorem d01_c_round (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53)
    (hn : Fintype.card n ≤ 2 ^ 26) (L : Matrix n n ℝ) (d : n → ℝ) (c : n → ℕ)
    (hcn : ∀ i, c i ≤ Fintype.card n)
    (Tcol : n → STree n) (hcol : ∀ k, (Tcol k).leaves.Nodup ∧ ∀ j, j ∉ (Tcol k).leaves → L j k = 0)
    (Tmv : n → STree n)
    (hmv : ∀ i, (Tmv i).leaves.Nodup ∧ (Tmv i).adds + 1 ≤ c i ∧ ∀ k, k ∉ (Tmv i).leaves → L i k = 0)
    (i : n) :
    gam R.u (c i + 1) * ∑ j, Gabs L d i j ≤ roundInfFl R Tcol Tmv L d c := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 10 ^ 15 := hu53.trans (by norm_num)
  have hu1' : R.u ≤ 1 := by linarith
  have h1u : 0 ≤ 1 - R.u := by linarith
  have hn' : Fintype.card n ≤ 67108864 := by have := hn; norm_num at this; exact this
  have hcu : ((c i + 1 : ℕ) : ℝ) * R.u < 1 := by
    have := small_mul_u hu hu53 (k := c i + 1) (by have := hcn i; norm_num; omega)
    linarith
  have hsumL : ∀ k, 0 ≤ ∑ j, |L j k| := fun k => Finset.sum_nonneg fun j _ => abs_nonneg _
  have hNn : ∀ k, (Tcol k).adds ≤ Fintype.card n := fun k => by
    have h1 := (hcol k).1.length_le_card
    rw [STree.length_leaves] at h1
    omega
  have hcolge : ∀ k, (1 - R.u) ^ Fintype.card n * ∑ j, |L j k| ≤ colsumFl R Tcol L k := fun k =>
    comp_ge_sum R h0 hu1' (fun j => |L j k|) (fun j => abs_nonneg _) (Tcol k) (hcol k).1
      (fun j hj => by simp [(hcol k).2 j hj]) (hNn k)
  have hw0 : ∀ k, 0 ≤ (1 - R.u) ^ Fintype.card n * ∑ j, |L j k| := fun k =>
    mul_nonneg (pow_nonneg h1u _) (hsumL k)
  have hwge : ∀ k, |d k| * ((1 - R.u) ^ Fintype.card n * ∑ j, |L j k|) * (1 - R.u) ≤
      R.rnd (|d k| * colsumFl R Tcol L k) := fun k =>
    R.le_rnd_mul h0 hu1' (abs_nonneg _) (hw0 k) le_rfl (hcolge k)
  have htge : ∀ k, |L i k| * (|d k| * ((1 - R.u) ^ Fintype.card n * ∑ j, |L j k|) * (1 - R.u)) *
      (1 - R.u) ≤ R.rnd (|L i k| * R.rnd (|d k| * colsumFl R Tcol L k)) := fun k =>
    R.le_rnd_mul h0 hu1' (abs_nonneg _) (mul_nonneg (mul_nonneg (abs_nonneg _) (hw0 k)) h1u)
      le_rfl (hwge k)
  have hG1 : (1 - R.u) ^ (c i + Fintype.card n + 2) * ∑ j, Gabs L d i j ≤
      G1Fl R Tcol Tmv L d i := by
    have hsupp : ∀ k, k ∉ (Tmv i).leaves →
        R.rnd (|L i k| * R.rnd (|d k| * colsumFl R Tcol L k)) = 0 :=
      fun k hk => by rw [(hmv i).2.2 k hk, abs_zero, zero_mul, rnd_zero R h0]
    have hnn : ∀ k, 0 ≤ R.rnd (|L i k| * R.rnd (|d k| * colsumFl R Tcol L k)) := fun k =>
      (mul_nonneg (mul_nonneg (abs_nonneg _) (mul_nonneg (mul_nonneg (abs_nonneg _) (hw0 k)) h1u))
        h1u).trans (htge k)
    have h := comp_ge_sum R h0 hu1' (fun k => R.rnd (|L i k| * R.rnd (|d k| * colsumFl R Tcol L k)))
      hnn (Tmv i) (hmv i).1 hsupp (K := c i) (by have := (hmv i).2.1; omega)
    refine le_trans ?_ h
    have hGs : ∑ j, Gabs L d i j = ∑ k, |L i k| * |d k| * ∑ j, |L j k| := by
      unfold Gabs
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      simp only [Real.norm_eq_abs]
    rw [hGs, Finset.mul_sum, Finset.mul_sum]
    refine Finset.sum_le_sum fun k _ => ?_
    refine le_trans (le_of_eq ?_) (mul_le_mul_of_nonneg_left (htge k) (pow_nonneg h1u _))
    ring
  have hGs0 : 0 ≤ ∑ j, Gabs L d i j :=
    Finset.sum_nonneg fun j _ => Finset.sum_nonneg fun k _ => by positivity
  have hci : ((c i + 1 : ℕ) : ℝ) * R.u ≤ 1 / 2 := small_mul_u hu hu53 (by have := hcn i; norm_num; omega)
  have hgc := (gamHat_bounds R h0 (by linarith) hci).1
  have hgc0 : 0 ≤ gam R.u (c i + 1) := gam_nonneg hu hcu
  have hp3 : 0 ≤ gam R.u (c i + 1) * (1 - R.u) ^ 2 := mul_nonneg hgc0 (sq_nonneg _)
  have hp4 : 0 ≤ (1 - R.u) ^ (c i + Fintype.card n + 2) * ∑ j, Gabs L d i j :=
    mul_nonneg (pow_nonneg h1u _) hGs0
  have hrr := R.le_rnd_mul h0 hu1' hp3 hp4 hgc hG1
  have hM2 := Finset.le_sup' (fun i => R.rnd (gamHat R (c i + 1) * G1Fl R Tcol Tmv L d i))
    (Finset.mem_univ i)
  have h2N : ((2 * Fintype.card n + 8 : ℕ) : ℝ) * R.u ≤ 1 / 2 := small_mul_u hu hu53 (by norm_num; omega)
  have hg2 := (gamHat_bounds R h0 (by linarith) h2N).1
  have hg20 : 0 ≤ gam R.u (2 * Fintype.card n + 8) := gam_nonneg hu (by linarith)
  have hq2 : 0 ≤ 1 + gam R.u (2 * Fintype.card n + 8) * (1 - R.u) ^ 2 :=
    add_nonneg zero_le_one (mul_nonneg hg20 (sq_nonneg _))
  have hc2 : (1 + gam R.u (2 * Fintype.card n + 8) * (1 - R.u) ^ 2) * (1 - R.u) ≤
      R.rnd (1 + gamHat R (2 * Fintype.card n + 8)) :=
    R.le_rnd_of_le h0 hu1' (by linarith) hq2
  have hp5 := mul_nonneg (mul_nonneg hp3 hp4) h1u
  have hy1 := R.le_rnd_mul h0 hu1' hp5 (mul_nonneg hq2 h1u) (hrr.trans hM2) hc2
  have hp6 := mul_nonneg (mul_nonneg hp5 (mul_nonneg hq2 h1u)) h1u
  have hy2 := R.le_rnd_of_le h0 hu1'
    (mul_le_mul_of_nonneg_right hy1 (by linarith : (0 : ℝ) ≤ 1 + 8 * R.u))
    (mul_nonneg hp6 (by linarith))
  have hy3 := R.le_rnd_of_le h0 hu1'
    (mul_le_mul_of_nonneg_right hy2 (by linarith : (0 : ℝ) ≤ 1 + 8 * R.u))
    (mul_nonneg (mul_nonneg (mul_nonneg hp6 (by linarith)) h1u) (by linarith))
  have hr2 := lambda_r2 hu hu53 (hcn i) hn
  have hC : gam R.u (c i + 1) * ∑ j, Gabs L d i j ≤ roundInfFl R Tcol Tmv L d c := by
    refine le_trans ?_ hy3
    calc gam R.u (c i + 1) * ∑ j, Gabs L d i j ≤ gam R.u (c i + 1) * (∑ j, Gabs L d i j) *
          ((1 - R.u) ^ (c i + Fintype.card n + 9) *
            (1 + gam R.u (2 * Fintype.card n + 8) * (1 - R.u) ^ 2) * (1 + 8 * R.u) ^ 2) :=
          le_mul_of_one_le_right (mul_nonneg hgc0 hGs0) hr2
      _ = _ := by ring
  exact hC

/-- **D-01 (c)**, real data: the computed `β` satisfies `β ≥ ‖E‖_∞ + 2u max_i |S_ii|`, in the form
`Σ_j |E_ij| + 2u |S_i'i'| ≤ β` for all `i`, `i'`. -/
theorem d01_c (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) (nb : ℕ)
    (hm : Fintype.card n + nb + 1 ≤ 2 ^ 27) (hn : Fintype.card n ≤ 2 ^ 26)
    {T : n → n → STree n} {L : Matrix n n ℝ} (hT : PatternTrees T L) (d : n → ℝ)
    (Sp S : Matrix n n ℝ) (hSp : ∀ i j, Sp i j = Sp j i) (c : n → ℕ)
    (hc : ∀ i j, (T i j).adds + 1 ≤ c i ∧ (T i j).adds + 1 ≤ c j)
    (hcn : ∀ i, c i ≤ Fintype.card n) (Trow : n → STree (n ⊕ Fin (nb + 1)))
    (hrow : ∀ i, (Trow i).leaves.Nodup ∧
      ∀ j, Sum.inl j ∉ (Trow i).leaves → ecLow R T L d Sp i j = 0)
    (Tcol : n → STree n) (hcol : ∀ k, (Tcol k).leaves.Nodup ∧ ∀ j, j ∉ (Tcol k).leaves → L j k = 0)
    (Tmv : n → STree n)
    (hmv : ∀ i, (Tmv i).leaves.Nodup ∧ (Tmv i).adds + 1 ≤ c i ∧ ∀ k, k ∉ (Tmv i).leaves → L i k = 0)
    (i i' : n) :
    ∑ j, |(L * (dg d : Matrix n n ℝ) * Lᴴ - Sp) i j| + 2 * R.u * |S i' i'| ≤
      betaFl R nb Trow Tcol Tmv T L d Sp S c := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 10 ^ 15 := hu53.trans (by norm_num)
  have hu1' : R.u ≤ 1 := by linarith
  have h1u : 0 ≤ 1 - R.u := by linarith
  have hn' : Fintype.card n ≤ 67108864 := by have := hn; norm_num at this; exact this
  have hcu : ∀ i, ((c i + 1 : ℕ) : ℝ) * R.u < 1 := fun i => by
    have := small_mul_u hu hu53 (k := c i + 1) (by have := hcn i; norm_num; omega)
    linarith
  -- Step A: the row sums of `E`
  have hEsym : ∀ i j, (L * (dg d : Matrix n n ℝ) * Lᴴ - Sp) i j = (L * (dg d : Matrix n n ℝ) * Lᴴ - Sp) j i := fun i j => by
    rw [Matrix.sub_apply, Matrix.sub_apply, ldl_apply, ldl_apply, hSp i j]
    congr 1
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [star_trivial, RCLike.ofReal_real_eq_id, id_eq]
    ring
  have hGsym : ∀ i j, Gabs L d i j = Gabs L d j i := fun i j => by
    unfold Gabs
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  have hA : ∑ j, |(L * (dg d : Matrix n n ℝ) * Lᴴ - Sp) i j| ≤ (∑ j, ecLow R T L d Sp i j) / (1 - R.u) +
      gam R.u (c i + 1) * ∑ j, Gabs L d i j := by
    rw [Finset.sum_div, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun j _ => ?_
    unfold ecLow
    split_ifs with hji
    · exact d01_a h0 hT d Sp i j (hc i j).1 (hcu i)
    · rw [hEsym i j, hGsym i j]
      exact d01_a h0 hT d Sp j i (hc j i).2 (hcu i)
  have hS10 : 0 ≤ ∑ j, ecLow R T L d Sp i j :=
    Finset.sum_nonneg fun j _ => by unfold ecLow; split_ifs <;> exact abs_nonneg _
  have hB := d01_c_resid h0 hu53 nb hm T L d Sp Trow hrow i
  have hGs0 : 0 ≤ ∑ j, Gabs L d i j :=
    Finset.sum_nonneg fun j _ => Finset.sum_nonneg fun k _ => by positivity
  have hgc0 : 0 ≤ gam R.u (c i + 1) := gam_nonneg hu (hcu i)
  have hC := d01_c_round h0 hu53 hn L d c hcn Tcol hcol Tmv hmv i
  -- Step D: the shift term
  have hMS := Finset.le_sup' (fun i => |S i i|) (Finset.mem_univ i')
  have hS0 : 0 ≤ 2 * R.u * |S i' i'| := by positivity
  have hD : 2 * R.u * |S i' i'| * (1 - R.u) ≤ shiftFl R S :=
    R.le_rnd_mul h0 hu1' (by positivity) (abs_nonneg _) le_rfl hMS
  -- Step E: `β`
  have hrI0 : 0 ≤ residInfFl R nb Trow T L d Sp := (div_nonneg hS10 h1u).trans hB
  have hrO0 : 0 ≤ roundInfFl R Tcol Tmv L d c := (mul_nonneg hgc0 hGs0).trans hC
  have hsh0 : 0 ≤ shiftFl R S := (mul_nonneg hS0 h1u).trans hD
  have hsum0 := add_nonneg hrI0 hrO0
  have hz1 := R.le_rnd_of_le h0 hu1' (le_refl (residInfFl R nb Trow T L d Sp +
    roundInfFl R Tcol Tmv L d c)) hsum0
  have hz2 := R.le_rnd_of_le h0 hu1' (add_le_add hz1 (le_refl (shiftFl R S)))
    (add_nonneg (mul_nonneg hsum0 h1u) hsh0)
  have hz3 := R.le_rnd_of_le h0 hu1'
    (mul_le_mul_of_nonneg_right hz2 (by linarith : (0 : ℝ) ≤ 1 + 8 * R.u))
    (mul_nonneg (mul_nonneg (add_nonneg (mul_nonneg hsum0 h1u) hsh0) h1u) (by linarith))
  have hk3 : 1 ≤ (1 - R.u) ^ 3 * (1 + 8 * R.u) := pow3_mul_ge hu (by linarith)
  have hk2 : 0 ≤ (1 - R.u) ^ 2 * (1 + 8 * R.u) := mul_nonneg (sq_nonneg _) (by linarith)
  have e1 : residInfFl R nb Trow T L d Sp + roundInfFl R Tcol Tmv L d c ≤
      (residInfFl R nb Trow T L d Sp + roundInfFl R Tcol Tmv L d c) *
        ((1 - R.u) ^ 3 * (1 + 8 * R.u)) := le_mul_of_one_le_right hsum0 hk3
  have e2 : 2 * R.u * |S i' i'| ≤ shiftFl R S * ((1 - R.u) ^ 2 * (1 + 8 * R.u)) := by
    calc 2 * R.u * |S i' i'| ≤ 2 * R.u * |S i' i'| * ((1 - R.u) ^ 3 * (1 + 8 * R.u)) :=
          le_mul_of_one_le_right hS0 hk3
      _ = 2 * R.u * |S i' i'| * (1 - R.u) * ((1 - R.u) ^ 2 * (1 + 8 * R.u)) := by ring
      _ ≤ shiftFl R S * ((1 - R.u) ^ 2 * (1 + 8 * R.u)) := mul_le_mul_of_nonneg_right hD hk2
  have hfin : residInfFl R nb Trow T L d Sp + roundInfFl R Tcol Tmv L d c + 2 * R.u * |S i' i'| ≤
      betaFl R nb Trow Tcol Tmv T L d Sp S c := by
    refine le_trans ?_ hz3
    calc residInfFl R nb Trow T L d Sp + roundInfFl R Tcol Tmv L d c + 2 * R.u * |S i' i'| ≤
          (residInfFl R nb Trow T L d Sp + roundInfFl R Tcol Tmv L d c) *
            ((1 - R.u) ^ 3 * (1 + 8 * R.u)) + shiftFl R S * ((1 - R.u) ^ 2 * (1 + 8 * R.u)) :=
          add_le_add e1 e2
      _ = _ := by ring
  linarith

/-- **D-01**, real data, the count with the computed `β` (parts (a) to (c) and Step 5): with the
stored shifted matrix `S = A - σ I + diag Δ`, `|Δ_i| ≤ 2u |S_ii|` (the rounding of the diagonal),
`S_P = P S Pᴴ` for a permutation (unitary) `P` and `L` invertible,
`#{λ(A) < σ - β} ≤ ν ≤ #{λ(A) ≤ σ + β}`. -/
theorem d01_count_fl (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) (nb : ℕ)
    (hm : Fintype.card n + nb + 1 ≤ 2 ^ 27) (hn : Fintype.card n ≤ 2 ^ 26)
    {A : Matrix n n ℝ} (hA : A.IsHermitian) (σ : ℝ) (Δ : n → ℝ)
    (P : Matrix n n ℝ) (hP1 : Pᴴ * P = 1) (hP2 : P * Pᴴ = 1)
    {T : n → n → STree n} {L : Matrix n n ℝ} (hL : IsUnit L) (hT : PatternTrees T L) (d : n → ℝ)
    (hΔ : ∀ i, |Δ i| ≤ 2 * R.u * |(A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) i i|) (c : n → ℕ)
    (hc : ∀ i j, (T i j).adds + 1 ≤ c i ∧ (T i j).adds + 1 ≤ c j)
    (hcn : ∀ i, c i ≤ Fintype.card n) (Trow : n → STree (n ⊕ Fin (nb + 1)))
    (hrow : ∀ i, (Trow i).leaves.Nodup ∧ ∀ j, Sum.inl j ∉ (Trow i).leaves →
      ecLow R T L d (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) i j = 0)
    (Tcol : n → STree n) (hcol : ∀ k, (Tcol k).leaves.Nodup ∧ ∀ j, j ∉ (Tcol k).leaves → L j k = 0)
    (Tmv : n → STree n)
    (hmv : ∀ i, (Tmv i).leaves.Nodup ∧ (Tmv i).adds + 1 ≤ c i ∧ ∀ k, k ∉ (Tmv i).leaves → L i k = 0) :
    cntLt hA (σ - betaFl R nb Trow Tcol Tmv T L d (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ)
      (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) c) ≤ (Finset.univ.filter fun i => d i < 0).card ∧
    (Finset.univ.filter fun i => d i < 0).card ≤ cntLe hA (σ + betaFl R nb Trow Tcol Tmv T L d
      (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) c) := by
  have hσ : (σ • (1 : Matrix n n ℝ)).IsHermitian := by
    show (σ • (1 : Matrix n n ℝ))ᴴ = σ • 1
    rw [conjTranspose_smul, conjTranspose_one, star_trivial]
  have hdg : ∀ e : n → ℝ, (dg e : Matrix n n ℝ).IsHermitian := fun e => by
    unfold dg
    exact isHermitian_diagonal _
  have hM : (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)).IsHermitian := (hA.sub hσ).add (hdg Δ)
  have hSpH := isHermitian_mul_mul_conjTranspose P hM
  have hSp : ∀ i j, (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) i j =
      (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) j i := fun i j => by
    have := congrFun (congrFun hSpH.eq j) i
    rw [conjTranspose_apply, star_trivial] at this
    exact this
  set s := 2 * R.u * Finset.univ.sup' Finset.univ_nonempty
    (fun i => |(A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) i i|) with hs
  obtain ⟨i', -, hi'⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty
    (fun i => |(A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) i i|)
  set β := betaFl R nb Trow Tcol Tmv T L d (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ)
    (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) c with hβ
  have hrows : ∀ i, ∑ j, ‖(L * (dg d : Matrix n n ℝ) * Lᴴ - P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) i j‖ ≤
      β - s := fun i => by
    have := d01_c h0 hu53 nb hm hn hT d
      (P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ)
      (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) hSp c hc hcn Trow hrow Tcol hcol Tmv hmv i i'
    rw [← hβ] at this
    rw [hs, hi']
    simp only [Real.norm_eq_abs]
    linarith
  have hΔ' : ∀ i, |Δ i| ≤ s := fun i => (hΔ i).trans (by
    rw [hs]
    exact mul_le_mul_of_nonneg_left (Finset.le_sup' (fun i =>
      |(A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) i i|) (Finset.mem_univ i))
      (by have := R.u_nonneg; positivity))
  have hEh : (L * (dg d : Matrix n n ℝ) * Lᴴ - P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ).IsHermitian :=
    (isHermitian_mul_mul_conjTranspose L (hdg d)).sub hSpH
  have h := lemma4_perm_rowSum hA σ (β - s) s P hP1 hP2 L hL d Δ hΔ'
    (L * (dg d : Matrix n n ℝ) * Lᴴ - P * (A - σ • (1 : Matrix n n ℝ) + (dg Δ : Matrix n n ℝ)) * Pᴴ) hEh hrows (by
      simp only [RCLike.ofReal_real_eq_id, id_eq]
      abel)
  simpa using h

end PartC

end CertifyMathlib.Counting
