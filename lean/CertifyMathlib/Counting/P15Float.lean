import CertifyMathlib.Enclosure.FloatAccounting
import CertifyMathlib.Counting.Thermo

/-!
# Proposition 15, floating-point corollary (Lemmas A and B)

Source: `docs/proofs/P15_summed_bounds.md` in the code repository and `combine_certified`,
`modulus_upper` in `certify_ed/correlations.py`. The arithmetic is the standard model of
`CertifyMathlib.Enclosure.FloatModel` (a rounding `R` with `eta = 0`, hypothesis (H3) of the
document) with `u ≤ 2⁻¹⁰` (binary64 has `u = 2⁻⁵³`), and the upward neighbour `succ` of a rounded
result (`numpy.nextafter(·, +∞)`) is any function with `z ≤ succ (rnd z)`.

* `upFl`: `fp.up(x) = fl(x (1 + 8u))`; `modulusUpper`: `modulus_upper(z)`, with its three branches.
* `modulusUpper_ge`: **Lemma A**, `|z| ≤ modulus_upper(z)`.
* `dotFlC`, `dotFlC_error`: **Lemma B**, `numpy.sum(a * v)` as complex products (four real products
  and two real sums, `cmul`) summed in any order with componentwise rounding (`crnd`), and
  `|fl(Σ a_k v_k) - Σ a_k v_k| ≤ γ_{m+2} Σ |a_k| |v_k|`.
* `sumUp`, `sumUp_ge`: the sums accumulated with `add_up` from `0` are upper bounds of the exact sums.
* `combineBound`, `p15_fl`: the bound `B` that `combine_certified` returns, and inequality (2),
  `|Σ a_k tr(P O_k) - val| ≤ B`.

A summation tree has at least one leaf, so `p15_fl` needs `m ≥ 1`; the case `m = 0` (no
coefficient: the code returns the value `0` and the bound `add_up(0, 0)`) is `p15_fl_zero`.
-/

namespace CertifyMathlib.Counting

open CertifyMathlib.Enclosure.FP

section P15Float

variable (R : Rounding ℝ)

/-- `fp.up(x) = fl(x (1 + 8u))` (`1 + 8u` is a binary64 number). -/
noncomputable def upFl (x : ℝ) : ℝ := R.rnd (x * (1 + 8 * R.u))

/-- `modulus_upper(z)`: `|x|` or `|y|` when the other component is `0`, `up(fl(|x| + |y|))` when a
component is below `2⁻⁵⁰⁰`, and `up(fl(√(fl(fl(x²) + fl(y²)))))` otherwise. -/
noncomputable def modulusUpper (z : ℂ) : ℝ :=
  if z.im = 0 then |z.re| else if z.re = 0 then |z.im|
  else if min |z.re| |z.im| < (2 : ℝ)⁻¹ ^ 500 then upFl R (R.rnd (|z.re| + |z.im|))
  else upFl R (R.rnd (Real.sqrt (R.rnd (R.rnd (z.re * z.re) + R.rnd (z.im * z.im)))))

variable {R}

theorem le_upFl (h0 : R.eta = 0) (hu1 : R.u ≤ 1) {w z : ℝ} (hz : 0 ≤ z) (hzw : z ≤ w) :
    z * (1 + 8 * R.u) * (1 - R.u) ≤ upFl R w := by
  have hu := R.u_nonneg
  exact R.le_rnd_of_le h0 hu1 (mul_le_mul_of_nonneg_right hzw (by linarith))
    (mul_nonneg hz (by linarith))

theorem norm_eq_sqrt_re_im (z : ℂ) : ‖z‖ = Real.sqrt (z.re * z.re + z.im * z.im) := by
  rw [Complex.norm_def, Complex.normSq_apply]

/-- **Lemma A**: `|z| ≤ modulus_upper(z)`. -/
theorem modulusUpper_ge (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) (z : ℂ) :
    ‖z‖ ≤ modulusUpper R z := by
  have hu := R.u_nonneg
  have hu1' : R.u ≤ 1 := by linarith
  have hc3 : 1 ≤ (1 - R.u) ^ 3 * (1 + 8 * R.u) := by nlinarith [sq_nonneg R.u, mul_nonneg hu (sq_nonneg R.u)]
  have hc2 : 1 ≤ (1 - R.u) ^ 2 * (1 + 8 * R.u) := by nlinarith [sq_nonneg R.u]
  rw [norm_eq_sqrt_re_im]
  unfold modulusUpper
  split_ifs with h1 h2 h3
  · rw [h1, mul_zero, add_zero, Real.sqrt_mul_self_eq_abs]
  · rw [h2, mul_zero, zero_add, Real.sqrt_mul_self_eq_abs]
  · -- the branch `up(fl(|x| + |y|))`
    have hs : Real.sqrt (z.re * z.re + z.im * z.im) ≤ |z.re| + |z.im| := by
      calc Real.sqrt (z.re * z.re + z.im * z.im) ≤ Real.sqrt ((|z.re| + |z.im|) ^ 2) :=
            Real.sqrt_le_sqrt (by
              have e1 := abs_mul_abs_self z.re
              have e2 := abs_mul_abs_self z.im
              have e3 := mul_nonneg (abs_nonneg z.re) (abs_nonneg z.im)
              nlinarith)
        _ = |z.re| + |z.im| := Real.sqrt_sq (by positivity)
    have hq := R.le_rnd_of_le h0 hu1' (le_refl (|z.re| + |z.im|)) (by positivity)
    have hup := le_upFl h0 hu1' (mul_nonneg (by positivity) (by linarith)) hq
    have h0' : 0 ≤ |z.re| + |z.im| := by positivity
    calc Real.sqrt (z.re * z.re + z.im * z.im) ≤ |z.re| + |z.im| := hs
      _ ≤ (|z.re| + |z.im|) * ((1 - R.u) ^ 2 * (1 + 8 * R.u)) := le_mul_of_one_le_right h0' hc2
      _ = (|z.re| + |z.im|) * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) := by ring
      _ ≤ _ := hup
  · -- the main branch `up(fl(√(fl(fl(x²) + fl(y²)))))`
    have hx := R.le_rnd_of_le h0 hu1' (le_refl (z.re * z.re)) (mul_self_nonneg _)
    have hy := R.le_rnd_of_le h0 hu1' (le_refl (z.im * z.im)) (mul_self_nonneg _)
    have hA : 0 ≤ z.re * z.re + z.im * z.im := add_nonneg (mul_self_nonneg _) (mul_self_nonneg _)
    have h1u : 0 ≤ 1 - R.u := by linarith
    have hsum0 : 0 ≤ (z.re * z.re + z.im * z.im) * (1 - R.u) := mul_nonneg hA h1u
    have hp := R.le_rnd_of_le h0 hu1' (by linarith : (z.re * z.re + z.im * z.im) * (1 - R.u) ≤
      R.rnd (z.re * z.re) + R.rnd (z.im * z.im)) hsum0
    have hr0 : 0 ≤ Real.sqrt (z.re * z.re + z.im * z.im) := Real.sqrt_nonneg _
    have e : Real.sqrt (z.re * z.re + z.im * z.im) * (1 - R.u) =
        Real.sqrt ((z.re * z.re + z.im * z.im) * (1 - R.u) * (1 - R.u)) := by
      rw [mul_assoc (z.re * z.re + z.im * z.im), Real.sqrt_mul hA, Real.sqrt_mul_self h1u]
    have hsq : Real.sqrt (z.re * z.re + z.im * z.im) * (1 - R.u) ≤
        Real.sqrt (R.rnd (R.rnd (z.re * z.re) + R.rnd (z.im * z.im))) := by
      rw [e]
      exact Real.sqrt_le_sqrt hp
    have hs := R.le_rnd_of_le h0 hu1' hsq (mul_nonneg hr0 h1u)
    have hup := le_upFl h0 hu1' (mul_nonneg (mul_nonneg hr0 h1u) h1u) hs
    calc Real.sqrt (z.re * z.re + z.im * z.im) ≤
          Real.sqrt (z.re * z.re + z.im * z.im) * ((1 - R.u) ^ 3 * (1 + 8 * R.u)) :=
          le_mul_of_one_le_right hr0 hc3
      _ = Real.sqrt (z.re * z.re + z.im * z.im) * (1 - R.u) * (1 - R.u) * (1 + 8 * R.u) *
          (1 - R.u) := by ring
      _ ≤ _ := hup

theorem modulusUpper_nonneg (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) (z : ℂ) :
    0 ≤ modulusUpper R z :=
  (norm_nonneg z).trans (modulusUpper_ge h0 hu1 z)

variable (R)

/-- `numpy.sum(a * v)`: complex products `cmul`, summed along `T` with componentwise rounding. -/
noncomputable def dotFlC {m : ℕ} (T : STree (Fin m)) (a v : Fin m → ℂ) : ℂ :=
  T.comp (crnd R) (fun k => cmul R (a k) (v k))

variable {R}

/-- **Lemma B**: `|fl(Σ a_k v_k) - Σ a_k v_k| ≤ γ_{m+2} Σ |a_k| |v_k|`, for any summation order. -/
theorem dotFlC_error (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) {m : ℕ} (T : STree (Fin m))
    (hT : FullTree T) (hm : ((m + 2 : ℕ) : ℝ) * R.u < 1) (a v : Fin m → ℂ) :
    ‖dotFlC R T a v - ∑ k, a k * v k‖ ≤ gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖ := by
  have hu := R.u_nonneg
  have hlen : T.adds + 1 ≤ m := by
    rw [← STree.length_leaves]
    simpa using hT.1.length_le_card
  have hle : 3 + T.adds ≤ m + 2 := by omega
  have hk : ((3 + T.adds : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hle) hu) hm
  have h3 : ((3 : ℕ) : ℝ) * R.u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 3 ≤ m + 2)) hu) hm
  have hleaf : ∀ k, ‖cmul R (a k) (v k) - a k * v k‖ ≤ gam R.u 3 * ‖a k * v k‖ + 0 := fun k => by
    rw [add_zero]
    exact cmul_err R h0 hu1 (a k) (v k)
  have h := STree.comp_error (cRounding R h0) (a := 3) (β := 0) le_rfl (fun k => a k * v k)
    (fun k => cmul R (a k) (v k)) hleaf (fun _ => trivial) T hk
  have hsupp : ∀ k, k ∉ T.leaves → a k * v k = 0 := fun k hk => absurd (hT.2 k) hk
  rw [STree.eval_eq_finset_sum _ T hT.1 hsupp, STree.absEval_eq_finset_sum _ T hT.1 hsupp] at h
  simp only [mul_zero, add_zero] at h
  have hg := gam_mono hu hle hm
  have hS : 0 ≤ ∑ k, ‖a k * v k‖ := Finset.sum_nonneg fun k _ => norm_nonneg _
  calc ‖dotFlC R T a v - ∑ k, a k * v k‖ ≤ gam R.u (3 + T.adds) * ∑ k, ‖a k * v k‖ := h
    _ ≤ gam R.u (m + 2) * ∑ k, ‖a k * v k‖ := mul_le_mul_of_nonneg_right hg hS
    _ = gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖ := by simp only [norm_mul]

variable (R)

/-- `Σ↑`: the terms accumulated with `add_up(s, x) = succ(fl(s + x))`, starting from `0`. -/
noncomputable def sumUp (succ : ℝ → ℝ) (l : List ℝ) : ℝ :=
  l.foldl (fun s x => succ (R.rnd (s + x))) 0

variable {R}

theorem foldl_addUp_ge (succ : ℝ → ℝ) (hsucc : ∀ z, z ≤ succ (R.rnd z)) :
    ∀ (l : List ℝ) (s s' : ℝ), s' ≤ s →
      s' + l.sum ≤ l.foldl (fun s x => succ (R.rnd (s + x))) s
  | [], s, s', h => by simpa using h
  | x :: l, s, s', h => by
    simp only [List.foldl_cons, List.sum_cons]
    have := foldl_addUp_ge succ hsucc l (succ (R.rnd (s + x))) (s' + x)
      ((by linarith : s' + x ≤ s + x).trans (hsucc _))
    linarith

theorem sumUp_ge (succ : ℝ → ℝ) (hsucc : ∀ z, z ≤ succ (R.rnd z)) (l : List ℝ) :
    l.sum ≤ sumUp R succ l := by
  have := foldl_addUp_ge succ hsucc l 0 0 le_rfl
  simpa [sumUp] using this

variable (R)

/-- The bound `B` returned by `combine_certified`:
`add_up(Σ↑ up(|a_k|↑ b_k), up(2 γ̂_{m+2} Σ↑ up(|a_k|↑ |v_k|↑)))`. -/
noncomputable def combineBound (succ : ℝ → ℝ) {m : ℕ} (a v : Fin m → ℂ) (b : Fin m → ℝ) : ℝ :=
  succ (R.rnd (sumUp R succ (List.ofFn fun k => upFl R (R.rnd (modulusUpper R (a k) * b k))) +
    upFl R (R.rnd (R.rnd (2 * gamHat R (m + 2)) *
      sumUp R succ (List.ofFn fun k =>
        upFl R (R.rnd (modulusUpper R (a k) * modulusUpper R (v k))))))))

variable {R}

theorem up_mul_ge (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000) {x y x' y' : ℝ} (hx' : 0 ≤ x')
    (hy' : 0 ≤ y') (hx : x' ≤ x) (hy : y' ≤ y) : x' * y' ≤ upFl R (R.rnd (x * y)) := by
  have hu := R.u_nonneg
  have hu1' : R.u ≤ 1 := by linarith
  have h1 := R.le_rnd_mul h0 hu1' hx' hy' hx hy
  have h2 := le_upFl h0 hu1' (mul_nonneg (mul_nonneg hx' hy') (by linarith)) h1
  have hc2 : 1 ≤ (1 - R.u) ^ 2 * (1 + 8 * R.u) := by nlinarith [sq_nonneg R.u]
  calc x' * y' ≤ x' * y' * ((1 - R.u) ^ 2 * (1 + 8 * R.u)) :=
        le_mul_of_one_le_right (mul_nonneg hx' hy') hc2
    _ = x' * y' * (1 - R.u) * (1 + 8 * R.u) * (1 - R.u) := by ring
    _ ≤ _ := h2

/-- **Proposition 15, floating-point corollary (2)**: `|Σ a_k tr(P O_k) - val| ≤ B`, for the value
`val = fl(Σ a_k v_k)` summed along any tree `T` and the bound `B` of `combine_certified`, when
`|tr(P O_k) - v_k| ≤ b_k`. -/
theorem p15_fl (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 10) (succ : ℝ → ℝ)
    (hsucc : ∀ z, z ≤ succ (R.rnd z)) {m : ℕ} (T : STree (Fin m)) (hT : FullTree T)
    (hm : ((m + 2 : ℕ) : ℝ) * R.u ≤ 1 / 2) (a v t : Fin m → ℂ) (b : Fin m → ℝ)
    (hb : ∀ k, ‖t k - v k‖ ≤ b k) :
    ‖∑ k, a k * t k - dotFlC R T a v‖ ≤ combineBound R succ a v b := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 1000 := hu53.trans (by norm_num)
  have hu1' : R.u ≤ 1 := by linarith
  have hb0 : ∀ k, 0 ≤ b k := fun k => (norm_nonneg _).trans (hb k)
  have hm' : ((m + 2 : ℕ) : ℝ) * R.u < 1 := by linarith
  have hg0 : 0 ≤ gam R.u (m + 2) := gam_nonneg hu hm'
  -- (1) and Lemma B
  have h1 : ‖∑ k, a k * t k - ∑ k, a k * v k‖ ≤ ∑ k, ‖a k‖ * b k := by
    rw [← Finset.sum_sub_distrib]
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun k _ => ?_)
    rw [← mul_sub, norm_mul]
    exact mul_le_mul_of_nonneg_left (hb k) (norm_nonneg _)
  have h2 := dotFlC_error h0 hu1 T hT hm' a v
  -- the two sums of the bound
  have hB1 : ∑ k, ‖a k‖ * b k ≤
      sumUp R succ (List.ofFn fun k => upFl R (R.rnd (modulusUpper R (a k) * b k))) := by
    refine le_trans ?_ (sumUp_ge succ hsucc _)
    rw [List.sum_ofFn]
    exact Finset.sum_le_sum fun k _ =>
      up_mul_ge h0 hu1 (norm_nonneg _) (hb0 k) (modulusUpper_ge h0 hu1 (a k)) le_rfl
  set S2 := sumUp R succ (List.ofFn fun k =>
    upFl R (R.rnd (modulusUpper R (a k) * modulusUpper R (v k)))) with hS2
  have hB2 : ∑ k, ‖a k‖ * ‖v k‖ ≤ S2 := by
    refine le_trans ?_ (sumUp_ge succ hsucc _)
    rw [List.sum_ofFn]
    exact Finset.sum_le_sum fun k _ =>
      up_mul_ge h0 hu1 (norm_nonneg _) (norm_nonneg _) (modulusUpper_ge h0 hu1 (a k))
        (modulusUpper_ge h0 hu1 (v k))
  have hA0 : 0 ≤ ∑ k, ‖a k‖ * ‖v k‖ := Finset.sum_nonneg fun k _ => by positivity
  -- the rounding term
  have hgh := (gamHat_bounds R h0 (by linarith) (k := m + 2) hm).1
  have h1u : 0 ≤ 1 - R.u := by linarith
  have hgu : 0 ≤ gam R.u (m + 2) * (1 - R.u) ^ 2 := mul_nonneg hg0 (sq_nonneg _)
  have hx1 : 2 * (gam R.u (m + 2) * (1 - R.u) ^ 2) * (1 - R.u) ≤ R.rnd (2 * gamHat R (m + 2)) :=
    R.le_rnd_of_le h0 hu1' (by linarith) (mul_nonneg zero_le_two hgu)
  have hx2 : 2 * (gam R.u (m + 2) * (1 - R.u) ^ 2) * (1 - R.u) * (∑ k, ‖a k‖ * ‖v k‖) *
      (1 - R.u) ≤ R.rnd (R.rnd (2 * gamHat R (m + 2)) * S2) :=
    R.le_rnd_mul h0 hu1' (mul_nonneg (mul_nonneg zero_le_two hgu) h1u) hA0 hx1 hB2
  have hx3 := le_upFl h0 hu1'
    (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg zero_le_two hgu) h1u) hA0) h1u) hx2
  have hc : 1 ≤ 2 * (1 - R.u) ^ 5 * (1 + 8 * R.u) := by
    have h5 : 1 - 5 * R.u ≤ (1 - R.u) ^ 5 := by
      have := one_sub_mul_le_pow hu hu1' 5
      push_cast at this
      linarith
    have h6 := mul_le_mul_of_nonneg_right h5 (by linarith : (0 : ℝ) ≤ 1 + 8 * R.u)
    nlinarith
  have hB3 : gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖ ≤
      upFl R (R.rnd (R.rnd (2 * gamHat R (m + 2)) * S2)) := by
    refine le_trans ?_ hx3
    have hp : 0 ≤ gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖ := mul_nonneg hg0 hA0
    calc gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖ ≤
          (gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖) * (2 * (1 - R.u) ^ 5 * (1 + 8 * R.u)) :=
          le_mul_of_one_le_right hp hc
      _ = _ := by ring
  have hfin := hsucc (sumUp R succ (List.ofFn fun k => upFl R (R.rnd (modulusUpper R (a k) * b k))) +
    upFl R (R.rnd (R.rnd (2 * gamHat R (m + 2)) * S2)))
  calc ‖∑ k, a k * t k - dotFlC R T a v‖ ≤
        ‖∑ k, a k * t k - ∑ k, a k * v k‖ + ‖∑ k, a k * v k - dotFlC R T a v‖ :=
        norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ ∑ k, ‖a k‖ * b k + gam R.u (m + 2) * ∑ k, ‖a k‖ * ‖v k‖ := by
        rw [norm_sub_rev (∑ k, a k * v k)]
        exact add_le_add h1 h2
    _ ≤ _ := by
        unfold combineBound
        rw [← hS2]
        linarith

/-- **Proposition 15, floating-point corollary (2)**, the case `m = 0`: the value is `0` and the
bound `add_up(0, 0)` of `combine_certified` is at least `|Σ_∅ - 0| = 0`. -/
theorem p15_fl_zero (succ : ℝ → ℝ) (hsucc : ∀ z, z ≤ succ (R.rnd z)) (a t : Fin 0 → ℂ) :
    ‖∑ k, a k * t k - 0‖ ≤ succ (R.rnd (0 + 0)) := by
  simpa using hsucc 0

end P15Float

end CertifyMathlib.Counting
