import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Complex.Exponential

/-!
# Lemma 17: dyadic interval arithmetic for exp and ln

`docs/proofs/L17_dyadic_exp_ln.md` (`certify_ed/ivl.py`). An interval is a pair of integers
`(lo, hi)` standing for `[lo 2^-P, hi 2^-P]`; Python's `a // b` and `-((-a) // b)` for `b > 0` are
`⌊a / b⌋` and `⌈a / b⌉` of the real quotient, which is how the integer steps are written here
(`P` is arbitrary; the code uses `P = 192`, `N = 40`, `M = 60`).

* (i) `from_fraction` encloses its argument (`from_fraction_mem`);
* (ii) the product of two grid intervals with the corner products rounded outward encloses every
  product of members (`mul_mem_corners`, `grid_mul_mem`), and the squaring step of a nonnegative
  interval (`sq_step`);
* (iii) exp: the lower Taylor terms `term_k^↓ = ⌊⌊term_{k-1} t / 2^P⌋ / k⌋` stay below
  `2^P τ^k / k!` and the upper ones above (`termDn_le`, `termUp_ge`), so for a grid point
  `0 ≤ t ≤ 2^P` the interval `(S_N^↓(t), S_N^↑(t) + rem)` contains `exp(t 2^-P)`
  (`exp_point_mem`, the remainder from `Real.exp_bound'`); `s` squarings give `exp(2^s y)`
  (`sqIter_mem`), which with `⌊x / 2^s⌋ ≤ x / 2^s ≤ ⌈x / 2^s⌉` gives the enclosure of
  `exp(x 2^-P)` for `x ≥ 0` (`exp_nonneg_mem`), and the reciprocal step gives `x < 0`
  (`exp_neg_mem`);
* (iv) ln: the lower atanh sum `A^↓(u) = 2 Σ_{j ≤ M} ⌊pw_j / (2j+1)⌋` is at most
  `2^P (log(1 + x) - log(1 - x))` for `x = u 2^-P < 1` (`lnSeriesDn_le`), and
  `log(1 + x) - log(1 - x) = log m` for `x = (m - 1)/(m + 1)` (`log_eq_atanh_form`).

Not formalized: the upper atanh sum with its tail term `⌈⌈8 pw_{M+1} / 7⌉ / (2M+3)⌉`, the
`k · ln 2` decomposition with the enclosure of `ln 2`, the halving count identity
`⌈⌈a/b⌉/c⌉ = ⌈a/(bc)⌉`, and (v), the conversion to binary64.
-/

open Finset Real

namespace CertifyMathlib.Enclosure.Dyadic

variable (P : ℕ)

/-! ### (i) and (ii): grid operations -/

/-- (i) `from_fraction(q) = (⌊q 2^P⌋, ⌈q 2^P⌉)` contains `q`. -/
theorem from_fraction_mem (q : ℝ) :
    ((⌊q * 2 ^ P⌋ : ℤ) : ℝ) / 2 ^ P ≤ q ∧ q ≤ ((⌈q * 2 ^ P⌉ : ℤ) : ℝ) / 2 ^ P := by
  have h2 : (0 : ℝ) < 2 ^ P := by positivity
  constructor
  · rw [div_le_iff₀ h2]
    exact Int.floor_le _
  · rw [le_div_iff₀ h2]
    exact Int.le_ceil _

/-- The product of members of two intervals lies between the extreme corner products. -/
theorem mul_mem_corners {x y xl xh yl yh : ℝ} (hxl : xl ≤ x) (hxh : x ≤ xh) (hyl : yl ≤ y)
    (hyh : y ≤ yh) :
    min (min (xl * yl) (xl * yh)) (min (xh * yl) (xh * yh)) ≤ x * y ∧
      x * y ≤ max (max (xl * yl) (xl * yh)) (max (xh * yl) (xh * yh)) := by
  have hl : ∀ z, min (z * yl) (z * yh) ≤ z * y ∧ z * y ≤ max (z * yl) (z * yh) := by
    intro z
    rcases le_total 0 z with hz | hz
    · exact ⟨min_le_of_left_le (mul_le_mul_of_nonneg_left hyl hz),
        le_max_of_le_right (mul_le_mul_of_nonneg_left hyh hz)⟩
    · exact ⟨min_le_of_right_le (mul_le_mul_of_nonpos_left hyh hz),
        le_max_of_le_left (mul_le_mul_of_nonpos_left hyl hz)⟩
  have hm : min (xl * y) (xh * y) ≤ x * y ∧ x * y ≤ max (xl * y) (xh * y) := by
    rcases le_total 0 y with hy | hy
    · exact ⟨min_le_of_left_le (mul_le_mul_of_nonneg_right hxl hy),
        le_max_of_le_right (mul_le_mul_of_nonneg_right hxh hy)⟩
    · exact ⟨min_le_of_right_le (mul_le_mul_of_nonpos_right hxh hy),
        le_max_of_le_left (mul_le_mul_of_nonpos_right hxl hy)⟩
  obtain ⟨l1, u1⟩ := hl xl
  obtain ⟨l2, u2⟩ := hl xh
  constructor
  · refine le_trans ?_ hm.1
    exact le_min (min_le_of_left_le l1) (min_le_of_right_le l2)
  · refine hm.2.trans ?_
    exact max_le (le_max_of_le_left u1) (le_max_of_le_right u2)

/-- (ii) **Grid product**: `(⌊min p / 2^P⌋, ⌈max p / 2^P⌉)`, `p` the four corner products of the
grid integers, contains `x y` for `x ∈ [xl, xh] 2^-P`, `y ∈ [yl, yh] 2^-P`. -/
theorem grid_mul_mem {x y : ℝ} {xl xh yl yh : ℤ} (hxl : (xl : ℝ) / 2 ^ P ≤ x)
    (hxh : x ≤ (xh : ℝ) / 2 ^ P) (hyl : (yl : ℝ) / 2 ^ P ≤ y) (hyh : y ≤ (yh : ℝ) / 2 ^ P) :
    ((⌊((min (min (xl * yl) (xl * yh)) (min (xh * yl) (xh * yh)) : ℤ) : ℝ) / 2 ^ P⌋ : ℤ) : ℝ) /
        2 ^ P ≤ x * y ∧
      x * y ≤ ((⌈((max (max (xl * yl) (xl * yh)) (max (xh * yl) (xh * yh)) : ℤ) : ℝ) / 2 ^ P⌉ :
        ℤ) : ℝ) / 2 ^ P := by
  have h2 : (0 : ℝ) < 2 ^ P := by positivity
  have hx1 : (xl : ℝ) ≤ x * 2 ^ P := by rwa [div_le_iff₀ h2] at hxl
  have hx2 : x * 2 ^ P ≤ (xh : ℝ) := by rwa [le_div_iff₀ h2] at hxh
  have hy1 : (yl : ℝ) ≤ y * 2 ^ P := by rwa [div_le_iff₀ h2] at hyl
  have hy2 : y * 2 ^ P ≤ (yh : ℝ) := by rwa [le_div_iff₀ h2] at hyh
  obtain ⟨c1, c2⟩ := mul_mem_corners hx1 hx2 hy1 hy2
  have e : x * 2 ^ P * (y * 2 ^ P) = x * y * 2 ^ P * 2 ^ P := by ring
  rw [e] at c1 c2
  push_cast
  constructor
  · rw [div_le_iff₀ h2]
    refine (Int.floor_le _).trans ?_
    rw [div_le_iff₀ h2]
    exact c1
  · rw [le_div_iff₀ h2]
    refine le_trans ?_ (Int.le_ceil _)
    rw [le_div_iff₀ h2]
    exact c2

/-- The squaring step of a nonnegative interval: `(⌊lo² / 2^P⌋, ⌈hi² / 2^P⌉)` contains `e²`. -/
theorem sq_step {e : ℝ} {lo hi : ℤ} (hlo : 0 ≤ lo) (h1 : (lo : ℝ) / 2 ^ P ≤ e)
    (h2 : e ≤ (hi : ℝ) / 2 ^ P) :
    ((⌊(lo : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) / 2 ^ P ≤ e ^ 2 ∧
      e ^ 2 ≤ ((⌈(hi : ℝ) ^ 2 / 2 ^ P⌉ : ℤ) : ℝ) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have hl0 : (0 : ℝ) ≤ (lo : ℝ) / 2 ^ P := div_nonneg (by exact_mod_cast hlo) hP.le
  have he0 : 0 ≤ e := hl0.trans h1
  constructor
  · calc ((⌊(lo : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) / 2 ^ P ≤ ((lo : ℝ) ^ 2 / 2 ^ P) / 2 ^ P :=
          div_le_div_of_nonneg_right (Int.floor_le _) hP.le
      _ = ((lo : ℝ) / 2 ^ P) ^ 2 := by field_simp
      _ ≤ e ^ 2 := pow_le_pow_left₀ hl0 h1 2
  · calc e ^ 2 ≤ ((hi : ℝ) / 2 ^ P) ^ 2 := pow_le_pow_left₀ he0 h2 2
      _ = ((hi : ℝ) ^ 2 / 2 ^ P) / 2 ^ P := by field_simp
      _ ≤ ((⌈(hi : ℝ) ^ 2 / 2 ^ P⌉ : ℤ) : ℝ) / 2 ^ P :=
          div_le_div_of_nonneg_right (Int.le_ceil _) hP.le

/-! ### (iii) exp -/

/-- `term_k^↓(t)`: `term_0 = 2^P`, `term_k = ⌊⌊term_{k-1} t / 2^P⌋ / k⌋`. -/
noncomputable def termDn (t : ℤ) : ℕ → ℤ
  | 0 => 2 ^ P
  | k + 1 => ⌊((⌊(termDn t k : ℝ) * t / 2 ^ P⌋ : ℤ) : ℝ) / ((k : ℝ) + 1)⌋

/-- `term_k^↑(t)`: `term_0 = 2^P`, `term_k = ⌈⌈term_{k-1} t / 2^P⌉ / k⌉`. -/
noncomputable def termUp (t : ℤ) : ℕ → ℤ
  | 0 => 2 ^ P
  | k + 1 => ⌈((⌈(termUp t k : ℝ) * t / 2 ^ P⌉ : ℤ) : ℝ) / ((k : ℝ) + 1)⌉

/-- `rem = ⌈2 ⌈term_N^↑ t / 2^P⌉ / (N + 1)⌉`. -/
noncomputable def remUp (t : ℤ) (N : ℕ) : ℤ :=
  ⌈2 * ((⌈(termUp P t N : ℝ) * t / 2 ^ P⌉ : ℤ) : ℝ) / ((N : ℝ) + 1)⌉

theorem termDn_le (t : ℤ) (ht : 0 ≤ t) :
    ∀ k, (termDn P t k : ℝ) ≤ 2 ^ P * ((t : ℝ) / 2 ^ P) ^ k / k.factorial
  | 0 => by simp [termDn]
  | k + 1 => by
    have ih := termDn_le t ht k
    have hP : (0 : ℝ) < 2 ^ P := by positivity
    have htr : (0 : ℝ) ≤ t := by exact_mod_cast ht
    have hk : (0 : ℝ) < (k : ℝ) + 1 := by positivity
    simp only [termDn]
    calc ((⌊((⌊(termDn P t k : ℝ) * t / 2 ^ P⌋ : ℤ) : ℝ) / ((k : ℝ) + 1)⌋ : ℤ) : ℝ)
        ≤ ((⌊(termDn P t k : ℝ) * t / 2 ^ P⌋ : ℤ) : ℝ) / ((k : ℝ) + 1) := Int.floor_le _
      _ ≤ ((termDn P t k : ℝ) * t / 2 ^ P) / ((k : ℝ) + 1) :=
          div_le_div_of_nonneg_right (Int.floor_le _) hk.le
      _ ≤ (2 ^ P * ((t : ℝ) / 2 ^ P) ^ k / k.factorial * t / 2 ^ P) / ((k : ℝ) + 1) := by
          gcongr
      _ = 2 ^ P * ((t : ℝ) / 2 ^ P) ^ (k + 1) / (k + 1).factorial := by
          rw [Nat.factorial_succ]
          push_cast
          have hc : (2 : ℝ) ^ P ≠ 0 := hP.ne'
          have hf : (k.factorial : ℝ) ≠ 0 := by positivity
          generalize (2 : ℝ) ^ P = c at hc ⊢
          simp only [div_pow]
          field_simp
          ring

theorem termUp_ge (t : ℤ) (ht : 0 ≤ t) :
    ∀ k, 2 ^ P * ((t : ℝ) / 2 ^ P) ^ k / k.factorial ≤ (termUp P t k : ℝ)
  | 0 => by simp [termUp]
  | k + 1 => by
    have ih := termUp_ge t ht k
    have hP : (0 : ℝ) < 2 ^ P := by positivity
    have htr : (0 : ℝ) ≤ t := by exact_mod_cast ht
    have hk : (0 : ℝ) < (k : ℝ) + 1 := by positivity
    simp only [termUp]
    calc 2 ^ P * ((t : ℝ) / 2 ^ P) ^ (k + 1) / (k + 1).factorial
        = (2 ^ P * ((t : ℝ) / 2 ^ P) ^ k / k.factorial * t / 2 ^ P) / ((k : ℝ) + 1) := by
          rw [Nat.factorial_succ]
          push_cast
          have hc : (2 : ℝ) ^ P ≠ 0 := hP.ne'
          have hf : (k.factorial : ℝ) ≠ 0 := by positivity
          generalize (2 : ℝ) ^ P = c at hc ⊢
          simp only [div_pow]
          field_simp
          ring
      _ ≤ ((termUp P t k : ℝ) * t / 2 ^ P) / ((k : ℝ) + 1) := by gcongr
      _ ≤ ((⌈(termUp P t k : ℝ) * t / 2 ^ P⌉ : ℤ) : ℝ) / ((k : ℝ) + 1) :=
          div_le_div_of_nonneg_right (Int.le_ceil _) hk.le
      _ ≤ ((⌈((⌈(termUp P t k : ℝ) * t / 2 ^ P⌉ : ℤ) : ℝ) / ((k : ℝ) + 1)⌉ : ℤ) : ℝ) :=
          Int.le_ceil _

theorem termDn_nonneg (t : ℤ) (ht : 0 ≤ t) : ∀ k, 0 ≤ termDn P t k
  | 0 => by simp [termDn]
  | k + 1 => by
    have ih := termDn_nonneg t ht k
    simp only [termDn]
    refine Int.floor_nonneg.mpr (div_nonneg ?_ (by positivity))
    have : (0 : ℝ) ≤ (termDn P t k : ℝ) * t / 2 ^ P :=
      div_nonneg (mul_nonneg (by exact_mod_cast ih) (by exact_mod_cast ht)) (by positivity)
    exact_mod_cast Int.floor_nonneg.mpr this

/-- **(iii), one point**: for a grid point `0 ≤ t ≤ 2^P` (`τ = t 2^-P ≤ 1`),
`S_N^↓(t) 2^-P ≤ exp τ ≤ (S_N^↑(t) + rem) 2^-P`. -/
theorem exp_point_mem (t : ℤ) (ht : 0 ≤ t) (ht1 : (t : ℝ) ≤ 2 ^ P) (N : ℕ) :
    ((∑ k ∈ range (N + 1), termDn P t k : ℤ) : ℝ) / 2 ^ P ≤ exp ((t : ℝ) / 2 ^ P) ∧
      exp ((t : ℝ) / 2 ^ P) ≤ ((∑ k ∈ range (N + 1), termUp P t k + remUp P t N : ℤ) : ℝ) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have htr : (0 : ℝ) ≤ t := by exact_mod_cast ht
  set τ := (t : ℝ) / 2 ^ P with hτ
  have hτ0 : 0 ≤ τ := div_nonneg htr hP.le
  have hτ1 : τ ≤ 1 := by rw [hτ, div_le_one hP]; exact ht1
  push_cast
  constructor
  · rw [div_le_iff₀ hP]
    calc ∑ k ∈ range (N + 1), (termDn P t k : ℝ) ≤ ∑ k ∈ range (N + 1), 2 ^ P * τ ^ k / k.factorial :=
          Finset.sum_le_sum fun k _ => termDn_le P t ht k
      _ = (∑ k ∈ range (N + 1), τ ^ k / k.factorial) * 2 ^ P := by
          rw [Finset.sum_mul]
          exact Finset.sum_congr rfl fun k _ => by ring
      _ ≤ exp τ * 2 ^ P := mul_le_mul_of_nonneg_right (Real.sum_le_exp_of_nonneg hτ0 _) hP.le
  · rw [le_div_iff₀ hP]
    have hb := Real.exp_bound' hτ0 hτ1 (n := N + 1) (by omega)
    have hfac : (0 : ℝ) < (N + 1).factorial := by positivity
    have hrem : τ ^ (N + 1) * ((N + 1 : ℕ) + 1 : ℝ) / ((N + 1).factorial * (N + 1 : ℕ)) ≤
        2 * (τ ^ (N + 1) / (N + 1).factorial) := by
      rw [div_le_iff₀ (by positivity)]
      have hp : 0 ≤ τ ^ (N + 1) := pow_nonneg hτ0 _
      have : τ ^ (N + 1) * ((N + 1 : ℕ) + 1 : ℝ) ≤
          2 * τ ^ (N + 1) * ((N + 1 : ℕ) : ℝ) := by
        push_cast
        nlinarith
      calc τ ^ (N + 1) * ((N + 1 : ℕ) + 1 : ℝ) ≤ 2 * τ ^ (N + 1) * ((N + 1 : ℕ) : ℝ) := this
        _ = 2 * (τ ^ (N + 1) / (N + 1).factorial) * ((N + 1).factorial * (N + 1 : ℕ)) := by
            field_simp
    have hR : 2 ^ P * (2 * (τ ^ (N + 1) / (N + 1).factorial)) ≤ (remUp P t N : ℝ) := by
      have h1 := termUp_ge P t ht N
      have hN : (0 : ℝ) < (N : ℝ) + 1 := by positivity
      unfold remUp
      refine le_trans ?_ (Int.le_ceil _)
      rw [le_div_iff₀ hN]
      have h2 : 2 ^ P * τ ^ N / N.factorial * t / 2 ^ P ≤
          ((⌈(termUp P t N : ℝ) * t / 2 ^ P⌉ : ℤ) : ℝ) :=
        le_trans (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right h1 htr) hP.le)
          (Int.le_ceil _)
      have e : 2 ^ P * (2 * (τ ^ (N + 1) / (N + 1).factorial)) * ((N : ℝ) + 1) =
          2 * (2 ^ P * τ ^ N / N.factorial * t / 2 ^ P) := by
        rw [hτ, Nat.factorial_succ]
        push_cast
        have hc : (2 : ℝ) ^ P ≠ 0 := hP.ne'
        have hf : (N.factorial : ℝ) ≠ 0 := by positivity
        generalize (2 : ℝ) ^ P = c at hc ⊢
        simp only [div_pow]
        field_simp
        ring
      rw [e]
      linarith
    have hS : 2 ^ P * ∑ k ∈ range (N + 1), τ ^ k / k.factorial ≤
        ∑ k ∈ range (N + 1), (termUp P t k : ℝ) := by
      rw [Finset.mul_sum]
      exact Finset.sum_le_sum fun k _ => by
        have := termUp_ge P t ht k
        rw [mul_div_assoc] at this
        exact this
    calc exp τ * 2 ^ P ≤ ((∑ m ∈ range (N + 1), τ ^ m / m.factorial) +
          τ ^ (N + 1) * ((N + 1 : ℕ) + 1 : ℝ) / ((N + 1).factorial * (N + 1 : ℕ))) * 2 ^ P := by
          have := hb
          push_cast at this ⊢
          exact mul_le_mul_of_nonneg_right this hP.le
      _ ≤ ((∑ m ∈ range (N + 1), τ ^ m / m.factorial) + 2 * (τ ^ (N + 1) / (N + 1).factorial)) *
            2 ^ P := mul_le_mul_of_nonneg_right (by linarith) hP.le
      _ = 2 ^ P * ∑ m ∈ range (N + 1), τ ^ m / m.factorial +
            2 ^ P * (2 * (τ ^ (N + 1) / (N + 1).factorial)) := by ring
      _ ≤ ∑ k ∈ range (N + 1), (termUp P t k : ℝ) + (remUp P t N : ℝ) := add_le_add hS hR

/-- `s` squarings of an interval. -/
noncomputable def sqIter : ℕ → ℤ × ℤ → ℤ × ℤ
  | 0, p => p
  | s + 1, p => sqIter s (⌊(p.1 : ℝ) ^ 2 / 2 ^ P⌋, ⌈(p.2 : ℝ) ^ 2 / 2 ^ P⌉)

/-- `s` squarings of an enclosure of `exp y` enclose `exp(2^s y)`. -/
theorem sqIter_mem : ∀ (s : ℕ) (p : ℤ × ℤ) (y : ℝ), 0 ≤ p.1 → (p.1 : ℝ) / 2 ^ P ≤ exp y →
    exp y ≤ (p.2 : ℝ) / 2 ^ P →
    ((sqIter P s p).1 : ℝ) / 2 ^ P ≤ exp (2 ^ s * y) ∧ exp (2 ^ s * y) ≤ ((sqIter P s p).2 : ℝ) / 2 ^ P
  | 0, p, y, _, h1, h2 => by simpa [sqIter] using ⟨h1, h2⟩
  | s + 1, p, y, h0, h1, h2 => by
    obtain ⟨l, u⟩ := sq_step P h0 h1 h2
    have e : exp y ^ 2 = exp (2 * y) := by rw [sq, ← Real.exp_add, two_mul]
    rw [e] at l u
    have h0' : 0 ≤ ⌊(p.1 : ℝ) ^ 2 / 2 ^ P⌋ := Int.floor_nonneg.mpr (by positivity)
    have ih := sqIter_mem s (⌊(p.1 : ℝ) ^ 2 / 2 ^ P⌋, ⌈(p.2 : ℝ) ^ 2 / 2 ^ P⌉) (2 * y) h0' l u
    have e2 : (2 : ℝ) ^ s * (2 * y) = 2 ^ (s + 1) * y := by ring
    rw [e2] at ih
    simpa [sqIter] using ih

/-- **(iii) exp for `x ≥ 0`**: with `t_lo ≤ x / 2^s ≤ t_hi` (the halvings) and `t_hi ≤ 2^P`, the
`s`-fold squaring of `(S_N^↓(t_lo), S_N^↑(t_hi) + rem(t_hi))` contains `exp(x 2^-P)`. -/
theorem exp_nonneg_mem (x : ℝ) (s N : ℕ) {tlo thi : ℤ} (hlo0 : 0 ≤ tlo)
    (hlo : (tlo : ℝ) ≤ x / 2 ^ s) (hhi : x / 2 ^ s ≤ (thi : ℝ)) (hhi1 : (thi : ℝ) ≤ 2 ^ P) :
    ((sqIter P s (∑ k ∈ range (N + 1), termDn P tlo k,
        ∑ k ∈ range (N + 1), termUp P thi k + remUp P thi N)).1 : ℝ) / 2 ^ P ≤ exp (x / 2 ^ P) ∧
      exp (x / 2 ^ P) ≤ ((sqIter P s (∑ k ∈ range (N + 1), termDn P tlo k,
        ∑ k ∈ range (N + 1), termUp P thi k + remUp P thi N)).2 : ℝ) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have hthi0 : 0 ≤ thi := by
    have : (0 : ℝ) ≤ thi := (by exact_mod_cast hlo0 : (0 : ℝ) ≤ tlo).trans (hlo.trans hhi)
    exact_mod_cast this
  obtain ⟨l1, -⟩ := exp_point_mem P tlo hlo0 ((hlo.trans hhi).trans hhi1) N
  obtain ⟨-, u1⟩ := exp_point_mem P thi hthi0 hhi1 N
  set y := x / 2 ^ s / 2 ^ P with hy
  have hl : exp ((tlo : ℝ) / 2 ^ P) ≤ exp y :=
    Real.exp_le_exp.mpr (div_le_div_of_nonneg_right hlo hP.le)
  have hu : exp y ≤ exp ((thi : ℝ) / 2 ^ P) :=
    Real.exp_le_exp.mpr (div_le_div_of_nonneg_right hhi hP.le)
  have h0 : 0 ≤ ∑ k ∈ range (N + 1), termDn P tlo k :=
    Finset.sum_nonneg fun k _ => termDn_nonneg P tlo hlo0 k
  have := sqIter_mem P s (∑ k ∈ range (N + 1), termDn P tlo k,
    ∑ k ∈ range (N + 1), termUp P thi k + remUp P thi N) y h0 (l1.trans hl) (hu.trans u1)
  have e : (2 : ℝ) ^ s * y = x / 2 ^ P := by
    rw [hy]
    field_simp
  rwa [e] at this

/-- **(iii) exp for `x < 0`**: from an enclosure `(e_lo, e_hi)`, `e_lo > 0`, of `exp y`, the interval
`(⌊2^{2P} / e_hi⌋, ⌈2^{2P} / e_lo⌉)` contains `exp(-y)`. -/
theorem exp_neg_mem {y : ℝ} {elo ehi : ℤ} (hlo0 : 0 < elo) (h1 : (elo : ℝ) / 2 ^ P ≤ exp y)
    (h2 : exp y ≤ (ehi : ℝ) / 2 ^ P) :
    ((⌊(2 : ℝ) ^ P * 2 ^ P / ehi⌋ : ℤ) : ℝ) / 2 ^ P ≤ exp (-y) ∧
      exp (-y) ≤ ((⌈(2 : ℝ) ^ P * 2 ^ P / elo⌉ : ℤ) : ℝ) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have hl : (0 : ℝ) < elo := by exact_mod_cast hlo0
  have hl' : 0 < (elo : ℝ) / 2 ^ P := div_pos hl hP
  have hh' : 0 < (ehi : ℝ) / 2 ^ P := hl'.trans_le (h1.trans h2)
  have hh : (0 : ℝ) < ehi := by
    have e : (ehi : ℝ) = (ehi / 2 ^ P) * 2 ^ P := (div_mul_cancel₀ _ hP.ne').symm
    rw [e]
    exact mul_pos hh' hP
  have e1 : (2 : ℝ) ^ P * 2 ^ P / ehi / 2 ^ P = ((ehi : ℝ) / 2 ^ P)⁻¹ := by
    field_simp
  have e2 : (2 : ℝ) ^ P * 2 ^ P / elo / 2 ^ P = ((elo : ℝ) / 2 ^ P)⁻¹ := by
    field_simp
  rw [Real.exp_neg]
  constructor
  · calc ((⌊(2 : ℝ) ^ P * 2 ^ P / ehi⌋ : ℤ) : ℝ) / 2 ^ P ≤ (2 : ℝ) ^ P * 2 ^ P / ehi / 2 ^ P :=
          div_le_div_of_nonneg_right (Int.floor_le _) hP.le
      _ = ((ehi : ℝ) / 2 ^ P)⁻¹ := e1
      _ ≤ (exp y)⁻¹ := inv_anti₀ (exp_pos y) h2
  · calc (exp y)⁻¹ ≤ ((elo : ℝ) / 2 ^ P)⁻¹ := inv_anti₀ hl' h1
      _ = (2 : ℝ) ^ P * 2 ^ P / elo / 2 ^ P := e2.symm
      _ ≤ ((⌈(2 : ℝ) ^ P * 2 ^ P / elo⌉ : ℤ) : ℝ) / 2 ^ P :=
          div_le_div_of_nonneg_right (Int.le_ceil _) hP.le

/-! ### (iv) ln: the lower atanh sum -/

/-- `pw_j^↓(u)`: `pw_0 = u`, `pw_{j+1} = ⌊pw_j ⌊u² / 2^P⌋ / 2^P⌋`. -/
noncomputable def pwDn (u : ℤ) : ℕ → ℤ
  | 0 => u
  | j + 1 => ⌊(pwDn u j : ℝ) * ((⌊(u : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) / 2 ^ P⌋

/-- `A^↓(u) = 2 Σ_{j ≤ M} ⌊pw_j / (2j+1)⌋`. -/
noncomputable def lnSeriesDn (u : ℤ) (M : ℕ) : ℤ :=
  2 * ∑ j ∈ range (M + 1), ⌊(pwDn P u j : ℝ) / (2 * (j : ℝ) + 1)⌋

theorem pwDn_le (u : ℤ) (hu : 0 ≤ u) :
    ∀ j, (pwDn P u j : ℝ) ≤ 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * j + 1)
  | 0 => by
    have hP : (0 : ℝ) < 2 ^ P := by positivity
    simp only [pwDn, mul_zero, zero_add, pow_one]
    rw [mul_div_cancel₀ _ hP.ne']
  | j + 1 => by
    have ih := pwDn_le u hu j
    have hP : (0 : ℝ) < 2 ^ P := by positivity
    have hur : (0 : ℝ) ≤ u := by exact_mod_cast hu
    have hq0 : (0 : ℝ) ≤ ((⌊(u : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) := by
      exact_mod_cast Int.floor_nonneg.mpr (by positivity : (0 : ℝ) ≤ (u : ℝ) ^ 2 / 2 ^ P)
    have hq : ((⌊(u : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) ≤ (u : ℝ) ^ 2 / 2 ^ P := Int.floor_le _
    have hb0 : 0 ≤ 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * j + 1) :=
      mul_nonneg hP.le (pow_nonneg (div_nonneg hur hP.le) _)
    simp only [pwDn]
    calc ((⌊(pwDn P u j : ℝ) * ((⌊(u : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) / 2 ^ P⌋ : ℤ) : ℝ)
        ≤ (pwDn P u j : ℝ) * ((⌊(u : ℝ) ^ 2 / 2 ^ P⌋ : ℤ) : ℝ) / 2 ^ P := Int.floor_le _
      _ ≤ 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * j + 1) * ((u : ℝ) ^ 2 / 2 ^ P) / 2 ^ P := by
          apply div_le_div_of_nonneg_right _ hP.le
          exact (mul_le_mul_of_nonneg_right ih hq0).trans (mul_le_mul_of_nonneg_left hq hb0)
      _ = 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * (j + 1) + 1) := by
          have hc : (2 : ℝ) ^ P ≠ 0 := hP.ne'
          generalize (2 : ℝ) ^ P = c at hc ⊢
          simp only [div_pow]
          field_simp
          ring

/-- **(iv), lower end**: `A^↓(u) 2^-P ≤ log(1 + x) - log(1 - x)` for `x = u 2^-P < 1`. -/
theorem lnSeriesDn_le (u : ℤ) (hu : 0 ≤ u) (hx : (u : ℝ) / 2 ^ P < 1) (M : ℕ) :
    (lnSeriesDn P u M : ℝ) / 2 ^ P ≤
      log (1 + (u : ℝ) / 2 ^ P) - log (1 - (u : ℝ) / 2 ^ P) := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have hur : (0 : ℝ) ≤ u := by exact_mod_cast hu
  set x := (u : ℝ) / 2 ^ P with hxdef
  have hx0 : 0 ≤ x := div_nonneg hur hP.le
  have hs := hasSum_log_sub_log_of_abs_lt_one (x := x) (by rw [abs_of_nonneg hx0]; exact hx)
  have hle := sum_le_hasSum (range (M + 1))
    (fun i _ => mul_nonneg (mul_nonneg (by norm_num) (by positivity)) (pow_nonneg hx0 _) :
      ∀ i ∉ range (M + 1), 0 ≤ (2 : ℝ) * (1 / (2 * i + 1)) * x ^ (2 * i + 1))
    hs
  rw [div_le_iff₀ hP]
  unfold lnSeriesDn
  push_cast
  calc 2 * ∑ j ∈ range (M + 1), ((⌊(pwDn P u j : ℝ) / (2 * (j : ℝ) + 1)⌋ : ℤ) : ℝ)
      ≤ 2 * ∑ j ∈ range (M + 1), 2 ^ P * x ^ (2 * j + 1) / (2 * (j : ℝ) + 1) := by
        apply mul_le_mul_of_nonneg_left _ (by norm_num)
        refine Finset.sum_le_sum fun j _ => (Int.floor_le _).trans ?_
        exact div_le_div_of_nonneg_right (pwDn_le P u hu j) (by positivity)
    _ = (∑ j ∈ range (M + 1), (2 : ℝ) * (1 / (2 * j + 1)) * x ^ (2 * j + 1)) * 2 ^ P := by
        rw [Finset.mul_sum, Finset.sum_mul]
        refine Finset.sum_congr rfl fun j _ => ?_
        field_simp
    _ ≤ (log (1 + x) - log (1 - x)) * 2 ^ P := mul_le_mul_of_nonneg_right hle hP.le

/-- The argument reduction of `_ln_point`: `log m = log(1 + x) - log(1 - x)` for
`x = (m - 1)/(m + 1)`, `m > 0`. -/
theorem log_eq_atanh_form {m : ℝ} (hm : 0 < m) :
    log (1 + (m - 1) / (m + 1)) - log (1 - (m - 1) / (m + 1)) = log m := by
  have hm1 : 0 < m + 1 := by linarith
  have e1 : 1 + (m - 1) / (m + 1) = 2 * m / (m + 1) := by
    field_simp
    ring
  have e2 : 1 - (m - 1) / (m + 1) = 2 / (m + 1) := by
    field_simp
    ring
  rw [e1, e2, ← Real.log_div (by positivity) (by positivity)]
  congr 1
  field_simp

end CertifyMathlib.Enclosure.Dyadic
