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

* (iii) halving count: `⌈⌈x⌉ / c⌉ = ⌈x / c⌉` for a positive integer `c` (`ceil_ceil_div`,
  `ceil_ceil_div_div`), so the loop `t ← ⌈t / 2⌉` gives `t_hi = ⌈x / 2^s⌉` (`halveIter_eq`);
* (ii) quotient: for `0 ∉ Y` the corner quotients rounded outward enclose `x / y`
  (`div_mem_corners`, `grid_div_mem`);
* (iv) the upper atanh sum with its tail term `⌈⌈8 pw_{M+1} / 7⌉ / (2M+3)⌉` is at least
  `2^P (log(1 + x) - log(1 - x))` for `x = u 2^-P ≤ 0.35` (`pwUp_ge`, `atanh_tail_le`,
  `lnSeriesUp_ge`); `L2 = (A^↓(⌊2^P/3⌋), A^↑(⌈2^P/3⌉))` contains `2^P ln 2` for `P ≥ 6`
  (`ln2_mem`); the grid arguments `u_lo`, `u_hi` bracket `(m - 1)/(m + 1)` and stay below `0.35`
  (`u_grid_bounds`); and `(A^↓(u_lo), A^↑(u_hi)) + k · L2` contains `2^P log(2^k m)`
  (`ln_point_mem`);
* (v) `to_float_down` and `to_float_up` return the largest binary64 at or below and the smallest at
  or above the grid value, given a correctly rounded nearest value and `nextafter`
  (`to_float_down_spec`, `to_float_up_spec`); `from_float` encloses its argument but has width `0`
  only when `f 2^P` is an integer (`from_fraction_tiny`, `two_pow_neg_1074_grid`).

Not formalized: the bit-length decomposition `v 2^-P = 2^k m` of `_ln_point` (taken as the
hypotheses of `ln_point_mem` and `u_grid_bounds`), and the binary64 facts of (v) (taken as
hypotheses on the set of binary64 values).
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

/-! ### (iii) the halving count -/

/-- `⌈⌈x⌉ / c⌉ = ⌈x / c⌉` for a positive integer `c` (Graham, Knuth, Patashnik (3.11)). -/
theorem ceil_ceil_div (x : ℝ) {c : ℕ} (hc : 0 < c) :
    ⌈((⌈x⌉ : ℤ) : ℝ) / c⌉ = ⌈x / c⌉ := by
  have hc' : (0 : ℝ) < c := by exact_mod_cast hc
  apply le_antisymm
  · rw [Int.ceil_le, div_le_iff₀ hc']
    have h1 : x ≤ ((⌈x / c⌉ : ℤ) : ℝ) * c := by
      have := Int.le_ceil (x / c)
      rwa [div_le_iff₀ hc'] at this
    have h2 : ⌈x⌉ ≤ ⌈x / c⌉ * (c : ℤ) := by
      rw [Int.ceil_le]
      push_cast
      exact h1
    exact_mod_cast h2
  · exact Int.ceil_mono (div_le_div_of_nonneg_right (Int.le_ceil x) hc'.le)

/-- The identity in the form of the document: `⌈⌈a/b⌉/c⌉ = ⌈a/(bc)⌉` for `b > 0` and a positive
integer `c`. -/
theorem ceil_ceil_div_div (a b : ℝ) {c : ℕ} (hc : 0 < c) :
    ⌈((⌈a / b⌉ : ℤ) : ℝ) / c⌉ = ⌈a / (b * c)⌉ := by
  rw [ceil_ceil_div _ hc, div_div]

/-- The halving loop of `_exp_point_nonneg`: `t ← ⌈t / 2⌉`, started at `x`. -/
noncomputable def halveIter (x : ℤ) : ℕ → ℤ
  | 0 => x
  | s + 1 => ⌈((halveIter x s : ℤ) : ℝ) / 2⌉

/-- After `s` halvings the loop variable is `t_hi = ⌈x / 2^s⌉`. -/
theorem halveIter_eq (x : ℤ) : ∀ s, halveIter x s = ⌈(x : ℝ) / 2 ^ s⌉
  | 0 => by simp [halveIter]
  | s + 1 => by
    have h := ceil_ceil_div ((x : ℝ) / 2 ^ s) (c := 2) (by norm_num)
    rw [Nat.cast_ofNat] at h
    rw [halveIter, halveIter_eq x s, h, div_div, ← pow_succ]

/-! ### (ii) the quotient -/

/-- For `0 ∉ [yl, yh]`, the quotient of members lies between the extreme corner quotients. -/
theorem div_mem_corners {x y xl xh yl yh : ℝ} (hxl : xl ≤ x) (hxh : x ≤ xh) (hyl : yl ≤ y)
    (hyh : y ≤ yh) (h0 : 0 < yl ∨ yh < 0) :
    min (min (xl / yl) (xl / yh)) (min (xh / yl) (xh / yh)) ≤ x / y ∧
      x / y ≤ max (max (xl / yl) (xl / yh)) (max (xh / yl) (xh / yh)) := by
  have hinv : 1 / yh ≤ 1 / y ∧ 1 / y ≤ 1 / yl := by
    rcases h0 with h | h
    · have hy : 0 < y := lt_of_lt_of_le h hyl
      exact ⟨one_div_le_one_div_of_le hy hyh, one_div_le_one_div_of_le h hyl⟩
    · have hy : y < 0 := lt_of_le_of_lt hyh h
      exact ⟨one_div_le_one_div_of_neg_of_le h hyh, one_div_le_one_div_of_neg_of_le hy hyl⟩
  obtain ⟨c1, c2⟩ := mul_mem_corners hxl hxh hinv.1 hinv.2
  simp only [mul_one_div] at c1 c2
  rw [min_comm (xl / yh), min_comm (xh / yh)] at c1
  rw [max_comm (xl / yh), max_comm (xh / yh)] at c2
  exact ⟨c1, c2⟩

/-- (ii) **Grid quotient** (`__truediv__`): for `0 ∉ Y`, the interval
`(min ⌊x_i 2^P / y_j⌋, max ⌈x_i 2^P / y_j⌉)` over the four corners contains `x / y` for
`x ∈ [xl, xh] 2^-P`, `y ∈ [yl, yh] 2^-P`. -/
theorem grid_div_mem {x y : ℝ} {xl xh yl yh : ℤ} (hxl : (xl : ℝ) / 2 ^ P ≤ x)
    (hxh : x ≤ (xh : ℝ) / 2 ^ P) (hyl : (yl : ℝ) / 2 ^ P ≤ y) (hyh : y ≤ (yh : ℝ) / 2 ^ P)
    (h0 : 0 < yl ∨ yh < 0) :
    ((min (min ⌊(xl : ℝ) * 2 ^ P / yl⌋ ⌊(xl : ℝ) * 2 ^ P / yh⌋)
        (min ⌊(xh : ℝ) * 2 ^ P / yl⌋ ⌊(xh : ℝ) * 2 ^ P / yh⌋) : ℤ) : ℝ) / 2 ^ P ≤ x / y ∧
      x / y ≤ ((max (max ⌈(xl : ℝ) * 2 ^ P / yl⌉ ⌈(xl : ℝ) * 2 ^ P / yh⌉)
        (max ⌈(xh : ℝ) * 2 ^ P / yl⌉ ⌈(xh : ℝ) * 2 ^ P / yh⌉) : ℤ) : ℝ) / 2 ^ P := by
  have h2 : (0 : ℝ) < 2 ^ P := by positivity
  have hx1 : (xl : ℝ) ≤ x * 2 ^ P := by rwa [div_le_iff₀ h2] at hxl
  have hx2 : x * 2 ^ P ≤ (xh : ℝ) := by rwa [le_div_iff₀ h2] at hxh
  have hy1 : (yl : ℝ) ≤ y * 2 ^ P := by rwa [div_le_iff₀ h2] at hyl
  have hy2 : y * 2 ^ P ≤ (yh : ℝ) := by rwa [le_div_iff₀ h2] at hyh
  have h0' : (0 : ℝ) < yl ∨ (yh : ℝ) < 0 := by
    rcases h0 with h | h
    · exact Or.inl (by exact_mod_cast h)
    · exact Or.inr (by exact_mod_cast h)
  obtain ⟨c1, c2⟩ := div_mem_corners hx1 hx2 hy1 hy2 h0'
  have e : x * 2 ^ P / (y * 2 ^ P) = x / y := by
    rw [mul_div_mul_right _ _ h2.ne']
  rw [e] at c1 c2
  have fl : ∀ a b : ℤ, ((⌊(a : ℝ) * 2 ^ P / b⌋ : ℤ) : ℝ) / 2 ^ P ≤ (a : ℝ) / b := fun a b => by
    rw [div_le_iff₀ h2]
    refine (Int.floor_le _).trans (le_of_eq ?_)
    ring
  have cl : ∀ a b : ℤ, (a : ℝ) / b ≤ ((⌈(a : ℝ) * 2 ^ P / b⌉ : ℤ) : ℝ) / 2 ^ P := fun a b => by
    rw [le_div_iff₀ h2]
    refine le_trans (le_of_eq ?_) (Int.le_ceil _)
    ring
  push_cast
  constructor
  · rw [← min_div_div_right h2.le, ← min_div_div_right h2.le, ← min_div_div_right h2.le]
    exact le_trans (min_le_min (min_le_min (fl _ _) (fl _ _)) (min_le_min (fl _ _) (fl _ _))) c1
  · rw [← max_div_div_right h2.le, ← max_div_div_right h2.le, ← max_div_div_right h2.le]
    exact le_trans c2 (max_le_max (max_le_max (cl _ _) (cl _ _)) (max_le_max (cl _ _) (cl _ _)))

/-! ### (iv) ln: the upper atanh sum with its tail term -/

/-- `pw_j^↑(u)`: `pw_0 = u`, `pw_{j+1} = ⌈pw_j ⌈u² / 2^P⌉ / 2^P⌉`. -/
noncomputable def pwUp (u : ℤ) : ℕ → ℤ
  | 0 => u
  | j + 1 => ⌈(pwUp u j : ℝ) * ((⌈(u : ℝ) ^ 2 / 2 ^ P⌉ : ℤ) : ℝ) / 2 ^ P⌉

/-- `A^↑(u) = 2 (Σ_{j ≤ M} ⌈pw_j / (2j+1)⌉ + ⌈⌈8 pw_{M+1} / 7⌉ / (2M+3)⌉)`. -/
noncomputable def lnSeriesUp (u : ℤ) (M : ℕ) : ℤ :=
  2 * (∑ j ∈ range (M + 1), ⌈(pwUp P u j : ℝ) / (2 * (j : ℝ) + 1)⌉ +
    ⌈((⌈8 * (pwUp P u (M + 1) : ℝ) / 7⌉ : ℤ) : ℝ) / (2 * (M : ℝ) + 3)⌉)

theorem pwUp_ge (u : ℤ) (hu : 0 ≤ u) :
    ∀ j, 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * j + 1) ≤ (pwUp P u j : ℝ)
  | 0 => by
    have hP : (0 : ℝ) < 2 ^ P := by positivity
    simp only [pwUp, mul_zero, zero_add, pow_one]
    rw [mul_div_cancel₀ _ hP.ne']
  | j + 1 => by
    have ih := pwUp_ge u hu j
    have hP : (0 : ℝ) < 2 ^ P := by positivity
    have hur : (0 : ℝ) ≤ u := by exact_mod_cast hu
    have hq : (u : ℝ) ^ 2 / 2 ^ P ≤ ((⌈(u : ℝ) ^ 2 / 2 ^ P⌉ : ℤ) : ℝ) := Int.le_ceil _
    have hq0 : (0 : ℝ) ≤ (u : ℝ) ^ 2 / 2 ^ P := by positivity
    have hb0 : 0 ≤ 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * j + 1) :=
      mul_nonneg hP.le (pow_nonneg (div_nonneg hur hP.le) _)
    have hpw0 : (0 : ℝ) ≤ (pwUp P u j : ℝ) := hb0.trans ih
    simp only [pwUp]
    calc 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * (j + 1) + 1)
        = 2 ^ P * ((u : ℝ) / 2 ^ P) ^ (2 * j + 1) * ((u : ℝ) ^ 2 / 2 ^ P) / 2 ^ P := by
          have hc : (2 : ℝ) ^ P ≠ 0 := hP.ne'
          generalize (2 : ℝ) ^ P = c at hc ⊢
          simp only [div_pow]
          field_simp
          ring
      _ ≤ (pwUp P u j : ℝ) * ((⌈(u : ℝ) ^ 2 / 2 ^ P⌉ : ℤ) : ℝ) / 2 ^ P := by
          apply div_le_div_of_nonneg_right _ hP.le
          exact (mul_le_mul_of_nonneg_left hq hb0).trans (mul_le_mul_of_nonneg_right ih
            (hq0.trans hq))
      _ ≤ ((⌈(pwUp P u j : ℝ) * ((⌈(u : ℝ) ^ 2 / 2 ^ P⌉ : ℤ) : ℝ) / 2 ^ P⌉ : ℤ) : ℝ) :=
          Int.le_ceil _

/-- The tail of the atanh series: for `0 ≤ x` with `x² ≤ 1/8`,
`log(1+x) - log(1-x) ≤ Σ_{j ≤ M} 2 x^{2j+1}/(2j+1) + 2 (8/7) x^{2M+3}/(2M+3)`. -/
theorem atanh_tail_le {x : ℝ} (hx0 : 0 ≤ x) (hx2 : x ^ 2 ≤ 1 / 8) (M : ℕ) :
    log (1 + x) - log (1 - x) ≤
      ∑ j ∈ range (M + 1), (2 : ℝ) * (1 / (2 * j + 1)) * x ^ (2 * j + 1) +
        2 * (8 / 7) * x ^ (2 * M + 3) / (2 * M + 3) := by
  have hx1 : x < 1 := by nlinarith
  have hs := hasSum_log_sub_log_of_abs_lt_one (x := x) (by rw [abs_of_nonneg hx0]; exact hx1)
  have ht := (hasSum_nat_add_iff' (M + 1)).mpr hs
  have hq0 : 0 ≤ x ^ 2 := sq_nonneg x
  have hq1 : x ^ 2 < 1 := by linarith
  have hg := (hasSum_geometric_of_lt_one hq0 hq1).mul_left
    (2 * (1 / (2 * (M : ℝ) + 3)) * x ^ (2 * M + 3))
  have hle := hasSum_le (fun n => ?_) ht hg
  · have hinv : (1 - x ^ 2)⁻¹ ≤ 8 / 7 := by
      rw [inv_le_comm₀ (by linarith) (by norm_num)]
      linarith
    have hM : (0 : ℝ) < 2 * M + 3 := by positivity
    have hp : 0 ≤ 2 * (1 / (2 * (M : ℝ) + 3)) * x ^ (2 * M + 3) := by positivity
    have h3 := mul_le_mul_of_nonneg_left hinv hp
    have e : 2 * (1 / (2 * (M : ℝ) + 3)) * x ^ (2 * M + 3) * (8 / 7) =
        2 * (8 / 7) * x ^ (2 * M + 3) / (2 * M + 3) := by
      field_simp
    linarith
  · have hM : (0 : ℝ) < 2 * M + 3 := by positivity
    have e1 : x ^ (2 * (n + (M + 1)) + 1) = x ^ (2 * M + 3) * (x ^ 2) ^ n := by
      rw [← pow_mul, ← pow_add]
      congr 1
      ring
    have e2 : (2 * ((n + (M + 1) : ℕ) : ℝ) + 1) = 2 * n + (2 * M + 3) := by
      push_cast
      ring
    simp only [e1, e2]
    have hd : 1 / (2 * (n : ℝ) + (2 * M + 3)) ≤ 1 / (2 * M + 3) :=
      one_div_le_one_div_of_le hM (by linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)])
    have hp : 0 ≤ x ^ (2 * M + 3) * (x ^ 2) ^ n := by positivity
    calc 2 * (1 / (2 * (n : ℝ) + (2 * M + 3))) * (x ^ (2 * M + 3) * (x ^ 2) ^ n)
        ≤ 2 * (1 / (2 * (M : ℝ) + 3)) * (x ^ (2 * M + 3) * (x ^ 2) ^ n) := by
          apply mul_le_mul_of_nonneg_right _ hp
          linarith
      _ = 2 * (1 / (2 * (M : ℝ) + 3)) * x ^ (2 * M + 3) * (x ^ 2) ^ n := by ring

/-- **(iv), upper end**: `log(1 + x) - log(1 - x) ≤ A^↑(u) 2^-P` for `x = u 2^-P ≤ 0.35`. -/
theorem lnSeriesUp_ge (u : ℤ) (hu : 0 ≤ u) (hx : (u : ℝ) / 2 ^ P ≤ 7 / 20) (M : ℕ) :
    log (1 + (u : ℝ) / 2 ^ P) - log (1 - (u : ℝ) / 2 ^ P) ≤
      (lnSeriesUp P u M : ℝ) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have hur : (0 : ℝ) ≤ u := by exact_mod_cast hu
  set x := (u : ℝ) / 2 ^ P with hxdef
  have hx0 : 0 ≤ x := div_nonneg hur hP.le
  have hx2 : x ^ 2 ≤ 1 / 8 := by nlinarith
  have ht := atanh_tail_le hx0 hx2 M
  have hM : (0 : ℝ) < 2 * M + 3 := by positivity
  rw [le_div_iff₀ hP]
  unfold lnSeriesUp
  push_cast
  refine (mul_le_mul_of_nonneg_right ht hP.le).trans ?_
  rw [add_mul, Finset.sum_mul, mul_add, Finset.mul_sum]
  apply add_le_add
  · refine Finset.sum_le_sum fun j _ => ?_
    have hj : (0 : ℝ) < 2 * (j : ℝ) + 1 := by positivity
    have h1 := pwUp_ge P u hu j
    have h2 : (pwUp P u j : ℝ) / (2 * (j : ℝ) + 1) ≤
        ((⌈(pwUp P u j : ℝ) / (2 * (j : ℝ) + 1)⌉ : ℤ) : ℝ) := Int.le_ceil _
    have h3 : 2 ^ P * x ^ (2 * j + 1) / (2 * (j : ℝ) + 1) ≤
        (pwUp P u j : ℝ) / (2 * (j : ℝ) + 1) := div_le_div_of_nonneg_right h1 hj.le
    have e : 2 * (1 / (2 * (j : ℝ) + 1)) * x ^ (2 * j + 1) * 2 ^ P =
        2 * (2 ^ P * x ^ (2 * j + 1) / (2 * (j : ℝ) + 1)) := by
      field_simp
    rw [e]
    linarith
  · have h1 := pwUp_ge P u hu (M + 1)
    have e1 : 2 * (M + 1) + 1 = 2 * M + 3 := by ring
    rw [e1] at h1
    have h2 : 8 * (pwUp P u (M + 1) : ℝ) / 7 ≤
        ((⌈8 * (pwUp P u (M + 1) : ℝ) / 7⌉ : ℤ) : ℝ) := Int.le_ceil _
    have h3 : ((⌈8 * (pwUp P u (M + 1) : ℝ) / 7⌉ : ℤ) : ℝ) / (2 * (M : ℝ) + 3) ≤
        ((⌈((⌈8 * (pwUp P u (M + 1) : ℝ) / 7⌉ : ℤ) : ℝ) / (2 * (M : ℝ) + 3)⌉ : ℤ) : ℝ) :=
      Int.le_ceil _
    have h4 : 8 * (2 ^ P * x ^ (2 * M + 3)) / 7 / (2 * (M : ℝ) + 3) ≤
        ((⌈8 * (pwUp P u (M + 1) : ℝ) / 7⌉ : ℤ) : ℝ) / (2 * (M : ℝ) + 3) := by
      apply div_le_div_of_nonneg_right _ hM.le
      linarith
    have e : 2 * (8 / 7) * x ^ (2 * M + 3) / (2 * (M : ℝ) + 3) * 2 ^ P =
        2 * (8 * (2 ^ P * x ^ (2 * M + 3)) / 7 / (2 * (M : ℝ) + 3)) := by
      field_simp
    rw [e]
    linarith

/-- `log(1 + x) - log(1 - x)` is monotone on `[0, 1)`. -/
theorem atanh_form_mono {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) (hb : b < 1) :
    log (1 + a) - log (1 - a) ≤ log (1 + b) - log (1 - b) := by
  have h1 : log (1 + a) ≤ log (1 + b) := log_le_log (by linarith) (by linarith)
  have h2 : log (1 - b) ≤ log (1 - a) := log_le_log (by linarith) (by linarith)
  linarith

theorem log_two_eq_atanh_form :
    log (1 + 1 / 3) - log (1 - 1 / 3) = log 2 := by
  have h := log_eq_atanh_form (m := (2 : ℝ)) two_pos
  have e : ((2 : ℝ) - 1) / (2 + 1) = 1 / 3 := by norm_num
  rwa [e] at h

/-- **`ln 2` enclosure** (`ln2_interval`): `L2 = (A^↓(⌊2^P/3⌋), A^↑(⌈2^P/3⌉))` contains
`2^P ln 2` for `P ≥ 6` (the code has `P = 192`). -/
theorem ln2_mem (hP6 : 6 ≤ P) (M : ℕ) :
    (lnSeriesDn P ⌊(2 : ℝ) ^ P / 3⌋ M : ℝ) / 2 ^ P ≤ log 2 ∧
      log 2 ≤ (lnSeriesUp P ⌈(2 : ℝ) ^ P / 3⌉ M : ℝ) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have h64 : (64 : ℝ) ≤ 2 ^ P := by
    have : (2 : ℝ) ^ 6 ≤ 2 ^ P := pow_le_pow_right₀ (by norm_num) hP6
    norm_num at this
    exact this
  constructor
  · have hu0 : 0 ≤ ⌊(2 : ℝ) ^ P / 3⌋ := Int.floor_nonneg.mpr (by positivity)
    have hfl : ((⌊(2 : ℝ) ^ P / 3⌋ : ℤ) : ℝ) ≤ 2 ^ P / 3 := Int.floor_le _
    have hx3 : ((⌊(2 : ℝ) ^ P / 3⌋ : ℤ) : ℝ) / 2 ^ P ≤ 1 / 3 := by
      rw [div_le_iff₀ hP]
      linarith
    have hx0 : (0 : ℝ) ≤ ((⌊(2 : ℝ) ^ P / 3⌋ : ℤ) : ℝ) / 2 ^ P :=
      div_nonneg (by exact_mod_cast hu0) hP.le
    refine (lnSeriesDn_le P _ hu0 (by linarith) M).trans ?_
    rw [← log_two_eq_atanh_form]
    exact atanh_form_mono hx0 hx3 (by norm_num)
  · have hu0 : 0 ≤ ⌈(2 : ℝ) ^ P / 3⌉ := Int.ceil_nonneg (by positivity)
    have hcl : (2 : ℝ) ^ P / 3 ≤ ((⌈(2 : ℝ) ^ P / 3⌉ : ℤ) : ℝ) := Int.le_ceil _
    have hcu : ((⌈(2 : ℝ) ^ P / 3⌉ : ℤ) : ℝ) < 2 ^ P / 3 + 1 := Int.ceil_lt_add_one _
    have hx3 : 1 / 3 ≤ ((⌈(2 : ℝ) ^ P / 3⌉ : ℤ) : ℝ) / 2 ^ P := by
      rw [le_div_iff₀ hP]
      linarith
    have hx35 : ((⌈(2 : ℝ) ^ P / 3⌉ : ℤ) : ℝ) / 2 ^ P ≤ 7 / 20 := by
      rw [div_le_iff₀ hP]
      linarith
    refine le_trans ?_ (lnSeriesUp_ge P _ hu0 hx35 M)
    rw [← log_two_eq_atanh_form]
    exact atanh_form_mono (by norm_num) hx3 (by linarith)

/-! ### (iv) assembly of `_ln_point` -/

/-- The grid argument `u`: from `m_lo ≤ m 2^P ≤ m_hi` (with `m ≥ 1`, `0 < m_lo`, `m_hi ≤ 2·2^P`),
`u_lo = max(0, ⌊(m_lo - 2^P) 2^P / (m_lo + 2^P)⌋)` and
`u_hi = max(0, ⌈(m_hi - 2^P) 2^P / (m_hi + 2^P)⌉)` bracket `2^P (m - 1)/(m + 1)`, and
`u_hi ≤ 0.35 · 2^P` for `P ≥ 6` (the runtime check `U_MAX` of `_atanh2_series`). -/
theorem u_grid_bounds (hP6 : 6 ≤ P) {m : ℝ} {mlo mhi : ℤ} (hm : 1 ≤ m) (hmlo0 : 0 < mlo)
    (hmlo : (mlo : ℝ) ≤ m * 2 ^ P) (hmhi : m * 2 ^ P ≤ mhi) (hmhi2 : (mhi : ℝ) ≤ 2 * 2 ^ P) :
    ((max ⌊((mlo : ℝ) - 2 ^ P) * 2 ^ P / (mlo + 2 ^ P)⌋ 0 : ℤ) : ℝ) / 2 ^ P ≤ (m - 1) / (m + 1) ∧
      (m - 1) / (m + 1) ≤ ((max ⌈((mhi : ℝ) - 2 ^ P) * 2 ^ P / (mhi + 2 ^ P)⌉ 0 : ℤ) : ℝ) / 2 ^ P ∧
      ((max ⌈((mhi : ℝ) - 2 ^ P) * 2 ^ P / (mhi + 2 ^ P)⌉ 0 : ℤ) : ℝ) / 2 ^ P ≤ 7 / 20 := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have h64 : (64 : ℝ) ≤ 2 ^ P := by
    have : (2 : ℝ) ^ 6 ≤ 2 ^ P := pow_le_pow_right₀ (by norm_num) hP6
    norm_num at this
    exact this
  have hmlo0' : (0 : ℝ) < mlo := by exact_mod_cast hmlo0
  have hm1 : 0 < m + 1 := by linarith
  have hw0 : 0 ≤ (m - 1) / (m + 1) := div_nonneg (by linarith) hm1.le
  have hmhi0 : (0 : ℝ) < mhi := by nlinarith
  refine ⟨?_, ?_, ?_⟩
  · push_cast
    rw [← max_div_div_right hP.le]
    refine max_le ?_ (by simpa using hw0)
    rw [div_le_iff₀ hP]
    refine (Int.floor_le _).trans ?_
    rw [div_le_iff₀ (by positivity : (0 : ℝ) < mlo + 2 ^ P)]
    have e : (m - 1) / (m + 1) * 2 ^ P * (mlo + 2 ^ P) =
        (m - 1) * 2 ^ P * (mlo + 2 ^ P) / (m + 1) := by ring
    rw [e, le_div_iff₀ hm1]
    nlinarith [mul_le_mul_of_nonneg_left hmlo hP.le]
  · push_cast
    rw [← max_div_div_right hP.le]
    refine le_max_of_le_left ?_
    rw [le_div_iff₀ hP]
    refine le_trans ?_ (Int.le_ceil _)
    rw [le_div_iff₀ (by positivity : (0 : ℝ) < mhi + 2 ^ P)]
    have e : (m - 1) / (m + 1) * 2 ^ P * (mhi + 2 ^ P) =
        (m - 1) * 2 ^ P * (mhi + 2 ^ P) / (m + 1) := by ring
    rw [e, div_le_iff₀ hm1]
    nlinarith [mul_le_mul_of_nonneg_left hmhi hP.le]
  · push_cast
    rw [← max_div_div_right hP.le]
    refine max_le ?_ (by rw [zero_div]; norm_num)
    rw [div_le_iff₀ hP]
    have hc := Int.ceil_lt_add_one (((mhi : ℝ) - 2 ^ P) * 2 ^ P / (mhi + 2 ^ P))
    have hq : ((mhi : ℝ) - 2 ^ P) * 2 ^ P / (mhi + 2 ^ P) ≤ 2 ^ P / 3 := by
      rw [div_le_iff₀ (by positivity)]
      nlinarith
    linarith

/-- **(iv) `_ln_point`.** Let `z = 2^k m` with `1 ≤ m` (the code has `m < 2`), let `0 ≤ u_lo`, `u_hi 2^-P ≤ 0.35` bracket
`u(m) = (m - 1)/(m + 1)` (`u_grid_bounds`), and let `L2 = (A^↓(⌊2^P/3⌋), A^↑(⌈2^P/3⌉))`. Then
`2^P log z` lies in `(A^↓(u_lo), A^↑(u_hi)) + k · L2`, where `k · L2` is `scale_int` (ends swapped
for `k < 0`). -/
theorem ln_point_mem (hP6 : 6 ≤ P) (M : ℕ) {z m : ℝ} (k : ℤ) (hm1 : 1 ≤ m)
    (hz : z = (2 : ℝ) ^ k * m) {ulo uhi : ℤ} (hulo0 : 0 ≤ ulo) (huhi : (uhi : ℝ) / 2 ^ P ≤ 7 / 20)
    (hlo : (ulo : ℝ) / 2 ^ P ≤ (m - 1) / (m + 1)) (hhi : (m - 1) / (m + 1) ≤ (uhi : ℝ) / 2 ^ P) :
    ((lnSeriesDn P ulo M : ℝ) + min (k * lnSeriesDn P ⌊(2 : ℝ) ^ P / 3⌋ M)
        (k * lnSeriesUp P ⌈(2 : ℝ) ^ P / 3⌉ M)) / 2 ^ P ≤ log z ∧
      log z ≤ ((lnSeriesUp P uhi M : ℝ) + max (k * lnSeriesDn P ⌊(2 : ℝ) ^ P / 3⌋ M)
        (k * lnSeriesUp P ⌈(2 : ℝ) ^ P / 3⌉ M)) / 2 ^ P := by
  have hP : (0 : ℝ) < 2 ^ P := by positivity
  have hm0 : 0 < m := by linarith
  have hlogz : log z = k * log 2 + log m := by
    rw [hz, Real.log_mul (zpow_ne_zero _ two_ne_zero) hm0.ne', Real.log_zpow]
  have hw := log_eq_atanh_form hm0
  have hw0 : 0 ≤ (m - 1) / (m + 1) := div_nonneg (by linarith) (by linarith)
  have hw1 : (m - 1) / (m + 1) < 1 := by rw [div_lt_one (by linarith)]; linarith
  have hulo0' : (0 : ℝ) ≤ (ulo : ℝ) / 2 ^ P := div_nonneg (by exact_mod_cast hulo0) hP.le
  have huhi0 : (0 : ℝ) ≤ uhi := by
    have := hw0.trans hhi
    rwa [le_div_iff₀ hP, zero_mul] at this
  obtain ⟨l2lo, l2hi⟩ := ln2_mem P hP6 M
  have hA1 := (lnSeriesDn_le P ulo hulo0 (by linarith) M).trans
    (atanh_form_mono hulo0' hlo hw1)
  have hA2 := (atanh_form_mono hw0 hhi (by linarith)).trans
    (lnSeriesUp_ge P uhi (by exact_mod_cast huhi0) huhi M)
  rw [hw] at hA1 hA2
  set a := (lnSeriesDn P ⌊(2 : ℝ) ^ P / 3⌋ M : ℝ) with ha
  set b := (lnSeriesUp P ⌈(2 : ℝ) ^ P / 3⌉ M : ℝ) with hb
  have hk : min ((k : ℝ) * a) (k * b) / 2 ^ P ≤ k * log 2 ∧
      (k : ℝ) * log 2 ≤ max ((k : ℝ) * a) (k * b) / 2 ^ P := by
    rw [div_le_iff₀ hP, le_div_iff₀ hP]
    rw [div_le_iff₀ hP] at l2lo
    rw [le_div_iff₀ hP] at l2hi
    rcases le_total 0 (k : ℝ) with h | h
    · constructor
      · refine (min_le_left _ _).trans ?_
        nlinarith
      · refine le_trans ?_ (le_max_right _ _)
        nlinarith
    · constructor
      · refine (min_le_right _ _).trans ?_
        nlinarith
      · refine le_trans ?_ (le_max_left _ _)
        nlinarith
  rw [hlogz, add_div, add_div]
  push_cast
  rw [← ha, ← hb]
  constructor <;> linarith [hk.1, hk.2]

/-! ### (v) conversion to binary64 -/

/-- (v) `to_float_down`. Let `F` be the set of finite binary64 values, `f ∈ F` a nearest element to
`q` (`float(Fraction)` is correctly rounded), and, when `f > q`, `fm = nextafter(f, -∞) ∈ F`, the
element of `F` below `f` (`fm < f`, nothing of `F` in between). The returned value
`if f ≤ q then f else fm` is the largest element of `F` at or below `q`. -/
theorem to_float_down_spec (F : Set ℝ) {q f fm : ℝ} (hf : f ∈ F)
    (hnear : ∀ g ∈ F, |f - q| ≤ |g - q|) (hfm : q < f → fm ∈ F ∧ fm < f ∧ ∀ g ∈ F, g < f → g ≤ fm) :
    (if f ≤ q then f else fm) ∈ F ∧ (if f ≤ q then f else fm) ≤ q ∧
      ∀ g ∈ F, g ≤ q → g ≤ (if f ≤ q then f else fm) := by
  by_cases h : f ≤ q
  · rw [ite_eq_left h]
    refine ⟨hf, h, fun g hg hgq => ?_⟩
    by_contra hlt'
    have hlt := not_le.mp hlt'
    have := hnear g hg
    rw [abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)] at this
    linarith
  · rw [ite_eq_right h]
    have h' := not_le.mp h
    obtain ⟨hfmF, hfmlt, hgap⟩ := hfm h'
    refine ⟨hfmF, ?_, fun g hg hgq => hgap g hg (by linarith)⟩
    by_contra hlt'
    have hlt := not_le.mp hlt'
    have := hnear fm hfmF
    rw [abs_of_pos (by linarith), abs_of_pos (by linarith)] at this
    linarith

/-- (v) `to_float_up`, the mirror image: the returned value is the smallest element of `F` at
or above `q`. -/
theorem to_float_up_spec (F : Set ℝ) {q f fp : ℝ} (hf : f ∈ F)
    (hnear : ∀ g ∈ F, |f - q| ≤ |g - q|) (hfp : f < q → fp ∈ F ∧ f < fp ∧ ∀ g ∈ F, f < g → fp ≤ g) :
    (if q ≤ f then f else fp) ∈ F ∧ q ≤ (if q ≤ f then f else fp) ∧
      ∀ g ∈ F, q ≤ g → (if q ≤ f then f else fp) ≤ g := by
  by_cases h : q ≤ f
  · rw [ite_eq_left h]
    refine ⟨hf, h, fun g hg hgq => ?_⟩
    by_contra hlt'
    have hlt := not_le.mp hlt'
    have := hnear g hg
    rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)] at this
    linarith
  · rw [ite_eq_right h]
    have h' := not_le.mp h
    obtain ⟨hfpF, hfplt, hgap⟩ := hfp h'
    refine ⟨hfpF, ?_, fun g hg hgq => hgap g hg (by linarith)⟩
    by_contra hlt'
    have hlt := not_le.mp hlt'
    have := hnear fp hfpF
    rw [abs_of_neg (by linarith), abs_of_neg (by linarith)] at this
    linarith

/-- `from_float` encloses its argument but is exact (width 0) only when `f 2^P` is an integer:
for `0 < f 2^P < 1` (for example the binary64 value `2^-1074` with `P = 192`) it returns `(0, 1)`. -/
theorem from_fraction_tiny {f : ℝ} (h0 : 0 < f * 2 ^ P) (h1 : f * 2 ^ P < 1) :
    ⌊f * 2 ^ P⌋ = 0 ∧ ⌈f * 2 ^ P⌉ = 1 := by
  constructor
  · rw [Int.floor_eq_iff]
    push_cast
    constructor <;> linarith
  · rw [Int.ceil_eq_iff]
    push_cast
    constructor <;> linarith

theorem two_pow_neg_1074_grid : 0 < (2 : ℝ) ^ (-1074 : ℤ) * 2 ^ (192 : ℕ) ∧
    (2 : ℝ) ^ (-1074 : ℤ) * 2 ^ (192 : ℕ) < 1 := by
  have e : (2 : ℝ) ^ (-1074 : ℤ) * 2 ^ (192 : ℕ) = (2 : ℝ) ^ (-882 : ℤ) := by
    rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]
    norm_num
  rw [e]
  exact ⟨zpow_pos two_pos _, zpow_lt_one_of_neg₀ (by norm_num) (by norm_num)⟩

end CertifyMathlib.Enclosure.Dyadic
