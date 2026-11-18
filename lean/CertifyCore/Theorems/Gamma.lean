/-
Kernel-checked facts about the rounding constant gamma_k = k u / (1 - k u) with the binary64 unit roundoff
u = 2^-53 (`certify_ed/fp.py`: `U`, `gamma`), as used by the Theorem 1 accounting
(`docs/proofs/thm1_fp-accounting.md`, lemmas Lambda1 to Lambda3; Higham 2002, Lemmas 3.1 and 3.3).
Core Lean only; the rationals are `Rat` of core.
-/
namespace CertifyCore.Thm

/-- The unit roundoff of binary64, exactly. -/
def u : Rat := 1 / 2 ^ 53

/-- `fp.gamma(k)` evaluated exactly: k u / (1 - k u). -/
def gamma (k : Nat) : Rat := (k : Rat) * u / (1 - (k : Rat) * u)

theorem div_le_div_of {a b c d : Rat} (hb : 0 < b) (hd : 0 < d) (h : a * d ≤ c * b) : a / b ≤ c / d := by
  have hb' : b ≠ 0 := Rat.ne_of_gt hb
  have hd' : d ≠ 0 := Rat.ne_of_gt hd
  have ib : b * b⁻¹ = 1 := Rat.mul_inv_cancel b hb'
  have id : d * d⁻¹ = 1 := Rat.mul_inv_cancel d hd'
  have pbd : 0 ≤ b⁻¹ * d⁻¹ := Rat.le_of_lt (Rat.mul_pos (Rat.inv_pos.mpr hb) (Rat.inv_pos.mpr hd))
  have e1 : a / b = (a * d) * (b⁻¹ * d⁻¹) := by rw [Rat.div_def]; grind
  have e2 : c / d = (c * b) * (b⁻¹ * d⁻¹) := by rw [Rat.div_def]; grind
  rw [e1, e2]
  exact Rat.mul_le_mul_of_nonneg_right h pbd

theorem le_div_of {a c d : Rat} (hd : 0 < d) (h : a * d ≤ c) : a ≤ c / d := by
  have : a / 1 ≤ c / d := div_le_div_of (by decide) hd (by grind)
  grind

theorem div_le_of {a b c : Rat} (hb : 0 < b) (h : a ≤ c * b) : a / b ≤ c := by
  have : a / b ≤ c / 1 := div_le_div_of hb (by decide) (by grind)
  grind

theorem two53_pos : (0 : Rat) < 2 ^ 53 := by decide +kernel

/-- gamma_k = k / (2^53 - k): the form used in the proofs. -/
theorem gamma_eq (k : Nat) (hk : (k : Rat) < 2 ^ 53) : gamma k = (k : Rat) / (2 ^ 53 - k) := by
  unfold gamma u
  have h1 : (0 : Rat) < 2 ^ 53 - k := by grind
  have h2 : (2 : Rat) ^ 53 ≠ 0 := Rat.ne_of_gt two53_pos
  grind

theorem cast_nonneg (k : Nat) : (0 : Rat) ≤ k := by exact_mod_cast Nat.zero_le k

theorem gamma_nonneg (k : Nat) (hk : (k : Rat) < 2 ^ 53) : 0 ≤ gamma k := by
  rw [gamma_eq k hk]
  have h1 : (0 : Rat) < 2 ^ 53 - k := by grind
  have := le_div_of h1 (show (0 : Rat) * (2 ^ 53 - k) ≤ k by grind [cast_nonneg k])
  exact this

/-- Monotonicity: m <= n implies gamma_m <= gamma_n. -/
theorem gamma_mono (m n : Nat) (hmn : m ≤ n) (hn : (n : Rat) < 2 ^ 53) : gamma m ≤ gamma n := by
  have hmn' : (m : Rat) ≤ n := by exact_mod_cast hmn
  have hm : (m : Rat) < 2 ^ 53 := by grind
  rw [gamma_eq m hm, gamma_eq n hn]
  apply div_le_div_of (by grind) (by grind)
  have : (m : Rat) * 2 ^ 53 ≤ n * 2 ^ 53 := Rat.mul_le_mul_of_nonneg_right hmn' (Rat.le_of_lt two53_pos)
  grind

/-- Higham, Lemma 3.3: (1 + gamma_m)(1 + gamma_n) <= 1 + gamma_{m+n}, i.e.
gamma_m + gamma_n + gamma_m gamma_n <= gamma_{m+n}. -/
theorem gamma_add (m n : Nat) (h : (m : Rat) + n < 2 ^ 53) :
    gamma m + gamma n + gamma m * gamma n ≤ gamma (m + n) := by
  have hm0 := cast_nonneg m
  have hn0 := cast_nonneg n
  have hm : (m : Rat) < 2 ^ 53 := by grind
  have hn : (n : Rat) < 2 ^ 53 := by grind
  have hmn : ((m + n : Nat) : Rat) < 2 ^ 53 := by push_cast; exact h
  rw [gamma_eq m hm, gamma_eq n hn, gamma_eq (m + n) hmn]
  have h1 : (0 : Rat) < 2 ^ 53 - m := by grind
  have h2 : (0 : Rat) < 2 ^ 53 - n := by grind
  have h3 : (0 : Rat) < 2 ^ 53 - (m + n : Nat) := by push_cast; grind
  have lhs : (m : Rat) / (2 ^ 53 - m) + n / (2 ^ 53 - n) + (m / (2 ^ 53 - m)) * (n / (2 ^ 53 - n))
      = (m * (2 ^ 53 - n) + n * (2 ^ 53 - m) + m * n) / ((2 ^ 53 - m) * (2 ^ 53 - n)) := by grind
  rw [lhs]
  apply div_le_div_of (Rat.mul_pos h1 h2) h3
  push_cast
  have : 0 ≤ (m : Rat) * n * 2 ^ 53 := Rat.mul_nonneg (Rat.mul_nonneg hm0 hn0) (Rat.le_of_lt two53_pos)
  grind

/-- Lambda2, second part: gamma_a + u (1 + gamma_a) <= gamma_{a+1}. -/
theorem gamma_succ (a : Nat) (h : (a : Rat) + 1 < 2 ^ 53) : gamma a + u * (1 + gamma a) ≤ gamma (a + 1) := by
  have ha0 := cast_nonneg a
  have ha : (a : Rat) < 2 ^ 53 := by grind
  have ha1 : ((a + 1 : Nat) : Rat) < 2 ^ 53 := by push_cast; exact h
  rw [gamma_eq a ha, gamma_eq (a + 1) ha1]
  unfold u
  have h1 : (0 : Rat) < 2 ^ 53 - a := by grind
  have h3 : (0 : Rat) < 2 ^ 53 - (a + 1 : Nat) := by push_cast; grind
  have lhs : (a : Rat) / (2 ^ 53 - a) + 1 / 2 ^ 53 * (1 + a / (2 ^ 53 - a))
      = (a * 2 ^ 53 + 2 ^ 53) / ((2 ^ 53 - a) * 2 ^ 53) := by
    have : (2 : Rat) ^ 53 ≠ 0 := Rat.ne_of_gt two53_pos
    grind
  rw [lhs]
  apply div_le_div_of (Rat.mul_pos h1 two53_pos) h3
  push_cast
  have : 0 ≤ ((2 : Rat) ^ 53 - a) * 2 ^ 53 := Rat.mul_nonneg (Rat.le_of_lt h1) (Rat.le_of_lt two53_pos)
  grind

/-- k u <= gamma_k. -/
theorem gamma_ge (k : Nat) (hk : (k : Rat) < 2 ^ 53) : (k : Rat) * u ≤ gamma k := by
  have hk0 := cast_nonneg k
  rw [gamma_eq k hk]
  unfold u
  have h1 : (0 : Rat) < 2 ^ 53 - k := by grind
  have e : (k : Rat) * (1 / 2 ^ 53) = k / 2 ^ 53 := by
    have : (2 : Rat) ^ 53 ≠ 0 := Rat.ne_of_gt two53_pos
    grind
  rw [e]
  apply div_le_div_of two53_pos h1
  have : 0 ≤ (k : Rat) * k := Rat.mul_nonneg hk0 hk0
  grind

/-- gamma_k <= 2 k u when 2 k <= 2^53 (k u <= 1/2), as the accounting uses (gamma_a <= 2 a u). -/
theorem gamma_le_two (k : Nat) (hk : 2 * (k : Rat) ≤ 2 ^ 53) : gamma k ≤ 2 * k * u := by
  have hk0 := cast_nonneg k
  have hk' : (k : Rat) < 2 ^ 53 := by
    rcases Nat.eq_zero_or_pos k with h0 | hpos
    · subst h0; simp; exact two53_pos
    · have : (1 : Rat) ≤ k := by exact_mod_cast hpos
      grind
  rw [gamma_eq k hk']
  unfold u
  have h1 : (0 : Rat) < 2 ^ 53 - k := by grind
  have e : 2 * (k : Rat) * (1 / 2 ^ 53) = (2 * k) / 2 ^ 53 := by
    have : (2 : Rat) ^ 53 ≠ 0 := Rat.ne_of_gt two53_pos
    grind
  rw [e]
  apply div_le_div_of h1 two53_pos
  have : 0 ≤ (k : Rat) * (2 ^ 53 - 2 * k) := Rat.mul_nonneg hk0 (by grind)
  grind

/-! ### small helpers on Rat (core Lean has few lemmas on powers and max) -/

theorem rpow_add (x : Rat) (a b : Nat) : x ^ (a + b) = x ^ a * x ^ b := by
  induction b with
  | zero => rw [Nat.add_zero, Rat.pow_zero]; grind
  | succ b ih => rw [← Nat.add_assoc, Rat.pow_succ, Rat.pow_succ, ih]; grind

theorem rpow_mul (x : Rat) (a b : Nat) : x ^ (a * b) = (x ^ a) ^ b := by
  induction b with
  | zero => rw [Nat.mul_zero, Rat.pow_zero, Rat.pow_zero]
  | succ b ih => rw [Nat.mul_succ, rpow_add, ih, Rat.pow_succ]

theorem rpow_le_pow_left {x y : Rat} (h0 : 0 ≤ x) (h1 : x ≤ y) (n : Nat) : x ^ n ≤ y ^ n := by
  induction n with
  | zero => rw [Rat.pow_zero, Rat.pow_zero]; exact Rat.le_refl
  | succ n ih =>
    rw [Rat.pow_succ, Rat.pow_succ]
    have a := Rat.mul_le_mul_of_nonneg_right ih h0
    have b := Rat.mul_le_mul_of_nonneg_left h1 (Rat.pow_nonneg (n := n) (Rat.le_trans h0 h1))
    exact Rat.le_trans a b

theorem one_pow' (n : Nat) : (1 : Rat) ^ n = 1 := by
  induction n with
  | zero => rw [Rat.pow_zero]
  | succ n ih => rw [Rat.pow_succ, ih]; grind

theorem one_le_rpow {x : Rat} (h : 1 ≤ x) (n : Nat) : 1 ≤ x ^ n := by
  have := rpow_le_pow_left (by decide) h n
  rw [one_pow' n] at this
  exact this

theorem le_max_l (a b : Rat) : a ≤ max a b := by rw [Rat.max_def]; split <;> grind
theorem le_max_r (a b : Rat) : b ≤ max a b := by rw [Rat.max_def]; split <;> grind
theorem max_lt_of (a b c : Rat) (h : max a b < c) : a < c ∧ b < c := by
  have := le_max_l a b; have := le_max_r a b; constructor <;> grind

end CertifyCore.Thm
