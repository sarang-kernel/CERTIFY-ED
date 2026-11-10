import CertifyCore.Theorems.Gamma
/-
The finite arithmetic of INT-05 (`docs/proofs/int05_underflow.md`): the binary64 grid, the exactness of
additions whose result lies below 2^-1022 (the eta = 0 clause of the model (M-U)), the additive term of a sum of
k rounded products (Lemma U1), the norm underflow factor of `fp.norm_underflow_term` (Lemma U3), and why
`+ ULP_MIN` covers the rounding of `math.ldexp(t, -1075)`. Kernel-checked, core Lean only.
-/
namespace CertifyCore.Thm

/-- The binary64 numbers: m 2^e with |m| < 2^53 and -1074 <= e <= 971, written m 2^k / 2^1074 with
k = e + 1074 in [0, 2045]. (Infinities and NaN are not numbers here.) -/
def IsBinary64 (q : Rat) : Prop :=
  ∃ (m : Int) (k : Nat), m.natAbs < 2 ^ 53 ∧ k ≤ 2045 ∧ q = ((m * 2 ^ k : Int) : Rat) / 2 ^ 1074

/-- Every binary64 number is an integer multiple of 2^-1074. -/
theorem binary64_on_grid (q : Rat) (h : IsBinary64 q) : ∃ z : Int, q = (z : Rat) / 2 ^ 1074 := by
  obtain ⟨m, k, _, _, hq⟩ := h
  exact ⟨m * 2 ^ k, hq⟩

/-- An integer multiple z 2^-1074 with |z| < 2^53 is a binary64 number (a subnormal one, or zero, or a
normal one of the lowest binade). -/
theorem small_multiple_binary64 (z : Int) (hz : z.natAbs < 2 ^ 53) : IsBinary64 ((z : Rat) / 2 ^ 1074) :=
  ⟨z, 0, hz, by decide, by simp⟩

/-- (M-U), the clause eta = 0 for + and -: if x and y are binary64 numbers and |x + y| < 2^-1022, then x + y
is itself a binary64 number, so round-to-nearest returns it exactly (the same holds for x - y = x + (-y)). -/
theorem add_exact_below (x y : Rat) (hx : IsBinary64 x) (hy : IsBinary64 y)
    (hlo : -(1 / 2 ^ 1022) < x + y) (hhi : x + y < 1 / 2 ^ 1022) : IsBinary64 (x + y) := by
  obtain ⟨a, ha⟩ := binary64_on_grid x hx
  obtain ⟨b, hb⟩ := binary64_on_grid y hy
  have hs : x + y = ((a + b : Int) : Rat) / 2 ^ 1074 := by rw [ha, hb]; push_cast; grind
  rw [hs] at hlo hhi ⊢
  apply small_multiple_binary64
  have p : (0 : Rat) < 2 ^ 1074 := by decide +kernel
  have h1 : ((a + b : Int) : Rat) < 2 ^ 52 := by
    have : ((a + b : Int) : Rat) / 2 ^ 1074 * 2 ^ 1074 < 1 / 2 ^ 1022 * 2 ^ 1074 :=
      Rat.mul_lt_mul_of_pos_right hhi p
    have e1 : ((a + b : Int) : Rat) / 2 ^ 1074 * 2 ^ 1074 = ((a + b : Int) : Rat) := Rat.div_mul_cancel (Rat.ne_of_gt p)
    have e2 : (1 : Rat) / 2 ^ 1022 * 2 ^ 1074 = 2 ^ 52 := by decide +kernel
    grind
  have h2 : -(2 : Rat) ^ 52 < ((a + b : Int) : Rat) := by
    have : -(1 / 2 ^ 1022) * 2 ^ 1074 < ((a + b : Int) : Rat) / 2 ^ 1074 * 2 ^ 1074 :=
      Rat.mul_lt_mul_of_pos_right hlo p
    have e1 : ((a + b : Int) : Rat) / 2 ^ 1074 * 2 ^ 1074 = ((a + b : Int) : Rat) := Rat.div_mul_cancel (Rat.ne_of_gt p)
    have e2 : -((1 : Rat) / 2 ^ 1022) * 2 ^ 1074 = -(2 ^ 52) := by decide +kernel
    grind
  have e52 : (2 : Rat) ^ 52 = 4503599627370496 := by decide +kernel
  rw [e52] at h1 h2
  have i1 : a + b < 4503599627370496 := by
    by_cases hc : a + b < 4503599627370496
    · exact hc
    · have h : 4503599627370496 ≤ a + b := by omega
      have := (Rat.intCast_le_intCast (a := 4503599627370496) (b := a + b)).mpr h
      push_cast at this h1
      grind
  have i2 : -4503599627370496 < a + b := by
    by_cases hc : -4503599627370496 < a + b
    · exact hc
    · have h : a + b ≤ -4503599627370496 := by omega
      have := (Rat.intCast_le_intCast (a := a + b) (b := -4503599627370496)).mpr h
      push_cast at this h2
      grind
  have : (2 : Nat) ^ 53 = 9007199254740992 := by decide
  rw [this]
  omega

/-- Lemma U1, additive part: if every rounded product carries an absolute error eta_j with |eta_j| <= u_min and
is then multiplied by a factor (1 + theta_j) with |theta_j| <= g <= 1 (the later additions), the accumulated
additive error is at most k u_min (1 + g) in magnitude, k the number of products. -/
theorem additive_sum_bound (ps : List (Rat × Rat)) (umin g : Rat) (hu : 0 ≤ umin) (hg1 : g ≤ 1)
    (hp : ∀ p ∈ ps, -umin ≤ p.1 ∧ p.1 ≤ umin ∧ -g ≤ p.2 ∧ p.2 ≤ g) :
    -((ps.length : Rat) * umin * (1 + g)) ≤ (ps.map (fun p => p.1 * (1 + p.2))).sum ∧
    (ps.map (fun p => p.1 * (1 + p.2))).sum ≤ (ps.length : Rat) * umin * (1 + g) := by
  induction ps with
  | nil => simp
  | cons p t ih =>
    have ⟨a1, a2, a3, a4⟩ := hp p (by simp)
    have ⟨i1, i2⟩ := ih (fun q hq => hp q (by simp [hq]))
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    have f0 : 0 ≤ 1 + p.2 := by grind
    have up1 : p.1 * (1 + p.2) ≤ umin * (1 + p.2) := Rat.mul_le_mul_of_nonneg_right a2 f0
    have up2 : umin * (1 + p.2) ≤ umin * (1 + g) := Rat.mul_le_mul_of_nonneg_left (by grind) hu
    have lo1 : -umin * (1 + p.2) ≤ p.1 * (1 + p.2) := Rat.mul_le_mul_of_nonneg_right a1 f0
    have lo2 : umin * (1 + p.2) ≤ umin * (1 + g) := up2
    constructor <;> grind

/-- Lemma U3, the factor of `fp.norm_underflow_term`: (1 + 2 gamma_{N+2})^2 >= (1 + gamma_N)/(1 - gamma_N) for
N u <= 1/3, so sqrt(N u_min)(1 + 2 gamma_{N+2}) >= sqrt(N u_min) sqrt((1 + gamma_N)/(1 - gamma_N)). -/
theorem norm_underflow_factor (N : Nat) (hN : 3 * (N : Rat) + 6 ≤ 2 ^ 53) :
    (1 + gamma N) / (1 - gamma N) ≤ (1 + 2 * gamma (N + 2)) ^ 2 := by
  have hN0 := cast_nonneg N
  have hlt : ((N + 2 : Nat) : Rat) < 2 ^ 53 := by push_cast; grind
  have g0 := gamma_nonneg N (by grind)
  have mono := gamma_mono N (N + 2) (by omega) hlt
  have ghalf : gamma N ≤ 1 / 2 := by
    rw [gamma_eq N (by grind)]
    apply div_le_of (by grind)
    grind
  have d : 0 < 1 - gamma N := by grind
  apply div_le_of d
  -- (1 + 2g')^2 (1 - g) >= (1 + 2g)^2 (1 - g) = 1 + 3g - 4g^3 >= 1 + g
  have s1 : (1 + 2 * gamma N) ^ 2 ≤ (1 + 2 * gamma (N + 2)) ^ 2 := by
    have h0 : 0 ≤ 1 + 2 * gamma N := by grind
    have := Rat.mul_le_mul_of_nonneg_left (show 1 + 2 * gamma N ≤ 1 + 2 * gamma (N + 2) by grind) h0
    have := Rat.mul_le_mul_of_nonneg_right (show 1 + 2 * gamma N ≤ 1 + 2 * gamma (N + 2) by grind) (by grind : (0 : Rat) ≤ 1 + 2 * gamma (N + 2))
    grind
  have s2 := Rat.mul_le_mul_of_nonneg_right s1 (Rat.le_of_lt d)
  have g3 : gamma N * gamma N * gamma N ≤ gamma N * (1 / 4) := by
    have q : gamma N * gamma N ≤ 1 / 4 := by
      have := Rat.mul_le_mul_of_nonneg_left ghalf g0
      have := Rat.mul_le_mul_of_nonneg_right ghalf (by grind : (0 : Rat) ≤ 1 / 2)
      grind
    have := Rat.mul_le_mul_of_nonneg_left q g0
    grind
  grind

/-- `math.ldexp(t, -1075)` rounds t 2^-1075 to the nearest multiple of 2^-1074 (the subnormal grid), an error of
at most u_min = 2^-1075; adding ULP_MIN = 2^-1074 = 2 u_min therefore gives an upper bound. -/
theorem ulp_min_covers (t r : Rat) (hr : t - 1 / 2 ^ 1075 ≤ r) : t ≤ r + 1 / 2 ^ 1074 := by
  have : (1 : Rat) / 2 ^ 1075 ≤ 1 / 2 ^ 1074 := by decide +kernel
  grind

end CertifyCore.Thm
