import CertifyCore.Theorems.Gamma
/-
The finite arithmetic of Lemma 17 (`docs/proofs/L17_dyadic_exp_ln.md`, `certify_ed/ivl.py`): the remainder
constant 8/7 of the ln series for arguments u <= 0.35, the tail bound of the atanh series and of the exp
series in finite form, the range of the series argument (u(m) <= 1/3 for m in [1, 2], and ceil(2^P/3) <= U_MAX
for P = 192), and the nested-ceiling identity used by the halving count. Kernel-checked, core Lean only.
-/
namespace CertifyCore.Thm

/-- 1/(1 - x^2) <= 8/7 for 0 <= x <= 0.35 (x^2 <= 0.1225 <= 1/8). -/
theorem ln_remainder_const (x : Rat) (h0 : 0 ≤ x) (h1 : x ≤ 35 / 100) : 1 / (1 - x ^ 2) ≤ 8 / 7 := by
  have xx : x * x ≤ 35 / 100 * x := Rat.mul_le_mul_of_nonneg_right h1 h0
  have hx2 : x ^ 2 ≤ 1225 / 10000 := by grind
  apply div_le_of (by grind)
  grind

/-- The constant 9/8 of the 3.1 code is not enough at x = 0.35: 1/(1 - 0.35^2) > 9/8. -/
theorem ln_remainder_old_const_too_small : ¬ ((1 : Rat) / (1 - (35 / 100) ^ 2) ≤ 9 / 8) := by decide +kernel

/-- Partial geometric sum: sum_{i<K} q^i (1 - q) = 1 - q^K. -/
def geom (q : Rat) : Nat → Rat
  | 0 => 0
  | K + 1 => geom q K + q ^ K

theorem geom_mul (q : Rat) (K : Nat) : geom q K * (1 - q) = 1 - q ^ K := by
  induction K with
  | zero => simp only [geom, Rat.pow_zero]; grind
  | succ K ih => simp only [geom]; rw [Rat.pow_succ]; grind

theorem geom_le (q : Rat) (h0 : 0 ≤ q) (h1 : q < 1) (K : Nat) : geom q K ≤ 1 / (1 - q) := by
  apply le_div_of (by grind)
  rw [geom_mul]
  have : 0 ≤ q ^ K := Rat.pow_nonneg h0
  grind

/-- atanh tail in finite form: sum_{j=M+1}^{M+K} x^(2j+1)/(2j+1). -/
def atanhTail (x : Rat) (M : Nat) : Nat → Rat
  | 0 => 0
  | K + 1 => atanhTail x M K + x ^ (2 * (M + 1 + K) + 1) / (2 * (M + 1 + K) + 1 : Nat)

theorem atanhTail_le_geom (x : Rat) (M : Nat) (h0 : 0 ≤ x) (K : Nat) :
    atanhTail x M K ≤ x ^ (2 * M + 3) / (2 * M + 3 : Nat) * geom (x ^ 2) K := by
  induction K with
  | zero => simp [atanhTail, geom]
  | succ K ih =>
    simp only [atanhTail, geom]
    have hpow : x ^ (2 * (M + 1 + K) + 1) = x ^ (2 * M + 3) * (x ^ 2) ^ K := by
      rw [← rpow_mul, ← rpow_add]; congr 1; omega
    have hd1 : (0 : Rat) < ((2 * M + 3 : Nat) : Rat) := by push_cast; have := cast_nonneg M; grind
    have hd2 : ((2 * M + 3 : Nat) : Rat) ≤ ((2 * (M + 1 + K) + 1 : Nat) : Rat) := by exact_mod_cast (by omega)
    have tnn : 0 ≤ x ^ (2 * M + 3) * (x ^ 2) ^ K := Rat.mul_nonneg (Rat.pow_nonneg h0) (Rat.pow_nonneg (Rat.pow_nonneg h0))
    have term : x ^ (2 * (M + 1 + K) + 1) / ((2 * (M + 1 + K) + 1 : Nat) : Rat) ≤
        x ^ (2 * M + 3) * (x ^ 2) ^ K / ((2 * M + 3 : Nat) : Rat) := by
      rw [hpow]
      apply div_le_div_of (by grind) hd1
      exact Rat.mul_le_mul_of_nonneg_left hd2 tnn
    have e : x ^ (2 * M + 3) / ((2 * M + 3 : Nat) : Rat) * (geom (x ^ 2) K + (x ^ 2) ^ K) =
        x ^ (2 * M + 3) / ((2 * M + 3 : Nat) : Rat) * geom (x ^ 2) K + x ^ (2 * M + 3) * (x ^ 2) ^ K / ((2 * M + 3 : Nat) : Rat) := by
      rw [Rat.div_def, Rat.div_def]; grind
    rw [e]
    grind

/-- The ln remainder of Lemma 17 (iv): for 0 <= x <= 0.35 and every K,
sum_{j=M+1}^{M+K} x^(2j+1)/(2j+1) <= (8/7) x^(2M+3)/(2M+3). -/
theorem ln_tail_bound (x : Rat) (M K : Nat) (h0 : 0 ≤ x) (h1 : x ≤ 35 / 100) :
    atanhTail x M K ≤ 8 / 7 * (x ^ (2 * M + 3) / (2 * M + 3 : Nat)) := by
  have t := atanhTail_le_geom x M h0 K
  have q0 : 0 ≤ x ^ 2 := Rat.pow_nonneg h0
  have xx : x * x ≤ 35 / 100 * x := Rat.mul_le_mul_of_nonneg_right h1 h0
  have q1 : x ^ 2 < 1 := by grind
  have g := geom_le (x ^ 2) q0 q1 K
  have c := ln_remainder_const x h0 h1
  have T0 : 0 ≤ x ^ (2 * M + 3) / ((2 * M + 3 : Nat) : Rat) := by
    apply le_div_of (by push_cast; have := cast_nonneg M; grind)
    have := Rat.pow_nonneg (n := 2 * M + 3) h0; grind
  have := Rat.mul_le_mul_of_nonneg_left (Rat.le_trans g c) T0
  grind

/-- Factorial (core Lean has none). -/
def fact : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

theorem fact_pos (n : Nat) : 0 < fact n := by
  induction n with
  | zero => decide
  | succ n ih => simp only [fact]; exact Nat.mul_pos (Nat.succ_pos n) ih

/-- exp tail in finite form: sum_{j<K} t^(N+1+j)/(N+1+j)!. -/
def expTail (t : Rat) (N : Nat) : Nat → Rat
  | 0 => 0
  | K + 1 => expTail t N K + t ^ (N + 1 + K) / (fact (N + 1 + K) : Rat)

/-- (N+1+j)! >= (N+1)! (N+2)^j. -/
theorem fact_ge (N j : Nat) : fact (N + 1) * (N + 2) ^ j ≤ fact (N + 1 + j) := by
  induction j with
  | zero => simp
  | succ j ih =>
    have e : N + 1 + (j + 1) = (N + 1 + j) + 1 := by omega
    rw [e]
    simp only [fact]
    rw [Nat.pow_succ]
    have : N + 2 ≤ N + 1 + j + 1 := by omega
    calc fact (N + 1) * ((N + 2) ^ j * (N + 2)) = (fact (N + 1) * (N + 2) ^ j) * (N + 2) := by
          rw [Nat.mul_assoc]
      _ ≤ fact (N + 1 + j) * (N + 1 + j + 1) := Nat.mul_le_mul ih this
      _ = (N + 1 + j + 1) * fact (N + 1 + j) := Nat.mul_comm _ _

/-- The exp remainder of Lemma 17 (iii): for 0 <= t <= 1/2 and every K,
sum_{k=N+1}^{N+K} t^k/k! <= 2 t^(N+1)/(N+1)!. -/
theorem exp_tail_bound (t : Rat) (N K : Nat) (h0 : 0 ≤ t) (h1 : t ≤ 1 / 2) :
    expTail t N K ≤ 2 * (t ^ (N + 1) / (fact (N + 1) : Rat)) := by
  -- each term is at most T * (t/(N+2))^j <= T * (1/2)^j
  have F0 : (0 : Rat) < (fact (N + 1) : Rat) := by exact_mod_cast fact_pos (N + 1)
  have T0 : 0 ≤ t ^ (N + 1) / (fact (N + 1) : Rat) := by
    apply le_div_of F0; have := Rat.pow_nonneg (n := N + 1) h0; grind
  have key : ∀ K, expTail t N K ≤ t ^ (N + 1) / (fact (N + 1) : Rat) * geom (1 / 2) K := by
    intro K
    induction K with
    | zero => simp [expTail, geom]
    | succ K ih =>
      simp only [expTail, geom]
      have fg : ((fact (N + 1) * (N + 2) ^ K : Nat) : Rat) ≤ (fact (N + 1 + K) : Rat) := by exact_mod_cast fact_ge N K
      push_cast at fg
      have pN : (0 : Rat) < ((N : Rat) + 2) ^ K := Rat.pow_pos (by have := cast_nonneg N; grind)
      have tk : t ^ K ≤ (1 / 2) ^ K * 1 := by
        have : t ^ K ≤ (1 / 2) ^ K := rpow_le_pow_left h0 h1 K
        grind
      have tk2 : t ^ K ≤ (1 / 2) ^ K * ((N : Rat) + 2) ^ K := by
        have one : (1 : Rat) ≤ ((N : Rat) + 2) ^ K := one_le_rpow (x := (N : Rat) + 2) (by have := cast_nonneg N; grind) K
        have := Rat.mul_le_mul_of_nonneg_left one (Rat.pow_nonneg (n := K) (by grind : (0 : Rat) ≤ 1 / 2))
        grind
      have term : t ^ (N + 1 + K) / (fact (N + 1 + K) : Rat) ≤ t ^ (N + 1) / (fact (N + 1) : Rat) * (1 / 2) ^ K := by
        have e1 : t ^ (N + 1 + K) = t ^ (N + 1) * t ^ K := rpow_add _ _ _
        rw [e1]
        have e2 : t ^ (N + 1) / (fact (N + 1) : Rat) * (1 / 2) ^ K = t ^ (N + 1) * (1 / 2) ^ K / (fact (N + 1) : Rat) := by
          rw [Rat.div_def, Rat.div_def]; grind
        rw [e2]
        apply div_le_div_of (by exact_mod_cast fact_pos (N + 1 + K)) F0
        have p1 : 0 ≤ t ^ (N + 1) := Rat.pow_nonneg h0
        have a1 := Rat.mul_le_mul_of_nonneg_left tk2 p1
        have a2 := Rat.mul_le_mul_of_nonneg_left fg (Rat.mul_nonneg p1 (Rat.pow_nonneg (n := K) (by grind : (0 : Rat) ≤ 1 / 2)))
        have a3 := Rat.mul_le_mul_of_nonneg_right a1 (Rat.le_of_lt F0)
        grind
      have e : t ^ (N + 1) / (fact (N + 1) : Rat) * (geom (1 / 2) K + (1 / 2) ^ K) =
          t ^ (N + 1) / (fact (N + 1) : Rat) * geom (1 / 2) K + t ^ (N + 1) / (fact (N + 1) : Rat) * (1 / 2) ^ K := by grind
      rw [e]
      grind
  have g := geom_le (1 / 2) (by grind) (by grind) K
  have g2 : geom (1 / 2) K ≤ 2 := by
    have : (1 : Rat) / (1 - 1 / 2) = 2 := by decide +kernel
    grind
  have := Rat.mul_le_mul_of_nonneg_left g2 T0
  have := key K
  grind

/-- The series argument u(m) = (m - 1)/(m + 1) of Lemma 17 (iv) is at most 1/3 for 1 <= m <= 2 (in grid units
X = 2^P: (m - X)/(m + X) <= 1/3 for X <= m <= 2X). -/
theorem ln_argument_range (X m : Rat) (hX : 0 < X) (h1 : X ≤ m) (h2 : m ≤ 2 * X) : (m - X) / (m + X) ≤ 1 / 3 := by
  apply div_le_of (by grind)
  grind

/-- ceil(2^192/3) <= U_MAX = floor(0.35 * 2^192): the runtime check of `ivl.py` admits every argument the
decomposition produces. -/
theorem umax_admits : (2 ^ 192 + 2) / 3 ≤ 35 * 2 ^ 192 / 100 := by decide +kernel

/-- Nested ceilings, as used by the halving count of (iii): ceil(ceil(a/b)/c) = ceil(a/(bc)) for b, c > 0,
with ceil(a/b) written (a + b - 1)/b in natural numbers. -/
theorem ceil_ceil (a b c : Nat) (hb : 0 < b) (hc : 0 < c) :
    ((a + b - 1) / b + c - 1) / c = (a + b * c - 1) / (b * c) := by
  have e : (a + b - 1) / b + (c - 1) = (a + b - 1 + b * (c - 1)) / b := by
    rw [Nat.add_mul_div_left _ _ hb]
  have e2 : a + b - 1 + b * (c - 1) = a + b * c - 1 := by
    have h1 : b * (c - 1) = b * c - b := Nat.mul_sub_one b c
    have h2 : b ≤ b * c := Nat.le_mul_of_pos_right b hc
    omega
  have e3 : (a + b - 1) / b + c - 1 = (a + b - 1) / b + (c - 1) := Nat.add_sub_assoc hc _
  rw [e3, e, e2, Nat.div_div_eq_div_mul]

end CertifyCore.Thm
