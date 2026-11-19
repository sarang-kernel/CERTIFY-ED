import Mathlib.Analysis.Normed.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Complex.Basic
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

/-!
# A formal model of binary64 rounding and Higham's error lemmas

The model is a set of hypotheses about an abstract rounding operation, not an implementation: every
theorem below holds for any operation that satisfies them.

`Rounding V` is a rounding `rnd : V → V` on a normed group `V` (`ℝ` for real data, `ℂ` for complex
data) with a set of representable numbers `IsFloat`, a unit roundoff `u` and an absolute underflow
term `eta`, such that

* every result is representable (`isFloat_rnd`),
* `‖rnd z - z‖ ≤ u ‖z‖ + eta` for every exact result `z` (the underflow-tolerant model M-U of
  `docs/proofs/int05_underflow.md`),
* `‖rnd (a + b) - (a + b)‖ ≤ u ‖a + b‖` for representable `a`, `b` (a sum that underflows is exact,
  so additions carry no absolute term).

`Binary64` adds the binary64 constants `u = 2⁻⁵³`, `eta = 2⁻¹⁰⁷⁵`, monotonicity, and the two
neighbours `succ`, `pred` of a representable number with `z ≤ succ (rnd z)` and
`pred (rnd z) ≤ z`, which is what the code's outward rounding (`fp.add_up`, `fp.sub_down`, one
`nextafter` step after a round-to-nearest operation) needs. The standard model of
`docs/proofs/thm1_fp-accounting.md` is the case `eta = 0` (no underflow).

Proved here: Higham's Lemma 3.1 (`higham_lemma_3_1`), the algebra of `γ_k = k u / (1 - k u)` of
Lemma 3.3 (`gam_succ`, `gam_add`, `gam_mono`, `mul_gam_le`, `gam_div_one_sub`), the error of a sum
evaluated in any order (a binary summation tree, `STree.comp_error`), and the dot-product bound
(Higham eq. (3.4) and Lemma U1 of INT-05, `dot_error`).
-/

open Finset

namespace CertifyMathlib.Enclosure.FP

/-! ### The constants `γ_k` (Higham 2002, Lemma 3.3) -/

/-- `γ_k = k u / (1 - k u)`. -/
noncomputable def gam (u : ℝ) (k : ℕ) : ℝ := (k : ℝ) * u / (1 - (k : ℝ) * u)

variable {u : ℝ}

theorem gam_zero : gam u 0 = 0 := by simp [gam]

theorem one_add_gam {k : ℕ} (hk : (k : ℝ) * u < 1) : 1 + gam u k = 1 / (1 - (k : ℝ) * u) := by
  have hne : (1 - (k : ℝ) * u) ≠ 0 := by linarith
  unfold gam
  field_simp
  ring

theorem gam_nonneg (hu : 0 ≤ u) {k : ℕ} (hk : (k : ℝ) * u < 1) : 0 ≤ gam u k :=
  div_nonneg (mul_nonneg (Nat.cast_nonneg k) hu) (by linarith)

theorem gam_mono (hu : 0 ≤ u) {j k : ℕ} (hjk : j ≤ k) (hk : (k : ℝ) * u < 1) :
    gam u j ≤ gam u k := by
  have hj : (j : ℝ) ≤ k := by exact_mod_cast hjk
  have h1 : (j : ℝ) * u ≤ k * u := mul_le_mul_of_nonneg_right hj hu
  unfold gam
  exact div_le_div₀ (mul_nonneg (Nat.cast_nonneg k) hu) h1 (by linarith) (by linarith)

/-- `γ_k + u (1 + γ_k) ≤ γ_{k+1}`. -/
theorem gam_succ (hu : 0 ≤ u) {k : ℕ} (hk : ((k + 1 : ℕ) : ℝ) * u < 1) :
    gam u k + u * (1 + gam u k) ≤ gam u (k + 1) := by
  push_cast at hk
  have hkn : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hd1 : 0 < 1 - (k : ℝ) * u := by nlinarith
  have hd2 : 0 < 1 - ((k : ℝ) + 1) * u := by linarith
  have hne := hd1.ne'
  have e : gam u k + u * (1 + gam u k) = ((k : ℝ) + 1) * u / (1 - (k : ℝ) * u) := by
    unfold gam
    field_simp
    ring
  rw [e]
  unfold gam
  push_cast
  exact div_le_div_of_nonneg_left (by positivity) hd2 (by nlinarith)

/-- `(1 + γ_j)(1 + γ_k) ≤ 1 + γ_{j+k}`, equivalently `γ_j + γ_k + γ_j γ_k ≤ γ_{j+k}`. -/
theorem gam_add (hu : 0 ≤ u) {j k : ℕ} (h : ((j + k : ℕ) : ℝ) * u < 1) :
    gam u j + gam u k + gam u j * gam u k ≤ gam u (j + k) := by
  have h' := h
  push_cast at h'
  have hjn : (0 : ℝ) ≤ j := Nat.cast_nonneg j
  have hkn : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hj : (j : ℝ) * u < 1 := by nlinarith
  have hk : (k : ℝ) * u < 1 := by nlinarith
  have hdj : 0 < 1 - (j : ℝ) * u := by linarith
  have hdk : 0 < 1 - (k : ℝ) * u := by linarith
  have hd : 0 < 1 - ((j + k : ℕ) : ℝ) * u := by linarith
  have e1 : gam u j + gam u k + gam u j * gam u k = (1 + gam u j) * (1 + gam u k) - 1 := by ring
  have e2 : gam u (j + k) = (1 + gam u (j + k)) - 1 := by ring
  rw [e1, e2, one_add_gam hj, one_add_gam hk, one_add_gam h, one_div_mul_one_div]
  have hle : 1 - ((j + k : ℕ) : ℝ) * u ≤ (1 - (j : ℝ) * u) * (1 - (k : ℝ) * u) := by
    push_cast
    nlinarith [mul_nonneg (mul_nonneg hjn hkn) (mul_nonneg hu hu)]
  have := one_div_le_one_div_of_le hd hle
  linarith

theorem one_add_gam_mul_le (hu : 0 ≤ u) {j k : ℕ} (h : ((j + k : ℕ) : ℝ) * u < 1) :
    (1 + gam u j) * (1 + gam u k) ≤ 1 + gam u (j + k) := by
  have := gam_add hu h
  nlinarith

/-- `γ_k + u ≤ γ_{k+1} (1 - u)`, used for the factors `(1 + δ)⁻¹`. -/
theorem gam_div_one_sub (hu : 0 ≤ u) {k : ℕ} (hk : ((k + 1 : ℕ) : ℝ) * u < 1) :
    gam u k + u ≤ gam u (k + 1) * (1 - u) := by
  push_cast at hk
  have hkn : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hd1 : 0 < 1 - (k : ℝ) * u := by nlinarith
  have hd2 : 0 < 1 - ((k : ℝ) + 1) * u := by linarith
  have key : gam u (k + 1) * (1 - u) - (gam u k + u) =
      (k : ℝ) * u * u / ((1 - (k : ℝ) * u) * (1 - ((k : ℝ) + 1) * u)) := by
    unfold gam
    push_cast
    field_simp
    ring
  have : 0 ≤ (k : ℝ) * u * u / ((1 - (k : ℝ) * u) * (1 - ((k : ℝ) + 1) * u)) :=
    div_nonneg (mul_nonneg (mul_nonneg hkn hu) hu) (mul_pos hd1 hd2).le
  linarith

/-- `c γ_k ≤ γ_{c k}`. -/
theorem mul_gam_le (hu : 0 ≤ u) {c k : ℕ} (hc : 1 ≤ c) (h : ((c * k : ℕ) : ℝ) * u < 1) :
    (c : ℝ) * gam u k ≤ gam u (c * k) := by
  have hcr : (1 : ℝ) ≤ c := by exact_mod_cast hc
  have hkn : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  push_cast at h ⊢
  have hk : (k : ℝ) * u < 1 := by nlinarith [mul_nonneg hkn hu]
  unfold gam
  push_cast
  rw [mul_div_assoc', ← mul_assoc]
  exact div_le_div_of_nonneg_left (mul_nonneg (mul_nonneg (by linarith) hkn) hu)
    (by linarith) (by nlinarith [mul_nonneg hkn hu])

theorem u_le_gam_one (hu : 0 ≤ u) (h1 : u < 1) : u ≤ gam u 1 := by
  unfold gam
  push_cast
  rw [one_mul, le_div_iff₀ (by linarith)]
  nlinarith

/-! ### Higham's Lemma 3.1 -/

/-- One factor `(1 + δ)^{±1}`: `true` for the power `+1`, `false` for `-1`. -/
noncomputable def fac (p : ℝ × Bool) : ℝ := if p.2 then 1 + p.1 else (1 + p.1)⁻¹

theorem fac_step (hu : 0 ≤ u) {k : ℕ} (hk : ((k + 1 : ℕ) : ℝ) * u < 1) {θ : ℝ}
    (hθ : |θ| ≤ gam u k) (p : ℝ × Bool) (hp : |p.1| ≤ u) :
    |fac p * (1 + θ) - 1| ≤ gam u (k + 1) := by
  have hk' := hk
  push_cast at hk'
  have hkn : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hu1 : u < 1 := by nlinarith
  have hγ : 0 ≤ gam u k := gam_nonneg hu (by nlinarith)
  rcases p with ⟨δ, b⟩
  have hδ := abs_le.mp hp
  cases b
  · -- the factor `(1 + δ)⁻¹`
    have h1 : 0 < 1 + δ := by linarith
    simp only [fac, Bool.false_eq_true, ite_false]
    have e : (1 + δ)⁻¹ * (1 + θ) - 1 = (θ - δ) / (1 + δ) := by
      field_simp
      ring
    rw [e, abs_div, abs_of_pos h1, div_le_iff₀ h1]
    have h2 := gam_div_one_sub hu hk
    have h3 : |θ - δ| ≤ gam u k + u := (abs_sub _ _).trans (add_le_add hθ hp)
    have h4 : gam u (k + 1) * (1 - u) ≤ gam u (k + 1) * (1 + δ) :=
      mul_le_mul_of_nonneg_left (by linarith) (gam_nonneg hu (by push_cast; linarith))
    linarith
  · -- the factor `1 + δ`
    simp only [fac, ite_true]
    have e : (1 + δ) * (1 + θ) - 1 = θ + δ + δ * θ := by ring
    rw [e]
    have h2 := gam_succ hu hk
    have h3 : |δ * θ| ≤ u * gam u k := by
      rw [abs_mul]
      exact mul_le_mul hp hθ (abs_nonneg _) hu
    calc |θ + δ + δ * θ| ≤ |θ| + |δ| + |δ * θ| := abs_add_three _ _ _
      _ ≤ gam u k + u + u * gam u k := by linarith
      _ ≤ gam u (k + 1) := by linarith

/-- **Higham's Lemma 3.1.** A product of `k` factors `(1 + δ_i)^{±1}` with `|δ_i| ≤ u` and
`k u < 1` equals `1 + θ_k` with `|θ_k| ≤ γ_k`. -/
theorem higham_lemma_3_1 (hu : 0 ≤ u) :
    ∀ l : List (ℝ × Bool), (∀ p ∈ l, |p.1| ≤ u) → (l.length : ℝ) * u < 1 →
      |(l.map fac).prod - 1| ≤ gam u l.length
  | [], _, _ => by simp [gam]
  | p :: l, hp, hl => by
    have hl1 : ((l.length + 1 : ℕ) : ℝ) * u < 1 := by simpa using hl
    have hl2 : (l.length : ℝ) * u < 1 := by
      have : (l.length : ℝ) * u ≤ ((l.length + 1 : ℕ) : ℝ) * u := by
        push_cast
        nlinarith
      linarith
    have ih := higham_lemma_3_1 hu l (fun q hq => hp q (List.mem_cons_of_mem _ hq)) hl2
    have hstep := fac_step hu hl1 (θ := (l.map fac).prod - 1) ih p (hp p List.mem_cons_self)
    simp only [List.map_cons, List.prod_cons, List.length_cons]
    simpa using hstep

/-! ### The rounding model -/

/-- An abstract rounding on a normed group `V`: see the module docstring. -/
structure Rounding (V : Type*) [NormedAddCommGroup V] where
  rnd : V → V
  IsFloat : V → Prop
  isFloat_rnd : ∀ v, IsFloat (rnd v)
  u : ℝ
  eta : ℝ
  u_nonneg : 0 ≤ u
  eta_nonneg : 0 ≤ eta
  err : ∀ v, ‖rnd v - v‖ ≤ u * ‖v‖ + eta
  add_err : ∀ a b, IsFloat a → IsFloat b → ‖rnd (a + b) - (a + b)‖ ≤ u * ‖a + b‖

/-- The binary64 instance of the model on `ℝ`, with outward neighbours. -/
structure Binary64 extends Rounding ℝ where
  u_eq : u = (2 : ℝ)⁻¹ ^ 53
  eta_eq : eta = (2 : ℝ)⁻¹ ^ 1075
  mono : Monotone rnd
  succ : ℝ → ℝ
  pred : ℝ → ℝ
  le_succ_rnd : ∀ z, z ≤ succ (rnd z)
  pred_rnd_le : ∀ z, pred (rnd z) ≤ z

namespace Binary64

variable (B : Binary64)

/-- `fp.add_up`: the successor of the rounded sum. -/
def addUp (a b : ℝ) : ℝ := B.succ (B.rnd (a + b))

/-- `fp.sub_down`: the predecessor of the rounded difference. -/
def subDown (a b : ℝ) : ℝ := B.pred (B.rnd (a - b))

theorem le_addUp (a b : ℝ) : a + b ≤ B.addUp a b := B.le_succ_rnd _

theorem subDown_le (a b : ℝ) : B.subDown a b ≤ a - b := B.pred_rnd_le _

/-- An outward interval `[subDown μ r, addUp μ r]` contains `[μ - r, μ + r]`. -/
theorem outward_interval (μ r x : ℝ) (hx : |x - μ| ≤ r) :
    B.subDown μ r ≤ x ∧ x ≤ B.addUp μ r := by
  rw [abs_le] at hx
  exact ⟨(B.subDown_le μ r).trans (by linarith), (by linarith : x ≤ μ + r).trans (B.le_addUp μ r)⟩

end Binary64

namespace Rounding

variable {V : Type*} [NormedAddCommGroup V] (R : Rounding V)

/-- Standard model, real data: a nonnegative exact result loses at most a factor `1 - u`. -/
theorem rnd_ge_of_nonneg (R : Rounding ℝ) (h0 : R.eta = 0) {z : ℝ} (hz : 0 ≤ z) :
    z * (1 - R.u) ≤ R.rnd z := by
  have h := R.err z
  rw [h0, add_zero, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg hz] at h
  have := (abs_le.mp h).1
  linarith

theorem rnd_le_of_nonneg (R : Rounding ℝ) (h0 : R.eta = 0) {z : ℝ} (hz : 0 ≤ z) :
    R.rnd z ≤ z * (1 + R.u) := by
  have h := R.err z
  rw [h0, add_zero, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg hz] at h
  have := (abs_le.mp h).2
  linarith

theorem rnd_le_abs (R : Rounding ℝ) (h0 : R.eta = 0) (z : ℝ) :
    R.rnd z ≤ z + R.u * |z| := by
  have h := R.err z
  rw [h0, add_zero, Real.norm_eq_abs, Real.norm_eq_abs] at h
  linarith [(abs_le.mp h).2]

end Rounding

/-! ### Sums in any order: binary summation trees -/

/-- A summation order: a binary tree whose leaves are the indices of the terms. -/
inductive STree (α : Type*) where
  | leaf : α → STree α
  | node : STree α → STree α → STree α

namespace STree

variable {α : Type*}

/-- The number of additions of the tree (leaves minus one). -/
def adds : STree α → ℕ
  | leaf _ => 0
  | node l r => l.adds + r.adds + 1

/-- The leaves, from left to right. -/
def leaves : STree α → List α
  | leaf i => [i]
  | node l r => l.leaves ++ r.leaves

theorem length_leaves : ∀ T : STree α, T.leaves.length = T.adds + 1
  | leaf _ => rfl
  | node l r => by simp [leaves, adds, length_leaves l, length_leaves r]; ring

variable {V : Type*} [NormedAddCommGroup V]

/-- The exact sum along the tree. -/
def eval (t : α → V) : STree α → V
  | leaf i => t i
  | node l r => l.eval t + r.eval t

/-- The sum of the norms of the terms. -/
def absEval (t : α → V) : STree α → ℝ
  | leaf i => ‖t i‖
  | node l r => l.absEval t + r.absEval t

/-- The computed sum: every addition rounded, the leaves are the computed terms. -/
def comp (rnd : V → V) (t : α → V) : STree α → V
  | leaf i => t i
  | node l r => rnd (l.comp rnd t + r.comp rnd t)

theorem eval_eq_sum (t : α → V) : ∀ T : STree α, T.eval t = (T.leaves.map t).sum
  | leaf i => by simp [eval, leaves]
  | node l r => by simp [eval, leaves, eval_eq_sum t l, eval_eq_sum t r]

theorem absEval_eq_sum (t : α → V) : ∀ T : STree α, T.absEval t = (T.leaves.map fun i => ‖t i‖).sum
  | leaf i => by simp [absEval, leaves]
  | node l r => by simp [absEval, leaves, absEval_eq_sum t l, absEval_eq_sum t r]

theorem norm_eval_le (t : α → V) : ∀ T : STree α, ‖T.eval t‖ ≤ T.absEval t
  | leaf _ => le_rfl
  | node l r => (norm_add_le _ _).trans (add_le_add (norm_eval_le t l) (norm_eval_le t r))

theorem absEval_nonneg (t : α → V) : ∀ T : STree α, 0 ≤ T.absEval t
  | leaf _ => norm_nonneg _
  | node l r => add_nonneg (absEval_nonneg t l) (absEval_nonneg t r)

theorem comp_isFloat (R : Rounding V) (t : α → V) (ht : ∀ i, R.IsFloat (t i)) :
    ∀ T : STree α, R.IsFloat (T.comp R.rnd t)
  | leaf i => ht i
  | node _ _ => R.isFloat_rnd _

/-- **The error of a sum in any order.** If every computed term `t̂ i` is representable and
within `γ_a ‖t i‖ + β` of the exact term, the computed sum along `T` is within
`γ_{a + adds T} ∑ ‖t i‖ + (adds T + 1)(1 + γ_{adds T}) β` of the exact sum. -/
theorem comp_error (R : Rounding V) {a : ℕ} {β : ℝ} (hβ : 0 ≤ β) (t th : α → V)
    (hleaf : ∀ i, ‖th i - t i‖ ≤ gam R.u a * ‖t i‖ + β) (hfl : ∀ i, R.IsFloat (th i)) :
    ∀ T : STree α, ((a + T.adds : ℕ) : ℝ) * R.u < 1 →
      ‖T.comp R.rnd th - T.eval t‖ ≤ gam R.u (a + T.adds) * T.absEval t +
        ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * β
  | leaf i, _ => by
    simp only [comp, eval, absEval, adds, add_zero, Nat.cast_zero, zero_add, gam_zero, one_mul]
    exact hleaf i
  | node l r, h => by
    have hu := R.u_nonneg
    have hl : ((a + l.adds : ℕ) : ℝ) * R.u < 1 := by
      refine lt_of_le_of_lt ?_ h
      apply mul_le_mul_of_nonneg_right _ hu
      exact_mod_cast (by simp only [adds]; omega)
    have hr : ((a + r.adds : ℕ) : ℝ) * R.u < 1 := by
      refine lt_of_le_of_lt ?_ h
      apply mul_le_mul_of_nonneg_right _ hu
      exact_mod_cast (by simp only [adds]; omega)
    have IHl := comp_error R hβ t th hleaf hfl l hl
    have IHr := comp_error R hβ t th hleaf hfl r hr
    set m := a + l.adds + r.adds with hm
    have hmn : a + (node l r).adds = m + 1 := by simp only [adds]; omega
    rw [hmn] at h ⊢
    have hm1 : (m : ℝ) * R.u < 1 := by
      refine lt_of_le_of_lt ?_ h
      apply mul_le_mul_of_nonneg_right _ hu
      exact_mod_cast Nat.le_succ m
    have gl : gam R.u (a + l.adds) ≤ gam R.u m := gam_mono hu (by omega) hm1
    have gr : gam R.u (a + r.adds) ≤ gam R.u m := gam_mono hu (by omega) hm1
    have Sl := absEval_nonneg t l
    have Sr := absEval_nonneg t r
    -- additive parts
    have hal : ((l.adds : ℝ)) * R.u < 1 := by
      refine lt_of_le_of_lt ?_ hm1
      apply mul_le_mul_of_nonneg_right _ hu
      exact_mod_cast (by omega)
    have har : ((r.adds : ℝ)) * R.u < 1 := by
      refine lt_of_le_of_lt ?_ hm1
      apply mul_le_mul_of_nonneg_right _ hu
      exact_mod_cast (by omega)
    set Bl := ((l.adds : ℝ) + 1) * (1 + gam R.u l.adds) * β with hBl
    set Br := ((r.adds : ℝ) + 1) * (1 + gam R.u r.adds) * β with hBr
    have hBl0 : 0 ≤ Bl := mul_nonneg (mul_nonneg (by positivity)
      (by linarith [gam_nonneg hu hal])) hβ
    have hBr0 : 0 ≤ Br := mul_nonneg (mul_nonneg (by positivity)
      (by linarith [gam_nonneg hu har])) hβ
    set Δ := (l.comp R.rnd th + r.comp R.rnd th) - (l.eval t + r.eval t) with hΔ
    have hΔle : ‖Δ‖ ≤ gam R.u m * (l.absEval t + r.absEval t) + (Bl + Br) := by
      have e : Δ = (l.comp R.rnd th - l.eval t) + (r.comp R.rnd th - r.eval t) := by
        rw [hΔ]; abel
      rw [e]
      refine (norm_add_le _ _).trans ?_
      have h1 : gam R.u (a + l.adds) * l.absEval t ≤ gam R.u m * l.absEval t :=
        mul_le_mul_of_nonneg_right gl Sl
      have h2 : gam R.u (a + r.adds) * r.absEval t ≤ gam R.u m * r.absEval t :=
        mul_le_mul_of_nonneg_right gr Sr
      linarith
    have hround := R.add_err (l.comp R.rnd th) (r.comp R.rnd th)
      (comp_isFloat R th hfl l) (comp_isFloat R th hfl r)
    have hsum : ‖l.comp R.rnd th + r.comp R.rnd th‖ ≤
        (l.absEval t + r.absEval t) + ‖Δ‖ := by
      have e : l.comp R.rnd th + r.comp R.rnd th = (l.eval t + r.eval t) + Δ := by
        rw [hΔ]; abel
      rw [e]
      refine (norm_add_le _ _).trans ?_
      have := (norm_add_le (l.eval t) (r.eval t)).trans
        (add_le_add (norm_eval_le t l) (norm_eval_le t r))
      linarith
    have hnode : ‖(node l r).comp R.rnd th - (node l r).eval t‖ ≤
        R.u * ((l.absEval t + r.absEval t) + ‖Δ‖) + ‖Δ‖ := by
      have e : (node l r).comp R.rnd th - (node l r).eval t =
          (R.rnd (l.comp R.rnd th + r.comp R.rnd th) - (l.comp R.rnd th + r.comp R.rnd th)) + Δ := by
        simp only [comp, eval]
        rw [hΔ]
        abel
      rw [e]
      refine (norm_add_le _ _).trans ?_
      have := mul_le_mul_of_nonneg_left hsum hu
      linarith
    have gsucc := gam_succ hu (k := m) (by exact_mod_cast h)
    have hγm : 0 ≤ gam R.u m := gam_nonneg hu hm1
    -- the additive part grows by at most the factor `1 + u` per level
    have hBadd : (1 + R.u) * (Bl + Br) ≤
        (((node l r).adds : ℝ) + 1) * (1 + gam R.u (node l r).adds) * β := by
      have hnl : (1 + R.u) * (1 + gam R.u l.adds) ≤ 1 + gam R.u (node l r).adds := by
        have hs := gam_succ hu (k := l.adds) (by
          refine lt_of_le_of_lt ?_ h
          apply mul_le_mul_of_nonneg_right _ hu
          exact_mod_cast (by omega))
        have hmono : gam R.u (l.adds + 1) ≤ gam R.u (node l r).adds :=
          gam_mono hu (by simp only [adds]; omega) (by
            refine lt_of_le_of_lt ?_ h
            apply mul_le_mul_of_nonneg_right _ hu
            exact_mod_cast (by simp only [adds]; omega))
        nlinarith
      have hnr : (1 + R.u) * (1 + gam R.u r.adds) ≤ 1 + gam R.u (node l r).adds := by
        have hs := gam_succ hu (k := r.adds) (by
          refine lt_of_le_of_lt ?_ h
          apply mul_le_mul_of_nonneg_right _ hu
          exact_mod_cast (by omega))
        have hmono : gam R.u (r.adds + 1) ≤ gam R.u (node l r).adds :=
          gam_mono hu (by simp only [adds]; omega) (by
            refine lt_of_le_of_lt ?_ h
            apply mul_le_mul_of_nonneg_right _ hu
            exact_mod_cast (by simp only [adds]; omega))
        nlinarith
      have hcount : ((node l r).adds : ℝ) + 1 = ((l.adds : ℝ) + 1) + ((r.adds : ℝ) + 1) := by
        simp only [adds]; push_cast; ring
      rw [hcount, hBl, hBr]
      have hl0 : (0 : ℝ) ≤ (l.adds : ℝ) + 1 := by positivity
      have hr0 : (0 : ℝ) ≤ (r.adds : ℝ) + 1 := by positivity
      have k1 := mul_le_mul_of_nonneg_left hnl (mul_nonneg hl0 hβ)
      have k2 := mul_le_mul_of_nonneg_left hnr (mul_nonneg hr0 hβ)
      nlinarith
    simp only [absEval]
    calc ‖(node l r).comp R.rnd th - (node l r).eval t‖
        ≤ R.u * ((l.absEval t + r.absEval t) + ‖Δ‖) + ‖Δ‖ := hnode
      _ = R.u * (l.absEval t + r.absEval t) + (1 + R.u) * ‖Δ‖ := by ring
      _ ≤ R.u * (l.absEval t + r.absEval t) +
            (1 + R.u) * (gam R.u m * (l.absEval t + r.absEval t) + (Bl + Br)) :=
          add_le_add le_rfl (mul_le_mul_of_nonneg_left hΔle (by linarith))
      _ = (gam R.u m + R.u * (1 + gam R.u m)) * (l.absEval t + r.absEval t) +
            (1 + R.u) * (Bl + Br) := by ring
      _ ≤ gam R.u (m + 1) * (l.absEval t + r.absEval t) +
            (((node l r).adds : ℝ) + 1) * (1 + gam R.u (node l r).adds) * β :=
          add_le_add (mul_le_mul_of_nonneg_right gsucc (add_nonneg Sl Sr)) hBadd

/-- A tree with no repeated leaf whose leaves cover the support of `t` sums to `∑ t`. -/
theorem eval_eq_finset_sum [DecidableEq α] [Fintype α] (t : α → V) (T : STree α)
    (hnd : T.leaves.Nodup) (hsupp : ∀ i, i ∉ T.leaves → t i = 0) : T.eval t = ∑ i, t i := by
  rw [eval_eq_sum, ← List.sum_toFinset _ hnd]
  exact Finset.sum_subset (Finset.subset_univ _) (fun i _ hi => hsupp i (by simpa using hi))

theorem absEval_le_finset_sum [DecidableEq α] [Fintype α] (t : α → V) (T : STree α)
    (hnd : T.leaves.Nodup) : T.absEval t ≤ ∑ i, ‖t i‖ := by
  rw [absEval_eq_sum, ← List.sum_toFinset _ hnd]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
    (fun i _ _ => norm_nonneg _)

theorem absEval_eq_finset_sum [DecidableEq α] [Fintype α] (t : α → V) (T : STree α)
    (hnd : T.leaves.Nodup) (hsupp : ∀ i, i ∉ T.leaves → t i = 0) :
    T.absEval t = ∑ i, ‖t i‖ := by
  rw [absEval_eq_sum, ← List.sum_toFinset _ hnd]
  exact Finset.sum_subset (Finset.subset_univ _)
    (fun i _ hi => by rw [hsupp i (by simpa using hi), norm_zero])

end STree

/-! ### Dot products -/

section Dot

variable {𝕂 : Type*} [NormedField 𝕂] {α : Type*}

/-- **Dot-product error** (Higham eq. (3.4); Lemma U1 of INT-05). The products `rnd (a i * b i)`
are summed along `T` with `k = adds T + 1` terms:
`‖fl(∑ a_i b_i) - ∑ a_i b_i‖ ≤ γ_k ∑ ‖a_i b_i‖ + k (1 + γ_{k-1}) eta`. -/
theorem dot_error (R : Rounding 𝕂) (a b : α → 𝕂) (T : STree α)
    (h : ((T.adds + 1 : ℕ) : ℝ) * R.u < 1) :
    ‖T.comp R.rnd (fun i => R.rnd (a i * b i)) - T.eval (fun i => a i * b i)‖ ≤
      gam R.u (T.adds + 1) * T.absEval (fun i => a i * b i) +
        ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * R.eta := by
  have hu := R.u_nonneg
  have h1 : (1 : ℝ) * R.u < 1 := by
    refine lt_of_le_of_lt ?_ h
    apply mul_le_mul_of_nonneg_right _ hu
    exact_mod_cast (by omega)
  have hleaf : ∀ i, ‖R.rnd (a i * b i) - a i * b i‖ ≤ gam R.u 1 * ‖a i * b i‖ + R.eta := by
    intro i
    have := R.err (a i * b i)
    have hg := u_le_gam_one hu (by simpa using h1)
    have := mul_le_mul_of_nonneg_right hg (norm_nonneg (a i * b i))
    linarith
  have := STree.comp_error R R.eta_nonneg (fun i => a i * b i) (fun i => R.rnd (a i * b i))
    hleaf (fun i => R.isFloat_rnd _) T (by rw [add_comm]; exact h)
  rw [add_comm 1 T.adds] at this
  exact this

end Dot

end CertifyMathlib.Enclosure.FP
