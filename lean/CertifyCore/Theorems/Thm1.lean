import CertifyCore.Theorems.Gamma
/-
The scalar inequalities of the Theorem 1 accounting (`docs/proofs/thm1_fp-accounting.md`, section 2) with
u = 2^-53 exactly: the norm guards (Lambda4), the |A||x| guard (Lambda5) and its complex-width correction
(gamma_{w+3} in place of gamma_{w+1}, VERDICTS.md thm1 correction), the E guard (Lambda6), and the constants of
steps 2 to 5 (the factors (1 + 4u), (1 + 8u)). Kernel-checked, core Lean only.
-/
namespace CertifyCore.Thm

theorem u_pos : (0 : Rat) < u := by unfold u; decide +kernel

theorem gamma_cast (k : Nat) : gamma k = ((k : Rat) * u) / (1 - (k : Rat) * u) := rfl

/-! ### constants with u fixed (decided by evaluation in the kernel) -/

/-- (1 - gamma_3)(1 + 4u) >= 1: the factor (1 + 4u) covers three roundings (Lambda1 with k = 3). -/
theorem one_add_4u_covers_three : (1 - gamma 3) * (1 + 4 * u) ≥ 1 := by
  unfold gamma u; decide +kernel

/-- (1 - u)^3 (1 + 4u) >= 1: three relative roundings down, undone by (1 + 4u). -/
theorem one_sub_u_cube_mul : (1 - u) ^ 3 * (1 + 4 * u) ≥ 1 := by unfold u; decide +kernel

/-- (1 + 8u) covers six roundings: (1 - u)^6 (1 + 8u) >= 1 (`fp._GUARD`, `up`). -/
theorem guard_covers_six : (1 - u) ^ 6 * (1 + 8 * u) ≥ 1 := by unfold u; decide +kernel

/-- Step 5 of the accounting: the product of the slacks is at least 1 - 3u. -/
theorem step5_chain :
    (1 + 3 * u) * (1 - u) ^ 2 ≥ 1 + u - 5 * u ^ 2 ∧
    (1 - u) ^ 2 * (1 + 4 * u) ≥ 1 + 2 * u - 7 * u ^ 2 ∧
    (1 + u - 5 * u ^ 2) * (1 + 2 * u - 7 * u ^ 2) ≥ 1 - 3 * u := by
  unfold u; decide +kernel

/-! ### Lambda5: the |A||x| guard, for every width w <= 2^50 -/

/-- Lambda5: (1 - gamma_w)(1 - u)^3 (1 + 4u) / (1 - gamma_{w+1}(1 - u)^2) >= 1 + u, for 0 <= w <= 2^50
(the document states it for 1 <= w <= 2^26). -/
theorem lambda5 (w : Nat) (hw : (w : Rat) ≤ 2 ^ 50) :
    (1 - gamma w) * (1 - u) ^ 3 * (1 + 4 * u) / (1 - gamma (w + 1) * (1 - u) ^ 2) ≥ 1 + u := by
  have hw0 := cast_nonneg w
  have hlt : (w : Rat) + 1 < 2 ^ 53 := by grind
  have hs := gamma_succ w hlt
  have g0 := gamma_nonneg w (by grind)
  have g1le : gamma (w + 1) ≤ 2 * ((w + 1 : Nat) : Rat) * u := gamma_le_two (w + 1) (by push_cast; grind)
  have gle : gamma w ≤ 2 * (w : Rat) * u := gamma_le_two w (by grind)
  have hu : u = 1 / 2 ^ 53 := rfl
  have wu : (w : Rat) * u ≤ 1 / 8 := by rw [hu]; grind
  push_cast at g1le
  have d0 : 0 < 1 - gamma (w + 1) * (1 - u) ^ 2 := by
    have : gamma (w + 1) * (1 - u) ^ 2 ≤ gamma (w + 1) := by
      have h1 : (1 - u) ^ 2 ≤ 1 := by rw [hu]; decide +kernel
      have := Rat.mul_le_mul_of_nonneg_left h1 (gamma_nonneg (w + 1) (by push_cast; grind))
      grind
    unfold u at *
    grind
  rw [ge_iff_le]
  apply le_div_of d0
  -- the inequality is linear in g = gamma w once gamma_{w+1} >= g + u (1 + g) is used
  have key : (1 + u) * (1 - gamma (w + 1) * (1 - u) ^ 2) ≤ (1 + u) * (1 - (gamma w + u * (1 + gamma w)) * (1 - u) ^ 2) := by
    have h1 : (gamma w + u * (1 + gamma w)) * (1 - u) ^ 2 ≤ gamma (w + 1) * (1 - u) ^ 2 :=
      Rat.mul_le_mul_of_nonneg_right hs (by rw [hu]; decide +kernel)
    have h2 : (0 : Rat) ≤ 1 + u := by rw [hu]; decide +kernel
    have := Rat.mul_le_mul_of_nonneg_left (show 1 - gamma (w + 1) * (1 - u) ^ 2 ≤ 1 - (gamma w + u * (1 + gamma w)) * (1 - u) ^ 2 by grind) h2
    exact this
  have gw : gamma w ≤ 1 / 4 := by unfold u at *; grind
  unfold u at *
  grind

/-- The complex-width correction (VERDICTS.md, thm1): with the moduli rounded once more, the guard needs
gamma_{w+3} in the division; Lambda5 with w + 2 in place of w is exactly that inequality. -/
theorem lambda5_complex (w : Nat) (hw : (w : Rat) + 2 ≤ 2 ^ 50) :
    (1 - gamma (w + 2)) * (1 - u) ^ 3 * (1 + 4 * u) / (1 - gamma (w + 3) * (1 - u) ^ 2) ≥ 1 + u := by
  have := lambda5 (w + 2) (by push_cast; exact hw)
  simpa using this

/-! ### Lambda6: the E guard -/

/-- The largest k with k u <= 1/3 is 3002399751580330; Lambda6 holds up to one less. -/
def lambda6Max : Nat := 3002399751580329

/-- Lambda6: (1 - u)^6 (1 + 4u)(1 + u / gamma_{w+2}) >= 1 for w + 2 <= 3002399751580329. -/
theorem lambda6 (w : Nat) (hw : (w : Rat) + 2 ≤ lambda6Max) :
    (1 - u) ^ 6 * (1 + 4 * u) * (1 + u / gamma (w + 2)) ≥ 1 := by
  have hw0 := cast_nonneg w
  have hk : ((w + 2 : Nat) : Rat) < 2 ^ 53 := by push_cast; unfold lambda6Max at hw; grind
  rw [gamma_eq (w + 2) hk]
  push_cast
  have hpos : (0 : Rat) < w + 2 := by grind
  have hM : (0 : Rat) < 2 ^ 53 - (w + 2) := by push_cast at hk; grind
  have e : u / ((w + 2) / (2 ^ 53 - (w + 2))) = (2 ^ 53 - (w + 2)) / ((w + 2) * 2 ^ 53) := by
    unfold u
    have : (2 : Rat) ^ 53 ≠ 0 := Rat.ne_of_gt two53_pos
    have : ((w : Rat) + 2) ≠ 0 := Rat.ne_of_gt hpos
    have : ((2 : Rat) ^ 53 - (w + 2)) ≠ 0 := Rat.ne_of_gt hM
    grind
  rw [e]
  have hden : (0 : Rat) < (w + 2) * 2 ^ 53 := Rat.mul_pos hpos two53_pos
  have e2 : (1 : Rat) + (2 ^ 53 - (w + 2)) / ((w + 2) * 2 ^ 53) = ((w + 2) * 2 ^ 53 + 2 ^ 53 - (w + 2)) / ((w + 2) * 2 ^ 53) := by
    have : ((w : Rat) + 2) * 2 ^ 53 ≠ 0 := Rat.ne_of_gt hden
    grind
  rw [e2]
  have c6 : (0 : Rat) ≤ (1 - u) ^ 6 * (1 + 4 * u) := by unfold u; decide +kernel
  -- c6 * X / D >= 1  <=  c6 * X >= D, linear in w
  have lin : ((w : Rat) + 2) * 2 ^ 53 ≤ (1 - u) ^ 6 * (1 + 4 * u) * (((w : Rat) + 2) * 2 ^ 53 + 2 ^ 53 - (w + 2)) := by
    unfold lambda6Max at hw
    unfold u
    grind
  have := le_div_of hden (show (1 : Rat) * ((w + 2) * 2 ^ 53) ≤ (1 - u) ^ 6 * (1 + 4 * u) * ((w + 2) * 2 ^ 53 + 2 ^ 53 - (w + 2)) by grind)
  have e3 : (1 - u) ^ 6 * (1 + 4 * u) * ((((w : Rat) + 2) * 2 ^ 53 + 2 ^ 53 - (w + 2)) / ((w + 2) * 2 ^ 53))
      = (1 - u) ^ 6 * (1 + 4 * u) * ((w + 2) * 2 ^ 53 + 2 ^ 53 - (w + 2)) / ((w + 2) * 2 ^ 53) := by
    rw [Rat.div_def, Rat.div_def]; grind
  rw [ge_iff_le, e3]
  exact this

/-- At the single width w + 2 = 3002399751580330 = floor(2^53 / 3), which the document's condition
(w + 2) u <= 1/3 admits, Lambda6 fails (by an amount of order u^2). -/
theorem lambda6_fails_at_boundary :
    ¬ ((1 - u) ^ 6 * (1 + 4 * u) * (1 + u / gamma 3002399751580330) ≥ 1) := by
  unfold gamma u; decide +kernel

/-! ### Lambda4: the norm guards, for 1 <= n <= 2^26 and N in {n, 2n} -/

/-- Powers of a in [0, h]: a^(i+1) <= h a^i. -/
theorem pow_chain (a h : Rat) (h0 : 0 ≤ a) (h1 : a ≤ h) :
    a * a ≤ h * a ∧ a * a * a ≤ h * (a * a) ∧ a * a * a * a ≤ h * (a * a * a) ∧
    a * a * a * a * a ≤ h * (a * a * a * a) ∧ 0 ≤ a * a ∧ 0 ≤ a * a * a ∧ 0 ≤ a * a * a * a ∧
    0 ≤ a * a * a * a * a := by
  have p2 : 0 ≤ a * a := Rat.mul_nonneg h0 h0
  have p3 : 0 ≤ a * a * a := Rat.mul_nonneg p2 h0
  have p4 : 0 ≤ a * a * a * a := Rat.mul_nonneg p3 h0
  have p5 : 0 ≤ a * a * a * a * a := Rat.mul_nonneg p4 h0
  refine ⟨?_, ?_, ?_, ?_, p2, p3, p4, p5⟩
  · have := Rat.mul_le_mul_of_nonneg_right h1 h0; grind
  · have := Rat.mul_le_mul_of_nonneg_right h1 p2; grind
  · have := Rat.mul_le_mul_of_nonneg_right h1 p3; grind
  · have := Rat.mul_le_mul_of_nonneg_right h1 p4; grind

theorem sq_nonneg' (x : Rat) : 0 ≤ x ^ 2 := by
  rcases (Rat.le_total (a := 0) (b := x)) with h | h
  · have := Rat.mul_nonneg h h; grind
  · have h' : 0 ≤ -x := by grind
    have := Rat.mul_nonneg h' h'; grind

theorem sq_le_sq_of {x y : Rat} (h0 : 0 ≤ x) (h : x ≤ y) : x ^ 2 ≤ y ^ 2 := by
  have hy : 0 ≤ y := Rat.le_trans h0 h
  have a1 := Rat.mul_le_mul_of_nonneg_left h h0
  have a2 := Rat.mul_le_mul_of_nonneg_right h hy
  grind

/-- gamma_{n+2} >= (a + 2u)(1 + a + 2u) with a = n u. -/
theorem gamma_n2_lower (n : Nat) (hn : (n : Rat) ≤ 2 ^ 26) :
    ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) ≤ gamma (n + 2) := by
  have hn0 := cast_nonneg n
  unfold gamma
  push_cast
  have hx0 : 0 ≤ ((n : Rat) + 2) * u := Rat.mul_nonneg (by grind) (Rat.le_of_lt u_pos)
  have hx1 : ((n : Rat) + 2) * u ≤ 1 / 2 := by unfold u; grind
  have hd : 0 < 1 - ((n : Rat) + 2) * u := by grind
  apply le_div_of hd
  have x3 : 0 ≤ (((n : Rat) + 2) * u) * (((n : Rat) + 2) * u) * (((n : Rat) + 2) * u) :=
    Rat.mul_nonneg (Rat.mul_nonneg hx0 hx0) hx0
  grind

/-- Lambda4, upper guard, N = 2n. -/
theorem lambda4_upper_2n (n : Nat) (hn : (n : Rat) ≤ 2 ^ 26) :
    (1 + 3 * u) ^ 2 ≤ (1 - gamma (2 * n)) * ((1 - u) * (1 + gamma (n + 2) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4) ^ 2 := by
  have hn0 := cast_nonneg n
  have hq := gamma_n2_lower n hn
  have hq0 : 0 ≤ ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) :=
    Rat.mul_nonneg (by have := Rat.mul_nonneg hn0 (Rat.le_of_lt u_pos); have := u_pos; grind)
      (by have := Rat.mul_nonneg hn0 (Rat.le_of_lt u_pos); have := u_pos; grind)
  -- K0 <= K
  have cpos : (0 : Rat) ≤ (1 - u) * (1 + 8 * u) * (1 - u) ^ 4 := by unfold u; decide +kernel
  have dpos : (0 : Rat) ≤ (1 - u) ^ 2 := by unfold u; decide +kernel
  have kk : (1 - u) * (1 + ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4
      ≤ (1 - u) * (1 + gamma (n + 2) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4 := by
    have m1 := Rat.mul_le_mul_of_nonneg_right hq dpos
    have m2 := Rat.mul_le_mul_of_nonneg_left (show 1 + ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 - u) ^ 2
      ≤ 1 + gamma (n + 2) * (1 - u) ^ 2 by grind) cpos
    grind
  have k0pos : 0 ≤ (1 - u) * (1 + ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4 := by
    have := Rat.mul_nonneg hq0 dpos
    have := Rat.mul_nonneg (show (0 : Rat) ≤ 1 + ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 - u) ^ 2 by grind) cpos
    grind
  have sq := sq_le_sq_of k0pos kk
  -- 1 - gamma_{2n} = (1 - 4a)/(1 - 2a) >= 0
  have ha : (n : Rat) * u ≤ 1 / 134217728 := by unfold u; grind
  have ha0 : 0 ≤ (n : Rat) * u := Rat.mul_nonneg hn0 (Rat.le_of_lt u_pos)
  have g2 : 1 - gamma (2 * n) = (1 - 4 * ((n : Rat) * u)) / (1 - 2 * ((n : Rat) * u)) := by
    unfold gamma; push_cast
    have : 1 - 2 * ((n : Rat) * u) ≠ 0 := by grind
    grind
  have g2pos : 0 ≤ 1 - gamma (2 * n) := by
    rw [g2]; apply le_div_of (by grind); grind
  have step := Rat.mul_le_mul_of_nonneg_left sq g2pos
  refine Rat.le_trans ?_ step
  rw [g2]
  have e : (1 - 4 * ((n : Rat) * u)) / (1 - 2 * ((n : Rat) * u)) *
      ((1 - u) * (1 + ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4) ^ 2
      = ((1 - 4 * ((n : Rat) * u)) *
      ((1 - u) * (1 + ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4) ^ 2) /
      (1 - 2 * ((n : Rat) * u)) := by
    rw [Rat.div_def, Rat.div_def]; grind
  rw [e]
  apply le_div_of (by grind)
  -- the polynomial inequality in a = n u on [0, 2^-27]
  generalize hA : (n : Rat) * u = a at *
  have pc := pow_chain a (1 / 134217728) ha0 ha
  unfold u at *
  grind

/-- Lambda4, upper guard, N = n (follows from N = 2n, since gamma_n <= gamma_{2n}). -/
theorem lambda4_upper_n (n : Nat) (hn : (n : Rat) ≤ 2 ^ 26) :
    (1 + 3 * u) ^ 2 ≤ (1 - gamma n) * ((1 - u) * (1 + gamma (n + 2) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4) ^ 2 := by
  have h := lambda4_upper_2n n hn
  have hn0 := cast_nonneg n
  have m : gamma n ≤ gamma (2 * n) := gamma_mono n (2 * n) (by omega) (by push_cast; grind)
  have s0 : 0 ≤ ((1 - u) * (1 + gamma (n + 2) * (1 - u) ^ 2) * (1 + 8 * u) * (1 - u) ^ 4) ^ 2 := sq_nonneg' _
  have := Rat.mul_le_mul_of_nonneg_right (show 1 - gamma (2 * n) ≤ 1 - gamma n by grind) s0
  grind

/-- Lambda4, lower guard, N = 2n: (1 + gamma_N) [(1 + u)(1 - gamma_{n+2}(1 + u)^2)(1 + u)^3/(1 + 8u)]^2 <= (1 - 3u)^2,
the squared form of norm2_lower(v) <= ||v|| (1 - 3u); the bracket is non-negative. -/
theorem lambda4_lower_2n (n : Nat) (hn : (n : Rat) ≤ 2 ^ 26) :
    0 ≤ 1 - gamma (n + 2) * (1 + u) ^ 2 ∧
    (1 + gamma (2 * n)) * ((1 + u) * (1 - gamma (n + 2) * (1 + u) ^ 2) * (1 + u) ^ 3 / (1 + 8 * u)) ^ 2 ≤ (1 - 3 * u) ^ 2 := by
  have hn0 := cast_nonneg n
  have hq := gamma_n2_lower n hn
  have hqu : gamma (n + 2) ≤ 2 * ((n + 2 : Nat) : Rat) * u := gamma_le_two (n + 2) (by push_cast; grind)
  push_cast at hqu
  have ha : (n : Rat) * u ≤ 1 / 134217728 := by unfold u; grind
  have ha0 : 0 ≤ (n : Rat) * u := Rat.mul_nonneg hn0 (Rat.le_of_lt u_pos)
  have upos := u_pos
  have e2 : (1 + u) ^ 2 ≤ 2 := by unfold u; decide +kernel
  have q0 : 0 ≤ gamma (n + 2) := gamma_nonneg (n + 2) (by push_cast; grind)
  have Lpos : 0 ≤ 1 - gamma (n + 2) * (1 + u) ^ 2 := by
    have := Rat.mul_le_mul_of_nonneg_left e2 q0
    have : gamma (n + 2) ≤ 1 / 4 := by unfold u at *; grind
    grind
  refine ⟨Lpos, ?_⟩
  have L0 : 1 - gamma (n + 2) * (1 + u) ^ 2 ≤ 1 - ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 + u) ^ 2 := by
    have := Rat.mul_le_mul_of_nonneg_right hq (show (0 : Rat) ≤ (1 + u) ^ 2 from sq_nonneg' _)
    grind
  have c8 : (0 : Rat) < 1 + 8 * u := by unfold u; decide +kernel
  have c4 : (0 : Rat) ≤ (1 + u) * (1 + u) ^ 3 := by unfold u; decide +kernel
  -- rewrite the bracket as ((1 + u)^4 L) / (1 + 8u)
  have eb : ∀ L : Rat, (1 + u) * L * (1 + u) ^ 3 / (1 + 8 * u) = ((1 + u) * (1 + u) ^ 3 * L) / (1 + 8 * u) := by
    intro L; rw [Rat.div_def, Rat.div_def]; grind
  rw [eb]
  have B0 : 0 ≤ (1 + u) * (1 + u) ^ 3 * (1 - gamma (n + 2) * (1 + u) ^ 2) := Rat.mul_nonneg c4 Lpos
  have Bm : (1 + u) * (1 + u) ^ 3 * (1 - gamma (n + 2) * (1 + u) ^ 2) ≤
      (1 + u) * (1 + u) ^ 3 * (1 - ((n : Rat) * u + 2 * u) * (1 + (n : Rat) * u + 2 * u) * (1 + u) ^ 2) :=
    Rat.mul_le_mul_of_nonneg_left L0 c4
  have sq := sq_le_sq_of B0 Bm
  have g2 : 1 + gamma (2 * n) = 1 / (1 - 2 * ((n : Rat) * u)) := by
    unfold gamma; push_cast
    have : 1 - 2 * ((n : Rat) * u) ≠ 0 := by grind
    grind
  rw [g2]
  have dpos : 0 < 1 - 2 * ((n : Rat) * u) := by grind
  have e3 : ∀ X : Rat, 1 / (1 - 2 * ((n : Rat) * u)) * (X / (1 + 8 * u)) ^ 2 = X ^ 2 / ((1 - 2 * ((n : Rat) * u)) * (1 + 8 * u) ^ 2) := by
    intro X
    have : 1 - 2 * ((n : Rat) * u) ≠ 0 := by grind
    have : 1 + 8 * u ≠ 0 := Rat.ne_of_gt c8
    rw [Rat.div_def, Rat.div_def, Rat.div_def]; grind
  rw [e3]
  have c82 : (0 : Rat) < (1 + 8 * u) ^ 2 := by unfold u; decide +kernel
  apply div_le_of (Rat.mul_pos dpos c82)
  refine Rat.le_trans sq ?_
  generalize hA : (n : Rat) * u = a at *
  have pc := pow_chain a (1 / 134217728) ha0 ha
  unfold u at *
  grind

/-- Lambda4, lower guard, N = n. -/
theorem lambda4_lower_n (n : Nat) (hn : (n : Rat) ≤ 2 ^ 26) :
    (1 + gamma n) * ((1 + u) * (1 - gamma (n + 2) * (1 + u) ^ 2) * (1 + u) ^ 3 / (1 + 8 * u)) ^ 2 ≤ (1 - 3 * u) ^ 2 := by
  have ⟨_, h⟩ := lambda4_lower_2n n hn
  have hn0 := cast_nonneg n
  have m : gamma n ≤ gamma (2 * n) := gamma_mono n (2 * n) (by omega) (by push_cast; grind)
  have := Rat.mul_le_mul_of_nonneg_right (show 1 + gamma n ≤ 1 + gamma (2 * n) by grind)
    (sq_nonneg' ((1 + u) * (1 - gamma (n + 2) * (1 + u) ^ 2) * (1 + u) ^ 3 / (1 + 8 * u)))
  grind

end CertifyCore.Thm
