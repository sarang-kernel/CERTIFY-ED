import CertifyCore.Kernel.Defs
import CertifyMathlib.Enclosure.Residual

/-!
# Refinement of the kernel residual check (N-05)

`CertifyCore.K.pairOK` (`CertifyCore/Kernel/Defs.lean`) is the residual check that `checkLowest` performs for every
claimed pair, in exact rational arithmetic on lists. This file proves, on Mathlib, that when it accepts a pair the
conclusion of Theorem 1 (`CertifyMathlib.Enclosure.residual_enclosure`) holds for the decoded rational data: the
matrix whose entry (i, j) is the sum of the values a row stores at column j, viewed as a real symmetric matrix, has
an eigenvalue within the recorded existence radius of the claimed value.
-/

open Matrix

namespace CertifyMathlib.Refine

open CertifyCore.K

/-- Entry `j` of a stored row: the sum of the values stored at column `j` (the expression of `denseRow`). -/
def entryQ (row : List (Nat × Rat)) (j : Nat) : Rat :=
  (row.filter (fun e => e.1 = j)).foldr (fun e s => e.2 + s) 0

/-- The decoded matrix of a kernel certificate, as a real matrix. -/
noncomputable def toMatrix (A : List (List (Nat × Rat))) : Matrix (Fin A.length) (Fin A.length) ℝ :=
  fun i j => (entryQ (A.getD i []) j : ℝ)

/-- The decoded vector of a pair, as a Euclidean vector of length `n`. -/
noncomputable def toVec (n : Nat) (x : List Rat) : EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 (fun i => (getR x i : ℝ))

theorem getR_eq_getD (x : List Rat) (j : Nat) : getR x j = x.getD j 0 := by
  induction x generalizing j with
  | nil => simp [getR]
  | cons a as ih =>
    cases j with
    | zero => simp [getR]
    | succ j => simp [getR, ih]

theorem getR_of_le : ∀ {x : List Rat} {j : Nat}, x.length ≤ j → getR x j = 0
  | [], _, _ => by simp [getR]
  | _ :: _, 0, h => by simp at h
  | _ :: as, j + 1, h => by
    simp only [getR]
    exact getR_of_le (x := as) (by simpa using h)

theorem entryQ_nil (j : Nat) : entryQ [] j = 0 := by simp [entryQ]

theorem entryQ_cons (k : Nat) (v : Rat) (t : List (Nat × Rat)) (j : Nat) :
    entryQ ((k, v) :: t) j = (if k = j then v else 0) + entryQ t j := by
  unfold entryQ
  by_cases h : k = j
  · simp [h]
  · simp [h]

/-- `rowDot` is the row of the decoded matrix applied to the decoded vector. -/
theorem rowDot_eq_sum (row : List (Nat × Rat)) {n : Nat} {x : List Rat} (hx : x.length = n) :
    rowDot row x = ∑ j : Fin n, entryQ row j * getR x j := by
  induction row with
  | nil => simp [rowDot, entryQ_nil]
  | cons e t ih =>
    obtain ⟨k, v⟩ := e
    simp only [rowDot, entryQ_cons, add_mul, Finset.sum_add_distrib, ih]
    congr 1
    by_cases hk : k < n
    · rw [Finset.sum_eq_single ⟨k, hk⟩]
      · simp
      · intro b _ hb
        have : k ≠ (b : Nat) := fun h => hb (Fin.ext h.symm)
        simp [this]
      · simp
    · have h0 : getR x k = 0 := getR_of_le (by omega)
      rw [h0, mul_zero]
      symm
      refine Finset.sum_eq_zero fun b _ => ?_
      have : k ≠ (b : Nat) := by have := b.isLt; omega
      simp [this]

/-- `dotL` of two lists of length `n` is the sum of the products of their entries. -/
theorem dotL_eq_sum : ∀ {n : Nat} (a b : List Rat), a.length = n → b.length = n →
    dotL a b = ∑ i : Fin n, getR a i * getR b i
  | 0, [], [], _, _ => by simp [dotL]
  | n + 1, x :: as, y :: bs, ha, hb => by
    rw [Fin.sum_univ_succ]
    simp only [List.length_cons, Nat.add_right_cancel_iff] at ha hb
    simp [dotL, getR, dotL_eq_sum as bs ha hb]
  | 0, _ :: _, _, ha, _ => by simp at ha
  | 0, [], _ :: _, _, hb => by simp at hb
  | _ + 1, [], _, ha, _ => by simp at ha
  | _ + 1, _ :: _, [], _, hb => by simp at hb

theorem getR_zipWith (g : Rat → Rat → Rat) : ∀ (l1 l2 : List Rat) (i : Nat), i < l1.length → i < l2.length →
    getR (List.zipWith g l1 l2) i = g (getR l1 i) (getR l2 i)
  | _ :: _, _ :: _, 0, _, _ => by simp [getR]
  | _ :: as, _ :: bs, i + 1, h1, h2 => by
    simp only [List.zipWith_cons_cons, getR]
    exact getR_zipWith g as bs i (by simpa using h1) (by simpa using h2)
  | [], _, _, h1, _ => by simp at h1
  | _ :: _, [], _, _, h2 => by simp at h2

theorem getR_map_rowDot (x : List Rat) : ∀ (A : List (List (Nat × Rat))) (i : Nat), i < A.length →
    getR (A.map (fun row => rowDot row x)) i = rowDot (A.getD i []) x
  | _ :: _, 0, _ => by simp [getR]
  | _ :: as, i + 1, h => by
    simp only [List.map_cons, getR, List.getD_cons_succ]
    exact getR_map_rowDot x as i (by simpa using h)
  | [], _, h => by simp at h

theorem length_residual (A : List (List (Nat × Rat))) {x : List Rat} (hx : x.length = A.length) (mu : Rat) :
    (CertifyCore.K.residual A x mu).length = A.length := by
  simp [CertifyCore.K.residual, matVec, hx]

theorem getR_residual (A : List (List (Nat × Rat))) {x : List Rat} (hx : x.length = A.length) (mu : Rat)
    (i : Fin A.length) :
    getR (CertifyCore.K.residual A x mu) i = rowDot (A.getD i []) x - mu * getR x i := by
  simp only [CertifyCore.K.residual, matVec]
  rw [getR_zipWith _ _ _ _ (by simp) (by rw [hx]; exact i.isLt), getR_map_rowDot x A i i.isLt]

theorem toVec_apply (n : Nat) (x : List Rat) (i : Fin n) : toVec n x i = (getR x i : ℝ) := rfl

theorem norm_sq_toVec {n : Nat} {x : List Rat} (hx : x.length = n) :
    ‖toVec n x‖ ^ 2 = ((dotL x x : Rat) : ℝ) := by
  rw [EuclideanSpace.real_norm_sq_eq, dotL_eq_sum x x hx hx]
  push_cast
  simp only [toVec_apply, sq]

/-- The residual of the decoded data is the decoded residual list. -/
theorem residual_vec (A : List (List (Nat × Rat))) {x : List Rat} (hx : x.length = A.length) (mu : Rat) :
    toEuclideanLin (toMatrix A) (toVec A.length x) - ((mu : ℝ) : ℝ) • toVec A.length x =
      toVec A.length (CertifyCore.K.residual A x mu) := by
  ext i
  have hc : (toEuclideanLin (toMatrix A) (toVec A.length x)) i = ∑ j, toMatrix A i j * toVec A.length x j := by
    rw [Matrix.toLpLin_apply]
    rfl
  rw [PiLp.sub_apply, PiLp.smul_apply, hc, smul_eq_mul, toVec_apply, toVec_apply, getR_residual A hx mu i,
    rowDot_eq_sum _ hx]
  simp only [toMatrix, toVec_apply]
  push_cast
  ring

theorem norm_sq_residual (A : List (List (Nat × Rat))) {x : List Rat} (hx : x.length = A.length) (mu : Rat) :
    ‖toEuclideanLin (toMatrix A) (toVec A.length x) - ((mu : ℝ) : ℝ) • toVec A.length x‖ ^ 2 =
      ((dotL (CertifyCore.K.residual A x mu) (CertifyCore.K.residual A x mu) : Rat) : ℝ) := by
  rw [residual_vec A hx mu, norm_sq_toVec (length_residual A hx mu)]

/-! ### The checks of `checkLowest` -/

/-- Exact symmetry of the decoded matrix, as a Boolean that the kernel can evaluate. `checkLowest` does not
test symmetry (cecheck does, in `H5.symmetric`), so the refinement theorems take this check as a hypothesis. -/
def symOK (A : List (List (Nat × Rat))) : Bool :=
  (List.range A.length).all fun i => (List.range A.length).all fun j =>
    decide (entryQ (A.getD i []) j = entryQ (A.getD j []) i)

theorem isHermitian_of_symOK {A : List (List (Nat × Rat))} (h : symOK A = true) : (toMatrix A).IsHermitian := by
  refine Matrix.IsHermitian.ext fun i j => ?_
  simp only [symOK, List.all_eq_true, List.mem_range, decide_eq_true_eq] at h
  simp only [toMatrix, star_trivial]
  exact_mod_cast h j j.isLt i i.isLt

/-- Vectors of the matrix's length and nonzero, as a Boolean. `pairOK` accepts a zero vector (the residual and
the vector are then both zero), so this is a hypothesis too. -/
def vecOK (c : Cert) : Bool :=
  c.pairs.all fun p => decide (p.x.length = c.A.length) && decide (dotL p.x p.x ≠ 0)

theorem pairOK_spec {c : Cert} {p : Pair} (h : pairOK c p = true) :
    0 ≤ p.rb ∧ dotL (p.r c) (p.r c) ≤ p.rb * p.rb ∧ 0 ≤ p.ex ∧
      dotL (p.r c) (p.r c) ≤ p.ex * p.ex * dotL p.x p.x ∧ p.ex ≤ p.rad ∧
      p.rad + c.shift ≤ p.radArr ∧ p.ex + c.shift ≤ p.exArr := by
  simpa [pairOK, Bool.and_eq_true, decide_eq_true_eq, and_assoc] using h

theorem toVec_ne_zero {n : Nat} {x : List Rat} (hx : x.length = n) (h0 : dotL x x ≠ 0) : toVec n x ≠ 0 := by
  intro hz
  apply h0
  have h2 := norm_sq_toVec hx
  rw [hz, norm_zero] at h2
  exact_mod_cast (by simpa using h2.symm : ((dotL x x : Rat) : ℝ) = 0)

/-- The bound that `pairOK` checks, in the norms of Theorem 1: `‖A x - μ x‖ / ‖x‖ ≤ ex`. -/
theorem residual_ratio_le (c : Cert) (p : Pair) (hn : p.x.length = c.A.length) (h : pairOK c p = true)
    (hxv : toVec c.A.length p.x ≠ 0) :
    ‖toEuclideanLin (toMatrix c.A) (toVec c.A.length p.x) - ((p.mu : ℝ) : ℝ) • toVec c.A.length p.x‖ /
      ‖toVec c.A.length p.x‖ ≤ (p.ex : ℝ) := by
  obtain ⟨-, -, hex0, hres, -, -, -⟩ := pairOK_spec h
  have hpos : 0 < ‖toVec c.A.length p.x‖ := norm_pos_iff.mpr hxv
  rw [div_le_iff₀ hpos]
  have hres' : ((dotL (CertifyCore.K.residual c.A p.x p.mu) (CertifyCore.K.residual c.A p.x p.mu) : Rat) : ℝ) ≤
      (p.ex : ℝ) * p.ex * ((dotL p.x p.x : Rat) : ℝ) := by
    have := hres
    simp only [Pair.r] at this
    exact_mod_cast this
  have hex : (0 : ℝ) ≤ p.ex := by exact_mod_cast hex0
  have hsq : ‖toEuclideanLin (toMatrix c.A) (toVec c.A.length p.x) - ((p.mu : ℝ) : ℝ) • toVec c.A.length p.x‖ ^ 2 ≤
      ((p.ex : ℝ) * ‖toVec c.A.length p.x‖) ^ 2 := by
    rw [norm_sq_residual c.A hn p.mu, mul_pow, norm_sq_toVec hn]
    nlinarith [hres']
  exact (pow_le_pow_iff_left₀ (norm_nonneg _) (by positivity) two_ne_zero).mp hsq

/-- **Refinement of the residual check.** If `pairOK` accepts a pair whose vector has the matrix's length and is
nonzero, and the decoded matrix is symmetric, then some eigenvalue of the decoded matrix lies within the recorded
existence radius `ex` of the claimed value (the conclusion of `residual_enclosure`). -/
theorem pairOK_theorem1 (c : Cert) (p : Pair) (hn : p.x.length = c.A.length) (hx0 : dotL p.x p.x ≠ 0)
    (hA : (toMatrix c.A).IsHermitian) (h : pairOK c p = true) :
    ∃ i, |hA.eigenvalues i - (p.mu : ℝ)| ≤ (p.ex : ℝ) := by
  have hxv := toVec_ne_zero hn hx0
  obtain ⟨i, hi⟩ := Enclosure.residual_enclosure hA (p.mu : ℝ) hxv
  exact ⟨i, le_trans hi (residual_ratio_le c p hn h hxv)⟩

open scoped Matrix.Norms.L2Operator in
/-- The same with the S1 model shift: for every symmetric `H` (the declared model) with `‖H - A‖₂ ≤ shift`,
some eigenvalue of `H` lies within the exported existence radius `exArr` of the claimed value. -/
theorem pairOK_theorem1_shift (c : Cert) (p : Pair) (hn : p.x.length = c.A.length) (hx0 : dotL p.x p.x ≠ 0)
    (h : pairOK c p = true) {H : Matrix (Fin c.A.length) (Fin c.A.length) ℝ} (hH : H.IsHermitian)
    (hσ : ‖H - toMatrix c.A‖ ≤ (c.shift : ℝ)) :
    ∃ i, |hH.eigenvalues i - (p.mu : ℝ)| ≤ (p.exArr : ℝ) := by
  have hxv := toVec_ne_zero hn hx0
  obtain ⟨i, hi⟩ := Enclosure.residual_enclosure_of_l2_opNorm hH (toMatrix c.A) hσ (p.mu : ℝ) hxv
  obtain ⟨-, -, -, -, -, -, hexA⟩ := pairOK_spec h
  have hexA' : (p.ex : ℝ) + c.shift ≤ p.exArr := by exact_mod_cast hexA
  refine ⟨i, le_trans hi ?_⟩
  have hr := residual_ratio_le c p hn h hxv
  exact le_trans (add_le_add hr le_rfl) hexA'

open scoped Matrix.Norms.L2Operator in
/-- The same bound by the exported radius `radArr` (from `ex ≤ rad` and `rad + shift ≤ radArr`). -/
theorem pairOK_theorem1_shift_rad (c : Cert) (p : Pair) (hn : p.x.length = c.A.length)
    (hx0 : dotL p.x p.x ≠ 0) (h : pairOK c p = true) {H : Matrix (Fin c.A.length) (Fin c.A.length) ℝ}
    (hH : H.IsHermitian) (hσ : ‖H - toMatrix c.A‖ ≤ (c.shift : ℝ)) :
    ∃ i, |hH.eigenvalues i - (p.mu : ℝ)| ≤ (p.radArr : ℝ) := by
  have hxv := toVec_ne_zero hn hx0
  obtain ⟨i, hi⟩ := Enclosure.residual_enclosure_of_l2_opNorm hH (toMatrix c.A) hσ (p.mu : ℝ) hxv
  obtain ⟨-, -, -, -, hr0, hradA, -⟩ := pairOK_spec h
  have h2 : (p.ex : ℝ) + c.shift ≤ p.radArr := by
    have h3 : (p.ex : ℝ) ≤ p.rad := by exact_mod_cast hr0
    have h4 : (p.rad : ℝ) + c.shift ≤ p.radArr := by exact_mod_cast hradA
    linarith
  refine ⟨i, le_trans hi ?_⟩
  have hr := residual_ratio_le c p hn h hxv
  exact le_trans (add_le_add hr le_rfl) h2

theorem pairs_ok_of_checkLowest {c : Cert} (h : checkLowest c = true) : ∀ p ∈ c.pairs, pairOK c p = true := by
  have h1 : c.pairs.all (pairOK c) = true := by
    unfold checkLowest at h
    simp only [Bool.and_eq_true] at h
    exact h.1.1.1.1.1.1.1
  simpa [List.all_eq_true] using h1

theorem le_maxL_init : ∀ (l : List Rat) (m : Rat), m ≤ maxL l m
  | [], m => by simp [maxL]
  | a :: as, m => by
    simp only [maxL]
    refine le_trans ?_ (le_maxL_init as _)
    split_ifs with hm
    · exact hm.le
    · exact le_rfl

theorem mem_le_maxL : ∀ (l : List Rat) (m : Rat), ∀ a ∈ l, a ≤ maxL l m
  | [], _, a, ha => by simp at ha
  | b :: bs, m, a, ha => by
    simp only [maxL]
    rcases List.mem_cons.mp ha with rfl | ha'
    · refine le_trans ?_ (le_maxL_init bs _)
      split_ifs with hm
      · exact le_rfl
      · exact not_lt.mp hm
    · exact mem_le_maxL bs _ a ha'

theorem target_of_checkLowest {c : Cert} (h : checkLowest c = true) : ∀ p ∈ c.pairs, p.radArr ≤ c.target := by
  have h1 : maxL (c.pairs.map (·.radArr)) 0 ≤ c.target := by
    unfold checkLowest at h
    simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    exact h.1.1.1.1.2
  intro p hp
  exact le_trans (mem_le_maxL _ 0 p.radArr (List.mem_map.mpr ⟨p, hp, rfl⟩)) h1

/-- **Theorem 1 for a certificate accepted by `checkLowest`.** With the decoded matrix exactly symmetric and every
vector of the right length and nonzero, every claimed pair has an eigenvalue of the decoded matrix within its
existence radius, and the existence radius is at most the radius. -/
theorem checkLowest_theorem1 (c : Cert) (hsym : symOK c.A = true) (hvec : vecOK c = true)
    (h : checkLowest c = true) :
    ∀ p ∈ c.pairs, ∃ i, |(isHermitian_of_symOK hsym).eigenvalues i - (p.mu : ℝ)| ≤ (p.ex : ℝ) ∧
      (p.ex : ℝ) ≤ p.rad := by
  intro p hp
  have hv : p.x.length = c.A.length ∧ dotL p.x p.x ≠ 0 := by
    simp only [vecOK, List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hvec
    exact hvec p hp
  have hok := pairs_ok_of_checkLowest h p hp
  obtain ⟨i, hi⟩ := pairOK_theorem1 c p hv.1 hv.2 (isHermitian_of_symOK hsym) hok
  obtain ⟨-, -, -, -, hr, -, -⟩ := pairOK_spec hok
  exact ⟨i, hi, by exact_mod_cast hr⟩

open scoped Matrix.Norms.L2Operator in
/-- **The S2 conclusion for a certificate accepted by `checkLowest`.** For every symmetric `H` within the recorded
model shift of the decoded matrix in the spectral norm (the content of S1, taken here as a hypothesis), every
claimed pair has an eigenvalue of `H` within its exported radius, and the exported radius meets the accuracy
target. The decoded matrix itself need not be symmetric here. -/
theorem checkLowest_S2 (c : Cert) (hvec : vecOK c = true) (h : checkLowest c = true)
    {H : Matrix (Fin c.A.length) (Fin c.A.length) ℝ} (hH : H.IsHermitian)
    (hσ : ‖H - toMatrix c.A‖ ≤ (c.shift : ℝ)) :
    ∀ p ∈ c.pairs, ∃ i, |hH.eigenvalues i - (p.mu : ℝ)| ≤ (p.radArr : ℝ) ∧ (p.radArr : ℝ) ≤ c.target := by
  intro p hp
  have hv : p.x.length = c.A.length ∧ dotL p.x p.x ≠ 0 := by
    simp only [vecOK, List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hvec
    exact hvec p hp
  obtain ⟨i, hi⟩ := pairOK_theorem1_shift_rad c p hv.1 hv.2 (pairs_ok_of_checkLowest h p hp) hH hσ
  exact ⟨i, hi, by exact_mod_cast target_of_checkLowest h p hp⟩

end CertifyMathlib.Refine
