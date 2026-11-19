import CertifyMathlib.Counting.Inertia
import CertifyMathlib.Refine.Residual

/-!
# Refinement of an exact inertia count (N-05, second part)

The exact counts of `cecheck` and of `CertifyCore.K.negAtSigma` come from fraction-free (Bareiss) elimination of
`2^s (A - σ I)`, and the number of negative eigenvalues is read off by Sylvester's law with Jacobi's rule. This
file proves the mathematical core on Mathlib and a kernel-evaluable rational elimination with its refinement:

* `ldl_of_pivots`: if Gaussian elimination without pivoting runs through a symmetric real matrix `M` with the
  nonzero pivots `ps` (`Pivots`), then `M = L diag(ps) Lᵀ` with `det L = 1`;
* `cntLt_of_pivots`: hence (Sylvester's law, `CertifyMathlib.Counting.cntLt_eq_of_factor`) the number of
  eigenvalues of `A` below `σ` is the number of negative pivots of `A - σ I`;
* `gaussQ`, `pivots_of_gaussQ`: the elimination on lists of rationals, as the kernel can evaluate it, and the
  statement that its pivots are those of `Pivots` for the decoded matrix;
* `count_of_gaussQ`: the count for a kernel certificate: if `gaussQ` returns pivots for the decoded `A - σ I`,
  then `#{λ < σ}` of the decoded matrix is the number of negative ones.

Not proved here: that `CertifyCore.K.bareiss` (integer division, the fraction-free form) returns the same sign
count as `gaussQ`. The fraction-free pivots are `d_k = π_1 ⋯ π_k` exactly when every division is exact, which is
Sylvester's determinant identity; for kernel data the agreement can be evaluated instead.
-/

open Matrix

namespace CertifyMathlib.Refine

open CertifyMathlib.Counting

/-! ### Gaussian elimination on real matrices -/

/-- The Schur complement of the entry `(0, 0)`. -/
noncomputable def schur {n : Nat} (M : Matrix (Fin (n + 1)) (Fin (n + 1)) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => M i.succ j.succ - M i.succ 0 * M 0 j.succ / M 0 0

/-- `Pivots n M ps`: Gaussian elimination without pivoting runs through `M` with the nonzero pivots `ps`. -/
inductive Pivots : (n : Nat) → Matrix (Fin n) (Fin n) ℝ → List ℝ → Prop
  | nil (M : Matrix (Fin 0) (Fin 0) ℝ) : Pivots 0 M []
  | cons {n : Nat} (M : Matrix (Fin (n + 1)) (Fin (n + 1)) ℝ) (ps : List ℝ) (h0 : M 0 0 ≠ 0)
      (hs : Pivots n (schur M) ps) : Pivots (n + 1) M (M 0 0 :: ps)

/-- The factor of one elimination step: first row `(1, 0, ..., 0)`, first column `M i 0 / M 0 0`, `L'` below. -/
noncomputable def stepL {n : Nat} (M : Matrix (Fin (n + 1)) (Fin (n + 1)) ℝ) (L' : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin (n + 1)) (Fin (n + 1)) ℝ :=
  Matrix.of fun i j => Fin.cases (motive := fun _ => ℝ) (if j = 0 then 1 else 0)
    (fun i' => Fin.cases (motive := fun _ => ℝ) (M i'.succ 0 / M 0 0) (fun j' => L' i' j') j) i

section step
variable {n : Nat} (M : Matrix (Fin (n + 1)) (Fin (n + 1)) ℝ) (L' : Matrix (Fin n) (Fin n) ℝ)

theorem stepL_zero_zero : stepL M L' 0 0 = 1 := by simp [stepL]
theorem stepL_zero_succ (j : Fin n) : stepL M L' 0 j.succ = 0 := by simp [stepL, Fin.succ_ne_zero]
theorem stepL_succ_zero (i : Fin n) : stepL M L' i.succ 0 = M i.succ 0 / M 0 0 := by simp [stepL]
theorem stepL_succ_succ (i j : Fin n) : stepL M L' i.succ j.succ = L' i j := by simp [stepL]

theorem det_stepL : (stepL M L').det = L'.det := by
  rw [Matrix.det_succ_row_zero, Fin.sum_univ_succ]
  have h0 : (stepL M L').submatrix Fin.succ (Fin.succAbove (0 : Fin (n + 1))) = L' := by
    ext i j
    simp [Matrix.submatrix_apply, stepL_succ_succ]
  rw [h0]
  simp [stepL_zero_zero, stepL_zero_succ]

end step

theorem ldl_apply_real {m : Nat} (L : Matrix (Fin m) (Fin m) ℝ) (d : Fin m → ℝ) (i j : Fin m) :
    (L * (dg d : Matrix (Fin m) (Fin m) ℝ) * Lᴴ) i j = ∑ k, L i k * d k * L j k := by
  rw [Matrix.mul_apply]
  refine Finset.sum_congr rfl fun k _ => ?_
  simp [dg, Matrix.mul_diagonal, Matrix.conjTranspose_apply]

/-- **One elimination step per pivot gives `M = L diag(ps) Lᵀ` with `det L = 1`**, and the number of negative
entries of the diagonal is the number of negative pivots. -/
theorem ldl_of_pivots {n : Nat} {M : Matrix (Fin n) (Fin n) ℝ} {ps : List ℝ} (h : Pivots n M ps) :
    M.IsHermitian → ∃ (L : Matrix (Fin n) (Fin n) ℝ) (d : Fin n → ℝ), L.det = 1 ∧
      M = L * (dg d : Matrix (Fin n) (Fin n) ℝ) * Lᴴ ∧
      (Finset.univ.filter fun i => d i < 0).card = ps.countP (fun x => decide (x < 0)) := by
  induction h with
  | nil M =>
    intro _
    refine ⟨1, Fin.elim0, by simp, ?_, by simp⟩
    ext i
    exact i.elim0
  | cons M ps h0 hs ih =>
    intro hM
    have hsym : ∀ a b, M b a = M a b := fun a b => by simpa using hM.apply a b
    have hS : (schur M).IsHermitian := Matrix.IsHermitian.ext fun i j => by
      simp only [schur, Matrix.of_apply, star_trivial]
      rw [hsym i.succ j.succ, hsym 0 j.succ, hsym i.succ 0]
      ring
    obtain ⟨L', d', hdet, hfac, hcnt⟩ := ih hS
    have hfac' : ∀ a b, ∑ k, L' a k * d' k * L' b k = schur M a b := fun a b => by
      rw [← ldl_apply_real, ← hfac]
    refine ⟨stepL M L', (Fin.cons (M 0 0) d' : Fin _ → ℝ), by rw [det_stepL, hdet], ?_, ?_⟩
    · ext i j
      rw [ldl_apply_real, Fin.sum_univ_succ]
      cases i using Fin.cases with
      | zero =>
        cases j using Fin.cases with
        | zero => simp [stepL_zero_zero, stepL_zero_succ]
        | succ j' =>
          simp only [stepL_zero_zero, stepL_zero_succ, stepL_succ_zero, Fin.cons_zero, Fin.cons_succ, zero_mul,
            Finset.sum_const_zero, add_zero, one_mul]
          rw [hsym j'.succ 0]
          field_simp
      | succ i' =>
        cases j using Fin.cases with
        | zero =>
          simp only [stepL_zero_zero, stepL_zero_succ, stepL_succ_zero, Fin.cons_zero, Fin.cons_succ, mul_zero,
            Finset.sum_const_zero, add_zero, mul_one]
          field_simp
        | succ j' =>
          simp only [stepL_succ_zero, stepL_succ_succ, Fin.cons_zero, Fin.cons_succ]
          rw [hfac' i' j']
          simp only [schur, Matrix.of_apply]
          rw [hsym j'.succ 0]
          field_simp
          ring
    · rw [Fin.card_filter_univ_succ']
      simp only [Fin.cons_zero, Fin.cons_succ, hcnt, List.countP_cons, decide_eq_true_eq]
      split_ifs <;> omega

/-- **Sylvester's law for Gaussian elimination**: the number of eigenvalues of `A` below `σ` is the number of
negative pivots of `A - σ I`. -/
theorem cntLt_of_pivots {n : Nat} {A : Matrix (Fin n) (Fin n) ℝ} (hA : A.IsHermitian) (σ : ℝ) {ps : List ℝ}
    (h : Pivots n (A - σ • (1 : Matrix (Fin n) (Fin n) ℝ)) ps) :
    cntLt hA σ = ps.countP (fun x => decide (x < 0)) := by
  have hM : (A - σ • (1 : Matrix (Fin n) (Fin n) ℝ)).IsHermitian := by
    refine Matrix.IsHermitian.ext fun i j => ?_
    have hij := hA.apply i j
    simp only [star_trivial] at hij ⊢
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply, hij, smul_eq_mul]
    by_cases e : i = j
    · subst e
      rfl
    · simp [e, Ne.symm e]
  obtain ⟨L, d, hdet, hfac, hcnt⟩ := ldl_of_pivots h hM
  have hL : IsUnit L := (Matrix.isUnit_iff_isUnit_det L).mpr (by rw [hdet]; exact isUnit_one)
  have hd' := isUnit_det_conjTranspose hL
  have key := (cntLt_eq_of_factor hA σ L (Lᴴ)⁻¹ (mul_nonsing_inv _ hd') (nonsing_inv_mul _ hd') d
    (by simpa using hfac)).1
  rw [key, hcnt]

/-! ### The elimination on lists, as the kernel evaluates it -/

/-- The elimination step on the rows below the first. -/
def elimRow (p : Rat) (tl0 : List Rat) : List Rat → List Rat
  | [] => []
  | a0 :: tl => List.zipWith (fun a b => a - a0 * b / p) tl tl0

/-- Gaussian elimination without pivoting on the rows of an `n × n` rational matrix: the pivots, or `none` when a
pivot is zero or the shape is wrong. -/
def gaussQ : Nat → List (List Rat) → Option (List Rat)
  | 0, rows => if rows.isEmpty then some [] else none
  | _ + 1, [] => none
  | n + 1, row0 :: rest =>
    match row0 with
    | [] => none
    | p :: tl0 =>
      if p = 0 then none else
      (gaussQ n (rest.map (elimRow p tl0))).map (p :: ·)

/-- An `n × n` list of rows. -/
def Square (n : Nat) (rows : List (List Rat)) : Prop := rows.length = n ∧ ∀ r ∈ rows, r.length = n

/-- Row `i` of a list of rows (`[]` beyond the end). -/
def rowAt : List (List Rat) → Nat → List Rat
  | [], _ => []
  | r :: _, 0 => r
  | _ :: rs, i + 1 => rowAt rs i

/-- The real matrix of a list of rows. -/
noncomputable def ofRows (n : Nat) (rows : List (List Rat)) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j => ((CertifyCore.K.getR (rowAt rows i) j : Rat) : ℝ)

theorem rowAt_eq_getD (rows : List (List Rat)) (i : Nat) : rowAt rows i = rows.getD i [] := by
  induction rows generalizing i with
  | nil => simp [rowAt]
  | cons a as ih =>
    cases i with
    | zero => simp [rowAt]
    | succ i => simp [rowAt, ih]

theorem rowAt_mem : ∀ (rows : List (List Rat)) (i : Nat), i < rows.length → rowAt rows i ∈ rows
  | _ :: _, 0, _ => by simp [rowAt]
  | _ :: rs, i + 1, h => by
    simp only [rowAt]
    exact List.mem_cons_of_mem _ (rowAt_mem rs i (by simpa using h))
  | [], _, h => by simp at h

theorem rowAt_map (g : List Rat → List Rat) : ∀ (rows : List (List Rat)) (i : Nat), i < rows.length →
    rowAt (rows.map g) i = g (rowAt rows i)
  | _ :: _, 0, _ => by simp [rowAt]
  | _ :: rs, i + 1, h => by
    simp only [List.map_cons, rowAt]
    exact rowAt_map g rs i (by simpa using h)
  | [], _, h => by simp at h

theorem gaussQ_cons (n : Nat) (p : Rat) (tl0 : List Rat) (rest : List (List Rat)) :
    gaussQ (n + 1) ((p :: tl0) :: rest) =
      if p = 0 then none else (gaussQ n (rest.map (elimRow p tl0))).map (p :: ·) := by
  simp only [gaussQ]

/-- **Refinement of the list elimination**: its pivots are those of `Pivots` for the decoded matrix. -/
theorem pivots_of_gaussQ : ∀ (n : Nat) (rows : List (List Rat)) (ps : List Rat), Square n rows →
    gaussQ n rows = some ps → Pivots n (ofRows n rows) (ps.map fun q : Rat => (q : ℝ))
  | 0, rows, ps, _, h => by
    have hps : ps = [] := by
      simp only [gaussQ] at h
      split_ifs at h
      · exact (Option.some.inj h).symm
    subst hps
    exact Pivots.nil _
  | n + 1, [], ps, _, h => by simp [gaussQ] at h
  | n + 1, [] :: rest, ps, _, h => by simp [gaussQ] at h
  | n + 1, (p :: tl0) :: rest, ps, hsq, h => by
    rw [gaussQ_cons] at h
    split_ifs at h with hp
    cases hg : gaussQ n (rest.map (elimRow p tl0)) with
    | none => rw [hg] at h; simp at h
    | some ps' =>
      rw [hg] at h
      simp only [Option.map_some, Option.some.injEq] at h
      subst h
      obtain ⟨hlen, hrow⟩ := hsq
      have hrest : rest.length = n := by simpa using hlen
      have htl0 : tl0.length = n := by
        have := hrow (p :: tl0) (by simp)
        simpa using this
      have hrow' : ∀ r ∈ rest, r.length = n + 1 := fun r hr => hrow r (by simp [hr])
      -- the rows below the first after one step form an n × n list
      have hsq' : Square n (rest.map (elimRow p tl0)) := by
        refine ⟨by simp [hrest], fun r hr => ?_⟩
        obtain ⟨r0, hr0, rfl⟩ := List.mem_map.mp hr
        have hl := hrow' r0 hr0
        match r0, hl with
        | a0 :: tl, hl =>
          simp only [elimRow, List.length_zipWith, htl0]
          simp at hl
          omega
      have ih := pivots_of_gaussQ n _ ps' hsq' hg
      have hM00 : ofRows (n + 1) ((p :: tl0) :: rest) 0 0 = (p : ℝ) := by
        simp [ofRows, rowAt, CertifyCore.K.getR]
      have hschur : schur (ofRows (n + 1) ((p :: tl0) :: rest)) = ofRows n (rest.map (elimRow p tl0)) := by
        ext i j
        have hi : (i : Nat) < rest.length := by rw [hrest]; exact i.isLt
        have hmem := hrow' _ (rowAt_mem rest i hi)
        obtain ⟨a0, tl, hat⟩ : ∃ a0 tl, rowAt rest i = a0 :: tl := by
          match h' : rowAt rest i with
          | [] => rw [h'] at hmem; simp at hmem
          | a0 :: tl => exact ⟨a0, tl, rfl⟩
        have htl : tl.length = n := by rw [hat] at hmem; simpa using hmem
        have hj1 : (j : Nat) < tl.length := by rw [htl]; exact j.isLt
        have hj2 : (j : Nat) < tl0.length := by rw [htl0]; exact j.isLt
        simp only [schur, ofRows, Matrix.of_apply, Fin.val_succ, Fin.val_zero]
        simp only [rowAt]
        rw [rowAt_map _ rest i hi, hat]
        simp only [elimRow, CertifyCore.K.getR]
        rw [getR_zipWith _ tl tl0 j hj1 hj2]
        push_cast
        ring
      have hp0 : ofRows (n + 1) ((p :: tl0) :: rest) 0 0 ≠ 0 := by rw [hM00]; exact_mod_cast hp
      have key := Pivots.cons (ofRows (n + 1) ((p :: tl0) :: rest)) _ hp0 (hschur ▸ ih)
      rw [hM00] at key
      simpa using key

/-- The rows of `A - σ I` for a kernel certificate's matrix, as `CertifyCore.K.shiftedInt` forms them before
scaling. -/
def denseRows (A : List (List (Nat × Rat))) (σ : Rat) : List (List Rat) :=
  (List.range A.length).map fun i => CertifyCore.K.denseRow A.length (A.getD i []) i σ

theorem square_denseRows (A : List (List (Nat × Rat))) (σ : Rat) : Square A.length (denseRows A σ) := by
  refine ⟨by simp [denseRows], fun r hr => ?_⟩
  simp only [denseRows, List.mem_map, List.mem_range] at hr
  obtain ⟨i, _, rfl⟩ := hr
  simp [CertifyCore.K.denseRow]

theorem ofRows_denseRows (A : List (List (Nat × Rat))) (σ : Rat) :
    ofRows A.length (denseRows A σ) = toMatrix A - (σ : ℝ) • (1 : Matrix (Fin A.length) (Fin A.length) ℝ) := by
  ext i j
  simp only [ofRows, Matrix.of_apply, rowAt_eq_getD, getR_eq_getD, denseRows, CertifyCore.K.denseRow,
    List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range, i.isLt, j.isLt, Option.map_some,
    Option.getD_some, toMatrix, Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply, smul_eq_mul]
  by_cases e : i = j
  · subst e
    simp [entryQ]
  · have e' : (i : Nat) ≠ (j : Nat) := fun h => e (Fin.ext h)
    simp [entryQ, e, e']

/-- **The exact count for a kernel certificate's matrix**: if the list elimination of the decoded `A - σ I`
returns pivots, the number of eigenvalues of the decoded (symmetric) matrix below `σ` is the number of negative
pivots. -/
theorem count_of_gaussQ (A : List (List (Nat × Rat))) (hsym : symOK A = true) (σ : Rat) {ps : List Rat}
    (h : gaussQ A.length (denseRows A σ) = some ps) :
    cntLt (isHermitian_of_symOK hsym) (σ : ℝ) = ps.countP (fun x => decide (x < 0)) := by
  have hp := pivots_of_gaussQ _ _ ps (square_denseRows A σ) h
  rw [ofRows_denseRows] at hp
  rw [cntLt_of_pivots _ _ hp]
  simp [List.countP_map, Function.comp_def]

end CertifyMathlib.Refine
