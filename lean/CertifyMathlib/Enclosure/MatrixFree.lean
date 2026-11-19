import CertifyMathlib.Enclosure.FloatAccounting
import Mathlib.LinearAlgebra.Matrix.Kronecker

/-!
# D-02: matrix-free products from the spec

`docs/proofs/D-02_matrix-free.md`. A spec operator is applied term by term: for every contribution
to an output entry `i` the code forms the amplitude `c_t · a₁ ⋯ a_L` by a chain of rounded products
from the exact coupling and the stored local amplitudes (exact, or rounded once), multiplies by the
source entry `x_j`, and adds the contributions to `y_i` in some order.

* `chain_err`: a chain `acc ← op(acc, f̂_l)` whose operation has relative error `γ_m` (`m = 1` for
  a real product, `m = 3` for a complex one, `cmul_err`) and whose stored factors have relative
  errors `γ_{e_l}` carries `γ_{k + Σ(m + e_l)}`;
* **Theorem D-02 (a)** (`matrixfree_entry_error`, `matrixfree_entry_error_c`): with at most `w_row`
  contributions per output entry, at most `L` letters per string and at most `ρ` rounded amplitudes
  per string, `|fl(H x)_i - (H_spec x)_i| ≤ γ_{W-1} (|H_spec|_abs |x|)_i` with
  `W = w_row + L + ρ + 1` (real data) or `W = w_row + 3L + ρ + 3` (complex data);
* Lemma N: `‖(A ⊗ B) x‖ ≤ ‖A‖ ‖B‖ ‖x‖` for the Kronecker product (`norm_kronecker_le`, from
  `A ⊗ B = (A ⊗ 1)(1 ⊗ B)`), a diagonal unitary sign factor preserves norms
  (`norm_diagonal_unit`), and the term-wise triangle bound `‖Σ_t c_t O_t‖ ≤ Σ_t |c_t| ‖O_t‖`
  (`norm_sum_terms_le`, `norm_list_prod_le`).

* **Theorem D-02 (b)** (`d02b_radius`, real data): the radius of `residuals` evaluated with
  `adapter.width = W - 2` bounds `‖H_spec x - μ x‖ / ‖x‖`, given (a) and the second pass of
  Lemma M3 as a hypothesis. The accounting has no spare `γ` index: the residual error index
  `γ_{W-1} + u (1 + γ_{W-1})` is covered by `γ_W` and the code's rounding factors only with a
  margin of second order in `u` (`d02b_key`, reduced to the polynomial inequality `d02b_poly`).

The count of contributions per output entry (Lemma M1) is a hypothesis (the size of the summation
tree).
-/

open Finset Matrix

namespace CertifyMathlib.Enclosure.FP

section Chain

variable {𝕂 : Type*} [RCLike 𝕂]

/-- A chain of products `acc ← op(acc, f_l)`. -/
def chainFl (op : 𝕂 → 𝕂 → 𝕂) : 𝕂 → List 𝕂 → 𝕂
  | c, [] => c
  | c, f :: fs => chainFl op (op c f) fs

/-- The rounding index of a chain: `Σ (m + e_l)`. -/
def chainIdx (m : ℕ) (fs : List (𝕂 × 𝕂 × ℕ)) : ℕ := (fs.map fun p => m + p.2.2).sum

/-- **A chain of products.** Each step `op(a, b)` has relative error `γ_m`; the stored factor `f̂_l`
of the exact `f_l` has relative error `γ_{e_l}`. -/
theorem chain_err {u : ℝ} (hu : 0 ≤ u) (op : 𝕂 → 𝕂 → 𝕂) (m : ℕ)
    (hop : ∀ a b, ‖op a b - a * b‖ ≤ gam u m * ‖a * b‖) :
    ∀ (fs : List (𝕂 × 𝕂 × ℕ)) (ch c : 𝕂) (k : ℕ), ‖ch - c‖ ≤ gam u k * ‖c‖ →
      (∀ p ∈ fs, ‖p.1 - p.2.1‖ ≤ gam u p.2.2 * ‖p.2.1‖) →
      ((k + chainIdx m fs : ℕ) : ℝ) * u < 1 →
      ‖chainFl op ch (fs.map Prod.fst) - c * (fs.map fun p => p.2.1).prod‖ ≤
        gam u (k + chainIdx m fs) * ‖c * (fs.map fun p => p.2.1).prod‖
  | [], ch, c, k, h, _, _ => by simpa [chainFl, chainIdx] using h
  | p :: fs, ch, c, k, h, hp, hk => by
    obtain ⟨fh, f, e⟩ := p
    have hidx : k + chainIdx m ((fh, f, e) :: fs) = k + e + m + chainIdx m fs := by
      simp only [chainIdx, List.map_cons, List.sum_cons]
      omega
    have hk' : ((k + e + m + chainIdx m fs : ℕ) : ℝ) * u < 1 := by
      rw [← hidx]
      exact hk
    have hke : ((k + e : ℕ) : ℝ) * u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : k + e ≤ k + e + m + chainIdx m fs))
        hu) hk'
    have hkem : ((k + e + m : ℕ) : ℝ) * u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right
        (by exact_mod_cast (by omega : k + e + m ≤ k + e + m + chainIdx m fs)) hu) hk'
    have hf : ‖fh - f‖ ≤ gam u e * ‖f‖ := hp (fh, f, e) List.mem_cons_self
    have hm := rel_mul hu (j := k) (k := e) hke h hf
    have hg0 : 0 ≤ gam u (k + e) := gam_nonneg hu hke
    have hm0 : 0 ≤ gam u m := gam_nonneg hu (lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : m ≤ k + e + m)) hu) hkem)
    have hadd : gam u (k + e) + gam u m + gam u (k + e) * gam u m ≤ gam u (k + e + m) :=
      gam_add hu hkem
    have hstep : ‖op ch fh - c * f‖ ≤ gam u (k + e + m) * ‖c * f‖ := by
      have h1 := hop ch fh
      have h2 : ‖ch * fh‖ ≤ (1 + gam u (k + e)) * ‖c * f‖ := by
        have := norm_le_add_norm_sub (ch * fh) (c * f)
        nlinarith
      calc ‖op ch fh - c * f‖ = ‖(op ch fh - ch * fh) + (ch * fh - c * f)‖ := by
            congr 1
            abel
        _ ≤ ‖op ch fh - ch * fh‖ + ‖ch * fh - c * f‖ := norm_add_le _ _
        _ ≤ gam u m * ((1 + gam u (k + e)) * ‖c * f‖) + gam u (k + e) * ‖c * f‖ :=
            add_le_add (h1.trans (mul_le_mul_of_nonneg_left h2 hm0)) hm
        _ ≤ gam u (k + e + m) * ‖c * f‖ := by
            nlinarith [mul_le_mul_of_nonneg_right hadd (norm_nonneg (c * f))]
    have ih := chain_err hu op m hop fs (op ch fh) (c * f) (k + e + m) hstep
      (fun q hq => hp q (List.mem_cons_of_mem _ hq)) hk'
    simp only [List.map_cons, chainFl, List.prod_cons]
    rw [← mul_assoc, hidx]
    exact ih

/-- A sum of computed terms along a tree with at most `wr` leaves, every term with relative
error `γ_a`: `γ_{a + wr - 1}`. -/
theorem tree_entry_error (R : Rounding 𝕂) (h0 : R.eta = 0) {wr a : ℕ}
    (hW : ((a + wr : ℕ) : ℝ) * R.u < 1) {κ : Type*} [DecidableEq κ] (t th : κ → 𝕂)
    (S : Finset κ) (hth : ∀ k ∈ S, ‖th k - t k‖ ≤ gam R.u a * ‖t k‖) {h : 𝕂}
    (hb : EntryBuilt R wr th S h) :
    ‖h - ∑ k ∈ S, t k‖ ≤ gam R.u (a + wr - 1) * ∑ k ∈ S, ‖t k‖ := by
  have hu := R.u_nonneg
  rcases hb with ⟨hS, hh⟩ | ⟨T, hnd, hmem, hlen, hh⟩
  · subst hS
    subst hh
    simp
  · have hW1 : ((a + wr - 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : a + wr - 1 ≤ a + wr)) hu) hW
    have hT : ((a + T.adds : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : a + T.adds ≤ a + wr)) hu) hW
    have ha : ((a : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : a ≤ a + wr)) hu) hW
    let t0 : κ → 𝕂 := fun k => if k ∈ S then t k else th k
    have hleaf : ∀ k, ‖th k - t0 k‖ ≤ gam R.u a * ‖t0 k‖ + 0 := by
      intro k
      by_cases hk : k ∈ S
      · simp only [t0, hk, ↓reduceIte, add_zero]
        exact hth k hk
      · simp only [t0, hk, ↓reduceIte, sub_self, norm_zero, add_zero]
        exact mul_nonneg (gam_nonneg hu ha) (norm_nonneg _)
    have herr := STree.comp_error (R.std h0) le_rfl t0 th hleaf (fun _ => trivial) T hT
    have hfs : T.leaves.toFinset = S := Finset.ext fun k => by simp [hmem]
    have he : T.eval t0 = ∑ k ∈ S, t k := by
      rw [STree.eval_eq_sum, ← List.sum_toFinset _ hnd, hfs]
      exact Finset.sum_congr rfl fun k hk => by simp [t0, hk]
    have hae : T.absEval t0 = ∑ k ∈ S, ‖t k‖ := by
      rw [STree.absEval_eq_sum, ← List.sum_toFinset _ hnd, hfs]
      exact Finset.sum_congr rfl fun k hk => by simp [t0, hk]
    have hg : gam R.u (a + T.adds) ≤ gam R.u (a + wr - 1) := gam_mono hu (by omega) hW1
    have hs0 : 0 ≤ ∑ k ∈ S, ‖t k‖ := Finset.sum_nonneg fun k _ => norm_nonneg _
    rw [hh]
    calc ‖T.comp R.rnd th - ∑ k ∈ S, t k‖ = ‖T.comp (R.std h0).rnd th - T.eval t0‖ := by
          rw [he]
          rfl
      _ ≤ gam R.u (a + T.adds) * T.absEval t0 +
            ((T.adds : ℝ) + 1) * (1 + gam R.u T.adds) * 0 := herr
      _ ≤ gam R.u (a + wr - 1) * ∑ k ∈ S, ‖t k‖ := by
          rw [hae, mul_zero, add_zero]
          exact mul_le_mul_of_nonneg_right hg hs0

end Chain

/-! ### Theorem D-02 (a) -/

section Entry

variable {κ : Type*} [DecidableEq κ]

/-- The computed contribution of a monomial: the chain of rounded products from the exact coupling
`c`, then the product with the source entry `x`. -/
noncomputable def contribFl (R : Rounding ℝ) (c : ℝ) (fs : List (ℝ × ℝ × ℕ)) (x : ℝ) : ℝ :=
  R.rnd (chainFl (fun a b => R.rnd (a * b)) c (fs.map Prod.fst) * x)

/-- The exact contribution `c f₁ ⋯ f_L x`. -/
def contribEx (c : ℝ) (fs : List (ℝ × ℝ × ℕ)) (x : ℝ) : ℝ := c * (fs.map fun p => p.2.1).prod * x

theorem rnd_mul_rel (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u < 1) (a b : ℝ) :
    ‖R.rnd (a * b) - a * b‖ ≤ gam R.u 1 * ‖a * b‖ := by
  have := R.err (a * b)
  rw [h0, add_zero] at this
  exact this.trans (mul_le_mul_of_nonneg_right (u_le_gam_one R.u_nonneg hu1) (norm_nonneg _))

/-- **Theorem D-02 (a), real data.** Each contribution to the output entry carries at most `L`
products and `ρ` rounded amplitudes, there are at most `w_row` contributions, so
`|fl(H x)_i - (H_spec x)_i| ≤ γ_{w_row + L + ρ} (|H_spec|_abs |x|)_i = γ_{W-1} (…)`. -/
theorem matrixfree_entry_error (R : Rounding ℝ) (h0 : R.eta = 0) {wr L ρ : ℕ} (hwr : 1 ≤ wr)
    (hW : ((wr + L + ρ + 1 : ℕ) : ℝ) * R.u < 1) (S : Finset κ) (c : κ → ℝ)
    (fs : κ → List (ℝ × ℝ × ℕ)) (x : κ → ℝ)
    (hfac : ∀ k ∈ S, ∀ p ∈ fs k, ‖p.1 - p.2.1‖ ≤ gam R.u p.2.2 * ‖p.2.1‖)
    (hidx : ∀ k ∈ S, chainIdx 1 (fs k) ≤ L + ρ) {h : ℝ}
    (hb : EntryBuilt R wr (fun k => contribFl R (c k) (fs k) (x k)) S h) :
    ‖h - ∑ k ∈ S, contribEx (c k) (fs k) (x k)‖ ≤
      gam R.u (wr + L + ρ) * ∑ k ∈ S, ‖contribEx (c k) (fs k) (x k)‖ := by
  have hu := R.u_nonneg
  have hW' : ((L + ρ + 1 + wr : ℕ) : ℝ) * R.u < 1 := by
    have e : L + ρ + 1 + wr = wr + L + ρ + 1 := by omega
    rw [e]
    exact hW
  have hu1 : R.u < 1 := lt_of_le_of_lt
    (le_mul_of_one_le_left hu (by exact_mod_cast (by omega : 1 ≤ L + ρ + 1 + wr))) hW'
  have hterm : ∀ k ∈ S, ‖contribFl R (c k) (fs k) (x k) - contribEx (c k) (fs k) (x k)‖ ≤
      gam R.u (L + ρ + 1) * ‖contribEx (c k) (fs k) (x k)‖ := by
    intro k hk
    have hik : ((0 + chainIdx 1 (fs k) : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by have := hidx k hk; omega :
        0 + chainIdx 1 (fs k) ≤ L + ρ + 1 + wr)) hu) hW'
    have hch := chain_err hu (fun a b => R.rnd (a * b)) 1 (rnd_mul_rel R h0 hu1) (fs k) (c k) (c k) 0
      (by simp [gam_zero]) (hfac k hk) hik
    have hx : ‖x k - x k‖ ≤ gam R.u 0 * ‖x k‖ := by simp [gam_zero]
    have hik1 : ((0 + chainIdx 1 (fs k) + 0 : ℕ) : ℝ) * R.u < 1 := by simpa using hik
    have hm := rel_mul hu (j := 0 + chainIdx 1 (fs k)) (k := 0) hik1 hch hx
    have hik2 : ((0 + chainIdx 1 (fs k) + 0 + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by have := hidx k hk; omega :
        0 + chainIdx 1 (fs k) + 0 + 1 ≤ L + ρ + 1 + wr)) hu) hW'
    have hr := rel_rnd R h0 (k := 0 + chainIdx 1 (fs k) + 0) hik2 hm
    have hLr : ((L + ρ + 1 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : L + ρ + 1 ≤ L + ρ + 1 + wr)) hu)
      hW'
    have hg := gam_mono hu (j := 0 + chainIdx 1 (fs k) + 0 + 1) (k := L + ρ + 1)
      (by have := hidx k hk; omega) hLr
    exact hr.trans (mul_le_mul_of_nonneg_right hg (norm_nonneg _))
  have := tree_entry_error R h0 (a := L + ρ + 1) hW' (fun k => contribEx (c k) (fs k) (x k))
    (fun k => contribFl R (c k) (fs k) (x k)) S hterm hb
  have e : L + ρ + 1 + wr - 1 = wr + L + ρ := by omega
  rwa [e] at this

/-- The computed contribution of a monomial, complex data (complex products `cmul`). -/
noncomputable def contribFlC (R : Rounding ℝ) (c : ℂ) (fs : List (ℂ × ℂ × ℕ)) (x : ℂ) : ℂ :=
  cmul R (chainFl (cmul R) c (fs.map Prod.fst)) x

/-- The exact contribution, complex data. -/
def contribExC (c : ℂ) (fs : List (ℂ × ℂ × ℕ)) (x : ℂ) : ℂ := c * (fs.map fun p => p.2.1).prod * x

/-- **Theorem D-02 (a), complex data**: `γ_{w_row + 3L + ρ + 2} = γ_{W-1}`. -/
theorem matrixfree_entry_error_c (R : Rounding ℝ) (h0 : R.eta = 0) (hu1 : R.u ≤ 1 / 1000)
    {wr L ρ : ℕ} (hwr : 1 ≤ wr) (hW : ((wr + 3 * L + ρ + 3 : ℕ) : ℝ) * R.u < 1) (S : Finset κ)
    (c : κ → ℂ) (fs : κ → List (ℂ × ℂ × ℕ)) (x : κ → ℂ)
    (hfac : ∀ k ∈ S, ∀ p ∈ fs k, ‖p.1 - p.2.1‖ ≤ gam R.u p.2.2 * ‖p.2.1‖)
    (hidx : ∀ k ∈ S, chainIdx 3 (fs k) ≤ 3 * L + ρ) {h : ℂ}
    (hb : EntryBuilt (cRounding R h0) wr (fun k => contribFlC R (c k) (fs k) (x k)) S h) :
    ‖h - ∑ k ∈ S, contribExC (c k) (fs k) (x k)‖ ≤
      gam R.u (wr + 3 * L + ρ + 2) * ∑ k ∈ S, ‖contribExC (c k) (fs k) (x k)‖ := by
  have hu := R.u_nonneg
  have hW' : ((3 * L + ρ + 3 + wr : ℕ) : ℝ) * R.u < 1 := by
    have e : 3 * L + ρ + 3 + wr = wr + 3 * L + ρ + 3 := by omega
    rw [e]
    exact hW
  have hop : ∀ a b : ℂ, ‖cmul R a b - a * b‖ ≤ gam R.u 3 * ‖a * b‖ := cmul_err R h0 hu1
  have hterm : ∀ k ∈ S, ‖contribFlC R (c k) (fs k) (x k) - contribExC (c k) (fs k) (x k)‖ ≤
      gam R.u (3 * L + ρ + 3) * ‖contribExC (c k) (fs k) (x k)‖ := by
    intro k hk
    have hik : ((0 + chainIdx 3 (fs k) : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by have := hidx k hk; omega :
        0 + chainIdx 3 (fs k) ≤ 3 * L + ρ + 3 + wr)) hu) hW'
    have hch := chain_err hu (cmul R) 3 hop (fs k) (c k) (c k) 0 (by simp [gam_zero]) (hfac k hk) hik
    have hik3 : ((0 + chainIdx 3 (fs k) + 3 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by have := hidx k hk; omega :
        0 + chainIdx 3 (fs k) + 3 ≤ 3 * L + ρ + 3 + wr)) hu) hW'
    have hm0 : 0 ≤ gam R.u 3 := gam_nonneg hu (lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 3 ≤ 0 + chainIdx 3 (fs k) + 3)) hu)
      hik3)
    have hg0 : 0 ≤ gam R.u (0 + chainIdx 3 (fs k)) := gam_nonneg hu hik
    have hadd := gam_add hu (j := 0 + chainIdx 3 (fs k)) (k := 3) hik3
    set A := chainFl (cmul R) (c k) ((fs k).map Prod.fst)
    set a := c k * ((fs k).map fun p => p.2.1).prod
    have hA : ‖A * x k‖ ≤ (1 + gam R.u (0 + chainIdx 3 (fs k))) * ‖a * x k‖ := by
      have h1 : ‖A * x k - a * x k‖ ≤ gam R.u (0 + chainIdx 3 (fs k)) * ‖a * x k‖ := by
        rw [← sub_mul, norm_mul, norm_mul, ← mul_assoc]
        exact mul_le_mul_of_nonneg_right hch (norm_nonneg _)
      have := norm_le_add_norm_sub (A * x k) (a * x k)
      linarith
    have hstep : ‖cmul R A (x k) - a * x k‖ ≤ gam R.u (0 + chainIdx 3 (fs k) + 3) * ‖a * x k‖ := by
      have h1 : ‖A * x k - a * x k‖ ≤ gam R.u (0 + chainIdx 3 (fs k)) * ‖a * x k‖ := by
        rw [← sub_mul, norm_mul, norm_mul, ← mul_assoc]
        exact mul_le_mul_of_nonneg_right hch (norm_nonneg _)
      calc ‖cmul R A (x k) - a * x k‖ = ‖(cmul R A (x k) - A * x k) + (A * x k - a * x k)‖ := by
            congr 1
            abel
        _ ≤ ‖cmul R A (x k) - A * x k‖ + ‖A * x k - a * x k‖ := norm_add_le _ _
        _ ≤ gam R.u 3 * ((1 + gam R.u (0 + chainIdx 3 (fs k))) * ‖a * x k‖) +
              gam R.u (0 + chainIdx 3 (fs k)) * ‖a * x k‖ :=
            add_le_add ((hop A (x k)).trans (mul_le_mul_of_nonneg_left hA hm0)) h1
        _ ≤ gam R.u (0 + chainIdx 3 (fs k) + 3) * ‖a * x k‖ := by
            nlinarith [mul_le_mul_of_nonneg_right hadd (norm_nonneg (a * x k))]
    have hLr : ((3 * L + ρ + 3 : ℕ) : ℝ) * R.u < 1 := lt_of_le_of_lt
      (mul_le_mul_of_nonneg_right
        (by exact_mod_cast (by omega : 3 * L + ρ + 3 ≤ 3 * L + ρ + 3 + wr)) hu) hW'
    have hg := gam_mono hu (j := 0 + chainIdx 3 (fs k) + 3) (k := 3 * L + ρ + 3)
      (by have := hidx k hk; omega) hLr
    exact hstep.trans (mul_le_mul_of_nonneg_right hg (norm_nonneg _))
  have := tree_entry_error (cRounding R h0) rfl (a := 3 * L + ρ + 3) hW'
    (fun k => contribExC (c k) (fs k) (x k)) (fun k => contribFlC R (c k) (fs k) (x k)) S hterm hb
  have e : 3 * L + ρ + 3 + wr - 1 = wr + 3 * L + ρ + 2 := by omega
  rwa [e] at this

end Entry

/-! ### Lemma N: norms of Kronecker products and of sums of terms -/

section KronNorm

variable {𝕂 : Type*} [RCLike 𝕂] {m p : Type*} [Fintype m] [Fintype p] [DecidableEq m]
  [DecidableEq p]

open scoped Kronecker

omit [DecidableEq m] [DecidableEq p] in
theorem norm_sq_prod_eq (x : EuclideanSpace 𝕂 (m × p)) :
    ‖x‖ ^ 2 = ∑ k, ‖(WithLp.toLp 2 (fun j => x (j, k)) : EuclideanSpace 𝕂 m)‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, Fintype.sum_prod_type_right]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [EuclideanSpace.norm_sq_eq]

omit [DecidableEq m] [DecidableEq p] in
theorem norm_sq_prod_eq' (x : EuclideanSpace 𝕂 (m × p)) :
    ‖x‖ ^ 2 = ∑ j, ‖(WithLp.toLp 2 (fun k => x (j, k)) : EuclideanSpace 𝕂 p)‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [EuclideanSpace.norm_sq_eq]

/-- `‖(A ⊗ 1) x‖ ≤ a ‖x‖` when `‖A v‖ ≤ a ‖v‖`. -/
theorem norm_kron_one_le (A : Matrix m m 𝕂) {a : ℝ} (ha : 0 ≤ a)
    (hA : ∀ v, ‖toEuclideanLin A v‖ ≤ a * ‖v‖) (x : EuclideanSpace 𝕂 (m × p)) :
    ‖toEuclideanLin (A ⊗ₖ (1 : Matrix p p 𝕂)) x‖ ≤ a * ‖x‖ := by
  let sl : p → EuclideanSpace 𝕂 m := fun k => WithLp.toLp 2 (fun j => x (j, k))
  have hcomp : ∀ i k, (toEuclideanLin (A ⊗ₖ (1 : Matrix p p 𝕂)) x) (i, k) =
      (toEuclideanLin A (sl k)) i := by
    intro i k
    rw [Matrix.toLpLin_apply, Matrix.toLpLin_apply]
    simp [sl, Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kroneckerMap_apply,
      Matrix.one_apply]
  have hy := norm_sq_prod_eq (toEuclideanLin (A ⊗ₖ (1 : Matrix p p 𝕂)) x)
  have hx := norm_sq_prod_eq x
  have hsl : ∀ k, (WithLp.toLp 2 (fun j => (toEuclideanLin (A ⊗ₖ (1 : Matrix p p 𝕂)) x) (j, k)) :
      EuclideanSpace 𝕂 m) = toEuclideanLin A (sl k) := fun k => by
    refine PiLp.ext fun i => ?_
    exact hcomp i k
  simp only [hsl] at hy
  have hsq : ‖toEuclideanLin (A ⊗ₖ (1 : Matrix p p 𝕂)) x‖ ^ 2 ≤ (a * ‖x‖) ^ 2 := by
    rw [hy, mul_pow, hx, Finset.mul_sum]
    refine Finset.sum_le_sum fun k _ => ?_
    rw [← mul_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) (hA (sl k)) 2
  exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg ha (norm_nonneg _))).mp hsq

/-- `‖(1 ⊗ B) x‖ ≤ b ‖x‖` when `‖B v‖ ≤ b ‖v‖`. -/
theorem norm_one_kron_le (B : Matrix p p 𝕂) {b : ℝ} (hb : 0 ≤ b)
    (hB : ∀ v, ‖toEuclideanLin B v‖ ≤ b * ‖v‖) (x : EuclideanSpace 𝕂 (m × p)) :
    ‖toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x‖ ≤ b * ‖x‖ := by
  let sl : m → EuclideanSpace 𝕂 p := fun j => WithLp.toLp 2 (fun k => x (j, k))
  have hcomp : ∀ j k, (toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x) (j, k) =
      (toEuclideanLin B (sl j)) k := by
    intro j k
    rw [Matrix.toLpLin_apply, Matrix.toLpLin_apply]
    simp [sl, Matrix.mulVec, dotProduct, Fintype.sum_prod_type, Matrix.kroneckerMap_apply,
      Matrix.one_apply]
  have hy := norm_sq_prod_eq' (toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x)
  have hx := norm_sq_prod_eq' x
  have hsl : ∀ j, (WithLp.toLp 2 (fun k => (toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x) (j, k)) :
      EuclideanSpace 𝕂 p) = toEuclideanLin B (sl j) := fun j => by
    refine PiLp.ext fun k => ?_
    exact hcomp j k
  simp only [hsl] at hy
  have hsq : ‖toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x‖ ^ 2 ≤ (b * ‖x‖) ^ 2 := by
    rw [hy, mul_pow, hx, Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    rw [← mul_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) (hB (sl j)) 2
  exact (sq_le_sq₀ (norm_nonneg _) (mul_nonneg hb (norm_nonneg _))).mp hsq

/-- **Lemma N (Kronecker factor):** `‖(A ⊗ B) x‖ ≤ ‖A‖ ‖B‖ ‖x‖`, with the norms as pointwise
bounds `‖A v‖ ≤ a ‖v‖`, `‖B w‖ ≤ b ‖w‖`. -/
theorem norm_kronecker_le (A : Matrix m m 𝕂) (B : Matrix p p 𝕂) {a b : ℝ} (ha : 0 ≤ a)
    (hb : 0 ≤ b) (hA : ∀ v, ‖toEuclideanLin A v‖ ≤ a * ‖v‖)
    (hB : ∀ v, ‖toEuclideanLin B v‖ ≤ b * ‖v‖) (x : EuclideanSpace 𝕂 (m × p)) :
    ‖toEuclideanLin (A ⊗ₖ B) x‖ ≤ a * b * ‖x‖ := by
  have e : A ⊗ₖ B = (A ⊗ₖ (1 : Matrix p p 𝕂)) * ((1 : Matrix m m 𝕂) ⊗ₖ B) := by
    rw [← Matrix.mul_kronecker_mul, Matrix.mul_one, Matrix.one_mul]
  have h1 := norm_kron_one_le A ha hA (toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x)
  have h2 := norm_one_kron_le B hb hB x
  rw [e]
  calc ‖toEuclideanLin ((A ⊗ₖ (1 : Matrix p p 𝕂)) * ((1 : Matrix m m 𝕂) ⊗ₖ B)) x‖
      = ‖toEuclideanLin (A ⊗ₖ (1 : Matrix p p 𝕂))
          (toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x)‖ := by
        rw [Matrix.toLpLin_mul_same]
        rfl
    _ ≤ a * ‖toEuclideanLin ((1 : Matrix m m 𝕂) ⊗ₖ B) x‖ := h1
    _ ≤ a * (b * ‖x‖) := mul_le_mul_of_nonneg_left h2 ha
    _ = a * b * ‖x‖ := by ring

/-- A diagonal sign factor (`|d_i| = 1`, the Jordan-Wigner parity) preserves norms. -/
theorem norm_diagonal_unit {n : Type*} [Fintype n] [DecidableEq n] (d : n → 𝕂)
    (hd : ∀ i, ‖d i‖ = 1) (x : EuclideanSpace 𝕂 n) :
    ‖toEuclideanLin (Matrix.diagonal d) x‖ = ‖x‖ := by
  have h : ‖toEuclideanLin (Matrix.diagonal d) x‖ ^ 2 = ‖x‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp [Matrix.toLpLin_apply, Matrix.mulVec_diagonal, norm_mul, hd i]
  have := congrArg Real.sqrt h
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at this

end KronNorm

section Terms

variable {𝕂 : Type*} [RCLike 𝕂] {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕂 E]

/-- A product of operators with pointwise bounds `‖O_l z‖ ≤ n_l ‖z‖` is bounded by `Π n_l`. -/
theorem norm_list_prod_le : ∀ (l : List ((E →ₗ[𝕂] E) × ℝ)),
    (∀ q ∈ l, 0 ≤ q.2 ∧ ∀ z, ‖q.1 z‖ ≤ q.2 * ‖z‖) →
    ∀ z, ‖(l.map Prod.fst).prod z‖ ≤ (l.map Prod.snd).prod * ‖z‖
  | [], _, z => by simp
  | q :: l, hq, z => by
    have ih := norm_list_prod_le l (fun r hr => hq r (List.mem_cons_of_mem _ hr))
    obtain ⟨h0, h1⟩ := hq q List.mem_cons_self
    simp only [List.map_cons, List.prod_cons, Module.End.mul_apply]
    calc ‖q.1 ((l.map Prod.fst).prod z)‖ ≤ q.2 * ‖(l.map Prod.fst).prod z‖ := h1 _
      _ ≤ q.2 * ((l.map Prod.snd).prod * ‖z‖) := mul_le_mul_of_nonneg_left (ih z) h0
      _ = q.2 * (l.map Prod.snd).prod * ‖z‖ := by ring

/-- **Lemma N (term-wise bound):** `‖Σ_t c_t O_t z‖ ≤ (Σ_t |c_t| n_t) ‖z‖` when
`‖O_t z‖ ≤ n_t ‖z‖`. -/
theorem norm_sum_terms_le {τ : Type*} [Fintype τ] (O : τ → E →ₗ[𝕂] E) (c : τ → 𝕂) (n : τ → ℝ)
    (hO : ∀ t z, ‖O t z‖ ≤ n t * ‖z‖) (z : E) :
    ‖∑ t, c t • O t z‖ ≤ (∑ t, ‖c t‖ * n t) * ‖z‖ := by
  refine (norm_sum_le _ _).trans ?_
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun t _ => ?_
  rw [norm_smul, mul_assoc]
  exact mul_le_mul_of_nonneg_left (hO t z) (norm_nonneg _)

end Terms

/-! ### Part (b): the Theorem 1 radius with `adapter.width = W - 2` -/

section PartB

theorem d02b_poly {x u : ℝ} (hx0 : 0 ≤ x) (hx : x ≤ 1 / 10 ^ 7) (hu : 0 ≤ u)
    (hu1 : u ≤ 1 / 10 ^ 15) :
    (1 - x - x * (1 - u) ^ 2) * (1 + u) * (1 - x - u) ≤
      (1 - x) * (1 - 2 * x) * ((1 - u) ^ 7 * (1 + 4 * u) ^ 2) := by
  have h7 := one_sub_mul_le_pow hu (by linarith) 7
  push_cast at h7
  have h16 : 1 + 8 * u ≤ (1 + 4 * u) ^ 2 := by nlinarith only [hu]
  have hP : 1 + u - 56 * u ^ 2 ≤ (1 - u) ^ 7 * (1 + 4 * u) ^ 2 := by
    have := mul_le_mul h7 h16 (by linarith) (pow_nonneg (by linarith) 7)
    nlinarith only [this]
  have hA : 0 ≤ (1 - x) * (1 - 2 * x) := mul_nonneg (by linarith) (by linarith)
  have hR := mul_le_mul_of_nonneg_left hP hA
  have hsq : 1 - 2 * u ≤ (1 - u) ^ 2 := by nlinarith only [hu]
  have hL1 : 1 - x - x * (1 - u) ^ 2 ≤ 1 - 2 * x + 2 * x * u := by
    nlinarith only [mul_le_mul_of_nonneg_left hsq hx0]
  have hB : 0 ≤ (1 + u) * (1 - x - u) := mul_nonneg (by linarith) (by linarith)
  have hL : (1 - x - x * (1 - u) ^ 2) * (1 + u) * (1 - x - u) ≤
      (1 - 2 * x + 2 * x * u) * (1 + u) * (1 - x - u) := by
    have := mul_le_mul_of_nonneg_right hL1 hB
    linarith only [this]
  have hk : 0 ≤ 1 - 4 * x + 2 * x ^ 2 - 55 * u + 166 * x * u - 110 * x ^ 2 * u + 2 * x * u ^ 2 := by
    have hx2 : x ^ 2 ≤ 1 := by nlinarith only [hx0, hx]
    have a1 : x ^ 2 * u ≤ u := by
      have := mul_le_mul_of_nonneg_right hx2 hu
      linarith only [this]
    nlinarith only [hx0, hx, hu, hu1, mul_nonneg hx0 hu, mul_nonneg (mul_nonneg hx0 hu) hu,
      sq_nonneg x, a1]
  have hd : (1 - 2 * x + 2 * x * u) * (1 + u) * (1 - x - u) ≤
      (1 - x) * (1 - 2 * x) * (1 + u - 56 * u ^ 2) := by
    have e : (1 - x) * (1 - 2 * x) * (1 + u - 56 * u ^ 2) -
        (1 - 2 * x + 2 * x * u) * (1 + u) * (1 - x - u) =
        u * (1 - 4 * x + 2 * x ^ 2 - 55 * u + 166 * x * u - 110 * x ^ 2 * u + 2 * x * u ^ 2) := by
      ring
    nlinarith only [e, mul_nonneg hu hk]
  linarith only [hL, hd, hR]

/-- **The second-order inequality of D-02 (b).** With `k = W - 1`, `2 ≤ k ≤ 2²⁶ + 1` and
`u ≤ 2⁻⁵³`, the residual error index `γ_k + u (1 + γ_k)` is covered by the code's factor:
`(γ_k + u (1 + γ_k)) (1 - γ_k (1 - u)²)(1 + u) ≤ γ_{k+1} (1 - u)⁷ (1 + 4u)² (1 - γ_k)`.
The margin is of second order in `u`: `γ_k + u (1 + γ_k)` and `γ_{k+1}` differ by about
`k u²`, and the code's factors give `1 + O(u²)`. -/
theorem d02b_key {u : ℝ} (hu : 0 ≤ u) (hu53 : u ≤ (2 : ℝ)⁻¹ ^ 53) {k : ℕ} (hk : k ≤ 2 ^ 26 + 1) :
    (gam u k + u * (1 + gam u k)) * ((1 - gam u k * (1 - u) ^ 2) * (1 + u)) ≤
      gam u (k + 1) * ((1 - u) ^ 7 * (1 + 4 * u) ^ 2) * (1 - gam u k) := by
  obtain ⟨hk1, -, -⟩ := gam_regime hu hu53 (k := k + 1) (by omega)
  have hu1 : u ≤ 1 / 10 ^ 15 := hu53.trans (by norm_num)
  push_cast at hk1
  set x := (k : ℝ) * u with hx
  have hx0 : 0 ≤ x := mul_nonneg (Nat.cast_nonneg k) hu
  have hxs : x ≤ 1 / 10 ^ 7 := by rw [hx]; nlinarith only [hk1, hu]
  have hA : 0 < 1 - x := by linarith
  have hB : 0 < 1 - x - u := by rw [hx]; nlinarith only [hk1, hu]
  have hg : gam u k = x / (1 - x) := by unfold gam; rw [hx]
  have hG : gam u (k + 1) = (x + u) / (1 - x - u) := by
    unfold gam; rw [hx]; push_cast; ring
  rw [hg, hG]
  have e1 : x / (1 - x) + u * (1 + x / (1 - x)) = (x + u) / (1 - x) := by
    field_simp; ring
  have e2 : 1 - x / (1 - x) * (1 - u) ^ 2 = (1 - x - x * (1 - u) ^ 2) / (1 - x) := by
    field_simp
  have e3 : 1 - x / (1 - x) = (1 - 2 * x) / (1 - x) := by
    field_simp; ring
  rw [e1, e2, e3]
  set P := (1 - u) ^ 7 * (1 + 4 * u) ^ 2 with hP
  have eL : (x + u) / (1 - x) * ((1 - x - x * (1 - u) ^ 2) / (1 - x) * (1 + u)) =
      (x + u) * (1 - x - x * (1 - u) ^ 2) * (1 + u) / ((1 - x) * (1 - x)) := by
    field_simp
  have eR : (x + u) / (1 - x - u) * P * ((1 - 2 * x) / (1 - x)) =
      (x + u) * P * (1 - 2 * x) / ((1 - x - u) * (1 - x)) := by
    field_simp
  rw [eL, eR, div_le_div_iff₀ (by positivity) (by positivity)]
  have hpoly := d02b_poly hx0 hxs hu hu1
  have hc : 0 ≤ (x + u) * (1 - x) := mul_nonneg (by linarith) hA.le
  have := mul_le_mul_of_nonneg_left hpoly hc
  calc (x + u) * (1 - x - x * (1 - u) ^ 2) * (1 + u) * ((1 - x - u) * (1 - x))
      = (x + u) * (1 - x) * ((1 - x - x * (1 - u) ^ 2) * (1 + u) * (1 - x - u)) := by ring
    _ ≤ (x + u) * (1 - x) * ((1 - x) * (1 - 2 * x) * P) := this
    _ = (x + u) * P * (1 - 2 * x) * ((1 - x) * (1 - x)) := by ring

/-- `_rounding_bound` from a given computed `(|H| |x|)_i` entry `ah`. -/
noncomputable def roundBoundOf (R : Rounding ℝ) (w : ℕ) (ah x μ : ℝ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (ah / R.rnd (1 - gamHat R (w + 1))) * (1 + 4 * R.u)) +
    R.rnd (|x| * |μ|)) * gamHat R (w + 2)) * (1 + 4 * R.u))

/-- **D-02 (b), one entry.** If the computed product entry has error `γ_{w+1} a` (Theorem D-02
(a) with `w = W - 2`) and the computed `(|H| |x|)_i` is at least `a (1 - γ_{w+1})` (Lemma M3),
the rounding bound evaluated with width `w` covers the residual entry. -/
theorem d02b_entry (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) {w : ℕ}
    (hw1 : 1 ≤ w) (hw : w ≤ 2 ^ 26) {y yh a ah x μ : ℝ} (ha0 : 0 ≤ a)
    (hy : |yh - y| ≤ gam R.u (w + 1) * a) (hya : |y| ≤ a)
    (hah : a * (1 - gam R.u (w + 1)) ≤ ah) :
    |R.rnd (yh - R.rnd (x * μ)) - (y - μ * x)| ≤ roundBoundOf R w ah x μ := by
  have hu := R.u_nonneg
  obtain ⟨hu1, hkw, hgw0, hgw1, hgw2, hGw, hGw2⟩ := small_consts hu hu53 hw
  have hu1' : R.u ≤ 1 := by linarith
  set g := gam R.u (w + 1) with hg
  set G := gam R.u (w + 2) with hG
  have hg0 : 0 ≤ g := by linarith
  have hG0 : 0 ≤ G := by linarith
  set q := |x| * |μ| with hq
  have hq0 : 0 ≤ q := by positivity
  have h1 := R.resid_entry h0 (x := x) (μ := μ) g hy hya
  -- the computed bound from below
  have hw1u : ((w + 1 : ℕ) : ℝ) * R.u ≤ 1 / 2 := by
    have : ((w + 1 : ℕ) : ℝ) * R.u ≤ ((w + 2 : ℕ) : ℝ) * R.u :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : w + 1 ≤ w + 2)) hu
    linarith
  have hw2u : ((w + 2 : ℕ) : ℝ) * R.u ≤ 1 / 2 := by linarith
  obtain ⟨hgh1, hgh1'⟩ := gamHat_bounds R h0 (by linarith) (k := w + 1) hw1u
  obtain ⟨hgh2, -⟩ := gamHat_bounds R h0 (by linarith) (k := w + 2) hw2u
  have hsq1 : (1 - R.u) ^ 2 ≤ 1 := by nlinarith
  have hC0 : 0 ≤ 1 - g * (1 - R.u) ^ 2 := by nlinarith [mul_le_mul_of_nonneg_left hsq1 hg0]
  set dh := (1 - g * (1 - R.u) ^ 2) * (1 + R.u) with hdh
  have hd : R.rnd (1 - gamHat R (w + 1)) ≤ dh := R.rnd_le_of_le h0 hu1' (by linarith) hC0
  have hd0 : 0 < R.rnd (1 - gamHat R (w + 1)) := by
    have := R.le_rnd_of_le h0 hu1' (w := 1 - gamHat R (w + 1)) (z := 1 - 3 * g)
      (by linarith) (by linarith)
    nlinarith
  have hdh0 : 0 < dh := lt_of_lt_of_le hd0 hd
  have hag : 0 ≤ a * (1 - g) := mul_nonneg ha0 (by linarith)
  have hz1 : a * (1 - g) / dh ≤ ah / R.rnd (1 - gamHat R (w + 1)) :=
    (div_le_div_of_nonneg_left hag hd0 hd).trans (div_le_div_of_nonneg_right hah hd0.le)
  have hX0 : 0 ≤ a * (1 - g) / dh := div_nonneg hag hdh0.le
  have hb1 := R.le_rnd_of_le h0 hu1' hz1 hX0
  have h1u : 0 ≤ 1 - R.u := by linarith
  have h4 : 0 ≤ 1 + 4 * R.u := by linarith
  have hb2 := R.le_rnd_mul h0 hu1' (mul_nonneg hX0 h1u) h4 hb1 le_rfl
  set B2 := a * (1 - g) / dh * (1 - R.u) * (1 + 4 * R.u) * (1 - R.u) with hB2
  have hB20 : 0 ≤ B2 := by rw [hB2]; positivity
  have hqh := R.le_rnd_of_le h0 hu1' (w := |x| * |μ|) (z := q) le_rfl hq0
  have hz3 : B2 + q * (1 - R.u) ≤ R.rnd (R.rnd (ah / R.rnd (1 - gamHat R (w + 1))) *
      (1 + 4 * R.u)) + R.rnd (|x| * |μ|) := add_le_add hb2 hqh
  have hz30 : 0 ≤ B2 + q * (1 - R.u) := add_nonneg hB20 (mul_nonneg hq0 h1u)
  have hb3 := R.le_rnd_of_le h0 hu1' hz3 hz30
  have hG2 : 0 ≤ G * (1 - R.u) ^ 2 := by positivity
  have hb4 := R.le_rnd_mul h0 hu1' (mul_nonneg hz30 h1u) hG2 hb3 hgh2
  have hb5 := R.le_rnd_mul h0 hu1' (mul_nonneg (mul_nonneg (mul_nonneg hz30 h1u) hG2) h1u) h4
    hb4 le_rfl
  -- the two coefficients
  have hkey := d02b_key hu hu53 (k := w + 1) (by omega)
  have hka : (g + R.u * (1 + g)) * a ≤ G * (1 - R.u) ^ 5 * (1 + 4 * R.u) * B2 := by
    have e : G * (1 - R.u) ^ 5 * (1 + 4 * R.u) * B2 =
        (G * ((1 - R.u) ^ 7 * (1 + 4 * R.u) ^ 2) * (1 - g)) * a / dh := by
      rw [hB2]; field_simp
    rw [e, le_div_iff₀ hdh0]
    have := mul_le_mul_of_nonneg_left hkey ha0
    calc (g + R.u * (1 + g)) * a * dh = a * ((g + R.u * (1 + g)) * dh) := by ring
      _ ≤ a * (G * ((1 - R.u) ^ 7 * (1 + 4 * R.u) ^ 2) * (1 - g)) := this
      _ = _ := by ring
  have hkq : (2 * R.u + R.u ^ 2) * q ≤ G * (1 - R.u) ^ 6 * (1 + 4 * R.u) * q := by
    apply mul_le_mul_of_nonneg_right _ hq0
    have h3u : 3 * R.u ≤ G := by
      have hku : ((w + 2 : ℕ) : ℝ) * R.u ≤ G := by
        rw [hG]; unfold gam
        rw [le_div_iff₀ (by linarith)]
        have : 0 ≤ ((w + 2 : ℕ) : ℝ) * R.u := by positivity
        nlinarith only [this]
      have : 3 * R.u ≤ ((w + 2 : ℕ) : ℝ) * R.u :=
        mul_le_mul_of_nonneg_right (by exact_mod_cast (by omega : 3 ≤ w + 2)) hu
      linarith
    have h6 := one_sub_mul_le_pow hu hu1' 6
    push_cast at h6
    have hX : 1 - 3 * R.u ≤ (1 - R.u) ^ 6 * (1 + 4 * R.u) := by
      have := mul_le_mul_of_nonneg_right h6 h4
      nlinarith only [this, hu, hu1]
    have h3 : 0 ≤ 1 - 3 * R.u := by linarith
    nlinarith only [mul_le_mul_of_nonneg_left hX hG0, mul_le_mul_of_nonneg_right h3u h3, hu,
      mul_le_mul_of_nonneg_left hu1 hu]
  unfold roundBoundOf
  refine h1.trans (le_trans ?_ hb5)
  calc (g + R.u * (1 + g)) * a + (2 * R.u + R.u ^ 2) * (|x| * |μ|)
      ≤ G * (1 - R.u) ^ 5 * (1 + 4 * R.u) * B2 + G * (1 - R.u) ^ 6 * (1 + 4 * R.u) * q := by
        rw [← hq]; linarith only [hka, hkq]
    _ = (B2 + q * (1 - R.u)) * (1 - R.u) * (G * (1 - R.u) ^ 2) * (1 - R.u) * (1 + 4 * R.u) *
        (1 - R.u) := by ring

end PartB

section PartBRadius

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The radius `rel` of `residuals` from a computed product `yh` and a computed `|H| |x|`
vector `ah` (any operator adapter). -/
noncomputable def residualRadiusOf (R : Rounding ℝ) (yh ah : ι → ℝ) (TR TE TX : STree ι)
    (w : ℕ) (x : ι → ℝ) (μ tR tE tX c1 c0 : ℝ) : ℝ :=
  R.rnd (R.rnd (R.rnd (R.rnd (R.rnd (norm2Up R TR tR (fun i => R.rnd (yh i - R.rnd (x i * μ))) +
    norm2Up R TE tE (fun i => roundBoundOf R w (ah i) (x i) μ)) + c1) / norm2Low R TX tX x) *
      (1 + 4 * R.u)) + c0)

/-- **Theorem D-02 (b)** (real data, standard model). If the computed product has entrywise error
`γ_{W-1} a_i` (Theorem D-02 (a), `matrixfree_entry_error`, with `a_i = (|H_spec|_abs |x|)_i`) and
the computed `|H| |x|` is at least `a_i (1 - γ_{W-1})` (Lemma M3), the radius of `residuals`
evaluated with `adapter.width = W - 2` bounds `‖H_spec x - μ x‖ / ‖x‖`. -/
theorem d02b_radius (R : Rounding ℝ) (h0 : R.eta = 0) (hu53 : R.u ≤ (2 : ℝ)⁻¹ ^ 53) {W : ℕ}
    (hW3 : 3 ≤ W) (hW : W ≤ 2 ^ 26 + 2) (hn : Fintype.card ι ≤ 2 ^ 26) (H : ι → ι → ℝ)
    (x : ι → ℝ) (μ : ℝ) {a yh ah : ι → ℝ} (hya : ∀ i, |∑ j, H i j * x j| ≤ a i)
    (hy : ∀ i, |yh i - ∑ j, H i j * x j| ≤ gam R.u (W - 1) * a i)
    (hah : ∀ i, a i * (1 - gam R.u (W - 1)) ≤ ah i) {TR TE TX : STree ι} (hTR : FullTree TR)
    (hTE : FullTree TE) (hTX : FullTree TX) {tR tE tX c1 c0 : ℝ} (htR : 0 ≤ tR) (htE : 0 ≤ tE)
    (htX : 0 ≤ tX) (hc1 : 0 ≤ c1) (hc0 : 0 ≤ c0) (hxl : 0 < norm2Low R TX tX x) :
    l2 (fun i => ∑ j, H i j * x j - μ * x i) / l2 x ≤
      residualRadiusOf R yh ah TR TE TX (W - 2) x μ tR tE tX c1 c0 := by
  have hu := R.u_nonneg
  have hu1 : R.u ≤ 1 / 1000 := hu53.trans (by norm_num)
  have hWe : W - 1 = W - 2 + 1 := by omega
  rw [hWe] at hy hah
  have hentry : ∀ i, |R.rnd (yh i - R.rnd (x i * μ)) - (∑ j, H i j * x j - μ * x i)| ≤
      roundBoundOf R (W - 2) (ah i) (x i) μ := fun i =>
    d02b_entry R h0 hu53 (by omega) (by omega) (le_trans (abs_nonneg _) (hya i)) (hy i) (hya i)
      (hah i)
  have hr : l2 (fun i => ∑ j, H i j * x j - μ * x i) ≤
      l2 (fun i => R.rnd (yh i - R.rnd (x i * μ))) +
        l2 (fun i => roundBoundOf R (W - 2) (ah i) (x i) μ) := by
    refine (l2_le_add _ (fun i => R.rnd (yh i - R.rnd (x i * μ)))).trans
      (add_le_add le_rfl (l2_le_of_abs_le ?_))
    intro i
    rw [abs_sub_comm]
    exact hentry i
  exact R.chain_final h0 (by linarith) hr (l2_nonneg _) (l2_nonneg _)
    (norm2Up_ge R h0 hu53 hn hTR htR _) (norm2Up_ge R h0 hu53 hn hTE htE _) hc1 hc0 hxl
    (norm2Low_le R h0 hu53 hn hTX htX x) (by linarith) (num_final hu hu1)

end PartBRadius


end CertifyMathlib.Enclosure.FP
