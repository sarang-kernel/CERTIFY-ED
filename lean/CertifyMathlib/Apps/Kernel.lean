import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.RingTheory.Localization.Module
import Mathlib.Algebra.GCDMonoid.Finset
import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Algebra.Order.Star.Real

/-!
# The certificate of the AKLT kernel equality `ker H_X = range U_X`

The E10 eta bounds take the ground space of each subgraph Hamiltonian `H_X = Σ_e P_e` (projectors onto
the largest pair spin) to be the range of the VBS map `U_X` (the kernel lemma,
`project_docs/notes/aklt_kernel_lemma.md`). `certify_ed/apps/aklt/kernel.py`, and for all twelve
subgraphs the compiled program `ceapps aklt-kernels` (`CertifyApps/Kernel.lean`), confirm the equality
by exact computation instead of the general argument: in the scaled basis the edge terms are integer
matrices `T e = D⁻¹ P_e D` (up to a positive factor), every `T e` annihilates the integer VBS columns `U`,
the rank of `U` modulo a prime `p` is at least `r`, and the common null space of the `T e` modulo `p` has
dimension at most `r`. This file proves that these computed facts imply the equality:

* `linearIndependent_int_of_zmod`: integer vectors that are linearly independent modulo `p` are
  linearly independent over `ℤ` (a primitive integer relation would survive modulo `p`);
* `rank_zmod_le_rank_rat`: the rank of an integer matrix modulo `p` is at most its rank over `ℚ`;
* `psd_sum_mulVec_eq_zero`: a vector killed by a sum of positive semidefinite matrices is killed by
  every term;
* `aklt_kernel_certificate`: the computed facts give `ker (Σ_e T e) = range U` over `ℚ`, of dimension
  `r`.
-/

namespace CertifyMathlib.Apps

open Matrix

/-- Integer vectors that are linearly independent modulo a prime are linearly independent over `ℤ`. -/
theorem linearIndependent_int_of_zmod {ι m : Type*} [Fintype ι] (p : ℕ) [hp : Fact p.Prime]
    (v : ι → m → ℤ) (h : LinearIndependent (ZMod p) (fun i j => ((v i j : ℤ) : ZMod p))) :
    LinearIndependent ℤ v := by
  rw [Fintype.linearIndependent_iff] at h ⊢
  intro g hg
  by_contra hne
  rw [not_forall] at hne
  obtain ⟨i0, hi0⟩ := hne
  have hG : Finset.univ.gcd g ≠ 0 := by
    intro h0
    rw [Finset.gcd_eq_zero_iff] at h0
    exact hi0 (h0 i0 (Finset.mem_univ _))
  obtain ⟨g', hg', hgcd⟩ := Finset.extract_gcd g ⟨i0, Finset.mem_univ _⟩
  have hterm : ∑ i, Finset.univ.gcd g • g' i • v i = ∑ i, g i • v i :=
    Finset.sum_congr rfl fun i _ => by rw [hg' i (Finset.mem_univ i), mul_smul]
  have hsum : Finset.univ.gcd g • ∑ i, g' i • v i = 0 := by
    rw [Finset.smul_sum, hterm, hg]
  have hsum' : ∑ i, g' i • v i = 0 := (smul_eq_zero.mp hsum).resolve_left hG
  have hmod := h (fun i => ((g' i : ℤ) : ZMod p)) (by
    funext j
    have hj := congrFun hsum' j
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at hj ⊢
    have := congrArg (Int.cast : ℤ → ZMod p) hj
    push_cast at this
    exact this)
  have hdvd : (p : ℤ) ∣ Finset.univ.gcd g' :=
    Finset.dvd_gcd fun i _ => (ZMod.intCast_zmod_eq_zero_iff_dvd _ p).mp (hmod i)
  rw [hgcd] at hdvd
  have h1 := Int.eq_one_of_dvd_one (by positivity) hdvd
  have h2 : (p : ℤ) ≠ 1 := by exact_mod_cast hp.out.one_lt.ne'
  exact h2 h1

/-- **The rank of an integer matrix modulo a prime is at most its rank over `ℚ`.** -/
theorem rank_zmod_le_rank_rat {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]
    (A : Matrix m n ℤ) (p : ℕ) [Fact p.Prime] :
    (A.map (Int.cast : ℤ → ZMod p)).rank ≤ (A.map (Int.cast : ℤ → ℚ)).rank := by
  rw [Matrix.rank_eq_finrank_span_cols, Matrix.rank_eq_finrank_span_cols]
  obtain ⟨κ, a, ha, hspan, hli⟩ :=
    exists_linearIndependent' (ZMod p) (A.map (Int.cast : ℤ → ZMod p)).col
  have : Finite κ := Finite.of_injective a ha
  have : Fintype κ := Fintype.ofFinite κ
  rw [← hspan, finrank_span_eq_card hli]
  have hZ : LinearIndependent ℤ (fun k => A.col (a k)) := by
    apply linearIndependent_int_of_zmod p
    convert hli using 1
    funext i j
    rfl
  let f : (m → ℤ) →ₗ[ℤ] (m → ℚ) := (Int.castAddHom ℚ).toIntLinearMap.compLeft m
  have hf : LinearMap.ker f = ⊥ := by
    rw [LinearMap.ker_eq_bot]
    intro x y hxy
    funext j
    have := congrFun hxy j
    simpa [f] using this
  have hQZ : LinearIndependent ℤ (fun k => (A.map (Int.cast : ℤ → ℚ)).col (a k)) := by
    have := hZ.map' f hf
    convert this using 1
    funext k j
    rfl
  have hQ : LinearIndependent ℚ (fun k => (A.map (Int.cast : ℤ → ℚ)).col (a k)) :=
    (LinearIndependent.iff_fractionRing ℤ ℚ).mp hQZ
  calc Fintype.card κ
      = Module.finrank ℚ (Submodule.span ℚ
          (Set.range (fun k => (A.map (Int.cast : ℤ → ℚ)).col (a k)))) := (finrank_span_eq_card hQ).symm
    _ ≤ _ := Submodule.finrank_mono (Submodule.span_mono (by
          rintro _ ⟨k, rfl⟩
          exact ⟨a k, rfl⟩))

/-- A vector killed by a sum of positive semidefinite matrices is killed by every term. -/
theorem psd_sum_mulVec_eq_zero {E N : Type*} [Fintype E] [Fintype N] (P : E → Matrix N N ℝ)
    (hP : ∀ e, (P e).PosSemidef) (y : N → ℝ) (h : (∑ e, P e) *ᵥ y = 0) (e : E) :
    P e *ᵥ y = 0 := by
  have h0 : ∑ e, star y ⬝ᵥ (P e *ᵥ y) = 0 := by
    rw [← dotProduct_sum, ← Matrix.sum_mulVec, h, dotProduct_zero]
  have hnn : ∀ e ∈ Finset.univ, 0 ≤ star y ⬝ᵥ (P e *ᵥ y) := fun e _ =>
    (hP e).dotProduct_mulVec_nonneg y
  have := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp h0 e (Finset.mem_univ e)
  exact (hP e).dotProduct_mulVec_zero_iff.mp this

/-- The edge terms stacked into one tall matrix: row `(e, i)` is row `i` of `T e`. -/
def stack {E N : Type*} (T : E → Matrix N N ℤ) : Matrix (E × N) N ℤ := fun ei j => T ei.1 ei.2 j

/-- **The AKLT kernel certificate.** Let the integer edge terms `T e` be similar, by one invertible real
matrix `D`, to positive semidefinite real matrices `P e` (in the AKLT case `D = diag(sqrt(binom(z, k)))`
and `P e` a positive multiple of the projector onto the largest pair spin), let `T e U = 0` for every
edge, and let `p` be a prime with `r ≤ rank_p U` and `#N ≤ r + rank_p (stack T)` (that is, the common
null space of the `T e` modulo `p` has dimension at most `r`). Then over `ℚ` the null space of
`Σ_e T e` is exactly the range of `U`, and it has dimension `r`. -/
theorem aklt_kernel_certificate {E N c : Type*} [Fintype E] [Fintype N] [DecidableEq N] [Fintype c]
    [DecidableEq c] (T : E → Matrix N N ℤ) (U : Matrix N c ℤ) (D : Matrix N N ℝ)
    (hD : IsUnit D.det) (P : E → Matrix N N ℝ) (hP : ∀ e, (P e).PosSemidef)
    (hsim : ∀ e, (T e).map (Int.cast : ℤ → ℝ) = D⁻¹ * P e * D) (hTU : ∀ e, T e * U = 0)
    (p : ℕ) [Fact p.Prime] (r : ℕ) (hr : r ≤ (U.map (Int.cast : ℤ → ZMod p)).rank)
    (hnul : Fintype.card N ≤ r + ((stack T).map (Int.cast : ℤ → ZMod p)).rank) :
    LinearMap.ker ((∑ e, T e).map (Int.cast : ℤ → ℚ)).mulVecLin =
        LinearMap.range (U.map (Int.cast : ℤ → ℚ)).mulVecLin ∧
      Module.finrank ℚ (LinearMap.range (U.map (Int.cast : ℤ → ℚ)).mulVecLin) = r := by
  set H := (∑ e, T e).map (Int.cast : ℤ → ℚ) with hH
  set Uq := U.map (Int.cast : ℤ → ℚ) with hUq
  set Sq := (stack T).map (Int.cast : ℤ → ℚ) with hSq
  -- the VBS range lies in the kernel
  have hHU : H * Uq = 0 := by
    have h1 : (∑ e, T e) * U = 0 := by
      rw [Matrix.sum_mul]; exact Finset.sum_eq_zero fun e _ => hTU e
    ext i j
    have hij := congrFun (congrFun h1 i) j
    simp only [Matrix.mul_apply, Matrix.zero_apply] at hij
    simp only [hH, hUq, Matrix.mul_apply, Matrix.map_apply, Matrix.zero_apply]
    exact_mod_cast hij
  have hle : LinearMap.range Uq.mulVecLin ≤ LinearMap.ker H.mulVecLin := by
    rintro _ ⟨x, rfl⟩
    rw [LinearMap.mem_ker, Matrix.mulVecLin_apply, Matrix.mulVecLin_apply, mulVec_mulVec, hHU,
      zero_mulVec]
  -- the kernel of the sum is the common kernel of the terms (positive semidefinite similarity)
  have hker : LinearMap.ker H.mulVecLin ≤ LinearMap.ker Sq.mulVecLin := by
    intro x hx
    rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hx ⊢
    set f := Rat.castHom ℝ
    set xr : N → ℝ := fun i => f (x i)
    have hcast : ∀ (M : Matrix N N ℤ), (M.map (Int.cast : ℤ → ℚ)).map f = M.map (Int.cast : ℤ → ℝ) := by
      intro M; ext i j; simp [f]
    have hmv : ∀ (M : Matrix N N ℤ) (i : N),
        f (((M.map (Int.cast : ℤ → ℚ)) *ᵥ x) i) = (M.map (Int.cast : ℤ → ℝ) *ᵥ xr) i := by
      intro M i
      rw [RingHom.map_mulVec, hcast]
      rfl
    have hsumR : (∑ e, T e).map (Int.cast : ℤ → ℝ) = D⁻¹ * (∑ e, P e) * D := by
      have : (∑ e, T e).map (Int.cast : ℤ → ℝ) = ∑ e, (T e).map (Int.cast : ℤ → ℝ) := by
        ext i j; simp [Matrix.sum_apply]
      rw [this, Finset.mul_sum, Finset.sum_mul]
      exact Finset.sum_congr rfl fun e _ => hsim e
    have hz : (∑ e, T e).map (Int.cast : ℤ → ℝ) *ᵥ xr = 0 := by
      funext i
      rw [← hmv, hx]
      simp
    have h1 : D⁻¹ *ᵥ ((∑ e, P e) *ᵥ (D *ᵥ xr)) = 0 := by
      rw [hsumR] at hz
      simp only [mulVec_mulVec]
      rw [← Matrix.mul_assoc]
      exact hz
    have hPz : (∑ e, P e) *ᵥ (D *ᵥ xr) = 0 := by
      calc (∑ e, P e) *ᵥ (D *ᵥ xr) = D *ᵥ (D⁻¹ *ᵥ ((∑ e, P e) *ᵥ (D *ᵥ xr))) := by
            rw [Matrix.mulVec_mulVec _ D D⁻¹, mul_nonsing_inv D hD, one_mulVec]
        _ = 0 := by rw [h1, mulVec_zero]
    funext ei
    obtain ⟨e, i⟩ := ei
    have hPe := psd_sum_mulVec_eq_zero P hP (D *ᵥ xr) hPz e
    have hTe : (T e).map (Int.cast : ℤ → ℝ) *ᵥ xr = 0 := by
      rw [hsim e, ← mulVec_mulVec, ← mulVec_mulVec, hPe, mulVec_zero]
    have h1 : f ((Sq *ᵥ x) (e, i)) = 0 := by
      have : (Sq *ᵥ x) (e, i) = ((T e).map (Int.cast : ℤ → ℚ) *ᵥ x) i := rfl
      rw [this, hmv, hTe]
      rfl
    simpa [f] using h1
  -- dimensions
  have hrn := LinearMap.finrank_range_add_finrank_ker Sq.mulVecLin
  rw [Module.finrank_fintype_fun_eq_card] at hrn
  have hrankS := rank_zmod_le_rank_rat (stack T) p
  have hrankU := rank_zmod_le_rank_rat U p
  have hk1 := Submodule.finrank_mono hker
  have hS' : Sq.rank = Module.finrank ℚ (LinearMap.range Sq.mulVecLin) := rfl
  have hU' : Uq.rank = Module.finrank ℚ (LinearMap.range Uq.mulVecLin) := rfl
  have hle2 : Module.finrank ℚ (LinearMap.ker H.mulVecLin) ≤
      Module.finrank ℚ (LinearMap.range Uq.mulVecLin) := by
    rw [← hSq] at hrankS
    rw [← hUq] at hrankU
    omega
  have heq := Submodule.eq_of_le_of_finrank_le hle hle2
  refine ⟨heq.symm, ?_⟩
  have h3 : Module.finrank ℚ (LinearMap.range Uq.mulVecLin) =
      Module.finrank ℚ (LinearMap.ker H.mulVecLin) := by rw [heq]
  rw [← hSq] at hrankS
  rw [← hUq] at hrankU
  omega

end CertifyMathlib.Apps
