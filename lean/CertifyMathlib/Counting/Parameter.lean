import CertifyMathlib.Counting.Inertia

/-!
# E-04 (certified parameter intervals)

Source: `docs/proofs/E-04_parameter_intervals.md` in the code repository. On a segment
`[c_k, c_{k+1}]` the certified operator is the affine interpolant `F_k(c) = H(c_k) + (c - c_k) H1`
with `‖H1‖₂ ≤ B` and the affinity defect `‖H(c_{k+1}) - F_k(c_{k+1})‖₂ ≤ d_k`. Norm bounds enter
as bounds on quadratic forms, as everywhere in this library.

* `weyl_affine` (in `CertifyMathlib.Counting.Inertia`): Proposition 1, first tube, and
  Proposition 2 (index preservation in counting form);
* `weyl_affine_right`: Proposition 1, second tube, `|λ_j(F_k(c)) - λ_j(H(c_{k+1}))| ≤
  B |c - c_{k+1}| + d_k`, in counting form;
* `qf_triangle`: the triangle inequality for quadratic-form bounds;
* `sampling_criterion`: Lemma 3 (a `2B`-Lipschitz margin that is below `-B D` at both ends of a
  sample interval of width `D` is negative on the whole interval).

Not formalised: Lemma 4 of the document (the term-wise bound on `‖H1‖₂`) and the continuity
argument for crossings.
-/

namespace CertifyMathlib.Counting

open Matrix

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

omit [DecidableEq n] in
theorem qf_triangle {A B C : Matrix n n 𝕜} {ρ₁ ρ₂ : ℝ} (h1 : ∀ x, |qf (A - B) x| ≤ ρ₁ * nsq x)
    (h2 : ∀ x, |qf (B - C) x| ≤ ρ₂ * nsq x) (x : n → 𝕜) :
    |qf (A - C) x| ≤ (ρ₁ + ρ₂) * nsq x := by
  have e : qf (A - C) x = qf (A - B) x + qf (B - C) x := by
    simp only [qf_sub]
    ring
  rw [e, add_mul]
  exact (abs_add_le _ _).trans (add_le_add (h1 x) (h2 x))

/-- **E-04, Proposition 1, second tube**, in counting form. -/
theorem weyl_affine_right {H₀ H₁ F Hn : Matrix n n 𝕜} (hF : F.IsHermitian)
    (hn : Hn.IsHermitian) (c ck ck1 B dk : ℝ) (hFeq : F = H₀ + ((c - ck : ℝ) : 𝕜) • H₁)
    (hB : ∀ x, |qf H₁ x| ≤ B * nsq x)
    (hd : ∀ x, |qf (Hn - (H₀ + ((ck1 - ck : ℝ) : 𝕜) • H₁)) x| ≤ dk * nsq x) (t : ℝ) :
    cntLt hF t ≤ cntLt hn (t + (B * |c - ck1| + dk)) ∧
    cntLt hn t ≤ cntLt hF (t + (B * |c - ck1| + dk)) ∧
    cntLe hF t ≤ cntLe hn (t + (B * |c - ck1| + dk)) ∧
    cntLe hn t ≤ cntLe hF (t + (B * |c - ck1| + dk)) := by
  have hd' : ∀ x, |qf (F - Hn) x| ≤ (B * |c - ck1| + dk) * nsq x := by
    intro x
    have e1 : qf (F - Hn) x = (c - ck1) * qf H₁ x +
        -qf (Hn - (H₀ + ((ck1 - ck : ℝ) : 𝕜) • H₁)) x := by
      rw [hFeq]
      simp only [qf_sub, qf_add, qf_smul]
      ring
    rw [e1]
    have h1 := hB x
    have h2 := hd x
    calc |(c - ck1) * qf H₁ x + -qf (Hn - (H₀ + ((ck1 - ck : ℝ) : 𝕜) • H₁)) x|
        ≤ |(c - ck1) * qf H₁ x| + |-qf (Hn - (H₀ + ((ck1 - ck : ℝ) : 𝕜) • H₁)) x| :=
          abs_add_le _ _
      _ = |c - ck1| * |qf H₁ x| + |qf (Hn - (H₀ + ((ck1 - ck : ℝ) : 𝕜) • H₁)) x| := by
          rw [abs_mul, abs_neg]
      _ ≤ |c - ck1| * (B * nsq x) + dk * nsq x :=
          add_le_add (mul_le_mul_of_nonneg_left h1 (abs_nonneg _)) h2
      _ = (B * |c - ck1| + dk) * nsq x := by ring
  have hd'' : ∀ x, |qf (Hn - F) x| ≤ (B * |c - ck1| + dk) * nsq x := fun x => by
    rw [abs_qf_sub_comm]
    exact hd' x
  exact ⟨weyl_cntLt hF hn _ hd' t, weyl_cntLt hn hF _ hd'' t, weyl_cntLe hF hn _ hd' t,
    weyl_cntLe hn hF _ hd'' t⟩

/-- **E-04, Lemma 3 (sampling criterion).** -/
theorem sampling_criterion (m : ℝ → ℝ) (B ci D : ℝ) (hB : 0 ≤ B)
    (hlip : ∀ c c', |m c - m c'| ≤ 2 * B * |c - c'|)
    (hend : max (m ci) (m (ci + D)) + B * D < 0) (c : ℝ) (hc : ci ≤ c ∧ c ≤ ci + D) :
    m c < 0 := by
  have h1 := le_max_left (m ci) (m (ci + D))
  have h2 := le_max_right (m ci) (m (ci + D))
  by_cases hmid : c - ci ≤ D / 2
  · have h3 := hlip c ci
    rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ c - ci)] at h3
    have h4 := (abs_le.mp h3).2
    nlinarith
  · have h3 := hlip c (ci + D)
    rw [abs_of_nonpos (by linarith : c - (ci + D) ≤ 0)] at h3
    have h4 := (abs_le.mp h3).2
    nlinarith

end CertifyMathlib.Counting
