import CertifyCore.Theorems.Gamma
/-
The crossing slack of E-04 (`docs/proofs/E-04_parameter_intervals.md`, Lemma 3, `certify_ed/crossings.py`):
if the tube ends alpha+ and beta- are B-Lipschitz in the coupling c, the margin m(c) = alpha+(c) - beta-(c) on a
sample interval of width D is at most max(m(c_i), m(c_{i+1})) + B D, so the criterion max(...) + B D < 0
separates the tubes on the whole interval. Kernel-checked, core Lean only.
-/
namespace CertifyCore.Thm

/-- Pointwise form: m(c) <= m0 + 2 B t and m(c) <= m1 + 2 B (D - t) with 0 <= t <= D imply
m(c) <= max(m0, m1) + B D. -/
theorem crossing_slack (B D t m0 m1 mc : Rat) (hB : 0 ≤ B) (_ht0 : 0 ≤ t) (_htD : t ≤ D)
    (h0 : mc ≤ m0 + 2 * B * t) (h1 : mc ≤ m1 + 2 * B * (D - t)) : mc ≤ max m0 m1 + B * D := by
  have mx0 : m0 ≤ max m0 m1 := le_max_l m0 m1
  have mx1 : m1 ≤ max m0 m1 := le_max_r m0 m1
  by_cases hc : 2 * t ≤ D
  · have := Rat.mul_le_mul_of_nonneg_left hc hB
    grind
  · have : D ≤ 2 * t := by grind
    have := Rat.mul_le_mul_of_nonneg_left (show 2 * (D - t) ≤ D by grind) hB
    grind

/-- Lemma 3 of E-04: alpha and beta B-Lipschitz on [ci, ci + D]; if max(m(ci), m(ci + D)) + B D < 0 then
alpha(c) < beta(c) for every c in [ci, ci + D]. -/
theorem crossing_separation (alpha beta : Rat → Rat) (B ci D : Rat) (hB : 0 ≤ B)
    (la : ∀ x y, ci ≤ x → x ≤ y → y ≤ ci + D → alpha y - alpha x ≤ B * (y - x) ∧ alpha x - alpha y ≤ B * (y - x))
    (lb : ∀ x y, ci ≤ x → x ≤ y → y ≤ ci + D → beta y - beta x ≤ B * (y - x) ∧ beta x - beta y ≤ B * (y - x))
    (hcrit : max (alpha ci - beta ci) (alpha (ci + D) - beta (ci + D)) + B * D < 0) :
    ∀ c, ci ≤ c → c ≤ ci + D → alpha c < beta c := by
  intro c h1 h2
  have ⟨a1, _⟩ := la ci c (Rat.le_refl) h1 h2
  have ⟨b1, b2⟩ := lb ci c (Rat.le_refl) h1 h2
  have ⟨a3, a4⟩ := la c (ci + D) h1 h2 (Rat.le_refl)
  have ⟨b3, b4⟩ := lb c (ci + D) h1 h2 (Rat.le_refl)
  have s := crossing_slack B D (c - ci) (alpha ci - beta ci) (alpha (ci + D) - beta (ci + D)) (alpha c - beta c)
    hB (by grind) (by grind) (by grind) (by grind)
  grind

end CertifyCore.Thm
