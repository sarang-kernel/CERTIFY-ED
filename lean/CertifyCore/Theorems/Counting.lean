import CertifyCore.Theorems.Gamma
/-
The counting arguments behind S3 and the sector assembly, over finite index sets and lists:
the pigeonhole step and the sorted-pairing lemma of Corollary 5 (`docs/proofs/cor5_pairing.md`), the completeness
count (an injection of k claimed eigenvalues into the eigenvalues below t, when at most k lie below t, is onto
the k lowest), the multiplicity count of C-16 (`docs/proofs/C16_exact_bracket.md`, the sums over the squarefree
parts), and the additivity of counts over sectors used by Theorem 8 (`docs/proofs/thm8_rho.md`).
Eigenvalues appear here as given nondecreasing sequences; that a matrix has such a sequence, and Sylvester's law,
are the subject of the Mathlib formalisation. Kernel-checked, core Lean only.
-/
namespace CertifyCore.Thm

/-- Pigeonhole: a map injective on {0, ..., k} takes a value >= k there. -/
theorem pigeon (k : Nat) : ∀ (f : Nat → Nat), (∀ i j, i ≤ k → j ≤ k → f i = f j → i = j) →
    ∃ i, i ≤ k ∧ k ≤ f i := by
  induction k with
  | zero => intro f _; exact ⟨0, Nat.le_refl 0, Nat.zero_le _⟩
  | succ k ih =>
    intro f inj
    apply Classical.byContradiction
    intro H
    have small : ∀ i, i ≤ k + 1 → f i ≤ k := by
      intro i hi
      have : ¬ (k + 1 ≤ f i) := fun h => H ⟨i, hi, h⟩
      omega
    let g : Nat → Nat := fun i => if f i = k then f (k + 1) else f i
    have fk1 : f (k + 1) ≠ k ∨ ∀ i, i ≤ k → f i ≠ k := by
      by_cases h : f (k + 1) = k
      · right; intro i hi hfi
        have := inj i (k + 1) (by omega) (Nat.le_refl _) (by rw [hfi, h])
        omega
      · left; exact h
    have ginj : ∀ i j, i ≤ k → j ≤ k → g i = g j → i = j := by
      intro i j hi hj hg
      simp only [g] at hg
      by_cases a : f i = k <;> by_cases b : f j = k <;> simp only [a, b, ite_true, ite_false] at hg
      · exact inj i j (by omega) (by omega) (by rw [a, b])
      · have := inj (k + 1) j (Nat.le_refl _) (by omega) hg; omega
      · have := inj i (k + 1) (by omega) (Nat.le_refl _) hg; omega
      · exact inj i j (by omega) (by omega) hg
    obtain ⟨i, hi, hgi⟩ := ih g ginj
    simp only [g] at hgi
    by_cases a : f i = k
    · simp only [a, ite_true] at hgi
      have s := small (k + 1) (Nat.le_refl _)
      have : f (k + 1) = k := by omega
      have := inj i (k + 1) (by omega) (Nat.le_refl _) (by rw [a, this])
      omega
    · simp only [a, ite_false] at hgi
      have := small i (by omega)
      omega

/-- Corollary 5, sorted pairing (uniform radius): if a and b are nondecreasing on {0, ..., m - 1}, pi is
injective with values in {0, ..., m - 1}, and |b (pi j) - a j| <= r for every j, then |b j - a j| <= r for
every j. -/
theorem sorted_pairing (m : Nat) (a b : Nat → Rat) (r : Rat) (p : Nat → Nat)
    (ha : ∀ i j, i ≤ j → j < m → a i ≤ a j) (hb : ∀ i j, i ≤ j → j < m → b i ≤ b j)
    (hp : ∀ i, i < m → p i < m) (inj : ∀ i j, i < m → j < m → p i = p j → i = j)
    (close : ∀ j, j < m → b (p j) - a j ≤ r ∧ a j - b (p j) ≤ r) :
    ∀ j, j < m → b j - a j ≤ r ∧ a j - b j ≤ r := by
  intro j hj
  constructor
  · -- some i <= j has p i >= j
    obtain ⟨i, hi, hpi⟩ := pigeon j p (fun x y hx hy h => inj x y (by omega) (by omega) h)
    have h1 := hb j (p i) hpi (hp i (by omega))
    have h2 := (close i (by omega)).1
    have h3 := ha i j hi hj
    grind
  · -- some i >= j has p i <= j
    let f : Nat → Nat := fun t => (m - 1) - p (m - 1 - t)
    have finj : ∀ x y, x ≤ m - 1 - j → y ≤ m - 1 - j → f x = f y → x = y := by
      intro x y hx hy h
      simp only [f] at h
      have px := hp (m - 1 - x) (by omega)
      have py := hp (m - 1 - y) (by omega)
      have : p (m - 1 - x) = p (m - 1 - y) := by omega
      have := inj (m - 1 - x) (m - 1 - y) (by omega) (by omega) this
      omega
    obtain ⟨t, ht, hft⟩ := pigeon (m - 1 - j) f finj
    simp only [f] at hft
    have pt := hp (m - 1 - t) (by omega)
    have hpj : p (m - 1 - t) ≤ j := by omega
    have h1 := hb (p (m - 1 - t)) j hpj hj
    have h2 := (close (m - 1 - t) (by omega)).2
    have h3 := ha j (m - 1 - t) (by omega) (by omega)
    grind

/-- Number of indices i < n with lam i < s. -/
def countBelow (lam : Nat → Rat) (s : Rat) : Nat → Nat
  | 0 => 0
  | n + 1 => countBelow lam s n + (if lam n < s then 1 else 0)

theorem countBelow_ge (lam : Nat → Rat) (s : Rat) (i : Nat) (h : ∀ j, j ≤ i → lam j < s) :
    ∀ n, i < n → i + 1 ≤ countBelow lam s n := by
  intro n hn
  induction n with
  | zero => omega
  | succ n ih =>
    simp only [countBelow]
    by_cases e : i < n
    · have := ih e; split <;> omega
    · have hin : i = n := by omega
      subst hin
      -- all of 0..i are below s
      have all : ∀ k, k ≤ i → k ≤ countBelow lam s k := by
        intro k hk
        induction k with
        | zero => simp [countBelow]
        | succ k ihk =>
          simp only [countBelow]
          have := ihk (by omega)
          have : lam k < s := h k (by omega)
          simp [this]; omega
      have := all i (Nat.le_refl _)
      have : lam i < s := h i (Nat.le_refl _)
      simp [this]; omega

/-- The count of Corollary 5 / Lemma 4: if at most k of the nondecreasing values lam_0, ..., lam_{n-1} lie below s,
then every index i with lam i < s satisfies i < k. -/
theorem below_index_lt (n k : Nat) (lam : Nat → Rat) (s : Rat) (hl : ∀ i j, i ≤ j → j < n → lam i ≤ lam j)
    (hc : countBelow lam s n ≤ k) : ∀ i, i < n → lam i < s → i < k := by
  intro i hi hs
  have all : ∀ j, j ≤ i → lam j < s := fun j hj => by have := hl j i hj hi; grind
  have := countBelow_ge lam s i all n hi
  omega

/-- Corollary 5, completeness and pairing (one radius): k claimed values mu (nondecreasing), an injection iota of
the claims into the n eigenvalues lam (nondecreasing) with |lam (iota j) - mu j| <= r and lam (iota j) < s, and at
most k eigenvalues below s. Then the claims are the k lowest eigenvalues, in order: |lam j - mu j| <= r. -/
theorem cor5_pairing (n k : Nat) (lam mu : Nat → Rat) (r s : Rat) (iota : Nat → Nat)
    (hl : ∀ i j, i ≤ j → j < n → lam i ≤ lam j) (hm : ∀ i j, i ≤ j → j < k → mu i ≤ mu j)
    (hi : ∀ j, j < k → iota j < n) (inj : ∀ i j, i < k → j < k → iota i = iota j → i = j)
    (close : ∀ j, j < k → lam (iota j) - mu j ≤ r ∧ mu j - lam (iota j) ≤ r)
    (below : ∀ j, j < k → lam (iota j) < s) (hc : countBelow lam s n ≤ k) :
    ∀ j, j < k → lam j - mu j ≤ r ∧ mu j - lam j ≤ r := by
  have into : ∀ j, j < k → iota j < k := fun j hj => below_index_lt n k lam s hl hc (iota j) (hi j hj) (below j hj)
  have hkn : k ≤ n ∨ k = 0 := by
    by_cases h0 : k = 0
    · right; exact h0
    · left
      -- k distinct indices below n
      obtain ⟨i, hik, hpi⟩ := pigeon (k - 1) iota (fun x y hx hy h => inj x y (by omega) (by omega) h)
      have := hi i (by omega)
      omega
  exact sorted_pairing k mu lam r iota hm (fun i j hij hj => hl i j hij (by
      rcases hkn with h | h
      · omega
      · omega)) into inj close

/-! ### C-16: multiplicity counts in a bracket -/

theorem sum_map_zero (l : List Nat) : (l.map (fun _ => (0 : Nat))).sum = 0 := by
  induction l with
  | nil => rfl
  | cons a t ih => simp [ih]

theorem sum_map_add' (l : List Nat) (f g : Nat → Nat) :
    (l.map (fun j => f j + g j)).sum = (l.map f).sum + (l.map g).sum := by
  induction l with
  | nil => rfl
  | cons a t ih => simp only [List.map_cons, List.sum_cons, ih]; omega

/-- Indicator sum: sum_{j < J} j [m = j] = m when m < J. -/
theorem indicator_sum (m : Nat) : ∀ J, m < J → ((List.range J).map (fun j => j * (if m = j then 1 else 0))).sum = m := by
  intro J hJ
  induction J with
  | zero => omega
  | succ J ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    by_cases h : m = J
    · subst h
      have z : ((List.range m).map (fun j => j * (if m = j then 1 else 0))).sum = 0 := by
        have : ∀ j ∈ List.range m, j * (if m = j then 1 else 0) = 0 := by
          intro j hj; rw [List.mem_range] at hj
          have : m ≠ j := by omega
          simp [this]
        rw [List.map_congr_left this, sum_map_zero]
      rw [z]; simp
    · rw [ih (by omega)]; simp [h]

theorem indicator_count (m : Nat) : ∀ J, m < J → ((List.range J).map (fun j => if m = j then 1 else 0)).sum = 1 := by
  intro J hJ
  induction J with
  | zero => omega
  | succ J ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    by_cases h : m = J
    · subst h
      have z : ((List.range m).map (fun j => if m = j then 1 else 0)).sum = 0 := by
        have : ∀ j ∈ List.range m, (if m = j then 1 else 0) = 0 := by
          intro j hj; rw [List.mem_range] at hj
          have : m ≠ j := by omega
          simp [this]
        rw [List.map_congr_left this, sum_map_zero]
      rw [z]; simp
    · rw [ih (by omega)]; simp [h]

/-- d_j: the number of distinct roots in the bracket with multiplicity j (the Sturm counts of s_j). -/
def dCount (rs : List (Rat × Nat)) (j : Nat) : Nat := (rs.filter (fun p => p.2 == j)).length

theorem sum_mul_dCount (rs : List (Rat × Nat)) (J : Nat) (hJ : ∀ p ∈ rs, p.2 < J) :
    ((List.range J).map (fun j => j * dCount rs j)).sum = (rs.map (·.2)).sum := by
  induction rs with
  | nil => simp [dCount]; exact sum_map_zero _
  | cons p t ih =>
    have ht : ∀ q ∈ t, q.2 < J := fun q hq => hJ q (by simp [hq])
    have key : ∀ j, j * dCount (p :: t) j = j * (if p.2 = j then 1 else 0) + j * dCount t j := by
      intro j; simp only [dCount, List.filter_cons]; split <;> simp_all <;> grind
    rw [List.map_congr_left (fun j _ => key j), sum_map_add', ih ht, indicator_sum p.2 J (hJ p (by simp))]
    simp only [List.map_cons, List.sum_cons]

theorem sum_dCount (rs : List (Rat × Nat)) (J : Nat) (hJ : ∀ p ∈ rs, p.2 < J) :
    ((List.range J).map (fun j => dCount rs j)).sum = rs.length := by
  induction rs with
  | nil => simp [dCount]; exact sum_map_zero _
  | cons p t ih =>
    have ht : ∀ q ∈ t, q.2 < J := fun q hq => hJ q (by simp [hq])
    have key : ∀ j, dCount (p :: t) j = (if p.2 = j then 1 else 0) + dCount t j := by
      intro j; simp only [dCount, List.filter_cons]; split <;> simp_all <;> omega
    rw [List.map_congr_left (fun j _ => key j), sum_map_add', ih ht, indicator_count p.2 J (hJ p (by simp))]
    simp; omega

/-- C-16 (i)-(iii) as counting: for the distinct roots in the bracket with their multiplicities (all < J),
sum_j d_j is the number of distinct roots, sum_j j d_j the number counted with multiplicity, and if these are
1 and g, the bracket holds exactly one root and its multiplicity is g ("exactly g-fold degenerate"). -/
theorem c16_single_root (rs : List (Rat × Nat)) (J g : Nat) (hJ : ∀ p ∈ rs, p.2 < J)
    (h1 : ((List.range J).map (fun j => dCount rs j)).sum = 1)
    (hg : ((List.range J).map (fun j => j * dCount rs j)).sum = g) :
    ∃ x, rs = [(x, g)] := by
  rw [sum_dCount rs J hJ] at h1
  rw [sum_mul_dCount rs J hJ] at hg
  match rs, h1 with
  | [p], _ => exact ⟨p.1, by simp at hg; rw [← hg]⟩

/-! ### Theorem 8: counts add over sectors -/

/-- The number of eigenvalues of the direct sum below a cut is the sum of the sectors' counts. -/
theorem pooled_count (Ls : List (List Rat)) (c : Rat) :
    ((Ls.flatten).filter (fun x => decide (x < c))).length = (Ls.map (fun L => (L.filter (fun x => decide (x < c))).length)).sum := by
  induction Ls with
  | nil => simp
  | cons L t ih =>
    simp only [List.flatten_cons, List.filter_append, List.length_append, List.map_cons, List.sum_cons, ih]

/-- The pooled dimension: the sector dimensions add up to the dimension of the direct sum (hypothesis T1). -/
theorem pooled_length (Ls : List (List Rat)) : (Ls.flatten).length = (Ls.map List.length).sum := by
  induction Ls with
  | nil => simp
  | cons L t ih => simp [List.flatten_cons, ih]

end CertifyCore.Thm
