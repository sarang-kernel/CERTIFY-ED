# Independent re-derivation: Proposition 16 (thermodynamic intervals from a complete whole-spectrum enclosure)

Fresh-context session, 2026-10-30. Sections 1 and 3 of `../P16_thermo_intervals.md` read first, with
`certify_ed/thermo.py`; proof written before section 2 was opened. Lemma 17 (enclosure property of every `ivl`
operation, re-derived in `independent/L17_dyadic_exp_ln.md`) is used as a black box: "X ∋ x and Y ∋ y imply
op(X, Y) ∋ op(x, y)" for +, −, ·, /, square, exp, ln, `scale_int`, `sum_intervals`, `intersect_nonneg` (for a
member ≥ 0), and `floats()` returns binary64 ends enclosing the grid interval.

## Statement checked

Under (H), F ∈ F̂, U ∈ Û, Var(E) ∈ V̂, C ∈ Ĉ, ln Z ∈ ln Ẑ, Z ∈ Ẑ, with the intervals defined in section 1.

## My proof

Write w_i := e^{−βe_i} (e_i = E_i − E_0 ≥ 0 because E_0 = min_k l_k ≤ E_i by (H)) and Z̃_exact := Σ_i w_i = Z e^{βE_0}.

*Inputs.* E_0, T, l_k, h_k are binary64, hence dyadic, and `from_float` represents them exactly (Lemma 17 (v));
a_k^↓ := ⌊(l_k − E_0)2ᴾ⌋ ≤ a_k and b_k^↑ := ⌈(h_k − E_0)2ᴾ⌉ ≥ b_k (`from_fraction` of the exact rational
differences), so [a_k^↓, b_k^↑] ∋ e_i for every eigenvalue E_i of interval k (there are exactly c_k of them by
(H)). β̂ = [1, 1]/[T, T] ∋ 1/T = β by the division rule (T > 0 and T ≥ 2⁻ᴾ so the grid interval of T excludes 0;
below that `from_float(T)` has lower end 0 and the division raises, as section 1 states).

*Weights.* For i in interval k: β e_i ∈ β̂ · [a_k^↓, b_k^↑] (product rule, both factors enclosed), so
−β e_i ∈ −(β̂·[a, b]) (negation) and w_i ∈ exp(−(β̂·[a, b])) (exp rule, exp increasing). Summing the c_k
members: Σ_{i∈k} w_i ∈ c_k · exp(·) = W_k (`scale_int`, c_k ≥ 1 an integer, exact). Hence Z̃_exact ∈ Σ_k W_k = Z̃
(`sum_intervals`, exact). Z̃.lo may be 0: the lower end of W_k is c_k·exp(−β̂.hi·b_k^↑) rounded down, which is 0 on
the grid when β b_k ≳ 133 (exp(−133) < 2⁻¹⁹²); this happens for every interval, including the one attaining
E_0 when it is wide. Then `ln_interval(Z̃)` raises ValueError and the code reports no value (`certify()` catches
ValueError, ZeroDivisionError and OverflowError and makes S7 INCONCLUSIVE, certify.py 1490–1491). The
conclusion below therefore holds whenever the quantities are defined, i.e. when Z̃.lo > 0; section 1 mentions
only the ZeroDivisionError of T < 2⁻ᴾ, not this second no-value case. (An earlier draft of this paragraph
claimed Z̃.lo ≥ 2ᴾ always; that was wrong — the lower end of an exp enclosure is exp of the *lower* end of the
argument interval, exp(−β b_k), not exp(−β a_k).)

*F.* F = −T ln Z = −T(ln Z̃_exact − βE_0) = E_0 − T ln Z̃_exact (βT = 1 exactly). With ln Z̃_exact ∈ ln Z̃ (ln rule,
Z̃.lo > 0), T ∈ [T, T] exact, E_0 ∈ E0_iv exact: F ∈ E0_iv − [T, T]·ln Z̃ = F̂ (product and subtraction rules). ✓

*U.* U = Σ_i E_i w_i / Z = E_0 + Σ_i e_i w_i / Z̃_exact (multiply numerator and denominator by e^{βE_0}). For i in
interval k, e_i w_i ∈ [a_k^↓, b_k^↑] · exp(·) and the c_k members sum into c_k·([a, b]·exp(·)) ⊆ W_k · [a_k^↓, b_k^↑]
(for integer c ≥ 1 and intervals X, Y: c·(X·Y) ⊆ (cX)·Y, since the corner products of (cX, Y) are c times those
of (X, Y) and ⌊cp/2ᴾ⌋ ≥ c⌊p/2ᴾ⌋, ⌈cp/2ᴾ⌉ ≤ c⌈p/2ᴾ⌉; the code forms `w * e` with w = W_k, i.e. (cX)·Y directly,
which encloses every member product c·x·y with x ∈ X, y ∈ Y — and Σ_{i∈k} e_i w_i is such a product? Not of a
single pair; but it lies in c_k·(X·Y) = {c·z : z ∈ X·Y} since each term e_i w_i ∈ X·Y and the mean of c_k members
of an interval is a member: Σ_{i∈k} e_i w_i = c_k·(mean) ∈ c_k·(X·Y) ⊆ (c_kX)·Y.) Hence Σ_i e_i w_i ∈ Σ_k W_k·[a, b]
= num1, Σ_i e_i w_i/Z̃_exact ∈ num1/Z̃ (division rule, 0 ∉ Z̃), and U ∈ E0_iv + num1/Z̃ = Û. ✓

*Var and C.* Var(E) = ⟨E²⟩ − ⟨E⟩² is invariant under the shift E → E − E_0, so Var(E) = Σ e_i² w_i/Z̃_exact −
(Σ e_i w_i/Z̃_exact)². As above Σ_i e_i² w_i ∈ Σ_k W_k·[a, b]² = num2 ([a, b]² by the square rule, a ≥ 0 so the
first case), the quotient ∈ num2/Z̃, the square of the mean ∈ (num1/Z̃)², and the difference ∈ num2/Z̃ − (num1/Z̃)².
Var(E) ≥ 0 (a variance), so Var(E) ∈ (…) ∩ [0, ∞) = V̂ (`intersect_nonneg` keeps every member ≥ 0; it raises only
if the whole interval is negative, which cannot happen because Var(E) is a member). C = β²Var(E) ∈ β̂²·V̂ = Ĉ. ✓

*Z.* ln Z = ln Z̃_exact − βE_0 ∈ ln Z̃ − β̂·E0_iv = ln Ẑ; Z = exp(ln Z) ∈ exp(ln Ẑ) = Ẑ (exp rule; an argument above
2¹⁶ raises OverflowError and the code reports Z = None while keeping ln Z). ✓

*Conversion.* `floats()` = (to_float_down(lo), to_float_up(hi)) ⊇ the grid interval ⊇ the exact value (Lemma 17 (v)). ∎

Where (H) enters: completeness (every eigenvalue in some interval, exactly c_k in interval k) is what makes the
weight sums exact enumerations; without it Z̃ would miss or double-count members. E_0 = min l_k (not min E_i) is
enough: e_i ≥ 0 is all that is used (so that [a_k^↓, b_k^↑]² takes the non-negative branch and exp arguments
are ≤ 0, keeping every W_k ≤ c_k·[1, 1] — no overflow in exp for the weights; only Ẑ = exp(ln Ẑ) can overflow).

## Comparison with section 2 of the original (read after the proof above was written)

Same proof line by line (shift, weights with the c_k-member argument, F in the form E_0 − T ln Z̃, U with the
corner-product argument for W_k·[a, b] ⊇ c_k·([a, b]·X_k), shift invariance of the variance, intersect with [0, ∞),
conversion). The original *does* treat the Z̃.lo = 0 no-value case (ValueError → INCONCLUSIVE) in its *Weights*
paragraph — the point I had flagged against section 1 — so the only imprecision is that section 1 names the
ZeroDivisionError (T < 2⁻ᴾ) but not this case. Same constants (none beyond the grid).

**Remark (added by the adversarial run).** β̂ = [1, 1]/[T, T] has width one grid unit, so for T > 2¹⁹² ≈ 6·10⁵⁷
(β below the grid) β̂ = (0, 2⁻ᴾ) and the enclosures are those of β ∈ [0, 2⁻¹⁹²]; with energy ranges b_k ≳ 10⁶⁰
this makes the lower end of every W_k zero and the code reports no value (INCONCLUSIVE). Not a soundness
issue (nothing false is stated), but the document could say that the method needs β·b_k and 2⁻¹⁹²·b_k below
≈ 130 for at least one interval.

**Code mapping (current `thermo.py`, 221 lines after the C-06 extension).** Section 3's line numbers refer to
the 112-line revision; current: `_read_intervals` 50–60, E_0 and shifted intervals 66–70, β̂ 86, W_k 88–91,
Z̃ 132, ln Ẑ / Ẑ 133–140, F̂ 141–144, Û 145–149, V̂ / Ĉ 150–156, conversion by `.floats()` throughout,
`thermodynamics` 161–178 (dimension check 171–172). Rounding: every step is an `ivl` operation; no float
arithmetic between `from_float` and `.floats()`. ✓

**Verdict: AGREES-WITH-REMARKS** (section 3 line numbers stale; section 1 omits the second no-value case).

## Runs

- `verify_P16.py --random 200 --seed 7`: 200 random enclosures (2000 cases) + 95 adversarial; 0 violations;
  point-spectrum max width 1.1e-16 (binary64 conversion), grid width 1.5e-191.
- `adversarial_P16_thermo_intervals.py --cases 200 --seed 31`: containment 160 random + 1120 adversarial
  (members at the ends, midpoints, spread; degenerate levels, counts 10⁶, scales 10^±150, offset 10²⁰⁰, T at
  2⁻¹⁹² and 10³⁰⁰, adjacent-float ends, n = 1, 40 narrow intervals); negative controls 80; no-value cases 100
  (ValueError only when β̂.hi·b_k > 130 for every interval); 0 violations.
