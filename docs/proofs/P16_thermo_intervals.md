# Proposition 16: thermodynamic intervals from a complete whole-spectrum enclosure (S7)

Implements: `certify_ed/thermo.py` (`thermodynamics`). Depends on Lemma 17 (`docs/proofs/L17_dyadic_exp_ln.md`)
for every arithmetic step and on Proposition 9 (`docs/theory.md`, Section 5) for the hypothesis. Notation as
in `docs/theory.md`, Section 9.3. The truncated-spectrum extension (C-06) is `docs/proofs/C06_truncated_thermodynamics.md`.

## 1. Statement

**Hypothesis (H).** A is Hermitian of dimension n with eigenvalues E_1 ≤ … ≤ E_n (with multiplicity). Given
are m intervals [l_k, h_k] with integer counts c_k ≥ 1, Σ_k c_k = n, such that every eigenvalue lies in one
of the intervals and interval k contains exactly c_k eigenvalues. The l_k, h_k are finite binary64 numbers
(taken as the exact dyadic rationals they are). k_B = 1. The temperatures T are finite binary64 numbers
with T > 0 and T ≥ 2^−192 (a smaller T is not representable on the grid of Lemma 17; the code raises
ZeroDivisionError and `certify()` reports INCONCLUSIVE). β = 1/T.

**Definitions.** Z = Σ_i e^{−βE_i}; F = −T ln Z; U = ⟨E⟩ = Σ_i E_i e^{−βE_i} / Z; Var(E) = ⟨E²⟩ − ⟨E⟩²;
C = β² Var(E). E_0 := min_k l_k; a_k := l_k − E_0 ≥ 0, b_k := h_k − E_0; e_i := E_i − E_0.

**Conclusion.** With interval arithmetic in the sense of Lemma 17 (every operation outward-rounded on the
grid, so that an operation on enclosures encloses the operation on any members), define
  β̂ = [1, 1] / [T, T]                                           (encloses β),
  W_k = c_k · exp(−β̂ · [a_k, b_k])                                 (encloses Σ_{i in interval k} e^{−βe_i}),
  Z̃ = Σ_k W_k                                                      (encloses Z e^{βE_0}),
  F̂ = E_0 − [T, T] · ln Z̃,
  Û = E_0 + (Σ_k W_k · [a_k, b_k]) / Z̃,
  V̂ = ((Σ_k W_k · [a_k, b_k]²) / Z̃ − ((Σ_k W_k · [a_k, b_k]) / Z̃)²) ∩ [0, ∞),
  Ĉ = β̂² · V̂,
  ln Ẑ = ln Z̃ − β̂ · E_0,  Ẑ = exp(ln Ẑ).
Then F ∈ F̂, U ∈ Û, Var(E) ∈ V̂, C ∈ Ĉ, ln Z ∈ ln Ẑ and Z ∈ Ẑ. The binary64 endpoints reported by the code
(`to_float_down`/`to_float_up`) contain the grid intervals, hence the exact values.

The proposition makes no claim about the *width* of the intervals; Section 5 says what it is in the point
spectrum case, and the record carries the sentence `LOOSENESS` about U and C.

## 2. Proof

Everything follows from the enclosure property of Lemma 17 (an operation on enclosures encloses the
operation on members) once the exact quantities are written as expressions in the enclosed inputs. We
verify each line.

*β.* T is exact on the grid (from_float), 1/T ∈ [1,1]/[T,T] by Lemma 17 (ii); T > 2^−192 keeps 0 out of
the divisor.

*Shift.* e^{−βE_i} = e^{−βE_0} e^{−βe_i} with e_i = E_i − E_0 ≥ 0, since E_0 ≤ l_k ≤ E_i for every i. The
shift makes every exp argument −βe_i ≤ 0, so the enclosures of Lemma 17 (iii) only ever divide by exp of
a nonnegative argument, and e ↦ e² is increasing on each [a_k, b_k] ⊂ [0, ∞) (used for V̂). a_k and b_k are
formed as from_fraction(l_k − E_0), from_fraction(h_k − E_0) of exact rationals: enclosures of width at most
one grid unit, taken as the interval [a_k^↓, b_k^↑] (lower end of the first, upper end of the second).

*Weights.* For E_i in interval k, e_i ∈ [a_k, b_k], so −βe_i ∈ −β̂·[a_k, b_k] (product of enclosures, negated
exactly) and e^{−βe_i} ∈ exp(−β̂·[a_k, b_k]) (Lemma 17 (iii): exp of an enclosure). The c_k eigenvalues of
interval k contribute Σ_{i∈k} e^{−βe_i} ∈ c_k · exp(−β̂·[a_k, b_k]) = W_k (a sum of c_k members of one
interval lies in c_k times the interval; `scale_int` is exact). Summing over k (exact):
Z e^{βE_0} = Σ_i e^{−βe_i} ∈ Σ_k W_k = Z̃. The exact value Z e^{βE_0} is positive; the lower end of Z̃ is
positive unless the lower end of every W_k rounds to 0 on the grid (an exp argument −βb_k below about −133
gives a lower end of 0 or 1 unit), which happens only when β(b_k − a_k) is that large for *every* interval,
i.e. the enclosure is far too wide for the temperature. In that case `ln_interval` raises ValueError and
`certify()` reports INCONCLUSIVE; otherwise ln Z̃ is defined.

*Free energy.* F = −T ln Z = −T (ln Z̃_exact − βE_0) where Z̃_exact = Z e^{βE_0} ∈ Z̃; and −T·(−βE_0) = E_0
exactly, so F = E_0 − T ln Z̃_exact ∈ E_0 − [T,T]·ln Z̃ by Lemma 17 (iv) (ln of an enclosure) and (i)–(ii).
The code uses this form rather than −T·ln Ẑ to avoid multiplying the enclosure of β by T, which would widen
the result by a grid unit.

*Partition function.* ln Z = ln Z̃_exact − βE_0 ∈ ln Z̃ − β̂·E_0 = ln Ẑ; Z = exp(ln Z) ∈ exp(ln Ẑ) = Ẑ
(Lemma 17 (iii)). When the conversion of Ẑ to binary64 overflows, the record carries Z = None and only
ln Z; nothing false is stated.

*Internal energy.* U = E_0 + ⟨e⟩ with ⟨e⟩ = Σ_i e_i e^{−βe_i} / Σ_i e^{−βe_i}. For E_i in interval k the
product e_i e^{−βe_i} is a product of a member of [a_k, b_k] and a member of exp(−β̂·[a_k, b_k]), so it lies
in the interval product [a_k, b_k] · X_k, X_k := exp(−β̂·[a_k, b_k]); the c_k members sum into c_k times
that set. The code forms W_k · [a_k, b_k] with W_k = c_k X_k; the corner products of W_k · [a_k, b_k] are
c_k times those of X_k · [a_k, b_k], and outward rounding of c_k p/2^P encloses c_k times the exact corner
value, so W_k · [a_k, b_k] ⊇ c_k · ([a_k, b_k] · X_k) ∋ Σ_{i∈k} e_i e^{−βe_i}. Summing over k and dividing by Z̃ (Lemma 17 (ii); the
quotient of an enclosure of the numerator by an enclosure of the denominator encloses the quotient, since
0 ∉ Z̃) gives ⟨e⟩ ∈ (Σ_k W_k · [a_k, b_k]) / Z̃, and U ∈ E_0 + that.

*Variance and heat capacity.* Var(E) = Var(e) because a shift leaves the variance unchanged
(⟨(e+E_0)²⟩ − ⟨e+E_0⟩² = ⟨e²⟩ − ⟨e⟩²). ⟨e²⟩ = Σ_i e_i² e^{−βe_i} / Z̃_exact; for E_i in interval k, e_i² ∈
[a_k, b_k]² = [a_k², b_k²] (square of an interval in [0, ∞), Lemma 17 (ii)), so as above ⟨e²⟩ ∈
(Σ_k W_k · [a_k, b_k]²) / Z̃. ⟨e⟩² ∈ (⟨e⟩-enclosure)² (Lemma 17 (ii), any sign case). Their difference
encloses Var(e). Var(e) = ⟨(e − ⟨e⟩)²⟩ ≥ 0, so intersecting with [0, ∞) keeps the exact value (Lemma 17 (ii),
intersect_nonneg; the enclosure before intersection has upper end ≥ Var(e) ≥ 0, so the intersection is
never empty). C = β² Var(E) ∈ β̂² · V̂ (square of β̂ ⊂ (0, ∞), then product). The identities U = ⟨E⟩ and
C = ∂U/∂T = β² Var(E) at k_B = 1 are the textbook canonical-ensemble relations (e.g. Huang, *Statistical
Mechanics*, 2nd ed., §7.1; or any text: C = ∂U/∂T with ∂β/∂T = −β², ∂U/∂β = −Var(E)).

*Conversion.* Each reported endpoint is to_float_down of the lower grid end and to_float_up of the upper
(Lemma 17 (v)), so the binary64 interval contains the grid interval. ∎

The proposition uses no property of the spectrum beyond (H): any n reals distributed as the counts say give
the same bounds, which is why the verification can draw random spectra consistent with the enclosure.

## 3. Where the code implements it

`certify_ed/thermo.py`, `thermodynamics` (line numbers of this revision):

* input validation, counts, completeness against `dimension`: lines 49–61; `E0 = min(lo)` (62),
  `shifted` and `e_ivs` = [a_k^↓, b_k^↑] (63–65; from_fraction floors the lower, ceils the upper end).
* β̂ = `one / T_iv` (74); the placeholder on line 72 is dead code kept from 3.1.
* W_k: `arg = -(beta * e_iv)`, `exp_interval(arg).scale_int(c)` (76–79); Z̃ = `sum_intervals(W)` (80).
* ln Ẑ, Ẑ: lines 83–89 (Z = None on OverflowError / ValueError from the binary64 conversion).
* F̂ = `E0_iv - T_iv * lnZt` (90–93).
* Û: `num1`, `e_mean = num1 / Zt`, `U = E0_iv + e_mean` (94–98).
* V̂, Ĉ: `num2` with `e.square()`, `e2_mean`, `var = (e2_mean - e_mean.square()).intersect_nonneg()`,
  `C = beta.square() * var` (99–105).
* Conversion: `.floats()` on every reported interval (Lemma 17 (v)).

Rounding direction: every operation is the outward-rounded operation of `ivl.py`; no floating-point
arithmetic occurs between the exact reading of the inputs (`from_float`) and the final conversion.

## 4. Computational verification

`docs/proofs/verify_P16.py` (run: `python docs/proofs/verify_P16.py --random 400 --seed 0`):

* 400 random enclosures (1–6 intervals, counts 1–3, energy scale 10^−2 … 10^3, widths 0 or 10^−14 … 10^−1
  relative, 30 % point intervals), two random temperatures each (10^−2 … 10^2 times the energy scale); for
  each temperature five spectra consistent with the enclosure (two random, all at the lower ends, all at
  the upper ends, each eigenvalue at a random end): 4000 cases. Ground truth: mpmath interval arithmetic
  (`mpmath.iv`) at 200 digits, so that the reference itself is an enclosure of the exact value (a point
  reference at 200 digits gives C = −1e−200 for a single level, where the exact C is 0); a case passes when
  the implementation's interval meets the reference interval (largest reference width over the run:
  1.5e−191, printed as `reference_max_width`).
* Adversarial enclosures (95 cases with their extreme spectra): the two-level system at T = 10^−3, 1, 10^6;
  a single four-fold level (C = 0 exactly, the lower end of Ĉ must be 0); touching intervals [0, 1], [1, 2];
  energies ±10^6 with a width 10^−9 at T = 10^3 and 10^7; a gap of 2000 at T = 1 (weight e^{−2000} ≈ 10^−869,
  below binary64 range, handled on the grid); nested intervals; energies 10^−300 (below the grid resolution
  2^−192, enclosed as [0, 1 unit]) at T = 2^−190; one wide interval [−1, 1] with 5 eigenvalues.
* Tightness: for the point spectrum {0, 1} the widths of F̂, Û, Ĉ are ≤ 1.1e−16 (one binary64 ulp of the
  final conversion).

Result (seed 0): 4000 random + 95 adversarial cases, 0 violations.

Two findings of the verification that changed the code: (a) a temperature 10^−300 (below the grid) made
`one / T_iv` divide by an interval containing 0: the hypothesis T ≥ 2^−192 is now stated here and the
exception is caught by `certify()`; (b) β·e beyond ~10^4 made `exp_interval` attempt 2^190 squarings: the
underflow shortcut and the argument limit of Lemma 17 (iii) (`_EXP_ARG_UNDER`, `_EXP_ARG_MAX`) were added.

## 5. Analytic cross-check

*Two-level system.* E ∈ {0, Δ}: Z = 1 + e^{−βΔ}, F = −T ln(1 + e^{−βΔ}), U = Δ e^{−βΔ}/(1 + e^{−βΔ}),
C = β²Δ² e^{−βΔ}/(1 + e^{−βΔ})² (the Schottky anomaly). With point intervals the enclosures have widths of a
few grid units before conversion and one ulp after (Section 4, tightness line); `tests/test_thermo.py::
test_two_level_system_against_closed_form` checks the closed form at five temperatures.

*Single level, g-fold.* Z = g e^{−βE}, F = E − T ln g, U = E, C = 0. Here Z̃ = g·[1, 1] exactly (exp(0) =
1 on the grid), ln Z̃ ∋ ln g, Û = [E, E] exactly and V̂ = [0, 0] ∩ [0, ∞) = [0, 0]: the enclosures reproduce
the closed form with zero width in U and C.

*Limits.* T → ∞ (β → 0): all weights → 1, Z̃ → [n, n], Û → mean energy, V̂ → the population variance of the
levels: the enclosure interpolates the high-temperature limit since exp(−β̂·[a, b]) → [1, 1]. T → 0: W for the
ground level → c_0·[1, 1] and all others → [0, ·]; F̂ → E_0 − T ln c_0, the residual entropy ln c_0 of a
c_0-fold ground state.

*Width in the point-spectrum case.* If every interval has zero width, each W_k has the width of the exp
enclosure (a few grid units, Lemma 17), so the widths of F̂, Û, Ĉ are O(n · 2^−192 · scale) before the
binary64 conversion: the conversion dominates, as observed.

## 6. Status

Derived by Claude on 2026-10-30; machine-checked (4000 random + 95 adversarial cases, 0 violations);
independent re-derivation: pending (Phase 2); Sarang's check (D-12): pending.
