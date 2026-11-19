# Independent re-derivation: Proposition 16′ (C-06, thermodynamics from the k lowest eigenvalues and a tail bound)

Written on 2026-10-30 before the original proof was read (not a human check). Sections 1 and 3 of `../C06_truncated_thermodynamics.md` read first, with
the `tail` branch of `thermo._rows` (lines 72–79, 94–130, 146, 151); proof written before section 2 was opened.
Proposition 16 and Lemma 17 as re-derived in `independent/P16_thermo_intervals.md`, `independent/L17_dyadic_exp_ln.md`.

## Statement checked

Lemma C (envelopes of φ(e) = e·e^{−βe}, ψ(e) = e²e^{−βe}); Proposition 16′ (i) omitted weight, (ii) omitted
moments, (iii) the intervals for Z, F, U, Var, C under (H′).

## My proof

**Lemma C.** φ′(e) = e^{−βe}(1 − βe): positive on [0, 1/β), zero at 1/β, negative after; φ(1/β) = e⁻¹/β; φ > 0 on
(0, ∞) and φ → 0 at ∞ (exponential beats linear). ψ′(e) = e·e^{−βe}(2 − βe): increasing on [0, 2/β], decreasing
after; ψ(2/β) = 4e⁻²/β²; ψ → 0 at ∞. A continuous function increasing up to e* and decreasing after attains its
supremum over [ℓ, μ] at the point of [ℓ, μ] nearest to e*, i.e. at clamp(e*, ℓ, μ), and its infimum over
[ℓ, μ] at an endpoint (min(φ(ℓ), φ(μ))); over [ℓ, ∞) the infimum is the limit 0 (not attained). ∎

**(i).** By (H′)(b), E_{k+1} ≥ L, hence e_j ≥ ℓ for all j > k (the eigenvalues are sorted), and by (d) e_j ≤ μ. So
e^{−βe_j} ∈ [e^{−βμ}, e^{−βℓ}] for each of the n − k omitted eigenvalues (β > 0; e^{−βμ} := 0 when μ = ∞), and their
sum lies in (n − k)·[e^{−βμ}, e^{−βℓ}] (n − k members of an interval). In particular Σ_{j>k} e^{−βE_j} =
e^{−βE_0}Σ_{j>k} e^{−βe_j} ≤ (n − k)e^{−βE_0}e^{−βℓ} = (n − k)e^{−βL}. ✓

**(ii).** For j > k, e_j ∈ [ℓ, μ], so φ(e_j) ∈ [inf_{[ℓ,μ]}φ, sup_{[ℓ,μ]}φ] and likewise for ψ; the sums of n − k members
lie in (n − k) times those intervals. ✓

**(iii).** Z e^{βE_0} = Σ_{j≤k} e^{−βe_j} + Σ_{j>k} e^{−βe_j}. The first sum lies in Σ_j W_j exactly as in Proposition 16
((H′)(a) is (H) for the k lowest eigenvalues). For the second, (i) gives membership of (n − k)·[e^{−βμ}, e^{−βℓ}];
the code's W_t = (n − k)·[exp(−β̂μ̂)^↓, exp(−β̂ℓ̂)^↑] encloses it: β̂ ∋ β, ℓ̂ ∋ ℓ (exact rational difference,
outward), μ̂ ∋ μ, so −β̂ℓ̂ ∋ −βℓ and exp(−β̂ℓ̂).hi ≥ e^{−βℓ} (exp rule); exp(−β̂μ̂).lo ≤ e^{−βμ} (or 0 when M is
absent); `scale_int(n − k)` exact. Hence Z e^{βE_0} ∈ Σ_j W_j + W_t = Z̃. The moments: Σ_{j≤k} e_j e^{−βe_j} ∈
Σ_j W_j·[a_j, b_j] (Proposition 16) and Σ_{j>k} e_j e^{−βe_j} = Σ_{j>k} φ(e_j) ∈ (n − k)·[inf φ, sup φ] ⊆ T_1 when
φ_inf ≤ inf_{[ℓ,μ]}φ and φ_sup ≥ sup_{[ℓ,μ]}φ, which the code's evaluation on enclosures gives: with T̂ = [T, T]
exact and 1/β = T, (α) if ℓ̂.lo ≥ T then ℓ ≥ 1/β, the clamp is ℓ, and sup φ = φ(ℓ) ≤ (ℓ̂·exp(−β̂ℓ̂)).hi; (β) if
μ̂.hi ≤ T then μ ≤ 1/β, sup φ = φ(μ) ≤ (μ̂·exp(−β̂μ̂)).hi; (γ) otherwise sup_{[ℓ,μ]}φ ≤ sup_{[0,∞)}φ = e⁻¹T ≤
(T̂·exp([−1, −1])).hi — a valid (possibly loose) bound in every case, including those where (α) or (β) hold but
were not detected because the enclosures straddle T. inf: 0 when μ = ∞ (correct lower bound since φ > 0), else
min(φ(ℓ), φ(μ)) ≥ min((ℓ̂·w_l).lo, (μ̂·w_m).lo). The same for ψ with 2T (`two_T = T̂·2` exact) and 4e⁻²T² ≥ sup ψ
((T̂²·exp([−2, −2])).scale_int(4).hi). So T_1 ∋ Σ_{j>k}φ(e_j), T_2 ∋ Σ_{j>k}ψ(e_j). With num1 = Σ_j W_j·[a_j, b_j] + T_1
∋ Σ_{all} e_j e^{−βe_j} and num2 likewise, the formulas for U, Var, C, F, ln Z, Z are those of Proposition 16 with
the complete sums replaced by these enclosures, and the same argument gives containment (division by Z̃ ∌ 0,
Var ≥ 0 for the intersection, F = E_0 − T ln Z̃_exact). Two-sidedness: every interval above has finite ends
(W_t.lo = 0 is a finite lower end; the upper ends are finite because ℓ > 0 and exp arguments ≤ 0), so Z, F, U,
Var, C are enclosed on both sides with or without M. ✓ ∎

Where the hypotheses enter: (b) with L > h_j ≥ l_j ≥ E_0 gives ℓ > 0 (so exp(−βℓ) < 1 and the tail weight bound
is meaningful; ℓ ≤ 0 is rejected by the code); (d) M ≥ L makes [ℓ, μ] non-empty (checked, ValueError otherwise);
(c) the dimension n fixes the number of omitted eigenvalues — if the true dimension were larger the tail count
would be wrong, which is why n is the operator's dimension, not a claimed number.

### Code (current `thermo.py`)

tail setup 72–79 (ℓ̂ positivity check 75–76, μ̂, e⁻¹, e⁻² enclosures), per temperature 94–130: `w_l`, `w_m` 98–99,
φ, ψ at the ends 100–102, regime (α)/(β)/(γ) 104–109, ψ regime 110–116, infima 117–119, W_tail/T1/T2 120–122,
Z̃ 127, moments 146 and 151; `thermodynamics_truncated` 181–221 (k ≤ n 195–196, L > max h 197–199, k = n delegates
202–204, M ≥ L 208–209). certify.py S7 branch 1452–1473 (`thermodynamics_truncated` with `upper_bound =
add_up(norm_upper, eps_s3)` upstream; ValueError/ZeroDivisionError/OverflowError → INCONCLUSIVE). Rounding:
every tail quantity is an `ivl` operation; the regime tests compare conservative ends (ℓ̂.lo against T̂.hi,
μ̂.hi against T̂.lo), so a regime is only claimed when it holds for every member. ✓

## Comparison with section 2 of the original (read after the proof above was written)

Identical in content: Lemma C by the derivatives, (i)–(ii) by membership of [ℓ, μ], (iii) by splitting the sums,
the conservative regime tests, the global maximum as the fallback, infima at the ends or 0. Same constants. Both
valid. One imprecision in the original's last sentence of (iii): "Z̃'s lower end is positive (the ground-level
term has exp(0) = 1 exactly)" — the lower end of W_j for the ground interval is c_j·exp(−β̂·b_j)^↓, not exp(0); it
is 0 on the grid when β b_j ≳ 133 (wide ground interval at low T), and then `ln_interval` raises and S7 is
INCONCLUSIVE, exactly as noted for Proposition 16. Not a soundness issue (nothing false is stated).

**Code mapping**: as in my code check (current `thermo.py` lines); section 3 gives no line numbers, the
description matches.

**Verdict: AGREES-WITH-REMARKS** (one incorrect side remark about Z̃'s lower end; the statement and the bounds
are unaffected).

## Runs

- `verify_C06.py --random 150 --seed 7`: 150 random + 75 adversarial, 7 negative controls detected; 0 violations.
- `adversarial_C06_truncated_thermodynamics.py --cases 200 --seed 43`: Lemma C 200 (mp grid, β and ℓ over six
  decades, μ − ℓ from 10⁻⁶ℓ to 10³ℓ); containment 276 random + 1868 adversarial (tail at L, at M, at the argmax of
  φ and ψ, spread; T = ℓ and 2ℓ exactly; M = L; n − k = 10⁶; β ℓ > 133; T = 10⁶; scales 10^±100; M absent;
  μ − ℓ = 10⁻¹⁵ with T between); negative controls 30; no-value cases 44 (all legitimate); 0 violations. (Two
  earlier runs reported violations that were artifacts of my test: an exact-equality check at 120 digits and
  a generator with k > n.)
