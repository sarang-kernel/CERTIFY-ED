# Independent re-derivation verdicts (Phase 2, rule D-12/D-59)

Fresh-context session, 2026-10-30. For every derivation document in `docs/proofs/` the statement (section 1) and the
code mapping (section 3) were read first, an independent proof was written in `docs/proofs/independent/<id>.md`,
then section 2 of the original was compared step by step, the original `verify_<id>.py` was run at a reduced
size and an independent adversarial search `docs/proofs/independent/adversarial_<id>.py` was run (exact arithmetic or
mpmath at 60 digits as ground truth). Verdict classes: AGREES, AGREES-WITH-REMARKS, GAP, ERROR. None of these
results is described as verified; Sarang's check (D-12) is pending for all of them.

| document | statement checked | protocol followed | verdict | remarks | original verify run (reduced) | independent adversarial counts | violations | date |
|---|---|---|---|---|---|---|---|---|
| appA_model-shift.md | Lemmas A1-A3 (builder error, symmetrisation, norm bound Delta_builder = gamma_W sqrt(rc)), Theorem A4 (Delta = max(\|\|D\|\|_1, \|\|D\|\|_inf) + Delta_builder), Theorem A5 (Weyl transfer) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Proof valid, same constants. Remark: Lemma A2 is stated in the max(a_ij, a_ji) form, which does not imply the 2-norm bound for an asymmetric \|H\|_abs (counterexample a=[[1,1],[0,0]]); the proof of A3 correctly uses the (a+a^T)/2 form instead - state it in A2. Two classical facts (entrywise monotonicity of the 2-norm, \|H^\| <= (1+gamma)(a+a^T)/2) used unnamed. Budget: one spare unit of W, not two. Code mapping agrees (certify.py 363-390, 459-478; fp.py 173-188). | run 2026-11-03 by the lead after the user's approval (`--cases 400`): 386 random + 681 adversarial, 139 negative controls, 0 violations (`docs/proofs/runs/appA_model-shift.txt`) | 16 cases seed 17 + 24 cases seed 18 (background batch): N1/W1/D1/A5 10 random + 30 adversarial each; 40 negative controls; 0 violations (~77 s per case: mp eigendecompositions of 81x81 spin-1 cases) | 0 | 2026-10-30 |
| C-03_entanglement.md | Entanglement entropy / spectrum certification (Props 1–3, Lemmas 3–5) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | _f_point raises for rational 0 < p < 2^-192 (record NOT_CHECKED, sound but avoidable); η_V Frobenius vs 2-norm (conservative) | verify_C-03.py 3: 21176 cases, 0 violations | adversarial_C-03_entanglement.py --cases 120 --seed 53: 0 violations | 0 | 2026-10-30 |
| C-04_correlation_lift.md | Lift of observables through the sector isometry (Lemma 0, Lemma 1, Prop 1, conclusion) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | GAP | Lemma 1 (ii) proof step "(1+γ̇_a)(1+γ̇_b) ≤ 1+γ̇_{a+b}" is false for the complex γ̇_m = 2γ_{m+4} (20u vs 12u at a=b=1); statement true via Higham constants (≤ γ_{a+b+4} ≤ γ̇_{a+b}/2); bounds unaffected. Prop 1 holds with g instead of 2g. | verify_C-04.py 5: 254 cases, 0 violations | adversarial_C-04_correlation_lift.py --cases 150 --seed 61: 0 violations | 0 | 2026-10-30 |
| C06_truncated_thermodynamics.md | Proposition 16' (C-06): omitted weight/moment envelopes (Lemma C) and two-sided intervals for Z, F, U, Var, C from the k lowest eigenvalues, L <= lambda_{k+1} and optionally M >= lambda_n | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Same proof. One wrong side remark in (iii): the lower end of Z~ is not always positive (ground interval's exp(-beta b)^v can be 0 on the grid); the code then raises and S7 is INCONCLUSIVE; no bound affected. | verify --random 150 --seed 7: 150 random + 75 adversarial, 7 negative controls; 0 violations | 200 cases seed 43: Lemma C 200; containment 276+1868; negative controls 30; no-value cases 44; 0 violations | 0 | 2026-10-30 |
| C16_exact_bracket.md | C-16: exact count, distinct values and multiplicities of the eigenvalues of the declared model in a rational bracket (charpoly, Yun, Sturm), and the three decision rules | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Proof identical and valid. Code: the certified sentence printed the outward-rounded binary64 bracket as the interval of the count; an eigenvalue within an ulp of an end made the sentence false (exact example: eigenvalues 1 + 2^-60 and 3/2, bracket [1 + 2^-52, 2]: 'exactly 1 in [1, 2]'). Fixed on this branch (wording names bracket_exact), negative-control test added; counts unchanged. | verify --random 100 --seed 7: A_planted_matrices: (300, 0) ; B_known_polynomials: (401, 0) ; C_sector_identity: (50, 0) ; D_models: (42, 0) ; random_cases: 751 ; adversarial_cases: 42 ; violations: 0 | 200 cases seed 47: roots_in_bracket 34+166 (known spectra, roots 2^-60 apart, ends on roots, factor 2), sturm_count 270, nudging 200; 0 violations | 0 | 2026-10-30 |
| cor5_pairing.md | Corollary 5 (completeness of a lowest claim, sorted pairing within uniform-radius groups, model shift Delta) and the sorted-pairing lemma | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES | Lemma proved by a different (counting) argument; Step 3 made explicit. Code mapping agrees at current lines certify.py 533-604, enclosure.py 597-626; all endpoints outward. | verify --cases 2000 --seed 7: lemma 2000+500, Cor5 174 (26 skipped), negatives 200; 0 violations | 300 cases seed 11: lemma 1786+941 exact; Cor5 202+998 exact; necessity 3/3 (H1a 41/300, H1c 277/300, H3= 42/300 failures when broken); code 13+82 (85 not CERTIFIED); 107 negative controls; 0 violations | 0 | 2026-10-30 |
| E-04_parameter_intervals.md | Eigenvalue tubes over a coupling interval (Props 1–2, Lemma 3 sampling criterion, Lemma 4, bounds for B) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | ERROR | certified_crossing_interval applied Lemma 3 across the tube jump at interior grid points: relation wrong by ≤ d_k^a + d_k^b (scalar counterexample). FIXED (looser: defects added to the slack) + negative-control test. Tubes unaffected. Lemma 4 stated too generally (true for the letters); sector branch Hermiticity-up-to-rounding remark. | verify_E-04.py 7: 1156 cases, 0 violations | adversarial_E-04_parameter_intervals.py --cases 120 --seed 71: 0 violations (pre-fix: 60 targeted crossing cases violate) | 0 | 2026-10-30 |
| E-07_davis_kahan.md | Davis–Kahan sin Θ bound s_DK, alignment, approximate-symmetry labels (C-12), fidelities (C-13), rounding lemma | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | GAP | Model-shift increment of ‖R‖_F stated as σ√(1+η) (2-norm bound) — Frobenius route needs σ√(g(1+η)); implemented bound valid via the 2-norm sin Θ theorem (interval separation holds). Remarks: np.abs(C)**2 vs re²+im² in Lemma 5(c); γ₀ = sep + w/2 relies on J = [a−ε, b+ε]. | verify_E-07.py 11: 557 cases, 0 violations | adversarial_E-07_davis_kahan.py --cases 100 --seed 83: 0 violations | 0 | 2026-10-30 |
| int05_underflow.md | (M-U) model; U1 dot products; U2 exact_dot; U3 norms with T_N; U4 \|Rc - r\| <= E + c_w u_min; U5 rel >= radius for any magnitude; U6 radii never shrink; U7 other additive terms | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | ERROR | Theorem U5 is false as stated ('no magnitude hypothesis'): when \|\|x\|\|^2 overflows (\|\|x\|\| > ~1.3e154) norm2_lower returned inf and rel = 2 ULP_MIN, a finite too-small radius; exact counterexample A = diag(1,2), mu = 1 + 1e-8, x = (1e160, 0) (true radius 1e-8, code 1e-323, group distinct). Pre-existing in the old chain. Corrected on this branch (looser): fp.norm2_lower returns 0 for a non-finite computed norm -> rel = inf; negative-control tests in tests/test_independent_check.py. Also a GAP in Lemma U4's proof for complex data (hypot absolute errors of \|a_ij\|, \|x_j\| not counted); statement holds via the factor 2 of the complex g. (M-U), U1-U3, U6, U7 agree. | verify --cases 3000 --seed 7: model 2853, U1 750, U2 950, U3 200773, U5 525, U6 375; 0 violations | seed 23: (M-U) 391, U1 300, U2 300, U3 300, U4 300, U5 300, U6 42, U7 100; 0 violations after the correction (11 U5 violations before it, all overflow cases) | 0 after correction (11 before: Theorem U5 overflow family) | 2026-10-30 |
| L17_dyadic_exp_ln.md | Lemma 17: enclosure property of the dyadic interval arithmetic (grid 2^-192), exp (halving + Taylor N=40 + squarings), ln (atanh series M=60, u <= 0.35, ln 2), float conversions | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES | Proofs identical step by step (nested-ceiling identity, tail bounds with 2 and 8/7, ln 2 = 2 atanh(1/3), nearest-float argument). Code lines of section 3 are current. | verify --random 600 --seed 7: grid_ops 600, exp 600 + 66 adversarial, ln 600 + 74 adversarial (1800 random + 140 adversarial), constants ok; 0 violations | 300 cases seed 29: arithmetic 206+94, exp 325, ln 319, float conversion 300+300, tightness 392+11; 0 violations | 0 | 2026-10-30 |
| P14_degeneracy_bracketing.md | Proposition 14 (a)-(e): bracketing inertia counts give nu_hi - nu_lo <= #I_out, #I_in <= nu_hi - nu_lo, and exactly g eigenvalues in I_in when J in I_in and nu_hi - nu_lo = g | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES | Identical proof. Remark: distinctness across several Theorem 2 groups in one cluster rests on the disjoint union intervals of enclose (implicit). Code mapping current. | verify --random 600 --seed 7: model 600+200, code 20 (75 CERTIFIED) + 6 adversarial; 0 violations | 300 cases seed 37: exact (a)-(e) 300+900; code 39+123 (47 not certified) vs exact model eigenvalues; 77 negative controls; 0 violations | 0 | 2026-10-30 |
| P15_summed_bounds.md | Proposition 15 (1)-(2): \|sum a_k tr(P O_k) - fl(sum a_k v_k)\| <= B = sum^ up(\|a\|^ b) + up(2 gamma_{m+2} sum^ up(\|a\|^ \|v\|^)); Lemma A modulus_upper >= \|z\|; Lemma B rounding of the complex dot product | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES | Identical proof. Section 3 line numbers off by one on the current file. D1/D2 correctly labelled as hypotheses. | verify --random 200 --seed 7: 461 random + 32 adversarial; 0 violations | 120 cases seed 41: Lemma A 60+300, Lemma B 20+100 (m <= 20000, cancellation, scales), (2) 20+101; 0 violations | 0 | 2026-10-30 |
| P16_thermo_intervals.md | Proposition 16: F, U, Var, C, ln Z, Z of a complete whole-spectrum enclosure lie in the dyadic interval expressions F^, U^, V^, C^, ln Z^, Z^ | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Same proof. Section 1 omits the no-value case Z~.lo = 0 (ValueError, INCONCLUSIVE), which section 2 treats; for T > 2^192 beta is below the grid and wide energy ranges give no value (remark). Section 3 line numbers are those of the 112-line revision (current thermo.py has 221 lines; mapping listed in the independent doc). | verify --random 200 --seed 7: 200 enclosures / 2000 cases + 95 adversarial; 0 violations | 200 cases seed 31: containment 160+1120, negative controls 80, no-value cases 100; 0 violations | 0 | 2026-10-30 |
| thm1_fp-accounting.md | Theorem 1-fl: rel >= \|\|Ax - mu x\|\|/\|\|x\|\| with stages (S2) absHX_up >= a_i, (S3) E_i >= \|Rc_i - r_i\|, (S4) norm guards 1 +- 3u, (S5) upper/rel | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | GAP | Main statement and (S3)-(S5) agree. (S2) for complex data is not justified: (Lambda5) assumes absHX >= a_i(1 - gamma_w), but the hypot roundings of \|a_ij\|, \|x_j\| give 1 - gamma_{w+2}; the route yields 1 - O(w u^2), not >= 1. Certified radius unaffected (Step 3 slack factor 2 for complex g). Close by restating (S2) for complex data or using gamma(w+3) for complex gw (looser). Remark: overflowing \|\|x\|\| gives NaN radius, handled fail-safe (IMP-22). Code mapping agrees (enclosure.py 286-302, 343-375; fp.py 149-162). | verify --cases 2000 --seed 7: lemmas 20000+12, rel 2000+246, stages 4000+480; 0 violations | 600 cases seed 19: (S2) real 50+350 (50 subnormal skipped), (S2) complex 150 tuned (no violation found), (S3) 50+500, (S4) 100+1075, (S5) 50+475, overflow controls 100; 0 violations | 0 | 2026-10-30 |
| thm2_polar-factor.md | Polar-factor lemma (a)-(d); Kahan inclusion for any Hermitian block; Theorem 2 radius rho = \|\|R\|\| + (\|\|A\|\|+\|\|Lambda\|\|) delta | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Proof valid, same constant. Code mapping holds at current lines (enclosure.py 476-511; certify.py 296-303; sectors.py 125-148); section 3 line numbers are those of track/backlog-proofs. Remark: sectors.audit_orthogonality has no rounding term for the computed product V_a^H V_b (input of the decomposition_shift eta) - filed under thm8_rho. | verify --cases 2000 --seed 7: 3452+860 lemma, 1726+430 Kahan, 1726+430 Thm2, 500+600 Kahan-block, 3291+809 code; 0 violations | 500 cases seed 3: 36 random + 355 adversarial (109 skipped eta>=1); code 26+225 (140 skipped Frobenius eta>=1); 129 negative controls; 0 violations | 0 | 2026-10-30 |
| thm2prime_eps-star.md | Theorem 2': distinct matching within eps* = \|\|R\|\|/sqrt(1-eta) + \|\|X^H R\|\|/(1-eta) + delta spread/sqrt(1-eta) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Proof identical in structure, valid, same constant. Code mapping holds at current lines (enclosure.py 497-511). Remark: the row 'E_norm[i] >= \|\|Rc_i - r_i\|\|' is justified under the standard model only; under INT-05 the missing sqrt(g) sqrt(n) c_w u_min is covered by norm_underflow_term(g) inside norm2_upper (ratio <= 1.4e-153 for g<=64, n<=1e9, w<=1e4) - the table should say so. No code change. | verify --cases 1500 --seed 7: 3786+1875 (413 skipped), 1262+625, 1122+549 (216 skipped); 0 violations | 400 cases seed 5: 29 random + 254 adversarial (117 skipped eta>=1); code 22+232; 80 remark checks; subnormal operator/vector cases included; 0 violations | 0 | 2026-10-30 |
| thm8_rho.md | Theorem 8: pooled sector claims match lambda_1..lambda_K(H_spec) within r + eps, eps = rho + m, rho = D + (\|\|H\|\|+\|\|B\|\|) delta; lambda_{K+1} >= E_cut - max beta - eps | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES | Same proof (Cor 5 per sector, initial segment via T3, sorted pairing, unitary polar factor + Weyl twice). Code mapping agrees (compose.py 51-178). Remark: sectors.audit_orthogonality (74-82) omits the rounding term of the computed product V_a^H V_b; it feeds only sectors.decomposition_shift, which no certified path calls (tests only). | verify --cases 60 --seed 7: 60+13 statement (7 skipped), 40+10 steps, 5 negatives; 0 violations | 150 cases seed 13: spectral core 50+240 (10 skipped); pooling exact 31+419; T3 necessity control 1; code 6+6 CERTIFIED vs exact spec eigenvalues; 18 negative controls; 0 violations | 0 | 2026-10-30 |
| E-05_variational.md | Rayleigh upper bound, existence radius, Temple lower bound, MPS contraction (Lemmas 1–5, Prop 1, V1–V3) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | GAP | Lemma 3 (v) proves the model-shift statements via Weyl/Rayleigh/Temple on the floating-point build H_ref, presupposing it is Hermitian (false in general at rounding level: boson chain with complex hopping 1.1e-16). Statements hold by the direct H_spec route with r ≥ r_stored + σ, which the code uses; the statement text "any r ≥ r_stored" is not established. No certified bound affected. | verify_E-05.py --cases 60 --seed 5: 310 cases, 0 violations | adversarial_E-05_variational.py --cases 80 --seed 91: 0 violations | 0 | 2026-10-30 |
| E-03_generalized.md | Generalized pencil: Cholesky reduction, B⁻¹-norm residual enclosure, inertia counting with collar, G1–G3 | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Forming error of M̂ stated as u\|M̂\| (is u(1+2u)\|M̂\|; covered by the up guard); ν < k paragraph describes the overlapping case (with disjoint enclosures it is an input inconsistency); indefinite-B refusal conservative; inertia_dense contract taken as hypothesis. | verify_E-03.py --cases 60 --seed 7: 303 cases, 0 violations | adversarial_E-03_generalized.py --cases 80 --seed 97: 0 violations | 0 | 2026-10-30 |
| C-07_moments.md | Moments tr(P O Hᵏ O)/g through stored vectors (Lemmas 1–4) and the zeroth/first-moment sum-rule consistency statements (Props 1–2) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES | — | verify_C-07.py --cases 40 --seed 9: 604 cases, 0 violations | adversarial_C-07_moments.py --cases 60 --seed 101: 0 violations | 0 | 2026-10-30 |
| D-01_sparse-residual.md | a-posteriori backward error of the stored sparse L D Lᴴ factors: entrywise (a), row-sum (b), β ≥ ‖E‖₂ + 2u max\|S_ii\| (c), inertia sandwich | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | (Λ-R1) first-order margin is 6u, not 5u (harmless); fl(\|z\|) via C `hypot` is a library function outside the one-rounding model (assume ≤ 3u relative error); the counted factorisation is L·Re(D)·Lᴴ (imaginary parts of complex pivots land in E); E_c is not Hermitian — the mirrored lower triangle is valid because the exact E is; (H1) Hermiticity supplied by the callers via `hermitian_part`, underflow hypothesis for L, D not scanned | verify_D-01.py --cases 60 --seed 11: 60 random + 17 adversarial cases, 4 skipped, 0 violations | independent/adversarial_D-01_sparse-residual.py --cases 40 --seed 103 (exact-rational residual, mp sandwich, corrupted factors): 0 violations; --cases 16 --seed 104: 0 violations | 0 | 2026-10-30 |
| D-02_matrix-free.md | matrix-free rounding width W = w_row + L + ρ + 1 (real) / w_row + 3L + ρ + 3 (complex): entrywise (a), Theorem-1 radius for the exact spec operator (b), spec norm bound (Lemma N) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | GAP | Lemma M3 omits the rounding of \|x_j\| for complex vectors: for a real spec applied to complex-typed claim vectors (reachable) the second-pass inequality (1 − γ_{W−1}) is 1–2 counts short, so absHX_up ≥ \|H_spec\|_abs\|x\| is not established; (b) unaffected (complex branch uses 2γ_{W+4}). The modulus model is not in section 1; measured NumPy complex \|z\| error on this host up to 1.88u (> one rounding). Lemma N evaluates the local norms on stored, not exact, amplitudes (covered by the guard). | verify_D-02.py --cases 40 --seed 13: 72 random + 18 adversarial cases, 0 skipped, 0 violations | independent/adversarial_D-02_matrix-free.py --cases 40 --seed 107 (independent mp H_spec, (a), (b), M1, M3 per family, Lemma N, exact letters, hypot accuracy): 0 violations | 0 | 2026-10-30 |
| E-02_non_hermitian.md | Lemma 1 (backward errors of right and left residuals), Theorem 2 (Bauer–Fike), Lemma 3 (inverse from a computed inverse), Theorem 4 (discs from a computed eigenvector matrix and their count), Propositions 5–7 (floating-point accounting of N1–N4) | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Own proofs by other routes (vector forms of Theorem 2 and Lemma 3; argument principle on a δ/2-neighbourhood instead of continuous root functions for the count); same constants and rounding budgets. Remarks: (1) Proposition 5 (ii) mixed the standard-model inequality with the additive budget (wording, applied); (2) (FP) did not mention fused multiply-adds in NumPy's complex products (bound 2u ≤ γ₃, applied); (3) with underflow_check=False H-mod is assumed for subnormal components (applied; the targeted Part D search found no violation); (4) the radius is uniform, per-disc radii not implemented; (5) N4 is certified when uninformative (one group), the summary says so. An instrument fault of verify_E-02.py (σ_min through MᴴM at 60 digits) was found and corrected; recorded runs are from the corrected script. | verify_E-02.py --cases 40 --seed 7: 539 random + 558 adversarial, negative controls detected, 0 violations | adversarial_E-02_non_hermitian.py --cases 120 --seed 113: 120 matrices (perturbed Jordan blocks, nearly parallel eigenvectors, graded scales, wrong and random supplied inverses), 360 centre pairs at the separation boundary, 59 opt-out matrices with subnormal complex components; negative controls 57/76 and 10/24; 0 violations | 0 | 2026-11-11 |
| fp_modulus-up.md | Lemma M (i)-(iv) (m ≥ \|z\|; m ≤ \|z\|(1 + 12u) + 2⁻¹⁰⁷³; ∞ only if \|z\|(1 + 12u) > Ω; exact for a zero component) and the corollary (no rounding count for upper moduli) | statement (section 1) and code excerpt only; own proof written; exact adversarial search; the author's proof not read (fresh-context session, 2026-11-18) | GAP | Lemma M (i)-(iv) AGREES: same statement by an own proof; the guard 1 + 4u would suffice, 12 is the least integer the count gives, the term 2⁻¹⁰⁷³ is needed. Corollary GAP: true only for finite moduli and products that do not underflow (a = x = 2⁻⁶⁰⁰ gives fl(m m) = 0; ∞ · 0 = nan); the same for its general sentence. Code against statement: object arrays took the real branch (m < \|z\| in 49 842 of 100 000 entries), integer input wraps at the most negative value, np.clongdouble wider than binary64 would be rounded (reasoning only). Resolved 2026-11-18 (UNVERIFIED, D-12): the corollary is stated for finite moduli with the term w·2⁻¹⁰⁷⁵, absent without underflow; TypeError for object, sparse and wider complex input; integers stepped up from 2⁵³; with the upper moduli in the bounds, a non-finite bound makes the records that rest on it INCONCLUSIVE (`tests/test_nonfinite_moduli.py`). | verify_modulus-up.py (defaults, third run, after the second review round): 199 953 random + 26 815 adversarial (2 000 over the subnormal band), 0 violations; (iv) 461 zero-component cases | adversarial_fp_modulus-up.py: 1 009 411 exact cases, 0 violations of (i)-(iv); lower margin 1 + 4u 0 violations in 963 014; corollary 6 784 sums without underflow, 0 violations; all controls detected (on the excerpt, and again on certify_ed.fp at 249b8ae of fix/upper-moduli) | 0 | 2026-11-18 |

## Required corrections

### thm1_fp-accounting.md

Statement-level gap, no certified bound affected. Section 1, (S2) "absHX_up_i ≥ a_i" and section 2, Step 2:
for complex data the moduli |a_ij| and |x_j| are each rounded once (hypot), so absHX_i ≥ a_i(1 − γ_{w+2}), not
a_i(1 − γ_w); (Λ5) then gives only absHX_up_i ≥ a_i(1 − γ_{w+2})(1 − u)³(1 + 4u)/(1 − γ_{w+1}(1 − u)²) ≥ a_i(1 − 8wu²),
which does not imply absHX_up_i ≥ a_i. Exact change to the proof (no code change): in (S2) write "absHX_up_i ≥ a_i
for real data and ≥ a_i(1 − 8wu²) for complex data", and in Step 3 note that the complex slack
g(1 − 3u) = 2γ_{w+6}(1 − 3u) ≥ γ_{w+4}(1 + 8wu²) absorbs it (true for (w + 6)u ≤ 1/4). Alternative (code, looser
bound, allowed immediately): in `enclosure._rounding_bound` use `gw = gamma(adapter.width + 3)` when `cplx`,
which restores absHX_up_i ≥ a_i(1 + u) for complex data by the (Λ5) argument with w + 2 in place of w. IMPLEMENTED on dev/3.2 (2026-11-03, lead integration: `gamma(width + 3)` for complex data, test `test_rounding_bound_uses_width_plus_three_for_complex_data`; a note in the document's section 3); the verification script's stage check
(S2) should then include complex cases with tuned moduli (see `independent/adversarial_thm1_fp-accounting.py`,
family `complex_S2_targeted`).

### int05_underflow.md

ERROR affecting a certified bound (the S2 Theorem 1 radius). Theorem U5 states rel ≥ ‖Ax − μx‖/‖x‖ "for every
finite stored A, x ≠ 0, μ (no magnitude hypothesis)". Counterexample (exact): A = diag(1, 2), μ = 1 + 10⁻⁸,
x = (10¹⁶⁰, 0). `np.linalg.norm(x)` overflows to inf, `norm2_lower(x)` returned inf, `upper` ≈ 10¹⁴⁴ is finite,
rel = upper/inf·(1 + 4u) + 2·ULP_MIN = 2⁻¹⁰⁷³ ≈ 10⁻³²³, while the true radius is 10⁻⁸ and no eigenvalue lies within
2⁻¹⁰⁷³ of μ; `enclose` marked the pair distinct. The INT-01 scan does not detect large magnitudes.

Code change (looser bound, implemented on this branch, commit noted in RESULT): `certify_ed/fp.py`, `norm2_lower`:
`if not math.isfinite(s): return 0.0` before the guarded formula, so that `rel` = ∞ (no enclosure) whenever the
sum of squares overflows. Negative-control tests: `tests/test_independent_check.py`
(`test_norm2_lower_is_zero_when_the_squared_norm_overflows`, `test_huge_claimed_vector_gives_no_finite_enclosure`),
which fail on the previous code (checked: 2 failed before, 2 passed after). No other bound becomes tighter.

Proof/statement change: either add the hypothesis "Σ|x_j|² < 2¹⁰²⁴ (the sum of squares does not overflow)" to
Theorem U5 and Lemma U3 (lower bound), or keep the statement and add to Lemma U3: "when fl(Σv_j²) = ∞ the code
returns norm2_lower = 0 ≤ ‖v‖". Also Lemma U4, complex data: add the paragraph on the absolute errors of the
hypot moduli |a_ij|, |x_j| (each ≤ u_min, multiplied by the other factor; absHX_i may fall short of a_i by up to
½ of the subnormal-modulus terms), covered by the factor 2 of g = 2γ_{w+6}: E_i ≥ 2γ_{w+6}(½a_i + q_i)(1 − 3u) ≥
γ_{w+4}(a_i + q_i) ≥ e_i. Not a code change.

### C16_exact_bracket.md

Code wording (certified statement string of the S8 evidence `exact_bracket`), not the proposition. `_record`
printed "exactly g eigenvalues of the declared model in [lo, hi]" with lo, hi the *outward-rounded* binary64 ends
of the counted rational bracket; an eigenvalue within one ulp outside the counted bracket lies in [lo, hi] and
makes the sentence false (exact example in `tests/test_independent_check.py`). Change implemented on this
branch (wording only; no count, bracket or bound changed): the sentence now reads "… in the exact bracket
[p/q, r/s] (about [lo, hi], width …)", and the FAILED reason likewise. Document change: in section 3, `_record`,
state that `bracket` is a display and the statement refers to `bracket_exact`.

### C-04_correlation_lift.md

Proof text only (no certified bound changes): in section 2, Lemma 1 (ii), replace the composition "(1 + γ̇_a)(1 + γ̇_b) ≤ 1 + γ̇_{a+b}, the same for γ̇" — false for complex data (γ̇_m = 2γ_{m+4}: a = b = 1 gives 20u > 12u) — by the composition of Higham's constants: each stage is ≤ γ_{m+2} (Lemma 3.5: √2γ₂ ≤ γ₃, plus γ_{m−1} for the summation), so the total is ≤ (1 + γ_{a+2})(1 + γ_{b+2}) − 1 ≤ γ_{a+b+4} ≤ 2γ_{a+b+4} = γ̇_{a+b} (docs/proofs/independent/C-04_correlation_lift.md, Lemma 1 (c)). Optionally note that Proposition 1 holds with g in place of 2g. Not implemented here (the author's text).

### E-04_parameter_intervals.md

IMPLEMENTED (looser): certify_ed/sweep.py certified_crossing_interval adds, for a sample interval ending at an interior grid point of a scan, that scan's affinity defect of the segment ending there (d^a + d^b) to the slack B·D (Lemma 3 applied to the eigenvalue branches of the segment family, whose value at the grid point differs from H(c_{k+1}) by ≤ d_k). Negative control: tests/test_independent_check.py::test_crossing_criterion_accounts_for_the_affinity_defect_at_interior_grid_points (scalar scans; pre-fix the run "<" [0, 1.045] hid a crossing at 0.99755). Document: amend section 1 (Lemma 3 remark: the implemented tube jumps at grid points; the criterion carries the defect terms), section 3 accordingly; restate Lemma 4 for the letters of the three local spaces (Hermitian or monomial) rather than for general matrices. Recommended, not implemented: symmetrise H1_s and the stored blocks in the sector branch before the ∞-norm bounds.

### E-07_davis_kahan.md

Proof text only (no certified bound changes): Proposition 1 with a model shift σ — section 3's "R changes by at most σ‖X‖₂ ≤ σ√(1+η)" is a 2-norm statement; in the Frobenius route of Lemma 1 the increment is ‖ΔX‖_F ≤ σ‖X‖_F ≤ σ√(g(1+η)). Either justify the implemented increment by the 2-norm sin Θ theorem (the Ritz spectrum lies in [a − w′, b + w′] and the rest outside its (dist − w′)-neighbourhood, so the interval hypothesis holds; ‖R′_spec‖₂ ≤ ‖R‖_F/√(1−η) + σ) — recommended, no code change — or use σ√(g(1+η)) in davis_kahan.sin_theta_davis_kahan line 57 (looser). Optional robustness: certified_fidelity squared moduli as C.real**2 + C.imag**2 (Lemma 5 (c) as stated); approximate_symmetry_label γ₀ from the window directly. Not implemented here.

### E-05_variational.md

Proof/statement text only (no certified bound changes). Section 2, Lemma 3 (v): replace the route through λ_j(H_ref) (Weyl, Rayleigh–Ritz and Temple applied to the floating-point build, which is Hermitian only up to rounding — e.g. a boson chain with complex hopping has max|H_ref − H_refᴴ| = 1.1e-16) by the direct route: θ_spec = ⟨ψ, H_spec ψ⟩/⟨ψ,ψ⟩ ∈ [θ − σ, θ + σ], r_spec ≤ r_stored + σ (the Rayleigh quotient minimises the residual over shifts), and Temple for the Hermitian H_spec with θ_hi + σ < ρ ≤ λ₂(H_spec): λ₁(H_spec) ≥ θ_lo − σ − (r_stored + σ)²/(ρ − σ − θ_hi). Section 1: state the Temple bound with r ≥ r_stored + σ (the value the code uses, variational.py line 350), not "any r ≥ r_stored". Alternative (code): symmetrise the spec build used by the variational adapter; then the H_ref route is valid as written. Not implemented here.

### D-02_matrix-free.md

Statement-level gap, no certified bound affected. Section 2, Lemma M3 counts the second pass fl(|H|_abs|x|) without
the rounding of |x_j| for a complex vector x ("the moduli of real amplitudes are exact; of complex ones one rounding
each (hypot) … then the product with |x_j|"). For a real spec (W = w_row + L + ρ + 1) applied to complex-typed
vectors — reachable, since `certify.py:893–898` uses `claim.eigenvectors` in the producer's dtype and
`enclosure._rounding_bound` takes the complex branch for them — the second pass carries w_i + L + ρ + h roundings
(h = roundings of one complex modulus) against the budget W − 1 = w_row + L + ρ, so fl(|H|_abs|x|)_i ≥
(|H_spec|_abs|x|)_i (1 − γ_{W−1}) and `absHX_up_i` ≥ (|H_spec|_abs|x|)_i are not established (h = 1: one count short;
h = 2: two). For a complex spec the 2L spare counts of W cover h ≤ L + 1. Statement (b) is unaffected: the complex
branch uses g = 2γ_{W+4} ≥ 2γ_W, and the shortfall of a few u in `absHX_up` is absorbed by the factor 2 exactly as in
Step 3 of `thm1_fp-accounting.md` (same mechanism as the (S2) gap recorded there).

Second, the modulus model: section 1's error model does not mention `np.abs` of a complex number, which Lemma M3 (and
Lemma N's |c_t|) rely on as "one rounding". On this host (Windows, NumPy's complex `absolute` → C `hypot`) the
measured relative error over 140 000 arguments reaches 1.88u (`independent/adversarial_D-02_matrix-free.py`,
informational line), i.e. more than one rounding, within 1 ulp. The hypothesis |fl(|z|) − |z|| ≤ h u |z| with h ≥ 2
should be stated (D-01 uses h = 3) and the counts redone with it; this also concerns `thm1_fp-accounting.md` Step 2
(its complex (S2) accounting assumes h = 1; its Step 3 slack still absorbs h ≤ 3).

Exact change to the proof (no code change): in Lemma M3 write the exponent as w_row + L + ρ + h_a + h_x with h_a = 0
for a real spec, h_x = 0 for a real vector, h_a, h_x ≤ h otherwise, and conclude (1 − γ_{W−1}) only for (real spec,
real x) and (complex spec, h ≤ L + 1); for (real spec, complex x) state that `absHX_up_i` ≥ (|H_spec|_abs|x|)_i
(1 − 2hu) and that (b) still follows from the factor 2 of the complex `gamma_dot`. Alternative (code, looser bound) — IMPLEMENTED on dev/3.2 (lead commit 2e1dc59, before this verdict landed): in
`enclosure._rounding_bound` `gw = gamma(adapter.width + 3)` when `cplx` (the same change recorded for thm1),
which restores the second-pass inequality for h ≤ 2 in every family; `tests/` should then include a real spec with
complex-typed vectors through `residuals` and, if the hypothesis h ≤ 2 is adopted, a `hypot` accuracy probe.

### E-02_non_hermitian.md

Written and checked on 2026-11-11 under the same protocol (a second derivation written from the statement alone, not
a human check). No statement-level gap. Wording changes applied to section 1 before delivery: Proposition 5 (ii) now
separates the standard-model inequality from the additive budget (2w + 10)u_min of E; (FP) states that NumPy's complex
products may use fused multiply-adds (error at most 2u, Jeannerod, Kornerup, Louvet and Muller 2017, below γ₃) and that
the opt-out `underflow_check=False` assumes H-mod also for subnormal components. Instrument correction in
`verify_E-02.py`: σ_min(A − μI) is computed with mpmath's SVD instead of the eigenvalues of (A − μI)ᴴ(A − μI), which at 60
digits does not resolve σ_min below about 10⁻³⁰‖A − μI‖ (a false report on a matrix with subnormal complex entries; the
exact checks held). The code (`certify_ed/nonhermitian.py`) carries the UNVERIFIED marker until Sarang's check.

### Upper moduli (2026-11-18; UNVERIFIED, D-12)

New derivation, not a human check: `fp_modulus-up.md` adds `fp.modulus_up` and `fp.abs_up` (Lemma M: m ≥ |z| from
correctly rounded ×, +, √ and exact scaling by powers of two; m ≤ |z|(1 + 12u) + 2⁻¹⁰⁷³); no bound uses them in that
change. The D-02 verdict above asks for a stated modulus hypothesis |fl(|z|) − |z|| ≤ hu|z| with h ≥ 2 and the counts
redone with it. Measured in exact arithmetic on the Windows reference host (NumPy 2.4.4, AVX512F), NumPy's complex
absolute value errs by −2.466u for z = 0.02019294356747192 − 0.5012539518584045i, −2.462u for z = 1.9986893956853402 +
0.07375207458660175i and +2.269u for z = 1.010573342221019 + 0.3330018761835207i; for z = 2⁻¹⁰⁷⁴(1 + i) it returns a
value below |z|. The 1.88u above came from 140 000 random arguments, and "within 1 ulp" is not established, so no
small h is adopted; the upper moduli replace the hypothesis. Machine-checked by `verify_modulus-up.py` (199 953
random + 26 815 adversarial cases, 0 violations; negative controls detected). Independent re-derivation: done
2026-11-18 (row fp_modulus-up.md above; verdict AGREES on Lemma M, GAP on the corollary, resolved as said there);
Sarang's check (D-12): pending.

Review rounds of 2026-11-18 (an EGM review of `fp_modulus-up.md`, code reviews of the branch, the independent
re-derivation, and a second round by three reviewers; UNVERIFIED, D-12). Resolved in the code: `modulus_up` rejects
sparse, object and wider complex input (TypeError), steps integers up from 2⁵³, returns a float for every scalar,
takes 0 + 0i on the unscaled branch, and works in passes of at most 2¹⁸ entries without copying its input
(temporaries 6.8 MiB measured, bounded by about 12 MiB, for every memory layout); `abs_up` uses only public scipy
calls (COO from `coords` where scipy has it, which covers a 1-D `coo_array`, from `row` and `col` before
scipy 1.13), steps integer and boolean sparse data up, and keeps its in-place
summation of duplicate entries; a test covers the subnormal band of results. Open items:
- Duplicate entries: direct calls of `davis_kahan.commutator_bound`, `sectors.audit_sector`,
  `correlations.compress_operator` and `eigenspace.certify_observable` form a product before the moduli sum the
  duplicates in place, and are not valid for a matrix with duplicate entries (as before; example in
  `fp_modulus-up.md`, What is not claimed). An `abs_up` that bounds the stored entries without summing is not written.
- Single precision: Lemma M and the corollary do not cover products formed in float32 or complex64; a false S2
  certificate for a float32 operator with float32 vectors was reproduced (as before). Refusing or widening such input
  at the entry points is not written here (not covered; the separate change `fix/single-precision`).
- The FP-M items that `fp.probe_float_environment` does not probe (correctly rounded square root, frexp, ldexp,
  nextafter) are tested with the test suite only, and the certificate does not name them; a run-time probe and an
  assumption line are not written.
- For P15's D-12 check: `P15_summed_bounds.md` prints (1 − u)³(1 + 8u) = 1 + 5u − 21u² + 13u³ − 8u⁴, where the
  coefficient is 23u³, and reports a worst main-branch overestimate of 2.2·10⁻¹⁵ relative, above the 12u ≈
  1.33·10⁻¹⁵ of Lemma M (ii) for the same computation (probably m²/|z|² − 1). P15 is not changed here.

### Non-finite bounds (2026-11-19; UNVERIFIED, D-12)

A bound of the floating-point accounting can be ∞ or nan: a product, a sum or a modulus overflows for operator entries
near the largest float (∞, and nan where an ∞ meets a zero or another ∞), for real and complex data. Resolved in the
code, each with a test in `tests/test_nonfinite_bounds.py` (real entries of 1.7·10³⁰⁸ and one step below the largest
float, the complex entry 1.5·10³⁰⁸(1 + i), whose modulus exceeds the largest float; controls at 10¹⁰⁰ certified): S2
and S3 of `certify()` are INCONCLUSIVE with the reason code `NONFINITE_BOUND` and a summary that says the radius is not
finite (before, the S2 summary of such a pair said that an eigenvalue lies within its radius), the residual bounds and
radii are stored as ∞, not nan, and S2 makes no accuracy statement, with the same code, when the sector lift shift is
not finite; a nonzero claimed vector whose norm binary64 does not resolve (its squared norm underflows or overflows)
has the reason code `UNRESOLVED_VECTOR_NORM` and the advice to normalise the vectors instead; `inertia_dense` and
`inertia_sparse` give no count on a non-finite backward error (the sparse guard is reached through a stand-in residual
bound only); a sector audit with a non-finite defect does not pass (inf ≤ tol · inf held before), so S1 on a sector
isometry is INCONCLUSIVE and the sector composition is INCONCLUSIVE with a summary that names the audit bounds; the
variational records (V1 to V3; V1 and V2 were CERTIFIED with a nan bound) and the moment records (M1 to M3, and a
sum-rule verdict on a nan endpoint) are INCONCLUSIVE with `NONFINITE_BOUND`; a non-finite observable bound is reported
as ∞. The label record of a cyclic symmetry carries its separation as a float, a note that it is not a certified lower
bound and `separation_rigorous: false`; its `certified` flag is unchanged (open, D-12). Not covered by a test: the
correlation records, the sweeps, the MPS contraction, the Ozaki prototype and the GPU path. Open, as at the base:
`certify(kind="lowest")` raises ValueError in `scipy.linalg.ldl` when the shifted matrix of S3 is not finite (couplings
of 10¹⁶⁰ without a sector isometry).

### Complex moduli (2026-11-18; UNVERIFIED, D-12)

The moduli of complex data in rigorous bounds now come from `fp.modulus_up` and `fp.abs_up` (section "Upper moduli"
above), and the analyses count no rounding for a modulus. New, not an independent re-derivation and not a human check.
What this changes in the documents (a correction note in each, marked UNVERIFIED, with a line after its status line
pointing to it): `thm1_fp-accounting.md` (S2) holds for complex data with the kept constant γ_{w+3} by (Λ5′), the
monotone form of (Λ5) with the division index k ≥ w + 1, checked at k = w + 3; `D-02_matrix-free.md` Lemma M3 holds
as first stated (h_a = h_x = 0), and Lemma N counts nothing for |c_t|; `D-01_sparse-residual.md` no longer needs
"≤ 3u" for the moduli (the factors (1 − u)³ are spare); `E-02_non_hermitian.md` no longer needs H-mod;
`appA_model-shift.md` no longer needs the extra index of its erratum of 10 Nov 2026; `E-05_variational.md` Lemmas 3
and 4, `C-07_moments.md`, `E-03_generalized.md`, `C-04_correlation_lift.md`, `E-04_parameter_intervals.md`,
`E-07_davis_kahan.md` (Lemma 5 (c) now formed as stated, which settles the remark in its row), `int05_underflow.md`
(the Lemma U4 remark on `hypot` moduli of subnormal components) and the draft `D-03_ozaki_draft.md` hold as written
with upper moduli. No code constant changes; the complex γ_{w+3} of `_rounding_bound` and the constants of D-01, D-02,
D-03 and E-02 keep spare counts. For complex128 data the bound values change in the last bits (each modulus lies at
or above the exact one and, for normal values, at most a relative 12u above it); for complex64 data the moduli were
float32 values before, so those bounds can move either way by more; an entry within a relative 12u of the largest
float has an infinite modulus now (S2 INCONCLUSIVE instead of CERTIFIED). The records that rest on such a modulus
state nothing, by the rules of the section above (`tests/test_nonfinite_moduli.py`), as the independent re-derivation
asks for a nan bound (suggested edit 1 of its row). Real data and complex data whose entries have a zero component are
unchanged, except for the arithmetic text of the non-Hermitian record, which names `fp.modulus_up` instead of the
hypothesis H-mod.

Review rounds of 2026-11-18 (UNVERIFIED, D-12). Resolved: the audit pin of `tests/test_abs_audit.py` covers every
spelling of a modulus, called or referenced, whatever the object it is taken from, the 1- and ∞-type norm orders and
hand-written moduli, and fails on stale lines of its list; a dynamic pin runs the main entry points on complex data
with numpy's modulus recorded outside the classified places (nothing recorded; with the source pin left out it
records 66 of the 74 routed moduli reverted one at a time, all but the 8 of the device path in `gpu.py`);
`fp.inf_norm_upper` has an exact and bit-for-bit test; the S1 invariants take each modulus once (S1 time back at the
base's: N = 20, T = 120: 7.3 s against 8.9 s at the base, 80.5 s before); the moduli of an S2 block are formed once.
Open items:
- `sectors.audit_label` divides by the separation of the computed roots of unity, a floating-point value, and its
  record's `certified` flag rests on it (the record says so: `separation_note`, `separation_rigorous: false`). A rigorous version needs a lower bound on 2 sin(π/n) and a bound on the
  distance of the computed roots from the exact ones: a D-12 item, not started.
- The GPU products were not run after the change (no CuPy on the reference host); the moduli are those of the CPU path
  by construction.
