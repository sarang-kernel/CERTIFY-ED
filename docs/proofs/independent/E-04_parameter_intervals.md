# Independent re-derivation: E-04 (certified parameter intervals / eigenvalue tubes)

Not a human check. Written from sections 1 and 3 of `docs/proofs/E-04_parameter_intervals.md` only (section 2 read
afterwards; the comparison is at the end). Notation as there: grid c₀ < … < c_m, F_k(c) = H(c_k) + (c − c_k)H1 on [c_k,
c_{k+1}], d_k ≥ ‖H(c_{k+1}) − F_k(c_{k+1})‖₂, B ≥ ‖H1‖₂, λ_j(·) the j-th smallest eigenvalue. Classical facts: Weyl's
inequality |λ_j(A + E) − λ_j(A)| ≤ ‖E‖₂ for Hermitian A, E (Horn & Johnson, Cor. 4.3.15); the singular values of a
Kronecker product are the products of the singular values; ρ(A) ≤ ‖A‖ for every induced norm.

## Proofs

**Proposition 1.** F_k(c) − H(c_k) = (c − c_k)H1 is Hermitian with ‖·‖₂ ≤ |c − c_k|‖H1‖₂ ≤ B|c − c_k|; Weyl gives the
first inequality. F_k(c) − H(c_{k+1}) = (c − c_{k+1})H1 + (F_k(c_{k+1}) − H(c_{k+1})) has norm ≤ B|c − c_{k+1}| + d_k;
Weyl gives the second. If λ_j(H(c_k)) ∈ [l_j, u_j] and λ_j(H(c_{k+1})) ∈ [l′_j, u′_j], both inequalities hold for the
same index j, so λ_j(F_k(c)) lies in each of the two widened intervals, hence in their intersection. Without S3
the enclosure contains λ_i(H(c_k)) for some i (Theorem 1); the first inequality with that i gives the tube
containment for λ_i(F_k(c)); the right-hand tube can only be intersected with it when it refers to the same i,
which is what rank certification at both points supplies (the code intersects whenever both points have an
interval; with uncertified ranks the intersection is a statement about "an eigenvalue" only if the two
enclosures refer to the same index — see the remark on `rank_certified` below). ∎

**Proposition 2.** Let A, C be Hermitian with ‖A − C‖₂ ≤ ε, so λ_j(C) ≤ λ_j(A) + ε and λ_j(A) ≤ λ_j(C) + ε for all j
(Weyl). If λ_j(A) < σ − ε then λ_j(C) < σ: #{λ(A) < σ − ε} ≤ #{λ(C) < σ}. If λ_j(C) < σ then λ_j(A) < σ + ε:
#{λ(C) < σ} ≤ #{λ(A) ≤ σ + ε}. Apply with C = H(c_k), A = F_k(c), ε = B|c − c_k|. The gap line: λ_{k+1}(H(c_k)) ≥ L
and Weyl give λ_{k+1}(F_k(c)) ≥ L − B|c − c_k| (and via the right point ≥ L′ − B|c − c_{k+1}| − d_k). ∎

**Lemma 3.** The ends of a tube on a fixed segment are max/min of finitely many functions const ± B|c − c_k|, each
B-Lipschitz, and max/min preserve the Lipschitz constant; so α⁺, β⁻ are B-Lipschitz on the segment and m = α⁺ − β⁻ is
2B-Lipschitz there. For c ∈ [c_i, c_{i+1}] the nearer sample point c* is at distance ≤ D/2, so
m(c) ≤ m(c*) + 2B·D/2 ≤ max(m(c_i), m(c_{i+1})) + BD < 0. ∎
*The hypothesis matters.* The tube function implemented by `tube_at` is B-Lipschitz on each open segment but
jumps at interior grid points: at c = c_{k+1} it evaluates the tube of segment k + 1 (whose value there is the
certificate interval of H(c_{k+1}), intersected with the widened next one), not the limit of the segment-k tube
(which refers to F_k(c_{k+1}) and carries d_k). For the eigenvalue branches themselves: g(c) := λ_a(F^a_k(c)) −
λ_b(F^b_{k′}(c)) is (B_a + B_b) ≤ 2B-Lipschitz on the closed segment, g(c_i) ≤ m(c_i) (the sample c_i lies in the
segment, including its left grid point, where `_segment` picks this segment), and at a right end that is an
interior grid point of scan a, λ_a(F^a_k(c_{k+1})) ≤ λ_a(H^a(c_{k+1})) + d^a_k ≤ α⁺(c_{k+1}) + d^a_k (Weyl with
‖F_k(c_{k+1}) − H(c_{k+1})‖₂ ≤ d_k; the right-segment tube at its left grid point contains λ_a(H^a(c_{k+1})) by
Proposition 1 with c = c_{k+1}); likewise λ_b ≥ β⁻ − d^b. Hence the sufficient criterion for "a < b on the whole
sample interval" is

    max(m(c_i), m(c_{i+1})) + B·D + d^a + d^b < 0,

with d^a (d^b) the affinity defect of the segment of scan a (b) ending at c_{i+1} when c_{i+1} is an interior grid
point of that scan, and 0 otherwise. Without the defect terms the criterion is wrong whenever d > 0: counterexample
with scalar families (B = 1, D = 1/200): scan a with F₀(c) = −1.9951 + c on [0, 1] and H_a(1) = −1.0951 (d₀ = 0.1),
scan b with H_b(c) = −c (d = 0). The segment-0 branches cross at c* = 0.99755 ∈ (0.995, 1), yet m(0.995) = −0.0051
and the grid-point margin from the right tube is −0.0951, so max + BD = −0.0001 < 0 and the relation "<" was
certified on [0.995, 1] (the current code certified "<" on [0, 1.045]). The defect terms are only zero for exactly
affine families (the common case), so the error is of size ≤ d^a + d^b, typically 0 or ~10⁻¹⁶·‖H‖ — but a
certified relation is a certified relation. **Fixed (looser):** `certified_crossing_interval` now adds d^a + d^b to
the slack of a sample interval ending at an interior grid point; negative-control test
`test_crossing_criterion_accounts_for_the_affinity_defect_at_interior_grid_points` in
`tests/test_independent_check.py` (fails on the previous code: the run '<' [0, 1.045]; passes now: '<' [0, 0.995],
not separated [0.995, 1]).
*Remark on crossings across a grid point.* "A crossing lies in every not-separated run flanked by opposite
relations" uses continuity of the branches; the piecewise family F_k is continuous within segments and jumps by at
most d_k in norm at grid points, so a run that contains a grid point with d_k > 0 may hold a jump of the model
rather than a crossing of the continuous family H(c) (about which E-04 makes no statement between grid points).

**Lemma 4.** For o₁ ⊗ … ⊗ o_m (on distinct sites; the embedding and any Jordan–Wigner sign string are unitary
factors or diagonal ±1 matrices, which do not increase ‖·‖₂), ‖O‖₂ = Π‖o_i‖₂. The second inequality
‖o‖₂ ≤ ‖|o|‖_∞ is **not** true for arbitrary matrices (o = [[1, 0], [1, 0]]: ‖o‖₂ = √2 > 1 = ‖o‖_∞); it holds for
every letter of the three local spaces because each letter matrix is either Hermitian (I, x, y, z; fermion n, z;
boson n: ‖o‖₂ = ρ(o) ≤ ‖o‖_∞) or monomial — at most one nonzero per row and per column (spin ±, fermion ±, boson
±): then oᴴo is diagonal with the squared entries, so ‖o‖₂ = max|o_ij| = ‖|o|‖_∞. Checked for all letters of
spin S = 1/2 … 7/2 (three conventions), fermions (both JW strings) and bosons n_max ≤ 10. The code's factor (1 + 4u)
covers the rounding of the float letter matrices (the √ amplitudes) relative to the exact letters. ∎

**Bounds used for B.** (i) ‖H1‖₂ ≤ ‖H1‖_∞ for the exact (Hermitian) H1, and ‖H1‖_∞ ≤ ‖H1_built‖_∞ + ‖H1 − H1_built‖_∞;
the builder-shift analysis gives |H1_built − H1| ≤ γ_W |H1|_abs entrywise, and for a Hermitian spec the multiset of
terms is closed under Hermitian conjugation (the letter strings are linearly independent, and the adjoint of a
string is a string), so |H1|_abs is symmetric and ‖γ_W|H1|_abs‖_∞ = γ_W √(‖|H1|_abs‖₁‖|H1|_abs‖_∞) ≤ the builder shift.
So row-sum + shift is an upper bound even when H1_built is not exactly Hermitian (which `h1_norm_bound` does not
check; the alternative route ‖H1_built‖₂ ≤ ‖H1_built‖_∞ would need Hermiticity of the built matrix). (ii) The
triangle bound Σ|b_t|‖O_t‖₂ with Lemma 4 is immediate. The minimum of two upper bounds is an upper bound. ∎

## Code (current `certify_ed/sweep.py`)

* `term_norm_bound` 75–83 (product of `inf_norm_upper(|o|)`, (1 + 4u), `up`); `h1_norm_bound` 86–98 (`add_up`/`up`
  in the triangle sum; row-sum + builder shift with `add_up`; min). `affinity_defect` 112–133: e_t exact in
  Fractions, |e_t| with two roundings covered by (1 + 4u) and `up`, Σ with `add_up`; `matrix_affinity_defect`
  101–109 for the stored block family (γ₃ entrywise rounding of the three operations, (1 + 4u), `up`).
* `certified_sweep` 172–289: B at 217 (sector: row-sum of the stored H1_s) / 219 (row-sum of the built H1 + builder
  shift, or the triangle bound); defects at 221 / 266; per point the S2 enclosures with the model shift
  (`_intervals`, 150–169, block radii for sector claims), `lower`/`upper` outward (255–256); `rank_certified`
  requires S2 CERTIFIED, S3 CERTIFIED and kind lowest/all (249); `next_eigenvalue_lower_bound` only when ranked.
* Proposition 1: `tube_at` 308–336: `widen = up(B·dist)` (322), `add_up(widen, d)` for the right point (324),
  `sub_down`/`add_up` of the outward ends (325–326), intersection by max/min, ValueError on an empty intersection.
  Proposition 2: `next_level_lower_bound_at` 339–352, the same widening, `sub_down`.
* Lemma 3: `certified_crossing_interval` 357–430 (after the fix): sample grid = union of both scans' grid points
  with `samples_per_segment` subdivisions (so no sample interval straddles a grid point of either scan), margins
  rounded up (`nextafter`), D rounded up, `slack = up(B·D)` plus — new — the end defects; the sign test
  `max(...) + slack < 0` is safe under round-to-nearest (the sum of two binary64 numbers rounds to a value of the
  same sign). B = max of the two scans' B (the margin is (B_a + B_b)-Lipschitz ≤ 2B).
* *Sector branch remark.* For `sector_fn` the family is "the stored block family" with H1_s = V₀ᴴH1V₀ and
  H_s(c_k) = V₀ᴴHV₀ in floating point, which are Hermitian only up to rounding, while the certificate at c_k refers
  to the Hermitian part of the stored block (certify.py line 413). Weyl needs a Hermitian family, so F_k should be
  read with the Hermitian parts; then B = `inf_norm_upper(H1_s)` ≥ ‖H1_s‖_∞ bounds ‖(H1_s)_h‖₂ only up to the
  asymmetry of H1_s (‖(H1_s)_h‖₂ ≤ ‖H1_s‖₂ ≤ √(‖H1_s‖₁‖H1_s‖_∞)), a relative deficit of the order of the rounding of
  the product V₀ᴴH1V₀ (~n·u); the same for `matrix_affinity_defect`. Symmetrising H1_s and the blocks before use
  would remove the remark (not implemented: a definitional change of the recorded family).

### Comparison with section 2 (read after the above)

Propositions 1 and 2, the affinity defects and the triangle bound agree with my derivation. Lemma 3's proof
(minimum of two affine functions of slopes ±2B) is correct and equivalent to my nearer-sample-point argument.

**ERROR (code mapping of Lemma 3 at interior grid points; fixed).** Section 1 asserts that every tube end is
B-Lipschitz and section 3 that `certified_crossing_interval` applies the criterion to them. The tube function
actually evaluated jumps at every interior grid point of a scan (`_segment` switches to the next segment, whose
tube at the grid point refers to H(c_{k+1}), not to F_k(c_{k+1})), so on the last sample interval before such a
point Lemma 3's hypothesis fails and its conclusion can be false by up to the segment's affinity defect: the
scalar counterexample above was certified "<" on [0, 1.045] although the segment-0 branches cross at 0.99755. The
correct criterion adds d^a + d^b (defects of the segments ending at the sample interval's right end, one per scan
with a grid point there), which is 0 for exactly affine families — so no run of the shipped tests or of
`verify_E-04.py` could see it (its part C checks Lemma 3 on tubes that are Lipschitz by construction). Implemented
(looser) in `sweep.certified_crossing_interval`, with the negative-control test
`test_crossing_criterion_accounts_for_the_affinity_defect_at_interior_grid_points` (fails on the previous code,
passes now); `tests/test_sweep.py` (7) still passes (the J1–J2 crossing interval is unchanged: that family is
exactly affine).

**Remarks.** (1) Section 2's bound (i) applies "‖A‖₂ ≤ ‖A‖_∞ for Hermitian A" to the *built* H1, whose exact
Hermiticity is not checked; the bound is nevertheless valid through the entrywise builder-shift estimate and the
symmetry of |H1|_abs for a Hermitian spec (my proof of Bound (i)). (2) Section 1's Lemma 4 states ‖o‖₂ ≤ ‖o‖_∞ for
general letters, which is false for non-normal matrices (o = [[1, 0], [1, 0]]); section 2 says the code bounds
both ‖o‖₁ and ‖o‖_∞ — it computes only `inf_norm_upper(|o|)` = ‖o‖_∞. The bound is right for every letter of the
three local spaces (Hermitian or monomial, checked exhaustively); the lemma should be stated for those letters.
(3) The "continuity and crossings" paragraph holds within a segment; across a grid point with d_k > 0 the piecewise
family jumps by ≤ d_k in norm, so a not-separated run containing such a grid point may hold a jump of the model
rather than a crossing. (4) Sector branch: H1_s and the stored blocks are Hermitian only up to rounding; B and the
matrix defect use ∞-norms of these (deficit of the order of the rounding of V₀ᴴH1V₀ relative to the Hermitian
parts the certificates refer to); symmetrising them would close it. (5) Intersecting the two grid points' tubes
when `rank_certified` is False (Theorem 1 indices possibly different at the two points) is not justified by
Proposition 1; the record flags it but the caveat text claims the tube "contains an eigenvalue of unidentified
rank". (6) Section 3 line numbers are those of `track/new-statements`.

**Verdict: ERROR** (crossing criterion applied outside Lemma 3's hypothesis at interior grid points; certified
relation wrong by ≤ d^a + d^b when a defect is non-zero; fixed, looser, negative control added). Tubes
(Propositions 1, 2) and B are unaffected.

### Runs

* `python docs/proofs/verify_E-04.py 7` (positional seed; fixed sizes): A 87, B 261, C 400 (criterion held in 327,
  each checked at 1001 points), D 108, E 300 tube evaluations, F negative controls detected; TOTAL 1156 cases,
  0 violations (67 s). Part C cannot see the grid-point defect (its tubes are Lipschitz by construction).
* `python docs/proofs/independent/adversarial_E-04_parameter_intervals.py --cases 120 --seed 71` (after the fix):
  Proposition 1 960 tube checks with d_k ∈ {0, 1e-12, 1e-3, 0.5} and c at/next to both grid points, Proposition 2
  2880 threshold counts, Lemma 3 / `certified_crossing_interval` 120 synthetic scans (60 random piecewise affine
  with defects and two grids, 60 targeted at the last sample interval before a defect — on the pre-fix code all 60
  targeted cases violate; the shipped run reports the fixed code), Lemma 4 145 (all letters, strings ≤ 81 dims),
  end to end TFIM N = 3 (grid 0.3/0.8/1.3, defects > 0) and N = 4 80 tube/gap-line checks, negative control B/20
  detected; 0 violations (the run time is dominated by mp eigensolves, ~15 min).
