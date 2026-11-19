# Independent re-derivation: D-02 (matrix-free residual bounds from the spec)

Independent re-derivation (rules D-12/D-59; not a human check). Written from sections 1 and 3 of `docs/proofs/D-02_matrix-free.md`
and the code (`certify_ed/matrixfree.py`, `construct._apply_string`, `construct.SpecOperator._apply`,
`localops.LocalSpace.shift_components/matrix`, `enclosure.OperatorAdapter.__init__` 89–131, `enclosure._rounding_bound`
295–311) before section 2 was read. Theorem 1-fl (`thm1_fp-accounting.md`, re-derived in the first track) is used as a
black box: it needs, for the operator at hand, (i) an entrywise bound |fl(Hx)_i − (H x)_i| ≤ γ_W (|H|_abs|x|)_i with
`gamma_dot(width + 2)` ≥ γ_W, and (ii) a computed second pass fl(|H|_abs|x|)_i ≥ (|H|_abs|x|)_i (1 − γ_{width+1}).

## Setting

Spec with N sites, local dimension q, couplings (c, sites) grouped in terms t with strings o_{t,1} ⋯ o_{t,L_t};
T = `spec.n_terms` = total number of couplings; L = max_t L_t. For a letter o, `shift_components(o)` returns pairs
(δ, a) with a[k] = ⟨k + δ|o|k⟩ as stored binary64 complex numbers; the exact components have â[k] with the same
sparsity pattern. A monomial of a coupling is a choice of one component per letter. `_apply_string` (construct.py
144–171) processes one monomial per pass: starting from amp = c (exact), for the letters from right to left it reads
the source digit, multiplies amp by the stored amplitude a[digit] (and by the Jordan–Wigner sign ±1 where the letter
is a fermion ladder), and shifts the digit; states for which some digit leaves [0, q) or some amplitude is zero are
masked out (`ok`). `_apply` (matrixfree.py 157–190) then forms contrib = a · x[src] (or |a| · |x[src]| for the
absolute pass) and does `y[rr] += contrib`.

**Hypotheses.** (S) standard model fl(a ∘ b) = (a ∘ b)(1 + δ), |δ| ≤ u, for +, −, ×, ÷, √ with normal results (no
underflow; this is the same hypothesis as for the stored-matrix path). (X) x finite. (C) every coupling c is taken as
stored (exact by definition of H_spec). (A) the stored amplitude of an *exact* letter equals the exact amplitude; the
stored amplitude of an *inexact* letter is the correctly rounded exact value: a[k] = â[k](1 + ε), |ε| ≤ u. (M) the
complex modulus |z| = `np.abs(z)` satisfies |fl(|z|) − |z|| ≤ h u |z| for a small integer h (library `hypot`; h = 3
if it is evaluated as a rounded square root of a rounded sum of two rounded squares, h = 2 for a 1-ulp `hypot`, h = 1
if correctly rounded); the modulus of a real number or of a purely imaginary number is exact (`hypot(x, ±0)` = |x|).

(A) is what `_exact_components` decides: it compares, in rational arithmetic, the square of the stored modulus with
the exact S(S + 1) − m(m ± 1) (times the convention scale) or with the boson level k, so "exact letter" means
*every* stored nonzero amplitude equals its exact value; for an inexact letter each stored amplitude comes from one
correctly rounded `np.sqrt` of an exactly representable argument (S(S + 1) − m(m ± 1) with S ≤ 7/2 and |m| ≤ S, or
k + 1 ≤ q) times exact powers of two, hence one rounding (`shift_components`, localops.py 165–200). The phases are
exact (real for z, ±, x, bosons, fermions; ±i/2 times a real for y).

## Lemma A (one pass, one rounded addition per contribution)

A monomial shifts every digit by a fixed amount (shifts on a repeated site add), so it is a translation of the
digit vector on the states it does not mask out: injective. Hence within one pass the targets `rr` are distinct and
`y[rr] += contrib` performs exactly one rounded addition per target; over all passes the output entry y_i is the
recursive floating-point sum of its w_i contributions in pass order, the first addition (0 + contrib) being exact:
w_i − 1 roundings. For the restricted state set the same holds (`index_of` is injective; targets outside the set are
dropped, which is the restriction of H to the set).

## Lemma B (per-row count): w_i ≤ w_row := Σ_t Π_{l ≤ L_t} c(o_{t,l}) · #couplings(t)

Fix a coupling and an output state i with digits k_1 … k_N. A monomial contributes to i only if its target is i.
Read the string from the left (the last letter applied): its component (δ, a) must have output digit k_site and a
nonzero amplitude at the source digit k_site − δ; by definition of c(o) (`letter_counts`: the maximum over target
digits k of the number of components with 0 ≤ k − δ < q and a[k − δ] ≠ 0) there are at most c(o_{t,1}) such
components. Given it, the digit before this letter acted is determined (k_site − δ), and the next letter is in the
same situation — also when it acts on the same site (then the required output digit is k_site − δ instead of k_site).
So at most Π_l c(o_{t,l}) monomials of the coupling reach i, each from at most one source (Lemma A): at most
Π_l c(o_{t,l}) contributions per coupling, and w_i ≤ w_row after summing over couplings. (The code's `ok & (amp != 0)`
skips exactly the masked monomials and nothing else: a product of nonzero binary64 numbers is nonzero under (S).)

For q = 2 every letter has c(o) = 1 (each shift component has one valid source digit for a given target digit; for x
and y the two components have disjoint target digits), so w_row = T, and for a Pauli/fermion string every coupling
reaches every state exactly once with a nonzero amplitude: w_i = T — the bound is sharp (Lemma M2 of the statement).

## Lemma C (one contribution)

Let â := c Π_l â_l[k_l] · (JW sign) be the exact amplitude of a monomial on a given source state. The code's amp
after the L_t multiplications is a_c = c Π_l a_l[k_l] · (sign) with every multiplication `amp * factor` a complex
times a real or purely imaginary number (every stored component is real or purely imaginary, see (A); the JW sign
is an exact ±1 factor): NumPy's complex product (ac − bd, ad + bc) then has one exactly zero summand per component
and each component is rounded once, so |fl(z f) − z f| ≤ u |z f| (two independent relative roundings of the two
components give a relative error ≤ u in modulus). With (A), |a_c − â| ≤ ((1 + u)^{L_t + ρ_t} − 1)|â| where
ρ_t := #{l : o_{t,l} inexact} ≤ ρ: a_c = â (1 + θ_{L_t+ρ_t}).

contrib = fl(a_c · x_j): real data (a_c real, x real; `.real` of the complex product (a_c x_j, 0) is exact): one
rounding, so contrib = â x_j (1 + θ_{L_t+ρ_t+1}). Complex data: a full complex product, |fl(zw) − zw| ≤ √2 γ₂ |z||w|
(Higham Lemma 3.5) ≤ γ₃ |z||w|, so contrib = â x_j (1 + θ_{L_t+ρ_t+3}).

## Theorem D-02 (a)

By Lemma A, fl(Hx)_i = Σ_k contrib_k (1 + θ^{(k)}_{w_i−1}) (recursive summation, Higham eq. (3.4) with w_i − 1
roundings; for complex data each addition rounds both components, which is again a relative error ≤ u in modulus).
With Lemma C every term is â_k x_{j_k} (1 + θ_{w_i−1+L_t+ρ_t+1}) (real) or (1 + θ_{w_i−1+L_t+ρ_t+3}) (complex), hence

    |fl(Hx)_i − (H_spec x)_i| ≤ γ_{w_i+L+ρ} Σ_k |â_k| |x_{j_k}| = γ_{w_i+L+ρ} (|H_spec|_abs |x|)_i        (real),
    |fl(Hx)_i − (H_spec x)_i| ≤ γ_{w_i+L+ρ+2} (|H_spec|_abs |x|)_i                                       (complex),

where (|H_spec|_abs|x|)_i is, by definition, the sum of the moduli of the exact contributions (Appendix A). Since
w_i ≤ w_row (Lemma B) and γ is increasing: real γ_{w_row+L+ρ} ≤ γ_{W−1} ≤ γ_W with W = w_row + L + ρ + 1; complex
γ_{w_row+L+ρ+2} ≤ γ_{W} with W = w_row + 3L + ρ + 3 (margin 2L + 1). This is (a). The adapter sets width = W − 2
(`rounding_width`, matrixfree.py 126; `min` with the 3.1 count at enclosure.py 116 only lowers it when the 3.1 count
is smaller, which is itself a proven width) and `_rounding_bound` uses `gamma_dot(width + 2, cplx)` = γ_W (real) or
2γ_{W+4} ≥ γ_W (complex): requirement (i) of Theorem 1-fl holds with one count to spare for real data. ∎

## Lemma M3 (the second pass fl(|H|_abs |x|))

`_apply(x, absolute=True)`: contrib = fl(fl(|a_c|) · fl(|x_j|)), summed as in Lemma A. Shortfall factors: a_c carries
(1 − u)^{L_t+ρ_t} (Lemma C); fl(|a_c|) ≥ |a_c|(1 − u)^{h_a} with h_a = 0 for a real or purely imaginary a_c (real
data: a_c is real) and h_a ≤ h otherwise; fl(|x_j|) ≥ |x_j|(1 − u)^{h_x}, h_x = 0 for real x, ≤ h for complex x; the
product one rounding; the sum w_i − 1 roundings (non-negative terms: fl(Σ) ≥ Σ (1 − u)^{w_i−1}). Altogether

    fl(|H|_abs|x|)_i ≥ (|H_spec|_abs|x|)_i (1 − u)^{w_i + L + ρ + h_a + h_x} ≥ (|H_spec|_abs|x|)_i (1 − γ_{w_i+L+ρ+h_a+h_x}).

Requirement (ii) needs this exponent ≤ width + 1 = W − 1:

* real data, real x: w_i + L + ρ ≤ w_row + L + ρ = W − 1 — holds, with no count to spare (sharp for q = 2 specs);
* real data, complex x (`cplx` is then true in `_rounding_bound` but W is the real-data W): w_i + L + ρ + h ≤ W − 1
  requires h ≤ 0 — the real-data W does **not** cover the modulus of a complex x. Whether this case can occur depends
  on the callers (Theorem 1-fl is applied to Ritz vectors of a real operator, which are real); recorded as a remark
  to be settled against section 2 and the callers;
* complex data, real x: w_i + L + ρ + h ≤ w_row + 3L + ρ + 2 ⇔ h ≤ 2L + 2 — holds for every h ≤ 4;
* complex data, complex x: w_i + L + ρ + 2h ≤ w_row + 3L + ρ + 2 ⇔ h ≤ L + 1 — holds for all L ≥ 2 under the
  3-rounding model h = 3, and for L = 1 only if the library `hypot` is within 1 ulp (h ≤ 2). Under the 3-rounding
  model, a spec whose strings all have one letter (a pure field term with complex data) is two counts short in the
  second pass: γ_{W+1} would be needed where γ_{W−1} is used. This is the delicate point of the document (see the
  comparison below).

Given (i) and (ii), (b) is Theorem 1-fl for the exact spec operator H_spec: rel ≥ ‖H_spec x − μx‖/‖x‖ with no model
shift, because the stored-amplitude roundings are inside W (they are part of (a)), not a perturbation of the operator.

## Lemma N (spec norm bound)

Each coupling contributes c · (JW string) · o_1 ⊗ ⋯ ⊗ o_L (operators on repeated sites multiply): the JW string is a
diagonal unitary, ‖o ⊗ o′‖₂ = ‖o‖₂‖o′‖₂, ‖o o′‖₂ ≤ ‖o‖₂‖o′‖₂, and the triangle inequality gives
‖H_spec‖₂ ≤ Σ_t |c_t| Π_l ‖o_{t,l}‖₂. For any matrix ‖o‖₂² ≤ ‖o‖₁ ‖o‖_∞ (Hölder / Horn–Johnson Cor. 5.6.10 (?) —
‖o‖₂² = ρ(oᴴo) ≤ ‖oᴴo‖_∞ ≤ ‖oᴴ‖_∞‖o‖_∞ = ‖o‖₁‖o‖_∞). Hence ‖H_spec‖₂ ≤ N_spec. (Section 1 cites Cor. 5.6.10; the
identity used is the standard submultiplicativity bound, the corollary number is not checked here.)

Code (`local_norm_bound`, matrixfree.py 100–105): M = |`local.matrix(letter)`| with the *stored* (rounded) amplitudes:
|M_ij| ≥ |ô_ij|(1 − u) (y: hypot(0, a) exact). A letter has at most two nonzeros per row and per column (x, y) or one
(z, n, ±, fermion), so each row/column sum has at most one rounding: rows ≥ R(1 − u)², cols ≥ C(1 − u)², the product
(1 − u), the square root (1 − u)^{1/2}·… ≥ √(RC)(1 − u)^{3.5}, times (1 + 8u) with one rounding and `up` (×(1 + 8u)):
local_norm_bound ≥ √(RC)(1 − u)^{4.5}(1 + 8u)² ≥ √(RC) = √(‖ô‖₁‖ô‖_∞) ≥ ‖ô‖₂. The sum (128–135): per coupling
|c| = `abs(complex(c))` (h roundings), Π of L bounds (L − 1 roundings), one product, and T − 1 additions of
non-negative terms: shortfall ≤ γ_{T+L+h−1} ≤ γ_{T+L+2} for h ≤ 3; `_norm = up(s · (1 + γ_{T+L+2}))` with the
roundings of fl(1 + γ), of the product and of `up` absorbed by the (1 + 8u) guard of `up`. So `norm_bound()` ≥ N_spec.
The adapter's other bound, `up(max |H|_abs 1) (1 + γ_{W})` (enclosure.py 117–120): the row sums of |H|_abs are the
second pass with x = 1 (|x_j| = 1 exact, product by 1 exact): shortfall exponent w_i − 1 + L + ρ + h_a ≤ w_row + L +
ρ + 2 ≤ W for both data types — covered by γ_W with the fl roundings of `up` and (1 + γ) inside the guard. Both bounds
are upper bounds on ‖H_spec‖₂, so the minimum is one. ∎

## Code (current files)

* `matrixfree.py`: `_exact_components` 41–82 (rational exactness test per stored component), `letter_counts` 85–97
  (c(o)), `local_norm_bound` 100–105, `MatrixFreeOperator.__init__` 111–137 (exact_letters, max_len = L, rho,
  terms_per_row = w_row with the coupling multiplicity, W, rounding_width = max(1, W − 2), _norm), `_apply` 157–190
  (full-space index shortcut and `y[rr] += contrib`; `np.add.at` in `SpecOperator._apply` sums the same distinct
  targets, so both give the same bits).
* `enclosure.OperatorAdapter.__init__` 101–123: width = min(3.1 count, rounding_width); g_norm = γ_{width+2};
  _norm = min(row-sum bound, norm_bound()). `_rounding_bound` 295–311: gamma_dot(width + 2), gamma(width + 1), the
  (1 + 4u) factors of Theorem 1-fl.
* Section 3's table names functions, not line numbers; all named objects exist at the places above.

## Comparison with section 2 (read after the proof above)

Lemma M1/M2, Step 1, Step 2, Lemma N and the "why no model shift" paragraph agree with my Lemmas A–C, Theorem (a)
and Lemma N: same combinatorial count (including the repeated-site argument), same per-contribution accounting
(section 2 charges γ₃ per complex product in the amplitude chain where one rounding suffices — every stored component
is real or purely imaginary — which is conservative), same recursive-sum argument, same norm inequality. The
constants match: real W − 1 suffices for (a) with one count of slack, complex `gamma_dot` = 2γ_{W+4} ≥ γ_W.

**Verdict: GAP** (Lemma M3; statement (b) unaffected).

1. *Lemma M3 omits the rounding of |x_j| for a complex vector.* The text counts "the moduli of real amplitudes are
   exact; of complex ones one rounding each (hypot) … then the product with |x_j| and the ≤ w_row − 1 additions",
   i.e. it treats |x_j| as exact. For a complex spec the over-count 3L of Step 1 leaves 2L ≥ 2 spare counts and the
   conclusion (1 − γ_{W−1}) survives (my Lemma M3, with h ≤ L + 1). For a **real spec applied to complex-typed
   vectors** — reachable: `certify.py:893–898` uses `claim.eigenvectors` with the producer's dtype, and
   `_rounding_bound` then takes the complex branch (`cplx = adapter.complex or np.iscomplexobj(X)`) — the second
   pass carries w_i + L + ρ + h_x roundings against a budget of W − 1 = w_row + L + ρ: one count short under the
   one-rounding modulus model, two under a 1-ulp `hypot`. So the inequality claimed in Lemma M3,
   fl(|H|_abs|x|)_i ≥ (|H_spec|_abs|x|)_i (1 − γ_{W−1}) and hence `absHX_up_i` ≥ (|H_spec|_abs|x|)_i, is not
   established in that case. This is the same structure as the (S2) gap recorded for `thm1_fp-accounting.md`
   (complex moduli outside the budget), transplanted to the matrix-free width. **(b) is unaffected**: in the complex
   branch `_rounding_bound` uses g = 2γ_{W+4} ≥ 2γ_W, and a shortfall of a few u in `absHX_up` is absorbed by the
   factor 2 exactly as in Step 3 of Theorem 1-fl.
2. *The modulus model is not stated and, as a one-rounding model, not met on this platform.* Section 1's error
   model covers `amp * factor`, `a * x[src]` and `y[rr] += contrib`, not `np.abs` of a complex number, which Lemma M3
   needs ("one rounding each (hypot)"). NumPy evaluates |z| with the C library `hypot`; my adversarial script
   measures its relative error on 140 000 arguments on this host (Windows, NumPy's complex `absolute`): maximum
   1.88u (seed 107; 1.84u with seed 1) — larger than one rounding (≤ u), within 1 ulp (≤ 2u) and within the three-rounding model (≤ 3u) used by
   `D-01_sparse-residual.md`. A hypothesis of the form |fl(|z|) − |z|| ≤ h u |z| with h ≥ 2 must therefore be
   stated, and the counts of Lemma M3 (and of the adapter row-sum norm bound, which has the room: exponent
   ≤ w_row + L + ρ + h ≤ W for h ≤ 2 real, h ≤ 2L + 2 complex) redone with it. With h = 2 the complex-spec case
   still holds (h ≤ L + 1 for L ≥ 1), the real-spec/complex-x case is two counts short.
3. *Lemma N, code.* Section 2 says `local_norm_bound` evaluates √(‖o‖₁‖o‖_∞) "on the moduli of the exact local
   matrix": it uses `local.matrix(letter)`, whose amplitudes are the stored (rounded) square roots; the shortfall
   (≤ 5.5u by my accounting, since every letter has at most two nonzeros per row and column) is covered by the
   (1 + 8u)² guard. The `norm_bound()` sum: with |c| from `abs(complex(c))` at h = 2 the roundings are
   T + L + 2 ≤ the γ index used; fine, but the text's "≤ T + L + 1 roundings" assumes h = 1.
4. *Section 1's W for complex data* (w_row + 3L + ρ + 3) is larger than needed for (a) by 2L + 1 (my Theorem (a):
   γ_{w_row+L+ρ+2} suffices). Harmless; it is this slack that saves Lemma M3 for complex specs.

Code mapping (step 3 of the protocol): section 3 names functions rather than lines; all exist as described
(`letter_counts`, `_exact_components`, `rho`, `terms_per_row`, `W`, `rounding_width`, `_apply`,
`OperatorAdapter.__init__` width/g_norm/min with `norm_bound()`, `_rounding_bound` gamma_dot(w + 2)/gamma(w + 1),
`local_norm_bound`, `amplitude_hypothesis`, `check_model_fidelity` matrix-free branch at `certify.py:433–446`).
Rounding directions: all inflations upward (`up`, (1 + γ), (1 + 8u)); `rounding_width` = max(1, W − 2) is an
integer; the `min` with the 3.1 count takes the smaller of two proven widths. No code change on this branch: the
correction is at proof/statement level, with the alternative code change (looser bound) recorded in VERDICTS.md.

## Runs

* `python docs/proofs/verify_D-02.py --cases 40 --seed 13` (reduced; default 150 cases): 72 random + 18 adversarial
  cases, 0 skipped, 0 violations.
* `python docs/proofs/independent/adversarial_D-02_matrix-free.py --cases 40 --seed 107`: 0 violations; Lemma M3 as
  stated held in all 43 (real spec, complex x) cases tried — the gap is in the proof, not (yet) exhibited numerically;
  `hypot` maximum relative error 1.88u (exact
  H_spec built independently as sparse mp operator products with explicit Jordan–Wigner strings; (a) entrywise for
  random, wide-range, near-eigenvector and tiny-imaginary-part vectors; Lemma M1 counts; (b) end-to-end radius for
  Ritz pairs, perturbed pairs and complex-typed vectors of a real spec; Lemma M3 as stated per family; Lemma N
  against mp spectral norms; exact-letter decisions; `hypot` accuracy).
