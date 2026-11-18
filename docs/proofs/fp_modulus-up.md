# fp.modulus_up: upper bounds on complex moduli

Derivation document in the D-59 format (the six sections of `docs/proofs/`; `docs/theory.md`, section 10). Implements
`certify_ed/fp.py` (`modulus_up`, `_modulus_up_into`, `_modulus_up_block`, `_modulus_up_scaled`, `abs_up`), for the
rigorous bounds that take the modulus of complex data. This change adds the functions; no bound uses them yet
(section 3).
**UNVERIFIED (D-12):** new mathematics of 2026-11-18; it is not checked by Sarang, and nothing below says it is.

**Context.** CERTIFY-ED certifies statements about the eigenpairs of a stored or declared matrix (enclosures, counts,
eigenspaces, derived quantities). A rigorous bound is a computed binary64 number that is provably at least the exact
quantity, every rounding included. Rule D-12 (`CONTRIBUTING.md`, "The mathematics rule"): new mathematics is marked
UNVERIFIED until Sarang has re-derived it. References: P15 Lemma A (`docs/proofs/P15_summed_bounds.md`), the
underflow model (M-U) (`docs/proofs/int05_underflow.md`), Theorem 1-fl (`docs/proofs/thm1_fp-accounting.md`), and
N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.

Notation: u = 2⁻⁵³; γ_k = ku/(1 − ku); Ω = (2 − 2⁻⁵²)·2¹⁰²³, the largest binary64 number; 2⁻¹⁰²², the smallest
positive normal number; 2⁻¹⁰⁷⁴, the smallest positive binary64 number. For z = x + iy with binary64 components,
|z| = √(x² + y²) is the exact modulus, and m = `modulus_up(z)` the computed value.

## 1 Statement

**Goal.** A function m = `modulus_up(z)` with m ≥ |z| for every finite z, from correctly rounded operations only, so
that a bound built from moduli needs no rounding count for them, and m close to |z| (a stated relative bound).

**Why.** Up to 3.2 the rounding analyses of the residual bounds used `numpy.abs` of complex numbers and counted it as h
roundings, that is, they took the computed modulus within a factor (1 ± u)^h of |z|: h = 1 in `thm1_fp-accounting.md`
(Step 2), h ≤ 2 in `D-02_matrix-free.md` (Lemma M3), h ≤ 3 in `D-01_sparse-residual.md` (Step 4) and
`E-02_non_hermitian.md` (hypothesis H-mod of (FP)), and an exact modulus in the other documents. No bound on the
error of NumPy's complex absolute value is established, in this project or in NumPy's documentation. On the Windows
reference host (NumPy 2.4.4, AVX512F) its relative error (to first order from the exact squares; the values to 40 digits
agree in the digits shown) is −2.466u for z = 0.02019294356747192 − 0.5012539518584045i, −2.462u for z =
1.9986893956853402 + 0.07375207458660175i and +2.269u for z = 1.010573342221019 + 0.3330018761835207i. For z = 2⁻¹⁰⁷⁴(1
+ i) it returns 2⁻¹⁰⁷⁴, below |z| = √2·2⁻¹⁰⁷⁴; a correctly rounded modulus returns the same value, so this case shows
that a rounded modulus is not an upper bound, not an error beyond rounding. `modulus_up` is an upper bound from
correctly rounded operations, so that an analysis that takes its moduli from it counts no rounding for them; the
routing of the bounds through it and the corrections of the documents are a separate change.

**Hypotheses (FP-M).** IEEE 754 binary64 arithmetic with the following properties. *Probed*: checked in every
`certify()` run by `fp.probe_float_environment`; a failed probe turns every CERTIFIED floating-point statement into
INCONCLUSIVE. *Tested*: not probed at run time; checked with known answers in exact arithmetic by
`tests/test_modulus_up.py::test_fp_m_hypotheses_on_this_host` on the host that runs the test suite, and assumed in a
run on a host where the suite has not run.

- Round to nearest, ties to even (probed).
- Gradual underflow: no flush to zero of results, no zeroing of subnormal operands (probed).
- Multiplication and addition are correctly rounded, subnormal results included (tested). With gradual underflow this
  is the model (M-U) of `int05_underflow.md`: fl(t) = t(1 + δ) + η with |δ| ≤ u, |η| ≤ 2⁻¹⁰⁷⁵ and δη = 0.
- `np.sqrt` is correctly rounded (IEEE 754-2008, §5.4.1, squareRoot) (tested).
- `np.frexp(v)` returns the exact exponent e with v = f·2^e, f ∈ [1/2, 1), subnormal v included (tested).
- `np.ldexp(v, k)` returns v·2^k when that number is a binary64 number, ∞ when it exceeds Ω, and a point of the
  subnormal grid within 2⁻¹⁰⁷⁴ of it when it is below 2⁻¹⁰²² (tested). IEEE 754-2008 scaleB (§5.3.3) is correctly
  rounded; NumPy calls the C library's `ldexp`, for which no rounding is assumed beyond this property.
- `np.nextafter(t, ∞)` of a number 0 ≤ t < 2⁻¹⁰²² is t + 2⁻¹⁰⁷⁴ (tested).
- Each NumPy ufunc call rounds each of its operations once, and the code makes one call per operation, so the
  roundings of the proof are the ones the code performs (tested through the correctly rounded results above). The code
  uses no fused multiply-add; the proof does not cover one.

**Inputs.** Real floating input: `np.abs(z)`, exact, in the input's own type. Integer and boolean input: the
absolute values as float64; from 2⁵³ on, where the conversion rounds to nearest, one step up (`np.nextafter`), so
m ≥ |z| (`np.abs` itself is not used on integers: it wraps at the most negative integer). complex64 and complex128:
the computation below; complex64 components are widened to binary64 exactly. A scalar gives a Python float (a NumPy
scalar for a real type wider than binary64, which a float would round). These are statements about the moduli only:
a bound formed in single precision is not covered (Risks and limits). Rejected with TypeError: scipy sparse
matrices (`abs_up` takes them), object arrays (their entries need not be binary64 numbers, and `np.abs` of them is
not an upper bound: m < |z| in 49 842 of 100 000 random entries, independent re-derivation, section 4), complex types
wider than complex128 (`np.clongdouble` where long double is wider than binary64: the components would be rounded on
the way to binary64), and every other dtype. Lemma M is stated for binary64 components.

**The computation.** For complex input (see Inputs), with x := |Re z| and y := |Im z| (exact), M := max(x, y) and
N := min(x, y), each entry is computed by itself (the array is processed in passes of at most 2¹⁸ entries along its
leading axes, without a copy of the input, which bounds the temporaries and does not change any value):

- *unscaled branch* (2⁻⁵⁰⁰ ≤ M ≤ 2⁵⁰⁰, or M = 0): m = fl(fl(√(fl(fl(x·x) + fl(y·y)))) · (1 + 8u));
- *scaled branch* (every other entry, ∞ and nan included; the last rule below then overrides the non-finite ones):
  with e the exponent of M (`np.frexp`), a = M·2⁻ᵉ, b̂ = the
  computed N·2⁻ᵉ (`np.ldexp`), m_s = fl(fl(√(fl(fl(a·a) + fl(b̂·b̂)))) · (1 + 8u)), m′ = `np.ldexp`(m_s, e), and
  m = m′ + 2⁻¹⁰⁷⁴ if m′ < 2⁻¹⁰²², else m = m′;
- *zero component*: m = y if x = 0, m = x if y = 0 (exact);
- *non-finite component*: m = ∞ if x or y is infinite; otherwise a nan component gives nan (as `np.abs` does).

**Lemma M.** For every z with finite binary64 components:

- (i) m ≥ |z|;
- (ii) if m is finite, m ≤ |z|(1 + 12u) + 2⁻¹⁰⁷³; if moreover |z| ≥ 2⁻¹⁰²², m ≤ |z|(1 + 12u);
- (iii) m = ∞ only if |z|(1 + 12u) > Ω;
- (iv) if x = 0 or y = 0, m = |z|.

*Remark (not used by the bounds).* When both components are nonzero and m is finite and at least 2⁻¹⁰²², m ≥
|z|(1 + 4u)(1 − 2⁻⁷³) > |z|(1 + 3u): the lower side of Step 3. `tests/test_modulus_up.py` pins the factor (1 + 8u)
with the weaker m ≥ |z|(1 + 3u); the independent re-derivation proves m ≥ |z|(1 + 4u) for two nonzero components.

**Corollary (no rounding count for moduli).** Let a_ij be the stored entries of row i of a matrix A, x_j the stored
entries of a vector x, and w the number of stored entries in row i. Let m_ij ≥ |a_ij| and m_j ≥ |x_j| be finite
binary64 numbers (from `modulus_up` or `abs_up`; exact for real data). Then the computed sum of the w products
m_ij m_j, each product and each addition rounded once in binary64, in any order, satisfies

    fl(Σ_j m_ij m_j) ≥ (1 − γ_w) Σ_j |a_ij||x_j| − w·2⁻¹⁰⁷⁵,

and the term w·2⁻¹⁰⁷⁵ is absent when no product m_ij m_j is a nonzero number below 2⁻¹⁰²² (no product underflows).
So a bound that needs only upper bounds on the moduli counts the products and the additions, and nothing for the
moduli. More generally, let E be built from non-negative arguments by sums and products along a fixed tree in which
every path from the root to a leaf passes at most d operations. If the arguments are finite and no product in the
tree underflows, the computed value at the moduli bounds is at least (1 − γ_d) times E at the exact moduli. If a
modulus bound is ∞, the computed value can be ∞ or nan (∞ · 0 = nan); the corollary then says nothing, and a caller
must treat such a value as no bound (Risks and limits). An analysis that uses the corollary carries the term
w·2⁻¹⁰⁷⁵ or a no-underflow hypothesis: Theorem 1 (S2) carries the underflow terms η of `int05_underflow.md`; for the
other analyses the treatment is the one each document states (not re-checked here).

**What is not claimed.**

- `modulus_up` is not a correctly rounded modulus and is not a lower bound. Where a bound needs a lower bound on a
  modulus (a modulus in a denominator), it does not apply.
- Duplicate entries. For a sparse matrix in CSR, CSC, BSR or COO format, `abs_up` first sums duplicate entries in place
  (`A.sum_duplicates()`, in the type of the data, as `abs(A)` does), and its moduli are those of the summed entries;
  the exact sum of the duplicates can differ from the stored sum by a rounding, and no bound refers to it. A product
  formed with A after the call uses the summed matrix, and these moduli bound its rounding. A product formed before
  the call uses the stored entries, whose rounding these moduli do not bound, so a bound that combines the two is not
  valid for a matrix with duplicate entries. Direct calls of `davis_kahan.commutator_bound`, `sectors.audit_sector`,
  `correlations.compress_operator` and `eigenspace.certify_observable` do this (they form a product, then take the
  moduli; `abs` sums in place in the same way): for the real 2 × 2 CSR matrix whose (0, 1) and (1, 0) positions each
  hold the two stored entries 2⁵³ and −2⁵³ + 2 (summing to 2), and S = diag(0, 0.7481628719198741),
  `commutator_bound(H, S)` returns 1.0000000000000024, below the exact ‖[H, S]‖₂ = 1.4963257438397481 (a reviewer's
  example, reproduced; with the duplicates summed first it returns 1.4963257438397513). Within `certify()` the
  operators that reach these functions are canonical: built by the library, or the private copy that `certify()` makes
  of a CSR operator whose indices are unsorted or repeated (CHANGELOG, 3.3.0.dev0, "The caller's matrix stays as
  given"). Open, as a separate change because it changes real-data bounds for such input: an `abs_up` that bounds the
  stored entries without summing them, valid for products with the summed and with the unsummed matrix.
  `tests/test_modulus_up.py` pins the summation as it is.

**Risks and limits.** One line per item, with its mitigation or "open".

- Platform. The items of FP-M marked *tested* are not probed at run time; the evidence for them is from one host
  (Windows, NumPy 2.4.4, AVX512F), and the arithmetic record of a certificate (`fp.float_env_record`) does not name
  them. Mitigation: `test_fp_m_hypotheses_on_this_host` runs with the test suite on every host that runs it. Open (a
  separate change, as it adds keys to every certificate): a run-time probe of these items and an assumption line.
- ∞ and nan. `modulus_up` returns ∞ for a finite z when |z|(1 + 12u) > Ω (Lemma M (iii)); for z = x + i with x one
  step below Ω it is ∞ where `np.abs` is finite. A dense product then forms ∞ · 0 = nan (a sparse product skips the
  structural zeros). A caller must treat a non-finite bound as no bound; this change has no callers. `modulus_up`
  raises no floating-point error under `np.errstate(all="raise")`; its callers may raise FloatingPointError then.
- Dtypes. As in Inputs; other dtypes are rejected with TypeError.
- Single precision. The moduli of float16, float32 and complex64 data are exact or upper bounds, but Lemma M and the
  corollary say nothing about products formed in single precision. With a float32 or complex64 operator and vectors
  of the same type, A x and |A||x| are rounded with u = 2⁻²⁴ while the bounds count u = 2⁻⁵³: a reviewer obtained a
  false S2 certificate of `certify()` for the float32 2 × 2 matrix [[1, 2⁻³⁰], [2⁻³⁰, 1]] with x = (1, 1) (enclosure
  [0.999999999999999, 1.000000000000001], eigenvalues 1 ∓ 2⁻³⁰), at the base of this change too. Not covered here;
  open (a separate change): refuse or widen every non-binary64 operator and vector at the entry points.
- Cost. On large arrays one `modulus_up` pass takes about four times as long as one `np.abs` pass (section 4); a call
  on a small array costs a fixed overhead of tens of microseconds (section 4). Its temporaries beyond the float64
  result are bounded by about 12 MiB (six arrays of 2¹⁸ float64 numbers) whatever the size and memory layout of the
  input; 6.8 MiB measured.
- Duplicate entries. As in What is not claimed.

**Alternatives considered.**

- `np.abs` (NumPy's complex absolute value): no error bound is established for it; on the reference host its error
  is −2.466u to +2.269u, and for 2⁻¹⁰⁷⁴(1 + i) it is below |z|. Not an upper bound.
- `np.abs` times a fixed outward factor: needs a bound on its error on every platform and SIMD path; none is
  established, and a factor sized on one host proves nothing for another.
- A correctly rounded hypot with one step up: one ulp above |z|, but NumPy has none; it would need a compiled
  dependency with a vectorised wrapper on every platform.
- `correlations.modulus_upper` (P15 Lemma A) as it is: scalar Python code (a loop over 10⁷ entries), up to √2 above
  |z| when a component is below 2⁻⁵⁰⁰ (for z = 10⁻³⁰⁰ + 3·10⁻³¹⁰i it is 2.7·10⁶u above |z|, an analytic value), and ∞
  when a square or the sum of the squares overflows, or, for a component below 2⁻⁵⁰⁰, when |x| + |y| rounded up
  overflows. With the scaling added it is `modulus_up`. When both components lie in [2⁻⁵⁰⁰, 2⁵⁰⁰] the two functions
  perform the same five operations and agree bit for bit (section 4).
- Why `combine_certified` keeps `modulus_upper`: it bounds a few coefficients and values per call, the scalar code
  has its own proof (P15, re-derived with verdict AGREES), and in the range where both apply the two functions agree
  bit for bit. Replacing it would change P15's records and need P15 re-derived, for no gain there. Two points of P15
  are logged for its D-12 check (VERDICTS.md): its expansion of (1 − u)³(1 + 8u) prints 13u³ for 23u³, and its
  reported main-branch overestimate 2.2·10⁻¹⁵ exceeds the 12u ≈ 1.33·10⁻¹⁵ of Lemma M (ii) for the same computation
  (probably a ratio of squares, m²/|z|² − 1 ≈ 2(m/|z| − 1)).
- Arb balls (python-flint `acb`, as `rigorous.py` uses them): rigorous and tight, but one Python object per entry
  (orders of magnitude slower for 10⁷ entries) and an optional dependency.
- The squares x² + y² without the root: suited where the quantity is itself a square and must be enclosed from both
  sides (such as ‖C‖_F² and ‖H‖_F²); a bound on Σ_j |a_ij||x_j| needs the moduli themselves.
- |x| + |y| rounded up: one addition, but up to √2 above |z| (at |x| = |y|), which widens every radius by up to 41 %.
- Directed rounding (toward +∞ for each operation): NumPy has no portable control of the rounding mode; a process-wide
  mode changes other code, and `fp.probe_float_environment` requires round to nearest.
- Always scaling (no unscaled branch): the same values, since the scaling is exact in the safe range, but slower
  (section 4).
- The guard 1 + 4u: it already gives (i) ((1 − u)³(1 + 4u) > 1; independent re-derivation, section 3) and would give
  (ii) with 8 for 12. 1 + 8u is used because it is the factor of `fp.up` and of P15 Lemma A, so that the two functions
  agree bit for bit, and it leaves margin. 1 + 2u does not give (i).
- Other thresholds than 2^±500: with them the squares lie in [2⁻¹⁰⁰⁰, 2¹⁰⁰⁰] and their sum below 2¹⁰⁰¹, normal with
  margins 2²² and 2²³; P15 Lemma A uses the same threshold; the independent re-derivation finds them conservative by
  factors 2¹⁰ and 2¹¹ (section 2.3 there).

## 2 Proof

**Common setting.** Entries with M = 0 take the unscaled branch, where the computed value is 0; Step 5 replaces it by
the exact 0, so Steps 1–4 concern M > 0. In both branches write a := M·2⁻ᵏ and b := N·2⁻ᵏ, with k = 0 in the unscaled
branch and k = e in the scaled branch, and r_s := √(a² + b²) = |z|·2⁻ᵏ. In both branches 2⁻⁵⁰⁰ ≤ a ≤ 2⁵⁰⁰ (a ∈ [1/2, 1)
when k = e), so r_s² ≥ a² ≥ 2⁻¹⁰⁰⁰, and r_s ≤ √2·a. The computed larger component is a itself: in the unscaled branch
nothing is computed, in the scaled branch a = f ∈ [1/2, 1) is a normal number and the scaling by a power of two is
exact. The computed smaller component b̂ equals b, except in the scaled branch with b < 2⁻¹⁰²², where `np.ldexp` may
round it to the subnormal grid: |b̂ − b| < 2⁻¹⁰⁷⁴, and then |b̂² − b²| ≤ (b + b̂)|b̂ − b| ≤ 2⁻¹⁰²¹·2⁻¹⁰⁷⁴.

**Step 1 (the sum of squares).** a² ∈ [2⁻¹⁰⁰⁰, 2¹⁰⁰⁰] is a normal number, so fl(a²) ∈ a²[1 − u, 1 + u] (correctly
rounded multiplication, FP-M). For b̂² the model with underflow of FP-M (`int05_underflow.md`) gives fl(b̂²) = b̂²(1 +
δ) + η with |δ| ≤ u and |η| ≤ 2⁻¹⁰⁷⁵. With the bound on |b̂² − b²|, fl(b̂²) ∈ [b²(1 − u) − 2⁻¹⁰⁷⁴, b²(1 + u) + 2⁻¹⁰⁷⁴].
The sum of the two lies in [r_s²(1 − u) − 2⁻¹⁰⁷⁴, r_s²(1 + u) + 2⁻¹⁰⁷⁴]. Since 2⁻¹⁰⁷⁴ ≤ 2⁻⁷⁴ r_s² ≤ 2⁻⁷³(1 − u) r_s², it
lies in r_s²[(1 − u)(1 − 2⁻⁷³), (1 + u)(1 + 2⁻⁷³)]. This sum is in [2⁻¹⁰⁰¹, 2¹⁰⁰²], a normal number, so the addition
rounds with a relative u: p := fl(fl(a²) + fl(b̂²)) ∈ r_s²[(1 − u)²(1 − 2⁻⁷³), (1 + u)²(1 + 2⁻⁷³)].

**Step 2 (the square root).** s := fl(√p) ∈ √p[1 − u, 1 + u] (correctly rounded `np.sqrt`, FP-M). With √(1 − t) ≥ 1 − t
and √(1 + t) ≤ 1 + t/2, s ∈ r_s[(1 − u)²(1 − 2⁻⁷³), (1 + u)²(1 + 2⁻⁷⁴)].

**Step 3 (the guard).** 1 + 8u = 1 + 2⁻⁵⁰ is a binary64 number, and s(1 + 8u) is normal and below Ω, so
m_s := fl(s(1 + 8u)) ∈ s(1 + 8u)[1 − u, 1 + u]. Lower side:

    m_s ≥ r_s (1 − u)³(1 + 8u)(1 − 2⁻⁷³) ≥ r_s (1 + 4u)(1 − 2⁻⁷³) ≥ r_s,

since (1 − u)³(1 + 8u) = 1 + 5u − 21u² + 23u³ − 8u⁴ ≥ 1 + 4u (u ≥ 21u² + 8u⁴) and 4u ≥ 2⁻⁷³(1 + 4u). Upper side:

    m_s ≤ r_s (1 + u)³(1 + 8u)(1 + 2⁻⁷⁴) = r_s (1 + 11u + 27u² + 25u³ + 8u⁴)(1 + 2⁻⁷⁴) ≤ r_s (1 + 12u),

since u ≥ 27u² + 25u³ + 8u⁴ + 2⁻⁷⁴(1 + 11u + 27u² + 25u³ + 8u⁴) for u = 2⁻⁵³. Also m_s ≤ √2·2⁵⁰⁰(1 + 12u) < Ω.

**Step 4 (scaling back).** Unscaled branch: k = 0 and m = m_s, so |z| = r_s ≤ m ≤ |z|(1 + 12u), and m ≥ 2⁻⁵⁰⁰ is
finite. Scaled branch: put v := m_s·2ᵉ (exact real number); by Step 3, |z| ≤ v ≤ |z|(1 + 12u). The ldexp clause of
FP-M gives three cases:

- (a) If 2⁻¹⁰²² ≤ v ≤ Ω, v is a binary64 number (the 53-bit significand of m_s times a power of two in range) and
  `np.ldexp` returns it; m′ = v ≥ 2⁻¹⁰²², no step is taken, m = v.
- (b) If v > Ω, `np.ldexp` returns ∞: m = ∞ ≥ |z|, and Ω < v ≤ |z|(1 + 12u), which is (iii).
- (c) If v < 2⁻¹⁰²², `np.ldexp` returns m′ with |m′ − v| < 2⁻¹⁰⁷⁴. If m′ < 2⁻¹⁰²² the code returns m = m′ + 2⁻¹⁰⁷⁴
  (the nextafter clause of FP-M), so v < m < v + 2⁻¹⁰⁷³. If m′ = 2⁻¹⁰²² (rounded up to the smallest normal
  number), m = m′ and v ≤ m < v + 2⁻¹⁰⁷⁴.

In cases (a) and (c), m ≥ v ≥ |z| and m < |z|(1 + 12u) + 2⁻¹⁰⁷³. If |z| ≥ 2⁻¹⁰²², then v ≥ 2⁻¹⁰²² and (a) or (b)
applies, which gives the second half of (ii). The unscaled branch never returns ∞, which completes (iii).

**Step 5 (zero and non-finite components).** If y = 0 then |z| = x and the code returns x; if x = 0 it returns y.
These assignments come after the branches, so they hold whatever the branches computed; this is (iv). An infinite
component makes |z| = ∞, and the code returns ∞. ∎

**Proof of the corollary.** Write P_j := m_ij m_j. Then P_j ≥ |a_ij||x_j| ≥ 0, by Lemma M (i) or because real moduli
are exact. By the model of FP-M, fl(P_j) ≥ P_j(1 − u) − 2⁻¹⁰⁷⁵, and fl(P_j) ≥ P_j(1 − u) when P_j is 0 or at least
2⁻¹⁰²² (an overflow to ∞ only helps). The sum of two non-negative binary64 numbers is rounded with one factor 1 + δ,
|δ| ≤ u, and no additive term: when the exact sum is below 2⁻¹⁰²² it is a binary64 number and is exact. In any order
of summation each term passes at most w − 1 additions, and every computed partial sum is non-negative, so

    fl(Σ_j m_ij m_j) ≥ (1 − u)^(w−1) Σ_j fl(P_j) ≥ (1 − u)^w Σ_j P_j − w·2⁻¹⁰⁷⁵ ≥ (1 − γ_w) Σ_j |a_ij||x_j| − w·2⁻¹⁰⁷⁵,

since (1 − u)^(w−1) ≤ 1 and (1 − u)^w ≥ 1 − wu ≥ 1 − γ_w (Higham 2002, Lemma 3.1). Without underflow the term
w·2⁻¹⁰⁷⁵ is not needed. The general form follows by the same induction over the tree, since E is non-decreasing in
each argument. The counterexample to the form without the term: a = x = 2⁻⁶⁰⁰ (exact moduli), fl(2⁻⁶⁰⁰·2⁻⁶⁰⁰) = 0 <
(1 − γ_1)·2⁻¹²⁰⁰ (independent re-derivation, section 2.7). ∎

**Tightness and slack.** The constant 12 is the first-order count 11 plus room for the second-order terms. The count
11 is the guard 8u plus 3u from the five roundings: the two squares and the sum together move the sum of squares by
at most a relative 2u, which the root halves to u; the root and the guard product cost u each. The independent
re-derivation finds 12 the least integer this count yields. The run of section 4 reaches 10.44u. The additive term
2⁻¹⁰⁷³ matters only for subnormal moduli, where the grid itself is that coarse: for z = 2⁻¹⁰⁷⁴(1 + i), m = 2⁻¹⁰⁷³
while |z| = 1.414·2⁻¹⁰⁷⁴.

## 3 Where the code implements it

`certify_ed/fp.py` on branch `fix/upper-moduli` (line numbers of the commit that adds this document).

| proof object | function | lines | code |
|---|---|---|---|
| constants | module | 29–30, 143–145 | `U = 2^-53`, `_GUARD = 1 + 8u`; `_SMALLEST_NORMAL = 2^-1022`; `_SAFE_LO, _SAFE_HI = 2^-500, 2^500`; `_CHUNK = 2^18` |
| Inputs | `modulus_up` | 191–206 | TypeError for sparse input (191); real floating input: `np.abs(z)`, a float for a scalar of at most binary64 precision (195–197); integers and booleans stepped up from 2⁵³ (198–203); TypeError for other kinds (204) and for complex types wider than complex128 (206) |
| passes, shapes | `modulus_up`, `_modulus_up_into` | 208–242 | a float for a complex scalar (208–212); the result C-ordered in the input's shape (213); passes of at most `_CHUNK` entries along the leading axes, no copy of the input (222–231); x, y the absolute values of the components (exact) |
| unscaled branch, Steps 1–3 | `_modulus_up_block` | 248–257 | warnings off (`np.errstate`); `fast = ((M >= 2^-500) & (M <= 2^500)) \| (M == 0)`; `x*x + y*y`, `np.sqrt`, `* _GUARD` |
| branch choice, ∞ | `_modulus_up_block` | 258–261 | the scaled branch on the entries outside the range (260); `inf` where a component is infinite (261) |
| Step 5, (iv) | `_modulus_up_block` | 262–266 | `copyto(m, y, where=x == 0)`, `copyto(m, x, where=y == 0)` |
| scaled branch: e, a, b̂ | `_modulus_up_scaled` | 151–154 | `e = np.frexp(np.maximum(x, y))[1]`; `np.ldexp(x, -e)`, `np.ldexp(y, -e)` |
| Steps 1–3 (scaled) | `_modulus_up_scaled` | 155–159 | squares, sum, `np.sqrt`, `np.multiply(m, _GUARD)` |
| Step 4 | `_modulus_up_scaled` | 161–164 | `np.ldexp(m, e)`; `m < 2^-1022` → `np.nextafter(m, inf)` |
| `abs_up` | `abs_up` | 269–291 | real floating data: `abs(A)`; complex, integer and boolean sparse data: `A.sum_duplicates()` in place (CSR, CSC, BSR, COO), data `modulus_up(A.data)`, result in A's format and class (COO from `row` and `col`); LIL, DOK and others converted to CSR; dense: `modulus_up(A)` |

No function of `certify_ed` other than these calls `modulus_up` or `abs_up` in this change; the bounds keep
`numpy.abs` and `abs`. The change that routes them lists the call sites, the audit of every modulus call and the
corrections of the documents in this section.

## 4 Computational verification

`docs/proofs/verify_modulus-up.py` (exact rationals, `fractions.Fraction`; every pass/fail decision is an exact
comparison):

- the checks: (i) m² ≥ x² + y²; (ii) m² ≤ (x² + y²)(1 + 12u)² when x² + y² ≥ 2⁻²⁰⁴⁴, and (m − 2⁻¹⁰⁷³)² ≤
  (x² + y²)(1 + 12u)² or m ≤ 2⁻¹⁰⁷³ for every finite m; (iii) (x² + y²)(1 + 12u)² > Ω² whenever m = ∞; (iv) m equal to
  the absolute value of the other component on every case with a zero component; and, apart from Lemma M, inf and
  nan as `np.abs` gives them and real floating input returned exactly;
- random cases: components with random significands and exponents over the whole range (subnormals included); for
  half of them the second component is |x|·f·2^s with f ∈ [0.5, 1) and s ∈ [−4, 4], so that the exponents differ by
  at most 5 and both components matter (where this scaling overflows, z is not finite and is dropped);
- adversarial cases: the three arguments of section 1 and their scalings by ten powers of two (2^k for k = −1074,
  −1060, −1022, −600, −511, −60, 60, 511, 1000, 1022); values near 2⁻¹⁰⁷⁴ and 2⁻¹⁰²² (the subnormal range and its
  top), 2⁻⁵³⁷ (whose square is 2⁻¹⁰⁷⁴), 2⁻⁵¹¹ (whose square is 2⁻¹⁰²²), 2⁻⁵⁰⁰ and 2⁵⁰⁰ (the ends of the unscaled
  branch), 2⁵¹¹ (whose square is near the top of the range), 2¹⁰²³ and Ω, each with its binary64 neighbours; zero
  components; equal components; components differing by 2⁶⁰ and 2⁻⁶⁰; results just below 2⁻¹⁰²²; and 2 000 cases
  with both components nonzero and |z| log-uniform over the subnormal band [2⁻¹⁰⁷⁴, 2⁻¹⁰²²];
- negative controls: `np.abs` on the three arguments (platform dependent; reported, with the threshold 2u, the
  one-ulp hypothesis of the withdrawn attempt); three broken variants, without the factor 1 + 8u (on a second random
  draw of 20 000 cases, seed + 1), without the step below 2⁻¹⁰²² and with the step only below 2⁻¹⁰³⁰ (on the
  adversarial cases), which check (i) must catch: a variant that is never caught fails the run;
- cross-check: bit-for-bit agreement with `correlations.modulus_upper` when both components lie in [2⁻⁵⁰⁰, 2⁵⁰⁰]
  (on the second draw), and Lemma A's bound on the adversarial cases, with its infinite results split by cause.

Third run of 2026-11-18, after the second review round (`python docs/proofs/verify_modulus-up.py`, defaults `--cases
200000 --seed 7`; Windows reference host, NumPy 2.4.4; output in `docs/proofs/runs/fp_modulus-up.txt`): 199 953 random
cases (the finite ones of 200 000 drawn) and 26 815 adversarial cases, 0 violations of (i), (ii) and (iii); 461 cases
with a zero component, 0 violations of (iv); inf, nan and real input as stated. 1 293 results are ∞ (6 random, 1 287
adversarial), each with |z|(1 + 12u) > Ω. The largest relative excess (m/|z| − 1)/u, computed to 40 digits from the
exact m²/|z|², is 10.441. Negative controls: `np.abs` is below |z| on 2 of the 3 arguments and outside 2u on all 3
(−2.466u, −2.462u, +2.269u, to first order from the exact squares); the variant without the factor 1 + 8u fails (i)
on 14 862 of 19 996 cases of the second draw, the variant without the step on 1 166 and the variant with the step
only below 2⁻¹⁰³⁰ on 87 of the 26 815 adversarial cases. Cross-check: 7 307 of 7 307 cases of the second draw agree
bit for bit with `modulus_upper`; Lemma A's bound holds on the adversarial cases (0 below |z|; the largest ratio m/|z|
is 1.414214, that is √2 to six decimals; 4 163 results ∞: 3 466 where a square overflows, 133 where the sum of the
squares overflows, 564 where |x| + |y| rounded up overflows with a component below 2⁻⁵⁰⁰). Run time about 4 minutes.
The independent re-derivation ran its own exact check on 1 009 411 cases (docs/proofs/independent/fp_modulus-up.md,
section 7).

`tests/test_modulus_up.py` pins (i), (ii) and (iv) on the adversarial grid, on random magnitudes and on 2 800 cases
over the subnormal band; the band test catches no step and a step taken only below 2⁻¹⁰²⁵, 2⁻¹⁰²⁶ or 2⁻¹⁰³⁰, while a
step only below 2⁻¹⁰²³ or 2⁻¹⁰²⁴ passes and is sound (there the rescaling loses at most 2⁻¹⁰⁷⁵ ≤ 4u·v, less than the
lower margin of the guard, (1 − u)³(1 + 8u) − 1 ≥ 5u − 21u²). It also pins the guard (m ≥ |z|(1 + 3u), the remark
after Lemma M); the dtypes of Inputs (TypeError for sparse, object and wider complex input; integers stepped up; a
float for every scalar); the passes (equal to one-entry calls at the boundaries; Fortran-ordered, transposed and
sliced input give the values of the C-ordered array, with temporaries below 12 MiB); zero entries on the unscaled
branch; the FP-M items marked *tested*; real input, inf and nan, shapes, the elementwise identity used by the GPU spot
check; `abs_up` on CSR, CSC, BSR, COO, DIA, LIL and dense input, with duplicates summed in place and the format kept,
and on integer and boolean sparse data (stepped up).

**Cost** (best and median of five, seconds, Windows reference host, branch `fix/upper-moduli` at commit 249b8ae). A
complex vector of 2²⁴ entries: `np.abs` 0.162 and 0.171, `modulus_up` 0.698 and 0.739 (about 4.3 times); the scaled
branch on every entry took 1.686 and 1.747 (first review round, same host). The `.data` of a complex sparse matrix
with 10⁷ stored entries: `np.abs` 0.098 and 0.102, `modulus_up` 0.427 and 0.431; the whole matrix, `abs(A)` 0.126 and
0.127, `abs_up(A)` 0.450 and 0.455. Real input of 2²² entries: 0.021 s for both. A call on a small array has a fixed
cost from the many small NumPy calls of the two branches: 96 µs on the 2 × 2 array [[0, −i], [i, 0]] (186 µs before
0 + 0i took the unscaled branch), 90 µs on a 2 × 2 array without zero entries and 102 µs on a complex scalar, against
about 2 µs for `np.abs` (0.2 µs for `abs` of a scalar). Temporaries beyond the float64 result (tracemalloc): 6.8 MiB
on 2²⁴ entries, and 6.8 MiB on C-ordered, Fortran-ordered, transposed and sliced arrays of 2¹⁹ × 8 entries (70.8 MiB
for the Fortran-ordered and transposed array and 30.8 MiB for the slice X[:, 2:5] before the passes avoided the copy).

## 5 Analytic cross-check

The values below are those of `modulus_up` on branch `fix/upper-moduli`, with the relative excesses computed to 40
digits from the exact ratio m²/|z|².

(a) *z = 3 + 4i.* x² + y² = 25 and √25 = 5 are exact; 5(1 + 2⁻⁵⁰) is a binary64 number, so m = 5(1 + 8u): the
relative excess is exactly the guard 8u. `np.abs` returns 5.

(b) *The three arguments of section 1.* m = 0.501660521890638, 2.000049666616023 and 1.064024778635116, relative
excesses 9.49u, 7.54u and 7.91u; `np.abs` returns 0.5016605218906374 (−2.47u), 2.000049666616021 (−2.46u) and
1.0640247786351154 (+2.27u).

(c) *z = 2⁻¹⁰⁷⁴(1 + i).* Scaled branch: e = −1073, a = b = 1/2, p = 1/2, s = fl(√(1/2)), m_s ≈ 0.7071(1 + 8u),
v = m_s·2⁻¹⁰⁷³ ≈ 1.414·2⁻¹⁰⁷⁴, `np.ldexp` rounds it to 2⁻¹⁰⁷⁴, and the step gives m = 2⁻¹⁰⁷³ ≥ |z|. Without the
step the result would be 2⁻¹⁰⁷⁴ < |z| (the second negative control catches such cases).

(d) *z = Ω + Ωi* and *z = Ω + i.* |z| > Ω, so no finite binary64 number bounds it; m = ∞. *z = Ω + 0i:* m = Ω exactly.

(e) *z = 2⁶⁰⁰(1 + i).* Scaled branch, m = 5.868301194789815·10¹⁸⁰, 9.10u above |z|; Lemma A returns ∞ (x² overflows).

## 6 Status

Derived on 2026-11-18; machine-checked (sampled, exact rational checks: 199 953 random + 26 815 adversarial cases, 0 violations; negative controls: broken variants detected in 14 862 of 19 996, 1 166 and 87 of 26 815 cases, `np.abs` below |z| on 2 of the 3 reference arguments); independent re-derivation: done 2026-11-18 (fresh-context session, verdict AGREES on Lemma M and GAP on the corollary, docs/proofs/independent/fp_modulus-up.md, docs/proofs/VERDICTS.md); resolved 2026-11-18 by stating the corollary for finite moduli with the underflow term w·2⁻¹⁰⁷⁵ (absent without underflow), and by rejecting inputs whose components are not binary64 numbers (TypeError) and stepping integer input up from 2⁵³;
Sarang's check (D-12): pending. UNVERIFIED (D-12) until that check.
