# RESULT: track/props-proofs (Props 14-16 + Lemma 17 derivations, C-06, C-16)

Branch `track/props-proofs` on `dev/3.2` (38b0212). Code head: b95d942 (the full suite ran there); branch head including this file: the commit that adds it (`git log -1`). Bundle: `track-props-proofs.bundle`;
patches: `patches_props_proofs.zip` (`git format-patch dev/3.2..track/props-proofs`, 6 patches).

## Item status

| Item | Status | Notes |
|---|---|---|
| 1. Props 14-16 and Lemma 17 | done | four derivation documents in `docs/proofs/` with verification scripts, pinning tests, two code corrections (ivl.py, correlations.py), statement strings and theory.md section 9 updated |
| 2. C-06 truncated-spectrum thermodynamics | done | `thermo.thermodynamics_truncated`, S7 CERTIFIED for 'lowest' claims with S3 CERTIFIED, evidence `tail_bound`, derivation `C06_truncated_thermodynamics.md`, scope matrix, schema, diagnostics, tests with negative controls |
| 3. C-16 exact degeneracy | done (sector path at the API only) | `certify_ed/exact_degeneracy.py`, S8 evidence `exact_bracket` (plain and U(1)-block claims through `certify()`; the symmetry-sector path `exact_bracket((N, Aim, G), a, b, g, den)` is implemented and tested but not wired for sector-isometry claims), derivation `C16_exact_bracket.md`, tests on the Heisenberg multiplets and a planted non-degenerate triplet |

## Proof documents (docs/proofs/)

Each has the six D-59 sections and the verbatim status line; each verification script prints case counts
and violations; the documents record the numbers of the runs on the committed code.

| Document | Script | Random | Adversarial | Violations |
|---|---|---|---|---|
| `L17_dyadic_exp_ln.md` | `verify_L17.py --random 2000` | 6000 | 140 | 0 |
| `P16_thermo_intervals.md` | `verify_P16.py --random 400` | 4000 | 95 | 0 |
| `P14_degeneracy_bracketing.md` | `verify_P14.py --random 3000` | 3060 | 1006 | 0 |
| `P15_summed_bounds.md` | `verify_P15.py --random 500` | 1111 | 32 | 0 |
| `C06_truncated_thermodynamics.md` | `verify_C06.py --random 300` | 300 | 150 (+7 negative controls, all detected) | 0 |
| `C16_exact_bracket.md` | `verify_C16.py --random 200` | 1501 | 42 (model levels) | 0 |

Ground truth: `fractions.Fraction` (grid operations, Prop 14 model, Prop 15 combinations, C-16 planted
matrices and polynomials), mpmath at 200 digits in interval mode (`mpmath.iv`) for Props 16/16' and Lemma 17,
exact `fmpq_mat` determinants for the sector polynomial identity.

### Findings of the derivations that changed code (all make bounds looser or guard a hang; none tightens)

1. `ivl.py`, Lemma 17 (iv): the atanh series remainder used 1 − u² ≥ 8/9 (valid for u ≤ 1/3), but the rounded
   upper argument u_hi can be 1/3 + 2⁻¹⁹²; the constant is now 8/7 (valid for u ≤ 0.35) with a runtime range
   check. The deficiency was far below one grid unit, so no enclosure produced by 3.1 was wrong in practice;
   the proof needed the weaker constant.
2. `ivl.py`, Lemma 17 (iii): `exp_interval` returns [0, 2⁻¹⁹²] for arguments below −1000 and raises
   OverflowError above 2¹⁶. Found by verify_P16: T = 2⁻¹⁹⁰ with an energy of 1 made the halving loop attempt
   2¹⁹⁰ squarings (a hang, not a wrong result). Also stated: T must be ≥ 2⁻¹⁹² (else ZeroDivisionError,
   caught by `certify()` as INCONCLUSIVE).
3. `correlations.py`, Prop 15: `combine_certified` bounded |a_k| with `np.abs` (a libm hypot, whose error is
   not provable from IEEE) and multiplied a nearest-rounded sum by (1 + 8u), which does not cover m > 8 terms.
   Now `modulus_upper` (IEEE +, ×, √ with the `up` guard; Lemma A) and directed rounding throughout, with the
   rounding term 2 γ_{m+2} Σ|a_k||v_k| (Lemma B proves γ_{m+2} suffices). The bound grows by a few ulps.
4. Documented hypotheses that cannot be proved from the code: (D1) the structure-factor phases are the
   floating-point values of exp(iθ)/N (libm accuracy); (D2) the two-point operator is the stored product matrix
   (exact for dyadic local entries, rounded for spin-1 x/y). Both are now record fields (`phase_hypothesis`,
   `operator_hypothesis`).
5. `degeneracy.py`: the S8 statement text says "in the open interval (…)" (I_in is open; the computed closed
   bracket lies inside the exact open one).

The statement strings keep "UNVERIFIED" in the D-12 sense (independent re-derivation pending) and now name the
derivation document; `docs/theory.md` section 9 points to `docs/proofs/`.

## C-06 (Proposition 16')

For a 'lowest' claim with S3 CERTIFIED (L = `next_eigenvalue_lower_bound`, declared model) and dimension n:
Σ_{j>k} e^{−βλ_j} ≤ (n − k) e^{−βL}; the omitted first and second moments are bounded on both sides by the
envelopes of φ(e) = e e^{−βe} and ψ(e) = e² e^{−βe} over [L, M] (M = operator norm bound + model shift) or
[L, ∞) (Lemma C: φ increases to 1/β and decreases after; ψ to 2/β). Z, F, U, C are two-sided in every case; the
intervals are informative when T ≪ (L − E₀)/ln(n − k) and the record says so (`TAIL_LOOSENESS`). The envelope
bounds are never weaker than adding [L, M] as an extra interval to Proposition 16 and much sharper when
M ≫ L (N = 8 Heisenberg at T = 0.05: C width 5e−4 against 5e−2). Wiring: `certify.py` S7 branch for
`claim.kind == "lowest"`; a 'lowest' claim with S3 CERTIFIED and no temperatures now gets NEEDS_TEMPERATURES
(before: NEEDS_WHOLE_SPECTRUM); `tests/test_diagnostics.py` adjusted accordingly and a 'subset' case added.
Negative controls (tests and verify_C06): subset claim (NOT_CHECKED), S3 skipped (S7 narrowed), S3 not
certified for a cut multiplet, planted L above λ_{k+1}, dimension too small, M below λ_n, input validation.

## C-16 (exact bracket)

What is decidable in exact arithmetic without any floating-point hypothesis: the number of eigenvalues of
the declared model in a rational bracket, counted with multiplicity, and the multiplicity of each distinct
eigenvalue there — via the characteristic polynomial of the exact scaled matrix den·K (similar to H; real
embedding for complex entries, multiplicities halved), Yun's squarefree factorization and Sturm sequences at
the bracket ends (nudged outward by 2⁻²⁰⁰ if an end is a root). Statements: "exactly g eigenvalues in [a, b],
all equal: one eigenvalue of multiplicity g" (`equal = True`), or "exactly g in [a, b], NOT all equal:
d distinct eigenvalues with multiplicities {…}" (`equal = False`), or FAILED when the count differs from g.
Wiring: with `Options.exact_recheck`, a declared spec, no sector isometry, python-flint and dimension ≤ 600,
every S8 cluster record gets `exact_bracket` (the bracket is Proposition 14's inner bracket); the S8 summary
appends the counts; a contradiction makes S8 FAILED; schema 2.1 declares the key. Negative controls: a field of
2⁻³⁰ splits the N = 8 triplet into three levels 2⁻³⁰ apart — S8 certifies "exactly 3 in the bracket" (true) and
the exact bracket says `equal = False`, multiplicities {1: 3}; a bracket holding the triplet with g = 2 is FAILED.

## Files touched

New: `docs/proofs/{L17_dyadic_exp_ln,P14_degeneracy_bracketing,P15_summed_bounds,P16_thermo_intervals,
C06_truncated_thermodynamics,C16_exact_bracket}.md`, `docs/proofs/verify_{L17,P14,P15,P16,C06,C16}.py`,
`certify_ed/exact_degeneracy.py`, `tests/test_proofs_props.py`, `tests/test_thermo_truncated.py`,
`tests/test_exact_degeneracy.py`.
Modified: `certify_ed/ivl.py`, `certify_ed/thermo.py`, `certify_ed/correlations.py`, `certify_ed/degeneracy.py`,
`certify_ed/certify.py` (S7 and S8 blocks), `certify_ed/schema.py` (+ regenerated `schema/certificate-2.1.schema.json`),
`certify_ed/diagnostics.py`, `certify_ed/scope.py`, `certify_ed/presets.py` (text), `docs/theory.md` (section 9
header and Lemma 17 remainder), `docs/SCOPE_MATRIX.md` (thermodynamics and degeneracies rows), `CHANGELOG.md`,
`tests/test_diagnostics.py` (one expectation, C-06).

## Tests added

`tests/test_proofs_props.py` (16: the six scripts at reduced size, document structure, Lemma 17 pins — series
range, exp guards, exp∘ln ∋ z — Prop 15 pins — modulus bound, cancellation — Prop 16 pins — sub-grid T refused,
harness detects a false enclosure — Prop 14 pins — open-interval statement, harness detects a wrong count);
`tests/test_thermo_truncated.py` (5); `tests/test_exact_degeneracy.py` (5). Suite before this track: 953 tests.

## Not done and why

- Sector-isometry claims do not get `exact_bracket` through `certify()`: the sector path exists at the API
  (`exact_bracket((Are, Aim, D), a, b, g, den=ex.den)`, tested on the k = 0 sectors of the N = 8 chain) but
  wiring it needs the generators and fixed quantum numbers of the claim, which `certify()` does not carry for
  S8; noted in the scope matrix.
- `exact_bracket` is capped at dimension 600 (charpoly cost: 0.8 s at 256, 16 s at 400); N = 12 at S^z = 0
  (924) is refused with SIZE_LIMIT.
- The correlation lift (exactness of the two-point product operator, D2) is the other track's; here it is a
  documented hypothesis in the record.
- No experiments/ script or evidence-run record was regenerated (not in this track's scope).
- Mutation testing of the new status-affecting paths was not run; each has a negative-control test.

## Final checks

`pytest -q`: 979 passed, 2 skipped, 5 warnings in 2067.69s (0:34:27) (full suite at b95d942, with the result file itself not yet committed)
`ruff check certify_ed tests experiments conformance corpus docs/proofs`: All checks passed

Register note: documents and code use "derived", "machine-checked", "UNVERIFIED by independent
re-derivation"; the word "verified" is not used for any of this track's statements (the pinning test checks
the documents).
