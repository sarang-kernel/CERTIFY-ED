# CERTIFY-ED Validators

A *validator* is an independent check on the correctness of an ED result.
The point of having thirteen of them is that no single check, taken alone,
is sufficient evidence for a publication. Different validators catch
different failure modes, and a result that passes all of them is hard to
explain by anything other than correctness.

This document explains what each validator checks, what kind of bug it is
designed to catch, and what the pass criterion is.

## Failure-mode coverage matrix

| Validator | Algebraic bug | Algorithm bug | Numerical instability | Wrong physics |
|---|---|---|---|---|
| AnalyticValidator        | ✓ | ✓ | ✓ | ✓ |
| QuSpinValidator          | ✓ | ✓ |   | ✓ |
| HighPrecisionValidator   |   |   | ✓ |   |
| SparseDenseValidator     |   | ✓ | ✓ |   |
| JordanWignerValidator    | ✓ |   |   | ✓ |
| SpectralSumRuleValidator | ✓ |   | ✓ |   |
| OrthonormalityValidator  | ✓ | ✓ | ✓ |   |
| UnitarityValidator       | ✓ | ✓ | ✓ |   |
| ConservationLawValidator | ✓ |   |   | ✓ |
| SymmetrySectorValidator  | ✓ | ✓ |   | ✓ |
| ThermalLimitValidator    |   |   |   | ✓ |
| FiniteSizeScalingValidator|  |   |   | ✓ |
| ErrorInjectionValidator  | ✓ | ✓ | ✓ | ✓ |

A bug that affects multiple cells in this matrix is detected with high
probability. A bug that lives only in a single cell is still detected by
the validator that owns that cell.

## Validator details

### AnalyticValidator

Compares against twelve closed-form analytic results:

- single qubit in transverse field — eigenvalues ±h
- 2-site Heisenberg singlet/triplet — −3J/4, J/4 (×3)
- 3-site and 4-site Heisenberg PBC ground state — Bethe ansatz / SU(2)
- TFIM at h = 0 — −J(N−1) doubly-degenerate ground state
- TFIM at J = 0 — −hN
- 2-site XXZ at arbitrary Δ — direct calculation
- Majumdar-Ghosh point — exact dimer ground state
- free fermion OBC — cosine dispersion
- AKLT chain — gap structure
- cluster model at h = 0 — stabilizer Hamiltonian, E = −(N−2)
- 3-site Heisenberg full SU(2) Casimir spectrum

**Pass criterion:** every analytic comparison agrees to 10⁻¹².

**What it catches:** Hamiltonian-construction bugs, indexing bugs,
operator-ordering bugs, sign errors.

### QuSpinValidator

Cross-validates against QuSpin (Weinberg & Bukov, *SciPost Phys.* 2017).
QuSpin is an established, peer-reviewed ED package with a completely
independent implementation. Tests:

- TFIM 20-point sweep over h ∈ [0.1, 2.0] — 320 eigenvalue comparisons
- 4-site Heisenberg full spectrum
- XXZ anisotropy sweep over Δ ∈ {0, 0.5, 1, 1.5, 2}

**Pass criterion:** maximum eigenvalue disagreement < 10⁻¹⁰.

**What it catches:** any bug shared by neither code.

**Note:** QuSpin requires Python 3.10. If unavailable in your environment
the validator skips cleanly without reporting failure.

### HighPrecisionValidator

Cross-validates standard double-precision LAPACK against arbitrary-precision
arithmetic via `mpmath` at 50 decimal digits. Tests cover TFIM, Heisenberg,
and Kitaev chain on small systems. Because mpmath uses a different
algorithmic path (modified QR iteration with arbitrary-precision floats),
agreement to ≈10⁻¹⁵ rules out any systematic numerical-precision issue
in the LAPACK path.

**Pass criterion:** max absolute difference between double and 50-digit
< 10⁻¹⁰. Typically observe ≈10⁻¹⁵ (limited by double precision itself).

### SparseDenseValidator

Compares the lowest five eigenvalues from ARPACK Lanczos (an iterative
Krylov-subspace algorithm) against LAPACK direct diagonalization (Householder
tridiagonalization + QL/QR). These have nothing in common at the algorithm
level. Tests TFIM, Heisenberg, Kitaev chain, free fermion, XXZ at N = 6.

**Pass criterion:** max diff < 10⁻⁸ (Lanczos has lower precision than
direct methods, so tolerance is looser).

**What it catches:** algorithm-specific bugs.

### JordanWignerValidator

For models that are equivalent to free fermions under Jordan-Wigner, the
entire spectrum is given analytically by single-particle energies and the
2ᴺ many-body energies are obtained as sums of those. Tests:

- TFIM PBC ground state from Bogoliubov dispersion
- XX chain (XXZ at Δ=0) — full 2ᴺ spectrum
- free fermion OBC — full spectrum
- Kitaev chain — informational BdG cross-check (sign-convention sensitive)

**Pass criterion:** machine precision agreement on the three stringent tests.

### SpectralSumRuleValidator

Tests four basis-independent invariants over eleven models:

- tr(H) = Σₙ Eₙ
- tr(H²) = Σₙ Eₙ²
- tr(H³) = Σₙ Eₙ³
- ‖H‖₂ = max|Eₙ|

These are exact identities — any deviation is a bug or floating-point
catastrophe.

**Pass criterion:** max absolute error < 10⁻¹⁰.

### OrthonormalityValidator

Tests the three properties that any correct eigendecomposition must satisfy:

- V†V = 𝟙 (eigenvectors orthonormal)
- VV† = 𝟙 (eigenvectors complete)
- VDV† = H (spectral decomposition reconstructs H)

…across nine models.

**Pass criterion:** max deviation from identity / from H < 10⁻¹⁰.

### UnitarityValidator

If V and D are correct, exp(−iHt) = V exp(−iDt) V† must be unitary, must
satisfy U(t₁)U(t₂) = U(t₁+t₂), and must equal scipy.linalg.expm(−iHt). Tests
all three at three different times across seven models.

**Pass criterion:** max unitarity / group / expm-comparison error < 10⁻¹⁰.

### ConservationLawValidator

For each (model, conserved quantity) pair, verifies (a) [H, S] = 0 numerically
and (b) S is block-diagonal in any eigenbasis of H, blocked by H's
degenerate subspaces. Tests:

- Heisenberg, XXZ + total Sz
- TFIM + spin-flip parity (must conserve), TFIM + Sz (must NOT conserve)
- Cluster model + Z-parity (at h = 0)
- Free fermion + total particle number
- Kitaev chain + fermion parity

The "must NOT conserve" tests are crucial — they verify the validator
distinguishes real symmetries from spurious ones.

**Pass criterion:** behavior matches expectation; commutator < 10⁻¹⁰
when conservation is expected, and > 0.1 when it is not.

### SymmetrySectorValidator

For symmetric models, projects H into each symmetry sector, diagonalizes
each block independently, and verifies that the union of sector spectra
matches the full spectrum. Tests Heisenberg/Sz, TFIM/parity, XXZ/Sz.

**Pass criterion:** max difference between full spectrum and union of
sector spectra < 10⁻¹⁰. Because sector diagonalization uses smaller
matrices with different rounding, this is a non-trivial cross-check.

### ThermalLimitValidator

Tests asymptotic thermal observables:

- High T (β → 0): ⟨O⟩ → tr(O)/d, Z → d
- Low T (β → ∞): ⟨H⟩ → E₀

…across four spin models. Catches eigenvector-eigenvalue mis-pairings that
might otherwise hide.

### FiniteSizeScalingValidator

Tests that finite-size results trend toward known thermodynamic limits:

- Heisenberg PBC: E₀/N → ¼ − ln 2 ≈ −0.4431 (Bethe ansatz)
- Free fermion half-filled: E₀/N → −2t/π ≈ −0.6366
- TFIM at criticality (h/J = 1): E₀/N → −4/π ≈ −1.273

Tolerance is loose (5–15%) because finite-N values genuinely differ from
asymptotic ones — but the *trend* and the *order of magnitude* must match.

### ErrorInjectionValidator

The most important validator: it confirms that the framework actually
*detects* errors when they are present.

- Inject anti-Hermitian term → Hermiticity check must flag
- Diagonalize wrong matrix → residuals against true H must grow
- Add a corrupt oracle → consensus must fail
- Perturb eigenvector → residual must amplify
- Tamper with saved certificate → SHA-256 must mismatch on load
- Swap eigenvectors of non-degenerate states → residuals reveal mismatch

**Pass criterion:** all six errors must be detected. Anything less means
the framework has a blind spot.

## How to interpret a full-pipeline run

A "green" run looks like this:

- `pytest`: 53 / 53 passed
- All 13 validators: PASS or SKIPPED (only QuSpin can legitimately skip)
- Total individual validator tests: 78 / 78 passed (or 75 / 75 if QuSpin skipped, depending on environment)

A failing run is a real result. The validator that fails tells you what
class of bug is present. If multiple validators fail, the intersection of
their coverage matrix entries narrows down where to look.
