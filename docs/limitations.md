# Limitations

What a certificate cannot do, where the statements stop, and the known issues of this version.

## Sizes

| statement | limit | what happens above it |
|---|---|---|
| S1 reference matrix | `Options.reference_max_dim` = 2²² states (full space or U(1) block); the Kronecker cross-check up to `kron_cross_check_max_dim` = 2¹⁴ | no stored reference: S1 compares by probes where it can, the operator is applied matrix-free from the spec |
| S3, S8 inertia counts | `Options.inertia_max_dim` = 60 000 (one dense or sparse LDLᵀ per probe shift, 2n² entries dense) | NOT_CHECKED with `SIZE_LIMIT`; raise the limit if the factorization fits |
| S6 exact re-check | `Options.exact_max_dim` = 4 000 (exact integer matrix and residuals; needs python-flint) | NOT_CHECKED with `SIZE_LIMIT` |
| S2 cluster Gram matrix | `enclosure.EXACT_GRAM_BUDGET` = 2 × 10⁷ entries of n·g(g+1)/2 | the accurately rounded Gram entries are replaced by the γ_n bound (wider η) |

Whole-spectrum claims (`kind="all"`, S7) need every eigenvector, so they are bounded by what the producer
can store, a few thousand states in practice.

## NOT_CHECKED under budgets and memory caps

`Options(budget_seconds=..., budget_mb=...)` turns statements that would exceed a budget into NOT_CHECKED
(`BUDGET_TIME`, `BUDGET_MEMORY`) before they start; a `MemoryError` during S1's reference build and an
inertia count that is not attempted to completion are reported the same way (the statement says what was
not attempted and why). Under an operating-system memory cap the outcome of a borderline S3 can differ
between machines: the 3.1.0 evidence run certified a 4 × 4 J2 = 0.5 completeness check that the Linux run
left NOT_CHECKED, because a Windows job object counts committed memory where RLIMIT_AS counts address
space (`CHANGELOG`, 3.1.0). Neither outcome is wrong; a NOT_CHECKED statement is not a claim.

## The exact layer

S6 re-derives S2 radii and the S3 count in integer arithmetic for `lowest` claims with a spec, eigenvectors
and dimension at most `exact_max_dim`; for symmetry-adapted claims use `exact_recheck_sector_lowest`
directly. It needs python-flint and its mathematics (§7 of [theory.md](theory.md)) is marked UNVERIFIED.
`window`, `all` and `subset` claims have no exact path (`UNSUPPORTED_CLAIM`).

## Matrix-free producers

Without a spec, nothing bounds the rounding inside the producer's `matvec`: S2 is INCONCLUSIVE with
estimates only, and S3, S8 are NOT_CHECKED (`NEEDS_MATRIX`). With a spec, S1 compares on random probes and
the radii refer to the spec operator; the probe comparison is agreement on a few random vectors, not an
entrywise identity. See the [matrix-free cookbook page](cookbook/matrix_free.md).

## What the theory does not cover

Entanglement entropies, dynamical response, time evolution, level statistics, finite-size extrapolation
and phase boundaries are out of scope with reason codes (`NOT_IMPLEMENTED`, `OUT_OF_SCOPE_DYNAMICS`,
`OUT_OF_SCOPE_EXTRAPOLATION`); the [scope matrix](SCOPE_MATRIX.md) names what each would need. Since 3.2
three tools read saved certificates without adding a statement: `certify-ed levelstats` (interval r-ratios
and spacing histograms of a counted whole spectrum; the unfolding is declared, not certified),
`certify-ed crossing` (orders two certified eigenvalue intervals when they do not overlap) and
`certify-ed extrapolate` (a report whose fitted limit is explicitly not certified).

## Known issues

From the CHANGELOG (ids INT-xx; D-57):

- **INT-01 — resolved conservatively in 3.2.0.dev0.** The no-underflow hypothesis was stated and not enforced
  in 3.1.0. It is now checked on the inputs and on the reported radii and residual bounds
  (`UNDERFLOW_RISK`); `Options(underflow_check=False)` restores the 3.1.0 behaviour and records that it did.
- **INT-02, INT-03 — fixed (`fix/3.1.1`, merged).** A validator for the scope block; exported radius
  arrays rounded upward.
- **INT-04 — documentation.** The manuscript's Appendix A described the floating-point accounting of the
  Theorem 1 radius differently from the code; [theory.md](theory.md) §2 now has the code-facing list and
  the appendix is to be rewritten from it. The code's bound was sound throughout.
- **INT-05 — open.** The intermediate quantities of the Theorem 1 bound (the residual vectors and the
  entrywise rounding bound) fall below 2⁻⁹⁰⁰ for routine LAPACK output, where eigenvectors carry entries of
  order 10⁻²⁶⁵ that are numerical zeros of a symmetry sector. Covering this needs either a stronger
  hypothesis or the underflow-tolerant error model (Higham 2002, eq. 2.6) carried through the norm bounds —
  a design decision, not made in this version. The bounds are unaffected in every case examined; the
  statement is about what the hypothesis covers.
- Certificates made with 3.2.0.dev0 are not comparable by `source_hash` with the 3.1.0 evidence run (the
  computational source changed); a version the paper cites needs its own full run (D-44).
