# What "certified" means

A certificate is a list of statements about one claimed result, each with a status, its evidence and the
assumptions it rests on. This page says what the statuses mean, which hypotheses every floating-point
statement carries, what no certificate claims, which numbers in a certificate are run-dependent and which
results of the theory are still marked UNVERIFIED. The mathematics is in [theory.md](theory.md).

## The statements

| id | what it establishes | rests on |
|---|---|---|
| `S1.model_fidelity` | the producer's operator is the operator of the declared `ModelSpec`: entrywise for a stored matrix, by invariance and orthonormality of the isometry for a symmetry block, by agreement on random probes for a matrix-free producer | exact comparison with a reference built from the spec, with the builder's rounding bound (the *model shift*) |
| `S2.eigenpairs` | each claimed eigenvalue lies within a certified radius of a true eigenvalue, and the claimed pairs are matched to *distinct* eigenvalues | Theorem 1 (residual), Theorem 2 / 2′ (clusters), Theorem 3 (Kato–Temple) |
| `S3.completeness` | no eigenvalue is missing inside the claimed window (`lowest`, `window`) or the count equals the dimension (`all`); a lower bound on the first unclaimed eigenvalue (the gap) | Lemma 4 (inertia with backward error), Corollary 5 |
| `S4.eigenspaces` | projector error of each claimed cluster, certified observables, correlations and quantum numbers | Theorem 6 (Davis–Kahan), Proposition 7, Proposition 15 |
| `S5.corroboration` | recomputation by algorithmically distinct solvers agrees with the claim — evidence, reported as CORROBORATED, never CERTIFIED | — |
| `S6.exact_recheck` | the S2 radii and the S3 count of a `lowest` claim re-derived in integer arithmetic, with no floating-point assumption | §7 of theory.md (UNVERIFIED) |
| `S7.thermodynamics` | interval bounds on Z, F, U, C from a complete whole-spectrum enclosure | Proposition 16, Lemma 17 (UNVERIFIED) |
| `S8.degeneracy` | the number of eigenvalues in each cluster, by bracketing inertia counts | Proposition 14 (UNVERIFIED) |

## The statuses

- **CERTIFIED** — the statement holds for the stored numbers, under the assumptions listed with it. Nothing
  else in a certificate is a claim.
- **FAILED** — the statement is contradicted by what was computed: the producer's matrix differs from the
  spec, a certified radius exceeds the accuracy target, an inertia count finds an eigenvalue the claim does
  not contain. A FAILED statement is a finding about the claim, not about CERTIFY-ED.
- **INCONCLUSIVE** — the statement was attempted and the evidence neither establishes nor contradicts it:
  vectors too far from orthonormal to separate a cluster, an ill-posed "k lowest" at a multiplet, a
  factorization whose backward error is too large, an input the floating-point hypotheses do not cover.
  The summary says why; the [FAQ](faq_inconclusive.md) says what to do.
- **NOT_CHECKED** — the statement did not run, with a reason code: not selected (`NARROWED_BY_USER`,
  `DEFAULT_SELECTION`), an input it needs is missing (`NEEDS_EIGENVECTORS`, `NEEDS_SPEC`, `NEEDS_MATRIX`,
  ...), a budget or size limit (`BUDGET_TIME`, `BUDGET_MEMORY`, `SIZE_LIMIT`). The reason codes are listed
  in the [scope matrix](SCOPE_MATRIX.md).

Every statement id appears exactly once in the certificate's `scope` block (in scope, narrowed, or out of
scope); `Certificate.load` refuses a certificate whose scope block does not account for every statement
(INT-02, since the merge of `fix/3.1.1`).

## The hypotheses of the floating-point statements

S1–S5, S7 and S8 are rigorous under the standard model of floating-point arithmetic (Higham 2002, eq. 2.4):
IEEE 754 binary64, round-to-nearest, BLAS and sparse kernels whose inner products obey the γ_n bounds in any
summation order, no Strassen-type products, no reduced-precision modes, and **no underflow**.

- *Checked at run time*: `fp.probe_float_environment` tests gradual underflow, denormals-are-zero, the
  rounding mode (including round-toward-zero) and BLAS precision in the running process; a failed probe
  downgrades every CERTIFIED floating-point statement to INCONCLUSIVE and the certificate records which
  probe failed (`arithmetic.runtime_probe`).
- *Checked on the input (since 3.2.0.dev0)*: the no-underflow hypothesis requires every nonzero operand, radius
  and residual bound to have magnitude at least 2⁻⁹⁰⁰. `certify()` scans the claimed eigenvalues, the
  eigenvector entries, the operator's stored entries (or, matrix-free, its norm bound and a probe matvec),
  the model shift and, after the enclosure, the radii and residual bounds. A finding makes S2 and the
  statements that rest on it INCONCLUSIVE with reason code `UNDERFLOW_RISK`. `Options(underflow_check=False)`
  (`--assume-no-underflow`) assumes the hypothesis instead; the choice is recorded in
  `scope.options.underflow_check` and `arithmetic.underflow_check`. The intermediate quantities of the
  bounds are not scanned (known issue INT-05 in the CHANGELOG).
- *Not checked*: that the BLAS in use satisfies the γ_n bounds. It is the standard assumption for
  reference and vendor BLAS; the certificate names it under `arithmetic.assumptions`.

S6 makes none of these assumptions: it works in integer arithmetic on a scaled copy of the spec operator.

## What is not claimed

Every certificate lists these under `not_verified`:

- that the `ModelSpec` is the Hamiltonian the authors intend — only "producer operator equals the spec" is
  checked. A spec written from the paper, independently of the solver's input, is what makes S1 meaningful;
- correctness of the producer's code in general — only this result is checked;
- an interval-arithmetic proof, unless S6 ran: the bounds rely on the standard model;
- anything outside the certified window (S3), and completeness at all for a `subset` claim;
- eigenvector subspaces of isolated pairs beyond their residuals (unless `eigenspaces="all"`);
- entanglement entropies, dynamical response, time evolution, level statistics, finite-size
  extrapolations and phase boundaries: standing out-of-scope quantities, each with a reason code in
  `scope.out_of_scope` ([scope matrix](SCOPE_MATRIX.md)). The 3.2 tools `levelstats`, `crossing` and
  `extrapolate` derive records from certificates after the fact; an extrapolated value is never certified.

A radius is a bound, not an error estimate: the true eigenvalue is within it, and it is typically some
orders of magnitude larger than the solver's actual error.

## Run-stack values

Statuses and integer evidence (counts, degeneracies, numbers of missing eigenvalues) are the claims of a
certificate, and two correct runs on different machines agree on them. Floating-point evidence — radii,
residual bounds, gap bounds, model shifts, backward errors, timings — depends on the *run stack*
(operating system, BLAS, library versions, memory cap mechanism) and may differ in its digits between
runs; each value is a valid bound for the run that produced it. Quote such numbers from the certificate
you have, with the environment it records (`environment`: versions, source hash, git commit), and
compare two runs with `experiments/compare_runs.py`, which separates statuses and counts (strict) from
run-stack values (compared by digits, with explained differences). The evidence run of 3.1.0 is described
in `RESULT_evidence_run_3.1.0.md`.

## UNVERIFIED results

Results that are not textbook material are marked **UNVERIFIED** in [theory.md](theory.md) until a second
person has re-derived them (project rule D-12). A statement that rests on one says so in its `assumptions`
("UNVERIFIED — re-derivation pending, D-12"). As of 3.2.0.dev0:

- Theorem 2 in the stated form (cluster radius with non-orthonormal vectors) and Theorem 2′ (the sharper
  cluster radius): 15 876 random cases without a violation, no second derivation yet. S2 uses them only
  when Theorem 1 intervals of a cluster overlap.
- The sector-lift bound ε = ρ for a square isometry (`compose.py`).
- The exact layer (§7: S6 and the exact sector re-check).
- Proposition 14 (S8 degeneracy counts), Proposition 15 (linear combinations of certified observables,
  S4 extension), Proposition 16 and Lemma 17 (S7 thermodynamics).

Theorem 1, Theorem 3, Lemma 4, Corollary 5 and Theorem 6 are classical (Parlett 1998; Kato 1949, Temple
1928; Sylvester; Davis–Kahan) and carry no such mark.
