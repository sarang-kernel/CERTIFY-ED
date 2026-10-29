# Planted-error corpus

Small exact-diagonalization results, each with one known error, in a form any checker can be run on.
The corpus answers one question per case: *does the checker refuse to certify the statement that the
error makes false?* It is the downloadable form of the failure matrix of experiment E1
(`experiments/e1_failure_matrix.py`), extended to eleven error classes and three model families, and it
scores a checker's output without trusting it.

```
corpus/
  make_corpus.py        generator (deterministic; python corpus/make_corpus.py regenerates cases/)
  run_corpus.py         runner and scorer (python corpus/run_corpus.py [--checker "CMD {case} {out}"])
  cases/<case>/         one producer bundle per case + expected.json
  cases/index.json      every case with the SHA-256 of its arrays
  results/<checker>/    report.md, report.json and the per-case checker output of a run
```

The committed `cases/` is 1.7 MB (60 cases). Nothing in this directory is part of the computational
source fingerprint of the package (`certify_ed/provenance.py` covers `certify_ed/` and `experiments/`).

## Models

| key | model | dimension | claim |
|---|---|---|---|
| `heis8` | S=1/2 Heisenberg chain, N=8, periodic | 256 | the k lowest eigenpairs, k = 5 (singlet, triplet, singlet) |
| `j1j2_8` | J1-J2 chain, N=8, J2 = 0.5 (Majumdar-Ghosh point) | 256 | k = 5 (doublet ground state, triplet) |
| `tfim8` | transverse-field Ising chain, N=8, J = h = 1 | 256 | k = 5 |
| `heis10` | S=1/2 Heisenberg chain, N=10, periodic | 1024 | k = 5 |

Reference eigenpairs come from dense `scipy.linalg.eigh` on the package's reference operator
(`certify_ed.build_bitwise`); k is the smallest k >= 5 that does not cut a degenerate level. Every claim
declares `accuracy_target = 1e-8` in `meta.json`: the producer asserts eight digits.

## Error classes

Each case is a producer bundle (`docs/producer_bundle.md`: `eigenvalues.npz`, `eigenvectors.npz`,
`operator.npz`, `spec.json`, `meta.json`) whose *declared model* (`spec.json`) is always the intended one;
the error sits in the arrays or in the stored operator, as it would in a real run. The statement ids are
those of `docs/SCOPE_MATRIX.md`: S1 model fidelity, S2 eigenpairs, S3 completeness, S4 eigenspaces,
S8 degeneracy.

| class | what is planted | a correct checker must not certify | and may certify |
|---|---|---|---|
| `correct` | nothing (control) | - | S1, S2, S3 must be CERTIFIED; S4/S8 when clusters exist |
| `wrong_eigenvalue_1e-05` | eigenvalue of pair 1 shifted by 1e-5 (vector correct) | S2 (no radius below the declared accuracy 1e-8 exists for the pair) | S1, S3 |
| `wrong_eigenvalue_large` | the top eigenvalue shifted past the next level (by gap + 0.1) | S2, S3 (more than k eigenvalues lie below the top of the claim) | S1 |
| `unnormalised_isolated` | one isolated eigenvector scaled by 1.7 | - (Theorem 1 uses the residual divided by the norm; the direction is an eigenvector) | S1, S2, S3; a checker that requires unit vectors says INCONCLUSIVE, not FAILED |
| `unnormalised_cluster_member` | one member of a degenerate multiplet scaled by 1.7 | S2 (distinctness of the cluster, Theorem 2) and S4 (eigenspace, Theorem 3) for the vectors as claimed: the Gram matrix is 1.89 from the identity; *conditional*: a checker that re-orthonormalises and records it may certify | S1, S3 |
| `swapped_pairs` | eigenvectors of two non-degenerate levels interchanged | S2 (the residual of (mu_i, x_j) is the level spacing) | S1 |
| `missing_multiplet_member` | the claim 'the k lowest' stops after the first member of a multiplet | S3 (the gap above the claim is zero; members are missing) | S1, S2; S8 speaks only about clusters of claimed pairs |
| `duplicated_vector` | the ground-state pair listed twice (a Lanczos ghost), the top pair dropped | S2 (two identical vectors are not two distinct eigenpairs), S3 (one eigenvalue is missing) | S1 |
| `wrong_coupling` | producer operator with J -> -J on bond (0,1) (TFIM: field on site 0 doubled); eigenpairs correct for that operator | S1 | S2-S4 *for the producer's operator* (the certificate must say so) |
| `wrong_matrix_entry` | one symmetric pair of matrix entries changed by +0.3 | S1 | as above |
| `non_hermitian` | one lower-triangle entry changed by +0.2 without its transpose | S1 (the stored matrix is not the model) | S2-S4 for the Hermitian part if the certificate says so |
| `nan_eigenvector_entry` | one eigenvector entry is NaN | S2, S3, S4, S8; refusing the input is a correct answer | - |
| `nan_eigenvalue` | one eigenvalue is NaN | S2, S3, S4, S8; refusing the input is a correct answer | - |
| `mislabeled_sector` | the lowest states of one sector (Sz = 0; TFIM: even spin-flip parity) claimed as the model's lowest | S3 (eigenvalues of other sectors lie below the top of the claim) | S1, S2 |
| `subnormal_entries` | a numerically zero eigenvector entry replaced by 1e-310 | S2 under the standard floating-point model (no-underflow hypothesis not established); *conditional*: a model with gradual underflow (Higham 2002, eq. 2.6) may certify | S1 |
| `wrong_window_count` | window claim (a, b) with one interior non-degenerate level left out | S3 (the window holds k eigenvalues, the claim lists k-1) | S1, S2 |

Expectations are derived from the definitions of the statements, not from running any checker: a
statement is in `must_not_certify` when the planted error makes it false, or makes the hypotheses under
which it could be certified unverifiable for the claimed data. Classes that depend on the spectrum (a
multiplet, two isolated levels) are generated only for the models that have the structure.

### expected.json

```json
{"schema": "certify-ed/corpus-case/1.0", "case": "heis8_swapped_pairs", "model": "heisenberg_chain N=8",
 "error_class": "swapped_pairs", "claim_kind": "lowest", "parameters": {"k": 5, "pairs": [0, 4]},
 "description": "...", "note": "why the statements below are false",
 "must_not_certify": ["S2.eigenpairs"], "must_certify": ["S1.model_fidelity"],
 "may_certify": ["S3.completeness", ...], "refusal_ok": false, "conditional": false}
```

## Running a checker

```
python corpus/run_corpus.py                                        # certify-ed of this checkout
python corpus/run_corpus.py --checker "mychecker {case} --out {out}" --name mychecker
```

`{case}` is the case directory, `{out}` the path stem of the result the checker must write (`{out}.json`
or `{out}`), `{python}` the interpreter. The result is a CERTIFY-ED certificate or a flat object
`{"S1.model_fidelity": "CERTIFIED", ...}` (short ids `S1`..`S8` accepted; statuses CERTIFIED, FAILED,
INCONCLUSIVE, NOT_CHECKED). A checker that exits non-zero without a result has *refused* the case.

Per case the scorer says:

| verdict | meaning |
|---|---|
| detected | no must-not-certify statement is CERTIFIED and at least one is FAILED or INCONCLUSIVE |
| unsound | a must-not-certify statement is CERTIFIED: the error was certified as correct |
| silent | every must-not-certify statement is NOT_CHECKED: the checker did not look |
| refused | the input was rejected; a detection where `refusal_ok` is true, an error otherwise |
| conditional | CERTIFIED on a case whose `expected.json` allows it under a stronger model; reported, not scored |
| error | no readable result and refusal is not an acceptable answer |
| control | a case without planted error; *clean* when no statement in `must_certify` is FAILED or INCONCLUSIVE |

A **false alarm** is FAILED on a statement in `must_certify` (any case); INCONCLUSIVE on one is a **weak
alarm**. The summary line is

```
detected D / missed M (unsound U, silent S, error E) / false alarms F (weak W) of N planted cases, C controls (clean K); conditional X
```

and `missed = unsound + silent + error`. The exit code is 0 when nothing was missed and there is no false
alarm. `results/<name>/report.md` lists every case with the statuses the checker gave;
`report.json` holds the same rows; `results/<name>/<case>/checker_output.txt` the command, exit code and
output. The run of CERTIFY-ED 3.2.0.dev0 on this corpus is committed under `results/certify-ed/`.

## Revision history of the expectations

Expectations must not be tuned to the checker under test. One revision has been made, and it followed an
observed result, so it is recorded here and both scores are kept:

* **2026-10-29, `unnormalised_cluster_member`:** the first version of `expected.json` listed only S4 in
  `must_not_certify`, while its note already said that S2's distinctness for the cluster (Theorem 2) rests
  on the same near-orthonormality hypothesis as S4 (Theorem 3). The first run of CERTIFY-ED 3.2.0.dev0 on
  that version scored *detected 49 / missed 4 (unsound 0, silent 4, error 0) / false alarms 0*: the four
  `unnormalised_cluster_member` cases came out *silent* because S4 was NOT_CHECKED (it depends on S3, which
  was INCONCLUSIVE) while S2 was INCONCLUSIVE with the distinctness reason. The expectation was then
  changed to `[S2, S4]`, which is what the note had stated. Under the revised expectation the same run
  scores the four cases as *detected*. Both reports are committed:
  `results/certify-ed_expectations-before-revision/` (first run, original expectation) and
  `results/certify-ed/` (second run, revised expectation). A reader who holds that only S4 is falsified
  by this error should read CERTIFY-ED's score on this class as *silent*, not *detected*.

Any further change to a `must_not_certify` list must be added to this section with the score before and
after.

## Regenerating and verifying the corpus

`python corpus/make_corpus.py` rewrites `cases/`. The generator has no random element: fixed model
parameters, dense `eigh`, and planted changes that are stated arithmetic operations on the arrays. On the
same NumPy/SciPy/BLAS the arrays are bit-identical to the committed ones (`index.json` has their
SHA-256); on another stack `eigh` may return eigenvectors that differ in sign or, within a degenerate
multiplet, in basis, so the hashes may differ while every expectation still holds (they depend on the
spectrum, not on the basis). Compare with `python -c "import json; ..."` on the two `index.json` files, or
simply use the committed cases.

## Submitting a case

A case is a directory with the five bundle files and an `expected.json` as above. To propose one:

1. Plant one error by a mechanism a real producer could exhibit (wrong input, wrong convention, solver
   behaviour, post-processing); say which in `note`.
2. Keep the declared model (`spec.json`) the intended one; the error lives in the arrays or the stored
   operator.
3. Derive `must_not_certify` from the statement definitions (`docs/theory.md`, `docs/SCOPE_MATRIX.md`),
   not from a checker's output; put a statement in `must_certify` only when the error cannot affect it.
4. Keep it small (dimension <= 4096, k <= 20, under 200 kB) and deterministic; add the generator code to
   `make_corpus.py` (a new error class or model) rather than committing hand-made arrays.
5. Run `python corpus/run_corpus.py --only <case>` with CERTIFY-ED; if CERTIFY-ED misses it, say so in the
   pull request: a miss is a finding about the checker, not a reason to change the expectation.

Open an issue or a pull request against `dev/3.2` (`CONTRIBUTING.md`).
