# FAQ: INCONCLUSIVE and the reason codes

INCONCLUSIVE means the statement was attempted and the evidence neither establishes nor contradicts it.
NOT_CHECKED means it did not run, for a reason the certificate names. This page lists the summaries you
will meet, what each means and the next action. The examples at the end were run with certify-ed 3.2.0.dev0.

## INCONCLUSIVE summaries by statement

### S2.eigenpairs

| summary begins | meaning | next action |
|---|---|---|
| `each claimed value has an eigenvalue within its radius, but some claimed pairs could not be matched to distinct eigenvalues` | two claimed vectors are (nearly) the same, or the vectors of a cluster are too far from orthonormal (η ≥ 1) | re-orthonormalize the cluster's vectors (QR) and re-certify; check for a duplicated eigenpair in the producer's output |
| `each claimed value has an eigenvalue within its Theorem-1 radius ... but the cluster radius ... exceeds it` | the claim is accurate (existence radii meet the target) but the vectors are not orthonormal enough to certify *distinct* eigenvalues at that accuracy (orthonormality defect η named) | orthonormalize the degenerate level's vectors; or state the existence radii only |
| `residuals computed but rounding not boundable for this operator (estimates only)` | a matrix-free producer without a spec | declare a `ModelSpec` so that S1 can tie the matvec to it by probes ([matrix-free](cookbook/matrix_free.md)) |
| `certified radius of pair i is not finite` | a zero (or all-zero) claimed eigenvector | fix the producer's output |
| `claim contains non-finite values` | NaN or inf in the eigenvalues or eigenvectors | fix the producer's output; nothing is enclosed |
| `no-underflow hypothesis not established: nonzero magnitudes below 2^-900 in ...` (`UNDERFLOW_RISK`) | an input or a reported radius is below 2⁻⁹⁰⁰ in magnitude; the floating-point model does not cover it | look at the named location (`evidence.underflow.findings`): a planted or garbage entry should be removed, a genuinely tiny scale rescaled; or pass `Options(underflow_check=False)` to assume the hypothesis, which the certificate records |
| `the floating-point environment of this process violates the standard model (failed probes: ...)` | a library switched on flush-to-zero / denormals-are-zero, or changed the rounding mode | run in a clean process (import order matters; `-ffast-math` binaries are the usual cause); `certify-ed env` shows the probe |

### S3.completeness

| summary begins | meaning | next action |
|---|---|---|
| `the claim is complete below ..., but the highest claimed level is a (quasi-)degenerate multiplet of at least m eigenvalues ... of which only j are claimed: 'the k lowest' is ill-posed for this k` | the claim is correct as far as it goes, but the k-th level is degenerate and only part of it is claimed | claim the whole multiplet (k + m − j) or stop below it; see the [numpy cookbook page](cookbook/numpy_eigh.md) |
| `at least one unclaimed eigenvalue lies within ... above the highest claimed one` | an eigenvalue sits just above the claim, within the probe margin | if it belongs to the claim, add it; if not, the gap is too small to certify at this accuracy — tighten the eigenpairs (smaller residuals) |
| `factorization backward error (...) too large relative to the margin above the highest claimed value` | the LDLᵀ at the probe shift was not accurate enough to separate the claim from the rest | `Options(inertia_method="dense")` for n ≤ a few thousand; otherwise a smaller claim or sharper eigenpairs |
| `inertia counts m eigenvalues below σ but k claimed pairs lie below it: claimed pairs are not all distinct eigenpairs` | more pairs are claimed than eigenvalues exist there | a duplicated eigenpair in the claim; see S2 |
| `claimed pairs not all certified distinct (S2); the count matches but cannot be attributed` | S2 could not separate a cluster | fix S2 first |
| `counts could not be matched to the claimed set` (window claims) | the certified intervals do not all lie strictly inside the window, or touch its edges | widen the window beyond the enclosures, or shrink the claim |
| `some claimed pair has no finite enclosure (zero eigenvector)` | a zero vector | fix the producer's output |
| `S2 inconclusive under UNDERFLOW_RISK` | S3 rests on the S2 enclosures | see S2 |

### S4.eigenspaces, S8.degeneracy, S7.thermodynamics

| statement | summary | meaning | next action |
|---|---|---|---|
| S4 | `cluster vectors too far from orthonormal` / `cluster interval overlaps other claimed pairs or the window edge` / `residual not small relative to separation` | the Davis–Kahan condition (residual ≪ separation) fails for that cluster | sharper eigenpairs, or claim the neighbouring level too so that the separation is certified |
| S4 | `j/m clusters certified at projector level` with j < m | some clusters failed the above | per-cluster `reason` in `evidence.clusters` |
| S8 | `j/m clusters: eigenvalue count ... certified` with j < m | the bracketing inertia counts did not isolate the level | per-cluster records in `evidence.clusters`; often the same ill-posed multiplet as S3 |
| S7 | `thermodynamic intervals not formed: ...` | the interval arithmetic overflowed or a temperature was invalid | temperatures must be positive; very low T with a wide enclosure overflows e^(−E/T) |

## NOT_CHECKED reason codes

| code | the statement did not run because | next action |
|---|---|---|
| `NARROWED_BY_USER` | `--only` / `--skip` / a preset excluded it, or `Options(inertia=False)`, `eigenspaces="none"` | run with the default selection or a wider preset |
| `DEFAULT_SELECTION` | S5, S6, S7 run only when asked | `Options(corroborate=(...))`, `exact_recheck=True`, `temperatures=(...)`, or `--preset full` |
| `DEPENDENCY_NARROWED` | a statement it needs (S2 for S3, S3 for S4) was excluded | include the dependency |
| `NEEDS_EIGENVECTORS` | an eigenvalue-only claim | pass the eigenvectors |
| `NEEDS_SPEC` | no `ModelSpec` | declare the model ([quickstart](quickstart.md) §2) |
| `NEEDS_MATRIX` | matrix-free producer without a reference matrix | declare a spec (S1 by probes); S3/S8 then factorize the spec's matrix when it fits |
| `NEEDS_WHOLE_SPECTRUM` | S4 needs a certified complete window (S3); S7 needs `kind="all"` | make the completeness claim (`kind="lowest"` with S3 CERTIFIED, or `kind="all"`) |
| `NEEDS_TEMPERATURES`, `NEEDS_OBSERVABLES`, `NEEDS_DEPENDENCY` | an input or an optional package is missing | `Options.temperatures`, `Options.observables`; install python-flint / h5py / the corroborating solver |
| `NO_COMPLETENESS_CLAIM` | `kind="subset"` | choose `lowest`, `window` or `all` if that is what you claim |
| `NO_CLUSTERS` | no (quasi-)degenerate clusters among the claimed pairs | nothing to do; `eigenspaces="all"` treats every pair as a cluster |
| `SIZE_LIMIT` | dimension above `inertia_max_dim` / `exact_max_dim` | raise the limit if the factorization fits in memory |
| `BUDGET_TIME`, `BUDGET_MEMORY` | the budget was spent or the estimate exceeds it | raise the budget or narrow the scope deliberately |
| `UNSUPPORTED_CLAIM`, `UNSUPPORTED_MODEL`, `NOT_APPLICABLE` | not implemented for this claim kind / sector form / model | see the [scope matrix](SCOPE_MATRIX.md) |

`UNDERFLOW_RISK` is the one reason code attached to an INCONCLUSIVE statement rather than a NOT_CHECKED one.

## Examples

```python
"""What INCONCLUSIVE and FAILED look like: a duplicated vector, a missing eigenvalue, an ill-posed k, a tiny entry."""
import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.tfim_chain(8, 1.0, 0.8)
H = ce.build_bitwise(spec)
w, V = la.eigh(H.toarray())
print("lowest six:", np.array2string(w[:6], precision=6))

def show(tag, cert):
    for s in cert.statements[:3]:
        print(f"{tag:10s} {s['id']:18s} {s['status']:12s} {s['evidence'].get('reason_code') or ''}")
    s = cert.statements[1] if cert.statements[1]["status"] != "CERTIFIED" else cert.statements[2]
    print(f"{tag:10s} {s['id'].split('.')[0]} says: {s['summary'][:170]}")

# the same vector claimed twice: existence holds, distinctness does not
X = V[:, :3].copy(); X[:, 1] = X[:, 0]; mu = w[:3].copy(); mu[1] = mu[0]
show("duplicate", ce.certify(ce.Claim(mu, X, operator=H, kind="lowest"), spec))

# the third lowest eigenvalue left out of a 'lowest' claim (indices 0, 1, 3, 4): the inertia count catches it
idx = [0, 1, 3, 4]
show("missing", ce.certify(ce.Claim(w[idx], V[:, idx], operator=H, kind="lowest"), spec))

# 'the four lowest' when the fourth level is a doublet: the claim is complete but k is ill-posed
show("ill-posed", ce.certify(ce.Claim(w[:4], V[:, :4], operator=H, kind="lowest"), spec))

# a subnormal entry planted in an eigenvector: the no-underflow hypothesis is not established
X = V[:, :3].copy(); X[5, 2] = 2.0 ** -1060
show("underflow", ce.certify(ce.Claim(w[:3], X, operator=H, kind="lowest"), spec))
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
lowest six: [-9.365204 -9.317101 -7.756266 -7.490737 -7.490737 -6.533218]
duplicate  S1.model_fidelity  CERTIFIED    
duplicate  S2.eigenpairs      INCONCLUSIVE 
duplicate  S3.completeness    INCONCLUSIVE 
duplicate  S2 says: each claimed value has an eigenvalue within its radius, but some claimed pairs could not be matched to distinct eigenvalues (duplicate or non-orthogonal vectors)
missing    S1.model_fidelity  CERTIFIED    
missing    S2.eigenpairs      CERTIFIED    
missing    S3.completeness    FAILED       
missing    S3 says: at least 1 eigenvalue(s) below the highest claimed value are missing from the claim
ill-posed  S1.model_fidelity  CERTIFIED    
ill-posed  S2.eigenpairs      CERTIFIED    
ill-posed  S3.completeness    INCONCLUSIVE 
ill-posed  S3 says: the claim is complete below -7.490736746, but the highest claimed level is a (quasi-)degenerate multiplet of at least 2 eigenvalues within [-7.490736746, -7.490736559], o
underflow  S1.model_fidelity  CERTIFIED    
underflow  S2.eigenpairs      INCONCLUSIVE UNDERFLOW_RISK
underflow  S3.completeness    INCONCLUSIVE UNDERFLOW_RISK
underflow  S2 says: no-underflow hypothesis not established: nonzero magnitudes below 2^-900 in claim.eigenvectors (min 8.09e-320, 1 entry); the floating-point bounds assume operands and rad
```

The second case is FAILED, not INCONCLUSIVE: the inertia count found an eigenvalue below the highest claimed
value that the claim does not contain. That is a finding about the claim.
