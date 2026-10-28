# Quickstart: five minutes to a certificate

A certificate is a JSON file that says, statement by statement, what has been established about a claimed
exact-diagonalization result: that the producer's matrix is the declared model (S1), that each claimed
eigenvalue lies within a stated radius of a distinct eigenvalue (S2), that no eigenvalue is missing below
the claim and how large the gap to the next one is (S3), and more. Every statement is CERTIFIED, FAILED,
INCONCLUSIVE or NOT_CHECKED, with its evidence. [What certified means](what_certified_means.md) explains
the statuses; this page gets you to a first certificate.

## 1. Install

```text
pip install certify-ed              # numpy and scipy
pip install "certify-ed[all]"       # + python-flint (exact layer), quspin, h5py, matplotlib
```

From a checkout: `pip install -e .` in the repository root. The version used on this page is
`3.2.0.dev0`; `python -c "import certify_ed; print(certify_ed.__version__)"` prints yours.

## 2. One call

Three inputs: the model you intend (a `ModelSpec`, written independently of the solver), the producer's
matrix and the eigenpairs the solver returned. The example uses CERTIFY-ED's own builder for the matrix and
LAPACK (`scipy.linalg.eigh`) as the producer; the [cookbook](cookbook/index.md) replaces each with the
producer you have.

```python
"""Five minutes: a Heisenberg chain, scipy.linalg.eigh, one call, the certificate."""
import scipy.linalg as la

import certify_ed as ce

spec = ce.heisenberg_chain(8, 1.0)                     # the model you intend (declared independently of the solver)
H = ce.build_bitwise(spec)                             # your matrix; here CERTIFY-ED's own builder stands in
w, V = la.eigh(H.toarray())                            # the producer: any solver, here LAPACK

claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
cert = ce.certify(claim, spec)                        # Options() by default: S1-S4 and S8
print(cert.summary())
path = cert.save("quickstart_certificate.json")       # certificate.json + certificate.arrays.npz
print(path)
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
CERTIFY-ED certificate (certify-ed/certificate/2.1)
  subject : heisenberg_chain
  claim   : lowest claim, 4 eigenvalues
  [   CERTIFIED] S1.model_fidelity: producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 1.7e-13)
  [   CERTIFIED] S2.eigenpairs: 4 claimed values matched to distinct eigenvalues; max radius 4.11e-14 (+6.8e-14 model shift)
  [   CERTIFIED] S3.completeness: exactly 4 eigenvalues below -2.74250724; certified gap to the next >= 3.859e-01
  [   CERTIFIED] S4.eigenspaces: 1/1 clusters certified at projector level
  [   CERTIFIED] S8.degeneracy: 1/1 clusters: eigenvalue count in the level enclosure certified by bracketing inertia counts; degeneracies [3]
  scope   : preset -; in scope S1, S2, S3, S4, S8
  narrowed: S5 (default: not part of the default selection); S6 (default: not part of the default selection); S7 (default: not part of the default selection)
  out of scope: 6 quantities with reason codes (see scope.out_of_scope, docs/SCOPE_MATRIX.md)
  not verified:
    - that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)
    - correctness of the producer's code in general (only this result is checked)
    - interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')
    - anything outside the certified window [-inf, -2.7425072398402923]
    - eigenvector subspaces of isolated (non-cluster) pairs beyond their residuals
quickstart_certificate.json
```

Every line of the summary is a statement with a status. `(+6.8e-14 model shift)` is the bound on the
distance between the producer's matrix and the declared model that S1 established; the radius that refers
to the declared model is the sum. `narrowed: S5 ... S6 ... S7` says that corroboration, the exact re-check
and thermodynamics did not run because nothing asked for them; `not verified` lists what no statement
claims.

## 3. Read the certificate

`cert.save()` wrote `quickstart_certificate.json` (the statements, the scope block, the environment) and
`quickstart_certificate.arrays.npz` (eigenvalues, radii, residual bounds). `Certificate.load` verifies the
body hash and the scope block before handing it back.

```python
"""Read a saved certificate: statuses, radii, the scope block."""
import numpy as np

import certify_ed as ce

cert = ce.Certificate.load("quickstart_certificate.json")        # verifies the body hash and the scope block
for s in cert.statements:
    print(f"{s['id']:20s} {s['status']:12s} reason_code={s['evidence'].get('reason_code')}")
r = cert.arrays["radius"]
print("eigenvalues  ", np.array2string(cert.arrays["eigenvalues"], precision=10))
print("radii        ", np.array2string(r, precision=2))
print("max radius   ", f"{r.max():.3e}")
s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]
print("gap >=       ", f"{s3['evidence']['gap_lower_bound']:.6f}")
print("version      ", cert.body["environment"]["certify_ed"])
print("scope in     ", [x["id"] for x in cert.body["scope"]["in_scope"]])
print("underflow    ", cert.body["scope"]["options"]["underflow_check"])
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
S1.model_fidelity    CERTIFIED    reason_code=None
S2.eigenpairs        CERTIFIED    reason_code=None
S3.completeness      CERTIFIED    reason_code=None
S4.eigenspaces       CERTIFIED    reason_code=None
S8.degeneracy        CERTIFIED    reason_code=None
eigenvalues   [-3.6510934089 -3.1284190638 -3.1284190638 -3.1284190638]
radii         [8.08e-14 1.09e-13 1.09e-13 1.09e-13]
max radius    1.091e-13
gap >=        0.385912
version       3.2.0.dev0
scope in      ['S1.model_fidelity', 'S2.eigenpairs', 'S3.completeness', 'S4.eigenspaces', 'S8.degeneracy']
underflow     True
```

The radii are per claimed eigenvalue and include the model shift (`arrays["radius"]`, rounded upward); the
claimed values themselves are the producer's. `underflow True` records that the no-underflow hypothesis of
the floating-point bounds was checked on this input (see [What certified means](what_certified_means.md)).

## 4. Write the sentence

What the certificate above supports, in the register of a methods section:

> The four lowest eigenvalues of the eight-site Heisenberg chain (J = 1, periodic) obtained with LAPACK
> `eigh` were certified with CERTIFY-ED 3.2.0.dev0: the matrix equals the declared model entrywise; each
> claimed eigenvalue lies within 1.1 × 10⁻¹³ of a distinct eigenvalue of the model; exactly four eigenvalues
> lie below −2.742508, with a gap of at least 0.3859 to the fifth (S1–S3 CERTIFIED, inertia counts; bounds
> under the IEEE 754 standard model, checked at run time). Certificate: `quickstart_certificate.json`.

Take the numbers from the certificate, not from this page: `arrays["radius"].max()` for the radius,
`S3.completeness.evidence.gap_lower_bound` for the gap, `environment.certify_ed` for the version. A
statement that is INCONCLUSIVE or NOT_CHECKED is not mentioned as established; the
[FAQ on INCONCLUSIVE](faq_inconclusive.md) says what to do about each reason code.

## Next

- [Cookbook](cookbook/index.md): one page per producer path (arrays, `eigsh`, files, QuSpin, matrix-free,
  producer bundle).
- [What certified means](what_certified_means.md), [Limitations](limitations.md),
  [FAQ: INCONCLUSIVE](faq_inconclusive.md).
- [Presets and scope](presets.md): `Options.from_preset("lab-minimal")`, `--only`, `--skip`, budgets.
