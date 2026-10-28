# The producer bundle

A directory with `eigenvalues.<ext>`, optionally `eigenvectors.<ext>`, `operator.<ext>`, `basis.<ext>`,
`spec.json` and `meta.json`. Any code that writes files can produce one; `write_bundle` is the Python
helper and the format reference is [producer_bundle.md](../producer_bundle.md).

```python
"""The producer bundle: a directory any code can write; certify it from Python or the command line."""
import os
import subprocess
import sys

import scipy.linalg as la

import certify_ed as ce

spec = ce.spinless_fermion_chain(6, 1.0, 0.5)          # spinless fermions, L = 6
H = ce.build_bitwise(spec)
w, V = la.eigh(H.toarray())

path = ce.write_bundle("fermions6", w[:3], V[:, :3], operator=H, spec=spec, fmt="csv", kind="lowest",
                       meta={"description": "spinless fermions L=6, three lowest", "preset": "lab-minimal",
                             "producer": {"name": "my_code", "version": "0.1"}})
print(sorted(os.listdir(path)))
print(open(os.path.join(path, "meta.json")).read())

claim, bspec, meta = ce.claim_from_bundle(path)
cert = ce.certify(claim, bspec, ce.Options.from_preset(meta["preset"]))     # the bundle's preset: S1 + S2 only
print(cert.summary())

print("$ certify-ed import fermions6 --preset full")
print(subprocess.run([sys.executable, "-m", "certify_ed.cli", "import", path, "--preset", "full"],
                     capture_output=True, text=True).stdout)
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
['eigenvalues.csv', 'eigenvectors.csv', 'meta.json', 'operator.csv', 'spec.json']
{
 "schema": "certify-ed/producer-bundle/1.0",
 "kind": "lowest",
 "description": "spinless fermions L=6, three lowest",
 "preset": "lab-minimal",
 "producer": {
  "name": "my_code",
  "version": "0.1"
 }
}
CERTIFY-ED certificate (certify-ed/certificate/2.1)
  subject : spinless_fermion_chain
  claim   : spinless fermions L=6, three lowest
  [   CERTIFIED] S1.model_fidelity: producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 9.6e-14)
  [   CERTIFIED] S2.eigenpairs: 3 claimed values matched to distinct eigenvalues; max radius 2.04e-14 (+3.7e-14 model shift)
  [ NOT_CHECKED] S3.completeness: not selected (preset: not in preset lab-minimal)
  [ NOT_CHECKED] S4.eigenspaces: not selected (preset: not in preset lab-minimal)
  scope   : preset lab-minimal; in scope S1, S2
  narrowed: S3 (preset: not in preset lab-minimal); S4 (preset: not in preset lab-minimal); S5 (preset: not in preset lab-minimal); S6 (preset: not in preset lab-minimal); S7 (preset: not in preset lab-minimal); S8 (preset: not in preset lab-minimal)
  out of scope: 6 quantities with reason codes (see scope.out_of_scope, docs/SCOPE_MATRIX.md)
  not verified:
    - that the model specification is the Hamiltonian the authors intend (only producer operator == spec is checked)
    - correctness of the producer's code in general (only this result is checked)
    - interval-arithmetic proof: bounds rely on the standard floating-point model (see 'arithmetic')
$ certify-ed import fermions6 --preset full
S1.model_fidelity      CERTIFIED     producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 9.6e-14)
S2.eigenpairs          CERTIFIED     3 claimed values matched to distinct eigenvalues; max radius 2.04e-14 (+3.7e-14 model shift)
S3.completeness        CERTIFIED     exactly 3 eigenvalues below -2.249052566; certified gap to the next >= 4.170e-02
S4.eigenspaces         CERTIFIED     3/3 clusters certified at projector level
S5.corroboration       CORROBORATED  2 recomputation(s) agree with the claim to 1.2e-14
S6.exact_recheck       CERTIFIED     3 lowest eigenvalues re-derived in exact arithmetic: 3 within their residual radii (max 1.57e-14); next eigenvalue >= -2.267585416; no floating-point assumption
S7.thermodynamics      NOT_CHECKED   needs a whole-spectrum claim (kind 'all') whose enclosure is complete (S3)
S8.degeneracy          CERTIFIED     3/3 clusters: eigenvalue count in the level enclosure certified by bracketing inertia counts; degeneracies [1, 1, 1]
scope: preset full; narrowed: none
certificate: fermions6\certificate.json
```

- `meta.json` carries the kind of claim, a description, the producer record and, optionally, the scope the
  producer suggests (`preset`, `only`, `skip`, budgets). The command line uses those when it is given
  nothing else; `--preset full` on the command line overrides the bundle's `lab-minimal`.
- The `lab-minimal` certificate lists S3–S8 as *narrowed* with the reason, so a reader knows they were not
  attempted rather than failed.
- `full` adds S5 (two independent solvers, evidence rather than a certified statement), S6 (the exact
  integer re-check, no floating-point assumption; needs python-flint) and would add S7 for a whole-spectrum
  claim. The CSV operator was read back as a COO table; the certificate records which files were used.
