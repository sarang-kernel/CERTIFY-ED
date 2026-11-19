# Files: `.npz`, HDF5 and the command line

A producer in any language writes arrays to files; CERTIFY-ED reads `.npy .npz .h5 .csv .tsv .txt .dat
.json .mtx` ([the producer bundle](../producer_bundle.md) lists every format). The spec travels as
`ModelSpec.to_dict()` in JSON.

```python
"""Files: write eigenvalues, eigenvectors and the matrix to .npz (and HDF5), certify from the command line."""
import subprocess
import sys

import scipy.linalg as la

import certify_ed as ce

spec = ce.heisenberg_chain(6, 1.0)
H = ce.build_bitwise(spec)
w, V = la.eigh(H.toarray())

# one .npz with eigenvalues, eigenvectors and the CSR triplet of H (export_npz is the helper for producers)
ce.export_npz("chain6.npz", w[:4], V[:, :4], H)
with open("chain6_spec.json", "w") as f:
    import json
    json.dump(spec.to_dict(), f)
cmd = [sys.executable, "-m", "certify_ed.cli", "certify", "chain6.npz", "--spec", "chain6_spec.json",
       "--kind", "lowest", "-o", "chain6.certificate"]
print("$ certify-ed certify chain6.npz --spec chain6_spec.json --kind lowest -o chain6.certificate")
print(subprocess.run(cmd, capture_output=True, text=True).stdout)

# the same claim as a producer bundle directory in HDF5 (needs h5py), certified with `certify-ed import`
ce.write_bundle("chain6_h5", w[:4], V[:, :4], operator=H, spec=spec, fmt="h5", kind="lowest")
import os
print(sorted(os.listdir("chain6_h5")))
print("$ certify-ed import chain6_h5")
print(subprocess.run([sys.executable, "-m", "certify_ed.cli", "import", "chain6_h5"], capture_output=True, text=True).stdout)
```

Output (certify-ed 3.2.0.dev0, Python 3.11, numpy 2.4.4, scipy 1.17.1; Windows):

```text
$ certify-ed certify chain6.npz --spec chain6_spec.json --kind lowest -o chain6.certificate
S1.model_fidelity      CERTIFIED     producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 7.8e-14)
S2.eigenpairs          CERTIFIED     4 claimed values matched to distinct eigenvalues; max radius 3.47e-14 (+3.9e-14 model shift)
S3.completeness        CERTIFIED     exactly 4 eigenvalues below -1.561803399; certified gap to the next >= 5.562e-01
S4.eigenspaces         CERTIFIED     1/1 clusters certified at projector level
S8.degeneracy          CERTIFIED     1/1 clusters: eigenvalue count in the level enclosure certified by bracketing inertia counts; degeneracies [3]
scope: preset -; narrowed: S5 (default), S6 (default), S7 (default)
certificate: chain6.certificate.json

['certificate.arrays.npz', 'certificate.json', 'eigenvalues.h5', 'eigenvectors.h5', 'meta.json', 'operator.h5', 'spec.json']
$ certify-ed import chain6_h5
S1.model_fidelity      CERTIFIED     producer matrix equals the spec matrix entrywise (max |diff| 0.0e+00 <= 7.8e-14)
S2.eigenpairs          CERTIFIED     4 claimed values matched to distinct eigenvalues; max radius 3.47e-14 (+3.9e-14 model shift)
S3.completeness        CERTIFIED     exactly 4 eigenvalues below -1.561803399; certified gap to the next >= 5.562e-01
S4.eigenspaces         CERTIFIED     1/1 clusters certified at projector level
S8.degeneracy          CERTIFIED     1/1 clusters: eigenvalue count in the level enclosure certified by bracketing inertia counts; degeneracies [3]
scope: preset -; narrowed: S5 (default), S6 (default), S7 (default)
certificate: chain6_h5\certificate.json
```

- `certify-ed certify FILE` certifies a single file (eigenvalues, eigenvectors and the CSR triplet of `H`
  in one `.npz`, as `export_npz` writes it); `--eigenvectors`, `--operator`, `--basis` name separate files.
  `--kind lowest` says what completeness is claimed; the default is `subset` (accuracy only).
- `certify-ed import DIR` (same as `certify --from DIR`) reads a producer bundle directory, here written in
  HDF5 by `write_bundle` (`h5py` needed). The bundle's `spec.json` and `meta.json` supply the spec and the
  kind, so the command line needs nothing else; the certificate goes to `DIR/certificate.json`.
- The exit code is 0 unless a statement is FAILED.
