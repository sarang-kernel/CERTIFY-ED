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
