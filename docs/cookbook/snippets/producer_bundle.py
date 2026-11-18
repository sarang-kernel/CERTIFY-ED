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
