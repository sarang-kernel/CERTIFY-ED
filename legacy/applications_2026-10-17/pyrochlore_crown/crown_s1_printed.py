"""Spin-1 crown at the printed couplings (c1 = 0.053466, c3 = c12 = 0.082726, c5 = 0, embedding
constraint fixes c17): the all-singlet pair-spin configuration has the exact eigenvalue
-12 c1 + c17 E0(spin-1 Heisenberg ring of 6). E0 is enclosed rigorously by CERTIFY-ED (S2 + S3), so the
cluster's lowest eigenvalue is at most the value below, which lies under the published bound -1.632985.
Exact rational arithmetic for the couplings; the enclosure radius includes the model shift."""
import json, os, sys
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np, scipy.linalg as la
import certify_ed as ce

c1, c3 = F("0.053466"), F("0.082726")
c17 = (1 - 2 * c1 - 8 * c3) / 2
spec = ce.spin1_heisenberg_chain(6)          # periodic ring, J = 1
H = ce.build_bitwise(spec)
w, V = la.eigh(H.toarray())
cert = ce.certify(ce.Claim(w[:1], V[:, :1], operator=H, kind="lowest"), spec)
st = {s["id"]: s for s in cert.statements}
assert st["S2.eigenpairs"]["status"] == "CERTIFIED" and st["S3.completeness"]["status"] == "CERTIFIED"
ev = st["S2.eigenpairs"]["evidence"]
rad = F(ev["max_radius"]) + F(ev["model_shift"])
E0_hi = F(float(w[0])) + rad
val_hi = -12 * c1 + c17 * (F(float(w[0])) - rad) if c17 < 0 else -12 * c1 + c17 * E0_hi
published = F("-1.632985")
rec = {"c1": str(c1), "c3": str(c3), "c17": str(c17), "E0_ring": float(w[0]), "E0_radius": float(rad),
       "eigenvalue_upper": float(val_hi), "published_bound": float(published),
       "contradiction": bool(val_hi < published), "margin": float(published - val_hi)}
json.dump(rec, open(os.path.join(HERE, "crown_s1_printed.json"), "w"), indent=1)
print(rec)
