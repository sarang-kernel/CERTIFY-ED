import os, sys, json
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lattices import cross
from pw_general import certify_eta_blocks
g, L, C, R = cross()
out = certify_eta_blocks(g, L, C, R, [Fraction(1, 9), Fraction(39896, 1000000)])  # eta < 1/3; just above 0.19974^2
out["criterion"] = {"eta_upper": min(v["t_prime"] for v in out["thresholds"].values() if v["certified"]) ** 0.5}
out["criterion"]["holds"] = out["criterion"]["eta_upper"] < 1 / 3
print(out["criterion"])
json.dump(out, open(os.path.join(HERE, "cross_certificate.json"), "w"), indent=1)
