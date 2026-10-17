import os, sys, json, time
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lattices import square_octagon
from pw_general import certify_eta_blocks as certify_eta
res = {}
targets = {"par": Fraction(11267, 1000000), "perp": Fraction(25271, 1000000)}   # just above 0.10614^2, 0.15897^2
for k, t in targets.items():
    g, L, C, R = square_octagon(k)
    print("square-octagon", k, flush=True)
    res[k] = certify_eta(g, L, C, R, [t])
ep = res["par"]["thresholds"][str(targets["par"])]["t_prime"] ** 0.5
eq = res["perp"]["thresholds"][str(targets["perp"])]["t_prime"] ** 0.5
res["criterion"] = {"eta_par_upper": ep, "eta_perp_upper": eq, "4 eta_perp + 2 eta_par": 4 * eq + 2 * ep,
                    "holds": 4 * eq + 2 * ep < 1}
print(res["criterion"])
json.dump(res, open(os.path.join(HERE, "square_octagon_certificate.json"), "w"), indent=1)
