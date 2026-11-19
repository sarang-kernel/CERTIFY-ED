import json, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from crown_certify import exact_check
rec = json.load(open("crown_381414_817957.json"))
out = exact_check(rec["n1"], rec["n3"], rec["shift_K_units"])
print("exact-check lower bound:", out["lower_bound_exact_check"], "seconds", out["seconds"], flush=True)
json.dump(out, open("crown_381414_817957_exact.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
