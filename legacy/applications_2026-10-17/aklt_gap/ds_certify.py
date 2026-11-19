"""Certified eta for the decorated lattices (Pomata & Wei, arXiv:1911.01410, Method II):
decorated honeycomb eta = 0.1530329085 < 1/4, decorated square eta = 0.2203543174 < 1/4.
Exact bordered-inertia certificate in reduced VBS coordinates (pw_symm), with the mirror symmetry of
the decorated-square union and the spin flip. Kernel equality ker H_Gamma = VBS range: lemma in
docs_v2/notes/aklt_kernel_lemma.md (plus exact checks in kernel_modp.py for the honeycomb Gammas)."""
import json, os, sys
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lattices import decorated_square, decorated_square_reflection, decorated_honeycomb
from pw_symm import certify_eta_sym
log = lambda s: print(s, flush=True)
path = os.path.join(HERE, "decorated_certificate.json")
res = json.load(open(path)) if os.path.exists(path) else {}
which = sys.argv[1:] or ["honeycomb", "square"]
if "honeycomb" in which:
    g, L, C, R = decorated_honeycomb()
    log("decorated honeycomb: criterion t = 1/16, sharp t = 0.02342, negative control t = 0.0234")
    res["decorated_honeycomb"] = certify_eta_sym(g, L, C, R, [Fraction(1, 16), Fraction(2342, 100000), Fraction(234, 10000)], log=log)
    json.dump(res, open(path, "w"), indent=1)
if "square" in which:
    g, L, C, R = decorated_square()
    log("decorated square: criterion t = 1/16, sharp t = 0.04856")
    res["decorated_square"] = certify_eta_sym(g, L, C, R, [Fraction(1, 16), Fraction(4856, 100000)],
                                              sigmas=[decorated_square_reflection(g)], log=log)
    json.dump(res, open(path, "w"), indent=1)
