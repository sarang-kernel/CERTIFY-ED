"""Star lattice through the generic V-coordinate certifier (cross-check of star_certify.py)."""
import os, sys, json
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from pw_general import Graph, certify_eta
TRI = lambda a: [(a, a + 1), (a + 1, a + 2), (a, a + 2)]
g = Graph([3] * 9, TRI(0) + TRI(3) + TRI(6) + [(2, 3), (4, 6)])
out = certify_eta(g, [0, 1, 2], [3, 4, 5], [6, 7, 8], [Fraction(1, 16), Fraction(12331, 1000000)])
json.dump(out, open(os.path.join(HERE, "star_v.json"), "w"), indent=1)
