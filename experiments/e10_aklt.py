"""E10: Pomata-Wei spectral-gap criteria for 2D AKLT models (arXiv:1911.01410), certified in exact
arithmetic (certify_ed.apps.aklt), with the kernel lemma checked exactly where feasible.

quick: star, square-octagon (both overlaps), cross, decorated honeycomb; kernel check for the star.
full:  + decorated square (about 45 min); kernel checks for square-octagon and decorated honeycomb.

usage: python3 e10_aklt.py [quick|full]
"""
import sys

from harness import Run
from certify_ed.apps.aklt import certify_lattice, square_octagon_criterion, lattices
from certify_ed.apps.aklt.kernel import kernel_check

QUICK = ["star", "square_octagon_par", "square_octagon_perp", "cross", "decorated_honeycomb"]
FULL = QUICK + ["decorated_square"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    names = FULL if mode == "full" else QUICK
    with Run("e10_aklt") as run:
        run.record("mode", mode)
        recs = {}
        for name in names:
            r = certify_lattice(name, log=lambda s: run.log(s) if not s.startswith("    block") else None)
            recs[name] = r
            run.log(f"{name}: eta <= {r['eta_upper']:.10f} (published {r['eta_published']:.10f}); "
                    f"criterion holds: {r['criterion_holds']}")
            run.record("lattices", recs)
        if "square_octagon_par" in recs and "square_octagon_perp" in recs:
            so = square_octagon_criterion(recs["square_octagon_par"], recs["square_octagon_perp"])
            run.log(f"square-octagon: 4 eta_perp + 2 eta_par <= {so['sum_upper']:.6f}: {so['criterion_holds']}")
            run.record("square_octagon", so)
        kernels = {}
        todo = [("star", lattices.star, ["LC"])]
        if mode == "full":
            todo += [("square_octagon", lambda: lattices.square_octagon("par"), ["LC"]),
                     ("decorated_honeycomb", lattices.decorated_honeycomb, ["LC", "CR"])]
        for name, build, parts in todo:
            g, L, C, R = build()
            kernels[name] = [kernel_check(g, L + C if p == "LC" else C + R, log=run.log) for p in parts]
            run.record("kernel", kernels)


if __name__ == "__main__":
    main()
