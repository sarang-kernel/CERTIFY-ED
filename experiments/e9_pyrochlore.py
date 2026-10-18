"""E9: pyrochlore crown cluster (Kranitz & Penc, arXiv:2601.07800), reduced model c5 = 0.

  S1/2  certified lower bound on lambda_min(H_crown) at the published couplings, exact arithmetic
        (certify_ed.apps.pyrochlore.spin_half.certify_crown). [full]
  S1P   S = 1: rigorous contradiction at the printed couplings (all-singlet sector).
  S1L   S = 1: floating-point landscape of the 92 pair-spin classes at the re-optimised couplings
        (quick: classes whose full space has at most 729*125 states; full: all, including the
        1.1M-state S^z = 0 block of the all-S~=2 class).

usage: python3 e9_pyrochlore.py [quick|full]
"""
import sys

from harness import Run
from certify_ed.apps.pyrochlore.spin_half import certify_crown
from certify_ed.apps.pyrochlore.spin_one import printed_coupling_check, landscape

OPTIMUM = {"c1": 0.053975140682882464, "c3": 0.08288267138128608}   # crown_s1_opt3 (Nelder-Mead, 2026-10-17)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    with Run("e9_pyrochlore") as run:
        run.record("mode", mode)
        p = printed_coupling_check()
        run.log(f"S=1 printed couplings: eigenvalue <= {p['eigenvalue_upper']:.10f} vs published "
                f"{p['published_bound']}: contradiction {p['contradiction']} (margin {p['margin']:.3e})")
        run.record("spin1_printed", p)
        if mode == "full":
            r = certify_crown(log=run.log)
            run.record("spin_half", r)
        L = landscape(2, OPTIMUM["c1"], OPTIMUM["c3"]) if mode == "full" else None
        if L is None:
            import itertools
            from certify_ed.apps.pyrochlore.spin_one import canonical
            import numpy as np
            small = {canonical(cfg) for cfg in itertools.product((0, 2, 4), repeat=6)
                     if int(np.prod([ts + 1 for ts in cfg])) <= 125}
            L = landscape(2, OPTIMUM["c1"], OPTIMUM["c3"], classes=small)
        run.log(f"S=1 landscape at c1={OPTIMUM['c1']:.7f}, c3={OPTIMUM['c3']:.7f}: lowest class "
                f"{L[0][1]} at {L[0][0]:.8f} ({len(L)} classes evaluated, floating point)")
        run.record("spin1_landscape", {"couplings": OPTIMUM, "classes": [
            {"energy": e, "pair_spins2": cfg, "dim": d, "multiplicity": m} for e, cfg, d, m in L]})


if __name__ == "__main__":
    main()
