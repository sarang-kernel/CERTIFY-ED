"""E8: symveig 1.0.0 (Vehale & Goel, arXiv:2606.16217) against certify_ed.spectrum on symveig's own
benchmark configurations (same matrices: symveig's model builders), sector path.

Per configuration: max half-width of symveig's sector enclosures and of certify_ed.spectrum's, whether
each is complete with certified counts, wall-clock times, and the largest distance from a floating-point
reference eigenvalue (independent full eigvalsh) to the nearest enclosure (> 0 means the reference lies
outside; for tight enclosures this measures the reference's own error, not a failure of the enclosure).

usage: python3 e8_symveig_comparison.py [SYMVEIG_SRC]   (default: ../../../symveig/code/symveig-1.0.0)
"""
import os
import sys
import time

import numpy as np
import scipy.linalg as la

from harness import Run

SRC = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                      os.path.join(os.path.dirname(__file__), "..", "..", "..", "symveig", "code", "symveig-1.0.0"))
sys.path.insert(0, SRC)
from symveig import enclose_sectors                                      # noqa: E402
from symveig.models import heisenberg_1d, j1j2_heisenberg_1d, heisenberg_2d  # noqa: E402
from certify_ed.spectrum import enclose_spectrum                          # noqa: E402

CONFIGS = ([("1D Heisenberg OBC", L, lambda L=L: heisenberg_1d(L, periodic=False)) for L in (4, 6, 8, 10, 12)]
           + [("1D Heisenberg PBC", L, lambda L=L: heisenberg_1d(L)) for L in (4, 6, 8, 10, 12)]
           + [("1D J1-J2 PBC", L, lambda L=L: j1j2_heisenberg_1d(L)) for L in (4, 6, 8, 10, 12)]
           + [("2D Heisenberg 3x3 PBC", 9, lambda: heisenberg_2d(3, 3)),
              ("2D Heisenberg 4x3 PBC", 12, lambda: heisenberg_2d(4, 3))])


def outside(intervals, w):
    return max(max(min(max(lo - x, x - hi, 0.0) for lo, hi in intervals) for x in w), 0.0)


def main():
    with Run("e8_symveig_comparison") as run:
        run.log(f"symveig source: {SRC}")
        rows = []
        for name, L, build in CONFIGS:
            H, sec = build()
            S = sec["Sz"]
            t0 = time.time(); sv = enclose_sectors(H, S); t_sv = time.time() - t0
            t0 = time.time(); ce_ = enclose_spectrum(H, S); t_ce = time.time() - t0
            w = la.eigvalsh(H)
            row = {"model": name, "L": L, "dim": int(H.shape[0]),
                   "symveig_max_halfwidth": max(e.halfwidth for e in sv),
                   "certify_max_halfwidth": ce_.max_halfwidth,
                   "ratio": max(e.halfwidth for e in sv) / ce_.max_halfwidth,
                   "symveig_median_halfwidth": float(np.median([e.halfwidth for e in sv])),
                   "certify_median_halfwidth": float(np.median([iv.halfwidth for iv in ce_.intervals])),
                   "symveig_counted_claims": sum(e.multiplicity for e in sv if e.multiplicity > 1),
                   "certify_cluster_methods": sorted({iv.method for iv in ce_.intervals}),
                   "symveig_seconds": t_sv, "certify_seconds": t_ce,
                   "certify_complete": ce_.complete,
                   "symveig_reference_outside": outside([(e.lo, e.hi) for e in sv], w),
                   "certify_reference_outside": outside([(iv.lo, iv.hi) for iv in ce_.intervals], w)}
            rows.append(row)
            run.log(f"{name} L={L}: symveig {row['symveig_max_halfwidth']:.2e} ({t_sv:.2f}s), certify_ed "
                    f"{row['certify_max_halfwidth']:.2e} ({t_ce:.2f}s), ratio {row['ratio']:.1f}; medians "
                    f"{row['symveig_median_halfwidth']:.1e} / {row['certify_median_halfwidth']:.1e}; complete "
                    f"{ce_.complete}, reference outside by {row['certify_reference_outside']:.1e}")
            run.record("E8", rows)


if __name__ == "__main__":
    main()
