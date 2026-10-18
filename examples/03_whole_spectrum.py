"""Enclose every eigenvalue of a Heisenberg chain, sector by sector (the former symveig), and use the
symveig 1.0.0 API on the same engine."""
import numpy as np

import certify_ed as ce
from certify_ed.compat.symveig import enclose_sectors, summary_stats, heisenberg_1d

spec = ce.heisenberg_chain(10)
H = ce.build_bitwise(spec)
Sz = np.array([bin(s).count("1") for s in range(H.shape[0])], dtype=float)
enc = ce.enclose_spectrum(H, Sz)
print(f"{len(enc.intervals)} intervals for {enc.dimension} eigenvalues; complete: {enc.complete}; "
      f"max half-width {enc.max_halfwidth:.2e}; sectors {enc.sector_dims}")
print("lowest:", enc.intervals[0])

Hs, sec = heisenberg_1d(10)                            # symveig-style dense builder and S_z diagonal
print(summary_stats(enclose_sectors(Hs, sec["Sz"])))
