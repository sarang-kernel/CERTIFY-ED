"""The two applications on their quick instances: AKLT star lattice and the spin-1 pyrochlore check.
(The S = 1/2 crown certificate takes about 13 minutes: certify_ed.apps.pyrochlore.spin_half.certify_crown.)"""
from certify_ed.apps.aklt import certify_lattice
from certify_ed.apps.pyrochlore.spin_one import printed_coupling_check

r = certify_lattice("star", log=lambda s: None)
print(f"AKLT star: eta <= {r['eta_upper']:.10f} (published {r['eta_published']}); eta < 1/4: {r['criterion_holds']}")
p = printed_coupling_check()
print(f"pyrochlore S=1 at the printed couplings: an eigenvalue <= {p['eigenvalue_upper']:.7f} "
      f"< published bound {p['published_bound']}: {p['contradiction']}")
