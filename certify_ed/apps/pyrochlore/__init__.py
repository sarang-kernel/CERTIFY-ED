"""Certified Anderson-type lower bounds for the pyrochlore Heisenberg antiferromagnet from the 18-site
"crown" cluster of Kranitz & Penc (arXiv:2601.07800), reduced model with c5 = 0.

* spin_half.certify_crown: S = 1/2, e_GS >= lambda_min(H_crown) certified in exact arithmetic
  (S^z = 0 block, which contains every SU(2) multiplet; Z2^7 symmetry blocks; exact inertia).
* spin_one.printed_coupling_check: S = 1, at the printed couplings the all-singlet pair-spin sector has
  an eigenvalue below the published bound (six-ring energy enclosed by CERTIFY-ED).
* spin_one.landscape: floating-point lowest energy of each of the 92 pair-spin classes (estimates).
"""
from .model import PAIRS, HEX, TET_HEX, C3_SITES, bonds, spec_for, symmetry_generators

__all__ = ["PAIRS", "HEX", "TET_HEX", "C3_SITES", "bonds", "spec_for", "symmetry_generators"]
