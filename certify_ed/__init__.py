"""CERTIFY-ED: certificates for exact-diagonalization results.

A claimed spectrum (eigenvalues, optionally eigenvectors, from any producer) is checked against a
declared model and receives statements S1-S5 (model fidelity, enclosures, completeness, eigenspaces,
corroboration), each CERTIFIED / FAILED / INCONCLUSIVE / NOT_CHECKED with its evidence. Bounds are
rigorous under the IEEE 754 standard model, which is checked at run time; the exact layer
(`exact_recheck_lowest`, `exact_recheck_sector_lowest`) re-derives enclosures and counts in integer
arithmetic with no floating-point assumption. `enclose_spectrum` encloses a whole spectrum, sector by
sector (the former symveig package). Applications to published proofs live in `certify_ed.apps`.
"""

__version__ = "3.2.0.dev0"
__author__ = "Sarang Vehale, Ritu Goel"

from .localops import LocalSpace
from .spec import (ModelSpec, Term, spec_from_quspin, heisenberg_chain, tfim_chain,
                   heisenberg_square, spinless_fermion_chain, bose_hubbard_chain,
                   spin1_heisenberg_chain)
from .construct import (build_bitwise, build_kron, cross_construct, SpecOperator,
                        StateSet, full_state_set)
from .basis import (u1_state_set, magnetization_state_set, translation, spin_flip,
                    reflection, cyclic_isometry, sector_matrix)
from .enclosure import OperatorAdapter, enclose, kato_temple
from .inertia import inertia as inertia_count
from .certify import Claim, Options, certify
from .certificate_v2 import Certificate
from .api import quick
from .report import methods_paragraph
from .cost import estimate_seconds, auto_budget, plan_for_claim
from .operators import parse_operator
from .importers import (claim_from_quspin, claim_from_arrays, claim_from_npz,
                        claim_from_hdf5, export_npz, claim_from_file, claim_from_bundle)
from .io import read_bundle, write_bundle
from .fp import probe_float_environment
from .spectrum import enclose_spectrum, SpectrumEnclosure, SpectrumInterval
from .checker import exact_matrix, exact_eigenpair_radii, exact_recheck_lowest
from .exact_sectors import exact_recheck_sector_lowest, site_permutation

# backwards-compatible name used by v2 scripts
CertificateV2 = Certificate

__all__ = [
    "LocalSpace", "ModelSpec", "Term", "spec_from_quspin", "heisenberg_chain", "tfim_chain",
    "heisenberg_square", "spinless_fermion_chain", "bose_hubbard_chain", "spin1_heisenberg_chain",
    "build_bitwise", "build_kron", "cross_construct", "SpecOperator", "StateSet", "full_state_set",
    "u1_state_set", "magnetization_state_set", "translation", "spin_flip", "reflection",
    "cyclic_isometry", "sector_matrix", "OperatorAdapter", "enclose", "kato_temple", "inertia_count",
    "Claim", "Options", "certify", "quick", "Certificate", "CertificateV2", "methods_paragraph",
    "estimate_seconds", "auto_budget", "plan_for_claim", "parse_operator", "claim_from_quspin",
    "claim_from_arrays", "claim_from_npz", "claim_from_hdf5", "export_npz", "claim_from_file",
    "claim_from_bundle", "read_bundle", "write_bundle", "probe_float_environment",
    "enclose_spectrum", "SpectrumEnclosure", "SpectrumInterval", "exact_matrix",
    "exact_eigenpair_radii", "exact_recheck_lowest", "exact_recheck_sector_lowest", "site_permutation",
]
