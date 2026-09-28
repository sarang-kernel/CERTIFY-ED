"""CERTIFY-ED: Verified Exact Diagonalization."""

__version__ = "2.0.0.dev0"
__author__ = "Sarang Vehale"

from .hamiltonian import (
    # Operators
    pauli_matrices, spin_half_operators, spin_one_operators, fermion_operators,
    tensor_product, site_operator, jordan_wigner_operator,
    # Symbolic builder
    SymbolicHamiltonian,
    # 1D spin-1/2
    build_tfim, build_heisenberg, build_xxz, build_xyz, build_ssh,
    build_j1j2_chain, build_majumdar_ghosh, build_cluster_model,
    # Fermionic
    build_free_fermion_chain, build_kitaev_chain, build_hubbard_chain,
    # Spin-1
    build_aklt_chain, build_haldane_chain,
    # 2D
    build_tfim_2d, build_heisenberg_2d, build_kitaev_honeycomb,
    # Registry
    MODEL_REGISTRY, list_models, build_model,
)

from .oracles import Oracle, NumPyOracle, ScipyOracle, SparseOracle, MultiOracle
from .certificates import Certificate, load_certificate
from .observables import ObservableCalculator
from .symmetries import (
    total_sz_operator, total_sx_operator, parity_operator,
    z_parity_operator, translation_operator,
    fermion_number_operator, fermion_parity_operator,
    project_onto_sector, commutator, commutator_norm, check_conservation,
)

# ---------------------------------------------------------------------------
# v2 API: certification of claimed results from arbitrary producers
# ---------------------------------------------------------------------------
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
from .certificate_v2 import Certificate as CertificateV2
from .importers import (claim_from_quspin, claim_from_arrays, claim_from_npz,
                        claim_from_hdf5, export_npz)
