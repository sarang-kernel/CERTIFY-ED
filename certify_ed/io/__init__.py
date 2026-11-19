"""Universal producer import (the dd `if=` side): file formats and the producer bundle convention."""
from .formats import (ARRAY_EXT, DEFAULT_KEYS, OPERATOR_EXT, FormatError, read_array, read_basis, read_operator,
                      sniff_kind, write_array, write_basis, write_operator)
from .bundle import META_SCHEMA, STEMS, Bundle, bundle_files, read_bundle, write_bundle
# readers for other ED codes (3.2, F-02); each module states which parts of its layout are documented
from .hphi import claim_from_hphi, read_hphi_eigenvalue_file, read_hphi_eigenvec, read_hphi_energy
from .diagham import claim_from_diagham, read_diagham_spectrum, read_diagham_vector
from .spinpack import claim_from_spinpack, read_spinpack_def, read_spinpack_energies
from .alps import claim_from_alps, read_alps_spectrum, states_from_alps_basis

__all__ = ["ARRAY_EXT", "DEFAULT_KEYS", "OPERATOR_EXT", "FormatError", "read_array", "read_basis", "read_operator",
           "sniff_kind", "write_array", "write_basis", "write_operator", "META_SCHEMA", "STEMS", "Bundle",
           "bundle_files", "read_bundle", "write_bundle",
           "claim_from_hphi", "read_hphi_eigenvalue_file", "read_hphi_eigenvec", "read_hphi_energy",
           "claim_from_diagham", "read_diagham_spectrum", "read_diagham_vector",
           "claim_from_spinpack", "read_spinpack_def", "read_spinpack_energies",
           "claim_from_alps", "read_alps_spectrum", "states_from_alps_basis"]
