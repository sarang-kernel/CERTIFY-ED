"""Universal producer import (the dd `if=` side): file formats and the producer bundle convention."""
from .formats import (ARRAY_EXT, DEFAULT_KEYS, OPERATOR_EXT, FormatError, read_array, read_basis, read_operator,
                      sniff_kind, write_array, write_basis, write_operator)
from .bundle import META_SCHEMA, STEMS, Bundle, bundle_files, read_bundle, write_bundle

__all__ = ["ARRAY_EXT", "DEFAULT_KEYS", "OPERATOR_EXT", "FormatError", "read_array", "read_basis", "read_operator",
           "sniff_kind", "write_array", "write_basis", "write_operator", "META_SCHEMA", "STEMS", "Bundle",
           "bundle_files", "read_bundle", "write_bundle"]
