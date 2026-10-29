"""JSON Schemas (draft 2020-12) of the certificate body and of the producer bundle, and optional validation.

The schemas are defined here, as data, so that the enumerations (statement ids, statuses, reason codes,
claim kinds) are the ones the code uses; `schema/*.schema.json` in the repository root are rendered from
these definitions (`python -m certify_ed.schema --write schema/`) and a test checks that they are in sync.

Validation needs the optional `jsonschema` package (extra `schema`). `validate_certificate_body(body)`
raises `SchemaValidationError` when the body does not conform; when `jsonschema` is not installed it
runs the structural checks only and returns `None`, unless `require=True`. Nothing in this module changes
a certified number: it reads bodies and reports (listed in provenance.NON_COMPUTATIONAL).

What JSON Schema cannot express is checked by `structural_check(body)`: every statement id appears exactly
once in the scope block (in_scope, narrowed or out_of_scope), the reason codes named in `scope.reason_codes`
are the ones used, and the status in `scope.in_scope` equals the statement's status.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .certificate_v2 import CERT_SCHEMA, STATEMENT_IDS, STATUSES
from .io.bundle import META_SCHEMA
from .scope import REASON_CODES, SCOPE_SCHEMA
from .spec import SPEC_SCHEMA

__all__ = ["CERTIFICATE_SCHEMA", "PRODUCER_BUNDLE_SCHEMA", "SCHEMA_FILES", "SchemaValidationError",
           "have_jsonschema", "validate_certificate_body", "validate_bundle_meta", "structural_check",
           "schema_errors", "write_schema_files", "render"]

DRAFT = "https://json-schema.org/draft/2020-12/schema"
SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$", "description": "SHA-256 hex digest"}
NULLABLE_SHA256 = {"oneOf": [SHA256, {"type": "null"}]}
CLAIM_KINDS = ("subset", "lowest", "window", "all")
NARROWING_BY = ("default", "dependency", "only", "skip", "preset")
OPERATOR_KINDS = ("matrix", "spec", "blackbox", "stable")
STATEMENT_STATUSES = STATUSES + ("CORROBORATED",)


class SchemaValidationError(ValueError):
    """The body does not conform to the certificate schema (message lists the first errors)."""


def _number_or_special():
    return {"$ref": "#/$defs/number_or_special"}


def _statement_evidence_defs() -> Dict[str, dict]:
    """Known evidence fields per statement id (all optional: evidence is open, this documents it)."""
    num = _number_or_special()
    budget = {"$ref": "#/$defs/budget_record"}
    return {
        "S1.model_fidelity": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"},
            "spec_name": {"type": "string"}, "spec_sha256": SHA256,
            "comparison": {"type": "object"}, "reference": {"type": "object"},
            "internal_cross_construction": {"type": "object"}, "spec_invariants_vs_reference": {"type": "object"},
            "audit_space": {"type": "object"}, "sector_audit": {"type": "object"},
            "hermiticity_max_abs": num, "model_shift_bound": num, "estimate_mb": num}},
        "S2.eigenpairs": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"},
            "n_pairs": {"type": "integer", "minimum": 0}, "rigorous": {"type": "boolean"},
            "max_radius": num, "median_radius": num, "max_existence_radius": num,
            "model_shift": num, "sector_lift_shift": {"oneOf": [num, {"type": "null"}]},
            "radius_refers_to": {"type": "string"}, "accuracy_target": {"type": "object"},
            "groups": {"type": "array", "items": {"type": "object"}}, "budget": budget}},
        "S3.completeness": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"}, "reason": {"type": "string"},
            "dimension": {"type": "integer", "minimum": 0},
            "count_from_inertia": {"type": "integer", "minimum": 0},
            "claimed_certified_inside": {"type": "integer", "minimum": 0},
            "claimed_touching": {"type": "integer", "minimum": 0},
            "missing_at_least": {"type": "integer", "minimum": 0},
            "unclaimed_near_top": {"type": "integer", "minimum": 0},
            "window": {"$ref": "#/$defs/interval"}, "certified_window": {"$ref": "#/$defs/interval"},
            "complete_below": num, "gap_lower_bound": num, "next_eigenvalue_lower_bound": num,
            "next_eigenvalue_estimate": num, "top_shift": num,
            "gap_interval": {"$ref": "#/$defs/interval"},
            "vectorless": {"type": "boolean"}, "n_claimed": {"type": "integer", "minimum": 0},
            "count_in_window": {"type": "integer", "minimum": 0}, "spurious_at_least": {"type": "integer", "minimum": 0},
            "count_statement": {"type": "string"}, "pairing": {"type": "string"},
            "edges": {"type": "object", "additionalProperties": {"type": "object"}},
            "beta_a": num, "beta_b": num, "beta_effective": num,
            "counts": {"type": "array", "items": {"type": "object"}}, "top_multiplet": {"type": "object"},
            "narrowed": {"type": "object"}, "budget": budget}},
        "S4.eigenspaces": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"},
            "clusters": {"type": "array", "items": {"type": "object"}},
            "narrowed": {"type": "object"}, "budget": budget}},
        "S5.corroboration": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"},
            "oracles": {"type": "object"}, "independence": {"type": "object"},
            "tolerance": num, "note": {"type": "string"}}},
        "S6.exact_recheck": {"type": "object", "properties": {"reason_code": {"$ref": "#/$defs/reason_code"}}},
        "S7.thermodynamics": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"},
            "dimension": {"type": "integer", "minimum": 0}, "n_intervals": {"type": "integer", "minimum": 0},
            "energy_offset": num, "hypothesis": {"type": "string"}, "method": {"type": "string"},
            "looseness": {"type": "string"},
            "temperatures": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}},
            "rows": {"type": "array", "items": {"type": "object"}}}},
        "S8.degeneracy": {"type": "object", "properties": {
            "reason_code": {"$ref": "#/$defs/reason_code"},
            "clusters": {"type": "array", "items": {"type": "object"}}, "budget": budget}},
    }


def _certificate_schema() -> dict:
    num = _number_or_special()
    ev_defs = _statement_evidence_defs()
    statement_rules: List[dict] = []
    for sid, ev in ev_defs.items():
        statement_rules.append({"if": {"properties": {"id": {"const": sid}}},
                                "then": {"properties": {"evidence": ev}}})
    statement_rules.append({
        "if": {"not": {"properties": {"id": {"const": "S5.corroboration"}}}},
        "then": {"properties": {"status": {"enum": list(STATUSES)}}},
        "else": {"properties": {"status": {"enum": ["CORROBORATED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED"]}}}})
    return {
        "$schema": DRAFT,
        "$id": "urn:certify-ed:schema:certificate:2.1",
        "title": "CERTIFY-ED certificate body, schema 2.1",
        "description": ("The JSON body written by certify_ed.Certificate.save (certify-ed/certificate/2.1). "
                        "Arrays live in the NPZ sidecar named in `arrays.sidecar`; their SHA-256 hashes are in the body. "
                        "Unknown top-level and evidence fields are allowed (a 2.x writer may add fields; none is removed "
                        "within a major version). Enumerations (statuses, statement ids, reason codes, claim kinds) are "
                        "closed. docs/schema.md describes every field."),
        "type": "object",
        "required": ["schema", "subject", "claim", "statements", "scope", "integrity", "environment"],
        "properties": {
            "schema": {"const": CERT_SCHEMA},
            "subject": {"$ref": "#/$defs/subject"},
            "claim": {"$ref": "#/$defs/claim"},
            "statements": {"type": "array", "items": {"$ref": "#/$defs/statement"},
                           "description": "one record per statement that was reached, in S1..S8 order"},
            "enclosures": {"type": "array", "items": {"$ref": "#/$defs/enclosure"}},
            "kato_temple": {"type": "array", "items": {"$ref": "#/$defs/kato_temple_record"}},
            "arithmetic": {"$ref": "#/$defs/arithmetic"},
            "environment": {"$ref": "#/$defs/environment"},
            "hermitian_part": {"oneOf": [{"type": "null"}, {"type": "object", "properties": {
                "max_abs_asymmetry": num, "note": {"type": "string"}}}]},
            "operator_note": {"type": ["string", "null"]},
            "spectrum_enclosure": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/spectrum_enclosure"}]},
            "gaps": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/gaps_record"}],
                     "description": "since 3.2: certified gap intervals read off S2 and S3 (certify_ed.gaps); null unless "
                                    "both are CERTIFIED"},
            "scope": {"$ref": "#/$defs/scope"},
            "not_verified": {"type": "array", "items": {"type": "string"}},
            "timing": {"type": "object", "properties": {"total_seconds": {"type": "number", "minimum": 0}}},
            "arrays": {"$ref": "#/$defs/arrays_manifest"},
            "integrity": {"$ref": "#/$defs/integrity"},
        },
        "$defs": {
            "number_or_special": {
                "description": "a JSON number, or the strings 'nan', 'inf', '-inf' (JSON has no non-finite numbers)",
                "oneOf": [{"type": "number"}, {"enum": ["nan", "inf", "-inf"]}]},
            "interval": {"type": "array", "items": num, "minItems": 2, "maxItems": 2,
                         "description": "[lower, upper]; either end may be 'inf' / '-inf'"},
            "statement_id": {"enum": list(STATEMENT_IDS)},
            "status": {"enum": list(STATEMENT_STATUSES),
                       "description": "CORROBORATED is valid for S5.corroboration only (evidence, not a certified statement)"},
            "reason_code": {"enum": sorted(REASON_CODES),
                            "description": "certify_ed.scope.REASON_CODES; documented in docs/SCOPE_MATRIX.md"},
            "claim_kind": {"enum": list(CLAIM_KINDS)},
            "budget_record": {"type": "object", "properties": {
                "seconds": {"type": ["number", "null"]}, "mb": {"type": ["number", "null"]},
                "events": {"type": "array", "items": {"type": "object", "required": ["statement", "code"], "properties": {
                    "statement": {"$ref": "#/$defs/statement_id"},
                    "code": {"enum": ["BUDGET_TIME", "BUDGET_MEMORY"]},
                    "note": {"type": "string"}, "estimate_mb": {"type": "number"}}}}}},
            "model_spec": {
                "type": "object", "required": ["schema", "n_sites", "local_space", "terms"],
                "properties": {
                    "schema": {"const": SPEC_SCHEMA},
                    "n_sites": {"type": "integer", "minimum": 1},
                    "local_space": {"type": "object", "required": ["kind", "q"], "properties": {
                        "kind": {"enum": ["spin", "boson", "fermion"]}, "q": {"type": "integer", "minimum": 2},
                        "spin": {"type": "string"}, "convention": {"type": "string"},
                        "n_max": {"type": "integer"}, "jw_string": {"type": "string"}}},
                    "encoding": {"type": "object", "properties": {
                        "site_order": {"type": "string"}, "state_order": {"enum": ["descending", "ascending"]},
                        "base": {"type": "integer", "minimum": 2}}},
                    "terms": {"type": "array", "items": {
                        "type": "array", "minItems": 2, "maxItems": 2,
                        "prefixItems": [{"type": "string", "description": "operator letters, one per site"},
                                        {"type": "array", "items": {
                                            "type": "array", "minItems": 2,
                                            "prefixItems": [{"oneOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}]}],
                                            "items": {"type": "integer", "minimum": 0},
                                            "description": "[coefficient (real, or [re, im]), site, site, ...]"}}]}},
                    "name": {"type": "string"}, "parameters": {"type": "object"}, "notes": {"type": "string"}}},
            "subject": {
                "type": "object", "required": ["description", "operator"],
                "properties": {
                    "description": {"type": "string"},
                    "spec": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/model_spec"}]},
                    "spec_sha256": NULLABLE_SHA256,
                    "operator": {"type": "object", "required": ["kind", "dimension"], "properties": {
                        "kind": {"enum": list(OPERATOR_KINDS)},
                        "dimension": {"type": "integer", "minimum": 1},
                        "matrix_sha256": NULLABLE_SHA256,
                        "norm_upper_bound": num}},
                    "sector": {"oneOf": [{"type": "null"}, {"type": "object", "required": ["isometry_sha256", "dimension"],
                                                            "properties": {"isometry_sha256": SHA256,
                                                                           "dimension": {"type": "integer", "minimum": 1},
                                                                           "lift_shift": num}}]}}},
            "claim": {
                "type": "object", "required": ["kind", "n_eigenvalues", "eigenvalues", "has_eigenvectors"],
                "properties": {
                    "description": {"type": "string"},
                    "kind": {"$ref": "#/$defs/claim_kind"},
                    "window": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/interval"}]},
                    "n_eigenvalues": {"type": "integer", "minimum": 0},
                    "has_eigenvectors": {"type": "boolean"},
                    "producer": {"type": "object",
                                 "description": "free-form producer record (name, version, bundle path, files)"},
                    "accuracy_target": {"type": ["number", "null"]},
                    "eigenvalues": {"type": "array", "items": {"type": "number"},
                                    "description": "the claimed eigenvalues in the claim's order (finite)"},
                    "eigenvectors_sha256": NULLABLE_SHA256},
                "if": {"properties": {"kind": {"const": "window"}}},
                "then": {"required": ["window"], "properties": {"window": {"$ref": "#/$defs/interval"}}}},
            "statement": {
                "type": "object", "required": ["id", "status", "summary"],
                "properties": {
                    "id": {"$ref": "#/$defs/statement_id"},
                    "status": {"$ref": "#/$defs/status"},
                    "summary": {"type": "string"},
                    "evidence": {"type": "object"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "seconds": {"type": "number", "minimum": 0}},
                "allOf": statement_rules},
            "enclosure": {
                "type": "object", "required": ["index", "claimed", "lower", "upper", "radius"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0}, "claimed": {"type": "number"},
                    "lower": num, "upper": num, "radius": num, "existence_radius": num,
                    "residual_bound": num, "residual_computed": num,
                    "method": {"enum": ["single", "cluster", "residual interval"]},
                    "group": {"type": "integer", "minimum": 0}, "group_size": {"type": "integer", "minimum": 1},
                    "distinct": {"type": "boolean"}}},
            "kato_temple_record": {
                "type": "object", "required": ["index", "lower", "upper"],
                "properties": {"index": {"type": "integer", "minimum": 0}, "lower": num, "upper": num,
                               "width": num, "residual_width": num,
                               "isolation_interval": {"$ref": "#/$defs/interval"}}},
            "arithmetic": {
                "type": "object", "required": ["arithmetic", "unit_roundoff", "model"],
                "properties": {
                    "arithmetic": {"type": "string"}, "unit_roundoff": {"type": "number", "exclusiveMinimum": 0},
                    "model": {"type": "string"}, "assumptions": {"type": "array", "items": {"type": "string"}},
                    "complex_inner_product_constant": {"type": "string"},
                    "runtime_probe": {"type": "object", "required": ["standard_model_hypotheses_hold"], "properties": {
                        "standard_model_hypotheses_hold": {"type": "boolean"},
                        "failed": {"type": "array", "items": {"type": "string"}},
                        "checks": {"type": "object", "additionalProperties": {"type": "boolean"}}}}}},
            "environment": {
                "type": "object", "required": ["python", "numpy", "scipy", "created_utc"],
                "properties": {
                    "python": {"type": "string"}, "platform": {"type": "string"}, "machine": {"type": "string"},
                    "numpy": {"type": "string"}, "scipy": {"type": "string"},
                    "created_utc": {"type": "string", "description": "ISO 8601 with offset, seconds precision"},
                    "blas": {}, "certify_ed": {"type": "string"},
                    "source_hash": {"type": ["string", "null"]}, "source_dirty": {"type": ["boolean", "null"]},
                    "git_commit": {"type": ["string", "null"]},
                    "python_flint": {"type": "string"}, "flint_c_library": {"type": "string"},
                    "distribution_sha256": {"type": ["string", "null"],
                                            "description": "reserved: hash of the installed distribution (not written by 3.1)"}}},
            "spectrum_enclosure": {
                "type": "object", "required": ["complete", "intervals", "n_intervals"],
                "properties": {
                    "complete": {"type": "boolean"}, "n_intervals": {"type": "integer", "minimum": 0},
                    "max_halfwidth": num,
                    "intervals": {"type": "array", "items": {"type": "object", "required": ["lo", "hi", "count"], "properties": {
                        "lo": num, "hi": num, "count": {"type": "integer", "minimum": 0},
                        "counted": {"type": "boolean"}, "method": {"type": "string"}}}}}},
            "gaps_record": {
                "type": "object", "required": ["refers_to", "ranked", "levels", "gaps"],
                "description": ("levels: maximal chains of overlapping S2 enclosures (radii including the model shift), "
                                "each holding exactly `count` eigenvalues when S3 is CERTIFIED; gaps: intervals for the "
                                "differences of consecutive eigenvalues, ends rounded outward, upper end 'inf' when the next "
                                "eigenvalue is not claimed (then the lower end is S3's gap_lower_bound)"),
                "properties": {
                    "refers_to": {"type": "string"}, "ranked": {"type": "boolean"}, "rank_origin": {"type": "string"},
                    "n_levels": {"type": "integer", "minimum": 0}, "basis": {"type": "string"},
                    "levels": {"type": "array", "items": {
                        "type": "object", "required": ["lower", "upper", "members", "count"], "properties": {
                            "lower": num, "upper": num, "count": {"type": "integer", "minimum": 1},
                            "members": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                            "ranks": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2}}}},
                    "gaps": {"type": "array", "items": {"$ref": "#/$defs/gap_record"}},
                    "top_gap": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/gap_record"}]}}},
            "gap_record": {
                "type": "object", "required": ["interval", "lower_from", "upper_from"],
                "properties": {
                    "interval": {"$ref": "#/$defs/interval"},
                    "lower_from": {"enum": ["S2", "S3"]}, "upper_from": {"enum": ["S2", None]},
                    "within_level": {"type": "boolean"},
                    "levels": {"type": "array", "items": {"type": ["integer", "null"]}, "minItems": 2, "maxItems": 2},
                    "between_ranks": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2},
                    "note": {"type": "string"}}},
            "scope": {
                "type": "object",
                "required": ["schema", "preset", "in_scope", "narrowed", "out_of_scope", "reason_codes"],
                "properties": {
                    "schema": {"const": SCOPE_SCHEMA},
                    "preset": {"type": ["string", "null"]},
                    "selection": {"type": "object", "properties": {
                        "preset": {"type": ["string", "null"]},
                        "wanted": {"type": "array", "items": {"type": "string"}},
                        "implied": {"type": "object", "additionalProperties": {"type": "string"}},
                        "narrowed": {"type": "object"},
                        "only": {"oneOf": [{"type": "null"}, {"type": "array", "items": {"type": "string"}}]},
                        "skip": {"type": "array", "items": {"type": "string"}},
                        "narrowing_reason": {"type": "string"}}},
                    "budgets": {"$ref": "#/$defs/budget_record"},
                    "in_scope": {"type": "array", "items": {
                        "type": "object", "required": ["id", "status"], "properties": {
                            "id": {"$ref": "#/$defs/statement_id"},
                            "status": {"enum": [s for s in STATEMENT_STATUSES if s != "NOT_CHECKED"]},
                            "implied": {"type": ["string", "null"]}}}},
                    "narrowed": {"type": "array", "items": {
                        "type": "object", "required": ["id", "by", "reason", "reason_code"], "properties": {
                            "id": {"$ref": "#/$defs/statement_id"},
                            "by": {"enum": list(NARROWING_BY)},
                            "reason": {"type": "string"},
                            "reason_code": {"$ref": "#/$defs/reason_code"},
                            "status": {"const": "NOT_CHECKED"}}}},
                    "out_of_scope": {"type": "array", "items": {
                        "type": "object", "required": ["quantity", "reason_code"], "properties": {
                            "quantity": {"type": "string",
                                         "description": "a statement id (NOT_CHECKED statement) or a named quantity"},
                            "reason_code": {"$ref": "#/$defs/reason_code"},
                            "detail": {"type": "string"}, "implied": {"type": ["string", "null"]}}}},
                    "requested": {"type": "object", "properties": {
                        "kind": {"$ref": "#/$defs/claim_kind"}, "has_eigenvectors": {"type": "boolean"},
                        "has_spec": {"type": "boolean"}, "has_matrix": {"type": "boolean"}}},
                    "options": {"type": "object"},
                    "operator_scope": {"type": "string"},
                    "reason_codes": {"type": "object", "propertyNames": {"$ref": "#/$defs/reason_code"},
                                     "additionalProperties": {"type": "string"},
                                     "description": "the reason codes used in this certificate with their one-line meaning"}}},
            "arrays_manifest": {
                "type": "object", "required": ["sidecar", "sha256", "shapes", "dtypes"],
                "properties": {
                    "sidecar": {"type": ["string", "null"], "description": "NPZ file next to the JSON (null in HDF5 export)"},
                    "sha256": {"type": "object", "additionalProperties": SHA256},
                    "shapes": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "integer", "minimum": 0}}},
                    "dtypes": {"type": "object", "additionalProperties": {"type": "string", "description": "numpy dtype.str, e.g. '<f8'"}}}},
            "integrity": {
                "type": "object", "required": ["algorithm", "canonicalization", "body_sha256"],
                "properties": {
                    "algorithm": {"const": "sha256"},
                    "canonicalization": {"const": "certify-ed-canonical-json-1"},
                    "body_sha256": SHA256}},
        },
    }


def _bundle_schema() -> dict:
    return {
        "$schema": DRAFT,
        "$id": "urn:certify-ed:schema:producer-bundle:1.0",
        "title": "CERTIFY-ED producer bundle meta.json, schema 1.0",
        "description": ("meta.json of a producer bundle directory (docs/producer_bundle.md). Every field is optional; "
                        "the directory must hold eigenvalues.<ext>. spec.json, when present, conforms to $defs/model_spec."),
        "type": "object",
        "properties": {
            "schema": {"const": META_SCHEMA},
            "kind": {"enum": list(CLAIM_KINDS)},
            "window": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "accuracy_target": {"type": "number", "exclusiveMinimum": 0},
            "description": {"type": "string"},
            "producer": {"type": "object", "properties": {"name": {"type": "string"}, "version": {"type": "string"}}},
            "keys": {"type": "object", "properties": {k: {"type": "string"} for k in ("eigenvalues", "eigenvectors", "operator", "basis")}},
            "complex_pairs": {"type": "object", "additionalProperties": {"type": "boolean"}},
            "eigenvectors_layout": {"enum": ["columns", "rows"]},
            "layout": {"type": "object", "properties": {"operator": {"enum": ["auto", "coo", "dense", "csr"]}}},
            "operator_shape": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2},
            "one_based": {"type": "boolean"},
            "state_order": {"enum": ["descending", "ascending"]},
            "basis_label": {"type": "string"},
            "temperatures": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}},
            "preset": {"type": "string"},
            "only": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
            "skip": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
            "budget_seconds": {"type": "number", "minimum": 0},
            "budget_mb": {"type": "number", "minimum": 0},
        },
        "if": {"properties": {"kind": {"const": "window"}}, "required": ["kind"]},
        "then": {"required": ["window"]},
        "$defs": {"model_spec": _certificate_schema()["$defs"]["model_spec"]},
    }


CERTIFICATE_SCHEMA: dict = _certificate_schema()
PRODUCER_BUNDLE_SCHEMA: dict = _bundle_schema()
SCHEMA_FILES = {"certificate-2.1.schema.json": CERTIFICATE_SCHEMA,
                "producer-bundle-1.0.schema.json": PRODUCER_BUNDLE_SCHEMA}


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def write_schema_files(directory: str) -> List[str]:
    os.makedirs(directory, exist_ok=True)
    out = []
    for name, schema in SCHEMA_FILES.items():
        p = os.path.join(directory, name)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(render(schema))
        out.append(p)
    return out


def have_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
        return True
    except Exception:
        return False


def _validator(schema: dict):
    import jsonschema
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def schema_errors(body: Any, schema: dict) -> Optional[List[str]]:
    """List of 'path: message' strings (empty when valid); None when jsonschema is not installed."""
    if not have_jsonschema():
        return None
    v = _validator(schema)
    errs = sorted(v.iter_errors(body), key=lambda e: [str(p) for p in e.absolute_path])
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errs]


def structural_check(body: dict) -> List[str]:
    """Checks JSON Schema cannot express. Returns a list of problems (empty when fine)."""
    problems: List[str] = []
    sc = body.get("scope")
    if not isinstance(sc, dict):
        return ["scope: missing"]
    seen: Dict[str, int] = {sid: 0 for sid in STATEMENT_IDS}
    for x in sc.get("in_scope", []):
        seen[x.get("id")] = seen.get(x.get("id"), 0) + 1
    for x in sc.get("narrowed", []):
        seen[x.get("id")] = seen.get(x.get("id"), 0) + 1
    for x in sc.get("out_of_scope", []):
        q = x.get("quantity")
        if q in seen:
            seen[q] += 1
    for sid, n in seen.items():
        if sid in STATEMENT_IDS and n != 1:
            problems.append(f"scope: statement {sid} appears {n} times (must be exactly once)")
    by_id = {s.get("id"): s for s in body.get("statements", []) if isinstance(s, dict)}
    for x in sc.get("in_scope", []):
        st = by_id.get(x.get("id"))
        if st is None:
            problems.append(f"scope.in_scope: {x.get('id')} has no statement record")
        elif st.get("status") != x.get("status"):
            problems.append(f"scope.in_scope: {x.get('id')} status {x.get('status')} differs from statement {st.get('status')}")
        elif st.get("status") == "NOT_CHECKED":
            problems.append(f"scope.in_scope: {x.get('id')} is NOT_CHECKED")
    used = {x.get("reason_code") for x in sc.get("out_of_scope", []) + sc.get("narrowed", [])}
    declared = set(sc.get("reason_codes", {}) or {})
    for code in sorted(used - declared):
        problems.append(f"scope.reason_codes: {code} is used but not declared")
    for code in sorted(used | declared):
        if code not in REASON_CODES:
            problems.append(f"scope: unknown reason code {code!r}")
    for s in body.get("statements", []):
        if s.get("status") == "NOT_CHECKED":
            code = (s.get("evidence") or {}).get("reason_code")
            if code is not None and code not in REASON_CODES:
                problems.append(f"statements/{s.get('id')}: unknown reason code {code!r}")
    claim = body.get("claim", {})
    if isinstance(claim, dict) and isinstance(claim.get("eigenvalues"), list) and \
            claim.get("n_eigenvalues") is not None and len(claim["eigenvalues"]) != claim["n_eigenvalues"]:
        problems.append("claim: n_eigenvalues differs from len(eigenvalues)")
    return problems


def validate_certificate_body(body: dict, require: bool = False, structural: bool = True) -> Optional[bool]:
    """Validate a certificate body. Returns True when valid, None when jsonschema is missing and
    require is False (structural checks still run); raises SchemaValidationError (a ValueError) when the
    body does not conform, ImportError when require is True and jsonschema is missing."""
    errs = schema_errors(body, CERTIFICATE_SCHEMA)
    if errs is None:
        if require:
            raise ImportError("schema validation needs the 'jsonschema' package (pip install certify-ed[schema])")
        if structural:
            probs = structural_check(body)
            if probs:
                raise SchemaValidationError("certificate structure: " + "; ".join(probs))
        return None
    if structural:
        errs = errs + structural_check(body)
    if errs:
        raise SchemaValidationError(f"certificate does not conform to {CERT_SCHEMA}: " + "; ".join(errs[:8]) +
                                    (f" (+{len(errs) - 8} more)" if len(errs) > 8 else ""))
    return True


def validate_bundle_meta(meta: dict, spec: Optional[dict] = None, require: bool = False) -> Optional[bool]:
    """Validate a producer bundle's meta.json (and spec.json when given) against the bundle schema."""
    errs = schema_errors(meta, PRODUCER_BUNDLE_SCHEMA)
    if errs is None:
        if require:
            raise ImportError("schema validation needs the 'jsonschema' package (pip install certify-ed[schema])")
        return None
    if spec is not None:
        sub = dict(PRODUCER_BUNDLE_SCHEMA["$defs"]["model_spec"])
        sub["$schema"] = DRAFT
        errs += ["spec.json/" + e for e in (schema_errors(spec, sub) or [])]
    if errs:
        raise SchemaValidationError(f"producer bundle does not conform to {META_SCHEMA}: " + "; ".join(errs[:8]))
    return True


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m certify_ed.schema", description="render or check the JSON Schema files")
    p.add_argument("--write", metavar="DIR", help="write certificate-2.1.schema.json and producer-bundle-1.0.schema.json")
    p.add_argument("--check", metavar="DIR", help="exit 1 when the files in DIR differ from the definitions")
    p.add_argument("certificates", nargs="*", help="certificate JSON files to validate")
    a = p.parse_args(argv)
    rc = 0
    if a.write:
        for f in write_schema_files(a.write):
            print(f)
    if a.check:
        for name, schema in SCHEMA_FILES.items():
            path = os.path.join(a.check, name)
            try:
                with open(path, encoding="utf-8") as f:
                    same = f.read() == render(schema)
            except OSError:
                same = False
            print(f"{name}: {'in sync' if same else 'DIFFERS'}")
            rc |= 0 if same else 1
    for path in a.certificates:
        with open(path, encoding="utf-8") as f:
            body = json.load(f)
        try:
            r = validate_certificate_body(body)
            print(f"{path}: {'valid' if r else 'structure ok (jsonschema not installed)'}")
        except SchemaValidationError as e:
            print(f"{path}: INVALID: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
