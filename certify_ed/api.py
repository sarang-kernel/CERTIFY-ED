"""One-call entry point: `quick(H, w, V, spec, kind, **options)` builds the Claim and the Options and returns
the Certificate. It adds no check of its own: everything it returns is what `certify()` returns for the same
Claim and Options (provenance.NON_COMPUTATIONAL)."""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional

from .certify import Claim, Options, certify
from .spec import ModelSpec

__all__ = ["quick"]

_CLAIM_KEYS = {f.name for f in fields(Claim)} - {"eigenvalues", "eigenvectors", "operator", "kind"}
_OPTION_KEYS = {f.name for f in fields(Options)}


def quick(H, w, V=None, spec: Optional[Any] = None, kind: str = "lowest", **options):
    """Certify the claim "w (with vectors V) are the `kind` eigenpairs of H" in one call.

    H     : the producer's operator (dense array, scipy.sparse matrix, LinearOperator), or None to use the
            matrix certify-ed builds from `spec`
    w, V  : claimed eigenvalues and eigenvectors (one column per value); V may be None (then only the
            statements that need no vectors can run; the certificate says which and why)
    spec  : a ModelSpec, a ModelSpec.to_dict() dictionary, or None
    kind  : 'lowest' (the len(w) lowest eigenvalues), 'window' (with window=(a, b)), 'all', or 'subset'
    options: fields of Claim (window, states, sector_isometry, producer, accuracy_target, description) and
            of Options (preset, only, skip, budget_seconds, exact_recheck, progress, ...); `preset` applies
            the preset's option values first, then the other keywords override them.

    Returns the Certificate; `cert.summary()` is one sentence, `cert.report()` the full text, and the
    object renders as a table in Jupyter. Inputs that cannot be certified (NaN, non-Hermitian matrices,
    eigenvalue-only claims) do not raise: the statements say INCONCLUSIVE / NOT_CHECKED and carry a
    `next_action`, exactly as `certify()` does.
    """
    if isinstance(spec, dict):
        spec = ModelSpec.from_dict(spec)
    elif spec is not None and not isinstance(spec, ModelSpec):
        raise TypeError(f"spec must be a ModelSpec, a dict or None (got {type(spec).__name__})")
    claim_kw = {k: options.pop(k) for k in list(options) if k in _CLAIM_KEYS}
    unknown = sorted(k for k in options if k not in _OPTION_KEYS)
    if unknown:
        raise TypeError(f"unknown keyword(s) {', '.join(unknown)}; Claim fields: {', '.join(sorted(_CLAIM_KEYS))}; "
                        f"Options fields: {', '.join(sorted(_OPTION_KEYS))}")
    preset = options.pop("preset", None)
    opts = Options.from_preset(preset, **options) if preset else Options(**options)
    claim = Claim(w, V, operator=H, kind=kind, **claim_kw)
    return certify(claim, spec, opts)
