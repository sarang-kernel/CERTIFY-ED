"""One call from a QuSpin `hamiltonian` and its eigensolver output to a Certificate (roadmap F-01).

    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian
    from certify_ed.integrations.quspin import certify_hamiltonian

    basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pblock=1)
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)
    w, V = H.eigsh(k=4, which="SA")
    cert = certify_hamiltonian(H, w, V)            # kind="lowest" by default
    print(cert.summary())

What the wrapper does, and only this:
  * takes the basis from `H.basis` and the operator strings from the static list QuSpin keeps on the object
    (`H._static_opstr_list`; pass `static=` to override, which is also the route for a `hamiltonian` built from
    matrices rather than strings), and refuses time-dependent Hamiltonians (a non-empty dynamic list);
  * sorts the claimed eigenvalues in ascending order (with the matching columns of V) when `kind` is 'lowest'
    or 'window', because `H.eigsh` does not promise an order;
  * builds the Claim and the ModelSpec with `importers.claim_from_quspin` (which keeps the symmetry-block
    path: a `basis` with kblock / pblock / zblock blocks gives a Claim with `sector_isometry = basis.get_proj`,
    so S1 audits the block the same way as a direct `claim_from_quspin` call);
  * calls `certify()` with Options built from `**options` exactly as `certify_ed.quick` does.

The ModelSpec is rebuilt from the same operator-string list that QuSpin received, so S1 compares QuSpin's
matrix construction with certify-ed's, not the user's intention with the string list. To certify against an
independently written model, pass `spec=` (e.g. `certify_ed.heisenberg_chain(L)`); then the string list is used
only for the local-space conventions of the basis.

This module adds no check of its own (provenance.NON_COMPUTATIONAL).
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional

import numpy as np

from ..certify import Claim, Options, certify
from ..importers import QUSPIN_HINT, _looks_like_quspin, claim_from_quspin
from ..precision import check_values, claimed_reals
from ..spec import ModelSpec

__all__ = ["certify_hamiltonian", "static_list_of"]

_CLAIM_KEYS = {f.name for f in fields(Claim)} - {"eigenvalues", "eigenvectors", "operator", "kind", "window",
                                                 "states", "sector_isometry"}
_OPTION_KEYS = {f.name for f in fields(Options)}


def static_list_of(H) -> list:
    """The operator-string list a QuSpin `hamiltonian` was built from (its `_static_opstr_list`).

    Raises ValueError when the object carries no string list (a hamiltonian built from matrices) or has
    time-dependent terms (a non-empty dynamic list): neither can be turned into a ModelSpec."""
    dyn = getattr(H, "_dynamic_opstr_list", None) or getattr(H, "dynamic", None)
    if dyn:
        raise ValueError("time-dependent hamiltonian (non-empty dynamic list): certify_hamiltonian certifies "
                         "static operators only; build a hamiltonian with the static list alone")
    static = getattr(H, "_static_opstr_list", None)
    if not static:
        raise ValueError("this hamiltonian keeps no operator-string list (it was built from matrices or the "
                         "QuSpin version does not store it): pass static=[...] (the list given to hamiltonian) "
                         "or spec=ModelSpec")
    return [[op, [list(c) for c in cps]] for op, cps in static]


def certify_hamiltonian(H, w, V=None, basis=None, static=None, spec: Optional[Any] = None,
                        kind: str = "lowest", window=None, **options):
    """Certify "w (with vectors V) are the `kind` eigenpairs of the QuSpin hamiltonian H" in one call.

    H       : a quspin.operators.hamiltonian (static terms only)
    w, V    : the output of H.eigsh(...) or H.eigh() (V may be None: then only the statements that need no
              vectors can run, and the certificate says which)
    basis   : the QuSpin basis (default H.basis); symmetry blocks (kblock, pblock, zblock, ...) are audited
              through the basis's isometry (`basis.get_proj`), U(1) blocks (Nup, Nf, Nb) through the state list
    static  : the operator-string list (default: the one stored on H)
    spec    : an independently written ModelSpec (or its to_dict()); default: rebuilt from `static`
    kind    : 'lowest' (default), 'window' (with window=(a, b)), 'all', or 'subset'
    options : fields of Claim (producer, accuracy_target, description) and of Options (preset, only, skip,
              budget_seconds, exact_recheck, progress, ...); `preset` is applied first, then the other keywords

    Returns the Certificate (`cert.summary()`, `cert.report()`, `cert.save(path)`).
    """
    if not _looks_like_quspin(H) and not hasattr(H, "tocsr"):
        try:
            import quspin  # noqa: F401
        except ImportError as e:
            raise ImportError(QUSPIN_HINT) from e
        raise TypeError(f"H must be a QuSpin hamiltonian (got {type(H).__name__})")
    basis = basis if basis is not None else getattr(H, "basis", None)
    if basis is None:
        raise TypeError("H has no basis attribute: pass basis=")
    static = static if static is not None else static_list_of(H)
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

    check_values(w, "w")                  # before the conversion, which would round an inexact value (precision.py)
    w_given = np.asarray(w).reshape(-1)    # complex values are passed on as given, so that certify records what it drops
    w = claimed_reals(w).reshape(-1)
    X = None if V is None else np.asarray(V)
    if X is not None and X.ndim == 1:
        X = X[:, None]
    if X is not None and X.shape[1] != w.size:
        raise ValueError(f"V has {X.shape[1]} columns for {w.size} eigenvalues (QuSpin returns one column per value)")
    reordered = False
    if kind in ("lowest", "window") and w.size > 1:
        order = np.argsort(w, kind="stable")
        if np.any(order != np.arange(w.size)):
            w, w_given = w[order], w_given[order]
            X = None if X is None else X[:, order]
            reordered = True
    claim, built_spec = claim_from_quspin(H, basis, static, w_given if w_given.dtype.kind in "cO" else w, X, kind=kind,
                                          window=window, **claim_kw)
    claim.producer["entry_point"] = "certify_ed.integrations.quspin.certify_hamiltonian"
    claim.producer["eigenvalues_sorted_by_wrapper"] = bool(reordered)
    claim.producer["spec_source"] = "user" if spec is not None else "static list of H"
    return certify(claim, spec if spec is not None else built_spec, opts)
