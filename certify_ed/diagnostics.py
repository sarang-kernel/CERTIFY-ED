"""Explanatory diagnostics: a concrete `next_action` for every statement that is not CERTIFIED or FAILED.

`annotate_next_actions` is called once by `certify()` after all statements exist. For each statement with
status INCONCLUSIVE or NOT_CHECKED it writes evidence["next_action"]: one sentence saying what the user
(or the producer) can do so that the statement can be decided. The rules are keyed on the reason code
(scope.REASON_CODES) first, then on the wording of the statement's summary and evidence["reason"]
(the strings written by certify.py), then on the statement id; a final generic sentence makes sure the
field is never empty. Nothing here reads or changes a status, a bound or a count: the module is
presentation only (provenance.NON_COMPUTATIONAL).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .scope import classify_not_checked

__all__ = ["next_action", "annotate_next_actions", "NEXT_ACTION_BY_CODE", "entanglement_next_action"]

MORE_PAIRS = ("ask the producer for more eigenpairs (k+2 at least) so that the cluster at the top of the claim is "
              "complete and separated from the rest, or use kind='window' with an upper edge inside a gap")
REORTHO = ("re-orthonormalize the claimed eigenvectors (e.g. a thin QR of the block) and drop duplicate pairs, "
           "then certify again")
TIGHTEN = ("tighten the producer's convergence (smaller residuals ||H x - mu x||), or declare the accuracy the "
           "producer achieves with Claim(accuracy_target=...)")
NONFINITE = ("remove the non-finite eigenvalues / eigenvector columns listed in the evidence from the claim, or "
             "recompute them; NaN and inf cannot be enclosed")
FLINT = "install python-flint (pip install python-flint, or pip install 'certify-ed[exact]') for the exact statements"

# reason code -> next action (statement-independent wording; refined per statement below)
NEXT_ACTION_BY_CODE: Dict[str, str] = {
    "NARROWED_BY_USER": ("select the statement: remove it from Options.skip / --skip, add it to Options.only / "
                         "--only, or choose a preset that includes it (certify-ed presets)"),
    "DEFAULT_SELECTION": "switch the statement on through its option, a preset, or Options.only / --only",
    "DEPENDENCY_NARROWED": "also select the statements it needs (listed in the summary); dependencies are added by --only",
    "BUDGET_TIME": ("raise Options.budget_seconds / --budget-seconds (or use budget_seconds='auto' for a budget from "
                    "the cost model), or narrow the selection with Options.only so the remaining statements start in time"),
    "BUDGET_MEMORY": ("raise Options.budget_mb / --budget-mb above the estimate in the evidence, use "
                      "inertia_method='sparse', or certify one symmetry sector (Claim.states) instead of the full space"),
    "NEEDS_EIGENVECTORS": ("supply the eigenvectors (Claim.eigenvectors, one column per claimed value): kind='lowest', "
                           "S2, S3, S4, S6 and S8 need them"),
    "NEEDS_SPEC": ("declare the model: pass a ModelSpec (certify_ed.heisenberg_chain(...), spec_from_quspin(...), "
                   "ModelSpec.from_dict(...) or spec.json in the producer bundle)"),
    "NEEDS_MATRIX": ("supply the operator as a stored matrix (Claim.operator: dense or scipy.sparse), or declare a spec "
                     "so that a reference matrix of dimension <= reference_max_dim can be built"),
    "NEEDS_WHOLE_SPECTRUM": ("claim the whole spectrum (kind='all', every eigenpair with orthonormal vectors) so that S3 "
                             "can certify completeness; for S7 a 'lowest' claim with S3 CERTIFIED also suffices (the omitted "
                             "eigenvalues enter through the tail bound (n-k) exp(-beta lambda_{k+1}), C-06)"),
    "NEEDS_TEMPERATURES": "pass the temperatures: Options(temperatures=[...]) or --temperatures T1,T2 (k_B = 1)",
    "NEEDS_DEPENDENCY": FLINT,
    "MISSING_DEPENDENCY": FLINT,
    "NEEDS_OBSERVABLES": "request them: Options(observables={...}) or Options(correlations={...})",
    "NO_COMPLETENESS_CLAIM": ("make a completeness claim: kind='lowest' (the k lowest eigenvalues), kind='window' with "
                              "window=(a, b), or kind='all'"),
    "NO_CLUSTERS": "nothing is missing; set Options(eigenspaces='all') to certify isolated levels as well",
    "SIZE_LIMIT": "raise the dimension limit named in the summary (Options.inertia_max_dim / exact_max_dim)",
    "UNSUPPORTED_CLAIM": "use kind='lowest' for this statement",
    "UNSUPPORTED_MODEL": "restrict the statement to the model class it handles (see docs/SCOPE_MATRIX.md)",
    "NOT_IMPLEMENTED": "not available in this version (docs/SCOPE_MATRIX.md names the mathematics needed)",
    "UNDERFLOW_RISK": ("rescale the problem (e.g. divide the Hamiltonian by its largest coupling and multiply the "
                       "enclosures back), or flush entries below 2^-900 to zero if they are numerical noise; to "
                       "assume the no-underflow hypothesis instead, pass Options(underflow_check=False) / "
                       "--assume-no-underflow (recorded in the certificate)"),
    "FP_GUARD_RANGE": ("reduce the problem below 2^26 (a symmetry sector of smaller dimension, or a spec whose row "
                       "width bound is smaller: Options(matrix_free=True) counts the monomials per row); the guards are "
                       "not established above 2^26 (docs/proofs/thm1_fp-accounting.md)"),
    "NOT_APPLICABLE": "the statement does not apply to this input; see docs/SCOPE_MATRIX.md",
    "OUT_OF_SCOPE_DYNAMICS": ("no statement of certify-ed covers dynamics; cite the per-eigenpair certificate and report "
                              "the dynamical quantity as uncertified (docs/SCOPE_MATRIX.md)"),
    "OUT_OF_SCOPE_EXTRAPOLATION": ("certify each finite size separately and cite the certificates by hash; the "
                                   "extrapolation itself is not a statement about any finite matrix"),
    "DECLARED_ESTIMATE": ("report the declared estimate as the producer's own (with its sample count and method); the "
                          "certified statements of the record (Rayleigh upper bound, existence radius, Temple lower bound) "
                          "come from the trial vector, not from the samples"),
}

_GENERIC = {
    "S1.model_fidelity": "compare the producer's operator with the declared model entrywise (certify_ed.build_bitwise)",
    "S2.eigenpairs": TIGHTEN,
    "S3.completeness": MORE_PAIRS,
    "S4.eigenspaces": MORE_PAIRS,
    "S5.corroboration": "name other corroborating solvers in Options.corroborate ('jacobi', 'arpack')",
    "S6.exact_recheck": ("the exact count could not separate the claimed values from the rest: " + MORE_PAIRS),
    "S7.thermodynamics": ("raise the temperatures or tighten the whole-spectrum enclosure: the interval widths are too "
                          "large at the requested temperatures"),
    "S8.degeneracy": ("the cluster count could not be bracketed: " + MORE_PAIRS + "; or tighten the residuals"),
}


def _text(stmt: dict) -> str:
    ev = stmt.get("evidence") or {}
    return (str(stmt.get("summary", "")) + " || " + str(ev.get("reason", ""))).lower()


def _cluster_reasons(stmt: dict) -> List[str]:
    ev = stmt.get("evidence") or {}
    return [str(c.get("reason", "")).lower() for c in ev.get("clusters", []) if isinstance(c, dict)
            and c.get("status") != "CERTIFIED"]


def _inconclusive(stmt: dict, ctx: dict) -> str:
    sid, text = stmt["id"], _text(stmt)
    if (stmt.get("evidence") or {}).get("reason_code") == "FP_GUARD_RANGE":
        return NEXT_ACTION_BY_CODE["FP_GUARD_RANGE"]
    if "floating-point environment" in text and "standard model" in text:
        return ("run on a platform where the floating-point probe passes (no fast-math, no flush-to-zero; see "
                "arithmetic.runtime_probe.failed): the bounds are not certified under this environment")
    if "non-finite" in text:
        return NONFINITE
    if "zero eigenvector" in text or "no finite enclosure" in text or "is not finite" in text:
        return "drop or recompute the claimed pairs listed in the evidence: a zero vector has no enclosure"
    if sid == "S1.model_fidelity":
        return ("report this as a bug (the two reference builders of certify-ed disagree); meanwhile certify with "
                "Options(model_fidelity=False) and compare the producer matrix with certify_ed.build_bitwise directly")
    if sid == "S2.eigenpairs":
        if "rounding not boundable" in text:
            return ("supply the operator as a stored dense or scipy.sparse matrix, or declare a spec, so that the "
                    "rounding of the matrix-vector products can be bounded")
        if "orthonormality defect" in text or "not be matched to distinct" in text:
            return REORTHO
        return TIGHTEN
    if sid == "S3.completeness":
        if "full-spectrum claim" in text:
            return ("claim all eigenpairs (count = dimension) with orthonormal vectors, or use kind='lowest' for a "
                    "partial spectrum")
        if "not all certified distinct" in text or "not all distinct eigenpairs" in text:
            return REORTHO
        if "multiplet" in text or "unclaimed eigenvalue lies within" in text or "could not be classified" in text:
            return MORE_PAIRS
        if "backward error" in text:
            return ("the factorization's backward error is too large for the margin above the highest claimed value: "
                    + TIGHTEN + "; or set Options(inertia_method='dense')")
        if "inconclusive (" in text or "zero pivot" in text or "no conclusive inertia count" in text:
            return ("a pivot of the LDL^T factorization was too close to zero: set Options(inertia_method='dense'), "
                    "or ask the producer for more eigenpairs so the probe shift moves away from an eigenvalue")
        if "window edge" in text:
            return "move the window edges (a, b) away from eigenvalues, so the inertia counts at the edges are conclusive"
        if "counts could not be matched" in text or ctx.get("kind") == "window":
            return ("supply orthonormal eigenvectors for every eigenvalue in the window, or widen the window so every "
                    "claimed enclosure lies strictly inside it")
        return MORE_PAIRS
    if sid == "S4.eigenspaces":
        reasons = " ".join(_cluster_reasons(stmt))
        if "orthonormal" in reasons:
            return REORTHO
        if "overlaps" in reasons:
            return (MORE_PAIRS + "; or raise Options.cluster_tol so overlapping pairs form one cluster")
        if "residual not small" in reasons:
            return TIGHTEN + "; the separation from the other eigenvalues is in the evidence"
        return MORE_PAIRS
    if sid == "S5.corroboration":
        return ("the corroborating solvers disagree with the claim or could not run: inspect evidence.solvers, then "
                "fix the producer's result or name other solvers in Options.corroborate")
    return _GENERIC.get(sid, "inspect the evidence of this statement; docs/SCOPE_MATRIX.md lists what each statement needs")


def _not_checked(stmt: dict, ctx: dict) -> str:
    sid, text = stmt["id"], _text(stmt)
    code = classify_not_checked(stmt)
    m = re.search(r"(inertia_max_dim|exact_max_dim)\s+(\d+)", text)
    if code == "SIZE_LIMIT":
        if sid == "S6.exact_recheck" or "exact_max_dim" in text:
            return ("raise Options.exact_max_dim above the claim's dimension (the exact re-check needs O(n^2) integer "
                    "entries), or certify a symmetry sector with exact_recheck_sector_lowest")
        lim = f" (now {m.group(2)})" if m else ""
        return (f"raise Options.inertia_max_dim{lim} above the dimension, or set Options(inertia_method='sparse') so the "
                "factorization uses a sparse LDL^T")
    if code in ("NEEDS_DEPENDENCY", "MISSING_DEPENDENCY"):
        if sid == "S5.corroboration":
            return "name corroborating solvers: Options(corroborate=('jacobi', 'arpack')) or --preset benchmark"
        return FLINT
    if code == "NEEDS_WHOLE_SPECTRUM":
        if sid == "S4.eigenspaces":
            return ("a certified complete window (S3) is needed to bound the separations: see the next_action of "
                    "S3.completeness")
        return NEXT_ACTION_BY_CODE[code]
    if code == "UNSUPPORTED_CLAIM":
        if "symmetry-adapted" in text:
            return "call certify_ed.exact_recheck_sector_lowest on the sector block directly"
        return "use kind='lowest' (the exact re-check is implemented for 'lowest' claims)"
    if code == "NARROWED_BY_USER":
        if "inertia counting disabled" in text:
            return "set Options(inertia=True) (the lab-minimal preset switches it off)"
        if sid == "S4.eigenspaces" and ("not requested" in text or ctx.get("eigenspaces") == "none"):
            return "set Options(eigenspaces='clusters') or 'all' (the preset in use sets 'none')"
        if sid == "S1.model_fidelity" and "disabled" in text:
            return "set Options(model_fidelity=True)"
        return NEXT_ACTION_BY_CODE[code]
    if code == "BUDGET_MEMORY":
        if "did not fit in memory" in text:
            return ("certify one U(1) sector (Claim.states) or a symmetry-adapted block, or run on a machine with more "
                    "memory: the reference matrix built from the spec did not fit")
        if "not attempted to completion" in text:
            return ("the factorization was not attempted at this size: set Options(inertia_method='sparse'), raise the "
                    "memory available, or certify a symmetry sector")
        return NEXT_ACTION_BY_CODE[code]
    if code == "BUDGET_TIME":
        sec = ctx.get("budget_seconds")
        base = NEXT_ACTION_BY_CODE[code]
        return base if sec is None else f"{base}; this run's budget was {sec:g} s"
    if code == "NEEDS_EIGENVECTORS" and sid == "S2.eigenpairs":
        return (NEXT_ACTION_BY_CODE[code] + "; without vectors only inertia counts (kind='window' with a stored matrix) "
                "say anything about the spectrum")
    if code == "DEPENDENCY_NARROWED":
        nar = (stmt.get("evidence") or {}).get("narrowed") or {}
        return f"also select the statement(s) it needs: {nar.get('reason', 'see the summary')}"
    return NEXT_ACTION_BY_CODE.get(code) or _GENERIC.get(
        sid, "inspect the evidence of this statement; docs/SCOPE_MATRIX.md lists what each statement needs")


def next_action(stmt: dict, context: Optional[dict] = None) -> Optional[str]:
    """The next action for one statement record, or None when its status needs none (CERTIFIED, FAILED,
    CORROBORATED). Never returns an empty string for INCONCLUSIVE / NOT_CHECKED."""
    ctx = context or {}
    st = stmt.get("status")
    if st == "INCONCLUSIVE":
        out = _inconclusive(stmt, ctx)
    elif st == "NOT_CHECKED":
        out = _not_checked(stmt, ctx)
    else:
        return None
    return out or _GENERIC.get(stmt["id"], "inspect the evidence of this statement")


def annotate_next_actions(statements: List[dict], context: Optional[dict] = None) -> List[dict]:
    """Write evidence['next_action'] into every INCONCLUSIVE / NOT_CHECKED statement (in place; returns the list)."""
    for s in statements:
        act = next_action(s, context)
        if act is not None:
            ev = s.get("evidence")
            if not isinstance(ev, dict):
                ev = s["evidence"] = {}
            ev["next_action"] = act
    return statements


# ---- sub-records of S4 (3.2): entanglement records (C-03) carry their own next_action, keyed on reason_code and
# on the wording of the reason written by entanglement.py
def entanglement_next_action(rec: dict) -> str:
    """Next action for a NOT_CHECKED entanglement record (entanglement.certify_entanglement)."""
    code = rec.get("reason_code")
    text = str(rec.get("reason", "")).lower()
    if code == "NEEDS_SPEC":
        return NEXT_ACTION_BY_CODE["NEEDS_SPEC"]
    if code == "UNSUPPORTED_CLAIM":
        return ("certify the eigenvectors in the computational basis (full space, or a U(1) block through Claim.states) "
                "instead of the symmetry-adapted block; the entanglement statement is not lifted through sector isometries")
    if code == "SIZE_LIMIT":
        return ("choose a smaller subsystem A, or raise the limit: Options(entanglement={name: {'sites': [...], "
                "'max_dim_a': d}}) (the exact reduced-density stage costs about d_A^3 rational operations)")
    if "fannes-audenaert range" in text:
        return (TIGHTEN + "; the trace-distance bound T = s (+ eta/(1 - eta) for a cluster) must be <= 1 - 1/d_A: a smaller "
                "residual or a larger certified separation lowers s = rho/separation")
    if "proper subset" in text or "non-empty subset" in text:
        return "declare A as a non-empty proper subset of the sites: Options(entanglement={name: {'sites': [...]}})"
    if "eta >= 1" in text:
        return REORTHO
    return "inspect the entanglement record's reason; docs/SCOPE_MATRIX.md (entanglement row) lists what the statement needs"


# ---- standalone statement records (3.2): variational records (E-05, variational.py), generalized eigenproblem records
# (E-03, generalized.py) and moment records (C-07, moments.py) carry their own next_action per statement
def record_next_action(stmt: dict) -> Optional[str]:
    """Next action for a statement of a standalone record (status not CERTIFIED): keyed on the reason code, then on
    the wording of the summary. None for CERTIFIED and DECLARED-with-nothing-to-do statements."""
    st = stmt.get("status")
    ev = stmt.get("evidence") or {}
    code = ev.get("reason_code")
    text = str(stmt.get("summary", "")).lower()
    sid = str(stmt.get("id", ""))
    if st == "CERTIFIED":
        return None
    if st == "CERTIFIED_UNDER_HYPOTHESIS":
        names = ", ".join(h.get("name", "?") for h in stmt.get("hypotheses", []))
        return (f"the bound rests on the declared hypothesis ({names}); to remove it, certify the gap with an ED run of the "
                "same operator (certify-ed certify, kind 'lowest') and pass the certificate as gap_certificate / --gap-certificate")
    if code == "DECLARED_ESTIMATE":
        return NEXT_ACTION_BY_CODE["DECLARED_ESTIMATE"]
    if code in ("UNDERFLOW_RISK", "FP_GUARD_RANGE"):
        return NEXT_ACTION_BY_CODE[code]
    if code == "SIZE_LIMIT":
        if "contracted vector" in text or sid.startswith("V"):
            return ("the MPS contracts to a vector above the 2^22 limit: contract it yourself to a dense vector on a machine "
                    "that holds it, or reduce the system; a residual bound without the dense vector is not implemented")
        return "raise the dimension limit named in the summary (max_dim), or restrict to a smaller block"
    if code == "NEEDS_DEPENDENCY" and sid.startswith("V3"):
        return ("declare a lower bound on lambda_2 with its source (gap_hypothesis, gap_source) or supply a certificate of an "
                "ED run on the same operator (gap_certificate) for a certified Temple bound")
    if code == "NEEDS_EIGENVECTORS":
        return "supply the trial vector (dense, MPS tensors or sparse CI amplitudes); declared estimates alone certify nothing"
    if code == "UNSUPPORTED_MODEL":
        return "the statement needs a Hermitian A and a positive definite B (certified by inertia); see the record's reason"
    if code == "NOT_APPLICABLE":
        if "theta < rho" in text:
            return ("improve the trial state (lower its Rayleigh quotient below the gap bound rho) or supply a larger certified "
                    "rho; Temple's inequality needs theta < rho <= lambda_2")
        if "too small to resolve" in text:
            return "rescale the trial state (multiply by a power of two) so that its squared norm is a normal binary64 number"
        if "non-finite" in text:
            return "remove the non-finite entries from the trial state"
        if "no producer estimates" in text:
            return None
        return "the statement does not apply to this input; see the record's reason"
    if code in NEXT_ACTION_BY_CODE:
        return NEXT_ACTION_BY_CODE[code]
    return "inspect the record's reason; docs/SCOPE_MATRIX.md lists what the statement needs"


def annotate_record(rec: dict) -> dict:
    """Write evidence['next_action'] into every non-CERTIFIED statement of a standalone record (in place)."""
    for stmt in rec.get("statements", []):
        na = record_next_action(stmt)
        if na is not None:
            stmt.setdefault("evidence", {})["next_action"] = na
    return rec
