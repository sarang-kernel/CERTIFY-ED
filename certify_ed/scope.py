"""The scope block of a certificate, the reason codes, and the budget guard.

Every certificate says what was in scope, what the user narrowed (and why), and what is out of scope
with a reason code (docs/SCOPE_MATRIX.md). A budget (seconds, megabytes) turns statements that would
exceed it into NOT_CHECKED with reason BUDGET_TIME / BUDGET_MEMORY; it never raises.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .presets import SHORT_OF, STATEMENTS, Selection

SCOPE_SCHEMA = "certify-ed/scope/1.0"

# reason codes used in statements' evidence["reason_code"] and in scope.out_of_scope
REASON_CODES: Dict[str, str] = {
    "NARROWED_BY_USER": "the user excluded the statement (--only / --skip / preset)",
    "DEFAULT_SELECTION": "not part of the default selection; switched on by its option, a preset or --only",
    "DEPENDENCY_NARROWED": "the statement needs another statement that the user excluded",
    "BUDGET_TIME": "the time budget was spent before the statement could start",
    "BUDGET_MEMORY": "the statement's estimated memory exceeds the memory budget",
    "NEEDS_EIGENVECTORS": "the claim has eigenvalues only",
    "NEEDS_SPEC": "no model specification was declared",
    "NEEDS_MATRIX": "the operator is matrix-free and no stored reference matrix exists",
    "NEEDS_WHOLE_SPECTRUM": ("the claim is not a complete whole-spectrum claim (kind 'all', S3 certified) nor, for S7, a "
                             "'lowest' claim with S3 certified"),
    "NEEDS_TEMPERATURES": "no temperatures were given",
    "NEEDS_DEPENDENCY": "an input the statement needs was not named (a corroborating solver, h5py for a file)",
    "MISSING_DEPENDENCY": "an optional package the statement needs is not installed (python-flint for the exact "
                          "statements, QuSpin for its importer); the certificate says which extra to install",
    "NEEDS_OBSERVABLES": "no observables, correlations or structure factors were requested",
    "NO_COMPLETENESS_CLAIM": "the claim is a 'subset' claim: nothing is said about other eigenvalues",
    "NO_CLUSTERS": "no (quasi-)degenerate clusters among the claimed pairs",
    "SIZE_LIMIT": "dimension above the configured limit for this statement (inertia_max_dim, exact_max_dim, ...)",
    "UNSUPPORTED_CLAIM": "the statement is not implemented for this claim kind or sector form",
    "UNSUPPORTED_MODEL": "the model class is outside what the statement handles",
    "NOT_IMPLEMENTED": "a design slot: the mathematics is identified but not implemented",
    "OUT_OF_SCOPE_DYNAMICS": "dynamical response and time evolution are not statements about eigenpairs of one matrix",
    "OUT_OF_SCOPE_EXTRAPOLATION": "finite-size extrapolation and phase boundaries are not statements about any finite matrix",
    "NOT_APPLICABLE": "the statement does not apply to this input",
    "UNDERFLOW_RISK": ("a nonzero operand, radius or residual bound has magnitude below 2^-900: the no-underflow "
                       "hypothesis of the floating-point model is not established for this input "
                       "(Options(underflow_check=False) assumes it instead)"),
    "DECLARED_ESTIMATE": ("a producer-declared stochastic estimate (an energy or a variance from samples): recorded as "
                          "DECLARED, never certified; no statement rests on it (variational records, E-05)"),
}

# standing quantities labs report that no statement of this version certifies (docs/SCOPE_MATRIX.md)
OUT_OF_SCOPE_QUANTITIES = [
    {"quantity": "entanglement entropy and entanglement spectrum", "reason_code": "DEFAULT_SELECTION",
     "detail": ("since 3.2 (C-03) an S4 evidence record per cluster when Options.entanglement names a cut: certified "
                "interval for S(rho_A) (Fannes-Audenaert on the trace-norm bound 2T <= 2 s + 2 eta/(1 - eta) of the "
                "reduced state) and the entanglement spectrum (Weyl); not computed unless requested")},
    {"quantity": "dynamical response, spectral functions", "reason_code": "OUT_OF_SCOPE_DYNAMICS",
     "detail": "needs certified Krylov / continued-fraction error bounds; not a statement about one eigenpair set"},
    {"quantity": "time evolution", "reason_code": "OUT_OF_SCOPE_DYNAMICS",
     "detail": "needs certified propagator error bounds (Krylov/Chebyshev/Trotter)"},
    {"quantity": "level statistics", "reason_code": "NOT_IMPLEMENTED",
     "detail": ("no certificate statement; certify_ed.levelstats (3.2) computes interval r-ratios and spacing histograms "
                "from a complete counted spectrum (kind 'all', S3 CERTIFIED) after the fact, with the unfolding declared, "
                "not certified")},
    {"quantity": "finite-size extrapolation", "reason_code": "OUT_OF_SCOPE_EXTRAPOLATION",
     "detail": ("an extrapolation is a statement about a model of the sequence, not about any finite matrix; "
                "certify_ed.extrapolation_report (3.2) cites per-size certificates by hash and fits a declared model, "
                "and says that the fitted value is not certified")},
    {"quantity": "phase boundaries", "reason_code": "OUT_OF_SCOPE_EXTRAPOLATION",
     "detail": ("derived from extrapolated quantities; certified inputs at each size are what a certificate provides; "
                "certify_ed.crossings (3.2) orders two certified intervals at one size when they do not overlap")},
]

__all__ = ["SCOPE_SCHEMA", "REASON_CODES", "OUT_OF_SCOPE_QUANTITIES", "Budget", "build_scope_block",
           "classify_not_checked"]


class Budget:
    """Time and memory budget. `time_left()` says whether a statement may start; `memory_ok(mb)` compares an
    estimate with the budget. Both return (ok, note)."""

    def __init__(self, seconds: Optional[float] = None, mb: Optional[float] = None, t_start: Optional[float] = None):
        self.seconds = None if seconds is None else float(seconds)
        self.mb = None if mb is None else float(mb)
        self.t_start = time.time() if t_start is None else t_start
        self.events: List[dict] = []

    def elapsed(self) -> float:
        return time.time() - self.t_start

    def time_left(self, sid: str):
        if self.seconds is None:
            return True, ""
        el = self.elapsed()
        if el > self.seconds:
            note = f"time budget {self.seconds:g} s spent ({el:.1f} s elapsed) before {sid}"
            self.events.append({"statement": sid, "code": "BUDGET_TIME", "note": note})
            return False, note
        return True, ""

    def memory_ok(self, sid: str, estimate_mb: Optional[float], what: str = ""):
        if self.mb is None or estimate_mb is None:
            return True, ""
        if estimate_mb > self.mb:
            note = f"estimated {estimate_mb:.0f} MB for {what or sid} exceeds the memory budget {self.mb:g} MB"
            self.events.append({"statement": sid, "code": "BUDGET_MEMORY", "note": note, "estimate_mb": estimate_mb})
            return False, note
        return True, ""

    def to_dict(self) -> dict:
        return {"seconds": self.seconds, "mb": self.mb, "events": list(self.events)}


_TEXT_TO_CODE = (
    ("budget", "BUDGET_TIME"),
    ("eigenvalue-only", "NEEDS_EIGENVECTORS"),
    ("no vectors", "NEEDS_EIGENVECTORS"),
    ("no eigenvectors", "NEEDS_EIGENVECTORS"),
    ("with eigenvectors", "NEEDS_EIGENVECTORS"),
    ("no model specification", "NEEDS_SPEC"),
    ("needs a model specification", "NEEDS_SPEC"),
    ("matrix-free", "NEEDS_MATRIX"),
    ("no stored matrix", "NEEDS_MATRIX"),
    ("python-flint", "MISSING_DEPENDENCY"),
    ("not installed", "MISSING_DEPENDENCY"),
    ("subset", "NO_COMPLETENESS_CLAIM"),
    ("no clusters", "NO_CLUSTERS"),
    ("_max_dim", "SIZE_LIMIT"),
    ("above exact_max_dim", "SIZE_LIMIT"),
    ("symmetry-adapted", "UNSUPPORTED_CLAIM"),
    ("implemented for", "UNSUPPORTED_CLAIM"),
    ("disabled", "NARROWED_BY_USER"),
    ("not requested", "NARROWED_BY_USER"),
    ("not selected", "NARROWED_BY_USER"),
    ("temperatures", "NEEDS_TEMPERATURES"),
    ("whole spectrum", "NEEDS_WHOLE_SPECTRUM"),
    ("complete window", "NEEDS_WHOLE_SPECTRUM"),
    ("did not fit in memory", "BUDGET_MEMORY"),
    ("not attempted to completion", "BUDGET_MEMORY"),
)


def classify_not_checked(stmt: dict) -> str:
    """Reason code of a NOT_CHECKED statement: explicit evidence['reason_code'], else from its summary."""
    ev = stmt.get("evidence") or {}
    if ev.get("reason_code") in REASON_CODES:
        return ev["reason_code"]
    text = (stmt.get("summary", "") + " " + str(ev.get("reason", ""))).lower()
    for needle, code in _TEXT_TO_CODE:
        if needle in text:
            return code
    return "NOT_APPLICABLE"


def build_scope_block(selection: Selection, statements: List[dict], budget: Budget, options_record: dict,
                      operator_scope: str, requested: Optional[dict] = None) -> dict:
    """The mandatory scope section of a certificate."""
    by_id = {s["id"]: s for s in statements}
    in_scope, narrowed, out = [], [], []
    for short, meta in STATEMENTS.items():
        sid = meta["id"]
        st = by_id.get(sid)
        nar = selection.narrowing_of(short)
        if nar is not None:
            code = {"dependency": "DEPENDENCY_NARROWED", "default": "DEFAULT_SELECTION"}.get(nar["by"], "NARROWED_BY_USER")
            narrowed.append({"id": sid, "by": nar["by"], "reason": nar["reason"], "reason_code": code,
                             "status": st["status"] if st else "NOT_CHECKED"})
            continue
        if st is None:
            out.append({"quantity": sid, "reason_code": "NOT_APPLICABLE", "detail": "statement not reached"})
            continue
        if st["status"] == "NOT_CHECKED":
            code = classify_not_checked(st)
            out.append({"quantity": sid, "reason_code": code, "detail": st.get("summary", ""),
                        "implied": selection.implied.get(short)})
        else:
            in_scope.append({"id": sid, "status": st["status"], "implied": selection.implied.get(short)})
    requested_ent = bool((options_record or {}).get("entanglement"))
    out += [dict(q) for q in OUT_OF_SCOPE_QUANTITIES
            if not (requested_ent and q["quantity"].startswith("entanglement"))]
    return {"schema": SCOPE_SCHEMA, "preset": selection.preset, "selection": selection.to_dict(),
            "budgets": budget.to_dict(), "in_scope": in_scope, "narrowed": narrowed, "out_of_scope": out,
            "requested": requested or {}, "options": options_record, "operator_scope": operator_scope,
            "reason_codes": {k: REASON_CODES[k] for k in sorted({x["reason_code"] for x in out + narrowed})}}


def short_id(sid: str) -> str:
    return SHORT_OF.get(sid, sid)
