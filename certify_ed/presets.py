"""Statement selection and presets (the ffmpeg `-preset` model).

Everything is data: STATEMENTS says what each statement needs, PRESETS names sets of statements plus the
option values that make them useful. `resolve_selection` turns (preset, --only, --skip, options) into the
set of statements that will run, the statements implied by dependencies, and the statements narrowed out
with their reason. `certify()` consults the selection before every statement; a statement that is not
selected is reported NOT_CHECKED with the reason, never silently dropped, and the certificate's scope
block lists the whole decision (docs/presets.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# short id -> full id, what it needs from other statements, what it needs from the input, one line
STATEMENTS: Dict[str, dict] = {
    "S1": {"id": "S1.model_fidelity", "needs": (), "inputs": ("spec",),
           "what": "producer operator equals the declared model (entrywise, sector audit, or probes)"},
    "S2": {"id": "S2.eigenpairs", "needs": (), "inputs": ("eigenvectors",),
           "what": "each claimed value within a certified radius of a distinct eigenvalue (Thm 1-3)"},
    "S3": {"id": "S3.completeness", "needs": ("S2",), "inputs": ("eigenvectors", "matrix", "completeness claim"),
           "what": "no eigenvalue missing in the claimed window; gap to the next (Lemma 4, Cor 5, Prop 9)"},
    "S4": {"id": "S4.eigenspaces", "needs": ("S2", "S3"), "inputs": ("eigenvectors",),
           "what": "projector error, observables, correlations, structure factors, quantum numbers, entanglement (Thm 6, Prop 7, Prop 15, C-03)"},
    "S5": {"id": "S5.corroboration", "needs": (), "inputs": ("matrix",),
           "what": "recomputation by algorithmically distinct solvers (evidence, not a certified statement)"},
    "S6": {"id": "S6.exact_recheck", "needs": ("S2",), "inputs": ("spec", "eigenvectors", "python-flint"),
           "what": "S2 radii and the S3 count re-derived in integer arithmetic (Lemmas 10-12, Prop 13)"},
    "S7": {"id": "S7.thermodynamics", "needs": ("S2", "S3"), "inputs": ("whole spectrum", "temperatures"),
           "what": "interval bounds on Z, F, U, C from a complete enclosure (Prop 16; derived in docs/proofs, re-derivation pending)"},
    "S8": {"id": "S8.degeneracy", "needs": ("S2",), "inputs": ("eigenvectors", "matrix"),
           "what": "number of eigenvalues in each cluster by bracketing inertia counts (Prop 14; derived in docs/proofs, re-derivation pending)"},
}
SHORT_OF = {v["id"]: k for k, v in STATEMENTS.items()}
ORDER = tuple(STATEMENTS)

# statements that run when nothing is said: the certified core; S5/S6/S7 switch on through their options
DEFAULT_STATEMENTS = ("S1", "S2", "S3", "S4", "S8")

PRESETS: Dict[str, dict] = {
    "lab-minimal": {
        "statements": ("S1", "S2"),
        "options": {"inertia": False, "eigenspaces": "none"},
        "description": "is the matrix the declared model, and are the claimed pairs eigenpairs? (no factorization)"},
    "benchmark": {
        "statements": ("S1", "S2", "S3", "S5"),
        "options": {"corroborate": ("jacobi", "arpack"), "eigenspaces": "none"},
        "description": "a reference value for other methods: enclosures, completeness and a gap, plus corroboration"},
    "proof-lemma": {
        "statements": ("S1", "S2", "S3", "S6"),
        "options": {"exact_recheck": True, "eigenspaces": "none"},
        "description": "a number that enters a proof: S1-S3, then the exact integer re-check with no floating-point assumption"},
    "whole-spectrum": {
        "statements": ("S1", "S2", "S3", "S7"),
        "options": {"eigenspaces": "none", "whole_spectrum": True},
        "description": "every eigenvalue enclosed and counted (Prop 9); thermodynamic intervals when temperatures are given"},
    "labels": {
        "statements": ("S1", "S2", "S3", "S4", "S8"),
        "options": {"eigenspaces": "all"},
        "description": "quantum numbers, degeneracies and observables of every claimed level"},
    "full": {
        "statements": ORDER,
        "options": {"corroborate": ("jacobi", "arpack"), "exact_recheck": True, "eigenspaces": "all"},
        "description": "everything; S5-S7 run only when their inputs exist (corroboration solvers, python-flint, a whole spectrum)"},
}


@dataclass
class Selection:
    preset: Optional[str]
    wanted: List[str]                               # short ids that will be attempted, in ORDER
    implied: Dict[str, str] = field(default_factory=dict)      # short id -> "required by S3"
    narrowed: Dict[str, dict] = field(default_factory=dict)    # short id -> {"by": ..., "reason": ...}
    only: Optional[List[str]] = None
    skip: List[str] = field(default_factory=list)
    reason: str = ""

    def wants(self, sid: str) -> bool:
        return _short(sid) in self.wanted

    def narrowing_of(self, sid: str) -> Optional[dict]:
        return self.narrowed.get(_short(sid))

    def to_dict(self) -> dict:
        return {"preset": self.preset, "wanted": list(self.wanted), "implied": dict(self.implied),
                "narrowed": {k: dict(v) for k, v in self.narrowed.items()},
                "only": None if self.only is None else list(self.only), "skip": list(self.skip),
                "narrowing_reason": self.reason}


def _short(sid: str) -> str:
    sid = str(sid).strip()
    if sid in STATEMENTS:
        return sid
    if sid in SHORT_OF:
        return SHORT_OF[sid]
    up = sid.upper()
    if up in STATEMENTS:
        return up
    raise ValueError(f"unknown statement {sid!r}; known: {', '.join(ORDER)}")


def parse_statement_list(text) -> List[str]:
    """'S1,S2 s3' or a sequence -> ['S1', 'S2', 'S3'] (validated)."""
    if text is None:
        return []
    if isinstance(text, str):
        parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
    else:
        parts = list(text)
    return [_short(p) for p in parts]


def resolve_selection(preset: Optional[str] = None, only=None, skip=(), reason: str = "",
                      corroborate: bool = False, exact_recheck: bool = False, temperatures: bool = False,
                      degeneracy: bool = True) -> Selection:
    """The statements to attempt.

    base  = preset's statements, else DEFAULT_STATEMENTS plus S5/S6/S7 when their options are on;
    only  = restrict to these (dependencies are added and recorded as implied);
    skip  = remove these; statements needing a skipped one are narrowed with by='dependency'."""
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {', '.join(PRESETS)}")
    if preset is not None:
        base = set(PRESETS[preset]["statements"])
    else:
        base = set(DEFAULT_STATEMENTS)
        if corroborate:
            base.add("S5")
        if exact_recheck:
            base.add("S6")
        if temperatures:
            base.add("S7")
    if not degeneracy:
        base.discard("S8")
    only_l = parse_statement_list(only) if only is not None else None
    skip_l = parse_statement_list(skip)
    narrowed: Dict[str, dict] = {}
    implied: Dict[str, str] = {}
    if only_l is not None:
        wanted = set(only_l)
        changed = True
        while changed:
            changed = False
            for s in list(wanted):
                for d in STATEMENTS[s]["needs"]:
                    if d not in wanted:
                        wanted.add(d)
                        implied[d] = f"required by {s}"
                        changed = True
        for s in ORDER:
            if s not in wanted and s in base:
                narrowed[s] = {"by": "only", "reason": reason or f"not in --only {','.join(only_l)}"}
            elif s not in wanted and s not in base:
                narrowed[s] = {"by": "default", "reason": "not part of the default selection"}
    else:
        wanted = set(base)
        for s in ORDER:
            if s not in wanted:
                narrowed[s] = {"by": "preset" if preset else "default",
                               "reason": f"not in preset {preset}" if preset else "not part of the default selection"}
    for s in skip_l:
        if s in wanted:
            wanted.discard(s)
            narrowed[s] = {"by": "skip", "reason": reason or "skipped by the user"}
        elif s not in narrowed:
            narrowed[s] = {"by": "skip", "reason": reason or "skipped by the user"}
    # dependency closure of skips
    changed = True
    while changed:
        changed = False
        for s in list(wanted):
            missing = [d for d in STATEMENTS[s]["needs"] if d not in wanted]
            if missing:
                wanted.discard(s)
                narrowed[s] = {"by": "dependency", "reason": f"needs {', '.join(missing)}, which is not selected"}
                changed = True
    return Selection(preset, [s for s in ORDER if s in wanted], implied, narrowed, only_l, skip_l, reason)


def preset_options(name: str) -> dict:
    """Option values a preset sets (keyword arguments for Options)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; known: {', '.join(PRESETS)}")
    return dict(PRESETS[name]["options"], preset=name)


def format_presets(markdown: bool = False) -> str:
    """The presets table, for `certify-ed presets` and docs/presets.md."""
    rows = []
    for name, p in PRESETS.items():
        st = " ".join(p["statements"])
        opts = ", ".join(f"{k}={v!r}" for k, v in p["options"].items()) or "-"
        rows.append((name, st, opts, p["description"]))
    if markdown:
        out = ["| preset | statements | options set | use |", "|---|---|---|---|"]
        out += [f"| `{n}` | {s} | {o} | {d} |" for n, s, o, d in rows]
        out += ["", "| statement | needs | inputs | what it says |", "|---|---|---|---|"]
        out += [f"| {k} `{v['id']}` | {' '.join(v['needs']) or '-'} | {', '.join(v['inputs'])} | {v['what']} |"
                for k, v in STATEMENTS.items()]
        return "\n".join(out)
    w = max(len(r[0]) for r in rows)
    out = [f"{'preset':{w}s}  statements           use"]
    for n, s, o, d in rows:
        out.append(f"{n:{w}s}  {s:20s} {d}")
        out.append(f"{'':{w}s}  {'':20s} sets {o}")
    out.append("")
    out.append("statements (needs -> inputs):")
    for k, v in STATEMENTS.items():
        out.append(f"  {k} {v['id']:20s} needs {' '.join(v['needs']) or '-':6s} inputs: {', '.join(v['inputs'])}")
        out.append(f"     {v['what']}")
    out.append("")
    out.append("default (no preset): " + " ".join(DEFAULT_STATEMENTS) + " (+S5 with corroborate, +S6 with exact_recheck, "
               "+S7 with temperatures); --only S1,S2 / --skip S3 narrow; --budget-seconds/--budget-mb turn statements that "
               "would exceed the budget into NOT CHECKED (reason BUDGET)")
    return "\n".join(out)


def all_statement_ids() -> Iterable[str]:
    return (v["id"] for v in STATEMENTS.values())


def statements_of(short_ids: Sequence[str]) -> List[str]:
    return [STATEMENTS[s]["id"] for s in short_ids]
