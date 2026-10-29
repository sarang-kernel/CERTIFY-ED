"""Run a checker on every case of the planted-error corpus and score it (roadmap B-07).

    python corpus/run_corpus.py                                   # default checker: certify-ed of this checkout
    python corpus/run_corpus.py --checker "mychecker {case} --out {out}" --name mychecker
    python corpus/run_corpus.py --cases corpus/cases --results corpus/results/mychecker --timeout 600

The checker command is a template; `{case}` is replaced by the case directory (a producer bundle,
docs/producer_bundle.md), `{out}` by the path stem where the checker must write its result (`{out}.json` or
`{out}`), `{python}` by the running interpreter. The result file is either a CERTIFY-ED certificate
(`statements: [{id, status}, ...]`) or a flat object `{"S1.model_fidelity": "CERTIFIED", ...}`; statement ids
may be the long form or `S1` .. `S8`; statuses are CERTIFIED, FAILED, INCONCLUSIVE, NOT_CHECKED (CORROBORATED
is read as NOT_CHECKED: it is evidence, not a certified statement). A checker that exits non-zero without
writing a result has *refused* the case.

Scoring, per case, against `expected.json` (corpus/README.md):

  unsound      a statement in must_not_certify is CERTIFIED (the error was certified as correct)
  detected     no unsound statement and at least one must_not_certify statement is FAILED or INCONCLUSIVE
  silent       no unsound statement but every must_not_certify statement is NOT_CHECKED (the checker did not look)
  refused      the checker refused the input; counts as detected when expected.refusal_ok is true, else as error
  conditional  a must_not_certify statement is CERTIFIED on a case whose expected.json says a checker with a
               stronger model may do so; listed separately, never counted as unsound or detected
  error        the checker crashed or wrote nothing readable (and refusal is not an acceptable answer)
  false alarm  a statement in must_certify is FAILED (counted per statement); INCONCLUSIVE on a must_certify
               statement is a weak alarm (counted separately)
  control      a case without planted error: clean when no false or weak alarm

The summary line is `detected D / missed M (unsound U, silent S, error E) / false alarms F (weak W) of N
planted cases, C controls (clean K)`. The Markdown report lists every case with the statuses the checker gave.
This script reads bundles and result files and runs a command; it does not compute a bound.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LONG = {"S1": "S1.model_fidelity", "S2": "S2.eigenpairs", "S3": "S3.completeness", "S4": "S4.eigenspaces",
        "S5": "S5.corroboration", "S6": "S6.exact_recheck", "S7": "S7.thermodynamics", "S8": "S8.degeneracy"}
STATUSES = ("CERTIFIED", "FAILED", "INCONCLUSIVE", "NOT_CHECKED")
DEFAULT_CHECKER = '"{python}" -m certify_ed.cli certify --from "{case}" -o "{out}"'


def _norm_id(sid: str) -> str:
    sid = str(sid).strip()
    return LONG.get(sid, sid)


def _norm_status(st) -> str:
    s = str(st).strip().upper().replace(" ", "_")
    if s == "CORROBORATED":
        return "NOT_CHECKED"
    return s if s in STATUSES else f"UNKNOWN({st})"


def read_result(stem: str):
    """statement id -> status from `{stem}.json` or `{stem}`; None when no readable file exists."""
    for p in (stem + ".json", stem):
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                return None
            if isinstance(doc, dict) and isinstance(doc.get("statements"), list):
                return {_norm_id(s.get("id")): _norm_status(s.get("status")) for s in doc["statements"]
                        if isinstance(s, dict)}
            if isinstance(doc, dict):
                return {_norm_id(k): _norm_status(v) for k, v in doc.items() if _norm_id(k) in LONG.values()}
            return None
    return None


def score(expected: dict, statuses, refused: bool, crashed: bool) -> dict:
    must_not = [_norm_id(s) for s in expected.get("must_not_certify", [])]
    must = [_norm_id(s) for s in expected.get("must_certify", [])]
    control = expected.get("error_class") == "correct" or not must_not
    res = {"verdict": None, "unsound": [], "flagged": [], "false_alarms": [], "weak_alarms": [], "conditional": []}
    if statuses is None:
        if refused and expected.get("refusal_ok"):
            res["verdict"] = "refused"
        elif refused or crashed:
            res["verdict"] = "error"
        else:
            res["verdict"] = "error"
        return res
    for sid in must_not:
        st = statuses.get(sid, "NOT_CHECKED")
        if st == "CERTIFIED":
            (res["conditional"] if expected.get("conditional") else res["unsound"]).append(sid)
        elif st in ("FAILED", "INCONCLUSIVE"):
            res["flagged"].append(sid)
    for sid in must:
        st = statuses.get(sid, "NOT_CHECKED")
        if st == "FAILED":
            res["false_alarms"].append(sid)
        elif st == "INCONCLUSIVE":
            res["weak_alarms"].append(sid)
    if control:
        res["verdict"] = "control"
    elif res["unsound"]:
        res["verdict"] = "unsound"
    elif res["flagged"]:
        res["verdict"] = "detected"
    elif res["conditional"] and not res["flagged"]:
        res["verdict"] = "conditional"
    else:
        res["verdict"] = "silent"
    return res


def run_case(case_dir: str, checker: str, results_dir: str, timeout: float, env: dict) -> dict:
    cid = os.path.basename(case_dir.rstrip("/\\"))
    with open(os.path.join(case_dir, "expected.json"), encoding="utf-8") as f:
        expected = json.load(f)
    stem = os.path.join(results_dir, cid, "result")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    cmd = checker.format(case=case_dir, out=stem, python=sys.executable)
    t0 = time.time()
    try:
        proc = subprocess.run(shlex.split(cmd, posix=(os.name != "nt")) if os.name != "nt" else cmd, cwd=ROOT,
                              capture_output=True, text=True, timeout=timeout, env=env, shell=(os.name == "nt"))
        rc, out, err, timed_out = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as e:
        rc, out, err, timed_out = None, (e.stdout or ""), (e.stderr or ""), True
    seconds = time.time() - t0
    # no machine paths in the record: the case directory and the repository root become relative names
    for full, short in ((os.path.abspath(case_dir), f"corpus/cases/{cid}"), (os.path.abspath(results_dir), "results"),
                        (ROOT, ".")):
        out, err = out.replace(full, short), err.replace(full, short)
    statuses = read_result(stem)
    refused = statuses is None and rc not in (0, None)
    crashed = statuses is None and (rc in (0, None) or timed_out)
    sc = score(expected, statuses, refused, crashed)
    with open(os.path.join(results_dir, cid, "checker_output.txt"), "w", encoding="utf-8") as f:
        f.write(f"$ {cmd}\nexit code: {rc}{' (timeout)' if timed_out else ''}\n--- stdout\n{out}\n--- stderr\n{err}\n")
    return {"case": cid, "model": expected.get("model"), "error_class": expected.get("error_class"),
            "claim_kind": expected.get("claim_kind"), "k": (expected.get("parameters") or {}).get("k"),
            "must_not_certify": expected.get("must_not_certify", []),
            "must_certify": expected.get("must_certify", []), "conditional_case": bool(expected.get("conditional")),
            "refusal_ok": bool(expected.get("refusal_ok")), "exit_code": rc, "timed_out": timed_out,
            "seconds": round(seconds, 2), "statuses": statuses, "stderr_tail": err[-400:] if statuses is None else "",
            **sc}


def summarize(rows):
    planted = [r for r in rows if r["verdict"] != "control"]
    controls = [r for r in rows if r["verdict"] == "control"]
    n = {v: sum(1 for r in planted if r["verdict"] == v)
         for v in ("detected", "unsound", "silent", "refused", "error", "conditional")}
    fa = sum(len(r["false_alarms"]) for r in rows)
    wa = sum(len(r["weak_alarms"]) for r in rows)
    clean = sum(1 for r in controls if not r["false_alarms"] and not r["weak_alarms"])
    detected = n["detected"] + n["refused"]
    missed = n["unsound"] + n["silent"] + n["error"]
    return {"planted": len(planted), "controls": len(controls), "controls_clean": clean, "detected": detected,
            "detected_by_refusal": n["refused"], "missed": missed, "unsound": n["unsound"], "silent": n["silent"],
            "error": n["error"], "conditional": n["conditional"], "false_alarms": fa, "weak_alarms": wa,
            "line": (f"detected {detected} / missed {missed} (unsound {n['unsound']}, silent {n['silent']}, error "
                     f"{n['error']}) / false alarms {fa} (weak {wa}) of {len(planted)} planted cases, "
                     f"{len(controls)} controls (clean {clean}); conditional {n['conditional']}")}


def _short(sid):
    return sid.split(".")[0]


def markdown(rows, summary, name, checker, started) -> str:
    lines = [f"# Planted-error corpus: {name}", "",
             f"Checker command: `{checker}`  ", f"Run started {started}; {len(rows)} cases.", "",
             f"**{summary['line']}**", "",
             "Verdicts: *detected* (a statement the error falsifies is FAILED or INCONCLUSIVE), *unsound* (such a "
             "statement is CERTIFIED), *silent* (all of them NOT_CHECKED), *refused* (input rejected; accepted as a "
             "detection where expected.json allows it), *conditional* (CERTIFIED on a case that allows it under a "
             "stronger model), *error* (no readable result). False alarm: FAILED on a statement a correct checker "
             "certifies; weak alarm: INCONCLUSIVE on one.", "",
             "| case | class | k | verdict | S1 | S2 | S3 | S4 | S8 | notes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = r["statuses"] or {}
        cells = [st.get(LONG[s], "-") if r["statuses"] is not None else "-" for s in ("S1", "S2", "S3", "S4", "S8")]
        notes = []
        if r["unsound"]:
            notes.append("UNSOUND: " + ", ".join(_short(s) for s in r["unsound"]))
        if r["false_alarms"]:
            notes.append("false alarm: " + ", ".join(_short(s) for s in r["false_alarms"]))
        if r["weak_alarms"]:
            notes.append("weak alarm: " + ", ".join(_short(s) for s in r["weak_alarms"]))
        if r["conditional"]:
            notes.append("conditional: " + ", ".join(_short(s) for s in r["conditional"]))
        if r["statuses"] is None:
            tail = (r.get("stderr_tail") or "").strip()
            notes.append(f"exit {r['exit_code']}" + (" timeout" if r["timed_out"] else "") +
                         (f": {tail.splitlines()[-1][:160]}" if tail else ""))
        k = r.get("k")
        lines.append(f"| {r['case']} | {r['error_class']} | {k if k is not None else ''} | **{r['verdict']}** | "
                     + " | ".join(cells) + f" | {'; '.join(notes)} |")
    by_class = {}
    for r in rows:
        by_class.setdefault(r["error_class"], []).append(r["verdict"])
    lines += ["", "## By error class", "", "| class | cases | verdicts |", "|---|---|---|"]
    for cls, vs in by_class.items():
        counts = {}
        for v in vs:
            counts[v] = counts.get(v, 0) + 1
        lines.append(f"| {cls} | {len(vs)} | " + ", ".join(f"{v} {c}" for v, c in sorted(counts.items())) + " |")
    lines += ["", f"Total checker time {sum(r['seconds'] for r in rows):.0f} s."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checker", default=DEFAULT_CHECKER, help="command template with {case} {out} [{python}]")
    p.add_argument("--name", default=None, help="name of the checker in the report (default from the command)")
    p.add_argument("--cases", default=os.path.join(HERE, "cases"))
    p.add_argument("--results", default=None, help="results directory (default corpus/results/<name>)")
    p.add_argument("--only", default=None, help="comma-separated case ids or error classes")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per case")
    a = p.parse_args(argv)
    name = a.name or ("certify-ed" if a.checker == DEFAULT_CHECKER else a.checker.split()[0])
    results = a.results or os.path.join(HERE, "results", name.replace(" ", "_"))
    os.makedirs(results, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PYTHONSAFEPATH", None)
    cases = sorted(d for d in os.listdir(a.cases) if os.path.isfile(os.path.join(a.cases, d, "expected.json")))
    if a.only:
        sel = {s.strip() for s in a.only.split(",")}
        cases = [c for c in cases if c in sel or any(c.split("_", 1)[1].startswith(s) or c.endswith(s) for s in sel)]
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for c in cases:
        row = run_case(os.path.join(a.cases, c), a.checker, results, a.timeout, env)
        rows.append(row)
        st = row["statuses"] or {}
        print(f"{c:42s} {row['verdict']:12s} " + " ".join(f"{s}={st.get(LONG[s], '-')[:4]}" for s in ("S1", "S2", "S3", "S4", "S8"))
              + (f"  exit {row['exit_code']}" if row["statuses"] is None else ""), flush=True)
    summary = summarize(rows)
    report = {"schema": "certify-ed/corpus-report/1.0", "checker": name, "command": a.checker, "started_utc": started,
              "cases_dir": os.path.abspath(a.cases), "summary": summary, "rows": rows}
    with open(os.path.join(results, "report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    with open(os.path.join(results, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(rows, summary, name, a.checker, started))
    print(summary["line"])
    print(f"report: {os.path.join(results, 'report.md')}")
    return 0 if summary["missed"] == 0 and summary["false_alarms"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
