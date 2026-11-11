"""Audit/make_integration_record.py: collect the integration checks into results/integration.json.

Usage, from reverification_lean/:  python Audit/make_integration_record.py

Reads logs/integration/lake_build.log (output of `lake build CertifyCore CertifyMathlib CertifyApps cecheck ceapps`),
logs/integration/axiom_audit.log (output of `lake env lean Audit/AxiomAudit.lean`), logs/integration/final_checks.json
(written by Audit/run_final_checks.py), work/nanoda_final/nanoda_final.jsonl; the three log files are
copied to results/integration_logs/ (logs/ is not tracked); it reads (nanoda on the main theorems of the
modules rebuilt by the final build) and searches the library files for forbidden tokens.
"""
import glob
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT = os.path.join(HERE, "logs", "integration")
LIBS = ("CertifyCore", "CertifyMathlib", "CertifyApps")
TOKENS = {"sorry": r"\bsorry\b", "admit": r"\badmit\b", "native_decide": r"\bnative_decide\b",
          "axiom": r"^\s*(private\s+|protected\s+)?axiom\b", "skipKernelTC": r"skipKernelTC",
          "debug options": r"set_option\s+debug\.", "unsafe": r"\bunsafe\b", "implemented_by": r"implemented_by",
          "extern": r"@\[\s*extern", "decide +kernel": r"decide\s*\+kernel"}


def text_scan():
    out = {}
    for lib in LIBS:
        files = [os.path.join(HERE, lib + ".lean")] + glob.glob(os.path.join(HERE, lib, "**", "*.lean"), recursive=True)
        row = {"files": len(files), "lines": 0}
        row.update({k: 0 for k in TOKENS})
        for f in files:
            t = open(f, encoding="utf-8").read()
            row["lines"] += t.count("\n")
            t = re.sub(r"/-.*?-/", "", t, flags=re.S)
            t = re.sub(r"--[^\n]*", "", t)
            for k, p in TOKENS.items():
                row[k] += len(re.findall(p, t, flags=re.M))
        out[lib] = row
    return out


def main():
    keep = os.path.join(HERE, "results", "integration_logs")
    os.makedirs(keep, exist_ok=True)
    for name in ("lake_build.log", "axiom_audit.log", "final_checks.json"):
        shutil.copy2(os.path.join(INT, name), os.path.join(keep, name))
    log = open(os.path.join(INT, "lake_build.log"), encoding="utf-8", errors="replace").read().split("\n")
    warns = [l for l in log if l.startswith("warning:")]
    jobs = [re.search(r"Build completed successfully \((\d+) jobs\)", l) for l in log]
    jobs = [m.group(1) for m in jobs if m]
    build = {"command": "lake build CertifyCore CertifyMathlib CertifyApps cecheck ceapps",
             "exit_code": 0 if jobs else 1, "jobs": int(jobs[-1]) if jobs else None,
             "errors": sum(1 for l in log if l.startswith("error:") or ": error" in l),
             "warnings_module_system": sum(1 for l in warns if "designed for use with the module system" in l),
             "warnings_other": sum(1 for l in warns if "designed for use with the module system" not in l),
             "log": "results/integration_logs/lake_build.log"}
    audit = []
    for l in open(os.path.join(INT, "axiom_audit.log"), encoding="utf-8", errors="replace"):
        m = re.match(r"AUDIT (\S+): theorems=(\d+) auxiliary=(\d+) axioms=\[(.*)\] nonstandard=(\d+)", l.strip())
        if m:
            audit.append({"library": m.group(1), "theorems": int(m.group(2)), "auxiliary": int(m.group(3)),
                          "axioms": [a.strip() for a in m.group(4).split(",") if a.strip()],
                          "nonstandard": int(m.group(5))})
    fc = json.load(open(os.path.join(INT, "final_checks.json"), encoding="utf-8"))
    lc = fc["leanchecker_final"]
    shutil.copy2(os.path.join(HERE, "work", "nanoda_final", "nanoda_final.jsonl"), os.path.join(keep, "nanoda_recheck.jsonl"))
    nrecs = [json.loads(l) for l in open(os.path.join(keep, "nanoda_recheck.jsonl"), encoding="utf-8") if l.strip()]
    ni = fc.get("nanoda_int05", {})
    rec = {
        "date": fc.get("date"),
        "lean": "4.35.0",
        "mathlib": "v4.35.0 (commit 56a005855ea1)",
        "build": build,
        "axiom_audit": {"command": "lake env lean Audit/AxiomAudit.lean", "log": "results/integration_logs/axiom_audit.log",
                        "libraries": audit},
        "text_scan": text_scan(),
        "leanchecker_final": {"exit_code": lc["exit_code"], "seconds": lc["seconds"], "modules": lc["modules"],
                              "module_list": lc["module_list"], "output": lc["output"]},
        "nanoda_recheck": {"scope": "the main theorems of CertifyMathlib.Counting.D01Float and P15Float, which the final "
                                    "build rebuilt because enclosure modules they import had changed",
                           "theorems": len(nrecs), "accepted": sum(1 for r in nrecs if r.get("result") == "accepted"),
                           "records": nrecs},
        "nanoda_int05": {"theorem": "CertifyMathlib.Enclosure.FP.residual_enclosure_int05",
                         "mem_limit_mib": ni.get("mem_limit_mib"), "exit_code": ni.get("exit_code"),
                         "records": ni.get("records", [])},
    }
    with open(os.path.join(HERE, "results", "integration.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: rec[k] for k in ("build", "axiom_audit")}, indent=1)[:2000])


if __name__ == "__main__":
    main()
