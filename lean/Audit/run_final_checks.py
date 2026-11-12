"""Audit/run_final_checks.py: the second-checker runs made after the final build of the three libraries.

Usage, from reverification_lean/ with the environment of logs/lean_env.json:
    python Audit/run_final_checks.py <toolchain bin> <python with psutil> <out.json> [--reuse-nanoda-int05 <old.json>]

First, nanoda on `CertifyMathlib.Enclosure.FP.residual_enclosure_int05` with a memory limit of 6 GiB (the earlier run
stopped at the limit of 4 GiB during the export), through Drafts/Counting/nanoda_counting.py. Second, `leanchecker` on
every module of CertifyCore, CertifyMathlib and CertifyApps (the module list is taken from the .lean files). The exit
codes, times and outputs are written to <out.json>; nothing else is changed. With `--reuse-nanoda-int05 <old.json>`,
the nanoda record is copied from an earlier output of this script instead of being made again (for a rebuild
that did not change the module `CertifyMathlib.Enclosure.FloatUnderflow`); the copy keeps its date and says so.
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def modules():
    mods = set()
    for lib in ("CertifyCore", "CertifyMathlib", "CertifyApps"):
        mods.add(lib)
        for f in glob.glob(os.path.join(HERE, lib, "**", "*.lean"), recursive=True):
            mods.add(os.path.relpath(f, HERE)[:-5].replace(os.sep, "."))
    return sorted(mods)


def main():
    lean_bin, py, out = sys.argv[1], sys.argv[2], sys.argv[3]
    reuse = sys.argv[sys.argv.index("--reuse-nanoda-int05") + 1] if "--reuse-nanoda-int05" in sys.argv else None
    lake = os.path.join(lean_bin, "lake.exe" if os.name == "nt" else "lake")
    rec = {"date": time.strftime("%d %B %Y").lstrip("0")}
    if reuse:
        with open(reuse, encoding="utf-8") as f:
            old = json.load(f)
        rec["nanoda_int05"] = dict(old["nanoda_int05"], reused_from=os.path.relpath(reuse, HERE).replace(os.sep, "/"),
                                   reused_date=old.get("date"))
        leanchecker(lake, rec, out)
        return
    work = os.path.join(HERE, "work", "nanoda_int05")
    os.makedirs(os.path.join(work, "scratch"), exist_ok=True)
    lst = os.path.join(work, "theorems.txt")
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        f.write("CertifyMathlib.Enclosure.FloatUnderflow CertifyMathlib.Enclosure.FP.residual_enclosure_int05\n")
    t0 = time.time()
    p = subprocess.run([py, os.path.join(HERE, "Drafts", "Counting", "nanoda_counting.py"), "--lean-bin", lean_bin,
                        "--work", os.path.join(work, "scratch"), "--theorems", lst,
                        "--out", os.path.join(work, "nanoda_int05.jsonl"), "--mem-mib", "6144"],
                       cwd=HERE, capture_output=True, text=True, errors="replace")
    rec["nanoda_int05"] = {"exit_code": p.returncode, "seconds": round(time.time() - t0, 1),
                           "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:], "mem_limit_mib": 6144}
    try:
        with open(os.path.join(work, "nanoda_int05.jsonl"), encoding="utf-8") as f:
            rec["nanoda_int05"]["records"] = [json.loads(l) for l in f if l.strip()]
    except OSError:
        rec["nanoda_int05"]["records"] = []
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    leanchecker(lake, rec, out)


def leanchecker(lake, rec, out):
    mods = modules()
    t0 = time.time()
    p = subprocess.run([lake, "env", "leanchecker", *mods], cwd=HERE, capture_output=True, text=True, errors="replace")
    rec["leanchecker_final"] = {"exit_code": p.returncode, "seconds": round(time.time() - t0, 1), "modules": len(mods),
                                "module_list": mods, "output": (p.stdout + p.stderr)[-4000:]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
