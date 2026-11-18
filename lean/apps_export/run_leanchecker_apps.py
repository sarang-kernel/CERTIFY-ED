"""Replay the modules of CertifyMathlib.Apps with leanchecker and write results/apps/leanchecker_apps.json.

leanchecker replays every declaration of the named modules (and of everything they import) through the Lean
kernel of the toolchain; it is not independent of the kernel the build used, so it raises a theorem from K to K+
only (README.md). Only the wall time is recorded.

usage (from reverification_lean/, with the toolchain's bin on PATH): python apps_export/run_leanchecker_apps.py
"""
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["CertifyMathlib.Apps.Bordered", "CertifyMathlib.Apps.Kernel", "CertifyMathlib.Apps.Criterion",
           "CertifyMathlib.Apps"]


def main():
    cmd = ["lake", "env", "leanchecker"] + MODULES
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    text = re.sub(re.escape(ROOT), "reverification_lean", out or "", flags=re.IGNORECASE)[-3000:]
    rec = {"tool": "leanchecker (toolchain lean-4.35.0-windows)",
           "command": "lake env leanchecker " + " ".join(MODULES), "modules": MODULES,
           "exit_code": p.returncode, "seconds": round(time.time() - t0, 1), "output": text.strip() or "none",
           "trust_level": "K+"}
    with open(os.path.join(ROOT, "results", "apps", "leanchecker_apps.json"), "w", encoding="utf8",
              newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print(rec["exit_code"], rec["seconds"])


if __name__ == "__main__":
    main()
