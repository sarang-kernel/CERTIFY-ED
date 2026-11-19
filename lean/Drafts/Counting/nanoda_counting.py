"""Drafts/Counting/nanoda_counting.py

Second-kernel check of the counting track (CertifyMathlib.Counting) with nanoda, one theorem per run.

For every theorem named in the input file (one "module theorem" pair per line), the script runs
    leanexport <module> -- <theorem>  > <work>/<theorem>.ndjson
which writes the theorem together with every declaration it depends on (Lean core and Mathlib
included), then
    nanoda_bin <config.json>
with the permitted axioms propext, Classical.choice and Quot.sound (unpermitted_axiom_hard_error,
nat_extension and string_extension set to true, as in export/second_kernel.py). The peak private
memory of each process is sampled every 0.25 s; a process is stopped when it exceeds the memory
limit (default 4096 MiB) or the time limit (default 1800 s). The export file and the configuration
are removed after each run. Results are appended to a JSON lines file, one record per theorem, and a
theorem already present there with result "accepted" is skipped.

Usage, from lean/ (LEAN_PATH is taken from `lake env`):
    python Drafts/Counting/nanoda_counting.py --lean-bin <toolchain bin> --work <scratch dir>
        --theorems <list file> --out <results.jsonl> [--mem-mib 4096] [--timeout 1800]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AXIOMS = ["propext", "Classical.choice", "Quot.sound"]


def tool(lean_bin, name):
    return os.path.join(lean_bin, name + (".exe" if os.name == "nt" else ""))


def lean_path(lean_bin):
    env = dict(os.environ)
    env["PATH"] = lean_bin + os.pathsep + env.get("PATH", "")
    p = subprocess.run([tool(lean_bin, "lake"), "env", sys.executable, "-c",
                        "import os; print(os.environ.get('LEAN_PATH', ''))"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    lp = p.stdout.strip().splitlines()[-1]
    parts = [os.path.abspath(os.path.join(ROOT, q)) for q in lp.split(os.pathsep) if q]
    return os.pathsep.join(parts)


def run(args, env, timeout, mem_mib, stdout_path=None):
    t0 = time.time()
    out_f = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.PIPE
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=out_f, stderr=subprocess.PIPE, text=True)
    proc = psutil.Process(p.pid)
    peak = 0
    stopped = None
    while p.poll() is None:
        try:
            mi = proc.memory_info()
            peak = max(peak, getattr(mi, "private", mi.rss))
        except psutil.Error:
            pass
        if peak > mem_mib * 2 ** 20:
            stopped = "memory"
        elif time.time() - t0 > timeout:
            stopped = "time"
        if stopped:
            p.kill()
            break
        time.sleep(0.25)
    out, err = p.communicate()
    if stdout_path:
        out_f.close()
        out = ""
    text = ((out or "") + (err or ""))[-1500:]
    text = re.sub(re.escape(ROOT), "lean", text, flags=re.IGNORECASE)
    return dict(exit_code=p.returncode, seconds=round(time.time() - t0, 1),
                peak_private_mib=round(peak / 2 ** 20), stopped=stopped, output=text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean-bin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--theorems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-mib", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = a.lean_bin + os.pathsep + env.get("PATH", "")
    env["LEAN_PATH"] = lean_path(a.lean_bin)
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("result") == "accepted":
                    done.add(r["theorem"])
    with open(a.theorems, encoding="utf-8") as f:
        todo = [line.split() for line in f if line.strip() and not line.startswith("#")]
    for module, thm in todo:
        if thm in done:
            continue
        exp = os.path.join(a.work, thm.split(".")[-1] + ".ndjson")
        rec = dict(theorem=thm, module=module)
        e = run([tool(a.lean_bin, "leanexport"), module, "--", thm], env, a.timeout, a.mem_mib,
                stdout_path=exp)
        rec.update(export_seconds=e["seconds"], export_peak_private_mib=e["peak_private_mib"])
        size = os.path.getsize(exp) if os.path.exists(exp) else 0
        rec["export_bytes"] = size
        if e["stopped"] or e["exit_code"] != 0 or "PANIC" in e["output"] or size == 0:
            rec.update(result="not run", stage="export", reason=e["stopped"] or "export failed",
                       output=e["output"])
        else:
            cfg = os.path.join(a.work, thm.split(".")[-1] + ".config.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({"export_file_path": exp, "use_stdin": False, "permitted_axioms": AXIOMS,
                           "unpermitted_axiom_hard_error": True, "nat_extension": True,
                           "string_extension": True, "print_success_message": True}, f, indent=1)
            c = run([tool(a.lean_bin, "nanoda_bin"), cfg], env, a.timeout, a.mem_mib)
            os.remove(cfg)
            m = re.search(r"Checked (\d+) declarations with no typechecker errors", c["output"])
            ndecl = int(m.group(1)) if m else 0
            if c["stopped"]:
                result = "not run"
            elif c["exit_code"] == 0 and ndecl > 0:
                result = "accepted"
            else:
                result = "rejected"
            rec.update(result=result, declarations=ndecl, check_seconds=c["seconds"],
                       check_peak_private_mib=c["peak_private_mib"], output=c["output"])
            if c["stopped"]:
                rec["reason"] = c["stopped"]
        if os.path.exists(exp):
            os.remove(exp)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(thm, rec["result"], rec.get("export_seconds"), rec.get("check_seconds"), flush=True)


if __name__ == "__main__":
    main()
