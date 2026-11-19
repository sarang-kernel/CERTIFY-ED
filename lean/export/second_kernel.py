"""export/second_kernel.py

Checks the compiled CertifyCore library a second time, after `lake build CertifyCore cecheck`.

lean4lean   runs lean4lean.exe (a reimplementation of the Lean kernel written in Lean) on every CertifyCore
            module, one call per module, and as a control on the toolchain's own module Init.Prelude.
leanchecker runs leanchecker.exe (replay through the Lean kernel of the toolchain) once on the prefix
            CertifyCore, which covers every CertifyCore module.  It is not independent of the kernel that the
            build used.
nanoda      exports every theorem listed in results/core/axioms.txt with leanexport, together with every
            declaration it depends on, and checks the export from scratch with nanoda_bin (an independent
            implementation of the Lean type theory written in Rust), allowing only the axioms propext,
            Classical.choice and Quot.sound.

Usage (from lean/):
    python export/second_kernel.py --lean-bin <toolchain bin> [lean4lean|leanchecker|nanoda ...]
Writes results/core/second_kernel.json (merged with an existing file, so modes can be run separately).
The export files are written to work/core/second_kernel/ and removed after each check.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, ".lake", "build", "lib", "lean")
OUT = os.path.join(ROOT, "results", "core", "second_kernel.json")
WORK = os.path.join(ROOT, "work", "core", "second_kernel")
AXIOMS = ["propext", "Classical.choice", "Quot.sound"]


def modules():
    found = []
    for p in glob.glob(os.path.join(LIB, "CertifyCore", "**", "*.olean"), recursive=True):
        rel = os.path.relpath(p, LIB)[:-len(".olean")]
        found.append(rel.replace(os.sep, "."))
    return ["CertifyCore"] + sorted(found)


def theorems():
    names = []
    pat = re.compile(r"^'(.+)' (?:depends on axioms|does not depend on any axioms)")
    with open(os.path.join(ROOT, "results", "core", "axioms.txt"), encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:  # names may end in a prime, as one_pow'
                names.append(m.group(1))
    return names


def env(lean_bin):
    e = dict(os.environ)
    e["LEAN_PATH"] = LIB
    e["PATH"] = lean_bin + os.pathsep + e.get("PATH", "")
    return e


def run(args, lean_bin, timeout, stdout=None):
    t0 = time.time()
    p = subprocess.Popen(args, cwd=ROOT, env=env(lean_bin), stdout=stdout or subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
        code = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        code = "timeout"
    text = ((out or "") + (err or ""))[-1500:]
    for path, name in ((os.path.dirname(os.path.abspath(lean_bin)), "<toolchain>"), (ROOT, "lean")):
        text = re.sub(re.escape(path), name, text, flags=re.IGNORECASE)
    return code, round(time.time() - t0, 1), text


def tool(lean_bin, name):
    return os.path.join(lean_bin, name + (".exe" if os.name == "nt" else ""))


def per_module(lean_bin, name, mods, timeout):
    rows = []
    for m in mods:
        code, sec, text = run([tool(lean_bin, name), m], lean_bin, timeout)
        rows.append(dict(module=m, exit_code=code, seconds=sec, output=text))
        print(name, m, code, sec, flush=True)
    return rows


def nanoda(lean_bin, thm, timeout):
    os.makedirs(WORK, exist_ok=True)
    exp = os.path.join(WORK, thm + ".ndjson")
    with open(exp, "w", encoding="utf-8") as f:
        code, t_export, text = run([tool(lean_bin, "leanexport"), "CertifyCore", "--", thm], lean_bin, timeout,
                                   stdout=f)
    if code != 0 or "PANIC" in text:  # leanexport exits with 0 when the constant is missing, after a PANIC line
        os.remove(exp)
        return dict(theorem=thm, ok=False, stage="export", exit_code=code, export_s=t_export, output=text)
    size = os.path.getsize(exp)
    cfg = os.path.join(WORK, thm + ".config.json")
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump({"export_file_path": exp, "use_stdin": False, "permitted_axioms": AXIOMS,
                   "unpermitted_axiom_hard_error": True, "nat_extension": True, "string_extension": True,
                   "print_success_message": True}, f, indent=1)
    code, t_check, text = run([tool(lean_bin, "nanoda_bin"), cfg], lean_bin, timeout)
    os.remove(exp)
    os.remove(cfg)
    m = re.search(r"Checked (\d+) declarations with no typechecker errors", text)
    ndecl = int(m.group(1)) if m else 0
    return dict(theorem=thm, ok=code == 0 and ndecl > 0, exit_code=code, declarations=ndecl, export_s=t_export,
                check_s=t_check, export_bytes=size, output=text)


def save(out):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    os.replace(tmp, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean-bin", required=True)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("modes", nargs="*", default=["lean4lean", "leanchecker", "nanoda"])
    a = ap.parse_args()
    out = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            out = json.load(f)
    mods = modules()
    for mode in a.modes:
        if mode == "lean4lean":
            out["lean4lean"] = dict(modules=per_module(a.lean_bin, "lean4lean", mods, a.timeout),
                                    control=per_module(a.lean_bin, "lean4lean", ["Init.Prelude"], a.timeout))
        elif mode == "leanchecker":
            # one call: leanchecker replays every module whose name has the prefix CertifyCore
            row = per_module(a.lean_bin, "leanchecker", ["CertifyCore"], a.timeout)[0]
            out["leanchecker"] = dict(command="leanchecker CertifyCore", modules_covered=mods, **row)
        elif mode == "nanoda":
            done = {r["theorem"]: r for r in out.get("nanoda", []) if r.get("ok") and r.get("declarations", 0) > 0}
            rows = []
            for thm in theorems():
                rec = done.get(thm) or nanoda(a.lean_bin, thm, a.timeout)
                rows.append(rec)
                out["nanoda"] = rows + [r for r in out.get("nanoda", []) if r["theorem"] not in
                                        {x["theorem"] for x in rows}]
                save(out)
                print("nanoda", thm, rec["ok"], rec.get("check_s"), flush=True)
            out["nanoda"] = rows
        save(out)


if __name__ == "__main__":
    main()
