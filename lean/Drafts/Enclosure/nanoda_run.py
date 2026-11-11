"""Drafts/Enclosure/nanoda_run.py: nanoda runs of the enclosure library, one theorem per run.

Usage, from reverification_lean/:
    python Drafts/Enclosure/nanoda_run.py <toolchain bin> <LEAN_PATH> <results.jsonl> <work dir> '<json list of [module, theorem]>'

For each pair, leanexport writes the theorem and every declaration it depends on, and nanoda_bin checks the export with
only propext, Classical.choice and Quot.sound permitted. A run is stopped above 4 GiB of committed memory or 30 minutes.
The runs of the enclosure library are recorded in results/mathlib_enclosure.json (second_checker).
"""
import ctypes, ctypes.wintypes as wt, json, os, re, subprocess, sys, time
LB, LEAN_PATH, OUTJ, WORK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
jobs = json.loads(sys.argv[5])  # list of [module, theorem]
MEM_LIMIT = 4 * 2**30; TIME_LIMIT = 1800
AX = ["propext", "Classical.choice", "Quot.sound"]
class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]
k32 = ctypes.WinDLL("kernel32", use_last_error=True); psapi = ctypes.WinDLL("psapi", use_last_error=True)
k32.OpenProcess.restype = wt.HANDLE
def meminfo(pid):
    h = k32.OpenProcess(0x1000 | 0x0010, False, pid)
    if not h: return None
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb); k32.CloseHandle(h)
    return (c.PeakWorkingSetSize, c.PeakPagefileUsage, c.WorkingSetSize, c.PagefileUsage) if ok else None
env = dict(os.environ); env["LEAN_PATH"] = LEAN_PATH; env["PATH"] = LB + os.pathsep + env.get("PATH", "")
def run(args, stdout_path=None):
    t0 = time.time(); out = open(stdout_path, "w", encoding="utf-8") if stdout_path else subprocess.PIPE
    errf = open(os.path.join(WORK, "stderr.txt"), "w", encoding="utf-8")
    p = subprocess.Popen(args, env=env, stdout=out, stderr=errf, text=True)
    peak_ws = peak_pf = 0; killed = None
    while p.poll() is None:
        m = meminfo(p.pid)
        if m: peak_ws = max(peak_ws, m[0]); peak_pf = max(peak_pf, m[1])
        if peak_pf > MEM_LIMIT: p.kill(); killed = "memory"; break
        if time.time() - t0 > TIME_LIMIT: p.kill(); killed = "time"; break
        time.sleep(0.5)
    p.wait()
    so = "" if stdout_path else p.stdout.read()
    if stdout_path: out.close()
    errf.close(); se = open(os.path.join(WORK, "stderr.txt"), encoding="utf-8", errors="replace").read()
    return dict(code=p.returncode, killed=killed, seconds=round(time.time() - t0, 1),
                peak_working_set_mib=round(peak_ws / 2**20), peak_private_mib=round(peak_pf / 2**20),
                text=((so or "") + se)[-1500:])
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else []
for mod, thm in jobs:
    exp = os.path.join(WORK, "export.ndjson")
    e = run([os.path.join(LB, "leanexport.exe"), mod, "--", thm], exp)
    row = dict(theorem=thm, module=mod, export=dict((k, v) for k, v in e.items() if k != "text"))
    if e["code"] != 0 or e["killed"] or "PANIC" in e["text"]:
        row.update(result="not run", why="export failed or stopped", output=e["text"])
    else:
        row["export_bytes"] = os.path.getsize(exp)
        cfg = os.path.join(WORK, "config.json")
        json.dump({"export_file_path": exp, "use_stdin": False, "permitted_axioms": AX,
                   "unpermitted_axiom_hard_error": True, "nat_extension": True, "string_extension": True,
                   "print_success_message": True}, open(cfg, "w"), indent=1)
        c = run([os.path.join(LB, "nanoda_bin.exe"), cfg])
        row["check"] = dict((k, v) for k, v in c.items() if k != "text")
        m = re.search(r"Checked (\d+) declarations with no typechecker errors", c["text"])
        row["declarations"] = int(m.group(1)) if m else 0
        row["output"] = c["text"]
        if c["killed"]: row.update(result="not run", why="nanoda stopped at the %s limit" % c["killed"])
        elif c["code"] == 0 and m: row["result"] = "accepted"
        else: row["result"] = "rejected"
        os.remove(cfg)
    if os.path.exists(exp): os.remove(exp)
    res.append(row); json.dump(res, open(OUTJ + ".tmp", "w"), indent=1); os.replace(OUTJ + ".tmp", OUTJ)
    print(thm, row.get("result"), row["export"]["seconds"], row.get("check", {}).get("seconds"), flush=True)
