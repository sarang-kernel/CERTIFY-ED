"""Run every application target of ceapps, measure time and peak memory, and write results/apps/.

Each target is one run of the compiled Lean program `ceapps`; the program reads the run record, recomputes
the certified quantity and decides agreement itself (exact comparisons in Lean). This script only starts the
runs, measures wall time and the peak working set of the process (Windows: GetProcessMemoryInfo; elsewhere:
ru_maxrss of the child), copies the result files, checks that the two evidence runs hold the same values for
the compared fields, and collects summary.json.

usage: python run_all.py --exe <ceapps.exe> --archive <sources/archive_v5> --out <results/apps> --work <work/apps>
                         [--only id ...] [--collect-only]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

RUN_NEW = "code/certify-ed/results/run_full_20261105"
RUN_OLD = "code/certify-ed/results/run_full_20261022"
REC = {
    "e7": ("e7_exact_benchmarks_20261105_092337", "e7_exact_benchmarks_20261022_193038"),
    "e9": ("e9_pyrochlore_20261105_084835", "e9_pyrochlore_20261022_185936"),
    "e10": ("e10_aklt_20261105_071106", "e10_aklt_20261022_170609"),
    "e10k": ("e10_aklt_20261105_071106", "e10_aklt_20261022_170609"),
}

# id, record group, ceapps arguments (record path and output path are filled in)
TARGETS = [
    ("e7_chain_J2_0", "e7", ["e7", "chain_J2=0", "{rec}", "{out}", "1"]),
    ("e7_chain_J2_0.5", "e7", ["e7", "chain_J2=0.5", "{rec}", "{out}", "1"]),
    ("e7_square_J2_0", "e7", ["e7", "square_J2=0", "{rec}", "{out}", "1"]),
    ("e7_square_J2_0.5", "e7", ["e7", "square_J2=0.5", "{rec}", "{out}", "1"]),
    ("e9_crown_spin_half_lower_bound", "e9", ["crown-half", "{rec}", "{out}", "1"]),
    ("e9_crown_spin_one_upper_bound", "e9", ["crown-one", "{rec}", "{out}"]),
    ("e10_star", "e10", ["aklt", "star", "{rec}", "{out}", "1"]),
    ("e10_square_octagon_perp", "e10", ["aklt", "square_octagon_perp", "{rec}", "{out}", "1"]),
    ("e10_square_octagon_par", "e10", ["aklt", "square_octagon_par", "{rec}", "{out}", "1"]),
    ("e10_cross", "e10", ["aklt", "cross", "{rec}", "{out}", "1"]),
    ("e10_decorated_honeycomb", "e10", ["aklt", "decorated_honeycomb", "{rec}", "{out}", "1"]),
    ("e10_decorated_square", "e10", ["aklt", "decorated_square", "{rec}", "{out}", "1"]),
    ("e10_kernels", "e10k", ["aklt-kernels", "{rec}", "{out}"]),
]


def peak_memory_windows(proc):
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(wintypes.HANDLE(int(proc._handle)), ctypes.byref(c), c.cb)
    return int(c.PeakWorkingSetSize) if ok else None


def run_target(exe, args, log_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf8") as log:
        proc = subprocess.Popen([exe] + args, stdout=log, stderr=subprocess.STDOUT)
        rc = proc.wait()
        peak = peak_memory_windows(proc) if os.name == "nt" else None
    if peak is None and os.name != "nt":
        import resource
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    return rc, time.time() - t0, peak


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if "second" not in k}
    if isinstance(x, list):
        return [strip_timing(v) for v in x]
    return x


def records_identical(archive):
    """Exact equality, timing fields excluded, of the record fields the targets use, in the two evidence runs."""
    out = {}
    for grp, (new, old) in REC.items():
        a = json.load(open(os.path.join(archive, RUN_NEW, new, "data.json")))
        b = json.load(open(os.path.join(archive, RUN_OLD, old, "data.json")))
        keys = {"e7": ["E7"], "e9": ["spin_half", "spin1_printed"], "e10": ["lattices", "square_octagon"],
                "e10k": ["kernel"]}[grp]
        out[grp] = all(strip_timing(a[k]) == strip_timing(b[k]) for k in keys)
    return out


def headline(tid, r):
    """Record value, Lean value and agreement of a result file, for the summary table."""
    def ex(x):
        return None if x is None else x.get("exact")

    def dec(x):
        return None if x is None else x.get("decimal")
    if tid.startswith("e7"):
        v = r.get("vector") or {}
        return {"quantity": "lowest eigenvalue of the block enclosed around the recorded E0 with the recorded exact radius; "
                            "eigenvalues below the recorded count point",
                "record_value": {"E0": dec(r["record"]["E0"]), "exact_radius": dec(r["record"]["exact_radius"]),
                                 "count": r["record"]["count"], "delta": dec(r["record"]["delta"])},
                "lean_value": {"rayleigh_quotient": dec(v.get("rayleigh_quotient_rounded")), "radius": dec(v.get("radius")),
                               "radius_at_record_E0": dec(v.get("radius_at_record_E0")), "count": r["count_total"],
                               "delta": dec(r["delta_max"]), "status": r["lean_status"]},
                "agreement": r["agreement"]}
    if tid == "e9_crown_spin_half_lower_bound":
        return {"quantity": "certified lower bound on lambda_min of the reduced spin-1/2 crown",
                "record_value": {"lower_bound": dec(r["record"]["lower_bound"]),
                                 "lower_bound_exact": ex(r["record"]["lower_bound_exact"]),
                                 "delta": dec(r["record"]["delta"])},
                "lean_value": {"lower_bound": dec(r["lower_bound_lean"]), "lower_bound_exact": ex(r["lower_bound_lean"]),
                               "delta": dec(r["delta_max"]), "count": r["count_total"], "status": r["lean_status"]},
                "agreement": r["agreement"]}
    if tid == "e9_crown_spin_one_upper_bound":
        return {"quantity": "certified upper bound -12 c1 + c17 (E0 + r) on an eigenvalue of the spin-1 crown at the printed couplings",
                "record_value": {"eigenvalue_upper": dec(r["record"]["eigenvalue_upper"]),
                                 "E0_ring": dec(r["record"]["E0_ring"]), "E0_radius": dec(r["record"]["E0_radius"])},
                "lean_value": {"eigenvalue_upper": dec(r["eigenvalue_upper_lean"]),
                               "from_record_numbers": dec(r["eigenvalue_upper_from_record_numbers"]),
                               "ring_rayleigh_quotient": dec(r["ring"]["rayleigh_quotient_rounded"]),
                               "ring_radius": dec(r["ring"]["radius"])},
                "agreement": r["agreement"]}
    if tid.startswith("e10") and "lean" in r:
        L = r["lean"]
        return {"quantity": "certified upper bound t' on eta^2 at the recorded threshold, and the criterion",
                "record_value": {"t_prime": dec(r["record"]["t_prime"]), "eta_upper": dec(r["record"]["eta_upper"]),
                                 "positive": r["record"]["positive"], "unit_cosines": r["record"]["unit_cosines_at_least"],
                                 "criterion_holds": r["record"]["criterion_holds"]},
                "lean_value": {"t_prime": dec(L["t_prime"]), "eta_upper": dec(r["eta_upper_lean"]), "positive": L["positive"],
                               "unit_cosines": L["unit_cosines_at_least"], "criterion_holds": r["criterion_holds_lean"],
                               "record_t_prime_recertified": r["record_t_prime_recertified"]},
                "agreement": r["agreement"]}
    if tid == "e10_kernels":
        regs = r["regions"]
        rec = [x["record"] for x in regs if isinstance(x.get("record"), dict) and x["record"]]
        return {"quantity": "ker H_X = range U_X for Gamma_i = L u C and Gamma_j = C u R of the six lattices",
                "record_value": {"regions_checked": len(rec),
                                 "entries": [x["record_entry"] for x in rec],
                                 "all_equal": all(x.get("kernel_equals_vbs_range") for x in rec)},
                "lean_value": {"regions_checked": len(regs),
                               "all_kernels_equal_vbs_range": r["all_kernels_equal_vbs_range"],
                               "largest_dimension": max(x["lean"]["dim"] for x in regs),
                               "record_checks_agree": r["record_checks_agree"]},
                "agreement": bool(r["all_kernels_equal_vbs_range"] and r["record_checks_agree"])}
    if tid == "e10_square_octagon_criterion":
        return {"quantity": "4 eta_perp + 2 eta_par < 1",
                "record_value": {"sum_upper": dec(r["record"]["sum_upper"]), "criterion_holds": r["record"]["criterion_holds"]},
                "lean_value": {"sum_upper": dec(r["sum_upper_lean"]), "criterion_holds": r["criterion_holds_lean"]},
                "agreement": r["agreement"]}
    return {"agreement": r.get("agreement")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--collect-only", action="store_true", help="do not run; collect existing results")
    ap.add_argument("--skipped", default=None, help="JSON file: {id: reason} for targets not run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    meas_path = os.path.join(a.out, "measurements.json")
    meas = json.load(open(meas_path)) if os.path.exists(meas_path) else {}
    skipped = json.load(open(a.skipped)) if a.skipped else {}
    for tid, grp, args in TARGETS:
        if a.collect_only or (a.only and tid not in a.only) or tid in skipped:
            continue
        rec = os.path.join(a.archive, RUN_NEW, REC[grp][0], "data.json")
        out = os.path.join(a.out, tid + ".json")
        full = [x.format(rec=rec, out=out) for x in args]
        log = os.path.join(a.work, tid + ".log")
        print(f"running {tid}", flush=True)
        rc, sec, peak = run_target(a.exe, full, log)
        meas[tid] = {"exit_code": rc, "seconds": round(sec, 1), "peak_memory_bytes": peak, "log": os.path.basename(log)}
        shutil.copy(log, os.path.join(a.out, tid + ".log"))
        json.dump(meas, open(meas_path, "w"), indent=1, sort_keys=True)
        print(f"  {tid}: exit {rc}, {sec:.0f} s, peak {None if peak is None else round(peak / 2**30, 3)} GiB", flush=True)
    # square-octagon criterion from the two square-octagon results
    so = os.path.join(a.out, "e10_square_octagon_criterion.json")
    perp, par = (os.path.join(a.out, f"e10_square_octagon_{k}.json") for k in ("perp", "par"))
    if os.path.exists(perp) and os.path.exists(par) and not a.collect_only:
        rec = os.path.join(a.archive, RUN_NEW, REC["e10"][0], "data.json")
        rc, sec, peak = run_target(a.exe, ["aklt-so-criterion", perp, par, rec, so], os.path.join(a.work, "so_criterion.log"))
        meas["e10_square_octagon_criterion"] = {"exit_code": rc, "seconds": round(sec, 1), "peak_memory_bytes": peak}
        json.dump(meas, open(meas_path, "w"), indent=1, sort_keys=True)
    ident = records_identical(a.archive)
    rows = []
    for tid, grp, _ in TARGETS + [("e10_square_octagon_criterion", "e10", None)]:
        res_path = os.path.join(a.out, tid + ".json")
        entry = {"id": tid, "record_files": [f"{RUN_NEW}/{REC[grp][0]}/data.json", f"{RUN_OLD}/{REC[grp][1]}/data.json"],
                 "records_identical_in_both_runs": ident[grp]}
        if tid in skipped:
            entry.update({"agreement": None, "skipped_reason": skipped[tid], "seconds": None, "peak_memory_bytes": None})
        elif os.path.exists(res_path):
            r = json.load(open(res_path))
            entry.update(headline(tid, r))
            m = meas.get(tid, {})
            entry.update({"result_file": f"results/apps/{tid}.json", "seconds": m.get("seconds"),
                          "peak_memory_bytes": m.get("peak_memory_bytes"), "skipped_reason": None})
        else:
            entry.update({"agreement": None, "skipped_reason": "not run", "seconds": None, "peak_memory_bytes": None})
        rows.append(entry)
    checks = {}
    for name, key in (("compare_models.json", "all_equal"), ("compare_aklt.json", "all_equal"),
                      ("compare_chain20.json", "all_equal")):
        path = os.path.join(a.out, name)
        if os.path.exists(path):
            j = json.load(open(path))
            checks[name[:-5]] = {"file": f"results/apps/{name}", "all_equal": j.get(key)}
    st = os.path.join(a.out, "selftest.txt")
    if os.path.exists(st):
        text = open(st).read()
        checks["selftest"] = {"file": "results/apps/selftest.txt", "all_agree": "DISAGREE" not in text and "agree" in text}
    mathlib = None
    mp = os.path.join(a.out, "mathlib_apps.json")
    if os.path.exists(mp):
        j = json.load(open(mp))
        mathlib = {k: j[k] for k in ("module", "theorems", "axioms", "forbidden_tokens", "nanoda", "leanchecker") if k in j}
    summary = {"description": "Recomputation of the certified application numbers of certify-ed in compiled core Lean "
                              "(trust level N); see reverification_lean/APPS.md",
               "lean_program": "ceapps (reverification_lean/CertifyApps/Main.lean)",
               "targets": rows,
               "model_checks": checks,
               "mathlib_lemmas": mathlib,
               "agreeing": sum(1 for r in rows if r.get("agreement") is True),
               "disagreeing": sum(1 for r in rows if r.get("agreement") is False),
               "skipped": sum(1 for r in rows if r.get("skipped_reason"))}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ("agreeing", "disagreeing", "skipped")}))


if __name__ == "__main__":
    main()
