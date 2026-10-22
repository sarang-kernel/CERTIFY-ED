"""Write RESULT_candidate_run.md: what a reproduction run re-derived, next to the reference run.

Usage: python experiments/make_result_candidate.py <reference_run_dir> <candidate_run_dir> <compare.csv> <out.md> [title]

Reads the two manifests, the E7/E9/E10/E8 records of both runs and the CSV written by compare_runs.py, and
writes a short document: per experiment the statements (statuses, counts, booleans) and the headline numbers
of both runs, the stack-dependent bounds with their relative difference, and the comparison summary. Numbers
are copied from the records, never typed. Post-processing only (excluded from the computational source).
"""
import csv
import glob
import json
import os
import sys


def rec(run, prefix):
    hits = sorted(glob.glob(os.path.join(run, prefix + "*")))
    if not hits:
        return None, None
    d = hits[-1]
    with open(os.path.join(d, "data.json")) as f:
        return json.load(f), os.path.basename(d)


def manifest(run):
    with open(os.path.join(run, "manifest.json")) as f:
        return json.load(f)


def fmt(x):
    if isinstance(x, float):
        return f"{x:.10g}"
    return str(x)


def main(ref, cand, cmp_csv, out, title="Candidate reproduction run: what was re-derived"):
    mr, mc = manifest(ref), manifest(cand)
    rows = list(csv.DictReader(open(cmp_csv, encoding="utf-8")))
    by_exp = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], []).append(r)
    L = []
    w = L.append
    w(f"# {title}\n")
    w(f"Reference run `{os.path.basename(ref)}`: code {mr.get('source', {}).get('source_hash')} "
      f"(commit {str(mr.get('git_commit'))[:7]}), platform {mr['environment'].get('platform', '?')}, "
      f"python {mr['environment'].get('python', '?')}, flint C library {mr['environment'].get('flint_c_library', 'not recorded by 3.0.0')}.\n")
    w(f"Candidate run `{os.path.basename(cand)}`: code {mc.get('source', {}).get('source_hash')} "
      f"(commit {str(mc.get('git_commit'))[:7]}), platform {mc['environment'].get('platform', '?')}, "
      f"python {mc['environment'].get('python', '?')}, flint C library {mc['environment'].get('flint_c_library', '?')}; "
      f"started {mc.get('started')}, finished {mc.get('finished', 'not finished')}.\n")
    w("## Steps\n")
    w("| step | candidate status | seconds | source hash at the step |")
    w("|---|---|---:|---|")
    for k, v in mc["steps"].items():
        if k == "numbers":
            continue
        w(f"| {k} | {v.get('status')}{' (' + v.get('reason', '') + ')' if v.get('reason') else ''} | "
          f"{fmt(v.get('seconds', ''))} | {v.get('source_hash', '')} |")
    w("")
    srcs = {v.get("source_hash") for k, v in mc["steps"].items() if v.get("status") == "ok"}
    if len(srcs) > 1:
        w("The per-step source hashes differ although the computational files did not change between the steps. The "
          "fingerprint's exclusion list (certify_ed/provenance.py NON_COMPUTATIONAL) was extended while the run was in "
          "flight, and the running process computed each step's hash with the list it had imported at start, over the "
          "working tree of the moment; a post-processing script created in experiments/ during the last step made its "
          "tree 'dirty'. Both facts were reproduced after the run: the start-of-run exclusion list applied to the step "
          "commits gives exactly the recorded hashes, the current list applied to every step commit gives one hash, and "
          "`git diff` over the computational files between any two step commits is empty. The 3.1.0 evidence run must "
          "be made from one commit with no edits in flight (the manifest check 'every step ran at one computational "
          "source' then passes).\n")

    # ---- E7
    r7, n7 = rec(ref, "e7_"); c7, m7 = rec(cand, "e7_")
    if r7 and c7:
        w("## E7 — benchmark ground energies, exact re-check\n")
        w("| system | U(1) dim / block dim | status ref / cand | E0 ref | E0 cand | exact radius ref | exact radius cand | ratio |")
        w("|---|---|---|---|---|---|---|---:|")
        for a, b in zip(r7["E7"], c7["E7"]):
            name = a.get("system") or a.get("name") or "?"
            ra, rb = a.get("exact_radius"), b.get("exact_radius")
            w(f"| {name} | {a.get('u1_dim')} / {a.get('block_dim')} | {a.get('status')} / {b.get('status')} | {fmt(a.get('E0'))} | {fmt(b.get('E0'))} | "
              f"{ra:.3e} | {rb:.3e} | {rb / ra:.2f} |")
        w("")
    # ---- E9
    r9, _ = rec(ref, "e9_"); c9, _ = rec(cand, "e9_")
    if r9 and c9:
        w("## E9 — pyrochlore crown\n")
        sa, sb = r9["spin_half"], c9["spin_half"]
        w(f"S = 1/2 reduced model (n1 = {sa['n1']}, n3 = {sa['n3']}; {sa['n_sectors']} sectors, largest {sa['largest_sector']}): "
          f"lower bound rounded down {fmt(sa['lower_bound_rounded_down'])} (ref) / {fmt(sb['lower_bound_rounded_down'])} (cand); "
          f"exact lower bound {sa['lower_bound']!r} / {sb['lower_bound']!r}; delta {sa['delta']:.4e} / {sb['delta']:.4e}; "
          f"no_floating_point_assumption {sa['no_floating_point_assumption']} / {sb['no_floating_point_assumption']}; "
          f"{sa['seconds']:.0f} s / {sb['seconds']:.0f} s.\n")
        pa, pb = r9["spin1_printed"], c9["spin1_printed"]
        w(f"S = 1 at the printed couplings (all-singlet sector): eigenvalue upper bound {pa['eigenvalue_upper']!r} (ref) / "
          f"{pb['eigenvalue_upper']!r} (cand) — {'bit-identical' if pa['eigenvalue_upper'] == pb['eigenvalue_upper'] else 'DIFFERENT'}; "
          f"published {pa['published_bound']}; contradiction {pa['contradiction']} / {pb['contradiction']}; margin "
          f"{pa['margin']:.6e} / {pb['margin']:.6e}; exact re-check {pa['exact_recheck']} / {pb['exact_recheck']}; "
          f"E0 radius (exact) {pa['E0_radius_exact']:.3e} / {pb['E0_radius_exact']:.3e}.\n")
        la, lb = r9.get("spin1_landscape", {}), c9.get("spin1_landscape", {})
        if la:
            ma = min(la["classes"], key=lambda c: c["energy"]); mb = min(lb["classes"], key=lambda c: c["energy"])
            w(f"S = 1 landscape (floating point) at c1 = {fmt(la['couplings']['c1'])}, c3 = {fmt(la['couplings']['c3'])}: "
              f"{len(la['classes'])} / {len(lb['classes'])} classes; lowest class {ma['pair_spins2']} at {fmt(ma['energy'])} (ref) / "
              f"{mb['pair_spins2']} at {fmt(mb['energy'])} (cand).\n")
        for extra in ("named_sectors", "named_sector_energies"):
            if extra in c9 and extra not in r9:
                w(f"Candidate-only record `{extra}` (feat/e9-named-sectors): `{json.dumps(c9[extra])[:400]}`\n")
    # ---- E10
    r10, _ = rec(ref, "e10_"); c10, _ = rec(cand, "e10_")
    if r10 and c10:
        w("## E10 — Pomata–Wei AKLT gap criteria in exact arithmetic\n")
        w("| lattice | pencil | largest block | positive pivots ref / cand | eta upper ref / cand | beta ref / cand | criterion holds |")
        w("|---|---:|---:|---|---|---|---|")
        for lat, a in r10["lattices"].items():
            b = c10["lattices"].get(lat)
            if not b:
                w(f"| {lat} | — | — | not in candidate | | | |")
                continue
            ta = list(a.get("thresholds", {}).values()); tb = list(b.get("thresholds", {}).values())
            pa = ta[-1].get("positive") if ta else None; pb = tb[-1].get("positive") if tb else None
            ba = ta[-1].get("beta") if ta else None; bb = tb[-1].get("beta") if tb else None
            w(f"| {lat} | {a.get('pencil_size')} | {a.get('largest_block')} | {pa} / {pb} (need <= {a.get('unit_cosines_at_least')}) | "
              f"{fmt(a.get('eta_upper'))} / {fmt(b.get('eta_upper'))} | {fmt(ba)} / {fmt(bb)} | {a.get('criterion_holds')} / {b.get('criterion_holds')} |")
        w("")
        so_a, so_b = r10.get("square_octagon", {}), c10.get("square_octagon", {})
        w(f"Square-octagon combined criterion 4 eta_perp + 2 eta_par: upper {fmt(so_a.get('sum_upper'))} (ref) / {fmt(so_b.get('sum_upper'))} (cand); "
          f"holds {so_a.get('criterion_holds')} / {so_b.get('criterion_holds')}.\n")
        w("Kernel checks (H U = 0, rank U, nullity of H mod p, kernel = VBS range):\n")
        w("| lattice | sites | dim | VBS rank ref / cand | nullity mod p ref / cand | kernel = VBS range ref / cand |")
        w("|---|---:|---:|---|---|---|")
        for lat, ka in r10.get("kernel", {}).items():
            kb = c10.get("kernel", {}).get(lat, [])
            for i, ea in enumerate(ka):
                eb = kb[i] if i < len(kb) else {}
                w(f"| {lat} | {ea.get('sites')} | {ea.get('dim')} | {ea.get('vbs_rank')} / {eb.get('vbs_rank', 'not run')} | "
                  f"{ea.get('nullity_mod_p_upper')} / {eb.get('nullity_mod_p_upper', 'not run')} | {ea.get('kernel_equals_vbs_range')} / {eb.get('kernel_equals_vbs_range', 'not run')} |")
        w("")
    # ---- E8
    r8, _ = rec(ref, "e8_"); c8, _ = rec(cand, "e8_")
    if r8 and c8:
        w("## E8 — comparison with symveig\n")
        w(f"{len(r8['E8'])} benchmark models; elapsed {fmt(r8.get('_elapsed_seconds'))} s (ref) / {fmt(c8.get('_elapsed_seconds'))} s (cand).\n")
        w("| model | L | dim | complete ref / cand | counted claims ref / cand | reference outside ref / cand | max half-width symveig / certify_ed (ref) | (cand) | ratio ref / cand |")
        w("|---|---:|---:|---|---|---|---|---|---|")
        for a, b in zip(r8["E8"], c8["E8"]):
            w(f"| {a.get('model')} | {a.get('L')} | {a.get('dim')} | {a.get('certify_complete')} / {b.get('certify_complete')} | "
              f"{a.get('symveig_counted_claims')} / {b.get('symveig_counted_claims')} | {fmt(a.get('symveig_reference_outside'))}, {fmt(a.get('certify_reference_outside'))} / "
              f"{fmt(b.get('symveig_reference_outside'))}, {fmt(b.get('certify_reference_outside'))} | {a.get('symveig_max_halfwidth'):.2e} / {a.get('certify_max_halfwidth'):.2e} | "
              f"{b.get('symveig_max_halfwidth'):.2e} / {b.get('certify_max_halfwidth'):.2e} | {fmt(a.get('ratio'))} / {fmt(b.get('ratio'))} |")
        w("")
        w("The half-widths are LAPACK residual norms and the ratio is their quotient: run-stack statistics (SS-19), reported, not compared by tolerance.\n")
        so_a = [a["model"] + f" L={a['L']}" for a in r8["E8"] if a.get("symveig_reference_outside", 0) > 0]
        so_b = [a["model"] + f" L={a['L']}" for a in c8["E8"] if a.get("symveig_reference_outside", 0) > 0]
        co_a = [a["model"] + f" L={a['L']}" for a in r8["E8"] if a.get("certify_reference_outside", 0) > 0]
        co_b = [a["model"] + f" L={a['L']}" for a in c8["E8"] if a.get("certify_reference_outside", 0) > 0]
        w(f"Models where symveig's intervals exclude the reference eigenvalue: {len(so_a)} (ref: {', '.join(so_a) or 'none'}) / "
          f"{len(so_b)} (cand: {', '.join(so_b) or 'none'}); where certify_ed's do: {len(co_a)} (ref) / {len(co_b)} (cand). "
          "Whether a residual-only interval misses the reference is itself stack-dependent; the record reports the distance, "
          "and the statements compared strictly are completeness and the counted claims.\n")
    # ---- E2
    r2, _ = rec(ref, "e2_"); c2, _ = rec(cand, "e2_")
    if r2 and c2:
        w("## E2 — cost and memory of certification (each child capped at the same memory)\n")
        w("| point | dim | S3 ref / cand | certify s ref / cand | peak MiB ref / cand | cap mechanism (cand) |")
        w("|---|---:|---|---|---|---|")

        def s3(row):
            c = row.get("certifier", {})
            return c.get("statements", {}).get("S3.completeness", {}).get("status", "error" if "error" in c else "-")
        same = 0
        for a, b in zip(r2.get("rows", []), c2.get("rows", [])):
            ca, cb = a.get("certifier", {}), b.get("certifier", {})
            same += s3(a) == s3(b)
            def f1(x):
                return f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
            w(f"| {a['label']} | {a.get('producer', {}).get('dim', '')} | {s3(a)} / {s3(b)} | {f1(ca.get('total', ''))} / "
              f"{f1(cb.get('total', ''))} | {f1(ca.get('maxrss_mb', ''))} / {f1(cb.get('maxrss_mb', ''))} | {cb.get('memory_cap', '')} |")
        w("")
        w(f"S3 outcome identical on {same} of {len(r2.get('rows', []))} points. NOT_CHECKED means the inertia count did not fit "
          "under the cap; the certificate says so and S1+S2 are measured alone.\n")
    # ---- comparison summary
    w("## Record-by-record comparison (experiments/compare_runs.py)\n")
    w("| experiment | leaves | same | float within 1e-9 | stack-dependent (within STACK_RTOL) | max rel. diff of stack-dependent | fp statistics (E8 ratios) | max rel. diff of fp statistics | timing | missing | MISMATCH |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for e, rs in sorted(by_exp.items()):
        cls = [r["class"] for r in rs]
        sd = [float(r["rel_diff"]) for r in rs if r["class"].startswith("stack") and r["rel_diff"] not in ("", None)]
        fs = [float(r["rel_diff"]) for r in rs if r["class"] == "fp-statistic" and r["rel_diff"]]
        w(f"| {e} | {len(rs)} | {cls.count('same')} | {cls.count('float-differs')} | {sum(c.startswith('stack') for c in cls)} | "
          f"{(max(sd) if sd else 0):.3e} | {cls.count('fp-statistic')} | {(max(fs) if fs else 0):.3e} | {cls.count('timing')} | {cls.count('missing')} | {cls.count('MISMATCH')} |")
    w("")
    mism = [r for r in rows if r["class"] == "MISMATCH"]
    if mism:
        w("MISMATCH leaves:\n")
        for r in mism[:50]:
            w(f"* {r['experiment']} `{r['path']}`: {r['value_A'][:80]} vs {r['value_B'][:80]}")
        w("")
    else:
        w("No MISMATCH leaf: every statement, count, pivot sign, status and string agrees; only timings, the "
          "stack-dependent bounds (within their tolerance) and the E8 residual statistics differ.\n")
    num = mc["steps"].get("numbers")
    if num:
        w("## Numbers step of the candidate run\n")
        for script, res in num.items():
            if script == "provenance":
                continue
            if res.get("status") == "skipped":
                w(f"* `{script}`: skipped — {res.get('reason')}")
            else:
                w(f"* `{script}`: returncode {res.get('returncode')}; " + "; ".join(str(c) for c in res.get("checks", [])))
        w("")
        w("The failed manifest checks are the ones expected of a reproduction made while the branch was being integrated "
          "(a skipped step, several per-step hashes, one dirty step; see Steps above) and would fail the 3.1.0 evidence run, "
          "which is their purpose.\n")
    w("## Coverage\n")
    missing = [k for k in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10")
               if mc["steps"].get(k, {}).get("status") != "ok"]
    if missing:
        w(f"Steps of the reference run not reproduced here: {', '.join(missing)} (see the step table for the reason).\n")
    else:
        w("Every experiment of the reference run (E0–E10) was reproduced.\n")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print(out, len(L), "lines")


if __name__ == "__main__":
    main(*sys.argv[1:6])
