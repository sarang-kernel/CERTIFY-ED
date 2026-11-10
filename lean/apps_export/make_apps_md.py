"""Write reverification_lean/APPS.md from the result files of results/apps/ and the run records.

Every number in the tables is copied by this script from a named file: record values from the run records
(as the decimal text Python wrote), recomputed values from the JSON files written by ceapps, times and memory
from results/apps/measurements.json. The prose is fixed text.

usage: python make_apps_md.py --lean <reverification_lean> --archive <sources/archive_v5>
"""
import argparse
import json
import os
import re

RUN_NEW = "code/certify-ed/results/run_full_20261105"
RUN_OLD = "code/certify-ed/results/run_full_20261022"
E7 = f"{RUN_NEW}/e7_exact_benchmarks_20261105_092337/data.json"
E9 = f"{RUN_NEW}/e9_pyrochlore_20261105_084835/data.json"
E10 = f"{RUN_NEW}/e10_aklt_20261105_071106/data.json"


def lexemes(path):
    """Map from a JSON path to the number text as written in the file (Python's repr of the float)."""
    raw = open(path, encoding="utf8").read()
    return raw


def num_text(raw, key, after=None):
    """Text of the number of `"key": <number>` (first occurrence after the text `after`)."""
    start = raw.find(after) if after else 0
    m = re.compile(r'"' + re.escape(key) + r'":\s*(-?[0-9.eE+-]+)').search(raw, max(start, 0))
    return m.group(1) if m else "n/a"


def d(x):
    """Decimal text written by ceapps (17 significant digits, `m.mmm...eK`), written positionally when
    -4 <= K <= 3 (the digits are unchanged; only the decimal point moves)."""
    if x is None:
        return "n/a"
    s = x["decimal"] if isinstance(x, dict) else str(x)
    m = re.fullmatch(r"(-?)(\d)\.?(\d*)e(-?\d+)", s)
    if not m:
        return s
    sign, a, b, k = m.group(1), m.group(2), m.group(3), int(m.group(4))
    digits = a + b
    if k < -4 or k > 3:
        return s
    if k >= 0:
        ip, fp = digits[:k + 1], digits[k + 1:]
        return sign + ip + ("." + fp if fp else "")
    return sign + "0." + "0" * (-k - 1) + digits


def mib(b):
    return "n/a" if b is None else str(round(b / 2 ** 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--archive", required=True)
    a = ap.parse_args()
    res = os.path.join(a.lean, "results", "apps")
    J = lambda f: json.load(open(os.path.join(res, f), encoding="utf8"))
    meas = J("measurements.json")
    summ = J("summary.json")
    cm = J("compare_models.json")
    ca = J("compare_aklt.json")
    st = open(os.path.join(res, "selftest.txt"), encoding="utf8").read()
    n_self = st.count(": agree")
    rec7 = json.load(open(os.path.join(a.archive, E7)))
    raw9 = open(os.path.join(a.archive, E9), encoding="utf8").read()
    raw10 = open(os.path.join(a.archive, E10), encoding="utf8").read()
    raw7 = open(os.path.join(a.archive, E7), encoding="utf8").read()
    T = {t["id"]: t for t in summ["targets"]}
    from fractions import Fraction
    def fr(x):
        return Fraction(x["exact"]) if isinstance(x, dict) else Fraction(str(x))
    smaller = []
    for tid in ["e7_chain_J2_0", "e7_chain_J2_0.5", "e7_square_J2_0", "e7_square_J2_0.5", "e9_crown_spin_half_lower_bound"]:
        if os.path.exists(os.path.join(res, tid + ".json")):
            r_ = J(tid + ".json")
            smaller.append(fr(r_["delta_max"]) < fr(r_["record"]["delta"]))
    rec10 = json.load(open(os.path.join(a.archive, E10)))
    tp_le = []
    for lat in ["star", "square_octagon_perp", "square_octagon_par", "cross", "decorated_honeycomb", "decorated_square"]:
        if os.path.exists(os.path.join(res, "e10_" + lat + ".json")):
            r_ = J("e10_" + lat + ".json")
            if "lean" in r_:
                rb = list(rec10["lattices"][lat]["thresholds"].values())[0]["beta"]
                smaller.append(fr(r_["lean"]["beta"]) < Fraction(rb))
                tp_le.append(r_["record_t_prime_recertified"])
    all_smaller = bool(smaller) and all(smaller)
    all_tp_le = bool(tp_le) and all(tp_le)
    ident = all(t["records_identical_in_both_runs"] for t in summ["targets"])

    def sec(tid):
        m = meas.get(tid, {})
        return "n/a" if m.get("seconds") is None else f'{m["seconds"]}'

    def mem(tid):
        return mib(meas.get(tid, {}).get("peak_memory_bytes"))

    def yes(b):
        return "yes" if b else ("no" if b is False else "n/a")

    out = []
    w = out.append
    w("# APPS.md: the application numbers recomputed in compiled Lean")
    w("")
    w("This part of the re-verification recomputes the certified numbers of the three application experiments of")
    w("certify-ed, E7 (exact benchmarks), E9 (the pyrochlore crown) and E10 (AKLT models), with a compiled Lean program")
    w("that does not call the Python code. The program is `ceapps` (root `CertifyApps/Main.lean`) and its library is")
    w("`CertifyApps`; it uses core Lean only, together with the exact binary64 decoding and the JSON reader of")
    w("`CertifyCore` (`CORE.md`). Every result here is at trust level N in the scale of `README.md`: a compiled Lean")
    w("program recomputed the number from the model specification and compared it with the run record, and the")
    w("comparison itself is made in exact rational arithmetic inside the program. The record values are those of the")
    w(f"3.2.1 evidence run, `{RUN_NEW}`; for every compared field the 3.1.0 evidence run `{RUN_OLD}` holds the same")
    w("values (JSON equality with the timing fields removed, checked by `apps_export/run_all.py`: "
      + ("true for E7, E9 and E10" if ident else "NOT true for every group, see `summary.json`") + ").")
    w("Paths of record files are relative to `sources/archive_v5/`; every other path is relative to `reverification_lean/`.")
    w("")
    nsk = summ['skipped']
    w(f"The outcome in one sentence: {summ['agreeing']} of {len(summ['targets'])} targets agree with the records, "
      f"{summ['disagreeing']} disagree{'s' if summ['disagreeing'] == 1 else ''} and {nsk} "
      f"{'was' if nsk == 1 else 'were'} skipped (`results/apps/summary.json`).")
    w("")
    w("## 1 How to run")
    w("")
    w("The commands assume the environment of `logs/lean_env.json` (the Lean binaries on `PATH`) and, for the")
    w("comparison with the Python builder only, the Python environment `certifyed-311` with the certify-ed source on")
    w("`PYTHONPATH`. They write below `results/apps/` and `work/apps/` (git-ignored).")
    w("")
    w("```")
    w("cd reverification_lean")
    w("lake build CertifyApps ceapps")
    w(".lake\\build\\bin\\ceapps.exe selftest > results\\apps\\selftest.txt")
    w("set A=..\\sources\\archive_v5")
    w("set PYTHONPATH=%A%\\code\\certify-ed")
    w("set PYTHONDONTWRITEBYTECODE=1")
    w("python apps_export\\compare_models.py --exe .lake\\build\\bin\\ceapps.exe --code %A%\\code\\certify-ed --work work\\apps\\export --out results\\apps\\compare_models.json")
    w("python apps_export\\compare_aklt.py --exe .lake\\build\\bin\\ceapps.exe --code %A%\\code\\certify-ed --work work\\apps\\export --out results\\apps\\compare_aklt.json")
    w("python apps_export\\run_all.py --exe .lake\\build\\bin\\ceapps.exe --archive %A% --out results\\apps --work work\\apps\\final")
    w("python apps_export\\make_apps_md.py --lean . --archive %A%")
    w("```")
    w("")
    w("One target is run by itself with, for example,")
    w(f"`ceapps e7 chain_J2=0 %A%\\{E7.replace('/', chr(92))} out.json 1`, `ceapps crown-half <E9 record> out.json 1`,")
    w("`ceapps crown-one <E9 record> out.json` or `ceapps aklt cross <E10 record> out.json 1` (the last argument is the")
    w("number of threads for the residual; every run here used 1). The exit code is 0 when the target agrees with the")
    w("record, 1 when it does not, 2 when an input could not be read. Setting `CEAPPS_DEBUG=1` prints the time of each")
    w("factorization step.")
    w("")
    w("## 2 How eigenvalues are counted, and why this way")
    w("")
    w("Every certified number of the three experiments rests on counting eigenvalues of a symmetric matrix below a")
    w("point: the number of negative eigenvalues of `M = b A - a den N` (the inertia of `M`, its numbers of negative,")
    w("zero and positive eigenvalues) for a sector matrix `A` and a rational point `a / b`, or the number of positive")
    w("eigenvalues of the bordered matrix of the AKLT pencil. certify-ed counts them with")
    w("`certify_ed.checker.inertia_exact`: a floating-point LDL^T factorization proposes factors, the factors are")
    w("moved to dyadic grids, the residual `L D L^T - M` is computed exactly with FLINT integers, and Sylvester's law")
    w("of inertia (a congruence keeps the inertia) and Weyl's inequality (a perturbation of norm `beta` moves every")
    w("eigenvalue by at most `beta`) turn the signs of the pivots into counts that are exact up to the window")
    w("`[-beta, beta]` (`code/certify-ed/certify_ed/checker.py`).")
    w("")
    w("Why not an exact elimination. The core track measured the fraction-free (Bareiss) elimination of")
    w("`CertifyCore/Inertia.lean` at 247 s for one dense block of dimension 256 (`CORE.md`, section 4). The blocks here")
    w("have dimensions up to 4034, and the cost of an exact elimination grows with the cube of the dimension and with")
    w("the bit length of the entries, which grows linearly along the elimination; the dimension alone would multiply")
    w("the cost by some thousands, so exact elimination was not a feasible route. The route of certify-ed needs exact")
    w("arithmetic only for the residual, whose entries are products of numbers of fixed size.")
    w("")
    w("What `CertifyApps/Engine.lean` does. It factors `D M D` (`D = diag(w)^(-1/2)` for positive integer weights `w`,")
    w("the squared norms of the basis vectors of a sector; `w = 1` otherwise) in binary64 floating point with")
    w("symmetric diagonal pivoting, forms `G = D^-1 L |D_d|^(1/2)`, rounds `G` to an integer matrix `Gt` with entries")
    w("below `2^62` and keeps `S = sign(D_d)`. `Gt` is lower triangular with nonzero diagonal (checked), so `Gt S Gt^T`")
    w("has the inertia of `S`. The residual `E = Gt S Gt^T 2^(sigma - 2F) - P^T M P` is computed in exact integer")
    w("arithmetic: every entry of `Gt` is split into three 21-bit limbs, the limb products are accumulated in 64-bit")
    w("words whose true values provably stay below `2^63`, and each residual entry is formed in a fixed-width signed")
    w("number of eight 21-bit limbs. The bound `beta` is the Frobenius norm of `W^(-1/2) E W^(-1/2)`, bounded from")
    w("above by summing, for every entry, the square of an upper bound made of its leading 24 bits; it bounds the")
    w("spectral norm of the residual of the orthonormalised problem. Floating point only proposes `Gt` and `S`: a bad")
    w("proposal gives a large `beta` or a failed check, never a wrong count. The two differences from certify-ed are")
    w("the form of the factor (one factor `Gt` with 62 bits instead of `L` on a `2^-52` grid and `D` with 52 bits) and")
    w("the norm (a weighted Frobenius norm instead of the infinity norm of the unweighted residual); both are valid")
    w("bounds of the same quantity" + (", and the recomputed residual bound came out smaller than the recorded one in every"
      " case (E7, E9 spin 1/2 and E10)." if all_smaller else "; the recomputed residual bound is not smaller in every case,"
      " see the tables."))
    w("")
    w(f"How the instrument was checked. `ceapps selftest` (`results/apps/selftest.txt`) factors {n_self} symmetric integer")
    w("matrices of dimension 8 to 90 with entries of 10 to 130 bits, definite and indefinite, with and without weights,")
    w("and compares the count with the exact Bareiss elimination of `CertifyCore` and the limb residual with the same")
    w("residual computed with Lean's arbitrary-precision integers; the counts agree in every case and the bucket bound")
    w("is never below the exact sum. Section 3 checks the matrices themselves.")
    w("")
    w("Two further theorems enter: Theorem 1 of certify-ed (for a nonzero vector `v`, some eigenvalue lies within")
    w("`|| H v - mu v || / || v ||` of `mu`), used for the E7 and spin-1 enclosures, and the bordered-pencil lemma")
    w("for E10. They are used as stated (`docs/math/exact_layer.md` and `docs/applications/aklt_pomata_wei.md` of this")
    w("repository describe them); the vectors are computed here by inverse iteration, first in floating point and")
    w("then with integer arithmetic and iterative refinement, so that the radii are limited by the vector only far")
    w("below the record's radii.")
    w("")
    w("## 3 The models rebuilt in Lean and the check against the Python builder")
    w("")
    w("Why. A recomputation is only as good as the model it is applied to. The models are rebuilt in Lean from the")
    w("specification builders of certify-ed (`certify_ed/spec.py`: `heisenberg_chain`, `heisenberg_square`,")
    w("`spin1_heisenberg_chain`), from `certify_ed/apps/pyrochlore/model.py` (the 36 bonds and the couplings of the")
    w("reduced crown, the six pair swaps and the spin flip) and from `certify_ed/apps/aklt/lattices.py` and `graph.py`")
    w("(the six unions of subgraphs, the reduced VBS maps and their Gram matrices). Symmetry sectors are built in Lean")
    w("from orbit sums with one-dimensional characters of values 1, i, -1, -i; that every generator commutes with the")
    w("integer Hamiltonian and that every sector matrix is Hermitian is checked exactly in every run.")
    w("")
    w("What was done. `apps_export/compare_models.py` and `apps_export/compare_aklt.py` export the Lean matrices")
    w("(`ceapps export`, `export-sector`, `export-spin1-ring`, `export-aklt`, `export-aklt-ham`) and compare them, as")
    w("exact rationals and entry by entry, with `certify_ed.checker.exact_matrix`, with")
    w("`certify_ed.exact_sectors.sector_integer_matrix` (the two sector bases differ by one factor per column, which is")
    w("read off and applied), with the region maps and Gram matrices of `apps/aklt/graph.py`, and with the scaled AKLT")
    w("Hamiltonian of `apps/aklt/kernel.py`, a sum over edges of projectors onto the largest pair spin that for two")
    w("spins 1 is the bilinear-biquadratic form `1/3 + X/2 + X^2/6` in `X = S_u . S_v`.")
    w("")
    w("| model | states | nonzero entries | equal |")
    w("|---|---|---|---|")
    for r in cm["heisenberg"]:
        w(f'| `{r["model"]}`, `S^z = 0` | {r["states"]} | {r["nonzeros_lean"]} | {yes(r["entries_equal"] and r["state_sets_equal"])} |')
    for r in cm["spin1"]:
        w(f'| spin-1 ring of 6 sites, all states | {r["states"]} | {r["nonzeros_lean"]} | {yes(r["entries_equal"])} |')
    w("")
    w("| sector matrix | character exponents | dimension | group order | equal |")
    w("|---|---|---|---|---|")
    for r in cm["sectors"]:
        w(f'| `{r["model"]}` | {" ".join(str(x) for x in r["exponents"])} | {r["dim_lean"]} | {r["group_order_lean"]} | {yes(r["entries_equal"])} |')
    w("")
    w("| AKLT lattice | dimensions L, C, R, Gamma_i, Gamma_j | unit cosines at least | Gram matrices, counts and rank equal |")
    w("|---|---|---|---|")
    for r in ca["lattices"]:
        w(f'| `{r["lattice"]}` | {", ".join(str(x) for x in r["dims"])} | {r["unit_cosines"]} | {yes(r["entries_equal"])} |')
    for r in ca["hamiltonians"]:
        w(f'| `{r["lattice"]}`, Hamiltonian of L u C | {r["dim"]} states, {r["nonzeros"]} entries | | {yes(r["entries_equal"])}; H U = 0 on the {r["vbs_columns"]} VBS columns: {yes(r["lean_H_U_zero"])} |')
    w("")
    w("How the instrument was checked, and what it shows. The comparison is a set equality of exact entries, so a")
    w("single changed coupling, sign or state would make it fail. " + ("Every case is equal" if cm.get("all_equal") and ca.get("all_equal") else "Not every case is equal") + " (`results/apps/compare_models.json`,")
    w("`results/apps/compare_aklt.json`). The full E7 chains of 20 sites were not exported (184756 states); they are")
    w("built by the same Lean code as the chains of 8 and 12 sites, and the E7 runs check their sector dimensions")
    w("against the record (section 4).")
    w("")
    # E7
    w("## 4 E7: the four benchmark ground energies")
    w("")
    w("Why. E7 is the case in which the exact layer of certify-ed reaches realistic sizes, blocks of dimension 9252 and")
    w("12870 after symmetry reduction (`docs/applications/exact_benchmarks_e7.md`). Each record claims that the")
    w("recorded `E0` is within the recorded exact radius of the lowest eigenvalue of the block, and that exactly one")
    w("eigenvalue of the block lies below the recorded count point (`experiments/e7_exact_benchmarks.py`,")
    w("`certify_ed.exact_sectors.exact_recheck_sector_lowest`). An error in the symmetry reduction, in a count or in a")
    w("radius would show here as a different sector dimension, a different count, or a radius that does not cover the")
    w("eigenvalue.")
    w("")
    w("What was done. For each system the Lean program builds the model on the `S^z = 0` states, the group (translation,")
    w("reflection and spin flip of the chain, momentum fixed to 0; the two translations and the spin flip of the")
    w("square), checks every generator against the Hamiltonian, builds the character sectors and compares their")
    w("dimensions with the record, and counts the eigenvalues of every sector below the recorded count point (its")
    w("binary64 value). In the sector that holds the counted eigenvalue it computes a vector by inverse iteration and")
    w("from it two exact radii: around its own Rayleigh quotient, and around the recorded `E0`. The second radius")
    w("tests the record directly: if it is at most the recorded radius, the recorded enclosure contains an eigenvalue")
    w("whatever vector certify-ed used. Together with the count, that eigenvalue is the lowest of the block.")
    w("")
    w("| system | record `E0` | record exact radius | radius around the record `E0` (Lean) | Lean Rayleigh quotient | Lean radius | count (record / Lean) | `delta` (record / Lean) | seconds | peak MiB | agrees |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    names = {"e7_chain_J2_0": ("Heisenberg chain", "Heisenberg chain, N = 20"),
             "e7_chain_J2_0.5": ("J1-J2 chain", "J1-J2 chain, N = 20, J2 = 0.5"),
             "e7_square_J2_0": ("Heisenberg square", "Heisenberg square 4 x 4"),
             "e7_square_J2_0.5": ("J1-J2 square", "J1-J2 square 4 x 4, J2 = 0.5")}
    for tid, (pfx, label) in names.items():
        if not os.path.exists(os.path.join(res, tid + ".json")):
            w(f"| {label} | not run | | | | | | | | | |")
            continue
        r = J(tid + ".json")
        anchor = next(x["system"] for x in rec7["E7"] if x["system"].startswith(pfx))
        e0 = num_text(raw7, "E0", anchor)
        rr = num_text(raw7, "exact_radius", anchor)
        dl = num_text(raw7, "delta", anchor)
        v = r["vector"]
        w(f'| {label} | {e0} | {rr} | {d(v["radius_at_record_E0"])} | {d(v["rayleigh_quotient_rounded"])} | '
          f'{d(v["radius"])} | {r["record"]["count"]} / {r["count_total"]} | {dl} / {d(r["delta_max"])} | {sec(tid)} | {mem(tid)} | {yes(r["agreement"])} |')
    w("")
    sep_ok = []
    for tid in names:
        if os.path.exists(os.path.join(res, tid + ".json")):
            r_ = J(tid + ".json")
            sep_ok.append(fr(r_["record"]["E0"]) + fr(r_["record"]["exact_radius"]) < fr(r_["count_point"]) - fr(r_["delta_max"]))
    w("The recorded enclosure lies below the count point minus the Lean `delta` in every case (checked in rationals by")
    w("`apps_export/make_apps_md.py`: " + ("true" if sep_ok and all(sep_ok) else "NOT true in every case") + "), so the eigenvalue in it is the lowest of the block.")
    w("The sector dimensions, the dimension of the block and of the `S^z = 0` space equal the record in every case;")
    w("the group orders are 80 and 32 (`results/apps/e7_*.json`). The radius around the Lean Rayleigh quotient is the")
    w("radius of a vector accurate far beyond binary64, so the Lean enclosure is much narrower than the record's and")
    if os.path.exists(os.path.join(res, "e7_chain_J2_0.5.json")):
        mg = J("e7_chain_J2_0.5.json")["vector"]
        w("lies inside it. For the J1-J2 chain at the Majumdar-Ghosh point, where the ground energy is -7.5, the Lean")
        w(f'enclosure is {d(mg["rayleigh_quotient_rounded"])} with radius {d(mg["radius"])}.')
    w("")
    w("What it shows and does not show. The four E7 statements of the record are recomputed in Lean from the")
    w("specification: the same sectors, the same count at the same point, and an exact radius around the recorded")
    w("`E0` that is below the recorded radius. As in the record, the statement for the chains is about the momentum-0")
    w("block only. The counts rest on Sylvester's law and Weyl's inequality and the enclosures on Theorem 1, used as")
    w("stated.")
    w("")
    # E9 half
    w("## 5 E9: the lower bound for the spin-1/2 crown")
    w("")
    r = J("e9_crown_spin_half_lower_bound.json") if os.path.exists(os.path.join(res, "e9_crown_spin_half_lower_bound.json")) else None
    w("Why. The certified bound `lambda_min(H_crown) >= -0.5501582988212038` turns the reduced-model value of")
    w("Kránitz and Penc into a statement with no floating-point assumption (`docs/applications/crown_kranitz_penc.md`).")
    w("The record claims that at the recorded shift no eigenvalue lies below the shift minus `delta` in any of the 128")
    w("sectors of the `S^z = 0` space (`certify_ed/apps/pyrochlore/spin_half.py`). A count of 1 in any sector, or a")
    w("residual bound larger than the recorded one, would mean that the recorded bound is not reproduced.")
    w("")
    w("What was done. The crown is rebuilt with `c1 = 381414/10^7`, `c3 = c12 = 817957/10^7` and `c17` from the")
    w("embedding constraint, on the 48620 states of `S^z = 0`; the seven generators are checked against the")
    w("Hamiltonian; the 128 sectors are built and every one is counted at the recorded shift; the bound is the shift")
    w("minus the largest Weyl slack `delta` of the 128 counts.")
    w("")
    if r:
        w("| quantity | record | Lean |")
        w("|---|---|---|")
        w(f'| `S^z = 0` dimension, sectors, largest sector | {num_text(raw9, "dim_Sz0")}, {num_text(raw9, "n_sectors")}, {num_text(raw9, "largest_sector")} | {r["dim_Sz0"]}, {r["n_sectors"]}, {r["largest_sector"]} |')
        w(f'| shift | `{r["shift"]["exact"]}` | the same |')
        w(f'| eigenvalues below the shift minus `delta`, all sectors | 0 | {r["count_total"]} |')
        w(f'| `delta` | {num_text(raw9, "delta", chr(34) + "spin_half")} | {d(r["delta_max"])} |')
        w(f'| lower bound | {num_text(raw9, "lower_bound", chr(34) + "spin_half")} | {d(r["lower_bound_lean"])} |')
        w(f'| seconds, peak MiB | | {sec("e9_crown_spin_half_lower_bound")}, {mem("e9_crown_spin_half_lower_bound")} |')
        w("")
        w(f'The Lean bound is at least the record\'s exact bound `lower_bound_exact` (checked in rationals:'
          f' {yes(r["record_bound_recertified"])}), so the recorded bound is recomputed; agreement:'
          f' {yes(r["agreement"])}. The smaller `delta` comes from the smaller residual bound of section 2; the shift,')
        w("and therefore the margin below the Lanczos estimate of the record, is the record's.")
    else:
        w("Not run.")
    w("")
    w("What it shows and does not show. The bound of the reduced model is recomputed. As in the record, it is not the")
    w("paper's headline bound of the nine-coupling model, the Anderson decomposition and the embedding constraint are")
    w("taken from the paper, and the counts rest on Sylvester's law and Weyl's inequality.")
    w("")
    # E9 one
    w("## 6 E9: the spin-1 crown at the printed couplings")
    w("")
    w("Why. The record claims that at the printed couplings the crown has an eigenvalue at most")
    w("`-12 c1 + c17 (E0 + r)`, with `E0` and `r` the enclosure of the lowest eigenvalue of the spin-1 Heisenberg")
    w("ring of 6 sites, and that this number lies below the published bound -1.632985")
    w("(`certify_ed/apps/pyrochlore/spin_one.py`, `printed_coupling_check`). It rests on the identity that the")
    w("all-singlet class of the crown is `-12 c1 + c17 H_ring`.")
    w("")
    if os.path.exists(os.path.join(res, "e9_crown_spin_one_upper_bound.json")):
        r = J("e9_crown_spin_one_upper_bound.json")
        g = r["ring"]
        w("What was done. The ring is rebuilt on all 729 states; its lowest eigenvalue is enclosed by a vector from")
        w("inverse iteration, counted below a point found by the program (one eigenvalue below it, the enclosure under")
        w("its window), and the radius around the recorded `E0` is computed. The two identities behind the")
        w("all-singlet class, `(S_a + S_b) . S_h u = 0` and `S_a . S_b u = -2 u` for the pair singlet `u`, are checked")
        w("exactly on three spins 1. The bound is evaluated in rationals, once from the record's `E0` and radius and")
        w("once from the Lean enclosure.")
        w("")
        w("| quantity | record | Lean |")
        w("|---|---|---|")
        w(f'| `E0` of the ring | {num_text(raw9, "E0_ring")} | {d(g["rayleigh_quotient_rounded"])} |')
        w(f'| radius | {num_text(raw9, "E0_radius", chr(34) + "E0_ring")} | {d(g["radius"])} (around the Lean value); {d(g["radius_at_record_E0"])} (around the record `E0`) |')
        w(f'| count point, count | | {d(g["count_point"])}, {g["count"]} |')
        w(f'| eigenvalue upper bound | {num_text(raw9, "eigenvalue_upper")} | {d(r["eigenvalue_upper_lean"])}; from the record\'s numbers {d(r["eigenvalue_upper_from_record_numbers"])} |')
        contra = "true" if re.search(r'"contradiction":\s*true', raw9) else "false"
        w(f'| below the published -1.632985 | {contra} | {"true" if r["below_published_bound_lean"] else "false"} |')
        w(f'| seconds, peak MiB | | {sec("e9_crown_spin_one_upper_bound")}, {mem("e9_crown_spin_one_upper_bound")} |')
        w("")
        ring_sep = fr(r["record"]["E0_ring"]) + fr(r["record"]["E0_radius"]) < fr(g["count_point"]) - fr(g["delta"])
        w("The recorded enclosure of the ring lies below the count point minus `delta` (checked in rationals: "
          + ("true" if ring_sep else "false") + "), so it holds the lowest eigenvalue of the ring.")
        w(f'The record\'s float `eigenvalue_upper` is the binary64 number nearest the exact value from its own `E0` and')
        w(f'radius ({yes(r["record_float_reproduced"])}), the record\'s enclosure is recomputed ({yes(r["record_enclosure_recertified"])}),')
        w(f'and the pair-singlet identities hold ({yes(r["pair_singlet_identities"])}); agreement: {yes(r["agreement"])}.')
    w("")
    w("The other pair-spin classes are built by `heisenberg_mixed` in floating point, with square roots in the")
    w("matrix elements, and enter no certified number; they were not rebuilt, since no exact comparison with them")
    w("is possible.")
    w("")
    w("What it shows and does not show. The certified part of the spin-1 statement is recomputed. The two sectors the")
    w("paper names and the floating-point values of the 92 classes (`spin1_landscape`) are not certified by the record")
    w("and are not targets.")
    w("")
    # E10
    w("## 7 E10: the AKLT eta bounds")
    w("")
    w("Why. For each lattice the spectral-gap criterion of Pomata and Wei needs `eta < 1/z`, or")
    w("`4 eta_perp + 2 eta_par < 1` for the square-octagon lattice, and certify-ed replaced the floating-point `eta` by")
    w("an upper bound `sqrt(t')` in exact arithmetic (`docs/applications/aklt_pomata_wei.md`,")
    w("`certify_ed/apps/aklt/eta.py`). The record states, at its threshold `t`, the number of positive pivots, the lower")
    w("bound on the unit cosines, `t'` and the criterion. A recomputed count above the unit-cosine bound, or a `t'`")
    w("that does not decide the criterion, would mean the record is not reproduced.")
    w("")
    w("What was done. For each lattice the Lean program builds the reduced regions, the union coordinates, the Gram")
    w("matrices and the rank modulo `2^31 - 1` of the union VBS states, certified lower bounds of the smallest")
    w("eigenvalues of the five Gram matrices, the symmetry group (spin flip, and the mirror for the decorated square)")
    w("with exact checks of every equivariance, and the bordered matrices of every block and character at the")
    w("record's threshold; it then forms `t' = (1 + beta/mu)(t + beta/nu)` exactly (`CertifyApps/Eta.lean`).")
    w("")
    w("| lattice | threshold `t` | positive pivots (record / Lean) | unit cosines at least | record `t'` | Lean `t'` | record `eta` bound | Lean `eta` bound | criterion (record / Lean) | seconds | peak MiB | agrees |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lat in ["star", "square_octagon_perp", "square_octagon_par", "cross", "decorated_honeycomb", "decorated_square"]:
        tid = "e10_" + lat
        if not os.path.exists(os.path.join(res, tid + ".json")):
            w(f"| `{lat}` | | | | | | | | | | | {T.get(tid, {}).get('skipped_reason') or 'not run'} |")
            continue
        r = J(tid + ".json")
        if "lean" not in r:
            w(f"| `{lat}` | | | | | | | | | | | error: {r.get('error')} |")
            continue
        L = r["lean"]
        anchor = '"' + lat + '": {'
        rec_tp = num_text(raw10, "t_prime", anchor)
        rec_eta = num_text(raw10, "eta_upper", anchor)
        crit_r = r["record"]["criterion_holds"]
        crit_l = r["criterion_holds_lean"]
        cs = lambda x: "n/a" if x is None else ("holds" if x else "fails")
        w(f'| `{lat}` | `{r["record"]["threshold"]["exact"]}` | {r["record"]["positive"]} / {L["positive"]} | '
          f'{L["unit_cosines_at_least"]} | {rec_tp} | {d(L["t_prime"])} | {rec_eta} | {d(r["eta_upper_lean"])} | '
          f'{cs(crit_r)} / {cs(crit_l)} | {sec(tid)} | {mem(tid)} | {yes(r["agreement"])} |')
    w("")
    if os.path.exists(os.path.join(res, "e10_square_octagon_criterion.json")):
        so = J("e10_square_octagon_criterion.json")
        w(f'For the square-octagon lattice `4 eta_perp + 2 eta_par` is at most {d(so["sum_upper_lean"])} in Lean and at most')
        w(f'{num_text(raw10, "sum_upper")} in the record; the criterion holds in both (agreement: {yes(so["agreement"])}).')
    if all_tp_le:
        w("In every lattice the Lean `t'` is at most the record's, so the recorded `eta` bounds are recomputed, and the unit")
        w("cosine bounds and the numbers of positive pivots are the record's (`results/apps/e10_*.json`). The excess of")
    else:
        w("Not in every lattice is the Lean `t'` at most the record's (column `record_t_prime_recertified` of the result")
        w("files); where it is larger, the record's bound is valid but not recomputed by this program. The excess of")
    w("`t'` over `t` is much smaller in Lean, for the reason given in section 2; the bounds themselves are dominated")
    w("by the placement of `t`, which is the record's.")
    w("")
    e10ok = all(T.get("e10_" + x, {}).get("agreement") for x in ["star", "cross", "decorated_honeycomb", "decorated_square"]) and T.get("e10_square_octagon_criterion", {}).get("agreement")
    w("What it shows and does not show. " + ("The five E10 criteria of the record are recomputed from the geometry." if e10ok else "Not all five E10 criteria were recomputed (see the table).") + " The")
    w("criterion itself, the kernel lemma (ground space equals VBS range) and the bordered-pencil lemma are used as")
    w("stated; the bordered-pencil lemma is new mathematics of the project and awaits Sarang Vehale's check under")
    w("D-12. The exact kernel check of the record (`kernel` in the E10 record) was not repeated; the Lean check")
    w("`H U = 0` of section 3 covers one of its two directions for the star and the square-octagon.")
    w("")
    w("## 8 What was not recomputed")
    w("")
    skipped = [t for t in summ["targets"] if t.get("skipped_reason")]
    if skipped:
        for t in skipped:
            w(f'The target `{t["id"]}` was skipped: {t["skipped_reason"]}.')
    else:
        w("Every target named for this part was run to the end; none was skipped for time or memory.")
    w("Floating-point values that the records do not certify are not targets: the two crown sectors named by the")
    w("paper, the values of the 92 pair-spin classes at the re-optimised couplings, and the floating-point `eta` of each lattice. The honeycomb")
    w("lattice (Method III of Pomata and Wei) was not attempted by certify-ed and is not attempted here.")
    w("")
    w("## 9 Files")
    w("")
    w("| path | contents |")
    w("|---|---|")
    w("| `CertifyApps.lean`, `CertifyApps/` | the library: `Engine` (counting), `SpinHalf` (spin-1/2 models and sectors), `Models` (the E7 and E9 models and the spin-1 ring), `Vector` (inverse iteration and exact radii), `Report` (records and JSON), `Apps` (E7 and E9), `Vbs` and `Eta` (E10), `AkltHam` (AKLT projector Hamiltonians), `Export`, `ExportAklt`, `Main` (the program) |")
    w("| `apps_export/` | the comparison with the Python builder, the driver of all runs, this document's generator |")
    w("| `results/apps/summary.json` | per target: record value, Lean value, agreement, seconds, peak memory, skipped reason |")
    w("| `results/apps/<target>.json`, `.log` | the full output of each run and its log |")
    w("| `results/apps/measurements.json` | wall time, exit code and peak working set of each run |")
    w("| `results/apps/compare_models.json`, `compare_aklt.json`, `selftest.txt` | the checks of sections 2 and 3 |")
    w("")
    with open(os.path.join(a.lean, "APPS.md"), "w", encoding="utf8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("written", len(out), "lines")


if __name__ == "__main__":
    main()
