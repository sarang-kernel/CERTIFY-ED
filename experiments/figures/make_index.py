"""Render docs/figures/INDEX.md from the figures' <name>.numbers.json files.

Every number a caption prints is read from the numbers file the figure script wrote next to the figure
(which in turn read it from the record or document named as the data source); nothing is typed in.
Usage: python experiments/figures/make_index.py [--check]   (--check: exit 1 when INDEX.md is out of date)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

FIG = os.path.join(C.ROOT, "docs", "figures")
RUN_A = "run_full_20261022 (Windows, certify-ed 3.1.0 evidence run)"
RUN_B = "run_full_20261021 (Linux, certify-ed 3.0.0 run)"


def N(name):
    with open(os.path.join(FIG, name + ".numbers.json"), encoding="utf-8") as f:
        return json.load(f)


def fmt(x, d=1):
    return f"{x:.{d}f}"


def caption_certificate_anatomy(n):
    return ("What a certificate contains and how its statements depend on each other. (a) The JSON body: the "
            "subject and claim (model specification, claim kind, the k claimed eigenvalues and the hash of their "
            "vectors), one record per statement S1–S8 with a status (CERTIFIED, INCONCLUSIVE, NOT_CHECKED, FAILED; "
            "CORROBORATED for S5 only), the scope block (what ran, what the user narrowed, what is out of scope and "
            "why), integrity and environment (body hash, library versions, floating-point probe, source hash), and the "
            "arrays sidecar. The statuses shown are an example, not a record. (b) The floating-point chain: S1 fixes "
            "the matrix the enclosures refer to (its shift σ is added to every radius), S2 encloses each claimed "
            "value, S3 counts eigenvalues below a shift and orders the enclosures, S4 and S8 read clusters and "
            "brackets off S2 and S3; S6 re-checks S2 and S3 in exact arithmetic from the claim and the specification "
            "alone; S7 needs the whole-spectrum enclosure of a kind `all` claim; S5 is corroboration by a second "
            "solver, evidence rather than a certified statement.")


def caption_enclosure_geometry(n):
    return ("The enclosures on the real line (positions illustrative). Theorem 1 gives each claimed value μ an "
            "interval of radius r = ‖Ax − μx‖/‖x‖ containing an eigenvalue, existence only; when two intervals "
            "overlap the pair is a cluster and Theorem 2/2′ gives one cluster radius ε under which the members are "
            "matched to distinct eigenvalues, each member keeping its Theorem 1 existence radius. Theorem 3 "
            "(Kato–Temple) sharpens an isolated value inside its isolating interval (α, β), whose ends are the "
            "neighbouring enclosures ordered by S3. S3 (Lemma 4) counts the inertia ν of A − σI with backward error "
            "β: #{λ < σ − β} ≤ ν ≤ #{λ ≤ σ + β}; with ν = k and k distinct enclosures below σ − β the enclosures hold "
            "λ₁ … λ_k and λ_{k+1} ≥ σ − β (Corollary 5), the certified gap lower bound.")


def caption_inertia_completeness(n):
    return ("How inertia counts certify \"exactly k eigenvalues below σ\" (illustrative spectra; N(x) = #{λ < x} "
            "is the counting function). (a) A `lowest` claim: k distinct enclosures below σ − β give N(σ − β) ≥ k; "
            "the inertia ν = k of the LDLᴴ factorization of A − σI gives N(σ − β) ≤ ν; the sandwich closes, the "
            "enclosures hold λ₁ … λ_k and λ_{k+1} ≥ σ − β. (b) A window claim without eigenvectors: two counts at the "
            "edges give ν_b − ν_a = #{λ ∈ (a, b)} once two further counts at e ± δ per edge show the collar e ± β to "
            "be eigenvalue-free. (c) What the count refuses: ν > k means an eigenvalue below σ is missing from the "
            "claim (FAILED); an eigenvalue inside the collar leaves the count undecided (INCONCLUSIVE).")


def caption_rounding_accounting(n):
    return ("The floating-point accounting of the Theorem 1 radius, as implemented (docs/theory.md section 2, "
            "`enclosure.residuals`): the residual vectors, the bound on |A||x| and the entrywise rounding bound E, "
            "each with one factor (1 + 4u) covering the roundings of its line; the norms with the guards "
            "(1 ± γ_{n+2}) and (1 + 8u); the radius with a third (1 + 4u); and the model shift σ from S1, added with "
            "`add_up` (one ulp towards +∞) where the radius enters S3, S4 and the exported arrays — added, never "
            "multiplied. Where a line has more roundings than its factor covers, the slack of γ_{w+1} over γ_w, of "
            "γ_{w+2} over the entrywise error and of γ_{n+2} over γₙ/2 + u absorbs the rest; "
            "tests/test_theorem1_accounting.py pins each stage in exact rational arithmetic.")


def caption_failure_matrix(n):
    return (f"Fifteen results checked by three practices (E1 record, {RUN_A}): {n['cases']} cases of which "
            f"{n['wrong_results']} are wrong results, N = {n['N']} sites, K = {n['K']} claimed pairs, residual and "
            f"agreement tolerances {n['residual_tolerance']:g}. P1 (residual check with the producer's own matrix) "
            f"flags {n['P1_flags']} of them, P2 (recomputation from the intended model) flags {n['P2_flags']}, "
            f"certify-ed (P3) leaves {n['P3_not_certified']} not certified (FAILED or INCONCLUSIVE) and localizes the "
            f"error in {n['P3_localizes']} cases. The \"statement named\" column is derived from the diagnosis text "
            f"of the record (`P3_detail`) by keyword rules, because the record stores per-statement statuses for the "
            f"correct result only; F12 is a correct result whose single-vector observable is basis dependent and is "
            f"certified at the multiplet level (S4).")


def caption_cost_memory(n):
    a = n["Windows, certify-ed 3.1.0 (run_full_20261022)"]
    b = n["Linux, certify-ed 3.0.0 (run_full_20261021)"]
    return (f"Certifier wall time (a) and peak memory (b) against block dimension for the {a['blocks']} E2 blocks "
            f"on two stacks: {RUN_A} and {RUN_B}. Filled markers: S3 (completeness) CERTIFIED; open markers: S3 "
            f"NOT_CHECKED because the factorization did not fit under the {n['memory_cap_GiB']:g} GiB cap. On both "
            f"stacks S3 is certified for {a['S3_certified']} blocks up to dimension {a['max_dim_S3_certified']:,} and "
            f"not checked for {a['S3_not_checked']}; the largest block has dimension {a['largest_dim']:,}. Longest "
            f"certifier run: {a['max_seconds']:.0f} s (Windows), {b['max_seconds']:.0f} s (Linux); peak memory "
            f"{a['max_mib']:.0f} MiB and {b['max_mib']:.0f} MiB. Time is `certifier.total`, memory "
            f"`certifier.maxrss_mb`, both read from the records.")


def caption_two_stack(n):
    pc = n["per_class"]
    return (f"Record-by-record comparison of the two runs ({n['run_A']} against {n['run_B']}, "
            f"`{n['csv']}`): {n['leaves']:,} leaves of the eleven experiments' data.json files, each in one class. "
            f"{pc['same']:,} are identical, {pc['float-differs']} differ by a relative {1e-9:g} or less, "
            f"{pc['stack-dependent']} are stack-dependent statistics within 10 %, {pc['fp-statistic']} are "
            f"floating-point statistics, {pc['text-numbers']} are the same sentence with run-stack digits, "
            f"{pc['extended']} is an extension of the newer version, {pc['sample-dependent']} are random draws (E5 "
            f"samples, E6 mutants), {pc['explained']} are differences explained once by path pattern, {pc['timing']} "
            f"are timings or memory, {pc['missing']} are present on one side only; MISMATCH: {pc['MISMATCH']}.")


def caption_mutation(n):
    r, s = n["run_3_1_0"], n["dev_3_2_sample"]
    sc = r["survivor_categories"]
    return (f"Mutation testing. (a) E6 record of {RUN_A}: {r['n']} sampled mutants over {r['modules']} modules, "
            f"{r['killed']} killed and {r['survived']} survived ({r['timeouts']} timeouts; score {r['score']:.3f}); "
            f"the baseline suite took {r['baseline_seconds']:.0f} s. Survivors by category of classification.json: "
            f"{sc['needs inspection']} need inspection, {sc['heuristic or performance parameter (shift placement, method choice, caching)']} "
            f"are heuristic or performance parameters, {sc['rounding allowance in a worst-case bound (looser: sound; tighter: below test resolution)']} "
            f"are rounding allowances in a worst-case bound, {sc['no effect on statements (timing, messages)']} have no "
            f"effect on statements, {sc['strict vs non-strict comparison (differs only at exact equality)']} are "
            f"strict-versus-non-strict comparisons. (b) The {s['n']}-mutant sample on the merged dev/3.2 branch "
            f"(RESULT_integration_3.2.md): {s['killed']} killed; survivors {', '.join('`' + x + '`' for x in s['survivors'])}. "
            f"The two samples come from different code and sample sizes; the figure does not compare their scores.")


def caption_corpus_detection(n):
    return (f"certify-ed on the planted-error corpus (corpus/results/certify-ed/report.json, run started "
            f"{n['started_utc']}): {n['planted']} planted cases in {n['classes'] - 1} error classes and "
            f"{n['controls']} controls. Each cell counts, for the cases of a class, the status of one statement; a "
            f"hatched cell is a refused input (no certificate written); a thick frame marks the statements the corpus "
            f"expects not to be certified for that class. Verdicts: {n['detected']} detected (of which "
            f"{n['detected_by_refusal']} by refusal), {n['missed']} missed, {n['unsound']} unsound, {n['silent']} "
            f"silent, {n['error']} errors, {n['false_alarms']} false alarms ({n['weak_alarms']} weak), "
            f"{n['controls_clean']} of {n['controls']} controls clean.")


def caption_scope_vs_survey(n):
    q = n["quantities"]
    sc = n["status_counts"]
    top = sorted(q, key=lambda k: -q[k]["percent"])
    t1, t2 = top[0], top[1]
    return (f"What ED papers report against what a certificate states. Bars: the share of the n = {n['n']} papers of "
            f"the practice survey's analysis set reporting each quantity (model-coded; Wilson 95 % intervals), "
            f"from experiments/survey/survey_for_scope.json; colours: the status of the matching row of "
            f"docs/SCOPE_MATRIX.md. The two most reported quantities, {t1} ({q[t1]['percent']:g} %, "
            f"{q[t1]['wilson95_percent'][0]:g}–{q[t1]['wilson95_percent'][1]:g} %) and {t2} ({q[t2]['percent']:g} %), are "
            f"certified statements; of the {len(q)} coded categories {sc['certified statement']} are certified "
            f"statements, {sc['derived record, no status']} a derived record without status, "
            f"{sc['nothing certified']} have nothing certified (entanglement entropy: NOT_IMPLEMENTED; "
            f"fidelity: no row), {sc['out of scope by design']} are out of scope by design (dynamics, time evolution, "
            f"extrapolation) and {sc['not classifiable']} (\"other\") is not classifiable. {n['papers_fully_covered']} of "
            f"the {n['coded_papers']} papers report only quantities a statement covers (\"other\" excluded).")


def caption_roadmap_progress(n):
    sc = n["status_counts"]
    pw, dw = n["points_per_wave"], n["done_points_per_wave"]
    return (f"The {n['items']} roadmap items (ROADMAP_expansion_items.csv, status column `{n['status_column']}`) by "
            f"wave (a, points = adoption + rigor) and theme (b, item counts): {sc['done']} done, {sc['partial']} partial "
            f"({', '.join(n['partial_ids'])}), {sc['blocked']} blocked ({', '.join(n['blocked_ids'])}), "
            f"{sc['not started']} not started; {n['points_done']} of {n['points_total']} points done. Wave 0 ({pw['0']} "
            f"points) has {dw['0']} done, Wave 1 ({pw['1']}) has {dw['1']}, Wave 4 ({pw['4']}) has {dw['4']}; Waves 2 "
            f"and 3 ({pw['2']} and {pw['3']} points) are untouched. The wave of each item is read from the wave lists "
            f"of ROADMAP_expansion.md (experiments/figures/data/README.md).")


def caption_crown_sectors(n):
    L, p, ns = n["landscape"], n["printed"], n["named_sectors"]
    return (f"The pyrochlore-crown S = 1 model (E9 record, {RUN_A}). (a) The lowest eigenvalue of each of the "
            f"{L['classes']} pair-spin classes of the S^z = 0 block at the record's re-optimised couplings "
            f"(c1 = {L['couplings']['c1']:.6f}, c3 = {L['couplings']['c3']:.6f}), sorted; marker area grows with the "
            f"class dimension. The lowest class {tuple(t // 2 for t in L['min_class']['pair_spins2'])} (dimension "
            f"{L['min_class']['dim']:,}), the all-singlet class (dimension {L['all_singlet_class']['dim']}) and the largest "
            f"class (dimension {L['largest_class']['dim']:,}) lie within "
            f"{max(L['min_class']['energy'], L['all_singlet_class']['energy'], L['largest_class']['energy']) - min(L['min_class']['energy'], L['all_singlet_class']['energy'], L['largest_class']['energy']):.1e} "
            f"of each other. The record holds the landscape at these couplings only, not at the printed ones. "
            f"(b) At the printed couplings (c1 = {p['c1']}, c3 = {p['c3']}, c17 = {p['c17']}): the all-singlet-sector "
            f"eigenvalue is certified ≤ {p['eigenvalue_upper']:.7f} by the exact re-check ({p['exact_recheck']}; radius of the "
            f"spin-1 ring eigenvalue it rests on {p['E0_radius_exact']:.1e}; no floating-point assumption), {p['margin']:.1e} below the published lower "
            f"bound {p['published_bound']}; the two named sectors {tuple(ns[0]['pair_spins'])} (dimension {ns[0]['dim']:,}) "
            f"and {tuple(ns[1]['pair_spins'])} (dimension {ns[1]['dim']:,}) have lowest eigenvalues "
            f"{ns[0]['energy']:.7f} and {ns[1]['energy']:.7f}, floating-point values without a certified interval in "
            f"the record, drawn as such.")


def caption_aklt_eta(n):
    L = n["lattices"]
    cv = n["criterion_values"]
    return (f"The Pomata–Wei AKLT lattices (E10 record, {RUN_A}). (a) The certified upper bound on η for each lattice "
            f"(bar) against the published value (diamond); the largest difference is "
            f"{n['max_abs_diff_upper_minus_published']:.1e}, so bar end and diamond coincide at this scale and the values "
            f"are printed. Orange ticks: the gap-criterion threshold 1/z for the lattices whose criterion is η < 1/z "
            f"(z = {L['star']['criterion_z']} for the star, decorated honeycomb and decorated square, "
            f"z = {L['cross']['criterion_z']} for the cross); the two square-octagon rows share the joint criterion "
            f"4η⊥ + 2η∥ < 1. (b) The certified criterion values: star {cv['star [3.12.12] 4 η']:.5f}, cross "
            f"{cv['cross [4.6.12] 3 η']:.5f}, decorated honeycomb {cv['decorated honeycomb 4 η']:.5f}, decorated square "
            f"{cv['decorated square 4 η']:.5f}, square-octagon {cv['square-octagon 4 η⊥ + 2 η∥']:.5f}; every one is below 1. "
            f"The largest symmetry-adapted block is {L['decorated_square']['largest_block']:,} (decorated square, "
            f"{L['decorated_square']['seconds']:.0f} s).")


FIGURES = [  # (name, label, caption function, data source)
    ("fig_certificate_anatomy", "C1", caption_certificate_anatomy, "concept figure, no data"),
    ("fig_enclosure_geometry", "C2", caption_enclosure_geometry, "concept figure, no data (docs/theory.md sections 2–3)"),
    ("fig_inertia_completeness", "C3", caption_inertia_completeness, "concept figure, no data (docs/theory.md section 3)"),
    ("fig_rounding_accounting", "C4", caption_rounding_accounting, "concept figure, no data (docs/theory.md section 2)"),
    ("fig_failure_matrix", "E1", caption_failure_matrix, "{record}; run " + RUN_A),
    ("fig_cost_memory", "E2", caption_cost_memory, "{record_A} of " + RUN_A + "; {record_B} of " + RUN_B),
    ("fig_two_stack", "E3", caption_two_stack, "{csv} at the root of " + RUN_A),
    ("fig_mutation", "E4", caption_mutation, "{record} and {classification} of " + RUN_A + "; {integration_result} (dev/3.2)"),
    ("fig_corpus_detection", "E5", caption_corpus_detection, "{report} (repository, corpus run on dev/3.2)"),
    ("fig_scope_vs_survey", "E6", caption_scope_vs_survey, "experiments/survey/survey_for_scope.json, experiments/survey/survey_coded.csv, docs/SCOPE_MATRIX.md"),
    ("fig_roadmap_progress", "E7", caption_roadmap_progress, "{source}"),
    ("fig_crown_sectors", "A1", caption_crown_sectors, "{record} of " + RUN_A),
    ("fig_aklt_eta", "A2", caption_aklt_eta, "{record} of " + RUN_A),
]

HEADER = """# Figure index

Figures for the manuscript, the consolidated report and this documentation site. Each figure is one script in
`experiments/figures/` (listed in `provenance.NON_COMPUTATIONAL`: the scripts read records and documents and draw;
none computes a bound) and is written as PDF and PNG into this directory together with `<name>.numbers.json`,
the numbers the caption prints. This file is rendered by `experiments/figures/make_index.py` from those files; every
number below was read from the data source named in the entry. Concept figures carry no data. The evidence run is
`run_full_20261022` (Windows, certify-ed 3.1.0), compared with `run_full_20261021` (Linux, certify-ed 3.0.0); the
record directories are those of the project archive (`code/certify-ed-3.1/results/`). To regenerate:
`CERTIFY_RESULTS=<run dir> CERTIFY_RESULTS_B=<second run dir> python experiments/figures/<script>.py`, then
`python experiments/figures/make_index.py`. The smoke test is `tests/test_figures.py`.

Placement in the manuscript and the report: `MANUSCRIPT_PLACEMENT.md`.

"""


def render():
    out = [HEADER]
    for name, label, fn, source in FIGURES:
        n = N(name)
        out.append(f"## {label}  `{name}.pdf` / `{name}.png`\n\n")
        out.append(f"![{name}]({name}.png)\n\n")
        out.append(fn(n) + "\n\n")
        out.append(f"- data source: {source.format(**{k: v for k, v in n.items() if isinstance(v, str)})}\n")
        out.append(f"- script: `experiments/figures/{name}.py`\n\n")
    return "".join(out)


def main():
    text = render()
    path = os.path.join(FIG, "INDEX.md")
    if "--check" in sys.argv:
        current = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if current != text:
            print("INDEX.md is out of date: run python experiments/figures/make_index.py", file=sys.stderr)
            sys.exit(1)
        print("INDEX.md is in sync")
        return
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(path)


if __name__ == "__main__":
    main()
