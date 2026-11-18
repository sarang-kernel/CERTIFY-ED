"""make_bundle.py: collect the files cited by blog/BLOG.md into one text bundle for the number check.

Usage (from the root of the repository):

    python blog/make_bundle.py
    python tools/lint_docs.py --bundle blog/blog_sources_bundle.txt blog/BLOG.md

Every path written in backticks in BLOG.md is resolved first relative to the repository root and then relative to
sources/archive_v5/. Each existing text file is appended to blog/blog_sources_bundle.txt; binary files (images,
archives, PDFs) and folders are skipped and listed on standard output. For every figure in blog/figures/ the caption
numbers of its source (the <name>.numbers.json next to the source PNG and the INDEX.md of that folder) are added as
well, because the captions of BLOG.md copy their numbers from those files. The script reads only; it changes nothing
outside blog/blog_sources_bundle.txt. Standard library only.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARCHIVE = os.path.join(ROOT, "sources", "archive_v5")
BLOG = os.path.join(HERE, "BLOG.md")
OUT = os.path.join(HERE, "blog_sources_bundle.txt")
TEXT_EXT = {".md", ".txt", ".json", ".csv", ".tsv", ".log", ".tex", ".py", ".cff", ".toml"}
CODE_DOCS = os.path.join(ARCHIVE, "code", "certify-ed", "docs")

# figure in blog/figures -> (folder of the source figure, source file name)
FIGURES = {
    "fig_scope_vs_survey.png": (os.path.join(CODE_DOCS, "figures"), "fig_scope_vs_survey"),
    "fig_certificate_anatomy.png": (os.path.join(CODE_DOCS, "figures"), "fig_certificate_anatomy"),
    "fig_enclosure_geometry.png": (os.path.join(CODE_DOCS, "figures"), "fig_enclosure_geometry"),
    "fig_inertia_completeness.png": (os.path.join(CODE_DOCS, "figures"), "fig_inertia_completeness"),
    "fig_two_stack_3.1.0_vs_3.0.0.png": (os.path.join(CODE_DOCS, "figures"), "fig_two_stack"),
    "fig_corpus_detection.png": (os.path.join(CODE_DOCS, "figures"), "fig_corpus_detection"),
    "fig_roadmap_progress.png": (os.path.join(CODE_DOCS, "figures"), "fig_roadmap_progress"),
    "fig_failure_matrix_run_3.2.1.png": (os.path.join(CODE_DOCS, "figures_run_3.2.1"), "fig_failure_matrix"),
    "fig_cost_memory_run_3.2.1.png": (os.path.join(CODE_DOCS, "figures_run_3.2.1"), "fig_cost_memory"),
    "fig_mutation_run_3.2.1.png": (os.path.join(CODE_DOCS, "figures_run_3.2.1"), "fig_mutation"),
    "fig_two_stack_3.2.1_vs_3.1.0.png": (os.path.join(CODE_DOCS, "figures_run_3.2.1"), "fig_two_stack"),
    "fig_crown_sectors_run_3.2.1.png": (os.path.join(CODE_DOCS, "figures_run_3.2.1"), "fig_crown_sectors"),
    "fig_aklt_eta_run_3.2.1.png": (os.path.join(CODE_DOCS, "figures_run_3.2.1"), "fig_aklt_eta"),
    "fig_symveig_width_vs_dim_norm.png": (os.path.join(ROOT, "docs", "applications"), "symveig_e8"),
}


CODE = os.path.join(ARCHIVE, "code", "certify-ed")
PROOFS = os.path.join(CODE, "docs", "proofs")


def resolve(path):
    """Return the existing file for a cited path, or None.

    A path is tried relative to the repository root, to sources/archive_v5/, to the code clone (for names such as
    experiments/compare_runs.py) and to its folder of derivation documents (for bare names such as cor5_pairing.md).
    """
    for base in (ROOT, ARCHIVE, CODE, PROOFS):
        full = os.path.normpath(os.path.join(base, path))
        if os.path.isfile(full):
            return full
    return None


def main():
    text = open(BLOG, encoding="utf-8").read()
    cited = []
    for span in re.findall(r"`([^`]+)`", text):
        if "/" in span and " " not in span.strip() or span.endswith(".md"):
            cited.append(span.strip())
    # paths with a space are written in backticks too (the survey of horizons)
    cited += [s for s in re.findall(r"`([^`]*/[^`]*\.md)`", text) if " " in s]
    seen, parts, skipped = set(), [], []
    for c in cited:
        full = resolve(c)
        if full is None or full in seen:
            if full is None:
                skipped.append(c)
            continue
        seen.add(full)
        if os.path.splitext(full)[1].lower() not in TEXT_EXT:
            skipped.append(c)
            continue
        parts.append("\n\n===== " + c + "\n\n" + open(full, encoding="utf-8", errors="replace").read())
    for fig, (folder, stem) in FIGURES.items():
        for name in (stem + ".numbers.json", "INDEX.md", stem + ".md"):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and full not in seen:
                seen.add(full)
                parts.append("\n\n===== " + os.path.relpath(full, ROOT) + "\n\n"
                             + open(full, encoding="utf-8", errors="replace").read())
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(parts))
    print(f"{len(seen)} files, {sum(len(p) for p in parts)} characters -> {os.path.relpath(OUT, ROOT)}")
    for s in sorted(set(skipped)):
        print("skipped (not a text file or not found):", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
