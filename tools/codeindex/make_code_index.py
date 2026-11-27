"""make_code_index.py: write docs/project/CODE_INDEX.md, one row for every program file of certify-ed at one revision.

Usage:  python tools/codeindex/make_code_index.py [REPO] [--rev REV] [--out FILE] [--check]
        (defaults: REPO = archive_v5/code/certify-ed, REV = HEAD, FILE = docs/project/CODE_INDEX.md)

The files are read from the commit with `git ls-tree` and `git show`; nothing is checked out and nothing is written
to the repository. A program file is a tracked file ending in .py, .c or .sh. For each one the script records:

  the number of lines of the committed file;
  one sentence on what the file does: the first sentence of the module docstring, or, for the files listed in
  purposes.py (no docstring, a heading only, a vague first sentence, or a word the style rules exclude), the
  sentence written there by reading the file;
  the public top-level functions and classes, the number of test functions of a test file, the command-line
  arguments of a script (the first argument of every add_argument call), whether it has a __main__ block, and
  the files it reads and writes (open, json, numpy, savefig and similar calls; a name is shown when the code
  writes it as a constant);
  the package modules the file imports, and, for a package module, how many files import it;
  whether the file is part of the source fingerprint of certify_ed/provenance.py (`source`), excluded from it by
  NON_COMPUTATIONAL (`excluded`), part of the test fingerprint (`tests`), or outside both (`none`);
  the statement ids S1 to S8 named in the docstring and, for files under experiments/ and tests/ and run.py,
  the experiment ids E0 to E10 (elsewhere such labels name cases or errors, not experiments).

Every sentence is checked with tools/lint_docs.py; the script stops with exit code 2 and lists the files whose
sentence fails, so that a sentence for them can be written in purposes.py. --check exits with 1 when FILE differs
from what would be written. Standard library only.
"""
import ast
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from purposes import PURPOSE  # noqa: E402
import lint_docs  # noqa: E402

SUFFIXES = (".py", ".c", ".sh")
WIDTH = 118
ABBREV = {"eq", "eqs", "j", "al", "fig", "figs", "sec", "ref", "no", "cf", "vs", "etc", "e.g", "i.e", "prop", "thm",
          "dr", "st", "approx"}
KEEP_CASE = {"symveig", "certify", "numpy", "scipy", "quspin", "cev"}   # names written in lower case
GROUPS = [
    ("certify_ed/", "The package `certify_ed/`",
     "The library itself: the statements, the arithmetic they rest on, the importers, the applications and the "
     "command line."),
    ("experiments/", "Experiments `experiments/`",
     "The experiment scripts E0 to E10, the scripts outside the evidence runs, the number and table scripts, the "
     "comparison of runs and the figure scripts."),
    ("tests/", "Tests `tests/`", "The test suite; its files form the test fingerprint."),
    (("verifier/", "corpus/", "conformance/"), "Verifier, corpus and conformance",
     "The independent C verifier and its validation, the planted-error corpus and the reference producer bundles."),
    (("examples/", "run.py"), "Examples and the driver", "The example scripts and `run.py`."),
    ("docs/", "Scripts under `docs/`",
     "The scripts that test the derivations of `docs/proofs/`, the independent adversarial searches, the cookbook "
     "snippets and the API page generator."),
    ("lean/", "Scripts under `lean/`",
     "The Python side of the Lean formalization: the export of certificates and applications to Lean, the runs of "
     "the checkers and the second kernel, and the scripts that write the audit and result records."),
    ("tools/", "Repository tools `tools/`",
     "The scripts that write and check repository documents: the changelog, this index and the source hash, the "
     "documentation lint, the roadmap, the Lean status tables and the verification report."),
    ("legacy/", "Legacy code `legacy/`",
     "Code of earlier stages kept for the record: version 1, symveig 1.0.0 and the application scripts of "
     "17 October 2026. Nothing in the package imports it; the imports of `certify_ed` in the files of version 1 "
     "refer to the version 1 package, which had the same name."),
]


def git(repo, *args):
    return subprocess.run(["git", "-c", "core.filemode=false", *args], cwd=repo, capture_output=True,
                          check=True).stdout.decode("utf-8", errors="replace")


def non_computational(repo, rev):
    src = git(repo, "show", f"{rev}:certify_ed/provenance.py")
    m = re.search(r"NON_COMPUTATIONAL = (\(.*?\))\n\n", src, re.S)
    return ast.literal_eval(m.group(1)) if m else ()


def fingerprint(path, nc):
    if path.startswith(("certify_ed/", "experiments/")):
        excluded = any(path == x or path.startswith(x.rstrip("/") + "/") for x in nc)
        return "excluded" if excluded else "source"
    return "tests" if path.startswith("tests/") else "none"


def first_sentence(doc):
    """The first sentence of a docstring; reST headings (a line of = or -) are joined to the next paragraph."""
    if not doc:
        return ""
    paras = [p for p in re.split(r"\n\s*\n", doc.strip()) if p.strip()]
    clean = []
    for p in paras:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip() and not re.fullmatch(r"[=\-~^]{3,}", ln.strip())]
        if lines:
            clean.append(" ".join(lines))
    if not clean:
        return ""
    head = clean[0]
    if len(clean) > 1 and not re.search(r"[.:?!)]$", head) and len(head) < 70:
        nxt = clean[1]
        head = nxt if re.fullmatch(r"[\w./-]+\.py", head) else head + ": " + nxt
    head = re.sub(r"^[\w./-]+\.(py|c)\s*[:,]\s*", "", head)              # "name.py: ..." prefix
    head = re.sub(r"^[A-Z]\d+ fig_\w+:\s*", "", head)                    # figure-script prefix "E1 fig_x:"
    for m in re.finditer(r"[.?!](?=\s+[A-Z(`\"']|\s*$)", head):
        before = head[:m.start()]
        word = re.search(r"([\w.]+)$", before)
        w = word.group(1).lower() if word else ""
        if m.group(0) == "." and (w in ABBREV or len(w) == 1 or before.endswith(".")):
            continue
        head = head[:m.end()]
        break
    head = head.strip()
    first = head.split(" ", 1)[0]
    if head[:1].islower() and "_" not in first and first not in KEEP_CASE and not first.startswith("certify-ed"):
        head = head[:1].upper() + head[1:]
    return head


def sanitize(s):
    s = re.sub(r"\s*[\u2014\u2015]\s*", ", ", s)
    s = s.replace("\u2013", "-").replace("\u2012", "-").replace("|", "/")
    if s.count("`") % 2:
        s = s.replace("`", "")
    parts = s.split("`")                                              # escape < and > outside code spans
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("<", "&lt;").replace(">", "&gt;")
    return "`".join(parts)

WRITE_CALLS = {"dump", "savefig", "save", "savez", "savez_compressed", "to_csv", "write_text", "write_bytes",
               "copyfile", "copytree", "write_bundle", "write_array", "write_operator", "writerow"}
READ_CALLS = {"load", "read_csv", "read_text", "read_bytes", "File", "read_bundle", "read_array",
              "read_operator", "load_npz", "loadtxt", "genfromtxt"}


def literal_name(node):
    """A file name written as a constant in the code: "x.json", os.path.join(..., "x.json"), f"{a}_x.json"."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        v = node.value
    elif isinstance(node, ast.JoinedStr):
        v = "".join(p.value if isinstance(p, ast.Constant) else "{...}" for p in node.values)
    elif isinstance(node, ast.Call) and node.args and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
            "join", "Path"):
        return literal_name(node.args[-1])
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return literal_name(node.right)
    else:
        return None
    v = v.replace("\\", "/").rsplit("/", 1)[-1]
    return v if re.search(r"\.[A-Za-z0-9]{1,6}$", v) and " " not in v and len(v) < 60 else None


def scan_io(tree):
    reads, writes = [], []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fname = getattr(n.func, "attr", getattr(n.func, "id", ""))
        owner = getattr(getattr(n.func, "value", None), "id", "")
        if fname == "copy" and owner == "shutil":
            fname = "copyfile"
        if fname in ("save", "load") and owner not in ("np", "numpy", "json", "pickle", "torch", "sp", "sparse"):
            continue
        target = None
        if fname == "open" and n.args:
            mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
            mode = mode.value if isinstance(mode, ast.Constant) else (mode if isinstance(mode, str) else "r")
            target = writes if any(c in str(mode) for c in "wax") else reads
            name = literal_name(n.args[0])
        elif fname in WRITE_CALLS:
            target = writes
            args = n.args[1:] if fname in ("dump",) else n.args
            name = next((x for x in (literal_name(a) for a in args) if x), None)
        elif fname in READ_CALLS:
            target = reads
            name = next((x for x in (literal_name(a) for a in n.args) if x), None)
        else:
            continue
        target.append(name)
    return reads, writes


def io_text(reads, writes):
    parts = []
    for word, lst in (("reads", reads), ("writes", writes)):
        if not lst:
            continue
        named = []
        for x in lst:
            if x and x not in named:
                named.append(x)
        parts.append(f"{word} " + (names(named, 4) if named else "files"))
    return parts


def module_name(path):
    if not path.startswith("certify_ed/") or not path.endswith(".py"):
        return None
    m = path[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


def scan_python(path, src, modules):
    r = {"doc": "", "defs": [], "tests": 0, "options": [], "imports": set(), "reads": [], "writes": [],
         "main": False}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return r
    r["doc"] = ast.get_docstring(tree) or ""
    r["reads"], r["writes"] = scan_io(tree)
    r["main"] = any(isinstance(n, ast.If) and "__main__" in ast.dump(n.test) for n in tree.body)
    r["defs"] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and not n.name.startswith("_")]
    pkg = module_name(path)
    if pkg and not path.endswith("__init__.py"):
        pkg = pkg.rsplit(".", 1)[0]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            r["tests"] += 1
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value not in r["options"]:
                r["options"].append(a.value)
        elif isinstance(n, ast.Import):
            for al in n.names:
                if al.name.startswith("certify_ed"):
                    r["imports"].add(al.name)
        elif isinstance(n, ast.ImportFrom):
            if n.level and pkg:
                base = pkg.split(".")
                base = base[: len(base) - (n.level - 1)] if n.level > 1 else base
                mod = ".".join(base + ([n.module] if n.module else []))
            else:
                mod = n.module or ""
            if not mod.startswith("certify_ed"):
                continue
            subs = [f"{mod}.{al.name}" for al in n.names if f"{mod}.{al.name}" in modules]
            r["imports"].update(subs if subs else [mod])
    r["imports"].discard(module_name(path))
    return r


def names(items, limit=8):
    items = list(items)
    shown = ", ".join(f"`{x}`" for x in items[:limit])
    return shown + (f" (+{len(items) - limit})" if len(items) > limit else "")


def contents(path, r, importers):
    parts = []
    if r["tests"] and path.startswith(("tests/", "legacy/v1/tests/")):
        parts.append(f"{r['tests']} test functions")
    elif r["defs"]:
        parts.append("defines " + names(r["defs"]))
    if r["options"]:
        parts.append("arguments " + names(r["options"], 10))
    if r.get("main") and not path.startswith("tests/"):
        parts.append("runs as a script")
    parts += io_text(r.get("reads", []), r.get("writes", []))
    imps = sorted(m[len("certify_ed."):] if m != "certify_ed" else "certify_ed" for m in r["imports"])
    if imps:
        parts.append("imports " + names(imps, 6))
    mod = module_name(path)
    if mod and importers.get(mod):
        k = len(importers[mod])
        parts.append(f"imported by {k} file{'s' if k != 1 else ''}")
    doc = re.sub(r"^[A-Z]\d+ fig_\w+:", "", r["doc"])                  # figure codes are not experiment ids
    ids = set(re.findall(r"(?<!\()\b(S[1-8]|E(?:10|[0-9]))\b(?!\))", doc))   # "(S2)" labels a stage, not S2
    if not path.startswith(("experiments/", "tests/", "run.py")):   # elsewhere E1, E2 ... label cases or errors
        ids = {x for x in ids if x.startswith("S")}
    ids = sorted(ids, key=lambda x: (x[0], int(x[1:])))
    if ids:
        parts.append("names " + ", ".join(ids))
    return "; ".join(parts) if parts else "no top-level definitions"


def group_of(path):
    for prefixes, title, intro in GROUPS:
        p = prefixes if isinstance(prefixes, tuple) else (prefixes,)
        if any(path == x or path.startswith(x) for x in p):
            return title
    return "Other files"


def wrap(par):
    """Wrap a paragraph at WIDTH columns without breaking an inline code span."""
    tokens, buf = [], []
    for w in par.split(" "):
        buf.append(w)
        if " ".join(buf).count("`") % 2 == 0:
            tokens.append(" ".join(buf))
            buf = []
    tokens += [" ".join(buf)] if buf else []
    lines, cur = [], ""
    for t in tokens:
        if cur and len(cur) + 1 + len(t) > WIDTH:
            lines.append(cur)
            cur = t
        else:
            cur = f"{cur} {t}" if cur else t
    return "\n".join(lines + [cur])


def build(repo, rev):
    commit = git(repo, "rev-parse", "--short", rev + "^{commit}").strip()
    tracked = [p for p in git(repo, "ls-tree", "-r", "--name-only", rev).splitlines() if p.endswith(SUFFIXES)]
    nc = non_computational(repo, rev)
    modules = {module_name(p) for p in tracked if module_name(p)}
    rows = {}
    for p in tracked:
        src = git(repo, "show", f"{rev}:{p}")
        r = scan_python(p, src, modules) if p.endswith(".py") else {"doc": "", "defs": [], "tests": 0, "options": [],
                                                                    "imports": set(), "reads": [], "writes": [],
                                                                    "main": False}
        r["lines"] = src.count("\n") + (0 if src.endswith("\n") or not src else 1)
        r["fingerprint"] = fingerprint(p, nc)
        r["purpose"] = sanitize(PURPOSE.get(p) or first_sentence(r["doc"]))
        r["from_purposes"] = p in PURPOSE
        rows[p] = r
    importers = {}
    for p, r in rows.items():
        for m in r["imports"]:
            importers.setdefault(m, set()).add(p)
    failing = [p for p, r in rows.items() if not r["purpose"] or lint_docs.lint(r["purpose"])]
    return commit, nc, rows, importers, failing


def render(repo, rev):
    commit, nc, rows, importers, failing = build(repo, rev)
    if failing:
        sys.stderr.write("no usable sentence for (write one in purposes.py):\n  " + "\n  ".join(failing) + "\n")
        sys.exit(2)
    by_group = {}
    for p in sorted(rows):
        by_group.setdefault(group_of(p), []).append(p)
    counts = {k: sum(1 for p in rows if rows[p]["fingerprint"] == k) for k in ("source", "excluded", "tests", "none")}
    n_hand = sum(1 for r in rows.values() if r["from_purposes"])
    total_lines = sum(r["lines"] for r in rows.values())
    out = ["# Code index", ""]
    out.append(wrap(
        f"This index lists every program file of the certify-ed repository at commit `{commit}` (revision `{rev}` of "
        f"`archive_v5/code/certify-ed`): {len(rows)} files ending in `.py`, `.c` or `.sh`, with "
        f"{total_lines} lines in all. It is written by `tools/codeindex/make_code_index.py` from the committed "
        f"files, and `python tools/codeindex/make_code_index.py --check` reports whether it is up to date. Apart from "
        f"the introductions of the sections, which are part of the generator, and the sentences of "
        f"`tools/codeindex/purposes.py`, every entry is taken from the files."))
    out.append("")
    out.append(wrap(
        "The columns are the following. Lines is the number of lines of the committed file. What it does is the "
        "first sentence of the module docstring; for the "
        f"{n_hand} files listed in `tools/codeindex/purposes.py` it is a sentence written by reading the file, "
        "because the file has no docstring, its docstring opens with a heading only or a sentence too vague to "
        "stand alone, or the sentence uses a word that the style rules of this documentation exclude. Contents "
        "lists the public top-level functions and classes (or the number of test functions of a test file), the "
        "command-line arguments of a script, whether the file runs as a script (a `__main__` block), the files it "
        'reads and writes (named where the code gives the name as a constant, otherwise shown as "reads files" or '
        '"writes files"; files read or '
        "written through helpers such as `experiments/harness.py` are not traced), the package modules the file "
        "imports, how many files import a package "
        "module, the statement ids S1 to S8 the docstring names and, for the experiments and the tests, the "
        "experiment ids E0 to E10 it names; the ids show what a file says it "
        "serves and are not the result of a dependency analysis. Fingerprint says whether the file counts in the "
        "source fingerprint of `certify_ed/provenance.py` (`source`), is excluded from it by `NON_COMPUTATIONAL` "
        "(`excluded`), counts in the test fingerprint (`tests`), or is outside both (`none`); see "
        "`docs/project/software/05_provenance.md`."))
    out.append("")
    out.append(wrap(
        f"Of these program files, {counts['source']} count in the source fingerprint, {counts['excluded']} are excluded "
        f"by the {len(nc)} entries of `NON_COMPUTATIONAL`, {counts['tests']} count in the test fingerprint and "
        f"{counts['none']} are outside both."))
    out.append("")
    out.append("| Group | Files | Lines |")
    out.append("|---|---|---|")
    for _, title, _ in GROUPS:
        ps = by_group.get(title, [])
        if ps:
            out.append(f"| {title} | {len(ps)} | {sum(rows[p]['lines'] for p in ps)} |")
    for prefixes, title, intro in GROUPS:
        ps = by_group.get(title, [])
        if not ps:
            continue
        out += ["", f"## {title}", "", wrap(intro), "", "| File | Lines | What it does | Contents | Fingerprint |",
                "|---|---|---|---|---|"]
        for p in ps:
            r = rows[p]
            out.append(f"| `{p}` | {r['lines']} | {r['purpose']} | {contents(p, r, importers)} | {r['fingerprint']} |")
    if by_group.get("Other files"):
        raise SystemExit("files outside every group: " + ", ".join(by_group["Other files"]))
    return "\n".join(out) + "\n"


def main(argv):
    args = list(argv)
    repo = args.pop(0) if args and os.path.isdir(args[0]) else os.path.join(os.environ.get("CERTIFYED_ARCHIVE") or os.path.normpath(os.path.join(ROOT, "..", "archive", "archive_v5")), "code",
                                                                             "certify-ed")
    if not os.path.isdir(repo):
        sys.exit("code checkout not found at " + repo + ": set CERTIFYED_ARCHIVE to the unpacked archive v5, or pass a checkout of the code as the first argument (this repository works: . --rev a814d28)")
    rev = args[args.index("--rev") + 1] if "--rev" in args else "HEAD"
    out = args[args.index("--out") + 1] if "--out" in args else os.path.join(ROOT, "docs", "project", "CODE_INDEX.md")
    text = render(repo, rev)
    probs = lint_docs.lint(text)
    if probs:
        for i, kind, line in probs:
            sys.stderr.write(f"{out}:{i}: {kind}: {line}\n")
        return 2
    if "--check" in args:
        with open(out, encoding="utf-8") as f:
            same = f.read() == text
        print(f"{out}: {'up to date' if same else 'out of date'}")
        return 0 if same else 1
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{out}: {text.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
