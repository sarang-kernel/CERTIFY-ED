"""Classify surviving mutants of E6 by what the mutated line does (rules below), and list the ones that
need human inspection. Output: results/<latest e6>/classification.json"""
from pathlib import Path
import ast, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run = sorted(glob.glob(os.path.join(ROOT, "experiments/results/e6_mutation_*")))[-1 if len(sys.argv) < 2 else int(sys.argv[1])]
d = json.loads(Path(run, "data.json").read_text())

ROUNDING = re.compile(r"gamma|gamma_dot|\bU\b|\bup\(|_GUARD|1e-1[0-9]|\* *U\b|safe_sqrt_upper|norm2_lower|inf_norm_upper|fro_upper")
TIMING = re.compile(r"time\.time|\[:\d+\]|f\"|f'|msg|note|summary|reason")
HEUR = re.compile(r"gap_fractions|tries|range\(4\)|dense_max|2500|EXACT_GRAM_BUDGET|4e7|2 \*\* 24|step = |cache|DEFAULT_REL|probe_vectors|seed|limit_denominator|scale = max")

def func_of(path, line):
    tree = ast.parse(open(path).read())
    best = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.lineno <= line <= n.end_lineno:
            if best is None or n.lineno > best.lineno:
                best = n
    return best.name if best else "<module>"

rows, counts = [], {}
for m in d["mutants"]:
    if m["status"] != "survived":
        continue
    path = os.path.join(ROOT, "certify_ed", m["module"])
    src = Path(path).read_text().splitlines()[m["line"] - 1]
    mut = m["mutation"]
    if TIMING.search(src) and not ROUNDING.search(src):
        cat = "no effect on statements (timing, messages)"
    elif HEUR.search(src):
        cat = "heuristic or performance parameter (shift placement, method choice, caching)"
    elif ROUNDING.search(src):
        cat = "rounding allowance in a worst-case bound (looser: sound; tighter: below test resolution)"
    elif re.match(r"(Lt|LtE|Gt|GtE)->", mut):
        cat = "strict vs non-strict comparison (differs only at exact equality)"
    else:
        cat = "needs inspection"
    counts[cat] = counts.get(cat, 0) + 1
    rows.append({"module": m["module"], "line": m["line"], "function": func_of(path, m["line"]),
                 "mutation": mut, "category": cat, "source": src.strip()})
out = {"run": os.path.basename(run), "summary": d["summary"], "survivor_categories": counts, "survivors": rows}
json.dump(out, open(os.path.join(run, "classification.json"), "w"), indent=1)
print(json.dumps(counts, indent=1))
for r in rows:
    if r["category"] == "needs inspection":
        print(f"{r['module']}:{r['line']} [{r['function']}] {r['mutation']} | {r['source'][:90]}")
