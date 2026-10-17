"""E6: mutation analysis of CERTIFY-ED itself.

Measures how strong the test suite is: each mutant changes one operator or constant in the
verification core (arithmetic, comparison, boolean and sign mutations), the v2 test suite runs
against it, and the mutant is 'killed' if any test fails or errors. Survivors are either test gaps or
equivalent mutants (changes with no observable effect); they are listed for inspection.

usage: python3 e6_mutation.py [n_mutants] [workers]
"""
import ast
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from harness import Run, ROOT

TARGETS = ["construct.py", "enclosure.py", "inertia.py", "eigenspace.py", "sectors.py", "compose.py",
           "fp.py", "localops.py", "spec.py", "basis.py", "su2.py", "checker.py", "certify.py",
           "exact_sectors.py"]

BINSWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
           ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult}
CMPSWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
           ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                out.add(id(node.body[0].value))
    return out


def mutation_sites(tree):
    doc = _docstring_nodes(tree)
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in BINSWAP:
            sites.append(("binop", node))
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMPSWAP:
            sites.append(("compare", node))
        elif isinstance(node, ast.BoolOp):
            sites.append(("boolop", node))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.Not)):
            sites.append(("unary", node))
        elif (isinstance(node, ast.Constant) and type(node.value) in (int, float) and id(node) not in doc
              and not isinstance(node.value, bool)):
            sites.append(("constant", node))
    return sites


def describe(kind, node):
    if kind == "binop":
        return f"{type(node.op).__name__}->{BINSWAP[type(node.op)].__name__}"
    if kind == "compare":
        return f"{type(node.ops[0]).__name__}->{CMPSWAP[type(node.ops[0])].__name__}"
    if kind == "boolop":
        return f"{type(node.op).__name__}->{'Or' if isinstance(node.op, ast.And) else 'And'}"
    if kind == "unary":
        return f"drop {type(node.op).__name__}"
    return f"constant {node.value!r}->{_newconst(node.value)!r}"


def _newconst(v):
    if isinstance(v, int):
        return v + 1 if v != 1 else 2
    return v * 2 if v != 0 else 1.0


def apply_mutation(source, index):
    tree = ast.parse(source)
    kind, node = mutation_sites(tree)[index]
    desc = describe(kind, node)
    line = getattr(node, "lineno", None)
    if kind == "binop":
        node.op = BINSWAP[type(node.op)]()
    elif kind == "compare":
        node.ops = [CMPSWAP[type(node.ops[0])]()]
    elif kind == "boolop":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "unary":
        # replace the UnaryOp by its operand: walk parents
        for parent in ast.walk(tree):
            for field, value in ast.iter_fields(parent):
                if value is node:
                    setattr(parent, field, node.operand)
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        if v is node:
                            value[i] = node.operand
    else:
        node.value = _newconst(node.value)
    return ast.unparse(tree), desc, line


def run_tests(workdir, timeout):
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            "tests/v2"], cwd=workdir, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    except subprocess.TimeoutExpired:
        return "timeout", None, time.time() - t
    first = next((l.split()[1] for l in p.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))), None)
    return ("survived" if p.returncode == 0 else "killed"), first, time.time() - t


def main():
    n_mut = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with Run("e6_mutation") as run:
        rng = random.Random(run.seed)
        pkg = os.path.join(ROOT, "certify_ed")
        pool = []
        counts = {}
        for f in TARGETS:
            src = open(os.path.join(pkg, f)).read()
            n = len(mutation_sites(ast.parse(src)))
            counts[f] = n
            pool += [(f, i) for i in range(n)]
        sample = rng.sample(pool, min(n_mut, len(pool)))
        run.log(f"{len(pool)} mutation sites in {len(TARGETS)} modules; sampling {len(sample)}")
        run.record("sites_per_module", counts)
        # one private copy of the repository per worker
        copies = []
        for w in range(workers):
            d = tempfile.mkdtemp(prefix=f"mut{w}_")
            shutil.copytree(os.path.join(ROOT, "certify_ed"), os.path.join(d, "certify_ed"))
            shutil.copytree(os.path.join(ROOT, "tests"), os.path.join(d, "tests"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            copies.append(d)
        # baseline: unmutated (unparsed) sources must pass
        base_status, base_first, base_t = run_tests(copies[0], 900)
        run.record("baseline", {"status": base_status, "seconds": base_t})
        run.log(f"baseline {base_status} in {base_t:.0f}s")
        if base_status != "survived":
            raise SystemExit("baseline test run fails; aborting")
        timeout = max(120, 4 * base_t)
        results = []

        def work(job):
            k, (f, i) = job
            d = copies[k % workers]
            path = os.path.join(d, "certify_ed", f)
            orig = open(os.path.join(pkg, f)).read()
            mutated, desc, line = apply_mutation(orig, i)
            with open(path, "w") as fh:
                fh.write(mutated)
            try:
                status, first, secs = run_tests(d, timeout)
            finally:
                with open(path, "w") as fh:
                    fh.write(orig)
            return {"module": f, "site": i, "line": line, "mutation": desc, "status": status,
                    "first_failure": first, "seconds": secs}

        # each worker processes its own slice sequentially (one copy per worker)
        slices = [[(k, s) for k, s in enumerate(sample) if k % workers == w] for w in range(workers)]

        def worker(sl):
            out = []
            for job in sl:
                r = work(job)
                out.append(r)
                run.log(f"{r['module']}:{r['line']} {r['mutation']}: {r['status']}"
                        + (f" ({r['first_failure']})" if r["first_failure"] else ""))
            return out

        with ThreadPoolExecutor(workers) as ex:
            for out in ex.map(worker, slices):
                results += out
        killed = sum(r["status"] in ("killed", "timeout") for r in results)
        run.record("mutants", results)
        run.record("summary", {"n": len(results), "killed": killed,
                               "timeouts": sum(r["status"] == "timeout" for r in results),
                               "survived": sum(r["status"] == "survived" for r in results),
                               "score": killed / max(1, len(results))})
        run.log(f"mutation score {killed}/{len(results)}")
        for d in copies:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
