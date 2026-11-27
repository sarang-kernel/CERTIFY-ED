"""E6: mutation analysis of CERTIFY-ED itself.

Measures how strong the test suite is: each mutant changes one operator or constant in the
verification core (arithmetic, comparison, boolean and sign mutations), the v2 test suite runs
against it, and the mutant is 'killed' if any test fails or errors. Survivors are either test gaps or
equivalent mutants (changes with no observable effect); they are listed for inspection.

usage: python3 e6_mutation.py [n_mutants] [workers]
"""
from pathlib import Path
import ast
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from harness import Run, ROOT

TARGETS = ["construct.py", "enclosure.py", "inertia.py", "eigenspace.py", "sectors.py", "compose.py",
           "fp.py", "localops.py", "spec.py", "basis.py", "su2.py", "checker.py", "certify.py",
           "exact_sectors.py", "spectrum.py"]

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


def _tracked_files():
    """Files git tracks in ROOT (relative paths), or [] when git is unavailable."""
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode:
        return []
    return [p for p in out.stdout.decode("utf-8").split("\0") if p and os.path.isfile(os.path.join(ROOT, p))]


_REMOVALS = (os.unlink, os.remove, os.rmdir)


def _force_writable(func, path, _exc):
    """Error handler of shutil.rmtree: git marks its object files read-only, and Windows refuses to delete a read-only
    file, so after a failed removal (os.unlink, os.remove or os.rmdir) the owner's permission bits are set (read and
    write for a file, read, write and search for a directory) and the removal is tried once more. Other failed calls
    (os.scandir, os.lstat, os.open) are not retried, and nothing the retry raises propagates: whatever is left is
    reported by _remove_tree. A symbolic link is not passed to chmod, which would change its target. For a directory
    the bits matter on Windows only: on POSIX the removal of a directory depends on the mode of its parent, which is
    not changed."""
    if func not in _REMOVALS:
        return
    try:
        if not os.path.islink(path):
            os.chmod(path, stat.S_IRWXU if os.path.isdir(path) else stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except Exception:
        pass


def _remove_tree(path):
    """Delete one repository copy, including the read-only files of a git repository a test created inside it.

    Returns True when nothing is left at path."""
    if not os.path.exists(path):
        return True
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force_writable)
    else:
        shutil.rmtree(path, onerror=_force_writable)
    return not os.path.exists(path)


def _remove_copies(copies):
    """Delete every repository copy; returns the paths that could not be removed."""
    return [d for d in copies if not _remove_tree(d)]


def run_tests(workdir, timeout):
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                            "tests"], cwd=workdir, capture_output=True, text=True, timeout=timeout,
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
            src = Path(pkg, f).read_text()
            n = len(mutation_sites(ast.parse(src)))
            counts[f] = n
            pool += [(f, i) for i in range(n)]
        sample = rng.sample(pool, min(n_mut, len(pool)))
        run.log(f"{len(pool)} mutation sites in {len(TARGETS)} modules; sampling {len(sample)}")
        run.record("sites_per_module", counts)
        # one private copy of the repository per worker: every tracked file (the suite reads pyproject.toml,
        # experiments/ and docs/ as well as certify_ed/, tests/ and examples/; a copy without them fails or
        # skips tests and measures a weaker suite than the one shipped)
        tracked = _tracked_files()
        run.record("copy", {"files": len(tracked), "source": "git ls-files" if tracked else "fallback directories"})
        copies = []
        try:
            _mutate_all(run, pkg, sample, tracked, copies, workers)
        finally:
            # the copies are removed also when the baseline fails (SystemExit) or the run stops early; up to 3.2.3 they
            # were removed only after a complete run and with ignore_errors=True, which on Windows silently kept the
            # read-only files of the git repository that a test creates inside each copy
            left = _remove_copies(copies)
            run.record("cleanup", {"copies": len(copies), "left": left})
            if left:
                run.log(f"could not remove {len(left)} repository copies: {left}")


def _mutate_all(run, pkg, sample, tracked, copies, workers):
    """Make one repository copy per worker (each appended to copies as soon as it exists, so that the caller can
    remove it whatever happens), run the baseline in the first copy and then every sampled mutant."""
    for w in range(workers):
        # 3.2.3: the copies live next to the repository (or in CERTIFY_ED_E6_COPY_DIR), not in the system temp
        # directory: in the sandbox of the 3.2.2 run git could not create a repository under the temp directory,
        # so test_imp05 of tests/test_review_fixes.py failed in the copy and the baseline was refused
        d = tempfile.mkdtemp(prefix=f"mut{w}_", dir=os.environ.get("CERTIFY_ED_E6_COPY_DIR") or os.path.dirname(ROOT))
        copies.append(d)  # registered before it is filled, so that a failed copy is removed too
        if tracked:
            for rel in tracked:
                os.makedirs(os.path.dirname(os.path.join(d, rel)) or d, exist_ok=True)
                shutil.copy2(os.path.join(ROOT, rel), os.path.join(d, rel))
        else:
            for sub in ("certify_ed", "tests", "examples", "experiments", "docs"):
                shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                                ignore=shutil.ignore_patterns("__pycache__", "results"))
            shutil.copy2(os.path.join(ROOT, "pyproject.toml"), d)
    # baseline: unmutated (unparsed) sources must pass
    base_status, base_first, base_t = run_tests(copies[0], 3600)   # the 3.2 suite takes ~25 min on 8 cores
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
        orig = Path(pkg, f).read_text()
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


if __name__ == "__main__":
    main()
