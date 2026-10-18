"""Re-run surviving mutants of an E6 run against the current test suite.

usage: python3 e6_recheck.py RUN_DIR [workers] [module:site ...]
Writes RUN_DIR/recheck.json. The mutation sites must refer to the same source as the original run
(checked by hashing the target modules)."""
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_mutation import apply_mutation, run_tests, ROOT, mutation_sites
import ast


def main():
    run = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    only = set(sys.argv[3:])
    d = json.loads(Path(run, "data.json").read_text())
    surv = [m for m in d["mutants"] if m["status"] == "survived"]
    if only:
        surv = [m for m in surv if f"{m['module']}:{m['site']}" in only]
    pkg = os.path.join(ROOT, "certify_ed")
    for mod, n in d["sites_per_module"].items():
        src = Path(pkg, mod).read_text()
        assert len(mutation_sites(ast.parse(src))) == n, f"{mod} changed since the run"
    def work(m):
        tmp = tempfile.mkdtemp(prefix="e6r_")
        try:
            shutil.copytree(os.path.join(ROOT, "certify_ed"), os.path.join(tmp, "certify_ed"))
            shutil.copytree(os.path.join(ROOT, "tests"), os.path.join(tmp, "tests"))
            for extra in ("pyproject.toml", "setup.cfg", "pytest.ini", "conftest.py"):
                if os.path.exists(os.path.join(ROOT, extra)):
                    shutil.copy(os.path.join(ROOT, extra), tmp)
            path = os.path.join(tmp, "certify_ed", m["module"])
            new, desc, line = apply_mutation(open(path).read(), m["site"])
            assert line == m["line"], (m, line)
            Path(path).write_text(new)
            status, first, secs = run_tests(tmp, 900)
            r = dict(m, recheck=status, recheck_first_failure=first, recheck_seconds=secs)
            print(f"{m['module']}:{m['line']} {m['mutation']}: {status} {first or ''}", flush=True)
            return r
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(work, surv))
    out = os.path.join(run, "recheck.json")
    old = json.loads(Path(out).read_text()) if os.path.exists(out) else {}
    for r in res:
        old[f"{r['module']}:{r['site']}"] = r
    json.dump(old, open(out, "w"), indent=1)
    k = sum(r["recheck"] != "survived" for r in res)
    print(f"{k}/{len(res)} previously surviving mutants now killed")


if __name__ == "__main__":
    main()
