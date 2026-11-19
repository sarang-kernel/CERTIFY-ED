"""Check that pinning tests kill their mutants (3.2.2, roadmap N-02).

For each (module, site index) the mutant of e6_mutation.apply_mutation is written into a scratch copy of certify_ed,
and the given test files run against that copy. The expected mutation and line are compared with the recorded
ones first, so a site index that no longer points at the recorded node is reported as 'site moved', not run.
The defaults are the eight survivors of the 3.2 sample named in E6_inspection_worksheet_3.2.0.md, with the site
indices of results/run_full_20261104/e6_mutation_20261104_163828 (the same in the 3.2.1 run), and the test file
tests/test_e6_pins.py.

usage: python experiments/e6_pin_check.py [--tests tests/test_e6_pins.py] [--out e6_pin_check.json]
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from e6_mutation import apply_mutation, describe, mutation_sites  # noqa: E402

# (module, site index, mutation, line on dev/3.2 = v3.2.0 for these modules)
SURVIVORS_3_2 = [
    ("checker.py", 494, "Add->Sub", 535),
    ("sectors.py", 1, "Or->And", 57),
    ("eigenspace.py", 11, "Mult->Div", 151),
    ("eigenspace.py", 55, "Div->Mult", 147),
    ("eigenspace.py", 73, "Mult->Div", 129),
    ("exact_sectors.py", 26, "FloorDiv->Mult", 202),
    ("localops.py", 232, "constant 0.0->1.0", 189),
    ("spectrum.py", 38, "And->Or", 103),
]


def check(tests, mutants=SURVIVORS_3_2, timeout=3600):
    out = []
    for mod, idx, want_desc, want_line in mutants:
        src = open(os.path.join(ROOT, "certify_ed", mod), encoding="utf-8").read()
        kind, node = mutation_sites(ast.parse(src))[idx]
        desc, line = describe(kind, node), getattr(node, "lineno", None)
        rec = {"module": mod, "site": idx, "mutation": desc, "line": line}
        if (desc, line) != (want_desc, want_line):
            rec.update(outcome="site moved", expected={"mutation": want_desc, "line": want_line})
            out.append(rec)
            continue
        with tempfile.TemporaryDirectory(prefix="e6pin_") as scratch:
            shutil.copytree(os.path.join(ROOT, "certify_ed"), os.path.join(scratch, "certify_ed"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            os.makedirs(os.path.join(scratch, "tests"))
            for f in ["tests/conftest.py"] + list(tests):
                shutil.copy(os.path.join(ROOT, f), os.path.join(scratch, "tests", os.path.basename(f)))
            mutated, _, _ = apply_mutation(src, idx)
            with open(os.path.join(scratch, "certify_ed", mod), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(mutated)
            env = dict(os.environ, PYTHONPATH=scratch, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
            env.pop("PYTHONSAFEPATH", None)
            t0 = time.time()
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                                os.path.join(scratch, "bt")] + [os.path.join("tests", os.path.basename(f)) for f in tests],
                               cwd=scratch, capture_output=True, text=True, env=env, timeout=timeout)
            rec.update(outcome="killed" if p.returncode != 0 else "survived", returncode=p.returncode,
                       failed_tests=[l.split()[1] for l in p.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))],
                       summary=(p.stdout.strip().splitlines() or [""])[-1], seconds=round(time.time() - t0, 1))
        out.append(rec)
        print(f"{mod}:{line} {desc}: {rec['outcome']}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", nargs="+", default=["tests/test_e6_pins.py"])
    ap.add_argument("--out", default="e6_pin_check.json")
    a = ap.parse_args(argv)
    res = check(a.tests)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tests": a.tests, "mutants": res}, fh, indent=1)
        fh.write("\n")
    return 0 if all(r["outcome"] == "killed" for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
