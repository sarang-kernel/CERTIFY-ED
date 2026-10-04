"""Reproducibility harness shared by all experiments.

Each run writes experiments/results/<name>_<timestamp>/ with
  data.json         all numbers reported in the paper (machine-readable)
  environment.json  interpreter, library versions, BLAS, CPU, seed, git commit
  run.log           timestamped log
  figures/          figures generated from data.json
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _jsonable(x):
    from certify_ed.certificate_v2 import _jsonable as j
    return j(x)


class Run:
    def __init__(self, name: str, seed: int = 12345, outdir: str = None):
        self.name = name
        self.seed = seed
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = outdir or os.path.join(HERE, "results")
        self.dir = os.path.join(base, f"{name}_{ts}")
        os.makedirs(os.path.join(self.dir, "figures"), exist_ok=True)
        self.data = {}
        self._log = open(os.path.join(self.dir, "run.log"), "w")
        self.t0 = time.time()

    def __enter__(self):
        np.random.seed(self.seed)
        env = {"experiment": self.name, "seed": self.seed, "python": sys.version,
               "platform": platform.platform(), "processor": platform.processor(),
               "cpu_count": os.cpu_count(), "numpy": np.__version__, "git_commit": _git_commit(),
               "started": _dt.datetime.now().isoformat(timespec="seconds")}
        import scipy
        env["scipy"] = scipy.__version__
        for mod in ("quspin", "mpmath", "flint", "matplotlib"):
            try:
                env[mod] = __import__(mod).__version__
            except Exception:
                env[mod] = "not installed"
        try:
            from threadpoolctl import threadpool_info
            env["blas"] = [{k: d.get(k) for k in ("internal_api", "version", "num_threads")}
                           for d in threadpool_info()]
        except Exception:
            pass
        try:
            with open("/proc/cpuinfo") as f:
                env["cpu_model"] = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), None)
            with open("/proc/meminfo") as f:
                kb = next(int(l.split()[1]) for l in f if l.startswith("MemTotal"))
            env["memory_GiB"] = round(kb / 2 ** 20, 1)
        except Exception:
            pass
        env["threads_env"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                            "MKL_NUM_THREADS")}
        with open(os.path.join(self.dir, "environment.json"), "w") as f:
            json.dump(env, f, indent=2)
        self.log(f"=== {self.name} started; output {self.dir}")
        return self

    def log(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def record(self, key, value):
        self.data[key] = _jsonable(value)
        self.save()

    def save(self):
        with open(os.path.join(self.dir, "data.json"), "w") as f:
            json.dump(self.data, f, indent=1)

    def fig_path(self, name: str) -> str:
        return os.path.join(self.dir, "figures", name)

    def __exit__(self, *exc):
        self.data["_elapsed_seconds"] = time.time() - self.t0
        self.save()
        self.log(f"=== finished in {time.time() - self.t0:.1f}s")
        self._log.close()
        return False
