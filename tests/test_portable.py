"""experiments/portable.py: the platform layer of the experiments (memory cap, peak memory, CPU, RAM)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, EXP)
import portable  # noqa: E402


def test_probes_return_plausible_values():
    cpu, mem, rss = portable.cpu_model(), portable.memory_gib(), portable.peak_rss_mb()
    assert cpu is None or (isinstance(cpu, str) and cpu)
    assert mem is None or 0.1 < mem < 1e5
    assert rss > 1.0                                   # this process holds more than 1 MiB


def test_posix_preexec_only_on_posix():
    assert (portable.posix_preexec(2 ** 30) is None) == (sys.platform == "win32")


def test_memory_cap_makes_large_allocations_fail():
    # child caps itself at 1 GiB, a 50 MiB allocation succeeds and a 2 GiB one raises MemoryError
    code = (f"import sys; sys.path.insert(0, {EXP!r})\n"
            "import numpy as np, portable\n"
            "mech = portable.cap_memory(2 ** 30)\n"
            "a = np.ones(50 * 2 ** 20 // 8)\n"
            "try:\n    b = np.ones(2 * 2 ** 30 // 8); print('RESULT', mech, 'allocated')\n"
            "except MemoryError:\n    print('RESULT', mech, 'MemoryError')\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("RESULT"))
    mech = line.split()[1:-1]
    assert " ".join(mech) in ("RLIMIT_AS", "job object"), line
    assert line.endswith("MemoryError"), line
