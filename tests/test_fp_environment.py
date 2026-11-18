"""The run-time probe of the floating-point model, including a real negative control: a shared
library compiled with -ffast-math that sets flush-to-zero/denormals-are-zero when loaded."""
import os, platform, shutil, subprocess, sys, textwrap
import pytest

from certify_ed.fp import probe_float_environment


def test_probe_passes_in_clean_process():
    r = probe_float_environment()
    assert r["standard_model_hypotheses_hold"], r


@pytest.mark.skipif(shutil.which("gcc") is None or platform.machine() not in ("x86_64", "AMD64")
                    or sys.platform == "win32", reason="needs gcc on x86-64 Linux/macOS (.so build)")
def test_probe_detects_fast_math_library(tmp_path):
    src = tmp_path / "fm.c"
    # what crtfastmath.o's constructor does in libraries built with -ffast-math by GCC < 13:
    # set FTZ (bit 15) and DAZ (bit 6) in MXCSR for the whole process
    src.write_text("#include <xmmintrin.h>\nvoid set_ftz_daz(void){ _mm_setcsr(_mm_getcsr() | 0x8040); }\n")
    lib = tmp_path / "libfm.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(lib), str(src)], check=True)
    code = textwrap.dedent(f"""
        import ctypes, json, sys
        sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))!r})
        from certify_ed.fp import probe_float_environment
        before = probe_float_environment()["standard_model_hypotheses_hold"]
        ctypes.CDLL({str(lib)!r}).set_ftz_daz()
        after = probe_float_environment()
        print(json.dumps([before, after["standard_model_hypotheses_hold"], after["failed"]]))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    import json
    before, after, failed = json.loads(out.strip().splitlines()[-1])
    assert before is True
    assert after is False and "gradual_underflow" in failed and "subnormal_operands" in failed


def test_certificate_downgrades_under_ftz(monkeypatch):
    import scipy.linalg as la
    import certify_ed as ce
    cmod = sys.modules["certify_ed.certify"]
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    bad = {"standard_model_hypotheses_hold": False, "checks": {"gradual_underflow": False},
           "failed": ["gradual_underflow"]}
    monkeypatch.setattr(cmod, "probe_float_environment", lambda: bad)
    c = ce.certify(ce.Claim(w[:2], V[:, :2], operator=H, kind="lowest"), spec)
    assert all(s["status"] != "CERTIFIED" for s in c.statements)
    assert any("floating-point environment" in x for x in c.body["not_verified"])


# ---------------------------------------------------------------- directed rounding modes (IMP-24)
_ROUNDING_SNIPPET = {
    "linux": ("libm = ctypes.CDLL('libm.so.6')\n"
              "MODES = {'up': 0x800, 'down': 0x400, 'zero': 0xC00, 'nearest': 0}\n"
              "setround = libm.fesetround\n"),
    "win32": ("libm = ctypes.CDLL('ucrtbase.dll')\n"
              "MODES = {'up': 0x200, 'down': 0x100, 'zero': 0x300, 'nearest': 0}\n"
              "setround = libm.fesetround\n"),
}


def _probe_under_rounding(mode):
    """Run fp.probe_float_environment in a subprocess with the rounding mode set through the C runtime
    (glibc libm on Linux, ucrtbase on Windows); returns (holds, failed_checks)."""
    code = "\n".join([
        "import ctypes, json, sys",
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})",
        "from certify_ed.fp import probe_float_environment",
        _ROUNDING_SNIPPET[sys.platform],
        f"assert setround(MODES[{mode!r}]) == 0",
        "r = probe_float_environment()",
        "setround(MODES['nearest'])",
        'print(json.dumps([r["standard_model_hypotheses_hold"], r["failed"]]))'])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    import json
    return json.loads(out.strip().splitlines()[-1])


needs_fesetround = pytest.mark.skipif(sys.platform not in _ROUNDING_SNIPPET,
                                      reason="fesetround through ctypes: glibc or ucrtbase only")


@needs_fesetround
def test_probe_detects_round_toward_zero():
    """Round-toward-zero passed every probe before 3.1.0 (IMP-24): the 1 + 3*2^-54 probe and its mirror catch it."""
    holds, failed = _probe_under_rounding("zero")
    assert holds is False
    assert "round_to_nearest_not_toward_zero_pos" in failed and "round_to_nearest_not_toward_zero_neg" in failed


@needs_fesetround
@pytest.mark.parametrize("mode,expect", [("up", "round_to_nearest_up"), ("down", "round_to_nearest_down")])
def test_probe_detects_directed_rounding(mode, expect):
    holds, failed = _probe_under_rounding(mode)
    assert holds is False and expect in failed


@needs_fesetround
def test_probe_passes_after_rounding_mode_restored():
    holds, failed = _probe_under_rounding("nearest")
    assert holds is True and failed == []
