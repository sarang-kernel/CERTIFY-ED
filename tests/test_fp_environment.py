"""The run-time probe of the floating-point model, including a real negative control: a shared
library compiled with -ffast-math that sets flush-to-zero/denormals-are-zero when loaded."""
import os, shutil, subprocess, sys, textwrap
import pytest

from certify_ed.fp import probe_float_environment


def test_probe_passes_in_clean_process():
    r = probe_float_environment()
    assert r["standard_model_hypotheses_hold"], r


@pytest.mark.skipif(shutil.which("gcc") is None or os.uname().machine not in ("x86_64", "AMD64"),
                    reason="needs gcc on x86-64")
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
