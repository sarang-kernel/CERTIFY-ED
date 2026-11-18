"""The audit of complex moduli (18 Nov 2026) stays true: every modulus call in certify_ed (abs(), np.abs and the other
spellings of _is_abs) is listed in tests/abs_audit_allowlist.txt, where each was classified as real data, as not part
of a bound, or as one of the three complex uses that need no upper modulus (Arb balls; one open question). The moduli of complex data in rigorous bounds
go through fp.modulus_up or fp.abs_up (docs/proofs/fp_modulus-up.md), so reverting one of them to abs() fails here."""
import ast
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# every spelling of a modulus that the audit covers: abs(), np.abs / np.absolute (also through xp, cp, cupy, numpy),
# hypot and fabs of numpy and math, cmath.polar, operator.abs, x.__abs__(), and the 1- and infinity-norms of
# np.linalg.norm / scipy.linalg.norm / scipy.sparse.linalg.norm (ord 1, inf, -inf), which sum or compare moduli
_MODULES = {"np", "numpy", "xp", "cp", "cupy", "math", "cmath", "operator"}
_NAMES = {"abs", "absolute", "hypot", "fabs", "polar"}
_ORDS = {"1", "-1", "np.inf", "-np.inf", "numpy.inf", "-numpy.inf", "math.inf", "-math.inf", "inf", "-inf",
         "float('inf')", "-float('inf')", "'inf'"}


def _norm_ord(call):
    for kw in call.keywords:
        if kw.arg == "ord":
            return ast.unparse(kw.value).replace(" ", "")
    return ast.unparse(call.args[1]).replace(" ", "") if len(call.args) >= 2 else None


def _is_abs(call):
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id == "abs" or (fn.id == "norm" and _norm_ord(call) in _ORDS)
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr == "__abs__" or (fn.attr == "norm" and _norm_ord(call) in _ORDS):
        return True
    root = fn.value
    while isinstance(root, ast.Attribute):
        root = root.value
    return fn.attr in _NAMES and isinstance(root, ast.Name) and root.id in _MODULES


def _imported_modulus_names():
    """from numpy/math/cmath/operator/... import hypot, fabs, ... (a bare name the pattern above would miss); abs and
    norm imported under their own names are caught by the pattern itself."""
    out = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "certify_ed")):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in (
                            "numpy", "math", "cmath", "operator", "scipy", "cupy"):
                        out += [(os.path.relpath(path, ROOT), a.name) for a in node.names
                                if a.name in _NAMES | {"norm"} and (a.asname or a.name) not in ("abs", "norm")]
    return out


def _visit(node, scope, rel, out):
    for ch in ast.iter_child_nodes(node):
        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _visit(ch, scope + [ch.name], rel, out)
            continue
        if isinstance(ch, ast.Call) and _is_abs(ch):
            out[(rel, ".".join(scope) or "<module>", ast.unparse(ch))] += 1
        _visit(ch, scope, rel, out)


def _abs_calls():
    out = Counter()
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "certify_ed")):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                _visit(ast.parse(open(path, encoding="utf-8").read()), [], rel, out)
    return out


def _allowed():
    out = Counter()
    with open(os.path.join(ROOT, "tests", "abs_audit_allowlist.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f, scope, expr, _cls = line.rstrip("\n").split("\t")
            out[(f, scope, expr)] += 1
    return out


def test_every_abs_call_is_classified():
    calls, allowed = _abs_calls(), _allowed()
    extra = {k: n - allowed[k] for k, n in calls.items() if n > allowed[k]}
    assert not extra, f"unclassified abs() calls (route complex moduli in bounds through fp.modulus_up): {extra}"


def test_no_modulus_function_is_imported_under_another_name():
    """A modulus function imported under an alias would escape the pattern of _is_abs."""
    assert not _imported_modulus_names()


def test_the_pattern_sees_every_spelling():
    """Negative control of the scanner: each spelling below is reported."""
    src = ("abs(z); np.abs(z); np.absolute(z); xp.abs(z); cp.abs(z); np.hypot(x, y); math.hypot(x, y); math.fabs(x); "
           "cmath.polar(z); operator.abs(z); z.__abs__(); np.linalg.norm(z, 1); np.linalg.norm(z, ord=np.inf); "
           "scipy.sparse.linalg.norm(A, -np.inf); norm(z, 1); np.linalg.norm(z); np.linalg.norm(z, 2); np.sqrt(z)")
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]
    flagged = [ast.unparse(c) for c in calls if _is_abs(c)]
    assert len(flagged) == 15, flagged
    assert not any(s in flagged for s in ("np.linalg.norm(z)", "np.linalg.norm(z, 2)", "np.sqrt(z)"))


def test_gpu_path_takes_no_numpy_modulus():
    """gpu.py forms the moduli with fp.modulus_up / fp.abs_up only, so that the device path stays bit for bit equal to
    the CPU path (the kernels cannot run without CuPy; this pins the host side)."""
    assert not [k for k in _abs_calls() if k[0] == "certify_ed/gpu.py"]
