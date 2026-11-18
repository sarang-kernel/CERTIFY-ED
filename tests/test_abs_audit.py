"""The audit of complex moduli (18 Nov 2026) stays true: every modulus in certify_ed (abs(), np.abs and the other
spellings of _is_abs, called or referenced) is listed in tests/abs_audit_allowlist.txt, where each was classified as real
data, as not part of a bound, or as one of the complex uses that need no upper modulus (Arb balls, squares of the
components; one open question), and the list holds nothing else. The moduli of complex data in rigorous bounds go
through fp.modulus_up or fp.abs_up (docs/proofs/fp_modulus-up.md), so reverting one of them to abs() fails here; the
dynamic pin of tests/test_modulus_up.py (numpy's modulus made to raise on complex input) covers the spellings that no
source pattern sees."""
import ast
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# every spelling of a modulus that the audit covers, whatever the object it is taken from (np, numpy, xp, cp,
# self.cp, math, cmath, operator, builtins, ...): abs, absolute, hypot, fabs, polar and __abs__, called or referenced
# as a value (map(abs, ...), key=abs, f = np.abs, np.abs.reduce, np.vectorize(abs)), getattr(m, "abs"); the norms
# norm, vector_norm, matrix_norm and cond with an order other than None, 2 or 'fro' (1, inf, -inf, ... sum or compare
# moduli); onenormest; isclose and allclose (they compare moduli of differences); and a square root of an expression
# that reads both .real and .imag, or calls .conj() (a hand-written modulus)
_NAMES = {"abs", "absolute", "hypot", "fabs", "polar", "__abs__"}
_BARE_NAMES = {"abs", "hypot", "fabs", "polar"}        # 'absolute' is also a parameter name in certify_ed
_NORMS = {"norm": ("ord", 1), "vector_norm": ("ord", None), "matrix_norm": ("ord", None), "cond": ("p", 1)}
_SAFE_ORDS = {"None", "2", "2.0", "'fro'"}
_OTHER = {"onenormest", "isclose", "allclose"}


def _name(fn):
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _norm_ord(call, kw_name, pos):
    for kw in call.keywords:
        if kw.arg == kw_name:
            return ast.unparse(kw.value).replace(" ", "").replace('"', "'")
    if pos is not None and len(call.args) > pos:
        return ast.unparse(call.args[pos]).replace(" ", "").replace('"', "'")
    return None


def _hand_written_modulus(call):
    if _name(call.func) != "sqrt":
        return False
    attrs = {n.attr for a in call.args for n in ast.walk(a) if isinstance(n, ast.Attribute)}
    return ("real" in attrs and "imag" in attrs) or "conj" in attrs or "conjugate" in attrs


def _is_abs(call):
    name = _name(call.func)
    if name in _NAMES or name in _OTHER:
        return True
    if name in _NORMS:
        kw_name, pos = _NORMS[name]
        o = _norm_ord(call, kw_name, pos)
        return o is not None and o not in _SAFE_ORDS
    if name == "getattr" and len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        return call.args[1].value in _NAMES
    return _hand_written_modulus(call)


def _is_reference(node, imported_absolute):
    if isinstance(node, ast.Attribute):
        return node.attr in _NAMES
    if isinstance(node, ast.Name):
        return node.id in _BARE_NAMES or (node.id == "absolute" and imported_absolute)
    return False


def _imports_absolute(tree):
    return any(isinstance(n, ast.ImportFrom) and any(a.name == "absolute" for a in n.names) for n in ast.walk(tree))


def _imported_modulus_names():
    """from numpy/math/cmath/operator/... import hypot, fabs, ... under another name (a bare name the patterns above
    would miss); names imported under their own names are caught by the patterns themselves."""
    out = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "certify_ed")):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in (
                            "numpy", "math", "cmath", "operator", "scipy", "cupy", "builtins"):
                        out += [(os.path.relpath(path, ROOT), a.name) for a in node.names
                                if a.name in _NAMES | set(_NORMS) | _OTHER and (a.asname or a.name) != a.name]
    return out


def _visit(node, scope, rel, out, imported_absolute, func_position=False):
    for field, value in ast.iter_fields(node):
        children = value if isinstance(value, list) else [value]
        for ch in children:
            if not isinstance(ch, ast.AST):
                continue
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _visit(ch, scope + [ch.name], rel, out, imported_absolute)
                continue
            in_func = isinstance(node, ast.Call) and field == "func"
            if isinstance(ch, ast.Call) and _is_abs(ch):
                out[(rel, ".".join(scope) or "<module>", ast.unparse(ch))] += 1
            elif not in_func and _is_reference(ch, imported_absolute):
                out[(rel, ".".join(scope) or "<module>", "<reference> " + ast.unparse(ch))] += 1
            _visit(ch, scope, rel, out, imported_absolute)


def _scan(src, rel="<source>"):
    out = Counter()
    tree = ast.parse(src)
    _visit(tree, [], rel, out, _imports_absolute(tree))
    return out


def _abs_calls():
    out = Counter()
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "certify_ed")):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                out += _scan(open(path, encoding="utf-8").read(), rel)
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
    """The moduli found and the allowlist are equal as multisets: a new modulus must be classified, and a removed or
    changed one must leave the list (a stale line would let a later call with the same key pass)."""
    calls, allowed = _abs_calls(), _allowed()
    extra = {k: n - allowed[k] for k, n in calls.items() if n > allowed[k]}
    stale = {k: n - calls[k] for k, n in allowed.items() if n > calls[k]}
    assert not extra, f"unclassified moduli (route complex moduli in bounds through fp.modulus_up): {extra}"
    assert not stale, f"allowlist lines without a modulus in certify_ed (remove them): {stale}"


def test_no_modulus_function_is_imported_under_another_name():
    """A modulus function imported under an alias would escape the patterns of _is_abs and _is_reference."""
    assert not _imported_modulus_names()


def test_the_pattern_sees_every_spelling():
    """Negative control of the scanner: each spelling below is reported, and the ones at the end are not."""
    seen = ["abs(z)", "np.abs(z)", "np.absolute(z)", "xp.abs(z)", "cp.abs(z)", "self.xp.abs(z)", "self._cp.absolute(z)",
            "self.cp.abs(z)", "builtins.abs(z)", "np.hypot(x, y)", "math.hypot(x, y)", "math.fabs(x)", "cmath.polar(z)",
            "operator.abs(z)", "z.__abs__()", "np.linalg.norm(z, 1)", "np.linalg.norm(z, 1.0)",
            "np.linalg.norm(z, ord=np.inf)", "np.linalg.norm(z, np.Inf)", "np.linalg.norm(z, ord=3)",
            "scipy.sparse.linalg.norm(A, -np.inf)", "la.norm(z, 1)", "norm(z, 1)", "np.linalg.vector_norm(z, ord=1)",
            "np.linalg.matrix_norm(A, ord=1)", "np.linalg.cond(A, 1)", "np.linalg.cond(A, p=np.inf)",
            "onenormest(A)", "np.isclose(a, b)", "np.allclose(a, b)", "getattr(np, 'abs')(z)",
            "np.sqrt(z.real ** 2 + z.imag ** 2)", "np.sqrt((z * z.conj()).real)", "math.sqrt(x.real * x.real + x.imag * x.imag)"]
    refs = ["map(abs, zs)", "sorted(zs, key=abs)", "f = np.abs", "np.abs.reduce(z)", "np.vectorize(abs)(z)",
            "a = abs\na(z)", "self.cp.absolute.reduce(z)", "fns = [math.fabs, np.hypot]"]
    for s in seen:
        assert sum(_scan(s).values()) >= 1, s
    for s in refs:
        assert any(k[2].startswith("<reference> ") for k in _scan(s)), s
    assert sum(_scan("from numpy import absolute\ng = absolute").values()) == 1
    for s in ("np.linalg.norm(z)", "np.linalg.norm(z, 2)", "np.linalg.norm(A, 'fro')", "np.linalg.norm(z, ord=None)",
              "np.linalg.vector_norm(z)", "np.linalg.matrix_norm(A)", "np.linalg.cond(A)", "np.sqrt(z)",
              "np.sqrt(x * x + y * y)", "op.matmul(x, absolute=True)", "def f(absolute):\n    return absolute",
              "abs_up(A)", "modulus_up(z)"):
        assert not _scan(s), s


def test_gpu_path_takes_no_numpy_modulus():
    """gpu.py forms the moduli with fp.modulus_up / fp.abs_up only, so that the device path stays bit for bit equal to
    the CPU path (the kernels cannot run without CuPy; this pins the host side)."""
    assert not [k for k in _abs_calls() if k[0] == "certify_ed/gpu.py"]
