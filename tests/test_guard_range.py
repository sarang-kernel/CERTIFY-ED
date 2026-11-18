"""3.2.2 (roadmap N-01, finding L-3 of the Lean formalization): the floating-point guards refuse to work outside the
range in which they are established.

The guards of Theorem 1 (docs/proofs/thm1_fp-accounting.md: Lambda4 for the norms, Lambda5 for |A||x|, Lambda6 for
E) and the sparse residual of D-01 (hypothesis H3) hold for vectors of n <= 2^26 entries and row widths w <= 2^26.
Lambda4 with N = 2n fails from n = 84886744 on. The tests check the guard functions at 2^26 (allowed) and
2^26 + 1 (refused) with inputs that allocate nothing large (a zero-stride view, a fake adapter), and the statements of
certify() with the limit lowered for a small operator and with a spec whose row width bound is above 2^26.
"""
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed.fp as fp
from certify_ed import Claim, Options, certify, heisenberg_chain, build_bitwise
from certify_ed.construct import SpecOperator, StateSet
from certify_ed.enclosure import OperatorAdapter, _rounding_bound, rayleigh_interval
from certify_ed.inertia import inertia
from certify_ed.ldl_residual import sparse_ldl_residual_bound
from certify_ed.localops import LocalSpace
from certify_ed.scope import REASON_CODES
from certify_ed.spec import ModelSpec

LIMIT = 2 ** 26


def _status(cert):
    return {s["id"]: (s["status"], s["evidence"].get("reason_code")) for s in cert.statements}


# ---------------------------------------------------------------- the range itself

def test_range_constant_and_reason_code():
    assert fp.GUARD_RANGE_MAX == LIMIT == 67108864
    assert "FP_GUARD_RANGE" in REASON_CODES


@pytest.mark.parametrize("q", ["n", "w"])
def test_check_at_and_above_the_limit(q):
    fp.check_guard_range(**{q: LIMIT})                     # allowed: the guards are stated for <= 2^26
    assert fp.guard_range_findings(**{q: LIMIT}) == []
    with pytest.raises(fp.GuardRangeError) as e:
        fp.check_guard_range(**{q: LIMIT + 1}, where="test")
    assert isinstance(e.value, ValueError)
    assert e.value.to_dict() == {"quantity": q, "value": LIMIT + 1, "limit": LIMIT, "where": "test"}
    assert fp.guard_range_findings(**{q: LIMIT + 1})[0]["value"] == LIMIT + 1
    fp.check_guard_range()                                 # nothing given: nothing to check


def test_norm_guards_refuse_more_than_2_26_entries_before_computing():
    v = np.broadcast_to(np.float64(1.0), (LIMIT + 1,))      # zero-stride view: no memory is allocated
    assert v.size == LIMIT + 1 and v.strides == (0,)
    for f in (fp.norm2_upper, fp.norm2_lower, fp.fro_upper):
        with pytest.raises(fp.GuardRangeError):
            f(v)


def test_norm_guards_unchanged_inside_the_range(monkeypatch):
    """At and below the (lowered) limit the norm guards return the same numbers as with the 2^26 limit."""
    rng = np.random.default_rng(5)
    v = rng.standard_normal(8)
    ref = (fp.norm2_upper(v), fp.norm2_lower(v), fp.fro_upper(v.reshape(2, 4)))
    monkeypatch.setattr(fp, "GUARD_RANGE_MAX", 8)
    assert (fp.norm2_upper(v), fp.norm2_lower(v), fp.fro_upper(v.reshape(2, 4))) == ref
    with pytest.raises(fp.GuardRangeError):
        fp.norm2_upper(np.ones(9))
    with pytest.raises(fp.GuardRangeError):
        fp.fro_upper(sp.csr_matrix(np.ones((3, 3))))        # the sparse data array has 9 entries


class _FakeAdapter:
    """What _rounding_bound and rayleigh_interval read from an adapter, with any row width (no matrix is built)."""
    kind = "matrix"
    complex = False

    def __init__(self, width):
        self.width = width

    def abs_matvec(self, X):
        return 2.0 * np.abs(X)

    def matvec(self, X):
        return 2.0 * X


def test_rounding_bound_at_and_above_the_width_limit():
    X = np.array([[1.0], [0.5]])
    mu = np.array([2.0])
    E = _rounding_bound(_FakeAdapter(LIMIT), mu, X.copy())
    g = fp.gamma_dot(LIMIT + 2, False)
    gw = fp.gamma(LIMIT + 1)
    expected = ((2.0 * np.abs(X) / (1.0 - gw)) * (1 + 4 * fp.U) + np.abs(X) * 2.0) * g * (1 + 4 * fp.U)
    assert np.array_equal(E, expected)
    with pytest.raises(fp.GuardRangeError) as e:
        _rounding_bound(_FakeAdapter(LIMIT + 1), mu, X.copy())
    assert e.value.quantity == "w" and "_rounding_bound" in e.value.where


def test_rayleigh_interval_refuses_width_above_the_limit():
    with pytest.raises(fp.GuardRangeError):
        rayleigh_interval(_FakeAdapter(LIMIT + 1), np.array([1.0, 0.5]))


def test_sparse_residual_refuses_dimension_above_the_limit(monkeypatch):
    A = sp.csr_matrix(np.diag([1.0, -2.0, 3.0, 4.0]))
    L = sp.identity(4, format="csr")
    rb = sparse_ldl_residual_bound(L, np.array([1.0, -2.0, 3.0, 4.0]), A, np.arange(4))
    assert rb.beta >= 0
    monkeypatch.setattr(fp, "GUARD_RANGE_MAX", 3)
    with pytest.raises(fp.GuardRangeError):
        sparse_ldl_residual_bound(L, np.array([1.0, -2.0, 3.0, 4.0]), A, np.arange(4))


@pytest.mark.parametrize("method", ["dense", "sparse"])
def test_inertia_reports_no_count_outside_the_range(monkeypatch, method):
    A = sp.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0]))
    ok = inertia(A, 2.5, method=method)
    assert ok.conclusive and ok.negative == 2
    monkeypatch.setattr(fp, "GUARD_RANGE_MAX", 3)       # dense: ||R||_F has 16 entries; sparse: n = 4 > 3
    r = inertia(A, 2.5, method=method)
    assert not r.conclusive and r.negative is None and not r.resource_failure
    assert "the range of the floating-point guards" in r.note


# ---------------------------------------------------------------- certify()

@pytest.fixture(scope="module")
def chain8():
    spec = heisenberg_chain(8, 1.0)
    H = build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


def test_certify_inside_the_range_is_unchanged(chain8, monkeypatch):
    spec, H, w, V = chain8
    claim = Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
    st = _status(certify(claim, spec, Options()))
    assert st["S2.eigenpairs"] == ("CERTIFIED", None) and st["S3.completeness"] == ("CERTIFIED", None)
    monkeypatch.setattr(fp, "GUARD_RANGE_MAX", 2 ** 20)        # above every array normed here: same statuses
    assert _status(certify(claim, spec, Options())) == st


def test_certify_refuses_a_norm_longer_than_the_limit_inside_the_dimension_range(chain8, monkeypatch):
    """n = 256 and w within a limit of 300, but the cluster bound norms the 256 x 2 block of a degenerate pair
    (||X_g||_F over 512 entries): fp.GuardRangeError inside enclose() makes S2 INCONCLUSIVE with FP_GUARD_RANGE."""
    spec, H, w, V = chain8
    claim = Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
    monkeypatch.setattr(fp, "GUARD_RANGE_MAX", 300)
    cert = certify(claim, None, Options())
    st = _status(cert)
    assert st["S2.eigenpairs"] == ("INCONCLUSIVE", "FP_GUARD_RANGE")
    assert st["S3.completeness"] == ("INCONCLUSIVE", "FP_GUARD_RANGE")
    f = next(s for s in cert.statements if s["id"] == "S2.eigenpairs")["evidence"]["guard_range"]["findings"]
    assert f[0]["where"] == "fp.norm2_upper" and f[0]["value"] > 300


@pytest.mark.parametrize("kind", ["lowest", "all"])
def test_certify_above_the_range_is_never_certified(chain8, monkeypatch, kind):
    """Dimension 256 with the limit lowered to 255: S2 and everything that rests on it is INCONCLUSIVE with
    FP_GUARD_RANGE, before any enclosure is formed."""
    spec, H, w, V = chain8
    k = w.size if kind == "all" else 4
    claim = Claim(eigenvalues=w[:k], eigenvectors=V[:, :k], operator=H, kind=kind)
    monkeypatch.setattr(fp, "GUARD_RANGE_MAX", 255)
    cert = certify(claim, spec, Options(temperatures=(1.0,) if kind == "all" else ()))
    st = _status(cert)
    for sid in ("S2.eigenpairs", "S3.completeness", "S4.eigenspaces", "S8.degeneracy"):
        assert st[sid] == ("INCONCLUSIVE", "FP_GUARD_RANGE"), (sid, st[sid])
    if kind == "all":
        assert st["S7.thermodynamics"] == ("INCONCLUSIVE", "FP_GUARD_RANGE")
    s2 = next(s for s in cert.statements if s["id"] == "S2.eigenpairs")
    f = s2["evidence"]["guard_range"]["findings"]
    assert f[0]["quantity"] == "n" and f[0]["value"] == 256 and f[0]["limit"] == 255
    assert "next_action" in s2["evidence"]
    assert all(s["status"] != "CERTIFIED" for s in cert.statements if s["id"][:2] in ("S2", "S3", "S4", "S7", "S8"))


def _zstring_spec(n_sites):
    """One string of n_sites Pauli z letters: diagonal, row width bound n_terms 2^max_len + 2 max_len + 2
    (enclosure.OperatorAdapter, spec branch) with n_terms = 1."""
    local = LocalSpace.spin_space("1/2", "pauli")
    static = [["z" * n_sites, [[1.0] + list(range(n_sites))]]]
    return ModelSpec.from_static(n_sites, local, static, name="zstring")


@pytest.mark.parametrize("n_sites,inside", [(25, True), (26, False)])
def test_spec_operator_row_width_at_the_limit(n_sites, inside):
    """A spec operator on four basis states of n_sites spins: width 2^25 + 52 is inside the range, 2^26 + 54 is not.
    Nothing of size 2^n_sites is built (the states are given)."""
    spec = _zstring_spec(n_sites)
    states = StateSet(np.arange(4, dtype=np.int64), spec.state_order, "test")
    op = SpecOperator(spec, states)
    width = OperatorAdapter(op).width
    assert spec.n_terms == 1 and width == 2 ** n_sites + 2 * n_sites + 2
    d = op.matvec(np.ones(4))
    claim = Claim(eigenvalues=d, eigenvectors=np.eye(4), kind="subset", states=states, accuracy_target=float("inf"))
    cert = certify(claim, spec, Options(model_fidelity=False))
    s2 = _status(cert)["S2.eigenpairs"]
    if inside:
        assert s2 == ("CERTIFIED", None)
    else:
        assert s2 == ("INCONCLUSIVE", "FP_GUARD_RANGE")
        f = next(s for s in cert.statements if s["id"] == "S2.eigenpairs")["evidence"]["guard_range"]["findings"]
        assert f == [{"quantity": "w", "value": width, "limit": LIMIT,
                      "where": "certify: dimension and row width of the operator"}]
