"""C-02: eigenvalue-only window claims counted by inertia alone (Lemma 4 at three shifts per edge, the count
inequality of Proposition 14 on the edge collars). S2 stays NOT_CHECKED: nothing pairs a claimed value with an
eigenvalue. Negative controls: a planted wrong count is FAILED, an edge on an eigenvalue is INCONCLUSIVE, and
no window with a count different from the claim is ever CERTIFIED."""
import numpy as np
import pytest
import scipy.linalg as la

import certify_ed as ce
from certify_ed.certificate_v2 import Certificate


@pytest.fixture(scope="module")
def heis8():
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w = la.eigvalsh(H.toarray())
    return spec, H, w


def _window_cert(spec, H, mu, a, b, **opts):
    return ce.certify(ce.Claim(np.asarray(mu, dtype=float), None, operator=H, kind="window", window=(float(a), float(b))),
                      spec, ce.Options(**opts))


def _s3(cert):
    return cert.statement("S3.completeness")


def test_vectorless_count_matches_dense_spectrum(heis8):
    spec, H, w = heis8
    # levels of the N = 8 ring: 1, 3, 1, 6, 6, 3, ...; windows between levels, at several places
    for lo_i, hi_i in ((4, 11), (0, 5), (1, 17), (11, 17)):
        assert w[hi_i] - w[hi_i - 1] > 1e-6 and (lo_i == 0 or w[lo_i] - w[lo_i - 1] > 1e-6)
        a = 0.5 * (w[lo_i - 1] + w[lo_i]) if lo_i > 0 else w[0] - 0.1
        b = 0.5 * (w[hi_i - 1] + w[hi_i])
        n_true = int(np.sum((w > a) & (w < b)))
        cert = _window_cert(spec, H, w[lo_i:hi_i], a, b)
        s3 = _s3(cert)
        assert s3["status"] == "CERTIFIED", s3["summary"]
        ev = s3["evidence"]
        assert ev["vectorless"] is True and ev["count_in_window"] == n_true == hi_i - lo_i
        assert ev["edges"]["a"]["exact"] and ev["edges"]["b"]["exact"]
        assert "no claimed value is paired" in s3["summary"] and "pairing" in ev
        # the eigenvalue-free collars really are eigenvalue-free, and each contains its edge
        for name, e in (("a", a), ("b", b)):
            lo, hi = ev["edges"][name]["eigenvalue_free"]
            assert lo < e < hi and not np.any((w >= lo) & (w <= hi))
        assert cert.status_of("S2.eigenpairs") == "NOT_CHECKED"
        assert cert.statement("S2.eigenpairs")["evidence"]["reason_code"] == "NEEDS_EIGENVECTORS"
        assert cert.status_of("S4.eigenspaces") == "NOT_CHECKED"
        assert cert.body["gaps"] is None
        assert cert.verify_scope()["ok"]


def test_vectorless_planted_wrong_count_fails(heis8):
    spec, H, w = heis8
    a, b = 0.5 * (w[3] + w[4]), 0.5 * (w[10] + w[11])        # 7 eigenvalues
    missing = _s3(_window_cert(spec, H, w[4:10], a, b))        # 6 claimed
    assert missing["status"] == "FAILED" and missing["evidence"]["missing_at_least"] == 1
    assert "not claimed" in missing["summary"]
    extra = _s3(_window_cert(spec, H, np.r_[w[4:11], -2.5], a, b))   # 8 claimed
    assert extra["status"] == "FAILED" and extra["evidence"]["spurious_at_least"] == 1
    assert "not eigenvalues in the window" in extra["summary"]
    # the claimed values themselves do not matter without vectors: garbage values with the right count pass the count
    garbage = _s3(_window_cert(spec, H, np.linspace(a, b, 9)[1:-1], a, b))
    assert garbage["status"] == "CERTIFIED" and "no claimed value is paired" in garbage["summary"]


def test_vectorless_edge_on_eigenvalue_is_inconclusive(heis8):
    spec, H, w = heis8
    b = 0.5 * (w[10] + w[11])
    for a in (w[4], w[4] + 1e-13, w[4] - 1e-12):
        s3 = _s3(_window_cert(spec, H, w[4:11], a, b))
        assert s3["status"] == "INCONCLUSIVE", s3["summary"]
        assert "window edge" in s3["summary"] and "eigenvalue" in s3["summary"]
        assert s3["evidence"]["edges"]["a"]["exact"] is False
        assert "count_in_window" not in s3["evidence"]
        assert "move the window edges" in s3["evidence"]["next_action"]


def test_vectorless_never_certifies_a_wrong_count(heis8):
    """Negative control over many windows: CERTIFIED implies count_in_window == claimed == dense count."""
    spec, H, w = heis8
    rng = np.random.default_rng(7)
    n_cert = 0
    for _ in range(12):
        a, b = sorted(rng.uniform(w[0] - 0.2, w[-1] + 0.2, size=2))
        n_true = int(np.sum((w > a) & (w < b)))
        k = max(0, n_true + int(rng.integers(-1, 2)))
        s3 = _s3(_window_cert(spec, H, np.linspace(a, b, k + 2)[1:-1], a, b))
        if s3["status"] == "CERTIFIED":
            n_cert += 1
            assert s3["evidence"]["count_in_window"] == n_true == k
        elif s3["status"] == "FAILED":
            assert k != n_true
    assert n_cert > 0


def test_vectorless_respects_gates_and_options(heis8):
    spec, H, w = heis8
    a, b = 0.5 * (w[3] + w[4]), 0.5 * (w[10] + w[11])
    off = _s3(_window_cert(spec, H, w[4:11], a, b, inertia=False))
    assert off["status"] == "NOT_CHECKED" and off["evidence"]["reason_code"] == "NARROWED_BY_USER"
    small = _s3(_window_cert(spec, H, w[4:11], a, b, inertia_max_dim=10))
    assert small["status"] == "NOT_CHECKED" and small["evidence"]["reason_code"] == "SIZE_LIMIT"
    budget = _s3(_window_cert(spec, H, w[4:11], a, b, budget_seconds=0.0))
    assert budget["status"] == "NOT_CHECKED" and budget["evidence"]["reason_code"] == "BUDGET_TIME"
    skipped = _s3(_window_cert(spec, H, w[4:11], a, b, skip=["S3"], narrowing_reason="t"))
    assert skipped["status"] == "NOT_CHECKED"
    # a 'lowest' eigenvalue-only claim is unchanged: NOT_CHECKED, needs eigenvectors
    low = ce.certify(ce.Claim(w[:4], None, operator=H, kind="lowest"), spec, ce.Options())
    assert _s3(low)["status"] == "NOT_CHECKED" and _s3(low)["evidence"]["reason_code"] == "NEEDS_EIGENVECTORS"
    # an inverted window is refused when the Claim is built (C-14's parse_window), before any statement
    with pytest.raises(ValueError, match="a < b"):
        _window_cert(spec, H, w[4:11], b, a)


def test_vectorless_certificate_saves_and_validates(heis8, tmp_path):
    spec, H, w = heis8
    a, b = 0.5 * (w[3] + w[4]), 0.5 * (w[10] + w[11])
    cert = _window_cert(spec, H, w[4:11], a, b)
    path = cert.save(str(tmp_path / "vectorless"))
    loaded = Certificate.load(path)
    assert loaded.validate_schema() in (True, None)
    assert loaded.statement("S3.completeness")["evidence"]["count_in_window"] == 7
    assert "7 eigenvalues" in loaded.summary() or "S3" in loaded.summary()
