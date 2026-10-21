"""Regression tests for the fixes of the external review of 2026-10-21 (ids IMP-xx, SS-xx, MS-Nxx refer to
REVIEW_CERTIFY-ED_2026-10-21). Each test names the finding it closes; the attack cases of the review
are in test_adversarial.py."""
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce


def _st(cert):
    return {s["id"]: s["status"] for s in cert.statements}


def _stmt(cert, sid):
    return next(s for s in cert.statements if s["id"] == sid)


@pytest.fixture(scope="module")
def heis6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    return spec, H, w, V


# ------------------------------------------------------------------ IMP-18: zero test vector in the exact layer
def test_imp18_radius_exact_returns_none_for_zero_vector(heis6):
    from certify_ed.checker import exact_matrix, radius_exact, _float_vector_to_int
    spec, H, w, V = heis6
    ex = exact_matrix(spec)
    yr, yi, _ = _float_vector_to_int(np.zeros(H.shape[0]))
    assert radius_exact(ex, yr, yi, w[0]) is None
    yr, yi, _ = _float_vector_to_int(V[:, 0])
    assert radius_exact(ex, yr, yi, w[0]) is not None


def test_imp18_exact_radii_report_zero_vectors(heis6):
    spec, H, w, V = heis6
    X = np.column_stack([V[:, 0], np.zeros(H.shape[0])])
    r = ce.exact_eigenpair_radii(spec, [w[0], w[0]], X)
    assert r["zero_test_vectors"] == [1]
    assert math.isinf(r["radii"][1]) and r["radii"][0] < 1e-12


def test_imp18_exact_recheck_lowest_inconclusive_for_zero_vector(heis6):
    spec, H, w, V = heis6
    r = ce.exact_recheck_lowest(spec, np.array([w[0] + 0.05]), np.zeros((H.shape[0], 1)))
    assert r["status"] == "INCONCLUSIVE" and "x != 0" in r["reason"]


def test_imp18_certify_S6_inconclusive_for_zero_vector(heis6):
    spec, H, w, V = heis6
    c = ce.certify(ce.Claim(np.array([w[0] + 0.05]), np.zeros((H.shape[0], 1)), operator=H, kind="lowest"),
                   spec, ce.Options(exact_recheck=True, inertia=False))
    s6 = _stmt(c, "S6.exact_recheck")
    assert s6["status"] == "INCONCLUSIVE" and "zero test vector" in s6["summary"]


def test_imp18_exact_sector_recheck_inconclusive_for_wrong_sector_vector():
    """A k=pi eigenvector claimed in the k=0 sector: its exact projection onto k=0 is zero."""
    from certify_ed.basis import translation, cyclic_isometry
    spec = ce.heisenberg_chain(10)
    st = ce.u1_state_set(spec, 5)
    H = ce.build_bitwise(spec, st)
    T = translation(spec)
    V0, Vpi = cyclic_isometry(st, T, 0), cyclic_isometry(st, T, 5)
    w0 = la.eigvalsh((V0.conj().T @ H @ V0).toarray())
    wpi, Ypi = la.eigh((Vpi.conj().T @ H @ Vpi).toarray())
    x = np.asarray(np.real_if_close(Vpi @ Ypi[:, :1]))
    r = ce.exact_recheck_sector_lowest(spec, st, [T], {0: 0}, np.array([w0[0] + 0.01]), x, next_hint=w0[1])
    assert r["status"] == "INCONCLUSIVE" and r.get("zero_test_vectors") == [0]
