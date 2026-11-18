"""E-03 generalized eigenproblems (certify_ed.generalized): random pencils against scipy.linalg.eigh(A, B), certified
lower bound on lambda_min(B), B^-1-norm residual radii, completeness by inertia of A - sigma B, negative controls
(indefinite B, wrong eigenvalue, swapped pairs, false declared lambda_min(B)), record schema."""
import json

import numpy as np
import pytest
import scipy.linalg as la

import certify_ed.generalized as G
from certify_ed.generalized import GENERALIZED_SCHEMA, certify_generalized, format_generalized
from certify_ed.schema import validate_record


def pencil(seed, n=40, cplx=False, cond=100.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
    A = A + A.conj().T
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0))
    B = (Q * np.logspace(0, -np.log10(cond), n)) @ Q.conj().T
    B = (B + B.conj().T) / 2
    w, V = la.eigh(A, B)
    return A, B, w, V


def stmt(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)


def exact_radius(A, B, mu, x):
    s = A @ x - mu * (B @ x)
    return np.sqrt(np.vdot(s, la.solve(B, s, assume_a="pos")).real) / np.sqrt(np.vdot(x, B @ x).real)


@pytest.mark.parametrize("cplx", [False, True])
def test_random_pencil_lowest_claim_certified(cplx):
    A, B, w, V = pencil(1, cplx=cplx)
    rec = certify_generalized(A, B, w[:5], V[:, :5], kind="lowest")
    assert rec["schema"] == GENERALIZED_SCHEMA
    g1, g2, g3 = (stmt(rec, s) for s in ("G1.B_positive_definite", "G2.eigenpairs", "G3.completeness"))
    assert g1["status"] == g2["status"] == g3["status"] == "CERTIFIED"
    assert 0 < g1["evidence"]["beta_c"] <= np.linalg.eigvalsh(B)[0]
    for i, e in enumerate(rec["enclosures"]):
        assert e["radius"] >= exact_radius(A, B, w[i], V[:, i]) * (1 - 1e-12)
        assert e["radius"] < 1e-8          # cond(B) = 100: sqrt(cond) times the 2-norm residual
        assert e["lower"] <= w[i] <= e["upper"]
    assert g3["evidence"]["count_from_inertia"] == 5
    assert g3["evidence"]["next_eigenvalue_lower_bound"] <= w[5]
    assert sum(w < g3["evidence"]["complete_below"]) == 5
    assert validate_record(rec) in (True, None)


def test_perturbed_vectors_keep_valid_radii():
    A, B, w, V = pencil(2, n=25)
    rng = np.random.default_rng(3)
    X = V[:, :3] + 1e-5 * rng.standard_normal((25, 3))
    rec = certify_generalized(A, B, w[:3], X, kind="lowest")
    for i, e in enumerate(rec["enclosures"]):
        r_ex = exact_radius(A, B, w[i], X[:, i])
        # beta_c = 0.5 lambda_min(B) (fraction 0.5): the radius is at most sqrt(2 cond(B)) times the exact B^-1-norm radius
        assert r_ex <= e["radius"] <= r_ex * (1 + 1e-8) * np.sqrt(2 * np.linalg.cond(B)) + 1e-14
        assert abs(w[i] - w[np.argmin(np.abs(w - w[i]))]) <= e["radius"]
    assert stmt(rec, "G3.completeness")["status"] == "CERTIFIED"


def test_negative_control_indefinite_B_refused():
    A, B, w, V = pencil(4, n=20)
    Bi = B - 2 * np.linalg.eigvalsh(B)[-1] * np.outer(V[:, 0], V[:, 0]) / (V[:, 0] @ V[:, 0])
    rec = certify_generalized(A, Bi, w[:2], V[:, :2], kind="lowest")
    assert all(s["status"] == "NOT_CHECKED" and s["evidence"]["reason_code"] == "UNSUPPORTED_MODEL" for s in rec["statements"])
    assert "not positive definite" in stmt(rec, "G1.B_positive_definite")["summary"]
    assert "next_action" in stmt(rec, "G1.B_positive_definite")["evidence"]
    with pytest.raises(ValueError, match="not Hermitian"):
        certify_generalized(A + np.triu(np.ones((20, 20)), 1), B, w[:2], V[:, :2])


def test_negative_control_wrong_eigenvalue_and_swapped_pairs():
    A, B, w, V = pencil(5, n=20)
    mu = w[:3].copy()
    mu[1] += 0.3 * (w[3] - w[0])
    rec = certify_generalized(A, B, mu, V[:, :3], kind="lowest")
    e = rec["enclosures"][1]
    assert e["radius"] >= abs(mu[1] - w[1]) * 0.99 and e["radius"] >= exact_radius(A, B, mu[1], V[:, 1]) * (1 - 1e-12)
    assert stmt(rec, "G3.completeness")["status"] != "CERTIFIED"
    rec = certify_generalized(A, B, w[:3], V[:, [1, 0, 2]], kind="lowest")
    g3 = stmt(rec, "G3.completeness")
    assert g3["status"] == "INCONCLUSIVE"       # the swapped radii are of the order of the level spacing
    assert g3["evidence"]["reason_code"] in ("NOT_IMPLEMENTED", "NOT_APPLICABLE")
    for i, e in enumerate(rec["enclosures"]):
        assert e["radius"] >= exact_radius(A, B, w[i], V[:, [1, 0, 2]][:, i]) * (1 - 1e-12)


def test_missing_eigenvalue_detected_by_inertia():
    A, B, w, V = pencil(6, n=20)
    rec = certify_generalized(A, B, w[[0, 1, 3]], V[:, [0, 1, 3]], kind="lowest")
    g3 = stmt(rec, "G3.completeness")
    assert g3["status"] == "INCONCLUSIVE" and g3["evidence"]["missing_at_least"] >= 1
    rec = certify_generalized(A, B, w[[0, 1, 3]], V[:, [0, 1, 3]], kind="subset")
    assert stmt(rec, "G3.completeness")["evidence"]["reason_code"] == "NO_COMPLETENESS_CLAIM"
    assert stmt(rec, "G2.eigenpairs")["status"] == "CERTIFIED"


def test_declared_lambda_min_B_is_a_named_hypothesis(monkeypatch):
    A, B, w, V = pencil(7, n=15)
    lam_min = np.linalg.eigvalsh(B)[0]
    monkeypatch.setattr(G, "certify_B_positive_definite",
                        lambda Bx, **kw: {"ok": False, "indefinite": False, "reason": "stub", "inertia": None, "inertia_at_zero": None,
                                          "beta_c": None, "beta": None})
    rec = certify_generalized(A, B, w[:2], V[:, :2], kind="lowest")
    assert stmt(rec, "G1.B_positive_definite")["status"] == "INCONCLUSIVE"
    assert all(stmt(rec, s)["status"] == "NOT_CHECKED" for s in ("G2.eigenpairs", "G3.completeness"))
    rec = certify_generalized(A, B, w[:2], V[:, :2], kind="lowest", lambda_min_B=0.5 * lam_min, lambda_min_B_source="construction")
    for sid in ("G1.B_positive_definite", "G2.eigenpairs", "G3.completeness"):
        s = stmt(rec, sid)
        assert s["status"] == "CERTIFIED_UNDER_HYPOTHESIS" and s["hypotheses"][0]["name"] == "declared_lambda_min_B"
        assert "HYPOTHESIS" in s["summary"]
    for i, e in enumerate(rec["enclosures"]):
        assert e["radius"] >= exact_radius(A, B, w[i], V[:, i]) * (1 - 1e-12)
    # false declaration (100 x lambda_max(B) > lambda_min): since ||s||_{B^-1} >= ||s|| / sqrt(lambda_max(B)), every radius
    # undercuts the exact B^-1-norm residual; the record names the hypothesis
    rng = np.random.default_rng(8)
    X = V[:, :2] + 1e-4 * rng.standard_normal((15, 2))
    bad = certify_generalized(A, B, w[:2], X, kind="lowest", lambda_min_B=100 * np.linalg.eigvalsh(B)[-1], lambda_min_B_source="WRONG")
    assert stmt(bad, "G2.eigenpairs")["status"] == "CERTIFIED_UNDER_HYPOTHESIS"
    assert stmt(bad, "G2.eigenpairs")["hypotheses"][0]["source"] == "WRONG"
    assert all(e["radius"] < exact_radius(A, B, w[i], X[:, i]) for i, e in enumerate(bad["enclosures"]))
    assert validate_record(bad) in (True, None)


def test_size_limit_and_json_roundtrip():
    A, B, w, V = pencil(9, n=12)
    rec = certify_generalized(A, B, w[:1], V[:, :1], max_dim=10)
    assert all(s["status"] == "NOT_CHECKED" and s["evidence"]["reason_code"] == "SIZE_LIMIT" for s in rec["statements"])
    rec = certify_generalized(A, B, w[:2], V[:, :2])
    text = json.dumps(rec)
    assert "G3.completeness" in text and "CERTIFIED" in format_generalized(rec)
