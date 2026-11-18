"""E-02 non-Hermitian eigenproblems (certify_ed.nonhermitian; every statement UNVERIFIED): backward errors (N1), the
certified bound on ||V^-1|| (N2), Bauer-Fike discs (N3) and their counts (N4), checked against exact spectra; reason
codes for singular or ill-conditioned eigenvector matrices, partial claims, underflow and size; the CLI; the record schema;
the derivation document and its verification script; the planted-error cases of corpus/nonhermitian."""
import importlib.util
import json
import os
import re
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la

from certify_ed.nonhermitian import (NONHERMITIAN_SCHEMA, NONHERMITIAN_STATEMENT_IDS, certify_nonhermitian,
                                     disc_groups, format_nonhermitian, inverse_bound, residual_bounds)
from certify_ed.schema import validate_record
from certify_ed.scope import REASON_CODES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROOFS = os.path.join(ROOT, "docs", "proofs")


def _status(rec):
    return {s["id"]: s["status"] for s in rec["statements"]}


def _code(rec, sid):
    return next(s for s in rec["statements"] if s["id"] == sid)["evidence"].get("reason_code")


def _cf(z):
    z = complex(z)
    return (Fraction(z.real), Fraction(z.imag))


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def test_random_real_matrix_with_conjugate_pairs_is_certified_and_the_discs_hold_the_eigenvalues():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((8, 8))
    w, V = np.linalg.eig(A)
    assert np.iscomplexobj(w) and np.any(np.abs(w.imag) > 0)
    rec = certify_nonhermitian(A, w, V)
    assert rec["schema"] == NONHERMITIAN_SCHEMA
    assert _status(rec) == {sid: "CERTIFIED" for sid in NONHERMITIAN_STATEMENT_IDS}
    rho = rec["radius"]["rho"]
    lam = la.eigvals(A)
    assert all(np.min(np.abs(l_ - w)) <= rho for l_ in lam)        # floating-point sanity; the exact checks follow
    assert len(rec["discs"]) == 8 and all(d["isolated"] for d in rec["discs"])
    assert "UNVERIFIED" in rec["statements"][0]["summary"] and "UNVERIFIED" in format_nonhermitian(rec)
    assert validate_record(rec) in (True, None)


def test_bauer_fike_equality_family_radius_is_at_least_the_exact_distance():
    """A = [[a, b], [0, a]] (eigenvalue a twice), x = (1, +-tau), mu = a +- b tau: ||V^-1 R||_2 = b tau = |mu - a| exactly,
    so the certified radius must be >= b tau, while the radius without the conditioning factor, ||R||_F = sqrt(2) b tau^2,
    and the backward error b tau^2 / sqrt(1 + tau^2) are both too small (negative controls)."""
    for b, tau in ((1.0, 2.0 ** -10), (2.0 ** 10, 2.0 ** -5), (2.0 ** -10, 2.0 ** -20)):
        a = 1.0
        A = np.array([[a, b], [0.0, a]])
        V = np.array([[1.0, 1.0], [tau, -tau]])
        mu = np.array([a + b * tau, a - b * tau])
        rec = certify_nonhermitian(A, mu, V)
        assert _status(rec)["N3.inclusion"] == "CERTIFIED"
        dist = Fraction(b) * Fraction(tau)
        assert Fraction(rec["radius"]["rho"]) >= dist
        # the two discs overlap: one group of two discs holds the double eigenvalue a
        assert [d["group"] for d in rec["discs"]] == [0, 0]
        rr = residual_bounds(A, mu, V)
        assert Fraction(float(np.sqrt(np.sum(rr["upper"] ** 2)))) < dist       # Bauer-Fike without ||V^-1|| fails
        assert Fraction(rec["pairs"][0]["eps_right"]) < dist                   # the backward error is no enclosure radius


def test_exact_planted_spectrum_with_a_jordan_block_is_never_overcounted():
    S = np.array([[1, 2, 0, 1], [0, 1, -1, 0], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=float)
    S = S @ np.array([[1, 0, 0, 0], [1, 1, 0, 0], [-2, 0, 1, 0], [0, 1, 1, 1]], dtype=float)
    Si = np.round(np.linalg.inv(S))
    assert np.array_equal(Si @ S, np.eye(4))
    J = np.array([[0.5, 1, 0, 0], [0, 0.5, 0, 0], [0, 0, -1.25, 0], [0, 0, 0, 2.0]])
    A = S @ J @ Si                                                    # small integers and dyadics: exact
    assert np.array_equal(A, np.round(A * 4) / 4)
    w, V = np.linalg.eig(A)
    rec = certify_nonhermitian(A, w, V)
    exact = [(Fraction(1, 2), Fraction(0))] * 2 + [(Fraction(-5, 4), Fraction(0)), (Fraction(2), Fraction(0))]
    st = _status(rec)
    if st["N4.counting"] == "CERTIFIED":
        r2 = Fraction(rec["radius"]["rho"]) ** 2
        gof = [d["group"] for d in rec["discs"]]
        for lam in exact:
            assert len({gof[i] for i in range(4) if _d2(lam, _cf(w[i])) <= r2}) == 1
        # the double eigenvalue 1/2 is never reported as two isolated discs each holding a simple eigenvalue
        near = [i for i in range(4) if abs(w[i] - 0.5) < 0.1]
        assert not all(rec["discs"][i]["isolated"] for i in near)
    else:
        assert _code(rec, "N2.eigenvector_inverse") == "ILL_CONDITIONED_EIGENVECTORS"


def test_duplicated_vector_makes_the_eigenvector_matrix_uncertifiable():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5))
    w, V = np.linalg.eig(A)
    V[:, 1], w[1] = V[:, 0], w[0]
    rec = certify_nonhermitian(A, w, V)
    st = _status(rec)
    assert st["N1.backward_errors"] == "CERTIFIED"
    assert st["N2.eigenvector_inverse"] == "INCONCLUSIVE"
    assert _code(rec, "N2.eigenvector_inverse") == "ILL_CONDITIONED_EIGENVECTORS"
    assert st["N3.inclusion"] == st["N4.counting"] == "NOT_CHECKED"
    assert _code(rec, "N3.inclusion") == "NEEDS_DEPENDENCY"
    assert all(s["evidence"].get("next_action") for s in rec["statements"] if s["status"] != "CERTIFIED")
    assert "ILL_CONDITIONED_EIGENVECTORS" in REASON_CODES
    assert validate_record(rec) in (True, None)


def test_left_vectors_give_left_backward_errors_that_bound_the_exact_left_residual():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    w, Y, V = la.eig(A, left=True, right=True)
    rec = certify_nonhermitian(A, w, V, left_eigenvectors=Y)
    assert _status(rec)["N1.backward_errors"] == "CERTIFIED"
    Af = [[_cf(A[i, j]) for j in range(6)] for i in range(6)]
    for j in (0, 3):
        y = [_cf(Y[i, j]) for i in range(6)]
        m = _cf(np.conj(w[j]))
        r = []
        for i in range(6):                                            # (A^H y - conj(mu) y)_i exactly
            s = [Fraction(0), Fraction(0)]
            for l_ in range(6):
                a = (Af[l_][i][0], -Af[l_][i][1])
                s[0] += a[0] * y[l_][0] - a[1] * y[l_][1]
                s[1] += a[0] * y[l_][1] + a[1] * y[l_][0]
            s[0] -= m[0] * y[i][0] - m[1] * y[i][1]
            s[1] -= m[0] * y[i][1] + m[1] * y[i][0]
            r.append(s[0] ** 2 + s[1] ** 2)
        ny2 = sum(c[0] ** 2 + c[1] ** 2 for c in y)
        assert Fraction(rec["pairs"][j]["eps_left"]) ** 2 * ny2 >= sum(r)


def test_subset_claim_gets_backward_errors_only():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((6, 6))
    w, V = np.linalg.eig(A)
    rec = certify_nonhermitian(A, w[:2], V[:, :2])
    st = _status(rec)
    assert rec["claim"]["kind"] == "subset" and st["N1.backward_errors"] == "CERTIFIED"
    for sid in NONHERMITIAN_STATEMENT_IDS[1:]:
        assert st[sid] == "NOT_CHECKED" and _code(rec, sid) == "NEEDS_WHOLE_SPECTRUM"
    with pytest.raises(ValueError):
        certify_nonhermitian(A, w[:2], V[:, :2], kind="all")


def test_underflow_scan_and_its_opt_out():
    rng = np.random.default_rng(9)
    A = rng.standard_normal((4, 4))
    w, V = np.linalg.eig(A)
    V = V.copy()
    V[0, 0] = 1e-310 if not np.iscomplexobj(V) else complex(1e-310, 0)
    rec = certify_nonhermitian(A, w, V)
    assert all(s["status"] == "INCONCLUSIVE" and s["evidence"]["reason_code"] == "UNDERFLOW_RISK" for s in rec["statements"])
    rec2 = certify_nonhermitian(A, w, V, underflow_check=False)
    assert any("underflow_check=False" in a for a in rec2["statements"][0]["assumptions"])


def test_size_limit_and_input_checks():
    A = np.eye(3) + np.triu(np.ones((3, 3)), 1)
    w, V = np.linalg.eig(A)
    rec = certify_nonhermitian(A, w, V, max_dim=2)
    assert all(s["status"] == "NOT_CHECKED" and s["evidence"]["reason_code"] == "SIZE_LIMIT" for s in rec["statements"])
    with pytest.raises(ValueError):
        certify_nonhermitian(np.ones((2, 3)), [1, 2], np.eye(2))
    with pytest.raises(ValueError):
        certify_nonhermitian(np.eye(2), [np.nan, 1.0], np.eye(2))
    with pytest.raises(ValueError):
        certify_nonhermitian(np.eye(2), [1.0, 1.0], np.eye(3))


def test_inverse_bound_and_disc_groups_directly():
    V = np.array([[1.0, 1.0], [0.0, 2.0 ** -30]])
    inv = inverse_bound(V)
    assert inv["ok"] and inv["beta"] >= 2.0 ** 30 and inv["theta"] < 1
    sing = inverse_bound(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert not sing["ok"] and sing["reason"]
    g = disc_groups(np.array([0.0, 1.0, 1.0 + 1e-3j, 5.0]), 0.1)
    assert g["groups"] == [[0], [1, 2], [3]] and g["isolated"] == [0, 3]
    # touching discs (distance exactly 2 rho) are never certified disjoint
    g2 = disc_groups(np.array([0.0, 0.25]), 0.125)
    assert g2["groups"] == [[0, 1]]


def test_cli_round_trip(tmp_path):
    from certify_ed.cli import main
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    w, Y, V = la.eig(A, left=True, right=True)
    p = tmp_path / "claim.npz"
    np.savez(p, operator=A, eigenvalues=w, eigenvectors=V, left_eigenvectors=Y)
    out = tmp_path / "rec.json"
    assert main(["nonhermitian", "--operator", str(p), "--left-eigenvectors", str(p), "--json", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert _status(rec) == {sid: "CERTIFIED" for sid in NONHERMITIAN_STATEMENT_IDS}
    assert rec["claim"]["left_eigenvectors_sha256"] is not None
    assert validate_record(rec) in (True, None)
    assert main(["nonhermitian", "--operator", str(p), "--kind", "subset", "--json", str(out)]) == 0
    assert _status(json.loads(out.read_text(encoding="utf-8")))["N2.eigenvector_inverse"] == "NOT_CHECKED"


def test_record_hash_detects_a_forged_status():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((4, 4))
    w, V = np.linalg.eig(A)
    V[:, 1], w[1] = V[:, 0], w[0]
    rec = certify_nonhermitian(A, w, V)
    bad = json.loads(json.dumps(rec))
    bad["statements"][1]["status"] = "CERTIFIED"
    with pytest.raises(Exception):
        validate_record(bad)


# ------------------------------------------------------------------------------------- derivation document and corpus
def test_derivation_document_has_the_six_sections_and_the_status_line():
    text = open(os.path.join(PROOFS, "E-02_non_hermitian.md"), encoding="utf-8").read()
    for num, title in enumerate(("Statement", "Proof", "Where the code implements it", "Computational verification",
                                 "Analytic cross-check", "Status"), start=1):
        assert re.search(rf"^## {num}\.? {re.escape(title)}", text, flags=re.M), title
    status = text[text.rindex("## 6"):]
    assert "Sarang's check (D-12): pending" in status and "not a human check" in status
    cleaned = status.lower().replace("unverified", "").replace("machine-checked", "")
    assert "verified" not in cleaned
    table = open(os.path.join(PROOFS, "VERDICTS.md"), encoding="utf-8").read()
    rows = [l_ for l_ in table.splitlines() if l_.startswith("|") and "E-02_non_hermitian.md" in l_.split("|")[1]]
    assert len(rows) == 1
    assert os.path.exists(os.path.join(PROOFS, "independent", "E-02_non_hermitian.md"))
    assert os.path.exists(os.path.join(PROOFS, "independent", "adversarial_E-02_non_hermitian.py"))
    assert os.path.exists(os.path.join(PROOFS, "runs", "verify_E-02.txt"))


def test_verification_script_reports_zero_violations_and_detects_the_negative_controls():
    pytest.importorskip("mpmath")
    spec = importlib.util.spec_from_file_location("verify_E_02", os.path.join(PROOFS, "verify_E-02.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.run(2, seed=1, quick=True)
    assert r["violations"] == 0 and r["ok"], r
    assert r["random_cases"] > 0 and r["adversarial_cases"] > 0
    assert all(hit > 0 for hit, tot in r["negative_controls"].values())


def test_planted_error_cases_of_the_nonhermitian_corpus():
    spec = importlib.util.spec_from_file_location("nh_run_cases", os.path.join(ROOT, "corpus", "nonhermitian", "run_cases.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.run(verbose=False)
    verdicts = {r_["case"]: r_["verdict"] for r_ in rows}
    assert len(verdicts) >= 10
    assert not [c for c, v in verdicts.items() if v.startswith(("unsound", "alarm"))], verdicts
    assert verdicts["nh_duplicated_vector"] == "detected" and verdicts["nh_subnormal_entry"] == "detected"
    assert all(v == "clean" for c, v in verdicts.items() if c.startswith("nh_correct") or c == "nh_bauer_fike_tight")
