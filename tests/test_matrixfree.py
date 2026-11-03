"""D-02: matrix-free residual bounds from the spec (docs/proofs/D-02_matrix-free.md).

Pins: bit identity of MatrixFreeOperator with SpecOperator and with the stored matrix up to rounding; the
combinatorial counts (w_row = n_terms for spin-1/2, 2^L factors for spin-1 x/y, rho for rounded amplitudes); the
spec norm bound against the dense spectral norm; the adapter width and the radii (never larger than the 3.1 spec
path, always containing the eigenvalue); certify(matrix_free=True) end to end on a memmap vector with S1 naming the
amplitude hypothesis, S2 CERTIFIED, S3 NOT_CHECKED (NEEDS_MATRIX); negative controls (wrong coefficient, planted
wrong vector); and the scaled-coupling / cancellation adversarial cases."""
import os

import numpy as np
import pytest

import certify_ed as ce
from certify_ed.construct import SpecOperator, build_bitwise
from certify_ed.enclosure import OperatorAdapter, enclose
from certify_ed.localops import LocalSpace
from certify_ed.matrixfree import MatrixFreeOperator, letter_counts
from certify_ed.spec import ModelSpec, Term


@pytest.mark.parametrize("mk", [lambda: ce.heisenberg_chain(8), lambda: ce.tfim_chain(7), lambda: ce.spin1_heisenberg_chain(4, D=0.3),
                                lambda: ce.spinless_fermion_chain(7, V=0.4), lambda: ce.bose_hubbard_chain(4, 2),
                                lambda: ce.heisenberg_square(2, 3, J2=0.2)])
def test_bit_identity_with_spec_operator_and_agreement_with_matrix(mk):
    spec = mk()
    mf, so = MatrixFreeOperator(spec), SpecOperator(spec)
    H = build_bitwise(spec)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(spec.dim) + (1j * rng.standard_normal(spec.dim) if mf.dtype == np.complex128 else 0.0)
    assert np.array_equal(mf.matvec(x), so.matvec(x))
    assert np.array_equal(mf.abs_matvec(np.abs(x)), so.abs_matvec(np.abs(x)))
    assert np.array_equal(mf.abs_rmatvec(np.abs(x)), so.abs_rmatvec(np.abs(x)))
    Hx = H @ x
    assert np.abs(mf.matvec(x) - Hx).max() <= 1e-12 * (np.abs(Hx).max() + 1)
    X = np.column_stack([x, 2 * x])
    assert np.array_equal(mf.matvec(X)[:, 0], mf.matvec(x))


def test_counts_spin_half_equal_n_terms_and_spin_one_doubles():
    spec = ce.heisenberg_chain(8)
    mf = MatrixFreeOperator(spec)
    assert mf.terms_per_row == spec.n_terms and mf.rho == 0 and all(mf.exact_letters.values())
    assert mf.W == spec.n_terms + 2 + 0 + 1          # L = 2, real data
    assert OperatorAdapter(mf).width == mf.W - 2 < OperatorAdapter(SpecOperator(spec)).width
    half = LocalSpace.spin_space("1/2", "spin")
    one = LocalSpace.spin_space("1", "spin")
    assert letter_counts(half, "x") == 1 and letter_counts(one, "x") == 2 and letter_counts(one, "z") == 1
    s1 = ce.spin1_heisenberg_chain(4)
    m1 = MatrixFreeOperator(s1)
    assert m1.rho == 2 and not m1.exact_letters["+"] and m1.exact_letters["z"]
    xx = ModelSpec(3, one, [Term("xx", [(1.0, (0, 1)), (1.0, (1, 2))])])
    assert MatrixFreeOperator(xx).terms_per_row == 2 * 4


def test_norm_bound_dominates_spectral_norm_and_adapter_keeps_the_smaller():
    for spec in (ce.heisenberg_chain(7), ce.tfim_chain(6), ce.spin1_heisenberg_chain(3), ce.spinless_fermion_chain(6)):
        mf = MatrixFreeOperator(spec)
        H = build_bitwise(spec).toarray()
        true = np.linalg.norm(H, 2)
        assert true <= mf.norm_bound()
        ad = OperatorAdapter(mf)
        assert true <= ad.norm_upper() <= mf.norm_bound()


def test_radii_contain_eigenvalues_and_are_not_wider_than_the_3_1_spec_path():
    spec = ce.heisenberg_chain(9)
    H = build_bitwise(spec).toarray()
    w, V = np.linalg.eigh(H)
    E_new = enclose(OperatorAdapter(MatrixFreeOperator(spec)), w[:6], V[:, :6])
    E_old = enclose(OperatorAdapter(SpecOperator(spec)), w[:6], V[:, :6])
    assert np.all(E_new.radius <= E_old.radius)
    for i in range(6):
        lo, hi = E_new.interval(i)
        assert np.any((w >= lo) & (w <= hi))


def _magnon_claim(N, tmp_path):
    spec = ce.heisenberg_chain(N)
    n = 2 ** N
    path = os.path.join(str(tmp_path), f"magnon{N}.npy")
    x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(n, 1))
    x[:] = 0.0
    up = n - 1
    for j in range(N):
        s = up & ~(1 << (N - 1 - j))
        x[(n - 1 - s), 0] = (-1.0) ** j / np.sqrt(N)
    x.flush()
    del x
    return spec, np.load(path, mmap_mode="r"), N / 4.0 - 2.0


def test_certify_matrix_free_end_to_end_on_memmap_vector(tmp_path):
    spec, X, mu = _magnon_claim(10, tmp_path)
    c = ce.certify(ce.Claim(np.array([mu]), X, kind="subset"), spec, ce.Options(matrix_free=True))
    st = {s["id"]: s for s in c.statements}
    assert st["S1.model_fidelity"]["status"] == "CERTIFIED"
    assert st["S1.model_fidelity"]["evidence"]["model_shift_bound"] == 0.0
    hyp = st["S1.model_fidelity"]["evidence"]["amplitude_hypothesis"]
    assert hyp["terms_per_row"] == spec.n_terms and "exactly representable" in st["S1.model_fidelity"]["summary"]
    assert st["S2.eigenpairs"]["status"] == "CERTIFIED"
    assert c.arrays["radius"][0] < 1e-13
    assert st["S3.completeness"]["status"] == "NOT_CHECKED"
    # a 'lowest' claim without a stored matrix reports the reason code NEEDS_MATRIX
    c_low = ce.certify(ce.Claim(np.array([mu]), np.asarray(X), kind="lowest"), spec, ce.Options(matrix_free=True))
    s3 = {s["id"]: s for s in c_low.statements}["S3.completeness"]
    assert s3["status"] == "NOT_CHECKED" and s3["evidence"].get("reason_code") == "NEEDS_MATRIX"
    # the reference-matrix path gives the same eigenvalue enclosure, wider by the model shift
    c0 = ce.certify(ce.Claim(np.array([mu]), np.asarray(X), kind="subset"), spec, ce.Options())
    assert {s["id"]: s["status"] for s in c0.statements}["S2.eigenpairs"] == "CERTIFIED"
    assert c.arrays["radius"][0] <= c0.arrays["radius"][0]


def test_certify_matrix_free_spin_one_records_rounded_amplitudes(tmp_path):
    spec = ce.spin1_heisenberg_chain(4, D=0.2)
    H = build_bitwise(spec).toarray()
    w, V = np.linalg.eigh(H)
    c = ce.certify(ce.Claim(w[:2], V[:, :2], kind="lowest"), spec, ce.Options(matrix_free=True))
    st = {s["id"]: s for s in c.statements}
    hyp = st["S1.model_fidelity"]["evidence"]["amplitude_hypothesis"]
    assert hyp["rounded_amplitudes_per_contribution"] == 2 and "rounded once" in st["S1.model_fidelity"]["summary"]
    assert st["S2.eigenpairs"]["status"] == "CERTIFIED"


# ---------------------------------------------------------------- negative controls

def test_negative_control_wrong_coefficient_changes_the_residual_and_fails_s2():
    spec = ce.heisenberg_chain(8)
    H = build_bitwise(spec).toarray()
    w, V = np.linalg.eigh(H)
    good = ce.certify(ce.Claim(w[:3], V[:, :3], kind="subset"), spec, ce.Options(matrix_free=True))
    wrong = ce.heisenberg_chain(8)
    c0, sites = wrong.terms[0].couplings[0]
    wrong.terms[0].couplings[0] = (c0 * 1.01, sites)
    bad = ce.certify(ce.Claim(w[:3], V[:, :3], kind="subset", accuracy_target=1e-8), wrong, ce.Options(matrix_free=True))
    assert bad.arrays["radius"].min() > 1e3 * good.arrays["radius"].max()
    assert {s["id"]: s["status"] for s in bad.statements}["S2.eigenpairs"] in ("FAILED", "INCONCLUSIVE")


def test_negative_control_planted_wrong_eigenvector_fails_s2():
    spec = ce.heisenberg_chain(8)
    H = build_bitwise(spec).toarray()
    w, V = np.linalg.eigh(H)
    planted = V[:, 0] + 1e-2 * V[:, 5]
    bad = ce.certify(ce.Claim(w[:1], planted[:, None], kind="subset", accuracy_target=1e-8), spec, ce.Options(matrix_free=True))
    assert {s["id"]: s["status"] for s in bad.statements}["S2.eigenpairs"] in ("FAILED", "INCONCLUSIVE")
    assert bad.arrays["radius"][0] > 1e-4


# ---------------------------------------------------------------- adversarial

def test_cancelling_terms_and_scaled_couplings():
    half = LocalSpace.spin_space("1/2", "spin")
    cancel = ModelSpec(4, half, [Term("zz", [(1.0, (0, 1)), (-1.0, (0, 1))]), Term("xx", [(0.5, (1, 2)), (0.5, (2, 3))])])
    mf = MatrixFreeOperator(cancel)
    H = build_bitwise(cancel).toarray()
    x = np.random.default_rng(1).standard_normal(16)
    assert np.abs(mf.matvec(x) - H @ x).max() < 1e-14
    assert mf.terms_per_row == 4          # the cancelling pair still costs two roundings per row
    for scale in (1e150, 1e-150):
        sc = ModelSpec(4, half, [Term("zz", [(scale, (i, (i + 1) % 4)) for i in range(4)]),
                                 Term("xx", [(scale, (i, (i + 1) % 4)) for i in range(4)])])
        Hs = build_bitwise(sc).toarray()
        w, V = np.linalg.eigh(Hs)
        E = enclose(OperatorAdapter(MatrixFreeOperator(sc)), w[:2], V[:, :2])
        # at 1e-150 the radius is the INT-05 additive floor (~1e-161), not a relative quantity
        assert np.all(np.isfinite(E.radius)) and np.all(E.radius <= max(1e-12 * scale, 1e-150))
        for i in range(2):
            lo, hi = E.interval(i)
            assert np.any((w >= lo) & (w <= hi))
