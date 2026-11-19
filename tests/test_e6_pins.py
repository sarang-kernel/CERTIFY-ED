"""Pinning tests for the soundness-relevant survivors of the 3.2 mutation sample (roadmap N-02, 3.2.2).

E6_inspection_worksheet_3.2.0.md lists seven mutants classed 'soundness-relevant test gap' (a bound, a count or a
flag that no test pinned) and one 'test gap, false FAILED possible' (checker.py:535). The 3.2.0 and 3.2.1 E6 runs
(results/run_full_20261104/e6_mutation_20261104_163828, site indices of experiments/e6_mutation.py) let all eight
survive. Each test below pins the formula at its site to an exact expected value on a small planted case, as
tests/test_backlog_proofs.py does for the cluster bound. The mutants are those of experiments/e6_mutation.py
(apply_mutation at the recorded site index). Sites (module:line on dev/3.2 = v3.2.0 for these modules, mutation,
site index):

    sectors.py:57          Or->And          1    complex-data flag of audit_sector
    eigenspace.py:151      Mult->Div       11    rounding term of the label bound scaled by g
    eigenspace.py:147      Div->Mult       55    kappa = 1/sqrt(1 - eta) - 1
    eigenspace.py:129      Mult->Div       73    main term 2 g ||O|| s of the observable bound
    exact_sectors.py:202   FloorDiv->Mult  26    count of a complex block from its real embedding
    localops.py:189        0.0 -> 1.0     232    amplitude vector of the fermion lowering letter
    spectrum.py:103        And->Or         38    'counted' flag of a merged interval
    checker.py:535         Add->Sub       494    separation test of the witness count below the top claim
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from certify_ed.checker import certify_by_counts
from certify_ed.eigenspace import EigenspaceCertificate, certify_observable, certify_quantum_numbers
from certify_ed.exact_sectors import _count_sector_below
from certify_ed.fp import U, fro_upper, gamma_dot, inf_norm_upper, up
from certify_ed.localops import LocalSpace
from certify_ed.sectors import audit_sector
from certify_ed.spectrum import SpectrumInterval, merge_intervals


# ---------------------------------------------------------------- sectors.py:57

def _audit_formula(H, V, Hs, cplx):
    """The three defects of sectors.audit_sector with the complex flag given explicitly (same operations, same order)."""
    H, V, Hs = sp.csr_matrix(H), sp.csr_matrix(V), sp.csr_matrix(Hs)
    ds = V.shape[1]
    G = (V.conj().T @ V - sp.identity(ds, format="csr")).tocsr()
    wv = int(np.diff(sp.csc_matrix(V).indptr).max())
    iso = up(fro_upper(G) + gamma_dot(wv, cplx) * fro_upper(V) ** 2)
    HV = H @ V
    D = (HV - V @ Hs).tocsr()
    wh = int(np.diff(H.indptr).max())
    rnd = gamma_dot(wh + int(np.diff(Hs.indptr).max()) + 2, cplx) * (
        fro_upper(abs(H) @ abs(V)) + fro_upper(abs(V) @ abs(Hs)))
    inv = up(fro_upper(D) + rnd)
    C = (V.conj().T @ HV - Hs).tocsr()
    comp = up(fro_upper(C) + gamma_dot(wh + wv + 2, cplx) * fro_upper(abs(V).T @ abs(H) @ abs(V)))
    return float(iso), float(inv), float(comp)


def _one_complex_input(which):
    """A one-dimensional sector e_0 of a 3 x 3 Hermitian H with exactly one of (V, H, Hs) stored as complex."""
    H = np.diag([1.0, 2.0, 3.0])
    V = np.array([[1.0], [0.0], [0.0]])
    Hs = np.array([[1.0]])
    if which == "H":
        H = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1j], [0.0, -1j, 3.0]])    # complex outside the sector
    elif which == "V":
        V = np.array([[1j], [0.0], [0.0]])                                 # V^H H V = 1 stays real
    elif which == "Hs":
        Hs = np.array([[1.0 + 0.0j]])                                      # complex dtype, real value
    return H, V, Hs


@pytest.mark.parametrize("which", ["H", "V", "Hs"])
def test_audit_sector_uses_complex_constants_when_any_input_is_complex(which):
    H, V, Hs = _one_complex_input(which)
    a = audit_sector(sp.csr_matrix(H), sp.csr_matrix(V), sp.csr_matrix(Hs))
    expected = _audit_formula(H, V, Hs, cplx=True)
    assert (a.isometry_defect, a.invariance_defect, a.compression_defect) == expected
    # the pin is sensitive: the real constants give a strictly smaller isometry defect on this case
    assert _audit_formula(H, V, Hs, cplx=False)[0] < expected[0]
    assert a.passed


def test_audit_sector_real_data_keeps_real_constants():
    H, V, Hs = _one_complex_input(None)
    a = audit_sector(sp.csr_matrix(H), sp.csr_matrix(V), sp.csr_matrix(Hs))
    assert (a.isometry_defect, a.invariance_defect, a.compression_defect) == _audit_formula(H, V, Hs, cplx=False)


# ---------------------------------------------------------------- eigenspace.py:151, :147

def _qn_formula(ec, X, S, S_norm):
    """certify_quantum_numbers' bound written out (same operations, same order)."""
    Xg = X[:, ec.members]
    eta = ec.eta
    kappa = 1.0 / math.sqrt(1.0 - eta) - 1.0
    n = Xg.shape[0]
    cplx = np.iscomplexobj(Xg)
    width = n if not sp.issparse(S) else int(np.diff(sp.csr_matrix(S).indptr).max())
    g = len(ec.members)
    rnd = gamma_dot(width + n + 4, cplx) * S_norm * (1 + eta) * g
    rnd += 10 * g ** 2 * U * S_norm * (1 + eta)
    return up(2 * ec.sin_theta ** 2 * S_norm + (2 * kappa + kappa ** 2) * (1 + eta) * S_norm + rnd)


def _cert(g, eta=0.0, s=0.0, projector_error=0.0):
    return EigenspaceCertificate(list(range(g)), [0.0] * g, 1.0, 0.0, eta, s, projector_error)


def test_label_bound_rounding_term_scales_with_cluster_size():
    """eta = 0 and s = 0: the bound is the rounding term alone, gamma_dot(w + n + 4) ||S|| g + 10 g^2 u ||S||."""
    X = np.eye(6)[:, :3]
    S = sp.diags([1.0, 1.0, -1.0, 0.5, 0.0, -0.5]).tocsr()
    ec = _cert(3)
    rec = certify_quantum_numbers(ec, X, S, "Sz")
    S_norm = inf_norm_upper(S)
    assert rec["bound"] == float(_qn_formula(ec, X, S, S_norm))
    g, n = 3, 6
    assert rec["bound"] >= g * gamma_dot(1 + n + 4, False) * S_norm + 10 * g * g * U * S_norm


def test_label_bound_kappa_term_with_planted_eta():
    """eta = 0.1 planted: kappa = 1/sqrt(0.9) - 1 > 0.054 and the term (2 kappa + kappa^2)(1 + eta)||S|| dominates."""
    X = np.eye(4)[:, :2]
    S = sp.diags([1.0, -1.0, 2.0, -2.0]).tocsr()
    ec = _cert(2, eta=0.1, s=0.05)
    rec = certify_quantum_numbers(ec, X, S, "Sz")
    S_norm = inf_norm_upper(S)
    assert rec["bound"] == float(_qn_formula(ec, X, S, S_norm))
    kappa_lo = 0.054                       # 1/sqrt(0.9) - 1 = 0.05409...
    assert rec["bound"] >= (2 * kappa_lo + kappa_lo ** 2) * 1.1 * S_norm + 2 * 0.05 ** 2 * S_norm


# ---------------------------------------------------------------- eigenspace.py:129

def test_observable_bound_main_term():
    """|tr(P O) - tr(X^H O X)| <= 2 g ||O|| ||X X^H - P|| + rounding: g = 2, ||O|| = 3, projector error 1/4 gives 3."""
    X = np.eye(5)[:, :2]
    O = sp.diags([1.0, 2.0, 3.0, -1.0, 0.5]).tocsr()
    ec = _cert(2, projector_error=0.25)
    rec = certify_observable(ec, X, O, "O", O_norm=3.0)
    Xg = X[:, [0, 1]]
    absOX = abs(O) @ np.abs(Xg)
    rnd = gamma_dot(1 + 5 + 2, False) * float(np.sum(np.abs(Xg) * absOX)) * (1 + 8 * U)
    assert rec["bound"] == float(up(2 * 2 * 3.0 * 0.25 + rnd))
    assert rec["bound"] >= 3.0
    assert rec["value"] == 3.0


# ---------------------------------------------------------------- exact_sectors.py:202

def _obj(a):
    out = np.empty(np.shape(a), dtype=object)
    out[...] = np.asarray(a, dtype=object)
    return out


@pytest.mark.parametrize("g,count", [(Fraction(-2), 0), (Fraction(0), 1), (Fraction(1, 2), 1), (Fraction(2), 2)])
def test_complex_sector_block_count_through_real_embedding(g, count):
    """M = [[0, i], [-i, 0]] (eigenvalues -1, 1) as a Gaussian-integer block: its real embedding has every eigenvalue
    twice, and the count is nu / 2."""
    Are = _obj([[0, 0], [0, 0]])
    Aim = _obj([[0, 1], [-1, 0]])
    D = _obj([1, 1])
    nu, beta = _count_sector_below(Are, Aim, D, g, 1)
    assert nu == count
    assert beta is not None and beta >= 0


def test_real_sector_block_count():
    """Control: a real block [[2, 1], [1, 2]] (eigenvalues 1, 3) takes the factor-1 path."""
    Are = _obj([[2, 1], [1, 2]])
    Aim = _obj([[0, 0], [0, 0]])
    D = _obj([1, 1])
    assert _count_sector_below(Are, Aim, D, Fraction(2), 1)[0] == 1
    assert _count_sector_below(Are, Aim, D, Fraction(4), 1)[0] == 2


# ---------------------------------------------------------------- localops.py:189

def _pairs(comps):
    return [(d, [complex(a) for a in amp]) for d, amp in comps]


def test_fermion_letter_amplitudes_are_pinned():
    """<k + delta|o|k> per digit k, zero where the output digit is out of range (the docstring's contract). The
    builders (construct._apply_string, SpecOperator, matrixfree) also mask out-of-range digits, so the lowering
    letter's entry at k = 0 is consumed by no builder found; the pin is what holds the contract."""
    f = LocalSpace.fermion_space()
    assert _pairs(f.shift_components("+")) == [(1, [1.0, 0.0])]
    assert _pairs(f.shift_components("-")) == [(-1, [0.0, 1.0])]
    assert _pairs(f.shift_components("n")) == [(0, [0.0, 1.0])]
    assert _pairs(f.shift_components("z")) == [(0, [-0.5, 0.5])]
    assert f.exact_scaled_components("-") == [(-1, [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))])]


def test_fermion_shift_components_vanish_out_of_range():
    for local in (LocalSpace.fermion_space(), LocalSpace.fermion_space("lower")):
        for letter in local.letters:
            for delta, amp in local.shift_components(letter):
                for k in range(local.q):
                    if not 0 <= k + delta < local.q:
                        assert amp[k] == 0, (letter, k)


# ---------------------------------------------------------------- spectrum.py:103

def test_merged_interval_is_counted_only_when_both_parts_are():
    counted = SpectrumInterval(0.0, 1.0, 1, True, "single")
    uncounted = SpectrumInterval(0.5, 2.0, 1, False, "existence-only")
    se = merge_intervals([counted, uncounted], 2)
    assert len(se.intervals) == 1
    m = se.intervals[0]
    assert (m.count, m.counted, m.method) == (2, False, "merged-across-sectors")
    assert se.complete is False
    se2 = merge_intervals([counted, SpectrumInterval(0.5, 2.0, 1, True, "single")], 2)     # control
    assert se2.intervals[0].counted is True and se2.complete is True


# ---------------------------------------------------------------- checker.py:535

def _counts_with_witness(witness_delta):
    """One claimed component [0, 1] (k = 1). The count above it is 2 (an unclaimed eigenvalue at or above the top
    claim: INCONCLUSIVE), so certify_by_counts adds the witness count at g = 0 - 64; that count reports nu = 1 with the
    given delta."""
    comps = [[Fraction(0), Fraction(1), 1]]

    def count_at(g):
        if g > 1:
            return 2, Fraction(1, 10 ** 9)
        return 1, Fraction(witness_delta)
    return certify_by_counts(comps, None, count_at)


@pytest.mark.parametrize("dd", [64, 100])
def test_witness_window_meeting_the_top_component_is_not_separated(dd):
    """dd >= 64: the window [g - dd, g + dd] around g = -64 reaches the component's lower edge 0 (dd = 64 exactly at
    the closed end). The count nu = 1 there may be the claimed eigenvalue itself, so nothing is missing below the
    top claim: INCONCLUSIVE, never FAILED."""
    status, enclosures, next_lower, counts = _counts_with_witness(dd)
    w = [c for c in counts if c.get("witness_below_top")]
    assert len(w) == 1 and w[0]["point_exact"] == -64 and w[0]["separated"] is False
    assert status == "INCONCLUSIVE"


def test_witness_window_below_the_top_component_gives_failed():
    """dd = 1: the window [-65, -63] is separated from [0, 1], and one eigenvalue below -63 is not claimed: FAILED."""
    status, _, _, counts = _counts_with_witness(1)
    w = [c for c in counts if c.get("witness_below_top")]
    assert len(w) == 1 and w[0]["separated"] is True
    assert status == "FAILED"
