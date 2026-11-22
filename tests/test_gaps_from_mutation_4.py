"""Tests closing the gaps found by the mutation analysis of the 2026-10-21 run (E6 record
e6_mutation_20261021_073237; the inspection verdicts are in that record's inspection.json)."""
import os
import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from conftest import needs_flint


def test_two_by_two_pivot_with_positive_determinant_is_counted_exactly():
    """inertia._count_block_diagonal and checker.negative_count_exact: a 2x2 block with det > 0 has two
    eigenvalues of the sign of its trace. Bunch-Kaufman 2x2 pivots are usually indefinite (det < 0),
    so this branch was not exercised (mutant inertia.py:78, index i+1 -> i-1, survived)."""
    from certify_ed.inertia import _count_block_diagonal
    from certify_ed.checker import negative_count_exact
    for blk, neg in (([[-2.0, 1.0], [1.0, -3.0]], 2), ([[2.0, 1.0], [1.0, 3.0]], 0), ([[1.0, 2.0], [2.0, 1.0]], 1)):
        D = np.zeros((4, 4))
        D[0, 0] = 5.0                 # a 1x1 pivot before the block
        D[1:3, 1:3] = blk
        D[3, 3] = -1.0                # and a negative 1x1 pivot after it
        expected = neg + 1
        assert _count_block_diagonal(D)[0] == expected
        assert negative_count_exact(D)[0] == expected
        assert expected == int(np.sum(np.linalg.eigvalsh(D) < 0))


@needs_flint
def test_exact_residual_with_huge_entries_uses_the_negative_exponent_branch():
    """checker.residual_inf_norm_exact: when 2 pL + pD < 0 (|D| above about 2^156) the scale is a
    fraction. Mutants of that branch (checker.py:112) survived because no test reached it."""
    from certify_ed.checker import residual_inf_norm_exact
    rng = np.random.default_rng(11)
    n = 8
    B = rng.integers(-5, 6, (n, n)).astype(float)
    A = (B + B.T + 30 * np.eye(n)) * 2.0 ** 170           # integer-valued, entries ~ 2^175
    lu, d, _ = la.ldl(A, lower=True, hermitian=True)
    e, Dt = residual_inf_norm_exact(A, lu, d)
    # exact residual of the dyadic factors, in rationals
    pL = 52
    Li = [[Fraction(int(np.rint(np.ldexp(x, pL))), 2 ** pL) for x in row] for row in lu]
    D = [[Fraction(x) for x in row] for row in Dt]
    LD = [[sum(Li[i][k] * D[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    E = [[sum(LD[i][k] * Li[j][k] for k in range(n)) - Fraction(A[i, j]) for j in range(n)] for i in range(n)]
    exact = max(sum(abs(x) for x in row) for row in E)
    assert Fraction(e) >= exact
    assert e <= 2 * float(exact) + 1e-300
    assert e < 1e-9 * np.abs(A).max()


def test_dekker_two_product_is_error_free():
    """fp._two_product: p + e == a*b exactly. A wrong splitting constant (mutant fp.py:183) leaves the
    sums plausible but not exact, which no bound-level test could see."""
    from certify_ed.fp import _two_product
    rng = np.random.default_rng(5)
    a = rng.standard_normal(500) * 10.0 ** rng.integers(-8, 8, 500)
    b = rng.standard_normal(500) * 10.0 ** rng.integers(-8, 8, 500)
    p, e = _two_product(a, b)
    for ai, bi, pi, ei in zip(a, b, p, e):
        assert Fraction(float(pi)) + Fraction(float(ei)) == Fraction(float(ai)) * Fraction(float(bi))


def test_tfim_chain_has_nearest_neighbour_bonds():
    """spec.tfim_chain: zz bonds between nearest neighbours only (mutant spec.py:244, range 1 -> 2)."""
    for boundary, nb in (("periodic", 6), ("open", 5)):
        spec = ce.tfim_chain(6, boundary=boundary)
        zz = [t for t in spec.terms if t.op == "zz"]
        bonds = sorted(tuple(sorted(s)) for t in zz for _c, s in t.couplings)
        assert len(bonds) == nb
        assert all((j - i) % 6 in (1, 5) for i, j in bonds)


def test_su2_structural_check_in_mixed_pauli_convention():
    """su2.su2_structural with 'pauli_pm1' letters (sigma^z = 2 S^z but sigma^+- = S^+-): the scale of
    x/y/z letters and of +- letters differ, so the products must be scaled by f*g (mutant su2.py:56)."""
    from certify_ed.su2 import su2_structural
    for conv in ("spin", "pauli", "pauli_pm1"):
        ok, why = su2_structural(ce.heisenberg_chain(6, convention=conv))
        assert ok, (conv, why)
    ok, _ = su2_structural(ce.heisenberg_chain(6, Delta=0.5, convention="pauli_pm1"))
    assert not ok


def test_sector_audit_reports_a_small_compression_defect_for_a_correct_sector():
    """sectors.audit_sector: the compression defect ||V^H H V - Hs|| is a reported diagnostic; for a
    correct block it is at rounding level, and for a wrong block it is O(||H||) (mutant sectors.py:68)."""
    from certify_ed.sectors import audit_sector
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, X = la.eigh(H.toarray())
    V = sp.csr_matrix(X[:, :3])                                  # an invariant subspace (lowest three)
    Hs = sp.csr_matrix(np.diag(w[:3]))
    good = audit_sector(H, V, Hs)
    assert good.passed and good.compression_defect < 1e-11
    bad = audit_sector(H, V, (Hs + sp.identity(Hs.shape[0])).tocsr())
    assert not bad.passed and bad.compression_defect > 1.0


@pytest.mark.skipif(sys.platform not in ("linux", "win32"), reason="uses fesetround through ctypes (glibc or ucrtbase)")
def test_probe_detects_round_up_mode():
    """fp.probe_float_environment under the directed rounding mode 'toward +inf': the
    round_to_nearest_up probe must fail (mutant fp.py:159 made it pass by testing 1 - 2^-60).
    Runs on glibc (libm.so.6, FE_UPWARD 0x800) and on Windows (ucrtbase, FE_UPWARD 0x200; IMP-24)."""
    lib, fe_upward = ("libm.so.6", 0x800) if sys.platform == "linux" else ("ucrtbase.dll", 0x200)
    code = textwrap.dedent(f"""
        import ctypes, json, sys
        sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
        from certify_ed.fp import probe_float_environment
        libm = ctypes.CDLL({lib!r})
        assert libm.fesetround({fe_upward}) == 0
        r = probe_float_environment()
        libm.fesetround(0)
        print(json.dumps([r["standard_model_hypotheses_hold"], r["failed"]]))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    import json
    holds, failed = json.loads(out.strip().splitlines()[-1])
    assert holds is False and "round_to_nearest_up" in failed
