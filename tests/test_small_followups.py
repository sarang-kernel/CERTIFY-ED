"""Small follow-ups: a sector isometry with a stored zero outside its support (restrict_to_u1 raised IndexError),
the reason code of S8 when the S2 radii of a cluster overflowed (S8 had none while S2 named the overflow), claimed
values with a non-finite imaginary part (the real part alone was certified), and (approximate) symmetries
given as an iterator or as a mapping with further keys (passed on without a private copy or a widening record)."""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.basis import digit_sum
from certify_ed.precision import claimed_reals
from certify_ed.sectors import restrict_to_u1


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _sector_isometry(spec, total, stored_zero_row=None):
    full = ce.full_state_set(spec)
    idx = full.index_of(ce.u1_state_set(spec, total).states_in_order())[0]
    rows, cols, vals = list(idx), list(range(idx.size)), [1.0] * idx.size
    if stored_zero_row is not None:
        rows.append(stored_zero_row)
        cols.append(0)
        vals.append(0.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(full.size, idx.size)), idx


def test_restrict_to_u1_ignores_a_stored_zero_outside_the_support():
    """A stored zero (as a cancelled sum leaves) in a row after the last row of the support: the restriction is that of
    the isometry without it (before: IndexError from the row look-up)."""
    spec = ce.heisenberg_chain(4)
    V0, idx = _sector_isometry(spec, 1)
    last = int(ce.full_state_set(spec).size) - 1
    assert last > int(idx.max())
    V1, _ = _sector_isometry(spec, 1, stored_zero_row=last)
    assert V1.nnz == V0.nnz + 1
    r0, r1 = restrict_to_u1(spec, V0), restrict_to_u1(spec, V1)
    assert r0 is not None and r1 is not None
    assert (r0[1] != r1[1]).nnz == 0 and r1[1].nnz == V0.nnz


@pytest.mark.parametrize("J", [1e200, 1e300])
def test_s8_names_overflowed_s2_radii(J):
    """Couplings of 1e200 and above: the S2 radii overflow, the cluster is not certified distinct, and S8 names the
    overflow as S2 does (before: INCONCLUSIVE without a code)."""
    spec = ce.heisenberg_chain(4, J=J)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["evidence"]["reason_code"] == "NONFINITE_BOUND"
    s8 = cert.statement("S8.degeneracy")
    assert s8["status"] == "INCONCLUSIVE" and s8["evidence"]["reason_code"] == "NONFINITE_BOUND", s8


def _chain6():
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    return spec, H, w, V


@pytest.mark.parametrize("imag", [np.inf, -np.inf, np.nan])
def test_claimed_value_with_non_finite_imaginary_part_is_not_certified(imag):
    """A claimed value whose imaginary part is not finite is a non-finite value: S2 is INCONCLUSIVE as for a nan claimed
    value, and the claim record holds nan for it (before: the imaginary part was dropped and the true eigenvalue w[1] +
    inf j was certified as w[1])."""
    spec, H, w, V = _chain6()
    lam = w[:3].astype(complex)
    lam[1] = complex(w[1], imag)
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    s2 = cert.statement("S2.eigenpairs")
    assert s2["status"] == "INCONCLUSIVE" and s2["evidence"]["non_finite_eigenvalues"] == [1], s2
    assert np.isnan(cert.body["claim"]["eigenvalues"][1])


def test_claimed_values_with_finite_or_zero_imaginary_parts_convert_as_before():
    """The 3.x conversion is kept for finite values (the real part, with numpy's ComplexWarning), and a real nan claimed
    value is INCONCLUSIVE as before."""
    spec, H, w, V = _chain6()
    cert = ce.certify(ce.Claim(eigenvalues=w[:3] + 0j, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    lam = w[:3].copy()
    lam[1] = np.nan
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "INCONCLUSIVE"
    assert np.array_equal(claimed_reals([1.0, 2 + 0j]), [1.0, 2.0])
    out = claimed_reals(np.array([1 + 2j, 3 + np.inf * 1j, np.nan + 0j]))
    assert out[0] == 1.0 and np.isnan(out[1]) and np.isnan(out[2])


def test_exact_recheck_and_generalized_see_a_non_finite_imaginary_part():
    """The exact re-checks (full space and symmetry sector) give INCONCLUSIVE and certify_generalized refuses, as for a
    nan value (before: all three used the real part)."""
    from certify_ed.checker import exact_recheck_lowest
    from certify_ed.generalized import certify_generalized
    spec, H, w, V = _chain6()
    assert exact_recheck_lowest(spec, w[:1], V[:, :1], next_hint=w[1])["status"] == "CERTIFIED"
    lam = np.array([complex(w[0], np.inf)])
    assert exact_recheck_lowest(spec, lam, V[:, :1], next_hint=w[1])["status"] == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        certify_generalized(H, sp.identity(H.shape[0], format="csr"), lam, V[:, :1], lambda_min_B=1.0)
    import scipy.linalg as la

    from certify_ed.basis import cyclic_isometry, translation
    from certify_ed.exact_sectors import exact_recheck_sector_lowest
    st = ce.u1_state_set(spec, 3)
    T = translation(spec)
    V0 = cyclic_isometry(st, T, 0)
    ws, Ys = la.eigh((V0.conj().T @ ce.build_bitwise(spec, st) @ V0).toarray())
    x = np.real_if_close(V0 @ Ys[:, :1])
    assert exact_recheck_sector_lowest(spec, st, [T], {0: 0}, ws[:1], x, next_hint=ws[1])["status"] == "CERTIFIED"
    lam_s = np.array([complex(ws[0], np.inf)])
    assert exact_recheck_sector_lowest(spec, st, [T], {0: 0}, lam_s, x, next_hint=ws[1])["status"] == "INCONCLUSIVE"


def _widened(cert):
    rec = (cert.body.get("arithmetic") or {}).get("input_widening") or {}
    return [(c["input"], c["dtype"], c["source_sha256"]) for c in rec.get("converted", [])]


def test_symmetry_given_as_an_iterator_is_recorded_as_a_tuple_is():
    """A symmetry entry read once (an iterator): its float32 diagonal is converted and recorded as for a tuple
    (before: no widening record)."""
    spec, H, w, V = _chain6()
    full = ce.full_state_set(spec)
    Sd = digit_sum(full.states_in_order().astype(np.int64), spec.q, spec.n_sites).astype(np.float32)
    runs = []
    for entry in ((Sd, None), iter((Sd, None))):
        cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                          ce.Options(seed=1, symmetries={"d": entry}))
        runs.append((_widened(cert), cert.statement("S4.eigenspaces")["status"]))
    assert runs[0] == runs[1]
    assert [x[0] for x in runs[1][0]] == ["options.symmetries[d]"]


def test_approximate_symmetry_given_as_a_mapping_with_a_further_key():
    """An approximate symmetry read by index from a mapping that has a further key: its float32 matrix is converted
    and recorded, and the caller's mapping and matrix are left as given (before: passed on without either)."""
    spec, H, w, V = _chain6()
    full = ce.full_state_set(spec)
    Sd = digit_sum(full.states_in_order().astype(np.int64), spec.q, spec.n_sites) - 3.0
    O = sp.csr_matrix(sp.diags(Sd.astype(np.float32)))
    data, indices = O.data.copy(), O.indices.copy()
    runs = []
    for entry in ((O, None, None), {0: O, 1: None, 2: None, "note": "a further key"}):
        cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                          ce.Options(seed=1, approximate_symmetries={"m": entry}))
        runs.append((_widened(cert), cert.statement("S4.eigenspaces")["status"]))
        if isinstance(entry, dict):
            assert entry[0] is O and sorted(entry, key=str) == [0, 1, 2, "note"]
    assert runs[0] == runs[1]
    assert [x[0] for x in runs[1][0]] == ["options.approximate_symmetries[m]"]
    assert O.dtype == np.float32 and np.array_equal(O.data, data) and np.array_equal(O.indices, indices)


@pytest.mark.parametrize("bad", [complex(0.0, np.inf), complex(np.nan, 0.0)])
def test_sector_composition_with_a_non_finite_claimed_value_is_inconclusive(bad):
    """A sector value w + inf j or nan: the sector is INCONCLUSIVE with the indices recorded, and so is the
    composition (before: w + inf j was certified as w, and nan made the sector MISSING and the composition FAILED)."""
    import scipy.linalg as la

    from certify_ed.compose import SectorResult, certify_low_energy
    spec = ce.heisenberg_chain(6)
    H = ce.build_bitwise(spec)
    full = ce.full_state_set(spec)
    w = np.linalg.eigvalsh(H.toarray())
    E_cut = float((w[3] + w[4]) / 2)
    secs, hit = [], None
    for nup in range(spec.n_sites + 1):
        idx = full.index_of(ce.u1_state_set(spec, nup).states_in_order())[0]
        Vs = sp.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))), shape=(H.shape[0], idx.size))
        Hs = (Vs.T @ H @ Vs).tocsr()
        ws, Ys = la.eigh(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        ev = ws[sel].astype(complex)
        if sel.size and hit is None:
            hit = f"nup={nup}"
            ev[0] = complex(ev[0].real, np.inf) if np.isinf(bad.imag) else complex(np.nan, 0.0)
        secs.append(SectorResult(f"nup={nup}", Vs, Hs, ev, Ys[:, sel]))
    res = certify_low_energy(spec, secs, E_cut)
    assert res["status"] == "INCONCLUSIVE", res["summary"]
    rec = next(p for p in res["sectors"] if p["label"] == hit)
    assert rec["status"] == "INCONCLUSIVE" and rec["non_finite_eigenvalues"] == [0]
