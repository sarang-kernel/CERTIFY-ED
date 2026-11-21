"""Small follow-ups: a sector isometry with a stored zero outside its support (restrict_to_u1 raised IndexError),
the reason code of S8 when the S2 radii of a cluster overflowed (S8 had none while S2 named the overflow), claimed
values with a non-finite imaginary part (the real part alone was certified), and (approximate) symmetries
given as an iterator or as a mapping with further keys (passed on without a private copy or a widening record), and
the message for a moments operator with entries that are not finite."""
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
    """The 3.x conversion is kept for finite values (the real part, with numpy's ComplexWarning; the claim record now
    says how many finite nonzero imaginary parts were dropped and the largest), and a real nan claimed value is
    INCONCLUSIVE as before."""
    spec, H, w, V = _chain6()
    cert = ce.certify(ce.Claim(eigenvalues=w[:3] + 0j, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    assert "dropped_imaginary_parts" not in cert.body["claim"]
    lam = w[:3].astype(complex)
    lam[1] = complex(w[1], 0.5)
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"           # the real parts, as in 3.x
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.5}
    lam = w[:3].copy()
    lam[1] = np.nan
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "INCONCLUSIVE"
    assert np.array_equal(claimed_reals(np.array([1.0, 2 + 0j])), [1.0, 2.0])
    with pytest.raises(TypeError):                # a list of Python complex numbers, refused as in 3.x
        claimed_reals([1.0, 2 + 5j])
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
        cert = ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest"), spec,
                          ce.Options(seed=1, symmetries={"d": entry}))
        s4 = cert.statement("S4.eigenspaces")
        runs.append((_widened(cert), s4["status"], s4["summary"]))
    assert runs[0] == runs[1] and runs[0][1] != "NOT_CHECKED"       # four pairs: S4 reads the entries
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
        cert = ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest"), spec,
                          ce.Options(seed=1, approximate_symmetries={"m": entry}))
        s4 = cert.statement("S4.eigenspaces")
        runs.append((_widened(cert), s4["status"], s4["summary"]))
        if isinstance(entry, dict):
            assert entry[0] is O and sorted(entry, key=str) == [0, 1, 2, "note"]
    assert runs[0] == runs[1] and runs[0][1] != "NOT_CHECKED"
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
    assert rec["status"] == "INCONCLUSIVE" and rec["non_finite_eigenvalues"] == [0] and rec["n_non_finite"] == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_moments_name_entries_that_are_not_finite(sparse):
    """certified_moments given an H with an infinite entry refuses it as not finite (before: 'not Hermitian' with a
    nan difference, the wrong cause); a finite H with the same pattern is accepted."""
    from certify_ed.moments import certified_moments
    H = np.diag([1.0, 2.0, 3.0])
    x = np.eye(3)[:, :1]
    O = np.diag([1.0, 0.0, 0.0])
    certified_moments(sp.csr_matrix(H) if sparse else H, O, x, 1e-12, 1e-12, K=2, sum_rules=False)
    Hb = H.copy()
    Hb[2, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite entries"):
        certified_moments(sp.csr_matrix(Hb) if sparse else Hb, O, x, 1e-12, 1e-12, K=2, sum_rules=False)
    if sparse:                                    # repeated entries 1e308 + 1e308: finite parts, an infinite sum
        D = sp.csr_matrix((np.array([1.0, 1e308, 1e308, 1e308, 1e308, 2.0, 3.0]), np.array([0, 1, 1, 0, 0, 1, 2]),
                           np.array([0, 3, 6, 7])), shape=(3, 3))
        with pytest.raises(ValueError, match="non-finite entries"):
            certified_moments(D, O, x, 1e-12, 1e-12, K=2, sum_rules=False)


def test_other_entry_points_see_a_non_finite_imaginary_part():
    """certified_sweep with a solver that returns w0 + inf j certifies no S2 at that point; exact_eigenpair_radii gives
    an infinite radius and lists the index; enclose and write_bundle refuse the value and corroborate is INCONCLUSIVE
    (before: all of them used w0)."""
    import tempfile

    from certify_ed.checker import exact_eigenpair_radii
    from certify_ed.corroborate import corroborate
    from certify_ed.enclosure import OperatorAdapter, enclose
    from certify_ed.io.bundle import write_bundle
    from certify_ed.sweep import certified_sweep

    def solver(Hc, k):
        ww, VV = np.linalg.eigh(Hc.toarray())
        ww = ww.astype(complex)
        ww[0] = complex(ww[0].real, np.inf)
        return ww[:k], VV[:, :k]
    sweep = certified_sweep(lambda c: ce.heisenberg_chain(4, J=c), [1.0, 1.1], k=1, solver=solver)
    assert sweep["points"][0]["S2"] != "CERTIFIED", sweep["points"][0]
    spec, H, w, V = _chain6()
    bad = np.array([complex(w[0], np.inf)])
    r = exact_eigenpair_radii(spec, bad, V[:, :1])
    assert r["radii"] == [np.inf] and r["non_finite_values"] == [0]
    with pytest.raises(ValueError, match="finite"):
        enclose(OperatorAdapter(H), bad, V[:, :1])
    assert corroborate(H, np.array([complex(w[0], np.inf), w[1]]))["status"] == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="finite"):
        write_bundle(tempfile.mkdtemp(), bad, V[:, :1], H)


def test_certify_hamiltonian_sees_a_non_finite_imaginary_part():
    """The QuSpin entry point: w[1] + inf j makes S2 INCONCLUSIVE, as through certify (before: CERTIFIED as w[1])."""
    pytest.importorskip("quspin")
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian

    from certify_ed.integrations import quspin as cq
    spec = ce.heisenberg_chain(6)
    Hq = hamiltonian(spec.to_quspin_static(), [], basis=spin_basis_1d(6, pauli=0), dtype=np.float64, check_herm=False,
                     check_symm=False, check_pcon=False)
    w, V = Hq.eigh()
    cert = cq.certify_hamiltonian(Hq, w[:3], V[:, :3], kind="lowest", seed=1)
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    lam = w[:3].astype(complex)
    lam[1] = complex(w[1], np.inf)
    cert = cq.certify_hamiltonian(Hq, lam, V[:, :3], kind="lowest", seed=1)
    assert cert.statement("S2.eigenpairs")["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_symmetry_given_as_a_mapping_is_read_as_its_pair(dtype):
    """Options.symmetries entries are read by index: a mapping {0: S, 1: lattice} gives the S4 record of the tuple
    entry, in float32 and in binary64 (before: binary64 raised a TypeError, and the private copy of the float32 entry,
    a mapping on this branch before the fix, raised as well); an empty iterator is refused with its reason."""
    spec, H, w, V = _chain6()
    full = ce.full_state_set(spec)
    Sd = digit_sum(full.states_in_order().astype(np.int64), spec.q, spec.n_sites).astype(dtype)
    runs = []
    for entry in ((Sd, None), {0: Sd, 1: None}):
        cert = ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest"), spec,
                          ce.Options(seed=1, symmetries={"d": entry}))
        s4 = cert.statement("S4.eigenspaces")
        runs.append((s4["status"], s4["summary"], _widened(cert)))
    assert runs[0] == runs[1]
    with pytest.raises(ValueError, match="empty iterator"):
        ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest"), spec,
                   ce.Options(seed=1, symmetries={"d": iter(())}))


@pytest.mark.parametrize("scale", [1e-170, 1e155])
def test_s8_takes_the_s2_code_for_an_unresolved_vector_norm(scale):
    """Vectors whose squared norms underflow or overflow: S2 is INCONCLUSIVE with UNRESOLVED_VECTOR_NORM and S8 gives
    the same code for the clusters S2 did not certify distinct (before this branch: no code; on its first round:
    NONFINITE_BOUND, although nothing overflowed for 1e-170)."""
    spec = ce.heisenberg_chain(4)
    H = ce.build_bitwise(spec).tocsr()
    w, V = np.linalg.eigh(H.toarray())
    cert = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3] * scale, operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["evidence"]["reason_code"] == "UNRESOLVED_VECTOR_NORM"
    s8 = cert.statement("S8.degeneracy")
    assert s8["status"] == "INCONCLUSIVE" and s8["evidence"]["reason_code"] == "UNRESOLVED_VECTOR_NORM", s8


def test_variational_names_entries_that_are_not_finite():
    """certify_variational refuses an operator with an infinite entry as non-finite (before: 'not Hermitian' with a nan
    difference)."""
    from certify_ed.variational import certify_variational
    H = np.diag([1.0, 2.0, 3.0])
    H[2, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite entries"):
        certify_variational(H, np.eye(3)[:, :1])


def test_plan_counts_a_mapping_entry_as_a_tuple_entry():
    """The planner's narrow-input memory counts a float32 symmetry diagonal given in a mapping as it counts the tuple
    entry (before: tuples and lists only, while certify widens the mapping's diagonal too)."""
    from certify_ed.cost import narrow_input_mb
    spec, H, w, V = _chain6()
    Sd = np.arange(H.shape[0], dtype=np.float32)
    claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
    a = narrow_input_mb(claim, ce.Options(symmetries={"d": (Sd, None)}))
    b = narrow_input_mb(claim, ce.Options(symmetries={"d": {0: Sd, 1: None}}))
    assert a > 0 and a == b
