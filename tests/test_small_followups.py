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
from conftest import eigh_dense, needs_flint


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


@needs_flint
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

    from certify_ed.basis import cyclic_isometry, translation
    from certify_ed.exact_sectors import exact_recheck_sector_lowest
    st = ce.u1_state_set(spec, 3)
    T = translation(spec)
    V0 = cyclic_isometry(st, T, 0)
    ws, Ys = eigh_dense((V0.conj().T @ ce.build_bitwise(spec, st) @ V0).toarray())
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
        ws, Ys = eigh_dense(Hs.toarray())
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
    assert "dropped_imaginary_parts" not in res        # a non-finite part is not a dropped one


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


def test_other_entry_points_see_a_non_finite_imaginary_part(tmp_path):
    """certified_sweep with a solver that returns w0 + inf j certifies no S2 at that point; exact_eigenpair_radii gives
    an infinite radius and lists the index; enclose and write_bundle refuse the value and corroborate is INCONCLUSIVE
    (before: all of them used w0)."""
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
        write_bundle(str(tmp_path), bad, V[:, :1], H)


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

    class OnlyIter:
        def __iter__(self):
            return iter((Sd, None))
    # entries read as their items (a values view, a class with __iter__ only) count too; an iterator is not read
    for ent in ({"s": Sd, "l": None}.values(), OnlyIter()):
        assert narrow_input_mb(claim, ce.Options(symmetries={"d": ent})) == a
        assert narrow_input_mb(claim, ce.Options(approximate_symmetries={"d": ent})) == a
    it = iter((Sd, None))
    assert narrow_input_mb(claim, ce.Options(symmetries={"d": it})) == 0.0 and next(it) is Sd


def test_dropped_imaginary_parts_are_listed_under_not_verified():
    """Finite nonzero imaginary parts (the statements concern the real parts, as in 3.x) are listed under not_verified,
    so that the report shows them (before: only in the claim record); real claimed values add no such line."""
    spec, H, w, V = _chain6()
    lam = w[:3].astype(complex)
    lam[1], lam[2] = complex(w[1], 0.25), complex(w[2], -0.5)       # the largest magnitude from a negative part
    cert = ce.certify(ce.Claim(eigenvalues=lam, eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 2, "max_abs": 0.5}
    lines = [s for s in cert.body["not_verified"] if "imaginary parts" in s]
    assert len(lines) == 1 and "2 claimed value(s)" in lines[0] and "0.5" in lines[0], lines
    assert lines[0] in cert.report()
    real = ce.certify(ce.Claim(eigenvalues=w[:3], eigenvectors=V[:, :3], operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    assert not any("imaginary" in s for s in real.body["not_verified"])


def test_sweep_and_composition_record_dropped_imaginary_parts():
    """certified_sweep passes complex solver values to the claim as given, so the certificate records the dropped part;
    the sector composition records it in the sector's record (before: dropped with no record at both)."""

    from certify_ed.compose import SectorResult, certify_low_energy
    from certify_ed.sweep import certified_sweep

    def solver(Hc, k):
        ww, VV = np.linalg.eigh(Hc.toarray())
        ww = ww.astype(complex)
        ww[0] = complex(ww[0].real, 0.25)
        return ww[:k], VV[:, :k]
    sweep = certified_sweep(lambda c: ce.heisenberg_chain(4, J=c), [1.0, 1.1], k=1, solver=solver)
    cert = sweep["_certificates"][0]
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
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
        ws, Ys = eigh_dense(Hs.toarray())
        sel = np.where(ws < E_cut)[0]
        ev = ws[sel].astype(complex)
        if sel.size and hit is None:
            hit = f"nup={nup}"
            ev[0] = complex(ev[0].real, -0.125)
        secs.append(SectorResult(f"nup={nup}", Vs, Hs, ev, Ys[:, sel]))
    res = certify_low_energy(spec, secs, E_cut)
    rec = next(p for p in res["sectors"] if p["label"] == hit)
    assert rec["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.125}
    assert all("dropped_imaginary_parts" not in p for p in res["sectors"] if p["label"] != hit)
    # and once for the composition, in its record and its summary
    assert res["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.125, "sectors": [hit]}
    assert "the imaginary parts of 1 claimed value(s) in 1 sector(s), largest in magnitude 0.125" in res["summary"]


def test_certify_hamiltonian_records_dropped_imaginary_parts():
    """The QuSpin entry point passes complex values to the claim as given (sorted by their real parts), so the record
    is made there too (before: S2 CERTIFIED with no record)."""
    pytest.importorskip("quspin")
    from quspin.basis import spin_basis_1d
    from quspin.operators import hamiltonian

    from certify_ed.integrations import quspin as cq
    spec = ce.heisenberg_chain(6)
    Hq = hamiltonian(spec.to_quspin_static(), [], basis=spin_basis_1d(6, pauli=0), dtype=np.float64, check_herm=False,
                     check_symm=False, check_pcon=False)
    w, V = Hq.eigh()
    lam = w[:3].astype(complex)
    lam[1] = complex(w[1], 0.5)
    order = [2, 0, 1]                            # unsorted: the wrapper sorts values and vectors together
    cert = cq.certify_hamiltonian(Hq, lam[order], V[:, :3][:, order], kind="lowest", seed=1)
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.5}
    assert any("imaginary parts of 1 claimed value(s)" in s for s in cert.body["not_verified"])


def test_s8_has_no_code_where_s2_has_none():
    """A zero claimed vector has an infinite radius by Theorem 1, not through an overflow: S2 gives no reason code, and
    S8 now gives none either (on the previous round: NONFINITE_BOUND)."""
    spec, H, w, V = _chain6()
    X = V[:, :4].copy()
    X[:, 2] = 0.0
    cert = ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=X, operator=H, kind="lowest"), spec,
                      ce.Options(seed=1))
    s2, s8 = cert.statement("S2.eigenpairs"), cert.statement("S8.degeneracy")
    assert s2["status"] == "INCONCLUSIVE" and s2["evidence"].get("reason_code") is None, s2
    assert s8["status"] == "INCONCLUSIVE" and s8["evidence"].get("reason_code") is None, s8


@pytest.mark.parametrize("entry", ["no key 1", "no key 0", None, 5], ids=["no-key-1", "no-key-0", "none", "number"])
def test_symmetry_entry_without_its_pair_is_refused_with_the_reason(entry):
    """A mapping without the key 0 or 1, and an entry without a length, are refused with the message of a sequence of
    the wrong length (before: a bare KeyError or TypeError)."""
    spec, H, w, V = _chain6()
    full = ce.full_state_set(spec)
    Sd = digit_sum(full.states_in_order().astype(np.int64), spec.q, spec.n_sites).astype(float)
    ent = {0: Sd} if entry == "no key 1" else {1: None, 2: Sd} if entry == "no key 0" else entry
    with pytest.raises(ValueError, match=r"must be \(S_diagonal, lattice\)"):
        ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest"), spec,
                   ce.Options(seed=1, symmetries={"d": ent}))


def test_real_nan_at_exact_eigenpair_radii_and_write_bundle(tmp_path):
    """A real nan claimed value: exact_eigenpair_radii gives an infinite radius and lists the index (before: a ValueError
    from the rational conversion), write_bundle refuses it (before: the bundle was written and its readers refused it)."""
    from certify_ed.checker import exact_eigenpair_radii
    from certify_ed.io.bundle import write_bundle
    spec, H, w, V = _chain6()
    bad = np.array([np.nan])
    r = exact_eigenpair_radii(spec, bad, V[:, :1])
    assert r["radii"] == [np.inf] and r["non_finite_values"] == [0]
    # finite claimed values: the record has the keys of main (no empty non_finite_values in S6's evidence)
    ok = exact_eigenpair_radii(spec, w[:1], V[:, :1])
    assert set(ok) == {"radii", "hermitian", "den", "nnz", "zero_test_vectors"} and np.isfinite(ok["radii"][0])
    with pytest.raises(ValueError, match="finite"):
        write_bundle(str(tmp_path), bad, V[:, :1], H)


def test_remaining_conversion_sites_see_a_non_finite_imaginary_part():
    """The conversion sites the other tests do not reach: enclosure.residuals (no finite upper bound), the Ozaki
    residuals (not applicable), and (the two tests below) the interval-arithmetic radii and the ALPS reader; and
    corroborate is INCONCLUSIVE for an infinite real claimed value (before: CORROBORATED with an infinite tolerance)."""
    from certify_ed.corroborate import corroborate
    from certify_ed.enclosure import OperatorAdapter, residuals
    from certify_ed.ozaki import NotApplicable, ozaki_residual, residual_report
    spec, H, w, V = _chain6()
    bad = np.array([complex(w[0], np.inf)])
    rep = residuals(OperatorAdapter(H), bad, V[:, :1])
    assert not np.isfinite(rep.upper[0]) and not np.isfinite(rep.rel_bound[0])
    with pytest.raises(NotApplicable):
        ozaki_residual(H, bad, V[:, :1])
    with pytest.raises(NotApplicable):
        residual_report(H, bad, V[:, :1])
    for v in (np.inf, -np.inf):
        cor = corroborate(H, np.array([v, w[1]]))
        assert cor["status"] == "INCONCLUSIVE" and cor["non_finite_values"] == [0], cor
    assert corroborate(H, w[:2])["status"] == "CORROBORATED"


def test_interval_radii_see_a_non_finite_imaginary_part():
    """The interval-arithmetic residual radii (python-flint) of a value with an infinite imaginary part: not finite."""
    pytest.importorskip("flint")
    from certify_ed.rigorous import arb_residual_radii
    spec, H, w, V = _chain6()
    assert not np.isfinite(arb_residual_radii(H, np.array([complex(w[0], np.inf)]), V[:, :1])[0])


def test_alps_reader_refuses_a_non_finite_imaginary_part(tmp_path):
    """The ALPS reader (h5py) refuses energies with an infinite imaginary part."""
    h5py = pytest.importorskip("h5py")
    from certify_ed.io.alps import read_alps_spectrum
    from certify_ed.io.formats import FormatError
    spec, H, w, V = _chain6()
    path = str(tmp_path / "spectrum.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("spectrum/sectors/0/energies", data=np.array([complex(w[0], np.inf), w[1]]))
    with pytest.raises(FormatError, match="non-finite"):
        read_alps_spectrum(path, with_vectors=False)


def test_object_array_of_complex_scalars_is_masked_and_recorded():
    """An object array holding numpy complex scalars: the non-finite imaginary part is seen (nan) and a finite one is
    recorded (before: the float conversion read the real parts, with no mask and no record)."""
    from certify_ed.precision import dropped_imaginary_parts
    obj = np.array([np.complex128(complex(1.0, np.inf)), np.complex128(complex(2.0, -0.5))], dtype=object)
    out = claimed_reals(obj)
    assert np.isnan(out[0]) and out[1] == 2.0
    assert dropped_imaginary_parts(obj) == {"count": 1, "max_abs": 0.5}
    assert dropped_imaginary_parts(np.array([1.0, 2.0], dtype=object)) is None


@pytest.mark.parametrize("form", ["values-view", "iterable-only", "three-items"])
def test_symmetry_entry_forms_without_indexing(form):
    """An entry that can be iterated but not indexed (a dict.values() view, a class with __iter__ only) is read as its
    items, as 3.x unpacked it (on the previous round: TypeError); a sequence of three items is refused with the
    message of a wrong length."""
    spec, H, w, V = _chain6()
    full = ce.full_state_set(spec)
    Sd = digit_sum(full.states_in_order().astype(np.int64), spec.q, spec.n_sites).astype(float)

    class OnlyIter:
        def __init__(self, *items):
            self.items = items

        def __iter__(self):
            return iter(self.items)

    def run(ent):
        return ce.certify(ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest"), spec,
                          ce.Options(seed=1, symmetries={"d": ent})).statement("S4.eigenspaces")
    if form == "three-items":
        with pytest.raises(ValueError, match=r"must be \(S_diagonal, lattice\)"):
            run((Sd, None, None))
        return
    ent = {"s": Sd, "l": None}.values() if form == "values-view" else OnlyIter(Sd, None)
    ref, got = run((Sd, None)), run(ent)
    assert got["status"] == ref["status"] == "CERTIFIED" and got["summary"] == ref["summary"]


def test_malformed_symmetry_entry_is_refused_whatever_s4_does():
    """A malformed entry of Options.symmetries is refused when certify starts, also when S4 never reads it (a claim
    with only S2 selected, or no cluster to check): before, it was refused only when S4 reached its loop."""
    spec, H, w, V = _chain6()
    for only in (["S2.eigenpairs"], None):
        with pytest.raises(ValueError, match=r"must be \(S_diagonal, lattice\)"):
            ce.certify(ce.Claim(eigenvalues=w[:1], eigenvectors=V[:, :1], operator=H, kind="lowest"), spec,
                       ce.Options(seed=1, symmetries={"d": (None, None, None)}, only=only))


def test_wrappers_record_dropped_parts_of_an_object_array():
    """certified_sweep (and certify_hamiltonian, by the same rule) passes an object array of numpy complex scalars to
    the claim as given, so the record is made (on the previous round: only complex arrays were passed on)."""
    from certify_ed.sweep import certified_sweep

    def solver(Hc, k):
        ww, VV = np.linalg.eigh(Hc.toarray())
        obj = np.array([np.complex128(complex(v, 0.0)) for v in ww], dtype=object)
        obj[0] = np.complex128(complex(ww[0], -0.375))
        return obj[:k], VV[:, :k]
    cert = certified_sweep(lambda c: ce.heisenberg_chain(4, J=c), [1.0, 1.1], k=1, solver=solver)["_certificates"][0]
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.375}


def test_corroborate_tolerance_of_a_non_finite_claim_is_schema_valid():
    """corroborate's record for a non-finite claimed value has tolerance nan (none applies), which the certificate
    schema's number_or_special form accepts; None, written before, is not valid under it."""
    import math

    jsonschema = pytest.importorskip("jsonschema")
    from certify_ed.certificate_v2 import _jsonable
    from certify_ed.corroborate import corroborate
    from certify_ed.schema import CERTIFICATE_SCHEMA
    spec, H, w, V = _chain6()
    sub = {"$defs": CERTIFICATE_SCHEMA["$defs"], "$ref": "#/$defs/number_or_special"}
    for v in (np.nan, np.inf, complex(w[0], np.inf)):
        cor = corroborate(H, np.array([v, w[1]]))
        assert cor["status"] == "INCONCLUSIVE" and math.isnan(cor["tolerance"]), cor
        jsonschema.validate(_jsonable(cor["tolerance"]), sub)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(None, sub)


@pytest.mark.parametrize("form", ["tuple", "iterable-only", "no key 0", "no key 1", "None", "number", "string",
                                  "bare sparse matrix"])
def test_malformed_approximate_symmetry_entry_is_refused_where_it_is_read(form):
    """A malformed entry of Options.approximate_symmetries (empty, a mapping without the keys it is read by, None, a
    number, a string, a bare sparse matrix) is refused with the form of an entry where S4 reads it (before: an IndexError, KeyError or
    TypeError there); certify passes such an entry on as given, so a run whose statements do not read it certifies as
    before (tests/test_input_unchanged.py)."""
    class _Empty:
        def __iter__(self):
            return iter(())
    spec, H, w, V = _chain6()
    full = ce.full_state_set(spec)
    Sd = digit_sum(full.states_in_order().astype(np.int64), spec.q, spec.n_sites).astype(float)
    entry = {"tuple": (), "iterable-only": _Empty(), "no key 0": {1: None}, "no key 1": {0: Sd, 2: None},
             "None": None, "number": 3, "string": "Sz", "bare sparse matrix": sp.diags(Sd).tocsr()}[form]
    claim = ce.Claim(eigenvalues=w[:4], eigenvectors=V[:, :4], operator=H, kind="lowest")
    with pytest.raises(ValueError, match=r"approximate_symmetries\[m\] must be \(S, lattice"):
        ce.certify(claim, spec, ce.Options(seed=1, approximate_symmetries={"m": entry}))
    only_s2 = ce.certify(claim, spec, ce.Options(seed=1, only=["S2.eigenpairs"], approximate_symmetries={"m": entry}))
    assert only_s2.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    # the forms that are read: a one-item tuple, a values view, a mapping with a further key
    for ok in ((Sd,), {"s": Sd, "l": None}.values(), {0: Sd, 1: None, 2: None, "note": 1}):
        s4 = ce.certify(claim, spec, ce.Options(seed=1, approximate_symmetries={"m": ok})).statement("S4.eigenspaces")
        assert s4["status"] == "CERTIFIED"


def test_claim_from_file_and_spectrum_invariants_read_claimed_values_as_certify_does(tmp_path):
    """claim_from_file keeps complex claimed values, so certify records a dropped imaginary part (before: converted to
    real with no record); spectrum_invariants of a value with a non-finite imaginary part compares nothing (before: the
    real part alone was compared)."""
    import warnings

    from certify_ed.importers import claim_from_file
    from certify_ed.invariants import compare_invariants, spec_invariants, spectrum_invariants
    spec, H, w, V = _chain6()
    ev = w[:1].astype(complex)
    ev[0] = complex(w[0], 0.25)
    np.save(str(tmp_path / "ev.npy"), ev)
    np.save(str(tmp_path / "vec.npy"), V[:, :1])
    claim = claim_from_file(str(tmp_path / "ev.npy"), str(tmp_path / "vec.npy"), kind="subset")
    claim.operator = H
    cert = ce.certify(claim, spec, ce.Options(seed=1))
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    assert cert.body["claim"]["dropped_imaginary_parts"] == {"count": 1, "max_abs": 0.25}
    np.save(str(tmp_path / "real.npy"), w[:1])
    assert claim_from_file(str(tmp_path / "real.npy")).eigenvalues.dtype == np.float64
    np.save(str(tmp_path / "zero_imag.npy"), w[:2].astype(complex))     # no imaginary part to record: float, as at 3.x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", getattr(np, "exceptions", np).ComplexWarning)
        zc = claim_from_file(str(tmp_path / "zero_imag.npy"))
    assert zc.eigenvalues.dtype == np.float64 and np.array_equal(zc.eigenvalues, w[:2])
    full = np.linalg.eigvalsh(H.toarray())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = full.astype(complex)
        bad[0] = complex(full[0], np.inf)
        cmp_bad = compare_invariants(spec_invariants(spec), spectrum_invariants(bad))
    assert cmp_bad["consistent"] is None
    assert compare_invariants(spec_invariants(spec), spectrum_invariants(full))["consistent"] is True
    assert spectrum_invariants(full) == spectrum_invariants(list(full))


@pytest.mark.parametrize("fn", ["variational", "moments"])
def test_spare_capacity_of_a_csr_operator_is_not_read(fn):
    """A canonical CSR operator with spare capacity past nnz holding nan (preallocated buffers): the finiteness check
    reads the values held, as precision does, so it is accepted with the record of the pruned matrix (before this
    round: refused as non-finite; at main: accepted)."""
    from certify_ed.moments import certified_moments
    from certify_ed.variational import certify_variational
    A = sp.csr_matrix(np.diag([1.0, 2.0]))
    A.data, A.indices = np.r_[A.data, np.nan], np.r_[A.indices, 0]
    assert A.nnz == 2 and A.data.size == 3
    B = sp.csr_matrix(np.diag([1.0, 2.0]))
    if fn == "variational":
        psi = np.array([1.0, 0.1])
        ra, rb = certify_variational(A, psi), certify_variational(B, psi)
        assert [s["summary"] for s in ra["statements"]] == [s["summary"] for s in rb["statements"]]
    else:
        x = np.eye(2)[:, :1]
        ra = certified_moments(A, B, x, 1e-12, 1e-12, K=1, sum_rules=False)
        rb = certified_moments(B, B, x, 1e-12, 1e-12, K=1, sum_rules=False)
        assert ra["moments"] == rb["moments"]
    with pytest.raises(ValueError, match="non-finite entries"):        # a held nan is still refused
        C = sp.csr_matrix(np.diag([1.0, 2.0]))
        C.data[1] = np.nan
        (certify_variational(C, np.array([1.0, 0.1])) if fn == "variational" else
         certified_moments(C, B, np.eye(2)[:, :1], 1e-12, 1e-12, K=1, sum_rules=False))


@pytest.mark.parametrize("cls", ["matrix", "array"])
@pytest.mark.parametrize("fmt", ["csr", "csc", "bsr", "coo", "dia", "lil", "dok"])
def test_stored_values_in_every_format(fmt, cls):
    """precision._stored_values reads the values a sparse matrix holds, in every format and class: a held nan is seen,
    and values a format keeps without holding them (spare capacity past nnz in CSR, CSC and BSR; DIA padding) are not.
    certify_variational refuses an operator with a held nan in every format and accepts the one with a nan kept but
    not held (the round-1 probe of these formats read A.data)."""
    from certify_ed.precision import _stored_values
    from certify_ed.variational import certify_variational
    if not hasattr(sp, f"{fmt}_{cls}"):
        pytest.skip(f"scipy has no {fmt}_{cls}")
    cons = getattr(sp, f"{fmt}_{cls}")

    def make(M, **kw):                            # BSR in 1 x 1 blocks, so that it holds the entries of the others
        return cons(M, blocksize=(1, 1), **kw) if fmt == "bsr" and not isinstance(M, tuple) else cons(M, **kw)
    D = np.diag([1.0, 2.0, 3.0]) + np.diag([0.5, 0.5], 1) + np.diag([0.5, 0.5], -1)
    psi = np.array([1.0, 0.1, 0.0])
    A = make(D)
    assert sorted(_stored_values(A).ravel().tolist()) == sorted(D[D != 0].tolist())
    Dn = D.copy()
    Dn[1, 1] = np.nan
    An = make(Dn)
    assert int(np.isnan(_stored_values(An)).sum()) == 1
    with pytest.raises(ValueError, match="non-finite entries"):
        certify_variational(An, psi)
    if fmt in ("csr", "csc", "bsr", "dia"):
        K = make(D)
        if fmt == "dia":                          # offset 1 stores its entries in columns 1.., column 0 is padding
            d = K.data.copy()
            d[list(K.offsets).index(1), 0] = np.nan
            K = make((d, K.offsets.copy()), shape=K.shape)
        else:                                     # one more entry (block) past indptr[-1]
            K.data = np.concatenate([K.data, np.full((1,) + K.data.shape[1:], np.nan)])
            K.indices = np.r_[K.indices, 0].astype(K.indices.dtype)
        assert int(np.isnan(_stored_values(K)).sum()) == 0
        ref = certify_variational(make(D), psi)
        got = certify_variational(K, psi)
        assert [s["summary"] for s in got["statements"]] == [s["summary"] for s in ref["statements"]]


@pytest.mark.parametrize("form", ["real object array", "complex, zero imaginary parts"])
def test_wrappers_pass_a_real_object_array_as_floats(monkeypatch, form):
    """Values without an imaginary part (a real-valued object array of Python floats, a complex array whose imaginary
    parts are all zero) carry nothing to record: the sweep passes their float form to the claim, as main did; values
    with a nonzero imaginary part, finite or not, are passed as given."""
    import warnings

    from certify_ed.precision import carries_imaginary_parts
    from certify_ed.sweep import certified_sweep
    assert not carries_imaginary_parts(np.array([1.0, 2.0], dtype=object))
    assert carries_imaginary_parts(np.array([np.complex128(1 + 0.5j), 2.0], dtype=object))
    assert not carries_imaginary_parts(np.array([1.0 + 0j, 2.0]))
    assert carries_imaginary_parts(np.array([1.0 + 0.5j])) and carries_imaginary_parts(np.array([complex(1.0, np.nan)]))
    assert not carries_imaginary_parts(np.array([1.0, 2.0]))

    def solver(Hc, k):
        ww, VV = np.linalg.eigh(Hc.toarray())
        if form == "real object array":
            return np.array([float(v) for v in ww[:k]], dtype=object), VV[:, :k]
        return ww[:k].astype(complex), VV[:, :k]
    import sys
    cmod = sys.modules["certify_ed.certify"]                    # the module (the package exports the function)
    seen, Claim0 = [], cmod.Claim

    def spy(eigenvalues, *a, **kw):
        seen.append(np.asarray(eigenvalues).dtype.kind)
        return Claim0(eigenvalues, *a, **kw)
    monkeypatch.setattr(cmod, "Claim", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", getattr(np, "exceptions", np).ComplexWarning)   # the 3.x conversion warns
        sweep = certified_sweep(lambda c: ce.heisenberg_chain(4, J=c), [1.0, 1.1], k=1, solver=solver)
    assert seen and set(seen) == {"f"}, seen                   # the float form reached the claim
    cert = sweep["_certificates"][0]
    assert cert.statement("S2.eigenpairs")["status"] == "CERTIFIED"
    assert "dropped_imaginary_parts" not in cert.body["claim"]
