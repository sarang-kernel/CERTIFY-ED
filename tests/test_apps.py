"""Applications (certify_ed.apps) on their smallest instances, and the command line."""
import json
from fractions import Fraction

import numpy as np
import scipy.linalg as la

import certify_ed as ce
from conftest import needs_flint


@needs_flint
def test_aklt_star_certified_and_reproduces_published_eta():
    from certify_ed.apps.aklt import certify_lattice
    r = certify_lattice("star", log=lambda s: None)
    assert abs(r["eta_float"] - 0.1110430220) < 1e-9          # geometry check
    assert r["eta_published"] <= r["eta_upper"] < r["eta_published"] + 1e-5
    assert r["criterion_holds"] is True


@needs_flint
def test_aklt_threshold_below_eta_squared_is_not_certified():
    from certify_ed.apps.aklt import certify_lattice
    r = certify_lattice("star", thresholds=[Fraction(1232, 100000)], check_geometry=False, log=lambda s: None)
    assert r["eta_upper"] is None                               # 0.0123 < eta^2 = 0.012330...


@needs_flint
def test_aklt_kernel_lemma_star_exact():
    from certify_ed.apps.aklt import lattices
    from certify_ed.apps.aklt.kernel import kernel_check
    g, L, C, R = lattices.star()
    r = kernel_check(g, L + C, log=lambda s: None)
    assert r["kernel_equals_vbs_range"] and r["vbs_rank"] == 16


def test_pyrochlore_model_embedding_and_symmetries():
    from certify_ed.apps.pyrochlore.model import spec_for, couplings, symmetry_generators, K_SCALE
    c1, c3, c17 = couplings(381414, 817957)
    assert 2 * c1 + 8 * c3 + 2 * c17 == 1
    spec, _ = spec_for(381414, 817957, scale=K_SCALE)          # asserts exact binary64 couplings
    assert all(complex(c).imag == 0 and float(complex(c).real).is_integer() for t in spec.terms for c, _ in t.couplings)
    assert len(symmetry_generators(spec)) == 7


@needs_flint
def test_pyrochlore_spin1_printed_coupling_contradiction():
    from certify_ed.apps.pyrochlore.spin_one import printed_coupling_check
    r = printed_coupling_check()
    assert r["contradiction"] and r["margin"] > 4e-3


def test_cli_list_env_and_spectrum(tmp_path, capsys):
    from certify_ed.cli import main
    assert main(["list"]) == 0
    assert main(["env"]) == 0
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    Sz = np.array([bin(s).count("1") for s in range(H.shape[0])], dtype=float)
    import scipy.sparse as sp
    Hc = sp.csr_matrix(H)
    path = tmp_path / "m.npz"
    np.savez(path, H_data=Hc.data, H_indices=Hc.indices, H_indptr=Hc.indptr, H_shape=np.array(Hc.shape), Sz=Sz)
    out = tmp_path / "s.json"
    assert main(["spectrum", str(path), "--sectors", "Sz", "-o", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["complete"] and sum(r["count"] for r in res["intervals"]) == 256


@needs_flint
def test_cli_certify_with_exact_recheck(tmp_path):
    from certify_ed.cli import main
    spec = ce.heisenberg_chain(8)
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    k = 1
    while w[k] - w[k - 1] < 1e-8:
        k += 1
    claim = tmp_path / "claim.npz"
    ce.export_npz(str(claim), w[:k], V[:, :k], H)
    sj = tmp_path / "spec.json"
    sj.write_text(json.dumps(spec.to_dict()))
    assert main(["certify", str(claim), "--spec", str(sj), "--kind", "lowest", "--exact"]) == 0
    cert = ce.Certificate.load(str(tmp_path / "claim.certificate.json"))
    st = {s["id"]: s["status"] for s in cert.statements}
    assert st["S2.eigenpairs"] == "CERTIFIED" and st["S3.completeness"] == "CERTIFIED"
    assert st["S6.exact_recheck"] == "CERTIFIED"


# ------------------------------------------------------------------ geometry of every AKLT lattice
@needs_flint
def test_aklt_lattice_geometries_reproduce_published_eta():
    """Every reconstructed subgraph geometry reproduces Pomata & Wei's floating-point eta to 1e-9
    (decorated square: about 6 minutes, covered by E10)."""
    from certify_ed.apps.aklt import LATTICES, eta_float
    for name in ("star", "square_octagon_par", "square_octagon_perp", "cross", "decorated_honeycomb"):
        g, L, C, R = LATTICES[name][0]()
        assert not (set(L) & set(C) or set(C) & set(R) or set(L) & set(R)), name
        deg = [0] * len(g.z)
        for u, v in g.edges:
            deg[u] += 1; deg[v] += 1
        assert all(d <= z for d, z in zip(deg, g.z)), name
        assert abs(eta_float(g, L, C, R)["eta"] - LATTICES[name][2]) < 1e-9, name


@needs_flint
def test_aklt_decorated_square_mirror_is_an_automorphism():
    from certify_ed.apps.aklt import lattices
    g, L, C, R = lattices.decorated_square()
    p = lattices.decorated_square_reflection(g)
    E = {frozenset(e) for e in g.edges}
    assert all(frozenset((p[u], p[v])) in E for u, v in g.edges)
    assert [sorted(p[x] for x in X) for X in (L, C, R)] == [L, C, R]


# ------------------------------------------------------------------ pyrochlore S = 1 tools
def test_pyrochlore_spin1_all_singlet_block_is_the_ring():
    from certify_ed.apps.pyrochlore.spin_one import config_hamiltonian
    c1, c3 = 0.053466, 0.082726
    c17 = (1 - 2 * c1 - 8 * c3) / 2
    H, _ = config_hamiltonian(2, (0,) * 6, c1, c3, c17)
    ring = ce.build_bitwise(ce.spin1_heisenberg_chain(6)).toarray()
    e_block = la.eigvalsh(H.toarray())[0]
    e_ring = la.eigvalsh(ring)[0]
    assert abs(e_block - (-12 * c1 + c17 * e_ring)) < 1e-12


def test_pyrochlore_canonical_class_is_dihedral_invariant():
    from certify_ed.apps.pyrochlore.spin_one import canonical
    cfg = (0, 2, 4, 4, 0, 2)
    rots = [tuple(cfg[(i + r) % 6] for i in range(6)) for r in range(6)]
    assert len({canonical(c) for c in rots + [c[::-1] for c in rots]}) == 1


@needs_flint
def test_pyrochlore_spin1_printed_check_is_exact():
    from certify_ed.apps.pyrochlore.spin_one import printed_coupling_check
    r = printed_coupling_check()
    assert r["no_floating_point_assumption"] and r["exact_recheck"] == "CERTIFIED"
    assert r["E0_radius_exact"] <= r["E0_radius_standard_model"]


@needs_flint
def test_aklt_default_threshold_is_sharp():
    """The default threshold sits a relative 1e-9 above the floating-point eta^2: the certified bound
    agrees with the published eta to the 10th decimal."""
    from certify_ed.apps.aklt import certify_lattice, _threshold_above
    r = certify_lattice("star", log=lambda s: None)
    assert r["eta_float"] <= r["eta_upper"] < r["eta_float"] + 1e-9
    assert abs(r["eta_upper"] - r["eta_published"]) < 2e-10
    t = _threshold_above(0.1, 1e-9)
    assert float(t) >= 0.01 * (1 + 1e-9)
