"""Applications (certify_ed.apps) on their smallest instances, and the command line."""
import json
from fractions import Fraction

import numpy as np
import pytest
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
