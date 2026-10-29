"""F-01: one-call QuSpin wrapper certify_ed.integrations.quspin.certify_hamiltonian.

The wrapper must give the same statuses as the direct claim_from_quspin + certify() route, keep the symmetry-block
audit (S1 through the basis isometry), handle unsorted eigsh output, and never certify a planted error."""
import numpy as np
import pytest

import certify_ed as ce
from certify_ed.integrations import quspin as cq

from conftest import needs_quspin

pytestmark = needs_quspin


def _status(cert, sid):
    return {s["id"].split(".")[0]: s["status"] for s in cert.statements}[sid]


def _statuses(cert):
    return [(s["id"], s["status"]) for s in cert.statements]


def _ham(static, basis, dtype=np.float64):
    from quspin.operators import hamiltonian
    return hamiltonian(static, [], basis=basis, dtype=dtype, check_herm=False, check_symm=False, check_pcon=False)


@pytest.fixture(scope="module")
def chain8():
    from quspin.basis import spin_basis_1d
    spec = ce.heisenberg_chain(8)
    static = spec.to_quspin_static()
    bases = {"full": spin_basis_1d(8, pauli=0),
             "u1": spin_basis_1d(8, Nup=4, pauli=0),
             "k": spin_basis_1d(8, Nup=4, kblock=0, pauli=0),
             "kp": spin_basis_1d(8, Nup=4, kblock=0, pblock=1, pauli=0)}
    out = {}
    for name, b in bases.items():
        H = _ham(static, b, np.complex128 if name == "k" else np.float64)
        w, V = H.eigh()
        out[name] = (b, H, w, V)
    return spec, static, out


@pytest.mark.parametrize("block", ["full", "u1", "k", "kp"])
def test_wrapper_matches_direct_route(chain8, block):
    spec, static, data = chain8
    b, H, w, V = data[block]
    k = 3
    cert = cq.certify_hamiltonian(H, w[:k], V[:, :k])
    claim, sp_ = ce.claim_from_quspin(H, b, static, w[:k], V[:, :k], kind="lowest")
    direct = ce.certify(claim, sp_)
    assert _statuses(cert) == _statuses(direct)
    assert _status(cert, "S1") == "CERTIFIED", cert.summary()
    assert _status(cert, "S2") == "CERTIFIED"
    prod = cert.body["claim"]["producer"]
    assert prod["name"] == "QuSpin"
    assert prod["entry_point"].endswith("certify_hamiltonian")
    assert prod["spec_source"] == "static list of H"


def test_symmetry_block_keeps_sector_audit(chain8):
    spec, static, data = chain8
    b, H, w, V = data["kp"]
    cert = cq.certify_hamiltonian(H, w[:2], V[:, :2])
    s1 = [s for s in cert.statements if s["id"].startswith("S1")][0]
    assert s1["status"] == "CERTIFIED"
    assert "sector_audit" in s1["evidence"], "S1 must carry the sector audit of the symmetry block"
    assert "isometry defect" in s1["summary"], s1["summary"]
    assert cert.body["claim"]["producer"]["symmetry_blocks"] == {"kblock": 0, "pblock": 1}


def test_unsorted_eigsh_output_is_sorted(chain8):
    spec, static, data = chain8
    b, H, w, V = data["u1"]
    order = np.array([2, 0, 1])
    cert = cq.certify_hamiltonian(H, w[order], V[:, order])
    assert cert.body["claim"]["producer"]["eigenvalues_sorted_by_wrapper"] is True
    assert _status(cert, "S2") == "CERTIFIED"
    assert _status(cert, "S3") == "CERTIFIED"
    assert np.allclose(cert.body["claim"]["eigenvalues"], w[:3])


def test_spec_override_is_used(chain8):
    spec, static, data = chain8
    b, H, w, V = data["u1"]
    cert = cq.certify_hamiltonian(H, w[:2], V[:, :2], spec=spec)
    assert cert.body["claim"]["producer"]["spec_source"] == "user"
    assert _status(cert, "S1") == "CERTIFIED"
    # a different model is detected: the static list says J=1, the spec J=1.3
    other = ce.heisenberg_chain(8, J=1.3)
    cert2 = cq.certify_hamiltonian(H, w[:2], V[:, :2], spec=other)
    assert _status(cert2, "S1") == "FAILED"


def test_eigsh_end_to_end_with_options(chain8):
    spec, static, data = chain8
    b, H, w, V = data["kp"]
    ws, Vs = H.eigsh(k=3, which="SA", v0=np.ones(b.Ns))
    cert = cq.certify_hamiltonian(H, ws, Vs, only=["S1", "S2"], narrowing_reason="test", description="eigsh run")
    assert _status(cert, "S1") == "CERTIFIED"
    assert _status(cert, "S2") == "CERTIFIED"
    assert cert.body["scope"]["narrowed"], "only= must narrow the scope"
    assert cert.body["claim"]["description"] == "eigsh run"


def test_negative_control_planted_eigenvalue_not_certified(chain8):
    spec, static, data = chain8
    b, H, w, V = data["kp"]
    bad = w[:3].copy()
    bad[1] += 1e-3
    cert = cq.certify_hamiltonian(H, bad, V[:, :3])
    assert _status(cert, "S2") != "CERTIFIED"


def test_negative_control_planted_vector_not_certified(chain8):
    spec, static, data = chain8
    b, H, w, V = data["u1"]
    X = V[:, :3].copy()
    X[:, 0] = V[:, 5]              # a correct vector of the wrong eigenvalue
    cert = cq.certify_hamiltonian(H, w[:3], X)
    assert _status(cert, "S2") != "CERTIFIED"


def test_dynamic_hamiltonian_refused(chain8):
    from quspin.operators import hamiltonian
    spec, static, data = chain8
    b = data["u1"][0]
    H = hamiltonian(static, [["z", [[1.0, 0]], lambda t: t, []]], basis=b, dtype=np.float64,
                    check_herm=False, check_symm=False, check_pcon=False)
    with pytest.raises(ValueError, match="time-dependent"):
        cq.certify_hamiltonian(H, np.zeros(1))


def test_bad_inputs_raise(chain8):
    spec, static, data = chain8
    b, H, w, V = data["u1"]
    with pytest.raises(TypeError, match="unknown keyword"):
        cq.certify_hamiltonian(H, w[:2], V[:, :2], nonsense=1)
    with pytest.raises(ValueError, match="columns"):
        cq.certify_hamiltonian(H, w[:2], V[:, :3])
    with pytest.raises(TypeError, match="QuSpin hamiltonian"):
        cq.certify_hamiltonian(np.eye(3), np.ones(1))
    with pytest.raises(TypeError, match="spec must be"):
        cq.certify_hamiltonian(H, w[:2], V[:, :2], spec="heisenberg")


def test_static_list_of_round_trips(chain8):
    spec, static, data = chain8
    H = data["full"][1]
    assert cq.static_list_of(H) == [[op, [list(c) for c in cps]] for op, cps in static]
