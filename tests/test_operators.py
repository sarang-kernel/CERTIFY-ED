"""Operator ergonomics (roadmap C-15): Pauli strings, site-index shorthand and QuSpin-style operator lists
become the sparse operators S4 certifies, checked against matrices built explicitly from Kronecker products
on 4 sites (site 0 is the most significant digit; the default state order is descending, i.e. the row index
of basis state s is dim - 1 - s). Local matrices are written in the spec's digit convention (localops.py):
digit k <-> m = k - S, so digit 0 is spin down and sigma^z = diag(-1, +1), S^+ |0> = |1>."""
import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.operators import (OperatorSyntaxError, operator_from_static, parse_operator, parse_terms,
                                  resolve_correlation_request, resolve_observables)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, 1j], [-1j, 0]], dtype=complex)       # (S+ - S-)/i in the digit basis
SZ = np.array([[-1, 0], [0, 1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
SP = np.array([[0, 0], [1, 0]], dtype=complex)      # S^+ in the digit convention m = k - 1/2: |0> -> |1>
SM = SP.T.copy()
PAULI = {"I": I2, "X": SX, "Y": SY, "Z": SZ}
N = 4
DIM = 2 ** N
PERM = np.arange(DIM - 1, -1, -1)                   # ascending integer order -> descending state order


def kron_string(ops):
    """ops: list of 2x2 matrices for sites 0..N-1 (site 0 the most significant factor), in the spec's order."""
    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return M[PERM][:, PERM]


def spin_op(letter):
    return {"x": SX / 2, "y": SY / 2, "z": SZ / 2, "+": SP, "-": SM, "I": I2}[letter]


def site(letter, i, pauli=False):
    ops = [I2] * N
    ops[i] = PAULI[letter] if pauli else spin_op(letter)
    return kron_string(ops)


@pytest.fixture(scope="module")
def spec():
    return ce.heisenberg_chain(N)


def _dense(O):
    return O.toarray() if sp.issparse(O) else np.asarray(O)


def test_pauli_string_against_kron(spec):
    for word in ("XIZY", "IIII", "ZZZZ", "XYZI", "IXIX"):
        got = _dense(parse_operator(word, spec, name=word))
        exp = kron_string([PAULI[c] for c in word])
        assert np.allclose(got, exp, atol=0, rtol=0), word
    got = _dense(parse_operator("X0 Z2", spec, name="X0 Z2"))
    assert np.array_equal(got, kron_string([SX, I2, SZ, I2]))
    got = _dense(parse_operator("0.5 X0 Z2", spec, name="c"))
    assert np.array_equal(got, 0.5 * kron_string([SX, I2, SZ, I2]))


def test_site_index_shorthand_against_kron(spec):
    assert np.array_equal(_dense(parse_operator("Sz_0 Sz_3", spec)), site("z", 0) @ site("z", 3))
    assert np.array_equal(_dense(parse_operator("Sx_1", spec)), site("x", 1))
    assert np.array_equal(_dense(parse_operator("z_0 z_3", spec)), site("z", 0) @ site("z", 3))
    sym = _dense(parse_operator("0.5 S+_0 S-_3 + 0.5 S-_0 S+_3", spec))
    exp = 0.5 * (site("+", 0) @ site("-", 3) + site("-", 0) @ site("+", 3))
    assert np.allclose(sym, exp, atol=0, rtol=0)
    assert np.allclose(_dense(parse_operator("Sz_0 Sz_1 - 0.25 I", spec)),
                       site("z", 0) @ site("z", 1) - 0.25 * np.eye(DIM))
    assert np.array_equal(_dense(parse_operator({"letter": "z", "sites": [0, 3]}, spec)), site("z", 0) @ site("z", 3))
    assert np.array_equal(_dense(parse_operator({"op": "zz", "sites": [0, 3], "coeff": 2.0}, spec)),
                          2 * site("z", 0) @ site("z", 3))


def test_quspin_style_operator_list_against_kron(spec):
    static = [["zz", [[1.0, 0, 1], [0.5, 1, 2]]], ["x", [[0.3, 0]]], ["+-", [[0.25, 0, 3]]], ["-+", [[0.25, 0, 3]]]]
    got = _dense(operator_from_static(spec, static))
    exp = (site("z", 0) @ site("z", 1) + 0.5 * site("z", 1) @ site("z", 2) + 0.3 * site("x", 0)
           + 0.25 * site("+", 0) @ site("-", 3) + 0.25 * site("-", 0) @ site("+", 3))
    assert np.allclose(got, exp, atol=0, rtol=0)
    assert np.allclose(_dense(parse_operator(static, spec)), exp, atol=0, rtol=0)
    # the full Hamiltonian of the spec, written as its own operator list, is the S1 reference matrix
    H = parse_operator(spec.to_quspin_static(), spec)
    assert abs(H - ce.build_bitwise(spec)).max() == 0


def test_non_hermitian_products_are_refused_with_the_theorem6_message(spec):
    with pytest.raises(ValueError, match="not Hermitian"):
        parse_operator("S+_0 S-_3", spec, name="hop")
    with pytest.raises(ValueError, match="not Hermitian"):
        parse_operator("X0 Y0", spec, name="xy")           # sigma^x sigma^z on one site: i sigma^y... anti-Hermitian part
    with pytest.raises(ValueError, match="not Hermitian"):
        resolve_correlation_request(spec, "C", "S+_0 S-_3")
    # explicit opt-out for callers that want the raw operator
    O = parse_operator("S+_0 S-_3", spec, name="hop", hermitian=False)
    assert np.array_equal(_dense(O), site("+", 0) @ site("-", 3))


def test_syntax_errors_name_the_token(spec):
    for bad in ("Sq_0", "Sz_9", "XIZ", "XIZYI", "Sz_0 &", ""):
        with pytest.raises((OperatorSyntaxError, ValueError)):
            parse_terms(bad, spec)
    with pytest.raises(ValueError, match="spec"):
        parse_operator("Sz_0", None)
    spin1 = ce.spin1_heisenberg_chain(3)
    with pytest.raises(OperatorSyntaxError, match="spin-1/2"):
        parse_operator("XIZ", spin1)
    assert np.allclose(_dense(parse_operator("z_0 z_1", spin1)), _dense(ce.build_bitwise(
        ce.ModelSpec(3, spin1.local, [ce.Term("zz", [(1.0, (0, 1))])]))))


def test_pauli_convention_is_respected():
    tf = ce.tfim_chain(N, J=1.0, h=0.5)          # 'pauli' convention: the letters already are sigma
    got = _dense(parse_operator("ZZII", tf))
    assert np.array_equal(got, kron_string([SZ, SZ, I2, I2]))
    assert np.array_equal(_dense(parse_operator("z_0 z_1", tf)), got)   # same thing in this convention


def test_correlation_request_strings(spec):
    assert resolve_correlation_request(spec, "C", "Sz_0 Sz_3") == {"letter": "z", "sites": [0, 3]}
    assert resolve_correlation_request(spec, "C", "z_1 z_2 z_3") == {"letter": "z", "sites": [1, 2, 3]}
    d = {"letter": "x", "sites": [0, 1]}
    assert resolve_correlation_request(spec, "C", d) is d
    with pytest.raises(OperatorSyntaxError):
        resolve_correlation_request(spec, "C", "Sz_0 Sx_1")


def test_observables_and_correlations_from_text_in_certify(spec):
    """End to end: S4 certifies tr(P O) for operators given as text, and the values agree with the matrix path."""
    H = ce.build_bitwise(spec)
    w, V = la.eigh(H.toarray())
    obs_text = {"zz01": "Sz_0 Sz_1", "pauli": "ZZII", "hop": "0.5 S+_0 S-_1 + 0.5 S-_0 S+_1",
                "list": [["zz", [[1.0, 0, 1]]]]}
    obs_mat = {name: parse_operator(v, spec) for name, v in obs_text.items()}
    c_text = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec,
                        ce.Options.from_preset("labels", observables=obs_text,
                                               correlations={"Czz": "Sz_0 Sz_2", "Cd": {"letter": "z", "sites": [0, 2]}}))
    c_mat = ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec,
                       ce.Options.from_preset("labels", observables=obs_mat,
                                              correlations={"Czz": {"letter": "z", "sites": [0, 2]},
                                                            "Cd": {"letter": "z", "sites": [0, 2]}}))
    s4t = next(s for s in c_text.statements if s["id"] == "S4.eigenspaces")
    s4m = next(s for s in c_mat.statements if s["id"] == "S4.eigenspaces")
    assert s4t["status"] == s4m["status"] == "CERTIFIED"
    for ct, cm in zip(s4t["evidence"]["clusters"], s4m["evidence"]["clusters"]):
        for name in obs_text:
            assert ct["observables"][name]["value"] == cm["observables"][name]["value"]
            assert ct["observables"][name]["bound"] == cm["observables"][name]["bound"]
        assert ct["correlations"]["Czz"]["pairs"] == cm["correlations"]["Czz"]["pairs"]
        assert ct["correlations"]["Czz"]["pairs"] == ct["correlations"]["Cd"]["pairs"]
    # a Pauli ZZ expectation is four times the S^z S^z one (sigma = 2 S)
    cl = s4t["evidence"]["clusters"][0]
    assert np.isclose(cl["observables"]["pauli"]["value"], 4 * cl["observables"]["zz01"]["value"])
    # refusal: a non-Hermitian text observable is an error before any statement runs
    with pytest.raises(ValueError, match="not Hermitian"):
        ce.certify(ce.Claim(w, V, operator=H, kind="all"), spec, ce.Options(observables={"bad": "S+_0 S-_1"}))
    assert resolve_observables(spec, {}) == {}
