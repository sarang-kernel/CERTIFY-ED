"""Registry of every public function of the modules that take stored matrices or vectors (the methods record's
entry-point table, kept here as code). Each listed function is called with random float32 and complex64 data whose
products round in single precision, with a sparse matrix that stores a repeated entry whose single-precision sum
rounds, and its result is compared bit for bit with the result for the exact binary64 copy of the same data. Every
other public name of those modules is listed in EXEMPT with the reason it takes no stored matrix or vector (or why
its output is the input as given). A public name in neither list fails test_registry_is_complete, so a new entry
point has to be added here, with a call, before it can be released."""
import importlib
import inspect
import json
import os
import tempfile
from dataclasses import is_dataclass

import numpy as np
import pytest
import scipy.sparse as sp

import certify_ed as ce
from certify_ed.certify import Claim, Options, certify

MODULES = ["certify_ed.basis", "certify_ed.certificate_v2", "certify_ed.certify", "certify_ed.checker",
           "certify_ed.compat.symveig", "certify_ed.compose", "certify_ed.construct", "certify_ed.correlations",
           "certify_ed.davis_kahan", "certify_ed.eigenspace", "certify_ed.enclosure", "certify_ed.entanglement",
           "certify_ed.exact_sectors", "certify_ed.fp", "certify_ed.generalized", "certify_ed.importers",
           "certify_ed.inertia", "certify_ed.invariants", "certify_ed.io.bundle", "certify_ed.io.formats",
           "certify_ed.ldl_residual", "certify_ed.moments", "certify_ed.nonhermitian", "certify_ed.ozaki",
           "certify_ed.precision", "certify_ed.rigorous", "certify_ed.sectors", "certify_ed.spectrum",
           "certify_ed.sweep", "certify_ed.variational"]


# every other module of the package, with the reason it is not in MODULES; test_every_module_is_listed fails for a
# module in none of MODULES, EXEMPT_MODULES (exact module names: a package key is its __init__ only) and EXEMPT_TREES
EXEMPT_MODULES = {
    "certify_ed": "package namespace: re-exports names of the listed modules",
    "certify_ed.api": "quick() wraps certify() (registered), which converts its inputs",
    "certify_ed.cli": "command-line front end: reads files with io.formats and importers and calls certify() and the "
                      "record functions (registered); one PrecisionError handler (tests/test_single_precision.py)",
    "certify_ed.compat": "package namespace (compat.symveig is listed)",
    "certify_ed.conformance": "checks producer bundles through io.bundle and certify() (registered)",
    "certify_ed.corroborate": "S5 corroboration: independent eigensolvers as evidence, no rigorous bound; certify() "
                              "passes it the converted operator",
    "certify_ed.cost": "time and memory model from sizes and dtypes; narrow_input_mb and the plan are tested in "
                       "tests/test_single_precision.py",
    "certify_ed.crossings": "works on certificates",
    "certify_ed.degeneracy": "degeneracy_counts hands the matrix to inertia.inertia (registered), which converts it; "
                             "its own arithmetic is on the enclosure's binary64 radii",
    "certify_ed.diagnostics": "text for statements",
    "certify_ed.exact_degeneracy": "exact arithmetic on the ExactMatrix built from the spec",
    "certify_ed.external": "runs an external verifier on a certificate file",
    "certify_ed.extrapolation": "works on certificates",
    "certify_ed.gaps": "works on the enclosure's binary64 radii",
    "certify_ed.gpu": "DeviceCSR refuses data that are not float64 or complex128; OperatorAdapter passes it the "
                      "converted matrix (D-05; no device here)",
    "certify_ed.integrations": "package namespace",
    "certify_ed.integrations.pauli": "builds model specs; no stored matrix",
    "certify_ed.integrations.quspin": "wraps certify(); checks the solver's eigenvalues (check_values)",
    "certify_ed.io": "package namespace (io.formats and io.bundle are listed)",
    "certify_ed.io.alps": "parses the producer's files into float64 arrays and builds a claim for certify()",
    "certify_ed.io.diagham": "parses the producer's files into float64 arrays (binary '<f8') and builds a claim",
    "certify_ed.io.hphi": "parses the producer's files into float64 / complex128 arrays ('<c16') and builds a claim",
    "certify_ed.io.spinpack": "parses the producer's text files into float64 arrays and builds a claim",
    "certify_ed.ivl": "dyadic interval arithmetic on Python integers (exact)",
    "certify_ed.levelstats": "works on certificates",
    "certify_ed.localops": "local operator definitions of the spec",
    "certify_ed.matrixfree": "applies the spec operator matrix-free (its own binary64 arithmetic)",
    "certify_ed.operators": "parses operator expressions into binary64 sparse matrices of the spec",
    "certify_ed.parallel": "runs certify() (registered) in workers",
    "certify_ed.presets": "statement selection",
    "certify_ed.provenance": "source hashes",
    "certify_ed.recorder": "records producer calls and hands their arrays to certify() (registered)",
    "certify_ed.report": "formats certificates",
    "certify_ed.schema": "JSON schemas",
    "certify_ed.scope": "scope block of a certificate",
    "certify_ed.spec": "model specifications",
    "certify_ed.su2": "SU(2) structure of the spec",
    "certify_ed.testing": "random specs for the tests",
    "certify_ed.thermo": "dyadic interval arithmetic on enclosures",
    "certify_ed.viewer": "renders certificates",
}

# subtrees exempt as a whole: a key covers every module below it
EXEMPT_TREES = {
    "certify_ed.apps": "application studies (AKLT, pyrochlore) that build their own binary64 operators from models; "
                       "not imported by the package (no certify_ed/apps/__init__.py)",
}

N, K = 8, 2
WIDE = {"float32": np.float64, "complex64": np.complex128}
DROP = ("input_widening", "record_sha256", "environment", "timing", "producer", "bundle", "files", "path",
        "op",       # OperatorAdapter.op: the caller's object as given
        "hash")     # certificate hashes cover the body, which records the conversion


# ------------------------------------------------------------------------------------------------------------ data
class Data:
    """One precision of the test data; `wide` is the exact binary64 copy of `narrow`."""


def _make(dtype):
    rng = np.random.default_rng(20261119)
    cplx = dtype == "complex64"
    wd = WIDE[dtype]

    def rnd(*shape):
        a = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
        return a.astype(dtype)

    def herm():
        U = np.triu(rnd(N, N), 1)
        return U + U.conj().T + np.diag(rng.standard_normal(N).astype(np.float32)).astype(dtype)   # exact, Hermitian

    def repeated(M):
        """COO of M with (0, 0) stored once more, as t: the matrix is M + t e0 e0^T, summed in binary64."""
        C = sp.coo_matrix(M)
        t = np.float32(np.spacing(np.float32(abs(M[0, 0].real))) * 0.375)
        assert np.float32(M[0, 0].real) + t != np.float64(M[0, 0].real) + np.float64(t)   # single precision rounds
        data = np.concatenate([C.data, np.array([t], dtype=dtype)])
        return sp.coo_matrix((data, (np.r_[C.row, 0], np.r_[C.col, 0])), shape=M.shape)

    A = herm()
    S = repeated(A)
    Aeff = sp.coo_matrix((S.data.astype(wd), (S.row, S.col)), shape=S.shape).toarray()   # summed in binary64
    w, V = np.linalg.eigh(Aeff)
    O = repeated(herm())
    Bm = np.diag((2.0 + np.abs(rng.standard_normal(N))).astype(np.float32)).astype(dtype)
    Bo = np.triu(rnd(N, N) * np.asarray(0.05, dtype=dtype), 1)
    B = Bm + Bo + Bo.conj().T
    import scipy.linalg as la
    wg, Vg = la.eigh(Aeff, B.astype(wd))
    Anh = rnd(N, N)
    wn, Vn = np.linalg.eig(Anh.astype(wd))
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)) + (1j * rng.standard_normal((N, N)) if cplx else 0))
    Viso = Q.astype(dtype)
    Hs = Viso.astype(wd).conj().T @ Aeff @ Viso.astype(wd)
    Hs = ((Hs + Hs.conj().T) / 2).astype(dtype)
    spec = ce.heisenberg_chain(3)
    full = ce.full_state_set(spec)
    idx = full.index_of(ce.u1_state_set(spec, 1).states_in_order())[0]
    Q3, _ = np.linalg.qr(rng.standard_normal((idx.size, idx.size)))
    Vu = sp.coo_matrix((Q3.ravel().astype(np.float32).astype(dtype),
                        (np.repeat(idx, idx.size), np.tile(np.arange(idx.size), idx.size))), shape=(N, idx.size))
    narrow = dict(A=A, S=S, X=V[:, :K].astype(dtype), O=O, B=B, Xg=Vg[:, :K].astype(dtype), Anh=Anh,
                  Xnh=Vn.astype(np.complex64), V=Viso, Hs=Hs, Vu=Vu, Sd=rng.standard_normal(N).astype(np.float32),
                  T=[rnd(1, 2, 2), rnd(2, 2, 2), rnd(2, 2, 1)])
    out = {}
    for prec in ("narrow", "wide"):
        D = Data()
        for k, v in narrow.items():
            if prec == "wide":
                v = _wide(v)
            setattr(D, k, v)
        D.mu, D.mug, D.mun, D.spec, D.dtype = w[:K], wg[:K], wn, spec, dtype
        D.Xw = V[:, :K]                                   # binary64 vectors given to low-level products
        D.tmp = tempfile.mkdtemp(prefix=f"reg_{dtype}_{prec}_")
        D.ec = lambda D=D: _ec(D)
        out[prec] = D
    return out


def _wide(v):
    if isinstance(v, list):
        return [_wide(a) for a in v]
    if sp.issparse(v):
        W = v.copy()
        W.data = v.data.astype(WIDE.get(v.dtype.name, v.dtype))
        return W
    if isinstance(v, np.ndarray) and v.dtype.name in WIDE:
        return v.astype(WIDE[v.dtype.name])
    return v


def _ec(D):
    from certify_ed.eigenspace import certify_eigenspace
    return certify_eigenspace(_wide(D.X), [0], np.asarray(D.mu), 1e-6, 1e-6, 1.0)


_CACHE = {}


def _data(dtype):
    if dtype not in _CACHE:
        _CACHE[dtype] = _make(dtype)
    return _CACHE[dtype]


# --------------------------------------------------------------------------------------------------- comparison
def canon(o):
    """A JSON-able picture of a result in which every float is its exact hex form and every array its dtype, shape and
    bytes; records lose the fields that differ by design (the widening record, hashes over it, times, paths)."""
    from certify_ed.certificate_v2 import Certificate
    if isinstance(o, Certificate):
        return {"body": canon(o.body), "arrays": {k: canon(np.asarray(v)) for k, v in sorted(o.arrays.items())}}
    if isinstance(o, dict):
        return {str(k): canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))
                if str(k) not in DROP and not any(w in str(k) for w in ("second", "time", "wall"))}
    if isinstance(o, (list, tuple)):
        return [canon(v) for v in o]
    if sp.issparse(o):
        C = sp.csr_matrix(o, copy=True)
        C.sum_duplicates()
        return {"sparse": [canon(C.data), canon(C.indices.astype(np.int64)), canon(C.indptr.astype(np.int64)),
                           list(C.shape)]}
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            return [canon(v) for v in o.ravel().tolist()]
        return [o.dtype.str, list(o.shape), np.ascontiguousarray(o).tobytes().hex()]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o).hex()
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real).hex(), float(o.imag).hex()]
    if hasattr(o, "to_dict") and callable(o.to_dict):
        return canon(o.to_dict())
    if is_dataclass(o) or hasattr(o, "__dict__"):
        return {"type": type(o).__name__, **canon({k: v for k, v in vars(o).items() if not callable(v)})}
    return repr(o)


def _first_difference(a, b, path=""):
    if type(a) is not type(b):
        return path or "/"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}"
            d = _first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else (path or "/")


# ------------------------------------------------------------------------------------------------------ recipes
def _cert(claim, **kw):
    return certify(claim, options=Options(**kw) if kw else None)


def _files(D):
    """The data as files (scipy COO npz, npz row/col/data, HDF5 row/col/data group), written as stored."""
    from certify_ed.io.formats import write_array
    p = D.tmp
    write_array(os.path.join(p, "ev.npy"), np.asarray(D.mu))
    write_array(os.path.join(p, "X.npy"), D.X, role="eigenvectors")
    sp.save_npz(os.path.join(p, "H_scipy.npz"), D.S, compressed=False)
    np.savez(os.path.join(p, "H_rcd.npz"), row=D.S.row, col=D.S.col, data=D.S.data, shape=np.array(D.S.shape))
    h5py = pytest.importorskip("h5py")
    with h5py.File(os.path.join(p, "H.h5"), "w") as f:
        g = f.create_group("H")
        for k, v in (("row", D.S.row), ("col", D.S.col), ("data", D.S.data), ("shape", np.array(D.S.shape))):
            g.create_dataset(k, data=v)
    return p


def _r_claim_from_file(D):
    from certify_ed.importers import claim_from_file
    p = _files(D)
    return [certify(claim_from_file(os.path.join(p, "ev.npy"), os.path.join(p, "X.npy"), os.path.join(p, f)))
            for f in ("H_scipy.npz", "H_rcd.npz", "H.h5")]


def _r_bundle(D):
    from certify_ed.io.bundle import read_bundle, write_bundle
    out = []
    for fmt in ("npz", "h5"):
        path = os.path.join(D.tmp, f"bundle_{fmt}")
        write_bundle(path, np.asarray(D.mu), D.X, operator=D.S, fmt="npz", operator_fmt=fmt, kind="subset")
        out.append(certify(read_bundle(path).claim))
    return out


def _r_export_npz(D):
    from certify_ed.importers import claim_from_npz, export_npz
    path = os.path.join(D.tmp, "export.npz")
    export_npz(path, np.asarray(D.mu), D.X, D.S)
    return certify(claim_from_npz(path))


def _r_claim_from_hdf5(D):
    h5py = pytest.importorskip("h5py")
    from certify_ed.importers import claim_from_hdf5
    path = os.path.join(D.tmp, "claim.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("eigenvalues", data=np.asarray(D.mu))
        f.create_dataset("eigenvectors", data=D.X)
        g = f.create_group("H")
        Cn = _noncanonical_csr(D.S)
        for k, v in (("data", Cn.data), ("indices", Cn.indices), ("indptr", Cn.indptr), ("shape", np.array(Cn.shape))):
            g.create_dataset(k, data=v)
    return certify(claim_from_hdf5(path))


def _noncanonical_csr(S):
    """CSR with the COO's stored entries, the repeated one kept (rows ordered stably)."""
    order = np.argsort(S.row, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(S.row, minlength=S.shape[0]))])
    return sp.csr_matrix((S.data[order], S.col[order], indptr), shape=S.shape)


def _r_write_operator(D):
    from certify_ed.io.formats import read_operator, write_operator
    out = []
    for ext in (".npz", ".mtx", ".json"):
        path = os.path.join(D.tmp, "op" + ext)
        write_operator(path, D.S)
        out.append(certify(Claim(D.mu, D.X, read_operator(path), kind="subset")))
    return out


def _r_compose(D):
    from certify_ed.compose import SectorResult, certify_low_energy
    spec = D.spec
    H = ce.build_bitwise(spec).toarray()
    rng = np.random.default_rng(5)
    Hf = (H + np.diag(rng.standard_normal(N).astype(np.float32))).astype(D.A.dtype)   # exact in single precision
    full = ce.full_state_set(spec)
    secs = []
    Hw = Hf.astype(WIDE.get(Hf.dtype.name, Hf.dtype))
    for nup in range(4):
        ix = full.index_of(ce.u1_state_set(spec, nup).states_in_order())[0]
        V = sp.csr_matrix((np.ones(ix.size), (ix, np.arange(ix.size))), shape=(N, ix.size)).astype(D.A.dtype)
        Hs = Hf[np.ix_(ix, ix)]
        ws, Ys = np.linalg.eigh(Hw[np.ix_(ix, ix)])
        Y = Ys.astype(np.float32 if D.dtype == "float32" else np.complex64).astype(D.A.dtype)
        secs.append(SectorResult(f"nup={nup}", V, sp.coo_matrix(Hs), ws, Y))
    E_cut = float(np.sort(np.linalg.eigvalsh(Hw))[3]) + 1e-3
    return certify_low_energy(spec, secs, E_cut, H_full=Hf, label_symmetries=("translation",))


def _r_sweep(D):
    from certify_ed.sweep import certified_sweep
    spec_fn = lambda h: ce.tfim_chain(3, 1.0, h)                         # noqa: E731
    dt = D.A.dtype

    def solver(H, k):
        w, V = np.linalg.eigh(H.toarray() if sp.issparse(H) else H)
        return w[:k], V[:, :k].astype(np.float32).astype(dt)            # vectors stored in the test's precision
    return certified_sweep(spec_fn, [0.5, 0.6], k=1, solver=solver)


def _r_certified_moments(D):
    from certify_ed.moments import certified_moments
    return certified_moments(D.S, D.O, D.X[:, :1], 1e-6, 1e-6, K=2)


def _r_moments_from_certificate(D):
    from certify_ed.moments import moments_from_certificate
    w, V = np.linalg.eigh(_wide(D.S).toarray())
    cw = certify(Claim(w, V, _wide(D.S), kind="all"), options=Options(store_eigenvectors=True, eigenspaces="all"))
    return moments_from_certificate(cw, D.S, D.O, K=1, members=[0])


def _adapter(D):
    from certify_ed.enclosure import OperatorAdapter
    return OperatorAdapter(D.S)


def _r_adapter(D):
    ad = _adapter(D)
    return [ad.A, ad.norm_upper(), ad.matvec(D.Xw), ad.abs_matvec(D.Xw), ad.both_products(D.Xw)]


def _streamed(D):
    from certify_ed.enclosure import StreamedColumns, read_columns
    s = StreamedColumns(D.X, dtype=WIDE[D.dtype])
    return [read_columns(s, [1, 0]), s[:, 0]]


R = {}   # name -> recipe(D)
R.update({
    "certify_ed.certificate_v2.matrix_sha256": lambda D: [matrix_sha(D.S), matrix_sha(D.A), matrix_sha(_noncanonical_csr(D.S))],
    "certify_ed.certify.check_model_fidelity": lambda D: _fidelity(D),
    "certify_ed.certify.certify": lambda D: [certify(Claim(D.mu, D.X, D.S, kind="subset"),
                                                     options=Options(eigenspaces="all", observables={"O": D.O},
                                                                     symmetries={"d": (D.Sd, None)},
                                                                     approximate_symmetries={"s": (D.O, None)},
                                                                     store_eigenvectors=True)),
                                             certify(Claim(D.mu, D.X, D.A, kind="subset", sector_isometry=None))],
    "certify_ed.compat.symveig.enclose_global": lambda D: [_enc(ce_sym().enclose_global(D.A)),
                                                           _enc(ce_sym().enclose_global(D.Anh, symmetrize=True))],
    "certify_ed.compat.symveig.enclose_sectors": lambda D: _enc(ce_sym().enclose_sectors(_blocked(D), _sdiag())),
    "certify_ed.compose.certify_low_energy": _r_compose,
    "certify_ed.correlations.hermitian_parts": lambda D: _corr().hermitian_parts(sp.coo_matrix(D.Anh), "O"),
    "certify_ed.correlations.certify_observable_general": lambda D: _corr().certify_observable_general(D.ec(), D.X, D.Anh, "O"),
    "certify_ed.correlations.certify_observables_batch": lambda D: _corr().certify_observables_batch(
        D.ec(), D.X, {"O": D.O, "N": D.Anh}),
    "certify_ed.correlations.certify_correlations": lambda D: _corr().certify_correlations(D.ec(), D.X, D.spec, "z", [0, 1, 2]),
    "certify_ed.correlations.compress_operator": lambda D: _corr().compress_operator(D.V, D.O),
    "certify_ed.correlations.certify_observable_lifted": lambda D: _corr().certify_observable_lifted(
        D.ec(), D.X, D.V, D.O, "O", 1e-7),
    "certify_ed.correlations.certify_correlations_lifted": lambda D: _corr().certify_correlations_lifted(
        D.ec(), D.X, D.spec, D.V, "z", [0, 1], 1e-7),
    "certify_ed.davis_kahan.commutator_bound": lambda D: _dk().commutator_bound(D.S, D.O),
    "certify_ed.davis_kahan.approximate_symmetry_label": lambda D: _dk().approximate_symmetry_label(
        D.ec(), D.X, D.O, "S", 1e-3, [1.0, 2.0]),
    "certify_ed.davis_kahan.certified_fidelity": lambda D: _dk().certified_fidelity(D.X[:, :1], 1e-6, 1e-6, D.Xg[:, :1], 1e-6, 1e-6),
    "certify_ed.davis_kahan.labels_hook": lambda D: _labels(D),
    "certify_ed.eigenspace.commutes_diagonal": lambda D: [_es().commutes_diagonal(D.S, np.zeros(N)),
                                                          _es().commutes_diagonal(D.S, np.arange(N))],
    "certify_ed.eigenspace.certify_eigenspace": lambda D: _es().certify_eigenspace(D.X, [0], np.asarray(D.mu), 1e-6, 1e-6, 1.0),
    "certify_ed.eigenspace.certify_observable": lambda D: _es().certify_observable(D.ec(), D.X, D.O, "O"),
    "certify_ed.eigenspace.certify_quantum_numbers": lambda D: _es().certify_quantum_numbers(
        D.ec(), D.X, sp.diags(D.Sd), "S"),
    "certify_ed.eigenspace.rotation_spread": lambda D: _es().rotation_spread(D.X, [0, 1], D.O, trials=4),
    "certify_ed.enclosure.OperatorAdapter": _r_adapter,
    "certify_ed.enclosure.StreamedColumns": _streamed,
    "certify_ed.enclosure.residuals": lambda D: _enc_mod().residuals(_adapter(D), D.mu, D.X, keep_R=True),
    "certify_ed.enclosure.nonfinite_columns": lambda D: _enc_mod().nonfinite_columns(D.X),
    "certify_ed.enclosure.enclose": lambda D: _enc_mod().enclose(_adapter(D), D.mu, D.X).to_records(),
    "certify_ed.enclosure.rayleigh_interval": lambda D: _enc_mod().rayleigh_interval(_adapter(D), D.X[:, 0]),
    "certify_ed.enclosure.kato_temple": lambda D: _enc_mod().kato_temple(_adapter(D), D.X[:, 0], D.mu[0] - 0.5, D.mu[1]),
    "certify_ed.entanglement.reduced_density_exact": lambda D: _ent_rde(D),
    "certify_ed.entanglement.certify_entanglement": lambda D: _ent_mod().certify_entanglement(
        D.ec(), D.X, ce.full_state_set(D.spec).states_in_order(), 3, 2, [0]),
    "certify_ed.entanglement.entanglement_records": lambda D: _ent_mod().entanglement_records(
        D.ec(), D.X, D.spec, Claim(D.mu, D.X, D.S, kind="subset"), {"A": {"sites": [0]}}),
    "certify_ed.fp.norm2_upper": lambda D: [_fp().norm2_upper(D.X[:, 0]), _fp().norm2_upper(D.A.ravel())],
    "certify_ed.fp.norm2_lower": lambda D: [_fp().norm2_lower(D.X[:, 0]), _fp().norm2_lower(D.A.ravel())],
    "certify_ed.fp.fro_upper": lambda D: [_fp().fro_upper(D.A), _fp().fro_upper(sp.csr_matrix(D.A))],
    "certify_ed.fp.inf_norm_upper": lambda D: [_fp().inf_norm_upper(D.A), _fp().inf_norm_upper(D.S)],
    "certify_ed.fp.max_nnz_per_row": lambda D: [_fp().max_nnz_per_row(D.A), _fp().max_nnz_per_row(sp.csr_matrix(D.A))],
    "certify_ed.fp.tiny_nonzero": lambda D: _fp().tiny_nonzero(D.X),
    "certify_ed.fp.min_nonzero_magnitude": lambda D: _fp().min_nonzero_magnitude(D.X),
    "certify_ed.fp.underflow_scan": lambda D: _fp().underflow_scan([("X", D.X), ("A", D.A), ("t", D.X, 0.05)]),
    "certify_ed.fp.exact_sum": lambda D: _fp().exact_sum(D.X[:, 0].real),
    "certify_ed.fp.exact_dot": lambda D: _fp().exact_dot(D.X[:, 0], D.X[:, 1]),
    "certify_ed.fp.exact_gram": lambda D: _fp().exact_gram(D.X),
    "certify_ed.generalized.certify_B_positive_definite": lambda D: _gen().certify_B_positive_definite(D.B),
    "certify_ed.generalized.pencil_residual_radii": lambda D: _gen().pencil_residual_radii(D.A, D.B, D.mug, D.Xg, 0.5),
    "certify_ed.generalized.pencil_count_below": lambda D: _gen().pencil_count_below(D.A, D.B, float(D.mug[0]) - 1e-3, 0.5),
    "certify_ed.generalized.certify_generalized": lambda D: _gen().certify_generalized(D.S, D.B, D.mug, D.Xg, kind="lowest"),
    "certify_ed.importers.claim_from_arrays": lambda D: certify(ce_imp().claim_from_arrays(D.mu, D.X, D.S)),
    "certify_ed.importers.claim_from_npz": _r_export_npz,
    "certify_ed.importers.export_npz": _r_export_npz,
    "certify_ed.importers.claim_from_hdf5": _r_claim_from_hdf5,
    "certify_ed.importers.claim_from_file": _r_claim_from_file,
    "certify_ed.importers.claim_from_bundle": lambda D: [_bundle_certs(D)],
    "certify_ed.inertia.inertia_dense": lambda D: _inr().inertia_dense(D.A, float(D.mu[0]) + 1e-3),
    "certify_ed.inertia.inertia_sparse": lambda D: _inr().inertia_sparse(D.S, float(D.mu[0]) + 1e-3),
    "certify_ed.inertia.inertia": lambda D: [_inr().inertia(D.S, float(D.mu[0]) + 1e-3),
                                             _inr().inertia(D.A, float(D.mu[0]) + 1e-3)],
    "certify_ed.io.bundle.read_bundle": _r_bundle,
    "certify_ed.io.bundle.write_bundle": _r_bundle,
    "certify_ed.io.formats.write_operator": _r_write_operator,
    "certify_ed.moments.hermitian_matrix_with_bound": lambda D: ce_mom().hermitian_matrix_with_bound([D.S, D.A], [N, N]),
    "certify_ed.moments.certified_moments": _r_certified_moments,
    "certify_ed.moments.moments_from_certificate": _r_moments_from_certificate,
    "certify_ed.nonhermitian.residual_bounds": lambda D: ce_nh().residual_bounds(D.Anh, D.mun, D.Xnh),
    "certify_ed.nonhermitian.matrix_norm2_upper": lambda D: ce_nh().matrix_norm2_upper(D.Anh),
    "certify_ed.nonhermitian.inverse_bound": lambda D: ce_nh().inverse_bound(D.Xnh),
    "certify_ed.nonhermitian.disc_groups": lambda D: ce_nh().disc_groups(
        D.mun.astype(np.complex64) if D.A.dtype.name in WIDE else D.mun.astype(np.complex64).astype(np.complex128), 1e-3),
    "certify_ed.nonhermitian.certify_nonhermitian": lambda D: ce_nh().certify_nonhermitian(sp.coo_matrix(D.Anh), D.mun, D.Xnh,
                                                                                         kind="all"),
    "certify_ed.ozaki.ozaki_gram": lambda D: ce_oz().ozaki_gram(D.X),
    "certify_ed.ozaki.ozaki_column_dots": lambda D: ce_oz().ozaki_column_dots(D.X, D.Xg),
    "certify_ed.ozaki.ozaki_residual": lambda D: ce_oz().ozaki_residual(D.S, D.mu, D.X),
    "certify_ed.ozaki.residual_report": lambda D: ce_oz().residual_report(D.S, D.mu, D.X),
    "certify_ed.rigorous.arb_residual_radii": lambda D: _arb().arb_residual_radii(D.S, D.mu, D.X),
    "certify_ed.rigorous.arb_eigenvalues": lambda D: _arb().arb_eigenvalues(D.S),
    "certify_ed.sectors.audit_sector": lambda D: ce_sec().audit_sector(D.S, D.V, D.Hs),
    "certify_ed.sectors.audit_orthogonality": lambda D: ce_sec().audit_orthogonality(
        [sp.csr_matrix(D.V[:, :3]), sp.coo_matrix(D.V[:, 3:])]),
    "certify_ed.sectors.audit_decomposition": lambda D: ce_sec().audit_decomposition(
        D.S, [sp.coo_matrix(D.V[:, :4]), sp.csr_matrix(D.V[:, 4:])],
        [sp.coo_matrix(D.Hs[:4, :4]), sp.csr_matrix(D.Hs[4:, 4:])]),
    "certify_ed.sectors.trace_moments_check": lambda D: ce_sec().trace_moments_check(D.S, [sp.coo_matrix(D.Hs), D.O]),
    "certify_ed.sectors.restrict_to_u1": lambda D: ce_sec().restrict_to_u1(D.spec, D.Vu)[1],
    "certify_ed.sectors.audit_label": lambda D: ce_sec().audit_label(D.spec, D.Vu, "translation"),
    "certify_ed.spectrum.enclose_spectrum": lambda D: [_spec_enc(ce_spm().enclose_spectrum(D.S)),
                                                       _spec_enc(ce_spm().enclose_spectrum(
                                                           D.A, eigh_result=_eigh_narrow(D)))],
    "certify_ed.sweep.h1_norm_bound": lambda D: ce_sw().h1_norm_bound(ce.tfim_chain(3, 0.0, 1.0), D.S),
    "certify_ed.sweep.matrix_affinity_defect": lambda D: ce_sw().matrix_affinity_defect(D.S, D.A, 0.3, D.O),
    "certify_ed.sweep.certified_sweep": _r_sweep,
    "certify_ed.variational.state_from_dense": lambda D: ce_var().state_from_dense(D.X[:, 0]),
    "certify_ed.variational.state_from_sparse_ci": lambda D: ce_var().state_from_sparse_ci([0, 3, 5], D.X[:3, 0], N),
    "certify_ed.variational.state_from_mps": lambda D: ce_var().state_from_mps(D.T),
    "certify_ed.variational.certify_variational": lambda D: [ce_var().certify_variational(D.S, D.X[:, 0]),
                                                             ce_var().certify_variational(D.A, D.X[:, 0])],
})


def _narrow(D):
    return D.A.dtype.name in WIDE


def _swap_matrix(D):
    """(0, 0) stored as 1 and 2^-25, (0, 1) = (1, 0) = 0.5, (1, 1) = 1: the swap of the two sites is not a symmetry
    of the stored matrix (exact sum 1 + 2^-25), but it is of the single-precision sum."""
    M = sp.coo_matrix((np.array([1.0, 2.0 ** -25, 0.5, 0.5, 1.0], dtype=D.dtype), ([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])),
                      shape=(2, 2))
    return M if _narrow(D) else _wide(M)


def _mod(name):
    return importlib.import_module(name)


def _r_exact_recheck_lowest(D):
    pytest.importorskip("flint")
    return _mod("certify_ed.checker").exact_recheck_lowest(D.spec, D.mu, D.X)


R.update({
    "certify_ed.eigenspace.commutes_signed_permutation": lambda D: [
        _es().commutes_signed_permutation(D.S, np.arange(N)[::-1].copy(), np.ones(N, dtype=np.int64)),
        _es().commutes_signed_permutation(_swap_matrix(D), np.array([1, 0]), np.array([1, 1]))],
    "certify_ed.construct.hermitian_part": lambda D: [_mod("certify_ed.construct").hermitian_part(sp.coo_matrix(D.Anh)),
                                                      _mod("certify_ed.construct").hermitian_part(D.S)],
    "certify_ed.construct.matrix_difference": lambda D: _mod("certify_ed.construct").matrix_difference(
        D.S, _wide(D.S).tocsr()),
    "certify_ed.basis.sector_matrix": lambda D: _mod("certify_ed.basis").sector_matrix(D.S, sp.coo_matrix(D.V)),
    "certify_ed.invariants.matrix_invariants": lambda D: [_mod("certify_ed.invariants").matrix_invariants(D.S),
                                                          _mod("certify_ed.invariants").matrix_invariants(D.A)],
    "certify_ed.checker.exact_eigenpair_radii": lambda D: _mod("certify_ed.checker").exact_eigenpair_radii(
        D.spec, D.mu, D.X),
    "certify_ed.checker.exact_recheck_lowest": _r_exact_recheck_lowest,
})


def matrix_sha(M):
    from certify_ed.certificate_v2 import matrix_sha256
    return matrix_sha256(M)


def _fidelity(D):
    from certify_ed.certify import check_model_fidelity
    spec = D.spec
    H = ce.build_bitwise(spec)
    Hn = sp.coo_matrix(H.toarray().astype(D.A.dtype))
    return check_model_fidelity(Claim(D.mu, D.X, Hn, kind="subset"), spec, Options())


def ce_sym():
    from certify_ed.compat import symveig
    return symveig


def _enc(encs):
    return [(e.midpoint, e.halfwidth, e.multiplicity, e.counted) for e in encs]


def _blocked(D):
    M = D.A.copy()
    M[:4, 4:] = 0
    M[4:, :4] = 0
    return M


def _sdiag():
    return np.array([0] * 4 + [1] * 4, dtype=float)


def _corr():
    from certify_ed import correlations
    return correlations


def _dk():
    from certify_ed import davis_kahan
    return davis_kahan


def _labels(D):
    from certify_ed.davis_kahan import labels_hook
    ec = D.ec()
    labels_hook(ec, D.X, {"s": (D.O, None), "d": (D.Sd, None)}, D.S, [1.0, 2.0])
    return ec


def _es():
    from certify_ed import eigenspace
    return eigenspace


def _enc_mod():
    from certify_ed import enclosure
    return enclosure


def _ent_mod():
    from certify_ed import entanglement
    return entanglement


def _ent_rde(D):
    from certify_ed.entanglement import bipartition_indices, reduced_density_exact
    a, b, dA, dB = bipartition_indices(ce.full_state_set(D.spec).states_in_order(), 3, 2, [0])
    return reduced_density_exact(D.X, a, b, dA, dB)


def _fp():
    from certify_ed import fp
    return fp


def _gen():
    from certify_ed import generalized
    return generalized


def ce_imp():
    from certify_ed import importers
    return importers


def _bundle_certs(D):
    from certify_ed.importers import claim_from_bundle
    from certify_ed.io.bundle import write_bundle
    path = os.path.join(D.tmp, "bundle_cfb")
    write_bundle(path, np.asarray(D.mu), D.X, operator=D.S, basis=None, fmt="npz", kind="subset")
    return certify(claim_from_bundle(path)[0])


def _inr():
    from certify_ed import inertia
    return inertia


def ce_mom():
    from certify_ed import moments
    return moments


def ce_nh():
    from certify_ed import nonhermitian
    return nonhermitian


def ce_oz():
    from certify_ed import ozaki
    return ozaki


def _arb():
    pytest.importorskip("flint")
    from certify_ed import rigorous
    return rigorous


def ce_sec():
    from certify_ed import sectors
    return sectors


def ce_spm():
    from certify_ed import spectrum
    return spectrum


def _spec_enc(e):
    return [(iv.lo, iv.hi, iv.count, iv.counted, iv.method) for iv in e.intervals]


def _eigh_narrow(D):
    w, V = np.linalg.eigh(_wide(D.A))
    return w, V.astype(np.float32 if D.dtype == "float32" else np.complex64).astype(D.X.dtype)


def ce_sw():
    from certify_ed import sweep
    return sweep


def ce_var():
    from certify_ed import variational
    return variational


EXEMPT = {
    # certificate_v2
    "certify_ed.certificate_v2.array_sha256": "hash of an array as given (its dtype is part of the hash by design); "
                                              "the records hash the converted data",
    "certify_ed.certificate_v2.environment_record": "no input",
    "certify_ed.certificate_v2.Certificate": "container of a certificate (no arithmetic on stored data)",
    # certify
    "certify_ed.certify.Claim": "container; certify() converts its arrays",
    "certify_ed.certify.Options": "container; certify() converts its matrices",
    "certify_ed.certify.parse_window": "two numbers",
    # compat.symveig
    "certify_ed.compat.symveig.Enclosure": "result container",
    "certify_ed.compat.symveig.summary_stats": "takes enclosures",
    "certify_ed.compat.symveig.heisenberg_1d": "builder (binary64 by construction)",
    "certify_ed.compat.symveig.j1j2_heisenberg_1d": "builder (binary64 by construction)",
    "certify_ed.compat.symveig.heisenberg_2d": "builder (binary64 by construction)",
    # compose
    "certify_ed.compose.SectorResult": "container; certify_low_energy converts its arrays",
    # correlations
    "certify_ed.correlations.site_operator": "builder (binary64 by construction)",
    "certify_ed.correlations.modulus_upper": "a scalar",
    "certify_ed.correlations.combine_certified": "certified values and bounds (scalars)",
    "certify_ed.correlations.structure_factor": "takes correlation records",
    "certify_ed.correlations.positions_chain": "an integer",
    "certify_ed.correlations.q_chain": "an integer",
    # davis_kahan
    "certify_ed.davis_kahan.sin_theta_davis_kahan": "scalars",
    "certify_ed.davis_kahan.fidelity_from_certificates": "takes certificates (stored vectors: binary64, or integers "
                                                         "as given, converted by certified_fidelity)",
    # eigenspace
    "certify_ed.eigenspace.EigenspaceCertificate": "record container",
    # enclosure
    "certify_ed.enclosure.is_streamed": "a predicate",
    "certify_ed.enclosure.column_blocks": "integers",
    "certify_ed.enclosure.read_columns": "returns the stored columns as given (a reader, no arithmetic)",
    "certify_ed.enclosure.ResidualReport": "result container",
    "certify_ed.enclosure.Group": "result container",
    "certify_ed.enclosure.EnclosureSet": "result container",
    "certify_ed.enclosure.cluster_groups": "takes an enclosure set",
    # entanglement
    "certify_ed.entanglement.bipartition_indices": "integer states",
    "certify_ed.entanglement.enclose_psd_eigenvalues": "exact rational matrices (Fractions from reduced_density_exact)",
    "certify_ed.entanglement.entropy_interval": "rationals",
    "certify_ed.entanglement.binary_entropy_upper": "a rational",
    "certify_ed.entanglement.fannes_audenaert_bound": "a rational and an integer",
    # fp
    "certify_ed.fp.GuardRangeError": "exception",
    "certify_ed.fp.guard_range_findings": "integers",
    "certify_ed.fp.check_guard_range": "integers",
    "certify_ed.fp.gamma": "a scalar",
    "certify_ed.fp.gamma_dot": "a scalar",
    "certify_ed.fp.sub_down": "directed rounding by nextafter in the operands' own format: a valid bound in any "
                              "IEEE format; callers pass binary64 data",
    "certify_ed.fp.add_up": "directed rounding by nextafter in the operands' own format (see sub_down)",
    "certify_ed.fp.sub_up": "directed rounding by nextafter in the operands' own format (see sub_down)",
    "certify_ed.fp.add_down": "directed rounding by nextafter in the operands' own format (see sub_down)",
    "certify_ed.fp.modulus_up": "an upper modulus in any IEEE format: the absolute value in the data's own type for "
                                "real data (exact), binary64 for complex data (Lemma M); callers pass binary64 data",
    "certify_ed.fp.abs_up": "entrywise modulus_up (see modulus_up); callers pass binary64 data, and a narrower sparse "
                            "matrix would have its repeated entries summed in its own precision",
    "certify_ed.fp.up": "a scalar, converted with float()",
    "certify_ed.fp.norm_underflow_term": "an integer",
    "certify_ed.fp.residual_entry_count": "integers",
    "certify_ed.fp.residual_additive_term": "integers",
    "certify_ed.fp.gram_additive_term": "integers",
    "certify_ed.fp.is_complex": "a predicate",
    "certify_ed.fp.float_env_record": "no input",
    "certify_ed.fp.probe_float_environment": "no input",
    "certify_ed.fp.safe_sqrt_upper": "a scalar, converted with float()",
    # generalized
    "certify_ed.generalized.format_generalized": "formats a record",
    # importers
    "certify_ed.importers.require_quspin": "no input",
    "certify_ed.importers.claim_from_quspin": "needs QuSpin objects; a scipy matrix in a narrower format is passed "
                                              "through as given (test_single_precision)",
    # inertia
    "certify_ed.inertia.InertiaResult": "result container",
    "certify_ed.inertia.count_bounds": "takes an inertia result",
    # io.bundle
    "certify_ed.io.bundle.Bundle": "container",
    "certify_ed.io.bundle.bundle_files": "a path",
    # io.formats
    "certify_ed.io.formats.FormatError": "exception",
    "certify_ed.io.formats.read_array": "returns the array as stored (dtype kept by design; certify() converts it)",
    "certify_ed.io.formats.write_array": "writes the array as given",
    "certify_ed.io.formats.read_operator": "returns the matrix as stored (a single-precision COO keeps its entries); "
                                           "covered through claim_from_file and write_operator",
    "certify_ed.io.formats.read_basis": "returns the basis as stored; covered through the bundle recipes",
    "certify_ed.io.formats.write_basis": "writes the basis as given (entries kept, see write_operator)",
    "certify_ed.io.formats.sniff_kind": "a path",
    # ldl_residual
    "certify_ed.ldl_residual.ResidualBound": "result container",
    "certify_ed.ldl_residual.sparse_ldl_residual_bound": "internal bound on the binary64 factors of inertia_sparse; "
                                                         "fails closed for other formats (test_fail_closed)",
    # moments
    "certify_ed.moments.format_moments": "formats a record",
    # nonhermitian
    "certify_ed.nonhermitian.additive_term": "integers",
    "certify_ed.nonhermitian.format_nonhermitian": "formats a record",
    # ozaki
    "certify_ed.ozaki.NotApplicable": "exception",
    "certify_ed.ozaki.split_beta": "an integer",
    "certify_ed.ozaki.Split": "result container",
    "certify_ed.ozaki.split_columns": "internal split of binary64 data; fails closed (test_fail_closed)",
    "certify_ed.ozaki.split_complex_columns": "internal split of binary64 data; fails closed (test_fail_closed)",
    "certify_ed.ozaki.RowSplit": "result container",
    "certify_ed.ozaki.split_rows": "internal split of binary64 data; fails closed (test_fail_closed)",
    "certify_ed.ozaki.sum2_coefficient": "an integer",
    "certify_ed.ozaki.GramResult": "result container",
    "certify_ed.ozaki.OzakiResidual": "result container",
    # precision
    "certify_ed.precision.PrecisionError": "exception",
    "certify_ed.precision.classify_format": "the format decision (test_single_precision)",
    "certify_ed.precision.is_narrow_float": "a predicate on a dtype",
    "certify_ed.precision.require_binary64": "the fail-closed check (test_fail_closed)",
    "certify_ed.precision.target_dtype": "a dtype mapping",
    "certify_ed.precision.source_sha256": "hash of the input as given (by design)",
    "certify_ed.precision.binary64": "the conversion itself (test_single_precision)",
    "certify_ed.precision.summed": "sums the repeated entries of binary64 data on a copy, after binary64; fails closed "
                                   "for narrower formats (test_duplicate_entries)",
    "certify_ed.precision.needs_summing": "a predicate on the sparse format and flags, no output data",
    "certify_ed.precision.check_values": "a check, no output",
    "certify_ed.precision.widening_record": "builds the record",
    # sectors
    "certify_ed.sectors.SectorAudit": "result container",
    "certify_ed.sectors.DecompositionAudit": "result container",
    "certify_ed.sectors.decomposition_shift": "takes an audit and two norms",
    "certify_ed.sectors.lift_radius": "scalars and an audit",
    "certify_ed.sectors.symmetry_matrix": "builder (signed permutation, exact)",
    # spectrum
    "certify_ed.spectrum.SpectrumInterval": "result container",
    "certify_ed.spectrum.SpectrumEnclosure": "result container",
    "certify_ed.spectrum.intervals_from_enclosure": "takes an enclosure set",
    "certify_ed.spectrum.merge_intervals": "takes intervals",
    # sweep
    "certify_ed.sweep.derivative_spec": "model specs",
    "certify_ed.sweep.term_norm_bound": "a model spec",
    "certify_ed.sweep.affinity_defect": "model specs (exact rationals)",
    "certify_ed.sweep.tube_at": "takes a sweep record",
    "certify_ed.sweep.next_level_lower_bound_at": "takes a sweep record",
    "certify_ed.sweep.certified_crossing_interval": "takes sweep records",
    "certify_ed.sweep.format_sweep": "formats a record",
    "certify_ed.sweep.format_crossing_interval": "formats a record",
    # variational
    "certify_ed.variational.TrialState": "container",
    "certify_ed.variational.gap_from_certificate": "takes a certificate",
    "certify_ed.variational.format_variational": "formats a record",
    # basis
    "certify_ed.basis.CyclicSymmetry": "symmetry container",
    "certify_ed.basis.cyclic_isometry": "builds a binary64 isometry from integer states",
    "certify_ed.basis.digit_sum": "integer states",
    "certify_ed.basis.magnetization_state_set": "state set of the spec",
    "certify_ed.basis.reflection": "symmetry of the spec",
    "certify_ed.basis.spin_flip": "symmetry of the spec",
    "certify_ed.basis.translation": "symmetry of the spec",
    "certify_ed.basis.u1_state_set": "state set of the spec",
    # checker (exact layer: every stored binary64 or narrower value is taken as the exact rational it is)
    "certify_ed.checker.ExactMatrix": "container of the exact spec matrix",
    "certify_ed.checker.certify_by_counts": "exact counts on the ExactMatrix",
    "certify_ed.checker.check_semidefinite_exact": "exact integer residual of the factors of an integer matrix",
    "certify_ed.checker.check_semidefinite_lower_bound": "ball-arithmetic residual of the stored factors: the bound "
                                                         "holds whatever precision the factorization used",
    "certify_ed.checker.exact_dense": "exact matrix",
    "certify_ed.checker.exact_matrix": "built from the spec",
    "certify_ed.checker.inertia_exact": "exact inertia of the ExactMatrix",
    "certify_ed.checker.inertia_exact_int": "exact inertia of an integer matrix",
    "certify_ed.checker.is_hermitian_exact": "exact matrix",
    "certify_ed.checker.negative_count_exact": "exact rational count on the stored values",
    "certify_ed.checker.radius_exact": "exact integer arithmetic on the test vector's integers",
    "certify_ed.checker.residual_inf_norm_exact": "exact integer arithmetic on the stored values",
    "certify_ed.checker.residual_inf_norm_upper": "ball arithmetic on the exact rationals of the stored values",
    "certify_ed.checker.scaled_test_vector": "any test vector gives a valid interval; numpy promotes the division to "
                                             "binary64",
    # construct
    "certify_ed.construct.SpecOperator": "applies the spec operator (its own binary64 products)",
    "certify_ed.construct.StateSet": "container of integer basis states",
    "certify_ed.construct.build_bitwise": "builds the spec matrix in binary64",
    "certify_ed.construct.build_kron": "builds the spec matrix in binary64",
    "certify_ed.construct.cross_construct": "builds the spec matrix twice and compares",
    "certify_ed.construct.full_state_set": "state set of the spec",
    "certify_ed.construct.product_tolerance": "a tolerance from the spec and a float",
    # exact_sectors
    "certify_ed.exact_sectors.FiniteGroup": "symmetry group of the spec",
    "certify_ed.exact_sectors.check_commutes": "exact matrix of the spec",
    "certify_ed.exact_sectors.exact_count_below": "exact counts of the spec",
    "certify_ed.exact_sectors.exact_recheck_sector_lowest": "exact arithmetic; the vectors become exact integers; "
                                                            "lam is checked (check_values, tests/test_single_precision.py)",
    "certify_ed.exact_sectors.orbit_columns": "integer orbits",
    "certify_ed.exact_sectors.sector_integer_matrix": "exact matrix of the spec",
    "certify_ed.exact_sectors.sector_matrices": "exact matrices of the spec",
    "certify_ed.exact_sectors.site_permutation": "symmetry of the spec",
    # invariants
    "certify_ed.invariants.compare_invariants": "compares two records",
    "certify_ed.invariants.spec_invariants": "from the spec",
    "certify_ed.invariants.spectrum_invariants": "converts the eigenvalues with dtype=float (exact) before its sums",
}


def _public_names():
    names = []
    for m in MODULES:
        mod = importlib.import_module(m)
        for n, obj in vars(mod).items():
            if n.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                names.append(f"{m}.{n}")
    return sorted(names)


def test_registry_is_complete():
    """Every public function and class of MODULES is in the registry (called with single-precision data) or in EXEMPT
    with a reason; a new public name fails here until it is listed."""
    found = set(_public_names())
    listed = set(R) | set(EXEMPT)
    assert not (set(R) & set(EXEMPT)), sorted(set(R) & set(EXEMPT))
    missing, stale = sorted(found - listed), sorted(listed - found)
    assert not missing and not stale, f"not listed: {missing}; listed but not public: {stale}"


def _package_modules(root=None):
    """Every module of the package by its file (packages without __init__.py, such as apps, included); `root` is the
    package directory (default: the installed certify_ed)."""
    root = root or os.path.dirname(ce.__file__)
    out = []
    for d, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(d, f), os.path.dirname(root))[:-3].replace(os.sep, ".")
                out.append(rel[:-len(".__init__")] if rel.endswith(".__init__") else rel)
    return sorted(out)


def _covered(m):
    return m in MODULES or m in EXEMPT_MODULES or any(m == k or m.startswith(k + ".") for k in EXEMPT_TREES)


def test_every_module_is_listed():
    """Every module of certify_ed is in MODULES (its public names checked above), in EXEMPT_MODULES with a reason (an
    exact module name) or below a key of EXEMPT_TREES; a new module fails here until it is listed."""
    mods = _package_modules()
    missing = [m for m in mods if not _covered(m)]
    stale = [k for k in EXEMPT_MODULES if k not in mods] + \
        [k for k in EXEMPT_TREES if not any(m == k or m.startswith(k + ".") for m in mods)]
    assert not missing and not stale, f"not listed: {missing}; exempt but absent: {stale}"
    assert not set(MODULES) & (set(EXEMPT_MODULES) | set(EXEMPT_TREES))
    assert all(EXEMPT_MODULES.values()) and all(EXEMPT_TREES.values())
    # a made-up module is not covered, at the top level or in a namespace package
    for fake in ("certify_ed.zz_not_a_module", "certify_ed.io.zz_not_a_module", "certify_ed.compat.zz_not_a_module",
                 "certify_ed.integrations.zz_not_a_module"):
        assert not _covered(fake), fake


def test_a_new_module_file_fails_the_module_check(tmp_path):
    """A copy of the package's file tree with one more file: the check reports exactly that module."""
    root = os.path.dirname(ce.__file__)
    copy = tmp_path / "certify_ed"
    for d, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                p = copy / os.path.relpath(os.path.join(d, f), root)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("")
    (copy / "zz_new_module.py").write_text("def norm_bound(v):\n    return 0.0\n")
    (copy / "io" / "zz_new_reader.py").write_text("")
    mods = _package_modules(str(copy))
    assert sorted(m for m in mods if not _covered(m)) == ["certify_ed.io.zz_new_reader", "certify_ed.zz_new_module"]
    assert set(mods) - {"certify_ed.io.zz_new_reader", "certify_ed.zz_new_module"} == set(_package_modules())


@pytest.mark.parametrize("name", sorted(R))
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_single_precision_input_gives_the_binary64_result(dtype, name):
    D = _data(dtype)
    a, b = canon(R[name](D["narrow"])), canon(R[name](D["wide"]))
    assert a == b, f"{name}: first difference at {_first_difference(a, b)}"


# ------------------------------------------------------------------------------------------------- fail closed
@pytest.mark.parametrize("module, call", [
    ("certify_ed.inertia", lambda D: __import__("certify_ed.inertia", fromlist=["x"]).inertia_dense(D.A, 0.0)),
    ("certify_ed.inertia", lambda D: __import__("certify_ed.inertia", fromlist=["x"]).inertia_sparse(D.S, 0.0)),
    ("certify_ed.sectors", lambda D: __import__("certify_ed.sectors", fromlist=["x"]).audit_sector(D.S, D.V, D.Hs)),
    ("certify_ed.enclosure", lambda D: __import__("certify_ed.enclosure", fromlist=["x"]).OperatorAdapter(D.S)),
    ("certify_ed.enclosure", lambda D: __import__("certify_ed.enclosure", fromlist=["x"]).residuals(
        __import__("certify_ed.enclosure", fromlist=["x"]).OperatorAdapter(_wide(D.S)), D.mu, D.X)),
])
def test_fail_closed_when_an_entry_point_misses_its_conversion(monkeypatch, module, call):
    """With the module's conversion disabled (binary64 returning its argument, as a forgotten call would), the
    lowest layer that applies the binary64 constants refuses the single-precision data instead of returning a
    bound."""
    from certify_ed.precision import PrecisionError
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "binary64", lambda a, *args, **kw: a)
    with pytest.raises(PrecisionError, match="did not convert"):
        call(_data("float32")["narrow"])


def test_low_level_products_and_splits_refuse_single_precision():
    from certify_ed.enclosure import OperatorAdapter
    from certify_ed.ldl_residual import sparse_ldl_residual_bound
    from certify_ed.ozaki import split_columns, split_complex_columns, split_rows
    from certify_ed.precision import PrecisionError
    D = _data("float32")["narrow"]
    ad = OperatorAdapter(D.S)
    for f in (ad.matvec, ad.abs_matvec, ad.both_products):
        with pytest.raises(PrecisionError, match="did not convert"):
            f(D.X)
    for f, args in ((split_columns, (D.X, 20)), (split_complex_columns, (D.X, 20)), (split_rows, (D.S, 20)),
                    (sparse_ldl_residual_bound, (sp.csr_matrix(D.A), np.ones(N), D.S, np.arange(N)))):
        with pytest.raises(PrecisionError, match="did not convert"):
            f(*args)
    assert json.dumps(canon(ad.matvec(D.Xw))) == json.dumps(canon(OperatorAdapter(_wide(D.S)).matvec(D.Xw)))
    with pytest.raises(PrecisionError, match="did not convert"):     # moduli passed by a caller: binary64 too
        OperatorAdapter(_wide(D.S)).abs_matvec(D.Xw, np.abs(D.Xw).astype(np.float32))
    # caller-supplied products (the fast path of certify()) must be binary64 products too
    from certify_ed.enclosure import kato_temple, rayleigh_interval, residuals
    aw, x = OperatorAdapter(_wide(D.S)), D.Xw[:, 0]
    Ax, absAx = aw.matvec(x), aw.abs_matvec(x)
    HX, absHX = aw.matvec(D.Xw), aw.abs_matvec(D.Xw)
    for call in (lambda: rayleigh_interval(aw, x, Ax.astype(np.float32), absAx),
                 lambda: rayleigh_interval(aw, x, Ax, absAx.astype(np.float32)),
                 lambda: kato_temple(aw, x, -10.0, 10.0, Ax.astype(np.float32), absAx),
                 lambda: residuals(aw, D.mu, D.Xw, products=(HX.astype(np.float32), absHX)),
                 lambda: residuals(aw, D.mu, D.Xw, products=(HX, absHX.astype(np.float32)))):
        with pytest.raises(PrecisionError, match="did not convert"):
            call()
    assert json.dumps(canon(rayleigh_interval(aw, x, Ax, absAx))) == json.dumps(canon(rayleigh_interval(aw, x)))
