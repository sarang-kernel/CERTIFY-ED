"""Export certify-ed certificates for the core-Lean checker `cecheck`.

For every certificate listed in an index (the `index.json` written by
`verifier/make_validation_set.py`, or the mutant index written by `make_mutants.py`) this script writes a
directory `<out>/<case>/` holding

    cert.json            the certificate JSON, byte for byte as it was saved (not parsed, not rewritten)
    arr_<name>.bin       every member of the array sidecar as raw little-endian bytes in C order
    op_indptr.bin        the operator in canonical CSR form (sorted column indices, duplicates summed, explicit
    op_indices.bin       zeros removed): row pointers and column indices as little-endian int64, values as
    op_data.bin          little-endian float64
    manifest.json        dtype string and shape of every exported array, and how the operator was obtained

Nothing here decides anything. The checker recomputes every SHA-256 the certificate records over these bytes
(array hashes over dtype string + shape + bytes, the operator hash over the CSR triple, the body hash over its
own canonical serialisation of cert.json), so a byte the exporter got wrong makes a hash comparison fail; it
cannot make a check pass. What the exporter is trusted for: nothing about the numbers. It only selects which
operator to hand over: the producer bundle's `operator.npz`, or, for bundles without one, the operator
`certify_ed` itself builds (`claim_from_bundle`, or `construct.build_bitwise` for a spec-only bundle), in each
case replaced by its Hermitian part (A + A^H)/2 when the certificate records `hermitian_part`, exactly as
`certify_ed.certify` does. The checker accepts that operator only if its hash equals
`subject.operator.matrix_sha256`.

    python export/export_certs.py --index work/core/certs/index.json --certs work/core/certs --out work/core/export
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import scipy.sparse as sp


def canonical_csr(A) -> sp.csr_matrix:
    A = sp.csr_matrix(A, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    return A


def write_le(path: str, a: np.ndarray) -> None:
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    with open(path, "wb") as f:
        f.write(a.tobytes(order="C"))


def load_operator(bundle: str, cert: dict):
    """The operator certify_ed certified for this bundle (before the Hermitian part is taken), and its source."""
    npz = os.path.join(bundle, "operator.npz")
    if os.path.exists(npz):
        return sp.load_npz(npz), "operator.npz"
    from certify_ed.importers import claim_from_bundle
    claim, spec, _meta = claim_from_bundle(bundle)
    if claim.operator is not None:
        op = claim.operator
        return (op if sp.issparse(op) else sp.csr_matrix(np.asarray(op))), "claim_from_bundle (stored operator)"
    from certify_ed.construct import build_bitwise
    return build_bitwise(spec, claim.states), "construct.build_bitwise(spec)"


def export_one(case: str, cert_path: str, bundle: str, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(cert_path, os.path.join(out, "cert.json"))
    with open(cert_path, "rb") as f:
        cert = json.loads(f.read().decode("utf-8"))
    man = {"case": case, "arrays": {}, "operator": None}
    side = (cert.get("arrays") or {}).get("sidecar")
    if side:
        with np.load(os.path.join(os.path.dirname(cert_path), side)) as z:
            for name in z.files:
                a = z[name]
                dt = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
                fn = f"arr_{name}.bin"
                write_le(os.path.join(out, fn), a)
                man["arrays"][name] = {"dtype": dt.str, "shape": [int(d) for d in a.shape], "file": fn}
    if bundle:
        A, src = load_operator(bundle, cert)
        herm = cert.get("hermitian_part") is not None
        if herm:
            A = sp.csr_matrix(A)
            A = (A + A.conj().T) / 2          # certify_ed/certify.py: op = (opm + opm.conj().T) / 2
        A = canonical_csr(A)
        cplx = np.iscomplexobj(A.data)
        write_le(os.path.join(out, "op_indptr.bin"), A.indptr.astype(np.int64))
        write_le(os.path.join(out, "op_indices.bin"), A.indices.astype(np.int64))
        write_le(os.path.join(out, "op_data.bin"), A.data.astype(np.complex128 if cplx else np.float64))
        man["operator"] = {"shape": [int(A.shape[0]), int(A.shape[1])], "nnz": int(A.nnz),
                           "data_dtype": "<c16" if cplx else "<f8", "source": src, "hermitian_part": herm,
                           "indptr": "op_indptr.bin", "indices": "op_indices.bin", "data": "op_data.bin"}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index.json (list of rows with case, certificate, bundle, refused)")
    ap.add_argument("--certs", required=True, help="directory holding the certificates named in the index")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.index, encoding="utf-8") as f:
        index = json.load(f)
    n = 0
    for row in index:
        if row.get("refused"):
            continue
        bundle = row.get("bundle")
        if bundle and not os.path.isabs(bundle):
            bundle = os.path.join(os.path.dirname(os.path.abspath(a.index)), bundle)
        export_one(row["case"], os.path.join(a.certs, row["certificate"]), bundle, os.path.join(a.out, row["case"]))
        n += 1
    print(f"exported {n} certificates to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
