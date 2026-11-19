"""One-command exact re-check of a certify-ed certificate with `cecheck` (roadmap item N-04, trust level N).

    python Verify/cecheck_bundle.py CERT.json (--bundle BUNDLE_DIR | --operator OPERATOR.npz) [--arrays SIDECAR.npz]
                                    [--json VERDICT.json] [--out-dir DIR] [--exe CECHECK] [--timeout S] [--quiet]

The script exports the certificate the way `export/export_certs.py` does (the certificate JSON byte for byte, every
member of the array sidecar as raw little-endian bytes, the operator in canonical CSR form, replaced by its Hermitian
part when the certificate records `hermitian_part`), runs `.lake/build/bin/cecheck` on the exported directory, and
turns cecheck's list of checks into one verdict per statement of the certificate:

    CONFIRMED        a CERTIFIED status that the exact checks support (all of cecheck's checks for it passed)
    NOT_CONFIRMED    a CERTIFIED status for which a check failed
    REPRODUCED       a FAILED status whose recorded reason cecheck finds on the stored numbers
    REFUTED          REPRODUCED, and an exact inertia count contradicts the claim itself
    NOT_REPRODUCED   a FAILED status whose reason cecheck does not find
    NO_CLAIM         INCONCLUSIVE or NOT_CHECKED: nothing to confirm (the exact checks are listed all the same)
    NOT_RECHECKED    a statement cecheck does not check (S1, S4, S5, S6, S7; see CORE.md section 3)

plus `integrity` (the hash and consistency checks H1 to H5, C1 to C4, C7) and `kato_temple` (the K checks). Every
verdict has level N: cecheck is a compiled program whose source is in CertifyCore and whose correctness is not a
kernel theorem. The exit code is cecheck's (0 no check failed, 1 a check failed, 2 the input could not be read); the
script exits with 2 as well when it cannot export or run cecheck.

What the exporter is trusted for is said in `export/export_certs.py`: nothing about the numbers, because cecheck
recomputes every SHA-256 the certificate records over the exported bytes. With --operator the file must hold the
operator the certificate was issued for; cecheck accepts it only if its hash equals subject.operator.matrix_sha256.
A bundle without operator.npz needs certify_ed importable (PYTHONPATH), as for export_certs.py.

The verdict JSON (--json) holds the certificate's path, SHA-256 and recorded statuses, cecheck's path, SHA-256,
exit code and counts, the SHA-256 of cecheck's own report, and the per-statement verdicts with the checks behind
each. Its fields `verdict`, `level` and `statements` are the ones `certify-ed verify --external` copies into its
external-verification record:

    certify-ed verify CERT.json --bundle BUNDLE_DIR --external
        "python <repo>/lean/Verify/cecheck_bundle.py {certificate} --bundle {bundle} --json {report}"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, ".lake", "build", "bin", "cecheck.exe" if os.name == "nt" else "cecheck")
sys.path.insert(0, os.path.join(ROOT, "export"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from export_certs import canonical_csr, load_operator, write_le  # noqa: E402

NOT_RECHECKED = {"S1": "the model shift of S1 is taken from the record (CORE.md section 2)",
                 "S4": "eigenspace bounds are not re-checked", "S5": "corroboration is not re-checked",
                 "S6": "the exact re-check of certify-ed is not repeated as such",
                 "S7": "thermodynamic intervals are not re-checked"}
INTEGRITY_PREFIXES = ("H1.", "H2.", "H3.", "H4.", "H5.", "C1.", "C2.", "C3.", "C4.", "C7.")
S2_PREFIXES = ("E1.", "E2.", "E3.", "E4.", "E5.", "G", "S2.")
LEVEL = "N"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(cert_path: str, out: str, bundle: str | None = None, operator: str | None = None,
           arrays: str | None = None, case: str | None = None) -> dict:
    """Write the directory cecheck reads. Byte for byte the output of export_certs.export_one for the same
    certificate and bundle; --operator and --arrays only change where the operator and the sidecar are read."""
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(cert_path, os.path.join(out, "cert.json"))
    with open(cert_path, "rb") as f:
        cert = json.loads(f.read().decode("utf-8"))
    if case is None:
        base = os.path.basename(cert_path)
        case = base[:-5] if base.endswith(".json") else base
    man = {"case": case, "arrays": {}, "operator": None}
    side = (cert.get("arrays") or {}).get("sidecar")
    side_path = arrays or (os.path.join(os.path.dirname(cert_path), side) if side else None)
    if side_path:
        with np.load(side_path) as z:
            for name in z.files:
                a = z[name]
                dt = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
                fn = f"arr_{name}.bin"
                write_le(os.path.join(out, fn), a)
                man["arrays"][name] = {"dtype": dt.str, "shape": [int(d) for d in a.shape], "file": fn}
    if bundle or operator:
        if operator:
            A, src = sp.load_npz(operator), f"operator file {os.path.basename(operator)}"
        else:
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


def _sel(checks, pred):
    return [{"id": c["id"], "status": c["status"]} for c in checks if pred(c["id"])]


def _by_id(checks):
    return {c["id"]: c["status"] for c in checks}


def statement_verdicts(cert: dict, rep: dict) -> dict:
    """Per-statement verdicts from cecheck's report `rep` for the certificate body `cert` (see the module text)."""
    checks = rep.get("checks") or []
    st = _by_id(checks)
    integ = _sel(checks, lambda i: i.startswith(INTEGRITY_PREFIXES))
    integ_ok = all(c["status"] != "FAIL" for c in integ) and bool(integ)
    out = {"integrity": {"verdict": "PASS" if integ_ok else "FAIL", "checks": integ, "level": LEVEL}}
    for s in cert.get("statements") or []:
        sid, rec = s.get("id", ""), s.get("status")
        short = sid.split(".")[0]
        v = {"recorded": rec, "level": LEVEL}
        if short == "S2":
            cs = _sel(checks, lambda i: i.startswith(S2_PREFIXES))
            exact_ok = all(c["status"] != "FAIL" for c in cs if not c["id"].startswith("S2.")) and \
                any(c["id"].startswith(("E1.", "E2.")) for c in cs)
            if rec == "CERTIFIED":
                ok = integ_ok and st.get("S2.status.exact") == "PASS" and st.get("S2.status.rule") == "PASS" and \
                    all(c["status"] != "FAIL" for c in cs)
                v["verdict"] = "CONFIRMED" if ok else "NOT_CONFIRMED"
            elif rec == "FAILED":
                f = st.get("S2.failed")
                v["verdict"] = ("REFUTED" if st.get("S2.failed.refuted") == "REFUTED" else "REPRODUCED") \
                    if f == "REPRODUCED" else "NOT_REPRODUCED"
            else:
                v["verdict"] = "NO_CLAIM"
            v["exact_radius_checks_hold"] = exact_ok
            v["checks"] = cs
        elif short == "S3":
            cs = _sel(checks, lambda i: i.startswith("S3."))
            if rec == "CERTIFIED":
                ok = integ_ok and st.get("S3.status.exact") == "PASS" and st.get("S3.status.rule", "PASS") == "PASS"
                v["verdict"] = "CONFIRMED" if ok else "NOT_CONFIRMED"
            elif rec == "FAILED":
                v["verdict"] = "REFUTED" if st.get("S3.failed") == "REFUTED" else "NOT_REPRODUCED"
            else:
                v["verdict"] = "NO_CLAIM"
            v["checks"] = cs
        elif short == "S8":
            cs = _sel(checks, lambda i: i.startswith("S8."))
            if rec == "CERTIFIED":
                ok = integ_ok and bool(cs) and all(c["status"] == "PASS" for c in cs)
                v["verdict"] = "CONFIRMED" if ok else "NOT_CONFIRMED"
            else:
                v["verdict"] = "NO_CLAIM"
            v["checks"] = cs
            v["note"] = "every cluster bracket recorded as CERTIFIED is re-counted exactly"
        else:
            v["verdict"] = "NOT_RECHECKED"
            v["note"] = NOT_RECHECKED.get(short, "not checked by cecheck")
        out[sid] = v
    kt = _sel(checks, lambda i: i.startswith("K") and i.endswith(".kato_temple"))
    if kt:
        out["kato_temple"] = {"verdict": "CONFIRMED" if all(c["status"] == "PASS" for c in kt) else "NOT_CONFIRMED",
                              "checks": kt, "level": LEVEL}
    return out


def run(cert_path: str, bundle=None, operator=None, arrays=None, out_dir=None, exe=EXE, timeout=None,
        quiet=False) -> tuple[int, dict]:
    cert_path = os.path.abspath(cert_path)
    with open(cert_path, "rb") as f:
        cert = json.loads(f.read().decode("utf-8"))
    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="cecheck_bundle_")
        out_dir = os.path.join(tmp, "export")
    res = {"tool": "Verify/cecheck_bundle.py", "level": LEVEL,
           "certificate": {"path": cert_path, "sha256": sha256_file(cert_path),
                           "statuses": {s.get("id"): s.get("status") for s in cert.get("statements") or []}}}
    try:
        man = export(cert_path, out_dir, bundle=bundle, operator=operator, arrays=arrays)
        res["export"] = {"dir": os.path.abspath(out_dir), "operator": man["operator"],
                         "arrays": sorted(man["arrays"])}
        rep_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "cecheck_report.json")
        cmd = ([sys.executable, exe] if exe.endswith(".py") else [exe]) + [out_dir, "--json", rep_path, "--quiet"]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            code, err = r.returncode, (r.stderr or "")[-2000:]
        except subprocess.TimeoutExpired:
            code, err = 2, f"cecheck exceeded {timeout} s"
        res["cecheck"] = {"exe": os.path.abspath(exe), "sha256": sha256_file(exe) if os.path.isfile(exe) else None,
                          "exit_code": code, "seconds": round(time.time() - t0, 1), "stderr_tail": err}
        if code in (0, 1) and os.path.exists(rep_path):
            with open(rep_path, encoding="utf-8") as f:
                rep = json.load(f)
            res["cecheck"].update(verdict=rep.get("verdict"), counts=rep.get("counts"),
                                  report_sha256=sha256_file(rep_path), exact_counts=rep.get("exact_counts"))
            res["statements"] = statement_verdicts(cert, rep)
            res["verdict"] = "PASS" if code == 0 else "FAIL"
        else:
            res["verdict"] = "ERROR"
            code = 2
    except Exception as e:  # noqa: BLE001 - every failure to export or run is an ERROR verdict, never a pass
        res["verdict"] = "ERROR"
        res["error"] = f"{type(e).__name__}: {e}"
        code = 2
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if not quiet:
        print(f"{res['verdict']}  {cert_path}  (cecheck exit code {res.get('cecheck', {}).get('exit_code')}, level N)")
        for sid, v in (res.get("statements") or {}).items():
            print(f"  {sid:24s} {str(v.get('recorded', '')):13s} {v['verdict']}")
        if res.get("error"):
            print(f"  error: {res['error']}")
    return code, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("certificate", help="certificate JSON written by certify-ed")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bundle", help="producer bundle directory the certificate was issued for")
    g.add_argument("--operator", help="operator file (scipy .npz) the certificate was issued for")
    ap.add_argument("--arrays", help="array sidecar (default: arrays.sidecar next to the certificate)")
    ap.add_argument("--json", help="write the per-statement verdicts here")
    ap.add_argument("--out-dir", dest="out_dir", help="keep the exported directory here (default: a temporary one)")
    ap.add_argument("--exe", default=EXE, help="the cecheck binary (default: .lake/build/bin/cecheck)")
    ap.add_argument("--timeout", type=float, help="seconds before cecheck is stopped (verdict ERROR)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    code, res = run(a.certificate, bundle=a.bundle, operator=a.operator, arrays=a.arrays, out_dir=a.out_dir,
                    exe=a.exe, timeout=a.timeout, quiet=a.quiet)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, sort_keys=True)
            f.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
