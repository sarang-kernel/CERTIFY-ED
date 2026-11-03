"""Validate the verifier against certify-ed on the corpus and the conformance fixtures, with mutation tests.

    python verifier/validation/run_validation.py [--certs verifier/validation/certs] [--python PY] [--cc CC]
                                                 [--skip-produce] [--mutants N]

Steps (verifier/README.md, section Validation):
  1. build verifier/build/cev with verifier/build.py;
  2. unless --skip-produce, run verifier/make_validation_set.py with the given interpreter (certify-ed of this
     checkout, Options(store_eigenvectors=True)) on corpus/cases and conformance/reference;
  3. run cev on every certificate with the bundle's operator.npz; every certificate must exit 0, every S2/S3
     status certify-ed FAILED must be REPRODUCED;
  4. mutation: for N certificates (default 20) three consistent forgeries each (body hash and array hashes
     recomputed so that only the semantic check can catch them, except the hash mutants) and one plain hash
     flip: (a) one stored radius lowered below the exact residual (existence_radius and residual_bound of one pair / 1e4, record and arrays),
     (b) one array hash altered with the body hash recomputed, (c) the body hash itself altered, (d) one
     non-CERTIFIED S2/S3 status set to CERTIFIED (statement and scope block); cev must exit 1 for each;
  5. write RESULTS.md and results.json next to this script.

Standard library only (zipfile, struct, json, hashlib); nothing here imports certify_ed or numpy. The .npy
writer below is a 30-line re-implementation of the format for the mutants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from build import build  # noqa: E402  (verifier/build.py)


# ----------------------------------------------------------------- npz read/write (stdlib)
def read_npz(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            data = z.read(name)
            assert data[:6] == b"\x93NUMPY"
            major = data[6]
            if major == 1:
                hl = struct.unpack("<H", data[8:10])[0]; off = 10
            else:
                hl = struct.unpack("<I", data[8:12])[0]; off = 12
            header = data[off:off + hl].decode("latin1")
            descr = header.split("'descr':")[1].split("'")[1]
            fortran = "True" in header.split("'fortran_order':")[1].split(",")[0]
            shape_txt = header.split("'shape':")[1].split("(")[1].split(")")[0]
            shape = tuple(int(s) for s in shape_txt.replace(" ", "").split(",") if s)
            out[name[:-4] if name.endswith(".npy") else name] = (descr, fortran, shape, data[off + hl:])
    return out


def npy_bytes(descr, fortran, shape, raw):
    shp = "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"
    header = "{'descr': '%s', 'fortran_order': %s, 'shape': %s, }" % (descr, "True" if fortran else "False", shp)
    pad = 64 - ((10 + len(header) + 1) % 64)
    header = header + " " * pad + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + raw


def write_npz(path, arrays):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, (descr, fortran, shape, raw) in arrays.items():
            z.writestr(name + ".npy", npy_bytes(descr, fortran, shape, raw))


def array_sha256(descr, fortran, shape, raw):
    """certify-ed's array hash: dtype.str + json.dumps(list(shape)) + C-order bytes (a Fortran-ordered 2-D
    member is transposed first)."""
    if fortran and len(shape) == 2:
        it = int("".join(ch for ch in descr if ch.isdigit()))
        r, c = shape
        raw = b"".join(raw[(j * r + i) * it:(j * r + i + 1) * it] for i in range(r) for j in range(c))
    h = hashlib.sha256()
    h.update(descr.encode()); h.update(json.dumps(list(shape)).encode()); h.update(raw)
    return h.hexdigest()


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def reseal(body):
    b = dict(body); b.pop("integrity", None)
    digest = hashlib.sha256(canonical(b).encode("utf-8")).hexdigest()
    b["integrity"] = {"algorithm": "sha256", "canonicalization": "certify-ed-canonical-json-1", "body_sha256": digest}
    return b


def f64(raw, i):
    return struct.unpack_from("<d", raw, 8 * i)[0]


def set_f64(raw, i, v):
    raw = bytearray(raw); struct.pack_into("<d", raw, 8 * i, v); return bytes(raw)


# ----------------------------------------------------------------- mutants
def mutate_radius(body, arrays):
    """Consistent forgery: the existence_radius and residual_bound of the pair with the largest residual bound
    are divided by 1e4 in the enclosure record and in the arrays; hashes recomputed. Only the recomputation from
    the operator and the stored vectors (R2/R3) can catch it. (A bound lowered by a small factor can stay a valid
    upper bound of the exact residual, since certify-ed's bound carries rounding terms above the computed
    residual; such a forgery is not an error and the verifier does not flag it.)"""
    body = json.loads(json.dumps(body)); arrays = dict(arrays)
    d, f, s, raw_res = arrays["residual_bound"]
    i = max(range(s[0]), key=lambda j: f64(raw_res, j))
    rec = next(r for r in body["enclosures"] if r["index"] == i)
    for key in ("existence_radius", "residual_bound"):
        old = rec[key]
        new = old / 1e4
        rec[key] = new
        d2, f2, s2, raw = arrays[key]
        shift = f64(raw, i) - old   # model shift carried by the stored array (0 for residual_bound)
        arrays[key] = (d2, f2, s2, set_f64(raw, i, new + max(shift, 0.0)))
    for key in arrays:
        body["arrays"]["sha256"][key] = array_sha256(*arrays[key])
    return reseal(body), arrays, f"pair {i}: existence_radius and residual_bound / 1e4 (record + arrays, hashes resealed)"


def mutate_array_hash(body, arrays):
    body = json.loads(json.dumps(body))
    key = "radius" if "radius" in body["arrays"]["sha256"] else sorted(body["arrays"]["sha256"])[0]
    h = body["arrays"]["sha256"][key]
    body["arrays"]["sha256"][key] = ("0" if h[0] != "0" else "1") + h[1:]
    return reseal(body), arrays, f"arrays.sha256[{key}] first hex digit changed, body resealed"


def mutate_body_hash(body, arrays):
    body = json.loads(json.dumps(body))
    h = body["integrity"]["body_sha256"]
    body["integrity"]["body_sha256"] = h[:-1] + ("0" if h[-1] != "0" else "1")
    return body, arrays, "integrity.body_sha256 last hex digit changed"


def mutate_status(body, arrays):
    body = json.loads(json.dumps(body))
    target = None
    for sid in ("S2.eigenpairs", "S3.completeness"):
        st = next((s for s in body["statements"] if s["id"] == sid), None)
        if st and st["status"] != "CERTIFIED" and st["status"] != "NOT_CHECKED":
            target = st; break
    if target is None:
        return None
    old = target["status"]
    target["status"] = "CERTIFIED"
    for e in body["scope"]["in_scope"]:
        if e["id"] == target["id"]:
            e["status"] = "CERTIFIED"
    return reseal(body), arrays, f"{target['id']} {old} -> CERTIFIED (statement + scope block, body resealed)"


MUTATIONS = [("radius", mutate_radius), ("array_hash", mutate_array_hash), ("body_hash", mutate_body_hash),
             ("status", mutate_status)]


# ----------------------------------------------------------------- driver
def run_cev(cev, cert, operator, json_out):
    cmd = [cev, cert, "--quiet", "--json", json_out] + (["--operator", operator] if operator else [])
    env = dict(os.environ)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    summ = json.load(open(json_out)) if r.returncode in (0, 1) and os.path.exists(json_out) else None
    return r.returncode, summ, r.stderr.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--certs", default=os.path.join(HERE, "certs"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cc")
    ap.add_argument("--skip-produce", action="store_true")
    ap.add_argument("--mutants", type=int, default=20)
    a = ap.parse_args()

    cev = build(a.cc, quiet=True)
    if not a.skip_produce:
        env = dict(os.environ); env.pop("PYTHONSAFEPATH", None)
        r = subprocess.run([a.python, os.path.join(ROOT, "verifier", "make_validation_set.py"), "--out", a.certs],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
            return 2
    index = json.load(open(os.path.join(a.certs, "index.json")))
    outdir = os.path.join(HERE, "reports"); os.makedirs(outdir, exist_ok=True)

    rows = []
    n_pass = n_fail = n_err = 0
    n_failed_statuses = n_reproduced = 0
    not_reproduced = []
    for row in index:
        if row["refused"]:
            rows.append({"case": row["case"], "certify_ed": "refused", "verifier": "n/a"}); continue
        cert = os.path.join(a.certs, row["certificate"])
        op = os.path.join(ROOT, row["bundle"], "operator.npz")
        rc, summ, err = run_cev(cev, cert, op if os.path.exists(op) else None, os.path.join(outdir, row["case"] + ".json"))
        if summ is None:
            n_err += 1; rows.append({"case": row["case"], "verifier": f"error: {err[-200:]}"}); continue
        n_pass += rc == 0; n_fail += rc == 1
        rep = {c["id"]: c["status"] for c in summ["checks"] if c["status"] in ("REPRODUCED", "NOT_REPRODUCED")}
        for sid in ("S2.eigenpairs", "S3.completeness"):
            if row["statuses"].get(sid) == "FAILED":
                n_failed_statuses += 1
                key = sid.split(".")[0] + ".failed"
                if rep.get(key) == "REPRODUCED":
                    n_reproduced += 1
                else:
                    not_reproduced.append((row["case"], sid, rep.get(key, "no record")))
        rows.append({"case": row["case"], "origin": row["origin"], "statuses": row["statuses"], "verifier_exit": rc,
                     "counts": summ["counts"], "fails": [c["id"] for c in summ["checks"] if c["status"] == "FAIL"],
                     "skips": [c["id"] for c in summ["checks"] if c["status"] == "SKIP"], "reproduced": rep})

    # mutants: the first N certificates that admit a status mutation (non-CERTIFIED S2 or S3), corpus order
    mut_dir = os.path.join(HERE, "mutants"); os.makedirs(mut_dir, exist_ok=True)
    candidates = [r for r in index if not r["refused"] and any(
        r["statuses"].get(s) not in ("CERTIFIED", "NOT_CHECKED", None) for s in ("S2.eigenpairs", "S3.completeness"))]
    candidates = candidates[:a.mutants]
    mut_rows = []; caught = {m: 0 for m, _ in MUTATIONS}; total = {m: 0 for m, _ in MUTATIONS}
    for row in candidates:
        cert = os.path.join(a.certs, row["certificate"])
        body = json.load(open(cert, encoding="utf-8"))
        arrays = read_npz(os.path.join(a.certs, body["arrays"]["sidecar"]))
        op = os.path.join(ROOT, row["bundle"], "operator.npz")
        for mname, fn in MUTATIONS:
            res = fn(body, arrays)
            if res is None:
                continue
            mbody, marrays, what = res
            stem = os.path.join(mut_dir, f"{row['case']}__{mname}")
            mbody["arrays"]["sidecar"] = os.path.basename(stem) + ".arrays.npz"
            mbody = reseal(mbody) if mname != "body_hash" else mbody
            with open(stem + ".json", "w", encoding="utf-8") as f:
                f.write(canonical(mbody))
            write_npz(stem + ".arrays.npz", marrays)
            rc, summ, err = run_cev(cev, stem + ".json", op if os.path.exists(op) else None, stem + ".cev.json")
            total[mname] += 1
            fails = [c["id"] for c in summ["checks"] if c["status"] == "FAIL"] if summ else []
            ok = rc == 1 and fails
            caught[mname] += bool(ok)
            mut_rows.append({"case": row["case"], "mutation": mname, "what": what, "exit": rc, "caught_by": fails,
                             "error": err[-200:] if summ is None else ""})

    results = {"verifier": os.path.relpath(cev, ROOT), "certificates": len(index),
               "refused_by_certify_ed": sum(r["refused"] for r in index), "verified": n_pass + n_fail + n_err,
               "verifier_pass": n_pass, "verifier_fail": n_fail, "verifier_error": n_err,
               "failed_statuses_S2_S3": n_failed_statuses, "reproduced": n_reproduced, "not_reproduced": not_reproduced,
               "mutants": {m: {"total": total[m], "caught": caught[m]} for m, _ in MUTATIONS},
               "rows": rows, "mutant_rows": mut_rows}
    with open(os.path.join(HERE, "results.json"), "w", newline="\n") as f:
        json.dump(results, f, indent=1, sort_keys=True)
        f.write("\n")
    # markdown
    L = ["# Verifier validation results", "",
         f"Certificates produced by certify-ed: {len(index)} ({results['refused_by_certify_ed']} refused by certify-ed, no certificate written; "
         f"{results['verified']} verified).", "",
         f"- verifier exit 0 (every check passed): {n_pass}", f"- verifier exit 1 (a check failed): {n_fail}",
         f"- verifier error (unreadable input): {n_err}",
         f"- S2/S3 statuses certify-ed FAILED: {n_failed_statuses}; reproduced by the verifier from the stored data: {n_reproduced}; "
         f"not reproduced: {len(not_reproduced)}", ""]
    L += ["## Mutation tests", "", "| mutation | mutants | caught (exit 1 with a FAIL check) |", "|---|---|---|"]
    for m, _ in MUTATIONS:
        L.append(f"| {m} | {total[m]} | {caught[m]} |")
    L += ["", "| case | mutation | what | exit | caught by |", "|---|---|---|---|---|"]
    for r in mut_rows:
        L.append(f"| {r['case']} | {r['mutation']} | {r['what']} | {r['exit']} | {', '.join(r['caught_by']) or '-'} |")
    L += ["", "## Per certificate", "", "| case | certify-ed S1 S2 S3 S4 S8 | verifier exit | pass/fail/skip | reproduced | skipped |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "statuses" not in r:
            L.append(f"| {r['case']} | {r.get('certify_ed', r.get('verifier'))} | - | - | - | - |"); continue
        st = " ".join(v[0] for v in r["statuses"].values())
        c = r["counts"]
        L.append(f"| {r['case']} | {st} | {r['verifier_exit']} | {c['pass']}/{c['fail']}/{c['skip']} | "
                 f"{', '.join(k + '=' + v for k, v in r['reproduced'].items()) or '-'} | {', '.join(r['skips']) or '-'} |")
    with open(os.path.join(HERE, "RESULTS.md"), "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"verified {results['verified']}: pass {n_pass}, fail {n_fail}, error {n_err}; FAILED statuses {n_failed_statuses}, "
          f"reproduced {n_reproduced}; mutants " + ", ".join(f"{m} {caught[m]}/{total[m]}" for m, _ in MUTATIONS))
    bad = n_fail or n_err or not_reproduced or any(caught[m] != total[m] for m, _ in MUTATIONS)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
