import CertifyCore.Json
import CertifyCore.Dyadic
import CertifyCore.Sha256
import CertifyCore.Inertia
/-
Reading one exported certificate directory (written by `export/export_certs.py`): the certificate JSON as raw
bytes, the sidecar arrays as raw little-endian bytes with their dtype and shape, and the operator's canonical
CSR arrays. The hash functions here reproduce certify-ed's serialisations byte for byte:
`array_sha256` = SHA-256(dtype string ++ json.dumps(list(shape)) ++ C-order bytes) and
`matrix_sha256` = SHA-256(json.dumps(list(shape)) ++ hex(array_sha256(indptr as <i8)) ++ hex(array_sha256(indices
as <i8)) ++ hex(array_sha256(data as <f8))) (`certify_ed/certificate_v2.py`).
-/
namespace CertifyCore

structure ArrRec where
  name : String
  dtype : String
  shape : Array Nat
  raw : ByteArray
  deriving Inhabited

structure OpRec where
  shape : Array Nat
  nnz : Nat
  dataDtype : String
  source : String
  hermitianPart : Bool
  indptr : ByteArray
  indices : ByteArray
  data : ByteArray
  deriving Inhabited

structure Bundle where
  dir : String
  certBytes : ByteArray
  cert : JV
  arrays : Array ArrRec
  op? : Option OpRec
  deriving Inhabited

/-- `json.dumps(list(shape))`: "[1024, 5]". -/
def shapeStr (s : Array Nat) : String := "[" ++ ", ".intercalate (s.toList.map toString) ++ "]"

def arraySha (dtype : String) (shape : Array Nat) (raw : ByteArray) : String :=
  Sha256.hex (dtype.toUTF8 ++ (shapeStr shape).toUTF8 ++ raw)

def OpRec.sha (o : OpRec) : String :=
  let n := o.shape[0]!
  let h1 := arraySha "<i8" #[n + 1] o.indptr
  let h2 := arraySha "<i8" #[o.nnz] o.indices
  let h3 := arraySha o.dataDtype #[o.nnz] o.data
  Sha256.hex ((shapeStr o.shape).toUTF8 ++ h1.toUTF8 ++ h2.toUTF8 ++ h3.toUTF8)

def jNat? (j : JV) : Option Nat := j.num? >>= String.toNat?

def jShape? (j : JV) : Option (Array Nat) := do
  let xs ← j.arr?
  xs.mapM jNat?

def readJson (path : String) : IO JV := do
  let b ← IO.FS.readBinFile path
  match Json.parse b with
  | .ok v => return v
  | .error e => throw (IO.userError s!"{path}: {e}")

def loadBundle (dir : String) : IO Bundle := do
  let certBytes ← IO.FS.readBinFile (dir ++ "/cert.json")
  let cert ← match Json.parse certBytes with
    | .ok v => pure v
    | .error e => throw (IO.userError s!"cert.json: {e}")
  let man ← readJson (dir ++ "/manifest.json")
  let mut arrays := #[]
  match man.get? "arrays" with
  | some (JV.obj kvs) =>
    for (name, rec_) in kvs do
      let dt := ((rec_.get? "dtype").bind JV.str?).getD ""
      let sh := ((rec_.get? "shape").bind jShape?).getD #[]
      let fn := ((rec_.get? "file").bind JV.str?).getD ""
      let raw ← IO.FS.readBinFile (dir ++ "/" ++ fn)
      arrays := arrays.push { name, dtype := dt, shape := sh, raw }
  | _ => pure ()
  let op? ← match man.get? "operator" with
    | some o@(JV.obj _) => do
      let f := fun k => ((o.get? k).bind JV.str?).getD ""
      let ip ← IO.FS.readBinFile (dir ++ "/" ++ f "indptr")
      let ix ← IO.FS.readBinFile (dir ++ "/" ++ f "indices")
      let da ← IO.FS.readBinFile (dir ++ "/" ++ f "data")
      pure (some { shape := ((o.get? "shape").bind jShape?).getD #[], nnz := ((o.get? "nnz").bind jNat?).getD 0,
                   dataDtype := f "data_dtype", source := f "source",
                   hermitianPart := ((o.get? "hermitian_part").bind JV.bool?).getD false,
                   indptr := ip, indices := ix, data := da : OpRec })
    | _ => pure none
  return { dir, certBytes, cert, arrays, op? }

def Bundle.array? (b : Bundle) (name : String) : Option ArrRec := b.arrays.find? (·.name == name)

/-- The float64 members of an array as exact values. -/
def ArrRec.f64s (a : ArrRec) : Array F64 := Id.run do
  let n := a.raw.size / 8
  let mut out := Array.mkEmpty n
  for i in [0:n] do
    out := out.push (F64.ofBits (readU64 a.raw (8 * i)))
  return out

/-- Decode the operator into an exact CSR matrix; checks the structural invariants of canonical CSR. -/
def OpRec.toCsr (o : OpRec) : Except String Csr := do
  if o.shape.size != 2 then throw "operator shape is not 2-D"
  let n := o.shape[0]!
  if o.shape[1]! != n then throw "operator is not square"
  if o.dataDtype != "<f8" then throw s!"operator data dtype {o.dataDtype} (only real <f8 is re-checked)"
  if o.indptr.size != 8 * (n + 1) || o.indices.size != 8 * o.nnz || o.data.size != 8 * o.nnz then
    throw "operator array sizes do not match shape and nnz"
  let mut indptr := Array.mkEmpty (n + 1)
  for i in [0:n+1] do
    let v := readI64 o.indptr (8 * i)
    if v < 0 then throw "negative row pointer"
    indptr := indptr.push v.toNat
  if indptr[0]! != 0 || indptr[n]! != o.nnz then throw "row pointers do not span the entries"
  let mut indices := Array.mkEmpty o.nnz
  let mut data := Array.mkEmpty o.nnz
  for t in [0:o.nnz] do
    let c := readI64 o.indices (8 * t)
    if c < 0 || c ≥ n then throw "column index out of range"
    indices := indices.push c.toNat
    match F64.ofBits (readU64 o.data (8 * t)) with
    | .fin d => data := data.push d
    | _ => throw "non-finite operator entry"
  for i in [0:n] do
    if indptr[i]! > indptr[i+1]! then throw "row pointers decrease"
    for t in [indptr[i]!:indptr[i+1]!] do
      if t > indptr[i]! && indices[t]! ≤ indices[t-1]! then throw "column indices not strictly increasing (not canonical CSR)"
      if data[t]!.m == 0 then throw "explicit zero (not canonical CSR)"
  return { n, indptr, indices, data }

/-- Exact symmetry: every stored (i, j, v) has a stored (j, i, v). -/
def Csr.isSymmetric (A : Csr) : Bool := Id.run do
  for i in [0:A.n] do
    for t in [A.indptr[i]!:A.indptr[i+1]!] do
      let j := A.indices[t]!
      let v := A.data[t]!
      let mut found := false
      for s in [A.indptr[j]!:A.indptr[j+1]!] do
        if A.indices[s]! == i then
          found := Dy.beq A.data[s]! v
      if !found then return false
  return true

/-- Exact infinity norm max_i sum_j |a_ij| (an upper bound of the spectral norm of a symmetric matrix). -/
def Csr.infNorm (A : Csr) : Dy := Id.run do
  let mut best := Dy.zero
  for i in [0:A.n] do
    let mut s := Dy.zero
    for t in [A.indptr[i]!:A.indptr[i+1]!] do
      s := s + A.data[t]!.abs
    best := Dy.max best s
  return best

/-- y = A x, exact. -/
def Csr.mulVec (A : Csr) (x : Array Dy) : Array Dy := Id.run do
  let mut y := Array.mkEmpty A.n
  for i in [0:A.n] do
    let mut s := Dy.zero
    for t in [A.indptr[i]!:A.indptr[i+1]!] do
      s := s + A.data[t]! * x[A.indices[t]!]!
    y := y.push s
  return y

end CertifyCore
