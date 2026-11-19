import CertifyCore.Check
/-
`cecheck kdata DIR NAME`: after the full exact check of DIR has passed, write the certificate's numbers as Lean
literals for the kernel-checked S2/S3 theorems (`CertifyCore/Kernel/`). Every literal is the exact value of a
binary64 number whose bytes the same run hashed and compared with the certificate.
-/
namespace CertifyCore

def lit (d : Dy) : String :=
  let n := d.normalize
  s!"q ({n.m}) ({n.e})"

def litList (xs : Array Dy) : String := "[" ++ ", ".intercalate (xs.toList.map lit) ++ "]"

def kernelData (b : Bundle) (name : String) : Except String String := do
  let cert := b.cert
  let stmts := ((cert.get? "statements").bind JV.arr?).getD #[]
  if (cert.path? ["claim", "kind"]).bind JV.str? != some "lowest" then throw "only claims of kind lowest"
  let some op := b.op? | throw "no operator"
  let A ← op.toCsr
  let getF := fun (nm : String) => ((b.array? nm).map ArrRec.f64s).getD #[]
  let mus ← (getF "eigenvalues").mapM (fun f => match f with | .fin d => .ok d | _ => .error "non-finite")
  let k := mus.size
  let radA ← (getF "radius").mapM (fun f => match f with | .fin d => .ok d | _ => .error "non-finite")
  let exA ← (getF "existence_radius").mapM (fun f => match f with | .fin d => .ok d | _ => .error "non-finite")
  let some X := b.array? "eigenvectors" | throw "no vectors"
  let xd ← X.f64s.mapM (fun f => match f with | .fin d => .ok d | _ => .error "non-finite")
  let recs := ((((cert.get? "enclosures").bind JV.arr?).getD #[]).map parseRec)
  if !(recs.all (·.distinct)) then throw "not all pairs distinct"
  let s2ev := evidenceOf stmts "S2.eigenpairs"
  let s3ev := evidenceOf stmts "S3.completeness"
  let shift := ((s2ev.get? "model_shift").bind jdy?).getD Dy.zero
  let some target := (s2ev.path? ["accuracy_target", "value"]).bind jdy? | throw "no target"
  let some sigma := (s3ev.get? "top_shift").bind jdy? | throw "no top shift"
  let some bnd := (s3ev.get? "next_eigenvalue_lower_bound").bind jdy? | throw "no bound"
  let some gap := (s3ev.get? "gap_lower_bound").bind jdy? | throw "no gap"
  -- scale: power of two that makes A - sigma I integral
  let mut emin : Int := sigma.normalize.e
  for v in A.data do
    let e := v.normalize.e
    if e < emin then emin := e
  let scale : Nat := (-emin).toNat
  let ngroups := recs.foldl (fun m rc => max m (rc.group + 1)) 0
  let groups := (List.range ngroups).map (fun g => (recs.filter (·.group == g)).map (·.index))
  let mut out := s!"import CertifyCore.Kernel.Defs\n\n/-! Literal data of the certificate `{name}`, written by `cecheck kdata` after its exact check passed\n(body SHA-256 {((cert.path? ["integrity", "body_sha256"]).bind JV.str?).getD "-"},\noperator SHA-256 {op.sha}). -/\n\nnamespace CertifyCore.K.Data\nopen CertifyCore.K\n\n"
  out := out ++ s!"def {name} : Cert where\n  A := ["
  let rows := (List.range A.n).map (fun i =>
    "[" ++ ", ".intercalate ((A.row i).toList.map (fun (j, v) => s!"({j}, {lit v})")) ++ "]")
  out := out ++ ",\n    ".intercalate rows ++ "]\n  pairs := ["
  let ps := (List.range k).map (fun j =>
    let rc := recs[j]!
    let xs := (List.range A.n).toArray.map (fun i => xd[i * k + j]!)
    s!"\{ mu := {lit mus[j]!}, rb := {lit (rc.residualBound.getD Dy.zero)}, ex := {lit (rc.existence.getD Dy.zero)}, rad := {lit (rc.radius.getD Dy.zero)}, radArr := {lit radA[j]!}, exArr := {lit exA[j]!}, group := {rc.group}, x := {litList xs} }")
  out := out ++ ",\n    ".intercalate ps ++ "]\n"
  out := out ++ s!"  groups := {groups.map (fun g => g.toList)}\n  shift := {lit shift}\n  target := {lit target}\n  sigma := {lit sigma}\n  bound := {lit bnd}\n  gap := {lit gap}\n  scale := {scale}\n\nend CertifyCore.K.Data\n"
  return out

end CertifyCore
