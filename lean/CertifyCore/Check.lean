import CertifyCore.Load
/-
The checks of `cecheck`. Every comparison is exact: stored binary64 numbers are decoded to dyadic rationals
(`F64.ofBits`, `decToF64`), every sum, product and square is exact, square roots are never taken (an
inequality r >= sqrt(q) is checked as r >= 0 and r^2 >= q), and where a theorem's radius contains square roots
of computed quantities the checker uses its own rational upper bounds with 160-bit precision (`sqrtUp`,
`sqrtDown`). Eigenvalue counts are exact inertias of A - sigma I (`CertifyCore.Inertia`).
-/
namespace CertifyCore

inductive St where
  | pass | fail | skip | info | reproduced | refuted | notReproduced
  deriving BEq, Inhabited

def St.str : St → String
  | .pass => "PASS" | .fail => "FAIL" | .skip => "SKIP" | .info => "INFO"
  | .reproduced => "REPRODUCED" | .refuted => "REFUTED" | .notReproduced => "NOT_REPRODUCED"

structure Check where
  id : String
  st : St
  detail : String

/-! ### exact helpers -/

def isqrt (n : Nat) : Nat := Id.run do
  if n < 2 then return n
  let mut x := 2 ^ ((Nat.log2 n) / 2 + 1)
  repeat
    let y := (x + n / x) / 2
    if y ≥ x then break
    x := y
  return x

def precBits : Nat := 160

/-- A rational s with s >= sqrt(q) (q >= 0). -/
def sqrtUp (q : Rat) : Rat :=
  let N := (q * ((4 : Rat) ^ precBits)).floor.toNat
  mkRat ((isqrt N + 1 : Nat) : Int) (2 ^ precBits)

/-- A rational s with 0 <= s <= sqrt(q) (q >= 0). -/
def sqrtDown (q : Rat) : Rat :=
  if q ≤ 0 then 0 else
  let N := (q * ((4 : Rat) ^ precBits)).floor.toNat
  mkRat ((isqrt N : Nat) : Int) (2 ^ precBits)

/-- r >= sqrt(q) exactly, for r rational and q >= 0. -/
def geSqrt (r q : Rat) : Bool := r ≥ 0 && r * r ≥ q

/-- A short decimal for reports (never used in a decision). -/
def ratSci (q : Rat) : String := (Dy.toSci ⟨(q * (2 : Rat) ^ 200).floor, -200⟩ 4)

def dySum (xs : Array Dy) : Dy := xs.foldl Dy.add Dy.zero
def dot (x y : Array Dy) : Dy := Id.run do
  let mut s := Dy.zero
  for i in [0:x.size] do
    s := s + x[i]! * y[i]!
  return s

def jdy? (j : JV) : Option Dy := do
  let l ← j.num?
  match ← decToF64 l with
  | .fin d => some d
  | _ => none

def jrat? (j : JV) : Option Rat := (jdy? j).map Dy.toRat

/-! ### the run -/

structure Run where
  b : Bundle
  checks : IO.Ref (Array Check)
  counts : IO.Ref (Array (Rat × Inertia × Nat))   -- memoised exact counts: shift, inertia, largest block
  countLog : IO.Ref (Array String)

def Run.add (r : Run) (id : String) (st : St) (detail : String) : IO Unit :=
  r.checks.modify (·.push ⟨id, st, detail⟩)

def Run.req (r : Run) (id : String) (ok : Bool) (detail : String) : IO Bool := do
  r.add id (if ok then .pass else .fail) detail
  return ok

/-- Exact inertia of A - t I, memoised. -/
def Run.count (r : Run) (A : Csr) (gens : Array (Array Nat)) (t : Dy) : IO Inertia := do
  let tq := t.toRat
  -- every eigenvalue lies in [-||A||_inf, ||A||_inf] (Gershgorin): no factorisation outside that interval
  let ai := A.infNorm.toRat
  if tq < -ai then return ⟨0, 0, A.n⟩
  if tq > ai then return ⟨A.n, 0, 0⟩
  for (q, i, _) in (← r.counts.get) do
    if q == tq then return i
  let t0 ← IO.monoMsNow
  let (res, _, big) ← IO.ofExcept (A.inertiaShiftSym gens t)
  let t1 ← IO.monoMsNow
  r.counts.modify (·.push (tq, res, big))
  r.countLog.modify (·.push s!"shift {t.toSci 17}: negative {res.neg}, zero {res.zero}, positive {res.pos}; symmetry generators {gens.size}, largest block {big}, {t1 - t0} ms")
  return res

def statusOf (stmts : Array JV) (sid : String) : Option String :=
  (stmts.find? (fun s => (s.get? "id").bind JV.str? == some sid)).bind (fun s => (s.get? "status").bind JV.str?)

def evidenceOf (stmts : Array JV) (sid : String) : JV :=
  ((stmts.find? (fun s => (s.get? "id").bind JV.str? == some sid)).bind (·.get? "evidence")).getD JV.null

structure Rec where
  index : Nat
  claimed : Option Dy
  radius : Option Dy
  lower : Option Dy
  upper : Option Dy
  existence : Option Dy
  residualBound : Option Dy
  group : Nat
  method : String
  distinct : Bool
  deriving Inhabited

def parseRec (j : JV) : Rec :=
  { index := ((j.get? "index").bind jNat?).getD 0, claimed := (j.get? "claimed").bind jdy?,
    radius := (j.get? "radius").bind jdy?, lower := (j.get? "lower").bind jdy?, upper := (j.get? "upper").bind jdy?,
    existence := (j.get? "existence_radius").bind jdy?, residualBound := (j.get? "residual_bound").bind jdy?,
    group := ((j.get? "group").bind jNat?).getD 0, method := ((j.get? "method").bind JV.str?).getD "",
    distinct := ((j.get? "distinct").bind JV.bool?).getD false }

/-- Main check of one exported certificate. Returns the checks and the count log. -/
def checkBundle (b : Bundle) : IO (Array Check × Array String) := do
  let r : Run := { b, checks := ← IO.mkRef #[], counts := ← IO.mkRef #[], countLog := ← IO.mkRef #[] }
  let cert := b.cert
  -- H1: body hash over the canonical serialisation without `integrity`
  let recBody := (cert.path? ["integrity", "body_sha256"]).bind JV.str?
  let hb := Sha256.hex (Json.canonical (cert.erase "integrity"))
  discard <| r.req "H1.body_sha256" (some hb == recBody) s!"computed {hb}"
  r.add "H1.canonical_file" .info (if Json.canonical cert == b.certBytes then "the file is the canonical serialisation of its own content" else "the file is not in canonical form (allowed; the hash is over the canonical form)")
  -- H2: arrays
  let shaRec := ((cert.path? ["arrays", "sha256"]).getD JV.null)
  let listed : Array String := match shaRec with
    | JV.obj kvs => kvs.map (·.1)
    | _ => #[]
  let mut h2ok := true
  for a in b.arrays do
    let recH := (shaRec.get? a.name).bind JV.str?
    let recDt := (cert.path? ["arrays", "dtypes", a.name]).bind JV.str?
    let recSh := (cert.path? ["arrays", "shapes", a.name]).bind jShape?
    let h := arraySha a.dtype a.shape a.raw
    let ok := some h == recH && recDt == some a.dtype && recSh == some a.shape &&
      a.raw.size == 8 * a.shape.foldl (· * ·) 1
    if !ok then h2ok := false
    discard <| r.req s!"H2.{a.name}" ok (if ok then "hash, dtype and shape agree" else s!"computed {h}, recorded {recH.getD "-"}")
  for nm in listed do
    if !(b.arrays.any (·.name == nm)) then
      h2ok := false
      discard <| r.req s!"H2.{nm}" false "listed in the certificate but not present"
  -- H3: claimed eigenvector hash
  match (cert.path? ["claim", "eigenvectors_sha256"]).bind JV.str?, b.array? "eigenvectors" with
  | some h, some X => discard <| r.req "H3.eigenvectors_sha256" (arraySha X.dtype X.shape X.raw == h) "claim.eigenvectors_sha256 equals the hash of the stored vectors"
  | some _, none => r.add "H3.eigenvectors_sha256" .skip "vectors not stored"
  | none, _ => r.add "H3.eigenvectors_sha256" .skip "no eigenvector hash recorded"
  let stmts := ((cert.get? "statements").bind JV.arr?).getD #[]
  -- H4: spec hash
  match cert.path? ["subject", "spec"], (cert.path? ["subject", "spec_sha256"]).bind JV.str? with
  | some spec, some h =>
    if !spec.isNull then
      let hs := Sha256.hex (Json.canonical spec)
      discard <| r.req "H4.spec_sha256" (hs == h) s!"computed {hs}"
  | _, _ => r.add "H4.spec_sha256" .skip "no specification recorded"
  -- C7: scope statuses
  let inScope := ((cert.path? ["scope", "in_scope"]).bind JV.arr?).getD #[]
  let scopeOk := inScope.all (fun e => match (e.get? "id").bind JV.str?, (e.get? "status").bind JV.str? with
    | some sid, some st => statusOf stmts sid == some st
    | some sid, none => (statusOf stmts sid).isSome
    | _, _ => false)
  discard <| r.req "C7.scope_statuses" scopeOk s!"{inScope.size} statements in scope"
  -- arrays as exact numbers
  let getF := fun (nm : String) => (b.array? nm).map ArrRec.f64s
  let some mus := getF "eigenvalues" | do
      r.add "C1.eigenvalues" .fail "no eigenvalue array"; return (← r.checks.get, ← r.countLog.get)
  let k := mus.size
  -- C1: claim eigenvalues = array
  let claimEv := ((cert.path? ["claim", "eigenvalues"]).bind JV.arr?).getD #[]
  let c1 := claimEv.size == k && (List.range k).all (fun i =>
    match (claimEv[i]!.num?.bind decToF64), mus[i]! with
    | some (.fin a), .fin b' => Dy.beq a b'
    | _, _ => false)
  let nEv := (cert.path? ["claim", "n_eigenvalues"]).bind jNat?
  discard <| r.req "C1.eigenvalues" (c1 && nEv == some k) s!"{k} claimed values equal the stored array exactly"
  let some muD := mus.mapM F64.fin? | do
      r.add "C1.finite" .info "non-finite claimed value"; return (← r.checks.get, ← r.countLog.get)
  -- records
  let recs := ((((cert.get? "enclosures").bind JV.arr?).getD #[]).map parseRec)
  let radA := (getF "radius").getD #[]
  let exA := (getF "existence_radius").getD #[]
  let rbA := (getF "residual_bound").getD #[]
  let s2ev := evidenceOf stmts "S2.eigenpairs"
  let shift : Dy := ((s2ev.get? "model_shift").bind jdy?).getD Dy.zero
  let sectorShift := (s2ev.get? "sector_lift_shift").bind jdy?
  if sectorShift.isSome then r.add "S.sector" .info "sector certificate: the re-check concerns the sector block"
  -- C3/C4: records vs arrays
  if recs.size == k && radA.size == k && exA.size == k && rbA.size == k then
    let mut c3 := true
    let mut c4 := true
    for i in [0:k] do
      let rc := recs[i]!
      let mu := muD[i]!
      match rc.claimed, rc.radius, rc.lower, rc.upper, rc.existence, rc.residualBound with
      | some cl, some rad, some lo, some up, some ex, some rb =>
        if !(rc.index == i && Dy.beq cl mu && Dy.le lo (mu - rad) && Dy.le (mu + rad) up && Dy.le ex rad) then c3 := false
        match radA[i]!, exA[i]!, rbA[i]! with
        | .fin ra, .fin ea, .fin rba =>
          if !(Dy.le (rad + shift) ra && Dy.le (ex + shift) ea && Dy.beq rb rba) then c4 := false
        | _, _, _ => c4 := false
      | _, _, _, _, _, _ => c3 := false
    discard <| r.req "C3.enclosure_records" c3 "claimed = eigenvalue; lower <= mu - r; upper >= mu + r; existence radius <= radius (exact)"
    discard <| r.req "C4.arrays_vs_records" c4 s!"arrays.radius >= record radius + model shift {shift.toSci}; same for existence_radius; residual_bound equal"
  else
    if recs.size > 0 || radA.size > 0 then
      discard <| r.req "C3.enclosure_records" false s!"{recs.size} records, {radA.size} radii for {k} pairs"
  -- H5 and the operator
  let opRecH := (cert.path? ["subject", "operator", "matrix_sha256"]).bind JV.str?
  let some op := b.op? | do
      r.add "H5.operator_sha256" .skip "no operator exported"; return (← r.checks.get, ← r.countLog.get)
  let hOp := op.sha
  let h5 ← r.req "H5.operator_sha256" (some hOp == opRecH) s!"computed {hOp} ({op.source}{if op.hermitianPart then ", Hermitian part" else ""})"
  let A ← match op.toCsr with
    | .ok A => pure A
    | .error e => do
      r.add "H5.csr" .fail e; return (← r.checks.get, ← r.countLog.get)
  let sym := A.isSymmetric
  discard <| r.req "H5.symmetric" sym "the operator is exactly symmetric"
  if !h5 || !sym then
    r.add "R.operator" .skip "operator not the certified one or not symmetric: no recomputation"
    return (← r.checks.get, ← r.countLog.get)
  let dim := (cert.path? ["subject", "operator", "dimension"]).bind jNat?
  discard <| r.req "C2.dimension" (dim == some A.n) s!"dimension {A.n}"
  let gens := Csr.symmetryGenerators (A.shiftedInt Dy.zero)
  r.add "I.symmetry" .info s!"{gens.size} involutive basis permutations commute exactly with the operator (block reduction of the inertia computations)"
  -- eigenvectors
  let some X := b.array? "eigenvectors" | do
      r.add "R.vectors" .skip "no eigenvectors stored: no residual recomputation"; return (← r.checks.get, ← r.countLog.get)
  if X.dtype != "<f8" || X.shape != #[A.n, k] then
    discard <| r.req "C2.eigenvectors" false s!"eigenvectors {X.dtype} {shapeStr X.shape}, expected <f8 [{A.n}, {k}]"
    return (← r.checks.get, ← r.countLog.get)
  let xf := X.f64s
  let some xd := xf.mapM F64.fin? | do
      r.add "R.vectors" .info "non-finite vector entry"; return (← r.checks.get, ← r.countLog.get)
  let cols : Array (Array Dy) := (List.range k).toArray.map (fun j => (List.range A.n).toArray.map (fun i => xd[i * k + j]!))
  let res : Array (Array Dy) := (List.range k).toArray.map (fun j =>
    let ax := A.mulVec cols[j]!
    (List.range A.n).toArray.map (fun i => ax[i]! - muD[j]! * cols[j]![i]!))
  let rsq := res.map (fun v => dot v v)
  let xsq := cols.map (fun v => dot v v)
  -- E1, E2: per pair, exact
  let mut e1 := true
  let mut e2 := true
  let mut worst : Rat := 0
  let mut e1bad := #[]
  for i in [0:k] do
    let rc := recs[i]!
    match rc.residualBound, rc.existence with
    | some rb, some ex =>
      if !(geSqrt rb.toRat (rsq[i]!.toRat)) then
        e1 := false
        e1bad := e1bad.push i
      if xsq[i]!.isZero then e2 := false
      else
        if !(Dy.sign ex ≥ 0 && Dy.le (rsq[i]!) (ex.sq * xsq[i]!)) then e2 := false
        if ex.m != 0 then
          let q := rsq[i]!.toRat / (ex.sq.toRat * xsq[i]!.toRat)
          if q > worst then worst := q
    | _, _ => e1 := false
  discard <| r.req "E1.residual_bound" e1 (if e1 then s!"every recorded residual bound >= the exact ||A x - mu x|| ({k} pairs)" else s!"recorded residual bound below the exact residual norm for pairs {e1bad}")
  discard <| r.req "E2.existence_radius" e2 s!"every recorded existence radius >= the exact ||A x - mu x|| / ||x||; largest ratio exact/recorded {ratSci (sqrtUp worst)}"
  -- groups
  let ngroups := recs.foldl (fun m rc => max m (rc.group + 1)) 0
  let ainf := A.infNorm
  let hnRec := (cert.path? ["subject", "operator", "norm_upper_bound"]).bind jrat?
  let mut groupOk : Array Bool := Array.replicate ngroups true
  for g in [0:ngroups] do
    let mem := (recs.filter (·.group == g)).map (·.index)
    if mem.size == 0 then continue
    let rc0 := recs[mem[0]!]!
    let some rad := rc0.radius | continue
    if rc0.method == "single" || rc0.method == "existence-only" then
      -- Theorem 1 per member: radius >= ||r|| / ||x||
      let ok := mem.all (fun i => match recs[i]!.radius with
        | some ri => !xsq[i]!.isZero && Dy.sign ri ≥ 0 && Dy.le rsq[i]! (ri.sq * xsq[i]!)
        | none => false)
      if !ok then groupOk := groupOk.set! g false
      discard <| r.req s!"E4.G{g}.theorem1" ok s!"{rc0.method} group {mem}: radius >= exact Theorem-1 radius"
    else if rc0.method == "cluster" then
      -- exact Theorem 2 / 2' with own rational witnesses
      let gram := Id.run do
        let mut s := Dy.zero
        for a in mem do
          for c in mem do
            let v := dot cols[a]! cols[c]! - (if a == c then Dy.one else Dy.zero)
            s := s + v.sq
        return s
      let rf2 := dySum (mem.map (fun i => rsq[i]!))
      let xr2 := Id.run do
        let mut s := Dy.zero
        for a in mem do
          for c in mem do
            s := s + (dot cols[a]! res[c]!).sq
        return s
      let mus := mem.map (fun i => muD[i]!)
      let lam := mus.foldl (fun m v => Dy.max m v.abs) Dy.zero
      let spread := (mus.foldl Dy.max mus[0]!) - (mus.foldl Dy.min mus[0]!)
      let etaU := sqrtUp gram.toRat
      if etaU ≥ 1 then
        groupOk := groupOk.set! g false
        discard <| r.req s!"E3.G{g}.cluster" false s!"cluster {mem}: exact orthonormality defect >= 1"
      else
        let sd := sqrtDown (1 - etaU)
        let deltaU := 1 - sd
        let rU := sqrtUp rf2.toRat
        let xrU := sqrtUp xr2.toRat
        let rho := rU + (ainf.toRat + lam.toRat) * deltaU
        let eps := rU / sd + xrU / (1 - etaU) + deltaU * spread.toRat / sd
        let bound := if rho < eps then rho else eps
        let ok := rad.toRat ≥ bound
        if !ok then groupOk := groupOk.set! g false
        discard <| r.req s!"E3.G{g}.cluster" ok s!"cluster {mem}: exact Theorem 2 radius <= {ratSci rho}, Theorem 2' radius <= {ratSci eps}, recorded radius {rad.toSci}; exact eta <= {ratSci etaU}"
        -- recorded cluster arithmetic (C verifier's G checks), exact
        let gev := (((s2ev.get? "groups").bind JV.arr?).getD #[]).find? (fun gj =>
          ((((gj.get? "members").bind JV.arr?).getD #[]).filterMap jNat?).qsort (· < ·) == mem.qsort (· < ·))
        match gev with
        | some gj =>
          let bd := (gj.get? "bound").getD JV.null
          let q := fun (key : String) => (bd.get? key).bind jrat?
          match (gj.get? "eta").bind jrat?, q "R_fro", q "XR", q "delta", q "spread", q "rho", q "eps_star", hnRec with
          | some eta, some rf, some xr, some de, some sp, some rh, some es, some hn =>
            let okEta : Bool := geSqrt eta gram.toRat && eta < 1
            let okR := geSqrt rf rf2.toRat
            let okXR : Bool := geSqrt xr xr2.toRat
            let okSp : Bool := sp ≥ spread.toRat
            let okDe : Bool := de ≥ 1 || (1 - de) * (1 - de) ≤ 1 - eta
            let okRho : Bool := rh ≥ rf + (hn + lam.toRat) * de
            let a := es - xr / (1 - eta)
            let bb := rf + de * sp
            let okEs : Bool := a ≥ 0 && a * a * (1 - eta) ≥ bb * bb
            let okRad : Bool := rad.toRat ≥ (if rh < es then rh else es)
            let okHn : Bool := hn ≥ ainf.toRat
            let all := okEta && okR && okXR && okSp && okDe && okRho && okEs && okRad
            if !all then groupOk := groupOk.set! g false
            discard <| r.req s!"G{g}.record" all s!"recorded eta {okEta}, R_fro {okR}, XR {okXR}, spread {okSp}, delta {okDe}, rho {okRho}, eps_star {okEs}, radius {okRad}"
            r.add s!"G{g}.norm" (if okHn then .pass else .info) (if okHn then "recorded ||H|| bound >= exact infinity norm" else "recorded ||H|| bound below the exact infinity norm (not a contradiction: the infinity norm only bounds ||H||_2 from above)")
          | _, _, _, _, _, _, _, _ => r.add s!"G{g}.record" .info "no complete recorded cluster bound"
        | none => r.add s!"G{g}.record" .info "cluster not listed in the S2 evidence"
  -- E5: disjointness of distinct groups (record radii)
  let mut ivs : Array (Nat × Rat × Rat) := #[]
  for g in [0:ngroups] do
    let mem := recs.filter (·.group == g)
    if mem.size > 0 && mem[0]!.distinct then
      let lo := mem.foldl (fun m rc => match rc.radius with
        | some rr => let v := (muD[rc.index]! - rr).toRat; if v < m then v else m
        | none => m) ((muD[mem[0]!.index]!).toRat)
      let hi := mem.foldl (fun m rc => match rc.radius with
        | some rr => let v := (muD[rc.index]! + rr).toRat; if v > m then v else m
        | none => m) ((muD[mem[0]!.index]!).toRat)
      ivs := ivs.push (g, lo, hi)
  let disj := ivs.all (fun (g1, l1, h1) => ivs.all (fun (g2, l2, h2) => g1 == g2 || h1 < l2 || h2 < l1))
  discard <| r.req "E5.disjoint" disj s!"{ivs.size} groups marked distinct have pairwise disjoint enclosures (exact)"
  let allDistinct := recs.all (·.distinct)
  let exactS2 := e1 && e2 && groupOk.all id && disj
  -- enclosure ends with the model shift (arrays.radius)
  let radD : Array Dy := radA.map (fun f => match f with | .fin d => d | _ => ⟨1, 2000⟩)
  let lowM := (List.range k).toArray.map (fun i => muD[i]! - radD[i]!)
  let upM := (List.range k).toArray.map (fun i => muD[i]! + radD[i]!)
  -- S2 status
  let target := (s2ev.path? ["accuracy_target", "value"]).bind jdy?
  let probeOk := ((cert.path? ["arithmetic", "runtime_probe", "standard_model_hypotheses_hold"]).bind JV.bool?).getD false
  let ufFind := (((cert.path? ["arithmetic", "underflow_check", "findings"]).bind JV.arr?).getD #[]).size
  match statusOf stmts "S2.eigenpairs" with
  | some "CERTIFIED" =>
    let maxR := radD.foldl Dy.max Dy.zero
    let okT := match target with
      | some t => radA.all F64.isFinite && Dy.le maxR t
      | none => false
    discard <| r.req "S2.status.exact" (exactS2 && allDistinct && okT) s!"CERTIFIED: all pairs distinct {allDistinct}, exact radius checks {exactS2}, max radius {maxR.toSci} <= target {(target.map Dy.toSci).getD "-"}: {okT}"
    discard <| r.req "S2.status.rule" (probeOk && ufFind == 0) s!"CERTIFIED needs the runtime probe ({probeOk}) and no underflow finding ({ufFind} recorded)"
  | some "FAILED" =>
    match target with
    | none => r.add "S2.failed" .notReproduced "no accuracy target"
    | some t =>
      let tq := (t - shift).toRat
      -- pairs whose exact Theorem-1 radius plus the model shift exceeds the target
      let over := (List.range k).toArray.filter (fun i => !xsq[i]!.isZero && (tq < 0 || tq * tq * xsq[i]!.toRat < rsq[i]!.toRat))
      if over.size == 0 then r.add "S2.failed" .notReproduced "every exact Theorem-1 radius plus the model shift meets the target"
      else
        r.add "S2.failed" .reproduced s!"pairs {over}: the exact residual radius ||A x - mu x|| / ||x|| plus the model shift exceeds the target {t.toSci}"
        -- refutation: no eigenvalue of the stored operator within target + shift of mu_i
        let mut refuted := #[]
        for i in over do
          let lo := muD[i]! - t - shift
          let hi := muD[i]! + t + shift
          let cl ← r.count A gens lo
          let ch ← r.count A gens hi
          let inside := (ch.neg + ch.zero) - cl.neg
          if inside == 0 then refuted := refuted.push i
        if refuted.size > 0 then
          r.add "S2.failed.refuted" .refuted s!"pairs {refuted}: exact inertia counts show no eigenvalue of the operator in [mu - target - shift, mu + target + shift]: no eigenvalue of the declared model lies within the target of the claimed value"
        else r.add "S2.failed.refuted" .info "an eigenvalue lies within the target of every such claimed value (the claim is inaccurate only in its vector or pairing)"
  | some st => r.add "S2.status" .info s!"{st}: no claim to confirm; exact radius checks {exactS2}, all pairs distinct {allDistinct}"
  | none => pure ()
  -- S3
  let s3ev := evidenceOf stmts "S3.completeness"
  let kind := ((cert.path? ["claim", "kind"]).bind JV.str?).getD ""
  let counts := ((s3ev.get? "counts").bind JV.arr?).getD #[]
  let mut ci := 0
  for c in counts do
    match (c.get? "sigma").bind jdy? with
    | some sg =>
      let ex ← r.count A gens sg
      let recN := (c.get? "negative").bind jNat?
      r.add s!"S3.count{ci}" .info s!"shift {sg.toSci 17}: exact negative {ex.neg}, zero {ex.zero}; recorded {(recN.map toString).getD "inconclusive"}{if recN == some ex.neg then " (equal)" else if recN.isNone then "" else " (differs: an eigenvalue within beta of the shift)"}"
    | none => pure ()
    ci := ci + 1
  match statusOf stmts "S3.completeness", kind with
  | some "CERTIFIED", "lowest" =>
    match (s3ev.get? "top_shift").bind jdy?, (s3ev.get? "next_eigenvalue_lower_bound").bind jdy?, (s3ev.get? "gap_lower_bound").bind jdy? with
    | some sg, some bnd, some gl =>
      let u := upM.foldl Dy.max upM[0]!
      let ex ← r.count A gens sg
      let ok := exactS2 && allDistinct && Dy.lt u bnd && Dy.le (bnd + shift) sg && ex.neg == k && Dy.le gl (bnd - u)
      discard <| r.req "S3.status.exact" ok s!"CERTIFIED (lowest): {k} distinct enclosures below {u.toSci} < bound {bnd.toSci}; bound + shift <= top shift; exact count below the top shift {ex.neg} (= k: {ex.neg == k}); gap lower bound {gl.toSci} <= bound - top enclosure: {Dy.le gl (bnd - u)}"
    | _, _, _ => discard <| r.req "S3.status.exact" false "CERTIFIED (lowest) without top_shift, next_eigenvalue_lower_bound and gap_lower_bound"
    discard <| r.req "S3.status.rule" (probeOk && ufFind == 0) s!"CERTIFIED S3 inherits the floating-point hypotheses of S2: runtime probe {probeOk}, underflow findings {ufFind}"
  | some "CERTIFIED", "window" =>
    match counts[0]?.bind (fun c => (c.get? "sigma").bind jdy?), counts[1]?.bind (fun c => (c.get? "sigma").bind jdy?),
          ((s3ev.get? "certified_window").bind JV.arr?) with
    | some a, some bw, some cw =>
      match cw[0]?.bind jdy?, cw[1]?.bind jdy? with
      | some lw, some hw =>
        let ca ← r.count A gens a
        let cb ← r.count A gens bw
        let nOp := cb.neg - (ca.neg + ca.zero)
        let cin := (List.range k).filter (fun i => Dy.lt lw lowM[i]! && Dy.lt upM[i]! hw) |>.length
        let ok := exactS2 && allDistinct && Dy.le (a + shift) lw && Dy.le hw (bw - shift) && cb.neg ≥ ca.neg + ca.zero && nOp == cin
        discard <| r.req "S3.status.exact" ok s!"CERTIFIED (window): exact count in ({a.toSci}, {bw.toSci}) = {nOp}, distinct claimed enclosures inside the certified window = {cin}"
      | _, _ => discard <| r.req "S3.status.exact" false "certified window not finite"
    | _, _, _ => discard <| r.req "S3.status.exact" false "CERTIFIED (window) without two counts and a certified window"
    discard <| r.req "S3.status.rule" (probeOk && ufFind == 0) s!"CERTIFIED S3 inherits the floating-point hypotheses of S2: runtime probe {probeOk}, underflow findings {ufFind}"
  | some "CERTIFIED", "all" =>
    discard <| r.req "S3.status.exact" (exactS2 && allDistinct && k == A.n) s!"CERTIFIED (all): {k} distinct pairs, dimension {A.n}"
  | some "CERTIFIED", other => discard <| r.req "S3.status.exact" false s!"CERTIFIED for claim kind {other}: not re-checkable"
  | some "FAILED", "lowest" =>
    match (s3ev.get? "witness_shift").bind jdy? with
    | some t =>
      let top := (List.range k).foldl (fun bi i => if Dy.lt muD[bi]! muD[i]! then i else bi) 0
      let c := (List.range k).filter (fun i => Dy.lt lowM[i]! t) |>.length
      let below := Dy.le t lowM[top]!
      let n ← r.count A gens (t - shift)
      if below && n.neg > c then
        r.add "S3.failed" .refuted s!"exact count: {n.neg} eigenvalues of the operator below witness shift - model shift, but only {c} claimed enclosures reach below the witness shift, which lies below the top claimed enclosure: at least {n.neg - c} of the lowest eigenvalues are not claimed"
      else r.add "S3.failed" .notReproduced s!"exact count {n.neg}, claimed enclosures below the witness {c}, witness below the top enclosure {below}"
    | none => r.add "S3.failed" .notReproduced "no witness shift recorded"
  | some "FAILED", "window" =>
    match ((cert.path? ["claim", "window"]).bind JV.arr?) with
    | some w =>
      match w[0]?.bind jdy?, w[1]?.bind jdy? with
      | some a, some bw =>
        let ca ← r.count A gens (a + shift)
        let cb ← r.count A gens (bw - shift)
        let nOp := cb.neg - (ca.neg + ca.zero)
        let touch := (List.range k).filter (fun i => Dy.lt a upM[i]! && Dy.lt lowM[i]! bw) |>.length
        if cb.neg ≥ ca.neg + ca.zero && nOp > touch then
          r.add "S3.failed" .refuted s!"exact count: at least {nOp} eigenvalues of the declared model in the claimed window ({a.toSci}, {bw.toSci}), but only {touch} claimed enclosures meet it"
        else r.add "S3.failed" .notReproduced s!"exact count {nOp}, claimed enclosures meeting the window {touch}"
      | _, _ => r.add "S3.failed" .notReproduced "window not finite"
    | none => r.add "S3.failed" .notReproduced "no claimed window"
  | some st, _ => r.add "S3.status" .info s!"{st}: no claim to confirm"
  | none, _ => pure ()
  -- Kato-Temple records: exact Rayleigh quotient and residual, exact isolation count
  let kt := ((cert.get? "kato_temple").bind JV.arr?).getD #[]
  let mut ki := 0
  for e in kt do
    let iso := ((e.get? "isolation_interval").bind JV.arr?).getD #[]
    match (e.get? "index").bind jNat?, iso[0]?.bind jdy?, iso[1]?.bind jdy?, (e.get? "lower").bind jrat?, (e.get? "upper").bind jrat? with
    | some i, some al, some be, some lo, some hi =>
      if i < k && !xsq[i]!.isZero then
        let x := cols[i]!
        let ax := A.mulVec x
        let num := (dot x ax).toRat
        let den := xsq[i]!.toRat
        let th := num / den
        let r2 := ((dot ax ax).toRat - 2 * th * num + th * th * den) / den
        let ca ← r.count A gens al
        let cb ← r.count A gens be
        let one := cb.neg ≥ ca.neg + ca.zero && cb.neg - (ca.neg + ca.zero) == 1
        let inside := al.toRat < th && th < be.toRat
        let okLo := inside && lo ≤ th - r2 / (be.toRat - th)
        let okHi := inside && hi ≥ th + r2 / (th - al.toRat)
        discard <| r.req s!"K{ki}.kato_temple" (one && okLo && okHi) s!"pair {i}: exactly one eigenvalue of the operator in the isolation interval {one}; exact Rayleigh quotient inside {inside}; recorded [{ratSci lo}, {ratSci hi}] contains the exact Kato-Temple interval: lower {okLo}, upper {okHi}"
      else discard <| r.req s!"K{ki}.kato_temple" false "pair index out of range or zero vector"
    | _, _, _, _, _ => r.add s!"K{ki}.kato_temple" .info "incomplete record"
    ki := ki + 1
  -- S8: degeneracy brackets
  let s8 := evidenceOf stmts "S8.degeneracy"
  let cl8 := ((s8.get? "clusters").bind JV.arr?).getD #[]
  let mut gi := 0
  for cj in cl8 do
    if (cj.get? "status").bind JV.str? == some "CERTIFIED" then
      let mem := (((cj.get? "members").bind JV.arr?).getD #[]).filterMap jNat?
      let cs := ((cj.get? "counts").bind JV.arr?).getD #[]
      let bi := ((cj.get? "bracket_inner").bind JV.arr?).getD #[]
      let deg := (cj.get? "degeneracy").bind jNat?
      match cs[0]?.bind (fun c => (c.get? "sigma").bind jdy?), cs[1]?.bind (fun c => (c.get? "sigma").bind jdy?), bi[0]?.bind jdy?, bi[1]?.bind jdy? with
      | some s1, some s2, some il, some ih =>
        let c1 ← r.count A gens s1
        let c2 ← r.count A gens s2
        let nOp := c2.neg - (c1.neg + c1.zero)
        let inside := mem.all (fun i => i < k && Dy.lt il lowM[i]! && Dy.lt upM[i]! ih)
        let grpDistinct := mem.all (fun i => i < k && recs[i]!.distinct)
        let ok := exactS2 && grpDistinct && inside && Dy.le s1 (il - shift) && Dy.le (ih + shift) s2 &&
          c2.neg ≥ c1.neg + c1.zero && nOp == mem.size && deg == some mem.size
        discard <| r.req s!"S8.C{gi}.exact" ok s!"cluster {mem}: exact count in ({s1.toSci 12}, {s2.toSci 12}) = {nOp}; members' enclosures inside the inner bracket {inside}; degeneracy {(deg.map toString).getD "-"}"
      | _, _, _, _ => discard <| r.req s!"S8.C{gi}.exact" false "incomplete bracket record"
    gi := gi + 1
  return (← r.checks.get, ← r.countLog.get)

end CertifyCore
