import CertifyApps.Eta
import CertifyApps.AkltHam
import Std.Data.HashMap
/-!
# Exact check of the AKLT kernel equality `ker H_X = range U_X` for the subgraphs of E10

For a region `X` (here `Gamma_i = L u C` and `Gamma_j = C u R` of every lattice), `H_X` is the sum over the
edges inside `X` of the projectors onto the largest total spin of the pair, written in the scaled basis of
`Vbs.lean` as integer matrices `Ht_e` (`scaledPmax`), and `U_X` is the reduced VBS map of `X`
(`reducedRegion`). The equality is shown by two exact computations, as in `certify_ed/apps/aklt/kernel.py`:

* `>=`: every `Ht_e` annihilates every reduced VBS column, checked in integers edge by edge
  (`vbsEdgeCheck`), and the rank of `U_X` is at least the rank modulo `p = 2^31 - 1` of its Gram matrix;
* `<=`: the null space of `H_X` over the rationals is the common null space of the `Ht_e` (the edge terms
  are similar to positive semidefinite projectors), whose dimension is at most its dimension modulo `p`
  (the rank of an integer matrix modulo `p` is at most its rank over the rationals).

The common null space modulo `p` is not computed per `S^z` block of the full matrix as in `kernel.py`,
which is out of reach for the 12-site regions of the cross and the decorated square (16777216 and
4100625 states), but site by site (`kernelDimModP`): after the sites `s_1 .. s_k` have been added, the
null space `K_k` of the edge terms inside `{s_1 .. s_k}` is kept as a subspace of `F_p^r (x) F`, where `F`
is the space of the frontier sites (those with an edge to a site not yet added) and `F_p^r` stands for the
interior sites through an injective linear map. Adding the site `w` gives the candidate space
`K_k (x) V_w`, in which the edges from the frontier to `w` impose linear equations; their null space is
`K_{k+1}`. Sites without further edges then move to the interior, and the interior is compressed to a set
of coordinates on which the restriction is injective (independent rows). Every step is exact linear
algebra over `F_p`, so the result is the dimension of the common null space modulo `p`.
-/
namespace CertifyApps

open CertifyCore

/-! ## Arithmetic modulo the Mersenne prime `2^31 - 1` -/

def kP : Nat := 2147483647

@[inline] def kP64 : UInt64 := 2147483647

/-- `x mod (2^31 - 1)` for `x < 2^63`. -/
@[inline] def mred (x : UInt64) : UInt64 :=
  let y := (x &&& kP64) + (x >>> 31)
  let y := (y &&& kP64) + (y >>> 31)
  if y ≥ kP64 then y - kP64 else y

@[inline] def mmul (a b : Nat) : Nat := (mred (a.toUInt64 * b.toUInt64)).toNat

def mpow (a e : Nat) : Nat := Id.run do
  let mut r := 1
  let mut b := a % kP
  let mut k := e
  while k > 0 do
    if k % 2 == 1 then r := mmul r b
    b := mmul b b
    k := k / 2
  return r

def minv (a : Nat) : Nat := mpow a (kP - 2)

def modP (x : Int) : Nat := (x % (kP : Int)).toNat

/-! ## Incremental row echelon form modulo `p` -/

/-- Rows with a leading 1 at their pivot column and zeros at the pivot columns of the earlier rows. -/
structure Ech where
  ncols : Nat
  rows : Array Nat := #[]
  piv : Array Nat := #[]

def Ech.reduce (E : Ech) (v0 : Array Nat) : Array Nat := Id.run do
  let n := E.ncols
  let mut v := v0
  for i in [0:E.piv.size] do
    let pc := E.piv[i]!
    let c := v[pc]!
    if c != 0 then
      let cu := (kP - c).toUInt64
      let base := i * n
      for j in [pc:n] do
        let y := E.rows[base + j]!
        if y != 0 then
          v := v.set! j (mred (v[j]!.toUInt64 + cu * y.toUInt64)).toNat
  return v

/-- Reduce and insert `v`; the flag says whether `v` was independent of the stored rows. -/
def Ech.insert (E : Ech) (v0 : Array Nat) : Ech × Bool := Id.run do
  let v := E.reduce v0
  let n := E.ncols
  let mut pc := n
  for j in [0:n] do
    if pc == n && v[j]! != 0 then pc := j
  if pc == n then return (E, false)
  let inv := minv v[pc]!
  let w := v.map (fun x => if x == 0 then 0 else mmul x inv)
  return ({ E with rows := E.rows ++ w, piv := E.piv.push pc }, true)

/-- A basis of the null space of the stored rows, as the columns of a flat `ncols x k` array. -/
def Ech.nullspace (E : Ech) : Array Nat × Nat := Id.run do
  let n := E.ncols
  let r := E.piv.size
  let mut isPiv : Array Bool := Array.replicate n false
  for pc in E.piv do isPiv := isPiv.set! pc true
  let free := (Array.range n).filter (fun j => !isPiv[j]!)
  let k := free.size
  let order := (Array.range r).qsort (fun a b => E.piv[a]! > E.piv[b]!)
  let mut N : Array Nat := Array.replicate (n * k) 0
  for fi in [0:k] do
    let mut x : Array Nat := Array.replicate n 0
    x := x.set! free[fi]! 1
    for i in order do
      let pc := E.piv[i]!
      let base := i * n
      let mut s : UInt64 := 0
      for j in [pc+1:n] do
        let y := E.rows[base + j]!
        if y != 0 then
          let xj := x[j]!
          if xj != 0 then s := mred (s + y.toUInt64 * xj.toUInt64)
      x := x.set! pc (if s == 0 then 0 else (kP64 - s).toNat)
    for j in [0:n] do
      N := N.set! (j * k + fi) x[j]!
  return (N, k)

/-! ## Site order and the site-by-site null space -/

def lexLt3 (a b : Nat × Int × Nat) : Bool :=
  a.1 < b.1 || (a.1 == b.1 && (a.2.1 < b.2.1 || (a.2.1 == b.2.1 && a.2.2 < b.2.2)))

/-- Greedy site order: from `start`, always add the site that leaves the smallest frontier space, then
the one with most edges to the sites already added, then the smallest label. -/
def greedyOrder (g : Graph) (X : Array Nat) (start : Nat) : Array Nat := Id.run do
  let inner := g.edges.filter (fun (u, v) => X.contains u && X.contains v)
  let adj := fun (s : Nat) => inner.filterMap (fun (u, v) =>
    if u == s then some v else if v == s then some u else none)
  let mut order : Array Nat := #[start]
  for _ in [1:X.size] do
    let mut best : Option (Nat × Int × Nat) := none
    for w in X do
      if !order.contains w then
        let dn := order.push w
        let front := dn.filter (fun f => (adj f).any (fun b => !dn.contains b))
        let DF := front.foldl (fun a f => a * (g.z[f]! + 1)) 1
        let conn : Int := ((adj w).filter (fun b => order.contains b)).size
        let key : Nat × Int × Nat := (DF, -conn, w)
        best := match best with
          | none => some key
          | some b0 => if lexLt3 key b0 then some key else some b0
    match best with
    | some (_, _, w) => order := order.push w
    | none => pure ()
  return order

structure KStep where
  site : Nat
  r : Nat
  frontierDim : Nat
  m : Nat
  rows : Nat
  ncols : Nat
  rank : Nat
  m2 : Nat
  deriving Inhabited

def KStep.json (s : KStep) : JV :=
  jobj [("site", jnat s.site), ("interior_rank", jnat s.r), ("frontier_dim", jnat s.frontierDim),
    ("kernel_dim_before", jnat s.m), ("equations", jnat s.rows), ("unknowns", jnat s.ncols),
    ("rank", jnat s.rank), ("kernel_dim_after", jnat s.m2)]

/-- Dimension modulo `2^31 - 1` of the common null space of the edge terms of the region `X`, sites added
in `order`; with the size of every step. -/
def kernelDimModP (g : Graph) (X : Array Nat) (order : Array Nat) : Nat × Array KStep := Id.run do
  let inner := g.edges.filter (fun (u, v) => X.contains u && X.contains v)
  let posO := fun (s : Nat) => (order.idxOf? s).getD 0
  let dim := fun (s : Nat) => g.z[s]! + 1
  let mut oedges : Array (Nat × Nat × Array Nat) := #[]
  for (u, v) in inner do
    let (a, b) := if posO u < posO v then (u, v) else (v, u)
    let (P, _) := scaledPmax g.z[a]! g.z[b]!
    let flat := P.foldl (fun acc row => acc ++ row.map modP) #[]
    oedges := oedges.push (a, b, flat)
  let laterOf := fun (s : Nat) => (oedges.filter (fun (a, _, _) => a == s)).map (fun (_, b, _) => b)
  let mut r := 1
  let mut F : Array Nat := #[]
  let mut B : Array Nat := #[1]
  let mut m := 1
  let mut stats : Array KStep := #[]
  for k in [0:order.size] do
    let w := order[k]!
    let dw := dim w
    let dFs := F.map dim
    let DF := dFs.foldl (· * ·) 1
    let rowsB := r * DF
    let ncols := m * dw
    let mut E : Ech := { ncols := ncols }
    let mut nrows := 0
    for (a, b, H) in oedges do
      if b == w then
        let iu := (F.idxOf? a).getD 0
        let du := dFs[iu]!
        let strideU := (dFs.extract (iu + 1) dFs.size).foldl (· * ·) 1
        let hn := du * dw
        for q in [0:rowsB] do
          if (q / strideU) % du == 0 then
            for a2 in [0:du] do
              for s2 in [0:dw] do
                let hr := (a2 * dw + s2) * hn
                let mut v : Array Nat := Array.replicate ncols 0
                let mut nz := false
                for a1 in [0:du] do
                  let rowB := (q + a1 * strideU) * m
                  for s1 in [0:dw] do
                    let h := H[hr + a1 * dw + s1]!
                    if h != 0 then
                      for j in [0:m] do
                        let bj := B[rowB + j]!
                        if bj != 0 then
                          let idx := j * dw + s1
                          v := v.set! idx (mred (v[idx]!.toUInt64 + h.toUInt64 * bj.toUInt64)).toNat
                          nz := true
                nrows := nrows + 1
                if nz then E := (E.insert v).1
    let rank := E.piv.size
    let m2 := ncols - rank
    stats := stats.push ⟨w, r, DF, m, nrows, ncols, rank, m2⟩
    if k + 1 == order.size then return (m2, stats)
    let (C, _) := E.nullspace
    -- B' = (B (x) I) C: rows q * dw + s, columns m2
    let mut Bp : Array Nat := Array.replicate (rowsB * dw * m2) 0
    for q in [0:rowsB] do
      for j in [0:m] do
        let bj := B[q * m + j]!
        if bj != 0 then
          for s in [0:dw] do
            let crow := (j * dw + s) * m2
            let orow := (q * dw + s) * m2
            for kk in [0:m2] do
              let c := C[crow + kk]!
              if c != 0 then
                Bp := Bp.set! (orow + kk) (mred (Bp[orow + kk]!.toUInt64 + bj.toUInt64 * c.toUInt64)).toNat
    -- the frontier after adding w; sites without further edges move to the interior
    let F2 := F.push w
    let d2 := F2.map dim
    let done := order.extract 0 (k + 1)
    let isLeave := fun (f : Nat) => (laterOf f).all (fun b => done.contains b)
    let leaveIx := (Array.range F2.size).filter (fun i => isLeave F2[i]!)
    let stayIx := (Array.range F2.size).filter (fun i => !isLeave F2[i]!)
    let DF2 := d2.foldl (· * ·) 1
    let DL := leaveIx.foldl (fun a i => a * d2[i]!) 1
    let DS := stayIx.foldl (fun a i => a * d2[i]!) 1
    let mut st2 : Array Nat := Array.replicate F2.size 1
    let mut acc := 1
    let mut i := F2.size
    while i > 0 do
      i := i - 1
      st2 := st2.set! i acc
      acc := acc * d2[i]!
    let ncolM := DS * m2
    let nrowM := r * DL
    let mut Mx : Array Nat := Array.replicate (nrowM * ncolM) 0
    for qq in [0:r * DF2] do
      let rho := qq / DF2
      let phi := qq % DF2
      let mut iL := 0
      for li in leaveIx do iL := iL * d2[li]! + (phi / st2[li]!) % d2[li]!
      let mut iS := 0
      for si in stayIx do iS := iS * d2[si]! + (phi / st2[si]!) % d2[si]!
      let rowM := rho * DL + iL
      for kk in [0:m2] do
        Mx := Mx.set! (rowM * ncolM + iS * m2 + kk) Bp[qq * m2 + kk]!
    -- interior coordinates on which the restriction is injective: a maximal set of independent rows
    let mut E2 : Ech := { ncols := ncolM }
    let mut sel : Array Nat := #[]
    for x in [0:nrowM] do
      let row := Mx.extract (x * ncolM) ((x + 1) * ncolM)
      if row.any (· != 0) then
        let (E2', isNew) := E2.insert row
        E2 := E2'
        if isNew then sel := sel.push x
    let rNew := sel.size
    let mut Bn : Array Nat := Array.replicate (rNew * ncolM) 0
    for ii in [0:rNew] do
      let x := sel[ii]!
      for t in [0:ncolM] do
        Bn := Bn.set! (ii * ncolM + t) Mx[x * ncolM + t]!
    r := rNew
    F := stayIx.map (fun i => F2[i]!)
    B := Bn
    m := m2
  return (m, stats)

/-! ## The `>=` direction: edge terms annihilate the VBS columns -/

/-- `Ht_e U_X = 0` for every edge `e` inside the region, in integers. -/
def vbsEdgeCheck (g : Graph) (reg : Region) : Bool := Id.run do
  let sites := reg.sites
  let pos := fun (s : Nat) => (sites.idxOf? s).getD 0
  let dims := sites.map (fun s => g.z[s]! + 1)
  let mut strides : Array Nat := Array.replicate sites.size 1
  let mut acc := 1
  let mut i := sites.size
  while i > 0 do
    i := i - 1
    strides := strides.set! i acc
    acc := acc * dims[i]!
  let inner := g.edges.filter (fun (u, v) => sites.contains u && sites.contains v)
  for (u, v) in inner do
    let (P, _) := scaledPmax g.z[u]! g.z[v]!
    let iu := pos u
    let iv := pos v
    let du := dims[iu]!
    let dv := dims[iv]!
    let su := strides[iu]!
    let sv := strides[iv]!
    for col in reg.cols do
      let mut groups : Std.HashMap Nat (Array Int) := {}
      for (row, amp) in col do
        let a := (row / su) % du
        let b := (row / sv) % dv
        let rest := row - a * su - b * sv
        let arr := (groups.getD rest (Array.replicate (du * dv) 0)).modify (a * dv + b) (· + amp)
        groups := groups.insert rest arr
      for (_, vec) in groups.toList do
        for rp in [0:du * dv] do
          let mut s : Int := 0
          for cp in [0:du * dv] do
            let x := vec[cp]!
            if x != 0 then s := s + P[rp]![cp]! * x
          if s != 0 then return false
  return true

/-- Number of states of the region in each `S^z` block (total number of down virtual spins). -/
def blockSizes (g : Graph) (sites : Array Nat) : Array Nat := Id.run do
  let mut poly : Array Nat := #[1]
  for s in sites do
    let z := g.z[s]!
    let mut np : Array Nat := Array.replicate (poly.size + z) 0
    for i in [0:poly.size] do
      for k in [0:z+1] do
        np := np.modify (i + k) (· + poly[i]!)
    poly := np
  return poly

structure KernelOut where
  lattice : String
  part : String
  sites : Nat
  dim : Nat
  vbsColumns : Nat
  vbsRankLower : Nat
  edgesAnnihilate : Bool
  nullityUpper : Nat
  equal : Bool
  order : Array Nat
  steps : Array KStep
  blocks : Array (Nat × Nat × Nat)
  seconds : Float
  deriving Inhabited

def KernelOut.json (k : KernelOut) : JV :=
  jobj [("lattice", jstr k.lattice), ("region", jstr k.part), ("sites", jnat k.sites), ("dim", jnat k.dim),
    ("vbs_columns", jnat k.vbsColumns), ("vbs_rank_lower", jnat k.vbsRankLower),
    ("edge_terms_annihilate_vbs", jbool k.edgesAnnihilate), ("nullity_mod_p_upper", jnat k.nullityUpper),
    ("prime", jnat kP), ("kernel_equals_vbs_range", jbool k.equal),
    ("site_order", jarr (k.order.map jnat)), ("steps", jarr (k.steps.map KStep.json)),
    ("blocks", jarr (k.blocks.map (fun (b, sz, c) => jarr #[jnat b, jnat sz, jnat c]))),
    ("seconds", jsec k.seconds)]

def kernelCheck (lat : Lattice) (part : String) (log : String → IO Unit) : IO KernelOut := do
  let t0 ← IO.monoMsNow
  let g := lat.g
  let X := (if part == "LC" then lat.L ++ lat.C else lat.C ++ lat.R).qsort (· < ·)
  let reg := reducedRegion g X
  let t1 ← IO.monoMsNow
  let edgesOk := vbsEdgeCheck g reg
  let rk := rankModP reg.G
  let t2 ← IO.monoMsNow
  log s!"  {lat.name} {part}: {X.size} sites, {reg.cols.size} VBS columns, edges annihilate {edgesOk}, rank >= {rk} [{(t1 - t0)} ms + {(t2 - t1)} ms]"
  let order := greedyOrder g X X[0]!
  let (nul, steps) := kernelDimModP g X order
  let t3 ← IO.monoMsNow
  let sizes := blockSizes g X
  let mut cnt : Array Nat := Array.replicate sizes.size 0
  for c in [0:reg.cols.size] do
    let b := reg.inner + reg.downs[c]!
    cnt := cnt.modify b (· + 1)
  let blocks := (Array.range sizes.size).map (fun b => (b, sizes[b]!, cnt[b]!))
  let dimN := sizes.foldl (· + ·) 0
  let equal := edgesOk && nul ≤ rk
  log s!"  {lat.name} {part}: order {order}, nullity mod p <= {nul}; kernel = VBS range: {equal} [{(t3 - t2)} ms]"
  return ⟨lat.name, part, X.size, dimN, reg.cols.size, rk, edgesOk, nul, equal, order, steps, blocks,
    (t3 - t0).toFloat / 1000.0⟩

/-- All twelve regions; comparison with the four kernel checks in the E10 record. -/
def runAkltKernels (recordPath outPath : String) : IO UInt32 := do
  let t0 ← IO.monoMsNow
  let rec_ ← readJsonFile recordPath
  let kern := rec_.get? "kernel"
  let names := #["star", "square_octagon_par", "square_octagon_perp", "cross", "decorated_honeycomb",
    "decorated_square"]
  let mut outs : Array JV := #[]
  let mut allEqual := true
  let mut allAgree := true
  for name in names do
    let some lat := latticeByName name | return 2
    for part in #["LC", "CR"] do
      let k ← kernelCheck lat part (fun s => IO.eprintln s)
      allEqual := allEqual && k.equal && k.nullityUpper == k.vbsRankLower
      -- the record's entry for this region, if it has one
      let recKey : Option (String × Nat) := match name, part with
        | "star", "LC" => some ("star", 0)
        | "square_octagon_par", "LC" => some ("square_octagon", 0)
        | "decorated_honeycomb", "LC" => some ("decorated_honeycomb", 0)
        | "decorated_honeycomb", "CR" => some ("decorated_honeycomb", 1)
        | _, _ => none
      let mut cmp : JV := .null
      match recKey with
      | some (rk, ix) =>
        match (kern.bind (·.get? rk)).bind (·.arr?) with
        | some xs =>
          let e := xs[ix]!
          let rSites := (e.get? "sites").bind jvNat
          let rDim := (e.get? "dim").bind jvNat
          let rRank := (e.get? "vbs_rank").bind jvNat
          let rNul := (e.get? "nullity_mod_p_upper").bind jvNat
          let rEq := (e.get? "kernel_equals_vbs_range").bind (·.bool?)
          let mut blocksOk := true
          match e.get? "blocks" with
          | some (.obj kvs) =>
            for (kk, v) in kvs do
              match kk.toNat?, v.arr? with
              | some b, some pr =>
                let sz := (pr[0]?.bind jvNat).getD 0
                let nl := (pr[1]?.bind jvNat).getD 0
                let mine := k.blocks.find? (fun (bb, _, _) => bb == b)
                blocksOk := blocksOk && (match mine with
                  | some (_, s2, c2) => s2 == sz && c2 == nl
                  | none => false)
              | _, _ => blocksOk := false
            blocksOk := blocksOk && kvs.size == (k.blocks.filter (fun (_, s2, _) => s2 > 0)).size
          | _ => blocksOk := false
          let agree := rSites == some k.sites && rDim == some k.dim && rRank == some k.vbsRankLower &&
            rNul == some k.nullityUpper && rEq == some k.equal && blocksOk
          allAgree := allAgree && agree
          cmp := jobj [("record_entry", jstr s!"kernel.{rk}[{ix}]"),
            ("sites", (rSites.map jnat).getD .null), ("dim", (rDim.map jnat).getD .null),
            ("vbs_rank", (rRank.map jnat).getD .null), ("nullity_mod_p_upper", (rNul.map jnat).getD .null),
            ("kernel_equals_vbs_range", (rEq.map jbool).getD .null),
            ("blocks_equal", jbool blocksOk), ("agreement", jbool agree)]
        | none =>
          allAgree := false
          cmp := jobj [("error", jstr "record entry missing")]
      | none => pure ()
      outs := outs.push (jobj [("lean", k.json), ("record", cmp)])
  let t1 ← IO.monoMsNow
  writeJson outPath (jobj [("target", jstr "E10 kernel equality ker H_X = range U_X for both subgraphs of all six lattices"),
    ("regions", jarr outs), ("all_kernels_equal_vbs_range", jbool allEqual),
    ("record_checks_agree", jbool allAgree), ("seconds", jsec ((t1 - t0).toFloat / 1000.0))])
  IO.eprintln s!"kernel checks: all equal {allEqual}; record comparisons agree {allAgree}"
  return (if allEqual && allAgree then 0 else 1)

end CertifyApps
