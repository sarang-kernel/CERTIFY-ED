import CertifyApps.Vbs
import CertifyApps.Report
/-!
# Exact upper bounds on the Pomata-Wei quantity eta (experiment E10)

The construction follows `certify_ed/apps/aklt/eta.py`. With the kernel lemma (the ground space of the AKLT
Hamiltonian of a subgraph is the range of its VBS map), the two subspaces `A = ker H_{L u C} ⊗ phys(R)` and
`B = phys(L) ⊗ ker H_{C u R}` are compared inside `V = range U_L ⊗ range U_C ⊗ range U_R`, in the reduced
coordinates of `Vbs.lean`. The squared cosines of their principal angles are the eigenvalues of the pencil
`(C G_B^-1 C^T, G_A)`; for a rational threshold `t = p/q` the bordered integer matrix
`M = [[-p G_A, q C], [q C^T, -q G_B]]` has, by Haynsworth's inertia additivity, as many positive eigenvalues
as there are pencil eigenvalues above `t`. A certified count (`Engine.certCount`) gives `pi` positive pivots
with a residual bound `q beta`, and the bordered-pencil lemma of the project gives at most `pi` pencil
eigenvalues above `t' = (1 + beta/mu)(t + beta/nu)`, `mu` and `nu` certified lower bounds of the smallest
eigenvalues of `G_B` and `G_A`. If `pi` is at most a lower bound of the number of unit cosines (the rank
modulo `2^31 - 1` of the union VBS states), then `eta^2 ≤ t'`.

Everything is split into blocks of equal total down count, and inside a block into the character sectors of
the symmetry group (lattice mirror where given, and the spin flip), whose equivariance on every ingredient is
checked exactly; blocks exchanged by the flip are counted once, as in `eta.py`. The residual bound of a
sector is the bound on `N^-1/2 E N^-1/2` with `N` the squared norms of its orbit-sum vectors (all at least
1), which bounds the residual of the orthonormalised block.

Not re-derived here: the kernel lemma, the bordered-pencil lemma and the finite-size criterion of the
paper; they are used as stated.
-/
namespace CertifyApps
open CertifyCore

/-! ## Lattices (`apps/aklt/lattices.py`) -/

structure Lattice where
  name : String
  g : Graph
  L : Array Nat
  C : Array Nat
  R : Array Nat
  sigmas : Array (Array Nat)
  deriving Inhabited

def starLattice : Lattice :=
  let tri := fun (a : Nat) => #[(a, a + 1), (a + 1, a + 2), (a, a + 2)]
  ⟨"star", Graph.make (Array.replicate 9 3) (tri 0 ++ tri 3 ++ tri 6 ++ #[(2, 3), (4, 6)]),
    #[0, 1, 2], #[3, 4, 5], #[6, 7, 8], #[]⟩

def squareOctagonLattice (perp : Bool) : Lattice :=
  let sq := fun (k : Nat) => (4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3)
  let sqE := fun (k : Nat) => let (n, e, s, w) := sq k; #[(n, e), (e, s), (s, w), (w, n)]
  let extra := if perp then (4 * 1 + 0, 4 * 2 + 2) else (4 * 1 + 1, 4 * 2 + 3)
  let edges := sqE 0 ++ sqE 1 ++ sqE 2 ++ #[(1, 4 * 1 + 3), extra]
  ⟨if perp then "square_octagon_perp" else "square_octagon_par", Graph.make (Array.replicate 12 3) edges,
    #[0, 1, 2, 3], #[4, 5, 6, 7], #[8, 9, 10, 11], #[]⟩

def crossLattice : Lattice := Id.run do
  let names := #["a0", "a1", "a2", "a3", "a4", "a5", "p2", "p3", "p4", "p5",
                 "b0", "b1", "b2", "b3", "b4", "b5", "q2", "q3", "q4", "q5"]
  let ix := fun (s : String) => (names.idxOf? s).getD 0
  let mut E : Array (String × String) := #[]
  for h in ["a", "b"] do
    for i in [0:6] do
      E := E.push (s!"{h}{i}", s!"{h}{(i + 1) % 6}")
  E := E.filter (fun (u, v) => !((u == "b0" && v == "b1") || (u == "b1" && v == "b0")))
  E := E ++ #[("a0", "b0"), ("a1", "b1"), ("b0", "b1")]
  for (h, x) in [("a", "p"), ("b", "q")] do
    for (i, j) in [(2, 3), (4, 5)] do
      E := E ++ #[(s!"{h}{i}", s!"{x}{i}"), (s!"{x}{i}", s!"{x}{j}"), (s!"{x}{j}", s!"{h}{j}")]
  let edges := E.map (fun (u, v) => (ix u, ix v))
  return ⟨"cross", Graph.make (Array.replicate 20 3) edges,
    #["a2", "a3", "a4", "a5", "p2", "p3", "p4", "p5"].map ix, #["a0", "a1", "b0", "b1"].map ix,
    #["b2", "b3", "b4", "b5", "q2", "q3", "q4", "q5"].map ix, #[]⟩

def decoratedHoneycombLattice : Lattice := Id.run do
  let names := #["v", "w", "d_vw", "d_vu", "d_wu", "d_wd", "u_w", "d_vd", "u",
                 "x", "d_ux", "d_xu", "d_ud", "d_xd", "uu", "ux"]
  let ix := fun (s : String) => (names.idxOf? s).getD 0
  let E := #[("v", "d_vw"), ("d_vw", "w"), ("v", "d_vu"), ("w", "d_wu"), ("v", "d_vd"), ("d_vd", "u"),
    ("w", "d_wd"), ("d_wd", "u_w"), ("u", "d_ux"), ("d_ux", "x"), ("x", "d_xu"), ("u", "d_ud"),
    ("d_ud", "uu"), ("x", "d_xd"), ("d_xd", "ux")]
  let z := names.map (fun n => if #["v", "w", "u_w", "u", "x", "uu", "ux"].contains n then 3 else 2)
  return ⟨"decorated_honeycomb", Graph.make z (E.map (fun (a, b) => (ix a, ix b))),
    (Array.range 7), #[7, 8], (Array.range 7).map (· + 9), #[]⟩

/-- Site labels of the decorated square: a vertex `(x, y)` or the decoration of the edge `(a, b)`. -/
abbrev DLabel := Bool × (Int × Int) × (Int × Int)

def decoratedSquareLattice : Lattice := Id.run do
  let pe := fun (x y : Int) => #[((x, y), (x + 1, y)), ((x, y), (x, y + 1)), ((x + 1, y), (x + 1, y + 1)),
    ((x, y + 1), (x + 1, y + 1))]
  let giV : Array (Int × Int) := #[(0, 0), (1, 0), (0, 1), (1, 1)]
  let giE := pe 0 0 ++ #[((1, 1), (1, 2)), ((1, 1), (2, 1)), ((0, -1), (0, 0)), ((-1, 0), (0, 0))]
  let gjV : Array (Int × Int) := #[(1, 1), (2, 1), (1, 2), (2, 2)]
  let gjE := pe 1 1 ++ #[((0, 2), (1, 2)), ((1, 2), (1, 3)), ((2, 1), (3, 1)), ((2, 0), (2, 1))]
  let mut sites : Array DLabel := #[]
  let mut z : Array Nat := #[]
  let mut edges : Array (Nat × Nat) := #[]
  let mut memI : Array Nat := #[]
  let mut memJ : Array Nat := #[]
  for (tag, vs, es) in [(0, giV, giE), (1, gjV, gjE)] do
    for v in vs do
      let lab : DLabel := (true, v, (0, 0))
      if !sites.contains lab then
        sites := sites.push lab
        z := z.push 4
      let k := (sites.idxOf? lab).getD 0
      if tag == 0 then
        if !memI.contains k then memI := memI.push k
      else
        if !memJ.contains k then memJ := memJ.push k
    for (a, b) in es do
      let lab : DLabel := (false, a, b)
      if !sites.contains lab then
        sites := sites.push lab
        z := z.push 2
      let d := (sites.idxOf? lab).getD 0
      if tag == 0 then
        if !memI.contains d then memI := memI.push d
      else
        if !memJ.contains d then memJ := memJ.push d
      for e in [a, b] do
        if vs.contains e then
          let vi := (sites.idxOf? (true, e, (0, 0))).getD 0
          if !edges.contains (vi, d) then edges := edges.push (vi, d)
  let Cs := memI.filter (fun k => memJ.contains k)
  let Ls := (memI.filter (fun k => !Cs.contains k)).qsort (· < ·)
  let Rs := (memJ.filter (fun k => !Cs.contains k)).qsort (· < ·)
  -- the mirror (x, y) -> (y, x)
  let sw := fun (p : Int × Int) => (p.2, p.1)
  let perm := sites.map (fun (isV, a, b) =>
    if isV then (sites.idxOf? (true, sw a, (0, 0))).getD 0
    else match sites.idxOf? (false, sw a, sw b) with
      | some k => k
      | none => (sites.idxOf? (false, sw b, sw a)).getD 0)
  return ⟨"decorated_square", Graph.make z edges, Ls, Cs.qsort (· < ·), Rs, #[perm]⟩

def latticeByName (name : String) : Option Lattice :=
  match name with
  | "star" => some starLattice
  | "square_octagon_par" => some (squareOctagonLattice false)
  | "square_octagon_perp" => some (squareOctagonLattice true)
  | "cross" => some crossLattice
  | "decorated_honeycomb" => some decoratedHoneycombLattice
  | "decorated_square" => some decoratedSquareLattice
  | _ => none

/-! ## Symmetries on reduced coordinates -/

/-- Image of a slot under a site permutation. -/
def slotImage (g : Graph) (sigma : Array Nat) (sl : Slot) : Option Slot :=
  match sl with
  | .free s k => some (.free sigma[s]! k)
  | .edge ei s =>
      let (u, v) := g.edges[ei]!
      let a := sigma[u]!
      let b := sigma[v]!
      match g.edges.idxOf? (a, b) with
      | some e2 => some (.edge e2 sigma[s]!)
      | none => (g.edges.idxOf? (b, a)).map (fun e2 => .edge e2 sigma[s]!)

/-- Permutation of reduced configurations induced by a site permutation and/or the flip (`reduced_perm`). -/
def reducedPerm (g : Graph) (slots : Array Slot) (R : Reduction) (sigma : Option (Array Nat)) (flip : Bool) :
    Option (Array Nat) := Id.run do
  let nb := slots.size
  let mut imgPos : Array Nat := Array.range nb
  match sigma with
  | none => pure ()
  | some sg =>
      for k in [0:nb] do
        match slotImage g sg slots[k]! with
        | none => return none
        | some im => match slots.idxOf? im with
          | none => return none
          | some p => imgPos := imgPos.set! k p
  if (imgPos.qsort (· < ·)) != Array.range nb then return none
  let mut perm : Array Nat := #[]
  for f in R.reps do
    let mut out := 0
    for k in [0:nb] do
      let mut bt := (f >>> (nb - 1 - k)) % 2
      if flip then bt := 1 - bt
      if bt == 1 then out := out ||| (1 <<< (nb - 1 - imgPos[k]!))
    perm := perm.push (R.red out)
  if (perm.qsort (· < ·)) != Array.range R.nred then return none
  return some perm

def gramInvariant (G : Array (Array Int)) (p : Array Nat) : Bool :=
  (Array.range G.size).all (fun a => (Array.range G.size).all (fun b => G[p[a]!]![p[b]!]! == G[a]![b]!))

structure Act where
  mask : Array Bool
  flip : Bool
  pL : Array Nat
  pC : Array Nat
  pR : Array Nat
  pI : Array Nat
  pJ : Array Nat
  sC : Int
  deriving Inhabited

/-! ## The certificate -/

structure EtaOut where
  ok : Bool
  msg : String
  json : JV
  positive : Nat
  nUnit : Nat
  tPrime : Rat
  certified : Bool
  etaUpper : Rat

def certifyEta (lat : Lattice) (t : Rat) (threads : Nat) (log : String → IO Unit) : IO EtaOut := do
  let t0 ← IO.monoMsNow
  let elapsed := fun (u : Nat) => s!"[{(u - t0) / 1000}s]"
  let fail := fun (m : String) => (⟨false, m, .null, 0, 0, 0, false, 0⟩ : EtaOut)
  let g := lat.g
  let rL := reducedRegion g lat.L
  let rC := reducedRegion g lat.C
  let rR := reducedRegion g lat.R
  let Ui := unionCoords g #[lat.L, lat.C] #[rL, rC]
  let Uj := unionCoords g #[lat.C, lat.R] #[rC, rR]
  let fi := Ui.cols.size
  let fj := Uj.cols.size
  let Gi := gramOfCoords Ui #[rL.G, rC.G]
  let Gj := gramOfCoords Uj #[rC.G, rR.G]
  let UU := unionCoords g #[lat.L, lat.C, lat.R] #[rL, rC, rR]
  let nUnit := rankModP (gramPlain UU.cols)
  log s!"  reduced dims L {rL.n}, C {rC.n}, R {rR.n}; Gamma boundaries {fi}, {fj}; unit cosines >= {nUnit} {elapsed (← IO.monoMsNow)}"
  let some lamL ← lambdaMinLower rL.G | return fail "lambda_min(G_L) not certified"
  let some lamC ← lambdaMinLower rC.G | return fail "lambda_min(G_C) not certified"
  let some lamR ← lambdaMinLower rR.G | return fail "lambda_min(G_R) not certified"
  let some lamI ← lambdaMinLower Gi | return fail "lambda_min(G_i) not certified"
  let some lamJ ← lambdaMinLower Gj | return fail "lambda_min(G_j) not certified"
  let nu := lamI * lamR
  let mu := lamL * lamJ
  let nL := rL.n
  let nC := rC.n
  let nR := rR.n
  -- Z[l, c, b] and Y[c, r, d]
  let mut Z : Array Int := Array.replicate (nL * nC * fi) 0
  for b in [0:fi] do
    for (key, v) in Ui.cols[b]! do
      let l2 := key / nC
      let c := key % nC
      for l in [0:nL] do
        Z := Z.modify ((l * nC + c) * fi + b) (· + rL.G[l]![l2]! * v)
  let mut Y : Array Int := Array.replicate (nC * nR * fj) 0
  for d in [0:fj] do
    for (key, v) in Uj.cols[d]! do
      let c2 := key / nR
      let r2 := key % nR
      for c in [0:nC] do
        let gc := rC.G[c]![c2]!
        if gc != 0 then
          for r in [0:nR] do
            Y := Y.modify ((c * nR + r) * fj + d) (· + gc * rR.G[r]![r2]! * v)
  let zAt := fun (l c b : Nat) => Z[(l * nC + c) * fi + b]!
  let yAt := fun (c r d : Nat) => Y[(c * nR + r) * fj + d]!
  -- group elements: products of the generators (site permutations, then the flip)
  let n := g.z.size
  let ident := Array.range n
  let gens : Array (Array Nat × Bool) := lat.sigmas.map (fun s => (s, false)) ++ #[(ident, true)]
  let mut acts : Array Act := #[]
  for m in [0:2 ^ gens.size] do
    let mask := (Array.range gens.size).map (fun i => (m >>> (gens.size - 1 - i)) % 2 == 1)
    let mut sig := ident
    let mut fl := false
    for i in [0:gens.size] do
      if mask[i]! then
        let (s, f) := gens[i]!
        sig := sig.map (fun x => s[x]!)
        fl := fl != f
    let sigO := if sig == ident then none else some sig
    let some pL := reducedPerm g rL.slots rL.red sigO fl | return fail "a symmetry does not act on the L slots"
    let some pC := reducedPerm g rC.slots rC.red sigO fl | return fail "a symmetry does not act on the C slots"
    let some pR := reducedPerm g rR.slots rR.red sigO fl | return fail "a symmetry does not act on the R slots"
    let some pI := reducedPerm g Ui.slots Ui.red sigO fl | return fail "a symmetry does not act on the Gamma_i slots"
    let some pJ := reducedPerm g Uj.slots Uj.red sigO fl | return fail "a symmetry does not act on the Gamma_j slots"
    if !(gramInvariant rL.G pL && gramInvariant rC.G pC && gramInvariant rR.G pR && gramInvariant Gi pI &&
        gramInvariant Gj pJ) then return fail "a Gram matrix is not invariant"
    -- equivariance of Z and Y up to a sign
    let mut zs : Int := 0
    let mut zOk := true
    for l in [0:nL] do
      for c in [0:nC] do
        for b in [0:fi] do
          let a1 := zAt pL[l]! pC[c]! pI[b]!
          let a0 := zAt l c b
          if a0 != 0 || a1 != 0 then
            let s := if a1 == a0 then 1 else if a1 == -a0 then -1 else 0
            if s == 0 || (zs != 0 && s != zs) then zOk := false
            else zs := s
    let mut ys : Int := 0
    let mut yOk := true
    for c in [0:nC] do
      for r in [0:nR] do
        for d in [0:fj] do
          let a1 := yAt pC[c]! pR[r]! pJ[d]!
          let a0 := yAt c r d
          if a0 != 0 || a1 != 0 then
            let s := if a1 == a0 then 1 else if a1 == -a0 then -1 else 0
            if s == 0 || (ys != 0 && s != ys) then yOk := false
            else ys := s
    if !(zOk && yOk) then return fail "Z or Y is not equivariant"
    let zsign : Int := if zs == 0 then 1 else zs
    let ysign : Int := if ys == 0 then 1 else ys
    acts := acts.push ⟨mask, fl, pL, pC, pR, pI, pJ, zsign * ysign⟩
  log s!"  symmetry group of order {acts.size} checked exactly; C signs {acts.map (·.sC)} {elapsed (← IO.monoMsNow)}"
  -- blocks of equal total down count
  let keyA := fun (b r : Nat) => Ui.singlets + Ui.downs[b]! + rR.downs[r]! + rR.inner
  let keyB := fun (l d : Nat) => Uj.singlets + rL.downs[l]! + Uj.downs[d]! + rL.inner
  let mut keys : Array Nat := #[]
  for b in [0:fi] do
    for r in [0:nR] do
      if !keys.contains (keyA b r) then keys := keys.push (keyA b r)
  let sortedKeys := keys.qsort (· < ·)
  let p := t.num
  let q : Int := ((t.den : Nat) : Int)
  let mut posTotal := 0
  let mut beta : Rat := 0
  let mut done : Array (Nat × Nat) := #[]
  let mut blockLog : Array JV := #[]
  let mut largest := 0
  for key in sortedKeys do
    let ra := ((Array.range (fi * nR)).filter (fun x => keyA (x / nR) (x % nR) == key)).map (fun x => (x / nR, x % nR))
    let cb := ((Array.range (nL * fj)).filter (fun x => keyB (x / fj) (x % fj) == key)).map (fun x => (x / fj, x % fj))
    let posA := fun (br : Nat × Nat) => ra.idxOf? br
    let posB := fun (ld : Nat × Nat) => cb.idxOf? ld
    -- elements that map this block to itself
    let selfMap := fun (a : Act) => ra.all (fun (b, r) => (posA (a.pI[b]!, a.pR[r]!)).isSome)
    let gHere := acts.filter (fun a => !a.flip || selfMap a)
    let pureFlip := acts.find? (fun a => a.flip && (Array.range (a.mask.size - 1)).all (fun i => !a.mask[i]!))
    let mut partner : Option Nat := none
    if !(gHere.any (·.flip)) then
      match pureFlip with
      | some fa =>
          let (b0, r0) := ra[0]!
          let im := (fa.pI[b0]!, fa.pR[r0]!)
          partner := sortedKeys.find? (fun k2 => keyA im.1 im.2 == k2)
      | none => pure ()
    match partner with
    | some pk =>
        match done.find? (fun (k2, _) => k2 == pk) with
        | some (_, cnt) =>
            posTotal := posTotal + cnt
            done := done.push (key, cnt)
            blockLog := blockLog.push (jobj [("key", jnat key), ("partner", jnat pk), ("positive", jnat cnt)])
            continue
        | none => pure ()
    | none => pure ()
    -- characters over the generators used by gHere
    let genUsed := (Array.range gens.size).filter (fun i => gHere.any (fun a => a.mask[i]!))
    let chiVal := fun (ch : Nat) (a : Act) => Id.run do
      let mut v : Int := 1
      for k in [0:genUsed.size] do
        if a.mask[genUsed[k]!]! && (ch >>> k) % 2 == 1 then v := -v
      return v
    let mut posBlock := 0
    -- orbit-sum vectors per character
    let mut chars : Array (Array (Array (Nat × Int)) × Array (Array (Nat × Int))) := #[]
    for ch in [0:2 ^ genUsed.size] do
      let mut seenA : Array Bool := Array.replicate ra.size false
      let mut avecs : Array (Array (Nat × Int)) := #[]
      for k in [0:ra.size] do
        if !seenA[k]! then
          let (b, r) := ra[k]!
          let mut vec : Array (Nat × Int) := #[]
          for a in gHere do
            let kk := (posA (a.pI[b]!, a.pR[r]!)).getD 0
            seenA := seenA.set! kk true
            match vec.findIdx? (fun (x, _) => x == kk) with
            | some i => vec := vec.modify i (fun (x, c) => (x, c + chiVal ch a))
            | none => vec := vec.push (kk, chiVal ch a)
          let nzv := vec.filter (fun (_, c) => c != 0)
          if nzv.size > 0 then avecs := avecs.push nzv
      let mut seenB : Array Bool := Array.replicate cb.size false
      let mut bvecs : Array (Array (Nat × Int)) := #[]
      for k in [0:cb.size] do
        if !seenB[k]! then
          let (l, d) := cb[k]!
          let mut vec : Array (Nat × Int) := #[]
          for a in gHere do
            let kk := (posB (a.pL[l]!, a.pJ[d]!)).getD 0
            seenB := seenB.set! kk true
            match vec.findIdx? (fun (x, _) => x == kk) with
            | some i => vec := vec.modify i (fun (x, c) => (x, c + chiVal ch a * a.sC))
            | none => vec := vec.push (kk, chiVal ch a * a.sC)
          let nzv := vec.filter (fun (_, c) => c != 0)
          if nzv.size > 0 then bvecs := bvecs.push nzv
      chars := chars.push (avecs, bvecs)
    let sumA := chars.foldl (fun s (av, _) => s + av.size) 0
    let sumB := chars.foldl (fun s (_, bv) => s + bv.size) 0
    if sumA != ra.size || sumB != cb.size then return fail s!"orbit-sum bases of block {key} are incomplete"
    -- cross matrix by rows of A: Cm_ch[x][y] = sum c1 c2 sum_c Z[l2, c, b1] Y[c, r1, d2]
    let mut owner : Array (Array (Nat × Nat × Int)) := Array.replicate ra.size #[]
    for ci in [0:chars.size] do
      let (av, _) := chars[ci]!
      for x in [0:av.size] do
        for (k1, c1) in av[x]! do
          owner := owner.modify k1 (·.push (ci, x, c1))
    let mut cms : Array (Array (Array Int)) := chars.map (fun (av, bv) => Array.replicate av.size (Array.replicate bv.size 0))
    if cb.size > 0 then
      for k1 in [0:ra.size] do
        if owner[k1]!.size > 0 then
          let (b1, r1) := ra[k1]!
          let row := cb.map (fun (l2, d2) => Id.run do
            let mut s : Int := 0
            for c in [0:nC] do
              let zz := zAt l2 c b1
              if zz != 0 then s := s + zz * yAt c r1 d2
            return s)
          for (ci, x, c1) in owner[k1]! do
            let (_, bv) := chars[ci]!
            for y in [0:bv.size] do
              let s := bv[y]!.foldl (fun acc (k2, c2) => acc + c2 * row[k2]!) (0 : Int)
              if s != 0 then cms := cms.modify ci (·.modify x (·.modify y (· + c1 * s)))
    for ci in [0:chars.size] do
      let (av, bv) := chars[ci]!
      if av.size > 0 then
        let nA := av.size
        let nB := bv.size
        let gaEntry := fun (x y : Nat) => Id.run do
          let mut s : Int := 0
          for (k1, c1) in av[x]! do
            let (b1, r1) := ra[k1]!
            for (k2, c2) in av[y]! do
              let (b2, r2) := ra[k2]!
              s := s + c1 * c2 * Gi[b1]![b2]! * rR.G[r1]![r2]!
          return s
        let gbEntry := fun (x y : Nat) => Id.run do
          let mut s : Int := 0
          for (k1, c1) in bv[x]! do
            let (l1, d1) := cb[k1]!
            for (k2, c2) in bv[y]! do
              let (l2, d2) := cb[k2]!
              s := s + c1 * c2 * rL.G[l1]![l2]! * Gj[d1]![d2]!
          return s
        let dim := nA + nB
        let mut rows : Array (Array Int) := Array.replicate dim (Array.replicate dim 0)
        for x in [0:nA] do
          for y in [x:nA] do
            let v := -p * gaEntry x y
            rows := rows.modify x (·.set! y v)
            rows := rows.modify y (·.set! x v)
        for x in [0:nB] do
          for y in [x:nB] do
            let v := -q * gbEntry x y
            rows := rows.modify (nA + x) (·.set! (nA + y) v)
            rows := rows.modify (nA + y) (·.set! (nA + x) v)
        let cm := cms[ci]!
        for x in [0:nA] do
          for y in [0:nB] do
            let v := q * cm[x]![y]!
            rows := rows.modify x (·.set! (nA + y) v)
            rows := rows.modify (nA + y) (·.set! x v)
        let w := (av.map (fun v => v.foldl (fun s (_, c) => s + c.natAbs * c.natAbs) 0)) ++
          (bv.map (fun v => v.foldl (fun s (_, c) => s + c.natAbs * c.natAbs) 0))
        if dim > largest then largest := dim
        if (← IO.getEnv "CEAPPS_DRYRUN").isSome then
          log s!"    block {key} chi {ci}: size {dim} (dry run, not counted) {elapsed (← IO.monoMsNow)}"
          continue
        let r ← certCount (.de dim rows) w threads
        if !r.ok then return fail s!"block {key} character {ci}: {r.msg}"
        posTotal := posTotal + r.pos
        posBlock := posBlock + r.pos
        let bq := r.beta / (q : Rat)
        if bq > beta then beta := bq
        log s!"    block {key} chi {ci}: size {dim}, {r.pos} positive pivots, beta {ratSci bq 3} [{r.secFactor + r.secResidual}s] {elapsed (← IO.monoMsNow)}"
        blockLog := blockLog.push (jobj [("key", jnat key), ("character", jnat ci), ("size", jnat dim),
          ("positive", jnat r.pos), ("beta", jrat bq), ("seconds", jsec (r.secFactor + r.secResidual))])
    done := done.push (key, posBlock)
  let tPrime := (1 + beta / mu) * (t + beta / nu)
  let dry := (← IO.getEnv "CEAPPS_DRYRUN").isSome
  let certified : Bool := !dry && decide (posTotal ≤ nUnit)
  let etaUp := sqrtUp tPrime
  let t1 ← IO.monoMsNow
  log s!"  t = {ratSci t 6}: {posTotal} positive pivots (need <= {nUnit}); beta = {ratSci beta 3}; t' = {ratSci tPrime 11}; certified {certified} {elapsed t1}"
  let js := jobj [("reduced_dims", jarr #[jnat nL, jnat nC, jnat nR]), ("gamma_boundaries", jarr #[jnat fi, jnat fj]),
    ("pencil_size", jnat (fi * nR)), ("unit_cosines_at_least", jnat nUnit), ("group_order", jnat acts.size),
    ("largest_block", jnat largest), ("lambda_min_lower", jobj [("L", jrat lamL), ("C", jrat lamC), ("R", jrat lamR),
      ("i", jrat lamI), ("j", jrat lamJ)]), ("nu_lower", jrat nu), ("mu_lower", jrat mu),
    ("threshold", jrat t), ("positive", jnat posTotal), ("beta", jrat beta), ("t_prime", jrat tPrime),
    ("certified", jbool certified), ("eta_upper", jrat etaUp), ("blocks", jarr blockLog),
    ("seconds", jsec ((t1 - t0).toFloat / 1000.0))]
  return ⟨true, "", js, posTotal, nUnit, tPrime, certified, etaUp⟩

/-- Record entry of one lattice and the Lean recomputation at the record's threshold. -/
def runAklt (name recordPath outPath : String) (threads : Nat) : IO UInt32 := do
  let t0 ← IO.monoMsNow
  let some lat := latticeByName name | IO.eprintln s!"unknown lattice {name}"; return 2
  let rec_ ← readJsonFile recordPath
  let some lr := (rec_.get? "lattices").bind (·.get? name) | IO.eprintln "no record"; return 2
  let some ths := (lr.get? "thresholds") | return 2
  let (tKey, tRec) := match ths with
    | .obj kvs => match kvs[0]? with
      | some (k, v) => (k, v)
      | none => ("", JV.null)
    | _ => ("", JV.null)
  let some t := parseRatStr tKey | IO.eprintln "no threshold"; return 2
  let some tpRec := (tRec.get? "t_prime_exact").bind jvRatStr | return 2
  let some posRec := (tRec.get? "positive").bind jvNat | return 2
  let some unitRec := (lr.get? "unit_cosines_at_least").bind jvNat | return 2
  let etaUpRec := (lr.get? "eta_upper").bind jvNumRat
  let critZ := (lr.get? "criterion_z").bind jvNat
  let critRec := (lr.get? "criterion_holds").bind (·.bool?)
  IO.eprintln s!"{name}: threshold {tKey}"
  let out ← certifyEta lat t threads (fun s => IO.eprintln s)
  if !out.ok then
    IO.eprintln s!"{name}: {out.msg}"
    writeJson outPath (jobj [("target", jstr s!"E10 {name}"), ("error", jstr out.msg)])
    return 1
  let critLean : Option Bool := critZ.map (fun z => out.certified && out.tPrime < 1 / ((z * z : Nat) : Rat))
  let recert := out.certified && out.tPrime ≤ tpRec
  let agree := out.certified && out.positive == posRec && out.nUnit == unitRec &&
    (critZ.isNone || critLean == critRec)
  let t1 ← IO.monoMsNow
  let js := jobj [("target", jstr s!"E10 {name}"), ("lean", out.json),
    ("record", jobj [("threshold", jrat t), ("t_prime", jrat tpRec), ("positive", jnat posRec),
      ("unit_cosines_at_least", jnat unitRec), ("eta_upper", (etaUpRec.map jrat).getD .null),
      ("criterion_z", (critZ.map jnat).getD .null), ("criterion_holds", (critRec.map jbool).getD .null)]),
    ("criterion_holds_lean", (critLean.map jbool).getD .null),
    ("record_t_prime_recertified", jbool recert),
    ("eta_upper_lean", jrat out.etaUpper),
    ("agreement", jbool agree), ("seconds", jsec ((t1 - t0).toFloat / 1000.0))]
  writeJson outPath js
  IO.eprintln s!"{name}: Lean t' {ratSci out.tPrime 12} vs record {ratSci tpRec 12}; agreement {agree}; record t' re-certified {recert}"
  return (if agree then 0 else 1)

/-- The square-octagon criterion `4 eta_perp + 2 eta_par < 1` from the two result files. -/
def runSquareOctagonCriterion (perpPath parPath recordPath outPath : String) : IO UInt32 := do
  let a ← readJsonFile perpPath
  let b ← readJsonFile parPath
  let rec_ ← readJsonFile recordPath
  let get := fun (j : JV) => (j.path? ["lean", "t_prime", "exact"]).bind jvRatStr
  let some tp := get a | return 2
  let some tq := get b | return 2
  let okA := ((a.path? ["lean", "certified"]).bind (·.bool?)).getD false
  let okB := ((b.path? ["lean", "certified"]).bind (·.bool?)).getD false
  let sp := sqrtUp tp
  let sq := sqrtUp tq
  -- the rational square-root bounds are checked exactly: t' <= s^2 and s >= 0
  let sqrtOk := sp ≥ 0 && sq ≥ 0 && tp ≤ sp * sp && tq ≤ sq * sq
  let total := 4 * sp + 2 * sq
  let holds := okA && okB && sqrtOk && total < 1
  let recSum := (rec_.path? ["square_octagon", "sum_upper"]).bind jvNumRat
  let recHolds := (rec_.path? ["square_octagon", "criterion_holds"]).bind (·.bool?)
  let agree := recHolds == some holds
  writeJson outPath (jobj [("target", jstr "E10 square-octagon criterion 4 eta_perp + 2 eta_par < 1"),
    ("sum_upper_lean", jrat total), ("criterion_holds_lean", jbool holds),
    ("sqrt_upper_perp", jrat sp), ("sqrt_upper_par", jrat sq), ("square_root_bounds_checked", jbool sqrtOk),
    ("record", jobj [("sum_upper", (recSum.map jrat).getD .null), ("criterion_holds", (recHolds.map jbool).getD .null)]),
    ("lean_sum_at_most_record_sum", jbool (match recSum with | some r => total ≤ r | none => false)),
    ("agreement", jbool agree)])
  IO.eprintln s!"square-octagon: 4 eta_perp + 2 eta_par <= {ratSci total 12}: {holds}; agreement {agree}"
  return (if agree then 0 else 1)

end CertifyApps
