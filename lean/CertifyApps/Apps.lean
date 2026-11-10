import CertifyApps.Models
import CertifyApps.Vector
import CertifyApps.Report
/-!
# Recomputation of the certified numbers of experiments E7 and E9

E7 (`experiments/e7_exact_benchmarks.py`): for each of the four benchmark systems the claim is that the
recorded `E0` encloses the lowest eigenvalue of the stated block, with the recorded exact radius, and that one
eigenvalue lies below the recorded count point. Here the block is split into the same character sectors,
the eigenvalues below the recorded count point are counted in every sector (`Engine.certCount`), and a
vector is computed in the sector that holds the eigenvalue, to high precision, from which the exact radius
around the recorded `E0` and around the vector's own Rayleigh quotient follow.

E9 spin 1/2 (`apps/pyrochlore/spin_half.py`): the claim is that no eigenvalue of the crown lies below the
recorded shift minus `delta` in any of the 128 sectors of the `S^z = 0` space. Here every sector is counted at
the recorded shift, and the bound `shift - delta` is formed with the residual bound of this program.

E9 spin 1 (`apps/pyrochlore/spin_one.py`, `printed_coupling_check`): the claim is that the crown at the
printed couplings has the eigenvalue `-12 c1 + c17 E0(ring)` with `E0(ring)` enclosed by the recorded radius.
-/
namespace CertifyApps
open CertifyCore

/-- Exponent `k` (value `i^k`) of the character index `j` (value `exp(2 pi i j / order)`), if Gaussian. -/
def indexToExp (j order : Nat) : Option Nat := if (4 * j) % order == 0 then some ((4 * j / order) % 4) else none
def expToIndex (k order : Nat) : Nat := (k * order / 4) % order

/-- All character exponent vectors: `fixed[i] = some j` fixes generator `i` to the index `j`. -/
def charChoices (gens : Array Sym) (fixed : Array (Option Nat)) : Array (Array Nat) := Id.run do
  let mut acc : Array (Array Nat) := #[#[]]
  for i in [0:gens.size] do
    let o := gens[i]!.order
    let opts : Array Nat := match fixed[i]! with
      | some j => match indexToExp j o with | some k => #[k] | none => #[]
      | none => ((Array.range o).filterMap (fun j => indexToExp j o))
    let mut nxt : Array (Array Nat) := #[]
    for a in acc do
      for k in opts do
        nxt := nxt.push (a.push k)
    acc := nxt
  return acc

structure SectorCount where
  label : Array Nat
  dim : Nat
  real : Bool
  count : Nat
  delta : Rat
  ok : Bool
  msg : String
  secs : Float
  beta : Rat
  deriving Inhabited

/-- Count the eigenvalues of `H` restricted to the sector below `g`, with the Weyl slack `delta`. -/
def countSector (s : Sector) (den : Nat) (g : Rat) (threads : Nat) : IO SectorCount := do
  let t0 ← IO.monoMsNow
  let (M, w, real) := s.shifted den g
  let r ← certCount M w threads
  let t1 ← IO.monoMsNow
  let b : Rat := (g.den : Rat) * (den : Rat)
  let delta := r.beta / b
  let cnt := if real then r.neg else r.neg / 2
  let ok := r.ok && (real || r.neg % 2 == 0)
  return ⟨s.label, s.dim, real, cnt, delta, ok, r.msg, (t1 - t0).toFloat / 1000.0, r.beta⟩

def sectorJson (gens : Array Sym) (c : SectorCount) : JV :=
  jobj [("quantum_numbers", .obj ((Array.range gens.size).map (fun i =>
            (gens[i]!.name, jnat (expToIndex c.label[i]! gens[i]!.order))))),
        ("dim", jnat c.dim), ("real", jbool c.real), ("count", jnat c.count), ("delta", jrat c.delta),
        ("ok", jbool c.ok), ("message", jstr c.msg), ("seconds", jsec c.secs)]

/-- Real form of a sector matrix without shift (for the vector): `[[Re, -Im], [Im, Re]]` if complex. -/
def Sector.realForm (s : Sector) : SpRows × Array Nat :=
  if s.isReal then (s.re, s.nrm) else
  let n := s.dim
  let top := (Array.range n).map (fun i => s.re[i]! ++ s.im[i]!.map (fun (j, v) => (j + n, -v)))
  let bot := (Array.range n).map (fun i => s.im[i]! ++ s.re[i]!.map (fun (j, v) => (j + n, v)))
  (top ++ bot, s.nrm ++ s.nrm)

/-- A model with its symmetry data, ready for sectors. -/
structure Prepared where
  m : Heis
  states : Array Nat
  gens : Array Sym
  G : Group
  ob : Orbits
  commuteOk : Bool
  dimFixed : Nat
  sectors : Array Sector
  hermitianOk : Bool
  seconds : Float

def prepare (m : Heis) (upCount : Nat) (gens : Array Sym) (fixed : Array (Option Nat)) : IO Prepared := do
  let t0 ← IO.monoMsNow
  let states := statesFixedWeight m.N upCount
  let commuteOk := gens.all (fun g => commutes m states g)
  let G := Group.generate m.N gens
  let ob := orbits m.N states G
  -- dimension of the subspace with the fixed quantum numbers: orbit sums of the subgroup they generate
  let fixedIdx := (Array.range gens.size).filter (fun i => fixed[i]!.isSome)
  let subGens := fixedIdx.map (fun i => gens[i]!)
  let Gs := Group.generate m.N subGens
  let obs := orbits m.N states Gs
  let subExp := fixedIdx.map (fun i => (indexToExp (fixed[i]!.getD 0) gens[i]!.order).getD 0)
  let dimFixed := match Gs.character subExp with
    | some chi => (Array.range obs.reps.size).foldl (fun acc k =>
        if obs.stab[k]!.all (fun g => chi[g]! % 4 == 0) then acc + 1 else acc) 0
    | none => 0
  let mut sectors : Array Sector := #[]
  for ex in charChoices gens fixed do
    match G.character ex with
    | none => pure ()
    | some chi =>
        let s := buildSector m states ob chi ex
        if s.dim > 0 then sectors := sectors.push s
  let hermOk := sectors.all (·.hermitian)
  let t1 ← IO.monoMsNow
  return ⟨m, states, gens, G, ob, commuteOk && ob.ok, dimFixed, sectors, hermOk, (t1 - t0).toFloat / 1000.0⟩

/-! ## E7 -/

def e7System (key : String) : Option (String × Heis × Array Sym × Array (Option Nat)) :=
  match key with
  | "chain_J2=0" => some ("Heisenberg chain", chainModel 20 0 1,
      #[translationChain 20, reflectionChain 20, spinFlip 20], #[some 0, none, none])
  | "chain_J2=0.5" => some ("J1-J2 chain", chainModel 20 1 2,
      #[translationChain 20, reflectionChain 20, spinFlip 20], #[some 0, none, none])
  | "square_J2=0" => some ("Heisenberg square", squareModel 4 4 0 1,
      #[squareTx 4 4, squareTy 4 4, spinFlip 16], #[none, none, none])
  | "square_J2=0.5" => some ("J1-J2 square", squareModel 4 4 1 2,
      #[squareTx 4 4, squareTy 4 4, spinFlip 16], #[none, none, none])
  | _ => none

def findRecordRow (rows : Array JV) (pfx : String) : Option JV :=
  rows.find? (fun r => match r.get? "system" with
    | some v => (v.str?.getD "").startsWith pfx
    | none => false)

def runE7 (key recordPath outPath : String) (threads : Nat) : IO UInt32 := do
  let t0 ← IO.monoMsNow
  let some (pfx, m, gens, fixed) := e7System key | IO.eprintln s!"unknown system {key}"; return 2
  let rec_ ← readJsonFile recordPath
  let some rows := (rec_.get? "E7").bind (·.arr?) | IO.eprintln "no E7 rows"; return 2
  let some row := findRecordRow rows pfx | IO.eprintln s!"no record row for {pfx}"; return 2
  let some e0 := (row.get? "E0").bind jvNumRat | IO.eprintln "no E0"; return 2
  let some rad := (row.get? "exact_radius").bind jvNumRat | IO.eprintln "no radius"; return 2
  let some cnt0 := (row.get? "counts").bind (·.arr?) |>.bind (·[0]?) | IO.eprintln "no counts"; return 2
  let some gc := (cnt0.get? "point").bind jvNumRat | IO.eprintln "no point"; return 2
  let some recCount := (cnt0.get? "count").bind jvNat | return 2
  let some recDelta := (cnt0.get? "delta").bind jvNumRat | return 2
  let recBlock := (row.get? "block_dim").bind jvNat
  let recU1 := (row.get? "u1_dim").bind jvNat
  let recSectors := ((row.get? "sectors").bind (·.arr?)).getD #[]
  let den := 4 * m.D
  IO.eprintln s!"E7 {key}: building model and sectors"
  let pr ← prepare m (m.N / 2) gens fixed
  IO.eprintln s!"  {pr.states.size} states, group order {pr.G.elems.size}, block {pr.dimFixed}, sectors {pr.sectors.map (·.dim)} [{pr.seconds}s]"
  -- counts at the recorded point
  let mut counts : Array SectorCount := #[]
  for s in pr.sectors do
    let c ← countSector s den gc threads
    IO.eprintln s!"  sector {c.label} dim {c.dim}: {c.count} below the point, delta {ratSci c.delta 4}, {c.secs}s {c.msg}"
    counts := counts.push c
  let total := counts.foldl (fun a c => a + c.count) 0
  let dmax := counts.foldl (fun a c => if c.delta > a then c.delta else a) 0
  let countsOk := counts.all (·.ok)
  -- the sector holding the counted eigenvalue
  let gsIdx := (Array.range counts.size).find? (fun i => counts[i]!.count ≥ 1)
  let mut vecJson : JV := .null
  let mut encOk := false
  let mut recEncOk := false
  let mut inside := false
  match gsIdx with
  | none => pure ()
  | some gi =>
    let s := pr.sectors[gi]!
    let (A, N) := s.realForm
    let delta0 := roundDyadic ((gc - e0) / 8) 40
    let sigma := roundDyadic (e0 - delta0) 60
    let tv0 ← IO.monoMsNow
    let ep := inverseIteration A N den sigma 8 10 160
    let tv1 ← IO.monoMsNow
    if ep.ok then
      let mu := roundDyadic ep.mu 120
      let r2 := (radiusSq A N den ep.z mu).getD 1000
      let r := sqrtUp r2
      let r2rec := (radiusSq A N den ep.z e0).getD 1000
      let rrec := sqrtUp r2rec
      encOk := mu + r < gc - dmax
      recEncOk := rrec ≤ rad
      inside := e0 - rad ≤ mu - r && mu + r ≤ e0 + rad
      vecJson := jobj [("sector", sectorJson gens counts[gi]!), ("shift", jrat sigma),
        ("rayleigh_quotient_rounded", jrat mu), ("radius", jrat r), ("radius_at_record_E0", jrat rrec),
        ("high_precision_steps", jnat ep.steps), ("last_refinement_relative_bits", jint ep.lastRel),
        ("seconds", jsec ((tv1 - tv0).toFloat / 1000.0))]
    else
      vecJson := jobj [("error", jstr ep.msg)]
  -- comparison of sector dimensions with the record
  let mut dimsAgree := true
  for c in counts do
    let qn := (Array.range gens.size).map (fun i => (gens[i]!.name, expToIndex c.label[i]! gens[i]!.order))
    let match_ := recSectors.find? (fun rs => match rs.get? "quantum_numbers" with
      | some q => qn.all (fun (nm, j) => ((q.get? nm).bind jvNat) == some j)
      | none => false)
    match match_ with
    | some rs => if ((rs.get? "dim").bind jvNat) != some c.dim then dimsAgree := false
    | none => dimsAgree := false
  if recSectors.size != counts.size then dimsAgree := false
  let certified := pr.commuteOk && pr.hermitianOk && countsOk && total == 1 && encOk
  let agree := certified && total == recCount && recEncOk && dimsAgree &&
    recBlock == some pr.dimFixed && recU1 == some pr.states.size
  let t1 ← IO.monoMsNow
  let out := jobj [
    ("target", jstr s!"E7 {key}"), ("system", (row.get? "system").getD .null),
    ("model", jobj [("sites", jnat m.N), ("bonds", jnat m.bonds.size), ("coupling_denominator", jnat m.D),
        ("Hint_scale", jnat den)]),
    ("u1_dim", jnat pr.states.size), ("block_dim", jnat pr.dimFixed), ("group_order", jnat pr.G.elems.size),
    ("symmetries_commute_exactly", jbool pr.commuteOk), ("sector_matrices_hermitian", jbool pr.hermitianOk),
    ("sectors", jarr (counts.map (sectorJson gens))),
    ("count_point", jrat gc), ("count_total", jnat total), ("delta_max", jrat dmax),
    ("next_eigenvalue_lower_bound", jrat (gc - dmax)),
    ("vector", vecJson),
    ("record", jobj [("E0", jrat e0), ("exact_radius", jrat rad), ("count", jnat recCount),
        ("delta", jrat recDelta), ("block_dim", (recBlock.map jnat).getD .null),
        ("u1_dim", (recU1.map jnat).getD .null)]),
    ("lean_status", jstr (if certified then "CERTIFIED" else "NOT_CERTIFIED")),
    ("record_enclosure_recertified", jbool recEncOk), ("lean_enclosure_inside_record_enclosure", jbool inside),
    ("sector_dims_agree", jbool dimsAgree),
    ("agreement", jbool agree), ("seconds", jsec ((t1 - t0).toFloat / 1000.0))]
  writeJson outPath out
  IO.eprintln s!"E7 {key}: count {total} (record {recCount}), record enclosure re-certified {recEncOk}, agreement {agree}"
  return (if agree then 0 else 1)

/-! ## E9, spin 1/2: the crown lower bound -/

def runCrownHalf (recordPath outPath : String) (threads : Nat) (onlyLargest : Nat := 0) : IO UInt32 := do
  let t0 ← IO.monoMsNow
  let rec_ ← readJsonFile recordPath
  let some sh := rec_.get? "spin_half" | IO.eprintln "no spin_half"; return 2
  let some g := (sh.get? "shift").bind jvRatStr | IO.eprintln "no shift"; return 2
  let some lbRec := (sh.get? "lower_bound_exact").bind jvRatStr | IO.eprintln "no bound"; return 2
  let some lbFloat := (sh.get? "lower_bound").bind jvNumRat | return 2
  let some dRec := (sh.get? "delta").bind jvNumRat | return 2
  let n1 := ((sh.get? "n1").bind jvNat).getD 0
  let n3 := ((sh.get? "n3").bind jvNat).getD 0
  let m := crownModel n1 n3
  let den := 4 * m.D
  IO.eprintln s!"crown S=1/2: n1={n1} n3={n3}; building sectors"
  let pr ← prepare m 9 crownGenerators (Array.replicate 7 none)
  let dims := pr.sectors.map (·.dim)
  IO.eprintln s!"  {pr.states.size} states, group order {pr.G.elems.size}, {pr.sectors.size} sectors, largest {dims.foldl max 0}, sum {dims.foldl (· + ·) 0} [{pr.seconds}s]"
  -- order the sectors by dimension (small first) so that progress is visible
  let order := (Array.range pr.sectors.size).qsort (fun a b => pr.sectors[a]!.dim < pr.sectors[b]!.dim)
  let order := if onlyLargest > 0 then order.extract (order.size - onlyLargest) order.size else order
  let mut counts : Array SectorCount := #[]
  for k in order do
    let s := pr.sectors[k]!
    let c ← countSector s den g threads
    IO.eprintln s!"  sector {c.label} dim {c.dim}: {c.count} below the shift, delta {ratSci c.delta 4}, {c.secs}s {c.msg}"
    counts := counts.push c
  let total := counts.foldl (fun a c => a + c.count) 0
  let dmax := counts.foldl (fun a c => if c.delta > a then c.delta else a) 0
  let allOk := counts.all (·.ok) && counts.size == pr.sectors.size
  let lbLean := g - dmax
  let certified := pr.commuteOk && pr.hermitianOk && allOk && total == 0 &&
    (dims.foldl (· + ·) 0) == pr.states.size
  let recert := certified && lbLean ≥ lbRec
  let t1 ← IO.monoMsNow
  let out := jobj [
    ("target", jstr "E9 crown S=1/2 lower bound"),
    ("model", jobj [("sites", jnat 18), ("bonds", jnat m.bonds.size), ("c1", jrat (mkRat n1 (10 ^ 7))),
        ("c3", jrat (mkRat n3 (10 ^ 7))), ("c17", jrat (mkRat ((10 ^ 7 : Int) - 2 * n1 - 8 * n3) (2 * 10 ^ 7))),
        ("Hint_scale", jnat den)]),
    ("dim_Sz0", jnat pr.states.size), ("n_sectors", jnat pr.sectors.size), ("largest_sector", jnat (dims.foldl max 0)),
    ("sector_dims_sum", jnat (dims.foldl (· + ·) 0)),
    ("symmetries_commute_exactly", jbool pr.commuteOk), ("sector_matrices_hermitian", jbool pr.hermitianOk),
    ("sectors_counted", jnat counts.size),
    ("sectors", jarr (counts.map (sectorJson pr.gens))),
    ("shift", jrat g), ("count_total", jnat total), ("delta_max", jrat dmax),
    ("lower_bound_lean", jrat lbLean),
    ("record", jobj [("lower_bound_exact", jrat lbRec), ("lower_bound", jrat lbFloat), ("delta", jrat dRec),
        ("dim_Sz0", (sh.get? "dim_Sz0").getD .null), ("n_sectors", (sh.get? "n_sectors").getD .null),
        ("largest_sector", (sh.get? "largest_sector").getD .null)]),
    ("lean_status", jstr (if certified then "CERTIFIED" else "NOT_CERTIFIED")),
    ("record_bound_recertified", jbool recert),
    ("agreement", jbool (recert && ((sh.get? "n_sectors").bind jvNat) == some pr.sectors.size &&
        ((sh.get? "largest_sector").bind jvNat) == some (dims.foldl max 0))),
    ("seconds", jsec ((t1 - t0).toFloat / 1000.0))]
  writeJson outPath out
  IO.eprintln s!"crown S=1/2: total count {total}, Lean bound {ratSci lbLean}, record {ratSci lbRec}, re-certified {recert}"
  return (if recert then 0 else 1)

/-! ## E9, spin 1: the all-singlet eigenvalue at the printed couplings -/

/-- Exact check of the pair-singlet identities on three spin-1 sites `a, b, h`:
`(S_a + S_b) . S_h (u ⊗ e) = 0` and `S_a . S_b (u ⊗ e) = -2 (u ⊗ e)` for every basis vector `e` of site `h`,
with `u = |1,-1> - |0,0> + |-1,1>` the pair singlet. -/
def pairSingletCheck : Bool := Id.run do
  let mab : Spin1 := ⟨3, #[(0, 1, 1)]⟩
  let mah : Spin1 := ⟨3, #[(0, 2, 1), (1, 2, 1)]⟩
  -- u ⊗ e_k: digits (k_a, k_b, k_h) with weights u(1,-1) = 1 (digits 0,2), u(0,0) = -1 (1,1), u(-1,1) = 1 (2,0)
  for kh in [0:3] do
    let v : Array (Nat × Int) := #[(0 * 9 + 2 * 3 + kh, 1), (1 * 9 + 1 * 3 + kh, -1), (2 * 9 + 0 * 3 + kh, 1)]
    for (mm, expect) in [(mah, (0 : Int)), (mab, -2)] do
      let mut acc : Array Int := Array.replicate 27 0
      for (s, c) in v do
        for (t, h) in mm.apply s do
          acc := acc.modify t (· + c * h)
      for (s, c) in v do
        acc := acc.modify s (· - expect * c)
      if acc.any (· != 0) then return false
  return true

def runCrownOne (recordPath outPath : String) : IO UInt32 := do
  let t0 ← IO.monoMsNow
  let rec_ ← readJsonFile recordPath
  let some p := rec_.get? "spin1_printed" | IO.eprintln "no spin1_printed"; return 2
  let some c1 := (p.get? "c1").bind jvRatStr | return 2
  let some c3 := (p.get? "c3").bind jvRatStr | return 2
  let some c17r := (p.get? "c17").bind jvRatStr | return 2
  let some e0 := (p.get? "E0_ring").bind jvNumRat | return 2
  let some rad := (p.get? "E0_radius").bind jvNumRat | return 2
  let some upRec := (p.get? "eigenvalue_upper").bind jvNumRat | return 2
  let some pub := (p.get? "published_bound").bind jvNumRat | return 2
  let c17 := (1 - 2 * c1 - 8 * c3) / 2
  -- the ring
  let ring := spin1Ring 6
  let A := ring.matrix
  let n := A.size
  let N : Array Nat := Array.replicate n 1
  let symOk := spSymmetric n A
  let sigma := roundDyadic (e0 - 1/20) 60
  let ep := inverseIteration A N 1 sigma 8 10 160
  let mu := roundDyadic ep.mu 120
  let r := sqrtUp ((radiusSq A N 1 ep.z mu).getD 1000)
  let rrec := sqrtUp ((radiusSq A N 1 ep.z e0).getD 1000)
  -- count: find a point with exactly one eigenvalue below it and the enclosure under its window
  let mut pt : Rat := 0
  let mut cnt : Nat := 0
  let mut dl : Rat := 0
  let mut found := false
  let mut tried : Array JV := #[]
  for k in [1:12] do
    if !found then
      let g := roundDyadic (mu + 1 / (2 ^ k : Nat)) 40
      let Mg : IMat := .sp n ((Array.range n).map (fun i => (A[i]!.map (fun (j, v) => (j, (g.den : Int) * v))).push (i, -g.num)))
      let cr ← certCount Mg #[] 1
      let d := cr.beta / (g.den : Rat)
      tried := tried.push (jobj [("point", jrat g), ("count", jnat cr.neg), ("delta", jrat d), ("ok", jbool cr.ok)])
      if cr.ok && cr.neg == 1 && mu + r < g - d then
        pt := g; cnt := cr.neg; dl := d; found := true
  let lowestOk := found && ep.ok && symOk
  -- the all-singlet eigenvalue bound, from the record's numbers and from this program's enclosure
  let upFromRecord := -12 * c1 + (if c17 > 0 then c17 * (e0 + rad) else c17 * (e0 - rad))
  let upLean := -12 * c1 + (if c17 > 0 then c17 * (mu + r) else c17 * (mu - r))
  let recFloatMatches := ratNearestF64 upFromRecord == some upRec
  let pairOk := pairSingletCheck
  let recEnc := rrec ≤ rad
  let agree := lowestOk && pairOk && recEnc && recFloatMatches && upLean < pub && upFromRecord < pub &&
    c17 == c17r
  let t1 ← IO.monoMsNow
  let out := jobj [
    ("target", jstr "E9 crown S=1 all-singlet upper bound at the printed couplings"),
    ("c1", jrat c1), ("c3", jrat c3), ("c17", jrat c17), ("c17_record", jrat c17r),
    ("ring", jobj [("sites", jnat 6), ("dim", jnat n), ("matrix_symmetric", jbool symOk),
        ("rayleigh_quotient_rounded", jrat mu), ("radius", jrat r), ("radius_at_record_E0", jrat rrec),
        ("count_point", jrat pt), ("count", jnat cnt), ("delta", jrat dl), ("count_attempts", jarr tried),
        ("lowest_certified", jbool lowestOk)]),
    ("pair_singlet_identities", jbool pairOk),
    ("eigenvalue_upper_from_record_numbers", jrat upFromRecord),
    ("eigenvalue_upper_lean", jrat upLean),
    ("record", jobj [("E0_ring", jrat e0), ("E0_radius", jrat rad), ("eigenvalue_upper", jrat upRec),
        ("published_bound", jrat pub)]),
    ("record_float_reproduced", jbool recFloatMatches), ("record_enclosure_recertified", jbool recEnc),
    ("below_published_bound_lean", jbool (upLean < pub)), ("margin_lean", jrat (pub - upLean)),
    ("agreement", jbool agree), ("seconds", jsec ((t1 - t0).toFloat / 1000.0))]
  writeJson outPath out
  IO.eprintln s!"crown S=1: Lean upper bound {ratSci upLean}, record {ratSci upRec}, agreement {agree}"
  return (if agree then 0 else 1)

end CertifyApps
