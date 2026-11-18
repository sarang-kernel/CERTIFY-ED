import CertifyApps.Engine
import CertifyApps.Vector
import Std.Data.HashMap
/-!
# VBS states of AKLT models on finite graphs, in exact integer arithmetic

This is the construction of `certify_ed/apps/aklt/graph.py`, rewritten. A union graph carries at every site
`v` a spin `z_v / 2`, `z_v` its degree in the infinite lattice; its `z_v` virtual spins 1/2 sit on its
`z_v` slots, each either a union edge or a free slot (an edge that leaves the union). For a region `X`
the VBS map `U_X` sends a configuration of the virtual spins on the slots leaving `X` to the physical
state of `X`, with a singlet `(up down - down up)` on every edge inside `X`. In the scaled basis
`D = diag(sqrt(binom(z, k)))` (digit `k` = number of down virtual spins of the site) the amplitudes are
the integers `c_z / binom(z, k)` with `c_z = lcm_k binom(z, k)`, and the physical metric is
`diag(binom(z, k))`; every Gram matrix below is an integer matrix. A boundary configuration enters a VBS
state only through the number of down spins among the slots of each site, so configurations are reduced
to one representative per count pattern.
-/
namespace CertifyApps

structure Graph where
  z : Array Nat
  edges : Array (Nat × Nat)
  free : Array Nat
  deriving Inhabited

def Graph.make (z : Array Nat) (edges : Array (Nat × Nat)) : Graph := Id.run do
  let mut deg : Array Nat := Array.replicate z.size 0
  for (u, v) in edges do
    deg := deg.modify u (· + 1)
    deg := deg.modify v (· + 1)
  return ⟨z, edges, (Array.range z.size).map (fun i => z[i]! - deg[i]!)⟩

/-- A virtual slot: a union edge (by index) at one of its ends, or the `k`-th free slot of a site. -/
inductive Slot where
  | edge (e : Nat) (site : Nat)
  | free (site : Nat) (k : Nat)
  deriving Inhabited, BEq, Repr

def Slot.site : Slot → Nat
  | .edge _ s => s
  | .free s _ => s

/-- Ordered boundary slots of a region (`Graph.boundary_slots`). -/
def boundarySlots (g : Graph) (region : Array Nat) : Array Slot := Id.run do
  let mut out : Array Slot := #[]
  for ei in [0:g.edges.size] do
    let (u, v) := g.edges[ei]!
    if region.contains u != region.contains v then
      out := out.push (.edge ei (if region.contains u then u else v))
  for s in region.qsort (· < ·) do
    for k in [0:g.free[s]!] do
      out := out.push (.free s k)
  return out

/-- Reduction of boundary configurations to down counts per site (`slot_reduction`). Slot `i` is bit
`nb - 1 - i` of a full configuration. -/
structure Reduction where
  nb : Nat
  groups : Array (Array Nat)
  radix : Array Nat
  nred : Nat
  reps : Array Nat
  downs : Array Nat
  deriving Inhabited

def Reduction.ofSlots (slots : Array Slot) : Reduction := Id.run do
  let nb := slots.size
  let mut sites : Array Nat := #[]
  for sl in slots do
    if !sites.contains sl.site then sites := sites.push sl.site
  let groups := sites.map (fun s => (Array.range nb).filter (fun i => slots[i]!.site == s))
  let radix := groups.map (·.size + 1)
  let nred := radix.foldl (· * ·) 1
  -- representatives: for each count vector, downs on the largest slot indices of each group
  let mut reps : Array Nat := #[]
  let mut downs : Array Nat := #[]
  for r in [0:nred] do
    -- decode r in mixed radix, first group most significant
    let mut rr := r
    let mut cnt : Array Nat := Array.replicate groups.size 0
    let mut gi := groups.size
    while gi > 0 do
      gi := gi - 1
      cnt := cnt.set! gi (rr % radix[gi]!)
      rr := rr / radix[gi]!
    let mut f : Nat := 0
    let mut tot := 0
    for gi2 in [0:groups.size] do
      let grp := groups[gi2]!
      let c := cnt[gi2]!
      tot := tot + c
      for t in [0:c] do
        let i := grp[grp.size - 1 - t]!
        f := f ||| (1 <<< (nb - 1 - i))
    reps := reps.push f
    downs := downs.push tot
  return ⟨nb, groups, radix, nred, reps, downs⟩

/-- Reduced index of a full configuration. -/
def Reduction.red (R : Reduction) (f : Nat) : Nat := Id.run do
  let mut r := 0
  for gi in [0:R.groups.size] do
    let mut c := 0
    for i in R.groups[gi]! do
      if (f >>> (R.nb - 1 - i)) % 2 == 1 then c := c + 1
    r := r * R.radix[gi]! + c
  return r

def binom (n k : Nat) : Nat := Id.run do
  if k > n then return 0
  let mut r := 1
  for i in [0:k] do
    r := r * (n - i) / (i + 1)
  return r

def siteAmp (z : Nat) : Array Nat :=
  let c := (Array.range (z + 1)).foldl (fun a k => Nat.lcm a (binom z k)) 1
  (Array.range (z + 1)).map (fun k => c / binom z k)

/-- A region with its reduced VBS map, Gram matrix and down counts (`reduced_region`). -/
structure Region where
  sites : Array Nat
  slots : Array Slot
  red : Reduction
  n : Nat
  /-- Gram matrix in the physical metric, dense. -/
  G : Array (Array Int)
  downs : Array Nat
  inner : Nat
  /-- reduced VBS columns: (physical row index, amplitude) -/
  cols : Array (Array (Nat × Int))
  deriving Inhabited

def reducedRegion (g : Graph) (X : Array Nat) : Region := Id.run do
  let sites := X.qsort (· < ·)
  let pos := fun (s : Nat) => (sites.idxOf? s).getD 0
  let inner := g.edges.filter (fun (u, v) => sites.contains u && sites.contains v)
  let slots := boundarySlots g sites
  let R := Reduction.ofSlots slots
  let nb := slots.size
  let dims := sites.map (fun s => g.z[s]! + 1)
  -- strides, site 0 most significant
  let mut strides : Array Nat := Array.replicate sites.size 1
  let mut acc := 1
  let mut i := sites.size
  while i > 0 do
    i := i - 1
    strides := strides.set! i acc
    acc := acc * dims[i]!
  let amps := sites.map (fun s => siteAmp g.z[s]!)
  let wts := sites.map (fun s => (Array.range (g.z[s]! + 1)).map (fun k => binom g.z[s]! k))
  let mut cols : Array (Array (Nat × Int)) := #[]
  for bconf in R.reps do
    let mut col : Array (Nat × Int) := #[]
    for sconf in [0:2 ^ inner.size] do
      let mut dn : Array Nat := Array.replicate sites.size 0
      let mut sign : Int := 1
      for ei in [0:inner.size] do
        let (u, v) := inner[ei]!
        if (sconf >>> (inner.size - 1 - ei)) % 2 == 0 then dn := dn.modify (pos v) (· + 1)
        else
          dn := dn.modify (pos u) (· + 1)
          sign := -sign
      for k in [0:nb] do
        if (bconf >>> (nb - 1 - k)) % 2 == 1 then dn := dn.modify (pos slots[k]!.site) (· + 1)
      let mut amp : Int := sign
      for q in [0:sites.size] do
        amp := amp * (amps[q]![dn[q]!]! : Nat)
      let row := (Array.range sites.size).foldl (fun a q => a + dn[q]! * strides[q]!) 0
      col := col.push (row, amp)
    -- merge duplicate rows, drop zeros
    let sortedCol := col.qsort (fun a b => a.1 < b.1)
    let mut merged : Array (Nat × Int) := #[]
    for (r, a) in sortedCol do
      if merged.size > 0 && merged[merged.size - 1]!.1 == r then
        merged := merged.modify (merged.size - 1) (fun (x, y) => (x, y + a))
      else merged := merged.push (r, a)
    cols := cols.push (merged.filter (fun (_, a) => a != 0))
  let weight := fun (row : Nat) => Id.run do
    let mut w := 1
    for q in [0:sites.size] do
      w := w * wts[q]![(row / strides[q]!) % dims[q]!]!
    return w
  let n := cols.size
  let mut G : Array (Array Int) := Array.replicate n (Array.replicate n 0)
  for a in [0:n] do
    for b in [a:n] do
      -- merge of two sorted sparse columns
      let ca := cols[a]!
      let cb := cols[b]!
      let mut s : Int := 0
      let mut ia := 0
      let mut ib := 0
      while ia < ca.size && ib < cb.size do
        let (ra, va) := ca[ia]!
        let (rb, vb) := cb[ib]!
        if ra == rb then
          s := s + (weight ra : Nat) * va * vb
          ia := ia + 1
          ib := ib + 1
        else if ra < rb then ia := ia + 1 else ib := ib + 1
      G := G.modify a (·.set! b s)
      G := G.modify b (·.set! a s)
  return ⟨sites, slots, R, n, G, R.downs, inner.size, cols⟩

/-- VBS states of the union of `parts` in the product of the parts' reduced coordinates (`union_coords`):
columns as sparse maps from the mixed-radix index of the part coordinates (first part most significant)
to integers; also the reduced down counts of the union, the number of singlets inside the union, and the
union's boundary slots. -/
structure Union where
  cols : Array (Array (Nat × Int))
  downs : Array Nat
  singlets : Nat
  slots : Array Slot
  red : Reduction
  sizes : Array Nat
  deriving Inhabited

def unionCoords (g : Graph) (parts : Array (Array Nat)) (regs : Array Region) : Union := Id.run do
  let Y := (parts.foldl (fun a p => a ++ p) #[]).qsort (· < ·)
  let whereP := fun (s : Nat) => ((Array.range parts.size).find? (fun pi => parts[pi]!.contains s)).getD 0
  let inY := fun (s : Nat) => Y.contains s
  let yslots := boundarySlots g Y
  let yred := Reduction.ofSlots yslots
  let cut := (Array.range g.edges.size).filter (fun ei =>
    let (u, v) := g.edges[ei]!
    inY u && inY v && whereP u != whereP v)
  let slotPos := fun (pi : Nat) (sl : Slot) => (regs[pi]!.slots.idxOf? sl).getD 0
  let nsl := regs.map (·.slots.size)
  let ymap := yslots.map (fun sl => (whereP sl.site, slotPos (whereP sl.site) sl))
  let cmap := cut.map (fun ei =>
    let (u, v) := g.edges[ei]!
    ((whereP u, slotPos (whereP u) (.edge ei u)), (whereP v, slotPos (whereP v) (.edge ei v))))
  let sizes := regs.map (·.n)
  let mut out : Array (Array (Nat × Int)) := #[]
  for yb in yred.reps do
    let mut col : Array (Nat × Int) := #[]
    for sconf in [0:2 ^ cut.size] do
      let mut full : Array Nat := Array.replicate regs.size 0
      for k in [0:yslots.size] do
        if (yb >>> (yslots.size - 1 - k)) % 2 == 1 then
          let (pi, kk) := ymap[k]!
          full := full.modify pi (· ||| (1 <<< (nsl[pi]! - 1 - kk)))
      let mut sign : Int := 1
      for ci in [0:cut.size] do
        let ((pu, ku), (pv, kv)) := cmap[ci]!
        if (sconf >>> (cut.size - 1 - ci)) % 2 == 0 then
          full := full.modify pv (· ||| (1 <<< (nsl[pv]! - 1 - kv)))
        else
          full := full.modify pu (· ||| (1 <<< (nsl[pu]! - 1 - ku)))
          sign := -sign
      let mut key := 0
      for pi in [0:regs.size] do
        key := key * sizes[pi]! + regs[pi]!.red.red full[pi]!
      col := col.push (key, sign)
    let sortedCol := col.qsort (fun a b => a.1 < b.1)
    let mut merged : Array (Nat × Int) := #[]
    for (r, a) in sortedCol do
      if merged.size > 0 && merged[merged.size - 1]!.1 == r then
        merged := merged.modify (merged.size - 1) (fun (x, y) => (x, y + a))
      else merged := merged.push (r, a)
    out := out.push (merged.filter (fun (_, a) => a != 0))
  let singlets := (g.edges.filter (fun (u, v) => inY u && inY v)).size
  return ⟨out, yred.downs, singlets, yslots, yred, sizes⟩

/-- Digits of a mixed-radix key. -/
def keyDigits (sizes : Array Nat) (key : Nat) : Array Nat := Id.run do
  let mut d : Array Nat := Array.replicate sizes.size 0
  let mut k := key
  let mut i := sizes.size
  while i > 0 do
    i := i - 1
    d := d.set! i (k % sizes[i]!)
    k := k / sizes[i]!
  return d

/-- Gram matrix of union columns in the metric `⊗ Gs` (`gram_of_coords`). -/
def gramOfCoords (u : Union) (Gs : Array (Array (Array Int))) : Array (Array Int) := Id.run do
  let n := u.cols.size
  let dig := u.cols.map (fun c => c.map (fun (k, v) => (keyDigits u.sizes k, v)))
  let mut G : Array (Array Int) := Array.replicate n (Array.replicate n 0)
  for a in [0:n] do
    for b in [a:n] do
      let mut s : Int := 0
      for (ka, va) in dig[a]! do
        for (kb, vb) in dig[b]! do
          let mut p : Int := va * vb
          for f in [0:Gs.size] do
            if p != 0 then p := p * Gs[f]![ka[f]!]![kb[f]!]!
          s := s + p
      G := G.modify a (·.set! b s)
      G := G.modify b (·.set! a s)
  return G

/-- Rank modulo the prime `p` of a dense integer matrix (Gaussian elimination in words). -/
def rankModP (A : Array (Array Int)) (p : Nat := 2147483647) : Nat := Id.run do
  let n := A.size
  if n == 0 then return 0
  let m := A[0]!.size
  let P : UInt64 := p.toUInt64
  let mut M : Array Nat := Array.replicate (n * m) 0
  for i in [0:n] do
    for j in [0:m] do
      let v := A[i]![j]! % (p : Int)
      M := M.set! (i * m + j) (if v < 0 then (v + p).toNat else v.toNat)
  let powmod := fun (a e : UInt64) => Id.run do
    let mut r : UInt64 := 1
    let mut b := a % P
    let mut k := e
    while k > 0 do
      if k % 2 == 1 then r := r * b % P
      b := b * b % P
      k := k / 2
    return r
  let mut rank := 0
  let mut row := 0
  for col in [0:m] do
    if row < n then
      let mut piv := n
      for i in [row:n] do
        if piv == n && M[i * m + col]! != 0 then piv := i
      if piv < n then
        if piv != row then
          for j in [0:m] do
            let a := M[piv * m + j]!
            M := M.set! (piv * m + j) M[row * m + j]!
            M := M.set! (row * m + j) a
        let inv := powmod (M[row * m + col]!.toUInt64) (P - 2)
        for i in [row+1:n] do
          let f := M[i * m + col]!
          if f != 0 then
            let c := (f.toUInt64 * inv) % P
            for j in [col:m] do
              let x := M[i * m + j]!.toUInt64
              let y := M[row * m + j]!.toUInt64
              M := M.set! (i * m + j) ((x + (P - (c * y) % P)) % P).toNat
        row := row + 1
        rank := rank + 1
  return rank

/-- `T^T T` of sparse columns over integer keys (for the rank of the union VBS states). -/
def gramPlain (cols : Array (Array (Nat × Int))) : Array (Array Int) := Id.run do
  let n := cols.size
  let mut byKey : Std.HashMap Nat (Array (Nat × Int)) := {}
  for j in [0:n] do
    for (k, v) in cols[j]! do
      byKey := byKey.insert k ((byKey.getD k #[]).push (j, v))
  let mut G : Array (Array Int) := Array.replicate n (Array.replicate n 0)
  for (_, ent) in byKey.toList do
    for (i, a) in ent do
      for (j, b) in ent do
        G := G.modify i (·.modify j (· + a * b))
  return G

/-- A certified rational lower bound of the smallest eigenvalue of a positive definite integer matrix:
`c - beta` for the shift `c` (half of a floating-point estimate) at which the certified count of
`G - c I` finds no negative eigenvalue. `none` if the count fails. -/
def lambdaMinLower (G : Array (Array Int)) : IO (Option Rat) := do
  let n := G.size
  let rows : SpRows := G.map (fun (r : Array Int) => (Array.range n).filterMap (fun (j : Nat) =>
    let v : Int := r[j]!
    if v != 0 then some (j, v) else none))
  let N := Array.replicate n 1
  let ep := inverseIteration rows N 1 0 12 0 60
  if !ep.ok then return none
  let est := ep.mu
  if est ≤ 0 then return none
  -- c = est / 2, halved again while the certified count still finds an eigenvalue below c
  let mut c := roundDyadic (est / 2) 30
  for _ in [0:12] do
    if c > 0 then
      let M : IMat := .de n ((Array.range n).map (fun i =>
        (Array.range n).map (fun j => ((c.den : Nat) : Int) * G[i]![j]! - (if i == j then c.num else 0))))
      let r ← certCount M #[] 1
      if r.ok && r.neg == 0 then
        let lb := c - r.beta / ((c.den : Nat) : Rat)
        if lb > 0 then return some lb
      c := roundDyadic (c / 2) 40
  return none

end CertifyApps
