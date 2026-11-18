import CertifyCore.Dyadic
/-
Exact inertia of a symmetric matrix with dyadic entries, by fraction-free (Bareiss) symmetric elimination.

For a symmetric integer matrix M, Bareiss elimination with symmetric (diagonal) pivoting produces the leading
principal minors d_1, d_2, ... of P M P^T. With d_0 = 1, the LDL^T factorisation of P M P^T has D_k = d_k / d_{k-1},
so by Sylvester's law of inertia the number of negative eigenvalues of M is the number of sign changes in
d_0, d_1, ..., d_r, where r is the rank. When every remaining diagonal entry is zero but an off-diagonal entry
M_ij is not, the congruence that adds row and column j to row and column i makes the diagonal entry 2 M_ij
nonzero and keeps the inertia; because a Bareiss entry is a minor of the original matrix, linear in its row and
in its column, the update stays exact. When the remaining block is zero, its size is the number of zero
eigenvalues. Every division is exact (Sylvester's determinant identity); `inertiaInt` checks this and reports an
error if a remainder is ever nonzero, so a coding error cannot silently give a count.

A shift sigma and the operator's entries are binary64 numbers, hence dyadic; the block A - sigma I is scaled by a
power of two to integers, which does not change the inertia. The operator is split into the connected
components of its sparsity graph (a symmetric permutation to block-diagonal form), and the inertias add.
-/
namespace CertifyCore

structure Inertia where
  neg : Nat
  zero : Nat
  pos : Nat
  deriving Inhabited, Repr, BEq

instance : Add Inertia := ⟨fun a b => ⟨a.neg + b.neg, a.zero + b.zero, a.pos + b.pos⟩⟩

/-- Inertia of a symmetric integer matrix (rows). `Except` carries the exactness check. -/
def inertiaInt (M0 : Array (Array Int)) : Except String Inertia := Id.run do
  let n := M0.size
  let mut M := M0
  let mut prev : Int := 1
  let mut neg := 0
  let mut pos := 0
  let mut k := 0
  while k < n do
    -- pivot: first remaining nonzero diagonal entry
    let mut p := n
    for i in [k:n] do
      if p == n && M[i]![i]! != 0 then p := i
    if p == n then
      -- all remaining diagonal entries zero: look for a nonzero off-diagonal entry
      let mut pi := n
      let mut pj := n
      for i in [k:n] do
        if pi == n then
          for j in [i+1:n] do
            if pi == n && M[i]![j]! != 0 then
              pi := i
              pj := j
      if pi == n then
        return .ok ⟨neg, n - k, pos⟩
      -- congruence: row pi += row pj, then column pi += column pj (columns >= k)
      let rj := M[pj]!
      M := M.modify pi (fun ri => Id.run do
        let mut r := ri
        for c in [k:n] do
          r := r.set! c (r[c]! + rj[c]!)
        return r)
      for r in [k:n] do
        M := M.modify r (fun row => row.set! pi (row[pi]! + row[pj]!))
      p := pi
    -- symmetric swap p <-> k
    if p != k then
      M := M.swapIfInBounds p k
      for r in [0:n] do
        M := M.modify r (fun row => row.swapIfInBounds p k)
    let d := M[k]![k]!
    if d * prev < 0 then neg := neg + 1 else pos := pos + 1
    let rowk := M[k]!
    for i in [k+1:n] do
      let mik := M[i]![k]!
      let mut row := M[i]!
      M := M.set! i #[]
      for j in [k+1:n] do
        let num := d * row[j]! - mik * rowk[j]!
        let q := num / prev
        if q * prev != num then
          return .error s!"inexact Bareiss division at step {k} ({i},{j})"
        row := row.set! j q
      M := M.set! i row
    prev := d
    k := k + 1
  return .ok ⟨neg, 0, pos⟩



/-- A sparse symmetric integer matrix (rows of (column, value)). -/
abbrev SpInt := Array (Array (Nat × Int))

/-- Value at (i, j) of a sparse row matrix (rows sorted by column). -/
def spGet (M : SpInt) (i j : Nat) : Int := Id.run do
  let row := M[i]!
  let mut lo := 0
  let mut hi := row.size
  while lo < hi do
    let mid := (lo + hi) / 2
    let (c, v) := row[mid]!
    if c == j then return v
    if c < j then lo := mid + 1 else hi := mid
  return 0

/-- Whether the permutation g (an involution) leaves M invariant: M[g i][g j] = M[i][j] for every entry. -/
def invariantUnder (M : SpInt) (g : Array Nat) : Bool := Id.run do
  let n := M.size
  if g.size != n then return false
  for i in [0:n] do
    if g[g[i]!]! != i then return false
    if M[i]!.size != M[g[i]!]!.size then return false
    for (j, v) in M[i]! do
      if spGet M g[i]! g[j]! != v then return false
  return true

/-- Connected components of a sparse symmetric matrix. -/
def spComponents (M : SpInt) : Array (Array Nat) := Id.run do
  let n := M.size
  let mut comp : Array Nat := Array.replicate n n
  let mut out : Array (Array Nat) := #[]
  for s in [0:n] do
    if comp[s]! == n then
      let c := out.size
      comp := comp.set! s c
      let mut stack := #[s]
      let mut members := #[]
      while stack.size > 0 do
        let v := stack.back!
        stack := stack.pop
        members := members.push v
        for (w, _) in M[v]! do
          if comp[w]! == n then
            comp := comp.set! w c
            stack := stack.push w
      out := out.push (members.qsort (· < ·))
  return out

/-- Dense block of M on the index set `c` (sorted). -/
def denseBlock (M : SpInt) (c : Array Nat) : Array (Array Int) := Id.run do
  let n := M.size
  let mut pos : Array Nat := Array.replicate n n
  for li in [0:c.size] do
    pos := pos.set! c[li]! li
  let mut out := Array.mkEmpty c.size
  for gi in c do
    let mut row := Array.replicate c.size (0 : Int)
    for (j, v) in M[gi]! do
      if pos[j]! < c.size then row := row.set! pos[j]! v
    out := out.push row
  return out

/-- Inertia of M by components and dense Bareiss. Also returns the largest block size. -/
def spInertia (M : SpInt) : Except String (Inertia × Nat) := do
  let mut tot : Inertia := ⟨0, 0, 0⟩
  let mut big := 0
  for c in spComponents M do
    big := max big c.size
    let r ← inertiaInt (denseBlock M c)
    tot := tot + r
  return (tot, big)

/-- Parity of the number of set bits. -/
def parity (x : Nat) : Bool := Id.run do
  let mut v := x
  let mut p := false
  while v > 0 do
    if v % 2 == 1 then p := !p
    v := v / 2
  return p

/-- Block reduction by an abelian group G = <g_1, ..., g_m> of commuting involutive permutations that leave M
invariant. For a character chi of G (a sign per generator) and an orbit O with representative r on whose
stabiliser chi is trivial, v_{chi,O} = sum_{k in G} chi(k) e_{k r}. These vectors form a basis, and
v_{chi,O}^T M v_{chi',O'} = 0 for chi != chi', so M is congruent to the direct sum over chi of the matrices
B_chi[O1][O2] = |G| sum_{k in G} chi(k) M[r1][k r2]; the factor |G| > 0 is dropped (inertia unchanged).
`elems` lists the group elements as (generator-exponent bitmask, permutation). -/
def charBlocks (M : SpInt) (gens : Array (Array Nat)) : Except String (Array SpInt) := do
  let n := M.size
  let m := gens.size
  -- all 2^m group elements
  let mut elems : Array (Nat × Array Nat) := #[(0, (List.range n).toArray)]
  for l in [0:m] do
    let g := gens[l]!
    let cur := elems
    for (mask, p) in cur do
      elems := elems.push (mask ||| (1 <<< l), p.map (fun i => g[i]!))
  -- orbits: representative = minimal index; for each x, an element index k with k(rep) = x
  let mut rep : Array Nat := Array.replicate n n
  let mut reps : Array Nat := #[]
  for x in [0:n] do
    if rep[x]! == n then
      reps := reps.push x
      for (_, p) in elems do
        if rep[p[x]!]! == n then rep := rep.set! p[x]! x
  let chiOf : Nat → Nat → Int := fun chi mask => if parity (chi &&& mask) then -1 else 1
  let mut blocks := #[]
  for chi in [0:2 ^ m] do
    -- orbits whose stabiliser chi is trivial on
    let mut idx : Array Nat := Array.replicate n n
    let mut basis : Array Nat := #[]
    for r in reps do
      let ok := elems.all (fun (mask, p) => p[r]! != r || chiOf chi mask == 1)
      if ok then
        idx := idx.set! r basis.size
        basis := basis.push r
    -- B[O1][O2] = sum over entries j of row r1, k in G with k r2 = j: chi(k) M[r1][j]
    let mut B : SpInt := Array.mkEmpty basis.size
    for r1 in basis do
      let mut acc : Array (Nat × Int) := #[]
      for (j, v) in M[r1]! do
        let r2 := rep[j]!
        let b2 := idx[r2]!
        if b2 < basis.size then
          for (mask, p) in elems do
            if p[r2]! == j then
              let w := chiOf chi mask * v
              match acc.findIdx? (fun e => e.1 == b2) with
              | some t => acc := acc.modify t (fun e => (e.1, e.2 + w))
              | none => acc := acc.push (b2, w)
      let row := (acc.filter (fun e => e.2 != 0)).qsort (fun a b => a.1 < b.1)
      B := B.push row
    -- exact symmetry of the block (a cheap consistency check of the construction)
    for i in [0:B.size] do
      for (j, v) in B[i]! do
        if spGet B j i != v then throw "character block is not symmetric"
    blocks := blocks.push B
  -- dimension check: the bases of all characters together span R^n
  let total := blocks.foldl (fun s B => s + B.size) 0
  if total != n then throw s!"character bases have total size {total}, not {n}"
  return blocks

/-- Bit complement and bit reversal of log2(n)-bit indices, the candidate symmetries for a spin-1/2 basis. -/
def candidateInvolutions (n : Nat) : Array (Array Nat) := Id.run do
  let nb := Nat.log2 n
  if n == 0 || 2 ^ nb != n then return #[]
  let compl := (List.range n).toArray.map (fun i => n - 1 - i)
  let rev := (List.range n).toArray.map (fun i => Id.run do
    let mut r := 0
    for b in [0:nb] do
      if (i >>> b) &&& 1 == 1 then r := r ||| (1 <<< (nb - 1 - b))
    return r)
  return #[compl, rev]


/-- A real symmetric sparse matrix in CSR form with exact entries. -/
structure Csr where
  n : Nat
  indptr : Array Nat
  indices : Array Nat
  data : Array Dy
  deriving Inhabited

namespace Csr

def row (A : Csr) (i : Nat) : Array (Nat × Dy) := Id.run do
  let mut out := #[]
  for t in [A.indptr[i]!:A.indptr[i+1]!] do
    out := out.push (A.indices[t]!, A.data[t]!)
  return out

/-- Connected components of the sparsity graph (each sorted increasingly). -/
def components (A : Csr) : Array (Array Nat) := Id.run do
  let mut comp : Array Nat := Array.replicate A.n A.n
  let mut out : Array (Array Nat) := #[]
  for s in [0:A.n] do
    if comp[s]! == A.n then
      let c := out.size
      comp := comp.set! s c
      let mut stack := #[s]
      let mut members := #[]
      while stack.size > 0 do
        let v := stack.back!
        stack := stack.pop
        members := members.push v
        for t in [A.indptr[v]!:A.indptr[v+1]!] do
          let w := A.indices[t]!
          if comp[w]! == A.n then
            comp := comp.set! w c
            stack := stack.push w
      out := out.push (members.qsort (· < ·))
  return out

/-- A - sigma I scaled by a power of two to integers, as a sparse symmetric integer matrix. -/
def shiftedInt (A : Csr) (sigma : Dy) : SpInt := Id.run do
  let mut emin : Int := sigma.e
  for v in A.data do
    if v.m != 0 && v.e < emin then emin := v.e
  let toInt : Dy → Int := fun v => if v.m == 0 then 0 else v.m * Dy.pow2 (v.e - emin).toNat
  let s := toInt sigma
  let mut M : SpInt := Array.mkEmpty A.n
  for i in [0:A.n] do
    let mut row : Array (Nat × Int) := #[]
    let mut hasDiag := false
    for t in [A.indptr[i]!:A.indptr[i+1]!] do
      let j := A.indices[t]!
      if j == i then
        hasDiag := true
        row := row.push (j, toInt A.data[t]! - s)
      else row := row.push (j, toInt A.data[t]!)
    if !hasDiag then row := row.push (i, -s)
    M := M.push ((row.filter (fun e => e.2 != 0)).qsort (fun a b => a.1 < b.1))
  return M

/-- The generators among `cands` that leave the pattern-and-value structure of A - sigma I invariant
(sigma I is invariant under any permutation, so this is a property of A), kept only if they commute with the
generators already kept and enlarge the group. -/
def symmetryGenerators (M : SpInt) : Array (Array Nat) := Id.run do
  let mut gens : Array (Array Nat) := #[]
  for g in candidateInvolutions M.size do
    if invariantUnder M g then
      let commutes := gens.all (fun h => (List.range M.size).all (fun i => g[h[i]!]! == h[g[i]!]!))
      let isNew := gens.all (fun h => h != g) && !(gens.size == 2)
      if commutes && isNew then
        -- reject g if it equals a product of kept generators (the group must have order 2^m)
        let prods := gens.foldl (fun acc h => acc ++ acc.map (fun p => p.map (fun i => h[i]!))) #[(List.range M.size).toArray]
        if !(prods.any (fun p => p == g)) then gens := gens.push g
  return gens

/-- Exact inertia of A - sigma I with the symmetry reduction; returns the inertia, the number of symmetry
generators used and the largest dense block factorised. -/
def inertiaShiftSym (A : Csr) (gens : Array (Array Nat)) (sigma : Dy) : Except String (Inertia × Nat × Nat) := do
  let M := shiftedInt A sigma
  let blocks ← charBlocks M gens
  let mut tot : Inertia := ⟨0, 0, 0⟩
  let mut big := 0
  for B in blocks do
    let (r, b) ← spInertia B
    tot := tot + r
    big := max big b
  return (tot, gens.size, big)

/-- Exact inertia of A - sigma I (sigma dyadic), block by block over the connected components. -/
def inertiaShift (A : Csr) (comps : Array (Array Nat)) (sigma : Dy) : Except String Inertia := Id.run do
  let mut total : Inertia := ⟨0, 0, 0⟩
  let mut pos : Array Nat := Array.replicate A.n 0
  for c in comps do
    let m := c.size
    for li in [0:m] do
      pos := pos.set! c[li]! li
    -- common exponent of the block's entries and of sigma
    let mut emin : Int := sigma.e
    for gi in c do
      for t in [A.indptr[gi]!:A.indptr[gi+1]!] do
        let v := A.data[t]!
        if v.m != 0 && v.e < emin then emin := v.e
    let toInt : Dy → Int := fun v => if v.m == 0 then 0 else v.m * Dy.pow2 (v.e - emin).toNat
    let s := toInt sigma
    let mut M : Array (Array Int) := Array.replicate m (Array.replicate m 0)
    for li in [0:m] do
      let gi := c[li]!
      let mut row := Array.replicate m (0 : Int)
      for t in [A.indptr[gi]!:A.indptr[gi+1]!] do
        row := row.set! (pos[A.indices[t]!]!) (toInt A.data[t]!)
      row := row.set! li (row[li]! - s)
      M := M.set! li row
    match inertiaInt M with
    | .ok r => total := total + r
    | .error e => return .error e
  return .ok total

end Csr
end CertifyCore
