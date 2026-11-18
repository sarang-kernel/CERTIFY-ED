import CertifyApps.Vbs
/-!
# AKLT Hamiltonians in the scaled basis (bilinear, biquadratic and higher terms)

The edge term of the AKLT model is the projector `P_{S_u + S_v}` onto the largest total spin of the pair, a
polynomial in `X = S_u . S_v`: `P = prod_{J < S_u + S_v} (X - x_J) / (x_{Jmax} - x_J)` with
`x_J = (J (J + 1) - S_u (S_u + 1) - S_v (S_v + 1)) / 2`; for two spins 1 it is
`1/3 + X/2 + X^2/6`, the bilinear-biquadratic form. In the scaled basis of `Vbs.lean`
(`D = diag(sqrt(binom(z, k)))`, digit `k` = number of down virtual spins, `S = z / 2`), `D^-1 S^+ D` has the
entry `z - k + 1` at `(k - 1, k)`, `D^-1 S^- D` the entry `k` at `(k, k - 1)`, and `S^z = S - k`, so every
matrix here is rational (`apps/aklt/kernel.py`, `scaled_pmax`, `hamiltonian_region`).
-/
namespace CertifyApps

abbrev QMat := Array (Array Rat)

def qmatMul (A B : QMat) : QMat :=
  let n := A.size
  let m := B[0]!.size
  let k := B.size
  (Array.range n).map (fun i => (Array.range m).map (fun j =>
    (Array.range k).foldl (fun s t => s + A[i]![t]! * B[t]![j]!) 0))

def qkron (A B : QMat) : QMat :=
  let n := A.size
  let m := B.size
  (Array.range (n * m)).map (fun i => (Array.range (n * m)).map (fun j =>
    A[i / m]![j / m]! * B[i % m]![j % m]!))

def qeye (n : Nat) : QMat := (Array.range n).map (fun i => (Array.range n).map (fun j => if i == j then 1 else 0))

/-- Scaled `S^z`, `S^+`, `S^-` of a site of degree `z`. -/
def scaledOps (z : Nat) : QMat × QMat × QMat :=
  let n := z + 1
  let S : Rat := mkRat z 2
  let Zm := (Array.range n).map (fun (i : Nat) => (Array.range n).map (fun (j : Nat) => if i == j then S - ((i : Nat) : Rat) else 0))
  let Pm := (Array.range n).map (fun (i : Nat) => (Array.range n).map (fun (j : Nat) => if j ≥ 1 && i == j - 1 then ((z - j + 1 : Nat) : Rat) else 0))
  let Mm := (Array.range n).map (fun (i : Nat) => (Array.range n).map (fun (j : Nat) => if i ≥ 1 && j == i - 1 then ((i : Nat) : Rat) else 0))
  (Zm, Pm, Mm)

/-- `(Pint, den)` with `Pint / den` the scaled projector onto the largest total spin of two sites. -/
def scaledPmax (z1 z2 : Nat) : Array (Array Int) × Nat := Id.run do
  let (Z1, P1, M1) := scaledOps z1
  let (Z2, P2, M2) := scaledOps z2
  let half : Rat := mkRat 1 2
  let ZZ := qkron Z1 Z2
  let PM := qkron P1 M2
  let MP := qkron M1 P2
  let d := ZZ.size
  let X := (Array.range d).map (fun i => (Array.range d).map (fun j => ZZ[i]![j]! + half * (PM[i]![j]! + MP[i]![j]!)))
  let S1 : Rat := mkRat z1 2
  let S2 : Rat := mkRat z2 2
  let xs := fun (J : Rat) => (J * (J + 1) - S1 * (S1 + 1) - S2 * (S2 + 1)) / 2
  let Jmax := S1 + S2
  let mut Pm := qeye d
  let mut J : Rat := if S1 ≥ S2 then S1 - S2 else S2 - S1
  while J < Jmax do
    let c := 1 / (xs Jmax - xs J)
    let F := (Array.range d).map (fun i => (Array.range d).map (fun j => (X[i]![j]! - (if i == j then xs J else 0)) * c))
    Pm := qmatMul Pm F
    J := J + 1
  let den := Pm.foldl (fun a row => row.foldl (fun b v => Nat.lcm b v.den) a) 1
  return (Pm.map (fun row => row.map (fun v => (v * (den : Rat)).num)), den)

/-- Sparse rows of `den H_X` (`H_X = sum` of the edge projectors inside the region) in the scaled basis,
sites sorted, site 0 most significant: the matrix of `hamiltonian_region`. -/
def hamiltonianRegion (g : Graph) (X : Array Nat) : Array (Array (Nat × Int)) × Nat := Id.run do
  let sites := X.qsort (· < ·)
  let pos := fun (s : Nat) => (sites.idxOf? s).getD 0
  let dims := sites.map (fun s => g.z[s]! + 1)
  let mut strides : Array Nat := Array.replicate sites.size 1
  let mut acc := 1
  let mut i := sites.size
  while i > 0 do
    i := i - 1
    strides := strides.set! i acc
    acc := acc * dims[i]!
  let N := acc
  let inner := g.edges.filter (fun (u, v) => sites.contains u && sites.contains v)
  let mut mats : Array ((Nat × Nat) × (Array (Array Int) × Nat)) := #[]
  for (u, v) in inner do
    let key := (g.z[u]!, g.z[v]!)
    if !(mats.any (fun (k, _) => k == key)) then mats := mats.push (key, scaledPmax key.1 key.2)
  let den := mats.foldl (fun a (_, (_, d)) => Nat.lcm a d) 1
  let mut rows : Array (Array (Nat × Int)) := Array.replicate N #[]
  for (u, v) in inner do
    let some (_, (P, d)) := mats.find? (fun (k, _) => k == (g.z[u]!, g.z[v]!)) | continue
    let f : Int := ((den / d : Nat) : Int)
    let iu := pos u
    let iv := pos v
    let dv := dims[iv]!
    for col in [0:N] do
      let a := (col / strides[iu]!) % dims[iu]!
      let b := (col / strides[iv]!) % dims[iv]!
      let cp := a * dv + b
      for rp in [0:P.size] do
        let c := P[rp]![cp]!
        if c != 0 then
          let a2 := rp / dv
          let b2 := rp % dv
          let tgt := col + a2 * strides[iu]! + b2 * strides[iv]! - a * strides[iu]! - b * strides[iv]!
          rows := rows.modify tgt (·.push (col, f * c))
  return (rows, den)

/-- Exact check `H_X U_X = 0` on the reduced VBS columns of a region (the VBS range lies in the kernel). -/
def vbsInKernel (g : Graph) (X : Array Nat) : Bool × Nat × Nat := Id.run do
  let (rows, _) := hamiltonianRegion g X
  let reg := reducedRegion g X
  -- column-oriented: H col = sum over entries (tgt, src, val) with src in the column support
  let N := rows.size
  let mut bySrc : Array (Array (Nat × Int)) := Array.replicate N #[]
  for tgt in [0:N] do
    for (src, v) in rows[tgt]! do
      bySrc := bySrc.modify src (·.push (tgt, v))
  let mut ok := true
  for col in reg.cols do
    let mut acc : Std.HashMap Nat Int := {}
    for (r, a) in col do
      for (tgt, v) in bySrc[r]! do
        acc := acc.insert tgt (acc.getD tgt 0 + v * a)
    if acc.toList.any (fun (_, v) => v != 0) then ok := false
  return (ok, N, reg.n)

end CertifyApps
