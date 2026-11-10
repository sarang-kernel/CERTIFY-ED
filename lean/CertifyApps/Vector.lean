import CertifyApps.Engine
/-!
# Eigenvector proposals and exact residual radii

The setting is a real symmetric integer matrix `A` (sparse rows) and positive integer weights `N`, with the
eigenvalue problem `A z = lambda den N z`; for a symmetry sector `A` is the sector matrix of `Hint = den H`
in the orthogonal orbit-sum basis of squared norms `N`, and the vector `v = sum_r z_r w_r` of the full space
satisfies `|| H v - mu v ||^2 = sum_i (A z - mu den N z)_i^2 / N_i / den^2` and `|| v ||^2 = sum_i N_i z_i^2`.
By Theorem 1 of certify-ed (Krylov and Bogolyubov) some eigenvalue of `H` restricted to the sector lies
within `|| H v - mu v || / || v ||` of `mu`, for any nonzero `v` and any real `mu`.

The vector is a proposal and may come from anywhere. Here it is computed by inverse iteration at a shift,
first in floating point, then in integer arithmetic: each step solves the shifted system by iterative
refinement (a floating-point solve with the factorization of the shifted matrix, an exact integer residual,
repeated), so that the vector is accurate far beyond binary64. The radius is then computed exactly from the
integer vector, whatever its accuracy.
-/
namespace CertifyApps

abbrev SpRows := Array (Array (Nat × Int))

/-- `A z` exactly. -/
def spMulInt (A : SpRows) (z : Array Int) : Array Int :=
  A.map (fun row => row.foldl (fun s (j, v) => s + v * z[j]!) 0)

/-- Bit length of the largest absolute entry. -/
def maxBits (z : Array Int) : Nat := z.foldl (fun m x => max m (if x == 0 then 0 else Nat.log2 x.natAbs + 1)) 0

/-- Shift every entry right (rounding to nearest) or left so that the largest has `p` bits. -/
def normaliseBits (z : Array Int) (p : Nat) : Array Int :=
  let b := maxBits z
  if b == 0 then z
  else if b > p then
    let s := b - p
    let h : Int := (2 : Int) ^ (s - 1)
    let d : Int := (2 : Int) ^ s
    z.map (fun x => if x ≥ 0 then (x + h) / d else -((-x + h) / d))
  else
    let d : Int := (2 : Int) ^ (p - b)
    z.map (· * d)

/-- Floating-point copy of an integer vector divided by `2^e`. -/
def intVecToFloat (z : Array Int) (e : Nat) : FloatArray := Id.run do
  let mut f := fzeros z.size
  for i in [0:z.size] do
    f := f.set! i (intToFloatScaled z[i]! e)
  return f

/-- A factorization of the scaled shifted matrix `D B D 2^-sigmaB`, `D = N^-1/2`, for solving `B y = c`. -/
structure ShiftSolver where
  n : Nat
  B : SpRows
  f : Ldl
  dscale : FloatArray
  sigmaB : Nat

def ShiftSolver.make (B : SpRows) (N : Array Nat) : Option ShiftSolver := Id.run do
  let n := B.size
  let M : IMat := .sp n B
  let mx := M.maxAbs
  if mx == 0 then return none
  let sigmaB := Nat.log2 mx + 1
  let mut d := fzeros n
  for i in [0:n] do
    d := d.set! i (1.0 / (Float.ofNat N[i]!).sqrt)
  let mut a := fzeros (n * n)
  for i in [0:n] do
    let r := M.row i
    for j in [0:i+1] do
      let v := r[j]!
      if v != 0 then a := a.set! (i * n + j) (intToFloatScaled v sigmaB * d.get! i * d.get! j)
  let f := ldlFactor n a
  if !f.ok then return none
  return some ⟨n, B, f, d, sigmaB⟩

/-- Approximately `B^-1 c` as an integer vector times `2^-k` for the returned `k`: floating point. -/
def ShiftSolver.solveF (s : ShiftSolver) (c : Array Int) : Array Int × Int := Id.run do
  let e := maxBits c
  -- x solves (D B D 2^-sigmaB) x = D c 2^-e, then B^-1 c = D x 2^(e - sigmaB)
  let mut rhs := intVecToFloat c e
  for i in [0:s.n] do
    rhs := rhs.set! i (rhs.get! i * s.dscale.get! i)
  let x := ldlSolve s.f rhs
  -- y = D x, returned as round(y 2^60) with scale 2^(e - sigmaB - 60)
  let mut y : Array Int := Array.replicate s.n 0
  for i in [0:s.n] do
    y := y.set! i (floatToFixedInt (x.get! i * s.dscale.get! i) 60)
  return (y, (e : Int) - (s.sigmaB : Int) - 60)

/-- High-precision solution of `B y = c` by iterative refinement: returns `y` (integer) and `k` with
`B (y 2^k) ≈ c`, and the bit length of the final exact residual relative to `c`. -/
def ShiftSolver.solveHP (s : ShiftSolver) (c : Array Int) (prec : Nat) (steps : Nat := 12) :
    Array Int × Int × Int := Id.run do
  let (y0, k0) := s.solveF c
  -- work at scale 2^k with k = k0 - prec (so y has about prec + 60 bits)
  let k : Int := k0 - (prec : Int)
  let mut y : Array Int := y0.map (· * (2 : Int) ^ prec)
  -- residual r = c 2^-k - B y  (integer since -k ≥ 0 is ensured below)
  let ck : Array Int := if k ≤ 0 then c.map (· * (2 : Int) ^ (-k).toNat) else c
  let cb := maxBits ck
  let mut rel : Int := 0
  for _ in [0:steps] do
    let by_ := spMulInt s.B y
    let r := (Array.range s.n).map (fun i => ck[i]! - by_[i]!)
    let rb := maxBits r
    rel := (rb : Int) - (cb : Int)
    if rb == 0 || rel < -((prec : Int) + 20) then break
    let (d, kd) := s.solveF r
    -- d 2^kd ≈ B^-1 r  (r at scale 2^k): add d 2^(kd) to y (exact shift if kd ≥ 0, rounding otherwise)
    if kd ≥ 0 then
      let m := (2 : Int) ^ kd.toNat
      y := (Array.range s.n).map (fun i => y[i]! + d[i]! * m)
    else
      let q := (2 : Int) ^ (-kd).toNat
      let h := q / 2
      y := (Array.range s.n).map (fun i =>
        let x := d[i]!
        y[i]! + (if x ≥ 0 then (x + h) / q else -((-x + h) / q)))
  return (y, k, rel)

/-- Exact squared radius `|| H v - mu v ||^2 / || v ||^2` for `mu = p/q`, and `|| v ||^2 > 0`. -/
def radiusSq (A : SpRows) (N : Array Nat) (den : Nat) (z : Array Int) (mu : Rat) : Option Rat := Id.run do
  let n := A.size
  let W := N.foldl (fun a b => Nat.lcm a b) 1
  let p := mu.num
  let q : Int := mu.den
  let az := spMulInt A z
  let mut num : Int := 0
  let mut x2 : Int := 0
  for i in [0:n] do
    let ri := q * az[i]! - p * (den : Int) * ((N[i]! : Nat) : Int) * z[i]!
    num := num + ri * ri * ((W / N[i]! : Nat) : Int)
    x2 := x2 + ((N[i]! : Nat) : Int) * z[i]! * z[i]!
  if x2 == 0 then return none
  -- r^2 = num / W / (q^2 den^2 x2)
  return some (mkRat num ((W : Nat) * (q.natAbs ^ 2) * (den ^ 2) * x2.natAbs))

/-- Rayleigh quotient `z^T A z / (den z^T N z)`. -/
def rayleigh (A : SpRows) (N : Array Nat) (den : Nat) (z : Array Int) : Rat :=
  let az := spMulInt A z
  let num := (Array.range z.size).foldl (fun s i => s + z[i]! * az[i]!) (0 : Int)
  let dd := (Array.range z.size).foldl (fun s i => s + ((N[i]! : Nat) : Int) * z[i]! * z[i]!) (0 : Int)
  mkRat num ((den : Nat) * dd.natAbs) * (if dd < 0 then -1 else 1)

/-- Round a rational to a dyadic with `k` bits after the point (nearest). -/
def roundDyadic (x : Rat) (k : Nat) : Rat :=
  let s : Int := (2 : Int) ^ k
  let num := x.num * s
  let d : Int := x.den
  let q := if num ≥ 0 then (2 * num + d) / (2 * d) else -((2 * (-num) + d) / (2 * d))
  mkRat q (2 ^ k)

structure EigenProposal where
  ok : Bool
  msg : String
  z : Array Int
  mu : Rat
  steps : Nat
  lastRel : Int

/-- Inverse iteration for the eigenvalue of `A z = lambda den N z` nearest the shift `sigma` (rational).
`fsteps` floating-point steps, then `hsteps` high-precision steps with `prec` bits. -/
def inverseIteration (A : SpRows) (N : Array Nat) (den : Nat) (sigma : Rat)
    (fsteps hsteps prec : Nat) : EigenProposal := Id.run do
  let n := A.size
  let a := sigma.num
  let b : Int := sigma.den
  -- B = b A - a den N
  let B : SpRows := (Array.range n).map (fun i =>
    (A[i]!.map (fun (j, v) => (j, b * v))).push (i, -(a * (den : Int) * ((N[i]! : Nat) : Int))))
  match ShiftSolver.make B N with
  | none => return ⟨false, "factorization of the shifted matrix failed", #[], 0, 0, 0⟩
  | some s =>
    -- start: all ones times 2^60
    let mut z : Array Int := Array.replicate n ((2 : Int) ^ 60)
    for _ in [0:fsteps] do
      let c := (Array.range n).map (fun i => ((N[i]! : Nat) : Int) * z[i]!)
      let (y, _) := s.solveF c
      z := normaliseBits y 60
    let mut rel : Int := 0
    let mut steps := 0
    for _ in [0:hsteps] do
      let c := (Array.range n).map (fun i => ((N[i]! : Nat) : Int) * z[i]!)
      let (y, _, r) := s.solveHP c prec
      rel := r
      z := normaliseBits y prec
      steps := steps + 1
    return ⟨true, "", z, rayleigh A N den z, steps, rel⟩

end CertifyApps
