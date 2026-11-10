import CertifyApps
import CertifyCore.Inertia
open CertifyApps

/-- Deterministic pseudo-random integers (64-bit LCG). -/
def lcg (x : UInt64) : UInt64 := x * 6364136223846793005 + 1442695040888963407

/-- A symmetric test matrix with entries of about `bits` bits, diagonal shifted by `s` times the
entry scale (indefinite for moderate `s`). -/
def testMatrix (n bits : Nat) (s : Int) (seed : UInt64) : IMat := Id.run do
  let mut st := seed
  let mut a : Array (Array Int) := Array.replicate n (Array.replicate n 0)
  for i in [0:n] do
    for j in [0:i+1] do
      st := lcg st
      let base : Int := ((st >>> 20).toNat % 2001 : Nat)
      let v : Int := (base - 1000) * (2 : Int) ^ bits / 1000 + (if i == j then s * (2 : Int) ^ bits else 0)
      a := a.modify i (·.set! j v)
      a := a.modify j (·.set! i v)
  return .de n a

def fmt (q : Rat) : String := toString (ratToFloat q)

def selftest : IO UInt32 := do
  let mut bad := 0
  for (n, bits, s, wts) in [(8, 10, (0 : Int), false), (20, 40, 3, false), (40, 70, 6, false), (40, 130, 6, false),
      (30, 90, 0, true), (60, 50, 10, false), (60, 100, 12, true), (90, 60, 0, false)] do
    let M := testMatrix n bits s (n.toUInt64 + bits.toUInt64)
    let w : Array Nat := if wts then (Array.range n).map (fun i => 2 ^ (i % 5)) else #[]
    let r ← certCount M w 2
    -- exact elimination of the same matrix
    let rows := match M with | .de _ rows => rows | _ => #[]
    let ex := CertifyCore.inertiaInt rows
    -- exact residual by integers, for comparison with the bucket bound
    let ww := if w.size == n then w else Array.replicate n 1
    let W := lcmAll ww
    let wq := ww.map (fun x => W / x)
    let mx := M.maxAbs
    let sigma0 := Nat.log2 mx + 1
    let mut a := fzeros (n * n)
    for i in [0:n] do
      let rr := M.row i
      for j in [0:i+1] do
        a := a.set! (i * n + j) (intToFloatScaled rr[j]! sigma0 / (Float.ofNat ww[i]!).sqrt / (Float.ofNat ww[j]!).sqrt)
    let f := ldlFactor n a
    let ds : FloatArray := Id.run do
      let mut d := fzeros n
      for i in [0:n] do d := d.set! i (1.0 / (Float.ofNat ww[i]!).sqrt)
      return d
    let lb := makeLimbs f ds
    let F := lb.F[0]!
    let rr : Int := 2 * F - (sigma0 : Int)
    let (u, v) : Nat × Nat := if rr ≥ 0 then (0, rr.toNat) else ((-rr).toNat, 0)
    let froEx := residualExactInt M f.perm lb u v wq
    let bound := match residualRows M f.perm lb u v wq 0 n with
      | some (b, _) => b
      | none => 0
    let okBound := bound ≥ froEx && (bound : Rat) ≤ (froEx : Rat) * (1 + 1 / 2 ^ 20) + 1
    match ex with
    | .ok e =>
        let agree := r.ok && r.neg == e.neg && e.zero == 0 && okBound
        IO.println s!"n={n} bits={bits} s={s} weights={wts}: neg {r.neg} (exact {e.neg}), beta {fmt r.beta}, bucket bound / exact sum {ratToFloat (mkRat bound (max froEx 1))}, u={u} v={v}: {if agree then "agree" else "DISAGREE"}"
        if !agree then bad := bad + 1
    | .error err => IO.println s!"exact elimination failed: {err}"; bad := bad + 1
  return (if bad == 0 then 0 else 1)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["bench", ns, ts] =>
      let n := ns.toNat!
      let M := testMatrix n 60 4 7
      let r ← certCount M #[] ts.toNat!
      IO.println s!"n={n} ok={r.ok} {r.msg} neg={r.neg} pos={r.pos} beta={fmt r.beta} tf={r.secFactor} tr={r.secResidual}"
      return 0
  | ["selftest"] => selftest
  | ["e7", key, recPath, out, ts] => runE7 key recPath out ts.toNat!
  | ["crown-half", recPath, out, ts] => runCrownHalf recPath out ts.toNat!
  | ["crown-half-largest", recPath, out, ts, k] => runCrownHalf recPath out ts.toNat! k.toNat!
  | ["crown-one", recPath, out] => runCrownOne recPath out
  | ["crown-sizes"] => do
      let pr ← prepare (crownModel 381414 817957) 9 crownGenerators (Array.replicate 7 none)
      IO.println s!"{pr.sectors.map (·.dim)}"
      return 0
  | ["aklt", name, recPath, out, ts] => runAklt name recPath out ts.toNat!
  | ["aklt-so-criterion", perp, par, recPath, out] => runSquareOctagonCriterion perp par recPath out
  | ["export-aklt", name, out] => exportAklt name out
  | ["export-aklt-ham", name, out] => exportAkltHam name out
  | ["export", key, out] => exportHeis key out
  | ["export-sector", key, ex, out] => exportSector key ex out
  | ["export-spin1-ring", out] => exportSpin1Ring out
  | _ => IO.println "usage: ceapps bench n threads | selftest"; return 2
