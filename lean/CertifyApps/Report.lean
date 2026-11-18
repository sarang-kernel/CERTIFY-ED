import CertifyCore.Json
import CertifyCore.Dyadic
import CertifyApps.Engine
/-!
# Reading run records and writing results

Record values are read from the `data.json` files of the run records with the JSON reader of `CertifyCore`;
a decimal number is taken as the binary64 number nearest to it (the value Python's `json.load` gives it,
`CertifyCore.decToF64`), and a string `"p/q"` as the exact rational. Results are written as JSON with every
exact rational as a `"p/q"` string next to a 17-digit decimal rendering for reading.
-/
namespace CertifyApps
open CertifyCore

def readJsonFile (path : String) : IO JV := do
  let b ← IO.FS.readBinFile path
  match Json.parse b with
  | .ok v => return v
  | .error e => throw (IO.userError s!"{path}: {e}")

/-- The exact value of a JSON number as the nearest binary64 number. -/
def jvNumRat (j : JV) : Option Rat :=
  match j.num? with
  | some lex => match decToF64 lex with
    | some (F64.fin d) => some d.toRat
    | _ => none
  | none => none

def parseNatStr (s : String) : Option Nat := s.toNat?

def parseIntStr (s : String) : Option Int :=
  if s.startsWith "-" then (s.drop 1).toString.toNat?.map (fun n => -(n : Int)) else s.toNat?.map (fun n => (n : Int))

/-- `"p/q"` or an integer string as a rational. -/
def parseRatStr (s : String) : Option Rat :=
  match s.splitOn "/" with
  | [p] => (parseIntStr p).map (fun z => (z : Rat))
  | [p, q] => match parseIntStr p, parseNatStr q with
    | some a, some b => if b == 0 then none else some (mkRat a b)
    | _, _ => none
  | _ => none

def jvRatStr (j : JV) : Option Rat := j.str?.bind parseRatStr

def jvNat (j : JV) : Option Nat := j.num?.bind (·.toNat?)

/-- Decimal rendering of a rational with `digits` significant digits (rounded to nearest; reports only). -/
def ratSci (q : Rat) (digits : Nat := 17) : String := Id.run do
  if q == 0 then return "0"
  let neg := q < 0
  let a : Rat := if neg then -q else q
  -- find k with 10^(digits-1) <= a 10^k < 10^digits
  let lg := (Nat.log2 a.num.natAbs : Int) - (Nat.log2 a.den : Int)    -- log2 a, within 1
  let mut k : Int := (digits : Int) - 1 - (lg * 3 / 10)
  let lo : Rat := (10 ^ (digits - 1) : Nat)
  let hi : Rat := (10 ^ digits : Nat)
  let scale := fun (k : Int) => if k ≥ 0 then a * ((10 ^ k.toNat : Nat) : Rat) else a / ((10 ^ (-k).toNat : Nat) : Rat)
  let mut v := scale k
  let mut guard := 0
  while (v < lo || v ≥ hi) && guard < 200 do
    if v < lo then k := k + 1 else k := k - 1
    v := scale k
    guard := guard + 1
  let n := (v + 1/2).floor.toNat
  let (n2, k2) := if n ≥ 10 ^ digits then (n / 10, k - 1) else (n, k)
  let s := toString n2
  let e := (digits : Int) - 1 - k2
  let mant := if s.length > 1 then String.ofList (s.toList.take 1) ++ "." ++ String.ofList (s.toList.drop 1) else s
  return (if neg then "-" else "") ++ mant ++ "e" ++ toString e

def ratStr (q : Rat) : String := if q.den == 1 then toString q.num else s!"{q.num}/{q.den}"

def jstr (s : String) : JV := .str s
def jnat (n : Nat) : JV := .num (toString n)
def jint (n : Int) : JV := .num (toString n)
def jbool (b : Bool) : JV := .bool b
/-- An exact rational with its decimal rendering. -/
def jrat (q : Rat) : JV := .obj #[("exact", .str (ratStr q)), ("decimal", .str (ratSci q))]
def jsec (x : Float) : JV := .num (toString ((x * 1000.0).round / 1000.0))
def jobj (kvs : List (String × JV)) : JV := .obj kvs.toArray
def jarr (xs : Array JV) : JV := .arr xs

def writeJson (path : String) (v : JV) : IO Unit := do
  IO.FS.writeBinFile path (Json.canonical v)

/-- Round a rational to the nearest binary64 number (ties to even, via `CertifyCore`'s
`nearestF64`), returned as a rational; `none` on overflow. Used to test whether a record float is
the binary64 number nearest an exact value. -/
def ratNearestF64 (q : Rat) : Option Rat :=
  if q == 0 then some 0 else
  let neg := q < 0
  let a := if neg then -q else q
  match nearestF64 a.num.natAbs a.den with
  | some d => some (if neg then -d.toRat else d.toRat)
  | none => none

end CertifyApps
