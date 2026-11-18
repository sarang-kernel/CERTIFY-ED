/-
Exact dyadic rationals m * 2^e and the two conversions the checker needs: the IEEE-754 binary64 bit pattern of
a stored number to its exact value, and a decimal lexeme of the certificate JSON to the binary64 number nearest
to it (round half to even, as Python's `float()` and `json.load` do). Every value the checker compares is one
of these exact numbers or an exact sum and product of them; no floating-point operation is used anywhere.
-/
namespace CertifyCore

/-- The exact number `m * 2^e`. Not normalised: equal values may have different representations. -/
structure Dy where
  m : Int
  e : Int
  deriving Inhabited, Repr

namespace Dy

@[inline] def pow2 (k : Nat) : Int := Int.ofNat (2 ^ k)

def zero : Dy := ⟨0, 0⟩
def one : Dy := ⟨1, 0⟩
def ofInt (z : Int) : Dy := ⟨z, 0⟩

def add (a b : Dy) : Dy :=
  if a.m == 0 then b else if b.m == 0 then a
  else if a.e ≤ b.e then ⟨a.m + b.m * pow2 (b.e - a.e).toNat, a.e⟩
  else ⟨a.m * pow2 (a.e - b.e).toNat + b.m, b.e⟩

def neg (a : Dy) : Dy := ⟨-a.m, a.e⟩
def sub (a b : Dy) : Dy := add a (neg b)
def mul (a b : Dy) : Dy := if a.m == 0 || b.m == 0 then zero else ⟨a.m * b.m, a.e + b.e⟩
def sq (a : Dy) : Dy := mul a a
def abs (a : Dy) : Dy := ⟨a.m.natAbs, a.e⟩
/-- Multiplication by 2^k (k may be negative): exact. -/
def scale2 (a : Dy) (k : Int) : Dy := ⟨a.m, a.e + k⟩

instance : Add Dy := ⟨add⟩
instance : Sub Dy := ⟨sub⟩
instance : Mul Dy := ⟨mul⟩
instance : Neg Dy := ⟨neg⟩

def sign (a : Dy) : Int := a.m.sign
def isZero (a : Dy) : Bool := a.m == 0
/-- a < b, a ≤ b exactly. -/
def lt (a b : Dy) : Bool := (sub a b).m < 0
def le (a b : Dy) : Bool := (sub a b).m ≤ 0
def beq (a b : Dy) : Bool := (sub a b).m == 0
def max (a b : Dy) : Dy := if lt a b then b else a
def min (a b : Dy) : Dy := if lt a b then a else b

def toRat (a : Dy) : Rat :=
  if a.e ≥ 0 then ((a.m * pow2 a.e.toNat : Int) : Rat) else mkRat a.m (2 ^ (-a.e).toNat)

/-- Remove trailing zero bits of the mantissa (keeps numbers small after long sums). -/
def normalize (a : Dy) : Dy := Id.run do
  if a.m == 0 then return zero
  let mut m := a.m
  let mut e := a.e
  while m % 2 == 0 do
    m := m / 2
    e := e + 1
  return ⟨m, e⟩

def sum (xs : Array Dy) : Dy := xs.foldl add zero

/-- Decimal rendering with `digits` significant digits (for reports only, never for a decision). -/
def toSci (a : Dy) (digits : Nat := 6) : String :=
  let q := a.toRat
  if q == 0 then "0" else
  let neg := q < 0
  let q := if neg then -q else q
  -- find k with 10^(digits-1) <= q * 10^k < 10^digits
  Id.run do
    let mut k : Int := 0
    let mut v := q
    let lo : Rat := (10 ^ (digits - 1) : Nat)
    let hi : Rat := (10 ^ digits : Nat)
    let mut guard := 0
    while v < lo && guard < 4000 do
      v := v * 10; k := k + 1; guard := guard + 1
    while v ≥ hi && guard < 8000 do
      v := v / 10; k := k - 1; guard := guard + 1
    let n := v.floor.toNat
    let s := toString n
    let exp := (digits : Int) - 1 - k
    let mant := if s.length > 1 then String.ofList (s.toList.take 1) ++ "." ++ String.ofList (s.toList.drop 1) else s
    return (if neg then "-" else "") ++ mant ++ "e" ++ toString exp

end Dy

/-- A binary64 value: finite (exact), infinite, or NaN. -/
inductive F64 where
  | fin (d : Dy)
  | inf (negative : Bool)
  | nan
  deriving Inhabited

namespace F64

/-- Exact value of an IEEE-754 binary64 bit pattern. -/
def ofBits (b : UInt64) : F64 :=
  let negative := (b >>> 63) == 1
  let ex := ((b >>> 52) &&& 0x7ff).toNat
  let fr := (b &&& 0xfffffffffffff).toNat
  if ex == 2047 then (if fr == 0 then inf negative else nan)
  else
    let mant : Int := if ex == 0 then fr else fr + 2 ^ 52
    let e : Int := if ex == 0 then -1074 else (ex : Int) - 1075
    fin ⟨if negative then -mant else mant, e⟩

def fin? : F64 → Option Dy
  | fin d => some d
  | _ => none

def isFinite : F64 → Bool
  | fin _ => true
  | _ => false

end F64

/-- Little-endian unsigned 64-bit word at byte offset `off`. -/
def readU64 (b : ByteArray) (off : Nat) : UInt64 := Id.run do
  let mut x : UInt64 := 0
  for i in [0:8] do
    x := x ||| ((b.get! (off + i)).toUInt64 <<< (8 * i).toUInt64)
  return x

def readI64 (b : ByteArray) (off : Nat) : Int :=
  let u := (readU64 b off).toNat
  if u ≥ 2 ^ 63 then (u : Int) - 2 ^ 64 else u

/-- Nearest binary64 to the rational `num / den` (num ≥ 0, den > 0), round half to even;
`none` on overflow. Returned as an exact dyadic. -/
def nearestF64 (num den : Nat) : Option Dy := Id.run do
  if num == 0 then return some Dy.zero
  -- binary exponent e with 2^52 <= num / (den 2^e) < 2^53, clamped below at -1074 (subnormals)
  let mut e : Int := (Nat.log2 num : Int) - (Nat.log2 den : Int) - 52
  -- adjust so that the scaled quotient lies in [2^52, 2^53)
  let q : Int → Nat × Nat := fun e =>
    if e ≥ 0 then (num, den * 2 ^ e.toNat) else (num * 2 ^ (-e).toNat, den)
  for _ in [0:4] do
    let (a, b) := q e
    if a / b ≥ 2 ^ 53 then e := e + 1
    else if a / b < 2 ^ 52 then e := e - 1
  if e < -1074 then e := -1074
  let (a, b) := q e
  let mut m := a / b
  let r := a % b
  if 2 * r > b || (2 * r == b && m % 2 == 1) then m := m + 1
  if m == 2 ^ 53 then
    m := 2 ^ 52
    e := e + 1
  if e > 971 then return none
  return some ⟨m, e⟩

/-- Value of a JSON number lexeme as Python's `float()` reads it (nearest binary64). -/
def decToF64 (lex : String) : Option F64 := Id.run do
  let cs := lex.toList
  let (neg, cs) := match cs with
    | '-' :: t => (true, t)
    | t => (false, t)
  let mut digits : Nat := 0
  let mut exp10 : Int := 0
  let mut i := 0
  let arr := cs.toArray
  -- integer part
  while i < arr.size && arr[i]!.isDigit do
    digits := 10 * digits + (arr[i]!.toNat - 48)
    i := i + 1
  if i < arr.size && arr[i]! == '.' then
    i := i + 1
    while i < arr.size && arr[i]!.isDigit do
      digits := 10 * digits + (arr[i]!.toNat - 48)
      exp10 := exp10 - 1
      i := i + 1
  if i < arr.size && (arr[i]! == 'e' || arr[i]! == 'E') then
    i := i + 1
    let mut eneg := false
    if i < arr.size && (arr[i]! == '+' || arr[i]! == '-') then
      eneg := arr[i]! == '-'
      i := i + 1
    let mut ev : Nat := 0
    while i < arr.size && arr[i]!.isDigit do
      ev := 10 * ev + (arr[i]!.toNat - 48)
      i := i + 1
    exp10 := exp10 + (if eneg then -(ev : Int) else ev)
  if i != arr.size then return none
  if digits == 0 then return some (F64.fin Dy.zero)
  if exp10 > 400 then return some (F64.inf neg)
  if exp10 < -800 then return some (F64.fin Dy.zero)
  let (num, den) := if exp10 ≥ 0 then (digits * 10 ^ exp10.toNat, 1) else (digits, 10 ^ (-exp10).toNat)
  match nearestF64 num den with
  | some d => return some (F64.fin (if neg then d.neg else d))
  | none => return some (F64.inf neg)

end CertifyCore
