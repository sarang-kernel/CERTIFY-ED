/-
A JSON reader that keeps every number as the exact text it was written with, and the canonical serialisation
`certify-ed-canonical-json-1` of certify-ed (Python `json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False)`).

Why not `Lean.Json`: its numbers are decimal mantissa/exponent pairs, which forget whether a value was written
`1`, `1.0` or `1e0`; the canonical form needs the exact lexeme. A certificate file is itself canonical JSON
written by Python, so every number lexeme in it is Python's `repr` of the float (or the integer), and
re-emitting the lexeme reproduces the bytes that were hashed. The value of a lexeme is decoded separately
(`CertifyCore.Dyadic.decToF64`), as the nearest binary64 number, the value Python's `json.load` gives it.
Strings are compared and sorted by their UTF-8 bytes, which orders them by code point, as Python sorts `str` keys.
-/
namespace CertifyCore

inductive JV where
  | null
  | bool (b : Bool)
  | num (lex : String)
  | str (s : String)
  | arr (xs : Array JV)
  | obj (kvs : Array (String × JV))
  deriving Inhabited

namespace JV

def get? (j : JV) (k : String) : Option JV :=
  match j with
  | obj kvs => (kvs.find? (fun p => p.1 == k)).map (·.2)
  | _ => none

def arr? : JV → Option (Array JV)
  | arr xs => some xs
  | _ => none

def str? : JV → Option String
  | str s => some s
  | _ => none

def num? : JV → Option String
  | num s => some s
  | _ => none

def bool? : JV → Option Bool
  | bool b => some b
  | _ => none

def isNull : JV → Bool
  | null => true
  | _ => false

/-- Follow a path of object keys. -/
def path? (j : JV) : List String → Option JV
  | [] => some j
  | k :: ks => match j.get? k with
    | some v => v.path? ks
    | none => none

/-- The object without the member `k` (used to drop `integrity` before hashing the body). -/
def erase (j : JV) (k : String) : JV :=
  match j with
  | obj kvs => obj (kvs.filter (fun p => p.1 != k))
  | v => v

end JV

namespace Json

structure St where
  bytes : ByteArray
  pos : Nat

abbrev P := StateT St (Except String)

def peek : P (Option UInt8) := do
  let s ← get
  return if s.pos < s.bytes.size then some (s.bytes.get! s.pos) else none

def adv : P Unit := modify fun s => { s with pos := s.pos + 1 }

def fail {α} (msg : String) : P α := do
  let s ← get
  throw s!"JSON parse error at byte {s.pos}: {msg}"

def skipWs : P Unit := do
  repeat
    match (← peek) with
    | some c => if c == 32 || c == 9 || c == 10 || c == 13 then adv else break
    | none => break

def expectByte (c : UInt8) : P Unit := do
  match (← peek) with
  | some d => if d == c then adv else fail s!"expected byte {c}, found {d}"
  | none => fail "unexpected end"

def expectLit (s : String) : P Unit := do
  for c in s.toUTF8.toList do expectByte c

def isDigit (c : UInt8) : Bool := 48 ≤ c && c ≤ 57

def hexVal (c : UInt8) : Option Nat :=
  if 48 ≤ c && c ≤ 57 then some (c.toNat - 48)
  else if 97 ≤ c && c ≤ 102 then some (c.toNat - 87)
  else if 65 ≤ c && c ≤ 70 then some (c.toNat - 55)
  else none

def hex4 : P Nat := do
  let mut v := 0
  for _ in [0:4] do
    match (← peek) with
    | some c => match hexVal c with
      | some h =>
        v := 16 * v + h
        adv
      | none => fail "bad \\u escape"
    | none => fail "unexpected end in \\u escape"
  return v

/-- UTF-8 encoding of one code point. -/
def utf8 (cp : Nat) : List UInt8 :=
  if cp < 0x80 then [cp.toUInt8]
  else if cp < 0x800 then [(0xC0 + cp / 64).toUInt8, (0x80 + cp % 64).toUInt8]
  else if cp < 0x10000 then [(0xE0 + cp / 4096).toUInt8, (0x80 + (cp / 64) % 64).toUInt8, (0x80 + cp % 64).toUInt8]
  else [(0xF0 + cp / 262144).toUInt8, (0x80 + (cp / 4096) % 64).toUInt8, (0x80 + (cp / 64) % 64).toUInt8,
        (0x80 + cp % 64).toUInt8]

def parseString : P String := do
  expectByte 34
  let mut out : ByteArray := ByteArray.empty
  repeat
    match (← peek) with
    | none => fail "unterminated string"
    | some c =>
      if c == 34 then
        adv
        break
      else if c == 92 then
        adv
        match (← peek) with
        | some e =>
          adv
          if e == 34 then out := out.push 34
          else if e == 92 then out := out.push 92
          else if e == 47 then out := out.push 47
          else if e == 98 then out := out.push 8
          else if e == 102 then out := out.push 12
          else if e == 110 then out := out.push 10
          else if e == 114 then out := out.push 13
          else if e == 116 then out := out.push 9
          else if e == 117 then
            let hi ← hex4
            let mut cp := hi
            if 0xD800 ≤ hi && hi < 0xDC00 then
              expectByte 92
              expectByte 117
              let lo ← hex4
              if !(0xDC00 ≤ lo && lo < 0xE000) then fail "unpaired surrogate"
              cp := 0x10000 + (hi - 0xD800) * 1024 + (lo - 0xDC00)
            else if 0xDC00 ≤ hi && hi < 0xE000 then fail "unpaired surrogate"
            for b in utf8 cp do out := out.push b
          else fail "bad escape"
        | none => fail "unexpected end in escape"
      else if c < 32 then fail "control character in string"
      else
        out := out.push c
        adv
  match String.fromUTF8? out with
  | some s => return s
  | none => fail "string is not valid UTF-8"

/-- A number, returned as its exact text after checking the JSON number grammar. -/
def parseNumber : P String := do
  let s0 := (← get).pos
  if (← peek) == some 45 then adv
  match (← peek) with
  | some 48 => adv
  | some c => if isDigit c && c != 48 then
                adv
                while (← peek).any isDigit do adv
              else fail "bad number"
  | none => fail "bad number"
  if (← peek) == some 46 then
    adv
    if !((← peek).any isDigit) then fail "bad fraction"
    while (← peek).any isDigit do adv
  if (← peek) == some 101 || (← peek) == some 69 then
    adv
    if (← peek) == some 43 || (← peek) == some 45 then adv
    if !((← peek).any isDigit) then fail "bad exponent"
    while (← peek).any isDigit do adv
  let s ← get
  match String.fromUTF8? (s.bytes.extract s0 s.pos) with
  | some t => return t
  | none => fail "number text"

partial def parseValue : P JV := do
  skipWs
  match (← peek) with
  | none => fail "unexpected end"
  | some c =>
    if c == 123 then
      adv
      skipWs
      let mut kvs : Array (String × JV) := #[]
      if (← peek) == some 125 then
        adv
        return JV.obj kvs
      repeat
        skipWs
        let k ← parseString
        skipWs
        expectByte 58
        let v ← parseValue
        if kvs.any (fun p => p.1 == k) then fail s!"duplicate key {k}"
        kvs := kvs.push (k, v)
        skipWs
        match (← peek) with
        | some 44 => adv
        | some 125 => do adv; break
        | _ => fail "expected , or }"
      return JV.obj kvs
    else if c == 91 then
      adv
      skipWs
      let mut xs : Array JV := #[]
      if (← peek) == some 93 then
        adv
        return JV.arr xs
      repeat
        let v ← parseValue
        xs := xs.push v
        skipWs
        match (← peek) with
        | some 44 => adv
        | some 93 => do adv; break
        | _ => fail "expected , or ]"
      return JV.arr xs
    else if c == 34 then return JV.str (← parseString)
    else if c == 116 then
      expectLit "true"
      return JV.bool true
    else if c == 102 then
      expectLit "false"
      return JV.bool false
    else if c == 110 then
      expectLit "null"
      return JV.null
    else return JV.num (← parseNumber)

def parseTop : P JV := do
  let v ← parseValue
  skipWs
  let s ← get
  if s.pos != s.bytes.size then fail "trailing data"
  return v

def parse (b : ByteArray) : Except String JV :=
  match parseTop.run ⟨b, 0⟩ with
  | .ok (v, _) => .ok v
  | .error e => .error e

/-- Lexicographic order of UTF-8 byte strings (= code point order of the strings). -/
def bytesLt (a b : ByteArray) : Bool := Id.run do
  let n := min a.size b.size
  for i in [0:n] do
    let x := a.get! i
    let y := b.get! i
    if x < y then return true
    if x > y then return false
  return a.size < b.size

def hexDigit (n : Nat) : UInt8 := if n < 10 then (48 + n).toUInt8 else (87 + n).toUInt8

/-- Python `json.dumps(s, ensure_ascii=False)` of a string, appended to `out`. -/
def emitString (out : ByteArray) (s : String) : ByteArray := Id.run do
  let mut o := out.push 34
  for c in s.toUTF8.toList do
    if c == 34 then o := (o.push 92).push 34
    else if c == 92 then o := (o.push 92).push 92
    else if c == 10 then o := (o.push 92).push 110
    else if c == 13 then o := (o.push 92).push 114
    else if c == 9 then o := (o.push 92).push 116
    else if c == 8 then o := (o.push 92).push 98
    else if c == 12 then o := (o.push 92).push 102
    else if c < 32 then
      o := (((((o.push 92).push 117).push 48).push 48).push (hexDigit (c.toNat / 16))).push (hexDigit (c.toNat % 16))
    else o := o.push c
  return o.push 34

def appendStr (out : ByteArray) (s : String) : ByteArray := out ++ s.toUTF8

/-- Canonical serialisation (sorted keys, `,` and `:` separators, numbers as their lexeme). -/
partial def emit (out : ByteArray) : JV → ByteArray
  | JV.null => appendStr out "null"
  | JV.bool b => appendStr out (if b then "true" else "false")
  | JV.num l => appendStr out l
  | JV.str s => emitString out s
  | JV.arr xs => Id.run do
      let mut o := out.push 91
      let mut first := true
      for x in xs do
        if !first then o := o.push 44
        first := false
        o := emit o x
      return o.push 93
  | JV.obj kvs => Id.run do
      let sorted := kvs.qsort (fun p q => bytesLt p.1.toUTF8 q.1.toUTF8)
      let mut o := out.push 123
      let mut first := true
      for (k, v) in sorted do
        if !first then o := o.push 44
        first := false
        o := emitString o k
        o := o.push 58
        o := emit o v
      return o.push 125

def canonical (v : JV) : ByteArray := emit ByteArray.empty v

end Json
end CertifyCore
