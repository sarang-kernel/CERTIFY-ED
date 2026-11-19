import CertifyCore.Dyadic
/-!
# Counting eigenvalues from a proposed factorization and an exact residual

This is the counting step of every application number. For a symmetric integer matrix `M` the program

1. scales `M` by a power of two and factors it in binary64 floating point as `P^T M P ≈ L D L^T`
   (`L` unit lower triangular, `D` diagonal, `P` the permutation of symmetric diagonal pivoting);
2. forms `G = L |D|^(1/2)` and rounds it to an integer matrix `Gt` (entries below `2^62` in magnitude),
   and `S = sign(D)`;
3. computes the residual `E = Gt S Gt^T 2^(sigma - 2F) - P^T M P` in exact integer arithmetic, where
   `2^F` is the scale of `Gt` and `2^sigma` the scale of the floating-point copy of `M`;
4. returns the number of negative and positive entries of `S` and a rational `beta` with
   `beta ≥ ‖E‖₂`, namely the smaller of the exact infinity norm and an upper bound of the Frobenius norm.

Floating point only proposes `Gt` and `S`. Whatever they are, `Gt` is lower triangular with nonzero diagonal
(checked), so `Gt S Gt^T` has the inertia of `S` (Sylvester's law of inertia), and `Gt S Gt^T 2^(sigma-2F)`
equals `P^T M P + E` exactly. By Weyl's inequality every eigenvalue of `M + E'` (`E'` similar to `E`) lies
within `beta` of the corresponding eigenvalue of `M`, hence

  #(eigenvalues of M below -beta) ≤ neg ≤ #(eigenvalues of M below beta).

Sylvester's law and Weyl's inequality are used as stated; they are not formalised in this library.

The exact product `Gt S Gt^T` is computed with 21-bit limbs in 64-bit words: each entry of `Gt` is split
into three signed limbs, the 9 limb products (each below `2^42`) are accumulated by degree in five 64-bit
words with wrapping arithmetic, and since the true value of each accumulator is below `2^63` in magnitude
(at most `3 n 2^42` with `n < 2^19`, checked) the wrapped word determines it. The limbs are stored bit-cast
in `FloatArray`s (no floating-point operation touches them), biased by `2^21` so that every stored pattern
is a small positive number.
-/
namespace CertifyApps

/-! ## Small helpers -/

/-- Bit-cast read and write of a 64-bit word stored in a `FloatArray`. -/
@[inline] def rdW (a : FloatArray) (i : Nat) : UInt64 := (a.get! i).toBits
@[inline] def wrW (a : FloatArray) (i : Nat) (w : UInt64) : FloatArray := a.set! i (Float.ofBits w)

def fzeros (n : Nat) : FloatArray := Id.run do
  let mut x := FloatArray.emptyWithCapacity n
  for _ in [0:n] do
    x := x.push 0.0
  return x

/-- Largest `r` with `r * r ≤ n` (Newton's method on naturals, checked). -/
def isqrt (n : Nat) : Nat := Id.run do
  if n < 2 then return n
  let mut x := 2 ^ ((Nat.log2 n) / 2 + 1)
  let mut y := (x + n / x) / 2
  while y < x do
    x := y
    y := (x + n / x) / 2
  return x

/-- A rational upper bound of `sqrt q` for `q ≥ 0`, with `k` binary digits after the point. -/
def sqrtUp (q : Rat) (k : Nat := 160) : Rat :=
  if q ≤ 0 then 0 else
  let s : Nat := 4 ^ k
  -- c ≥ q 4^k, then sqrt q ≤ sqrt c / 2^k ≤ (isqrt c + 1) / 2^k
  let c : Nat := ((q.num.natAbs * s) + q.den - 1) / q.den
  mkRat ((isqrt c + 1 : Nat) : Int) (2 ^ k)

/-- A rational lower bound of `sqrt q` for `q ≥ 0`. -/
def sqrtDown (q : Rat) (k : Nat := 160) : Rat :=
  if q ≤ 0 then 0 else
  let c : Nat := (q.num.natAbs * 4 ^ k) / q.den
  mkRat ((isqrt c : Nat) : Int) (2 ^ k)

/-- Signed value of a 64-bit two's complement word. -/
@[inline] def sval (w : UInt64) : Int := w.toInt64.toInt

def pow21 : Int := 2097152
def pow42 : Int := 4398046511104
def pow63 : Int := 9223372036854775808
def pow84 : Int := 19342813113834066795298816

/-- Exact decoding of a finite binary64 number: (negative, mantissa, exponent), |x| = m 2^e. -/
@[inline] def decodeF (x : Float) : Bool × UInt64 × Int :=
  let b := x.toBits
  let neg := (b >>> 63) == 1
  let ex := (b >>> 52) &&& 0x7ff
  let fr := b &&& 0xfffffffffffff
  if ex == 0 then (neg, fr, -1074) else (neg, fr ||| 0x10000000000000, (ex.toNat : Int) - 1075)

/-- Exact rational value of a finite binary64 number. -/
def floatToRat (x : Float) : Rat :=
  let (neg, m, e) := decodeF x
  let v : Rat := if e ≥ 0 then ((m.toNat * 2 ^ e.toNat : Nat) : Rat) else mkRat (m.toNat : Int) (2 ^ (-e).toNat)
  if neg then -v else v

/-- `round(|x| 2^F)` for `|x| = m 2^e` (round half up), as a word; the caller guarantees it is below 2^62. -/
@[inline] def toFixed (m : UInt64) (e : Int) (F : Int) : UInt64 :=
  let s := e + F
  if s ≥ 0 then m <<< s.toNat.toUInt64
  else
    let t := (-s).toNat
    if t ≥ 64 then 0 else (m + ((1 : UInt64) <<< (t - 1).toUInt64)) >>> t.toUInt64

/-- `round(x 2^F)` as an integer, exactly, for a finite double `x` (round half away from zero). -/
def floatToFixedInt (x : Float) (F : Int) : Int :=
  let (neg, m, e) := decodeF x
  let s := e + F
  let mag : Nat :=
    if s ≥ 0 then m.toNat * 2 ^ s.toNat
    else
      let t := (-s).toNat
      (m.toNat + 2 ^ t / 2) / 2 ^ t
  if neg then -(mag : Int) else (mag : Int)

/-- A double close to the integer `z` times `2^(-sh)` (for proposals and reports only). -/
def intToFloatScaled (z : Int) (sh : Nat) : Float :=
  let a := z.natAbs
  if a == 0 then 0.0 else
  let bl := Nat.log2 a
  -- keep the top 63 bits, then scale
  let (top, drop) := if bl > 62 then (a >>> (bl - 62), bl - 62) else (a, 0)
  let f := (top.toUInt64.toFloat).scaleB ((drop : Int) - (sh : Int))
  if z < 0 then -f else f

/-- A double close to a rational (reports and proposals only). -/
def ratToFloat (q : Rat) : Float :=
  if q.num == 0 then 0.0 else
  let a := q.num.natAbs
  let sh : Int := (Nat.log2 a : Int) - (Nat.log2 q.den : Int) - 60
  -- q = a/den; scale to get 60+ significant bits in the integer quotient
  let (num, den) := if sh ≥ 0 then (a, q.den * 2 ^ sh.toNat) else (a * 2 ^ (-sh).toNat, q.den)
  let f := (Float.ofNat (num / den)).scaleB sh
  if q.num < 0 then -f else f

/-! ## Integer symmetric matrices -/

/-- A symmetric integer matrix: sparse rows (column, value) with both triangles stored, or dense rows. -/
inductive IMat where
  | sp (n : Nat) (rows : Array (Array (Nat × Int)))
  | de (n : Nat) (rows : Array (Array Int))
  deriving Inhabited

namespace IMat

def dim : IMat → Nat
  | sp n _ => n
  | de n _ => n

/-- Dense copy of row `i`. -/
def row (M : IMat) (i : Nat) : Array Int :=
  match M with
  | sp n rows => Id.run do
      let mut r : Array Int := Array.replicate n 0
      for (j, v) in rows[i]! do
        r := r.set! j (r[j]! + v)
      return r
  | de _ rows => rows[i]!

/-- Largest absolute value of an entry. -/
def maxAbs (M : IMat) : Nat := Id.run do
  let mut m : Nat := 0
  match M with
  | sp _ rows =>
      for r in rows do
        for (_, v) in r do
          if v.natAbs > m then m := v.natAbs
  | de _ rows =>
      for r in rows do
        for v in r do
          if v.natAbs > m then m := v.natAbs
  return m

/-- Exact symmetry check. -/
def isSymmetric (M : IMat) : Bool := Id.run do
  let n := M.dim
  match M with
  | sp _ _ =>
      for i in [0:n] do
        let ri := M.row i
        for j in [0:i] do
          if ri[j]! != (M.row j)[i]! then return false
      return true
  | de _ rows =>
      for i in [0:n] do
        for j in [0:i] do
          if rows[i]![j]! != rows[j]![i]! then return false
      return true

end IMat

/-- Symmetry check for sparse rows without forming dense rows (uses a scratch map per row). -/
def spSymmetric (n : Nat) (rows : Array (Array (Nat × Int))) : Bool := Id.run do
  -- every entry (i,j,v) must have a matching (j,i,v); compare sorted triples
  let mut a : Array (Nat × Nat × Int) := #[]
  let mut b : Array (Nat × Nat × Int) := #[]
  for i in [0:n] do
    for (j, v) in rows[i]! do
      if v != 0 then
        a := a.push (i, j, v)
        b := b.push (j, i, v)
  let lt : Nat × Nat × Int → Nat × Nat × Int → Bool := fun x y =>
    x.1 < y.1 || (x.1 == y.1 && (x.2.1 < y.2.1 || (x.2.1 == y.2.1 && x.2.2 < y.2.2)))
  let sa := a.qsort lt
  let sb := b.qsort lt
  return sa.size == sb.size && (Array.range sa.size).all (fun k => sa[k]! == sb[k]!)

/-! ## Floating-point LDL^T with symmetric diagonal pivoting -/

structure Ldl where
  n : Nat
  /-- `perm[k]` is the original index placed at position `k`. -/
  perm : Array Nat
  /-- Row-major n × n: strictly lower part holds `L`, diagonal holds `D`. -/
  a : FloatArray
  ok : Bool
  msg : String

@[inline] def fswap (a : FloatArray) (x y : Nat) : FloatArray :=
  let vx := a.get! x
  let vy := a.get! y
  (a.set! x vy).set! y vx

/-- `a[base + j] -= li * v[j]` for `j` in `[j, stop)`. -/
partial def rowUpd (a : FloatArray) (base j stop : Nat) (li : Float) (v : FloatArray) : FloatArray :=
  if j < stop then
    let x := a.get! (base + j) - li * v.get! j
    rowUpd (a.set! (base + j) x) base (j + 1) stop li v
  else a

partial def swapRange1 (a : FloatArray) (x y j stop : Nat) : FloatArray :=
  -- swap a[x+j] and a[y+j] for j in [j, stop)
  if j < stop then swapRange1 (fswap a (x + j) (y + j)) x y (j + 1) stop else a

/-- Factor the symmetric matrix whose lower triangle is in `a0` (row-major n × n). -/
def ldlFactor (n : Nat) (a0 : FloatArray) : Ldl := Id.run do
  let mut a := a0
  let mut perm := Array.range n
  let mut v := fzeros n
  for k in [0:n] do
    let mut p := k
    let mut best := (a.get! (k * n + k)).abs
    for i in [k+1:n] do
      let x := (a.get! (i * n + i)).abs
      if x > best then
        best := x
        p := i
    if best == 0.0 || !best.isFinite then
      return ⟨n, perm, a, false, s!"zero or non-finite pivot at step {k}"⟩
    if p != k then
      a := swapRange1 a (k * n) (p * n) 0 k
      a := fswap a (k * n + k) (p * n + p)
      for i in [k+1:p] do
        a := fswap a (i * n + k) (p * n + i)
      for i in [p+1:n] do
        a := fswap a (i * n + k) (i * n + p)
      perm := perm.swapIfInBounds k p
    let d := a.get! (k * n + k)
    for i in [k+1:n] do
      v := v.set! i (a.get! (i * n + k))
    for i in [k+1:n] do
      let li := v.get! i / d
      a := rowUpd a (i * n) (k + 1) (i + 1) li v
      a := a.set! (i * n + k) li
  return ⟨n, perm, a, true, ""⟩

/-- Solve `P^T M P y = b` approximately with the factorization (`b`, `y` in original coordinates,
scaled as the factored matrix). Floating point; used only to propose vectors. -/
def ldlSolve (f : Ldl) (b : FloatArray) : FloatArray := Id.run do
  let n := f.n
  let mut z := fzeros n
  for k in [0:n] do
    z := z.set! k (b.get! f.perm[k]!)
  -- forward: L w = z
  for i in [0:n] do
    let mut s := z.get! i
    for k in [0:i] do
      s := s - f.a.get! (i * n + k) * z.get! k
    z := z.set! i s
  -- diagonal
  for i in [0:n] do
    z := z.set! i (z.get! i / f.a.get! (i * n + i))
  -- backward: L^T y = w
  let mut i := n
  while i > 0 do
    i := i - 1
    let mut s := z.get! i
    for k in [i+1:n] do
      s := s - f.a.get! (k * n + i) * z.get! k
    z := z.set! i s
  let mut y := fzeros n
  for k in [0:n] do
    y := y.set! f.perm[k]! (z.get! k)
  return y

/-! ## Exact residual -/

/-- The rounded factor `Gt`: row `i` is `round(G_i 2^(F_i))` (a row-dependent power of two; the diagonal
congruence by `2^(-F_i)` keeps the factor lower triangular with nonzero diagonal). Each entry is packed into
one machine word below `2^63`: bit 62 is the sign, bits 0 to 61 the magnitude. -/
structure Limbs where
  n : Nat
  g : Array Nat
  /-- 1 where `S_k = -1`, 0 where `S_k = +1`. -/
  sg : ByteArray
  F : Array Int
  ok : Bool
  msg : String

@[inline] def triIdx (i : Nat) : Nat := i * (i + 1) / 2

/-- Round the factor `G = D^-1 L |D_d|^(1/2)` of `M` (the float factorization was of `D M D`,
`D = diag(dscale)`) to integers `round(G 2^F)` with a global `F` such that every entry is below `2^62`. -/
def makeLimbs (f : Ldl) (dscale : FloatArray) : Limbs := Id.run do
  let n := f.n
  let mut sq := fzeros n
  let mut sg := ByteArray.emptyWithCapacity n
  for k in [0:n] do
    let d := f.a.get! (k * n + k)
    sq := sq.set! k d.abs.sqrt
    sg := sg.push (if d < 0.0 then 1 else 0)
  let tot := triIdx n
  let mut g : Array Nat := Array.replicate tot 0
  let mut gmax : Float := 0.0
  for i in [0:n] do
    let ri := 1.0 / dscale.get! f.perm[i]!
    for k in [0:i+1] do
      let gk := ((if k == i then sq.get! k else f.a.get! (i * n + k) * sq.get! k) * ri).abs
      if gk > gmax then gmax := gk
  if !(gmax > 0.0) || !gmax.isFinite then
    return ⟨n, g, sg, #[], false, "the factor is zero or not finite"⟩
  let (_, c) := gmax.frExp            -- gmax < 2^c
  let F : Int := 61 - c
  for i in [0:n] do
    let base := triIdx i
    let ri := 1.0 / dscale.get! f.perm[i]!
    for k in [0:i+1] do
      let gk := (if k == i then sq.get! k else f.a.get! (i * n + k) * sq.get! k) * ri
      let (neg, m, e) := decodeF gk
      let mag := if m == 0 then 0 else toFixed m e F
      if k == i && mag == 0 then
        return ⟨n, g, sg, #[F], false, s!"rounded factor has a zero diagonal entry at {i}"⟩
      g := g.set! (base + k) (mag.toNat + (if neg && mag != 0 then 4611686018427387904 else 0))
  return ⟨n, g, sg, #[F], true, ""⟩

/-- Accumulate `sum_k s_k g_ik g_jk` over `k` in `[k, m)` with 21-bit limbs, by degree, in wrapping words. -/
partial def dot5 (G : Array Nat) (sg : ByteArray) (bi bj k m : Nat)
    (c0 c1 c2 c3 c4 : UInt64) : UInt64 × UInt64 × UInt64 × UInt64 × UInt64 :=
  if k < m then
    let x := G[bi + k]!.toUInt64
    let y := G[bj + k]!.toUInt64
    let mk : UInt64 := 2097151
    let xm := x &&& 4611686018427387903
    let ym := y &&& 4611686018427387903
    let a0 := xm &&& mk
    let a1 := (xm >>> 21) &&& mk
    let a2 := xm >>> 42
    let b0 := ym &&& mk
    let b1 := (ym >>> 21) &&& mk
    let b2 := ym >>> 42
    let t : UInt64 := (0 : UInt64) - (((x ^^^ y) >>> 62) ^^^ (sg.get! k).toUInt64)
    let d0 := a0 * b0
    let d1 := a0 * b1 + a1 * b0
    let d2 := a0 * b2 + a1 * b1 + a2 * b0
    let d3 := a1 * b2 + a2 * b1
    let d4 := a2 * b2
    dot5 G sg bi bj (k + 1) m (c0 + ((d0 ^^^ t) - t)) (c1 + ((d1 ^^^ t) - t)) (c2 + ((d2 ^^^ t) - t))
      (c3 + ((d3 ^^^ t) - t)) (c4 + ((d4 ^^^ t) - t))
  else (c0, c1, c2, c3, c4)

/-! ### Signed numbers in eight 21-bit limbs

An `L8` holds `sum_d w_d 2^(21 d)` with each `w_d` a two's complement 64-bit word. It is *normal* when
`w_0, ..., w_6` lie in `[0, 2^21)`; then `w_7` carries the sign and the value is below `2^168` in
magnitude. All operations below are exact on values below `2^160` in magnitude. -/

structure L8 where
  w0 : UInt64
  w1 : UInt64
  w2 : UInt64
  w3 : UInt64
  w4 : UInt64
  w5 : UInt64
  w6 : UInt64
  w7 : UInt64

namespace L8

@[inline] def asr21 (x : UInt64) : UInt64 := (x.toInt64 >>> 21).toUInt64
@[inline] def lo21 (x : UInt64) : UInt64 := x &&& 2097151

/-- Carry propagation to the normal form. -/
@[inline] def norm (a : L8) : L8 :=
  let c0 := asr21 a.w0
  let w1 := a.w1 + c0
  let c1 := asr21 w1
  let w2 := a.w2 + c1
  let c2 := asr21 w2
  let w3 := a.w3 + c2
  let c3 := asr21 w3
  let w4 := a.w4 + c3
  let c4 := asr21 w4
  let w5 := a.w5 + c4
  let c5 := asr21 w5
  let w6 := a.w6 + c5
  let c6 := asr21 w6
  ⟨lo21 a.w0, lo21 w1, lo21 w2, lo21 w3, lo21 w4, lo21 w5, lo21 w6, a.w7 + c6⟩

@[inline] def sub (a b : L8) : L8 :=
  ⟨a.w0 - b.w0, a.w1 - b.w1, a.w2 - b.w2, a.w3 - b.w3, a.w4 - b.w4, a.w5 - b.w5, a.w6 - b.w6, a.w7 - b.w7⟩

@[inline] def neg (a : L8) : L8 :=
  ⟨0 - a.w0, 0 - a.w1, 0 - a.w2, 0 - a.w3, 0 - a.w4, 0 - a.w5, 0 - a.w6, 0 - a.w7⟩

@[inline] def isNeg (a : L8) : Bool := a.w7 >>> 63 == 1

@[inline] def isZero (a : L8) : Bool :=
  (a.w0 ||| a.w1 ||| a.w2 ||| a.w3 ||| a.w4 ||| a.w5 ||| a.w6 ||| a.w7) == 0

@[inline] def get (a : L8) (d : Nat) : UInt64 :=
  match d with
  | 0 => a.w0 | 1 => a.w1 | 2 => a.w2 | 3 => a.w3 | 4 => a.w4 | 5 => a.w5 | 6 => a.w6 | _ => a.w7

/-- Multiply a normal `L8` by `2^u` (the result below `2^160` in magnitude): limbwise by `2^rho`
(`u = 21 q + rho`, products below `2^42`), normalise, then move every limb up by `q` places, the limbs
that pass the top being folded into the signed top word. -/
def shl (a : L8) (u : Nat) : L8 :=
  if u == 0 then a else
  let q := u / 21
  let rho := (u % 21).toUInt64
  let m := fun (x : UInt64) => x <<< rho
  let b := norm ⟨m a.w0, m a.w1, m a.w2, m a.w3, m a.w4, m a.w5, m a.w6, m a.w7⟩
  if q == 0 then b else
  let s := fun (d : Nat) => if d < q then (0 : UInt64) else b.get (d - q)
  -- top word: b_(7-q) + b_(8-q) 2^21 + ... + b_7 2^(21 q), exact while it fits in 63 bits
  let top := Id.run do
    let mut t : UInt64 := 0
    let mut d := 7
    while d + q ≥ 7 + 1 && d > 7 - q do
      t := (t <<< 21) + b.get d
      d := d - 1
    -- now d = 7 - q
    t := (t <<< 21) + b.get (7 - q)
    return t
  ⟨s 0, s 1, s 2, s 3, s 4, s 5, s 6, top⟩

/-- Bit length of the magnitude of a normal non-negative `L8` and its top 24 bits `t`
(so that the value is below `(t + 1) 2^(L - 24)` when `L > 24`, and equals `t` otherwise). -/
def top24 (a : L8) : Nat × UInt64 := Id.run do
  let mut d := 7
  while d > 0 && a.get d == 0 do
    d := d - 1
  let wd := a.get d
  let bl := (Nat.log2 wd.toNat) + 1                 -- bit length of the top word
  let L := 21 * d + (if wd == 0 then 0 else bl)
  if L ≤ 24 then
    -- the whole value fits in 24 bits: a.w0 + a.w1 2^21
    return (L, a.w0 + (a.w1 <<< 21))
  -- extract bits [L-24, L): from words d and d-1 (and d-2)
  let pos := L - 24
  let mut t : UInt64 := 0
  let wordAt := fun (e : Nat) => if e ≤ 7 then a.get e else (0 : UInt64)
  -- assemble 63 bits starting at limb pos/21
  let base := pos / 21
  let off := (pos % 21).toUInt64
  let v : UInt64 := wordAt base ||| (wordAt (base + 1) <<< 21) ||| (wordAt (base + 2) <<< 42)
  t := (v >>> off) &&& 16777215
  return (L, t)

end L8

/-- `|v| 2^r` with the sign of `v` as a normal `L8`; `none` if it does not fit below `2^160`. -/
def L8.ofIntShift (v : Int) (r : Nat) : Option L8 :=
  let a := v.natAbs
  if a == 0 then some ⟨0, 0, 0, 0, 0, 0, 0, 0⟩ else
  if Nat.log2 a + 1 + r > 159 then none else
  -- bits of a 2^r at limb d are bits [21 d - r, 21 d - r + 21) of a
  let x0 : UInt64 := a.toUInt64
  let a1 := if a < 18446744073709551616 then 0 else a >>> 64
  let x1 : UInt64 := a1.toUInt64
  let a2 := if a1 == 0 then 0 else a1 >>> 64
  let x2 : UInt64 := a2.toUInt64
  let bitsAt := fun (pos : Int) =>
    -- 21 bits of a starting at bit `pos` (pos may be negative: low bits are zero)
    if pos + 21 ≤ 0 then (0 : UInt64)
    else if pos < 0 then (x0 <<< (-pos).toNat.toUInt64) &&& 2097151
    else
      let p := pos.toNat
      let w := p / 64
      let o := (p % 64).toUInt64
      let lo := if w == 0 then x0 else if w == 1 then x1 else if w == 2 then x2 else 0
      let hi := if w == 0 then x1 else if w == 1 then x2 else 0
      let v := if o == 0 then lo else (lo >>> o) ||| (hi <<< (64 - o))
      v &&& 2097151
  let l := fun (d : Nat) => bitsAt ((21 * d : Nat) - (r : Int))
  let pos : L8 := ⟨l 0, l 1, l 2, l 3, l 4, l 5, l 6, l 7⟩
  some (if v < 0 then (L8.neg pos).norm else pos)

/-- Exact `(Gt S Gt^T)_{ij}` for `j ≤ i` as a normal `L8`. -/
@[inline] def prodL8 (lb : Limbs) (i j : Nat) : L8 :=
  let (c0, c1, c2, c3, c4) := dot5 lb.g lb.sg (triIdx i) (triIdx j) 0 (j + 1) 0 0 0 0 0
  L8.norm ⟨c0, c1, c2, c3, c4, 0, 0, 0⟩

/-- Exact entry `(Gt S Gt^T)_{ij}` for `j ≤ i` as an integer (slow path and tests). -/
def prodEntry (lb : Limbs) (i j : Nat) : Int :=
  let (c0, c1, c2, c3, c4) := dot5 lb.g lb.sg (triIdx i) (triIdx j) 0 (j + 1) 0 0 0 0 0
  sval c0 + sval c1 * pow21 + sval c2 * pow42 + sval c3 * pow63 + sval c4 * pow84

/-- Value of a normal `L8` as an integer (tests). -/
def L8.toInt (a : L8) : Int :=
  sval a.w7 * (2 : Int) ^ 147 + (a.w6.toNat * 2 ^ 126 + a.w5.toNat * 2 ^ 105 + a.w4.toNat * 2 ^ 84 +
    a.w3.toNat * 2 ^ 63 + a.w2.toNat * 2 ^ 42 + a.w1.toNat * 2 ^ 21 + a.w0.toNat : Nat)

/-- Two-word accumulators per power of two: `hi 2^64 + lo` at index `k` stands for that times `2^k`. -/
@[inline] def bucketAdd (lo hi : FloatArray) (k : Nat) (v : UInt64) : FloatArray × FloatArray :=
  let x := rdW lo k
  let y := x + v
  let lo := wrW lo k y
  if y < x then (lo, wrW hi k (rdW hi k + 1)) else (lo, hi)

/-- Partial residual over the rows `[r0, r1)`, exact. With `e_ij = (Gt S Gt^T)_ij 2^u - M_ij 2^v`
(`u, v ≥ 0`, one of them zero), every `e_ij` is bounded above as `|e_ij| < (t + 1) 2^(L - 24)` from its top
24 bits; the values `(t + 1)^2 (W/w_i)(W/w_j)` are summed per exponent `2(L - 24)` (twice for `i ≠ j`).
The result is an upper bound of `sum_ij e_ij^2 (W/w_i)(W/w_j)`. Returns `none` if an entry is out of range. -/
def residualRows (M : IMat) (perm : Array Nat) (lb : Limbs) (u v : Nat) (wq : Array Nat)
    (r0 r1 : Nat) : Option (Nat × Nat) := Id.run do
  let nb := 400
  let mut blo := fzeros nb
  let mut bhi := fzeros nb
  let mut maxL : Nat := 0
  let zero8 : L8 := ⟨0, 0, 0, 0, 0, 0, 0, 0⟩
  for i in [r0:r1] do
    let mrow := M.row perm[i]!
    let wi := wq[perm[i]!]!
    for j in [0:i+1] do
      let p := L8.shl (prodL8 lb i j) u
      let mv := mrow[perm[j]!]!
      let mL := if mv == 0 then some zero8 else L8.ofIntShift mv v
      match mL with
      | none => return none
      | some ml =>
        let e := (L8.sub p ml).norm
        if !e.isZero then
          let a := if e.isNeg then e.neg.norm else e
          let (L, t) := L8.top24 a
          if L > maxL then maxL := L
          let (k, base) := if L ≤ 24 then (0, t) else (2 * (L - 24), t + 1)
          if k ≥ nb then return none
          let wj := wq[perm[j]!]!
          let val : UInt64 := base * base * (wi * wj).toUInt64
          let (l1, h1) := bucketAdd blo bhi k val
          blo := l1
          bhi := h1
          if i != j then
            let (l2, h2) := bucketAdd blo bhi k val
            blo := l2
            bhi := h2
  let mut tot : Nat := 0
  for k in [0:nb] do
    let s := (rdW bhi k).toNat * 18446744073709551616 + (rdW blo k).toNat
    if s != 0 then tot := tot + s * 2 ^ k
  return some (tot, maxL)

/-- Result of a certified count. All quantities exact. `beta` bounds the spectral norm of
`W^(-1/2) E W^(-1/2)` (with `W = diag(w)`, `W = I` when no weights are given) in the units of `M`. -/
structure CertCount where
  n : Nat
  neg : Nat
  pos : Nat
  ok : Bool
  msg : String
  betaInf : Rat
  betaFro : Rat
  beta : Rat
  secFactor : Float
  secResidual : Float
  deriving Inhabited

/-- Split `[0, n)` into `t` row ranges of roughly equal work (row `i` costs about `i^2`). -/
def splitRows (n t : Nat) : Array (Nat × Nat) := Id.run do
  if t ≤ 1 || n < 64 then return #[(0, n)]
  let total : Nat := n * n * n
  let mut out : Array (Nat × Nat) := #[]
  let mut start := 0
  for c in [1:t+1] do
    let target := c * total / t
    let mut r := start
    while r < n && r * r * r < target do
      r := r + 1
    if c == t then r := n
    if r > start then
      out := out.push (start, r)
      start := r
  return out

def lcmAll (w : Array Nat) : Nat := w.foldl (fun a b => Nat.lcm a b) 1

/-- Certified count of the negative and positive eigenvalues of `M` up to the residual bound `beta`.
`w` (optional, positive integers) are metric weights: the bound is on `W^(-1/2) E W^(-1/2)`, and the
floating-point factorization is of `W^(-1/2) M W^(-1/2)`. -/
def certCount (M : IMat) (w : Array Nat := #[]) (threads : Nat := 1) : IO CertCount := do
  let n := M.dim
  let fail := fun (m : String) => (⟨n, 0, 0, false, m, 0, 0, 0, 0, 0⟩ : CertCount)
  if n == 0 then return ⟨0, 0, 0, true, "", 0, 0, 0, 0, 0⟩
  if n ≥ 2 ^ 19 then return fail "dimension too large for the limb accumulators"
  let w := if w.size == n then w else Array.replicate n 1
  if w.any (· == 0) then return fail "zero weight"
  let W := lcmAll w
  let wq : Array Nat := w.map (fun x => W / x)
  if wq.any (fun x => x * x ≥ 2 ^ 15) then return fail "weights spread too wide for the bucket bound"
  let t0 ← IO.monoMsNow
  let dscale : FloatArray := Id.run do
    let mut d := fzeros n
    for i in [0:n] do
      d := d.set! i (1.0 / (Float.ofNat w[i]!).sqrt)
    return d
  let mx := M.maxAbs
  if mx == 0 then return fail "zero matrix"
  let sigma0 : Nat := Nat.log2 mx + 1
  let mut a := fzeros (n * n)
  for i in [0:n] do
    let r := M.row i
    for j in [0:i+1] do
      let v := r[j]!
      if v != 0 then
        a := a.set! (i * n + j) (intToFloatScaled v sigma0 * dscale.get! i * dscale.get! j)
  let ta ← IO.monoMsNow
  let f := ldlFactor n a
  if !f.ok then return fail f.msg
  let perm := f.perm
  let tb ← IO.monoMsNow
  let lb := makeLimbs f dscale
  if !lb.ok then return fail lb.msg
  let t1 ← IO.monoMsNow
  if (← IO.getEnv "CEAPPS_DEBUG").isSome then
    IO.eprintln s!"  [certCount n={n}] build {ta - t0} ms, ldl {tb - ta} ms, limbs {t1 - tb} ms"
  -- Gt S Gt^T ≈ 2^(2F) 2^(-sigma) P^T M P; e = (Gt S Gt^T) 2^u - M 2^v = 2^v E with E the residual
  let F := lb.F[0]!
  let r : Int := 2 * F - (sigma0 : Int)
  let (u, v) : Nat × Nat := if r ≥ 0 then (0, r.toNat) else ((-r).toNat, 0)
  let ranges := splitRows n threads
  let tasks := ranges.map (fun (r0, r1) =>
    Task.spawn (fun _ => residualRows M perm lb u v wq r0 r1))
  let mut fro : Nat := 0
  for tk in tasks do
    match tk.get with
    | some (fr, _) => fro := fro + fr
    | none => return fail "a residual entry is out of the range of the limb arithmetic"
  let t2 ← IO.monoMsNow
  let mut neg := 0
  for k in [0:n] do
    if lb.sg.get! k == 1 then neg := neg + 1
  let bFro : Rat := mkRat ((isqrt fro + 1 : Nat) : Int) (2 ^ v * W)
  return ⟨n, neg, n - neg, true, "", bFro, bFro, bFro,
    (t1 - t0).toFloat / 1000.0, (t2 - t1).toFloat / 1000.0⟩

/-- Exact residual with integers (slow; tests only): `sum_ij e_ij^2 (W/w_i)(W/w_j)` exactly. -/
def residualExactInt (M : IMat) (perm : Array Nat) (lb : Limbs) (u v : Nat) (wq : Array Nat) : Nat := Id.run do
  let mut fro : Nat := 0
  for i in [0:lb.n] do
    let mrow := M.row perm[i]!
    for j in [0:i+1] do
      let e := prodEntry lb i j * (2 : Int) ^ u - mrow[perm[j]!]! * (2 : Int) ^ v
      let sq := e.natAbs * e.natAbs * wq[perm[i]!]! * wq[perm[j]!]!
      fro := fro + (if i == j then sq else 2 * sq)
  return fro

end CertifyApps
