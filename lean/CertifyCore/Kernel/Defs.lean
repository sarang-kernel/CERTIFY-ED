/-
Kernel-reducible versions of the S2 and S3 checks for a certificate of kind `lowest` (structural recursion over
lists only, rationals of core Lean), so that `decide +kernel` can evaluate them on the literal data of a small
certificate (`CertifyCore/Kernel/Data*.lean`, written by `cecheck kdata` from the bytes whose SHA-256 it checked).

What `checkLowest c = true` asserts, all in exact rational arithmetic on the stored numbers:
(1) every pair: residual_bound >= ||A x - mu x||, existence_radius >= ||A x - mu x|| / ||x||, existence radius
    <= radius, and the exported radius >= record radius + model shift (squares compared, no square root);
(2) every single group: radius >= ||A x - mu x|| / ||x|| (Theorem 1);
(3) every cluster: rationals rU >= ||R||_F, etaU >= ||X^T X - I||_F with etaU < 1, 0 < sD <= sqrt(1 - etaU),
    xrU >= ||X^T R||_F exist (they are computed and their defining inequalities checked) such that the
    recorded radius is >= min(rU + (||A||_inf + max|mu|)(1 - sD), rU/sD + xrU/(1 - etaU) + (1 - sD) spread/sD),
    the Theorem 2 / Theorem 2' radius with these upper bounds;
(4) the groups' intervals [mu - r, mu + r] are pairwise disjoint;
(5) S2: the largest exported radius is <= the accuracy target;
(6) S3: every exported enclosure lies below the recorded next-eigenvalue bound b, b + model shift <= the top
    shift sigma, the recorded gap is <= b minus the top of the enclosures, and the fraction-free (Bareiss)
    elimination of 2^s (A - sigma I), s chosen so that the matrix is integral, has nonzero leading principal
    minors with exactly k sign changes in the sequence 1, d_1, ..., d_n.
By Sylvester's law of inertia (with Jacobi's rule) the last item says that exactly k eigenvalues of A lie below
sigma; that step, and Theorems 1, 2 and 2' themselves, are not part of this core-Lean statement.
-/
namespace CertifyCore.K

/-- m 2^e as a rational. -/
def q (m : Int) (e : Int) : Rat :=
  match e with
  | Int.ofNat k => (m : Rat) * ((2 ^ k : Nat) : Rat)
  | Int.negSucc k => (m : Rat) / ((2 ^ (k + 1) : Nat) : Rat)

def dotL : List Rat → List Rat → Rat
  | a :: as, b :: bs => a * b + dotL as bs
  | _, _ => 0

def sumL : List Rat → Rat
  | [] => 0
  | a :: as => a + sumL as

def maxL : List Rat → Rat → Rat
  | [], m => m
  | a :: as, m => maxL as (if m < a then a else m)

def getR : List Rat → Nat → Rat
  | [], _ => 0
  | a :: _, 0 => a
  | _ :: as, i + 1 => getR as i

def rowDot : List (Nat × Rat) → List Rat → Rat
  | [], _ => 0
  | (j, v) :: t, x => v * getR x j + rowDot t x

def matVec (A : List (List (Nat × Rat))) (x : List Rat) : List Rat := A.map (fun row => rowDot row x)

def residual (A : List (List (Nat × Rat))) (x : List Rat) (mu : Rat) : List Rat :=
  List.zipWith (fun ax xi => ax - mu * xi) (matVec A x) x

def absR (a : Rat) : Rat := if a < 0 then -a else a

def infNorm (A : List (List (Nat × Rat))) : Rat := maxL (A.map (fun row => sumL (row.map (fun e => absR e.2)))) 0

/-- Newton's iteration for the integer square root, with fuel. -/
def isqrtAux : Nat → Nat → Nat → Nat
  | 0, _, x => x
  | f + 1, n, x => if (x + n / x) / 2 < x then isqrtAux f n ((x + n / x) / 2) else x

def isqrt (n : Nat) : Nat := if n < 2 then n else isqrtAux 4000 n n

def P : Nat := 160

def sqrtUp (a : Rat) : Rat := mkRat ((isqrt (a * ((4 : Rat) ^ P)).floor.toNat + 1 : Nat) : Int) (2 ^ P)
def sqrtDown (a : Rat) : Rat := if a ≤ 0 then 0 else mkRat ((isqrt (a * ((4 : Rat) ^ P)).floor.toNat : Nat) : Int) (2 ^ P)

structure Pair where
  mu : Rat
  x : List Rat
  rb : Rat
  ex : Rat
  rad : Rat
  radArr : Rat
  exArr : Rat
  group : Nat

structure Cert where
  A : List (List (Nat × Rat))
  pairs : List Pair
  groups : List (List Nat)
  shift : Rat
  target : Rat
  sigma : Rat
  bound : Rat
  gap : Rat
  scale : Nat

def Pair.r (c : Cert) (p : Pair) : List Rat := residual c.A p.x p.mu

def pairOK (c : Cert) (p : Pair) : Bool :=
  let r2 := dotL (p.r c) (p.r c)
  let x2 := dotL p.x p.x
  decide (0 ≤ p.rb) && decide (r2 ≤ p.rb * p.rb) && decide (0 ≤ p.ex) && decide (r2 ≤ p.ex * p.ex * x2) &&
  decide (p.ex ≤ p.rad) && decide (p.rad + c.shift ≤ p.radArr) && decide (p.ex + c.shift ≤ p.exArr)

def getP : List Pair → Nat → Option Pair
  | [], _ => none
  | a :: _, 0 => some a
  | _ :: as, i + 1 => getP as i

def gramDefect (xs : List (List Rat)) : Rat :=
  sumL ((List.range xs.length).flatMap (fun a => (List.range xs.length).map (fun b =>
    let v := dotL ((xs.getD a []) ) (xs.getD b []) - (if a = b then 1 else 0)
    v * v)))

def crossSq (xs rs : List (List Rat)) : Rat :=
  sumL (xs.flatMap (fun x => rs.map (fun r => let v := dotL x r; v * v)))

def groupOK (c : Cert) (g : List Nat) : Bool :=
  match g.filterMap (getP c.pairs) with
  | [] => false
  | [p] => let r2 := dotL (p.r c) (p.r c)
           decide (0 ≤ p.rad) && decide (r2 ≤ p.rad * p.rad * dotL p.x p.x)
  | ps@(p0 :: _) =>
    let xs := ps.map (·.x)
    let rs := ps.map (fun p => p.r c)
    let etaU := sqrtUp (gramDefect xs)
    let sD := sqrtDown (1 - etaU)
    let rU := sqrtUp (sumL (rs.map (fun r => dotL r r)))
    let xrU := sqrtUp (crossSq xs rs)
    let mus := ps.map (·.mu)
    let lam := maxL (mus.map absR) 0
    let spread := maxL mus p0.mu - (- maxL (mus.map (fun m => -m)) (-p0.mu))
    let rho := rU + (infNorm c.A + lam) * (1 - sD)
    let eps := rU / sD + xrU / (1 - etaU) + (1 - sD) * spread / sD
    decide (etaU < 1) && decide (0 < sD) && decide (sD * sD ≤ 1 - etaU) &&
    ps.all (fun p => decide (p.rad = p0.rad)) && decide ((if rho < eps then rho else eps) ≤ p0.rad)

/-- The union interval [min (mu - r), max (mu + r)] of a group (a group is never empty here). -/
def interval (c : Cert) (g : List Nat) : Rat × Rat :=
  match g.filterMap (getP c.pairs) with
  | [] => (0, 0)
  | p0 :: ps => (- maxL (ps.map (fun p => -(p.mu - p.rad))) (-(p0.mu - p0.rad)), maxL (ps.map (fun p => p.mu + p.rad)) (p0.mu + p0.rad))

def disjoint (c : Cert) : Bool :=
  c.groups.all (fun g => c.groups.all (fun h => g == h ||
    let (l1, u1) := interval c g
    let (l2, u2) := interval c h
    decide (u1 < l2) || decide (u2 < l1)))

/-- Fraction-free elimination without pivoting; returns (negative, positive) sign changes, or none when a
leading principal minor vanishes. -/
def bareiss : Nat → Int → List (List Int) → Option (Nat × Nat)
  | 0, _, _ => some (0, 0)
  | _, _, [] => some (0, 0)
  | f + 1, prev, row0 :: rest =>
    match row0 with
    | [] => none
    | p :: tl0 =>
      if p = 0 then none else
      let rest' := rest.map (fun r => match r with
        | [] => []
        | a0 :: tl => List.zipWith (fun a b => (p * a - a0 * b) / prev) tl tl0)
      match bareiss f p rest' with
      | none => none
      | some (ng, ps) => if p * prev < 0 then some (ng + 1, ps) else some (ng, ps + 1)

def denseRow (n : Nat) (row : List (Nat × Rat)) (i : Nat) (sigma : Rat) : List Rat :=
  (List.range n).map (fun j => (row.filter (fun e => e.1 = j)).foldr (fun e s => e.2 + s) 0 - (if i = j then sigma else 0))

def shiftedInt (c : Cert) : Option (List (List Int)) :=
  let n := c.A.length
  let D : Rat := ((2 ^ c.scale : Nat) : Rat)
  let rows := (List.range n).map (fun i => (denseRow n (c.A.getD i []) i c.sigma).map (fun v => v * D))
  if rows.all (fun r => r.all (fun v => v.den = 1)) then some (rows.map (fun r => r.map (·.num))) else none

def negAtSigma (c : Cert) : Option Nat :=
  match shiftedInt c with
  | none => none
  | some M => (bareiss (c.A.length + 1) 1 M).map (·.1)

def checkLowest (c : Cert) : Bool :=
  let k := c.pairs.length
  let ups := c.pairs.map (fun p => p.mu + p.radArr)
  c.pairs.all (pairOK c) && c.groups.all (groupOK c) && disjoint c &&
  decide (maxL (c.pairs.map (·.radArr)) 0 ≤ c.target) &&
  ups.all (fun u => decide (u < c.bound)) && decide (c.bound + c.shift ≤ c.sigma) &&
  ups.all (fun u => decide (c.gap ≤ c.bound - u)) &&
  (negAtSigma c == some k)

end CertifyCore.K
