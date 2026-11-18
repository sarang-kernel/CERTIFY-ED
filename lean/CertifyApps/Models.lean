import CertifyApps.SpinHalf
/-!
# The models of experiments E7 and E9, rebuilt from their specifications

Every model is written down here from the parameters in the experiment scripts and the specification
builders of certify-ed (`certify_ed/spec.py`: `heisenberg_chain`, `heisenberg_square`,
`spin1_heisenberg_chain`; `certify_ed/apps/pyrochlore/model.py`; `experiments/e7_exact_benchmarks.py`;
`experiments/e9_pyrochlore.py`). Couplings are exact rationals `c / D` with integer `c`.
-/
namespace CertifyApps

/-- Heisenberg chain of `N` sites, periodic, `J = 1` and next-nearest `J2 = j2num / j2den`
(`heisenberg_chain`: bonds `(i, i + r)` for `i < N - r` and `(i, (i + r) mod N)` for the last `r` sites). -/
def chainModel (N : Nat) (j2num j2den : Nat) : Heis := Id.run do
  let D := if j2num == 0 then 1 else j2den
  let mut b : Array (Nat × Nat × Int) := #[]
  for i in [0:N] do
    b := b.push (Nat.min i ((i + 1) % N), Nat.max i ((i + 1) % N), (D : Int))
  if j2num != 0 then
    for i in [0:N] do
      b := b.push (Nat.min i ((i + 2) % N), Nat.max i ((i + 2) % N), (j2num : Int))
  return ⟨N, b, D⟩

/-- Heisenberg model on the `Lx × Ly` periodic square lattice, sites `y Lx + x`, nearest-neighbour bonds
with `J = 1` and diagonal next-nearest bonds with `J2 = j2num / j2den`; bonds as a set of sorted pairs, as
in `heisenberg_square`. -/
def squareModel (Lx Ly : Nat) (j2num j2den : Nat) : Heis := Id.run do
  let D := if j2num == 0 then 1 else j2den
  let idx := fun (x y : Int) => (((x % Lx) + Lx) % Lx).toNat + Lx * (((y % Ly) + Ly) % Ly).toNat
  let mut nn : Array (Nat × Nat) := #[]
  let mut nnn : Array (Nat × Nat) := #[]
  for x in [0:Lx] do
    for y in [0:Ly] do
      for (dx, dy, kind) in [((1 : Int), (0 : Int), 0), (0, 1, 0), (1, 1, 1), (1, -1, 1)] do
        let a := idx x y
        let c := idx (x + dx) (y + dy)
        if a != c then
          let pr := (Nat.min a c, Nat.max a c)
          if kind == 0 then
            if !nn.contains pr then nn := nn.push pr
          else
            if !nnn.contains pr then nnn := nnn.push pr
  let mut b : Array (Nat × Nat × Int) := #[]
  for (i, j) in nn do b := b.push (i, j, (D : Int))
  if j2num != 0 then
    for (i, j) in nnn do b := b.push (i, j, (j2num : Int))
  return ⟨Lx * Ly, b, D⟩

def translationChain (N : Nat) : Sym := ⟨"translation", (Array.range N).map (fun i => (i + 1) % N), false, N⟩
def reflectionChain (N : Nat) : Sym := ⟨"reflection", (Array.range N).map (fun i => N - 1 - i), false, 2⟩
def spinFlip (N : Nat) : Sym := ⟨"spin_flip", Array.range N, true, 2⟩

def squareTx (Lx Ly : Nat) : Sym :=
  ⟨"Tx", (Array.range (Lx * Ly)).map (fun s => (s % Lx + 1) % Lx + Lx * (s / Lx)), false, Lx⟩
def squareTy (Lx Ly : Nat) : Sym :=
  ⟨"Ty", (Array.range (Lx * Ly)).map (fun s => s % Lx + Lx * ((s / Lx + 1) % Ly)), false, Ly⟩

/-! ## The pyrochlore crown (`apps/pyrochlore/model.py`), sites 1-based in the paper, 0-based here -/

def crownPairs : Array (Nat × Nat) := #[(1, 2), (6, 12), (9, 16), (17, 18), (7, 13), (3, 10)]
def crownHex : Array Nat := #[4, 5, 11, 15, 14, 8]
def crownTetHex : Array ((Nat × Nat) × (Nat × Nat)) :=
  #[((1, 2), (4, 5)), ((6, 12), (5, 11)), ((9, 16), (11, 15)), ((17, 18), (15, 14)), ((7, 13), (14, 8)),
    ((3, 10), (8, 4))]
def crownC3Sites : Array Nat := #[1, 12, 9, 18, 7, 10]

/-- The 36 bonds `(coupling, i, j)` of `model.bonds`, with symbolic couplings 1 = c1, 3 = c3, 12 = c12,
17 = c17, sites 1-based. -/
def crownBondList : Array (Nat × Nat × Nat) := Id.run do
  let mut b : Array (Nat × Nat × Nat) := #[]
  for (p, q) in crownPairs do b := b.push (1, p, q)
  for ((p, q), (h1, h2)) in crownTetHex do
    for o in [p, q] do
      for h in [h1, h2] do
        b := b.push (if crownC3Sites.contains o then 3 else 12, o, h)
  for i in [0:6] do
    b := b.push (17, crownHex[i]!, crownHex[(i + 1) % 6]!)
  return b

/-- The spin-1/2 crown with `c1 = n1 / 10^7`, `c3 = c12 = n3 / 10^7` and `c17 = (1 - 2 c1 - 8 c3) / 2`
(`model.couplings`), as integer couplings over `D = 2 10^7`. -/
def crownModel (n1 n3 : Nat) : Heis := Id.run do
  let D : Nat := 2 * 10 ^ 7
  let c1 : Int := 2 * n1
  let c3 : Int := 2 * n3
  let c17 : Int := (10 ^ 7 : Int) - 2 * n1 - 8 * n3    -- 2 c17 10^7
  let mut b : Array (Nat × Nat × Int) := #[]
  for (k, i, j) in crownBondList do
    let c := if k == 1 then c1 else if k == 17 then c17 else c3
    b := b.push (i - 1, j - 1, c)
  return ⟨18, b, D⟩

/-- `Z_2^7` generators of `model.symmetry_generators`: the six pair swaps and the spin flip. -/
def crownGenerators : Array Sym := Id.run do
  let mut g : Array Sym := #[]
  for (a, b) in crownPairs do
    let p := (Array.range 18).map (fun i => if i == a - 1 then b - 1 else if i == b - 1 then a - 1 else i)
    g := g.push ⟨s!"swap{a}-{b}", p, false, 2⟩
  g := g.push (spinFlip 18)
  return g

/-! ## Spin 1 -/

/-- A spin-1 Heisenberg model with integer couplings: `H = sum_b c_b S_i . S_j` in the basis of `S^z`
states; digit `k` of site `i` (base 3, site 0 most significant) is `m = 1 - k`. In this basis every entry is
an integer: `S^z_i S^z_j` is `m_i m_j` and each allowed exchange `S^+_i S^-_j / 2 + h.c.` has the entry
`sqrt(2) sqrt(2) / 2 = 1`. -/
structure Spin1 where
  N : Nat
  bonds : Array (Nat × Nat × Int)

def pow3 (k : Nat) : Nat := 3 ^ k

def Spin1.digit (m : Spin1) (s i : Nat) : Nat := (s / pow3 (m.N - 1 - i)) % 3

def Spin1.apply (m : Spin1) (s : Nat) : Array (Nat × Int) := Id.run do
  let mut diag : Int := 0
  let mut out : Array (Nat × Int) := #[]
  for (i, j, c) in m.bonds do
    let ki := m.digit s i
    let kj := m.digit s j
    let mi : Int := 1 - (ki : Int)
    let mj : Int := 1 - (kj : Int)
    diag := diag + c * mi * mj
    -- S^+_i S^-_j: m_i + 1, m_j - 1 (digit k_i - 1, k_j + 1)
    if ki ≥ 1 && kj ≤ 1 then
      let t := s - pow3 (m.N - 1 - i) + pow3 (m.N - 1 - j)
      out := out.push (t, c)
    if kj ≥ 1 && ki ≤ 1 then
      let t := s + pow3 (m.N - 1 - i) - pow3 (m.N - 1 - j)
      out := out.push (t, c)
  if diag != 0 then out := out.push (s, diag)
  return out

/-- Full sparse matrix of a spin-1 model (all `3^N` states). -/
def Spin1.matrix (m : Spin1) : Array (Array (Nat × Int)) := Id.run do
  let n := pow3 m.N
  let mut rows : Array (Array (Nat × Int)) := #[]
  for s in [0:n] do
    let ent := m.apply s
    -- merge duplicates
    let ent := ent.qsort (fun a b => a.1 < b.1)
    let mut r : Array (Nat × Int) := #[]
    for (t, v) in ent do
      if r.size > 0 && r[r.size - 1]!.1 == t then
        r := r.modify (r.size - 1) (fun (a, b) => (a, b + v))
      else r := r.push (t, v)
    rows := rows.push (r.filter (fun (_, v) => v != 0))
  return rows

/-- The spin-1 Heisenberg ring of `spin1_heisenberg_chain(N)` (periodic, `J = 1`). -/
def spin1Ring (N : Nat) : Spin1 :=
  ⟨N, (Array.range N).map (fun i => (Nat.min i ((i + 1) % N), Nat.max i ((i + 1) % N), (1 : Int)))⟩

end CertifyApps
