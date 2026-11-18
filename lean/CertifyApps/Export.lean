import CertifyApps.Apps
/-!
# Export of the rebuilt models for the entrywise comparison with the Python builder

`ceapps export <model> <file>` writes the integer matrix `Hint` of a rebuilt model on its state set, one
nonzero entry per line, and `ceapps export-sector <model> <exponents> <file>` the Gaussian-integer sector
matrix of one character together with its representatives and orbit sizes. The comparison itself is made by
`apps_export/compare_models.py` against `certify_ed.checker.exact_matrix` and
`certify_ed.exact_sectors.sector_integer_matrix`; the agreement must be exact.
-/
namespace CertifyApps

/-- Models available for export: (model, number of up spins, generators, fixed quantum numbers). -/
def exportModel (key : String) : Option (Heis × Nat × Array Sym × Array (Option Nat)) :=
  match key with
  | "chain8" => some (chainModel 8 0 1, 4, #[translationChain 8, reflectionChain 8, spinFlip 8], #[some 0, none, none])
  | "chain8_J2" => some (chainModel 8 1 2, 4, #[translationChain 8, reflectionChain 8, spinFlip 8], #[some 0, none, none])
  | "chain12_J2" => some (chainModel 12 1 2, 6, #[translationChain 12, reflectionChain 12, spinFlip 12], #[some 0, none, none])
  | "chain20" => some (chainModel 20 0 1, 10, #[translationChain 20, reflectionChain 20, spinFlip 20], #[some 0, none, none])
  | "chain20_J2" => some (chainModel 20 1 2, 10, #[translationChain 20, reflectionChain 20, spinFlip 20], #[some 0, none, none])
  | "square4" => some (squareModel 4 4 0 1, 8, #[squareTx 4 4, squareTy 4 4, spinFlip 16], #[none, none, none])
  | "square4_J2" => some (squareModel 4 4 1 2, 8, #[squareTx 4 4, squareTy 4 4, spinFlip 16], #[none, none, none])
  | "square42_J2" => some (squareModel 4 2 1 2, 4, #[squareTx 4 2, squareTy 4 2, spinFlip 8], #[none, none, none])
  | "crown" => some (crownModel 381414 817957, 9, crownGenerators, Array.replicate 7 none)
  | _ => none

def exportHeis (key path : String) : IO UInt32 := do
  let some (m, up, _, _) := exportModel key | IO.eprintln s!"unknown model {key}"; return 2
  let states := statesFixedWeight m.N up
  let h ← IO.FS.Handle.mk path .write
  h.putStrLn s!"{m.N} {states.size} {4 * m.D}"
  for s in states do
    h.putStrLn (toString s)
  for si in [0:states.size] do
    let ent := (m.apply states[si]!).filterMap (fun (t, v) => (findIdx states t).map (fun ti => (ti, v)))
    let ent := ent.qsort (fun a b => a.1 < b.1)
    -- merge duplicates
    let mut r : Array (Nat × Int) := #[]
    for (t, v) in ent do
      if r.size > 0 && r[r.size - 1]!.1 == t then r := r.modify (r.size - 1) (fun (a, b) => (a, b + v))
      else r := r.push (t, v)
    for (t, v) in r do
      if v != 0 then h.putStrLn s!"{t} {si} {v}"
  return 0

def exportSector (key expStr path : String) : IO UInt32 := do
  let some (m, up, gens, _) := exportModel key | IO.eprintln s!"unknown model {key}"; return 2
  let ex := (expStr.splitOn ",").map (fun x => x.toNat!) |>.toArray
  let states := statesFixedWeight m.N up
  let G := Group.generate m.N gens
  let ob := orbits m.N states G
  let some chi := G.character ex | IO.eprintln "not a character"; return 2
  let s := buildSector m states ob chi ex
  let h ← IO.FS.Handle.mk path .write
  h.putStrLn s!"{s.dim} {4 * m.D} {G.elems.size} {(gens.all (fun g => commutes m states g))} {s.hermitian}"
  h.putStrLn (String.intercalate " " (s.reps.toList.map toString))
  h.putStrLn (String.intercalate " " (s.nrm.toList.map toString))
  for a in [0:s.dim] do
    for (b, v) in s.re[a]! do h.putStrLn s!"R {a} {b} {v}"
    for (b, v) in s.im[a]! do h.putStrLn s!"I {a} {b} {v}"
  return 0

def exportSpin1Ring (path : String) : IO UInt32 := do
  let A := (spin1Ring 6).matrix
  let h ← IO.FS.Handle.mk path .write
  h.putStrLn s!"6 {A.size} 1"
  for i in [0:A.size] do
    for (j, v) in A[i]! do h.putStrLn s!"{i} {j} {v}"
  return 0

end CertifyApps
