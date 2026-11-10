import CertifyApps.Eta
import CertifyApps.AkltHam
/-!
# Export of the rebuilt AKLT ingredients for the comparison with `apps/aklt/graph.py` and `kernel.py`

`ceapps export-aklt <lattice> <file>` writes, as JSON, the Gram matrices `G_L, G_C, G_R, G_i, G_j`, the
reduced down counts, the singlet counts and the rank modulo `2^31 - 1` of the union VBS states;
`ceapps export-aklt-ham <lattice> <file>` writes `den H` of the subgraph `L u C` in the scaled basis and
the result of the exact check `H U = 0` on its reduced VBS columns.
-/
namespace CertifyApps
open CertifyCore

def jmat (G : Array (Array Int)) : JV := jarr (G.map (fun r => jarr (r.map jint)))
def jnats (a : Array Nat) : JV := jarr (a.map jnat)

def exportAklt (name path : String) : IO UInt32 := do
  let some lat := latticeByName name | return 2
  let g := lat.g
  let rL := reducedRegion g lat.L
  let rC := reducedRegion g lat.C
  let rR := reducedRegion g lat.R
  let Ui := unionCoords g #[lat.L, lat.C] #[rL, rC]
  let Uj := unionCoords g #[lat.C, lat.R] #[rC, rR]
  let Gi := gramOfCoords Ui #[rL.G, rC.G]
  let Gj := gramOfCoords Uj #[rC.G, rR.G]
  let UU := unionCoords g #[lat.L, lat.C, lat.R] #[rL, rC, rR]
  let nUnit := rankModP (gramPlain UU.cols)
  writeJson path (jobj [("lattice", jstr name), ("L", jnats lat.L), ("C", jnats lat.C), ("R", jnats lat.R),
    ("edges", jarr (g.edges.map (fun (u, v) => jarr #[jnat u, jnat v]))), ("z", jnats g.z),
    ("GL", jmat rL.G), ("GC", jmat rC.G), ("GR", jmat rR.G), ("Gi", jmat Gi), ("Gj", jmat Gj),
    ("downsL", jnats rL.downs), ("downsR", jnats rR.downs), ("downs_i", jnats Ui.downs), ("downs_j", jnats Uj.downs),
    ("singlets_i", jnat Ui.singlets), ("singlets_j", jnat Uj.singlets), ("innerL", jnat rL.inner),
    ("innerR", jnat rR.inner), ("unit_cosines_at_least", jnat nUnit)])
  return 0

def exportAkltHam (name path : String) : IO UInt32 := do
  let some lat := latticeByName name | return 2
  let X := lat.L ++ lat.C
  let (rows, den) := hamiltonianRegion lat.g X
  let (ok, N, nv) := vbsInKernel lat.g X
  let h ← IO.FS.Handle.mk path .write
  h.putStrLn s!"{N} {den} {nv} {ok}"
  for tgt in [0:rows.size] do
    let ent := rows[tgt]!.qsort (fun a b => a.1 < b.1)
    let mut merged : Array (Nat × Int) := #[]
    for (c, v) in ent do
      if merged.size > 0 && merged[merged.size - 1]!.1 == c then
        merged := merged.modify (merged.size - 1) (fun (x, y) => (x, y + v))
      else merged := merged.push (c, v)
    for (c, v) in merged do
      if v != 0 then h.putStrLn s!"{tgt} {c} {v}"
  return 0

end CertifyApps
