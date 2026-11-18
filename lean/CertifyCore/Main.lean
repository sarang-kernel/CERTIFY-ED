import CertifyCore.Check
import CertifyCore.KData
/-
`cecheck`: exact re-check of one exported certify-ed certificate (trust level N).

    cecheck DIR [--json OUT] [--quiet]

DIR is a directory written by `export/export_certs.py`. Exit code 0: no check failed; 1: at least one check
failed; 2: the inputs could not be read. The report lists every check with PASS, FAIL, SKIP, INFO, and for a
statement certify-ed set to FAILED whether the exact recomputation REPRODUCED or REFUTED it.
-/
open CertifyCore

def jstr (s : String) : JV := JV.str s

def report (case : String) (checks : Array Check) (log : Array String) (ms : Nat) : JV :=
  let cnt := fun (st : St) => (checks.filter (·.st == st)).size
  let verdict := if cnt .fail > 0 then "FAIL" else "PASS"
  JV.obj #[("case", jstr case), ("verdict", jstr verdict), ("exit_code", JV.num (if cnt .fail > 0 then "1" else "0")),
    ("counts", JV.obj #[("pass", JV.num (toString (cnt .pass))), ("fail", JV.num (toString (cnt .fail))),
       ("skip", JV.num (toString (cnt .skip))), ("info", JV.num (toString (cnt .info))),
       ("reproduced", JV.num (toString (cnt .reproduced))), ("refuted", JV.num (toString (cnt .refuted))),
       ("not_reproduced", JV.num (toString (cnt .notReproduced)))]),
    ("checks", JV.arr (checks.map (fun c => JV.obj #[("id", jstr c.id), ("status", jstr c.st.str), ("detail", jstr c.detail)]))),
    ("exact_counts", JV.arr (log.map jstr)),
    ("milliseconds", JV.num (toString ms))]

def main (args : List String) : IO UInt32 := do
  match args with
  | "bench" :: dir :: _ =>
    let b ← loadBundle dir
    let some op := b.op? | throw (IO.userError "no operator")
    let A ← IO.ofExcept op.toCsr
    let gens := Csr.symmetryGenerators (A.shiftedInt Dy.zero)
    IO.println s!"n={A.n} components={A.components.size} generators={gens.size}"
    return 0
  | ["kdata", dir, name, out] =>
    let b ← loadBundle dir
    let (checks, _) ← checkBundle b
    let fails := checks.filter (·.st == .fail)
    if fails.size > 0 then
      IO.eprintln s!"exact check failed ({fails.size} checks): no kernel data written"
      return 1
    match kernelData b name with
    | .ok s => IO.FS.writeFile out s; IO.println s!"wrote {out}"; return 0
    | .error e => IO.eprintln e; return 1
  | dir :: rest =>
    let mut jsonOut : Option String := none
    let mut quiet := false
    let mut rs := rest
    while rs.length > 0 do
      match rs with
      | "--json" :: p :: t => jsonOut := some p; rs := t
      | "--quiet" :: t => quiet := true; rs := t
      | _ :: t => rs := t
      | [] => rs := []
    let t0 ← IO.monoMsNow
    let b ← try loadBundle dir catch e => do
      IO.eprintln s!"cannot read {dir}: {e}"
      return 2
    let (checks, log) ← try checkBundle b catch e => do
      IO.eprintln s!"error while checking {dir}: {e}"
      return 2
    let t1 ← IO.monoMsNow
    let case := ((b.cert.path? ["claim", "description"]).bind JV.str?).getD dir
    let rep := report case checks log (t1 - t0)
    if let some p := jsonOut then
      IO.FS.writeBinFile p (Json.canonical rep)
    if !quiet then
      for c in checks do
        IO.println s!"{c.st.str}  {c.id}: {c.detail}"
      for l in log do
        IO.println s!"COUNT  {l}"
    let fails := (checks.filter (·.st == .fail)).size
    IO.println s!"{if fails > 0 then "FAIL" else "PASS"} ({checks.size} checks, {fails} failed, {t1 - t0} ms)"
    return (if fails > 0 then 1 else 0)
  | [] =>
    IO.println "usage: cecheck DIR [--json OUT] [--quiet]"
    return 2
