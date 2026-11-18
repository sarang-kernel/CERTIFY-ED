/-!
# CertifyCore

Core-Lean (no Mathlib) part of the re-verification of CERTIFY-ED.

* Trust level N: the compiled program `cecheck` (`CertifyCore/Main.lean`) re-checks exported certificates
  in exact arithmetic: `Sha256`, `Json` (canonical serialisation), `Dyadic` (exact binary64 values),
  `Inertia` (exact Bareiss inertia), `Load`, `Check`.
* Trust level K: kernel-checked theorems about the finite arithmetic behind the bounds, in
  `CertifyCore/Theorems/`.

See `reverification_lean/CORE.md`.
-/
namespace CertifyCore
def version : String := "1.0"
end CertifyCore
