# reverification_lean

Lean 4 re-verification of CERTIFY-ED. The toolchain is Lean 4.35.0 (file `lean-toolchain`), the same build that was
used for the re-verification of the mm3x3 project, and Mathlib is the release `v4.35.0` (commit `56a005855ea1`), pinned
in `lake-manifest.json`.

## Libraries and programs

| Target | What it is | Imports Mathlib |
|---|---|---|
| `CertifyCore` | core-Lean library: exact rational arithmetic, IEEE-754 decoding, SHA-256, exact residual and inertia checks, kernel theorems about the finite arithmetic of the bounds | no |
| `cecheck` | compiled program (root `CertifyCore/Main.lean`) that re-checks certificate data exported from certify-ed | no |
| `CertifyMathlib` | formal proofs of the mathematical statements on Mathlib (enclosures, counting, sector assembly, eigenspaces) | yes |
| `CertifyApps` and `ceapps` | compiled recomputation of the certified application numbers (crown, AKLT, exact benchmarks) | no |

## Building on Windows

The Lean binaries are in `<lean-4.35.0-windows>\bin`
(no elan). Put that directory and `C:\Program Files\Git\cmd` on `PATH`, set `MATHLIB_CACHE_DIR` to
`reverification_lean\.mathlib_cache`, and run from this folder:

```
lake update                    # only once: fetches Mathlib v4.35.0 and its dependencies, then the prebuilt cache
lake exe cache get             # prebuilt Mathlib files from cache.mathlib.org (9031 files)
lake build                     # CertifyCore and CertifyMathlib
lake build cecheck ceapps      # the two compiled programs
```

On this machine `curl` needed the option `ssl-no-revoke` (file `.curl/.curlrc`, selected with `CURL_HOME`), because the
certificate revocation server is not reachable from the build environment, and git needed `core.autocrlf=false` so that
the Mathlib sources keep the line endings that the cache hashes were computed from. A first elaboration of a file that
imports `Mathlib.Analysis.Matrix.Spectrum` took 249 s; later ones took 45 s with a peak of 1.55 GiB.
