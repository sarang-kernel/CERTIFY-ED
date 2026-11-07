import Mathlib.Analysis.Matrix.Spectrum
/-! Smoke test: Mathlib's spectral theorem for Hermitian matrices is available. -/
open Matrix
example {n : Type*} [Fintype n] [DecidableEq n] (A : Matrix n n ℂ) (hA : A.IsHermitian) :
    A = (hA.eigenvectorUnitary : Matrix n n ℂ) * diagonal (RCLike.ofReal ∘ hA.eigenvalues) *
      star (hA.eigenvectorUnitary : Matrix n n ℂ) := hA.spectral_theorem
