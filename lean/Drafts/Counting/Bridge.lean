import CertifyMathlib.Counting.QuadForm

open Matrix CertifyMathlib.Counting

#check @EuclideanSpace.inner_eq_star_dotProduct
#check @EuclideanSpace.norm_sq_eq
#check @Matrix.toEuclideanLin_apply
#check @Matrix.toLpLin_apply
#check @WithLp.ofLp_toLp

variable {𝕜 : Type*} [RCLike 𝕜] {n : Type*} [Fintype n] [DecidableEq n]

example (M : Matrix n n 𝕜) (x : n → 𝕜) :
    qf M x = RCLike.re (inner 𝕜 (WithLp.toLp 2 x : EuclideanSpace 𝕜 n)
      (Matrix.toEuclideanLin M (WithLp.toLp 2 x))) := by
  rw [EuclideanSpace.inner_eq_star_dotProduct]
  simp [qf, dotProduct_comm]

example (x : n → 𝕜) : nsq x = ‖(WithLp.toLp 2 x : EuclideanSpace 𝕜 n)‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq]
  simp [nsq]
