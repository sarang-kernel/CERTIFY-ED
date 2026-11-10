import CertifyMathlib.Enclosure.Spectral
import CertifyMathlib.Enclosure.Residual
import CertifyMathlib.Enclosure.KatoTemple
import CertifyMathlib.Enclosure.Weyl
import CertifyMathlib.Enclosure.Kahan
import CertifyMathlib.Enclosure.Cluster
import CertifyMathlib.Enclosure.ClusterPrime
import CertifyMathlib.Enclosure.Variational
import CertifyMathlib.Enclosure.Generalized
import CertifyMathlib.Enclosure.FloatModel
import CertifyMathlib.Enclosure.FloatAccounting
import CertifyMathlib.Enclosure.MatrixFree
import CertifyMathlib.Enclosure.ExpLn

/-!
# CertifyMathlib.Enclosure

Root module of the enclosure track: Theorem 1 (residual enclosure, with the multi-pair form and the
S1 model shift), Theorem 3 (Kato–Temple), Gershgorin for Hermitian matrices, Weyl's inequality and
the Appendix A model shift, Kahan's inclusion theorem, Theorem 2 (polar factor) and Theorem 2′ (ε*),
E-05 (Rayleigh–Ritz, Temple, the perturbed Rayleigh quotient) and E-03 (Cholesky reduction of a
pencil, with the eigenvalues counted with multiplicity).

The floating-point part: a model of binary64 rounding with Higham's lemmas (`FloatModel`), the
Theorem 1 accounting of `residuals` for real and complex data and Lemmas A1 to A3 with Theorem A4
(`FloatAccounting`), the matrix-free count of D-02 (`MatrixFree`) and the dyadic exp and ln of
Lemma 17 (`ExpLn`). See `ENCLOSURE.md`.
-/
