import CertifyMathlib.Apps.Bordered
import CertifyMathlib.Apps.Kernel
import CertifyMathlib.Apps.Criterion

/-!
# The lemmas behind the certified application numbers (E10)

* `CertifyMathlib.Apps.Bordered`: the bordered-pencil lemma (Haynsworth inertia additivity for the
  bordered matrix of `eta.py`), the Loewner bound `G_B⁻¹ ≤ (1 + β/μ)(G_B + β)⁻¹` and `eta_count`, the
  certified count of pencil eigenvalues above `t' = (1 + β/μ)(t + β/ν)`;
* `CertifyMathlib.Apps.Kernel`: rank modulo a prime against rank over `ℚ`, and `aklt_kernel_certificate`,
  which turns the exact computations of the kernel check into `ker H_X = range U_X`;
* `CertifyMathlib.Apps.Criterion`: the pencil count as the count of squared cosines, the composed
  bounds `e10_eta_bound` and `e10_criterion`, and the rational tests of the Pomata-Wei criterion.

See `APPS.md`, section 9.
-/
