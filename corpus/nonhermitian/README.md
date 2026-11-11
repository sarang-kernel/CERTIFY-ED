# Non-Hermitian cases (E-02)

Eleven small cases for the non-Hermitian claim kind (`certify_ed/nonhermitian.py`, derivation
`docs/proofs/E-02_non_hermitian.md`). They are kept apart from the Hermitian planted-error corpus in `corpus/cases/`,
whose runner and score are unchanged.

`make_cases.py` writes `cases/<name>.npz` (operator, claimed eigenvalues, claimed right eigenvectors, and the exact
eigenvalues as integer numerator and denominator arrays) and `cases/<name>.json` (what the case plants and which
statements must or must not certify). Every operator is built as `A = S J S^-1` with a unimodular integer `S` and a
dyadic `J`, so its eigenvalues are known exactly. `run_cases.py` runs the certifier on each case, checks the
expectations, and checks every CERTIFIED N3 and N4 conclusion against the exact eigenvalues in rational arithmetic.

| case | plants | expected |
|---|---|---|
| nh_correct_real | nothing (real, diagonalizable) | N1 to N4 CERTIFIED |
| nh_correct_complex | nothing (complex) | N1 to N4 CERTIFIED |
| nh_correct_conjugate_pairs | nothing (real with conjugate pairs) | N1 to N4 CERTIFIED |
| nh_correct_nonnormal | nothing (strongly non-normal, kappa(V) large) | N1 to N4 CERTIFIED, wide discs |
| nh_bauer_fike_tight | nothing (radius close to attained) | N1 to N4 CERTIFIED |
| nh_duplicated_vector | two equal claimed eigenvectors | N2 INCONCLUSIVE, N3 and N4 NOT_CHECKED |
| nh_defective | a Jordan block claimed as diagonalizable | nothing false certified |
| nh_wrong_eigenvalue | one eigenvalue moved | nothing false certified |
| nh_swapped_vectors | two eigenvectors exchanged | nothing false certified |
| nh_subnormal_entry | an operator entry below 2^-1022 | N1 to N4 INCONCLUSIVE (UNDERFLOW_RISK) unless the user opts out |
| nh_partial_claim | a subset of the spectrum | N2 to N4 NOT_CHECKED (NEEDS_WHOLE_SPECTRUM) |

Verdicts of `run_cases.py`: clean (control with every must-certify statement CERTIFIED), detected (planted error, a
must-not-certify statement INCONCLUSIVE or NOT_CHECKED), sound (planted error, nothing false certified), unsound, alarm.
The exit code is 0 when no case is unsound and every control is clean. Run on 11 Nov 2026: 5 clean, 2 detected
(nh_duplicated_vector, nh_subnormal_entry), 4 sound, 0 unsound, 0 alarms. The test `tests/test_nonhermitian.py` runs the
same check.

The new statements are UNVERIFIED (D-12): Sarang's check is pending.
