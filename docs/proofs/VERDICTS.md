# Independent re-derivation verdicts (Phase 2, rule D-12/D-59)

Fresh-context session, 2026-10-30. For every derivation document in `docs/proofs/` the statement (section 1) and the
code mapping (section 3) were read first, an independent proof was written in `docs/proofs/independent/<id>.md`,
then section 2 of the original was compared step by step, the original `verify_<id>.py` was run at a reduced
size and an independent adversarial search `docs/proofs/independent/adversarial_<id>.py` was run (exact arithmetic or
mpmath at 60 digits as ground truth). Verdict classes: AGREES, AGREES-WITH-REMARKS, GAP, ERROR. None of these
results is described as verified; Sarang's check (D-12) is pending for all of them.

| document | statement checked | protocol followed | verdict | remarks | original verify run (reduced) | independent adversarial counts | violations | date |
|---|---|---|---|---|---|---|---|---|
| thm2_polar-factor.md | Polar-factor lemma (a)-(d); Kahan inclusion for any Hermitian block; Theorem 2 radius rho = \|\|R\|\| + (\|\|A\|\|+\|\|Lambda\|\|) delta | 1→2→3→4→5 (sections 1 and 3 read first; own proof written; then section 2 compared; verify script run at reduced size; adversarial search) | AGREES-WITH-REMARKS | Proof valid, same constant. Code mapping holds at current lines (enclosure.py 476-511; certify.py 296-303; sectors.py 125-148); section 3 line numbers are those of track/backlog-proofs. Remark: sectors.audit_orthogonality has no rounding term for the computed product V_a^H V_b (input of the decomposition_shift eta) - filed under thm8_rho. | verify --cases 2000 --seed 7: 3452+860 lemma, 1726+430 Kahan, 1726+430 Thm2, 500+600 Kahan-block, 3291+809 code; 0 violations | 500 cases seed 3: 36 random + 355 adversarial (109 skipped eta>=1); code 26+225 (140 skipped Frobenius eta>=1); 129 negative controls; 0 violations | 0 | 2026-10-30 |

## Required corrections

None recorded.
