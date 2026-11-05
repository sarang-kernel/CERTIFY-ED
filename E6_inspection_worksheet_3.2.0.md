# E6 inspection worksheet, 3.2.0 evidence run (results/run_full_20261104, e6_mutation_20261104_163828)

64 mutants, 36 killed, 28 survived, 0 timeouts (3.1.0 run: 38 of 64). Readings below are the lead's suggestions (RA-25), not verdicts; the verdicts are Sarang's. The 3.2.1 run repeats E6 from tag v3.2.1 with the same seed; sites in cost.py and ldl_residual.py may shift by a few lines.

Classes: equivalent (no output can change), looser (the bound only grows), heuristic (parameters that decide where to look, not what a count says), test gap (a reachable path without a test), soundness-relevant test gap (a bound or count that could become wrong under the mutant, hidden by the margins of the fixtures), needs inspection.

| mutant | mutation | class | reading | suggested action |
|---|---|---|---|---|
| `construct.py:220` | 0.0 -> 1.0 in hermitian_part's early return | equivalent | when H == H^H exactly, (H + H^H)/2 == H bit for bit | none |
| `fp.py:40` | drop unary minus in _SQRT_HALF_2M537 | equivalent (dead constant) | the constant is defined and never used (git grep) | delete the constant |
| `certify.py:640` | and -> or (count == k, not distinct) after the shift loop | equivalent status | reached only when no CERTIFIED return happened; with `or` the same INCONCLUSIVE is returned with a different reason text | none, or pin the reason text |
| `checker.py:535` | gd - dd -> gd + dd in the witness separation test | test gap, false FAILED possible | a weaker separation condition can mark a witness separated and turn INCONCLUSIVE into FAILED; never a false CERTIFIED | test: witness point within dd of a component on the left side |
| `eigenspace.py:154` | sin_theta ** 2 -> sin_theta * 2 | looser | s < 1, so 2 s >= s^2: the label bound only grows | none |
| `checker.py:234` | RE != 0 -> RE != 1 in the exact-matrix deduplication | needs inspection | entries whose scaled real part equals 1 (and imaginary part 0) would be dropped; S6 would then report a false FAILED (the residual of the claimed vector is nonzero), not a false CERTIFIED. Survival means no test drives this reduceat path with such entries, or the path is reached only for duplicated (r, c) pairs | test: a spec with duplicated terms whose summed scaled entry is 1 |
| `inertia.py:129` | gamma_dot(2n + 2) -> gamma_dot(3n + 2) | looser | larger rounding constant for the dense LDL^T residual | none |
| `certify.py:519` | 1j * randn -> 1j / randn in the model-fidelity probe vector | heuristic | a different random complex probe; the probe is probabilistic either way | none |
| `sectors.py:57` | or -> and in the complex-data flag of audit_sector | soundness-relevant test gap | with a real V and a complex H (or the converse) the real rounding constants would be used for complex arithmetic (gamma_dot smaller by up to 2x) | test: audit_sector with real V, complex H; pin cplx |
| `fp.py:124` | (1 + gamma_{w+1}) -> (2 + gamma_{w+1}) in residual_entry_count | looser | larger entry count, larger additive term | none |
| `construct.py:139` | x & 1 -> x & 2 in the popcount fallback | not reached on this stack | numpy 2.4 has bitwise_count; the fallback is dead here | test forcing the fallback (monkeypatch hasattr) |
| `certify.py:493` | herm > 0 -> herm > 1 in the model-fidelity symmetrisation | needs inspection | herm is max|Hp - Hp^H|; with the mutant a producer matrix with asymmetry in (0, 1] is not symmetrised before D = Hh - Href. The 1/inf-norm bound on ||D||_2 holds for any D, so the shift stays an upper bound on ||Hp - Href||; Weyl's transfer needs the Hermitian part, which is what the statements refer to (Lemma A2) | pin: the record's model_shift_bound with a slightly non-Hermitian producer matrix |
| `certify.py:798` | max(1.0, ...) -> max(2.0, ...) in the window-edge scale | looser | larger collar scale | none |
| `eigenspace.py:151` | * len(members) -> / len(members) in the label rounding term | soundness-relevant test gap (rounding scale) | the rounding term of the label bound shrinks by g^2; invisible to value tests because the term is ~1e-14 | pinning test of the formula (as tests/test_backlog_proofs.py does for the cluster bound) |
| `certify.py:1039` | bad_x[:8] -> [:9] | message text |  | none |
| `sectors.py:179` | dropped `not` in restrict_to_u1 | outcome-equivalent | single-sector support: returns None (full-space fallback, slower); multi-sector: index_of fails, returns None at line 194 | none |
| `checker.py:521` | > -> >= in top_excess | heuristic | an extra witness count; FAILED still requires count > claimed at line 536 | none |
| `checker.py:443` | gap / 4 -> gap / 5 in tight() | heuristic | placement of the exact count points | none |
| `eigenspace.py:147` | 1/sqrt(1 - eta) -> 1*sqrt(1 - eta) in kappa | soundness-relevant test gap (rounding scale) | kappa becomes negative, the (2 kappa + kappa^2) term shrinks the label bound; invisible at eta ~ 1e-16 | pinning test with a planted eta ~ 0.1 |
| `eigenspace.py:103` | s < 1 -> s <= 1 | boundary | s == 1 exactly admitted (Davis-Kahan needs s < 1); measure zero | test with s == 1 exactly |
| `sectors.py:120` | rtol * scale -> rtol / scale in the trace/Frobenius consistency check | needs inspection | `consistent` feeds sectors.py:109 `passed`; a tighter tolerance can only turn passed False (more INCONCLUSIVE), a looser one is not what the mutant does | test with fH > 1 and a discrepancy between rtol/scale and rtol*scale |
| `enclosure.py:537` | == 'c' -> != 'c' in the bytes-per-entry of the finiteness scan | equivalent | only the block width changes | none |
| `exact_sectors.py:202` | nu // factor -> nu * factor | soundness-relevant test gap | factor = 2 for the real embedding of a complex block; the exact count would be 4x the truth and the S6 count comparison would FAIL (or match a wrong claim) | test: complex sector block through inertia_exact with factor 2 |
| `certify.py:558` | 16 rad -> 17 rad in the shift margin | heuristic |  | none |
| `localops.py:189` | [0.0, 1.0] -> [1.0, 1.0] in the fermion '-' letter | soundness-relevant test gap (model) | the fermion lowering operator's amplitude vector; the built matrix for any spec with a '-' letter on a fermion site changes | test: fermion hopping with '+-' letters against a hand-built 2-site matrix |
| `certify.py:1054` | is not -> is in sector_lift_shift | record field | the record carries None where the shift belongs (and the converse); statuses unchanged | test: the field is present for sector claims |
| `eigenspace.py:129` | a Mult -> Div in 2 g ||O|| s + rnd | soundness-relevant test gap | the main term of the Theorem 6 observable bound changes; value tests pass because the certified interval still contains the true value when the bound is loose | pinning test of the bound formula on a planted case |
| `spectrum.py:103` | m.counted and iv.counted -> or | soundness-relevant test gap | a merged interval is flagged counted when one side is; `complete` at line 108 then holds for a spectrum with an uncounted sector | test: merge a counted with an uncounted sector interval; complete must be False |

Counts by class: equivalent 3, equivalent status 1, test gap, false FAILED possible 1, looser 4, needs inspection 3, heuristic 4, soundness-relevant test gap 7, not reached on this stack 1, message text 1, outcome-equivalent 1, boundary 1, record field 1.

The seven soundness-relevant test gaps (sectors.py:57, eigenspace.py:151/147/129, exact_sectors.py:202, localops.py:189, spectrum.py:103) are all of the form 'a constant or branch that a formula pinning test would catch'; none changes a statement of the 3.2.0 run itself (the run's records were produced by the unmutated code), and the recommended tests belong to 3.2.2 together with the verdicts.
