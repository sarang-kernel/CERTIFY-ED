# Independent re-derivation: Theorem 2′ (sharper cluster radius ε*)

Fresh-context session, 2026-10-30. Sections 1 and 3 of `../thm2prime_eps-star.md` read first; proof written
before section 2 was opened. Disclosure: `docs/theory.md` §2, read for the notation, carries a proof *sketch* of
Theorem 2′ (steps (i)–(iii) in three lines); the proof below fills in every step and justifies each inequality.
Notation as in `independent/thm2_polar-factor.md`: G = XᴴX, η = ‖G − I‖ < 1, M = G^{1/2}, Q̂ = XM⁻¹,
δ = 1 − √(1 − η), R = AX − XΛ, Λ Hermitian with eigenvalues μ₁ ≤ … ≤ μ_g, spread = μ_g − μ₁.

## Statement checked

There are g distinct indices with |λ_{i_j} − μ_j| ≤ ε* := ‖R‖/√(1 − η) + ‖XᴴR‖/(1 − η) + δ·spread/√(1 − η).

## My proof

Facts from the polar-factor lemma (proved in `independent/thm2_polar-factor.md`): Q̂ᴴQ̂ = I, X = Q̂M,
‖M − I‖ ≤ δ, ‖M⁻¹‖ ≤ (1 − η)^{−1/2}, M Hermitian, G = M².

Let H′ := Q̂ᴴAQ̂ (Hermitian g × g) with eigenvalues θ₁ ≤ … ≤ θ_g.

*(i) Kahan step.* From AX = R + XΛ and X = Q̂M (M invertible): AQ̂ = RM⁻¹ + Q̂MΛM⁻¹. Multiply by the orthogonal
projector I − Q̂Q̂ᴴ, which annihilates Q̂: (I − Q̂Q̂ᴴ)AQ̂ = (I − Q̂Q̂ᴴ)RM⁻¹. The left side is AQ̂ − Q̂(Q̂ᴴAQ̂) =
AQ̂ − Q̂H′. Since ‖I − Q̂Q̂ᴴ‖ ≤ 1 (Hermitian projector), ‖AQ̂ − Q̂H′‖ ≤ ‖R‖‖M⁻¹‖ ≤ ‖R‖/√(1 − η). Kahan's inclusion
theorem (Parlett, *The Symmetric Eigenvalue Problem*, Thm 11.5.1; here H′ is the Rayleigh quotient, the classical
case) gives g distinct indices i_j with |λ_{i_j} − θ_j| ≤ ‖R‖/√(1 − η).

*(ii) Weyl step.* H′ = M⁻¹XᴴAXM⁻¹ = M⁻¹Xᴴ(R + XΛ)M⁻¹ = M⁻¹XᴴRM⁻¹ + M⁻¹GΛM⁻¹ = M⁻¹XᴴRM⁻¹ + MΛM⁻¹ (G = M²,
M⁻¹M² = M). Hence H′ − Λ = M⁻¹XᴴRM⁻¹ + (MΛ − ΛM)M⁻¹. For any scalar c, [M, Λ] = [M − I, Λ − cI] (the identity
commutes with everything), and ‖[B, C]‖ = ‖BC − CB‖ ≤ 2‖B‖‖C‖. With c = (μ₁ + μ_g)/2, ‖Λ − cI‖ = spread/2
(Hermitian Λ with spectrum in [μ₁, μ_g]). Therefore
‖H′ − Λ‖ ≤ ‖M⁻¹‖²‖XᴴR‖ + 2δ(spread/2)‖M⁻¹‖ ≤ ‖XᴴR‖/(1 − η) + δ·spread/√(1 − η).
Weyl's inequality for Hermitian matrices (Horn–Johnson, *Matrix Analysis*, Cor. 4.3.15 / Thm 4.3.1): the
ascending eigenvalues of H′ and Λ satisfy |θ_j − μ_j| ≤ ‖H′ − Λ‖ for every j.

*(iii)* |λ_{i_j} − μ_j| ≤ |λ_{i_j} − θ_j| + |θ_j − μ_j| ≤ ε*, with the distinct indices of (i). ∎

Hypothesis use: η < 1 for M⁻¹ and the bound ‖M⁻¹‖ ≤ (1 − η)^{−1/2}; Λ Hermitian for Weyl (a non-normal Λ would
break (ii)); A finite only so that the eigenvalues exist. For g = n the projector I − Q̂Q̂ᴴ is zero and the first
term could be dropped; the bound stays valid.

Remark check: R = 0 and spread = 0 give ε* = 0 (exact eigenvectors of one level, any non-orthogonality) while
ρ = (‖A‖ + ‖Λ‖)δ > 0 when η > 0; for orthonormal X (η = 0) ε* = ‖R‖ + ‖XᴴR‖ ≥ ρ = ‖R‖. Neither dominates,
min(ρ, ε*) is sound because each is a proven radius for *some* distinct matching (the two matchings may differ;
the statement "there are distinct indices within min(ρ, ε*)" is the disjunction, which is what the code uses: the
radius under which the members are matched to distinct eigenvalues).

## Comparison with section 2 of the original (read after the proof above was written)

Steps (i)–(iii) coincide with mine line by line (same projector argument, same commutator identity
[M, Λ] = [M − I, Λ − cI] with c the midpoint, same Weyl reference). Same constant ε*. Both proofs valid.

**Code mapping (current `dev/3.2` lines, `enclosure._cluster_bound`).** Rc 497, XR 498, xnorm 499, Rerr 500,
XR_bound 501–502, inv_sqrt 503, inv 504, spread 506, eps_star 507, min 511. Rounding counts re-checked:
XR_bound has 5 roundings inside `up` (two products, the `gd` product, product, two additions — the table says
"5 (… = 6 operations)" and relies on the (1 + 5u)² slack of the two `fro_upper` for the sixth and for `gd`; with
(1 − u)⁷(1 + 8u) ≥ 1 even six would be covered), eps_star has exactly six (four products, two additions) —
the cap of `up`. Directions: all towards +∞; (1 − η)^{−1/2} and (1 − η)^{−1} are increasing in η, so evaluating
them at the stored η ≥ exact η is sound.

**Gap in the code mapping (not in the proof).** The table states `res.E_norm[i] ≥ ‖Rc_i − r_i‖` "(Theorem 1
accounting, step 3–4)". That holds under the standard model only. Under the underflow-tolerant model this branch
adopted (INT-05), the per-column shortfall of Rc_i + E_i is √n·c_w·u_min (`fp.residual_additive_term`), and that
term is added to `res.upper` (enclosure.py 368) but *not* to `res.E_norm`. Hence `Rerr = norm2_upper(E_norm)`
(line 500) can fall short of ‖Rc − R‖_F by up to √g·√n·c_w·u_min ≈ 10⁻¹⁶⁰·√(gn)·c_w, and `XR_bound` by ‖X‖₂
times that. Consequence: the stated direction "exact ε* ≤ eps_star" is proved only when ε* ≳ 10⁻¹⁴⁵ (where the
≥ 2u relative slack of `up` after five roundings exceeds the missing term), or when no operation of the residual
underflows (then η = 0 in the model and E_norm suffices). The statement itself is unaffected; the fix is a looser
bound: add `math.sqrt(g) * residual_additive_term(n, adapter.width, cplx) + ULP_MIN` to `Rerr` (or the equivalent
term `xnorm * …` outside `up` in `XR_bound`). My adversarial run (subnormal operator entries 2⁻¹⁰³⁰…2⁻¹⁰⁷⁰,
subnormal vectors) found no actual violation: the relative slack of E over the true rounding error covers the
worst-case additive allowance in every case tried. Recorded in VERDICTS.md under Required corrections.

**Verdict: AGREES-WITH-REMARKS on the proof; GAP in the code mapping (INT-05 additive term missing in `Rerr`;
statement still holds; closing change given above).**

## Runs

- `verify_thm2prime_eps-star.py --cases 1500 --seed 7`: inequalities E1–E3 3786 random + 1875 adversarial
  (413 skipped), ε* 1262 + 625, implemented 1122 + 549 (216 skipped); 0 violations.
- `adversarial_thm2prime_eps-star.py --cases 400 --seed 5`: theorem 29 random + 254 adversarial (117 skipped,
  η ≥ 1), intermediate bounds 22 + 232, implemented ε* ≥ exact and radius = min 22 + 232 (29 skipped, Frobenius
  η ≥ 1), remark checks 80; 0 violations.
