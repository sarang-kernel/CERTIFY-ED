# AKLT models on two-dimensional lattices (Pomata and Wei)

Pomata and Wei (Physical Review Letters 124, 177203 (2020); arXiv:1911.01410) show that the AKLT models on several
two-dimensional lattices of degree 3, and on decorated lattices, have a spectral gap. For each lattice the gap follows
from a finite-size criterion of the form $\eta < 1/\tilde z$, or $4\eta_\perp + 2\eta_\parallel < 1$ for the
square-octagon lattice, where $\eta$ is computed from two small overlapping subgraphs of the lattice. The authors
computed $\eta$ in floating point, by full diagonalization or the Lanczos method. CERTIFY-ED replaces every $\eta$ by an
upper bound obtained in exact arithmetic and decides each criterion exactly. The code is `certify_ed/apps/aklt/`
(`lattices.py`, `graph.py`, `kernel.py`, `eta.py` and `__init__.py`); the run records are experiment E10 of the 3.1.0
evidence run, `code/certify-ed/results/run_full_20261022/e10_aklt_20261022_170609/data.json`, and of the 3.2.1 evidence
run, `code/certify-ed/results/run_full_20261105/e10_aklt_20261105_071106/data.json`. The two records agree in all 333
compared leaves except timings (`code/certify-ed/RESULT_evidence_run_3.2.1.md`); every bound below is the same in both.

## The quantity $\eta$

For two subspaces A and B of a Hilbert space with orthonormal bases $Q_A$ and $Q_B$, the cosines of the principal angles
between them are the singular values of $Q_A^H Q_B$. A cosine equal to 1 belongs to a direction in $A \cap B$. In the
criterion of Pomata and Wei, A and B are the ground spaces (the kernels) of the AKLT Hamiltonians $H_{\Gamma_A}$ and
$H_{\Gamma_B}$ of two overlapping subgraphs, embedded in the space of their union, and $\eta$ is the largest cosine that
is not 1. The criterion is an operator inequality of the form $\tilde H^2 \ge (1 - \tilde z \eta) \tilde H$ together
with its reduction to a finite subgraph; it is taken from the paper and not re-proved here
(`code/certify-ed/docs/applications.md`). Here $\tilde z$ is 4 for the star lattice [3.12.12], 3 for the cross lattice
[4.6.12], 4 for the decorated honeycomb and decorated square lattices, and 6 for the honeycomb lattice (Method III of
the paper).

## Why this application

The reason is the same as for the crown (`crown_kranitz_penc.md`): a published proof whose conclusion depends on a
floating-point eigenvalue computation. Pomata and Wei did check their numbers, by ARPACK residuals of about $10^{-13}$
(`review/REVIEW_E-manuscript.md`, finding A01), so the expected outcome was a confirmation, not a correction. What a
certificate adds to a residual check is that the count of ground states and of unit cosines is complete: a residual
shows that a computed vector is close to some eigenvector, not that no eigenvector was missed, and a missed unit cosine
or a missed cosine just below the threshold would change $\eta$.

## How the bound is computed

The VBS (valence-bond solid) states of a subgraph are written as exact integer vectors in the scaled basis
$D = \mathrm{diag}(\sqrt{\binom{z}{k}})$, where z is the degree of the site, twice its spin; boundary configurations are
reduced to the number of down spins on each site's free virtual spins, and each VBS map $U_X$ is checked to be injective
by the exact rank of its Gram matrix. The kernel lemma (below) identifies the ground space of each subgraph Hamiltonian
with the range of its VBS map. Both ground spaces are then reduced to
$V = \mathrm{range}\, U_L \otimes \mathrm{range}\, U_C \otimes \mathrm{range}\, U_R$, the product of the left, centre
and right parts; the parts of A and B outside V are orthogonal to the other subspace, so the reduction is exact (the
argument is in the docstring of `apps/aklt/eta.py`). The squared cosines are the eigenvalues of the pencil
$(C G_B^{-1} C^T, G_A)$, with $G_A$ and $G_B$ the Gram matrices of the two bases and C their cross Gram matrix. The code
never inverts $G_B$: it counts the pencil eigenvalues above a rational threshold $t = p/q$ through the inertia of the
bordered integer matrix

$$M = \begin{pmatrix} -p\, G_A & q\, C \\ q\, C^T & -q\, G_B \end{pmatrix}$$

(the bordered-pencil lemma, below). Everything is split into $S^z$ blocks, and the mirror symmetry of the decorated
square and the spin flip act by signed permutations that are checked exactly and give an exact integer congruence into
character blocks; blocks exchanged by the flip are factored once. A floating-point $LDL^T$ factorization of each block
only proposes factors; the residual of the factors is computed exactly with FLINT integers, which gives a bound $\beta$,
and the count is moved to the threshold $t' = (1 + \beta/\mu)(t + \beta/\nu)$, with $\mu$ and $\nu$ certified lower
bounds on the smallest eigenvalues of the Gram matrices. The number of unit cosines is bounded from below by the rank
modulo the prime $p = 2^{31} - 1$ of the VBS states of the union subgraph. If the number $\pi$ of positive pivots is at
most that rank, every pencil eigenvalue above $t'$ is a unit cosine, and every other squared cosine is at most $t'$, so
$\eta \le \sqrt{t'}$. The threshold is set just above the floating-point value, $t = \eta_{\rm float}^2 (1 + 10^{-9})$;
fallbacks at $10^{-7}$ and $10^{-5}$ exist and were not needed in either run. The criterion is decided by comparing the
exact rational $t'$ with $1/\tilde z^2$; for the square-octagon lattice the two square roots are bounded from above by
rationals (`apps/aklt/__init__.py`, `square_octagon_criterion`). No step depends on the floating-point model.

## The kernel lemma

**Statement.** Let $\Gamma$ be a finite simple graph whose vertex v carries spin $S_v$ with $2S_v \ge \deg_\Gamma(v)$,
and let $H_\Gamma = \sum_{(u,v) \in E(\Gamma)} P^{(uv)}_{S_u + S_v}$, where $P_J$ projects onto total spin J of the
pair. Then $\ker H_\Gamma$ is the span of the VBS states of $\Gamma$ with arbitrary boundary spins, that is the range of
the reduced VBS map $U_\Gamma$, and its dimension is $\prod_v (2S_v - \deg_\Gamma(v) + 1)$.

**Proof outline.** The spin-$S_v$ space is the space of homogeneous polynomials of degree $2S_v$ in two variables
$(a_v, b_v)$. For one edge, the polynomials divisible by $\omega_{uv} = a_u b_v - b_u a_v$ form exactly the sum of the
components of total spin below $S_u + S_v$, by a dimension count and because the decomposition of the pair is
multiplicity-free; so the kernel of $P_{S_u + S_v}$ consists of these polynomials. $H_\Gamma$ is a sum of positive
semidefinite terms, so its kernel is the intersection of the edge kernels, the polynomials divisible by every
$\omega_e$; the $\omega_e$ are irreducible and pairwise non-associate, and the polynomial ring has unique factorization,
so these are the multiples of $\Psi = \prod_e \omega_e$, which is the VBS state. A monomial cofactor is a boundary
configuration (`code/certify-ed/docs/applications.md`, section 3; `manuscript_v4/sections/appC_applications.tex`).

**Status.** Classical (Affleck, Kennedy, Lieb and Tasaki 1988; Kirillov and Korepin 1989), re-derived in the project as
a check and taken off the D-12 list (review finding M-11). It is also confirmed by computation, independently of the
proof: the direction "VBS range is contained in the kernel" by exact integer VBS states with $H U = 0$ and their exact
rank, the other direction by the nullity of H modulo $p = 2^{31} - 1$ per $S^z$ block (the nullity over the rationals is
at most the nullity modulo p). The review found the dimension formula on 13 of 13 test graphs (finding B5-3 of
`review/REVIEW_B_applications.md`).

| subgraph | sites | dimension | rank of U | nullity of H mod p | kernel equals VBS range | seconds in the 3.2.1 run |
|---|---|---|---|---|---|---|
| star (two triangles and a bridge) | 6 | 4096 | 16 | 16 | yes | 1 |
| square-octagon (two squares and a bridge) | 8 | 65536 | 64 | 64 | yes | 1003 |
| decorated honeycomb, first subgraph | 9 | 62208 | 36 | 36 | yes | 1014 |
| decorated honeycomb, second subgraph | 9 | 62208 | 36 | 36 | yes | 1008 |

The seconds are from `code/certify-ed/results/run_full_20261105/E10.log`. The subgraphs of the cross lattice (16.7
million states) and of the decorated square lattice (4.1 million states) are too large for this check and rely on the
lemma alone.

## The bordered-pencil lemma

**Statement.** Let $G_A, G_B$ be positive definite with $G_A \succeq \nu I$ and $G_B \succeq \mu I$, $t = p/q$ with
$q > 0$, M the bordered matrix above, and let factors satisfy $L \tilde D L^T = M + E$ with
$\lVert E \rVert \le q\beta$, where $\tilde D$ has $\pi$ positive eigenvalues. Then the number of eigenvalues of the
pencil $(C G_B^{-1} C^T, G_A)$ above $t' = (1 + \beta/\mu)(t + \beta/\nu)$ is at most $\pi$.

**Proof outline.** Haynsworth's inertia additivity applied to the block $-qG_B$ shows that M has as many positive
eigenvalues as $q(C G_B^{-1} C^T - t G_A)$, which by Sylvester's law equals the number of pencil eigenvalues above t.
For the perturbation, $M + E \succeq M - q\beta I$, the bordered matrix of the blocks $-(pG_A + q\beta I)$ and
$-q(G_B + \beta I)$; Haynsworth again, together with $(G_B + \beta I)^{-1} \succeq G_B^{-1}/(1 + \beta/\mu)$ and
$pG_A + q\beta I \preceq (p + q\beta/\nu) G_A$, bounds its positive count from below by the number of pencil eigenvalues
above $t'$ (`manuscript_v4/sections/appC_applications.tex`).

**Status.** New mathematics, marked UNVERIFIED (rule D-12); it has no derivation document among the 22, so the AKLT
bounds are "certified by the implementation, subject to it" (finding RA-20 in `report/07_applications.md`). The review
tested it on 3000 random instances, half of them with the worst-case perturbation $E = q\beta I$, with 0 violations and
equality in 1161 trials, and re-read the proof (finding B5-4); that is a check by an automated review, not the human
check that D-12 requires.

## Results

The certified bounds are rounded up at the tenth decimal; the criterion values are those of the generated table
`manuscript_v4/generated_3.1.0/tables/apps_aklt.tex` (identical in `manuscript_v4/generated/`). The value $\sqrt{t'}$
before rounding, the threshold $t'$, the lower bound on the number of unit cosines and the largest symmetry-adapted
block are from the 3.2.1 record; the seconds of each lattice's certification are those printed in
`code/certify-ed/results/run_full_20261105/E10.log` and `code/certify-ed/results/run_full_20261022/E10.log`.

| lattice | criterion | $\eta$ published | $\eta$ certified $\le$ (record) | rounded up | criterion value | $t'$ | unit cosines $\ge$ | largest block | seconds (3.2.1 / 3.1.0) |
|---|---|---|---|---|---|---|---|---|---|
| star [3.12.12] | $\eta < 1/4$ | 0.1110430220 | 0.11104302202338649 | 0.1110430221 | 0.44418 | 0.012330552740086294 | 32 | 70 | 0 / 0 |
| square-octagon, $\eta_\perp$ | $4\eta_\perp + 2\eta_\parallel < 1$ | 0.1589663310 | 0.15896633115208958 | 0.1589663312 | 0.84816 (sum) | 0.025270294439955796 | 256 | 420 | 3 / 4 |
| square-octagon, $\eta_\parallel$ | (same) | 0.1061446858 | 0.10614468587603454 | 0.1061446859 | | 0.011266694339722041 | 256 | 420 | 4 / 4 |
| cross [4.6.12] | $\eta < 1/3$ | 0.1997384500 | 0.19973845072185567 | 0.1997384508 | 0.59922 | 0.03989544869676715 | 256 | 1584 | 106 / 113 |
| decorated honeycomb | $\eta < 1/4$ | 0.1530329085 | 0.15303290857599913 | 0.1530329086 | 0.61214 | 0.023419071107230096 | 216 | 673 | 10 / 11 |
| decorated square | $\eta < 1/4$ | 0.2203543174 | 0.22035431930729685 | 0.2203543194 | 0.88142 | 0.04855602603738212 | 1728 | 4034 | 2368 / 2683 |
| honeycomb (Method III) | $\eta < 1/6$ | 0.1445124916 | not attempted | | | | | | |

The record gives the certified upper bound on $4\eta_\perp + 2\eta_\parallel$ as 0.8481546963604274, and every criterion
holds. The decorated square needs the group of order 4 (mirror and spin flip), which reduces the largest bordered block
from 8842 to 4034 (`E10.log`).

**How the instrument was checked.** First, before certifying, the code reproduces the published $\eta$ in floating point
for every lattice (the column $\eta_{\rm float}$ of the record agrees with the published value to the ten printed
decimals), which confirms that the subgraph geometry, read from the coordinates of the paper's TikZ figures, is the
paper's (finding B6-2). Second, a negative control: on the decorated honeycomb at $t$ = 0.0234, below $\eta^2$ ($\eta$ =
0.15297), the same code finds 218 positive pivots against 216 unit cosines and reports "not certified", so it does not
certify a false bound (`code/certify-ed/docs/applications.md`). Third, the review computed $\eta$ independently with its
own VBS construction for the star (0.11104302197) and the square-octagon (0.10614468577 and 0.15896633105) and found
that the thresholds and criteria of `apps/aklt/__init__.py` match the paper exactly (findings B5-1, B5-2, B6-1); it did
not recompute the cross and the decorated lattices (B5-8).

**What it shows.** Every criterion of the five lattices holds with a certified margin; the largest criterion value is
0.88142, for the decorated square. The certified bounds exceed the published values by 2.3e-11 to 1.9e-9
(`report/07_applications.md`); the excess combines the rounding of the published value, the margin of the threshold, and
the factorization term $t' - t$, which dominates on the cross and the decorated square. The largest excess, rounded up,
is 2.0e-9 (`manuscript_v4/generated_3.1.0/numbers_apps.tex`). The run-time statement check uses the threshold "excess <
5e-9", which was set after the run's data had been seen (commit `faf41db`); the manuscript therefore quotes the observed
maximum and names the check threshold separately (findings RA-23 and IMP-27).

**What it does not show.** The criterion itself (the operator inequality and its finite-size reduction) is taken from
the paper. The bounds rest on the bordered-pencil lemma, which awaits Sarang Vehale's check, and, for the cross and the
decorated square, on the kernel lemma without a computational confirmation. Older certificate files from 17 October 2026
in `legacy/applications_2026-10-17` also contain runs at $t = 1/\tilde z^2$; their $t'$ exceeds $1/\tilde z^2$ by a few
times $10^{-8}$, so those runs alone do not prove the strict inequality, and their "certified" flag records only the
eigenvalue count. The bounds of that day (for example 0.1110450359 for the star and 0.2203633381 for the decorated
square, after 46 minutes) were valid but looser; the unified pipeline of 18 October 2026 reproduced them, and the
threshold ladder of 21 October 2026 gave the values above (`report/07_applications.md`; next section).

![Certified upper bounds on eta against the published values, and the certified criterion
values](figures/fig_aklt_eta.png)

The figure (`code/certify-ed/docs/figures_run_3.2.1/fig_aklt_eta.png`, drawn from the 3.2.1 record) shows in panel (a)
the certified upper bound on $\eta$ of each lattice as a bar and the published value as a diamond, with the threshold
$1/\tilde z$ in orange; the bounds are printed to ten decimals, rounded to nearest. Panel (b) shows the certified
criterion values, which are below 1 for every lattice; they are printed rounded to nearest, while the table above rounds
them up, which is why the star appears as 0.44417 in the figure and 0.44418 in the table.

## The first certification scripts (17 October 2026)

The first AKLT certificates of the project were made with standalone scripts, kept in
`code/certify-ed/legacy/applications_2026-10-17/aklt_gap/` with their logs and certificate files and summarized in
`project_docs/notes/aklt_results.md` (version 0.2 of 21 October 2026). They used two formulations of the pencil count.
For the star, the square-octagon and the cross lattice the count is made on an integer matrix scaled with adjugates
(`pw_general.py`): the inverse of $G_B = G_L \otimes G_j$ is the Kronecker product of the two inverses, which are made
integer by multiplying with $\det G_L$ and $\det G_j$, so that $\det(G_L) \det(G_j)\, C G_B^{-1} C^T$ is an integer
matrix. For the two decorated lattices the scripts use the bordered matrix with Haynsworth inertia additivity, which
needs no inverse (`pw_reduced.py`, `pw_symm.py`, the latter with the lattice mirror and the spin flip as signed
permutations); this is the formulation that `apps/aklt/eta.py` now uses for every lattice. The table gives the state of
that day (`aklt_results.md`):

| lattice | certified upper bound, 17 October 2026 | kernel equality on that day | script |
|---|---|---|---|
| star [3.12.12] | 0.1110450359 | lemma and exact check (6 sites) | `star_certify.py`, `kernel_modp.py` |
| square-octagon | 0.1589685504 / 0.1061461257 (sum 0.84817) | lemma and exact check (8 sites) | `so_certify.py` |
| cross [4.6.12] | 0.1997398308 | lemma only (12 sites, 16.7M states) | `cross_certify.py` |
| decorated honeycomb | 0.1530359435 | lemma and exact check (both subgraphs, 9 sites) | `ds_certify.py` with `pw_symm` |
| decorated square | 0.2203633381 | lemma only (12 sites, 4.1M states) | `ds_certify.py` with `pw_symm` (mirror and spin flip; largest exact block 4034) |

The logs show the thresholds of these runs: the star was counted at $t$ = 12331/1000000 and at $t$ = 1/16, the latter
being $1/\tilde z^2$ (`star_certify.log`), and the decorated square took 2757 s in blocks of up to 4034 rows
(`ds_certify_square.log`). Every criterion held on that day. The unified code of 18 October 2026 reproduced these bounds
to 9 or 10 digits: 0.1110450359, 0.1589685504, 0.1061461257 and 0.1530359435 again, 0.1997398311 for the cross and
0.2203633377 for the decorated square (`code/certify-ed/results/run_full_final/E10.log`). The threshold
$t = \eta_{\rm float}^2 (1 + 10^{-9})$ of 21 October 2026 gave the sharper bounds of the table above; the `E10.log`
files of the run of that day (`code/certify-ed/results/run_full_20261021`), of the 3.1.0 run and of the 3.2.1 run print
the same six bounds. The kernel lemma is written out in `project_docs/notes/aklt_kernel_lemma.md`; the note of 21
October 2026 still lists it for Sarang Vehale's check under D-12, while the later review finding M-11 classifies it as
classical and takes it off that list (section "The kernel lemma" above).

## The honeycomb lattice and a note on the paper

The honeycomb lattice (Method III of the paper) was not attempted. Its space V has dimension $2^{26}$ (Table I of the
paper), and the final gap bound of 0.00266 also uses three further numerical constants $\gamma_p$, $\gamma^{[2p]}$,
$\gamma^{[4p]}$, each a separate eigenvalue computation on a plaquette cluster that would need its own certificate. It
is the natural next target on a larger machine. It is also the one case in which the paper's own precision argument is
weakest: a tolerance of $10^{-5}$ with an error of at most $10^{-4}$ inferred from a residual
(`review/REVIEW_E-manuscript.md`, finding A01). The same review noted that in the arXiv version of the paper the final
equation of the honeycomb proof gives $\gamma^{[2p]}\gamma^{[4p]}(1 - 6\eta)$ = 0.002660333395, while the conclusion
states 0.002550333395. The two numbers differ in one digit; which one is intended was not checked, and it is not a case
CERTIFY-ED computed. It is recorded as a possible future note to the authors (`report/07_applications.md`).

## Reproducing

From the root of the code clone (see `docs/math/README.md`): `python run.py --only E10` runs the experiment;
`certify_ed.apps.aklt.certify_lattice(name)` with one of the names `"star"`, `"square_octagon_perp"`,
`"square_octagon_par"`, `"cross"`, `"decorated_honeycomb"`, `"decorated_square"` certifies one lattice, and
`certify_ed.apps.aklt.kernel.kernel_check` runs the kernel confirmation. E10 took 5848 s in the 3.2.1 run
(`code/certify-ed/RESULT_evidence_run_3.2.1.md`).
