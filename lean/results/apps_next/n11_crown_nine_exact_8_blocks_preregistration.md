# Pre-registration: exact re-check of the nine-coupling crown, all 8 blocks (N-11)

Date: 2026-11-22. Code: CERTIFY-ED main 71f155df. Host: the Linux GPU host (CPU only, 8 processes under one pin of
16 cores, nice 10, BLAS threads 1), run venv with python-flint 0.9.0.

Question. At the printed couplings of Kranitz and Penc (arXiv:2601.07800, Eq. (D4), the parenthesized digit read as one
more decimal: reading "appended"), does the exact count of eigenvalues of the S^z = 0 crown Hamiltonian below the fixed
shift SHIFT_APPENDED = -23061663926287/41943040000000 vanish in every one of the 8 symmetry blocks? The shift is about
-0.5498329145023108 in units of H_0. It lies inside the floating-point enclosure -0.5498329145683885 <= lambda_min <=
-0.5498329144473251 (about 6.6e-11 above its lower end). The exact re-check of 11-12 Nov counted 3 of the 8 blocks
(count 0 each) before it was stopped.

Method. `experiments/n11_crown_nine.py appended --blocks i --cap-gib 16` for i = 0..7, one process each, each writing its
own run folder; then `experiments/n11_crown_nine.py --combine <8 folders>`, which checks that the readings, K, shift and
dimensions agree, that the blocks are 0..7 once each with dimensions summing to dim S^z = 0, and that every count is 0,
and gives the exact lower bound shift - max(delta)/K.

Stop rules. A run clock of 6 hours. No change of the shift inside this run. A block not finished in time is reported as
not finished (its process keeps nothing); finished blocks are kept.

Outcomes. (a) Every count 0: the exact lower bound (an exact rational) is recorded and compared with the floating-point
lower bound and with the paper's -0.549832(8). (b) A count above 0 in a block: an eigenvalue of that block lies below
the shift. That does not contradict the floating-point certificate, whose lower bound is below the shift, but this shift
then gives no exact bound; it is reported as such, and a run at a lower shift is a separate, pre-registered run.
(c) Blocks not finished: the result is partial and reported as such.
