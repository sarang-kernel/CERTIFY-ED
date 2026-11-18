# RESULT: D-05, GPU products for S2 (branch track/gpu-d05)

Date: 17 November 2026. Branch `track/gpu-d05` from `dev/3.3` at `c29e968`; the code commit is `cc4af9e` (computational
source hash `9f190f7e9047d953`, tests hash `af3552384494fc84`), followed by a second commit with this document, a
faster spot check in `certify_ed/gpu.py` and the field `device` in the certificate schema. Machine: a shared GPU node (8 NVIDIA H100 80GB HBM3, 208
cores, the GPUs shared with other jobs); this work used one GPU at a time through the node's job wrapper (12 GB of
device memory, 8 cores, nice 10, duty cycle 0.5 unless stated). Ubuntu 24.04, Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1,
CuPy 13.6.0, CUDA runtime 13.0 (13000), driver 13.2 (13020), NVRTC 13.0. The run records are in the archive of this
document.

## What the GPU computes

For a stored sparse operator, `Options(device="gpu")` forms the two products from which every S2 radius is built,
H X and |H| |X|, on the GPU (`certify_ed/gpu.py`), in `enclosure.residuals` and in the Kato-Temple step of
`certify()`. The kernels perform the operations of SciPy's CSR product in SciPy's order: each row sum starts from +0.0
and adds the products of the stored row in index order, each product and each sum rounded once to nearest
(`__dmul_rn`, `__dadd_rn`, NVRTC option `--fmad=false`); complex products are formed as SciPy forms them. Under binary64
round to nearest with gradual underflow these are the same floating-point operations as on the CPU, so the products,
and every bound built from them, equal the CPU's bit for bit. The bounds are those of
`docs/proofs/thm1_fp-accounting.md` applied to the same numbers; no rounding analysis is added.

Two checks guard the premise on each device: `gpu.probe()` once per process (+, - and * against NumPy on three
magnitude ranges, gradual underflow on fixed cases, the kernels against SciPy on random real and complex CSR matrices
with empty rows, unsorted indices, subnormal and large entries) and a spot check of 64 rows of every product against
SciPy. On any failure, and for dense, matrix-free and black-box operators, the CPU forms the products and S2's evidence
field `device` says why. Norms, Gram matrices, inertia counts and the exact layer stay on the CPU.

## The device's binary64 arithmetic (`experiments/d05_fp_probe_gpu.py`)

| check | cases | result |
|---|---|---|
| +, -, *, /, sqrt and the separately rounded a*b + c against the CPU, bit for bit | 4 families x 200000 | 0 mismatches |
| fma against the correctly rounded exact value (rationals) | 15524 finite cases | 0 mismatches |
| directed rounding: `__dadd_ru/rd`, `__dmul_ru/rd`, `__fma_ru/rd` against exact values | 4 x 3000 per operation | 0 mismatches |
| gradual underflow (subnormal operands and results) | 6 fixed cases | all as IEEE-754 prescribes |
| a*b + c with NVRTC's default flags | 4 x 200000 | contracted to fma (differs from the separately rounded value in 1115, 16955, 25716 and 7071 cases) |
| a*b + c with `--fmad=false` | 4 x 200000 | never contracted |

The four families are random bit patterns over the full exponent range, exponents within 2^-60 to 2^60, operands
whose products and sums fall into the subnormal range, and ties near 1. The check of cuBLAS `ddot` and cuSPARSE SpMV
against the gamma_n bound did not run: CuPy's wrappers launch elementwise kernels that need the CUDA headers, which
this host does not have. `certify_ed.gpu` uses neither library. The module's own probe held on the device used
(`ops_full_range`, `ops_narrow`, `ops_low`, `gradual_underflow`, `kernels_match_scipy`).

The default contraction matters: a kernel compiled with NVRTC's default flags fuses a*b + c, so its results are not
the CPU's. The bounds of `fp.py` would still hold (its model allows fused multiply-add and any summation order), but
the bit for bit equality, on which this path rests, would not; the kernels are therefore compiled with `--fmad=false`
and use the rounding intrinsics explicitly.

## Tests

`tests/test_gpu.py`: 17 passed on the H100 (the products of real and complex matrices with sorted and unsorted indices
and empty rows, one pass for both products, column blocks, enclosures and residual bounds, `certify()` with
`device="gpu"` against `device="cpu"`: same statuses, same records apart from timings and the `device` field, same
arrays; spot-check rejection, injected fallback, no device, dense and unknown devices). The full suite of the branch
with no device visible: 1222 passed, 70 skipped, 4 failed; plain `dev/3.3` in the same environment: 1218 passed, 57
skipped, the same 4 failed (see the Linux findings below).

Re-test of 18 November 2026 at the head of the branch after the reviews (`937ae1e`, source hash
`7fc028c29b1173fd`, tests hash `6bbd879b8887b05a`; one H100, CuPy 13.6.0, CUDA runtime 13.0): the device probe holds
(full range, narrow range, low range, gradual underflow, kernels equal to SciPy's), and `tests/test_gpu.py` gives 21
passed. `certify()` with `device="gpu"` against `device="cpu"` on a banded matrix of dimension 20000 (real and
complex, four claimed pairs, S2, S3 and Kato-Temple) and on the Heisenberg chain of 18 sites with its model gives the
same statuses, the same records apart from timings and the `device` field, and the same arrays bit for bit, with the
GPU products used (4, 4 and 2 products; spot checks on 128, 128 and 64 rows). For the same chain stored with every
entry split into two halves the device was not used (`|A| does not share the pattern of A`), and the CPU products
gave the same certificate as the CPU run.

## Products and residual bounds against the CPU (`experiments/d05_gpu_bench.py`)

Heisenberg chains in the full space (n = 2^N), k = 8 random vectors and values. CPU: SciPy's CSR product on one thread,
as certify-ed uses it. GPU: wall time including the transfers, the spot check and the sleeps of the duty cycle, and
busy time without the sleeps. `residuals` is `enclosure.residuals` on each path. In every row the products and the
residual bounds of the two paths are equal bit for bit.

| N | n | nnz | products CPU (s) | products GPU busy (s) | products GPU wall (s) | residuals CPU (s) | residuals GPU (s) |
|---|---|---|---|---|---|---|---|
| 18 | 262144 | 2621440 | 0.030 | 0.007 | 0.020 | 0.104 | 0.072 |
| 20 | 1048576 | 11164824 | 0.134 | 0.073 | 0.190 | 0.558 | 0.407 |
| 22 | 4194304 | 50331648 | 0.650 | 0.176 | 0.355 | 3.022 | 2.065 |
| 24 | 16777216 | 212695496 | 2.981 | 0.665 | 1.333 | 17.523 | 10.221 |
| 24, no duty cycle | 16777216 | 212695496 | 3.031 | 0.671 | 0.675 | 17.797 | 9.688 |

Rows N = 18 and 20 are from the run with the first spot check (ended 11:14 UTC); rows N = 22 and 24 from the run
with the spot check of the second commit (0.0012 s and 0.0009 s per product pair, against 0.130 s and 0.551 s
before). At N = 24 the products are 4.5 times faster on the GPU without the duty cycle and `residuals` 1.8 times. The
rest of `residuals` stays on the CPU: at N = 24 the residual formation took 0.68 s, the computed norms 1.04 s, the
column bounds of R and x 1.66 s, the elementwise bound E 0.90 s and its column bounds 0.81 s (all eight vectors at
once; `residuals` itself works in blocks of two columns here, because the default budget of 512 MB for four n x b
temporaries allows one column at n = 2^24 and a block has at least two). At the start and end of the run that ended at
11:14 UTC other jobs kept the GPUs at 13 % to 37 % utilization.

`certify()` with only S2 on the chain N = 18 (the four lowest eigenpairs from `eigsh`) gives the same certificate on
both paths; S2 took 0.134 s on the CPU and 0.123 s with the GPU products. At this size the products are a small part
of S2.

## What is not on the GPU

The reductions: `np.linalg.norm` sums in the order of the CPU's BLAS, which a GPU kernel does not reproduce bit for bit.
Moving the norms to the GPU would keep the bounds valid under the model of `fp.py` (any summation order, fused
multiply-add allowed) but would change the last bits of the certified numbers, so certificates of the two paths would
no longer be identical. The elementwise parts (R = H X - X diag(mu) and the bound E) can move without changing a bit
(they are the same IEEE operations); at N = 24 they are about 1.6 s of 17.5 s. Dense operators (cuBLAS sums in its own
order and may use emulation modes), matrix-free and spec operators (the term application would need a GPU
implementation), S3 inertia counts and the exact layer are not touched.

## Findings on Linux (present on plain dev/3.3)

These are the first runs of the 3.3 test suite on Linux; every evidence run so far was on the Windows workstation.

- `tests/test_parallel.py::test_low_energy_identical_for_1_2_4_workers` fails: the serial reference of
  `compose.certify_low_energy` is INCONCLUSIVE instead of CERTIFIED, in the full suite and when run alone.
- `tests/test_planner.py::test_predicted_inertia_memory_within_factor_two_of_measured_peak` fails for three cases in
  the full suite (measured peaks 140.6, 0.0 and 266.5 MB against predictions of 853.7, 131.9 and 1372.6 MB) and passes
  when the file runs alone, so the measurement in a fresh process is not reliable on this host after the rest of the
  suite.
- CuPy on this host compiles kernels only with `CUDA_PATH` set (to `/usr`) and cannot build its own elementwise kernels
  (no `cuda_fp16.h`).

## Status and open decisions

D-05 stands at the product path: opt-in, bit for bit equal to the CPU path, with probe and spot checks, and no change
of any bound or statement. Decisions for Sarang Vehale: whether this reading (no derivation document is needed,
because the GPU executes the CPU's operations and both checks test that on each device) is accepted; whether a next
step may move the reductions to the GPU at the price of certificates that differ from the CPU's in the last bits
(same model, different summation order), or only the elementwise parts, which keep the bits; and whether a release
with the GPU path needs its own evidence run on a GPU machine (D-44).
