# The environment and the code

This document sets up everything the other reproduction documents need: a Python environment with the library versions
of the evidence runs, a checkout of the code at the version to be reproduced, and the settings of the shell. The
commands are written for Windows PowerShell, because the primary machine of the project is a Windows workstation
(decision D-54); the POSIX form is given where it differs. Every version below is that of the 3.1.0, 3.2.0 and 3.2.1
evidence runs, read from their `manifest.json` (`environment`) and from `code/certify-ed/docs/reproducibility.md`.

## The reference versions

| Package | Version | Note |
|---|---|---|
| Python | 3.11.17 (Windows runs), 3.11.15 (Linux 3.0.0 run) | |
| numpy | 2.4.4 | |
| scipy | 1.17.1 | |
| python-flint | 0.9.0, with FLINT 3.6.0 | exact layer, Arb references, applications |
| quspin | 1.0.1 | installed without dependency resolution (below) |
| quspin-extensions | 0.1.8 | QuSpin's compiled part |
| parallel-sparse-tools | 0.2.5 | QuSpin's sparse kernels |
| h5py | 3.16.0 | HDF5 importers and export |
| matplotlib | 3.10.9 | figures |
| mpmath | 1.4.1 | high-precision checks in the derivation scripts |
| pytest | any recent version | the test suite |
| jsonschema | 4.18 or later | optional: schema validation |

QuSpin 1.0.1 declares the obsolete PyPI backport `importlib` as a requirement, which does not install on Python 3. Its
real runtime dependencies are therefore installed first and the wheel without dependency resolution (decision D-54;
`code/certify-ed/docs/reproducibility.md`). With conda:

```
conda create -n certifyed-311 python=3.11 -y
conda activate certifyed-311
pip install numpy==2.4.4 scipy==1.17.1 python-flint==0.9.0 h5py==3.16.0 matplotlib==3.10.9 mpmath==1.4.1 pytest "jsonschema>=4.18"
pip install quspin-extensions==0.1.8 parallel-sparse-tools==0.2.5 numba numexpr dill joblib six gmpy2
pip install quspin==1.0.1 --no-deps
```

The same sequence is used by `.github/workflows/ci.yml` and the container recipes in `containers/` of the code
repository. QuSpin is optional for using the package, but the experiments E0 to E5 need it, and without it those steps
are skipped and recorded as skipped.

## The code at the version to be reproduced

The code history is part of this repository, and the archive holds a full git clone of it at the time of archive v5,
`archive_v5/code/certify-ed/`. Work in a separate worktree of this repository, never in the archive copy, so that the
archive keeps its source hash. From the root of this repository:

```
git worktree add --detach ../work/certify-ed v3.2.1     # or v3.1.0 for the cited version, v3.2.0 for the failed run
cd ../work/certify-ed
git -c core.filemode=false status --porcelain          # must print nothing
```

The repository holds every tag of the code (`v1.0.0-baseline`, `v2-pre-homologation`, `v3.1.0-candidate`, `v3.1.0`,
`v3.2.0`, `v3.2.1` and later). Until 18 November 2026 this document cloned the git bundle `certify-ed_dev-3.2.bundle`
instead; its commits have the same ids here. The history does not contain the run records, because `results/` is
ignored by git; the records of every run are in the archive copy, `archive_v5/code/certify-ed/results/`, and are read
from there by the comparison and number commands of the other documents. The package is used from the checkout,
without installation, by putting the checkout on `PYTHONPATH`; `pip install -e .[all]` also works but is not needed.
On Windows the `certify-ed` console script exists only after installation, so the commands of these documents call
`python -m certify_ed.cli`, which is the same program.

## Shell settings

```
# PowerShell, from the root of the checkout
$env:PYTHONPATH = (Get-Location).Path
Remove-Item Env:PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
New-Item -ItemType Directory -Force -Path ..\pytest-tmp | Out-Null
```

```
# POSIX shell
export PYTHONPATH="$PWD"; unset PYTHONSAFEPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p ../pytest-tmp
```

Three settings matter. `PYTHONSAFEPATH` must not be set, because the experiments run scripts of `experiments/` as
subprocesses that import their neighbours (`harness.py`). The BLAS libraries run on one thread, as in every evidence
run; the driver sets this for its own children as well. pytest needs a temporary directory in a writable working
folder (`--basetemp ..\pytest-tmp`), because on the Windows workstation the default temporary directory under
`AppData` makes `tests/test_driver.py::test_reproduce_parallel_resume_and_numbers` fail for a reason unrelated to the
code (`docs/project/failures/04_failed_runs_and_attempts.md`).

## Checking the environment and the checkout

```
python -m certify_ed.cli env
python <repository>\tools\codeindex\source_hash.py . v3.1.0 v3.2.0 v3.2.1 HEAD
```

The first command prints the versions, the floating-point probe and the fingerprint of the checkout;
`standard_model_hypotheses_hold` must be `true` and `source_dirty` must be `false`. The second (`<repository>` is the folder of this
repository; the worktree, at an older commit, has no `tools/`) recomputes the source and test hashes of
each tag from the commits; they must be `acbef11af991c860`, `47446976d4dbfdb1` and `cb73dec3e4433a3b` for `v3.1.0`,
`v3.2.0` and `v3.2.1` (`docs/project/software/05_provenance.md`). A record produced by the checkout carries the fingerprint of
the working tree, so a record whose `source_hash` is `cb73dec3e4433a3b` with `source_dirty` false ran the
computational code of 3.2.1.

For the independent verifier a C compiler is needed: gcc, clang or cc on the path, or the variable `CEV_CC`. The
project used the MinGW gcc 14.2 that ships with GNU Octave (`code/certify-ed/verifier/README.md`); on the workstation
it is `C:\Program Files\GNU Octave\Octave-10.3.0\mingw64\bin\gcc.exe`. For the PDFs of the manuscript and the report,
a TeX distribution with `latexmk` and pandoc with xelatex are needed; they are not needed for any record.
