# Containers: the 3.1.0 evidence-run stack

`Dockerfile` and `certify-ed.def` (Apptainer) build the software stack on which the 3.1.0 evidence run
was made (`RESULT_evidence_run_3.1.0.md`, `docs/reproducibility.md`):

| component | version | why pinned |
|---|---|---|
| Python | 3.11 | the run's interpreter; QuSpin 1.0.1 installs on it |
| numpy | 2.4.4 | the BLAS it bundles decides the floating-point behaviour the run-time probe checks |
| scipy | 1.17.1 | ARPACK, sparse kernels, LAPACK routines used by S2-S5 |
| python-flint | 0.9.0 (FLINT 3.6.0) | the exact layer (S6) and Arb references |
| QuSpin | 1.0.1 (`--no-deps`, see below) | importers, label audits, E0-E6; the `quspin-extensions` and `parallel-sparse-tools` wheels next to it |
| h5py, matplotlib, mpmath | latest at build time | HDF5 input/export, figures, the S7 reference values |
| jsonschema | latest at build time | certificate schema validation (`docs/schema.md`) |

QuSpin 1.0.1 is installed with `pip install quspin==1.0.1 --no-deps` after its real dependencies, because
the wheel declares the obsolete PyPI backport `importlib` (docs/reproducibility.md, "Installing QuSpin 1.0.1").

Both recipes copy the repository checkout into `/opt/certify-ed` and install it editable with
`--no-deps`, so the certificates written inside record the commit and the computational source hash.
The images are not built in CI (this version has no registry); build them locally:

```
docker build -t certify-ed:3.1.0 -f containers/Dockerfile .
docker run --rm -v "$PWD":/work certify-ed:3.1.0 certify --from my_run

apptainer build certify-ed_3.1.0.sif containers/certify-ed.def
apptainer run certify-ed_3.1.0.sif certify --from my_run
apptainer exec certify-ed_3.1.0.sif python /opt/certify-ed/run.py --full --jobs 4 --out /work/run_full
```

Thread counts are pinned to one (the experiments measure time and memory single-threaded). A
reproduction of the evidence run inside the container should give the same statuses and the same
numbers to the precision `experiments/compare_runs.py` checks; a difference is a finding, not noise,
and goes into the RESULT file of the run (`docs/reproducibility.md`, "Comparing two machines").

The images reproduce the *stack*; they do not make a certificate more trustworthy than the code that
wrote it. `packaging/SIGNING.md` says how the release that goes into the image is signed.
