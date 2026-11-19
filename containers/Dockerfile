# The 3.1.0 evidence-run stack (containers/README.md): Python 3.11 with pinned numpy, scipy, python-flint,
# QuSpin and the optional layers. Build from the repository root:
#   docker build -t certify-ed:3.1.0 -f containers/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# git: certificates record the commit and the computational source hash when run from a checkout
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        numpy==2.4.4 \
        scipy==1.17.1 \
        python-flint==0.9.0 \
        h5py \
        matplotlib \
        mpmath \
        jsonschema \
        pytest \
        ruff \
        quspin-extensions parallel-sparse-tools dill joblib numba numexpr six gmpy2 \
    && pip install quspin==1.0.1 --no-deps

WORKDIR /opt/certify-ed
COPY . .
RUN pip install --no-deps -e . && certify-ed env

# a mounted working directory for bundles and run directories
WORKDIR /work
ENTRYPOINT ["certify-ed"]
CMD ["env"]
