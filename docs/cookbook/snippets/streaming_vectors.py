"""Eigenvectors larger than RAM: a numpy memmap (.npy on disk) or an HDF5 dataset, read in column blocks."""
import os
import tempfile

import numpy as np
import scipy.linalg as la

import certify_ed as ce

spec = ce.heisenberg_chain(12)
states = ce.u1_state_set(spec, 6, label="Sz=0")             # dimension 924
H = ce.build_bitwise(spec, states)                           # sparse: the streamed path applies (see the page)
w, V = la.eigh(H.toarray())                                  # the producer's full diagonalization

d = tempfile.mkdtemp()
path = os.path.join(d, "V.npy")
mm = np.lib.format.open_memmap(path, mode="w+", dtype=V.dtype, shape=V.shape)   # what a producer would write
mm[:] = V
mm.flush()
del mm

V_disk = np.load(path, mmap_mode="r")                        # never loaded whole by certify()
cert = ce.quick(H, w, V_disk, spec, kind="all", states=states, block_mb=64, plan=True)
print(cert.summary())
cert_mem = ce.quick(H, w, V, spec, kind="all", states=states)
print("same radii as in memory:", np.array_equal(cert.arrays["radius"], cert_mem.arrays["radius"]),
      "| same enclosures:", cert.body["enclosures"] == cert_mem.body["enclosures"])
print("plan recorded:", cert.body["plan"]["peak_mb"], "MB predicted peak,", round(cert.body["plan"]["total_seconds"], 1), "s predicted")
