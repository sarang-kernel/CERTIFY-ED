"""Read a saved certificate: statuses, radii, the scope block."""
import numpy as np

import certify_ed as ce

cert = ce.Certificate.load("quickstart_certificate.json")        # verifies the body hash and the scope block
for s in cert.statements:
    print(f"{s['id']:20s} {s['status']:12s} reason_code={s['evidence'].get('reason_code')}")
r = cert.arrays["radius"]
print("eigenvalues  ", np.array2string(cert.arrays["eigenvalues"], precision=10))
print("radii        ", np.array2string(r, precision=2))
print("max radius   ", f"{r.max():.3e}")
s3 = [s for s in cert.statements if s["id"] == "S3.completeness"][0]
print("gap >=       ", f"{s3['evidence']['gap_lower_bound']:.6f}")
print("version      ", cert.body["environment"]["certify_ed"])
print("scope in     ", [x["id"] for x in cert.body["scope"]["in_scope"]])
print("underflow    ", cert.body["scope"]["options"]["underflow_check"])
