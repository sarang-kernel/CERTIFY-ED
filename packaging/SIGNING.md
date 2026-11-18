# Signed releases

Release assets (the sdist, the wheel, `SHA256SUMS`, and the archived evidence run) are signed with
[sigstore](https://www.sigstore.dev/) using keyless signing: the signature is bound to an OpenID
identity (the release account's e-mail through GitHub or Google) and recorded in the public Rekor
transparency log, so there is no long-lived private key to protect or lose. The Python client is
`sigstore` (`pip install sigstore`).

## Signing (release manager)

```
cd dist
sha256sum certify_ed-X.Y.Z.tar.gz certify_ed-X.Y.Z-py3-none-any.whl > SHA256SUMS
python -m sigstore sign certify_ed-X.Y.Z.tar.gz certify_ed-X.Y.Z-py3-none-any.whl SHA256SUMS
```

`sign` opens a browser for the OIDC flow and writes one `<asset>.sigstore.json` bundle per asset
(certificate, signature and the Rekor entry). Upload the bundles next to the assets on GitHub and Zenodo.
In GitHub Actions the same command signs with the workflow's OIDC token (`id-token: write`), which binds
the signature to the repository and workflow instead of a person; the release workflow is not part of
this version and a release is signed by hand.

## Verifying (anyone)

```
python -m sigstore verify identity \
    --cert-identity sarangvehale2@gmail.com \
    --cert-oidc-issuer https://github.com/login/oauth \
    certify_ed-X.Y.Z.tar.gz
sha256sum -c SHA256SUMS
```

Both the identity and the issuer must match: a valid signature by someone else is a failure. The expected
identity for releases is the maintainer address above (`GOVERNANCE.md`); a change of identity is announced
in the CHANGELOG one release ahead.

## What a signature does and does not say

A signature says the file was produced by the holder of the identity at the time in the Rekor entry and
has not changed since. It does not say the computation behind a certificate was right; that is what
re-running `certify` does (`docs/schema.md`). The two mechanisms are independent: a certificate's
`integrity.body_sha256` protects the certificate body, the release signature protects the code that
wrote it.

## Recording the distribution hash in certificates (future work)

The certificate schema reserves `environment.distribution_sha256` for the SHA-256 of the installed
distribution. 3.1 does not write it: the hash of an installed package is not well defined (byte-compiled
files, platform line endings, editable installs), and the existing `environment.source_hash` (the git
blob hashes of the computational source, `provenance.py`) already identifies the code when run from a
checkout. A future version may write the `RECORD`-file hash of the installed wheel when the package is
not an editable install; until then the field stays null and the schema allows it.
