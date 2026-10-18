# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub:
**Security → Report a vulnerability** on this repository. We aim to acknowledge reports within 3 days.

Areas that are in scope:
- The HTTP server (`dynalm serve`): request parsing, admission control, resource exhaustion.
- Model loading: GGUF and SafeTensors parsing of untrusted files.
- `dynalm pull` and `dynalm models rm`.

## Deployment notes

- `dynalm serve` binds to `127.0.0.1` by default. On loopback it needs no key; it refuses browser
  requests from other sites (Host check, JSON-only requests).
- To listen on a network address (`--host 0.0.0.0`) it requires an API key (`--api-key-file`)
  unless you pass `--allow-unauthenticated`. There is no built-in TLS: put a TLS reverse proxy in
  front for anything beyond a trusted network. Details: [docs/security.md](docs/security.md) and
  [docs/deployment.md](docs/deployment.md).
- The shutdown endpoint only accepts requests from loopback.
- Model files are parsed defensively (bounds-checked headers, tensor extents validated against the file).
  Still, only load models from sources you trust.

## Verifying a download

Every release lists the SHA-256 of each archive in `SHA256SUMS`:

```bash
sha256sum -c SHA256SUMS --ignore-missing      # macOS: shasum -a 256 -c SHA256SUMS --ignore-missing
```

Releases after v0.1.7 also carry signed build provenance (GitHub artifact attestations). It
proves an archive was built by this repository's release workflow:

```bash
gh attestation verify dynalm-linux-x86_64.tar.gz --repo dynalm-project/dynalm
```

The Docker image (`ghcr.io/dynalm-project/dynalm:cpu`) carries build provenance and an SBOM
(software bill of materials): `docker buildx imagetools inspect ghcr.io/dynalm-project/dynalm:cpu --format '{{ json .SBOM }}'`.
