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
