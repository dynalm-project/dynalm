# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub:
**Security → Report a vulnerability** on this repository. We aim to acknowledge reports within 3 days.

Areas that are in scope:
- The HTTP server (`dynalm serve`): request parsing, admission control, resource exhaustion.
- Model loading: GGUF and SafeTensors parsing of untrusted files.
- `dynalm pull` and `dynalm rm`.

## Deployment notes

- `dynalm serve` binds to `127.0.0.1` by default. It has no authentication. Put it behind a reverse
  proxy with authentication before exposing it to a network (`--host 0.0.0.0`).
- The shutdown endpoint only accepts requests from loopback.
- Model files are parsed defensively (bounds-checked headers, tensor extents validated against the file).
  Still, only load models from sources you trust.
