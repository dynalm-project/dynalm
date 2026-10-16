#!/usr/bin/env python3
"""Scans files for credentials that must never be committed or shipped.

  python3 tools/ci/scan_secrets.py                 # git-tracked files + untracked, not ignored
  python3 tools/ci/scan_secrets.py <archive|dir>   # a release package (tar.gz, zip) or directory

Exit 1 on any finding. Scope: high-signal patterns (private keys, AWS, GitHub,
Hugging Face, OpenAI-style and Slack tokens, bearer tokens in files) and
forbidden file types (.env, key material, model weights). It is a guard
against accidents, not a complete secret scanner; it does not inspect Git
history.
"""
import os, re, subprocess, sys, tarfile, zipfile

PATTERNS = [
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b")),
    ("OpenAI-style key", re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b")),
    ("Slack token", re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("hard-coded bearer token", re.compile(rb"Authorization: Bearer [A-Za-z0-9._~+/-]{24,}")),
]
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|id_rsa|id_ed25519|.*\.pem|.*\.key|.*\.p12|.*\.gguf|.*\.safetensors)$")
# Test fixtures that intentionally contain fake credentials.
ALLOW = {"dynalm/tests/test_server.cpp", "tools/ci/scan_secrets.py"}


def scan(name, data, findings):
    if FORBIDDEN_NAMES.search(name):
        findings.append((name, "forbidden file type"))
    if name in ALLOW:
        return
    for label, rx in PATTERNS:
        if rx.search(data):
            findings.append((name, label))


def main():
    findings, n = [], 0
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            for root, _, files in os.walk(target):
                for f in files:
                    p = os.path.join(root, f)
                    with open(p, "rb") as fh:
                        scan(os.path.relpath(p, target), fh.read(), findings)
                    n += 1
        elif target.endswith((".tar.gz", ".tgz")):
            with tarfile.open(target) as t:
                for m in t.getmembers():
                    if m.isfile():
                        scan(m.name, t.extractfile(m).read(), findings)
                        n += 1
        elif target.endswith(".zip"):
            with zipfile.ZipFile(target) as z:
                for m in z.namelist():
                    if not m.endswith("/"):
                        scan(m, z.read(m), findings)
                        n += 1
        else:
            sys.exit(f"unsupported target: {target}")
    else:
        files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True,
                               check=True).stdout.split()
        for f in files:
            if os.path.isfile(f):
                with open(f, "rb") as fh:
                    scan(f, fh.read(), findings)
                n += 1
    for name, label in findings:
        print(f"FINDING  {name}: {label}")
    print(f"secret scan: {n} files, {len(findings)} findings")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
