#!/usr/bin/env python3
"""CycloneDX 1.5 SBOM of a DynaLM build, from the pins in the build files.

  python3 tools/ci/sbom.py [--ggml] > sbom.cdx.json

Components: DynaLM itself, cpp-httplib (version and SHA-256 from
dynalm/src/CMakeLists.txt) and, with --ggml (alias --metal; any GPU build:
Metal, Vulkan or CUDA), ggml (commit from cmake/Ggml.cmake). Scope: the third-party code compiled into the binaries.
System libraries (libc++, the macOS frameworks, Windows runtime) and build
tools are not listed.
"""
import json, re, sys, uuid
from pathlib import Path

root = Path(__file__).resolve().parents[2]
top = (root / "CMakeLists.txt").read_text()
version = re.search(r"project\(dynalm\s+VERSION\s+([0-9.]+)", top).group(1)
srv = (root / "dynalm/src/CMakeLists.txt").read_text()
httplib_ver = re.search(r"cpp-httplib/(v[0-9.]+)/httplib.h", srv).group(1)
httplib_sha = re.search(r"_httplib_sha256\s+([0-9a-f]{64})", srv).group(1)
components = [{
    "type": "library", "name": "cpp-httplib", "version": httplib_ver.lstrip("v"),
    "purl": f"pkg:github/yhirose/cpp-httplib@{httplib_ver}",
    "licenses": [{"license": {"id": "MIT"}}],
    "hashes": [{"alg": "SHA-256", "content": httplib_sha}],
    "externalReferences": [{"type": "distribution", "url": f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{httplib_ver}/httplib.h"}],
}]
if "--ggml" in sys.argv or "--metal" in sys.argv:
    g = (root / "cmake/Ggml.cmake").read_text()
    commit = re.search(r"set\(DYNALM_GGML_COMMIT\s+([0-9a-f]{40})", g).group(1)
    gver = re.search(r"set\(DYNALM_GGML_VERSION\s+([0-9.]+)", g).group(1)
    components.append({
        "type": "library", "name": "ggml", "version": gver,
        "purl": f"pkg:github/ggml-org/ggml@{commit}",
        "licenses": [{"license": {"id": "MIT"}}],
        "externalReferences": [{"type": "vcs", "url": "https://github.com/ggml-org/ggml", "comment": f"commit {commit}"}],
    })
bom = {
    "bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": f"urn:uuid:{uuid.uuid4()}", "version": 1,
    "metadata": {"component": {"type": "application", "name": "dynalm", "version": version,
                               "licenses": [{"license": {"id": "Apache-2.0"}}]},
                 "properties": [{"name": "scope", "value": "third-party code compiled into the binaries; system libraries not listed"}]},
    "components": components,
}
json.dump(bom, sys.stdout, indent=2)
print()
