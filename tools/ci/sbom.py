#!/usr/bin/env python3
"""CycloneDX 1.5 SBOM of a DynaLM build, from the pins in the build files.

  python3 tools/ci/sbom.py [--ggml] > sbom.cdx.json

Components: DynaLM itself, cpp-httplib (version and SHA-256 from
dynalm/src/CMakeLists.txt), the Unicode Character Database data the tokenizer
tables are generated from (dynalm/src/tokenizer/unicode_tables.inc) and, with
--ggml (alias --metal), ggml (commit from cmake/Ggml.cmake, DynaLM's patches from
cmake/patches as its pedigree): every build with
a GPU backend (Metal, Vulkan, CUDA) or ggml's CPU kernels (ENABLE_GGML_CPU,
on by default; tools/ci/package.sh reads it from the build's CMakeCache.txt).
Scope: the third-party code and data compiled into the binaries.
System libraries (libc++, the macOS frameworks, Windows runtime) and build
tools are not listed.
"""
import json, re, sys, uuid
from pathlib import Path

root = Path(__file__).resolve().parents[2]
top = (root / "CMakeLists.txt").read_text()
# The release version: project(VERSION) plus DYNALM_VERSION_SUFFIX (CMakeLists.txt).
version = re.search(r"project\(dynalm\s+VERSION\s+([0-9.]+)", top).group(1) + \
    re.search(r'set\(DYNALM_VERSION_SUFFIX\s+"([^"]*)"\)', top).group(1)
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
tables = (root / "dynalm/src/tokenizer/unicode_tables.inc").read_text()
ucd_ver = re.search(r"from Unicode ([0-9]+(?:\.[0-9]+)*)", tables).group(1)
components.append({
    "type": "data", "name": "Unicode Character Database", "version": ucd_ver,
    "licenses": [{"license": {"id": "Unicode-3.0"}}],
    "externalReferences": [{"type": "distribution", "url": f"https://www.unicode.org/Public/{ucd_ver}/ucd/"}],
})
if "--ggml" in sys.argv or "--metal" in sys.argv:
    g = (root / "cmake/Ggml.cmake").read_text()
    commit = re.search(r"set\(DYNALM_GGML_COMMIT\s+([0-9a-f]{40})", g).group(1)
    gver = re.search(r"set\(DYNALM_GGML_VERSION\s+([0-9.]+)", g).group(1)
    ggml = {
        "type": "library", "name": "ggml", "version": gver,
        "purl": f"pkg:github/ggml-org/ggml@{commit}",
        "licenses": [{"license": {"id": "MIT"}}],
        "externalReferences": [{"type": "vcs", "url": "https://github.com/ggml-org/ggml", "comment": f"commit {commit}"}],
    }
    # DynaLM's patches to the pinned tree (cmake/patches, applied by cmake/GgmlPatch.cmake).
    patches = sorted((root / "cmake/patches").glob("*.patch"))
    if patches:
        ggml["pedigree"] = {"patches": [{"type": "unofficial",
                                         "diff": {"url": f"cmake/patches/{p.name}"}} for p in patches]}
    components.append(ggml)
bom = {
    "bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": f"urn:uuid:{uuid.uuid4()}", "version": 1,
    "metadata": {"component": {"type": "application", "name": "dynalm", "version": version,
                               "licenses": [{"license": {"id": "Apache-2.0"}}]},
                 "properties": [{"name": "scope", "value": "third-party code compiled into the binaries; system libraries not listed"}]},
    "components": components,
}
json.dump(bom, sys.stdout, indent=2)
print()
