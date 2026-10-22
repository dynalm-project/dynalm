#!/usr/bin/env python3
"""Print DynaLM's release version from CMakeLists.txt: project(VERSION) plus
DYNALM_VERSION_SUFFIX (e.g. 0.2.0-beta.1). With --check-tag TAG, exit 1 unless
TAG is "v" + that version. With --prerelease, print "true" for a version with a
pre-release suffix and "false" for a stable one.

  python3 tools/ci/version.py [--check-tag v0.2.0-beta.1 | --prerelease]
"""
import re
import sys
from pathlib import Path

top = (Path(__file__).resolve().parents[2] / "CMakeLists.txt").read_text()
base = re.search(r"project\(dynalm\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", top).group(1)
suffix = re.search(r'set\(DYNALM_VERSION_SUFFIX\s+"([^"]*)"\)', top).group(1)
if suffix and not re.fullmatch(r"-(alpha|beta|rc)\.[0-9]+", suffix):
    sys.exit(f"DYNALM_VERSION_SUFFIX '{suffix}': expected '' or -alpha.N / -beta.N / -rc.N")
version = base + suffix
if len(sys.argv) == 3 and sys.argv[1] == "--check-tag":
    if sys.argv[2] != "v" + version:
        sys.exit(f"tag {sys.argv[2]} does not match the source version v{version} (CMakeLists.txt)")
    print(f"tag {sys.argv[2]} matches the source version")
elif len(sys.argv) == 2 and sys.argv[1] == "--prerelease":
    print("true" if suffix else "false")
else:
    print(version)
