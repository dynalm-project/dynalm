#!/usr/bin/env python3
"""Offline checks of the release versioning and workflow (no tags, no GitHub).

  python tools/test_release.py

- tools/ci/version.py: the version is project(VERSION) + DYNALM_VERSION_SUFFIX;
  pre-release detection; a tag must equal "v" + the version.
- .github/workflows/release.yml, by running its own shell steps on simulated
  inputs: the "Release kind and notes" step marks a pre-release tag as
  pre-release and not Latest, and requires the tag's release notes; the
  "Expected assets" step drops the Vulkan builds from a pre-release, keeps them
  for a stable release, and fails when any expected archive is missing.
- Structural rules: the release is a draft; publish runs only when every
  package job passed (no !cancelled()); package jobs check the tag and the
  installed binary's version; the Docker image job runs for stable tags only.
- The release notes for the current pre-release version exist.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = (ROOT / ".github/workflows/release.yml").read_text()
checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAILED: {what}")
        sys.exit(1)


def version_py(*args):
    return subprocess.run([sys.executable, str(ROOT / "tools/ci/version.py"), *args], capture_output=True, text=True)


def step_script(name):
    """The `run:` block of the workflow step `name`, dedented."""
    m = re.search(r"- name: " + re.escape(name) + r"\n(?:.*\n)*?\s+run: \|\n((?:\s{10}.*\n|\n)+)", WF)
    check(m is not None, f"workflow step '{name}' found")
    return "\n".join(l[10:] for l in m.group(1).splitlines())


# --- version.py -------------------------------------------------------------------
top = (ROOT / "CMakeLists.txt").read_text()
base = re.search(r"project\(dynalm\s+VERSION\s+([0-9.]+)", top).group(1)
suffix = re.search(r'set\(DYNALM_VERSION_SUFFIX\s+"([^"]*)"\)', top).group(1)
ver = version_py().stdout.strip()
check(ver == base + suffix, f"version.py prints project VERSION + suffix ({ver})")
check(version_py("--prerelease").stdout.strip() == ("true" if suffix else "false"), "pre-release detection")
check(version_py("--check-tag", "v" + ver).returncode == 0, "the matching tag is accepted")
check(version_py("--check-tag", "v" + base + ("" if suffix else "-beta.1")).returncode != 0, "a different tag is refused")
check(version_py("--check-tag", ver).returncode != 0, "a tag without 'v' is refused")
check('"@DYNALM_VERSION@"' in (ROOT / "dynalm/src/common/version.h.in").read_text(), "--version reads DYNALM_VERSION")

# --- workflow steps on simulated inputs ---------------------------------------------
kind = step_script("Release kind and notes")
assets = step_script("Expected assets")
ALL = ["dynalm-macos-arm64.tar.gz", "dynalm-macos-x86_64.tar.gz", "dynalm-linux-x86_64.tar.gz",
       "dynalm-linux-arm64.tar.gz", "dynalm-windows-x86_64.zip", "dynalm-windows-arm64.zip",
       "dynalm-linux-x86_64-vulkan.tar.gz", "dynalm-windows-x86_64-vulkan.zip"]


def run_assets(prerelease, present):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "dist").mkdir()
        for f in present:
            (Path(tmp) / "dist" / f).write_bytes(b"x")
        script = assets.replace("${{ steps.kind.outputs.prerelease }}", "true" if prerelease else "false")
        r = subprocess.run(["bash", "-e", "-c", script], cwd=tmp, capture_output=True, text=True)
        return r.returncode, sorted(os.listdir(Path(tmp) / "dist"))


rc, left = run_assets(True, ALL)
check(rc == 0 and not any("vulkan" in f for f in left) and len(left) == 6, "pre-release: Vulkan builds left out, 6 assets")
rc, left = run_assets(False, ALL)
check(rc == 0 and len(left) == 8, "stable: all 8 assets kept")
rc, _ = run_assets(True, [f for f in ALL if f != "dynalm-linux-arm64.tar.gz"])
check(rc != 0, "a missing archive fails the release")
rc, _ = run_assets(False, [f for f in ALL if "vulkan" not in f])
check(rc != 0, "stable: a missing Vulkan archive fails the release")


def run_kind(tag, notes):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "tools/ci").mkdir(parents=True)
        (Path(tmp) / ".github/release-notes").mkdir(parents=True)
        (Path(tmp) / "tools/ci/version.py").write_text((ROOT / "tools/ci/version.py").read_text())
        b, s = (tag[1:].split("-", 1) + [""])[:2]
        (Path(tmp) / "CMakeLists.txt").write_text(
            f'project(dynalm\n  VERSION {b}\n  LANGUAGES CXX)\nset(DYNALM_VERSION_SUFFIX "{"-" + s if s else ""}")\n')
        if notes:
            (Path(tmp) / f".github/release-notes/{tag}.md").write_text("notes\n")
        out = Path(tmp) / "out.txt"
        env = dict(os.environ, GITHUB_REF_NAME=tag, GITHUB_OUTPUT=str(out))
        r = subprocess.run(["bash", "-e", "-c", kind], cwd=tmp, env=env, capture_output=True, text=True)
        return r.returncode, dict(l.split("=", 1) for l in out.read_text().splitlines()) if out.exists() else {}


rc, o = run_kind("v0.2.0-beta.1", True)
check(rc == 0 and o.get("prerelease") == "true" and o.get("latest") == "false", "v0.2.0-beta.1: pre-release, not Latest")
rc, o = run_kind("v0.2.0", True)
check(rc == 0 and o.get("prerelease") == "false" and o.get("latest") == "true", "v0.2.0: stable, Latest")
rc, _ = run_kind("v0.2.0-rc.2", False)
check(rc != 0, "missing release notes fail the release")

# --- structure ----------------------------------------------------------------------
pub = WF[WF.index("\n  publish:"):WF.index("\n  docker:")]
check("!cancelled()" not in pub and "if: startsWith(github.ref, 'refs/tags/')" in pub,
      "publish needs every package job to pass (no !cancelled())")
check("draft: true" in pub, "the release is created as a draft")
check("prerelease: ${{ steps.kind.outputs.prerelease }}" in pub and "make_latest: ${{ steps.kind.outputs.latest }}" in pub,
      "pre-release flag and Latest come from the version")
check("body_path: ${{ steps.kind.outputs.notes }}" in pub, "release notes from .github/release-notes/<tag>.md")
pkg = WF[WF.index("\n  package:"):WF.index("\n  publish:")]
check('version.py --check-tag "$GITHUB_REF_NAME"' in pkg, "package jobs check the tag against the source version")
check('DYNALM_EXPECT_VERSION="${GITHUB_REF_NAME#v}"' in pkg, "package jobs check the installed binary's version")
check("fail-fast: false" in pkg, "every platform still runs (and reports) when one fails")
dock = WF[WF.index("\n  docker:"):]
check("if: startsWith(github.ref, 'refs/tags/') && !contains(github.ref_name, '-')" in dock,
      "the Docker image job runs for stable tags only")
pkgsh = (ROOT / "tools/ci/package.sh").read_text()
check("DYNALM_EXPECT_VERSION" in pkgsh and "version mismatch" in pkgsh, "package.sh compares the installed binary's version")

# --- release notes for the current pre-release ----------------------------------------
if suffix:
    check((ROOT / f".github/release-notes/v{ver}.md").is_file(), f"release notes for v{ver} exist")

print(f"all release workflow tests passed ({checks} checks)")
