#!/usr/bin/env python3
"""DynaCore boundary check (DD-068).

DynaCore is the low-level inference runtime; DynaLM is built on it. The
dependency runs one way only. This check fails when anything under dynacore/

  1. includes a project header that is not a DynaCore header, or
  2. names an LLM-platform concept in code or string literals: model families,
     file formats, tokenizers, HTTP/OpenAI, scheduling, the product name.

Comments are ignored, so a kernel may still say "matches gguf-py's reference".
Run from anywhere:  python tests/boundary/check_boundary.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "dynacore"

# Concepts DynaCore must not know (case-insensitive, matched on identifiers).
BANNED = [
    r"gguf", r"safetensors?", r"tokeni[sz]", r"chat_?template", r"https?\b", r"httplib", r"openai",
    r"scheduler", r"model_?ir", r"model_?config", r"model_?loader", r"sampler", r"prefix_?cache",
    r"dynalm", r"llama", r"qwen", r"gemma", r"\bphi\d*\b", r"mistral", r"mixtral", r"granite", r"deepseek",
]
# Approved generic concepts that contain a banned substring.
ALLOWED = [
    r"\bkLlama3\b", r"\bllama3_(low|high)_freq_factor\b",  # Llama-3 RoPE scaling: a published RoPE method
]

INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def strip_comments(src: str) -> str:
    """Removes // and /* */ comments, keeps string literals and line numbers."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * src.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def main() -> int:
    errors = []
    files = [p for p in CORE.rglob("*") if p.suffix in (".h", ".hpp", ".cpp", ".cc", ".inc", ".cu", ".cuh")
             and "data" not in p.relative_to(CORE).parts]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for m in INCLUDE.finditer(text):
            inc = m.group(1)
            ok = inc.startswith("dynacore/") or (path.parent / inc).exists()
            if not ok:
                line = text.count("\n", 0, m.start()) + 1
                errors.append(f"{rel}:{line}: includes non-DynaCore header \"{inc}\"")
        code = strip_comments(text)
        for allowed in ALLOWED:
            code = re.sub(allowed, "", code)
        for lineno, line in enumerate(code.splitlines(), 1):
            if line.lstrip().startswith("#include"):
                continue
            for pat in BANNED:
                if re.search(pat, line, re.I):
                    errors.append(f"{rel}:{lineno}: names an LLM-platform concept /{pat}/: {line.strip()}")
    if errors:
        print("DynaCore boundary violations (DynaCore must not depend on DynaLM, DD-068):")
        print("\n".join(errors))
        return 1
    print(f"boundary ok: {len(files)} DynaCore files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
