#!/usr/bin/env python3
"""Generates tokenizer golden fixtures from HF reference tokenizers.

    PYTHONUTF8=1 python tools/gen_tokenizer_golden.py models/hf/smollm2/tokenizer.json dynalm/tests/data/golden_smollm2.txt
    PYTHONUTF8=1 python tools/gen_tokenizer_golden.py models/hf/qwen25/tokenizer.json dynalm/tests/data/golden_qwen25.txt

Output format, one case per line:  <hex-encoded UTF-8 text>\t<space-separated ids>
(no special tokens added; special-token text in inputs is parsed as HF does).
"""
import sys

from tokenizers import Tokenizer

CORPUS = [
    "Hello world",
    "Hello, world! How are you?",
    " leading space",
    "  two leading spaces",
    "trailing space ",
    "trailing spaces   ",
    "I'm here, you're there, they'll go, we've done, he'd say, it's fine, don't.",
    "SHOUTING I'M HERE THEY'LL GO",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 1,000,000",
    "Tabs\tand\ttabs\t\tdouble",
    "Newlines\nsingle\n\ndouble\n\n\ntriple",
    "Mixed  \n  whitespace \t\n end",
    "   \n\n   ",
    "\n",
    " ",
    "Unicode: café naïve résumé Ünïcödé",
    "中文测试：你好，世界！",
    "日本語のテキストです。",
    "한국어 텍스트",
    "Emoji 😀🎉👍🏽 and 🇺🇸 flags",
    "Arabic: مرحبا بالعالم",
    "Hindi: नमस्ते दुनिया",
    "Math: ∑ x² ≤ ∞ ± √2",
    "def f(x):\n    return x ** 2  # square\n",
    "for (int i = 0; i < n; ++i) { a[i] += b[i]; }",
    "URL: https://example.com/path?q=1&r=two#frag",
    "email@example.com, @handle, #hashtag, $100, 50%",
    "!!! ??? ... --- ___ *** ~~~",
    "a" * 50,
    "The quick brown fox jumps over the lazy dog. " * 3,
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "Special text inline: <|endoftext|> in the middle",
    "Mr. O'Neil's 'quoted' text",
    "x",
    "0",
    "007 and 2024-09-30 12:34:56",
    " non-breaking space",
    "zero​width",
]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    with open(sys.argv[2], "w", newline="\n") as out:
        for text in CORPUS:
            ids = tok.encode(text, add_special_tokens=False).ids
            out.write(text.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)) + "\n")
    print(f"wrote {len(CORPUS)} cases to {sys.argv[2]}")


if __name__ == "__main__":
    main()
