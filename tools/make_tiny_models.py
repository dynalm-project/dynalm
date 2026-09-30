#!/usr/bin/env python3
"""Writes tiny random-weight GGUF models (one per architecture) plus golden
reference outputs from tools/ref_model.py, for adapter tests that need no
downloads.

    PYTHONUTF8=1 python tools/make_tiny_models.py tests/data

Each model: 2-3 layers, hidden 64, byte-level vocab of 260 tokens, f16
weights, fixed seed. Output: tests/data/tiny_<arch>.gguf and ref_tiny_<arch>.txt.
"""
import os
import subprocess
import sys

import numpy as np
from gguf import GGUFWriter

VOCAB = 260
HIDDEN = 64
FF = 96
PROMPT = [5, 17, 99, 3, 200, 42, 7, 128, 64, 11, 250, 33]


def byte_tokens():
    # GPT-2 byte-to-unicode mapping for the first 256 tokens, then specials.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = dict(zip(bs, cs))
    return [chr(m[b]) for b in range(256)] + ["<s>", "</s>", "<pad>", "<unk>"]


def spec(arch):
    """Per-architecture hyperparameters and tensor layout."""
    s = dict(layers=2, heads=4, kv_heads=2, head_dim=16, qkv_bias=False, qk_norm=False,
             post_norms=False, fused=False, window=0, tied=False, attn_cap=0.0, final_cap=0.0,
             base=10000.0)
    if arch == "qwen2":
        s.update(qkv_bias=True, tied=True, base=1000000.0)
    elif arch == "qwen3":
        s.update(qk_norm=True, head_dim=32)  # head_dim decoupled from hidden/heads
    elif arch == "gemma":
        s.update(kv_heads=1, head_dim=32, tied=True)
    elif arch == "gemma2":
        s.update(layers=3, post_norms=True, window=4, tied=True, attn_cap=50.0, final_cap=30.0)
    elif arch == "gemma3":
        s.update(layers=7, qk_norm=True, post_norms=True, window=4, tied=True, head_dim=32,
                 base=1000000.0)
    elif arch == "phi3":
        s.update(fused=True, window=6)
    return s


def make(arch, out_dir, rng):
    s = spec(arch)
    path = os.path.join(out_dir, f"tiny_{arch}.gguf")
    w = GGUFWriter(path, arch)
    w.add_name(f"tiny-{arch}")
    w.add_block_count(s["layers"])
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(FF)
    w.add_head_count(s["heads"])
    w.add_head_count_kv(s["kv_heads"])
    w.add_context_length(128)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(s["base"])
    w.add_key_length(s["head_dim"])
    w.add_value_length(s["head_dim"])
    w.add_rope_dimension_count(s["head_dim"])
    if s["window"]:
        w.add_sliding_window(s["window"])
    if s["attn_cap"]:
        w.add_attn_logit_softcapping(s["attn_cap"])
        w.add_final_logit_softcapping(s["final_cap"])
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(byte_tokens())
    w.add_token_types([1] * 256 + [3, 3, 3, 2])
    w.add_token_merges([])
    w.add_bos_token_id(256)
    w.add_eos_token_id(257)

    def t(name, *shape, scale=0.15, offset=0.0):
        w.add_tensor(name, (rng.standard_normal(shape) * scale + offset).astype(np.float16))

    def norm(name, n):
        t(name, n, scale=0.1, offset=1.0)

    q, kv = s["heads"] * s["head_dim"], s["kv_heads"] * s["head_dim"]
    # Gemma multiplies embeddings by sqrt(hidden); keep the residual stream O(1).
    t("token_embd.weight", VOCAB, HIDDEN, scale=0.06 if arch.startswith("gemma") else 0.5)
    norm("output_norm.weight", HIDDEN)
    if not s["tied"]:
        t("output.weight", VOCAB, HIDDEN)
    for l in range(s["layers"]):
        p = f"blk.{l}."
        norm(p + "attn_norm.weight", HIDDEN)
        if s["fused"]:
            t(p + "attn_qkv.weight", q + 2 * kv, HIDDEN)
        else:
            t(p + "attn_q.weight", q, HIDDEN)
            t(p + "attn_k.weight", kv, HIDDEN)
            t(p + "attn_v.weight", kv, HIDDEN)
        if s["qkv_bias"]:
            t(p + "attn_q.bias", q)
            t(p + "attn_k.bias", kv)
            t(p + "attn_v.bias", kv)
        if s["qk_norm"]:
            norm(p + "attn_q_norm.weight", s["head_dim"])
            norm(p + "attn_k_norm.weight", s["head_dim"])
        t(p + "attn_output.weight", HIDDEN, q)
        if s["post_norms"]:
            norm(p + "post_attention_norm.weight", HIDDEN)
            norm(p + "post_ffw_norm.weight", HIDDEN)
        norm(p + "ffn_norm.weight", HIDDEN)
        if s["fused"]:
            t(p + "ffn_up.weight", 2 * FF, HIDDEN)  # gate|up, as Phi-3 GGUFs store it
        else:
            t(p + "ffn_gate.weight", FF, HIDDEN)
            t(p + "ffn_up.weight", FF, HIDDEN)
        t(p + "ffn_down.weight", HIDDEN, FF)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def main():
    out = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    for i, arch in enumerate(["llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3"]):
        path = make(arch, out, np.random.default_rng(1000 + i))
        ref = os.path.join(out, f"ref_tiny_{arch}.txt")
        subprocess.run([sys.executable, os.path.join(here, "ref_model.py"), path, "--tokens",
                        *map(str, PROMPT), "--steps", "6", "--out", ref], check=True)
        print(f"{arch}: {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
