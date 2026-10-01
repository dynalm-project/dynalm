#!/usr/bin/env python3
"""Exports each tiny GGUF fixture as a Hugging Face model directory
(config.json, model.safetensors, tokenizer.json, tokenizer_config.json),
undoing the llama.cpp converter's transforms so the HF copy is what a real
HF checkpoint of the same weights would contain:

- Llama Q/K rows are un-permuted (GGUF permutes them for interleaved RoPE);
- Gemma RMSNorm weights are stored as w - 1 (GGUF folds HF's (1 + w));
- names, config keys and tokenizer files follow HF conventions.

The engine must load both forms to the same logits (test_safetensors).

    PYTHONUTF8=1 python tools/make_tiny_hf.py tests/data
"""
import json
import os
import struct
import sys

import numpy as np
from gguf import GGUFReader

ARCHES = ["llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3"]
MODEL_TYPE = {"gemma3": "gemma3_text"}


def field(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    return v.tobytes().decode() if v.dtype == np.uint8 else v[0].item()


def hf_name(arch, gname):
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gname == "output_norm.weight":
        return "model.norm.weight"
    if gname == "output.weight":
        return "lm_head.weight"
    _, l, rest = gname.split(".", 2)
    sandwich = arch in ("gemma2", "gemma3")
    m = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_qkv.weight": "self_attn.qkv_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.weight": "pre_feedforward_layernorm.weight" if sandwich else "post_attention_layernorm.weight",
        "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.gate_up_proj.weight" if arch == "phi3" else "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }[rest]
    return f"model.layers.{l}.{m}"


def unpermute(w, n_head):
    # Inverse of llama.cpp's permute(): rows (head, pair, half) -> (head, half, pair).
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, a in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        header[name] = {"dtype": "F16", "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def export(arch, data_dir):
    r = GGUFReader(os.path.join(data_dir, f"tiny_{arch}.gguf"))
    a = arch
    heads, kv_heads = field(r, f"{a}.attention.head_count"), field(r, f"{a}.attention.head_count_kv")
    head_dim = field(r, f"{a}.attention.key_length")
    out_dir = os.path.join(data_dir, f"hf_tiny_{arch}")
    os.makedirs(out_dir, exist_ok=True)

    tensors = {}
    for t in r.tensors:
        w = np.array(t.data, dtype=np.float32).reshape(list(reversed([int(x) for x in t.shape])))
        if arch == "llama" and t.name.endswith("attn_q.weight"):
            w = unpermute(w, heads)
        if arch == "llama" and t.name.endswith("attn_k.weight"):
            w = unpermute(w, kv_heads)
        if arch.startswith("gemma") and t.name.endswith("norm.weight"):
            w = w - 1.0
        tensors[hf_name(arch, t.name)] = w.astype(np.float16)
    write_safetensors(os.path.join(out_dir, "model.safetensors"), tensors)

    window = field(r, f"{a}.attention.sliding_window") or 0
    config = {
        "architectures": ["TinyForCausalLM"],
        "model_type": MODEL_TYPE.get(arch, arch),
        "vocab_size": len(r.fields["tokenizer.ggml.tokens"].data),
        "hidden_size": field(r, f"{a}.embedding_length"),
        "intermediate_size": field(r, f"{a}.feed_forward_length"),
        "num_hidden_layers": field(r, f"{a}.block_count"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": field(r, f"{a}.context_length"),
        "rms_norm_eps": field(r, f"{a}.attention.layer_norm_rms_epsilon"),
        "rope_theta": field(r, f"{a}.rope.freq_base"),
        "rope_scaling": None,
        "tie_word_embeddings": "lm_head.weight" not in tensors,
        "sliding_window": window or None,
        "torch_dtype": "float16",
        "bos_token_id": 256,
        "eos_token_id": 257,
    }
    if arch == "qwen2":
        config["use_sliding_window"] = False
    if field(r, f"{a}.attn_logit_softcapping"):
        config["attn_logit_softcapping"] = field(r, f"{a}.attn_logit_softcapping")
        config["final_logit_softcapping"] = field(r, f"{a}.final_logit_softcapping")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    tokens = [bytes(r.fields["tokenizer.ggml.tokens"].parts[i]).decode()
              for i in r.fields["tokenizer.ggml.tokens"].data]
    added = [{"id": i, "content": tokens[i], "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for i in range(256, len(tokens))]
    tok = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "byte_fallback": False,
                  "vocab": {t: i for i, t in enumerate(tokens[:256])}, "merges": []},
    }
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)
    tc = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>",
          "add_bos_token": False}
    tmpl = field(r, "tokenizer.chat_template")
    if tmpl:
        tc["chat_template"] = tmpl
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tc, f, ensure_ascii=False, indent=1)


def main():
    for arch in ARCHES:
        export(arch, sys.argv[1])
        print("wrote", f"hf_tiny_{arch}")


if __name__ == "__main__":
    main()
